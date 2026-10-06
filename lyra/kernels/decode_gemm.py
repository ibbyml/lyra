from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.tpu import ARBITRARY, PARALLEL
from jax.tree_util import register_static
from jaxtyping import Float, Int32

from lyra.kernels.common import (
    BF16,
    FP32,
    TILE_INNER,
    Implementation,
    current_tpu_target,
    dispatch_kernel,
    divides,
    mxu_dot,
    require,
    swiglu,
    tpu_vmem_limit_bytes,
)
from lyra.kernels.gemm import (
    _expert_ids,
    grouped_gemm_reference,
    normalize_bias,
)
from lyra.nn.quant import QArray, dequantize, is_scaled, quantize

_A = lambda x: cast(Array, x)


@register_static
@dataclass(frozen=True)
class DecodeGEMMConfig:
    tile_m: int | None = None
    tile_n: int | None = None
    tile_g: int = 2

    block_compute: int | None = None
    rhs_buffer_count: int = 2
    scale_k: int = 256
    scale_n: int = 256
    fwd_dtype: jnp.dtype | None = None


_DECODE_GEMM_CONFIGS = {
    (2048, 4096): DecodeGEMMConfig(tile_n=1024),
    (2048, 2048): DecodeGEMMConfig(tile_n=1024),
    (4096, 8192): DecodeGEMMConfig(tile_n=1024),
    (4096, 4096): DecodeGEMMConfig(tile_n=1024),
}


def _resolve_tiling(config: DecodeGEMMConfig, m: int, n: int) -> tuple[int, int, int]:
    tile_m = min(config.tile_m if config.tile_m is not None else m, m)
    block_compute = tile_m if config.block_compute is None else min(config.block_compute, tile_m)
    tile_n = config.tile_n if config.tile_n is not None else n
    tile_n = 128 * pl.cdiv(max(128, min(tile_n, n)), 128)
    return tile_m, tile_n, block_compute


def select_decode_gemm_config(
    m: int,
    groups: int,
    k: int,
    n: int,
    quant_dtype: jnp.dtype | None = None,
) -> DecodeGEMMConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    if min(m, groups, k, n) <= 0:
        return None
    config = _DECODE_GEMM_CONFIGS.get((k, n), DecodeGEMMConfig(tile_n=1024)) if target == "v6e" else DecodeGEMMConfig()
    quant_dtype = quant_dtype if quant_dtype is not None and is_scaled(quant_dtype) else None
    config = replace(config, fwd_dtype=quant_dtype)
    tile_m, tile_n, block_compute = _resolve_tiling(config, m, n)
    if groups % config.tile_g:
        return None
    if m % tile_m or tile_m % block_compute:
        return None
    if k % 128 or n % 128 or n % tile_n:
        return None
    if quant_dtype is not None and (k % config.scale_k or n % config.scale_n or tile_n % config.scale_n):
        return None
    return config


def _lhs_row_map(
    group_sizes: Array,
    *,
    tile_g: int,
    tile_m: int,
    row_tiles: int,
) -> Array:
    G = group_sizes.shape[0]

    group_batches = G // tile_g
    batch_work = jnp.sum(group_sizes.reshape(group_batches, tile_g), axis=-1)
    work_total = jnp.pad(jnp.cumsum(batch_work), (1, 0))

    min_row = work_total[:-1] // tile_m
    row_map = min_row[:, None] + jnp.arange(row_tiles)[None, :]
    max_row = jnp.concatenate([min_row[1:], jnp.array([row_tiles - 1])])
    map = jnp.clip(row_map, max=jnp.minimum(max_row[:, None], row_tiles - 1))

    return map


def _rhs_group_map(group_sizes: Array, *, tile_g: int) -> Array:
    G = group_sizes.shape[0]

    group_batches = G // tile_g

    work_mask = jnp.sum(group_sizes.reshape(group_batches, tile_g), axis=-1) > 0
    unique = jnp.sort(jnp.arange(group_batches) * work_mask, descending=True)
    mapping = jnp.maximum(jnp.cumsum(jnp.flip(work_mask)) - 1, 0)
    map = jnp.flip(unique[mapping])

    return map


def _pack_decode_rhs_scales(scale: Array, *, scale_n: int, tile_n: int) -> Array:
    G, KB, NB = scale.shape
    n = NB * scale_n
    n_tiles = n // tile_n

    require(divides(n, tile_n), f"scaled N={n} must be divisible by tile_n={tile_n}")

    expanded = jnp.repeat(scale, scale_n, axis=-1)
    packed = expanded.reshape(G, KB, n_tiles, tile_n).transpose(0, 2, 1, 3)
    packed_scales = packed.reshape(G, n_tiles, 1, KB * tile_n)

    return packed_scales


def _decode_gemm_kernel(
    # Prefetced
    lhs_row_map: Array,
    rhs_group_map: Array,
    group_sizes_ref: Array,
    # In refs
    lhs_hbm: Array,
    rhs_hbm: Array,
    rhs_scale_hbm: Array,
    # Out refs
    out_hbm: Array,
    # Scratch refs
    row_cursor_ref: Array,
    group_id_ref: Array,
    group_rem_ref: Array,
    *,
    # static
    dims: tuple[int, ...],
    tiling: tuple[int, ...],
    config: DecodeGEMMConfig,
):
    M, G, K, N = dims
    TILE_M, TILE_N, TILE_G, BLOCK_COMPUTE = tiling
    RHS_BUFFER_COUNT = config.rhs_buffer_count
    QUANTIZED = config.fwd_dtype is not None
    K_BLOCKS = K // config.scale_k

    COL_TILES = pl.cdiv(N, TILE_N)
    GROUP_BATCHES = G // TILE_G
    ROW_TILES = M // TILE_M

    def lhs_index_map(n, g, m):
        return (lhs_row_map[g, m], 0)

    def rhs_index_map(n, g, m):
        return (rhs_group_map[g], 0, n)

    def rhs_scale_index_map(n, g, m):
        return (rhs_group_map[g], n, 0, 0)

    def out_index_map(n, g, m):
        return (lhs_row_map[g, m], n)

    def decode_pipeline_core(
        lhs_ref: Array,
        rhs_ref: Array,
        out_ref: Array,
        rhs_scale_ref: Array | None = None,
    ):
        g, m = pl.program_id(1), pl.program_id(2)
        bn = out_ref.shape[-1]
        row_block = lhs_row_map[g, m]

        is_first = (g == 0) & (m == 0)
        row_cursor = jnp.where(is_first, 0, row_cursor_ref[0])
        group_id = jnp.where(is_first, 0, group_id_ref[0])
        group_rem = jnp.where(m == 0, group_sizes_ref[g * TILE_G], group_rem_ref[0])

        prev = jnp.maximum(g * ROW_TILES + m - 1, 0)
        prev_row_block = lhs_row_map[prev // ROW_TILES, prev % ROW_TILES]

        _init = is_first | (prev_row_block != row_block)

        @pl.when(_init)
        def _():
            out_ref[...] = jnp.zeros_like(out_ref)

        def outer_body(i, carry):
            row_cursor, group_id, group_rem = carry
            local_i = i - row_block * (TILE_M // BLOCK_COMPUTE)
            y = out_ref[pl.ds(local_i * BLOCK_COMPUTE, BLOCK_COMPUTE), :].astype(jnp.float32)
            x = lhs_ref[pl.ds(local_i * BLOCK_COMPUTE, BLOCK_COMPUTE), :]

            def cond_fn(val):
                _, row_cursor, group_id, _ = val
                local_group = group_id - g * TILE_G
                return (row_cursor < (i + 1) * BLOCK_COMPUTE) & (local_group < TILE_G)

            def body_fn(val):
                y, row_cursor, group_id, group_rem = val
                n_compute = jnp.maximum(
                    jnp.minimum(group_rem, ((i + 1) * BLOCK_COMPUTE - row_cursor).astype(jnp.int32)),
                    0,
                )
                group_done = n_compute >= group_rem
                local_group = group_id - g * TILE_G

                def _compute():
                    A = rhs_ref[local_group, :, :]
                    if QUANTIZED:
                        assert rhs_scale_ref is not None
                        A_scale = rhs_scale_ref[local_group, 0, :].reshape(K_BLOCKS, TILE_N)
                        xA = jnp.zeros_like(y)
                        for scale_idx in range(K_BLOCKS):
                            start = scale_idx * config.scale_k
                            x_block = BF16(x[:, start : start + config.scale_k])
                            A_block = BF16(A[start : start + config.scale_k, :])
                            partial = mxu_dot(x_block, A_block, dimension_numbers=TILE_INNER)
                            partial *= A_scale[scale_idx][None, :]
                            xA += partial
                    else:
                        xA = mxu_dot(x, A.astype(x.dtype), dimension_numbers=TILE_INNER).astype(y.dtype)
                    iota = lax.broadcasted_iota(jnp.int32, (BLOCK_COMPUTE, bn), 0) + i * BLOCK_COMPUTE
                    mask = (iota >= row_cursor) & (iota < row_cursor + n_compute)
                    return jnp.where(mask, xA, y)

                new_y = lax.cond(n_compute > 0, _compute, lambda: y)
                new_group_id = jnp.where(group_done, group_id + 1, group_id)
                next_rem = group_sizes_ref[jnp.clip(group_id + 1, max=G - 1)]
                new_group_rem = jnp.where(group_done, next_rem, group_rem - n_compute)
                new_cursor = row_cursor + n_compute
                return new_y, new_cursor, new_group_id, new_group_rem

            y, new_cursor, new_group_id, new_group_rem = lax.while_loop(cond_fn, body_fn, (y, row_cursor, group_id, group_rem))
            out_ref[pl.ds(local_i * BLOCK_COMPUTE, BLOCK_COMPUTE), :] = y.astype(out_ref.dtype)
            return new_cursor, new_group_id, new_group_rem

        start = jnp.maximum(row_cursor, row_block * TILE_M) // BLOCK_COMPUTE
        end = jnp.minimum(M, (row_block + 1) * TILE_M) // BLOCK_COMPUTE
        new_cursor, new_group_id, new_group_rem = lax.fori_loop(start, end, outer_body, (row_cursor, group_id, group_rem))
        row_cursor_ref[0], group_id_ref[0], group_rem_ref[0] = new_cursor, new_group_id, new_group_rem

    if QUANTIZED:

        def decode_pipeline(lhs_ref, rhs_ref, rhs_scale_ref, out_ref):
            decode_pipeline_core(lhs_ref, rhs_ref, out_ref, rhs_scale_ref)

        in_specs = [
            pl.BlockSpec((TILE_M, K), lhs_index_map),
            pl.BlockSpec((TILE_G, K, TILE_N), rhs_index_map, pl.Buffered(RHS_BUFFER_COUNT)),
            pl.BlockSpec((TILE_G, None, 1, K_BLOCKS * TILE_N), rhs_scale_index_map),
        ]
        refs = [lhs_hbm, rhs_hbm, rhs_scale_hbm, out_hbm]

    else:

        def decode_pipeline(lhs_ref, rhs_ref, out_ref):
            decode_pipeline_core(lhs_ref, rhs_ref, out_ref)

        in_specs = [
            pl.BlockSpec((TILE_M, K), lhs_index_map),
            pl.BlockSpec((TILE_G, K, TILE_N), rhs_index_map, pl.Buffered(RHS_BUFFER_COUNT)),
        ]
        refs = [lhs_hbm, rhs_hbm, out_hbm]

    pipeline = pltpu.emit_pipeline(
        body=decode_pipeline,
        grid=(COL_TILES, GROUP_BATCHES, ROW_TILES),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((TILE_M, TILE_N), out_index_map),
        dimension_semantics=(PARALLEL, ARBITRARY, ARBITRARY),
    )

    pipeline(*refs)


def _decode_gemm(
    lhs: Array,
    rhs: Array | QArray,
    group_sizes: Array,
    config: DecodeGEMMConfig,
) -> Array:
    M, K = lhs.shape
    G, _, N = rhs.shape
    TILE_M, TILE_N, BLOCK_COMPUTE = _resolve_tiling(config, M, N)
    TILE_G = config.tile_g

    if config.fwd_dtype is not None:
        if not isinstance(rhs, QArray):
            raise ValueError("FP8 decode GEMM requires a block-scaled QArray RHS")
        require(
            rhs.block == (config.scale_k, config.scale_n),
            f"FP8 decode RHS block must be {(config.scale_k, config.scale_n)}, got {rhs.block}",
        )
        require(rhs.dtype == config.fwd_dtype, f"FP8 decode RHS dtype must be {config.fwd_dtype}, got {rhs.dtype}")
        expected_scale = (G, K // config.scale_k, N // config.scale_n)
        require(rhs.scale.shape == expected_scale, f"FP8 decode RHS scale must be {expected_scale}, got {rhs.scale.shape}")
        require(not is_scaled(rhs.stype), f"FP8 decode RHS scale dtype must be unscaled, got {rhs.stype}")
        rhs_value = rhs.qval
        rhs_scale = _pack_decode_rhs_scales(rhs.scale, scale_n=config.scale_n, tile_n=TILE_N)
    else:
        if isinstance(rhs, QArray):
            raise ValueError("dense decode GEMM config received a QArray RHS")
        rhs_value = rhs
        rhs_scale = jnp.ones((1,), dtype=jnp.float32)

    ROW_TILES = M // TILE_M

    lhs_row_map = _lhs_row_map(group_sizes, tile_g=TILE_G, tile_m=TILE_M, row_tiles=ROW_TILES)
    rhs_group_map = _rhs_group_map(group_sizes, tile_g=TILE_G)

    inputs = (lhs, rhs_value, rhs_scale)

    fwd = pl.pallas_call(
        kernel=partial(
            _decode_gemm_kernel,
            dims=(M, G, K, N),
            tiling=(TILE_M, TILE_N, TILE_G, BLOCK_COMPUTE),
            config=config,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), inputs),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.SMEM((1,), jnp.int32),  # row cursor
                pltpu.SMEM((1,), jnp.int32),  # group id
                pltpu.SMEM((1,), jnp.int32),  # group remainder
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((M, N), dtype=lhs.dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="DECODE_GEMM",
    )

    out = fwd(lhs_row_map, rhs_group_map, group_sizes.astype(jnp.int32), *inputs)

    return _A(out)


def decode_gemm_reference(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "G K N"] | QArray,
    group_sizes: Int32[Array, " G"],
    *,
    bias: Array | None = None,
    fused_swiglu: bool = False,
    swiglu_beta: float = 1.702,
    swiglu_limit: float = 7.0,
    quant_dtype: jnp.dtype | None = None,
) -> Array:

    if quant_dtype is not None and is_scaled(quant_dtype):
        config = DecodeGEMMConfig(fwd_dtype=quant_dtype)
        if isinstance(rhs, QArray):
            rhs_q = rhs
        else:
            block = (math.gcd(rhs.shape[-2], config.scale_k), math.gcd(rhs.shape[-1], config.scale_n))
            rhs_q = quantize(rhs, quant_dtype, block)
        qval, scale = lax.optimization_barrier((rhs_q.qval, rhs_q.scale))
        rhs = dequantize(QArray(qval, scale, rhs_q.block), lhs.dtype)

    out = grouped_gemm_reference(
        lhs,
        rhs,
        group_sizes,
        bias=bias,
        fused_swiglu=fused_swiglu,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        quant_dtype=None,
    )

    return out


def decode_gemm(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "G K N"] | QArray,
    group_sizes: Int32[Array, " G"],
    *,
    bias: Array | None = None,
    fused_swiglu: bool = False,
    swiglu_beta: float = 1.702,
    swiglu_limit: float = 7.0,
    quant_dtype: jnp.dtype | None = None,
    implementation: Implementation = "auto",
) -> Array:

    M, _ = lhs.shape
    G, K, N = rhs.shape
    dtype = quant_dtype if quant_dtype is not None and is_scaled(quant_dtype) else None
    config = select_decode_gemm_config(m=M, groups=G, k=K, n=N, quant_dtype=dtype)

    if dtype is not None:
        scale_config = config or DecodeGEMMConfig(fwd_dtype=dtype)
        if isinstance(rhs, QArray):
            rhs_q = rhs
        else:
            block = (math.gcd(rhs.shape[-2], scale_config.scale_k), math.gcd(rhs.shape[-1], scale_config.scale_n))
            rhs_q = quantize(rhs, dtype, block)
        qval, scale = lax.optimization_barrier((rhs_q.qval, rhs_q.scale))
        rhs_kernel: Array | QArray = QArray(qval, scale, rhs_q.block)
        rhs_reference = dequantize(rhs_kernel, lhs.dtype)
    elif isinstance(rhs, QArray):
        rhs_kernel = dequantize(rhs, lhs.dtype)
        rhs_reference = rhs_kernel
    else:
        rhs_kernel = rhs
        rhs_reference = rhs.astype(lhs.dtype)
    out_dtype = lhs.dtype

    def reference():
        return decode_gemm_reference(
            lhs,
            rhs_reference,
            group_sizes,
            bias=bias,
            fused_swiglu=fused_swiglu,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )

    def kernel():
        assert config is not None
        out = FP32(_decode_gemm(lhs, rhs_kernel, group_sizes, config))
        if bias is not None:
            ids = _expert_ids(group_sizes, lhs.shape[0])
            out += FP32(normalize_bias(bias, G, N, jnp.float32)[ids, 0, :])
        if fused_swiglu:
            out = swiglu(out, swiglu_beta, swiglu_limit)
        return out.astype(out_dtype)

    tiling = None if config is None else f"m{config.tile_m}/g{config.tile_g}/n{config.tile_n} dtype={config.fwd_dtype or 'bf16'}"
    op_shape = f"m{M} g{G} k{K} n{N}"

    out = dispatch_kernel(
        "decode_gemm",
        implementation=implementation,
        shape=op_shape,
        config=config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
    )

    return out
