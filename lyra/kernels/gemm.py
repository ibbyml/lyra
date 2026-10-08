from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import Array, Ref, lax, named_scope
from jax.experimental import pallas as pl
from jax.experimental.pallas import MemoryRef
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.tpu import ARBITRARY, PARALLEL
from jax.tree_util import register_static
from jaxtyping import Float, Int32

from lyra.kernels.common import (
    BF16,
    FP32,
    TILE_INNER,
    TILE_LHS_T,
    TILE_RHS_T,
    Implementation,
    InputArray,
    InputRef,
    current_tpu_target,
    dispatch_kernel,
    log_kernel_resolution,
    mxu_dot,
    mxu_ragged_dot,
    swiglu,
    tpu_lanes,
    tpu_sublanes,
    tpu_vmem_limit_bytes,
)
from lyra.nn.quant import QArray, dequantize, is_scaled, maybe_dequant, quantize

_A = lambda x: cast(Array, x)
_TA = lambda xs: tuple(cast(Array, x) for x in xs)


@register_static
@dataclass(frozen=True)
class GroupedGEMMConfig:
    tile_m: int = 256
    tile_k: int = 256
    tile_n: int = 256

    tile_m_drhs: int = 256
    tile_k_drhs: int = 256
    tile_n_drhs: int = 256
    tile_m_dlhs: int = 256
    tile_k_dlhs: int = 256
    tile_n_dlhs: int = 256

    scale_k: int = 256
    scale_n: int = 256

    fwd_dtype: jnp.dtype | None = None
    bwd_dtype: jnp.dtype | None = None
    out_dtype: jnp.dtype = jnp.bfloat16

    fused_swiglu: bool = False
    swiglu_beta: float = 1.702
    swiglu_limit: float = 7.0

    rhs_buffer_count: int = 2
    rhs_buffer_count_dlhs: int = 2
    use_dlhs_kernel: bool = False
    zero_slack_rows: bool = False


_GEMM_CONFIGS = {
    (True, 32768, 8, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=2),
    (True, 65536, 8, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=2),
    (True, 262144, 8, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=3),
    (True, 32768, 16, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=2),
    (True, 65536, 16, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=2),
    (False, 32768, 8, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=2048, rhs_buffer_count=3),
    (False, 65536, 8, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=2048, rhs_buffer_count=2),
    (False, 262144, 8, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=2048, rhs_buffer_count=3),
    (False, 32768, 16, 2048, 2048): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, rhs_buffer_count=3),
    (False, 65536, 16, 2048, 2048): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, rhs_buffer_count=3),
    (True, 65536, 16, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (True, 131072, 16, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (False, 65536, 16, 1024, 2048): GroupedGEMMConfig(tile_m=4096, tile_k=1024, tile_n=2048, rhs_buffer_count=2),
    (False, 131072, 16, 1024, 2048): GroupedGEMMConfig(tile_m=4096, tile_k=1024, tile_n=2048, rhs_buffer_count=3),
    (True, 65536, 32, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (True, 131072, 32, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (False, 65536, 32, 1024, 2048): GroupedGEMMConfig(tile_m=4096, tile_k=1024, tile_n=2048, rhs_buffer_count=2),
    (False, 131072, 32, 1024, 2048): GroupedGEMMConfig(tile_m=4096, tile_k=1024, tile_n=2048, rhs_buffer_count=3),
    (True, 131072, 16, 2048, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=2048, tile_n=2048, fused_swiglu=True, rhs_buffer_count=2),
    (False, 131072, 16, 2048, 2048): GroupedGEMMConfig(tile_m=2048, tile_k=2048, tile_n=2048, rhs_buffer_count=3),
    (True, 65536, 16, 4096, 8192): GroupedGEMMConfig(tile_m=1024, tile_k=1024, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (True, 131072, 16, 4096, 8192): GroupedGEMMConfig(tile_m=1024, tile_k=1024, tile_n=1024, fused_swiglu=True, rhs_buffer_count=3),
    (False, 65536, 16, 4096, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=1024, tile_n=1024, rhs_buffer_count=3),
    (False, 131072, 16, 4096, 4096): GroupedGEMMConfig(tile_m=1024, tile_k=1024, tile_n=1024, rhs_buffer_count=3),
}

_GEMM_FP8_CONFIGS = {
    (True, 262144, 8, 2048, 4096): GroupedGEMMConfig(
        tile_m=1024, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, fused_swiglu=True, rhs_buffer_count=2
    ),
    (False, 262144, 8, 2048, 2048): GroupedGEMMConfig(
        tile_m=2048, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, rhs_buffer_count=2
    ),
    (True, 32768, 16, 2048, 4096): GroupedGEMMConfig(
        tile_m=1024, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, fused_swiglu=True, rhs_buffer_count=4
    ),
    (False, 32768, 16, 2048, 2048): GroupedGEMMConfig(
        tile_m=2048, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, rhs_buffer_count=2
    ),
    (True, 32768, 8, 2048, 4096): GroupedGEMMConfig(
        tile_m=1024, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, fused_swiglu=True, rhs_buffer_count=2
    ),
    (False, 32768, 8, 2048, 2048): GroupedGEMMConfig(
        tile_m=2048, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, rhs_buffer_count=2
    ),
    (True, 65536, 8, 2048, 4096): GroupedGEMMConfig(
        tile_m=1024, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, fused_swiglu=True, rhs_buffer_count=2
    ),
    (False, 65536, 8, 2048, 2048): GroupedGEMMConfig(
        tile_m=2048, tile_k=2048, tile_n=2048, fwd_dtype=jnp.float8_e4m3fn, rhs_buffer_count=2
    ),
}

_TGEMM_CONFIGS = {
    (32768, 8, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (32768, 8, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (65536, 8, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (65536, 8, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (131072, 8, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (131072, 8, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (262144, 8, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (262144, 8, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (32768, 16, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (32768, 16, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (65536, 16, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (65536, 16, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (65536, 16, 1024, 2048): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (131072, 16, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (131072, 16, 1024, 2048): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (65536, 32, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (65536, 32, 1024, 2048): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (131072, 32, 2048, 2048): GroupedGEMMConfig(tile_m_drhs=512, tile_k_drhs=2048, tile_n_drhs=2048),
    (131072, 32, 1024, 2048): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (131072, 16, 2048, 4096): GroupedGEMMConfig(tile_m_drhs=256, tile_k_drhs=2048, tile_n_drhs=4096),
    (65536, 16, 4096, 8192): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (131072, 16, 4096, 8192): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (65536, 16, 4096, 4096): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
    (131072, 16, 4096, 4096): GroupedGEMMConfig(tile_m_drhs=1024, tile_k_drhs=1024, tile_n_drhs=2048),
}

_DLHS_CONFIGS = {
    (32768, 8, 2048, 4096): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (32768, 8, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (65536, 8, 2048, 4096): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (65536, 8, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (32768, 16, 2048, 4096): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=4096, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (65536, 16, 2048, 4096): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=4096, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (32768, 16, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (65536, 16, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (65536, 16, 1024, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (131072, 16, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (131072, 16, 1024, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (65536, 32, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (65536, 32, 1024, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (131072, 32, 2048, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=2048, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (131072, 32, 1024, 2048): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=2048, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (131072, 16, 2048, 4096): GroupedGEMMConfig(
        tile_m_dlhs=2048, tile_k_dlhs=1024, tile_n_dlhs=4096, rhs_buffer_count_dlhs=3, use_dlhs_kernel=True
    ),
    (65536, 16, 4096, 8192): GroupedGEMMConfig(
        tile_m_dlhs=512, tile_k_dlhs=1024, tile_n_dlhs=8192, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (131072, 16, 4096, 8192): GroupedGEMMConfig(
        tile_m_dlhs=512, tile_k_dlhs=1024, tile_n_dlhs=8192, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (65536, 16, 4096, 4096): GroupedGEMMConfig(
        tile_m_dlhs=1024, tile_k_dlhs=2048, tile_n_dlhs=4096, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
    (131072, 16, 4096, 4096): GroupedGEMMConfig(
        tile_m_dlhs=1024, tile_k_dlhs=2048, tile_n_dlhs=4096, rhs_buffer_count_dlhs=2, use_dlhs_kernel=True
    ),
}


def grad_dtype(quant_dtype: jnp.dtype | None) -> jnp.dtype | None:
    if quant_dtype is None or not is_scaled(quant_dtype):
        return None
    return jnp.float8_e5m2


def select_grouped_gemm_config(
    m: int,
    groups: int,
    k: int,
    n: int,
    out_dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None,
    fused_swiglu: bool,
) -> GroupedGEMMConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    config = GroupedGEMMConfig()
    if target == "v6e":
        configs = _GEMM_FP8_CONFIGS if quant_dtype is not None else _GEMM_CONFIGS
        if tuned := configs.get((fused_swiglu, m, groups, k, n)):
            config = tuned
        if tuned := _TGEMM_CONFIGS.get((m, groups, k, n)):
            config = replace(
                config,
                tile_m_drhs=tuned.tile_m_drhs,
                tile_k_drhs=tuned.tile_k_drhs,
                tile_n_drhs=tuned.tile_n_drhs,
            )
        if quant_dtype is None and (tuned := _DLHS_CONFIGS.get((m, groups, k, n))):
            config = replace(
                config,
                tile_m_dlhs=tuned.tile_m_dlhs,
                tile_k_dlhs=tuned.tile_k_dlhs,
                tile_n_dlhs=tuned.tile_n_dlhs,
                rhs_buffer_count_dlhs=tuned.rhs_buffer_count_dlhs,
                use_dlhs_kernel=tuned.use_dlhs_kernel,
            )

    bwd_dtype = grad_dtype(quant_dtype)

    config = replace(
        config,
        fwd_dtype=quant_dtype,
        bwd_dtype=bwd_dtype,
        out_dtype=out_dtype,
        fused_swiglu=fused_swiglu,
    )

    out_n = n // 2 if fused_swiglu else n
    if m <= 0 or groups <= 0 or m % 128:
        return None
    if fused_swiglu and n % 2:
        return None
    if k % config.scale_k or n % config.scale_n:
        return None
    if config.tile_k % config.scale_k or config.tile_n % config.scale_n:
        return None
    if k % config.tile_k or out_n % config.tile_n:
        return None
    if config.tile_k_drhs % config.scale_k or config.tile_n_drhs % config.scale_n:
        return None
    if k % config.tile_k_drhs or n % config.tile_n_drhs:
        return None
    if config.use_dlhs_kernel:
        if config.tile_m_dlhs % 128:
            return None
        if k % config.tile_k_dlhs or n % config.tile_n_dlhs:
            return None
        if config.rhs_buffer_count_dlhs < 1:
            return None
    return config


def select_grouped_tgemm_config(
    m: int,
    groups: int,
    k: int,
    n: int,
    dtype: jnp.dtype,
    quant_dtype: jnp.dtype | None = None,
) -> GroupedGEMMConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    config = GroupedGEMMConfig()
    if target == "v6e" and (tuned := _TGEMM_CONFIGS.get((m, groups, k, n))):
        config = tuned
    config = replace(config, bwd_dtype=quant_dtype, out_dtype=dtype)
    if m <= 0 or groups <= 0 or m % 128:
        return None
    if k % config.scale_k or n % config.scale_n:
        return None
    if config.tile_k_drhs % config.scale_k or config.tile_n_drhs % config.scale_n:
        return None
    if k % config.tile_k_drhs or n % config.tile_n_drhs:
        return None
    return config


class MetadataShapes(NamedTuple):
    tile_groups: MemoryRef
    tile_offsets: MemoryRef


class Metadata(NamedTuple):
    tile_groups: Array | Ref
    tile_offsets: Array | Ref


def swiglu_vjp(
    pre_act: Array,
    grad: Array,
    beta: float = 1.702,
    limit: float = 7.0,
) -> Array:
    act = partial(swiglu, beta=beta, limit=limit)
    _, f_vjp = jax.vjp(act, pre_act)
    g = f_vjp(grad.astype(pre_act.dtype))[0]
    return g


def pack_rhs_scales(
    s: Array,
    *,
    scale_n: int,
    scales_per_k_tile: int,
    tile_n: int,
) -> Array:
    groups, k_scale_blocks, _ = s.shape
    if k_scale_blocks % scales_per_k_tile:
        raise ValueError(f"K scale blocks must divide compute K tiles: {k_scale_blocks=} {scales_per_k_tile=}")
    expanded = jnp.repeat(s, scale_n, axis=-1)
    n = expanded.shape[-1]
    if n % tile_n:
        raise ValueError(f"N={n} must be divisible by compute tile N={tile_n}")
    k_tiles = k_scale_blocks // scales_per_k_tile
    n_tiles = n // tile_n
    packed = expanded.reshape(groups, k_tiles, scales_per_k_tile, n_tiles, tile_n)
    packed = packed.transpose(0, 1, 3, 2, 4)
    return packed.reshape(groups * k_tiles * n_tiles, 1, scales_per_k_tile * tile_n)


def pack_tgemm_scales(
    s: Array,
    *,
    tile_features: int,
    scale_features: int,
    lanes: int,
    sublanes: int,
) -> Array:
    m, feature_scale_blocks = s.shape
    scales_per_tile = tile_features // scale_features
    if feature_scale_blocks % scales_per_tile:
        raise ValueError(f"feature scale blocks must divide compute tiles: {feature_scale_blocks=} {scales_per_tile=}")
    if scales_per_tile > lanes:
        raise ValueError(f"a compute tile has more scale blocks ({scales_per_tile}) than TPU lanes ({lanes})")
    if m % sublanes:
        raise ValueError(f"M={m} must be divisible by TPU sublanes={sublanes}")
    feature_tiles = feature_scale_blocks // scales_per_tile
    packed = s.reshape(m, feature_tiles, scales_per_tile).transpose(1, 0, 2)
    packed = jnp.pad(packed, ((0, 0), (0, 0), (0, lanes - scales_per_tile)))
    return packed.reshape(feature_tiles, m // sublanes, sublanes, lanes)


def pad_lhs_scale(s: Array, lanes: int) -> Array:
    blocks = s.shape[-1]
    if blocks > lanes:
        raise ValueError(f"K has more scale blocks ({blocks}) than TPU lanes ({lanes})")
    scale = jnp.pad(s, ((0, 0), (0, lanes - blocks)))
    return scale


def normalize_bias(bias: Array | None, groups: int, features: int, dtype: jnp.dtype) -> Array:
    if bias is None:
        return jnp.zeros((groups, 1, features), dtype=dtype)
    if bias.ndim == 2:
        bias = bias[:, None, :]
    if bias.shape != (groups, 1, features):
        raise ValueError(f"Unsupported bias shape: {bias.shape}")
    return bias.astype(dtype)


def fill_metadata(
    metadata: Metadata,
    dims: tuple[int, ...],
    tiling: tuple[int, ...],
    group_sizes: Array,
    dtype: jnp.dtype,
) -> int:
    _, G, _, _ = dims
    TILE_M = tiling[0]
    SUBLANES = tpu_sublanes(dtype)

    metadata.tile_offsets[0] = 0

    def outer_loop(group_idx, carry):
        group_size = group_sizes[group_idx]
        group_start, prev_tiles = carry
        group_end = group_start + group_size

        group_sublane_offset = group_start % SUBLANES
        aligned_group_size = group_size + group_sublane_offset

        aligned_tiles = pl.cdiv(aligned_group_size, TILE_M)
        is_compute_tile = jnp.logical_and(group_size > 0, group_idx >= 0)
        group_compute_tiles = jnp.where(is_compute_tile, aligned_tiles, 0)
        curr_tiles = prev_tiles + group_compute_tiles

        def inner_loop(tile_idx, tile_start):
            tile_sublane_offset = tile_start % SUBLANES
            block_offset = TILE_M - tile_sublane_offset
            local_offset = group_end - tile_start

            tile_offset = jnp.minimum(block_offset, local_offset)
            tile_end = tile_start + tile_offset

            metadata.tile_groups[tile_idx] = group_idx
            metadata.tile_offsets[tile_idx] = tile_start
            metadata.tile_offsets[tile_idx + 1] = tile_end

            return tile_end

        lax.fori_loop(prev_tiles, curr_tiles, inner_loop, group_start)
        return group_end, curr_tiles

    _, compute_tiles = lax.fori_loop(0, G, outer_loop, (0, 0))
    return compute_tiles


def _expert_ids(group_sizes: Array, total: int) -> Array:
    ids = jnp.repeat(
        jnp.arange(group_sizes.shape[0], dtype=jnp.int32),
        repeats=group_sizes,
        total_repeat_length=total,
    )
    return ids


def kernel_input(
    input: Array | QArray,
    dtype: jnp.dtype | None,
    block: tuple[int, ...],
) -> InputArray:
    if dtype is not None and is_scaled(dtype):
        _in = quantize(input, dtype, block)
    else:
        _in = input
    quantized = isinstance(_in, QArray)
    arr = InputArray(
        value=_in.qval if quantized else _in,
        scale=_in.scale if quantized else None,
        block=_in.block if quantized else None,
    )
    return arr


def gemm_core(
    lhs: Array,
    rhs: Array | QArray,
    group_sizes: Array,
    bias: Array | None = None,
    config: GroupedGEMMConfig | None = None,
) -> Array:
    dtype = lhs.dtype
    if config is not None and config.fwd_dtype is not None and is_scaled(config.fwd_dtype):
        lhs_q = quantize(lhs, config.fwd_dtype, (1, config.scale_k))
        rhs_q = quantize(rhs, config.fwd_dtype, (config.scale_k, config.scale_n))
        lhs_q, rhs_q = lax.optimization_barrier((lhs_q, rhs_q))
        lhs = dequantize(lhs_q, dtype)
        rhs = dequantize(rhs_q, dtype)
    rhs_f = maybe_dequant(rhs, dtype)
    scaled = config is not None and config.fwd_dtype is not None and is_scaled(config.fwd_dtype)
    recompute_dtype = jnp.float32 if scaled else dtype
    t = mxu_ragged_dot(lhs, rhs_f, group_sizes).astype(recompute_dtype)
    if bias is not None:
        G, _, N = rhs_f.shape
        ids = _expert_ids(group_sizes, lhs.shape[0])
        t += normalize_bias(bias, G, N, recompute_dtype)[ids, 0, :]
    return t.astype(dtype)


def group_bias_grad(grad: Array, group_sizes: Array, bias: Array | None) -> Array | None:
    if bias is None:
        return None
    expert_ids = _expert_ids(group_sizes, grad.shape[0])
    dbias = jax.ops.segment_sum(grad, expert_ids, num_segments=group_sizes.shape[0])
    if bias.ndim == 3:
        dbias = dbias[:, None, :]
    return dbias.astype(bias.dtype)


def _gemm_kernel_impl(
    group_sizes: Array,
    lhs_hbm: InputRef,
    rhs_hbm: InputRef,
    bias_hbm: Array,
    out_hbm: Array,
    partial_pipeline_scratch: Array,
    acc_pipeline_scratch: Array,
    metadata: Metadata,
    *,
    dims: tuple[int, ...],
    tiling: tuple[int, ...],
    has_bias: bool,
    config: GroupedGEMMConfig,
):
    _, _, K, N = dims
    TILE_M, TILE_K, TILE_N = tiling
    FUSED = config.fused_swiglu
    OUT_N = N // 2 if FUSED else N
    K_BLOCKS = K // TILE_K
    SCALE_K = config.scale_k
    SCALES_PER_K_TILE = TILE_K // SCALE_K

    lhs_quantized = lhs_hbm.scale is not None
    rhs_quantized = rhs_hbm.scale is not None

    dtype = lhs_hbm.value.dtype
    SUBLANES = tpu_sublanes(dtype)
    TILE_P = TILE_M // SUBLANES
    LANES = tpu_lanes()

    def gemm_pipeline_core(
        lhs_tile_ref: InputRef,
        rhs_tile_refs: tuple[InputRef, ...],
        bias_tile_refs: tuple[Ref, ...],
        out_tile_ref: Ref,
        partial_scratch: Array,
        acc_scratch: Array,
    ):
        j, k = pl.program_id(1), pl.program_id(2)

        def _matmul(is_first_k_step: bool, is_last_k_step: bool):
            lhs_tile = lhs_tile_ref.get_value().reshape(TILE_M, TILE_K)
            rhs_tiles = tuple(ref.get_value().reshape(TILE_K, TILE_N) for ref in rhs_tile_refs)

            if lhs_quantized or rhs_quantized:
                lhs_scales = lhs_tile_ref.get_scale().reshape(TILE_M, LANES) if lhs_quantized else None
                k_scale_iota = lax.broadcasted_iota(jnp.int32, (1, LANES), 1)
                rhs_scales = tuple(ref.get_scale().reshape(SCALES_PER_K_TILE, TILE_N) if rhs_quantized else None for ref in rhs_tile_refs)

                scaled_accs: list[Array | None] = [None] * len(rhs_tiles)
                for scale_idx in range(SCALES_PER_K_TILE):
                    start = scale_idx * SCALE_K
                    lhs_block = lhs_tile[:, start : start + SCALE_K]
                    rhs_blocks = tuple(rhs_tile[start : start + SCALE_K, :] for rhs_tile in rhs_tiles)
                    if lhs_block.dtype != rhs_blocks[0].dtype:
                        lhs_block = BF16(lhs_block)
                        rhs_blocks = tuple(BF16(rhs_block) for rhs_block in rhs_blocks)

                    lhs_s = None
                    if lhs_scales is not None:
                        global_scale_idx = k * SCALES_PER_K_TILE + scale_idx
                        lhs_s = jnp.sum(
                            lhs_scales * (k_scale_iota == global_scale_idx),
                            axis=-1,
                            keepdims=True,
                        )

                    for rhs_idx, (rhs_block, rhs_s) in enumerate(zip(rhs_blocks, rhs_scales)):
                        partial_acc = mxu_dot(lhs_block, rhs_block, dimension_numbers=TILE_INNER)
                        if lhs_s is not None:
                            partial_acc *= lhs_s
                        if rhs_s is not None:
                            partial_acc *= rhs_s[scale_idx][None, :]
                        previous = scaled_accs[rhs_idx]
                        scaled_accs[rhs_idx] = partial_acc if previous is None else previous + partial_acc

                accs = tuple(FP32(cast(Array, acc)) for acc in scaled_accs)
            else:
                accs = tuple(FP32(mxu_dot(lhs_tile, rhs_tile, dimension_numbers=TILE_INNER)) for rhs_tile in rhs_tiles)

            acc = jnp.concatenate(accs, axis=-1) if FUSED else accs[0]

            if not is_first_k_step:
                acc += acc_scratch[...]

            if is_last_k_step:
                if has_bias:
                    biases = tuple(ref[...].reshape(1, TILE_N) for ref in bias_tile_refs)
                    bias = jnp.concatenate(biases, axis=-1) if FUSED else biases[0]
                    acc += FP32(bias)
                if FUSED:
                    acc = swiglu(acc, config.swiglu_beta, config.swiglu_limit)

                tile_start = metadata.tile_offsets[j]
                tile_end = metadata.tile_offsets[j + 1]
                tile_offset = tile_start - (tile_start % SUBLANES)

                aligned_start = tile_start - tile_offset
                aligned_end = tile_end - tile_offset

                iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
                mask = jnp.logical_and(aligned_start <= iota, aligned_end > iota)
                out = jnp.where(mask, acc, 0)

                out = out.reshape(TILE_P, SUBLANES, TILE_N)
                out_tile_ref[...] = out.astype(out_tile_ref.dtype)

                zeros = jnp.zeros_like(partial_scratch)
                carry = jnp.where(j == 0, zeros, partial_scratch[...])
                out_tile_ref[0] += carry.astype(out_tile_ref.dtype)

                last_packet = aligned_end // SUBLANES
                ends_on_packet_boundary = (aligned_end % SUBLANES) == 0
                partial_scratch[...] = jnp.where(
                    ends_on_packet_boundary,
                    zeros,
                    FP32(out_tile_ref[last_packet]),
                )
            else:
                acc_scratch[...] = acc

        @jax.named_scope("matmul_full")  # Entire matmul in a single step (no inner k loop)
        def matmul_full():
            _matmul(is_first_k_step=True, is_last_k_step=True)

        @jax.named_scope("matmul_first")
        def matmul_first():
            _matmul(is_first_k_step=True, is_last_k_step=False)

        @jax.named_scope("matmul_middle")
        def matmul_middle():
            _matmul(is_first_k_step=False, is_last_k_step=False)

        @jax.named_scope("matmul_last")
        def matmul_last():
            _matmul(is_first_k_step=False, is_last_k_step=True)

        if K_BLOCKS == 1:
            matmul_full()
        else:
            is_first_k_step = k == 0
            is_last_k_step = k == (pl.num_programs(2) - 1)

            lax.cond(
                is_first_k_step,
                lambda: lax.cond(is_last_k_step, matmul_full, matmul_first),
                lambda: lax.cond(is_last_k_step, matmul_last, matmul_middle),
            )

    if FUSED:

        def fused_pipeline(
            lhs_tile_ref,
            rhs_gate_tile_ref,
            rhs_up_tile_ref,
            bias_gate_tile_ref,
            bias_up_tile_ref,
            out_tile_ref,
            partial_scratch,
            acc_scratch,
        ):
            gemm_pipeline_core(
                lhs_tile_ref,
                (rhs_gate_tile_ref, rhs_up_tile_ref),
                (bias_gate_tile_ref, bias_up_tile_ref),
                out_tile_ref,
                partial_scratch,
                acc_scratch,
            )

        pipeline_fn = fused_pipeline

    else:

        def base_pipeline(
            lhs_tile_ref,
            rhs_tile_ref,
            bias_tile_ref,
            out_tile_ref,
            partial_scratch,
            acc_scratch,
        ):
            gemm_pipeline_core(
                lhs_tile_ref,
                (rhs_tile_ref,),
                (bias_tile_ref,),
                out_tile_ref,
                partial_scratch,
                acc_scratch,
            )

        pipeline_fn = base_pipeline

    def lhs_index_map(i, j, k):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, k)

    def lhs_s_index_map(i, j, k):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, 0)

    def rhs_gate_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        return (tile_group, k, i)

    def rhs_up_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        return (tile_group, k, i + i_programs)

    def rhs_gate_s_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        scale_row = (tile_group * K_BLOCKS + k) * (N // TILE_N) + i
        return (scale_row, 0, 0)

    def rhs_up_s_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        scale_row = (tile_group * K_BLOCKS + k) * (N // TILE_N) + i + i_programs
        return (scale_row, 0, 0)

    def bias_gate_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        return (tile_group, 0, i)

    def bias_up_index_map(i, j, k):
        tile_group = metadata.tile_groups[j]
        return (tile_group, 0, i + i_programs)

    def out_index_map(i, j, k):
        last_tile = j == (pl.num_programs(1) - 1)
        packets_capped_end = metadata.tile_offsets[j + 1] // SUBLANES
        packets_bound_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = jnp.where(last_tile, packets_bound_end, packets_capped_end)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, i)

    lhs_specs = replace(
        lhs_hbm,
        value=pl.BlockSpec((pl.BoundedSlice(TILE_P), SUBLANES, TILE_K), lhs_index_map),
        scale=(pl.BlockSpec((pl.BoundedSlice(TILE_P), SUBLANES, LANES), lhs_s_index_map) if lhs_quantized else None),
    )
    rhs_gate_specs = replace(
        rhs_hbm,
        value=pl.BlockSpec(
            (None, TILE_K, TILE_N),
            rhs_gate_index_map,
            pl.Buffered(config.rhs_buffer_count),
        ),
        scale=(pl.BlockSpec((None, 1, SCALES_PER_K_TILE * TILE_N), rhs_gate_s_index_map) if rhs_quantized else None),
    )

    if FUSED:
        rhs_up_specs = replace(
            rhs_hbm,
            value=pl.BlockSpec(
                (None, TILE_K, TILE_N),
                rhs_up_index_map,
                pl.Buffered(config.rhs_buffer_count),
            ),
            scale=(pl.BlockSpec((None, 1, SCALES_PER_K_TILE * TILE_N), rhs_up_s_index_map) if rhs_quantized else None),
        )
        in_specs = (
            lhs_specs,
            rhs_gate_specs,
            rhs_up_specs,
            pl.BlockSpec((None, 1, TILE_N), bias_gate_index_map),
            pl.BlockSpec((None, 1, TILE_N), bias_up_index_map),
        )
        refs = [lhs_hbm, rhs_hbm, rhs_hbm, bias_hbm, bias_hbm]
        refs += [out_hbm]
    else:
        in_specs = (
            lhs_specs,
            rhs_gate_specs,
            pl.BlockSpec((None, 1, TILE_N), bias_gate_index_map),
        )
        refs = [lhs_hbm, rhs_hbm, bias_hbm, out_hbm]

    i_programs = OUT_N // TILE_N
    j_programs = fill_metadata(metadata, dims, tiling, group_sizes, dtype)
    k_programs = K // TILE_K

    pipeline = pltpu.emit_pipeline(
        body=pipeline_fn,
        grid=(i_programs, j_programs, k_programs),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((pl.BoundedSlice(TILE_P), SUBLANES, TILE_N), out_index_map),
        dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY),
    )
    scratches = [partial_pipeline_scratch, acc_pipeline_scratch]
    pipeline(*refs, scratches=scratches)  # pyright:ignore


def _gemm_kernel(
    group_sizes: Array,
    lhs_hbm: InputRef,
    rhs_hbm: InputRef,
    bias_hbm: Array,
    out_hbm: Array,
    partial_pipeline_scratch: Array,
    acc_pipeline_scratch: Array,
    metadata: Metadata,
    *,
    dims: tuple[int, ...],
    tiling: tuple[int, ...],
    has_bias: bool,
    config: GroupedGEMMConfig,
):
    _gemm_kernel_impl(
        group_sizes,
        lhs_hbm,
        rhs_hbm,
        bias_hbm,
        out_hbm,
        partial_pipeline_scratch,
        acc_pipeline_scratch,
        metadata,
        dims=dims,
        tiling=tiling,
        has_bias=has_bias,
        config=config,
    )


def _gemm(
    lhs: InputArray,
    rhs: InputArray,
    group_sizes: Array,
    bias: Array | None,
    config: GroupedGEMMConfig,
) -> Array:
    M, K = lhs.shape
    G, _, N = rhs.shape
    TILE_M, TILE_K, TILE_N = config.tile_m, config.tile_k, config.tile_n

    FUSED = config.fused_swiglu
    OUT_N = N // 2 if FUSED else N
    ACC_TILE_N = TILE_N * 2 if FUSED else TILE_N
    TILES_BOUND = G + pl.cdiv(M, TILE_M) - 1
    SUBLANES = tpu_sublanes(lhs.dtype)
    PACKETS = M // SUBLANES

    has_bias = bias is not None

    lhs = replace(
        lhs,
        value=lhs.value.reshape(PACKETS, SUBLANES, K),
        scale=pad_lhs_scale(lhs.get_scale(), tpu_lanes()).reshape(PACKETS, SUBLANES, -1) if lhs.scale is not None else None,
    )

    bias = normalize_bias(bias, G, N, jnp.float32)

    rhs_v, rhs_s = rhs.value, rhs.scale
    if rhs_s is not None:
        rhs_s = pack_rhs_scales(
            rhs_s,
            scale_n=config.scale_n,
            scales_per_k_tile=TILE_K // config.scale_k,
            tile_n=TILE_N,
        )

    rhs = replace(rhs, value=rhs_v, scale=rhs_s)
    inputs = (lhs, rhs, bias)
    acc_scratch_shape = (TILE_M, ACC_TILE_N)

    fwd = pl.pallas_call(
        kernel=partial(
            _gemm_kernel,
            dims=(M, G, K, N),
            tiling=(TILE_M, TILE_K, TILE_N),
            has_bias=has_bias,
            config=config,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), inputs),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((SUBLANES, TILE_N), dtype=jnp.float32),  # partials scratch
                pltpu.VMEM(acc_scratch_shape, dtype=jnp.float32),  # accumulator scratch
                MetadataShapes(
                    tile_groups=pltpu.SMEM((TILES_BOUND,), jnp.int32),
                    tile_offsets=pltpu.SMEM((TILES_BOUND + 1,), jnp.int32),
                ),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((PACKETS, SUBLANES, OUT_N), dtype=config.out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="GEMM",
    )

    out = fwd(group_sizes, *inputs)
    return _A(out).reshape(M, OUT_N)


def _gemm_dlhs_kernel(
    group_sizes: Array,
    grad_hbm: Array,
    rhs_hbm: Array,
    out_hbm: Array,
    partial_pipeline_scratch: Array,
    acc_pipeline_scratch: Array,
    metadata: Metadata,
    *,
    dims: tuple[int, ...],
    config: GroupedGEMMConfig,
):
    _, _, K, N = dims
    TILE_M = config.tile_m_dlhs
    TILE_K = config.tile_k_dlhs
    TILE_N = config.tile_n_dlhs
    N_BLOCKS = N // TILE_N

    dtype = grad_hbm.dtype
    SUBLANES = tpu_sublanes(dtype)
    TILE_P = TILE_M // SUBLANES

    def dlhs_pipeline(
        grad_tile_ref: Ref,
        rhs_tile_ref: Ref,
        out_tile_ref: Ref,
        partial_scratch: Array,
        acc_scratch: Array,
    ):
        j, i = pl.program_id(1), pl.program_id(2)

        def _matmul(is_first_n_step: bool, is_last_n_step: bool):
            grad_tile = BF16(grad_tile_ref[...].reshape(TILE_M, TILE_N))
            rhs_tile = BF16(rhs_tile_ref[...].reshape(TILE_K, TILE_N))
            acc = FP32(mxu_dot(grad_tile, rhs_tile, dimension_numbers=TILE_RHS_T))

            if not is_first_n_step:
                acc += acc_scratch[...]

            if is_last_n_step:
                tile_start = metadata.tile_offsets[j]
                tile_end = metadata.tile_offsets[j + 1]
                tile_offset = tile_start - (tile_start % SUBLANES)

                aligned_start = tile_start - tile_offset
                aligned_end = tile_end - tile_offset

                iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
                mask = jnp.logical_and(aligned_start <= iota, aligned_end > iota)
                out = jnp.where(mask, acc, 0)

                out_tile_ref[...] = out.reshape(TILE_P, SUBLANES, TILE_K).astype(out_tile_ref.dtype)

                zeros = jnp.zeros_like(partial_scratch)
                carry = jnp.where(j == 0, zeros, partial_scratch[...])
                out_tile_ref[0] += carry.astype(out_tile_ref.dtype)

                last_packet = aligned_end // SUBLANES
                ends_on_packet_boundary = (aligned_end % SUBLANES) == 0
                partial_scratch[...] = jnp.where(
                    ends_on_packet_boundary,
                    zeros,
                    FP32(out_tile_ref[last_packet]),
                )
            else:
                acc_scratch[...] = acc

        @jax.named_scope("matmul_full")
        def matmul_full():
            _matmul(is_first_n_step=True, is_last_n_step=True)

        @jax.named_scope("matmul_first")
        def matmul_first():
            _matmul(is_first_n_step=True, is_last_n_step=False)

        @jax.named_scope("matmul_middle")
        def matmul_middle():
            _matmul(is_first_n_step=False, is_last_n_step=False)

        @jax.named_scope("matmul_last")
        def matmul_last():
            _matmul(is_first_n_step=False, is_last_n_step=True)

        if N_BLOCKS == 1:
            matmul_full()
        else:
            is_first_n_step = i == 0
            is_last_n_step = i == (pl.num_programs(2) - 1)
            lax.cond(
                is_first_n_step,
                lambda: lax.cond(is_last_n_step, matmul_full, matmul_first),
                lambda: lax.cond(is_last_n_step, matmul_last, matmul_middle),
            )

    def grad_index_map(k, j, i):
        del k
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, i)

    def rhs_index_map(k, j, i):
        tile_group = metadata.tile_groups[j]
        return (tile_group, k, i)

    def out_index_map(k, j, i):
        del i
        last_tile = j == (pl.num_programs(1) - 1)
        packets_capped_end = metadata.tile_offsets[j + 1] // SUBLANES
        packets_bound_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = jnp.where(last_tile, packets_bound_end, packets_capped_end)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, k)

    k_programs = K // TILE_K
    ragged_programs = fill_metadata(
        metadata,
        dims,
        (TILE_M, TILE_K, TILE_N),
        group_sizes,
        dtype,
    )
    n_programs = N // TILE_N

    pipeline = pltpu.emit_pipeline(
        body=dlhs_pipeline,
        grid=(k_programs, ragged_programs, n_programs),
        in_specs=(
            pl.BlockSpec(
                (pl.BoundedSlice(TILE_P), SUBLANES, TILE_N),
                grad_index_map,
            ),
            pl.BlockSpec(
                (None, TILE_K, TILE_N),
                rhs_index_map,
                pl.Buffered(config.rhs_buffer_count_dlhs),
            ),
        ),
        out_specs=pl.BlockSpec(
            (pl.BoundedSlice(TILE_P), SUBLANES, TILE_K),
            out_index_map,
        ),
        dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY),
    )

    refs = [grad_hbm, rhs_hbm, out_hbm]
    scratches = [partial_pipeline_scratch, acc_pipeline_scratch]
    pipeline(*refs, scratches=scratches)  # pyright: ignore


def _gemm_dlhs(
    grad: Array,
    rhs: Array,
    group_sizes: Array,
    config: GroupedGEMMConfig,
) -> Array:
    M, N = grad.shape
    G, K, _ = rhs.shape
    TILE_M = config.tile_m_dlhs
    TILE_K = config.tile_k_dlhs

    SUBLANES = tpu_sublanes(grad.dtype)

    PACKETS = M // SUBLANES
    TILES_BOUND = G + pl.cdiv(M, TILE_M) - 1
    grad_packets = grad.reshape(PACKETS, SUBLANES, N)
    inputs = (grad_packets, rhs)

    dlhs = pl.pallas_call(
        kernel=partial(
            _gemm_dlhs_kernel,
            dims=(M, G, K, N),
            config=config,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), inputs),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((SUBLANES, TILE_K), dtype=jnp.float32),
                pltpu.VMEM((TILE_M, TILE_K), dtype=jnp.float32),
                MetadataShapes(
                    tile_groups=pltpu.SMEM((TILES_BOUND,), jnp.int32),
                    tile_offsets=pltpu.SMEM((TILES_BOUND + 1,), jnp.int32),
                ),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((PACKETS, SUBLANES, K), dtype=grad.dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="GEMM_DLHS",
    )

    out = _A(dlhs(group_sizes, *inputs)).reshape(M, K)
    if config.zero_slack_rows:
        # The pipeline only writes tiles covering the live rows (sum of group
        # sizes), so slack receive-backing rows past them keep uninitialized HBM.
        # Rows are group-contiguous from 0, so zero the tail. Gated because the
        # masking pass reads and rewrites all [M, K]; only the EP dropless path,
        # where M > sum(group_sizes), needs it.
        live_rows = group_sizes.sum()
        out = jnp.where(lax.broadcasted_iota(jnp.int32, (M, K), 0) < live_rows, out, 0)
    return out


def zero_empty_tgemm_groups(
    group_sizes: Array,
    out_ref: Array,
    zero_ref: Array,
    semaphore_ref: Array,
):
    G, K, N = out_ref.shape
    TILE_ZERO_K = zero_ref.shape[0]
    LANES = tpu_lanes()

    zero_ref = zero_ref.reshape(1, TILE_ZERO_K, LANES)
    zero_ref[...] = jnp.zeros_like(zero_ref)

    num_groups_to_zero = jnp.array(0, dtype=jnp.int32)
    for group_idx in range(G):
        should_copy = group_sizes[group_idx] == 0
        should_copy_int = should_copy.astype(jnp.int32)
        num_groups_to_zero += should_copy_int

        for k_tile in range(pl.cdiv(K, TILE_ZERO_K)):
            k_start = k_tile * TILE_ZERO_K
            k_size = min(TILE_ZERO_K, K - k_start)

            for n_start in range(0, N, LANES):
                src = zero_ref.at[pl.ds(0, should_copy_int), pl.ds(0, k_size)]
                dst = out_ref.at[
                    pl.ds(group_idx, should_copy_int),
                    pl.ds(k_start, k_size),
                    pl.ds(n_start, LANES),
                ]
                pltpu.make_async_copy(
                    src_ref=src,
                    dst_ref=dst,
                    sem=semaphore_ref.at[0],
                ).start(priority=1)

    return num_groups_to_zero


def wait_empty_tgemm_groups(
    num_groups_to_zero: Array,
    out_ref: Array,
    semaphore_ref: Array,
):
    dst = out_ref.at[pl.ds(0, num_groups_to_zero)]
    pltpu.make_async_copy(
        src_ref=dst,
        dst_ref=dst,
        sem=semaphore_ref.at[0],
    ).wait()


def _tgemm_kernel(
    group_sizes: Array,
    lhs_hbm_ref: InputRef,
    rhs_hbm_ref: InputRef,
    out_hbm_ref: Array,
    acc_scratch: Array,
    metadata: Metadata,
    zero_scratch: Array,
    zero_semaphore: Array,
    *,
    dims: tuple[int, ...],
    config: GroupedGEMMConfig,
):
    _, _, K, N = dims
    TILE_M, TILE_K, TILE_N = config.tile_m_drhs, config.tile_k_drhs, config.tile_n_drhs
    tiling = (TILE_M, TILE_K, TILE_N)
    dtype = lhs_hbm_ref.value.dtype
    SCALE_K = config.scale_k
    SCALE_N = config.scale_n
    lhs_quantized = lhs_hbm_ref.scale is not None
    rhs_quantized = rhs_hbm_ref.scale is not None

    SUBLANES = tpu_sublanes(dtype)
    TILE_P = TILE_M // SUBLANES
    LANES = tpu_lanes()
    LHS_SCALE_BLOCKS = TILE_K // SCALE_K
    RHS_SCALE_BLOCKS = TILE_N // SCALE_N

    TILES = fill_metadata(metadata, dims, tiling, group_sizes, dtype)
    ZERO_GROUPS = zero_empty_tgemm_groups(group_sizes, out_hbm_ref, zero_scratch, zero_semaphore)

    def tgemm_pipeline(
        lhs_tile_ref: InputRef,
        rhs_tile_ref: InputRef,
        out_tile_ref,
        acc_ref,
    ):
        j = pl.program_id(2)

        tile_start = metadata.tile_offsets[j]
        tile_end = metadata.tile_offsets[j + 1]
        tile_offset = tile_start - (tile_start % SUBLANES)

        aligned_start = tile_start - tile_offset
        aligned_end = tile_end - tile_offset

        lhs_tile = lhs_tile_ref.value.reshape(-1, TILE_K)[...]
        if lhs_quantized:
            lhs_s = lhs_tile_ref.get_scale().reshape(-1, LANES)[:, :LHS_SCALE_BLOCKS]
            lhs_s = jnp.repeat(lhs_s, SCALE_K, axis=-1)
            lhs_tile = FP32(lhs_tile) * lhs_s
        lhs_tile = BF16(lhs_tile)
        lhs_iota = lax.broadcasted_iota(jnp.int32, lhs_tile.shape, 0)
        lhs_mask = jnp.logical_and(aligned_start <= lhs_iota, lhs_iota < aligned_end)
        lhs_tile = jnp.where(lhs_mask, lhs_tile[...], 0)

        rhs_tile = rhs_tile_ref.value.reshape(-1, TILE_N)[...]
        if rhs_quantized:
            rhs_s = rhs_tile_ref.get_scale().reshape(-1, LANES)[:, :RHS_SCALE_BLOCKS]
            rhs_s = jnp.repeat(rhs_s, SCALE_N, axis=-1)
            rhs_tile = FP32(rhs_tile) * rhs_s
        rhs_tile = BF16(rhs_tile)
        rhs_iota = lax.broadcasted_iota(jnp.int32, rhs_tile.shape, 0)
        rhs_mask = jnp.logical_and(aligned_start <= rhs_iota, rhs_iota < aligned_end)
        rhs_tile = jnp.where(rhs_mask, rhs_tile[...], 0)

        def _matmul(is_new_group: bool, is_group_changing: bool):
            acc = mxu_dot(lhs_tile, rhs_tile, dimension_numbers=TILE_LHS_T)

            if not is_new_group:
                acc += acc_ref[...]

            if is_group_changing:
                out_tile_ref[...] = acc.astype(out_tile_ref.dtype)
            else:
                acc_ref[...] = acc

        @jax.named_scope("matmul_new_group_and_changing")
        def matmul_new_group_and_changing():
            _matmul(is_new_group=True, is_group_changing=True)

        @jax.named_scope("matmul_new_group")
        def matmul_new_group():
            _matmul(is_new_group=True, is_group_changing=False)

        @jax.named_scope("matmul")
        def matmul():
            _matmul(is_new_group=False, is_group_changing=False)

        @jax.named_scope("matmul_group_changing")
        def matmul_group_changing():
            _matmul(is_new_group=False, is_group_changing=True)

        prev_j = jnp.where(j > 0, j - 1, 0)
        is_first_j = j == 0
        group_changed = metadata.tile_groups[j] != metadata.tile_groups[prev_j]
        new_group = jnp.logical_or(is_first_j, group_changed)

        last_j = j == (pl.num_programs(2) - 1)
        next_j = jnp.where(last_j, j, j + 1)
        next_group = metadata.tile_groups[next_j]
        curr_group = metadata.tile_groups[j]
        group_is_changing = jnp.logical_or(last_j, curr_group != next_group)

        lax.cond(
            new_group,
            lambda: lax.cond(
                group_is_changing,
                matmul_new_group_and_changing,
                matmul_new_group,
            ),
            lambda: lax.cond(
                group_is_changing,
                matmul_group_changing,
                matmul,
            ),
        )

    def lhs_index_map(i, k, j):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, k)

    def rhs_index_map(i, k, j):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        packets_width = packets_end - packets_start
        return (pl.ds(packets_start, packets_width), 0, i)

    def lhs_s_index_map(i, k, j):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        return (k, pl.ds(packets_start, packets_end - packets_start), 0, 0)

    def rhs_s_index_map(i, k, j):
        packets_start = metadata.tile_offsets[j] // SUBLANES
        packets_end = pl.cdiv(metadata.tile_offsets[j + 1], SUBLANES)
        return (i, pl.ds(packets_start, packets_end - packets_start), 0, 0)

    def out_index_map(i, k, j):
        tile_group = metadata.tile_groups[j]
        return (tile_group, k, i)

    i_programs = N // TILE_N
    j_programs = K // TILE_K
    k_programs = TILES

    pipeline = pltpu.emit_pipeline(
        body=tgemm_pipeline,
        grid=(i_programs, j_programs, k_programs),
        in_specs=(
            replace(
                lhs_hbm_ref,
                value=pl.BlockSpec((pl.BoundedSlice(TILE_P), SUBLANES, TILE_K), lhs_index_map),
                scale=(
                    pl.BlockSpec(
                        (None, pl.BoundedSlice(TILE_P), SUBLANES, LANES),
                        lhs_s_index_map,
                    )
                    if lhs_quantized
                    else None
                ),
            ),
            replace(
                rhs_hbm_ref,
                value=pl.BlockSpec((pl.BoundedSlice(TILE_P), SUBLANES, TILE_N), rhs_index_map),
                scale=(
                    pl.BlockSpec(
                        (None, pl.BoundedSlice(TILE_P), SUBLANES, LANES),
                        rhs_s_index_map,
                    )
                    if rhs_quantized
                    else None
                ),
            ),
        ),
        out_specs=pl.BlockSpec((None, TILE_K, TILE_N), out_index_map),
    )

    refs = [lhs_hbm_ref, rhs_hbm_ref, out_hbm_ref]
    scratches = [acc_scratch]
    pipeline(*refs, scratches=scratches)  # pyright:ignore
    wait_empty_tgemm_groups(ZERO_GROUPS, out_hbm_ref, zero_semaphore)


def _tgemm(
    lhs: Array | QArray,
    rhs: Array | QArray,
    group_sizes: Array,
    config: GroupedGEMMConfig,
) -> Array:
    M, K = lhs.shape
    _, N = rhs.shape
    G = group_sizes.shape[0]
    TILE_M, TILE_K, TILE_N = config.tile_m_drhs, config.tile_k_drhs, config.tile_n_drhs
    lhs_input = kernel_input(lhs, config.bwd_dtype, (1, config.scale_k))
    rhs_input = kernel_input(rhs, config.bwd_dtype, (1, config.scale_n))

    SUBLANES = tpu_sublanes(lhs_input.dtype)
    LANES = tpu_lanes()

    TILES_BOUND = G + pl.cdiv(M, TILE_M) - 1

    target_zero_ref_bytes = 2 * 1024 * 1024
    TILE_ZERO_K = target_zero_ref_bytes // LANES // jnp.dtype(jnp.float32).itemsize
    TILE_ZERO_K = min(TILE_ZERO_K, K)
    TILE_ZERO_K = max(SUBLANES, (TILE_ZERO_K // SUBLANES) * SUBLANES)

    PACKETS = M // SUBLANES
    lhs_input = replace(
        lhs_input,
        value=lhs_input.value.reshape(PACKETS, SUBLANES, K),
        scale=(
            pack_tgemm_scales(
                lhs_input.scale,
                tile_features=TILE_K,
                scale_features=config.scale_k,
                lanes=LANES,
                sublanes=SUBLANES,
            )
            if lhs_input.scale is not None
            else None
        ),
    )
    rhs_input = replace(
        rhs_input,
        value=rhs_input.value.reshape(PACKETS, SUBLANES, N),
        scale=(
            pack_tgemm_scales(
                rhs_input.scale,
                tile_features=TILE_N,
                scale_features=config.scale_n,
                lanes=LANES,
                sublanes=SUBLANES,
            )
            if rhs_input.scale is not None
            else None
        ),
    )
    inputs = (lhs_input, rhs_input)

    bwd = pl.pallas_call(
        kernel=partial(
            _tgemm_kernel,
            dims=(M, G, K, N),
            config=config,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), inputs),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((TILE_K, TILE_N), dtype=jnp.float32),
                MetadataShapes(
                    tile_groups=pltpu.SMEM((TILES_BOUND,), jnp.int32),
                    tile_offsets=pltpu.SMEM((TILES_BOUND + 1,), jnp.int32),
                ),
                pltpu.VMEM((TILE_ZERO_K, LANES), dtype=jnp.float32),
                pltpu.SemaphoreType.DMA((1,)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((G, K, N), dtype=jnp.float32),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            disable_bounds_checks=True,
        ),
        name="TGEMM",
    )

    return bwd(group_sizes, *inputs)


def _f(
    lhs: Array,
    rhs: Array | QArray,
    group_sizes: Array,
    bias: Array | None,
    config: GroupedGEMMConfig,
) -> Array:
    out, _ = _f_fwd(lhs, rhs, group_sizes, bias, config)
    return out


def _f_fwd(
    lhs: Array,
    rhs: Array | QArray,
    group_sizes: Array,
    bias: Array | None,
    config: GroupedGEMMConfig,
):
    out = _gemm(
        lhs=kernel_input(lhs, config.fwd_dtype, (1, config.scale_k)),
        rhs=kernel_input(rhs, config.fwd_dtype, (config.scale_k, config.scale_n)),
        group_sizes=group_sizes,
        bias=bias,
        config=config,
    )
    bwd_ctx = (lhs, rhs, group_sizes, bias)
    return out, bwd_ctx


def _f_bwd(
    config: GroupedGEMMConfig,
    ctx: tuple,
    grad: Array,
):
    lhs, rhs, group_sizes, bias = _TA(ctx)

    if config.fused_swiglu:
        pre_act_out = gemm_core(lhs, rhs, group_sizes, bias, config)
        grad = swiglu_vjp(
            pre_act_out,
            grad,
            config.swiglu_beta,
            config.swiglu_limit,
        )

    grad = grad.astype(lhs.dtype)
    rhs_f = maybe_dequant(rhs, lhs.dtype)

    proj = "up" if config.fused_swiglu else "down"
    op_shape = f"m{lhs.shape[0]} g{group_sizes.shape[0]} k{lhs.shape[1]} n{grad.shape[1]}"

    if config.use_dlhs_kernel and not isinstance(rhs, QArray) and rhs.dtype == jnp.float32:
        log_kernel_resolution(
            f"grouped_gemm.dlhs.{proj}",
            shape=op_shape,
            impl="pallas",
            tiling=f"m{config.tile_m_dlhs}/k{config.tile_k_dlhs}/n{config.tile_n_dlhs} buf{config.rhs_buffer_count_dlhs}",
        )
        dlhs = _gemm_dlhs(
            grad=grad,
            rhs=rhs,
            group_sizes=group_sizes,
            config=config,
        )
    else:
        log_kernel_resolution(f"grouped_gemm.dlhs.{proj}", shape=op_shape, impl="xla")
        dlhs = _gemm_dlhs_reference(grad, rhs, group_sizes)

    log_kernel_resolution(
        f"grouped_gemm.tgemm.{proj}",
        shape=op_shape,
        impl="pallas",
        tiling=f"m{config.tile_m_drhs}/k{config.tile_k_drhs}/n{config.tile_n_drhs}",
    )
    drhs = _tgemm(lhs=lhs, rhs=grad, group_sizes=group_sizes, config=config).astype(rhs_f.dtype)

    dbias = group_bias_grad(grad, group_sizes, bias)

    return dlhs, drhs, None, dbias


_op = jax.custom_vjp(_f, nondiff_argnames=("config",))
_op.defvjp(_f_fwd, _f_bwd, optimize_remat=True)


def _gemm_dlhs_reference(
    grad: Array,
    rhs: Array | QArray,
    group_sizes: Array,
) -> Array:
    rhs_f = maybe_dequant(rhs, grad.dtype)
    rhs_t = BF16(rhs_f).swapaxes(1, 2)
    return _A(mxu_ragged_dot(BF16(grad), rhs_t, group_sizes)).astype(grad.dtype)


def grouped_tgemm_reference(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "M N"],
    group_sizes: Int32[Array, " G"],
    *,
    quant_dtype: jnp.dtype | None = None,
) -> Array:

    if quant_dtype is not None and is_scaled(quant_dtype):
        config = GroupedGEMMConfig(bwd_dtype=quant_dtype)
        lhs_block = (1, math.gcd(lhs.shape[-1], config.scale_k))
        rhs_block = (1, math.gcd(rhs.shape[-1], config.scale_n))
        lhs_q = quantize(lhs, quant_dtype, lhs_block)
        rhs_q = quantize(rhs, quant_dtype, rhs_block)
        # prevents xla from optimizing quantization away
        lhs_q, rhs_q = lax.optimization_barrier((lhs_q, rhs_q))
        lhs = dequantize(lhs_q, jnp.bfloat16)
        rhs = dequantize(rhs_q, jnp.bfloat16)

    expert_ids = _expert_ids(group_sizes, lhs.shape[0])

    def group_dot(group: Array) -> Array:
        lhs_f = lhs.astype(jnp.float32)
        rhs_group = jnp.where(expert_ids[:, None] == group, rhs, 0).astype(jnp.float32)
        return lax.dot(lhs_f, rhs_group, dimension_numbers=TILE_LHS_T)

    groups = jnp.arange(group_sizes.shape[0], dtype=jnp.int32)
    with named_scope("Grouped dot"):
        out = jax.vmap(group_dot)(groups)
    return out


def grouped_gemm_reference(
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
    fwd_dtype = quant_dtype if quant_dtype is not None and is_scaled(quant_dtype) else None
    config = GroupedGEMMConfig(
        fwd_dtype=fwd_dtype,
        out_dtype=lhs.dtype,
        fused_swiglu=fused_swiglu,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )
    if config.fwd_dtype is not None:
        lhs_block = (1, math.gcd(lhs.shape[-1], config.scale_k))
        rhs_block = (math.gcd(rhs.shape[-2], config.scale_k), math.gcd(rhs.shape[-1], config.scale_n))
        lhs_q = quantize(lhs, config.fwd_dtype, lhs_block)
        rhs_q = quantize(rhs, config.fwd_dtype, rhs_block)
        # Keep quantization visible to XLA.
        lhs_q, rhs_q = lax.optimization_barrier((lhs_q, rhs_q))
        lhs_f = dequantize(lhs_q, jnp.bfloat16)
        rhs_f = dequantize(rhs_q, jnp.bfloat16)
    else:
        lhs_f = lhs
        rhs_f = maybe_dequant(rhs, lhs.dtype)
    with named_scope("Ragged dot"):
        t = _A(mxu_ragged_dot(lhs_f, rhs_f, group_sizes))
    if bias is not None:
        with named_scope("Bias"):
            G, _, N = rhs_f.shape
            ids = _expert_ids(group_sizes, lhs.shape[0])
            t += FP32(normalize_bias(bias, G, N, jnp.float32)[ids, 0, :])
    if config.fused_swiglu:
        with named_scope("SwiGLU"):
            t = swiglu(t, config.swiglu_beta, config.swiglu_limit)

    return t.astype(lhs.dtype)


def grouped_tgemm(
    lhs: Float[Array, "M K"],
    rhs: Float[Array, "M N"],
    group_sizes: Int32[Array, " G"],
    *,
    quant_dtype: jnp.dtype | None = None,
    implementation: Implementation = "auto",
) -> Array:

    M, K = lhs.shape
    _, N = rhs.shape
    G = group_sizes.shape[0]
    dtype = quant_dtype if quant_dtype is not None and is_scaled(quant_dtype) else None

    config = select_grouped_tgemm_config(
        m=M,
        groups=G,
        k=K,
        n=N,
        dtype=lhs.dtype,
        quant_dtype=dtype,
    )

    def reference():
        return grouped_tgemm_reference(lhs, rhs, group_sizes, quant_dtype=dtype)

    def kernel():
        assert config is not None
        return _tgemm(lhs, rhs, group_sizes, config)

    tiling = None if config is None else f"m{config.tile_m_drhs}/k{config.tile_k_drhs}/n{config.tile_n_drhs}"
    op_shape = f"m{M} g{G} k{K} n{N}"

    out = dispatch_kernel(
        "grouped_tgemm",
        implementation=implementation,
        shape=op_shape,
        config=config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
    )

    return out


def grouped_gemm(
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

    M, K = lhs.shape
    G, _, N = rhs.shape
    dtype = quant_dtype if quant_dtype is not None and is_scaled(quant_dtype) else None

    config = select_grouped_gemm_config(
        m=M,
        groups=G,
        k=K,
        n=N,
        out_dtype=lhs.dtype,
        quant_dtype=dtype,
        fused_swiglu=fused_swiglu,
    )

    def reference():
        return grouped_gemm_reference(
            lhs,
            rhs,
            group_sizes,
            bias=bias,
            fused_swiglu=fused_swiglu,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            quant_dtype=dtype,
        )

    def kernel():
        assert config is not None
        return _A(_op(lhs, rhs, group_sizes, bias, config=config))

    proj = "up" if fused_swiglu else "down"
    tiling = None if config is None else f"m{config.tile_m}/k{config.tile_k}/n{config.tile_n} buf{config.rhs_buffer_count}"
    op_shape = f"m{M} g{G} k{K} n{N}"

    out = dispatch_kernel(
        f"grouped_gemm.fwd.{proj}",
        implementation=implementation,
        shape=op_shape,
        config=config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
    )

    return out
