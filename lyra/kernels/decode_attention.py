from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array, Ref, lax, named_scope
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.tpu import ARBITRARY, PARALLEL
from jax.tree_util import register_static
from jaxtyping import Float

from lyra.kernels.common import (
    BF16,
    LOG2_E,
    TILE_INNER,
    TILE_RHS_T,
    Implementation,
    current_tpu_target,
    dispatch_kernel,
    mxu_dot,
    tpu_vmem_limit_bytes,
)
from lyra.nn.quant import QArray, dequantize, is_scaled

_A = lambda x: cast(Array, x)

_DECODE_BLOCK = 128


@register_static
@dataclass(frozen=True)
class DecodeAttentionConfig:
    tile_kv: int = _DECODE_BLOCK
    kv_buffer_count: int = 2
    out_dtype: jnp.dtype = jnp.bfloat16
    cache_dtype: jnp.dtype | None = None


def select_decode_attention_config(
    head_dim: int,
    head_groups: int,
    kv_len: int,
    sliding_window: int,
    out_dtype: jnp.dtype,
    cache_dtype: jnp.dtype | None = None,
) -> DecodeAttentionConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    cache_dtype = cache_dtype if cache_dtype is not None and is_scaled(cache_dtype) else None
    config = DecodeAttentionConfig(out_dtype=out_dtype, cache_dtype=cache_dtype)
    if head_dim % 128:
        return None
    if head_groups <= 0 or config.kv_buffer_count < 1:
        return None
    if kv_len % config.tile_kv:
        return None
    return config


def decode_window(sliding_window: int, block: int = _DECODE_BLOCK) -> int:
    n_blocks = -(-(sliding_window + block - 1) // block)
    return n_blocks * block


def is_sw(sliding_window: int, kv_len: int, block: int = _DECODE_BLOCK) -> bool:
    return sliding_window > 0 and decode_window(sliding_window, block) < kv_len


def _decode_attention_kernel(
    # Prefetced
    start_block_ref: Ref,
    position_ref: Ref,
    # In refs
    q_ref: Ref,
    k_ref: Ref,
    k_scale_ref: Ref,
    v_ref: Ref,
    v_scale_ref: Ref,
    sinks_ref: Ref,
    # Out ref
    out_ref: Ref,
    # Scratch refs
    m_scratch: Array,
    l_scratch: Array,
    o_scratch: Array,
    *,
    sm_scale: float,
    sliding_window: int,
    tile_kv: int,
    quantized: bool,
):
    j = pl.program_id(2)
    j_programs = pl.num_programs(2)

    start_block = start_block_ref[0]
    position = position_ref[0]
    kv_base = (start_block + j) * tile_kv

    _init = j == 0
    _run = kv_base <= position
    _write = j == j_programs - 1

    @pl.when(_init)
    def _():
        m_scratch[...] = sinks_ref[...].astype(jnp.float32)
        l_scratch[...] = jnp.ones_like(l_scratch)
        o_scratch[...] = jnp.zeros_like(o_scratch)

    @pl.when(_run)
    def _():
        col = kv_base + lax.broadcasted_iota(jnp.int32, (1, tile_kv), 1)
        visible = col <= position
        if sliding_window > 0:
            visible = visible & (col > position - sliding_window)

        if quantized:
            k_tile = BF16(k_ref[...].astype(jnp.float32) * k_scale_ref[...].reshape(tile_kv, 1))
            v_tile = BF16(v_ref[...].astype(jnp.float32) * v_scale_ref[...].reshape(tile_kv, 1))
        else:
            k_tile = BF16(k_ref[...])
            v_tile = BF16(v_ref[...])

        qk = mxu_dot(BF16(q_ref[...]), k_tile, dimension_numbers=TILE_RHS_T)
        qk = jnp.where(visible, qk * sm_scale, -jnp.inf)

        m_prev = m_scratch[...]
        m_loc = jnp.max(qk, axis=-1, keepdims=True)
        m_curr = jnp.maximum(m_loc, m_prev)
        m_cf = jnp.exp(m_prev - m_curr)
        m_scratch[...] = m_curr

        a = lax.exp2((qk - m_curr) * LOG2_E)

        l_prev = l_scratch[...]
        l_curr = l_prev * m_cf + jnp.sum(a, axis=-1, keepdims=True)
        l_scratch[...] = l_curr

        o_prev = o_scratch[...]
        o_curr = o_prev * m_cf + mxu_dot(a, v_tile, dimension_numbers=TILE_INNER)
        o_scratch[...] = o_curr

    @pl.when(_write)
    def _():
        out_ref[...] = (o_scratch[...] * pltpu.reciprocal(l_scratch[...])).astype(out_ref.dtype)


def _decode_attention(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"] | QArray,
    v: Float[Array, "B K S H"] | QArray,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,
    config: DecodeAttentionConfig,
    dtype: jnp.dtype,
) -> Array:
    B, N, T, H = q.shape  # T == 1
    _, K, S, _ = k.shape
    HEAD_GROUPS = N // K
    TILE_KV = config.tile_kv
    QUANTIZED = config.cache_dtype is not None

    pos = jnp.asarray(position).astype(jnp.int32)

    if is_sw(sliding_window, S, TILE_KV):
        wlen = decode_window(sliding_window, TILE_KV)
        j_blocks = wlen // TILE_KV
        lo = pos - sliding_window + 1
        start = jnp.clip((lo // TILE_KV) * TILE_KV, 0, S - wlen)
        start_block = (start // TILE_KV).astype(jnp.int32).reshape(1)
    else:
        j_blocks = S // TILE_KV
        start_block = jnp.zeros((1,), jnp.int32)

    pos_vec = pos.reshape(1)
    q = q.reshape(B, K, HEAD_GROUPS, H)
    sinks = jnp.broadcast_to(sinks.reshape(K, HEAD_GROUPS)[None, :, :, None], (B, K, HEAD_GROUPS, 1)).astype(jnp.float32)

    def q_index_map(b, h, *_):
        return (b, h, 0, 0)

    def kv_index_map(b, h, j, start_block_ref, *_):
        return (b, h, start_block_ref[0] + j, 0)

    def kv_scale_index_map(b, h, j, start_block_ref, *_):
        return (b, h, start_block_ref[0] + j, 0, 0)

    def sinks_index_map(b, h, *_):
        return (b, h, 0, 0)

    def out_index_map(b, h, *_):
        return (b, h, 0, 0)

    if QUANTIZED:
        if not isinstance(k, QArray) or not isinstance(v, QArray):
            raise ValueError("quantized decode attention config requires QArray k/v")
        k_value, v_value = k.qval, v.qval
        k_scale = k.scale.reshape(B, K, S // TILE_KV, 1, TILE_KV)
        v_scale = v.scale.reshape(B, K, S // TILE_KV, 1, TILE_KV)
        k_scale_spec = pl.BlockSpec(
            (None, None, None, 1, TILE_KV),
            kv_scale_index_map,
            pl.Buffered(config.kv_buffer_count),
        )
        v_scale_spec = pl.BlockSpec(
            (None, None, None, 1, TILE_KV),
            kv_scale_index_map,
            pl.Buffered(config.kv_buffer_count),
        )
    else:
        if isinstance(k, QArray) or isinstance(v, QArray):
            raise ValueError("dense decode attention config received QArray k/v")
        k_value, v_value = k, v
        k_scale = v_scale = jnp.ones((1,), dtype=jnp.float32)
        k_scale_spec = v_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    fwd = pl.pallas_call(
        kernel=partial(
            _decode_attention_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            tile_kv=TILE_KV,
            quantized=QUANTIZED,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(B, K, j_blocks),
            in_specs=[
                pl.BlockSpec((None, None, HEAD_GROUPS, H), q_index_map),
                pl.BlockSpec((None, None, TILE_KV, H), kv_index_map, pl.Buffered(config.kv_buffer_count)),
                k_scale_spec,
                pl.BlockSpec((None, None, TILE_KV, H), kv_index_map, pl.Buffered(config.kv_buffer_count)),
                v_scale_spec,
                pl.BlockSpec((None, None, HEAD_GROUPS, 1), sinks_index_map),
            ],
            out_specs=pl.BlockSpec((None, None, HEAD_GROUPS, H), out_index_map),
            scratch_shapes=[
                pltpu.VMEM((HEAD_GROUPS, 1), jnp.float32),  # running max
                pltpu.VMEM((HEAD_GROUPS, 1), jnp.float32),  # running normalizer
                pltpu.VMEM((HEAD_GROUPS, H), jnp.float32),  # running output
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((B, K, HEAD_GROUPS, H), config.out_dtype),
        compiler_params=(
            pltpu.CompilerParams(
                dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY),
                vmem_limit_bytes=tpu_vmem_limit_bytes(),
            )
        ),
        name="DECODE_ATTENTION",
    )

    out = _A(fwd(start_block, pos_vec, q, k_value, k_scale, v_value, v_scale, sinks))
    out = out.reshape(B, N, T, H)  # this can probably be fused into the kernel at some point

    return out.astype(dtype)


def sliding_decode(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"],
    v: Float[Array, "B K S H"],
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,
    dtype: jnp.dtype,
) -> Array:
    B, N, T, H = q.shape
    _, K, S, _ = k.shape
    HEAD_GROUPS = N // K

    wlen = decode_window(sliding_window)
    pos = jnp.asarray(position).astype(jnp.int32)
    vis = pos - sliding_window + 1
    start = jnp.clip((vis // _DECODE_BLOCK) * _DECODE_BLOCK, 0, S - wlen)

    k_win = lax.dynamic_slice_in_dim(k, start, wlen, axis=2)
    v_win = lax.dynamic_slice_in_dim(v, start, wlen, axis=2)

    key_pos = start + jnp.arange(wlen, dtype=jnp.int32)
    visible = (key_pos <= pos) & (key_pos > pos - sliding_window)
    mask_ninf = jnp.asarray((-jnp.inf), dtype=dtype)
    mask = jnp.where(visible, 0.0, mask_ninf).astype(dtype)[None, :]

    with named_scope("QK"):
        q = q.astype(dtype).reshape(B, K, HEAD_GROUPS, T, H)
        qk = jnp.einsum("bkgth,bksh->bkgts", q, k_win.astype(dtype)) * sm_scale
        qk += mask

    with named_scope("Softmax"):
        sinks = sinks.reshape(K, HEAD_GROUPS)[None, :, :, None, None]
        sinks = jnp.broadcast_to(sinks, (B, K, HEAD_GROUPS, T, 1))
        virtual_max = lax.stop_gradient(jnp.maximum(jnp.max(qk, axis=-1, keepdims=True), sinks))
        sink_strength = jnp.exp(sinks - virtual_max)
        unnormalized = jnp.exp(qk - virtual_max)
        normalizer = jnp.reciprocal(jnp.sum(unnormalized, axis=-1, keepdims=True) + sink_strength)
        a = unnormalized * normalizer

    with named_scope("Out dot"):
        out = jnp.einsum("bkgts,bksh->bkgth", a, v_win.astype(dtype))
        out = out.astype(dtype).reshape(B, N, T, H)

    return out


def decode_attention_reference(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"] | QArray,
    v: Float[Array, "B K S H"] | QArray,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,
    dtype: jnp.dtype,
) -> Array:

    B, N, T, H = q.shape
    _, K, S, _ = k.shape
    HEAD_GROUPS = N // K
    dtype = dtype if dtype is not None else q.dtype

    if isinstance(k, QArray) or isinstance(v, QArray):
        assert isinstance(k, QArray) and isinstance(v, QArray)
        k = dequantize(k, jnp.bfloat16)
        v = dequantize(v, jnp.bfloat16)
    k, v = _A(k), _A(v)

    if is_sw(sliding_window, S):
        return sliding_decode(q, k, v, sinks, sm_scale, sliding_window, position, dtype=dtype)

    key_positions = jnp.arange(S)
    mask_ninf = jnp.asarray((-jnp.inf), dtype=dtype)
    visible = key_positions <= position
    if sliding_window > 0:
        visible &= key_positions > position - sliding_window
    mask = jnp.where(visible, 0.0, mask_ninf).astype(dtype)[None, :]

    with named_scope("QK"):
        q = q.astype(dtype).reshape(B, K, HEAD_GROUPS, T, H)
        qk = jnp.einsum("bkgth,bksh->bkgts", q, k.astype(dtype)) * sm_scale
        qk += mask

    with named_scope("Softmax"):
        sinks = sinks.reshape(K, HEAD_GROUPS)[None, :, :, None, None]
        sinks = jnp.broadcast_to(sinks, (B, K, HEAD_GROUPS, T, 1))
        virtual_max = lax.stop_gradient(jnp.maximum(jnp.max(qk, axis=-1, keepdims=True), sinks))
        sink_strength = jnp.exp(sinks - virtual_max)
        unnormalized = jnp.exp(qk - virtual_max)
        normalizer = jnp.reciprocal(jnp.sum(unnormalized, axis=-1, keepdims=True) + sink_strength)
        a = unnormalized * normalizer

    with named_scope("Out dot"):
        out = jnp.einsum("bkgts,bksh->bkgth", a, v.astype(dtype))
        out = out.astype(dtype).reshape(B, N, T, H)

    return out


def decode_attention(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"] | QArray,
    v: Float[Array, "B K S H"] | QArray,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,
    *,
    dtype: jnp.dtype | None = None,
    implementation: Implementation = "auto",
) -> Array:

    B, N, _, H = q.shape
    _, K, S, _ = k.shape
    HEAD_GROUPS = N // K
    dtype = dtype if dtype is not None else q.dtype
    cache_dtype = k.dtype if isinstance(k, QArray) else None

    config = select_decode_attention_config(
        head_dim=H,
        head_groups=HEAD_GROUPS,
        kv_len=S,
        sliding_window=sliding_window,
        out_dtype=jnp.dtype(dtype),
        cache_dtype=cache_dtype,
    )

    def reference():
        ref_k = dequantize(k, jnp.bfloat16) if isinstance(k, QArray) else k
        ref_v = dequantize(v, jnp.bfloat16) if isinstance(v, QArray) else v
        return decode_attention_reference(q, ref_k, ref_v, sinks, sm_scale, sliding_window, position, dtype)

    def kernel():
        assert config is not None
        return _decode_attention(q, k, v, sinks, sm_scale, sliding_window, position, config, dtype)

    tiling = None if config is None else f"kv{config.tile_kv} buf{config.kv_buffer_count}"
    op_shape = f"b{B} n{N} k{K} s{S} h{H} sw{sliding_window}"

    # The XLA reference was faster at every decode shape measured on v6e, so "auto" selects it.
    out = dispatch_kernel(
        "decode_attention",
        implementation=implementation,
        shape=op_shape,
        config=None if implementation == "auto" else config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
        fallback_reason="faster than Pallas for decode",
    )

    return out
