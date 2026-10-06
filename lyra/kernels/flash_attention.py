from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Ref, lax, named_scope
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.tpu import ARBITRARY, PARALLEL
from jax.tree_util import register_static
from jaxtyping import Array, Float

from lyra.kernels.common import (
    BF16,
    FP32,
    LOG2_E,
    TILE_INNER,
    TILE_LHS_T,
    TILE_RHS_T,
    Implementation,
    current_tpu_target,
    dispatch_kernel,
    mxu_dot,
    require,
    tpu_vmem_limit_bytes,
)

_A = lambda x: cast(Array, x)
_TA = lambda xs: tuple(cast(Array, x) for x in xs)


@register_static
@dataclass(frozen=True)
class FlashAttentionConfig:
    tile_q: int = 128
    tile_kv: int = 128

    tile_q_dq: int = 128
    tile_kv_dq: int = 128
    tile_q_dkv: int = 128
    tile_kv_dkv: int = 128

    out_dtype: jnp.dtype = jnp.bfloat16

    banded_fwd: bool = False
    banded_dq: bool = False
    banded_dkv: bool = False
    clamp_causal_prefetch: bool = False
    kv_major_dkv: bool = False
    heads_per_program_fwd: int = 1

    buffer_count: int = 2


_ATTENTION_CONFIGS = {
    (4096, 128, 0): FlashAttentionConfig(
        tile_q=1024,
        tile_kv=1024,
        tile_q_dq=1024,
        tile_kv_dq=1024,
        tile_q_dkv=1024,
        tile_kv_dkv=1024,
    ),
    (4096, 128, 128): FlashAttentionConfig(
        tile_q=512,
        tile_kv=1024,
        tile_q_dq=512,
        tile_kv_dq=512,
        tile_q_dkv=256,
        tile_kv_dkv=512,
        banded_fwd=True,
        banded_dq=True,
        banded_dkv=True,
    ),
}


def select_flash_attention_config(
    q_tokens: int,
    kv_tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    sliding_window: int,
    out_dtype: jnp.dtype,
    *,
    batch_size: int | None = None,
) -> FlashAttentionConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    if q_tokens != kv_tokens:
        return None
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        return None
    if head_dim % 128:
        return None

    config = FlashAttentionConfig(out_dtype=out_dtype)
    if target == "v6e" and (tuned := _ATTENTION_CONFIGS.get((q_tokens, head_dim, sliding_window))):
        config = replace(tuned, out_dtype=out_dtype)
        if (batch_size, q_heads, kv_heads, q_tokens, head_dim, sliding_window) == (4, 16, 4, 4096, 128, 128):
            config = replace(config, kv_major_dkv=jnp.dtype(out_dtype) == jnp.bfloat16)
        if (
            (batch_size, q_heads, kv_heads, q_tokens, head_dim) == (4, 16, 4, 4096, 128)
            and sliding_window in (0, 128)
            and jnp.dtype(out_dtype) == jnp.bfloat16
            and jax.device_count() == 1
        ):
            config = replace(
                config,
                heads_per_program_fwd=4,
                clamp_causal_prefetch=sliding_window == 0,
            )
    if config.buffer_count < 1:
        return None

    legal = (
        q_tokens % config.tile_q == 0
        and kv_tokens % config.tile_kv == 0
        and q_tokens % config.tile_q_dq == 0
        and kv_tokens % config.tile_kv_dq == 0
        and q_tokens % config.tile_q_dkv == 0
        and kv_tokens % config.tile_kv_dkv == 0
    )
    return config if legal else None


def _gqa_q(h: int, head_groups: int, g: int):
    return h * head_groups + g


def _gqa_kv(h: int, head_groups: int):
    return h // head_groups


def _is_compute_tile(q_idx, kv_idx, tile_q, tile_kv, sliding_window) -> Array:
    q_start = q_idx * tile_q
    kv_start = kv_idx * tile_kv

    causal_condition = (q_start + tile_q - 1) >= kv_start

    if sliding_window > 0:
        bandwidth = q_start - (sliding_window - 1)
        sw_condition = (kv_start + tile_kv - 1) >= bandwidth
        return jnp.logical_and(causal_condition, sw_condition)

    return causal_condition


def _banded_kv_index(q_idx, local_kv_idx, tile_q, tile_kv, sliding_window):
    q_start = q_idx * tile_q
    first_kv = jnp.maximum(0, (q_start - sliding_window + 1) // tile_kv)
    return first_kv + local_kv_idx


def _banded_kv_programs(q_tokens: int, tile_q: int, tile_kv: int, sliding_window: int) -> int:
    programs = 0
    for q_idx in range(q_tokens // tile_q):
        q_start = q_idx * tile_q
        first_kv = max(0, (q_start - sliding_window + 1) // tile_kv)
        last_kv = (q_start + tile_q - 1) // tile_kv
        programs = max(programs, last_kv - first_kv + 1)
    return programs


def _banded_q_index(kv_idx, local_q_idx, tile_q, tile_kv):
    first_q = (kv_idx * tile_kv) // tile_q
    return first_q + local_q_idx


def _clamp_causal_prefetch_index(q_idx, kv_idx, tile_q, tile_kv):
    last_valid_kv = (q_idx * tile_q + tile_q - 1) // tile_kv
    return jnp.minimum(kv_idx, last_valid_kv)


def _banded_q_programs(q_tokens: int, kv_tokens: int, tile_q: int, tile_kv: int, sliding_window: int) -> int:
    programs = 0
    q_programs = q_tokens // tile_q
    for kv_idx in range(kv_tokens // tile_kv):
        kv_start = kv_idx * tile_kv
        first_q = kv_start // tile_q
        last_q = min(q_programs - 1, (kv_start + tile_kv + sliding_window - 2) // tile_q)
        programs = max(programs, last_q - first_q + 1)
    return programs


def _tile_mask(q_idx, kv_idx, tile_q, tile_kv, sliding_window):
    q_start = q_idx * tile_q
    kv_start = kv_idx * tile_kv

    row = lax.broadcasted_iota(jnp.int32, (tile_q, 1), dimension=0)
    col = lax.broadcasted_iota(jnp.int32, (1, tile_kv), dimension=1)

    row_pos = row + q_start
    col_pos = col + kv_start

    mask = row_pos < col_pos

    if sliding_window > 0:
        sw_mask = col_pos < (row_pos - sliding_window + 1)
        mask = jnp.logical_or(mask, sw_mask)

    return _A(mask)


def _flash_attention_kernel(
    q_tile_ref: Ref,
    k_tile_ref: Ref,
    v_tile_ref: Ref,
    sinks_ref: Ref,
    out_ref: Ref,
    lse_ref: Ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    sm_scale: float,
    sliding_window: int,
    banded: bool,
    kv_programs: int,
    heads_per_program: int,
):
    h, i, u = pl.program_id(0), pl.program_id(1), pl.program_id(2)  # u: local kv-band slot
    j_programs = pl.num_programs(2)
    TILE_Q, TILE_KV = q_tile_ref.shape[-2], k_tile_ref.shape[0]
    rows = heads_per_program * TILE_Q

    j = _banded_kv_index(i, u, TILE_Q, TILE_KV, sliding_window) if banded else u

    _init = u == 0
    _run = jnp.logical_and(j < kv_programs, _is_compute_tile(i, j, TILE_Q, TILE_KV, sliding_window))
    _write = u == (j_programs - 1)

    @pl.when(_init)
    def _():
        if heads_per_program == 1:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sinks_ref[h])
        else:
            sink_values = jnp.stack([sinks_ref[h * heads_per_program + g] for g in range(heads_per_program)])
            m_scratch_ref[...] = jnp.broadcast_to(sink_values[:, None, None], (heads_per_program, TILE_Q, 1)).reshape(rows, 1)
        l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

    @pl.when(_run)
    def _():
        mask = _tile_mask(i, j, TILE_Q, TILE_KV, sliding_window)
        if heads_per_program > 1:
            mask = jnp.tile(mask, (heads_per_program, 1))

        qk_tile = mxu_dot(
            BF16(q_tile_ref[...].reshape(rows, q_tile_ref.shape[-1])) if heads_per_program > 1 else BF16(q_tile_ref[...]),
            BF16(k_tile_ref[...]),
            dimension_numbers=TILE_RHS_T,
        )

        qk_tile *= sm_scale
        qk_masked = jnp.where(mask, -jnp.inf, qk_tile)

        m_prev = m_scratch_ref[...]
        m_loc = jnp.max(qk_masked, axis=-1, keepdims=True)
        m_curr = jnp.maximum(m_loc, m_prev)
        m_cf = jnp.exp(m_prev - m_curr)
        m_scratch_ref[...] = FP32(m_curr)

        a_tile = lax.exp2((qk_masked - m_curr) * LOG2_E)

        l_prev = _A(l_scratch_ref[...])
        l_loc = jnp.sum(a_tile, axis=-1, keepdims=True)
        l_curr = l_prev * m_cf + l_loc
        l_scratch_ref[...] = FP32(l_curr)

        o_prev = _A(o_scratch_ref[...])
        o_loc = mxu_dot(a_tile, BF16(v_tile_ref[...]), dimension_numbers=TILE_INNER)
        o_curr = o_prev * m_cf + o_loc
        o_scratch_ref[...] = FP32(o_curr)

    @pl.when(_write)
    def _():
        normalizer = _A(l_scratch_ref[...])
        if heads_per_program == 1:
            out_ref[...] = (o_scratch_ref[...] * pltpu.reciprocal(normalizer)).astype(out_ref.dtype)
            lse_ref[...] = (m_scratch_ref[...] + jnp.log2(normalizer) / LOG2_E).astype(lse_ref.dtype).T
        else:
            output = (o_scratch_ref[...] * pltpu.reciprocal(normalizer)).astype(out_ref.dtype)
            lse = (m_scratch_ref[...] + jnp.log2(normalizer) / LOG2_E).astype(lse_ref.dtype)
            out_ref[...] = output.reshape(heads_per_program, TILE_Q, q_tile_ref.shape[-1])
            lse_ref[...] = lse.reshape(heads_per_program, TILE_Q, 1).transpose(0, 2, 1)


def _flash_attention(
    q: Array,
    k: Array,
    v: Array,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
):
    _, N, T, H = q.shape
    _, K, S, _ = k.shape
    TILE_Q, TILE_KV = config.tile_q, config.tile_kv
    HEAD_GROUPS = N // K
    heads_per_program = config.heads_per_program_fwd
    require(heads_per_program > 0 and HEAD_GROUPS % heads_per_program == 0, "forward head group must divide query heads per K/V head")

    i_programs = N // heads_per_program
    j_programs = T // TILE_Q
    total_kv_programs = S // TILE_KV
    k_programs = _banded_kv_programs(T, TILE_Q, TILE_KV, sliding_window) if config.banded_fwd else total_kv_programs

    def q_index_map(h, i, _):
        return (h, i, 0)

    def kv_index_map(h, i, u):
        kv_j = _banded_kv_index(i, u, TILE_Q, TILE_KV, sliding_window) if config.banded_fwd else u
        if config.clamp_causal_prefetch and sliding_window == 0 and not config.banded_fwd:
            kv_j = _clamp_causal_prefetch_index(i, kv_j, TILE_Q, TILE_KV)
        kv_j = jnp.minimum(kv_j, total_kv_programs - 1)
        return (_gqa_kv(h * heads_per_program, HEAD_GROUPS), kv_j, 0)

    def out_index_map(h, i, _):
        return (h, i, 0)

    def lse_index_map(h, i, _):
        return (h, 0, i)

    fwd = pl.pallas_call(
        kernel=partial(
            _flash_attention_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            banded=config.banded_fwd,
            kv_programs=total_kv_programs,
            heads_per_program=heads_per_program,
        ),
        grid=(i_programs, j_programs, k_programs),
        in_specs=[
            pl.BlockSpec(
                (None if heads_per_program == 1 else heads_per_program, TILE_Q, H), q_index_map, pl.Buffered(config.buffer_count)
            ),  # q spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # k spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # v spec
            pl.BlockSpec(memory_space=pltpu.SMEM),  # sinks spec
        ],
        out_specs=[
            pl.BlockSpec((None if heads_per_program == 1 else heads_per_program, TILE_Q, H), out_index_map),  # attn out spec
            pl.BlockSpec((None if heads_per_program == 1 else heads_per_program, 1, TILE_Q), lse_index_map),  # lse spec
        ],
        out_shape=(
            jax.ShapeDtypeStruct((N, T, H), config.out_dtype),  # attn out shape
            jax.ShapeDtypeStruct((N, 1, T), jnp.float32),  # lse shape (stored this way for efficient padding)
        ),
        scratch_shapes=(
            pltpu.VMEM((heads_per_program * TILE_Q, 1), dtype=jnp.float32),  # max scratch
            pltpu.VMEM((heads_per_program * TILE_Q, 1), dtype=jnp.float32),  # normalizer scratch
            pltpu.VMEM((heads_per_program * TILE_Q, H), dtype=jnp.float32),  # output scratch
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={
                "XLA_TPU_FORCE_LP_LLO_SCHEDULER": True  # slightly improves performance but experimental.
            },
        ),
        name=("FLASH_ATTENTION_GQA_FORWARD_BANDED" if config.banded_fwd else "FLASH_ATTENTION_GQA_FORWARD")
        if heads_per_program > 1
        else ("FLASH_ATTENTION_BANDED" if config.banded_fwd else "FLASH_ATTENTION"),
    )

    out, lse = jax.vmap(fwd, in_axes=(0, 0, 0, None))(q, k, v, sinks)

    return out, lse


def _flash_attention_bwd_dq_kernel(
    q_tile_ref: Ref,
    k_tile_ref: Ref,
    v_tile_ref: Ref,
    o_tile_ref: Ref,
    lse_tile_ref: Ref,
    g_tile_ref: Ref,
    dq_ref: Ref,
    dq_scratch,
    *,
    sm_scale: float,
    sliding_window: int,
    banded: bool,
    kv_programs: int,
):
    i, u = pl.program_id(1), pl.program_id(2)  # u: local kv-band slot
    j_programs = pl.num_programs(2)
    TILE_Q, TILE_KV = q_tile_ref.shape[0], k_tile_ref.shape[0]

    j = _banded_kv_index(i, u, TILE_Q, TILE_KV, sliding_window) if banded else u

    _init = u == 0
    _run = jnp.logical_and(j < kv_programs, _is_compute_tile(i, j, TILE_Q, TILE_KV, sliding_window))
    _write = u == (j_programs - 1)

    @pl.when(_init)
    def _():
        dq_scratch[...] = jnp.zeros_like(dq_scratch)

    @pl.when(_run)
    def _():
        mask = _tile_mask(i, j, TILE_Q, TILE_KV, sliding_window)
        k_tile = BF16(k_tile_ref[...])

        qk_tile = mxu_dot(BF16(q_tile_ref[...]), k_tile, dimension_numbers=TILE_RHS_T) * sm_scale
        qk_masked = jnp.where(mask, -jnp.inf, qk_tile)
        p_tile = lax.exp2((qk_masked - lse_tile_ref[...].T) * LOG2_E)
        a_tile = jnp.where(mask, 0.0, p_tile)

        do_tile = g_tile_ref[...]
        di_tile = jnp.sum(o_tile_ref[...] * do_tile, axis=-1, keepdims=True)
        da_tile = mxu_dot(do_tile, BF16(v_tile_ref[...]), dimension_numbers=TILE_RHS_T)
        dqk_tile = jnp.where(mask, 0.0, a_tile * (da_tile - di_tile))

        dq_scratch[...] += mxu_dot(dqk_tile, k_tile)

    @pl.when(_write)
    def _():
        dq_ref[...] = (dq_scratch[...] * sm_scale).astype(dq_ref.dtype)


def _flash_attention_bwd_dq(
    ctx: tuple,
    grad: Array,
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
):
    q, k, v, _, o, lse = _TA(ctx)
    _, N, T, H = q.shape
    _, K, S, _ = k.shape
    TILE_Q, TILE_KV = config.tile_q_dq, config.tile_kv_dq
    HEAD_GROUPS = N // K

    i_programs = N
    j_programs = T // TILE_Q
    total_kv_programs = S // TILE_KV
    k_programs = _banded_kv_programs(T, TILE_Q, TILE_KV, sliding_window) if config.banded_dq else total_kv_programs

    def q_index_map(h, i, _):
        return (h, i, 0)

    def kv_index_map(h, i, u):
        kv_j = _banded_kv_index(i, u, TILE_Q, TILE_KV, sliding_window) if config.banded_dq else u
        if config.clamp_causal_prefetch and sliding_window == 0 and not config.banded_dq:
            kv_j = _clamp_causal_prefetch_index(i, kv_j, TILE_Q, TILE_KV)
        kv_j = jnp.minimum(kv_j, total_kv_programs - 1)
        return (_gqa_kv(h, HEAD_GROUPS), kv_j, 0)

    def o_index_map(h, i, _):
        return (h, i, 0)

    def lse_index_map(h, i, _):
        return (h, 0, i)

    def g_index_map(h, i, _):
        return (h, i, 0)

    def dq_out_index_map(h, i, _):
        return (h, i, 0)

    bwd = pl.pallas_call(
        kernel=partial(
            _flash_attention_bwd_dq_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            banded=config.banded_dq,
            kv_programs=total_kv_programs,
        ),
        grid=(i_programs, j_programs, k_programs),
        in_specs=[
            pl.BlockSpec((None, TILE_Q, H), q_index_map, pl.Buffered(config.buffer_count)),  # q spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # k spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # v spec
            pl.BlockSpec((None, TILE_Q, H), o_index_map, pl.Buffered(config.buffer_count)),  # o spec
            pl.BlockSpec((None, 1, TILE_Q), lse_index_map, pl.Buffered(config.buffer_count)),  # lse spec
            pl.BlockSpec((None, TILE_Q, H), g_index_map, pl.Buffered(config.buffer_count)),  # g spec
        ],
        out_specs=[
            pl.BlockSpec((None, TILE_Q, H), dq_out_index_map),  # dq spec
        ],
        out_shape=(
            jax.ShapeDtypeStruct((N, T, H), q.dtype),  # dq shape,
        ),
        scratch_shapes=(pltpu.VMEM((TILE_Q, H), dtype=jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="FLASH_ATTENTION_BWD_DQ_BANDED" if config.banded_dq else "FLASH_ATTENTION_BWD_DQ",
    )

    (dq,) = jax.vmap(bwd, in_axes=(0, 0, 0, 0, 0, 0))(q, k, v, o, lse, grad)

    return _A(dq)


def _flash_attention_bwd_dkv_kernel(
    q_tile_ref: Ref,
    k_tile_ref: Ref,
    v_tile_ref: Ref,
    o_tile_ref: Ref,
    lse_tile_ref: Ref,
    g_tile_ref: Ref,
    dk_ref: Ref,
    dv_ref: Ref,
    dk_scratch,
    dv_scratch,
    *,
    sm_scale: float,
    sliding_window: int,
    banded: bool,
    q_programs: int,
    kv_major: bool,
):
    j, v, g = pl.program_id(1), pl.program_id(2), pl.program_id(3)  # v: local q-band slot
    k_programs, g_programs = pl.num_programs(2), pl.num_programs(3)
    TILE_KV, TILE_Q = k_tile_ref.shape[0], q_tile_ref.shape[0]
    k = _banded_q_index(j, v, TILE_Q, TILE_KV) if banded else v

    _init = jnp.logical_and((v == 0), (g == 0))
    _run = jnp.logical_and(k < q_programs, _is_compute_tile(k, j, TILE_Q, TILE_KV, sliding_window))
    _write = jnp.logical_and((v == (k_programs - 1)), (g == (g_programs - 1)))

    @pl.when(_init)
    def _():
        dk_scratch[...] = jnp.zeros_like(dk_scratch)
        dv_scratch[...] = jnp.zeros_like(dv_scratch)

    @pl.when(_run)
    def _():
        mask = _tile_mask(k, j, TILE_Q, TILE_KV, sliding_window)
        q_tile = BF16(q_tile_ref[...])
        k_tile = BF16(k_tile_ref[...])
        v_tile = BF16(v_tile_ref[...])
        do_tile = g_tile_ref[...]
        di_tile = jnp.sum(o_tile_ref[...] * do_tile, axis=-1, keepdims=True)

        if kv_major:
            # [KV,Q] scores feed dK/dV directly, avoiding score-tile transposes.
            qk_tile = mxu_dot(k_tile, q_tile, dimension_numbers=TILE_RHS_T) * sm_scale
            da_tile = mxu_dot(v_tile, do_tile, dimension_numbers=TILE_RHS_T)
            mask = mask.T
            lse_tile = lse_tile_ref[...]
            di_tile = di_tile.T
        else:
            qk_tile = mxu_dot(q_tile, k_tile, dimension_numbers=TILE_RHS_T) * sm_scale
            da_tile = mxu_dot(do_tile, v_tile, dimension_numbers=TILE_RHS_T)
            lse_tile = lse_tile_ref[...].T
        qk_masked = jnp.where(mask, -jnp.inf, qk_tile)
        p_tile = lax.exp2((qk_masked - lse_tile) * LOG2_E)
        a_tile = jnp.where(mask, 0.0, p_tile)
        dqk_tile = jnp.where(mask, 0.0, a_tile * (da_tile - di_tile))

        dimensions = TILE_INNER if kv_major else TILE_LHS_T
        dk_scratch[...] += mxu_dot(dqk_tile, q_tile, dimension_numbers=dimensions)
        dv_scratch[...] += mxu_dot(a_tile, do_tile, dimension_numbers=dimensions)

    @pl.when(_write)
    def _():
        dk_ref[...] = (dk_scratch[...] * sm_scale).astype(dk_ref.dtype)
        dv_ref[...] = dv_scratch[...].astype(dv_ref.dtype)


def _flash_attention_bwd_dkv(
    ctx: tuple,
    grad: Array,
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
):
    q, k, v, _, o, lse = _TA(ctx)
    _, K, S, H = k.shape
    _, N, T, _ = q.shape
    TILE_KV, TILE_Q = config.tile_kv_dkv, config.tile_q_dkv
    HEAD_GROUPS = N // K

    i_programs = K
    j_programs = S // TILE_KV
    total_q_programs = T // TILE_Q
    k_programs = _banded_q_programs(T, S, TILE_Q, TILE_KV, sliding_window) if config.banded_dkv else total_q_programs
    l_programs = HEAD_GROUPS

    def q_index_map(h, i, u, g):
        q_j = _banded_q_index(i, u, TILE_Q, TILE_KV) if config.banded_dkv else u
        q_j = jnp.minimum(q_j, total_q_programs - 1)
        return (_gqa_q(h, HEAD_GROUPS, g), q_j, 0)

    def kv_index_map(h, i, *_):
        return (h, i, 0)

    def lse_index_map(h, i, u, g):
        q_j = _banded_q_index(i, u, TILE_Q, TILE_KV) if config.banded_dkv else u
        q_j = jnp.minimum(q_j, total_q_programs - 1)
        return (_gqa_q(h, HEAD_GROUPS, g), 0, q_j)

    def dk_out_index_map(h, i, *_):
        return (h, i, 0)

    def dv_out_index_map(h, i, *_):
        return (h, i, 0)

    bwd = pl.pallas_call(
        kernel=partial(
            _flash_attention_bwd_dkv_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            banded=config.banded_dkv,
            q_programs=total_q_programs,
            kv_major=config.kv_major_dkv,
        ),
        grid=(i_programs, j_programs, k_programs, l_programs),
        in_specs=[
            pl.BlockSpec((None, TILE_Q, H), q_index_map, pl.Buffered(config.buffer_count)),  # q spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # k spec
            pl.BlockSpec((None, TILE_KV, H), kv_index_map, pl.Buffered(config.buffer_count)),  # v spec
            pl.BlockSpec((None, TILE_Q, H), q_index_map, pl.Buffered(config.buffer_count)),  # o spec
            pl.BlockSpec((None, 1, TILE_Q), lse_index_map, pl.Buffered(config.buffer_count)),  # lse spec
            pl.BlockSpec((None, TILE_Q, H), q_index_map, pl.Buffered(config.buffer_count)),  # g spec
        ],
        out_specs=[
            pl.BlockSpec((None, TILE_KV, H), dk_out_index_map),  # dk spec
            pl.BlockSpec((None, TILE_KV, H), dv_out_index_map),  # dv spec
        ],
        out_shape=(
            jax.ShapeDtypeStruct((K, S, H), k.dtype),  # dk shape
            jax.ShapeDtypeStruct((K, S, H), v.dtype),  # dv shape
        ),
        scratch_shapes=(
            pltpu.VMEM((TILE_KV, H), dtype=jnp.float32),  # dk tile scratch
            pltpu.VMEM((TILE_KV, H), dtype=jnp.float32),  # dv tile scratch
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, PARALLEL, ARBITRARY, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="FLASH_ATTENTION_BWD_DKV_BANDED" if config.banded_dkv else "FLASH_ATTENTION_BWD_DKV",
    )

    dk, dv = jax.vmap(bwd, in_axes=(0, 0, 0, 0, 0, 0))(q, k, v, o, lse, grad)

    return dk, dv


def _flash_attention_bwd_dsinks(ctx, grad):
    _, _, _, sinks, o, lse = _TA(ctx)
    dtype = sinks.dtype
    di = jnp.sum(o * grad, axis=-1, dtype=jnp.float32)
    sink_prob = lax.exp2((sinks[None, :, None] - lse.squeeze(-2)) * LOG2_E)
    dsinks = jnp.sum(-sink_prob * di, axis=(0, 2), dtype=jnp.float32)
    return dsinks.astype(dtype)


def _f(
    q: Array,
    k: Array,
    v: Array,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
):
    out, _ = _flash_attention(q, k, v, sinks, sm_scale, sliding_window, config)
    return out


def _f_fwd(
    q: Array,
    k: Array,
    v: Array,
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
):
    out, lse = _flash_attention(q, k, v, sinks, sm_scale, sliding_window, config)
    bwd_ctx = (q, k, v, sinks, out, lse)
    return out, bwd_ctx


def _f_bwd(
    sm_scale: float,
    sliding_window: int,
    config: FlashAttentionConfig,
    ctx: tuple,
    grad: Array,
):
    dq = _flash_attention_bwd_dq(ctx, grad, sm_scale, sliding_window, config)
    dk, dv = _flash_attention_bwd_dkv(ctx, grad, sm_scale, sliding_window, config)
    dsinks = _flash_attention_bwd_dsinks(ctx, grad)
    return dq, dk, dv, dsinks


_op = jax.custom_vjp(_f, nondiff_argnames=("sm_scale", "sliding_window", "config"))
_op.defvjp(_f_fwd, _f_bwd, optimize_remat=True)


def flash_attention_reference(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"],
    v: Float[Array, "B K S H"],
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,  # unused
    dtype: jnp.dtype,
) -> Array:
    del position

    B, N, T, H = q.shape
    _, K, S, _ = k.shape
    HEAD_GROUPS = N // K

    mask_ninf = jnp.asarray((-jnp.inf), dtype=dtype)
    fill = jnp.full((T, S), fill_value=mask_ninf, dtype=dtype)
    query_offset = S - T
    mask = jnp.triu(fill, k=1 + query_offset)
    if sliding_window > 0:
        mask += jnp.tril(fill, k=query_offset - sliding_window)

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


def flash_attention(
    q: Float[Array, "B N T H"],
    k: Float[Array, "B K S H"],
    v: Float[Array, "B K S H"],
    sinks: Array,
    sm_scale: float,
    sliding_window: int,
    position: Array,  # unused
    *,
    dtype: jnp.dtype | None = None,
    implementation: Implementation = "auto",
) -> Array:

    B, N, T, H = q.shape
    _, K, S, _ = k.shape
    dtype = dtype if dtype is not None else q.dtype

    config = select_flash_attention_config(
        q_tokens=T,
        kv_tokens=S,
        q_heads=N,
        kv_heads=K,
        head_dim=H,
        sliding_window=sliding_window,
        out_dtype=dtype,
        # The specialized dKV schedule was validated with BF16 inputs only.
        batch_size=B if q.dtype == k.dtype == v.dtype == jnp.bfloat16 else None,
    )

    def reference():
        return flash_attention_reference(q, k, v, sinks, sm_scale, sliding_window, position, dtype=dtype)

    def kernel():
        assert config is not None
        return _A(_op(q, k, v, sinks, sm_scale, sliding_window, config))

    win_type = "sliding" if sliding_window else "global"
    tiling = None if config is None else f"q{config.tile_q}/kv{config.tile_kv}" + (" banded" if config.banded_fwd else "")
    if tiling is not None and config is not None and config.kv_major_dkv:
        tiling += " dKV=kv-major"
    if tiling is not None and config is not None and config.clamp_causal_prefetch:
        tiling += " causal-prefetch"
    if tiling is not None and config is not None and config.heads_per_program_fwd > 1:
        tiling += f" heads{config.heads_per_program_fwd}"
    op_shape = f"b{B} n{N} k{K} t{T} s{S} h{H} sw{sliding_window}"

    out = dispatch_kernel(
        f"flash_attention.{win_type}",
        implementation=implementation,
        shape=op_shape,
        config=config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
    )

    return out
