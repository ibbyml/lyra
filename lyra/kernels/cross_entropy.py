from dataclasses import dataclass
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Ref, core, lax, named_scope
from jax.experimental import pallas as pl
from jax.experimental.hijax import HiPrim, NullAccum, RefAccum
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
    Reduction,
    current_tpu_target,
    dispatch_kernel,
    mxu_dot,
    tpu_vmem_limit_bytes,
)

_A = lambda x: cast(Array, x)
_TA = lambda xs: tuple(cast(Array, x) for x in xs)


@register_static
@dataclass(frozen=True)
class LinearCrossEntropyConfig:
    tile_t: int = 128
    tile_v: int = 128

    tile_t_dw: int = 128
    tile_v_dw: int = 128
    tile_t_dx: int = 128
    tile_v_dx: int = 128

    buffer_count: int = 2


_LINEAR_CE_CONFIGS = {
    (1024, 204800, 2048): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=4096, tile_t_dw=1024, tile_v_dw=1024, tile_t_dx=1024, tile_v_dx=2048
    ),
    (4096, 204800, 2048): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=4096, tile_t_dw=1024, tile_v_dw=2048, tile_t_dx=2048, tile_v_dx=1024
    ),
    (8192, 204800, 2048): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=4096, tile_t_dw=1024, tile_v_dw=1024, tile_t_dx=1024, tile_v_dx=2048
    ),
    (16384, 204800, 2048): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=4096, tile_t_dw=1024, tile_v_dw=2048, tile_t_dx=1024, tile_v_dx=2048
    ),
    (32768, 204800, 2048): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=4096, tile_t_dw=1024, tile_v_dw=2048, tile_t_dx=1024, tile_v_dx=2048
    ),
    (16384, 204800, 4096): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=2048, tile_t_dw=1024, tile_v_dw=1024, tile_t_dx=1024, tile_v_dx=1024
    ),
    (32768, 204800, 4096): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=2048, tile_t_dw=1024, tile_v_dw=1024, tile_t_dx=1024, tile_v_dx=1024
    ),
    (4096, 204800, 4096): LinearCrossEntropyConfig(
        tile_t=1024, tile_v=2048, tile_t_dw=1024, tile_v_dw=1024, tile_t_dx=1024, tile_v_dx=1024
    ),
}


def select_linear_cross_entropy_config(
    tokens: int,
    vocab: int,
    hidden: int,
) -> LinearCrossEntropyConfig | None:
    target = current_tpu_target()
    if target is None:
        return None
    config = LinearCrossEntropyConfig()
    if target == "v6e" and (tuned := _LINEAR_CE_CONFIGS.get((tokens, vocab, hidden))):
        config = tuned
    vocab_tiles = (config.tile_v, config.tile_v_dw, config.tile_v_dx)
    token_tiles = (config.tile_t, config.tile_t_dw, config.tile_t_dx)
    if any(vocab % tile for tile in vocab_tiles) or any(tokens % tile for tile in token_tiles):
        return None
    if hidden % 128 or config.buffer_count < 1:
        return None
    return config


def reduce_loss(loss: Array, reduction: Reduction, preferred_element_type: jnp.dtype | None = None) -> Array:
    if reduction == "mean":
        return jnp.mean(loss, dtype=preferred_element_type)
    if reduction == "sum":
        return jnp.sum(loss, dtype=preferred_element_type)


def grad_scale(tokens: int | Array, reduction: Reduction) -> Array | float:
    if reduction == "mean":
        return 1.0 / tokens
    if reduction == "sum":
        return 1.0


def target_mask(y_tile, idx, bw):
    start_window = idx * bw
    y_loc = y_tile - start_window
    gathered = jax.nn.one_hot(y_loc, num_classes=bw)
    return gathered


def _linear_cross_entropy_kernel(
    w_tile_ref: Ref,
    x_tile_ref: Ref,
    y_tile_ref: Ref,
    loss_ref: Ref,
    lse_ref: Ref,
    m_scratch: Array,
    t_scratch: Array,
    l_scratch: Array,
    *,
    vocab_size: int,
    reduction: Reduction,
):
    i, j = pl.program_id(0), pl.program_id(1)
    i_programs, j_programs = pl.num_programs(0), pl.num_programs(1)
    TILE_V, TILE_T = w_tile_ref.shape[0], x_tile_ref.shape[0]

    _init = jnp.logical_and((i == 0), (j == 0))
    valid_programs = (vocab_size + TILE_V - 1) // TILE_V
    partial_program, valid_in_partial = divmod(vocab_size, TILE_V)
    _mask_padding = jnp.logical_and(i == partial_program, valid_in_partial)

    _run = i < valid_programs
    _write = jnp.logical_and((i == i_programs - 1), (j == j_programs - 1))

    @pl.when(_init)
    def _():
        m_scratch[...] = jnp.full_like(m_scratch, -jnp.inf)
        t_scratch[...] = jnp.zeros_like(t_scratch)
        l_scratch[...] = jnp.zeros_like(l_scratch)

    @pl.when(_run)
    def _():
        slc = pl.ds(j * TILE_T, TILE_T)

        def _sm_tile(z_ref):
            z_ref[...] = mxu_dot(
                BF16(x_tile_ref[...]),
                BF16(w_tile_ref[...]),
                dimension_numbers=TILE_RHS_T,
            )

            t_mask = target_mask(y_tile_ref[...], i, TILE_V)
            t_loc = jnp.sum(z_ref[...] * t_mask, axis=-1, keepdims=True)

            @pl.when(_mask_padding)
            def _():
                padding = TILE_V - valid_in_partial
                z_ref[:, pl.ds(valid_in_partial, padding)] = jnp.full((TILE_T, padding), -jnp.inf, dtype=z_ref.dtype)

            z_tile = z_ref[...]
            m_prev = m_scratch[:, slc].T
            m_loc = jnp.max(z_tile, axis=-1, keepdims=True)
            m_curr = jnp.maximum(m_loc, m_prev)
            m_cf = jnp.exp(m_prev - m_curr)
            m_scratch[:, slc] = FP32(m_curr.T)
            t_scratch[:, slc] += FP32(t_loc.T)

            exp_logits = lax.exp2((z_tile - m_curr) * LOG2_E)
            l_loc = jnp.sum(exp_logits, axis=-1, keepdims=True)
            l_curr = l_scratch[:, slc].T * m_cf + l_loc
            l_scratch[:, slc] = FP32(l_curr.T)

        # Prevents duplicate tile materialization
        pl.run_scoped(
            _sm_tile,
            pltpu.VMEM((TILE_T, TILE_V), jnp.float32),
        )

    @pl.when(_write)
    def _():
        lse = m_scratch[...] + jnp.log2(l_scratch[...]) / LOG2_E
        loss_ref[0] = reduce_loss(lse - t_scratch[...], reduction, preferred_element_type=loss_ref.dtype)
        lse_ref[...] = lse.astype(lse_ref.dtype).squeeze()


def _linear_cross_entropy(
    w: Array,
    x: Array,
    y: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    V, _ = w.shape
    T, C = x.shape
    TILE_V, TILE_T = config.tile_v, config.tile_t

    i_programs = V // TILE_V
    j_programs = T // TILE_T

    def w_index_map(i, _):
        return (i, 0)

    def x_index_map(_, j):
        return (j, 0)

    def y_index_map(_, j):
        return (j, 0)

    fwd = pl.pallas_call(
        kernel=partial(
            _linear_cross_entropy_kernel,
            vocab_size=vocab_size,
            reduction=reduction,
        ),
        grid=(i_programs, j_programs),
        in_specs=[
            pl.BlockSpec((TILE_V, C), w_index_map),
            pl.BlockSpec((TILE_T, C), x_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), y_index_map, pl.Buffered(config.buffer_count)),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),  # loss spec
            pl.BlockSpec(memory_space=pltpu.VMEM),  # lse spec
        ],
        out_shape=(
            jax.ShapeDtypeStruct((1,), dtype=jnp.float32),  # loss shape
            jax.ShapeDtypeStruct((T,), dtype=jnp.float32),
        ),
        scratch_shapes=(
            pltpu.VMEM((1, T), dtype=jnp.float32),  # max scratch
            pltpu.VMEM((1, T), dtype=jnp.float32),  # targets scratch
            pltpu.VMEM((1, T), dtype=jnp.float32),  # normalizer scratch
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="LINEAR_CE",
    )

    loss, lse = _TA(fwd(w, x, y))

    return loss, lse


def _linear_cross_entropy_bwd_dw_kernel_impl(
    w_tile_ref: Array,
    x_tile_ref: Array,
    y_tile_ref: Array,
    lse_tile_ref: Array,
    grad_ref: Array,
    dw_acc_ref: Array | None,
    dw_ref: Array,
    dw_scratch: Array,
    *,
    vocab_size: int,
    reduction: Reduction,
):
    i, j = pl.program_id(0), pl.program_id(1)
    j_programs = pl.num_programs(1)
    TILE_V, TILE_T = w_tile_ref.shape[0], x_tile_ref.shape[0]
    scale = grad_scale(tokens=TILE_T * j_programs, reduction=reduction) * grad_ref[0, 0]

    valid_programs = (vocab_size + TILE_V - 1) // TILE_V
    partial_program, valid_in_partial = divmod(vocab_size, TILE_V)
    _mask_padding = jnp.logical_and(i == partial_program, valid_in_partial)

    _init = j == 0
    _run = i < valid_programs
    _write = j == (j_programs - 1)

    @pl.when(_init)
    def _():
        dw_scratch[...] = jnp.zeros_like(dw_scratch)

    @pl.when(_run)
    def _():
        def _accumulate_dw(dz_ref):
            x_tile = BF16(x_tile_ref[...])
            w_tile = BF16(w_tile_ref[...])
            z_tile = mxu_dot(x_tile, w_tile, dimension_numbers=TILE_RHS_T)
            lse_tile = lse_tile_ref[...][..., None]
            p_tile = lax.exp2((z_tile - lse_tile) * LOG2_E)
            t_mask = target_mask(y_tile_ref[...], i, TILE_V)
            dz_ref[...] = BF16(p_tile - t_mask)

            @pl.when(_mask_padding)
            def _():
                invalid = TILE_V - valid_in_partial
                dz_ref[:, pl.ds(valid_in_partial, invalid)] = jnp.zeros((TILE_T, invalid), dtype=dz_ref.dtype)

            dw_tile = mxu_dot(dz_ref[...], x_tile, dimension_numbers=TILE_LHS_T)
            dw_scratch[...] += FP32(dw_tile)

        # Prevents duplicate tile materialization
        pl.run_scoped(
            _accumulate_dw,
            pltpu.VMEM((TILE_T, TILE_V), jnp.bfloat16),
        )

    @pl.when(_write)
    def _():
        contribution = (dw_scratch[...] * scale).astype(dw_ref.dtype)
        dw_ref[...] = contribution if dw_acc_ref is None else dw_acc_ref[...] + contribution


def _linear_cross_entropy_bwd_dw_kernel(
    w_tile_ref: Array,
    x_tile_ref: Array,
    y_tile_ref: Array,
    lse_tile_ref: Array,
    grad_ref: Array,
    dw_ref: Array,
    dw_scratch: Array,
    *,
    vocab_size: int,
    reduction: Reduction,
):
    _linear_cross_entropy_bwd_dw_kernel_impl(
        w_tile_ref,
        x_tile_ref,
        y_tile_ref,
        lse_tile_ref,
        grad_ref,
        None,
        dw_ref,
        dw_scratch,
        vocab_size=vocab_size,
        reduction=reduction,
    )


def _linear_cross_entropy_bwd_dw_accum_kernel(
    w_tile_ref: Array,
    x_tile_ref: Array,
    y_tile_ref: Array,
    lse_tile_ref: Array,
    grad_ref: Array,
    dw_acc_ref: Array,
    dw_ref: Array,
    dw_scratch: Array,
    *,
    vocab_size: int,
    reduction: Reduction,
):
    _linear_cross_entropy_bwd_dw_kernel_impl(
        w_tile_ref,
        x_tile_ref,
        y_tile_ref,
        lse_tile_ref,
        grad_ref,
        dw_acc_ref,
        dw_ref,
        dw_scratch,
        vocab_size=vocab_size,
        reduction=reduction,
    )


def _linear_cross_entropy_bwd_dw(
    ctx: tuple,
    grad: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    w, x, y, lse = _TA(ctx)

    V, C = w.shape
    T, _ = x.shape
    TILE_V, TILE_T = config.tile_v_dw, config.tile_t_dw

    i_programs = V // TILE_V
    j_programs = T // TILE_T

    def w_index_map(i, _):
        return (i, 0)

    def x_index_map(_, j):
        return (j, 0)

    def y_index_map(_, j):
        return (j, 0)

    def lse_index_map(_, j):
        return (j, 0)

    def grad_index_map(i, j):
        return (0, 0)

    def dw_out_index_map(i, _):
        return (i, 0)

    bwd = pl.pallas_call(
        kernel=partial(
            _linear_cross_entropy_bwd_dw_kernel,
            vocab_size=vocab_size,
            reduction=reduction,
        ),
        grid=(i_programs, j_programs),
        in_specs=[
            pl.BlockSpec((TILE_V, C), w_index_map),
            pl.BlockSpec((TILE_T, C), x_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), y_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), lse_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((1, 1), grad_index_map),
        ],
        out_specs=pl.BlockSpec((TILE_V, C), dw_out_index_map),
        out_shape=jax.ShapeDtypeStruct((V, C), w.dtype),
        scratch_shapes=(
            pltpu.VMEM((TILE_V, C), dtype=jnp.float32),  # dw scratch
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="LINEAR_CE_BWD_DW",
    )
    dw = _A(bwd(w, x, y, lse[..., None], FP32(grad).reshape(1, 1)))

    return dw


def _linear_cross_entropy_bwd_dw_accum(
    ctx: tuple,
    grad: Array,
    dw_acc: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    w, x, y, lse = _TA(ctx)

    V, C = w.shape
    T, _ = x.shape
    TILE_V, TILE_T = config.tile_v_dw, config.tile_t_dw

    i_programs = V // TILE_V
    j_programs = T // TILE_T

    def w_index_map(i, _):
        return (i, 0)

    def x_index_map(_, j):
        return (j, 0)

    def y_index_map(_, j):
        return (j, 0)

    def lse_index_map(_, j):
        return (j, 0)

    def grad_index_map(i, j):
        return (0, 0)

    def dw_index_map(i, _):
        return (i, 0)

    bwd = pl.pallas_call(
        kernel=partial(
            _linear_cross_entropy_bwd_dw_accum_kernel,
            vocab_size=vocab_size,
            reduction=reduction,
        ),
        grid=(i_programs, j_programs),
        in_specs=[
            pl.BlockSpec((TILE_V, C), w_index_map),
            pl.BlockSpec((TILE_T, C), x_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), y_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), lse_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((1, 1), grad_index_map),
            pl.BlockSpec((TILE_V, C), dw_index_map, pl.Buffered(1)),
        ],
        out_specs=pl.BlockSpec((TILE_V, C), dw_index_map, pl.Buffered(1)),
        out_shape=jax.ShapeDtypeStruct((V, C), dw_acc.dtype),
        scratch_shapes=(pltpu.VMEM((TILE_V, C), dtype=jnp.float32),),
        input_output_aliases={5: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="LINEAR_CE_BWD_DW_ACCUM",
    )
    return _A(bwd(w, x, y, lse[..., None], FP32(grad).reshape(1, 1), dw_acc))


def _linear_cross_entropy_bwd_dx_kernel(
    x_tile_ref: Array,
    w_tile_ref: Array,
    y_tile_ref: Array,
    lse_tile_ref: Array,
    grad_ref: Array,
    dx_ref: Array,
    dx_scratch: Array,
    *,
    vocab_size: int,
    reduction: Reduction,
):
    j = pl.program_id(1)
    i_programs, j_programs = pl.num_programs(0), pl.num_programs(1)
    TILE_T, TILE_V = x_tile_ref.shape[0], w_tile_ref.shape[0]
    scale = grad_scale(tokens=TILE_T * i_programs, reduction=reduction) * grad_ref[0, 0]

    _init = j == 0
    valid_programs = (vocab_size + TILE_V - 1) // TILE_V
    partial_program, valid_in_partial = divmod(vocab_size, TILE_V)
    _mask_padding = jnp.logical_and(j == partial_program, valid_in_partial)

    _run = j < valid_programs
    _write = j == (j_programs - 1)

    @pl.when(_init)
    def _():
        dx_scratch[...] = jnp.zeros_like(dx_scratch)

    @pl.when(_run)
    def _():
        def _accumulate_dx(dz_ref):
            x_tile = BF16(x_tile_ref[...])
            w_tile = BF16(w_tile_ref[...])
            z_tile = mxu_dot(x_tile, w_tile, dimension_numbers=TILE_RHS_T)
            lse_tile = lse_tile_ref[...][..., None]
            p_tile = lax.exp2((z_tile - lse_tile) * LOG2_E)
            t_mask = target_mask(y_tile_ref[...], j, TILE_V)
            dz_ref[...] = BF16(p_tile - t_mask)

            @pl.when(_mask_padding)
            def _():
                invalid = TILE_V - valid_in_partial
                dz_ref[:, pl.ds(valid_in_partial, invalid)] = jnp.zeros((TILE_T, invalid), dtype=dz_ref.dtype)

            dx_tile = mxu_dot(dz_ref[...], w_tile, dimension_numbers=TILE_INNER)
            dx_scratch[...] += FP32(dx_tile)

        # Prevents duplicate tile materialization
        pl.run_scoped(
            _accumulate_dx,
            pltpu.VMEM((TILE_T, TILE_V), jnp.bfloat16),
        )

    @pl.when(_write)
    def _():
        dx_ref[...] = (dx_scratch[...] * scale).astype(dx_ref.dtype)


def _linear_cross_entropy_bwd_dx(
    ctx: tuple,
    grad: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    w, x, y, lse = _TA(ctx)

    T, C = x.shape
    V, _ = w.shape
    TILE_T, TILE_V = config.tile_t_dx, config.tile_v_dx

    i_programs = T // TILE_T
    j_programs = V // TILE_V

    def x_index_map(i, _):
        return (i, 0)

    def w_index_map(_, j):
        return (j, 0)

    def y_index_map(i, _):
        return (i, 0)

    def lse_index_map(i, _):
        return (i, 0)

    def grad_index_map(i, j):
        return (0, 0)

    def dx_out_index_map(i, _):
        return (i, 0)

    bwd = pl.pallas_call(
        kernel=partial(
            _linear_cross_entropy_bwd_dx_kernel,
            vocab_size=vocab_size,
            reduction=reduction,
        ),
        grid=(i_programs, j_programs),
        in_specs=[
            pl.BlockSpec((TILE_T, C), x_index_map),
            pl.BlockSpec((TILE_V, C), w_index_map, pl.Buffered(config.buffer_count)),
            pl.BlockSpec((TILE_T, None), y_index_map),
            pl.BlockSpec((TILE_T, None), lse_index_map),
            pl.BlockSpec((1, 1), grad_index_map),  # grad spec
        ],
        out_specs=pl.BlockSpec((TILE_T, C), dx_out_index_map),
        out_shape=jax.ShapeDtypeStruct((T, C), x.dtype),
        scratch_shapes=(
            pltpu.VMEM((TILE_T, C), dtype=jnp.float32),  # dx scratch
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(PARALLEL, ARBITRARY),
            vmem_limit_bytes=tpu_vmem_limit_bytes(),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": False},
        ),
        name="LINEAR_CE_BWD_DX",
    )

    dx = _A(bwd(x, w, y, lse[..., None], FP32(grad).reshape(1, 1)))

    return dx


def _f(
    w: Array,
    x: Array,
    y: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    loss, _ = _linear_cross_entropy(w, x, y, reduction, vocab_size, config)
    return loss


def _f_fwd(
    w: Array,
    x: Array,
    y: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    loss, lse = _linear_cross_entropy(w, x, y, reduction, vocab_size, config)
    bwd_ctx = (w, x, y, lse)
    return loss, bwd_ctx


def _f_bwd(
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
    ctx: tuple,
    grad: Array,
):
    dw = _linear_cross_entropy_bwd_dw(ctx, grad, reduction, vocab_size, config)
    dx = _linear_cross_entropy_bwd_dx(ctx, grad, reduction, vocab_size, config)
    return dw, dx, None


_op = jax.custom_vjp(_f, nondiff_argnames=("reduction", "vocab_size", "config"))
_op.defvjp(_f_fwd, _f_bwd, optimize_remat=True)


class _AccumulatingLinearCrossEntropy(HiPrim):
    reduction: Reduction
    vocab_size: int
    config: LinearCrossEntropyConfig

    def __init__(self, w_aval, x_aval, y_aval, reduction: Reduction, vocab_size: int, config: LinearCrossEntropyConfig):
        self.in_avals = (w_aval, x_aval, y_aval)
        self.out_aval = core.ShapedArray((1,), jnp.float32)
        self.params = {"reduction": reduction, "vocab_size": vocab_size, "config": config}
        super().__init__()

    def expand(self, *args):
        w, x, y = args
        return _f(w, x, y, self.reduction, self.vocab_size, self.config)

    def vjp_fwd(self, nzs_in, /, *args):
        del nzs_in
        w, x, y = args
        return _f_fwd(w, x, y, self.reduction, self.vocab_size, self.config)

    def vjp_bwd(self, ctx, grad, /, *arg_accums):
        w_acc, x_acc, y_acc = arg_accums
        del y_acc

        if isinstance(w_acc, RefAccum) and w_acc.ref is not None:
            updated = _linear_cross_entropy_bwd_dw_accum(
                ctx,
                grad,
                w_acc.ref[...],
                self.reduction,
                self.vocab_size,
                self.config,
            )
            w_acc.ref[...] = updated
        elif not isinstance(w_acc, NullAccum):
            w_acc.accum(_linear_cross_entropy_bwd_dw(ctx, grad, self.reduction, self.vocab_size, self.config))

        if not isinstance(x_acc, NullAccum):
            x_acc.accum(_linear_cross_entropy_bwd_dx(ctx, grad, self.reduction, self.vocab_size, self.config))

    def batch(self, axis_data, args, dims):
        del axis_data

        def unaccumulated(w, x, y):
            return _op(w, x, y, self.reduction, vocab_size=self.vocab_size, config=self.config)

        return jax.vmap(unaccumulated, in_axes=dims, out_axes=0)(*args), 0


def _accumulating_op(
    w: Array,
    x: Array,
    y: Array,
    reduction: Reduction,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
):
    primitive = _AccumulatingLinearCrossEntropy(
        jax.typeof(w),
        jax.typeof(x),
        jax.typeof(y),
        reduction,
        vocab_size,
        config,
    )
    return primitive(w, x, y)


def _use_accumulating_ce(
    w: Array,
    x: Array,
    y: Array,
    vocab_size: int,
    config: LinearCrossEntropyConfig,
) -> bool:
    return (
        current_tpu_target() == "v6e"
        and jax.device_count() == 1
        and w.shape == (204_800, 2_048)
        and x.shape == (16_384, 2_048)
        and y.shape == (16_384, 1)
        and vocab_size == 201_088
        and w.dtype == jnp.float32
        and x.dtype == jnp.bfloat16
        and y.dtype == jnp.int32
        and config.tile_t_dw == 1_024
        and config.tile_v_dw == 2_048
        and config.buffer_count == 2
    )


def linear_cross_entropy_reference(
    w: Array,
    x: Array,
    y: Array,
    vocab_size: int,
    reduction: Reduction,
    dtype: jnp.dtype,
) -> Array:
    with named_scope("Linear matmul"):
        z = jnp.einsum("tc,vc->tv", x, w[:vocab_size], preferred_element_type=dtype)
    with named_scope("Cross entropy"):
        lse = jax.nn.logsumexp(z, axis=-1, keepdims=True)
        t = jnp.take_along_axis(z, y, axis=-1)
        loss = reduce_loss((lse - t), reduction=reduction, preferred_element_type=dtype)
    return loss


def linear_cross_entropy(
    w: Float[Array, "V C"],
    x: Float[Array, "T C"],
    y: Float[Array, "T 1"],
    *,
    vocab_size: int | None = None,
    reduction: Reduction = "sum",
    dtype: jnp.dtype | None = None,
    implementation: Implementation = "auto",
) -> Array:

    V, C = w.shape
    T, _ = x.shape
    VSZ = V if vocab_size is None else vocab_size
    dtype = dtype if dtype is not None else w.dtype

    config = select_linear_cross_entropy_config(tokens=T, vocab=V, hidden=C)

    def reference():
        return linear_cross_entropy_reference(w, x, y, vocab_size=VSZ, reduction=reduction, dtype=dtype).squeeze()

    def kernel():
        assert config is not None
        op = _accumulating_op if _use_accumulating_ce(w, x, y, VSZ, config) else _op
        return _A(op(w, x, y, reduction, vocab_size=VSZ, config=config)).squeeze()

    tiling = None if config is None else f"t{config.tile_t}/v{config.tile_v} (fwd; dw t{config.tile_t_dw}/v{config.tile_v_dw})"
    op_shape = f"t{T} v{V} c{C}"

    out = dispatch_kernel(
        "linear_cross_entropy",
        implementation=implementation,
        shape=op_shape,
        config=config,
        tiling=tiling,
        kernel=kernel,
        reference=reference,
    )

    return out
