from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax import P, lax, shard_map
from jax.sharding import Mesh
from jaxtyping import Array, Int, PyTree

from lyra.generate import sample_tokens
from lyra.kernels.op import ce_op
from lyra.model import ModelConfig, ModelWeights, lm_head_weight, model_apply
from lyra.nn.params import ShardingRules, all_gather_params, spec_axis_names, weight_sharding
from lyra.training.metrics import PPL_CE_CLAMP
from lyra.training.optimizer import optimizer_step
from lyra.training.probe import Probe
from lyra.training.state import TrainState

if TYPE_CHECKING:
    from lyra.training.train import TrainingConfig

_A = lambda x: cast(Array, x)


def loss_parts_fn(config: ModelConfig):
    ce = ce_op(config.implementation)

    @jax.named_scope("loss")
    def loss_parts(params: ModelWeights, x: Array, y: Array) -> tuple[Array, Array]:
        B, T = x.shape
        h, aux_loss, _ = model_apply(x, params, cache=None, return_hidden_states=True, config=config)
        unemb = lm_head_weight(params.tok, config)
        ce_sum = ce(w=unemb, x=h.reshape(B * T, -1), y=y.reshape(B * T, 1), vocab_size=config.vocab_size, reduction="sum")
        return ce_sum, aux_loss

    return loss_parts


@dataclass(frozen=True)
class _StepSetup:
    microbatches: int
    global_tokens: int
    global_microbatches: int
    sharding: ShardingRules
    mesh: Mesh
    param_specs: PyTree
    loss_parts: Callable[..., tuple[Array, Array]]

    @classmethod
    def for_batch(cls, xs: Int[Array, "A B T"], config: ModelConfig) -> _StepSetup:
        A, B, T = xs.shape
        return cls(
            microbatches=A,
            global_tokens=A * B * T,
            global_microbatches=A * config.sharding.data_axis_size,
            sharding=config.sharding,
            mesh=config.sharding.get_mesh(),
            param_specs=weight_sharding(ModelWeights.spec(config)),
            loss_parts=loss_parts_fn(config),
        )

    def losses(self, params: ModelWeights, x: Array, y: Array) -> tuple[Array, tuple[Array, Array]]:
        ce_sum, aux_loss = self.loss_parts(params, x, y)
        ce_term, aux_term = ce_sum / self.global_tokens, aux_loss / self.global_microbatches
        return ce_term + aux_term, (ce_term, aux_term)

    def psum_data_axes(self, tree: PyTree) -> PyTree:
        axes = self.sharding.data_axis_names
        return jax.tree.map(lambda x: _A(lax.psum(x, axes)), tree) if axes else tree


def reduce_grads(loc_grads: PyTree, param_specs: PyTree, sharding: ShardingRules) -> PyTree:
    fsdp_axes = tuple(sharding.fsdp_axis_names)

    def reduce(grad, grad_spec):
        sharded = set(spec_axis_names(grad_spec))
        replicated = tuple(axis for axis in sharding.mesh_axis_names if axis not in sharded and axis not in fsdp_axes)
        if replicated:
            grad = _A(lax.psum(grad, replicated))
        if fsdp_axes:
            fsdp_dims = [dim for dim, part in enumerate(grad_spec) if set(spec_axis_names((part,))) & set(fsdp_axes)]
            if fsdp_dims:
                grad = _A(lax.psum_scatter(grad, fsdp_axes, scatter_dimension=fsdp_dims[0], tiled=True))
            else:
                grad = _A(lax.psum(grad, fsdp_axes))
        return grad

    return jax.tree.map(reduce, loc_grads, param_specs)


@jax.jit(static_argnames=("max_new_tokens", "config"), inline=True)
def sample_eval_step(
    model: ModelWeights, prompt: Array, prompt_length: Array, sample_index: Array, max_new_tokens: int, config: ModelConfig
) -> Array:
    key = jax.random.fold_in(jax.random.split(jax.random.key(config.seed))[1], sample_index)
    return sample_tokens(model, key, config, prompt, prompt_length, max_new_tokens)


@jax.jit(static_argnames=("spec", "config"), inline=True)
def validation_step(xs: Int[Array, "A B T"], ys: Int[Array, "A B T"], params: ModelWeights, spec: P, config: ModelConfig):
    setup = _StepSetup.for_batch(xs, config)

    @shard_map(in_specs=(setup.param_specs, spec, spec), out_specs=P(), mesh=setup.mesh, check_vma=False)
    def sharded_validation(sparams, local_xs, local_ys):
        sparams = all_gather_params(sparams, setup.param_specs, setup.sharding)

        def accumulate(totals, batch):
            loss, (ce, aux) = setup.losses(sparams, *batch)
            return jax.tree.map(jnp.add, totals, (loss, ce, aux)), None

        zero = jnp.zeros((), jnp.float32)
        totals, _ = lax.scan(accumulate, (zero, zero, zero), (local_xs, local_ys))
        return setup.psum_data_axes(totals)

    loss, ce, aux = sharded_validation(params, xs, ys)
    return {"eval_loss": loss, "eval_ce_loss": ce, "eval_aux_loss": aux}


def run_validation(
    val_step: Callable, batches: Iterator, params: ModelWeights, *, eval_batches: int, microbatches: int
) -> dict[str, Array]:
    totals: dict[str, Array] = {}
    for _ in range(eval_batches):
        xs, ys = zip(*(next(batches) for _ in range(microbatches)))
        for name, value in val_step(jnp.stack(xs), jnp.stack(ys), params).items():
            totals[name] = totals.get(name, 0.0) + value
    metrics = {name: value / eval_batches for name, value in totals.items()}
    metrics["eval_perplexity"] = jnp.exp(jnp.minimum(metrics["eval_ce_loss"], PPL_CE_CLAMP))
    return metrics


@jax.jit(static_argnames=("spec", "config", "probe"), inline=True)
def numerics_step(xs: Int[Array, "A B T"], params: ModelWeights, spec: P, config: ModelConfig, probe: Probe) -> dict[str, dict[str, Array]]:
    setup = _StepSetup.for_batch(xs, config)

    @shard_map(in_specs=(setup.param_specs, spec), out_specs=P(), mesh=setup.mesh, check_vma=False)
    def sharded_numerics(sparams, local_xs):
        sparams = all_gather_params(sparams, setup.param_specs, setup.sharding)
        probe.start()
        model_apply(local_xs[0], sparams, None, config, return_hidden_states=True, probe=probe)
        stats = probe.collect()
        axes = setup.sharding.data_axis_names
        return jax.tree.map(lambda x: lax.pmean(x, axes), stats) if axes else stats

    return sharded_numerics(params, xs)


@jax.jit(donate_argnames=("state",), static_argnames=("spec", "mcfg", "tcfg"), inline=True)
def acc_train_step(
    xs: Int[Array, "A B T"],
    ys: Int[Array, "A B T"],
    step: Int[Array, ""],
    state: TrainState,
    spec: P,
    mcfg: ModelConfig,
    tcfg: TrainingConfig,
):
    setup = _StepSetup.for_batch(xs, mcfg)

    @shard_map(in_specs=(setup.param_specs, spec, spec), out_specs=(P(), setup.param_specs), mesh=setup.mesh, check_vma=False)
    def sharded_gradient_accumulation(sparams, local_xs, local_ys):
        sparams = all_gather_params(sparams, setup.param_specs, setup.sharding)
        if setup.microbatches == 1:
            loss, vjp, (ce, aux) = jax.vjp(lambda p: setup.losses(p, local_xs[0], local_ys[0]), sparams, has_aux=True)
            grads = vjp(jnp.ones_like(loss))[0]
            totals = (loss, ce, aux)
        else:
            grad_refs = jax.tree.map(lambda p: jax.new_ref(jnp.zeros_like(p)), sparams)
            total_refs = tuple(jax.new_ref(jnp.zeros((), jnp.float32)) for _ in range(3))

            @jax.named_scope("microbatch_forward_backward")
            def accumulate(_, batch):
                loss, vjp, (ce, aux) = jax.vjp(lambda p: setup.losses(p, *batch), sparams, has_aux=True)
                cast(Any, vjp).with_refs(grad_refs)(jnp.ones_like(loss))
                for ref, value in zip(total_refs, (loss, ce, aux)):
                    ref[...] += value
                return (), None

            lax.scan(accumulate, (), (local_xs, local_ys))
            grads = jax.tree.map(jax.freeze, grad_refs)
            totals = tuple(jax.freeze(ref) for ref in total_refs)
        return setup.psum_data_axes(totals), reduce_grads(grads, setup.param_specs, setup.sharding)

    with jax.named_scope("gradient_accumulation"):
        (loss, ce, aux), grads = sharded_gradient_accumulation(state.params, xs, ys)
    with jax.named_scope("optimizer"):
        params, optimizer, opt_metrics = optimizer_step(state.params, grads, state.optimizer, step, tcfg)
    return TrainState(params, optimizer), {"loss": loss, "ce_loss": ce, "aux_loss": aux, **opt_metrics}
