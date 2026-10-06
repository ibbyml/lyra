from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jaxtyping import Array, PyTree

from lyra.model import ModelWeights
from lyra.nn.params import MaskedNode, Tag, is_masked, mask_spec_tree, merge_masked_tree, zero_masked_tree

if TYPE_CHECKING:
    from lyra.training.train import TrainingConfig

MUON_TAGS = {Tag.MATRIX}
ADAM_TAGS = {Tag.DEFAULT, Tag.BIAS, Tag.EMBEDDING, Tag.SCALAR}

_POLAR_EXPRESS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
POLAR_EXPRESS_COEFFS = tuple((a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_EXPRESS)


def mmap(fn, *trees):
    return jax.tree.map(lambda *leaves: MaskedNode() if any(map(is_masked, leaves)) else fn(*leaves), *trees, is_leaf=is_masked)


def mmap_with_path(fn, *trees):
    return jax.tree.map_with_path(
        lambda path, *leaves: MaskedNode() if any(map(is_masked, leaves)) else fn(path, *leaves), *trees, is_leaf=is_masked
    )


def _as_matrices(path, x: Array) -> tuple[Array, bool]:
    if any(getattr(key, "name", None) == "mlp_up" for key in path):
        return x.swapaxes(-3, -2), True
    return x, False


@register_dataclass
@dataclass
class MuonState:
    moment: PyTree


@register_dataclass
@dataclass
class AdamState:
    mu: PyTree
    nu: PyTree
    count: Array


@register_dataclass
@dataclass
class OptimizerState:
    muon: MuonState
    adam: AdamState

    @classmethod
    def init(cls, params: ModelWeights, spec: ModelWeights, tcfg: TrainingConfig) -> OptimizerState:
        muon_params = mask_spec_tree(params, spec, MUON_TAGS)
        adam_params = mask_spec_tree(params, spec, ADAM_TAGS)
        return cls(
            muon=MuonState(moment=zero_masked_tree(muon_params, dtype=tcfg.muon_moment_dtype)),
            adam=AdamState(
                mu=zero_masked_tree(adam_params, dtype=tcfg.adam_mu_dtype),
                nu=zero_masked_tree(adam_params, dtype=tcfg.adam_nu_dtype),
                count=jnp.zeros((), dtype=jnp.int32),
            ),
        )


def wsd_lr(step: Array, tcfg: TrainingConfig) -> tuple[Array, Array, Array]:
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup = jnp.minimum((step + 1.0) / max(tcfg.warmup_steps, 1), 1.0)
    decay_start = tcfg.warmup_steps + tcfg.stable_steps
    progress = jnp.clip((step - decay_start) / max(tcfg.decay_steps - 1, 1), 0.0, 1.0)
    decay = 0.5 * (1.0 + jnp.cos(jnp.pi * progress)) if tcfg.decay_type == "cosine" else 1.0 - progress
    scale = jnp.where(step < tcfg.warmup_steps, warmup, decay)
    return tcfg.adam_max_lr * scale, tcfg.muon_max_lr * scale, scale


def global_l2_norm(tree: PyTree) -> Array:
    squares = jax.tree.leaves(mmap(lambda g: jnp.vdot(g.astype(jnp.float32), g.astype(jnp.float32)), tree))
    return jnp.sqrt(sum(squares, start=jnp.zeros((), jnp.float32)))


def clip_per_matrix_rms(grads: PyTree, max_rms: float, eps: float = 1e-8) -> tuple[PyTree, dict[str, Array]]:
    def rms(path, g):
        matrices, _ = _as_matrices(path, g.astype(jnp.float32))
        return jnp.sqrt(jnp.mean(matrices * matrices, axis=(-2, -1), keepdims=True))

    def clip(path, g, r):
        matrices, swapped = _as_matrices(path, g)
        scale = jnp.where(jnp.isfinite(r), jnp.minimum(1.0, max_rms / (r + eps)), 1.0)
        clipped = matrices * scale
        return (clipped.swapaxes(-3, -2) if swapped else clipped).astype(g.dtype)

    rms_tree = mmap_with_path(rms, grads)
    all_rms = jnp.concatenate([jnp.ravel(r) for r in jax.tree.leaves(rms_tree)])
    metrics = {"muon_grad_rms_max": jnp.max(all_rms), "muon_grad_clip_frac": jnp.mean(all_rms > max_rms)}
    
    return mmap_with_path(clip, grads, rms_tree), metrics


def newtonschulz(X: Array, eps: float = 1e-7) -> Array:
    dtype = X.dtype
    tall = X.shape[-2] > X.shape[-1]
    X = X.astype(jnp.bfloat16)
    X /= 1.01 * jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps
    if tall:
        X = X.swapaxes(-1, -2)
    for a, b, c in POLAR_EXPRESS_COEFFS:
        A = X @ X.swapaxes(-1, -2)
        X = a * X + (b * A + c * (A @ A)) @ X
    if tall:
        X = X.swapaxes(-1, -2)
    return X.astype(dtype)


def muon_step(params, grads, state: MuonState, tcfg: TrainingConfig, lr: Array, enabled: Array):
    lr = jnp.where(enabled, lr, 0.0)
    momentum = jnp.where(enabled, tcfg.muon_momentum, 1.0)
    grads = mmap(lambda g: jnp.where(enabled, g, 0.0), grads)

    moment = mmap(
        lambda m, g: (momentum * m.astype(jnp.float32) + g.astype(jnp.float32)).astype(tcfg.muon_moment_dtype), state.moment, grads
    )
    directions = mmap(lambda m, g: g + tcfg.muon_momentum * m, moment, grads) if tcfg.muon_nesterov else moment

    def update(path, p, d):
        d, swapped = _as_matrices(path, d)
        step = lr * newtonschulz(d) * (math.sqrt(max(d.shape[-2], d.shape[-1])) * tcfg.muon_update_rms)
        step = step.swapaxes(-3, -2) if swapped else step
        return (p - (step + lr * tcfg.muon_wd * p)).astype(p.dtype)

    return mmap_with_path(update, params, directions), MuonState(moment=moment)


def adam_step(params, grads, state: AdamState, tcfg: TrainingConfig, lr: Array, enabled: Array, eps: float = 1e-8):
    b1, b2 = tcfg.adam_betas
    lr = jnp.where(enabled, lr, 0.0)
    keep1, keep2 = jnp.where(enabled, b1, 1.0), jnp.where(enabled, b2, 1.0)
    grads = mmap(lambda g: jnp.where(enabled, g.astype(jnp.float32), 0.0), grads)
    correction1, correction2 = 1.0 / (1.0 - b1 ** (state.count + 1)), 1.0 / (1.0 - b2 ** (state.count + 1))

    mu = mmap(lambda m, g: (keep1 * m.astype(jnp.float32) + (1.0 - keep1) * g).astype(tcfg.adam_mu_dtype), state.mu, grads)
    nu = mmap(lambda n, g: (keep2 * n.astype(jnp.float32) + (1.0 - keep2) * g * g).astype(tcfg.adam_nu_dtype), state.nu, grads)

    def update(p, m, n):
        m_hat, n_hat = m.astype(jnp.float32) * correction1, n.astype(jnp.float32) * correction2
        return (p - lr * m_hat / (jnp.sqrt(n_hat) + eps)).astype(p.dtype)

    count = state.count + enabled.astype(state.count.dtype)
    return mmap(update, params, mu, nu), AdamState(mu=mu, nu=nu, count=count)


def optimizer_step(params, grads, state: OptimizerState, step: Array, tcfg: TrainingConfig):
    adam_lr, muon_lr, lr_scale = wsd_lr(step, tcfg)
    
    with jax.named_scope("grad_clipping"):
        muon_grads = mmap(lambda g, _: g, grads, state.muon.moment)
        adam_grads = mmap(lambda g, _: g, grads, state.adam.mu)
        clipped, clip_metrics = clip_per_matrix_rms(muon_grads, tcfg.muon_grad_clip)
        grad_norm = global_l2_norm(grads)
        finite = jnp.isfinite(grad_norm)
        
    with jax.named_scope("muon"):
        muon_params, muon_state = muon_step(params, clipped, state.muon, tcfg, muon_lr, finite)
        
    with jax.named_scope("adam"):
        adam_params, adam_state = adam_step(params, grads, state.adam, tcfg, adam_lr, finite)

    metrics = {
        "adam_lr": adam_lr,
        "muon_lr": muon_lr,
        "lr_scale": lr_scale,
        "grad_norm": grad_norm,
        "muon_grad_norm": global_l2_norm(muon_grads),
        "adam_grad_norm": global_l2_norm(adam_grads),
        "grad_is_finite": finite,
        **clip_metrics,
    }
    return merge_masked_tree(muon_params, adam_params), OptimizerState(muon_state, adam_state), metrics
