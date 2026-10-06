"""Model diagnostics collected during JAX tracing."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.nn import logsumexp, softmax
from jaxtyping import Array

EPS = 1e-30
# Padded vocabulary logits should not affect summary moments.
MASKED_LOGIT = -1e20


Stats = dict[str, Array]


class Level(IntEnum):
    """How much to compute per recorded tensor."""

    OFF = 0
    BASIC = 1
    FULL = 2


def _finite(t: Array) -> tuple[Array, Array]:
    """Zero out non-finite entries so one NaN cannot erase every other stat."""
    mask = jnp.isfinite(t)
    return jnp.where(mask, t, 0.0), jnp.count_nonzero(~mask).astype(jnp.float32)


def tensor_stats(x: Array, *, level: Level = Level.FULL) -> Stats:
    """Summarize a tensor. All statistics are computed in float32."""
    t = x.astype(jnp.float32)
    safe, n_nonfinite = _finite(t)
    flat = safe.reshape(-1)

    mean = jnp.mean(flat)
    second = jnp.mean(flat * flat)
    variance = jnp.maximum(second - mean * mean, 0.0)

    stats = {
        "mean": mean,
        "rms": jnp.sqrt(second),
        "std": jnp.sqrt(variance),
        "absmax": jnp.max(jnp.abs(flat)),
        "n_nonfinite": n_nonfinite,
    }

    if level < Level.FULL:
        return stats

    centered = flat - mean
    fourth = jnp.mean(centered**4)
    stats["kurtosis"] = fourth / (variance * variance + EPS)

    if safe.ndim >= 2:
        lead = tuple(range(safe.ndim - 1))
        chan_rms = jnp.sqrt(jnp.mean(safe * safe, axis=lead))
        chan_mean = jnp.mean(safe, axis=lead)
        row_rms = jnp.sqrt(jnp.mean(safe * safe, axis=-1))
        stats |= {
            "chan_rms_max": jnp.max(chan_rms),
            "chan_rms_ratio": jnp.max(chan_rms) / (jnp.mean(chan_rms) + EPS),
            "chan_mean_absmax": jnp.max(jnp.abs(chan_mean)),
            "row_rms_max": jnp.max(row_rms),
        }

    return stats


def residual_stats(x: Array, delta: Array) -> Stats:
    """Summarize the residual branch and its effect on the stream."""
    x32 = x.astype(jnp.float32)
    d32 = delta.astype(jnp.float32)
    o32 = x32 + d32

    q_x = jnp.mean(x32 * x32)  # Residual stream
    q_delta = jnp.mean(d32 * d32)  # Layer delta
    q_out = jnp.mean(o32 * o32)

    cross_overlap = jnp.mean(x32 * d32)

    lead = tuple(range(o32.ndim - 1))
    chan_rms = jnp.sqrt(jnp.mean(o32 * o32, axis=lead))
    chan_mean = jnp.mean(o32, axis=lead)

    # Per-token cosine avoids large-norm tokens dominating the batch statistic.
    token_dot = jnp.sum(x32 * o32, axis=-1)
    token_x = jnp.sqrt(jnp.sum(x32 * x32, axis=-1))
    token_out = jnp.sqrt(jnp.sum(o32 * o32, axis=-1))
    token_cosine = token_dot / (token_x * token_out + EPS)

    x_rms = jnp.sqrt(q_x)
    d_rms = jnp.sqrt(q_delta)
    o_rms = jnp.sqrt(q_out)

    return {
        "x_rms": x_rms,
        "delta_rms": d_rms,
        "out_rms": o_rms,
        "branch_ratio": jnp.sqrt(q_delta / (q_x + EPS)),
        "correlation": cross_overlap / jnp.sqrt(q_x * q_delta + EPS),
        "input_output_cosine": (q_x + cross_overlap) / (jnp.sqrt(q_x * q_out) + EPS),
        "token_cosine_mean": jnp.mean(token_cosine),
        "growth": jnp.sqrt(q_out / (q_x + EPS)),
        "delta_mean": jnp.mean(d32),
        "out_mean": jnp.mean(o32),
        "out_absmax": jnp.max(jnp.abs(o32)),
        "out_chan_rms_ratio": jnp.max(chan_rms) / (jnp.mean(chan_rms) + EPS),
        "out_chan_mean_absmax": jnp.max(jnp.abs(chan_mean)),
        "branch_absorbed_frac": jnp.sum(((o32.astype(x.dtype) == x) & (d32 != 0)).astype(jnp.float32))
        / jnp.maximum(jnp.count_nonzero(d32), 1),
        "n_nonfinite": jnp.count_nonzero(~jnp.isfinite(o32)).astype(jnp.float32),
    }


def swiglu_stats(x: Array, limit: float, *, level: Level = Level.FULL) -> Stats:
    """Clipping of packed gate/value inputs; the gate has no lower bound."""
    bound = jnp.asarray(limit, dtype=x.dtype).astype(jnp.float32)
    gate, value = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    stats = {
        "gate_clip_frac": jnp.mean(gate > bound),
        "value_clip_frac": jnp.mean(jnp.abs(value) > bound),
    }
    if level >= Level.FULL:
        for name, raw, clipped in (
            ("gate", gate, jnp.minimum(gate, bound)),
            ("value", value, jnp.clip(value, -bound, bound)),
        ):
            stats[f"{name}_clip_energy_frac"] = jnp.sum(raw**2 - clipped**2) / (jnp.sum(raw**2) + EPS)
    return stats


def routing_stats(
    router_logits: Array,
    expert_scores: Array,
    group_sizes: Array,
    n_experts: int,
    n_active: int,
) -> Stats:
    """Summarize one MoE routing decision."""
    logits = router_logits.astype(jnp.float32)
    counts = group_sizes.astype(jnp.float32)
    assignments = jnp.sum(counts)
    load = counts / (assignments + EPS)
    hard_entropy = -jnp.sum(load * jnp.log(load + EPS))

    probs = softmax(logits, axis=-1)
    usage = jnp.mean(probs, axis=0)
    log_usage = jnp.log(usage + EPS)
    token_entropy = -jnp.sum(probs * jnp.log(probs + EPS), axis=-1)

    scores = expert_scores.astype(jnp.float32)
    return {
        "logit_rms": jnp.sqrt(jnp.mean(logits * logits)),
        "logit_absmax": jnp.max(jnp.abs(logits)),
        "lse_mean": jnp.mean(logsumexp(logits, axis=-1)),
        "load_max_ratio": jnp.max(load) * n_experts,
        "load_min_ratio": jnp.min(load) * n_experts,
        "load_std": jnp.std(load) * n_experts,
        "experts_unused": jnp.count_nonzero(counts == 0).astype(jnp.float32),
        "hard_usage_entropy": hard_entropy,
        "usage_entropy": -jnp.sum(usage * log_usage),
        "usage_entropy_uniform": jnp.asarray(math.log(n_experts), jnp.float32),
        "token_entropy": jnp.mean(token_entropy),
        "top1_score": jnp.mean(scores[..., 0]),
        "topk_score_min": jnp.mean(scores[..., -1]),
        "n_experts": jnp.asarray(float(n_experts)),
        "n_active": jnp.asarray(float(n_active)),
    }


def logit_stats(logits: Array) -> Stats:
    """Summarize logits while excluding masked vocabulary padding."""
    z = logits.astype(jnp.float32)
    flat = z.reshape(-1, z.shape[-1])
    width = flat.shape[-1]

    live = flat > MASKED_LOGIT
    row_count = jnp.maximum(jnp.count_nonzero(live, axis=-1).astype(jnp.float32), 1.0)
    n_live = jnp.mean(row_count)
    safe = jnp.where(live, flat, 0.0)
    row_mean = jnp.sum(safe, axis=-1) / row_count
    row_second = jnp.sum(safe * safe, axis=-1) / row_count

    lse = logsumexp(flat, axis=-1)
    log_probs = flat - lse[:, None]
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(jnp.where(live, probs * log_probs, 0.0), axis=-1)

    stats = {
        "rms": jnp.sqrt(jnp.mean(row_second)),
        "absmax": jnp.max(jnp.abs(safe)),
        "token_std": jnp.sqrt(jnp.mean(jnp.maximum(row_second - row_mean * row_mean, 0.0))),
        "max_mean": jnp.mean(jnp.max(flat, axis=-1)),
        "lse_mean": jnp.mean(lse),
        "random_target_ce": jnp.mean(lse - row_mean),
        "entropy": jnp.mean(entropy),
        "entropy_uniform": jnp.log(n_live),
        "masked_frac": 1.0 - n_live / width,
        "n_nonfinite": jnp.count_nonzero(jnp.isnan(flat) | jnp.isposinf(flat)).astype(jnp.float32),
    }

    return stats


@jax.tree_util.register_static
class Probe:
    """Collects named statistics while a model is traced. Call start() and collect() inside the traced function."""

    def __init__(self, *, level: Level = Level.FULL, logit_subsample_tokens: int = 64, routing_only: bool = False) -> None:
        self.level = level
        self.logit_subsample_tokens = logit_subsample_tokens
        self.routing_only = routing_only
        self._prefix = ""
        self._values: dict[str, Stats] = {}

    @property
    def enabled(self) -> bool:
        return self.level > Level.OFF

    @property
    def detailed(self) -> bool:
        """Whether to record activations, which costs more than routing statistics."""
        return self.enabled and not self.routing_only

    def scope(self, name: str) -> Probe:
        """A probe that records under `prefix.name` into the same store."""
        if not self.enabled:
            return self
        scoped = copy.copy(self)
        scoped._prefix = f"{self._prefix}.{name}" if self._prefix else name
        return scoped

    def start(self) -> None:
        self._values.clear()

    def collect(self) -> dict[str, Stats]:
        values = dict(self._values)
        self._values.clear()
        return values

    def _add(self, name: str, stats: Stats) -> None:
        self._values[f"{self._prefix}.{name}" if self._prefix else name] = stats

    def tensor(self, name: str, x: Array) -> None:
        if self.detailed:
            self._add(name, tensor_stats(x, level=self.level))

    def residual(self, name: str, x: Array, delta: Array) -> None:
        if self.detailed:
            self._add(name, residual_stats(x, delta))

    def swiglu(self, name: str, x: Array, *, limit: float) -> None:
        if self.detailed:
            self._add(name, swiglu_stats(x, limit, level=self.level))

    def scalars(self, name: str, stats: Mapping[str, Array]) -> None:
        if self.enabled and (self.detailed or name == "aux_loss"):
            self._add(name, {key: jnp.asarray(value, jnp.float32) for key, value in stats.items()})

    def routing(self, name: str, **kwargs) -> None:
        if self.enabled:
            self._add(name, routing_stats(**kwargs))

    def logits(self, name: str, logits: Array) -> None:
        if self.detailed:
            self._add(name, logit_stats(logits))

    def logit_subsample(self, name: str, hidden: Array, unemb: Array, *, vocab_size: int) -> None:
        """Logit statistics for a few evenly spaced tokens, since training never materializes the full logits."""
        if not self.detailed:
            return
        flat = hidden.reshape(-1, hidden.shape[-1])
        sample = flat[jnp.linspace(0, flat.shape[0] - 1, min(self.logit_subsample_tokens, flat.shape[0]), dtype=jnp.int32)]
        z = jnp.einsum("tc,vc->tv", sample.astype(jnp.float32), unemb.astype(jnp.float32)).at[:, vocab_size:].set(MASKED_LOGIT)
        self._add(name, logit_stats(z))


NO_PROBE = Probe(level=Level.OFF)


def to_floats(values: Mapping[str, Stats]) -> dict[str, dict[str, float]]:
    """Pull collected statistics off-device into plain floats."""
    host = jax.device_get(dict(values))
    return {path: {key: float(value) for key, value in stats.items()} for path, stats in host.items()}
