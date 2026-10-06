from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyra.model import ModelWeights, model_apply
from lyra.training.probe import NO_PROBE, Level, Probe, swiglu_stats
from lyra.variants import model_variant


def assert_unchanged(actual, expected) -> None:
    """Probing must not change the computation: bit-exact on CPU. On TPU, a probe changes how XLA fuses
    the model, which moves BF16 rounding, so compare by relative L2 error instead."""
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if jax.default_backend() == "cpu":
        np.testing.assert_array_equal(actual, expected)
        return
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite)
    error = np.linalg.norm(actual[finite] - expected[finite]) / max(np.linalg.norm(expected[finite]), 1e-30)
    assert error < 1e-2, f"relative L2 error {error:.2e}"


@pytest.mark.parametrize(("mlp", "level"), [("dense", Level.FULL), ("moe", Level.BASIC)])
def test_probe_records_and_preserves_model_output(mlp: str, level: Level) -> None:
    config = replace(model_variant("dev"), seq_len=8, n_layers=1, mlp_map=(mlp,))
    model = ModelWeights.init(jax.random.key(0), config)
    tokens = jnp.arange(config.seq_len, dtype=jnp.int32)[None, :]
    probe = Probe(level=level)

    def run(model: ModelWeights):
        probe.start()
        logits, aux, _ = model_apply(tokens, model, None, config, probe=probe)
        return logits, aux, probe.collect()

    logits, aux, stats = jax.jit(run)(model)
    normal_logits, normal_aux, _ = jax.jit(lambda m: model_apply(tokens, m, None, config))(model)

    assert_unchanged(logits, normal_logits)
    assert_unchanged(aux, normal_aux)
    if mlp == "moe":
        assert "layer.00.mlp.routing" in stats
    else:
        assert 0 <= stats["layer.00.mlp.swiglu"]["gate_clip_frac"] <= 1
        assert "gate_clip_energy_frac" in stats["layer.00.mlp.swiglu"]
    assert "layer.00.attn.residual" in stats
    assert stats["layer.00.attn.q"]["rms"] > 0
    assert stats["layer.00.attn.k"]["rms"] > 0
    assert "layer.00.attn.qk" not in stats
    removed = {"min", "max", "frac_zero", "chan_rms_min", "row_rms_min", "token_cosine_min", "predicted_out_rms"}
    assert all(removed.isdisjoint(values) for values in stats.values())
    assert "head.logits" in stats
    assert probe.collect() == {}


def test_routing_only_probe_skips_activations() -> None:
    probe = Probe(level=Level.BASIC, routing_only=True)
    probe.start()
    probe.tensor("hidden", jnp.ones((2, 2)))
    probe.scalars("aux_loss", {"loss": jnp.array(1.0)})
    assert list(probe.collect()) == ["aux_loss"]


def test_probe_preserves_gradients() -> None:
    config = replace(model_variant("dev"), seq_len=4, n_layers=1, mlp_map=("dense",))
    model = ModelWeights.init(jax.random.key(1), config)
    tokens = jnp.arange(config.seq_len, dtype=jnp.int32)[None, :]
    probe = Probe(level=Level.FULL)

    def loss(model: ModelWeights, enabled: bool):
        if enabled:
            probe.start()
        hidden, aux, _ = model_apply(tokens, model, None, config, return_hidden_states=True, probe=probe if enabled else NO_PROBE)
        if enabled:
            probe.collect()
        return jnp.sum(hidden) + aux

    normal = jax.grad(lambda m: loss(m, False))(model)
    probed = jax.grad(lambda m: loss(m, True))(model)
    for left, right in zip(jax.tree.leaves(normal), jax.tree.leaves(probed), strict=True):
        assert_unchanged(left, right)


def test_swiglu_clipping_matches_the_activation_bounds() -> None:
    # Negative gate values survive; values clip at both ends. Equal bounds survive.
    x = jnp.array([[-14.0, 7.0, 14.0, -14.0, 7.0, 14.0]])
    stats = jax.jit(swiglu_stats)(x, 7.0)
    np.testing.assert_allclose(stats["gate_clip_frac"], 1 / 3)
    np.testing.assert_allclose(stats["value_clip_frac"], 2 / 3)
    np.testing.assert_allclose(stats["gate_clip_energy_frac"], 1 / 3)
    np.testing.assert_allclose(stats["value_clip_energy_frac"], 2 / 3)
    assert all(float(value) == 0 for value in swiglu_stats(jnp.zeros_like(x), 7.0).values())
    rounded = swiglu_stats(jnp.array([[7, 14, 7, 14]], dtype=jnp.bfloat16), 6.99)
    assert float(rounded["gate_clip_frac"]) == 0.5
    assert float(rounded["value_clip_frac"]) == 0.5
    np.testing.assert_allclose(rounded["gate_clip_energy_frac"], 0.6)
    np.testing.assert_allclose(rounded["value_clip_energy_frac"], 0.6)
