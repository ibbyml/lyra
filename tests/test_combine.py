from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np

from lyra.model import bpermute

combine = import_module("lyra.kernels.combine")


def test_combine_preserves_reference_backward(monkeypatch) -> None:
    assignments = jnp.array([0, 2, 4, 7, 1, 3, 4, 7], jnp.int32)
    groups = jnp.bincount(assignments, length=8)
    permute, unpermute = bpermute(assignments, groups, 8)
    experts = jnp.arange(64, dtype=jnp.bfloat16).reshape(8, 8)
    scores = jnp.array([[0.25, 0.25, 0.5, 1.0], [0.5, 0.25, 0.25, 1.0]])

    def forward(experts, indices, weights, **kwargs):
        return combine.combine_reference(experts, weights, permute, indices)

    monkeypatch.setattr(combine, "_gather_reduce", forward)
    reference = lambda x, s: combine.combine_reference(x, s, permute, unpermute)
    candidate = lambda x, s: combine.combine(x, s, permute, unpermute)
    expected, reference_backward = jax.vjp(reference, experts, scores)
    actual, backward = jax.vjp(candidate, experts, scores)
    np.testing.assert_array_equal(actual, expected)
    for value, target in zip(backward(jnp.ones_like(actual)), reference_backward(jnp.ones_like(expected)), strict=True):
        np.testing.assert_array_equal(value, target)
        assert value.dtype == target.dtype


def test_combine_selector_requires_validated_shape_and_dtype(monkeypatch) -> None:
    monkeypatch.setattr(combine, "current_tpu_target", lambda: "v6e")
    experts = jax.ShapeDtypeStruct((65536, 2048), jnp.bfloat16)
    scores = jax.ShapeDtypeStruct((16384, 4), jnp.float32)
    assert combine.supports_combine(experts, scores)
    assert not combine.supports_combine(jax.ShapeDtypeStruct(experts.shape, jnp.float32), scores)
    assert not combine.supports_combine(experts, jax.ShapeDtypeStruct(scores.shape, jnp.bfloat16))
    assert not combine.supports_combine(jax.ShapeDtypeStruct((131072, 2048), jnp.bfloat16), scores)
    monkeypatch.setattr(combine, "current_tpu_target", lambda: "v5p")
    assert not combine.supports_combine(experts, scores)
