from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lyra.model import ModelWeights, model_apply
from lyra.variants import model_variant


@pytest.fixture(scope="module")
def dev_model():
    mcfg = model_variant("dev")
    model = ModelWeights.init(jax.random.key(0), mcfg)
    return model, mcfg


def test_forward_shape_and_finiteness(dev_model) -> None:
    model, mcfg = dev_model
    tokens = jnp.arange(mcfg.seq_len, dtype=jnp.int32)[None, :] % mcfg.vocab_size
    logits, aux_loss, cache = model_apply(tokens, model, None, mcfg)

    assert logits.shape == (1, mcfg.seq_len, mcfg.padded_vocab_size)
    assert cache is None  # NOCACHE mode
    # Real vocab logits are finite; the padding tail is masked to -inf.
    assert np.all(np.isfinite(np.asarray(logits[..., : mcfg.vocab_size])))
    assert np.all(np.asarray(logits[..., mcfg.vocab_size :]) == -np.inf)
    assert np.isfinite(float(aux_loss)) and float(aux_loss) >= 0.0


def test_forward_is_deterministic(dev_model) -> None:
    model, mcfg = dev_model
    tokens = jnp.ones((1, mcfg.seq_len), dtype=jnp.int32)
    a, _, _ = model_apply(tokens, model, None, mcfg)
    b, _, _ = model_apply(tokens, model, None, mcfg)
    assert np.array_equal(np.asarray(a), np.asarray(b))
