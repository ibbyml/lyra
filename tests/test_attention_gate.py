from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from lyra.model import ModelWeights, model_apply
from lyra.variants import lyra_dev


def test_attention_gate_starts_as_identity_and_learns() -> None:
    config = replace(
        lyra_dev,
        seq_len=8,
        vocab_size=32,
        padded_vocab_size=32,
        n_layers=2,
        mlp_map=("dense", "dense"),
        implementation="xla",
        compute_dtype=jnp.float32,
        gemm_dtype=jnp.float32,
    )
    tokens = jnp.arange(8, dtype=jnp.int32)[None, :]
    key = jax.random.key(0)
    base = ModelWeights.init(key, config)
    base_logits, _, _ = model_apply(tokens, base, None, config)

    gated_config = replace(config, use_sdpa_output_gate=True)
    gated = ModelWeights.init(key, gated_config)
    np.testing.assert_array_equal(np.asarray(base.tok.emb), np.asarray(gated.tok.emb))
    for original, changed in zip(base.layers, gated.layers, strict=True):
        np.testing.assert_array_equal(np.asarray(original.attn.q), np.asarray(changed.attn.q))
        np.testing.assert_array_equal(np.asarray(original.mlp.mlp_up), np.asarray(changed.mlp.mlp_up))
        assert np.all(np.asarray(changed.attn.sdpa_gate) == 0)
    gate_logits, _, _ = model_apply(tokens, gated, None, gated_config)
    np.testing.assert_array_equal(np.asarray(base_logits), np.asarray(gate_logits))

    gate_loss = lambda params: jnp.mean(model_apply(tokens, params, None, gated_config)[0] ** 2)
    gate_grads = jax.grad(gate_loss)(gated)
    assert float(jnp.linalg.norm(gate_grads.layers[0].attn.sdpa_gate)) > 0
    first = gated.layers[0]
    updated_first = replace(first, attn=replace(first.attn, sdpa_gate=first.attn.sdpa_gate - 0.01 * gate_grads.layers[0].attn.sdpa_gate))
    updated = replace(gated, layers=[updated_first, *gated.layers[1:]])
    assert float(gate_loss(updated)) < float(gate_loss(gated))
