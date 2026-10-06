from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp

from lyra.evals.common import Scorer
from lyra.generate import InferenceConfig, apply_inference_overrides, init_cache, sample_token, sample_tokens
from lyra.model import ModelWeights, model_apply
from lyra.nn.quant import QArray
from lyra.training.steps import sample_eval_step
from lyra.variants import model_variant


def test_overrides_apply_sampling_and_fp8() -> None:
    out = apply_inference_overrides(model_variant("dev"), InferenceConfig(weights="unused", temperature=0.3, top_k=17, fp8=True))
    assert out.sample_temperature == 0.3
    assert out.sample_topk == 17
    assert out.gemm_dtype == jnp.float8_e4m3fn
    assert out.quantize_mlp and out.quantize_cache


def test_overrides_are_noop_when_unset() -> None:
    mcfg = model_variant("dev")
    assert apply_inference_overrides(mcfg, InferenceConfig(weights="unused")) is mcfg


def test_greedy_is_argmax() -> None:
    cfg = replace(model_variant("dev"), sample_temperature=0.0)
    assert int(sample_token(jnp.array([0.1, 5.0, -2.0, 4.9]), jax.random.key(0), cfg)) == 1


def test_top_k_masks_to_the_allowed_set() -> None:
    cfg = replace(model_variant("dev"), sample_temperature=1.0, sample_topk=2)
    logits = jnp.array([0.0, 10.0, 0.5, 9.5, -3.0])
    assert {int(sample_token(logits, jax.random.key(s), cfg)) for s in range(50)} <= {1, 3}


def test_top_k_keeps_exactly_k_tied_logits() -> None:
    cfg = replace(model_variant("dev"), sample_temperature=1.0, sample_topk=1)
    assert len({int(sample_token(jnp.zeros(8), jax.random.key(s), cfg)) for s in range(128)}) == 1


def test_fp8_dev_model_and_cache_execute() -> None:
    base = replace(model_variant("dev"), seq_len=8, implementation="xla", sample_temperature=0.0)
    config = apply_inference_overrides(base, InferenceConfig(weights="unused", fp8=True))
    with jax.set_mesh(config.sharding.get_mesh()):
        model = ModelWeights.quantize(ModelWeights.init(jax.random.key(0), config), config)
        cache = init_cache(config, 8)
        logits, _, _ = jax.jit(lambda tokens: model_apply(tokens, model, None, config))(jnp.arange(8, dtype=jnp.int32)[None, :])

    assert isinstance(model.layers[4].mlp.mlp_up, QArray)
    assert isinstance(cache.layers[0].k, QArray)
    assert bool(jnp.isfinite(logits[..., : config.vocab_size]).all())


def test_sample_tokens_runs_eagerly() -> None:
    config = replace(model_variant("dev"), seq_len=8, implementation="xla", sample_temperature=0.0)
    with jax.set_mesh(config.sharding.get_mesh()):
        model = ModelWeights.init(jax.random.key(0), config)
        sampled = sample_tokens(model, jax.random.key(1), config, jnp.asarray([1, 2], jnp.int32), jnp.int32(2), 2)
    assert sampled.shape == (1, 2)


def test_fixed_prompt_sampling_replays_with_one_padded_shape() -> None:
    config = replace(model_variant("dev"), seq_len=8, vocab_size=128, padded_vocab_size=128, implementation="xla")
    with jax.set_mesh(config.sharding.get_mesh()):
        model = ModelWeights.init(jax.random.key(0), config)
        short = jnp.asarray([1, 2, 0], dtype=jnp.int32)
        long = jnp.asarray([1, 3, 4], dtype=jnp.int32)
        first = sample_eval_step(model, short, jnp.int32(2), jnp.int32(0), 3, config)
        repeated = sample_eval_step(model, short, jnp.int32(2), jnp.int32(0), 3, config)
        other = sample_eval_step(model, long, jnp.int32(3), jnp.int32(1), 3, config)

    assert first.shape == other.shape == (1, 3)
    assert jnp.array_equal(first, repeated)
    assert jnp.all((0 <= other) & (other < config.vocab_size))


def test_harmony_scorer_renders_assistant_final_header() -> None:
    scorer = Scorer(cast(ModelWeights, None), model_variant("dev"), harmony=True)
    assert scorer.harmony is not None
    context, continuation = scorer._encode("Question: 1 + 1?", " 2")
    rendered = scorer.harmony.decode_utf8(context + continuation)
    assert rendered.startswith("<|start|>user<|message|>Question: 1 + 1?<|end|><|start|>assistant")
    assert "<|channel|>final<|message|> 2" in rendered
