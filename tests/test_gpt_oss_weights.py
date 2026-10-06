from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from safetensors.flax import save_file

from lyra.gpt_oss import load_gpt_oss_model
from lyra.model import ModelConfig, model_apply
from lyra.nn.params import no_sharding


def _tiny_gpt_oss_config() -> ModelConfig:
    return ModelConfig(
        name="Tiny GPT-OSS",
        seq_len=16,
        n_embd=32,
        n_layers=1,
        mlp_map=("moe",),
        vocab_size=16,
        padded_vocab_size=32,
        norm_eps=1e-5,
        head_dim=4,
        n_attn_heads=2,
        n_kv_heads=1,
        sliding_window=4,
        n_routed_experts=2,
        n_shared_experts=0,
        n_active_experts=1,
        expert_mlp_widening=2.0,
        swiglu_beta=1.702,
        swiglu_limit=7.0,
        rope_base=150000.0,
        rope_scale=32.0,
        ntk_alpha=1.0,
        ntk_beta=32.0,
        sharding=no_sharding(),
        tie_embeddings=False,
        use_learned_xsa=False,
        use_qk_norm=False,
        use_qkv_bias=True,
        use_attn_out_bias=True,
        use_mlp_bias=True,
        use_router_bias=True,
        param_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        implementation="xla",
    )


def _write_tiny_checkpoint(path, config: ModelConfig) -> None:
    path.mkdir(exist_ok=True)

    q_dim = config.n_attn_heads * config.head_dim
    qkv_dim = q_dim + 2 * config.n_kv_heads * config.head_dim
    in_hidden = int(config.n_embd * config.expert_mlp_widening)
    out_hidden = in_hidden // 2
    bf16 = jnp.bfloat16

    tensors = {
        "embedding.weight": jnp.arange(config.vocab_size * config.n_embd, dtype=bf16).reshape(config.vocab_size, config.n_embd),
        "norm.scale": jnp.ones((config.n_embd,), dtype=bf16),
        "unembedding.weight": jnp.arange(config.vocab_size * config.n_embd, dtype=bf16).reshape(config.vocab_size, config.n_embd),
        "block.0.attn.norm.scale": jnp.ones((config.n_embd,), dtype=bf16),
        "block.0.attn.out.bias": jnp.arange(config.n_embd, dtype=bf16),
        "block.0.attn.out.weight": jnp.arange(config.n_embd * q_dim, dtype=bf16).reshape(config.n_embd, q_dim),
        "block.0.attn.qkv.bias": jnp.arange(qkv_dim, dtype=bf16),
        "block.0.attn.qkv.weight": jnp.arange(qkv_dim * config.n_embd, dtype=bf16).reshape(qkv_dim, config.n_embd),
        "block.0.attn.sinks": jnp.arange(config.n_attn_heads, dtype=bf16),
        "block.0.mlp.gate.bias": jnp.arange(config.n_experts, dtype=bf16),
        "block.0.mlp.gate.weight": jnp.arange(config.n_experts * config.n_embd, dtype=bf16).reshape(config.n_experts, config.n_embd),
        "block.0.mlp.mlp1_bias": jnp.zeros((config.n_experts, in_hidden), dtype=bf16),
        "block.0.mlp.mlp1_weight.blocks": jnp.full(
            (config.n_experts, in_hidden, config.n_embd // 32, 16),
            0x22,
            dtype=jnp.uint8,
        ),
        "block.0.mlp.mlp1_weight.scales": jnp.full(
            (config.n_experts, in_hidden, config.n_embd // 32),
            127,
            dtype=jnp.uint8,
        ),
        "block.0.mlp.mlp2_bias": jnp.zeros((config.n_experts, config.n_embd), dtype=bf16),
        "block.0.mlp.mlp2_weight.blocks": jnp.full(
            (config.n_experts, config.n_embd, out_hidden // 32, 16),
            0x22,
            dtype=jnp.uint8,
        ),
        "block.0.mlp.mlp2_weight.scales": jnp.full(
            (config.n_experts, config.n_embd, out_hidden // 32),
            127,
            dtype=jnp.uint8,
        ),
        "block.0.mlp.norm.scale": jnp.ones((config.n_embd,), dtype=bf16),
    }
    save_file(tensors, str(path / "model.safetensors"))


def test_load_tiny_gpt_oss_checkpoint(tmp_path):
    config = _tiny_gpt_oss_config()
    _write_tiny_checkpoint(tmp_path, config)

    weights = load_gpt_oss_model(tmp_path, config)
    layer = weights.layers[0]

    assert weights.tok.emb.shape == (config.padded_vocab_size, config.n_embd)
    assert weights.tok.unemb is not None
    assert weights.tok.unemb.shape == (config.padded_vocab_size, config.n_embd)
    assert layer.attn.q.shape == (config.n_embd, 8)
    assert layer.attn.k.shape == (config.n_embd, 4)
    assert layer.attn.v.shape == (config.n_embd, 4)
    assert layer.attn.q_bias is not None
    assert layer.attn.k_bias is not None
    assert layer.attn.v_bias is not None
    assert layer.attn.out_bias is not None
    assert layer.mlp.router_bias is not None
    assert layer.mlp.mlp_up.shape == (config.n_experts, config.n_embd, 2, 32)
    assert layer.mlp.mlp_down.shape == (config.n_experts, 32, config.n_embd)
    assert jax.device_get(layer.attn.q[1, 0]) == jax.device_get(jnp.asarray(1, dtype=jnp.bfloat16))
    assert jax.device_get(layer.attn.k[0, 0]) == jax.device_get(jnp.asarray(256, dtype=jnp.bfloat16))
    assert jax.device_get(layer.attn.v[0, 0]) == jax.device_get(jnp.asarray(384, dtype=jnp.bfloat16))
    assert jnp.array_equal(layer.attn.q_bias, jnp.arange(8, dtype=jnp.bfloat16))
    assert jnp.array_equal(layer.attn.k_bias, jnp.arange(8, 12, dtype=jnp.bfloat16))
    assert jnp.array_equal(layer.attn.v_bias, jnp.arange(12, 16, dtype=jnp.bfloat16))
    assert jnp.all(layer.mlp.mlp_up == jnp.asarray(1.0, dtype=jnp.bfloat16))


def test_load_pads_embedding_and_unembedding_without_changing_real_logits(tmp_path):
    config = _tiny_gpt_oss_config()
    _write_tiny_checkpoint(tmp_path, config)

    weights = load_gpt_oss_model(tmp_path, config)
    checkpoint_emb = jnp.arange(config.vocab_size * config.n_embd, dtype=jnp.bfloat16).reshape(config.vocab_size, config.n_embd)
    checkpoint_unemb = jnp.arange(config.vocab_size * config.n_embd, dtype=jnp.bfloat16).reshape(config.vocab_size, config.n_embd)

    assert weights.tok.unemb is not None
    assert jnp.array_equal(weights.tok.emb[: config.vocab_size], checkpoint_emb)
    assert jnp.array_equal(weights.tok.unemb[: config.vocab_size], checkpoint_unemb)
    assert jnp.count_nonzero(weights.tok.emb[config.vocab_size :]) == 0
    assert jnp.count_nonzero(weights.tok.unemb[config.vocab_size :]) == 0

    hidden = jnp.arange(config.n_embd, dtype=jnp.float32)
    loaded_logits = hidden @ weights.tok.unemb.astype(jnp.float32).T
    checkpoint_logits = hidden @ checkpoint_unemb.astype(jnp.float32).T
    assert jnp.array_equal(loaded_logits[: config.vocab_size], checkpoint_logits)
    assert jnp.count_nonzero(loaded_logits[config.vocab_size :]) == 0

    logits, _, _ = model_apply(jnp.asarray([[0, 1]], dtype=jnp.int32), weights, None, config)
    assert jnp.isfinite(logits[..., : config.vocab_size]).all()
    assert jnp.isneginf(logits[..., config.vocab_size :]).all()


def test_mxfp4_experts_stay_bf16_regardless_of_param_dtype(tmp_path):
    config = replace(_tiny_gpt_oss_config(), param_dtype=jnp.float32)
    _write_tiny_checkpoint(tmp_path, config)

    weights = load_gpt_oss_model(tmp_path, config)
    layer = weights.layers[0]

    assert layer.mlp.mlp_up.dtype == jnp.bfloat16
    assert layer.mlp.mlp_down.dtype == jnp.bfloat16
    # Non-expert weights still honor param_dtype.
    assert weights.tok.emb.dtype == jnp.float32
