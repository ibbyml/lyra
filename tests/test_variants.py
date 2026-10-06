from __future__ import annotations

import pytest

from lyra.variants import MODEL_VARIANTS


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_expert_bookkeeping(name: str) -> None:
    c = MODEL_VARIANTS[name]
    assert c.n_experts == c.n_routed_experts + c.n_shared_experts
    assert c.n_shared_experts <= c.n_active_experts <= c.n_experts
    assert c.n_active_routed_experts == c.n_active_experts - c.n_shared_experts
    assert c.n_active_routed_experts >= 0


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_layer_schedule(name: str) -> None:
    c = MODEL_VARIANTS[name]
    assert len(c.mlp_map) == c.n_layers
    assert set(c.mlp_map) <= {"dense", "moe"}
    assert c.n_moe_layers == c.mlp_map.count("moe")


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_attention_head_divisibility(name: str) -> None:
    c = MODEL_VARIANTS[name]
    assert c.n_attn_heads > 0 and c.n_kv_heads > 0 and c.head_dim > 0
    assert c.n_attn_heads % c.n_kv_heads == 0, "GQA needs whole query groups per KV head"
    # Tensor parallelism splits heads across the model axis.
    model_axis = c.sharding.model_axis_size
    assert c.n_attn_heads % model_axis == 0
    assert c.n_kv_heads % model_axis == 0


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_expert_axis_divides_experts(name: str) -> None:
    c = MODEL_VARIANTS[name]
    expert_axis = c.sharding.expert_axis_size
    assert c.n_experts % expert_axis == 0, f"{name}: {c.n_experts} experts across {expert_axis} EP shards"


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_model_axis_divides_moe_hidden(name: str) -> None:
    c = MODEL_VARIANTS[name]
    model_axis = c.sharding.model_axis_size
    out_hidden = int(c.n_embd * c.expert_mlp_widening) // 2
    assert c.n_embd % model_axis == 0, "down projection shards the embedding dim"
    assert out_hidden % model_axis == 0, "up projection shards the expert hidden dim"


@pytest.mark.parametrize("name", sorted(MODEL_VARIANTS))
def test_vocab_padding(name: str) -> None:
    c = MODEL_VARIANTS[name]
    assert c.padded_vocab_size >= c.vocab_size
