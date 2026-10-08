from dataclasses import replace

from lyra.model import ModelConfig
from lyra.nn.params import data_parallel, expert_parallel, full_moe, interleaved_moe, moe_outer_dense, no_sharding, tensor_parallel

lyra_dev = ModelConfig(
    name="Lyra Dev",
    seq_len=256,
    n_embd=16,
    n_layers=16,
    n_attn_heads=2,
    n_kv_heads=1,
    head_dim=8,
    sliding_window=0,
    n_routed_experts=7,
    n_shared_experts=1,
    n_active_experts=2,
    mlp_map=interleaved_moe(layers=16, first=4, every=4),
    sharding=no_sharding(),
)

lyra_small = ModelConfig(
    name="Lyra Small",
    use_sdpa_output_gate=True,
    mlp_map=interleaved_moe(layers=16, every=4, first=4),
    sharding=data_parallel(),
)

lyra_medium = ModelConfig(
    name="Lyra Medium",
    seq_len=4096,
    n_embd=4096,
    n_layers=32,
    tie_embeddings=False,
    n_attn_heads=32,
    n_kv_heads=8,
    head_dim=128,
    use_learned_xsa=True,
    use_sdpa_output_gate=True,
    mlp_map=interleaved_moe(32, first=4, every=4),
    n_routed_experts=31,
    sharding=expert_parallel(fsdp=1, expert=8),
)

lyra_large = replace(
    lyra_medium,
    name="Lyra Large",
    mlp_map=moe_outer_dense(32, start_dense=2),
    n_routed_experts=31,
    dense_mlp_widening=4.0,
    sharding=expert_parallel(fsdp=1, expert=16),
)

lyra_max = replace(
    lyra_medium,
    name="Lyra Max",
    mlp_map=moe_outer_dense(32, start_dense=2),
    n_routed_experts=63,
    dense_mlp_widening=4.0,
    sharding=expert_parallel(fsdp=1, expert=32),
)

gpt_oss_20b = ModelConfig(
    name="GPT-OSS 20B",
    rope_scale=32.0,
    n_embd=2880,
    n_layers=24,
    head_dim=64,
    n_attn_heads=64,
    n_kv_heads=8,
    mlp_map=full_moe(layers=24),
    n_routed_experts=32,
    n_shared_experts=0,
    n_active_experts=4,
    sharding=tensor_parallel(data_axis=4, model_axis=2),
    tie_embeddings=False,
    use_qk_norm=False,
    use_qkv_bias=True,
    use_attn_out_bias=True,
    use_mlp_bias=True,
    use_router_bias=True,
)

gpt_oss_120b = replace(
    gpt_oss_20b,
    name="GPT-OSS 120B",
    n_layers=36,
    mlp_map=full_moe(layers=36),
    n_routed_experts=128,
)

MODEL_VARIANTS: dict[str, ModelConfig] = {
    "dev": lyra_dev,
    "lyra-small": lyra_small,
    "lyra-medium": lyra_medium,
    "lyra-large": lyra_large,
    "lyra-max": lyra_max,
    "gpt-oss-20b": gpt_oss_20b,
    "gpt-oss-120b": gpt_oss_120b,
}


def model_variant(variant: str) -> ModelConfig:
    try:
        return MODEL_VARIANTS[variant]
    except KeyError:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(f"Unknown model variant: {variant!r} (choices: {choices})") from None
