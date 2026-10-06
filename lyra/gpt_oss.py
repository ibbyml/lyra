from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from safetensors import safe_open

if TYPE_CHECKING:
    from lyra.model import ModelConfig, MoEWeights

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _dequantize_mxfp4(blocks, scales) -> jax.Array:
    blocks = jnp.asarray(blocks, dtype=jnp.uint8)
    lut = jnp.asarray(FP4_VALUES, dtype=jnp.bfloat16)
    unpacked = jnp.stack((lut[blocks & 0x0F], lut[blocks >> 4]), axis=-1)
    scaled = jnp.ldexp(unpacked, (jnp.asarray(scales, dtype=jnp.int32) - 127)[..., None, None])
    return scaled.reshape(*blocks.shape[:-2], -1)


def _deinterleave_glu(x) -> jax.Array:
    return jnp.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1).reshape(*x.shape[:-1], 2, -1)


def load_gpt_oss_model(source: str | Path, config: ModelConfig):
    from lyra.model import ModelWeights

    path = Path(source).expanduser()
    if (path / "original").is_dir():
        path = path / "original"
    files = {}
    for file in sorted(path.glob("*.safetensors")):
        with safe_open(file, framework="flax") as f:
            files |= dict.fromkeys(f.keys(), file)

    def get(name: str):
        with safe_open(files[name], framework="flax") as f:
            return f.get_tensor(name)

    def place(value, spec: Any, dtype: jnp.dtype | None = None):
        assert tuple(value.shape) == spec.shape, (value.shape, spec.shape)
        value = jnp.asarray(value, dtype=dtype or spec.dtype)
        return jax.device_put(value, NamedSharding(jax.sharding.get_mesh(), spec.sharding)) if any(spec.sharding) else value

    def padded_vocab(name: str):
        value = get(name)
        return jnp.pad(value, ((0, config.padded_vocab_size - value.shape[0]), (0, 0)))

    spec = ModelWeights.spec(config)
    q_dim, kv_dim = config.n_attn_heads * config.head_dim, config.n_kv_heads * config.head_dim
    tok = replace(
        spec.tok,
        emb=place(padded_vocab("embedding.weight"), spec.tok.emb),
        unemb=place(padded_vocab("unembedding.weight"), spec.tok.unemb),
        norm_scale=place(get("norm.scale"), spec.tok.norm_scale),
    )
    layers = []
    for index, layer in enumerate(spec.layers):
        block, attn, mlp = f"block.{index}", layer.attn, cast("MoEWeights", layer.mlp)  # GPT-OSS is MoE in every layer.
        q, k, v = jnp.split(get(f"{block}.attn.qkv.weight"), [q_dim, q_dim + kv_dim])
        q_bias, k_bias, v_bias = jnp.split(get(f"{block}.attn.qkv.bias"), [q_dim, q_dim + kv_dim])
        attn = replace(
            attn,
            q=place(q.T, attn.q),
            k=place(k.T, attn.k),
            v=place(v.T, attn.v),
            q_bias=place(q_bias, attn.q_bias),
            k_bias=place(k_bias, attn.k_bias),
            v_bias=place(v_bias, attn.v_bias),
            sinks=place(get(f"{block}.attn.sinks"), attn.sinks),
            out=place(get(f"{block}.attn.out.weight").T, attn.out),
            out_bias=place(get(f"{block}.attn.out.bias"), attn.out_bias),
            norm_scale=place(get(f"{block}.attn.norm.scale"), attn.norm_scale),
        )
        mlp_up = _dequantize_mxfp4(get(f"{block}.mlp.mlp1_weight.blocks"), get(f"{block}.mlp.mlp1_weight.scales")).transpose(0, 2, 1)
        mlp_down = _dequantize_mxfp4(get(f"{block}.mlp.mlp2_weight.blocks"), get(f"{block}.mlp.mlp2_weight.scales")).transpose(0, 2, 1)
        mlp = replace(
            mlp,
            router=place(get(f"{block}.mlp.gate.weight").T, mlp.router),
            router_bias=place(get(f"{block}.mlp.gate.bias"), mlp.router_bias),
            mlp_up=place(_deinterleave_glu(mlp_up), mlp.mlp_up, jnp.bfloat16),
            mlp_up_bias=place(_deinterleave_glu(get(f"{block}.mlp.mlp1_bias")), mlp.mlp_up_bias),
            mlp_down=place(mlp_down, mlp.mlp_down, jnp.bfloat16),
            mlp_down_bias=place(get(f"{block}.mlp.mlp2_bias"), mlp.mlp_down_bias),
            norm_scale=place(get(f"{block}.mlp.norm.scale"), mlp.norm_scale),
        )
        layers.append(replace(layer, attn=attn, mlp=mlp))
    return ModelWeights.quantize(replace(spec, tok=tok, layers=layers), config)
