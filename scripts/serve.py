from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import tyro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Args:
    variant: str = "lyra-small"
    """Model variant."""
    weights: str | None = None
    """Checkpoint path; defaults to chkpt/<variant>."""
    format: Literal["orbax", "gpt-oss"] = "orbax"
    """Checkpoint format."""
    device: Literal["auto", "cpu", "tpu", "gpu"] = "auto"
    """JAX platform."""
    max_new_tokens: int = 256
    """Tokens to generate per turn."""
    max_cache_length: int | None = None
    """Prompt plus generation length; defaults to the model's context length."""
    temperature: float | None = None
    """Sampling temperature; defaults to the variant's."""
    top_k: int | None = None
    """Top-k sampling; defaults to the variant's."""
    fp8: bool = False
    """Store MLP weights and the KV cache in block-scaled FP8."""


def main() -> int:
    args = tyro.cli(Args)
    if args.device != "auto":
        os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else args.device

    import jax

    from lyra.generate import InferenceConfig, apply_inference_overrides, load_model, serve_model
    from lyra.gpt_oss import load_gpt_oss_model
    from lyra.variants import model_variant

    icfg = InferenceConfig(
        weights=args.weights or os.path.join(ROOT, "chkpt", args.variant),
        max_new_tokens=args.max_new_tokens,
        max_cache_length=args.max_cache_length,
        temperature=args.temperature,
        top_k=args.top_k,
        fp8=args.fp8,
    )
    mcfg = apply_inference_overrides(model_variant(args.variant), icfg)
    with jax.set_mesh(mcfg.sharding.get_mesh()):
        model = load_gpt_oss_model(icfg.weights, mcfg) if args.format == "gpt-oss" else load_model(mcfg, icfg)
        serve_model(model, mcfg, icfg, harmony=args.format == "gpt-oss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
