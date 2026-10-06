from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import tyro

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Args:
    variant: str = "lyra-small"
    """Model variant."""
    weights: str | None = None
    """Checkpoint path; defaults to chkpt/<variant>."""
    tasks: tuple[Literal["mmlu", "arc-challenge", "arc-easy", "hellaswag"], ...] = ("mmlu", "arc-challenge", "arc-easy", "hellaswag")
    """Evaluation tasks."""
    format: Literal["orbax", "gpt-oss"] = "orbax"
    """Checkpoint format."""
    device: Literal["auto", "cpu", "tpu", "gpu"] = "auto"
    """JAX platform."""
    batch_size: int = 8
    """Sequences per forward pass."""
    limit: int | None = None
    """Maximum documents per task."""
    fewshot: dict[str, int] = field(default_factory=dict)
    """Few-shot examples per task; tasks not listed are zero-shot."""
    fewshot_seed: int = 1234
    """Seed for choosing ARC and HellaSwag few-shot examples."""


def main() -> int:
    args = tyro.cli(Args)
    if args.device != "auto":
        os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else args.device

    import jax

    from lyra.evals import TASKS, Scorer, evaluate
    from lyra.gpt_oss import load_gpt_oss_model
    from lyra.model import ModelWeights
    from lyra.variants import model_variant

    config = model_variant(args.variant)
    weights = args.weights or os.path.join(ROOT, "chkpt", args.variant)
    with jax.set_mesh(config.sharding.get_mesh()):
        model = load_gpt_oss_model(weights, config) if args.format == "gpt-oss" else ModelWeights.load(weights, config)
        scorer = Scorer(model, config, batch_size=args.batch_size, harmony=args.format == "gpt-oss")
        results = []
        for task in args.tasks:
            shots = args.fewshot.get(task, 0)
            docs = TASKS[task](limit=args.limit, num_fewshot=shots, fewshot_seed=args.fewshot_seed)
            print(f"Running {task} ({shots}-shot) on {len(docs)} documents...", flush=True)
            results.append(evaluate(task, docs, scorer))
            print(results[-1], flush=True)

    print("\nResults")
    for result in results:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
