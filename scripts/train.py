from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal

import tyro
from etils import epath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Args:
    variant: str = "lyra-small"
    """Model variant."""
    run: str | None = None
    """Training preset; defaults to the variant's (lyra-small -> train-small)."""
    device: Literal["auto", "cpu", "tpu", "gpu"] = "auto"
    """JAX platform."""
    print_config: bool = False
    """Print the resolved configuration and exit."""
    seed: int | None = None
    """Initialization and data seed."""
    steps: int | None = None
    """Total optimizer steps, including any already run."""
    batch_size: int | None = None
    """Sequences per microbatch."""
    acc_steps: int | None = None
    """Microbatches per optimizer step."""
    data: str | None = None
    """Training data."""
    eval_data: str | None = None
    """Validation data."""
    out: str | None = None
    """Checkpoint directory (local or gs://)."""
    resume: str | None = None
    """Checkpoint, or run directory, to resume from."""
    eval_every: int | None = None
    """Validation interval in steps."""
    eval_batches: int | None = None
    """Validation batches per evaluation."""
    sample_every: int | None = None
    """Prompt sampling interval in steps."""
    sample_prompts: tuple[str, ...] | None = None
    """Prompts to sample."""
    numerics_every: int | None = None
    """Routing diagnostics interval in steps."""
    chkpt_every: int | None = None
    """Checkpoint interval in steps."""
    chkpt_keep: int | None = None
    """Checkpoints to keep."""
    decay_type: Literal["linear", "cosine"] | None = None
    """Learning-rate decay."""
    warmup_fraction: float | None = None
    """Fraction of steps spent warming up."""
    stable_fraction: float | None = None
    """Fraction of steps held at the peak learning rate."""
    adam_lr: float | None = None
    """Peak Adam learning rate."""
    muon_lr: float | None = None
    """Peak Muon learning rate."""
    grad_clip: float | None = None
    """Per-matrix Muon gradient RMS limit."""

FIELDS = {
    "steps": "steps",
    "batch_size": "batch_size",
    "acc_steps": "accumulation_steps",
    "eval_every": "eval_every",
    "eval_batches": "eval_batches",
    "sample_every": "sample_every",
    "sample_prompts": "sample_prompts",
    "numerics_every": "numerics_every",
    "chkpt_every": "checkpoint_every",
    "chkpt_keep": "checkpoint_keep",
    "decay_type": "decay_type",
    "warmup_fraction": "warmup_fraction",
    "stable_fraction": "stable_fraction",
    "adam_lr": "adam_max_lr",
    "muon_lr": "muon_max_lr",
    "grad_clip": "muon_grad_clip",
}
PATH_FIELDS = {"data": "data_path", "eval_data": "eval_data_path", "out": "checkpoint_path", "resume": "resume_from"}


def training_config(args: Args):
    from lyra.training.presets import TRAINING_PRESETS

    run = args.run or f"train-{args.variant.removeprefix('lyra-')}"
    if run not in TRAINING_PRESETS:
        raise SystemExit(f"error: no training preset {run!r}; pass --run (choices: {', '.join(TRAINING_PRESETS)})")
    config = TRAINING_PRESETS[run]
    overrides = {field: getattr(args, flag) for flag, field in FIELDS.items() if getattr(args, flag) is not None}
    overrides |= {
        field: epath.Path(getattr(args, flag)).expanduser() for flag, field in PATH_FIELDS.items() if getattr(args, flag) is not None
    }
    if args.seed is not None:
        overrides["data_loader"] = replace(config.data_loader, seed=args.seed)
    return replace(config, **overrides)


def main() -> int:
    args = tyro.cli(Args)
    if args.device != "auto":
        os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else args.device
    os.makedirs(cache := os.path.join(ROOT, ".jax-cache"), exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache)
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")

    import jax

    from lyra.model import ModelWeights
    from lyra.training.train import train
    from lyra.variants import model_variant

    tcfg = training_config(args)
    mcfg = model_variant(args.variant)
    if args.seed is not None:
        mcfg = replace(mcfg, seed=args.seed)
    if args.print_config:
        from wadler_lindig import pprint

        pprint(mcfg, hide_defaults=False)
        pprint(tcfg, hide_defaults=False)
        return 0

    print(f"Training {mcfg.name} for {tcfg.steps:,} steps")
    with jax.set_mesh(mcfg.sharding.get_mesh()):
        model = None if tcfg.resume_from is not None else ModelWeights.init(jax.random.key(mcfg.seed), mcfg)
        train(model, mcfg, tcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
