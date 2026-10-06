from lyra.training.train import CHECKPOINT_DIR, DATA_DIR, TrainingConfig


def _pretraining(variant: str, *, steps: int, accumulation_steps: int) -> TrainingConfig:
    return TrainingConfig(
        steps=steps,
        batch_size=4,
        accumulation_steps=accumulation_steps,
        decay_type="cosine",
        warmup_fraction=16 / 7630,
        stable_fraction=0.5,
        eval_data_path=DATA_DIR / "val",
        eval_every=steps // 100,
        eval_batches=16,
        sample_every=steps // 100,
        numerics_every=32,
        checkpoint_keep=3,
        checkpoint_path=CHECKPOINT_DIR / variant,
    )


TRAINING_PRESETS = {
    "train-dev": TrainingConfig(
        steps=100,
        batch_size=2,
        accumulation_steps=1,
        numerics_every=10,
        checkpoint_path=CHECKPOINT_DIR / "dev",
    ),
    "train-small": _pretraining("lyra-small", steps=308000, accumulation_steps=4),
    "train-medium": _pretraining("lyra-medium", steps=490000, accumulation_steps=16),
    "train-large": _pretraining("lyra-large", steps=353000, accumulation_steps=32),
    "train-max": _pretraining("lyra-max", steps=353000, accumulation_steps=32),
}
