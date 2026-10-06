from __future__ import annotations

import importlib
import json
from dataclasses import replace

import jax
import numpy as np
import pytest
from etils.epath import Path

from lyra.data import DataLoaderConfig
from lyra.model import ModelWeights
from lyra.nn.checkpoints import checkpoint_run_config, checkpoint_steps
from lyra.training.plotting import _routing_series
from lyra.training.state import load_train_state
from lyra.training.train import TrainHooks, TrainingConfig, train
from lyra.variants import lyra_dev

TINY = replace(lyra_dev, seq_len=4, n_layers=1, mlp_map=("dense",), vocab_size=32, padded_vocab_size=32, implementation="xla")


def _records(run: Path, kind: str) -> list[int]:
    return [row["step"] for row in map(json.loads, (run / "metrics.jsonl").read_text().splitlines()) if row["kind"] == kind]


def test_resumed_run_matches_uninterrupted_training(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(importlib.import_module("lyra.training.train"), "plot_run", lambda _: None)
    np.save(tmp_path / "train.npy", np.arange(513, dtype=np.int32) % 32)
    np.save(tmp_path / "val.npy", np.arange(129, dtype=np.int32)[::-1] % 32)
    config = TrainingConfig(
        steps=6,
        batch_size=1,
        accumulation_steps=2,
        warmup_fraction=1 / 6,
        stable_fraction=0.5,
        data_path=Path(tmp_path / "train.npy"),
        data_loader=DataLoaderConfig(worker_count=0, shuffle=True),
        eval_data_path=Path(tmp_path / "val.npy"),
        eval_every=2,
        eval_batches=2,
        checkpoint_every=2,
        checkpoint_keep=3,
        checkpoint_path=Path(tmp_path / "reference"),
    )

    class Hooks(TrainHooks):
        def __init__(self, fail_at: int | None = None) -> None:
            self.fail_at = fail_at
            self.checkpoints: list[int] = []

        def before_step(self, step, index, state, xs, ys) -> None:
            if step == self.fail_at:
                raise RuntimeError("simulated interruption")

        def on_checkpoint(self, step, checkpoint_path, run_path) -> None:
            assert _records(run_path, "train")[-1] == step
            self.checkpoints.append(step)

    with jax.set_mesh(TINY.sharding.get_mesh()):
        hooks = Hooks()
        reference = train(ModelWeights.init(jax.random.key(TINY.seed), TINY), TINY, config, hooks=hooks)
        assert hooks.checkpoints == [2, 4, 6]
        assert [path.name for path in checkpoint_steps(config.checkpoint_path)] == ["step_00000002", "step_00000004", "step_00000006"]

        interrupted = replace(config, checkpoint_path=Path(tmp_path / "interrupted"))
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(ModelWeights.init(jax.random.key(TINY.seed), TINY), TINY, interrupted, hooks=Hooks(fail_at=3))
        checkpoint = checkpoint_steps(interrupted.checkpoint_path)[-1]
        assert checkpoint.name == "step_00000002"
        assert [json.loads(row)["step"] for row in checkpoint_run_config(checkpoint)["run_artifacts"]["metrics.jsonl"].splitlines()] == [
            0,
            1,
            2,
        ]

        resumed = train(None, TINY, replace(interrupted, resume_from=interrupted.checkpoint_path), hooks=Hooks())
        for actual, wanted in zip(jax.tree.leaves(resumed), jax.tree.leaves(reference), strict=True):
            np.testing.assert_array_equal(actual, wanted)
        assert (
            load_train_state(interrupted.checkpoint_path, TINY, config).loader_state
            == load_train_state(config.checkpoint_path, TINY, config).loader_state
        )
        assert _records(interrupted.checkpoint_path, "train") == [1, 2, 3, 4, 5, 6]
        assert _records(interrupted.checkpoint_path, "eval") == [0, 2, 4, 6]


def test_training_records_routing_numerics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(importlib.import_module("lyra.training.train"), "plot_run", lambda _: None)
    np.save(tmp_path / "train.npy", np.arange(513, dtype=np.int32) % 32)
    model = replace(TINY, n_layers=2, mlp_map=("dense", "moe"))
    config = TrainingConfig(
        steps=5,
        batch_size=1,
        accumulation_steps=2,
        data_path=Path(tmp_path / "train.npy"),
        data_loader=DataLoaderConfig(worker_count=0),
        numerics_every=2,
        checkpoint_path=Path(tmp_path / "run"),
    )
    with jax.set_mesh(model.sharding.get_mesh()):
        train(ModelWeights.init(jax.random.key(model.seed), model), model, config)

    steps, busiest, least_used = _routing_series(config.checkpoint_path)
    assert steps == [1, 2, 4, 5]
    assert all(low <= 1.0 <= high for low, high in zip(least_used, busiest, strict=True))
