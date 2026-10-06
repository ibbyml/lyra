from __future__ import annotations

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from etils import epath

from lyra.data import DataLoaderConfig
from lyra.model import ModelWeights
from lyra.training.optimizer import wsd_lr
from lyra.training.presets import TRAINING_PRESETS
from lyra.training.train import TrainingConfig
from lyra.variants import model_variant
from scripts.train import Args, training_config


def _scales(config: TrainingConfig) -> jax.Array:
    return jnp.stack([wsd_lr(jnp.asarray(step), config)[2] for step in range(config.steps)])


def test_default_schedule_warms_up_then_cosine_decays() -> None:
    config = replace(TRAINING_PRESETS["train-dev"], steps=48)
    scales = _scales(config)
    assert (config.warmup_steps, config.stable_steps, config.decay_steps) == (2, 0, 46)
    assert jnp.allclose(scales[:3], jnp.array([0.5, 1.0, 1.0]))
    assert scales[3] < 1.0
    assert jnp.isclose(scales[-1], 0.0, atol=1e-6)
    assert jnp.all(jnp.diff(scales[1:]) <= 0)
    assert replace(config, steps=954).warmup_steps == 29


def test_linear_wsd_schedule_from_cli() -> None:
    config = training_config(Args(run="train-dev", steps=20, warmup_fraction=0.1, stable_fraction=0.8, decay_type="linear"))
    assert (config.warmup_steps, config.stable_steps, config.decay_steps) == (2, 16, 2)
    assert jnp.allclose(_scales(config), jnp.array([0.5] + [1.0] * 18 + [0.0]))


def test_single_step_uses_peak_lr() -> None:
    config = replace(TRAINING_PRESETS["train-dev"], steps=1)
    adam_lr, muon_lr, scale = wsd_lr(jnp.asarray(0), config)
    assert jnp.isclose(scale, 1.0)
    assert jnp.isclose(adam_lr, config.adam_max_lr)
    assert jnp.isclose(muon_lr, config.muon_max_lr)


def test_cli_paths_expand_home() -> None:
    config = training_config(Args(data="~/data", eval_data="~/val", out="~/checkpoints", resume="~/previous"))
    assert [str(path) for path in (config.data_path, config.eval_data_path, config.checkpoint_path, config.resume_from)] == [
        str(Path.home() / name) for name in ("data", "val", "checkpoints", "previous")
    ]


def test_validation_replays_the_same_batches_after_the_final_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    training_module = import_module("lyra.training.train")
    directory = epath.Path(tmp_path)
    config = replace(
        model_variant("dev"), seq_len=4, n_layers=1, mlp_map=("dense",), vocab_size=32, padded_vocab_size=32, implementation="xla"
    )
    eval_tokens = np.arange(31, -2, -1, dtype=np.int32) % config.vocab_size
    np.save(directory / "train.npy", np.arange(17, dtype=np.int32))
    np.save(directory / "eval.npy", eval_tokens)
    training = TrainingConfig(
        steps=3,
        batch_size=1,
        accumulation_steps=1,
        data_path=directory / "train.npy",
        data_loader=DataLoaderConfig(worker_count=0, shuffle=True),
        eval_data_path=directory / "eval.npy",
        eval_every=2,
        eval_batches=2,
        checkpoint_every=100,
        checkpoint_path=directory / "run",
    )
    final_checkpoint = training.checkpoint_path / "step_00000003"
    evaluations, checkpoint_ready = [], []
    run_validation = training_module.run_validation

    def observe_validation(val_step, batches, params, *, eval_batches, microbatches):
        seen = []
        checkpoint_ready.append(final_checkpoint.is_dir())

        def observe_step(xs, ys, weights):
            seen.append((np.asarray(xs), np.asarray(ys)))
            return val_step(xs, ys, weights)

        result = run_validation(observe_step, batches, params, eval_batches=eval_batches, microbatches=microbatches)
        evaluations.append(seen)
        return result

    monkeypatch.setattr(training_module, "run_validation", observe_validation)
    monkeypatch.setattr(training_module, "plot_run", lambda _: None)
    with jax.set_mesh(config.sharding.get_mesh()):
        state = training_module.train(ModelWeights.init(jax.random.key(0), config), config, training)

    assert int(state.optimizer.adam.count) == training.steps
    assert checkpoint_ready == [False, False, True]
    for seen in evaluations:
        assert len(seen) == training.eval_batches
        for index, (xs, ys) in enumerate(seen):
            start = index * config.seq_len
            np.testing.assert_array_equal(xs.ravel(), eval_tokens[start : start + config.seq_len])
            np.testing.assert_array_equal(ys.ravel(), eval_tokens[start + 1 : start + config.seq_len + 1])
    records = [json.loads(line) for line in (training.checkpoint_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records if record["kind"] == "eval"] == [0, 2, 3]
