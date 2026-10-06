from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint.experimental.v1 as ocp
import pytest
from etils import epath

from lyra.model import ModelWeights, model_apply
from lyra.nn.checkpoints import resolve_checkpoint
from lyra.training.state import init_train_state, load_train_state, train_state_checkpointables
from lyra.training.train import TrainHooks, TrainingConfig, check_resume, compile_step, resume_signature, train
from lyra.variants import model_variant


def _write_run_config(checkpoint: Path, run_config: dict) -> None:
    checkpoint.mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"custom_metadata": {"run_config": run_config}}))


def test_resume_rejects_different_settings(tmp_path: Path) -> None:
    loader = cast(Any, SimpleNamespace(token_count=1000))
    current = resume_signature(model_variant("dev"), TrainingConfig(), loader)
    checkpoint = epath.Path(tmp_path / "step_00000001")
    _write_run_config(tmp_path / "step_00000001", {"resume_signature": current})

    check_resume(checkpoint, current)
    gated = resume_signature(replace(model_variant("dev"), use_sdpa_output_gate=True), TrainingConfig(), loader)
    with pytest.raises(ValueError, match="model settings differ"):
        check_resume(checkpoint, gated)
    with pytest.raises(ValueError, match="training settings differ"):
        check_resume(checkpoint, resume_signature(model_variant("dev"), TrainingConfig(steps=1), loader))
    with pytest.raises(ValueError, match="data settings differ"):
        check_resume(checkpoint, resume_signature(model_variant("dev"), TrainingConfig(), cast(Any, SimpleNamespace(token_count=999))))


def test_resolve_checkpoint_picks_the_latest_completed_step(tmp_path: Path) -> None:
    for step in (2, 10):
        _write_run_config(tmp_path / f"step_{step:08d}", {})
    (tmp_path / "step_00000011").mkdir()  # An incomplete save.
    assert resolve_checkpoint(tmp_path).name == "step_00000010"
    assert resolve_checkpoint(tmp_path / "step_00000002").name == "step_00000002"
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "step_00000011")


def test_fresh_run_refuses_a_directory_with_checkpoints(tmp_path: Path) -> None:
    _write_run_config(tmp_path / "step_00000001", {})
    with pytest.raises(FileExistsError, match="already has checkpoints"):
        train(None, model_variant("dev"), TrainingConfig(checkpoint_path=epath.Path(tmp_path)))


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = replace(
        model_variant("dev"),
        seq_len=4,
        n_layers=1,
        mlp_map=("dense",),
        vocab_size=32,
        padded_vocab_size=32,
        implementation="xla",
        use_sdpa_output_gate=True,
    )
    training = TrainingConfig()
    with jax.set_mesh(config.sharding.get_mesh()):
        weights = ModelWeights.init(jax.random.key(0), config)
        weights.layers[0].attn.sdpa_gate = jnp.full((config.n_embd, config.n_attn_heads), 0.1)
        state = init_train_state(weights, config, training)
        ocp.save_checkpointables(str(tmp_path / "step_00000001"), train_state_checkpointables(state, b"position"))

        loaded = ModelWeights.load(str(tmp_path), config)
        resumed = load_train_state(str(tmp_path), config, training)
        tokens = jnp.arange(4, dtype=jnp.int32)[None, :]
        expected = model_apply(tokens, weights, None, config)[0]
        for restored in (loaded, resumed.state.params):
            np.testing.assert_array_equal(model_apply(tokens, restored, None, config)[0], expected)
        assert (resumed.step, resumed.loader_state) == (1, b"position")
        for actual, wanted in zip(jax.tree.leaves(resumed.state.optimizer), jax.tree.leaves(state.optimizer), strict=True):
            np.testing.assert_array_equal(actual, wanted)
        with pytest.raises(ValueError, match="does not match the model config"):
            ModelWeights.load(str(tmp_path), replace(config, use_sdpa_output_gate=False))


def test_compile_hooks_receive_stages() -> None:
    events = []

    class Hooks(TrainHooks):
        def on_lowered(self, lowered) -> None:
            events.append("lowered")

        def on_compiled(self, compiled, timing: dict[str, float]) -> None:
            assert timing["total_seconds"] >= 0
            events.append("compiled")

    compiled = compile_step(jax.jit(lambda x: x + 1), "smoke", (jnp.array(1),), hooks=Hooks())
    assert int(compiled(jnp.array(1))) == 2
    assert events == ["lowered", "compiled"]
