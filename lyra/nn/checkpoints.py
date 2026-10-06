from __future__ import annotations

import json
import os

import jax
import orbax.checkpoint.experimental.v1 as ocp
from etils.epath import Path
from orbax.checkpoint import path as checkpoint_paths


def checkpoint_path(source: str | os.PathLike[str]) -> Path:
    source = str(source)
    return Path(source if "://" in source else os.path.abspath(os.path.expanduser(source)))


def is_checkpoint(path: Path) -> bool:
    return (path / "_CHECKPOINT_METADATA").is_file() and checkpoint_paths.step.is_path_finalized(path)


def checkpoint_steps(directory: Path) -> list[Path]:
    steps = [path for path in directory.glob("step_*") if is_checkpoint(path)]
    return sorted(steps, key=lambda path: int(path.name.removeprefix("step_")))


def resolve_checkpoint(source: str | os.PathLike[str]) -> Path:
    path = checkpoint_path(source)
    if is_checkpoint(path):
        return path
    steps = checkpoint_steps(path)
    if not steps:
        raise FileNotFoundError(f"No completed checkpoint in {path}")
    return steps[-1]


def checkpoint_run_config(path: Path) -> dict:
    return json.loads((path / "_CHECKPOINT_METADATA").read_text())["custom_metadata"]["run_config"]


def _leaf_paths(tree) -> set[str]:
    return {jax.tree_util.keystr(path, simple=True, separator="/") for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]}


def load_checkpointables(path: Path, abstract_checkpointables: dict) -> dict:
    mismatch = f"Checkpoint {path} does not match the model config; load it with the variant it was trained with."
    saved = ocp.checkpointables_metadata(path).metadata
    if _leaf_paths(saved["params"]) != _leaf_paths(abstract_checkpointables["params"]):
        raise ValueError(mismatch)
    try:
        return ocp.load_checkpointables(path, abstract_checkpointables)
    except KeyError as exc:
        raise ValueError(mismatch) from exc
