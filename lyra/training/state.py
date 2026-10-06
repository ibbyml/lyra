from __future__ import annotations

from dataclasses import dataclass
from functools import cache, partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import orbax.checkpoint.experimental.v1 as ocp
from etils.epath import Path
from jax import P, ShapeDtypeStruct
from jax.sharding import NamedSharding
from jax.tree_util import register_dataclass

from lyra.model import ModelConfig, ModelWeights
from lyra.nn.checkpoints import checkpoint_steps, load_checkpointables, resolve_checkpoint
from lyra.nn.params import MaskedNode, abstract_tree_from_spec, is_masked, mask_spec_tree
from lyra.training.optimizer import ADAM_TAGS, MUON_TAGS, AdamState, MuonState, OptimizerState

if TYPE_CHECKING:
    from lyra.training.train import TrainingConfig


@register_dataclass
@dataclass(frozen=True)
class TrainState:
    params: ModelWeights
    optimizer: OptimizerState


@dataclass(frozen=True)
class RestoredTrainState:
    state: TrainState
    step: int
    path: Path
    loader_state: bytes


@cache
def _optimizer_initializer(mcfg: ModelConfig, tcfg: TrainingConfig):
    return jax.jit(partial(OptimizerState.init, spec=ModelWeights.spec(mcfg), tcfg=tcfg))


def init_train_state(params: ModelWeights, mcfg: ModelConfig, tcfg: TrainingConfig) -> TrainState:
    return TrainState(params, _optimizer_initializer(mcfg, tcfg)(params))


def train_state_checkpointables(state: TrainState, loader_state: bytes) -> dict:
    return {
        "params": state.params,
        "muon_moment": state.optimizer.muon.moment,
        "adam_mu": state.optimizer.adam.mu,
        "adam_nu": state.optimizer.adam.nu,
        "adam_count": state.optimizer.adam.count,
        "loader": {"state": loader_state.decode()},
    }


def save_checkpoint(directory: Path, step: int, state: TrainState, loader_state: bytes, run_config: dict, *, keep: int) -> None:
    """Save step_<step> and delete all but the `keep` most recent checkpoints."""
    path = directory / f"step_{step:08d}"
    if path.exists():  # A partial save from an interrupted run.
        path.rmtree()
    print(f"\n[checkpoint] Saving {path}")
    ocp.save_checkpointables(path, train_state_checkpointables(state, loader_state), custom_metadata={"run_config": run_config})
    for old in checkpoint_steps(directory)[:-keep]:
        old.rmtree()


def _abstract_masked(mask_tree, dtype: jnp.dtype):
    return jax.tree.map(
        lambda leaf: None if isinstance(leaf, MaskedNode) else ShapeDtypeStruct(leaf.shape, dtype, sharding=leaf.sharding),
        mask_tree,
        is_leaf=lambda leaf: is_masked(leaf) or isinstance(leaf, ShapeDtypeStruct),
    )


def _unmask(restored, mask_tree):
    return jax.tree.map(
        lambda leaf, mask: MaskedNode() if isinstance(mask, MaskedNode) else leaf,
        restored,
        mask_tree,
        is_leaf=lambda leaf: is_masked(leaf) or leaf is None,
    )


def load_train_state(source: str | Path, mcfg: ModelConfig, tcfg: TrainingConfig) -> RestoredTrainState:
    path = resolve_checkpoint(source)
    spec = ModelWeights.spec(mcfg)
    params = abstract_tree_from_spec(spec)
    muon_mask = mask_spec_tree(params, spec, MUON_TAGS)
    adam_mask = mask_spec_tree(params, spec, ADAM_TAGS)
    loaded = load_checkpointables(
        path,
        {
            "params": params,
            "muon_moment": _abstract_masked(muon_mask, tcfg.muon_moment_dtype),
            "adam_mu": _abstract_masked(adam_mask, tcfg.adam_mu_dtype),
            "adam_nu": _abstract_masked(adam_mask, tcfg.adam_nu_dtype),
            "adam_count": ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(jax.sharding.get_mesh(), P())),
            "loader": None,
        },
    )
    optimizer = OptimizerState(
        muon=MuonState(moment=_unmask(loaded["muon_moment"], muon_mask)),
        adam=AdamState(mu=_unmask(loaded["adam_mu"], adam_mask), nu=_unmask(loaded["adam_nu"], adam_mask), count=loaded["adam_count"]),
    )
    return RestoredTrainState(
        state=TrainState(loaded["params"], optimizer),
        step=int(path.name.removeprefix("step_")),
        path=path,
        loader_state=loaded["loader"]["state"].encode(),
    )
