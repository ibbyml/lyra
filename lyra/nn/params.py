from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from typing import Literal, TypeGuard, cast

import jax
import jax.numpy as jnp
from jax import P, ShapeDtypeStruct, lax
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from jax.tree_util import register_dataclass, register_pytree_node, register_static
from jaxtyping import Array, PyTree

from lyra.nn.checkpoints import load_checkpointables, resolve_checkpoint
from lyra.nn.quant import QArray


class Tag(Enum):
    """Which optimizer a parameter belongs to: Muon for MATRIX, Adam for the rest."""

    DEFAULT = auto()
    SCALAR = auto()
    BIAS = auto()
    MATRIX = auto()
    EMBEDDING = auto()


@register_dataclass
@dataclass(frozen=True)
class ArraySpec:
    shape: tuple[int, ...]
    dtype: jnp.dtype
    sharding: PartitionSpec
    initializer: Callable[..., Array]
    tag: Tag = Tag.DEFAULT


def arr(x) -> Array:
    return cast(Array, x)


def qarr(x) -> Array | QArray:
    return cast(Array | QArray, x)


def acast(x, dtype: jnp.dtype) -> Array:
    return arr(x).astype(dtype)


def wcast(x, dtype: jnp.dtype) -> Array | QArray:
    return x if isinstance(x, QArray) else arr(x).astype(dtype)


def is_spec(x: object) -> TypeGuard[ArraySpec]:
    return isinstance(x, ArraySpec)


def init_spec(key: Array, spec: ArraySpec) -> Array:
    value = spec.initializer(key, spec.shape, spec.dtype)
    if not spec_axis_names(spec.sharding):
        return value
    return jax.device_put(value, NamedSharding(jax.sharding.get_abstract_mesh(), spec.sharding))


def abstract_tree_from_spec(spec):
    mesh = jax.sharding.get_mesh()
    return jax.tree.map(
        lambda leaf: ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=NamedSharding(mesh, leaf.sharding)), spec, is_leaf=is_spec
    )


def weight_sharding(spec) -> PyTree:
    return jax.tree.map(lambda leaf: leaf.sharding, spec, is_leaf=is_spec)


# Masking: optimizer state for Muon and Adam covers disjoint subsets of the weights, with MaskedNode in the gaps.


class MaskedNode:
    pass


register_pytree_node(MaskedNode, flatten_func=lambda _: ([], None), unflatten_func=lambda *_: MaskedNode())


def is_masked(x: object) -> TypeGuard[MaskedNode]:
    return isinstance(x, MaskedNode)


def mask_spec_tree(tree, spec, tags) -> PyTree:
    return jax.tree.map(lambda p, s: p if s.tag in tags else MaskedNode(), tree, spec, is_leaf=is_spec)


def zero_masked_tree(tree, dtype: jnp.dtype | None = None) -> PyTree:
    return jax.tree.map(lambda x: x if is_masked(x) else jnp.zeros_like(x, dtype=dtype), tree, is_leaf=is_masked)


def merge_masked_tree(tree1, tree2) -> PyTree:
    return jax.tree.map(lambda x, y: y if is_masked(x) else x, tree1, tree2, is_leaf=is_masked)


# Weights


@cache
def _compiled_init(specs: tuple[ArraySpec, ...], treedef, shardings: tuple[NamedSharding, ...] | None):
    def initialize(key):
        keys = jax.random.split(key, len(specs))
        return jax.tree.unflatten(treedef, [init_spec(k, spec) for k, spec in zip(keys, specs)])

    return jax.jit(initialize, out_shardings=None if shardings is None else jax.tree.unflatten(treedef, list(shardings)))


def init_weights(key: Array, spec):
    specs, treedef = jax.tree.flatten(spec, is_leaf=is_spec)
    shardings = None
    if any(spec_axis_names(s.sharding) for s in specs):
        shardings = tuple(NamedSharding(jax.sharding.get_abstract_mesh(), s.sharding) for s in specs)
    return _compiled_init(tuple(specs), treedef, shardings)(key)


def load_weights(source, spec):
    return load_checkpointables(resolve_checkpoint(source), {"params": abstract_tree_from_spec(spec)})["params"]


MLPKind = Literal["dense", "moe"]


def full_moe(layers: int) -> tuple[MLPKind, ...]:
    return ("moe",) * layers


def interleaved_moe(layers: int, *, every: int, first: int) -> tuple[MLPKind, ...]:
    """An MoE layer at `first` and every `every` layers after it, counting from one."""
    return tuple("moe" if layer >= first and (layer - first) % every == 0 else "dense" for layer in range(1, layers + 1))


def moe_outer_dense(layers: int, start_dense: int = 0, end_dense: int = 0) -> tuple[MLPKind, ...]:
    return ("dense",) * start_dense + ("moe",) * (layers - start_dense - end_dense) + ("dense",) * end_dense  # type: ignore[return-value]


# Sharding


def spec_axis_names(spec) -> tuple[str, ...]:
    """Mesh axes a PartitionSpec shards over."""
    axes: list[str] = []
    for partition in spec:
        if isinstance(partition, tuple):
            axes.extend(axis for axis in partition if axis is not None)
        elif partition is not None:
            axes.append(partition)
    return tuple(axes)


def psum_axes(x: Array, config) -> Array:
    axes = config.sharding.model_axis_names
    return cast(Array, lax.psum(x, axes)) if axes else x


def all_gather_params(sparams, param_specs, sharding):
    fsdp_axes = sharding.fsdp_axis_names
    if not fsdp_axes:
        return sparams

    def gather(spec: PartitionSpec, x):
        for dim, part in enumerate(spec):
            if set(spec_axis_names((part,))) & set(fsdp_axes):
                x = lax.all_gather(x, fsdp_axes, axis=dim, tiled=True)
        return x

    return jax.tree.map(gather, param_specs, sparams, is_leaf=lambda s: isinstance(s, PartitionSpec))


@register_static
@dataclass(frozen=True, kw_only=True)
class ShardingRules:
    mesh_axis_names: tuple[str, ...]
    mesh_shape: tuple[int, ...]

    # Parameters
    emb_spec: P
    unemb_spec: P
    norm_scale_spec: P
    attn_q_spec: P
    attn_kv_spec: P
    attn_qkv_bias_spec: P
    attn_sink_spec: P
    attn_out_spec: P
    attn_out_bias_spec: P
    router_spec: P
    router_bias_spec: P
    moe_mlp_up_spec: P
    moe_mlp_down_spec: P
    moe_mlp_up_bias_spec: P
    moe_mlp_down_bias_spec: P
    dense_mlp_up_spec: P
    dense_mlp_down_spec: P
    dense_mlp_up_bias_spec: P
    dense_mlp_down_bias_spec: P

    # Activations
    batch_spec: P

    def get_mesh(self):
        return jax.make_mesh(self.mesh_shape, self.mesh_axis_names, axis_types=(AxisType.Auto,) * len(self.mesh_axis_names))

    def _size(self, axes: tuple[str, ...]) -> int:
        return math.prod(self.mesh_shape[self.mesh_axis_names.index(axis)] for axis in axes)

    # Each role is read off the first dimension of a representative spec.
    data_axis_names = property(lambda self: spec_axis_names(self.batch_spec[:1]))
    model_axis_names = property(lambda self: spec_axis_names(self.attn_out_spec[:1]))
    fsdp_axis_names = property(lambda self: spec_axis_names(self.emb_spec[:1]))
    expert_axis_names = property(lambda self: spec_axis_names(self.moe_mlp_up_spec[:1]))
    data_axis_size = property(lambda self: self._size(self.data_axis_names))
    model_axis_size = property(lambda self: self._size(self.model_axis_names))
    fsdp_axis_size = property(lambda self: self._size(self.fsdp_axis_names))
    expert_axis_size = property(lambda self: self._size(self.expert_axis_names))


def _sharding_rules(
    mesh_axis_names: tuple[str, ...],
    mesh_shape: tuple[int, ...],
    *,
    data: str | None = None,
    model: str | None = None,
    fsdp: str | None = None,
    expert: str | None = None,
) -> ShardingRules:
    m, f, x = model, fsdp, expert
    batch_axes = tuple(axis for axis in (data, x) if axis is not None)
    batch = batch_axes[0] if len(batch_axes) == 1 else (batch_axes or None)
    return ShardingRules(
        mesh_axis_names=mesh_axis_names,
        mesh_shape=mesh_shape,
        emb_spec=P(f, None),
        unemb_spec=P(f, None),
        norm_scale_spec=P(None),
        attn_q_spec=P(f, m),
        attn_kv_spec=P(f, m),
        attn_qkv_bias_spec=P(m),
        attn_sink_spec=P(m),
        attn_out_spec=P(m, f),
        attn_out_bias_spec=P(None),
        router_spec=P(f, None),
        router_bias_spec=P(None),
        moe_mlp_up_spec=P(x, f, None, m),
        moe_mlp_down_spec=P(x, m, f),
        moe_mlp_up_bias_spec=P(x, None, m),
        moe_mlp_down_bias_spec=P(x, None),
        dense_mlp_up_spec=P(f, None, m),
        dense_mlp_down_spec=P(m, f),
        dense_mlp_up_bias_spec=P(None, None, m),
        dense_mlp_down_bias_spec=P(None, None),
        batch_spec=P(batch, None),
    )


def no_sharding() -> ShardingRules:
    return _sharding_rules(("devices",), (1,))


def data_parallel() -> ShardingRules:
    return _sharding_rules(("devices",), (jax.device_count(),), data="devices")


def tensor_parallel(data_axis: int, model_axis: int) -> ShardingRules:
    return _sharding_rules(("data", "model"), (data_axis, model_axis), data="data", model="model")


def expert_parallel(fsdp: int, expert: int, model: int = 1) -> ShardingRules:
    return _sharding_rules(("fsdp", "expert", "model"), (fsdp, expert, model), data="fsdp", fsdp="fsdp", expert="expert", model="model")
