import jax
import jax.numpy as jnp
from jax import Array, lax

from lyra.nn.params import *
from lyra.nn.quant import QArray


def ep_dispatch_metadata(all_groups: Array, device_index: Array):
    axis_size, n_experts = all_groups.shape
    experts_per_device = n_experts // axis_size
    groups_by_route = all_groups.reshape(axis_size, axis_size, experts_per_device)

    local_groups_by_device = groups_by_route.sum(axis=0)
    local_bases = jnp.cumsum(local_groups_by_device, axis=-1) - local_groups_by_device
    source_prefixes = jnp.cumsum(groups_by_route, axis=0) - groups_by_route
    output_offsets_by_source = local_bases[None, ...] + source_prefixes

    input_offsets_by_source = jnp.cumsum(all_groups, axis=-1) - all_groups
    owned_start = device_index * experts_per_device
    return_offsets = lax.dynamic_slice(
        input_offsets_by_source,
        (0, owned_start),
        (axis_size, experts_per_device),
    ).reshape(n_experts)

    metadata = (
        input_offsets_by_source[device_index],
        all_groups[device_index],
        output_offsets_by_source[device_index].reshape(n_experts),
        groups_by_route[:, device_index, :].reshape(n_experts),
        output_offsets_by_source[:, device_index, :].reshape(n_experts),
        return_offsets,
        local_groups_by_device[device_index],
    )

    return metadata


def ep_grouped_mlp(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm):
    up = gmm(
        lhs=t,
        rhs=mlp_up,
        group_sizes=groups,
        bias=up_bias,
        fused_swiglu=True,
        swiglu_beta=config.swiglu_beta,
        swiglu_limit=config.swiglu_limit,
        quant_dtype=config.gemm_dtype,
    )
    down = gmm(
        lhs=up,
        rhs=mlp_down,
        group_sizes=groups,
        bias=(down_bias / config.sharding.model_axis_size) if down_bias is not None else None,
        quant_dtype=config.gemm_dtype,
    )
    out = psum_axes(down, config)
    return out


def ep_shared_weights(param, config):
    if param is None:
        return None
    if isinstance(param, QArray):
        raise NotImplementedError("Quantized shared experts are unsupported with expert parallelism.")
    axis = config.sharding.expert_axis_names[0]
    axis_size = config.sharding.expert_axis_size
    experts_per_device = config.n_experts // axis_size
    axis_index = lax.axis_index(axis)
    shared = []
    for global_expert in range(config.n_routed_experts, config.n_experts):
        owner, local_expert = divmod(global_expert, experts_per_device)
        local = param[local_expert]
        selected = jnp.where(axis_index == owner, local, jnp.zeros_like(local))
        shared.append(lax.psum(selected, axis))
    return jnp.stack(shared)


def ep_shared_mlp(x, mlp_up, mlp_down, up_bias, down_bias, config, gmm):
    shared_up = ep_shared_weights(mlp_up, config)
    shared_down = ep_shared_weights(mlp_down, config)
    shared_up_bias = ep_shared_weights(up_bias, config)
    shared_down_bias = ep_shared_weights(down_bias, config)
    assert shared_up is not None and shared_down is not None

    tokens = x.shape[0]
    groups = jnp.full((config.n_shared_experts,), tokens, dtype=jnp.int32)
    repeated = jnp.tile(x, (config.n_shared_experts, 1))
    out = ep_grouped_mlp(repeated, groups, shared_up, shared_down, shared_up_bias, shared_down_bias, config, gmm)
    return out.reshape(config.n_shared_experts, tokens, x.shape[-1]).sum(axis=0)


def ep_moe_gemm_ragged(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm):
    axis = config.sharding.expert_axis_names[0]
    axis_size = config.sharding.expert_axis_size
    N, C = t.shape
    all_groups = lax.all_gather(groups, axis, axis=0, tiled=False)
    metadata = ep_dispatch_metadata(all_groups, lax.axis_index(axis))
    input_offsets, send_sizes, output_offsets, recv_sizes, recv_offsets, return_offsets, local_groups = metadata

    max_received = axis_size * N
    received = lax.ragged_all_to_all(
        t,
        jnp.zeros((max_received, C), dtype=t.dtype),
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name=axis,
    )
    received_rows = jnp.sum(local_groups)
    received = jnp.where(jnp.arange(max_received)[:, None] < received_rows, received, 0)
    down = ep_grouped_mlp(received, local_groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
    down = jnp.where(jnp.arange(max_received)[:, None] < received_rows, down, 0)

    out = lax.ragged_all_to_all(
        down,
        jnp.zeros_like(t),
        recv_offsets,
        recv_sizes,
        return_offsets,
        send_sizes,
        axis_name=axis,
    )

    return out


def ep_moe_gemm_reference(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm):
    axis = config.sharding.expert_axis_names[0]
    axis_size = config.sharding.expert_axis_size
    n_experts = config.n_experts
    experts_per_device = n_experts // axis_size
    N, C = t.shape

    offsets = jnp.cumsum(groups) - groups
    expert_ids = jnp.repeat(jnp.arange(n_experts), groups, total_repeat_length=N)
    ranks = jnp.arange(N) - offsets[expert_ids]
    dispatched = jnp.zeros((n_experts, N, C), t.dtype).at[expert_ids, ranks].set(t, unique_indices=True)
    dispatched = dispatched.reshape(axis_size, experts_per_device, N, C)
    dispatched = lax.all_to_all(dispatched, axis, 0, 0, tiled=True)
    received = dispatched.swapaxes(0, 1).reshape(experts_per_device * axis_size * N, C)
    local_groups = jnp.full((experts_per_device,), axis_size * N, dtype=jnp.int32)

    down = ep_grouped_mlp(received, local_groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
    down = down.reshape(experts_per_device, axis_size, N, C).swapaxes(0, 1)
    down = lax.all_to_all(down, axis, 0, 0, tiled=True).reshape(n_experts, N, C)
    return down[expert_ids, ranks]


@jax.named_scope("ep_dispatch")
def ep_moe_gemm(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm):
    ragged = lambda t, groups: ep_moe_gemm_ragged(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
    reference = lambda t, groups: ep_moe_gemm_reference(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
    dispatch = lax.platform_dependent(t, groups, tpu=ragged, default=reference)
    return dispatch
