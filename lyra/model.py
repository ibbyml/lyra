from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cache, partial
from typing import TypeIs, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.nn import initializers as init
from jax.nn import logsumexp, softmax
from jax.tree_util import register_dataclass, register_static
from jaxtyping import Array, Int

from lyra.kernels.combine import combine, supports_combine
from lyra.kernels.common import Implementation, swiglu
from lyra.kernels.embs import embs_gather
from lyra.kernels.ep_moe import ep_moe_gemm, ep_shared_mlp
from lyra.kernels.op import Mode, attn_op, gmm_op
from lyra.nn.params import ArraySpec, MLPKind, ShardingRules, Tag, acast, arr, init_spec, init_weights, load_weights, psum_axes, qarr, wcast
from lyra.nn.quant import QArray, dot, einsum, quantize, qupdate
from lyra.tokenizer import PADDED_VOCAB_SIZE
from lyra.training.probe import NO_PROBE, Probe

NN_INNER = (((2,), (0,)), ((), ()))  # [B, T, C] @ [C, H] -> [B, T, H]
NN_RHS_T = (((2,), (1,)), ((), ()))  # [B, T, C] @ [H, C] -> [B, T, H]

Param = Array | ArraySpec
QParam = Array | ArraySpec | QArray
OptionalParam = Param | None


def exists[T](x: T | None) -> TypeIs[T]:
    return x is not None


@register_static
@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    # Core
    name: str
    seq_len: int = 4096
    n_embd: int = 2048
    n_layers: int = 16
    vocab_size: int = 201088
    tie_embeddings: bool = True
    padded_vocab_size: int = PADDED_VOCAB_SIZE
    seed: int = 42

    # Attention
    head_dim: int = 128
    n_attn_heads: int = 16
    n_kv_heads: int = 4
    sliding_window: int = 128
    use_qk_norm: bool = True
    use_qkv_bias: bool = False
    use_attn_out_bias: bool = False
    use_learned_xsa: bool = False
    use_sdpa_output_gate: bool = False

    # MLP
    mlp_map: tuple[MLPKind, ...]
    n_routed_experts: int = 15
    n_shared_experts: int = 1
    n_active_experts: int = 4
    expert_mlp_widening: float = 2.0
    dense_mlp_widening: float = 3.0
    swiglu_beta: float = 1.702
    swiglu_limit: float = 7.0
    load_balance_weight: float = 1e-2
    router_z_loss_weight: float = 1e-4
    use_mlp_bias: bool = False
    use_router_bias: bool = True

    # RoPE
    rope_base: float = 150000.0
    rope_scale: float = 1.0
    ntk_alpha: float = 1.0
    ntk_beta: float = 32.0

    # Dtypes & Quantization
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.bfloat16
    gemm_dtype: jnp.dtype = jnp.bfloat16

    quantize_attn: bool = False
    quantize_mlp: bool = False
    quantize_cache: bool = False
    quant_block: tuple[int, int] = (256, 256)

    # Sharding & Parameters
    sharding: ShardingRules

    router_logit_std: float = 0.50
    w_init: init.Initializer = field(default_factory=init.lecun_normal)
    b_init: init.Initializer = init.zeros
    norm_init: init.Initializer = init.ones

    # Kernels & ops
    mode: Mode = Mode.NOCACHE
    use_presorted_emb_gather: bool = True
    implementation: Implementation = "auto"

    # Misc
    sample_topk: int = 4
    sample_temperature: float = 0.8
    norm_eps: float = 1e-5

    @property
    def n_experts(self) -> int:
        return self.n_routed_experts + self.n_shared_experts

    @property
    def n_active_routed_experts(self) -> int:
        return self.n_active_experts - self.n_shared_experts

    @property
    def n_moe_layers(self) -> int:
        return self.mlp_map.count("moe")

    @property
    def residual_scale(self) -> float:
        return 1.0 / math.sqrt(2.0 * self.n_layers)

    @property
    def dense_residual_scale(self) -> float:
        return self.residual_scale * (self.n_active_experts**-0.5 if "moe" in self.mlp_map else 1.0)


# Parameter specs
def _matrix(config: ModelConfig, shape: tuple[int, ...], sharding, initializer=None) -> ArraySpec:
    return ArraySpec(shape, config.param_dtype, sharding, initializer or config.w_init, Tag.MATRIX)


def _bias(config: ModelConfig, shape: tuple[int, ...], sharding, enabled: bool) -> ArraySpec | None:
    return ArraySpec(shape, config.param_dtype, sharding, config.b_init, Tag.BIAS) if enabled else None


def _scalar(shape: tuple[int, ...], sharding, initializer) -> ArraySpec:
    return ArraySpec(shape, jnp.float32, sharding, initializer, Tag.SCALAR)


def stacked_init(base_init: init.Initializer) -> init.Initializer:
    def initializer(key, shape, dtype=None, out_sharding=None):
        n_experts, *expert_shape = shape
        value = jax.vmap(lambda k: base_init(k, tuple(expert_shape), dtype))(jax.random.split(key, n_experts))
        return value if out_sharding is None else jax.device_put(value, out_sharding)

    return initializer


def _quantize_block(value: Array, config: ModelConfig) -> QArray:
    k, n = config.quant_block
    return quantize(value, config.gemm_dtype, (math.gcd(value.shape[-2], k), math.gcd(value.shape[-1], n)))


# RoPE and norm
@cache
def precompute_freq_cis(config: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(int(config.seq_len * config.rope_scale), dtype=np.float64)
    inv_freq = 1.0 / (config.rope_base ** (np.arange(0, config.head_dim, 2, dtype=np.float64) / config.head_dim))
    concentration = 1.0

    # YaRN
    if config.rope_scale > 1.0:
        concentration = 0.1 * math.log(config.rope_scale) + 1.0
        dim_half = config.head_dim / 2.0
        log_base = math.log(config.rope_base) ** -1.0
        low = dim_half * math.log(config.seq_len / (config.ntk_beta * 2.0 * math.pi)) * log_base
        high = dim_half * math.log(config.seq_len / (config.ntk_alpha * 2.0 * math.pi)) * log_base
        ramp = 1.0 - np.clip((np.arange(int(dim_half), dtype=np.float64) - low) / (high - low), 0.0, 1.0)
        inv_freq = (1.0 - ramp) * (inv_freq / config.rope_scale) + ramp * inv_freq
    
    angles = np.einsum("i,j->ij", positions, inv_freq)
    cos = (np.cos(angles) * concentration).astype(np.float32)
    sin = (np.sin(angles) * concentration).astype(np.float32)
    
    return cos, sin


def rotate(x: Array, freq_cis: tuple[Array, ...]) -> Array:
    cos, sin = freq_cis
    x_1, x_2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x_1 * cos - x_2 * sin, x_2 * cos + x_1 * sin], axis=-1).astype(x.dtype)


def rms_norm(x: Array, scale: Array, eps: float) -> Array:
    t = x.astype(jnp.float32)
    return (t * lax.rsqrt(jnp.mean(t**2, axis=-1, keepdims=True) + eps) * scale).astype(x.dtype)


# KV cache


@register_dataclass
@dataclass
class LayerCache:
    k: Array | QArray
    v: Array | QArray


def cache_update(cache: LayerCache, k: Array, v: Array, start_pos: Array) -> LayerCache:
    start = (0, 0, start_pos, 0)
    if isinstance(cache.k, QArray):
        return LayerCache(k=qupdate(cache.k, k, start), v=qupdate(cast(QArray, cache.v), v, start))
    k_cache, v_cache = arr(cache.k), arr(cache.v)
    return LayerCache(
        k=lax.dynamic_update_slice(k_cache, k.astype(k_cache.dtype), start),
        v=lax.dynamic_update_slice(v_cache, v.astype(v_cache.dtype), start),
    )


@register_dataclass
@dataclass
class ModelCache:
    layers: tuple[LayerCache, ...]
    pos: int | Array

    @staticmethod
    def quantize(cache: ModelCache, config: ModelConfig) -> ModelCache:
        if not config.quantize_cache:
            return cache
        qkv = lambda a: quantize(a, config.gemm_dtype, block=(1, config.head_dim))
        return ModelCache(layers=tuple(LayerCache(k=qkv(layer.k), v=qkv(layer.v)) for layer in cache.layers), pos=cache.pos)


# Attention


@register_dataclass
@dataclass
class AttentionWeights:
    q: QParam
    k: QParam
    v: QParam
    out: QParam

    q_bias: OptionalParam
    k_bias: OptionalParam
    v_bias: OptionalParam
    out_bias: OptionalParam

    sinks: Param
    norm_scale: Param
    qnorm_scale: OptionalParam
    knorm_scale: OptionalParam
    xsa_alpha: OptionalParam
    sdpa_gate: OptionalParam

    @classmethod
    def spec(cls, config: ModelConfig) -> AttentionWeights:
        c, s = config, config.sharding
        q_dim, kv_dim = c.n_attn_heads * c.head_dim, c.n_kv_heads * c.head_dim
        return AttentionWeights(
            q=_matrix(c, (c.n_embd, q_dim), s.attn_q_spec),
            q_bias=_bias(c, (q_dim,), s.attn_qkv_bias_spec, c.use_qkv_bias),
            k=_matrix(c, (c.n_embd, kv_dim), s.attn_kv_spec),
            k_bias=_bias(c, (kv_dim,), s.attn_qkv_bias_spec, c.use_qkv_bias),
            v=_matrix(c, (c.n_embd, kv_dim), s.attn_kv_spec),
            v_bias=_bias(c, (kv_dim,), s.attn_qkv_bias_spec, c.use_qkv_bias),
            sinks=_scalar((c.n_attn_heads,), s.attn_sink_spec, c.b_init),
            out=_matrix(c, (q_dim, c.n_embd), s.attn_out_spec),
            out_bias=_bias(c, (c.n_embd,), s.attn_out_bias_spec, c.use_attn_out_bias),
            norm_scale=_scalar((c.n_embd,), s.norm_scale_spec, c.norm_init),
            qnorm_scale=_scalar((c.head_dim,), s.norm_scale_spec, c.norm_init) if c.use_qk_norm else None,
            knorm_scale=_scalar((c.head_dim,), s.norm_scale_spec, c.norm_init) if c.use_qk_norm else None,
            xsa_alpha=_scalar((c.n_attn_heads,), s.attn_sink_spec, c.b_init) if c.use_learned_xsa else None,
            # Zero-init, so 2 * sigmoid(0) = 1 (starts as identity).
            sdpa_gate=ArraySpec((c.n_embd, c.n_attn_heads), jnp.float32, s.attn_q_spec, init.zeros, Tag.DEFAULT)
            if c.use_sdpa_output_gate
            else None,
        )

    @staticmethod
    def quantize(w: AttentionWeights, config: ModelConfig) -> AttentionWeights:
        if not config.quantize_attn:
            return w
        q = lambda weight: weight if isinstance(weight, QArray) else _quantize_block(arr(weight), config)
        return replace(w, q=q(w.q), k=q(w.k), v=q(w.v), out=q(w.out))


@jax.named_scope("attention")
def attention_apply(
    x: Array,
    w: AttentionWeights,
    idx: int,
    pos: int | Array,
    freq_cis: tuple[Array, ...],
    cache: LayerCache | None,
    config: ModelConfig,
    *,
    probe: Probe = NO_PROBE,
) -> tuple[Array, LayerCache | None]:
    B, T, _ = x.shape
    dtype = x.dtype
    start_pos = jnp.asarray(pos)
    heads_first = partial(jnp.swapaxes, axis1=1, axis2=2)
    wc = partial(wcast, dtype=config.compute_dtype)
    ac = partial(acast, dtype=config.compute_dtype)

    h = rms_norm(x, arr(w.norm_scale), config.norm_eps)
    probe.tensor("norm_out", h)

    with jax.named_scope("qkv_projection"):
        q = dot(h, wc(w.q), dimension_numbers=NN_INNER).reshape(B, T, -1, config.head_dim)
        k = dot(h, wc(w.k), dimension_numbers=NN_INNER).reshape(B, T, -1, config.head_dim)
        v = dot(h, wc(w.v), dimension_numbers=NN_INNER).reshape(B, T, -1, config.head_dim)
    if exists(w.q_bias) and exists(w.k_bias) and exists(w.v_bias):
        q += ac(w.q_bias).reshape(1, 1, -1, config.head_dim)
        k += ac(w.k_bias).reshape(1, 1, -1, config.head_dim)
        v += ac(w.v_bias).reshape(1, 1, -1, config.head_dim)

    if exists(w.qnorm_scale) and exists(w.knorm_scale):
        with jax.named_scope("qknorm"):
            probe.tensor("q_pre_norm", q)
            probe.tensor("k_pre_norm", k)
            q = rms_norm(q, arr(w.qnorm_scale), config.norm_eps)
            k = rms_norm(k, arr(w.knorm_scale), config.norm_eps)

    with jax.named_scope("RoPE"):
        q, k = rotate(q, freq_cis), rotate(k, freq_cis)
    probe.tensor("q", q)
    probe.tensor("k", k)
    probe.tensor("v", v)

    q, k, v, sinks = heads_first(q), heads_first(k), heads_first(v), arr(w.sinks)
    current_v = v
    updated_cache = None
    if config.mode != Mode.NOCACHE:
        assert cache is not None
        updated_cache = cache_update(cache, k, v, start_pos)
        if config.mode == Mode.DECODE:
            k, v = updated_cache.k, updated_cache.v

    sliding_window = config.sliding_window if idx % 2 == 0 else 0
    probe.tensor("sinks", sinks)
    with jax.named_scope("attention_kernel"):
        t = attn_op(config.mode, config.implementation)(q, k, v, sinks, config.head_dim**-0.5, sliding_window, start_pos)
    probe.tensor("attn_out", t)

    if exists(w.sdpa_gate):
        gate = 2.0 * jax.nn.sigmoid(
            dot(h.astype(jnp.float32), arr(w.sdpa_gate), dimension_numbers=NN_INNER, preferred_element_type=jnp.float32),
        )
        t *= gate.swapaxes(1, 2)[..., None].astype(dtype)
        if probe.enabled:
            probe.scalars("sdpa_gate", {"mean": jnp.mean(gate), "std": jnp.std(gate)})

    if config.use_learned_xsa:
        with jax.named_scope("xsa"):
            v32 = current_v.astype(jnp.float32)
            vn = (v32 * lax.rsqrt(jnp.sum(v32 * v32, axis=-1, keepdims=True) + 1e-10)).astype(dtype)
            n_rep, n_kv_local = config.n_attn_heads // config.n_kv_heads, vn.shape[1]
            t_g = t.reshape(B, n_kv_local, n_rep, T, config.head_dim)
            vn_g = vn[:, :, None, :, :]
            coef = jnp.sum((t_g * vn_g).astype(jnp.float32), axis=-1, keepdims=True)
            alpha = jnp.tanh(arr(w.xsa_alpha).astype(jnp.float32)).reshape(1, n_kv_local, n_rep, 1, 1).astype(dtype)
            t = (t_g - alpha * coef.astype(dtype) * vn_g).reshape(B, -1, T, config.head_dim)

        probe.tensor("xsa_alpha", alpha)
        probe.tensor("xsa_out", t)

    with jax.named_scope("output_projection"):
        t = psum_axes(dot(t.swapaxes(1, 2).reshape(B, T, -1), wc(w.out), dimension_numbers=NN_INNER), config)
    if exists(w.out_bias):
        t += ac(w.out_bias)
    probe.tensor("out_proj", t)

    t *= config.residual_scale
    probe.residual("residual", x, t)

    return (x + t).astype(dtype), updated_cache


# MoE
@register_dataclass
@dataclass
class MoEWeights:
    router: Param
    router_bias: OptionalParam
    mlp_up: QParam
    mlp_up_bias: OptionalParam
    mlp_down: QParam
    mlp_down_bias: OptionalParam
    norm_scale: Param

    @classmethod
    def spec(cls, config: ModelConfig) -> MoEWeights:
        c, s = config, config.sharding
        hidden = int(c.n_embd * c.expert_mlp_widening) // 2
        router_init = init.variance_scaling(scale=c.router_logit_std**2, mode="fan_in", distribution="truncated_normal")
        return MoEWeights(
            router=ArraySpec((c.n_embd, c.n_routed_experts), jnp.float32, s.router_spec, router_init, Tag.DEFAULT),
            router_bias=ArraySpec((c.n_routed_experts,), jnp.float32, s.router_bias_spec, c.b_init, Tag.BIAS)
            if c.use_router_bias
            else None,
            mlp_up=_matrix(c, (c.n_experts, c.n_embd, 2, hidden), s.moe_mlp_up_spec, stacked_init(c.w_init)),
            mlp_up_bias=_bias(c, (c.n_experts, 2, hidden), s.moe_mlp_up_bias_spec, c.use_mlp_bias),
            mlp_down=_matrix(c, (c.n_experts, hidden, c.n_embd), s.moe_mlp_down_spec, stacked_init(c.w_init)),
            mlp_down_bias=_bias(c, (c.n_experts, c.n_embd), s.moe_mlp_down_bias_spec, c.use_mlp_bias),
            norm_scale=_scalar((c.n_embd,), s.norm_scale_spec, c.norm_init),
        )

    @staticmethod
    def quantize(w: MoEWeights, config: ModelConfig) -> MoEWeights:
        if not config.quantize_mlp:
            return w
        up, down = qarr(w.mlp_up), qarr(w.mlp_down)
        if not isinstance(up, QArray):
            up = _quantize_block(up.reshape(*up.shape[:-2], -1), config)
        if not isinstance(down, QArray):
            down = _quantize_block(down, config)
        return replace(w, mlp_up=up, mlp_down=down)


def calculate_aux_loss(router_logits: Array, expert_indices: Array, config: ModelConfig, *, probe: Probe = NO_PROBE) -> Array:
    F, E = router_logits.shape
    K = expert_indices.shape[-1]
    logits = router_logits.astype(jnp.float32)
    counts = jnp.bincount(jnp.ravel(expert_indices), minlength=E, length=E)
    tokens = jnp.asarray(F, dtype=jnp.float32)

    assigned = softmax(logits, axis=-1).sum(axis=0) / tokens
    assignments = lax.stop_gradient(counts / (tokens * K))
    lb_term = E * jnp.sum(assigned * assignments, dtype=jnp.float32)
    z_term = jnp.mean(jnp.square(logsumexp(logits, axis=-1)), dtype=jnp.float32)
    lb_loss, z_loss = config.load_balance_weight * lb_term, config.router_z_loss_weight * z_term
    aux_loss = lb_loss + z_loss

    if probe.enabled:
        mean_count = jnp.maximum(counts.mean(), 1.0)
        probe.scalars(
            "aux_loss",
            {
                "lb_term": lb_term,
                "lb_loss": lb_loss,
                "z_term": z_term,
                "z_loss": z_loss,
                "aux_loss": aux_loss,
                "load_cv": counts.std() / mean_count,
                "max_load_violation": counts.max() / mean_count - 1.0,
                "min_expert_count": counts.min(),
                "max_expert_count": counts.max(),
            },
        )
    return aux_loss.astype(jnp.float32)


def route_tokens(t: Array, w: MoEWeights, config: ModelConfig, *, probe: Probe = NO_PROBE) -> tuple[Array, Array, Array, Array]:
    tokens = t.shape[0]
    routed, shared = config.n_routed_experts, config.n_shared_experts
    with jax.named_scope("router"):
        router_logits = dot(lhs=t, rhs=arr(w.router), preferred_element_type=jnp.float32, precision=lax.Precision.HIGH)
    if exists(w.router_bias):
        router_logits += arr(w.router_bias)

    routed_logits, indices = lax.top_k(router_logits, config.n_active_routed_experts)
    scores = softmax(routed_logits, axis=-1)
    aux_loss = calculate_aux_loss(router_logits, indices, config, probe=probe)
    if probe.enabled:
        routed_groups = jnp.bincount(jnp.ravel(indices), minlength=routed, length=routed)
        probe.routing(
            "routing",
            router_logits=router_logits,
            expert_scores=scores,
            group_sizes=routed_groups,
            n_experts=routed,
            n_active=config.n_active_routed_experts,
        )

    if shared:
        shared_indices = jnp.broadcast_to(jnp.arange(routed, config.n_experts, dtype=indices.dtype)[None, :], (tokens, shared))
        indices = jnp.concatenate((indices, shared_indices), axis=-1)
        scores = jnp.concatenate((scores, jnp.ones((tokens, shared), dtype=scores.dtype)), axis=-1)
    active_map = jnp.ravel(indices)
    groups = jnp.bincount(active_map, minlength=config.n_experts, length=config.n_experts)
    
    return scores, active_map, groups, aux_loss


def bpermute(expert_ids: Array, group_sizes: Array, n_experts: int) -> tuple[Array, Array]:
    membership = jax.nn.one_hot(expert_ids, n_experts, dtype=jnp.int32)
    local_rank = jnp.take_along_axis(jnp.cumulative_sum(membership, axis=0) - 1, expert_ids[:, None], axis=1)
    destinations = (jnp.cumulative_sum(group_sizes) - group_sizes)[expert_ids] + local_rank[:, 0]
    
    assignments = jnp.arange(expert_ids.shape[0], dtype=destinations.dtype)
    permutation = jnp.empty_like(assignments).at[destinations].set(assignments, unique_indices=True)
    
    return permutation, destinations


def _fuse_glu[T](p: T) -> T:
    return p if isinstance(p, QArray) else p.reshape(*p.shape[:-2], -1)  # type: ignore


@jax.named_scope("moe")
def moe_apply(x: Array, w: MoEWeights, config: ModelConfig, *, probe: Probe = NO_PROBE) -> tuple[Array, Array]:
    B, T, C = x.shape
    tokens = B * T
    gmm = gmm_op(config.mode, config.implementation)

    xn = rms_norm(x, arr(w.norm_scale), config.norm_eps)
    probe.tensor("norm_out", xn)
    xn = xn.reshape(tokens, C)
    with jax.named_scope("route_tokens"):
        scores, active_map, groups, aux_loss = route_tokens(xn, w, config, probe=probe)

    expert_parallel = config.sharding.expert_axis_size > 1
    split_shared = expert_parallel and config.n_shared_experts > 0
    topk = config.n_active_experts
    if split_shared:
        topk = config.n_active_routed_experts
        active_map = active_map.reshape(tokens, -1)[:, :topk].reshape(-1)
        groups = groups.at[config.n_routed_experts :].set(0)
        scores = scores[:, :topk]

    with jax.named_scope("permute"):
        permute_map, unpermute_map = bpermute(active_map, groups, config.n_experts)
        t = xn[permute_map // topk, ...]
    probe.tensor("gemm_input", t)

    mlp_up, mlp_down = _fuse_glu(qarr(w.mlp_up)), qarr(w.mlp_down)
    up_bias = _fuse_glu(acast(w.mlp_up_bias, config.compute_dtype)) if exists(w.mlp_up_bias) else None
    down_bias = acast(w.mlp_down_bias, config.compute_dtype) if exists(w.mlp_down_bias) else None

    shared_out = None
    if expert_parallel:
        t = ep_moe_gemm(t, groups, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
        if split_shared:
            shared_out = ep_shared_mlp(xn, mlp_up, mlp_down, up_bias, down_bias, config, gmm)
    else:
        with jax.named_scope("expert_up"):
            t = gmm(
                lhs=t,
                rhs=mlp_up,
                group_sizes=groups,
                bias=up_bias,
                fused_swiglu=True,
                swiglu_beta=config.swiglu_beta,
                swiglu_limit=config.swiglu_limit,
                quant_dtype=config.gemm_dtype,
            )
        probe.tensor("up_out", t)
        
        with jax.named_scope("expert_down"):
            bias = down_bias / config.sharding.model_axis_size if exists(down_bias) else None
            t = psum_axes(gmm(lhs=t, rhs=mlp_down, group_sizes=groups, bias=bias, quant_dtype=config.gemm_dtype), config)
    
    probe.tensor("down_out", t)

    with jax.named_scope("unpermute_combination"):
        if not expert_parallel and config.implementation != "xla" and supports_combine(t, scores):
            t = combine(t, scores, permute_map, unpermute_map)
        else:
            t = einsum("fkc,fk->fc", t[unpermute_map].reshape(tokens, topk, C), scores)
        
        if shared_out is not None:
            t += shared_out # type: ignore
    
    t = t.reshape(B, T, C).astype(x.dtype) # type: ignore
    probe.tensor("combined", t)

    t *= config.residual_scale
    probe.residual("residual", x, t)
    return x + t, aux_loss


# Dense MLP


@register_dataclass
@dataclass
class DenseMLPWeights:
    mlp_up: QParam
    mlp_up_bias: OptionalParam
    mlp_down: QParam
    mlp_down_bias: OptionalParam
    norm_scale: Param

    @classmethod
    def spec(cls, config: ModelConfig) -> DenseMLPWeights:
        c, s = config, config.sharding
        hidden = int(c.n_embd * c.dense_mlp_widening) // 2
        return DenseMLPWeights(
            mlp_up=_matrix(c, (c.n_embd, 2, hidden), s.dense_mlp_up_spec),
            mlp_up_bias=_bias(c, (1, 2, hidden), s.dense_mlp_up_bias_spec, c.use_mlp_bias),
            mlp_down=_matrix(c, (hidden, c.n_embd), s.dense_mlp_down_spec),
            mlp_down_bias=_bias(c, (1, c.n_embd), s.dense_mlp_down_bias_spec, c.use_mlp_bias),
            norm_scale=_scalar((c.n_embd,), s.norm_scale_spec, c.norm_init),
        )

    @staticmethod
    def quantize(w: DenseMLPWeights, config: ModelConfig) -> DenseMLPWeights:
        if not config.quantize_mlp:
            return w
        up, down = qarr(w.mlp_up), qarr(w.mlp_down)
        if not isinstance(up, QArray):
            k, n = config.quant_block
            up = quantize(up, config.gemm_dtype, (math.gcd(up.shape[-3], k), 1, math.gcd(up.shape[-1], n)))
        if not isinstance(down, QArray):
            down = _quantize_block(down, config)
        
        return replace(w, mlp_up=up, mlp_down=down)


@jax.named_scope("dense_mlp")
def dense_mlp_apply(x: Array, w: DenseMLPWeights, config: ModelConfig, *, probe: Probe = NO_PROBE) -> Array:
    t = rms_norm(x, arr(w.norm_scale), config.norm_eps)
    probe.tensor("norm_out", t)
    
    with jax.named_scope("up_projection"):
        t = dot(t, wcast(w.mlp_up, config.compute_dtype), dimension_numbers=NN_INNER)
        if exists(w.mlp_up_bias):
            t += acast(w.mlp_up_bias, config.compute_dtype)
    t = t.reshape(*t.shape[:-2], -1)
    
    if probe.detailed:
        probe.tensor("up_out", t)
        probe.swiglu("swiglu", t, limit=config.swiglu_limit)
        
    with jax.named_scope("swiglu"):
        t = swiglu(t, beta=config.swiglu_beta, limit=config.swiglu_limit)
    probe.tensor("act_out", t)
    
    with jax.named_scope("down_projection"):
        t = psum_axes(dot(t, wcast(w.mlp_down, config.compute_dtype), dimension_numbers=NN_INNER), config)
        if exists(w.mlp_down_bias):
            t += acast(w.mlp_down_bias, config.compute_dtype)
    probe.tensor("down_out", t)

    t *= config.dense_residual_scale
    probe.residual("residual", x, t)
    
    return (x + t).astype(x.dtype)


MLPWeights = DenseMLPWeights | MoEWeights


def _mlp_apply(x: Array, w: MLPWeights, config: ModelConfig, *, probe: Probe) -> tuple[Array, Array]:
    if isinstance(w, MoEWeights):
        return moe_apply(x, w, config, probe=probe)
    return dense_mlp_apply(x, w, config, probe=probe), jnp.array(0.0, dtype=jnp.float32)


@partial(jax.checkpoint, static_argnames=("config",), prevent_cse=True)
def _remat_mlp_apply(x: Array, w: MLPWeights, config: ModelConfig) -> tuple[Array, Array]:
    return _mlp_apply(x, w, config, probe=NO_PROBE)


def mlp_apply(x: Array, w: MLPWeights, config: ModelConfig, *, probe: Probe = NO_PROBE) -> tuple[Array, Array]:
    if probe.detailed:
        return _mlp_apply(x, w, config, probe=probe)
    out, aux_loss = _remat_mlp_apply(x, w, config)
    if probe.enabled and isinstance(w, MoEWeights):
        B, T, C = x.shape
        route_tokens(rms_norm(x, arr(w.norm_scale), config.norm_eps).reshape(B * T, C), w, config, probe=probe)
    return out, aux_loss


# Model 

@register_dataclass
@dataclass
class TokenEmbeddings:
    emb: Param
    unemb: OptionalParam
    norm_scale: Param

    @classmethod
    def spec(cls, config: ModelConfig) -> TokenEmbeddings:
        c, s = config, config.sharding
        shape = (c.padded_vocab_size, c.n_embd)
        emb_init = init.variance_scaling(scale=1.0, mode="fan_out", in_axis=0, out_axis=1, distribution="truncated_normal")
        unemb_init = init.variance_scaling(scale=1.0, mode="fan_in", in_axis=1, out_axis=0, distribution="truncated_normal")
        return TokenEmbeddings(
            emb=ArraySpec(shape, c.param_dtype, s.emb_spec, emb_init, Tag.EMBEDDING),
            unemb=None if c.tie_embeddings else ArraySpec(shape, c.param_dtype, s.unemb_spec, unemb_init, Tag.EMBEDDING),
            norm_scale=_scalar((c.n_embd,), s.norm_scale_spec, c.norm_init),
        )


def lm_head_weight(tok: TokenEmbeddings, config: ModelConfig) -> Array:
    return arr(tok.emb if config.tie_embeddings else tok.unemb)


@register_dataclass
@dataclass
class LayerWeights:
    attn: AttentionWeights
    mlp: MLPWeights

    @classmethod
    def spec(cls, idx: int, config: ModelConfig) -> LayerWeights:
        mlp = MoEWeights.spec(config) if config.mlp_map[idx] == "moe" else DenseMLPWeights.spec(config)
        return LayerWeights(attn=AttentionWeights.spec(config), mlp=mlp)


@register_dataclass
@dataclass
class ModelWeights:
    tok: TokenEmbeddings
    layers: list[LayerWeights]

    @classmethod
    def spec(cls, config: ModelConfig) -> ModelWeights:
        return ModelWeights(tok=TokenEmbeddings.spec(config), layers=[LayerWeights.spec(idx, config) for idx in range(config.n_layers)])

    @classmethod
    def init(cls, key: Array, config: ModelConfig) -> ModelWeights:
        if not config.use_sdpa_output_gate:
            return init_weights(key, cls.spec(config))
        w = init_weights(key, cls.spec(replace(config, use_sdpa_output_gate=False)))
        gate_spec = AttentionWeights.spec(config).sdpa_gate
        assert isinstance(gate_spec, ArraySpec)
        gate = lambda idx: init_spec(jax.random.fold_in(key, idx), gate_spec)
        w.layers = [replace(layer, attn=replace(layer.attn, sdpa_gate=gate(idx))) for idx, layer in enumerate(w.layers)]
        return w

    @classmethod
    def load(cls, source: str, config: ModelConfig) -> ModelWeights:
        return load_weights(source, cls.spec(config))

    @staticmethod
    def quantize(w: ModelWeights, config: ModelConfig) -> ModelWeights:
        mlp = lambda m: MoEWeights.quantize(m, config) if isinstance(m, MoEWeights) else DenseMLPWeights.quantize(m, config)
        layers = [LayerWeights(attn=AttentionWeights.quantize(layer.attn, config), mlp=mlp(layer.mlp)) for layer in w.layers]
        return ModelWeights(tok=w.tok, layers=layers)


@jax.named_scope("model")
def model_apply(
    tokens: Int[Array, "B T"],
    model: ModelWeights,
    cache: ModelCache | None,
    config: ModelConfig,
    *,
    return_hidden_states: bool = False,
    probe: Probe = NO_PROBE,
) -> tuple[Array, Array, ModelCache | None]:
    T = tokens.shape[1]
    
    with jax.named_scope("token_embedding"):
        emb = arr(model.tok.emb)
        h = embs_gather(emb, tokens, emb.shape[0]) if config.use_presorted_emb_gather else jnp.take(emb, tokens, axis=0)
        h = h.astype(config.compute_dtype)
    probe.tensor("embs", h)

    start_pos = cache.pos if cache is not None else 0
    cos, sin = precompute_freq_cis(config)
    rope_slice = partial(lax.dynamic_slice_in_dim, start_index=start_pos, slice_size=T, axis=0)
    freq_cis = (rope_slice(cos)[None, :, None, :], rope_slice(sin)[None, :, None, :])

    aux_loss = jnp.array(0.0, dtype=jnp.float32)
    caches = []
    for idx, layer in enumerate(model.layers):
        layer_probe = probe.scope(f"layer.{idx:02d}")
        with jax.named_scope(f"layer_{idx:02d}"):
            layer_cache = cache.layers[idx] if cache is not None else None
            h, layer_cache = attention_apply(h, layer.attn, idx, start_pos, freq_cis, layer_cache, config, probe=layer_probe.scope("attn"))
            h, layer_aux = mlp_apply(h, layer.mlp, config, probe=layer_probe.scope("mlp"))
        aux_loss += layer_aux
        caches.append(layer_cache)
    
    if config.n_moe_layers:
        aux_loss /= config.n_moe_layers
    updated_cache = None if cache is None else ModelCache(layers=tuple(cast(list[LayerCache], caches)), pos=start_pos + T)

    head = probe.scope("head")
    head.tensor("pre_norm", h)
    
    with jax.named_scope("output_norm"):
        h = rms_norm(h, arr(model.tok.norm_scale), config.norm_eps)
    head.tensor("norm_out", h)

    unemb = lm_head_weight(model.tok, config)
    if return_hidden_states:
        head.logit_subsample("logits_subsample", h, unemb, vocab_size=config.vocab_size)
        return h, aux_loss, updated_cache
    
    logits = dot(
        lhs=h,
        rhs=acast(unemb, config.compute_dtype),
        dimension_numbers=NN_RHS_T,
        preferred_element_type=jnp.float32,
        precision=lax.Precision.HIGH,
    )
    logits = logits.at[..., config.vocab_size :].set(-jnp.inf)
    head.logits("logits", logits)
    return logits, aux_loss, updated_cache
