from dataclasses import replace
from functools import partial
from importlib import import_module
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from lyra.kernels import cross_entropy as ce
from lyra.kernels.flash_attention import FlashAttentionConfig, select_flash_attention_config
from lyra.kernels.flash_attention import _op as attention_op


def test_attention_specializations_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    attention = import_module("lyra.kernels.flash_attention")
    monkeypatch.setattr(attention, "current_tpu_target", lambda: "v6e")
    monkeypatch.setattr(jax, "device_count", lambda: 1)
    for window in (0, 128):
        config = select_flash_attention_config(4096, 4096, 16, 4, 128, window, jnp.bfloat16, batch_size=4)
        assert config is not None
        assert config.heads_per_program_fwd == 4
        assert config.clamp_causal_prefetch == (window == 0)
        assert config.kv_major_dkv == (window == 128)
        for batch, dtype in ((8, jnp.bfloat16), (4, jnp.float32)):
            fallback = select_flash_attention_config(4096, 4096, 16, 4, 128, window, dtype, batch_size=batch)
            assert fallback is not None
            assert fallback.heads_per_program_fwd == 1
            assert not fallback.clamp_causal_prefetch
            assert not fallback.kv_major_dkv
    monkeypatch.setattr(jax, "device_count", lambda: 2)
    config = select_flash_attention_config(4096, 4096, 16, 4, 128, 128, jnp.bfloat16, batch_size=4)
    assert config is not None and config.kv_major_dkv and config.heads_per_program_fwd == 1


@pytest.mark.parametrize("window", [0, 128])
def test_attention_interpreted_vjp_matches_base(monkeypatch: pytest.MonkeyPatch, window: int) -> None:
    monkeypatch.setattr(pl, "pallas_call", partial(pl.pallas_call, interpret=True))
    keys = jax.random.split(jax.random.key(17), 5)
    q = jax.random.normal(keys[0], (1, 4, 256, 128), jnp.bfloat16)
    k = jax.random.normal(keys[1], (1, 1, 256, 128), jnp.bfloat16)
    v = jax.random.normal(keys[2], k.shape, jnp.bfloat16)
    sinks = jax.random.normal(keys[3], (4,), jnp.float32)
    grad = jax.random.normal(keys[4], q.shape, jnp.bfloat16)
    base = FlashAttentionConfig(banded_fwd=bool(window), banded_dq=bool(window), banded_dkv=bool(window))
    selected = replace(base, heads_per_program_fwd=4, clamp_causal_prefetch=window == 0, kv_major_dkv=window == 128)

    def run(config: FlashAttentionConfig):
        attention = partial(attention_op, sm_scale=128**-0.5, sliding_window=window, config=config)
        out, backward = jax.vjp(attention, q, k, v, sinks)
        return out, backward(grad)

    expected = jax.device_get(jax.jit(lambda: run(base))())
    actual = jax.device_get(jax.jit(lambda: run(selected))())
    for result, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(result, reference)


def test_ce_static_vjp_and_ref_accumulation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ce.LinearCrossEntropyConfig()
    w = jnp.ones((2, 2), jnp.float32)
    x = jnp.ones((2, 2), jnp.bfloat16)
    y = jnp.zeros((2, 1), jnp.int32)

    def forward(w: jax.Array, x: jax.Array, y: jax.Array, reduction: str, vocab_size: int, received_config: Any):
        del y
        assert (reduction, vocab_size, received_config) == ("sum", 2, config)

        return (jnp.sum(w) + jnp.sum(x.astype(jnp.float32))).reshape(1), jnp.zeros((2,), jnp.float32)

    def dw(ctx: tuple, grad: jax.Array, reduction: str, vocab_size: int, received_config: Any):
        assert (reduction, vocab_size, received_config) == ("sum", 2, config)
        return jnp.full_like(ctx[0], 3 * grad.reshape(()))

    def dx(ctx: tuple, grad: jax.Array, reduction: str, vocab_size: int, received_config: Any):
        assert (reduction, vocab_size, received_config) == ("sum", 2, config)
        return jnp.full_like(ctx[1], 5 * grad.reshape(()))

    monkeypatch.setattr(ce, "_linear_cross_entropy", forward)
    monkeypatch.setattr(ce, "_linear_cross_entropy_bwd_dw", dw)
    monkeypatch.setattr(ce, "_linear_cross_entropy_bwd_dx", dx)
    monkeypatch.setattr(ce, "_linear_cross_entropy_bwd_dw_accum", lambda ctx, grad, acc, *args: acc + dw(ctx, grad, *args))

    for op in (ce._op, ce._accumulating_op):

        def loss(w_: jax.Array, x_: jax.Array, op=op):
            return op(w_, x_, y, "sum", vocab_size=2, config=config)

        _, pullback = jax.vjp(loss, w, x)
        dw_actual, dx_actual = pullback(jnp.full((1,), 2, jnp.float32))
        np.testing.assert_array_equal(dw_actual, jnp.full_like(w, 6))
        np.testing.assert_array_equal(dx_actual, jnp.full_like(x, 10))

    _, pullback = jax.vjp(lambda w_: ce._accumulating_op(w_, x, y, "sum", 2, config), w)
    dw_ref = jax.new_ref(jnp.full_like(w, 7))
    cast(Any, pullback).with_refs(dw_ref)(jnp.full((1,), 2, jnp.float32))
    np.testing.assert_array_equal(jax.freeze(dw_ref), jnp.full_like(w, 13))

    def batched_loss(w_: jax.Array):
        return jax.vmap(lambda x_: ce._accumulating_op(w_, x_, y, "sum", 2, config))(jnp.stack((x, x))).sum()

    np.testing.assert_array_equal(jax.jit(jax.grad(batched_loss))(w), jnp.full_like(w, 6))


def test_ce_accumulator_stages_aliased_call() -> None:
    config = ce.LinearCrossEntropyConfig()
    w = jnp.ones((128, 128), jnp.float32)
    x = jnp.ones((128, 128), jnp.bfloat16)
    y = jnp.zeros((128, 1), jnp.int32)
    lse = jnp.zeros((128,), jnp.float32)
    grad = jnp.ones((1,), jnp.float32)
    staged = str(jax.make_jaxpr(lambda acc: ce._linear_cross_entropy_bwd_dw_accum((w, x, y, lse), grad, acc, "sum", 127, config))(w))
    assert "input_output_aliases=((5, 0),)" in staged


def test_ce_gate_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ce, "current_tpu_target", lambda: "v6e")
    monkeypatch.setattr(jax, "device_count", lambda: 1)
    config = ce.LinearCrossEntropyConfig(tile_t_dw=1024, tile_v_dw=2048)
    w = cast(Any, jax.ShapeDtypeStruct((204800, 2048), jnp.float32))
    x = cast(Any, jax.ShapeDtypeStruct((16384, 2048), jnp.bfloat16))
    y = cast(Any, jax.ShapeDtypeStruct((16384, 1), jnp.int32))
    assert ce._use_accumulating_ce(w, x, y, 201088, config)
    assert not ce._use_accumulating_ce(w, x, y, 201087, config)
    assert not ce._use_accumulating_ce(w, x, y, 201088, replace(config, tile_v_dw=1024))
    monkeypatch.setattr(jax, "device_count", lambda: 2)
    assert not ce._use_accumulating_ce(w, x, y, 201088, config)
