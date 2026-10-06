from __future__ import annotations

from importlib import import_module

import jax.numpy as jnp
import numpy as np

from lyra.kernels.decode_gemm import _lhs_row_map
from lyra.nn.quant import QArray, quantize

decode_attention_module = import_module("lyra.kernels.decode_attention")


def test_quantized_decode_kernel_receives_quantized_cache(monkeypatch) -> None:
    q = jnp.ones((1, 2, 1, 128), dtype=jnp.bfloat16)
    kv = jnp.ones((1, 1, 128, 128), dtype=jnp.bfloat16)
    qkv = quantize(kv, jnp.float8_e4m3fn, (1, 128))
    sinks = jnp.zeros(2, dtype=jnp.float32)
    config = decode_attention_module.DecodeAttentionConfig(cache_dtype=jnp.float8_e4m3fn)
    received: dict[str, object] = {}

    monkeypatch.setattr(decode_attention_module, "select_decode_attention_config", lambda **_: config)

    def fake_kernel(kernel_q, kernel_k, kernel_v, *_):
        received["k"] = kernel_k
        received["v"] = kernel_v
        return jnp.zeros_like(kernel_q)

    monkeypatch.setattr(decode_attention_module, "_decode_attention", fake_kernel)
    monkeypatch.setattr(decode_attention_module, "dispatch_kernel", lambda *_args, kernel, **_kwargs: kernel())

    decode_attention_module.decode_attention(
        q,
        qkv,
        qkv,
        sinks,
        1.0,
        0,
        jnp.asarray(0, dtype=jnp.int32),
        implementation="pallas",
    )

    assert isinstance(received["k"], QArray)
    assert isinstance(received["v"], QArray)


def test_decode_gemm_row_map_builds() -> None:
    group_sizes = jnp.array([1, 1], dtype=jnp.int32)
    row_map = _lhs_row_map(group_sizes, tile_g=1, tile_m=1, row_tiles=2)
    np.testing.assert_array_equal(np.asarray(row_map), np.array([[0, 1], [1, 1]], dtype=np.int32))


def test_auto_decode_attention_uses_xla(monkeypatch) -> None:
    q = jnp.ones((1, 2, 1, 128), dtype=jnp.bfloat16)
    kv = jnp.ones((1, 1, 128, 128), dtype=jnp.bfloat16)
    sinks = jnp.zeros(2, dtype=jnp.float32)
    config = decode_attention_module.DecodeAttentionConfig()
    monkeypatch.setattr(decode_attention_module, "select_decode_attention_config", lambda **_: config)

    def fail(*_):
        raise AssertionError("auto dispatch selected the Pallas decode kernel")

    monkeypatch.setattr(decode_attention_module, "_decode_attention", fail)
    out = decode_attention_module.decode_attention(q, kv, kv, sinks, 1.0, 0, jnp.asarray(0, dtype=jnp.int32), implementation="auto")
    assert out.shape == q.shape
