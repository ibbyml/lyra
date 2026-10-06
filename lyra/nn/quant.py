from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_dataclass
from jaxtyping import Array

FP8_DTYPES = (jnp.float8_e4m3fn, jnp.float8_e5m2)
is_scaled = lambda dt: jnp.dtype(dt) in tuple(jnp.dtype(d) for d in FP8_DTYPES)


@register_dataclass
@dataclass(frozen=True)
class QArray:
    qval: Array
    scale: Array
    block: tuple[int, ...] = field(metadata={"static": True})

    dtype = property(lambda self: self.qval.dtype)
    shape = property(lambda self: self.qval.shape)
    ndim = property(lambda self: self.qval.ndim)
    stype = property(lambda self: self.scale.dtype)

    @property
    def ldim(self) -> int:
        return self.qval.ndim - len(self.block)

    def eff_block(self, axis: int) -> int:
        return 1 if axis < self.ldim else self.block[axis - self.ldim]


def tile(x: Array, block: tuple[int, ...]) -> Array:
    """[..., d0, d1] -> [..., d0//b0, b0, d1//b1, b1]. Pure reshape."""
    n = len(block)
    lead, tail = x.shape[: x.ndim - n], x.shape[x.ndim - n :]
    bad = [(d, b) for d, b in zip(tail, block) if d % b]
    if bad:
        raise ValueError(f"block {block} does not divide trailing dims {tail}")
    return x.reshape(*lead, *(v for d, b in zip(tail, block) for v in (d // b, b)))


def block_axes(ldim: int, sdim: int) -> tuple[int, ...]:
    axes = tuple(ldim + 2 * i + 1 for i in range(sdim))
    return axes


def quantize(
    x: Array | QArray,
    dtype: jnp.dtype,
    block: tuple[int, ...] = (128, 128),
    scale_dtype: jnp.dtype = jnp.float32,
) -> QArray:
    if isinstance(x, QArray):
        if x.block != block or x.dtype != dtype:
            raise ValueError(f"already quantized as {x.dtype}/{x.block}; requested {dtype}/{block}")
        return x
    if not is_scaled(dtype):
        raise ValueError(f"quantize target must be fp8, got {dtype}")
    if is_scaled(scale_dtype):
        raise ValueError("scale dtype must be unscaled (fp32/bf16)")

    sdim = len(block)
    ldim = x.ndim - sdim
    axes = block_axes(ldim, sdim)

    tiled = tile(x, block)
    fmax = float(jnp.finfo(dtype).max)
    amax = jnp.max(jnp.abs(tiled), axis=axes).astype(jnp.float32)
    scale = jnp.where(amax > 0, amax / fmax, 1.0).astype(scale_dtype)
    qinv = jnp.expand_dims(1.0 / scale.astype(jnp.float32), axes)
    qval = jnp.clip(tiled * qinv, -fmax, fmax).astype(dtype).reshape(x.shape)
    qx = QArray(qval, scale, block)

    return qx


def dequantize(
    qx: QArray,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Array:
    axes = block_axes(qx.ldim, len(qx.block))
    tiled = tile(qx.qval.astype(jnp.float32), qx.block)
    scale = jnp.expand_dims(qx.scale.astype(jnp.float32), axes)
    x = (tiled * scale).reshape(qx.shape).astype(dtype)
    return x


def qupdate(qx: QArray, x: Array, start: tuple) -> QArray:
    for axis, s in enumerate(start):
        b = qx.eff_block(axis)
        if isinstance(s, int) and s % b:
            raise ValueError(f"start {s} on axis {axis} is not aligned to block {b}")
    qnew = quantize(x, qx.dtype, qx.block, qx.stype)
    scale_start = tuple(s // qx.eff_block(axis) for axis, s in enumerate(start))
    qval = lax.dynamic_update_slice(qx.qval, qnew.qval, start)
    scale = lax.dynamic_update_slice(qx.scale, qnew.scale, scale_start)
    return QArray(qval, scale, qx.block)


def maybe_dequant(x: Array | QArray, dtype: jnp.dtype) -> Array:
    out = dequantize(x, dtype) if isinstance(x, QArray) else x
    return out


def dot(lhs: Array, rhs: Array | QArray, **kw):
    out = lax.dot(lhs, maybe_dequant(rhs, lhs.dtype), **kw)
    return out


def einsum(subscript: str, lhs: Array, rhs: Array | QArray, **kw):
    out = jnp.einsum(subscript, lhs, maybe_dequant(rhs, lhs.dtype), **kw)
    return out
