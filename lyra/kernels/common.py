import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu
from jax.nn import sigmoid
from jax.tree_util import register_dataclass
from jaxtyping import Array

Reduction = Literal["sum", "mean"]
Implementation = Literal["auto", "pallas", "xla"]

TILE_INNER = (((1,), (0,)), ((), ()))  # [M, K] @ [K, N] -> [M, N]
TILE_LHS_T = (((0,), (0,)), ((), ()))  # [K, M] @ [K, N] -> [M, N]
TILE_RHS_T = (((1,), (1,)), ((), ()))  # [M, K] @ [N, K] -> [M, N]

LOG2_E = math.log2(math.e)


@register_dataclass
@dataclass(frozen=True)
class InputArray:
    value: Array
    scale: Array | None = None
    block: tuple[int, ...] | None = field(default=None, metadata={"static": True})

    dtype = property(lambda self: self.value.dtype)
    shape = property(lambda self: self.value.shape)
    ndim = property(lambda self: self.value.ndim)

    def get_value(self) -> Array:
        return self.value[...]

    def get_scale(self) -> Array:
        assert self.scale is not None
        return self.scale[...]


InputRef = InputArray


def current_tpu_target():
    if not is_tpu_backend():
        return None
    return str(pltpu.get_tpu_info().chip_version)


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def divides(dividend: int, divisor: int) -> bool:
    return divisor != 0 and dividend % divisor == 0


def FP32(x: Array) -> Array:
    return x.astype(jnp.float32)


def BF16(x: Array) -> Array:
    return x.astype(jnp.bfloat16)


def is_tpu_backend() -> bool:
    return jax.default_backend() == "tpu"


def tpu_vmem_limit_bytes(fraction: float = 0.9) -> int:
    if not is_tpu_backend():
        return 1
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * fraction)


def tpu_sublanes(dtype: jnp.dtype) -> int:
    if not is_tpu_backend():
        return 1
    return pltpu.get_tpu_info().get_sublane_tiling(dtype)


def tpu_lanes() -> int:
    if not is_tpu_backend():
        return 1
    return pltpu.get_tpu_info().num_lanes


mxu_dot = partial(lax.dot, preferred_element_type=jnp.float32)
mxu_ragged_dot = partial(lax.ragged_dot, preferred_element_type=jnp.float32)


def swiglu(
    x: Array,
    beta: float = 1.702,
    limit: float = 7.0,
    shift: float = 1.0,
    interleaved: bool = False,
) -> Array:
    dtype = x.dtype
    if interleaved:
        glu, lin = x[..., ::2], x[..., 1::2]
    else:
        glu, lin = jnp.split(x, 2, axis=-1)
    glu = glu.clip(max=limit)
    lin = lin.clip(min=-limit, max=limit)
    gate = glu * sigmoid(beta * glu)
    out = (lin + shift) * gate
    return out.astype(dtype)


_KERNEL_LOG_SEEN: set[str] = set()
_KERNEL_LOG_ENABLED = os.environ.get("LYRA_LOG_KERNELS", "1") != "0"


def log_kernel_resolution(op: str, *, shape: str, impl: str, tiling: str | None = None) -> None:
    if not _KERNEL_LOG_ENABLED:
        return
    key = f"{op}|{shape}|{impl}|{tiling}"
    if key in _KERNEL_LOG_SEEN:
        return
    _KERNEL_LOG_SEEN.add(key)
    detail = impl if tiling is None else f"{impl:<6} {tiling}"
    print(f"[kernel] {op:<15} {shape:<26} -> {detail}", file=sys.stderr)


def dispatch_kernel(
    op: str,
    *,
    implementation: Implementation,
    shape: str,
    config: object | None,
    tiling: str | None,
    kernel: Callable[[], Array],
    reference: Callable[[], Array],
    fallback_reason: str | None = None,
) -> Array:
    if config is not None and implementation != "xla":
        assert tiling is not None
        log_kernel_resolution(op, shape=shape, impl="pallas", tiling=tiling)
        return kernel()
    if implementation == "pallas":
        raise ValueError(
            f"{op}: no Pallas tile configuration supports inputs ({shape}). "
            "Pass implementation='auto' or 'xla', or pad the inputs to a supported shape."
        )
    reason = fallback_reason or "no Pallas tiles for shape"
    fallback = f" ({reason})" if implementation == "auto" else ""
    log_kernel_resolution(op, shape=shape, impl=f"xla{fallback}")
    return reference()
