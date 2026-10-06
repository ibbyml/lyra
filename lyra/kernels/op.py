from enum import Enum, auto
from functools import partial

from lyra.kernels.common import Implementation
from lyra.kernels.cross_entropy import linear_cross_entropy
from lyra.kernels.decode_attention import decode_attention
from lyra.kernels.decode_gemm import decode_gemm
from lyra.kernels.flash_attention import flash_attention
from lyra.kernels.gemm import grouped_gemm


class Mode(Enum):
    NOCACHE = auto()
    PREFILL = auto()
    DECODE = auto()


def attn_op(mode: Mode, implementation: Implementation):
    op = flash_attention if mode in (Mode.NOCACHE, Mode.PREFILL) else decode_attention
    attn = partial(op, implementation=implementation)
    return attn


def gmm_op(mode: Mode, implementation: Implementation):
    op = grouped_gemm if mode in (Mode.NOCACHE, Mode.PREFILL) else decode_gemm
    gmm = partial(op, implementation=implementation)
    return gmm


def ce_op(implementation: Implementation):
    ce = partial(linear_cross_entropy, implementation=implementation)
    return ce
