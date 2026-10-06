from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

_A = lambda x: cast(Array, x)


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _embs_gather(emb: Array, tokens: Array, vocab: int) -> Array:
    return jnp.take(emb, tokens, axis=0)


def _embs_gather_fwd(emb: Array, tokens: Array, vocab: int):
    return jnp.take(emb, tokens, axis=0), (tokens, jnp.zeros((), emb.dtype))


def _embs_gather_bwd(vocab: int, res, g: Array):
    tokens, gd = res
    toks = tokens.reshape(-1)
    tgrads = g.reshape(toks.shape[0], -1).astype(jnp.float32)
    order = jnp.argsort(toks)
    emb_grad = jax.ops.segment_sum(tgrads[order], toks[order], num_segments=vocab, indices_are_sorted=True)
    out = (emb_grad.astype(gd.dtype), None)
    return out


_embs_gather.defvjp(_embs_gather_fwd, _embs_gather_bwd)


def embs_gather(emb: Array, tokens: Array, vocab: int):
    return _A(_embs_gather(emb, tokens, vocab))
