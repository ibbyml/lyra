# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MoE combine, adapted from MaxText's Apache-2.0 SparseCore gather/reduce.

Source: AI-Hypercomputer/maxtext
src/maxtext/kernels/gather_reduce_pallas.py.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from lyra.kernels.common import current_tpu_target, log_kernel_resolution


def _gather_reduce(
    op: jax.Array,
    idx: jax.Array,
    topk_weights: jax.Array | None = None,
    *,
    reduce_group_size: int,
    single_sc: bool = False,
    col_chunk_size: int = int(3.5 * 1024),
    row_chunk_size: int = 512,
    topk_wgt_zero_nan: bool = False,
    name: str = "MoE_Combine",
) -> jax.Array:
    """Gather BF16 assignment rows and accumulate weighted groups in FP32."""
    if op.dtype != jnp.bfloat16:
        raise ValueError(f"op.dtype must be bf16, but got {op.dtype}")
    if op.shape[0] % reduce_group_size != 0:
        raise ValueError(f"{op.shape[0]=} must be divisible by {reduce_group_size=}")

    if op.shape[1] % col_chunk_size:
        raise ValueError("columns must be divisible by col_chunk_size")
    sc_info = pltpu.get_tpu_info().sparse_core
    if sc_info is None:
        raise RuntimeError("SparseCore is not available on this TPU version.")

    [M] = idx.shape
    _, K = op.shape
    M_out = M // reduce_group_size

    if topk_weights is not None:
        topk_weights = topk_weights.flatten()

    @jax.jit
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((M_out, K), op.dtype),
        name=name,
        mesh=plsc.VectorSubcoreMesh(
            core_axis_name="core",
            subcore_axis_name="subcore",
            num_cores=1 if single_sc else sc_info.num_cores,
        ),
        compiler_params=pltpu.CompilerParams(needs_layout_passes=True, use_tc_tiling_on_sc=True),
    )
    def kernel(in_hbm_ref, idx_hbm_ref, weights_hbm_ref, out_hbm_ref):
        row_wave_size = row_chunk_size * lax.axis_size(("core", "subcore"))
        if M % row_wave_size:
            raise NotImplementedError(
                f"{M=} must be divisible by {row_chunk_size=} *"
                f" num_cores={lax.axis_size('core')} *"
                f" num_vector_subcores={lax.axis_size('subcore')} = {row_wave_size}"
            )
        num_row_chunks = M // row_wave_size
        num_col_chunks = K // col_chunk_size
        packing = 32 // jax.dtypes.itemsize_bits(op.dtype)

        subcore_first_row_chunk = lax.axis_index(("core", "subcore")) * num_row_chunks

        in_spec = pl.BlockSpec((row_chunk_size,), lambda i: (subcore_first_row_chunk + i,))
        in_specs = (in_spec,) * (1 + (weights_hbm_ref is not None))

        @functools.partial(pltpu.emit_pipeline, grid=(num_row_chunks,), in_specs=in_specs)
        def idx_pipeline(idx_ref, weights_ref=None):
            row_chunk_idx = subcore_first_row_chunk + pl.program_id(0)

            row_subchunk_size = sc_info.num_lanes
            out_rows_per_step = row_subchunk_size // reduce_group_size
            assert reduce_group_size * out_rows_per_step == sc_info.num_lanes
            num_row_subchunks = row_chunk_size // row_subchunk_size
            if row_chunk_size % row_subchunk_size:
                raise ValueError(f"row_chunk_size needs to be a multiple of {row_subchunk_size}, but got {row_chunk_size}")

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, num_col_chunks),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), col_chunk_size),
                    lambda r, c: (
                        lax.div(
                            idx_ref[pl.ds(r * row_subchunk_size, row_subchunk_size)],
                            packing,
                        ),
                        c,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (out_rows_per_step // packing, col_chunk_size),
                    lambda r, c: (row_chunk_idx * num_row_subchunks + r, c),
                ),
            )
            def data_pipeline(gather_ref, out_ref):
                gather_ref = gather_ref.bitcast(op.dtype)
                out_ref = out_ref.bitcast(op.dtype)

                row_slice = pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
                subchunk_idxs = idx_ref[row_slice]
                weights = None if weights_ref is None else weights_ref[row_slice].astype(jnp.float32)

                unpack_col_chunk = 32  # 32 seems to works best when tuning.

                @plsc.parallel_loop(0, col_chunk_size, step=unpack_col_chunk)
                def _(col_base):
                    accs = []
                    for reduce_group in range(out_rows_per_step):
                        acc = jnp.zeros((unpack_col_chunk,), dtype=jnp.float32)
                        products = []
                        for row_in_group in range(reduce_group_size):
                            row = reduce_group * reduce_group_size + row_in_group
                            row_data = gather_ref[pl.ds(row * packing, packing), pl.ds(col_base, unpack_col_chunk)].astype(jnp.float32)
                            if packing == 1:
                                row_data = row_data[0]
                            else:
                                assert packing == 2
                                # For dtypes narrower than 32-bit, we end up gathering multiple
                                # rows (since we had to bitcast to int32 before the gather).
                                # This uses the remainder of the packing to choose the only row
                                # we actually care about.
                                row_data = jnp.where(
                                    lax.rem(subchunk_idxs[row], 2) == 0,
                                    row_data[0],
                                    row_data[1],
                                )
                            if weights is not None:
                                row_data *= weights[row]
                                if topk_wgt_zero_nan:
                                    row_data = jnp.where(weights[row] == 0.0, jnp.zeros_like(row_data), row_data)
                            products.append(row_data)
                        if reduce_group_size == 4:
                            # Match XLA's sublane reduction tree before BF16 rounding.
                            acc = (products[0] + products[2]) + (products[1] + products[3])
                        else:
                            for product in products:
                                acc += product
                        accs.append(acc)
                    out = jnp.stack(accs, axis=0).astype(op.dtype)
                    out_ref[:, pl.ds(col_base, unpack_col_chunk)] = out

            data_pipeline(in_hbm_ref.bitcast(jnp.int32), out_hbm_ref.bitcast(jnp.int32))

        idx_pipeline(idx_hbm_ref, *([weights_hbm_ref] if weights_hbm_ref is not None else []))

    return kernel(op, idx, topk_weights)  # pylint: disable=no-value-for-parameter


def combine_reference(experts, scores, permute_map, unpermute_map):
    del permute_map
    gathered = experts[unpermute_map].reshape(*scores.shape, experts.shape[-1])
    return jnp.einsum("fkc,fk->fc", gathered, scores).astype(experts.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _combine(experts, scores, permute_map, unpermute_map, row_chunk_size):
    return _gather_reduce(
        experts,
        unpermute_map,
        scores,
        reduce_group_size=scores.shape[1],
        col_chunk_size=experts.shape[1],
        row_chunk_size=row_chunk_size,
    )


def _combine_fwd(experts, scores, permute_map, unpermute_map, row_chunk_size):
    out = _combine(experts, scores, permute_map, unpermute_map, row_chunk_size)
    return out, (experts, scores, permute_map, unpermute_map)


def _combine_bwd(row_chunk_size, context, grad):
    del row_chunk_size
    experts, scores, permute_map, unpermute_map = context
    # The inverse-map rewrite passed isolated checks but failed full-update parity.
    _, backward = jax.vjp(lambda x, s: combine_reference(x, s, permute_map, unpermute_map), experts, scores)
    dexperts, dscores = backward(grad)
    return dexperts, dscores, None, None


_combine.defvjp(_combine_fwd, _combine_bwd, optimize_remat=True)


def combine(experts, scores, permute_map, unpermute_map, *, row_chunk_size=512):
    """Combine bijectively permuted assignment rows, preserving the score VJP."""
    if experts.ndim != 2 or scores.ndim != 2 or experts.shape[0] != scores.size:
        raise ValueError("experts must contain one row per routing score")
    if permute_map.shape != (scores.size,) or unpermute_map.shape != (scores.size,):
        raise ValueError("permutation maps must cover every assignment row")
    log_kernel_resolution(
        "moe_combine",
        shape=f"f{scores.shape[0]} k{scores.shape[1]} c{experts.shape[1]}",
        impl="pallas",
        tiling=f"SparseCore rows{row_chunk_size}",
    )
    return _combine(experts, scores, permute_map, unpermute_map, row_chunk_size)


def supports_combine(experts, scores):
    return (
        experts.shape == (65536, 2048)
        and scores.shape == (16384, 4)
        and experts.dtype == jnp.bfloat16
        and scores.dtype == jnp.float32
        and current_tpu_target() == "v6e"
    )
