# Kernels

Each Pallas kernel has a plain JAX reference used as a test oracle and fallback. The model's `implementation` setting chooses between them:

- `pallas`: always use the kernel.
- `xla`: always use the reference.
- `auto` (default): use the kernel when it has a tile config for the shape, except where noted below.

## Attention

- [flash_attention.py](../../lyra/kernels/flash_attention.py): global and sliding-window attention with GQA and sinks. Sliding-window layers only visit tiles inside the window (forward and backward), and each GQA group shares its K/V tiles.
- [decode_attention.py](../../lyra/kernels/decode_attention.py): single-token attention over a BF16 or FP8 KV cache. It hasn't beaten XLA yet, so `auto` uses XLA.

## Cross-entropy

[cross_entropy.py](../../lyra/kernels/cross_entropy.py) fuses the output projection with the loss, streaming over vocabulary tiles so the full logits (6+ GB for one Small microbatch) are never stored.

## Mixture of experts

- [gemm.py](../../lyra/kernels/gemm.py): grouped expert matmuls with SwiGLU fused into the up projection. The backward pass recomputes the SwiGLU inputs.
- [combine.py](../../lyra/kernels/combine.py): SparseCore gather + weighted sum of expert outputs, adapted from [MaxText](https://github.com/AI-Hypercomputer/maxtext) (Apache-2.0). `auto` only uses it for Small at batch 4.
- [ep_moe.py](../../lyra/kernels/ep_moe.py): expert parallelism via ragged all-to-all.
- [decode_gemm.py](../../lyra/kernels/decode_gemm.py): expert matmuls for decoding, with BF16 or FP8 weights.

## Benchmarks

[RESULTS.md](../benchmarks/RESULTS.md) has every measurement: Small and Medium shapes, batch 4 and 8, decode, memory, and numerical checks. The Medium measurements predate its move to 31 + 1 experts and expert parallelism. Medium, Large, and Max run their experts through the expert-parallel path, whose shapes don't have tuned tiles yet.
