# Kernels

Lyra writes the operations that dominate training time and memory as Pallas kernels for TPU. Each one has a plain JAX reference that serves as a test oracle and as a fallback. The model's `implementation` setting picks between them: `pallas` always uses the kernel, `xla` always uses the reference, and `auto` (the default) uses the kernel when it has a tile configuration for the input shape and the reference otherwise. Each operation's `auto` exceptions are noted below.

## Attention

[flash_attention.py](../../lyra/kernels/flash_attention.py) computes global causal and sliding-window attention in tiles, without materializing the attention matrix. It supports grouped-query attention and attention sinks, and it has separate backward kernels for the query and key/value gradients.

- **Banded scheduling.** Sliding-window layers visit only the tiles that overlap the window, which skips most of the matrix work and memory traffic outside it, in both the forward and backward passes.
- **Grouped-query reuse.** The forward kernels process every query head in a group against the same key/value tiles, instead of reloading them for each head.

[decode_attention.py](../../lyra/kernels/decode_attention.py) attends from a single new token to the KV cache, reading only the window for sliding layers and supporting BF16 and FP8 caches. It hasn't beaten XLA at any shape measured so far, so `auto` uses the XLA reference for decoding.

## Linear cross-entropy

With a 200K-token vocabulary, the logits for one Small microbatch of 16,384 tokens would take over 6 GB even in BF16. [cross_entropy.py](../../lyra/kernels/cross_entropy.py) fuses the output projection with the loss: it streams over vocabulary tiles and keeps a running softmax normalizer, so the full logits are never stored. Separate backward kernels compute the input and weight gradients. At Small's training shape, the weight-gradient kernel adds directly into the gradient accumulator instead of producing a new gradient for each microbatch.

## Mixture of experts

[gemm.py](../../lyra/kernels/gemm.py) multiplies tokens grouped by expert, using each expert's actual token count. The up projection applies the clamped SwiGLU inside the matmul. The backward pass has its own kernels for the input and weight gradients, and it recomputes the SwiGLU inputs instead of storing them.

After the experts run, [combine.py](../../lyra/kernels/combine.py) gathers each token's expert outputs, applies the routing weights, and sums them back into token order on the TPU's SparseCore, without materializing the gathered tensor. It's adapted from [MaxText](https://github.com/AI-Hypercomputer/maxtext) under the Apache-2.0 license. `auto` uses it only for Small at batch size 4, the shape it was validated at, and its backward pass uses the JAX reference.

[ep_moe.py](../../lyra/kernels/ep_moe.py) handles expert parallelism, sending token groups between devices with ragged all-to-all communication. [decode_gemm.py](../../lyra/kernels/decode_gemm.py) handles the small token counts of generation, with BF16 or prequantized FP8 weights.

## Kernel speedups

Small's shapes (batch 4, context 4,096, 16 query and 4 key/value heads, 16 experts with top-4 routing) on one TPU v6e. Each time is the median of 21 calls after three warmups; "forward + backward" includes the forward work that autodiff needs.

| Operation | Pallas | XLA | Speedup |
| --- | ---: | ---: | ---: |
| Cross-entropy, forward | 5.07 ms | 7.33 ms | 1.45× |
| Cross-entropy, forward + backward | 21.42 ms | 24.82 ms | 1.16× |
| Global attention, forward | 2.27 ms | 6.11 ms | 2.69× |
| Global attention, forward + backward | 7.60 ms | 22.41 ms | 2.95× |
| Sliding-window attention, forward | 1.50 ms | 5.97 ms | 3.98× |
| Sliding-window attention, forward + backward | 4.25 ms | 22.40 ms | 5.27× |
| MoE up projection + SwiGLU | 1.51 ms | 7.26 ms | 4.80× |
| MoE down projection | 0.86 ms | 3.24 ms | 3.76× |
| Expert combine, forward | 1.06 ms | 3.02 ms | 2.86× |

The fused kernels also avoid large temporary buffers. Compiler-allocated temporary memory for one call:

| Operation | Pallas | XLA |
| --- | ---: | ---: |
| Cross-entropy, forward | 0.05 MiB | 3,142 MiB |
| Global attention, forward | 1.03 MiB | 2,081 MiB |
| Expert combine, forward | 0.34 MiB | 768 MiB |

These are per-op numbers. They show where the kernels help, but they don't add up to end-to-end training savings.

[RESULTS.md](../benchmarks/RESULTS.md) has every measurement, including Medium's shapes, batch size 8, decode, and the numerical checks.