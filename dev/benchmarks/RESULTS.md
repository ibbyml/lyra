# Benchmark results

[Training step](#training-step) · [Step profile](#step-profile) · [Numerical checks](#numerical-checks) · [Isolated kernels](#isolated-kernels)

## Training step

Lyra Small with 4 sequences per microbatch, 32 microbatches per step, and a 4096-token context: 524288 tokens per step, with FP32 weights and BF16 compute.

| Measurement | Result |
|---|---:|
| Mean step time | 13.782 s |
| Median step time | 13.764 s |
| Throughput | 38,041 tokens/s |
| Compiled peak memory | 28.814 GiB |
| Compiler temporaries | 26,238,912,736 bytes |
| Generated code | 164,669,440 bytes |

### Step profile

| Kernel | Calls/update | Mean ms/call | Self ms/update |
|---|---:|---:|---:|
| CE forward | 32 | 19.003 | 608.106 |
| CE dW + accumulation | 32 | 34.388 | 1100.407 |
| CE dX | 32 | 31.597 | 1011.099 |
| Global attention forward | 256 | 2.041 | 522.430 |
| Global attention dQ | 256 | 2.251 | 576.210 |
| Global attention dKV | 256 | 2.730 | 698.980 |
| Sliding attention forward | 256 | 1.281 | 327.996 |
| Sliding attention dQ | 256 | 1.179 | 301.890 |
| Sliding attention dKV | 256 | 1.222 | 312.768 |
| MoE up original forward | 128 | 1.559 | 199.518 |
| MoE up remat forward | 128 | 1.562 | 199.951 |
| MoE up dLHS | 128 | 1.622 | 207.679 |
| MoE up dW | 128 | 1.748 | 223.744 |
| MoE down original forward | 128 | 0.778 | 99.537 |
| MoE down remat forward | 128 | 0.777 | 99.501 |
| MoE down dLHS | 128 | 0.827 | 105.874 |
| MoE down dW | 128 | 0.915 | 117.102 |
| SparseCore combine forward | 128 | 0.803 | 102.835 |

### lyra-small-b4

B=4, T=4096, C=2048, 16Q/4KV, H=128; 16 groups, top-4.

| Case | Kernel ms | XLA ms | Speedup | Kernel temp MiB | XLA temp MiB |
|---|---:|---:|---:|---:|---:|
| linear_cross_entropy fwd T=4096 | 5.068 | 7.330 | 1.45× | 0.047 | 3142.250 |
| linear_cross_entropy fwd T=16384 | 19.427 | — | — | 0.094 | — |
| linear_cross_entropy grad call T=4096 | 21.419 | 24.815 | 1.16× | 0.062 | 4713.125 |
| linear_cross_entropy grad call T=16384 | 84.838 | — | — | 8.031 | — |
| flash_attention fwd global B=4 | 2.269 | 6.112 | 2.69× | 1.031 | 2080.779 |
| flash_attention grad call global B=4 | 7.596 | 22.412 | 2.95× | 128.125 | 8353.373 |
| flash_attention fwd sliding-128 B=4 | 1.501 | 5.970 | 3.98× | 1.031 | 2080.779 |
| flash_attention grad call sliding-128 B=4 | 4.249 | 22.395 | 5.27× | 128.125 | 8353.373 |
| grouped gemm up (fused swiglu) [bf16] | 1.513 | 7.258 | 4.80× | 0.000 | 2049.093 |
| grouped gemm down [bf16] | 0.861 | 3.240 | 3.76× | 0.000 | 1024.968 |
| grouped gemm up (fused swiglu) [fp8] | 18.473 | 18.187 | 0.98× | 1184.250 | 3715.406 |
| grouped gemm down [fp8] | 14.538 | 10.891 | 0.75× | 1056.656 | 1344.468 |
| grouped tgemm up weight-grad [bf16] | 1.944 | 2.126 | 1.09× | 0.000 | 0.000 |
| grouped tgemm down weight-grad [bf16] | 1.072 | 1.160 | 1.08× | 0.000 | 0.000 |
| grouped gemm dLHS up [bf16] | 1.523 | 3.154 | 2.07× | 0.000 | 257.031 |
| grouped gemm dLHS down [bf16] | 0.833 | 1.475 | 1.77× | 0.000 | 128.000 |
| decode gemm up [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.299 | 0.559 | 1.87× | 0.062 | 0.062 |
| decode gemm up [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.396 | — | — | 0.062 | — |
| decode gemm up [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.559 | — | — | 0.062 | — |
| decode gemm up [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.549 | — | — | 0.062 | — |
| decode gemm up [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.549 | — | — | 0.062 | — |
| decode gemm up [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.566 | — | — | 0.062 | — |
| decode gemm up [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.565 | — | — | 0.062 | — |
| decode gemm up [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.556 | — | — | 0.062 | — |
| decode gemm down [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.234 | 0.372 | 1.59× | 0.062 | 0.062 |
| decode gemm down [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.283 | — | — | 0.062 | — |
| decode gemm down [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.376 | — | — | 0.062 | — |
| decode gemm down [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.372 | — | — | 0.062 | — |
| decode gemm down [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.376 | — | — | 0.062 | — |
| decode gemm down [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.391 | — | — | 0.062 | — |
| decode gemm down [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.367 | — | — | 0.062 | — |
| decode gemm down [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.366 | — | — | 0.062 | — |
| decode_attention global B=4 | 0.475 | 0.147 | 0.31× | 0.000 | 0.000 |
| decode_attention sliding-128 B=4 | 0.150 | 0.150 | 1.00× | 0.000 | 0.000 |
| grouped gemm up (fused swiglu) [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 13.036 | 11.082 | 0.85× | 32.031 | 3075.156 |
| grouped gemm down [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 10.443 | 5.934 | 0.57× | 32.031 | 1729.781 |
| decode gemm up [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.253 | 2.147 | 8.49× | 0.124 | 1024.062 |
| decode gemm up [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.296 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.380 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.382 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.391 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.416 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.479 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.578 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.220 | 1.188 | 5.40× | 0.062 | 512.094 |
| decode gemm down [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.228 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.266 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.270 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.291 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.293 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.319 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.384 | — | — | 0.124 | — |
| decode_attention global FP8 cache B=4 | 0.516 | 0.223 | 0.43× | 0.000 | 0.000 |
| decode_attention sliding-128 FP8 cache B=4 | 0.215 | 0.194 | 0.90× | 0.000 | 0.000 |
| MoE combine forward [bf16/fp32] | 1.058 | 3.021 | 2.86× | 0.344 | 768.031 |
| MoE combine VJP [bf16/fp32] | 6.232 | 6.196 | 0.99× | 1144.562 | 1144.562 |

### lyra-small-b8

B=8, T=4096, C=2048, 16Q/4KV, H=128; 16 groups, top-4.

| Case | Kernel ms | XLA ms | Speedup | Kernel temp MiB | XLA temp MiB |
|---|---:|---:|---:|---:|---:|
| linear_cross_entropy fwd T=4096 | 5.072 | 7.321 | 1.44× | 0.047 | 3142.250 |
| linear_cross_entropy fwd T=32768 | 38.590 | — | — | 16.156 | — |
| linear_cross_entropy grad call T=4096 | 21.467 | 24.827 | 1.16× | 0.062 | 4713.125 |
| linear_cross_entropy grad call T=32768 | 180.969 | — | — | 32.031 | — |
| flash_attention fwd global B=4 | 2.263 | 6.003 | 2.65× | 1.031 | 2080.779 |
| flash_attention grad call global B=4 | 7.602 | 22.425 | 2.95× | 128.125 | 8353.373 |
| flash_attention fwd sliding-128 B=4 | 1.494 | 5.878 | 3.93× | 1.031 | 2080.779 |
| flash_attention grad call sliding-128 B=4 | 4.219 | 22.472 | 5.33× | 128.125 | 8353.373 |
| flash_attention fwd global B=8 | 5.022 | — | — | 2.031 | — |
| flash_attention grad call global B=8 | 15.809 | — | — | 256.125 | — |
| flash_attention fwd sliding-128 B=8 | 3.357 | — | — | 2.031 | — |
| flash_attention grad call sliding-128 B=8 | 9.246 | — | — | 256.125 | — |
| grouped gemm up (fused swiglu) [bf16] | 2.768 | 14.286 | 5.16× | 0.000 | 4098.093 |
| grouped gemm down [bf16] | 1.465 | 6.082 | 4.15× | 0.000 | 2049.968 |
| grouped gemm up (fused swiglu) [fp8] | 34.013 | 31.255 | 0.92× | 2176.187 | 6598.250 |
| grouped gemm down [fp8] | 27.510 | 19.506 | 0.71× | 2176.187 | 3648.468 |
| grouped tgemm up weight-grad [bf16] | 3.378 | 3.519 | 1.04× | 0.000 | 0.000 |
| grouped tgemm down weight-grad [bf16] | 1.815 | 1.867 | 1.03× | 0.000 | 0.000 |
| grouped gemm dLHS up [bf16] | 2.847 | 5.388 | 1.89× | 0.000 | 258.031 |
| grouped gemm dLHS down [bf16] | 1.454 | 2.333 | 1.60× | 0.000 | 128.000 |
| decode gemm up [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.277 | 0.538 | 1.94× | 0.062 | 0.062 |
| decode gemm up [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.368 | — | — | 0.062 | — |
| decode gemm up [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.529 | — | — | 0.062 | — |
| decode gemm up [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.537 | — | — | 0.062 | — |
| decode gemm up [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.541 | — | — | 0.062 | — |
| decode gemm up [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.556 | — | — | 0.062 | — |
| decode gemm up [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.551 | — | — | 0.062 | — |
| decode gemm up [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.536 | — | — | 0.062 | — |
| decode gemm down [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.194 | 0.363 | 1.87× | 0.062 | 0.062 |
| decode gemm down [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.249 | — | — | 0.062 | — |
| decode gemm down [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.375 | — | — | 0.062 | — |
| decode gemm down [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.369 | — | — | 0.062 | — |
| decode gemm down [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.378 | — | — | 0.062 | — |
| decode gemm down [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.374 | — | — | 0.062 | — |
| decode gemm down [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.369 | — | — | 0.062 | — |
| decode gemm down [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.372 | — | — | 0.062 | — |
| decode_attention global B=4 | 0.477 | 0.158 | 0.33× | 0.000 | 0.000 |
| decode_attention sliding-128 B=4 | 0.160 | 0.148 | 0.93× | 0.000 | 0.000 |
| decode_attention global B=8 | 0.769 | — | — | 0.000 | — |
| decode_attention sliding-128 B=8 | 0.187 | — | — | 0.000 | — |
| grouped gemm up (fused swiglu) [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 25.804 | 21.021 | 0.81× | 64.031 | 5766.156 |
| grouped gemm down [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 20.710 | 10.637 | 0.51× | 64.031 | 3331.531 |
| decode gemm up [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.246 | 2.154 | 8.77× | 0.124 | 1024.062 |
| decode gemm up [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.303 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.381 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.379 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.374 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.402 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.462 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.558 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.186 | 1.187 | 6.40× | 0.062 | 512.094 |
| decode gemm down [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.213 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.254 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.243 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.270 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.280 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.315 | — | — | 0.062 | — |
| decode gemm down [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.389 | — | — | 0.124 | — |
| decode_attention global FP8 cache B=4 | 0.517 | 0.208 | 0.40× | 0.000 | 0.000 |
| decode_attention sliding-128 FP8 cache B=4 | 0.191 | 0.182 | 0.95× | 0.000 | 0.000 |
| decode_attention global FP8 cache B=8 | 0.823 | — | — | 0.000 | — |
| decode_attention sliding-128 FP8 cache B=8 | 0.218 | — | — | 0.000 | — |

### lyra-medium-b4

B=4, T=4096, C=4096, 32Q/8KV, H=128; 16 groups, top-4.

| Case | Kernel ms | XLA ms | Speedup | Kernel temp MiB | XLA temp MiB |
|---|---:|---:|---:|---:|---:|
| linear_cross_entropy fwd T=4096 | 9.726 | 12.275 | 1.26× | 0.047 | 3142.250 |
| linear_cross_entropy fwd T=16384 | 36.204 | — | — | 0.094 | — |
| linear_cross_entropy grad call T=4096 | 42.569 | 43.743 | 1.03× | 0.062 | 6284.125 |
| linear_cross_entropy grad call T=16384 | 177.432 | — | — | 8.031 | — |
| flash_attention fwd global B=4 | 5.008 | 11.682 | 2.33× | 2.031 | 4097.215 |
| flash_attention grad call global B=4 | 15.774 | 41.632 | 2.64× | 256.125 | 16642.308 |
| flash_attention fwd sliding-128 B=4 | 3.345 | 11.672 | 3.49× | 2.031 | 4097.215 |
| flash_attention grad call sliding-128 B=4 | 9.234 | 41.744 | 4.52× | 256.125 | 16642.308 |
| grouped gemm up (fused swiglu) [bf16] | 6.220 | 24.023 | 3.86× | 0.000 | 4097.343 |
| grouped gemm down [bf16] | 4.326 | 8.713 | 2.01× | 0.000 | 2049.093 |
| grouped gemm up (fused swiglu) [fp8] | 68.933 | 53.491 | 0.78× | 4096.312 | 6944.718 |
| grouped gemm down [fp8] | 51.851 | 30.810 | 0.59× | 2336.312 | 3392.718 |
| grouped tgemm up weight-grad [bf16] | 7.655 | 8.114 | 1.06× | 0.000 | 0.000 |
| grouped tgemm down weight-grad [bf16] | 3.874 | 4.038 | 1.04× | 0.000 | 0.000 |
| grouped gemm dLHS up [bf16] | 6.639 | 19.835 | 2.99× | 0.000 | 1025.031 |
| grouped gemm dLHS down [bf16] | 3.205 | 6.336 | 1.98× | 0.000 | 513.031 |
| decode gemm up [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.566 | 1.626 | 2.87× | 0.062 | 0.062 |
| decode gemm up [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.933 | — | — | 0.062 | — |
| decode gemm up [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 1.631 | — | — | 0.062 | — |
| decode gemm up [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 1.624 | — | — | 0.062 | — |
| decode gemm up [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 1.636 | — | — | 0.062 | — |
| decode gemm up [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 1.716 | — | — | 0.062 | — |
| decode gemm up [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 1.634 | — | — | 0.062 | — |
| decode gemm up [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 1.633 | — | — | 0.062 | — |
| decode gemm down [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.373 | 0.886 | 2.37× | 0.062 | 0.062 |
| decode gemm down [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.554 | — | — | 0.062 | — |
| decode gemm down [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.904 | — | — | 0.062 | — |
| decode gemm down [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.902 | — | — | 0.062 | — |
| decode gemm down [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.899 | — | — | 0.062 | — |
| decode gemm down [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.912 | — | — | 0.062 | — |
| decode gemm down [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.903 | — | — | 0.062 | — |
| decode gemm down [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.919 | — | — | 0.062 | — |
| decode_attention global B=4 | 0.767 | 0.172 | 0.22× | 0.000 | 0.000 |
| decode_attention sliding-128 B=4 | 0.178 | 0.146 | 0.82× | 0.000 | 0.000 |
| grouped gemm up (fused swiglu) [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 50.664 | 33.509 | 0.66× | 32.031 | 7168.375 |
| grouped gemm down [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 40.116 | 17.551 | 0.44× | 32.031 | 2560.375 |
| decode gemm up [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.448 | 9.652 | 21.53× | 0.124 | 4096.062 |
| decode gemm up [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.596 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.866 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.869 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.898 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.970 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 1.143 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 1.411 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.289 | 4.433 | 15.36× | 0.124 | 2048.062 |
| decode gemm down [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.380 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.511 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.513 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.523 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.562 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.658 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.809 | — | — | 0.124 | — |
| decode_attention global FP8 cache B=4 | 0.809 | 0.223 | 0.28× | 0.000 | 0.000 |
| decode_attention sliding-128 FP8 cache B=4 | 0.195 | 0.162 | 0.83× | 0.000 | 0.000 |

### lyra-medium-b8

B=8, T=4096, C=4096, 32Q/8KV, H=128; 16 groups, top-4.

| Case | Kernel ms | XLA ms | Speedup | Kernel temp MiB | XLA temp MiB |
|---|---:|---:|---:|---:|---:|
| linear_cross_entropy fwd T=4096 | 9.712 | 12.265 | 1.26× | 0.047 | 3142.250 |
| linear_cross_entropy fwd T=32768 | 71.629 | — | — | 16.156 | — |
| linear_cross_entropy grad call T=4096 | 42.922 | 43.853 | 1.02× | 0.062 | 6284.125 |
| linear_cross_entropy grad call T=32768 | 365.093 | — | — | 32.031 | — |
| flash_attention fwd global B=4 | 5.016 | 11.722 | 2.34× | 2.031 | 4097.215 |
| flash_attention grad call global B=4 | 15.774 | 41.740 | 2.65× | 256.125 | 16642.308 |
| flash_attention fwd sliding-128 B=4 | 3.343 | 11.685 | 3.50× | 2.031 | 4097.215 |
| flash_attention grad call sliding-128 B=4 | 9.238 | 41.796 | 4.52× | 256.125 | 16642.308 |
| flash_attention fwd global B=8 | 9.796 | — | — | 4.031 | — |
| flash_attention grad call global B=8 | 31.168 | — | — | 516.156 | — |
| flash_attention fwd sliding-128 B=8 | 6.463 | — | — | 4.031 | — |
| flash_attention grad call sliding-128 B=8 | 18.170 | — | — | 516.156 | — |
| grouped gemm up (fused swiglu) [bf16] | 12.591 | 47.758 | 3.79× | 0.000 | 8194.343 |
| grouped gemm down [bf16] | 8.539 | 17.671 | 2.07× | 0.000 | 4098.093 |
| grouped gemm up (fused swiglu) [fp8] | 126.370 | 91.349 | 0.72× | 4736.062 | 14598.531 |
| grouped gemm down [fp8] | 98.357 | 50.732 | 0.52× | 4224.187 | 5824.468 |
| grouped tgemm up weight-grad [bf16] | 13.867 | 13.802 | 1.00× | 0.000 | 0.000 |
| grouped tgemm down weight-grad [bf16] | 6.972 | 6.850 | 0.98× | 0.000 | 0.000 |
| grouped gemm dLHS up [bf16] | 11.716 | 35.801 | 3.06× | 0.000 | 1026.031 |
| grouped gemm dLHS down [bf16] | 5.700 | 10.765 | 1.89× | 0.000 | 514.031 |
| decode gemm up [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.586 | 1.625 | 2.77× | 0.062 | 0.062 |
| decode gemm up [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.957 | — | — | 0.062 | — |
| decode gemm up [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 1.730 | — | — | 0.062 | — |
| decode gemm up [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 1.689 | — | — | 0.062 | — |
| decode gemm up [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 1.638 | — | — | 0.062 | — |
| decode gemm up [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 1.724 | — | — | 0.062 | — |
| decode gemm up [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 1.621 | — | — | 0.062 | — |
| decode gemm up [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 1.682 | — | — | 0.062 | — |
| decode gemm down [bf16] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.392 | 0.899 | 2.30× | 0.062 | 0.062 |
| decode gemm down [bf16] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.566 | — | — | 0.062 | — |
| decode gemm down [bf16] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.906 | — | — | 0.062 | — |
| decode gemm down [bf16] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.913 | — | — | 0.062 | — |
| decode gemm down [bf16] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.920 | — | — | 0.062 | — |
| decode gemm down [bf16] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.926 | — | — | 0.062 | — |
| decode gemm down [bf16] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.912 | — | — | 0.062 | — |
| decode gemm down [bf16] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.906 | — | — | 0.062 | — |
| decode_attention global B=4 | 0.764 | 0.163 | 0.21× | 0.000 | 0.000 |
| decode_attention sliding-128 B=4 | 0.188 | 0.157 | 0.83× | 0.000 | 0.000 |
| decode_attention global B=8 | 1.346 | — | — | 0.000 | — |
| decode_attention sliding-128 B=8 | 0.249 | — | — | 0.000 | — |
| grouped gemm up (fused swiglu) [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 100.723 | 64.786 | 0.64× | 64.031 | 12294.406 |
| grouped gemm down [fp8 prequantized] tile=256x256x256 scale=1x256/256x256 rhs-buffers=2 | 79.828 | 30.348 | 0.38× | 64.031 | 6787.656 |
| decode gemm up [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.425 | 9.617 | 22.62× | 0.124 | 4096.062 |
| decode gemm up [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.576 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.831 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.835 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.868 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.950 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 1.121 | — | — | 0.124 | — |
| decode gemm up [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 1.395 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=4 tg=2 tn=1024 bc=4 rhs-buffers=2 | 0.303 | 4.755 | 15.69× | 0.124 | 2048.062 |
| decode gemm down [fp8 prequantized] M=8 tg=2 tn=1024 bc=8 rhs-buffers=2 | 0.385 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=16 tg=2 tn=1024 bc=16 rhs-buffers=2 | 0.520 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=32 tg=2 tn=1024 bc=32 rhs-buffers=2 | 0.523 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=64 tg=2 tn=1024 bc=64 rhs-buffers=2 | 0.543 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=128 tg=2 tn=1024 bc=128 rhs-buffers=2 | 0.576 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=256 tg=2 tn=1024 bc=256 rhs-buffers=2 | 0.666 | — | — | 0.124 | — |
| decode gemm down [fp8 prequantized] M=512 tg=2 tn=1024 bc=512 rhs-buffers=2 | 0.812 | — | — | 0.124 | — |
| decode_attention global FP8 cache B=4 | 0.803 | 0.224 | 0.28× | 0.000 | 0.000 |
| decode_attention sliding-128 FP8 cache B=4 | 0.189 | 0.164 | 0.87× | 0.000 | 0.000 |
| decode_attention global FP8 cache B=8 | 1.419 | — | — | 0.000 | — |
| decode_attention sliding-128 FP8 cache B=8 | 0.245 | — | — | 0.000 | — |
