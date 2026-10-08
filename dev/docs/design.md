# Design

## Architecture

Lyra starts from the GPT-OSS decoder: 
- Grouped-Query Attention that alternates between full causal and 128-token sliding-window layers + learned attention sinks.
- RoPE + NTK-aware scaling (YaRN) with an initial context length of 4096, which is extended during post-training to 131072.
- Fused experts with a clamped SwiGLU which also uses a 1.0 bias on the linear half.

The Lyra variants change a few things:

- **Interleaved experts.** Small and Medium repeat three dense layers followed by one MoE layer. Large and Max use two dense layers, then MoE everywhere else. Fewer MoE layers means fewer expert weights to hold in memory.
- **Shared experts.** Each MoE layer has one shared expert that every token uses, plus routed experts chosen by the router. A token uses four experts in total: the shared one and its top three routed ones. Routing scores are normalized over the selected experts, and a load-balancing loss and a router z-loss are added to the language-modeling loss.
- **QKNorm.** Queries and keys are RMS-normalized before RoPE.
- **Attention output gate.** Each query head's attention output is scaled by `2 * sigmoid(h @ W)` before the output projection, where `h` is the normalized layer input. `W` starts at zero, so the gate starts as the identity, and it adds 524,288 parameters to Small.
- **Learned XSA (Medium and up).** After the gate, each head's output has its projection onto the token's own normalized value subtracted, scaled by a per-head `tanh(alpha)`. `alpha` starts at zero, so XSA also starts as the identity.
- **Tied embeddings (Small).** Small shares its input and output embeddings. The larger variants keep them separate.

Q, K, and V are stored as separate weights so that Muon orthogonalizes each projection on its own.

The `gpt-oss-20b` and `gpt-oss-120b` configurations reproduce the original models: biases, four routed experts per token, no shared experts, and no QKNorm.

### Variants

| Variant | Parameters | Width | Layers | MoE layers | Experts per MoE layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dev` | 3.33M | 16 | 16 | 4 | 7 + 1 shared |
| `lyra-small` | 1.62B | 2,048 | 16 | 4 | 15 + 1 shared |
| `lyra-medium` | 17.72B | 4,096 | 32 | 8 | 31 + 1 shared |
| `lyra-large` | 51.55B | 4,096 | 32 | 30 | 31 + 1 shared |
| `lyra-max` | 99.87B | 4,096 | 32 | 30 | 63 + 1 shared |
| `gpt-oss-20b` | 20.94B | 2,880 | 24 | 24 | 32 |
| `gpt-oss-120b` | 116.85B | 2,880 | 36 | 36 | 128 |

`dev` has a 256-token context and two active experts so it trains on CPU. All variants are defined in [variants.py](../../lyra/variants.py).

## Optimizer and precision

[optimizer.py](../../lyra/training/optimizer.py) splits the parameters between two optimizers:

- **Muon** (peak LR 1.5e-3) updates the weight matrices, treating each expert and each SwiGLU half as its own matrix. Nesterov momentum, 5 Newton–Schulz steps with Polar Express coefficients, per-matrix RMS clipping, and weight decay.
- **Adam** (peak LR 2e-4) updates everything else: embeddings, routers, norms, biases, sinks, the attention gate, and the XSA scales.

The schedule is ~0.21% warmup, a 50% plateau, then cosine decay to zero.

Weights are FP32 and compute is BF16. Muon momentum and Adam's first moment are BF16; Adam's second moment is FP32.

## Token budgets

Presets use [Chinchilla](https://arxiv.org/abs/2203.15556)'s 20 tokens per active parameter, counting non-expert weights plus four experts per MoE layer:

| Model | Active parameters | Tokens |
| --- | ---: | ---: |
| Small | 1.01B | 20.2B |
| Medium | 6.42B | 128.5B |
| Large | 9.24B | 185.1B |
| Max | 9.24B | 185.1B |

## Sharding

[Sharding rules](../../lyra/nn/params.py) set the device mesh and how each weight and activation is partitioned:

| Variants | Default mesh |
| --- | --- |
| `dev` | No Sharding (Single Device) |
| Small | DDP (Across devices) |
| Medium | EP (8 devices) |
| Large | EP (16 devices) |
| Max | EP (32 devices) |
| GPT-OSS | DDP + TP (4x2) |

Expert parallelism gives each device whole experts and splits the batch across all of them. Medium, Large, and Max are sized for about 95 GB of HBM per device (TPU v5p or v7x): the weights, gradients, and optimizer state take roughly 70 GB per device, leaving room for activations at one sequence per device.
