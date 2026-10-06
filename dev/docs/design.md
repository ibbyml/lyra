# Design

## Architecture

Lyra starts from the GPT-OSS decoder: grouped-query attention that alternates between full causal and 128-token sliding-window layers, learned attention sinks, rotary position embeddings, and a clamped SwiGLU. The context length is 4,096 tokens.

![Attention, layer schedule, and mixture of experts](../assets/architecture.svg)

The Lyra variants change a few things:

- **Interleaved experts.** Small and Medium repeat three dense layers followed by one MoE layer. Large and Max use two dense layers, then MoE everywhere else. Fewer MoE layers means fewer expert weights to hold in memory.
- **Shared experts.** Each MoE layer has one shared expert that every token uses, plus routed experts chosen by the router. A token uses four experts in total: the shared one and its top three routed ones. Routing scores are normalized over the selected experts, and a load-balancing loss and a router z-loss are added to the language-modeling loss.
- **QK normalization.** Queries and keys are RMS-normalized before RoPE.
- **Attention output gate (Small).** Each query head's attention output is scaled by `2 * sigmoid(h @ W)` before the output projection, where `h` is the normalized layer input. `W` starts at zero, so the gate starts as the identity, and it adds 524,288 parameters to Small. Medium, Large, and Max use learned XSA in the same position instead.
- **Tied embeddings (Small).** Small shares its input and output embeddings. The larger variants keep them separate.

Q, K, and V are stored as separate weights so that Muon orthogonalizes each projection on its own.

The `gpt-oss-20b` and `gpt-oss-120b` configurations reproduce the original models: biases, four routed experts per token, no shared experts, and no QK normalization.

### Variants

| Variant | Parameters | Width | Layers | MoE layers | Experts per MoE layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dev` | 3.33M | 16 | 16 | 4 | 7 + 1 shared |
| `lyra-small` | 1.62B | 2,048 | 16 | 4 | 15 + 1 shared |
| `lyra-medium` | 11.28B | 4,096 | 32 | 8 | 15 + 1 shared |
| `lyra-large` | 51.54B | 4,096 | 32 | 30 | 31 + 1 shared |
| `lyra-max` | 99.87B | 4,096 | 32 | 30 | 63 + 1 shared |
| `gpt-oss-20b` | 20.94B | 2,880 | 24 | 24 | 32 |
| `gpt-oss-120b` | 116.85B | 2,880 | 36 | 36 | 128 |

Parameter counts include every stored weight, including vocabulary padding. Small uses about 1.01B of its parameters per token. The `dev` variant has a 256-token context and two active experts, so a CPU can train it. Every model is defined in [variants.py](../../lyra/variants.py), using the fields of [ModelConfig](../../lyra/model.py).

### Why a gate on Small

Before the 500M-token run, I compared three architecture changes on Small, each trained for 50M tokens on one TPU v6e. The attention output gate reached a held-out loss of 4.601, against 4.662 for normalizing each residual branch and 4.663 for scaling up the input embeddings, and it ran about 2% slower than the embedding change (35,459 vs 36,220 tokens/s). That comparison had no ungated control, and it says nothing yet about larger models, so only Small uses the gate.

## Optimizer and precision

[optimizer.py](../../lyra/training/optimizer.py) splits the parameters between two optimizers:

- **Muon** updates the weight matrices, treating each expert and each half of a SwiGLU projection as its own matrix. It uses Nesterov momentum, five Newton–Schulz iterations with the Polar Express coefficients, per-matrix gradient RMS clipping, and weight decay.
- **Adam** updates everything else: embeddings, routers, norm scales, biases, attention sinks, and the attention gate.

Peak learning rates are 1.5e-3 for Muon and 2e-4 for Adam. Small through Max scale the 500M run's recipe: warmup over `16/7630` of the updates (~0.21%), a plateau for 50% of all updates, then cosine decay to zero. Validation and sampling run every 1% of the schedule, with 16 fixed accumulated validation batches. Routing diagnostics run every 32 updates, and checkpoints retain the latest three saves.

Weights are stored in FP32 and the forward and backward passes run in BF16. Muon's momentum and Adam's first moment are BF16, and Adam's second moment is FP32. Gradient accumulation gives large token batches without increasing activation memory: Small uses 4 microbatches of four sequences (65,536 tokens per update), Medium uses 16 (262,144 tokens), and Large and Max use 32 (524,288 tokens).

## Data

Training uses a 30B-token slice of ClimbMix, shuffled, tokenized with `o200k_harmony`, and stored as ArrayRecord shards. The [loader](../../lyra/data.py) uses Grain to stream fixed-size token blocks, and its position is saved with every checkpoint, so a resumed run continues on exactly the next batch. The last shard is held out for validation.

Pretraining budgets apply the [Chinchilla](https://arxiv.org/abs/2203.15556) rule of thumb of 20 training tokens per parameter to each MoE model's active parameters. We count all non-expert weights plus four active experts per MoE layer, using the real vocabulary size and counting tied embeddings once. From the current [weight specifications](../../lyra/model.py):

| Model | Active parameters | Minimum tokens (20×) | Preset tokens |
| --- | ---: | ---: | ---: |
| Small | 1,008,138,556 | 20,162,771,120 | 20,185,088,000 |
| Medium | 6,412,810,360 | 128,256,207,200 | 128,450,560,000 |
| Large | 9,234,701,218 | 184,694,024,360 | 185,073,664,000 |
| Max | 9,238,634,338 | 184,772,686,760 | 185,073,664,000 |

Steps are `ceil(20 × active_parameters / tokens_per_step)`, rounded up to the next 1,000 updates. Large and Max have the same active expert count and differ here only in router size. Recalculate these budgets when model shapes change.

`train-small` fits within the 21B tokens fetched by default. The larger presets require more data than the published 29.7B training tokens; the loader checks data sufficiency before training. The separate 500M preset is retired, with its saved configuration and results retained in the [run notes](../runs/lyra-small-500m).

## Sharding

[Sharding rules](../../lyra/nn/params.py) set the device mesh and how each weight and activation is partitioned:

| Variants | Default mesh |
| --- | --- |
| `dev` | One device |
| Small, Medium | Data parallel across all devices |
| Large, Max | Eight-way expert parallel |
| GPT-OSS | Four-way data parallel × two-way tensor parallel |

The mesh has to match the number of available devices. To run on a different layout, change the variant's sharding in [variants.py](../../lyra/variants.py). So far, only Small on a single TPU v6e has been trained end to end; the other layouts have not been benchmarked.
