# Lyra Small, 500M tokens

An end-to-end correctness run of the full stack before full pretraining: data loading, training, validation, sampling, and checkpointing on one TPU v6e. Completed 2026-10-05. Its recipe now underlies the Small through Max pretraining presets.

![Loss, gradient norm, and routing balance](metrics.png)

## Setup

- **Model:** `lyra-small`, 1.61B parameters (1.01B active per token), with the attention output gate.
- **Run:** the former `train-small-500m` preset. 7,630 updates of 65,536 tokens (batch 4 × 4 accumulation steps × 4,096 context), for 500,039,680 tokens.
- **Schedule:** 16 warmup updates, 3,815 at peak (Muon 1.5e-3, Adam 2e-4), then cosine decay to zero over the last 3,799.
- **Data:** the first two shards (600M tokens) of [lyra-climbmix-30b](https://huggingface.co/datasets/Ibbyml/lyra-climbmix-30b) at revision `346a345f`. Validation uses 16 fixed batches (262K tokens) from shard 99, which is never trained on.
- **Software:** JAX 0.11.2, libtpu 0.0.48.

## Results

| Training tokens | Validation loss |
| ---: | ---: |
| 0 | 12.62 |
| 65M | 3.84 |
| 130M | 3.57 |
| 195M | 3.45 |
| 260M | 3.37 |
| 330M | 3.27 |
| 395M | 3.15 |
| 460M | 3.07 |
| 500M | **3.057** |

Final validation perplexity is 21.3.

| Performance | |
| --- | ---: |
| Update time (median of 239) | 1.919 s |
| Throughput | 34,150 tokens/s |
| Model FLOPs utilization | 24.1% |
| Compiled peak memory | 28.8 GiB (90% of 32 GiB) |
| Compile time | 59 s |
| Wall clock, end to end | 4h 40m |

MFU uses the analytical method from the [kernel guide](../../docs/kernels.md#how-mfu-is-computed), with 6,475,926,528 FLOPs per token (including the gate) against 918 TFLOP/s. Wall clock includes compilation, 101 validation passes, 101 sampling passes, and checkpoints.

## Observations

**One loss spike.** At updates 4,805–4,807 (~315M tokens), training loss jumped from 3.3 to 5.8 and then returned to trend within five updates. Most of the gradient norm came from the Adam-managed parameters (embeddings, routers, norms, and gates): Adam's gradient norm reached 4.3 against Muon's 1.2, and Muon's clipping never engaged. Validation loss moved from 3.307 to 3.319 at the next evaluation and was back on trend by the one after.

**Routing stayed balanced.** All 60 routed experts were in use at the end of the run. Across the four MoE layers, the busiest expert carried 1.18–1.37× a uniform share of tokens and the least-used carried 0.69–0.85×.

**The residual stream grows.** The RMS of the final residual stream grew from 0.67 to 22.9 over the run, the largest per-channel mean reached 68, and an outlier channel appeared in layer 1's MLP activations at 16× the mean RMS. The loss curve shows no effect at this scale. These diagnostics are worth watching in longer runs.

## Samples

From the final checkpoint, using the fixed prompts the run samples every 5M tokens (top-k 4, temperature 0.8, 128 tokens). At 500M tokens the model is fluent and on topic, but repetitive and loose with facts.

> **To make a cup of tea, first**, you need to make a cup of tea by boiling water. You can do this by pouring boiling water into a cup of hot water and then pouring the boiling water into the hot water. The water will be hot enough to boil the tea, and then you can drink it.
> The next step is to add sugar to the hot water. […]

> **The history of astronomy begins with** Galileo Galilei. His work was published in 1619, and he was the first to use the word "astronomy" to refer to the study of celestial bodies in their natural environment. He also made significant contributions to the field of astrophysics, including the study of the formation of stars and the evolution of the solar system. […]

> **Here is a simple explanation of gravity:** When you are in a gravitational field, gravity pulls you toward your center of mass. The gravitational field will keep you from getting to your center of mass. Gravity is a force that pulls you toward your center of mass. Gravity also pulls you toward your center of mass. […]

All prompts, at every sampling point, are in `samples.jsonl`.

## Reproducing

```bash
uv run data/dataset.py --max-tokens 600000000
uv run scripts/train.py --variant lyra-small --steps 7630 \
  --eval-every 76 --eval-batches 4 --sample-every 76 --out chkpt/lyra-small-500m
```

This downloads the two training shards and the validation shard, then overrides the standard Small preset to reproduce the run's batch settings and learning-rate schedule. Evaluation and sampling land within one update of the original 5M-token marks. The current validation loop groups microbatches by accumulation, so four accumulated evaluation batches preserve the original 16 microbatches (262,144 tokens).

## Files

- `config.json`: the full model and training configuration.
- `metrics.jsonl`: training metrics every five updates, and validation about every 76.
- `samples.jsonl`: five fixed prompts every 5M tokens, from initialization to the final update.
- `numerics.jsonl`: expert routing statistics every 32 updates.
- `metrics.png`: drawn from these files by `python -m lyra.training.plotting dev/runs/lyra-small-500m`.

The residual-stream statistics above come from fuller per-layer diagnostics (23 MB) that aren't included here.
