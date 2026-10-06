# Usage

## Setup

Clone the repository and install the extra for your hardware:

```bash
git clone https://github.com/ibbyml/lyra.git
cd lyra

uv sync --extra tpu   # TPU
uv sync --extra cpu   # CPU
uv sync --extra gpu   # NVIDIA GPU, CUDA 13
```

Every script accepts `--help`.

### Short CPU run

To check that everything works without a TPU, download one training shard and train the tiny `dev` model for 20 steps:

```bash
uv run data/dataset.py --max-tokens 300000000 --no-val
uv run scripts/train.py --device cpu --variant dev --steps 20
uv run scripts/serve.py --device cpu --variant dev
```

The `dev` model is far too small to produce sensible text, but it runs the same training, checkpointing, and generation code as the real models.

## Data

The [published dataset](https://huggingface.co/datasets/Ibbyml/lyra-climbmix-30b) is 30B tokens of shuffled ClimbMix, tokenized with `o200k_harmony` and split into 100 ArrayRecord shards of 300M tokens each. Lyra holds out the last shard for validation and never trains on it.

```bash
uv run data/dataset.py                            # 21B training tokens, plus the validation shard
uv run data/dataset.py --max-tokens 2000000000    # Enough whole shards to cover 2B tokens
uv run data/dataset.py --max-tokens None          # All 29.7B training tokens
```

Training shards go to `data/train` and the validation shard goes to `data/val`. Pass `--no-val` to skip the validation shard. Downloads are pinned to a fixed dataset revision, and running the script again only fetches what's missing.

To build your own shards from raw ClimbMix, use [prepare_climbmix.py](../../data/prepare_climbmix.py).

Training reads the data once by default, and it checks before starting that there are enough tokens for the full schedule. Downloading less data doesn't shorten a run, so pair a smaller download with a smaller `--steps`.

## Training

`scripts/train.py` takes a model (`--variant`) and a run preset (`--run`). The preset defaults to the one matching the variant, so `--variant lyra-small` uses `train-small`:

| Preset | Batch | Accumulation | Steps | Tokens per step | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train-dev` | 2 | 1 | 100 | 512 | 51K |
| `train-small` | 4 | 4 | 308,000 | 65,536 | 20.19B |
| `train-medium` | 4 | 16 | 490,000 | 262,144 | 128.45B |
| `train-large` | 4 | 32 | 353,000 | 524,288 | 185.07B |
| `train-max` | 4 | 32 | 353,000 | 524,288 | 185.07B |

Tokens per step are sequences per microbatch × microbatches per step × context length. For example, `train-small` uses 4 × 4 × 4,096. Pretraining budgets cover 20 tokens per active parameter, rounded up to the next 1,000 updates; the [design guide](design.md#data) gives the counts. Medium, Large, and Max need more training data than the published 29.7B training tokens.

```bash
# The full Small run
uv run scripts/train.py

# A shorter run into its own directory
uv run scripts/train.py --steps 1000 --out chkpt/experiment

# Print the resolved configuration without training
uv run scripts/train.py --variant lyra-medium --print-config
```

`--batch-size` sets sequences per microbatch and `--acc-steps` sets microbatches per step. Small through Max share the 500M run's recipe: warmup over `16/7630` of the schedule (~0.21%), a plateau for 50% of the total updates, then cosine decay to zero over the remaining updates. `train-dev` keeps its short schedule with 3% warmup and no plateau. `--warmup-fraction`, `--stable-fraction`, and `--decay-type linear` change the schedule. Other changes go in [variants.py](../../lyra/variants.py) and [presets.py](../../lyra/training/presets.py).

The separate `train-small-500m` preset has been retired. The saved [run notes](../runs/lyra-small-500m) include a command for reproducing its schedule with `train-small` overrides.

### Validation and samples

Small through Max evaluate on `data/val` at initialization, every 1% of the run, and at completion: every 3,080 updates for Small, 4,900 for Medium, and 3,530 for Large and Max. Each evaluation replays 16 fixed accumulated batches. `--eval-data`, `--eval-every`, and `--eval-batches` override these settings.

These presets also sample fixed prompts on the same schedule and record routing diagnostics every 32 updates. `--sample-every` and `--numerics-every` change those intervals:

```bash
uv run scripts/train.py --steps 1000 --out chkpt/experiment \
  --eval-data data/val --eval-every 100 --sample-every 100 \
  --sample-prompts "The surprising thing about the ocean is" "To make a cup of tea, first"
```

Without `--sample-prompts`, the five prompts from the README's results are used. Each sample is 128 tokens of top-k sampling with a fixed seed.

### Outputs

Every run writes these files next to its checkpoints:

- `metrics.jsonl`: training loss, learning rates, gradient norms, and throughput every five steps, plus every validation result.
- `samples.jsonl`: each prompt's completion at each sampling point.
- `numerics.jsonl`: expert routing statistics for every MoE layer, recorded every `--numerics-every` steps.
- `metrics.png`: loss, gradient norm, and routing balance, drawn at the end of the run.

To redraw the plot for any run directory:

```bash
uv run python -m lyra.training.plotting chkpt/lyra-small
```

### Checkpoints and resuming

Runs save a checkpoint every 1,000 steps and at the end. Small through Max keep the three most recent checkpoints; `train-dev` keeps two. `--chkpt-every` and `--chkpt-keep` change both.

Checkpoints include the optimizer state, the data loader's position, and the run configuration. To resume, pass the original settings along with `--resume`:

```bash
uv run scripts/train.py --resume chkpt/lyra-small
```

This continues from the latest checkpoint in `chkpt/lyra-small`. Lyra checks the model, schedule, batch settings, and data against the checkpoint before restoring anything, and refuses to resume if they differ. `--steps` is always the total length of the schedule, including steps that have already run. To resume into a different directory, add `--out`.

To save checkpoints to Google Cloud Storage, pass `--out gs://YOUR_BUCKET/run` on a machine with Application Default Credentials. Logs are written locally under `logs/` and copied to the bucket with each checkpoint.

## Generation

`scripts/serve.py` loads a checkpoint and opens an interactive prompt. With no arguments it loads `chkpt/lyra-small`:

```bash
uv run scripts/serve.py --weights chkpt/experiment \
  --temperature 0.7 --top-k 50 --max-new-tokens 512
```

`--max-cache-length` sets the total prompt and generation length. `--fp8` stores the MLP weights and KV cache in block-scaled FP8. That saves memory, but it isn't faster at every shape; the [kernel guide](kernels.md) has the measurements.

## Evaluation

MMLU, ARC, and HellaSwag download from Hugging Face on first use:

```bash
uv run scripts/evals.py --weights chkpt/experiment \
  --tasks mmlu arc-challenge hellaswag \
  --fewshot mmlu 5 arc-challenge 25 hellaswag 10
```

Evaluation is zero-shot unless you pass `--fewshot`. `--limit` evaluates a subset of each task, and `--batch-size` controls memory use.

## GPT-OSS weights

Lyra can load OpenAI's original GPT-OSS checkpoints for generation and evaluation:

```bash
uvx --from huggingface_hub hf download openai/gpt-oss-20b \
  --include 'original/*' --local-dir chkpt/gpt-oss-20b

uv run scripts/serve.py --variant gpt-oss-20b --weights chkpt/gpt-oss-20b --format gpt-oss
uv run scripts/evals.py --variant gpt-oss-20b --weights chkpt/gpt-oss-20b --format gpt-oss
```

`--format gpt-oss` also turns on Harmony chat formatting. Both GPT-OSS variants expect eight devices, four-way data parallel and two-way tensor parallel. For other setups, change their sharding in [variants.py](../../lyra/variants.py).

## Python API

Load a Lyra checkpoint with the same variant it was trained with:

```python
import jax

from lyra.model import ModelWeights
from lyra.variants import model_variant

config = model_variant("lyra-small")
with jax.set_mesh(config.sharding.get_mesh()):
    model = ModelWeights.load("chkpt/lyra-small", config)
```

Loading fails with a clear error if the checkpoint's parameters don't match the config. GPT-OSS checkpoints load with `lyra.gpt_oss.load_gpt_oss_model(path, config)` instead.

### Diagnostics

A `Probe` records statistics about activations, residuals, routing, and logits as the model runs. Create it outside the traced function, then start and collect it inside:

```python
from lyra.model import model_apply
from lyra.training.probe import Probe

probe = Probe()

@jax.jit
def observe(tokens, weights):
    probe.start()
    output, auxiliary_loss, _ = model_apply(tokens, weights, None, config, probe=probe)
    return output, probe.collect()
```

The model runs without a probe by default, so this adds no cost to normal training. `Probe(routing_only=True)` records just the routing statistics; this is what training uses for `numerics.jsonl`.

`TrainHooks` in [train.py](../../lyra/training/train.py) adds callbacks around compilation, each step, and each checkpoint, for tooling that needs more than the built-in logs.
