# Usage

Every script accepts `--help`, which lists all of its flags.

## Setup

```bash
uv sync --extra tpu   # TPU
uv sync --extra cpu   # CPU
uv sync --extra gpu   # NVIDIA GPU, CUDA 13
```

### Short CPU run

Download one shard and train the tiny `dev` model for 20 steps:

```bash
uv run data/dataset.py --max-tokens 300000000 --no-val
uv run scripts/train.py --device cpu --variant dev --steps 20
uv run scripts/serve.py --device cpu --variant dev
```

## Data

```bash
uv run data/dataset.py                            # 21B training tokens + validation shard
uv run data/dataset.py --max-tokens 2000000000    # Enough shards to cover 2B tokens
uv run data/dataset.py --max-tokens None          # All 29.7B training tokens
```

Shards go to `data/train` and `data/val`. To build your own from raw ClimbMix, use [prepare_climbmix.py](../../data/prepare_climbmix.py).

Training checks that there's enough data for the full schedule, so pair a smaller download with a smaller `--steps`.

## Training

`--variant` picks the model and `--run` the preset, which defaults to the variant's (`lyra-small` → `train-small`):

| Preset | Batch | Accumulation | Steps | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| `train-dev` | 2 | 1 | 100 | 51K |
| `train-small` | 4 | 4 | 308,000 | 20.2B |
| `train-medium` | 8 | 8 | 490,000 | 128.5B |
| `train-large` | 16 | 8 | 353,000 | 185.1B |
| `train-max` | 32 | 4 | 353,000 | 185.1B |

Medium, Large, and Max need more than the published 29.7B tokens.

The batch is split across the data-parallel devices, so it has to be a multiple of their count. Medium, Large, and Max need 8, 16, and 32 devices with about 95 GB of HBM each (TPU v5p or v7x), and their presets put one sequence on each device.

```bash
uv run scripts/train.py                                          # Full Small run
uv run scripts/train.py --steps 1000 --out chkpt/experiment      # Shorter run
uv run scripts/train.py --variant lyra-medium --print-config     # Show config and exit

# Custom validation and sampling
uv run scripts/train.py --steps 1000 --out chkpt/experiment \
  --eval-data data/val --eval-every 100 --sample-every 100 \
  --sample-prompts "The surprising thing about the ocean is" "To make a cup of tea, first"
```

To change anything beyond the flags, edit [variants.py](../../lyra/variants.py) and [presets.py](../../lyra/training/presets.py).

### Outputs

Each run writes these next to its checkpoints:

- `metrics.jsonl`: loss, learning rates, gradient norms, throughput, and validation results.
- `samples.jsonl`: completions at each sampling point.
- `numerics.jsonl`: expert routing statistics per MoE layer.
- `metrics.png`: plots drawn at the end of the run. To redraw one, run `uv run python -m lyra.training.plotting chkpt/lyra-small`.

### Resuming

```bash
uv run scripts/train.py --resume chkpt/lyra-small
```

Pass the original settings along with `--resume`; Lyra refuses to resume if they don't match the checkpoint. `--steps` is the total schedule length, including steps already run.

For Google Cloud Storage, pass `--out gs://YOUR_BUCKET/run` with Application Default Credentials.

## Generation

```bash
uv run scripts/serve.py --weights chkpt/experiment \
  --temperature 0.7 --top-k 50 --max-new-tokens 512
```

`--fp8` stores the MLP weights and KV cache in FP8. It saves memory but isn't always faster; see the [benchmarks](../benchmarks/RESULTS.md).

## Evaluation

```bash
uv run scripts/evals.py --weights chkpt/experiment \
  --tasks mmlu arc-challenge hellaswag \
  --fewshot mmlu 5 arc-challenge 25 hellaswag 10
```

Evaluation is zero-shot unless you pass `--fewshot`.

## GPT-OSS weights

```bash
uvx --from huggingface_hub hf download openai/gpt-oss-20b \
  --include 'original/*' --local-dir chkpt/gpt-oss-20b

uv run scripts/serve.py --variant gpt-oss-20b --weights chkpt/gpt-oss-20b --format gpt-oss
uv run scripts/evals.py --variant gpt-oss-20b --weights chkpt/gpt-oss-20b --format gpt-oss
```

Both GPT-OSS variants expect eight devices (4x2 DDP + TP). For other setups, change their sharding in [variants.py](../../lyra/variants.py).

## Python API

```python
import jax

from lyra.model import ModelWeights
from lyra.variants import model_variant

config = model_variant("lyra-small")
with jax.set_mesh(config.sharding.get_mesh()):
    model = ModelWeights.load("chkpt/lyra-small", config)
```

GPT-OSS checkpoints load with `lyra.gpt_oss.load_gpt_oss_model(path, config)`. For custom diagnostics, see `Probe` in [probe.py](../../lyra/training/probe.py) and `TrainHooks` in [train.py](../../lyra/training/train.py).
