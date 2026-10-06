# Lyra

Lyra is an LLM pretraining stack in pure JAX, built and optimized for TPUs. It covers everything from loading tokenized data to sampling from and evaluating a trained MoE model in a compact and lean codebase.

## Quick Start

On a TPU machine, install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/ibbyml/lyra.git
cd lyra
uv sync --extra tpu

# Download the 21B-token training set and train Lyra Small
uv run data/dataset.py
uv run scripts/train.py

# Generate text and evaluate the saved checkpoint
uv run scripts/serve.py
uv run scripts/evals.py --weights chkpt/lyra-small --tasks mmlu arc-challenge hellaswag
```

The default run is 308,000 updates (~20.19B tokens), saving to `chkpt/lyra-small` every 1,000 updates and at completion. Use `--steps` for a shorter run and `--variant` to choose a model.

For a small local example, follow the [20-step CPU example](dev/docs/usage.md#short-cpu-run).

## Results

To check the full stack before the full pretraining run, Lyra Small was trained on 500M tokens on a single TPU v6e. That run's recipe is now shared by the pretraining presets. It took roughly ~4.5 hours and kept a sustained 35,000 TPS, High MFU (+25%), and remained healthy for the duration of the run.

![Lyra Small trained on 500M tokens: loss, gradient norm, and expert routing balance](dev/runs/lyra-small-500m/metrics.png)

`val` loss fell from 12.62 to 3.06. Routing stayed balanced, and every expert was in use at the end.

<details>
<summary>Samples from the final checkpoint</summary>

> **Prompt**: The surprising thing about the ocean is

> **Completion**: that it is not always the case that the ocean is the only place where the ocean can support life. There are also some places where the ocean can be a source of food, a source of water, a source for life. [...]

> **Prompt**: The history of astronomy begins with

> **Completion**: Galileo Galilei. His work was published in 1619, and he was the first to use the word "astronomy" to refer to the study of celestial bodies in their natural environment. […]

</details>

The [run notes](dev/runs/lyra-small-500m) have the full setup, metrics, and samples.

## Details

### Model

Lyra Small is a 1.6B-parameter mixture-of-experts transformer with about 1B parameters active per token.

![Attention, layer schedule, and mixture of experts](dev/assets/model.svg)

Attention is a Gated GQA and alternates between global and local windows (GLGL). It also has support for tanh XSA, Learned Attention Sinks, and a quantized KV Cache.

| Model | Parameters | Width | Context | Layers | Experts |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lyra-small` | 1.6B | 2048  | 4096 | 16 | 15 + 1 shared |
| `lyra-medium` | 11.3B | 4096 | 4096 | 32 | 15 + 1 shared |
| `lyra-large` | 51.5B | 4096 | 4096 | 32 | 31 + 1 shared |
| `lyra-max` | 99.9B | 4096 | 4096 | 32 | 63 + 1 shared |

Lyra can also load OpenAI's original `gpt-oss-20b` and `gpt-oss-120b` checkpoints for generation and evaluation. The [design guide](dev/docs/design.md) covers the architecture, optimizer, and sharding in more depth.

### Kernels

![Kernel speedups and MFU for Small and Medium](dev/assets/kernel-benchmarks.svg)

Lyra currently has optimized kernels for the following operations:

- [Flash attention](lyra/kernels/flash_attention.py) for global and sliding-window layers. Sliding-window layers visit only the tiles inside the window.
- [Fused linear cross-entropy](lyra/kernels/cross_entropy.py). Fuses the Output Head Linear + Cross Entropy.
- [Grouped expert matmuls](lyra/kernels/gemm.py) fuse SwiGLU into the up projection, and a [SparseCore combine](lyra/kernels/combine.py), adapted from MaxText, gathers the expert outputs.
- Decode [attention](lyra/kernels/decode_attention.py) and [matmul](lyra/kernels/decode_gemm.py) kernels handle generation, with BF16 or FP8 caches and weights.

Against XLA on one TPU v6e at Small's shapes (batch 4, context 4,096; cross-entropy uses 4,096 total tokens), from the [2026-09-27 benchmarks](dev/benchmarks/RESULTS.md#lyra-small-b4):

| Operation | Pallas ms | XLA ms | Speedup | Temp. memory saved |
| --- | ---: | ---: | ---: | ---: |
| Global attention, Forward | 2.269 | 6.112 | 2.69× | 99.95% |
| SW-attention, Forward | 1.501 | 5.970 | 3.98× | 99.95% |
| MoE up projection + SwiGLU | 1.513 | 7.258 | 4.80× | 100% |
| MoE down projection | 0.861 | 3.240 | 3.76× | 100% |
| Linear Cross-Entropy, Forward | 5.068 | 7.330 | 1.45× | >99.99% |
| Expert combine, forward | 1.058 | 3.021 | 2.86× | 99.96% |

Memory savings compare compiler-allocated temporary buffers per call. Cross-entropy uses BF16 in Pallas and FP32 in the XLA reference. The [kernel guide](dev/docs/kernels.md) includes the full measurements and methodology.

### Training

Muon updates the weight matrices, and Adam handles everything else (embeddings, routers, norms, and gates). Weights are stored in FP32 and compute runs in BF16, and gradient accumulation reaches half-million-token batches on a single chip. Data streams from ArrayRecord shards through Grain. Orbax checkpoints, saved locally or on GCS, include the optimizer and data-loader state along with the weights, so an interrupted run resumes where it stopped.

### Making changes

Every setting lives in one of two dataclasses. `ModelConfig` in [model.py](lyra/model.py) covers the architecture, precision, sharding, and kernels. `TrainingConfig` in [train.py](lyra/training/train.py) covers the schedule, optimizer, data, and checkpoints. The models in [variants.py](lyra/variants.py) and the runs in [presets.py](lyra/training/presets.py) are instances of these, so an experiment is usually a new entry in one of those two files. Common training settings are also available as flags:

```bash
uv run scripts/train.py --variant lyra-small --steps 1000 \
  --data data/my-training-set --eval-data data/my-validation-set --out chkpt/experiment
```

The [usage guide](dev/docs/usage.md) covers data preparation, resuming, sampling, evaluation, and loading GPT-OSS weights.

## Acknowledgements

Lyra builds on ideas, tools, and writing from a number of people and projects:

- OpenAI's [GPT-OSS](https://github.com/openai/gpt-oss), whose architecture and design choices provided much of the foundation for this project.
- The Google DeepMind team and authors of [How to Scale Your Model](https://jax-ml.github.io/scaling-book/), which was an invaluable reference throughout development, particularly for distributed training and roofline analysis.
- Special thanks to Patrick Toulme for his [blog](https://patricktoulme.substack.com/). His posts on TPUs, Pallas, and XLA fundamentally improved my understanding of TPU programming and compiler output, and directly informed much of the Pallas kernel work in Lyra.
- The [JAX LLM](https://github.com/jax-ml/jax-llm-examples) and [Tokamax](https://github.com/openxla/tokamax) teams for excellent examples of practical, high-performance JAX development.
- Andrej Karpathy and [nanochat](https://github.com/karpathy/nanochat), which helped inspire the spirit and direction of this project.
- Keller Jordan for [Muon](https://kellerjordan.github.io/posts/muon/) and [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt), both of which influenced the training and optimization work here.
- NVIDIA and the CLIMB team for making the [ClimbMix dataset](https://research.nvidia.com/labs/lpr/climb/) available.

## Citation

If you find Lyra useful in your research, please cite:

```bibtex
@misc{Lyra,
  author = {Ibrahim Khan},
  title = {Lyra: A performant LLM pretraining stack in pure JAX.},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ibbyml/lyra}
}
```

## License

[MIT](LICENSE)
