from __future__ import annotations

import json
import math
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from etils.epath import Path
from jax import P, ShapeDtypeStruct
from jax.stages import Compiled, Lowered, Wrapped
from jax.tree_util import register_static
from jaxtyping import Array

from lyra.data import DataLoader, DataLoaderConfig
from lyra.model import ModelConfig, ModelWeights
from lyra.nn.checkpoints import checkpoint_path, checkpoint_run_config, checkpoint_steps, resolve_checkpoint
from lyra.tokenizer import LyraTokenizer
from lyra.training.metrics import RULE, MetricsLogger, format_duration
from lyra.training.plotting import plot_run
from lyra.training.probe import Probe, to_floats
from lyra.training.state import TrainState, init_train_state, load_train_state, save_checkpoint
from lyra.training.steps import acc_train_step, numerics_step, run_validation, sample_eval_step, validation_step

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "chkpt"
DATA_DIR = ROOT / "data"
DEFAULT_SAMPLE_PROMPTS = (
    "The surprising thing about the ocean is",
    "To make a cup of tea, first",
    "In a small village by the sea,",
    "Here is a simple explanation of gravity:",
    "The history of astronomy begins with",
)
RUN_ARTIFACTS = ("metrics.jsonl", "samples.jsonl", "numerics.jsonl")
MODEL_FIELDS_IGNORED_ON_RESUME = ("w_init", "b_init", "norm_init", "mode", "sample_topk", "sample_temperature", "quantize_cache")
SCHEDULE_FIELDS = ("steps", "batch_size", "accumulation_steps", "decay_type", "warmup_fraction", "stable_fraction")


@register_static
@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    # Schedule
    steps: int = 20000
    batch_size: int = 4
    accumulation_steps: int = 64
    decay_type: Literal["linear", "cosine"] = "cosine"
    warmup_fraction: float = 0.03
    stable_fraction: float = 0.0

    # Muon
    muon_max_lr: float = 1.5e-3
    muon_momentum: float = 0.95
    muon_update_rms: float = 0.2
    muon_wd: float = 0.1
    muon_nesterov: bool = True
    muon_grad_clip: float = 1.0
    muon_moment_dtype: jnp.dtype = jnp.bfloat16

    # Adam
    adam_max_lr: float = 2e-4
    adam_betas: tuple[float, float] = (0.90, 0.95)
    adam_mu_dtype: jnp.dtype = jnp.bfloat16
    adam_nu_dtype: jnp.dtype = jnp.float32

    # Data
    data_path: Path = DATA_DIR / "train"
    data_loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    eval_data_path: Path | None = None
    eval_batches: int = 4

    log_every: int = 5
    eval_every: int | None = 500
    sample_every: int | None = None
    numerics_every: int | None = None
    checkpoint_every: int = 1000
    checkpoint_keep: int = 2

    sample_prompts: tuple[str, ...] = DEFAULT_SAMPLE_PROMPTS
    sample_tokens: int = 128
    checkpoint_path: Path = CHECKPOINT_DIR / "model"
    resume_from: Path | None = None

    @property
    def warmup_steps(self) -> int:
        return min(math.ceil(self.warmup_fraction * self.steps), self.steps - 1)

    @property
    def stable_steps(self) -> int:
        return round(self.stable_fraction * self.steps)

    @property
    def decay_steps(self) -> int:
        return self.steps - self.warmup_steps - self.stable_steps


def resume_signature(mcfg: ModelConfig, tcfg: TrainingConfig, loader: DataLoader) -> dict:
    model = {name: value for name, value in asdict(mcfg).items() if name not in MODEL_FIELDS_IGNORED_ON_RESUME}
    training = {name: value for name, value in asdict(tcfg).items() if name in SCHEDULE_FIELDS or name.startswith(("muon_", "adam_"))}
    data = {"tokens": loader.token_count, "shuffle": tcfg.data_loader.shuffle, "seed": tcfg.data_loader.seed}
    return json.loads(json.dumps({"model": model, "training": training, "data": data}, default=str))


def check_resume(path: Path, signature: dict) -> None:
    saved = checkpoint_run_config(path)["resume_signature"]
    differing = [section for section in signature if saved.get(section) != signature[section]]
    if differing:
        raise ValueError(f"The {' and '.join(differing)} settings differ from {path}. Resume with the original run configuration.")


@dataclass
class TrainHooks:
    compiler_options: dict[str, str | bool] | None = None

    def on_lowered(self, lowered: Lowered) -> None:
        pass

    def on_compiled(self, compiled: Compiled, timing: dict[str, float]) -> None:
        pass

    def before_step(self, step: int, index: int, state: TrainState, xs: np.ndarray, ys: np.ndarray) -> None:
        pass

    def step_scope(self, step: int) -> AbstractContextManager[None]:
        return nullcontext()

    def times_step(self, step: int) -> bool:
        return False

    def after_step(self, step: int, state: TrainState, aux: dict[str, Array], step_time: float | None) -> None:
        pass

    def on_checkpoint(self, step: int, checkpoint_path: Path, run_path: Path) -> None:
        pass


def compile_step(step: Wrapped, name: str, lower_args: tuple, *, hooks: TrainHooks | None = None) -> Compiled:
    start = time.perf_counter()
    lowered = step.lower(*lower_args)
    lower_seconds = time.perf_counter() - start
    if hooks is not None:
        hooks.on_lowered(lowered)
    start = time.perf_counter()
    options = hooks.compiler_options if hooks is not None else None
    compiled = lowered.compile(options) if options else lowered.compile()
    compile_seconds = time.perf_counter() - start
    print(f"Compiled {name} step in {lower_seconds + compile_seconds:.1f}s")
    if hooks is not None:
        hooks.on_compiled(
            compiled,
            {"lower_seconds": lower_seconds, "xla_compile_seconds": compile_seconds, "total_seconds": lower_seconds + compile_seconds},
        )
    return compiled


def train(model: ModelWeights | None, mcfg: ModelConfig, tcfg: TrainingConfig, *, hooks: TrainHooks | None = None) -> TrainState:
    hooks = hooks or TrainHooks()
    out = checkpoint_path(tcfg.checkpoint_path)
    run_dir = ROOT / "logs" / out.name if str(out).startswith("gs://") else out
    run_dir.mkdir(parents=True, exist_ok=True)
    if tcfg.resume_from is None and checkpoint_steps(out):
        raise FileExistsError(f"{out} already has checkpoints. Pass --resume, or choose another --out.")
    if tcfg.batch_size % (data_devices := mcfg.sharding.data_axis_size):
        raise ValueError(f"--batch-size {tcfg.batch_size} must be a multiple of the {data_devices} data-parallel devices.")

    loader = DataLoader(tcfg.data_path, tcfg.batch_size, mcfg.seq_len, config=tcfg.data_loader)
    loader.check_steps(tcfg.steps, tcfg.accumulation_steps)
    signature = resume_signature(mcfg, tcfg, loader)
    eval_loader = None
    if tcfg.eval_data_path is not None:
        eval_config = replace(tcfg.data_loader, shuffle=False, num_epochs=None, worker_count=0)
        eval_loader = DataLoader(tcfg.eval_data_path, tcfg.batch_size, mcfg.seq_len, config=eval_config)
        eval_start = eval_loader.get_state()

    if tcfg.resume_from is None:
        assert model is not None, "a new run needs initial weights"
        start_step = 0
        state = init_train_state(model, mcfg, tcfg)
        for name in RUN_ARTIFACTS:
            (run_dir / name).unlink(missing_ok=True)
    else:
        path = resolve_checkpoint(tcfg.resume_from)
        check_resume(path, signature)
        restored = load_train_state(path, mcfg, tcfg)
        state, start_step = restored.state, restored.step
        loader.set_state(restored.loader_state)
        for name, text in checkpoint_run_config(path)["run_artifacts"].items():
            (run_dir / name).write_text(text)
        print(f"Resumed from {path} at step {start_step:,}")

    run_config = json.loads(json.dumps({"model": asdict(mcfg), "training": asdict(tcfg), "resume_signature": signature}, default=str))
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    def copy_logs() -> None:
        if run_dir != out:
            for name in ("config.json", "metrics.png", *RUN_ARTIFACTS):
                if (run_dir / name).exists():
                    (out / name).write_bytes((run_dir / name).read_bytes())

    acc = tcfg.accumulation_steps
    tokens_per_step = acc * tcfg.batch_size * mcfg.seq_len
    spec = P(None, *mcfg.sharding.batch_spec)
    batch = ShapeDtypeStruct((acc, tcfg.batch_size, mcfg.seq_len), jnp.int32, sharding=spec)
    scalar = ShapeDtypeStruct((), jnp.int32)
    train_step = compile_step(acc_train_step, "train", (batch, batch, scalar, state, spec, mcfg, tcfg), hooks=hooks)
    metrics = MetricsLogger(run_dir, tokens_per_step=tokens_per_step, initial_step=start_step)

    def due(every: int | None, step: int) -> bool:
        return step == tcfg.steps or (every is not None and step % every == 0)

    if eval_loader is not None:
        val_step = compile_step(validation_step, "validation", (batch, batch, state.params, spec, mcfg))

        def evaluate(step: int) -> None:
            eval_loader.set_state(eval_start)
            metrics.log_eval(step, run_validation(val_step, eval_loader, state.params, eval_batches=tcfg.eval_batches, microbatches=acc))

    if tcfg.sample_every is not None:
        tokenizer = LyraTokenizer()
        encoded = [[tokenizer.eos_token, *tokenizer.encode_ordinary(prompt)] for prompt in tcfg.sample_prompts]
        width = max(map(len, encoded))
        prompts = [(jnp.asarray(ids + [tokenizer.pad_token] * (width - len(ids)), jnp.int32), jnp.int32(len(ids))) for ids in encoded]
        sample_args = (state.params, ShapeDtypeStruct((width,), jnp.int32), scalar, scalar, tcfg.sample_tokens, mcfg)
        sample_step = compile_step(sample_eval_step, "sampling", sample_args)

        def sample(step: int) -> None:
            for index, (prompt, (ids, length)) in enumerate(zip(tcfg.sample_prompts, prompts)):
                tokens = np.asarray(sample_step(state.params, ids, length, jnp.int32(index))[0]).tolist()
                completion = tokenizer.decode(tokens[: tokens.index(tokenizer.eos_token)] if tokenizer.eos_token in tokens else tokens)
                record = {"step": step, "total_tokens": step * tokens_per_step, "prompt": prompt, "completion": completion}
                with (run_dir / "samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"\n{RULE}\nstep {step:,} | {prompt}{completion}\n{RULE}\n")

    if tcfg.numerics_every is not None:
        probe_batch = ShapeDtypeStruct((1, tcfg.batch_size, mcfg.seq_len), jnp.int32, sharding=spec)
        numerics = compile_step(numerics_step, "numerics", (probe_batch, state.params, spec, mcfg, Probe(routing_only=True)))

    with metrics.paused():
        if eval_loader is not None:
            evaluate(start_step)
        if tcfg.sample_every is not None:
            sample(start_step)

    run_start = time.perf_counter()
    try:
        for step in range(start_step + 1, tcfg.steps + 1):
            xs, ys = (np.stack(arrays) for arrays in zip(*(next(loader) for _ in range(acc))))
            if tcfg.numerics_every is not None and (step == 1 or due(tcfg.numerics_every, step)):
                record = {"kind": "step", "step": step, "records": to_floats(numerics(xs[:1], state.params))}
                with (run_dir / "numerics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")

            hooks.before_step(step, step - 1, state, xs, ys)
            timed = hooks.times_step(step)
            step_start = time.perf_counter()
            with hooks.step_scope(step):
                state, aux = train_step(xs, ys, np.int32(step - 1), state)
                if timed:
                    jax.block_until_ready(aux["loss"])
            metrics.record(aux)
            hooks.after_step(step, state, aux, time.perf_counter() - step_start if timed else None)

            checkpoint = due(tcfg.checkpoint_every, step)
            evaluating = eval_loader is not None and due(tcfg.eval_every, step)
            sampling = tcfg.sample_every is not None and due(tcfg.sample_every, step)
            if step == start_step + 1 or step % tcfg.log_every == 0 or checkpoint or evaluating:
                metrics.log_train(step)
            with metrics.paused() if checkpoint or evaluating or sampling else nullcontext():
                if checkpoint:
                    artifacts = {name: (run_dir / name).read_text() for name in RUN_ARTIFACTS if (run_dir / name).exists()}
                    save_checkpoint(
                        out, step, state, loader.get_state(), run_config | {"run_artifacts": artifacts}, keep=tcfg.checkpoint_keep
                    )
                    hooks.on_checkpoint(step, out, run_dir)
                if evaluating:
                    evaluate(step)
                if sampling:
                    sample(step)
                if checkpoint:
                    copy_logs()
    finally:
        metrics.close()

    plot_run(run_dir)
    copy_logs()
    print(f"{RULE}\nTraining finished in {format_duration(time.perf_counter() - run_start)}\n{RULE}")
    return state
