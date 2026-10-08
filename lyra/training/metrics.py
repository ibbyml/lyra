from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import numpy as np
from etils.epath import Path

RULE = "─" * 60
PPL_CE_CLAMP = 20.0

TRAIN_COLUMNS = (
    ("step", "step", lambda v: f"{int(v):,}"),
    ("loss", "loss", lambda v: f"{v:.4f}"),
    ("ce_loss", "ce_loss", lambda v: f"{v:.4f}"),
    ("aux_loss", "aux", lambda v: f"{v:.4f}"),
    ("perplexity", "ppl", lambda v: f"{v:,.0f}"),
    ("step_time", "step_ms", lambda v: f"{v * 1e3:.1f}"),
    ("tps", "tok/s", lambda v: f"{v:,.0f}"),
    ("total_tokens", "tokens", lambda v: f"{v / 1e9:.3f}B" if v >= 1e9 else f"{v / 1e6:.3f}M"),
)


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s" if minutes else f"{seconds:.1f}s"


class MetricsLogger:
    def __init__(self, run_dir: Path, tokens_per_step: int = 0, initial_step: int = 0):
        self.tokens_per_step = tokens_per_step
        self.total_tokens = initial_step * tokens_per_step
        self.file = (run_dir / "metrics.jsonl").open("a")
        self.window: list[dict] = []
        self.window_start = time.perf_counter()
        self.printed_header = False

    def record(self, aux: dict) -> None:
        self.window.append(aux)
        self.total_tokens += self.tokens_per_step

    def log_train(self, step: int) -> None:
        if not self.window:
            return
        rows = jax.device_get(self.window)
        elapsed = time.perf_counter() - self.window_start
        averages = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        averages["perplexity"] = float(np.exp(min(averages["ce_loss"], PPL_CE_CLAMP)))
        averages["step_time"] = elapsed / len(rows)
        averages["tps"] = self.tokens_per_step * len(rows) / elapsed
        averages["total_tokens"] = self.total_tokens
        record = {"kind": "train", "step": step} | averages
        if not self.printed_header:
            print("  ".join(header.ljust(8) for _, header, _ in TRAIN_COLUMNS))
            self.printed_header = True
        print("  ".join(fmt(record[key]).ljust(8) for key, _, fmt in TRAIN_COLUMNS))
        self._write(record)
        self.window = []
        self.window_start = time.perf_counter()

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Leave evaluation, sampling, and checkpointing out of the step time and throughput."""
        jax.block_until_ready(self.window)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.window_start += time.perf_counter() - start

    def log_eval(self, step: int, metrics: dict) -> None:
        record = {"kind": "eval", "step": step} | {key: float(value) for key, value in metrics.items()}
        print(f"eval | step {step:,} | loss {record['eval_loss']:.4f} | ce_loss {record['eval_ce_loss']:.4f}")
        self._write(record)

    def _write(self, record: dict) -> None:
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
