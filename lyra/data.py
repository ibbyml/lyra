from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Self

import grain.python as grain
import jax
import numpy as np
from absl import flags
from etils.epath import Path

# random grain warning suppression
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module=r"multiprocessing\.resource_tracker")

TOKEN_DTYPE = np.dtype("<i4")
LENGTHS_SUFFIX = ".lengths.npy"


@dataclass(frozen=True, kw_only=True)
class DataLoaderConfig:
    shuffle: bool = False
    seed: int = 42
    num_epochs: int | None = 1
    worker_count: int = 2


class TokenBlocks(grain.RandomAccessDataSource):
    def __init__(self, path: Path, batch_size: int, seq_len: int):
        self.path = path
        self.shape = (batch_size, seq_len)
        self.size = batch_size * seq_len
        if path.suffix == ".npy":
            self.tokens = np.load(path, mmap_mode="r")
            self.token_count = len(self.tokens)
        else:
            shards = sorted(str(shard) for shard in path.glob("*.array_record")) if path.is_dir() else [str(path)]
            self.records = grain.ArrayRecordDataSource(shards)
            lengths = np.concatenate([np.load(shard + LENGTHS_SUFFIX) for shard in shards]).astype(np.int64)
            self.offsets = np.concatenate(([0], np.cumsum(lengths)))
            self.token_count = int(self.offsets[-1])

    def __len__(self) -> int:
        return (self.token_count - 1) // self.size

    def __repr__(self) -> str:
        return f"TokenBlocks({self.path}, shape={self.shape})"

    def _read(self, start: int, count: int) -> np.ndarray:
        if hasattr(self, "tokens"):
            return np.asarray(self.tokens[start : start + count], dtype=TOKEN_DTYPE)
        end, parts = start + count, []
        record = int(np.searchsorted(self.offsets, start, side="right")) - 1
        while start < end:
            tokens = np.frombuffer(self.records[record], dtype=TOKEN_DTYPE)
            take = min(end, int(self.offsets[record + 1])) - start
            local = start - int(self.offsets[record])
            parts.append(tokens[local : local + take])
            start += take
            record += 1
        return np.concatenate(parts)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        tokens = self._read(index * self.size, self.size + 1)
        return {"input_ids": tokens[:-1].reshape(self.shape), "labels": tokens[1:].reshape(self.shape)}


class DataLoader:
    def __init__(self, path: str | os.PathLike, batch_size: int, seq_len: int, *, config: DataLoaderConfig):
        source = TokenBlocks(Path(path), batch_size, seq_len)
        self.config = config
        self.token_count = source.token_count
        self.block_count = len(source)
        print(f"Loaded {self.token_count:,} tokens from {path} ({self.block_count:,} blocks of {batch_size} x {seq_len})")

        sampler = grain.IndexSampler(
            num_records=self.block_count,
            shuffle=config.shuffle,
            seed=config.seed,
            num_epochs=config.num_epochs,
            shard_options=grain.ShardOptions(shard_index=jax.process_index(), shard_count=jax.process_count(), drop_remainder=True),
        )
        if config.worker_count > 0 and not flags.FLAGS.is_parsed():
            flags.FLAGS.mark_as_parsed()
        loader = grain.DataLoader(
            data_source=source,
            sampler=sampler,
            worker_count=config.worker_count,
            worker_buffer_size=4,
            read_options=grain.ReadOptions(num_threads=2, prefetch_buffer_size=4),
        )
        self._iter = iter(loader)

    def check_steps(self, steps: int, accumulation_steps: int) -> None:
        needed = steps * accumulation_steps
        if self.config.num_epochs is not None and needed > self.block_count * self.config.num_epochs:
            raise ValueError(
                f"{steps:,} steps need {needed:,} blocks, but {self.token_count:,} tokens give {self.block_count:,} per epoch. "
                "Download more data or train for fewer steps."
            )

    def get_state(self) -> bytes:
        return self._iter.get_state()

    def set_state(self, state: bytes) -> None:
        self._iter.set_state(state)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        batch = next(self._iter)
        return batch["input_ids"], batch["labels"]
