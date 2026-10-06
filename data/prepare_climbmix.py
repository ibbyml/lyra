"""Tokenize ClimbMix into ArrayRecord shards in the same format as the published dataset."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from array_record.python import array_record_module as ar  # type: ignore

from lyra.data import LENGTHS_SUFFIX, TOKEN_DTYPE
from lyra.tokenizer import LyraTokenizer

DATA_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Args:
    output: Path = DATA_DIR / "train"
    """Shard directory."""
    max_tokens: int = 30_000_000_000
    """Tokens to write."""
    dataset: str = "karpathy/climbmix-400b-shuffle"
    """Hugging Face dataset with a text column."""
    tokens_per_shard: int = 300_000_000
    """Tokens per shard."""
    num_threads: int = os.cpu_count() or 1
    """Tokenizer threads."""


class ShardWriter:
    """Writes one record per document, starting a new shard every `tokens_per_shard` tokens."""

    def __init__(self, directory: Path, tokens_per_shard: int) -> None:
        self.directory = directory
        self.tokens_per_shard = tokens_per_shard
        self.index = 0
        self.writer = None
        self.lengths: list[int] = []
        self.count = 0

    def write(self, tokens: list[int]) -> None:
        while tokens:
            if self.writer is None:
                self.path = self.directory / f"shard-{self.index:05d}.array_record"
                self.writer = ar.ArrayRecordWriter(str(self.path), "group_size:1")
            chunk, tokens = tokens[: self.tokens_per_shard - self.count], tokens[self.tokens_per_shard - self.count :]
            self.writer.write(np.asarray(chunk, dtype=TOKEN_DTYPE).tobytes())
            self.lengths.append(len(chunk))
            self.count += len(chunk)
            if self.count == self.tokens_per_shard:
                self.close()

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        np.save(str(self.path) + LENGTHS_SUFFIX, np.asarray(self.lengths, dtype=np.int64))
        print(f"Wrote {self.path} ({self.count:,} tokens)", flush=True)
        self.index, self.writer, self.lengths, self.count = self.index + 1, None, [], 0


def documents(dataset: str, tokenizer: LyraTokenizer, num_threads: int) -> Iterator[list[int]]:
    from datasets import load_dataset

    for batch in load_dataset(dataset, split="train", streaming=True).iter(batch_size=4096):
        for tokens in tokenizer.encode_ordinary_batch(batch["text"], num_threads=num_threads):
            yield [*tokens, tokenizer.eos_token]


def main() -> int:
    args = tyro.cli(Args)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = LyraTokenizer()
    writer = ShardWriter(args.output, args.tokens_per_shard)
    remaining = args.max_tokens
    for tokens in documents(args.dataset, tokenizer, args.num_threads):
        tokens = tokens[:remaining]
        writer.write(tokens)
        remaining -= len(tokens)
        if not remaining:
            break
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
