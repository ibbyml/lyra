"""Download pre-tokenized ClimbMix shards. The last shard is held out for validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import tyro

from lyra.data import LENGTHS_SUFFIX

DATA_DIR = Path(__file__).resolve().parent
REPO = "Ibbyml/lyra-climbmix-30b"
REVISION = "346a345f0a0e2671da6ed8112b45f1197598ebb1"
SHARDS = 100
TOKENS_PER_SHARD = 300_000_000


@dataclass(frozen=True)
class Args:
    max_tokens: int | None = 21_000_000_000
    """Training tokens to download, rounded up to whole shards; None downloads every training shard."""
    val: bool = True
    """Also download the held-out validation shard."""


def training_shards(max_tokens: int | None) -> range:
    count = SHARDS - 1 if max_tokens is None else math.ceil(max_tokens / TOKENS_PER_SHARD)
    if count >= SHARDS:
        raise SystemExit(f"error: the dataset has {(SHARDS - 1) * TOKENS_PER_SHARD:,} training tokens")
    return range(count)


def download(shards: range | list[int], directory: Path) -> None:
    from huggingface_hub import snapshot_download

    names = [f"shard-{index:05d}.array_record" for index in shards]
    patterns = names + [name + LENGTHS_SUFFIX for name in names]
    print(f"Downloading {len(names)} shard(s) to {directory}")
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=directory, allow_patterns=patterns)


def main() -> int:
    args = tyro.cli(Args)
    download(training_shards(args.max_tokens), DATA_DIR / "train")
    if args.val:
        download([SHARDS - 1], DATA_DIR / "val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
