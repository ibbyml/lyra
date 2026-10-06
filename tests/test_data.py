from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data import dataset
from data.prepare_climbmix import ShardWriter
from lyra.data import DataLoader, DataLoaderConfig
from lyra.training.presets import TRAINING_PRESETS
from lyra.variants import lyra_small


def test_default_download_covers_the_small_run() -> None:
    preset = TRAINING_PRESETS["train-small"]
    required = preset.steps * preset.batch_size * preset.accumulation_steps * lyra_small.seq_len
    available = len(dataset.training_shards(dataset.Args().max_tokens)) * dataset.TOKENS_PER_SHARD
    assert required < available


def test_last_shard_is_held_out_for_validation() -> None:
    assert len(dataset.training_shards(600_000_000)) == 2
    assert dataset.SHARDS - 1 not in dataset.training_shards(None)
    with pytest.raises(SystemExit):
        dataset.training_shards(30_000_000_000)


def test_download_paths_match_the_small_preset() -> None:
    preset = TRAINING_PRESETS["train-small"]
    assert str(dataset.DATA_DIR / "train") == str(preset.data_path)
    assert str(dataset.DATA_DIR / "val") == str(preset.eval_data_path)


def test_shards_read_back_as_one_token_stream(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, tokens_per_shard=10)
    for start, length in ((0, 5), (5, 7), (12, 3), (15, 10)):
        writer.write(list(range(start, start + length)))
    writer.close()
    assert sorted(path.name for path in tmp_path.glob("*.array_record")) == [f"shard-0000{i}.array_record" for i in range(3)]

    loader = DataLoader(tmp_path, 2, 3, config=DataLoaderConfig(worker_count=0))
    assert (loader.token_count, loader.block_count) == (25, 4)
    for block in range(loader.block_count):
        x, y = next(loader)
        np.testing.assert_array_equal(x.ravel(), np.arange(block * 6, block * 6 + 6))
        np.testing.assert_array_equal(y.ravel(), np.arange(block * 6 + 1, block * 6 + 7))
    with pytest.raises(ValueError, match="Download more data"):
        loader.check_steps(steps=3, accumulation_steps=2)
