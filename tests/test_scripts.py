import os

from lyra.training.presets import TRAINING_PRESETS
from scripts import serve, train


def test_presets_save_where_serving_looks() -> None:
    for preset in TRAINING_PRESETS:
        variant = "dev" if preset == "train-dev" else preset.replace("train-", "lyra-")
        config = train.training_config(train.Args(variant=variant))
        assert str(config.checkpoint_path) == os.path.join(serve.ROOT, "chkpt", variant)
