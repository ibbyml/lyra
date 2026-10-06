from __future__ import annotations

import random
import re

from datasets import load_dataset

from lyra.evals.common import Doc, few_shot_prefix, select_limit

DATASET = "Rowan/hellaswag"
REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def _clean(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text.strip().replace(" [title]", ". "))
    return text.replace("  ", " ")


def _to_doc(row: dict, prefix: str = "") -> Doc:
    context = _clean(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
    endings = tuple(_clean(ending) for ending in row["endings"])
    return Doc(context=f"{prefix}{context}", choices=endings, gold=int(row["label"]), normalize_choices=True)


def load_docs(*, limit: int | None = None, num_fewshot: int = 0, fewshot_seed: int = 1234) -> list[Doc]:
    train = load_dataset(DATASET, split="train", revision=REVISION) if num_fewshot else []
    rng = random.Random(fewshot_seed)
    docs = []
    for row in select_limit(load_dataset(DATASET, split="validation", revision=REVISION), limit):
        shots = [_to_doc(train[index]) for index in rng.sample(range(len(train)), num_fewshot)]
        docs.append(_to_doc(row, few_shot_prefix(shots)))
    return docs
