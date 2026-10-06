from __future__ import annotations

import random

from datasets import load_dataset

from lyra.evals.common import Doc, few_shot_prefix, select_limit

DATASET = "allenai/ai2_arc"
REVISION = "210d026faf9955653af8916fad021475a3f00453"


def _to_doc(row: dict, prefix: str = "") -> Doc:
    gold = row["choices"]["label"].index(row["answerKey"])
    context = f"{prefix}Question: {row['question'].strip()}\nAnswer:"
    return Doc(context=context, choices=tuple(row["choices"]["text"]), gold=gold, normalize_choices=True)


def load_docs(*, subset: str, limit: int | None = None, num_fewshot: int = 0, fewshot_seed: int = 1234) -> list[Doc]:
    train = load_dataset(DATASET, subset, split="train", revision=REVISION) if num_fewshot else []
    rng = random.Random(fewshot_seed)
    docs = []
    for row in select_limit(load_dataset(DATASET, subset, split="test", revision=REVISION), limit):
        shots = [_to_doc(train[index]) for index in rng.sample(range(len(train)), num_fewshot)]
        docs.append(_to_doc(row, few_shot_prefix(shots)))
    return docs
