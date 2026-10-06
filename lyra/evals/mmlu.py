from __future__ import annotations

from collections import defaultdict

from datasets import load_dataset

from lyra.evals.common import Doc, select_limit

DATASET = "cais/mmlu"
REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
LETTERS = ("A", "B", "C", "D")


def _question(row: dict) -> str:
    choices = "".join(f"{letter}. {choice}\n" for letter, choice in zip(LETTERS, row["choices"]))
    return f"{row['question'].strip()}\n{choices}Answer:"


def load_docs(*, limit: int | None = None, num_fewshot: int = 0, fewshot_seed: int = 1234) -> list[Doc]:
    """Few-shot examples are the first rows of each subject's dev split, so the seed is unused."""
    shots: dict[str, str] = defaultdict(str)
    if num_fewshot:
        counts: dict[str, int] = defaultdict(int)
        for row in load_dataset(DATASET, "all", split="dev", revision=REVISION):
            if counts[row["subject"]] < num_fewshot:
                counts[row["subject"]] += 1
                shots[row["subject"]] += f"{_question(row)} {LETTERS[int(row['answer'])]}\n\n"

    docs = []
    for row in select_limit(load_dataset(DATASET, "all", split="test", revision=REVISION), limit):
        header = f"The following are multiple choice questions (with answers) about {row['subject'].replace('_', ' ')}.\n\n"
        docs.append(Doc(context=header + shots[row["subject"]] + _question(row), choices=LETTERS, gold=int(row["answer"])))
    return docs
