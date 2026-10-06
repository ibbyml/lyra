from functools import partial

from lyra.evals import arc, hellaswag, mmlu
from lyra.evals.common import Doc, EvalResult, Scorer, evaluate

TASKS = {
    "mmlu": mmlu.load_docs,
    "arc-challenge": partial(arc.load_docs, subset="ARC-Challenge"),
    "arc-easy": partial(arc.load_docs, subset="ARC-Easy"),
    "hellaswag": hellaswag.load_docs,
}

__all__ = ["TASKS", "Doc", "EvalResult", "Scorer", "evaluate"]
