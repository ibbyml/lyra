from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import P, shard_map

from lyra.model import ModelConfig, ModelWeights, model_apply
from lyra.nn.params import all_gather_params, weight_sharding
from lyra.tokenizer import LyraTokenizer


@dataclass(frozen=True)
class Doc:
    context: str
    choices: tuple[str, ...]
    gold: int
    normalize_choices: bool = False


@dataclass
class EvalResult:
    task: str
    num_docs: int
    acc: float
    acc_norm: float | None = None

    def __str__(self) -> str:
        norm = "" if self.acc_norm is None else f" acc_norm={self.acc_norm:.4f}"
        return f"{self.task}: acc={self.acc:.4f}{norm} (n={self.num_docs})"


@cache
def _compiled_scorer(config: ModelConfig):
    param_specs = weight_sharding(ModelWeights.spec(config))

    def score(model, tokens, targets, mask):
        model = all_gather_params(model, param_specs, config.sharding)
        logits, _, _ = model_apply(tokens, model, cache=None, config=config)
        logprobs = jax.nn.log_softmax(logits[..., : config.vocab_size].astype(jnp.float32), axis=-1)
        return (jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0] * mask).sum(axis=-1)

    sharded = shard_map(score, mesh=config.sharding.get_mesh(), in_specs=(param_specs, P(), P(), P()), out_specs=P(), check_vma=False)
    return jax.jit(sharded)


class Scorer:
    def __init__(self, model: ModelWeights, config: ModelConfig, *, batch_size: int = 8, harmony: bool = False) -> None:
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.tokenizer = LyraTokenizer()
        self.harmony = None
        if harmony:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding

            self.harmony = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def _encode(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        if self.harmony is None:
            return [self.tokenizer.eos_token, *self.tokenizer.encode_ordinary(context)], self.tokenizer.encode_ordinary(continuation)
        from openai_harmony import Conversation, Message, Role

        conversation = Conversation.from_messages([Message.from_role_and_content(Role.USER, context)])
        prompt = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)
        prompt += self.harmony.encode("<|channel|>final<|message|>", allowed_special="all")
        return prompt, self.harmony.encode(continuation)

    def score(self, docs: list[Doc]) -> list[list[float]]:
        requests = []
        for d, doc in enumerate(docs):
            context = doc.context.rstrip()
            for c, choice in enumerate(doc.choices):
                ctx, cont = self._encode(context, doc.context[len(context) :] + " " + choice)
                ctx = ctx[-(self.config.seq_len - len(cont)) :]
                requests.append((d, c, ctx + cont, len(ctx)))

        width = max(len(tokens) for _, _, tokens, _ in requests)
        scores = [[0.0] * len(doc.choices) for doc in docs]
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            tokens = np.full((self.batch_size, width), self.tokenizer.pad_token, dtype=np.int32)
            targets = np.zeros((self.batch_size, width), dtype=np.int32)
            mask = np.zeros((self.batch_size, width), dtype=np.float32)
            for row, (_, _, ids, cont_start) in enumerate(batch):
                tokens[row, : len(ids)] = ids
                targets[row, : len(ids) - 1] = ids[1:]
                mask[row, cont_start - 1 : len(ids) - 1] = 1.0
            sums = np.asarray(_compiled_scorer(self.config)(self.model, tokens, targets, mask))
            for row, (d, c, _, _) in enumerate(batch):
                scores[d][c] = float(sums[row])
        return scores


def evaluate(task: str, docs: list[Doc], scorer: Scorer) -> EvalResult:
    scores = scorer.score(docs)
    correct = [int(np.argmax(s)) == doc.gold for doc, s in zip(docs, scores)]
    acc_norm = None
    if docs[0].normalize_choices:
        normalized = [[s / len(choice) for s, choice in zip(ss, doc.choices)] for doc, ss in zip(docs, scores)]
        acc_norm = float(np.mean([int(np.argmax(s)) == doc.gold for doc, s in zip(docs, normalized)]))
    return EvalResult(task=task, num_docs=len(docs), acc=float(np.mean(correct)), acc_norm=acc_norm)


def few_shot_prefix(docs: list[Doc]) -> str:
    return "".join(f"{doc.context} {doc.choices[doc.gold]}\n\n" for doc in docs)


def select_limit(data, limit: int | None):
    return data if limit is None else data.select(range(min(limit, len(data))))
