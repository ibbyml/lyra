from __future__ import annotations

import pytest

from lyra.tokenizer import PADDED_VOCAB_SIZE, LyraTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> LyraTokenizer:
    return LyraTokenizer()


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def f(x): return x * 2  # code-ish\n",
        "unicode: café — naïve — 日本語",
    ],
)
def test_encode_decode_roundtrip(tokenizer: LyraTokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode_ordinary(text)) == text


def test_special_tokens_present(tokenizer: LyraTokenizer) -> None:
    assert tokenizer.eos_token == 199999
    # pad and eos must differ so packing/masking never confuses them.
    assert tokenizer.pad_token != tokenizer.eos_token
    assert 0 <= tokenizer.pad_token < PADDED_VOCAB_SIZE
    assert "<|reserved_200018|>" not in tokenizer._special_tokens
    assert tokenizer.decode([200018]) == "<|endofprompt|>"


def test_batch_matches_single(tokenizer: LyraTokenizer) -> None:
    texts = ["alpha", "beta gamma", "delta"]
    batched = tokenizer.encode_ordinary_batch(texts)
    assert batched == [tokenizer.encode_ordinary(t) for t in texts]


def test_single_token_bytes_roundtrip(tokenizer: LyraTokenizer) -> None:
    ids = tokenizer.encode_ordinary("tokenization")
    rebuilt = b"".join(tokenizer.decode_single_token_bytes(i) for i in ids)
    assert rebuilt.decode("utf-8") == "tokenization"
