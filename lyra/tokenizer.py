import tiktoken

PADDED_VOCAB_SIZE = 204800


class LyraTokenizer:
    def __init__(self):
        base = tiktoken.get_encoding("o200k_base")
        self._special_tokens = {
            **base._special_tokens,
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|reserved_200000|>": 200000,
            "<|reserved_200001|>": 200001,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|reserved_200004|>": 200004,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|reserved_200009|>": 200009,
            "<|reserved_200010|>": 200010,
            "<|reserved_200011|>": 200011,
            "<|call|>": 200012,
        } | {f"<|reserved_{i}|>": i for i in range(200013, PADDED_VOCAB_SIZE) if i not in base._special_tokens.values()}
        self.tokenizer = tiktoken.Encoding(
            name="o200k_harmony",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens=self._special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def encode_ordinary(self, text: str) -> list[int]:
        return self.tokenizer.encode_ordinary(text)

    def encode_ordinary_batch(self, texts: list[str], num_threads: int = 8) -> list[list[int]]:
        return self.tokenizer.encode_ordinary_batch(texts, num_threads=num_threads)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self.tokenizer.decode_single_token_bytes(token)

    @property
    def eos_token(self):
        return self._special_tokens["<|endoftext|>"]

    @property
    def pad_token(self):
        return self._special_tokens["<|reserved_200004|>"]
