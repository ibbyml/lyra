import codecs
import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array, P, lax, shard_map
from jax.sharding import NamedSharding

from lyra.kernels.op import Mode
from lyra.model import LayerCache, ModelCache, ModelConfig, ModelWeights, model_apply
from lyra.nn.params import all_gather_params, is_spec, weight_sharding
from lyra.nn.quant import QArray
from lyra.tokenizer import LyraTokenizer

GenFn = Callable[..., tuple[Array, ModelCache]]


@dataclass(frozen=True, kw_only=True)
class InferenceConfig:
    weights: str
    max_new_tokens: int = 256
    max_cache_length: int | None = None
    fp8: bool = False
    temperature: float | None = None
    top_k: int | None = None


def apply_inference_overrides(mcfg: ModelConfig, icfg: InferenceConfig) -> ModelConfig:
    if icfg.temperature is not None:
        mcfg = replace(mcfg, sample_temperature=icfg.temperature)
    if icfg.top_k is not None:
        mcfg = replace(mcfg, sample_topk=icfg.top_k)
    if icfg.fp8:
        mcfg = replace(mcfg, gemm_dtype=jnp.float8_e4m3fn, quantize_mlp=True, quantize_cache=True)
    return mcfg


def load_model(mcfg: ModelConfig, icfg: InferenceConfig) -> ModelWeights:
    mcfg = apply_inference_overrides(mcfg, icfg)
    return ModelWeights.quantize(ModelWeights.load(icfg.weights, mcfg), mcfg)


def sample_token(logits: Array, key: Array, mcfg: ModelConfig) -> Array:
    if mcfg.sample_temperature <= 0:
        return jnp.argmax(logits).astype(jnp.int32)
    logits = logits.astype(jnp.float32) / mcfg.sample_temperature
    if 0 < mcfg.sample_topk < logits.shape[-1]:
        _, top = lax.top_k(logits, mcfg.sample_topk)
        logits = jnp.full_like(logits, jnp.finfo(jnp.float32).min).at[top].set(logits[top])
    return jax.random.categorical(key, logits).astype(jnp.int32)


def generate(
    ctx: Array, model: ModelWeights, cache: ModelCache, key: Array, mcfg: ModelConfig, sample_pos: Array
) -> tuple[Array, ModelCache]:
    logits, _, updated = model_apply(tokens=ctx, model=model, cache=cache, config=mcfg)
    token = sample_token(lax.dynamic_index_in_dim(logits[0], sample_pos, axis=0, keepdims=False), key, mcfg)
    return token, cast(ModelCache, updated)


def sample_tokens(model: ModelWeights, key: Array, mcfg: ModelConfig, prompt: Array, prompt_length: Array, max_new_tokens: int) -> Array:
    prefill, decode = replace(mcfg, mode=Mode.PREFILL), replace(mcfg, mode=Mode.DECODE)
    param_specs = weight_sharding(ModelWeights.spec(mcfg))

    def sample(model_l, key_l, prompt_l, prompt_length_l):
        model_l = all_gather_params(model_l, param_specs, mcfg.sharding)
        kv_heads = mcfg.n_kv_heads // mcfg.sharding.model_axis_size
        buffer = jnp.zeros((1, kv_heads, prompt.shape[0] + max_new_tokens, mcfg.head_dim), dtype=mcfg.compute_dtype)
        cache = ModelCache(layers=tuple(LayerCache(k=buffer, v=buffer) for _ in range(mcfg.n_layers)), pos=jnp.int32(0))
        first, cache = generate(prompt_l[None, :], model_l, cache, jax.random.fold_in(key_l, 0), prefill, prompt_length_l - 1)
        cache = replace(cache, pos=prompt_length_l)

        def step(carry, index):
            token, cache = carry
            token, cache = generate(token.reshape(1, 1), model_l, cache, jax.random.fold_in(key_l, index), decode, jnp.int32(0))
            return (token, cache), token

        _, rest = lax.scan(step, (first, cache), jnp.arange(1, max_new_tokens))
        return jnp.concatenate([first[None], rest])[None]

    sharded = shard_map(sample, mesh=mcfg.sharding.get_mesh(), in_specs=(param_specs, P(), P(), P()), out_specs=P(), check_vma=False)
    return jax.jit(sharded)(model, key, prompt, prompt_length)


def _kv_head_spec(mcfg: ModelConfig) -> P:
    names = mcfg.sharding.model_axis_names
    return P(None, names[0] if len(names) == 1 else names, None, None) if names else P()


def init_cache(mcfg: ModelConfig, length: int) -> ModelCache:
    sharding = NamedSharding(mcfg.sharding.get_mesh(), _kv_head_spec(mcfg))
    buffer = jnp.zeros((1, mcfg.n_kv_heads, length, mcfg.head_dim), dtype=mcfg.compute_dtype, device=sharding)
    return ModelCache.quantize(ModelCache(layers=tuple(LayerCache(buffer, buffer) for _ in range(mcfg.n_layers)), pos=0), mcfg)


def _param_specs(model: ModelWeights, mcfg: ModelConfig):
    def spec(value, array_spec):
        if not isinstance(value, QArray):
            return array_spec.sharding
        # A fused FP8 up projection drops the gate/value axis, so its spec drops one entry too.
        parts = tuple(array_spec.sharding)
        p = array_spec.sharding if value.qval.ndim == len(parts) else P(*parts[:-1])
        return QArray(qval=cast(Any, p), scale=cast(Any, p), block=value.block)

    return jax.tree.map(spec, model, ModelWeights.spec(mcfg), is_leaf=lambda x: is_spec(x) or isinstance(x, QArray))


def sharded_generate(model: ModelWeights, cache: ModelCache, mcfg: ModelConfig) -> GenFn:
    param_specs = _param_specs(model, mcfg)
    kv = _kv_head_spec(mcfg)
    cache_specs = ModelCache(
        layers=tuple(LayerCache(k=jax.tree.map(lambda _: kv, layer.k), v=jax.tree.map(lambda _: kv, layer.v)) for layer in cache.layers),
        pos=cast(Any, P()),
    )

    @jax.jit
    @shard_map(
        mesh=mcfg.sharding.get_mesh(), in_specs=(P(), param_specs, cache_specs, P(), P()), out_specs=(P(), cache_specs), check_vma=False
    )
    def step(ctx, model_l, cache_l, key, sample_pos):
        model_l = all_gather_params(model_l, param_specs, mcfg.sharding)
        return generate(ctx, model_l, cache_l, key, mcfg, sample_pos)

    return lambda ctx, cache, key, sample_pos: step(ctx, model, cache, key, sample_pos)


def serve_model(model: ModelWeights, mcfg: ModelConfig, icfg: InferenceConfig, *, harmony: bool = False) -> None:
    mcfg = apply_inference_overrides(mcfg, icfg)
    capacity = icfg.max_cache_length or mcfg.seq_len
    prefill = sharded_generate(model, init_cache(mcfg, capacity), replace(mcfg, mode=Mode.PREFILL))
    decode = sharded_generate(model, init_cache(mcfg, capacity), replace(mcfg, mode=Mode.DECODE))
    tokenizer = LyraTokenizer()

    if harmony:
        from openai_harmony import Conversation, HarmonyEncodingName, Message, Role, StreamableParser, load_harmony_encoding

        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_tokens = set(encoding.stop_tokens_for_assistant_actions())

        def user_turn(prompt: str) -> list:
            return [Message.from_role_and_content(Role.USER, prompt)]

        def reply_turns(tokens: list[int]) -> list:
            return encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT, strict=False)

        def render(turns: list) -> list[int]:
            return encoding.render_conversation_for_completion(Conversation.from_messages(turns), Role.ASSISTANT)

        def text_stream() -> Callable[[int | None], str]:
            parser = StreamableParser(encoding, Role.ASSISTANT, strict=False)

            def final_channel(token: int | None) -> str:
                if token is None:
                    return ""
                parser.process(token)
                return (parser.last_content_delta or "") if parser.current_channel == "final" else ""

            return final_channel
    else:
        stop_tokens = {tokenizer.eos_token}

        def user_turn(prompt: str) -> list:
            return [[tokenizer.eos_token, *tokenizer.encode_ordinary(prompt)]]

        def reply_turns(tokens: list[int]) -> list:
            return [tokens]

        def render(turns: list) -> list[int]:
            return [token for turn in turns for token in turn]

        def text_stream() -> Callable[[int | None], str]:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            return lambda token: decoder.decode(b"" if token is None else tokenizer.decode_single_token_bytes(token), final=token is None)

    print("Type a prompt to talk to the model. Ctrl-D or 'exit' to quit.\n")
    key = jax.random.key(mcfg.seed)
    turns: list = []
    for turn in itertools.count():
        try:
            prompt = input("User: ").strip()
        except EOFError:
            break
        if prompt in ("exit", "quit"):
            break
        if not prompt:
            continue
        context = render([*turns, *user_turn(prompt)])
        if len(context) >= capacity:
            turns, context = [], render(user_turn(prompt))
            if len(context) >= capacity:
                print(f"\n[the prompt needs {len(context)} tokens but the context window is {capacity}]\n")
                continue
            print("\n[context window full; starting a new conversation]\n")

        turn_key = jax.random.fold_in(key, turn)
        padded = jnp.asarray([context + [tokenizer.pad_token] * (capacity - len(context))], dtype=jnp.int32)
        token, cache = prefill(padded, init_cache(mcfg, capacity), jax.random.split(turn_key)[1], jnp.int32(len(context) - 1))
        cache = replace(cache, pos=jnp.int32(len(context)))

        print()
        to_text, reply = text_stream(), []
        for index in range(min(icfg.max_new_tokens, capacity - len(context))):
            next_token = int(token.item())
            if next_token in stop_tokens:
                break
            reply.append(next_token)
            print(to_text(next_token), end="", flush=True)
            token, cache = decode(jnp.asarray([[next_token]], dtype=jnp.int32), cache, jax.random.fold_in(turn_key, index), jnp.int32(0))
        print(to_text(None) + "\n")
        turns = [*turns, *user_turn(prompt), *reply_turns(reply)]
