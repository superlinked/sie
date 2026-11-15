"""Qwen2FlashCrossEncoderAdapter's shared prefixes: one encoding of the tokens before the document.

The layout and the cache are checked on the host. The forward is checked on
the CPU against the Hugging Face model itself, with the attention step
replaced by a plain reference of varlen causal attention. On CUDA, flash
attention is checked against that reference, the shared forward against the
unshared one, and a graph over a cached prefix against the eager forward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sie_server.adapters.qwen2_flash_cross_encoder._prefix import (
    PrefixCache,
    PrefixEntry,
    plan_shared_forward,
    prefix_key,
)
from sie_server.adapters.qwen2_flash_cross_encoder.adapter import (
    QWEN3_DOCUMENT_SEPARATOR,
    Qwen2FlashCrossEncoderAdapter,
)
from sie_server.core.loader import load_model_config
from sie_server.types.inputs import Item

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_VOCAB = 96


def _flash_attn_available() -> bool:
    try:
        import flash_attn
    except ImportError:
        return False
    return torch.cuda.is_available()


needs_flash = pytest.mark.skipif(not _flash_attn_available(), reason="needs CUDA and flash-attn")


class _Tokenizer:
    """One token per character, inside the small model's vocabulary."""

    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [3 + ord(char) % (_VOCAB - 3) for char in text]

    def __call__(self, texts: list[str], **_kwargs: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [self.encode(text) for text in texts]}


def _entry(tokens: int, layers: int = 2, fill: float = 0.0) -> PrefixEntry:
    return PrefixEntry(keys=torch.full((layers, tokens, 2, 4), fill), values=torch.full((layers, tokens, 2, 4), fill))


# --------------------------------------------------------------------------- the layout


def test_rows_that_share_a_prefix_attend_to_one_copy_of_it() -> None:
    shared, other = [5, 6, 7], [8, 9]
    plan = plan_shared_forward([(shared, [10, 11]), (other, [12]), (shared, [13, 14, 15])], None)

    assert plan is not None
    # The shared prefix once, then each row: its tail, or the whole row when nothing shares its prefix.
    assert plan.input_ids.tolist() == [5, 6, 7, 10, 11, 8, 9, 12, 13, 14, 15]
    assert plan.positions.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    assert plan.cu_seqlens_q.tolist() == [0, 3, 5, 8, 11]
    assert plan.cu_seqlens_k.tolist() == [0, 3, 8, 11, 17]
    assert plan.k_index is not None
    assert plan.k_index.tolist() == [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 8, 9, 10]
    assert (plan.max_seqlen_q, plan.max_seqlen_k) == (3, 6)
    assert plan.last_index.tolist() == [4, 7, 10]
    assert plan.cached == []
    assert plan.store == []


def test_no_shared_forward_when_no_row_can_share() -> None:
    rows = [([5, 6], [7]), ([8, 9], [10])]
    assert plan_shared_forward(rows, None) is None
    # A first sight is not kept, so there is nothing to share yet.
    assert plan_shared_forward(rows, PrefixCache(2**20), entry_bytes_per_token=64) is None
    # Rows without a prefix (the mxbai format) never share.
    assert plan_shared_forward([([], [5, 6]), ([], [5, 6])], PrefixCache(2**20)) is None


def test_a_prefix_is_kept_on_its_second_sight_and_read_from_the_cache_after() -> None:
    cache = PrefixCache(2**20)
    prefix = [5, 6, 7]
    assert plan_shared_forward([(prefix, [8])], cache, entry_bytes_per_token=64) is None

    second = plan_shared_forward([(prefix, [9, 10])], cache, entry_bytes_per_token=64)
    assert second is not None
    assert second.store == [(prefix_key(prefix), 0, 3)]
    assert second.input_ids.tolist() == [5, 6, 7, 9, 10]
    assert second.cu_seqlens_k.tolist() == [0, 3, 8]
    entry = _entry(3)
    cache.put(prefix_key(prefix), entry)

    third = plan_shared_forward([(prefix, [11, 12])], cache, entry_bytes_per_token=64)
    assert third is not None
    assert third.cached == [entry]
    assert third.store == []
    assert third.input_ids.tolist() == [11, 12]
    assert third.positions.tolist() == [3, 4]
    assert third.cu_seqlens_q.tolist() == [0, 2]
    assert third.cu_seqlens_k.tolist() == [0, 5]
    # One row over one cached prefix attends to the keys in place: no gather.
    assert third.k_index is None
    assert cache.hits == 1


def test_cached_and_computed_prefixes_share_one_forward() -> None:
    cache = PrefixCache(2**20)
    kept, fresh = [5, 6], [7, 8, 9]
    entry = _entry(2)
    cache.put(prefix_key(kept), entry)

    plan = plan_shared_forward([(fresh, [10]), (kept, [11, 12]), (fresh, [13]), (kept, [14])], cache)

    assert plan is not None
    assert plan.cached == [entry]
    assert plan.input_ids.tolist() == [7, 8, 9, 10, 11, 12, 13, 14]
    assert plan.positions.tolist() == [0, 1, 2, 3, 2, 3, 3, 2]
    # Keys: the cached prefix (0, 1), then the packed stream from 2.
    assert plan.k_index is not None
    assert plan.k_index.tolist() == [2, 3, 4, 2, 3, 4, 5, 0, 1, 6, 7, 2, 3, 4, 8, 0, 1, 9]
    assert plan.cu_seqlens_k.tolist() == [0, 3, 7, 11, 15, 18]
    assert plan.last_index.tolist() == [3, 5, 6, 7]


def test_a_prefix_the_cache_cannot_hold_is_not_kept() -> None:
    cache = PrefixCache(max_bytes=100)
    prefix = [5, 6, 7]
    assert plan_shared_forward([(prefix, [8])], cache, entry_bytes_per_token=64) is None
    # Seen twice, but 3 tokens of 64 bytes do not fit in 100: it runs unshared.
    assert plan_shared_forward([(prefix, [8])], cache, entry_bytes_per_token=64) is None
    cache.put(prefix_key(prefix), _entry(3))
    assert len(cache) == 0


def test_the_cache_drops_the_least_recently_used_prefix_past_its_bytes() -> None:
    one = _entry(4).nbytes
    cache = PrefixCache(max_bytes=2 * one)
    first, second, third = _entry(4), _entry(4), _entry(4)
    cache.put(b"a", first)
    cache.put(b"b", second)
    assert cache.get(b"a") is first  # now the most recently used
    cache.put(b"c", third)

    assert cache.get(b"b") is None
    assert cache.get(b"a") is first
    assert cache.get(b"c") is third
    assert cache.held_bytes == 2 * one
    assert cache.evictions == 1
    assert cache.entry_ids() == {id(first), id(third)}
    cache.clear()
    assert len(cache) == 0
    assert cache.held_bytes == 0


def test_the_cache_holds_a_bounded_number_of_prefixes() -> None:
    cache = PrefixCache(max_bytes=2**30, max_entries=2)
    for name in (b"a", b"b", b"c"):
        cache.put(name, _entry(1))
    assert len(cache) == 2
    assert cache.get(b"a") is None


# --------------------------------------------------------------------------- options


def test_shared_prefixes_and_graphs_are_off_unless_a_profile_asks() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused")
    assert adapter._prefix_reuse is False
    assert adapter._cuda_graphs == "off"
    assert adapter._prefix_cache is None


@pytest.mark.parametrize("config", ["Qwen__Qwen3-Reranker-4B.yaml", "Qwen__Qwen3-Reranker-0.6B.yaml"])
def test_the_options_that_move_scores_ship_in_the_fast_profile_only(config: str) -> None:
    model = load_model_config(_MODELS_DIR / config)
    default, fast = model.resolve_profile("default"), model.resolve_profile("fast")

    assert "prefix_reuse" not in default.loadtime
    assert "cuda_graphs" not in default.loadtime
    # The fast profile is the default one plus the options: same adapter, precision and scoring.
    added = {"prefix_reuse": True, "prefix_cache_mb": 512, "cuda_graphs": "bucketed", "cuda_graph_tokens": 512}
    assert dict(fast.loadtime) == {**default.loadtime, **added}
    assert fast.adapter_path == default.adapter_path
    assert fast.compute_precision == default.compute_precision
    assert fast.max_batch_tokens == default.max_batch_tokens
    adapter = Qwen2FlashCrossEncoderAdapter("unused", **fast.loadtime)
    assert adapter._fused_kernels is True
    assert adapter._prefix_reuse is True
    assert adapter._cuda_graphs == "bucketed"
    assert adapter._cuda_graph_tokens == 512


def test_cuda_graphs_takes_off_or_bucketed() -> None:
    assert Qwen2FlashCrossEncoderAdapter("unused", cuda_graphs=False)._cuda_graphs == "off"
    with pytest.raises(ValueError, match="cuda_graphs must be"):
        Qwen2FlashCrossEncoderAdapter("unused", cuda_graphs="always")
    with pytest.raises(ValueError, match="prefix_cache_mb must be"):
        Qwen2FlashCrossEncoderAdapter("unused", prefix_cache_mb=-1)
    for tokens in (0, 63, 4096, True, "128"):
        with pytest.raises(ValueError, match="cuda_graph_tokens must be"):
            Qwen2FlashCrossEncoderAdapter("unused", cuda_graph_tokens=tokens)  # ty: ignore[invalid-argument-type]


def test_graphs_hold_longer_tails_for_a_narrower_model() -> None:
    from sie_server.adapters.qwen2_flash_cross_encoder.adapter import default_graph_tokens

    assert default_graph_tokens(1024) == 512  # Qwen3-Reranker-0.6B
    assert default_graph_tokens(2560) == 128  # Qwen3-Reranker-4B


def test_shared_prefixes_apply_to_the_qwen3_format_only() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", prefix_reuse=True, cuda_graphs="bucketed")
    adapter._resolve_serving_options("cuda")
    assert adapter._prefix_reuse is False
    assert adapter._prefix_cache is None
    assert adapter._prefix_graphs is None


def test_a_zero_cache_shares_inside_a_forward_only() -> None:
    adapter = Qwen2FlashCrossEncoderAdapter("unused", input_format="qwen3", prefix_reuse=True, prefix_cache_mb=0)
    adapter._resolve_serving_options("cuda")
    assert adapter._prefix_reuse is True
    assert adapter._prefix_cache is None


# --------------------------------------------------------------------------- the forward, against the model


def _small_reranker(dtype: torch.dtype = torch.float32, device: str = "cpu") -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
    )
    return Qwen3ForCausalLM(config).to(device, dtype).eval()


def _adapter(model: Any, device: str, **options: Any) -> Qwen2FlashCrossEncoderAdapter:
    """The adapter as ``load`` leaves it, around a small random model."""
    adapter = Qwen2FlashCrossEncoderAdapter("unused", input_format="qwen3", score_mode="log_softmax", **options)
    config = model.config
    adapter._model = model
    adapter._tokenizer = _Tokenizer()  # ty: ignore[invalid-assignment]
    adapter._device = device
    adapter._dtype = model.dtype
    adapter._num_heads = config.num_attention_heads
    adapter._num_kv_heads = config.num_key_value_heads
    adapter._hidden_size = config.hidden_size
    adapter._head_dim = config.head_dim
    adapter._no_token_id, adapter._yes_token_id = 1, 2
    rows = torch.tensor([1, 2], device=device)
    adapter._score_weight = model.lm_head.weight.index_select(0, rows).detach().float()
    adapter._pre_tokenize_templates()
    adapter._resolve_serving_options(device)
    return adapter


def _reference_attention(adapter: Qwen2FlashCrossEncoderAdapter) -> Any:
    """``_prefix_attention`` in plain PyTorch: causal varlen attention, the mask aligned to the bottom right."""

    def build(cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int) -> Any:
        _ = (max_q, max_k)
        scale = 1.0 / (adapter._head_dim**0.5)
        groups = adapter._num_heads // adapter._num_kv_heads

        def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            out = []
            for s in range(len(cu_q) - 1):
                q = query[cu_q[s] : cu_q[s + 1]].float()
                k = key[cu_k[s] : cu_k[s + 1]].float().repeat_interleave(groups, dim=1)
                v = value[cu_k[s] : cu_k[s + 1]].float().repeat_interleave(groups, dim=1)
                nq, nk = q.shape[0], k.shape[0]
                weights = torch.einsum("qhd,khd->hqk", q, k) * scale
                hidden_keys = torch.arange(nk, device=q.device)[None, :] > (
                    torch.arange(nq, device=q.device)[:, None] + nk - nq
                )
                weights = weights.masked_fill(hidden_keys, float("-inf")).softmax(-1)
                out.append(torch.einsum("hqk,khd->qhd", weights, v))
            return torch.cat(out).to(query.dtype)

        return attention

    return build


def _on_cpu(adapter: Qwen2FlashCrossEncoderAdapter, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the adapter without flash-attn: reference attention for shared forwards, the model itself for whole rows."""

    def whole_rows(
        input_ids: torch.Tensor, cu_seqlens: torch.Tensor, _max_seqlen: int, _total: int, batch_size: int, _pos: Any
    ) -> torch.Tensor:
        last = [
            adapter._model.model(input_ids=input_ids[cu_seqlens[i] : cu_seqlens[i + 1]][None]).last_hidden_state[0, -1]
            for i in range(batch_size)
        ]
        return adapter._project_score_logits(torch.stack(last))

    monkeypatch.setattr(adapter, "_prefix_attention", _reference_attention(adapter))
    monkeypatch.setattr(adapter, "_forward_flash", whole_rows)


def _model_scores(adapter: Qwen2FlashCrossEncoderAdapter, query: str, docs: list[str], instruction: str) -> np.ndarray:
    """P(yes) of each pair from the Hugging Face model's own forward, one whole row at a time."""
    scores = []
    for doc in docs:
        ids = adapter._build_input_ids(query, doc, instruction=instruction)
        with torch.inference_mode():
            logits = adapter._model(input_ids=torch.tensor([ids], device=adapter._device)).logits[0, -1].float()
        scores.append(torch.log_softmax(logits[[1, 2]], dim=-1)[1].exp().item())
    return np.asarray(scores, dtype=np.float32)


_DOCS = ["a short one", "a rather longer candidate document", "x", "another candidate with its own words"]


def test_a_shared_forward_scores_as_the_model_does(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(), "cpu", prefix_reuse=True, prefix_cache_mb=0)
    _on_cpu(adapter, monkeypatch)

    query = Item(text="which candidate answers this question")
    output = adapter.score_pairs([query] * len(_DOCS), [Item(text=d) for d in _DOCS], instruction="pick the answer")

    np.testing.assert_allclose(
        output.scores, _model_scores(adapter, query.text or "", _DOCS, "pick the answer"), atol=1e-5
    )
    # The rows are billed whole: sharing the prefix changes no token count.
    whole = [len(adapter._build_input_ids(query.text or "", d, instruction="pick the answer")) for d in _DOCS]
    assert output.input_token_counts == whole


def test_a_cached_prefix_scores_as_the_model_does(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(), "cpu", prefix_reuse=True)
    _on_cpu(adapter, monkeypatch)
    cache = adapter._prefix_cache
    assert cache is not None
    question = Item(text="is this message spam")

    for sight, doc in enumerate(_DOCS):
        output = adapter.score_pairs([question], [Item(text=doc)], instruction="answer yes or no")
        np.testing.assert_allclose(
            output.scores, _model_scores(adapter, question.text or "", [doc], "answer yes or no"), atol=1e-5
        )
        # Unshared on the first sight, kept on the second, read from the cache after.
        assert len(cache) == (0 if sight == 0 else 1)
    assert cache.stores == 1
    assert cache.hits == len(_DOCS) - 2
    entry = next(iter(cache._entries.values()))
    assert entry.keys.shape == (3, entry.tokens, 2, 16)
    assert entry.nbytes == entry.tokens * adapter._prefix_token_bytes()


def test_a_repeated_question_is_tokenized_once(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(), "cpu", prefix_reuse=True)
    _on_cpu(adapter, monkeypatch)
    tokenizer = adapter._tokenizer
    assert tokenizer is not None
    encoded: list[str] = []
    encode = tokenizer.encode
    monkeypatch.setattr(
        tokenizer, "encode", lambda text, **kwargs: encoded.append(text) or encode(text, **kwargs), raising=False
    )
    question = Item(text="is this message spam")

    documents = {QWEN3_DOCUMENT_SEPARATOR + doc for doc in _DOCS}

    def beside_documents() -> list[str]:  # the stub's batched call encodes each document through ``encode``
        return [text for text in encoded if text not in documents]

    first = adapter.score_pairs([question], [Item(text=_DOCS[0])], instruction="answer yes or no")
    # The prefix, and the question and the instruction alone for the token counts.
    assert len(beside_documents()) == 3
    again = adapter.score_pairs([question], [Item(text=_DOCS[1])], instruction="answer yes or no")

    assert len(beside_documents()) == 3
    assert first.input_token_counts is not None
    assert again.input_token_counts is not None
    grew = len(_DOCS[1]) - len(_DOCS[0])
    assert again.input_token_counts[0] - first.input_token_counts[0] == grew
    # Without shared prefixes nothing is remembered between requests.
    plain = _adapter(_small_reranker(), "cpu")
    assert plain._prefix_ids_memo == {}
    assert plain._text_tokens_memo == {}


def test_cached_computed_and_unshared_rows_score_in_one_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(), "cpu", prefix_reuse=True)
    _on_cpu(adapter, monkeypatch)
    kept, fresh, lone = Item(text="a question asked again and again"), Item(text="a new question"), Item(text="once")
    for doc in _DOCS[:2]:  # the second sight caches the first question's prefix
        adapter.score_pairs([kept], [Item(text=doc)], instruction="judge")

    queries = [fresh, kept, lone, fresh, kept]
    docs = [_DOCS[0], _DOCS[1], _DOCS[2], _DOCS[3], _DOCS[0]]
    output = adapter.score_pairs(queries, [Item(text=d) for d in docs], instruction="judge")

    expected = np.concatenate(
        [_model_scores(adapter, q.text or "", [d], "judge") for q, d in zip(queries, docs, strict=True)]
    )
    np.testing.assert_allclose(output.scores, expected, atol=1e-5)


def test_an_out_of_memory_forward_drops_the_cached_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(), "cpu", prefix_reuse=True)
    _on_cpu(adapter, monkeypatch)
    question = Item(text="is this message spam")
    for doc in _DOCS[:2]:
        adapter.score_pairs([question], [Item(text=doc)])
    cache = adapter._prefix_cache
    assert cache is not None
    assert len(cache) == 1

    def out_of_memory(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(adapter, "_run_shared", out_of_memory)
    with pytest.raises(RuntimeError, match="out of memory"):
        adapter.score_pairs([question], [Item(text=_DOCS[2])])
    assert len(cache) == 0


# --------------------------------------------------------------------------- on CUDA


@needs_flash
def test_flash_attention_aligns_the_causal_mask_to_the_bottom_right() -> None:
    torch.manual_seed(0)
    adapter = _adapter(_small_reranker(torch.bfloat16, "cuda"), "cuda", prefix_reuse=True)
    # Three sequences: a prefix alone, and two tails over prefix + tail keys.
    q_lengths, k_lengths = [5, 3, 7], [5, 8, 12]
    cu_q = torch.tensor([0, *np.cumsum(q_lengths)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, *np.cumsum(k_lengths)], dtype=torch.int32, device="cuda")
    query = torch.randn(sum(q_lengths), 4, 16, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(sum(k_lengths), 2, 16, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(sum(k_lengths), 2, 16, device="cuda", dtype=torch.bfloat16)

    flash = adapter._prefix_attention(cu_q, cu_k, max(q_lengths), max(k_lengths))(query, key, value)
    reference = _reference_attention(adapter)(cu_q, cu_k, 0, 0)(query, key, value)

    torch.testing.assert_close(flash.float(), reference.float(), atol=3e-2, rtol=3e-2)


@needs_flash
@pytest.mark.parametrize("fused", [False, True])
def test_shared_and_cached_prefixes_score_as_whole_rows_on_cuda(fused: bool) -> None:
    if fused:
        pytest.importorskip("triton")
    model = _small_reranker(torch.bfloat16, "cuda")
    whole = _adapter(model, "cuda", fused_kernels=fused)
    shared = _adapter(model, "cuda", fused_kernels=fused, prefix_reuse=True)
    query = Item(text="which candidate answers this question")
    docs = [Item(text=d) for d in _DOCS]

    expected = whole.score_pairs([query] * len(docs), docs, instruction="pick the answer").scores
    # Shared inside the forward, kept on the second sight, read from the cache on the third.
    for _ in range(3):
        got = shared.score_pairs([query] * len(docs), docs, instruction="pick the answer").scores
        np.testing.assert_allclose(got, expected, atol=2e-2)
    assert shared._prefix_cache is not None
    assert shared._prefix_cache.hits == 1
    # One row at a time over the cached prefix.
    for doc, want in zip(docs, expected, strict=True):
        got = shared.score_pairs([query], [doc], instruction="pick the answer").scores
        np.testing.assert_allclose(got, [want], atol=2e-2)


@needs_flash
def test_a_graph_over_a_cached_prefix_scores_as_the_eager_forward() -> None:
    pytest.importorskip("triton")
    model = _small_reranker(torch.bfloat16, "cuda")
    eager = _adapter(model, "cuda", fused_kernels=True, prefix_reuse=True)
    graphed = _adapter(
        model, "cuda", fused_kernels=True, prefix_reuse=True, cuda_graphs="bucketed", cuda_graph_tokens=128
    )
    assert graphed._prefix_graphs == {}
    assert graphed._graph_tokens == 128
    question = Item(text="is this message spam")
    # A tail is the space after "<Document>:", the document and the template's closing tokens.
    closing = len(graphed._document_separator_ids) + len(graphed._chat_suffix_ids)
    # Tails of 60 and 64 tokens (one graph), 100 and 128 (another), and 129, which is past the graphs.
    docs = [Item(text="d" * (n - closing)) for n in (60, 100, 64, 128, 129)]

    for adapter in (eager, graphed):  # two sights cache the prefix
        for doc in docs[:2]:
            adapter.score_pairs([question], [doc])
    for doc in docs:
        want = eager.score_pairs([question], [doc]).scores
        got = graphed.score_pairs([question], [doc]).scores
        np.testing.assert_allclose(got, want, atol=2e-2)

    assert graphed._prefix_graphs is not None
    ((entry, runner),) = graphed._prefix_graphs.values()
    assert graphed._prefix_cache is not None
    assert graphed._prefix_cache.entry_ids() == {id(entry)}
    # The first forward of each size records its graph; the next replays it; the long tail ran eagerly.
    assert runner.graph_count == 2
    assert runner.stats.recorded == 2
    assert runner.stats.replayed == 2
    assert runner.stats.eager == {}
    # Dropping the prefix drops its graphs.
    graphed._drop_prefixes()
    assert graphed._prefix_graphs == {}
    assert runner.graph_count == 0
