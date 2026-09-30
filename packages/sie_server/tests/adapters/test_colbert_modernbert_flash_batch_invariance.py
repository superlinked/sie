"""Late-interaction token vectors of the ModernBERT flash adapter, alone and batched.

The adapter packs a request's items into one unpadded token stream and keeps
them apart with ``cu_seqlens``: each item attends only to its own tokens,
within the sliding window on local layers, at positions that restart per
item. In float32 an item's token vectors therefore do not depend on which
other items share its request, and they equal the Hugging Face
``ModernBertModel`` forward of that item through the same head. These CPU
tests check both on tiny models laid out like the served checkpoints, with a
small reference standing in for flash-attn's varlen kernel.

In half precision the vectors do move with the batch: a different number of
packed rows makes cuBLAS pick a different matrix-multiply kernel, which
rounds differently. That is rounding, not the layout these tests pin down.
"""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import torch
from sie_server.adapters._pylate_dense import apply_dense_chain
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from sie_server.types.inputs import Item
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.nn import functional
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

_HIDDEN = 32
_TOKEN_DIM = 8
_MAX_LENGTH = 64
_WORDS = ["the", "a", "patient", "drug", "trial", "dose", "protein", "cell", "gene", "effect", "risk", "study"]


def _text(i: int) -> str:
    clause = " ".join((_WORDS[i % len(_WORDS)], *_WORDS[: 1 + (7 * i) % len(_WORDS)])) + " ,"
    return " ".join([clause] * (1 + i // 3)) + " ."


# From two words to past the window: rows of different lengths pack against
# each other, span several sliding windows, and the longest are truncated.
_TEXTS = [_text(i) for i in range(14)]
# Layouts of the served checkpoints, scaled down: which layers attend
# globally, the sliding window, and the RoPE bases of global and local layers.
_LAYOUTS = {
    # GTE-ModernColBERT-v1, Iso-ModernColBERT, Reason-ModernColBERT
    "modernbert-base": {"global_attn_every_n_layers": 3, "local_rope_theta": 10000.0},
    # mxbai-edge-colbert-v0-32m (Ettin), mLateOn (mmBERT)
    "ettin-mmbert": {"global_attn_every_n_layers": 3, "local_rope_theta": 160000.0},
    "all-global": {"global_attn_every_n_layers": 1, "local_rope_theta": 10000.0},
}


def _reference_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    causal: bool = False,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """``flash_attn_varlen_func``: attention within each sequence and, if given, its window."""
    assert not causal
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    bounds = cu_seqlens_q.tolist()
    assert max(end - start for start, end in pairwise(bounds)) == max_seqlen_q == max_seqlen_k
    out = torch.full_like(query, float("nan"))
    for start, end in pairwise(bounds):
        rows = [t[start:end].transpose(0, 1) for t in (query, key, value)]
        mask = None
        if window_size != (-1, -1):
            positions = torch.arange(end - start)
            mask = (positions[:, None] - positions[None, :]).abs() <= window_size[0]
        attended = functional.scaled_dot_product_attention(*rows, attn_mask=mask, scale=softmax_scale)
        out[start:end] = attended.transpose(0, 1)
    return out


@pytest.fixture(autouse=True)
def reference_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("flash_attn")
    module.flash_attn_varlen_func = _reference_varlen  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "flash_attn", module)


def _tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[Q] ", "[D] "]
    vocab = {token: index for index, token in enumerate([*specials, *_WORDS, ".", ","])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106 -- a vocabulary entry, not a secret
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[MASK]",  # noqa: S106 -- the served checkpoints pad with [MASK]
        unk_token="[UNK]",  # noqa: S106
        cls_token="[CLS]",  # noqa: S106
        sep_token="[SEP]",  # noqa: S106
        model_max_length=_MAX_LENGTH,
    )
    tokenizer.add_tokens(["[Q] ", "[D] "])
    return tokenizer


def _model(layout: str) -> ModernBertModel:
    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=32,
        hidden_size=_HIDDEN,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        local_attention=8,
        global_rope_theta=160000.0,
        max_position_embeddings=_MAX_LENGTH,
        pad_token_id=4,
        reference_compile=False,
        attn_implementation="eager",
        **_LAYOUTS[layout],
    )
    model = ModernBertModel(config).eval()
    # Larger attention logits, so a wrong position, window, RoPE base or
    # neighbouring item changes the outputs visibly.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "Wqkv" in name:
                parameter.mul_(8.0)
    return model


def _adapter(layout: str) -> ColBERTModernBERTFlashAdapter:
    adapter = ColBERTModernBERTFlashAdapter(
        "tiny",
        token_dim=_TOKEN_DIM,
        compute_precision="float32",
        max_seq_length=_MAX_LENGTH,
        query_max_length=16,
        query_prefix="[Q] ",
        doc_prefix="[D] ",
        skip_special_tokens=False,
        doc_punctuation_skiplist=True,
    )
    torch.manual_seed(1)
    adapter._model, adapter._tokenizer, adapter._device = _model(layout), _tokenizer(), "cpu"
    adapter._dense_chain = [torch.randn(16, _HIDDEN), torch.randn(_TOKEN_DIM, 16)]
    adapter._doc_skiplist_ids = {adapter._tokenizer.convert_tokens_to_ids(t) for t in (".", ",")}
    return adapter


def _encode(adapter: ColBERTModernBERTFlashAdapter, texts: list[str], *, is_query: bool) -> list[np.ndarray]:
    items = [Item(text=text) for text in texts]
    return adapter.encode(items, ["multivector"], is_query=is_query).multivector


def _reference(adapter: ColBERTModernBERTFlashAdapter, texts: list[str], *, is_query: bool) -> list[np.ndarray]:
    """The Hugging Face forward of a right-padded batch, through the adapter's head and token filter."""
    tokenizer = adapter._tokenizer
    assert tokenizer is not None
    prefixed = adapter._extract_texts([Item(text=text) for text in texts], None, is_query=is_query)
    batch = tokenizer(
        prefixed,
        max_length=adapter._query_max_length if is_query else adapter._max_seq_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        hidden = adapter._model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        vectors = functional.normalize(apply_dense_chain(hidden, adapter._dense_chain)[..., :_TOKEN_DIM], dim=-1)
    skip = set() if is_query else adapter._doc_skiplist_ids
    out = []
    for row, ids, mask in zip(vectors, batch["input_ids"], batch["attention_mask"], strict=True):
        keep = mask.bool() & torch.tensor([int(i) not in skip for i in ids])
        out.append(row[keep].numpy())
    return out


def _assert_same(got: list[np.ndarray], want: list[np.ndarray], *, atol: float) -> None:
    assert [g.shape for g in got] == [w.shape for w in want]
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, atol=atol, rtol=0)


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("is_query", [False, True])
def test_token_vectors_do_not_depend_on_the_other_items_in_a_request(layout: str, is_query: bool) -> None:
    adapter = _adapter(layout)
    alone = [_encode(adapter, [text], is_query=is_query)[0] for text in _TEXTS]
    for size in (2, 4, 8):
        batched: list[Any] = [None] * len(_TEXTS)
        # Interleave long and short rows, so each request mixes lengths.
        order = [i for pair in zip(range(7), range(13, 6, -1), strict=True) for i in pair]
        for start in range(0, len(order), size):
            request = order[start : start + size]
            for i, vectors in zip(
                request, _encode(adapter, [_TEXTS[i] for i in request], is_query=is_query), strict=True
            ):
                batched[i] = vectors
        _assert_same(batched, alone, atol=1e-5)


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("is_query", [False, True])
def test_token_vectors_match_the_hugging_face_forward(layout: str, is_query: bool) -> None:
    adapter = _adapter(layout)
    want = _reference(adapter, _TEXTS, is_query=is_query)
    assert any(len(w) > 8 for w in want)  # rows span more than one sliding window
    _assert_same(_encode(adapter, _TEXTS, is_query=is_query), want, atol=1e-5)
    _assert_same([_encode(adapter, [text], is_query=is_query)[0] for text in _TEXTS], want, atol=1e-5)
