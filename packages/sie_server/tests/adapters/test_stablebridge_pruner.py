from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sie_server.adapters.stablebridge_pruner import (
    DEFAULT_HIGHLIGHT_THRESHOLD,
    DEFAULT_PRUNE_THRESHOLD,
    PruningHead,
    StablebridgePrunerAdapter,
)
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item


class TestPruningHead:
    def test_forward_shape_and_range(self) -> None:
        head = PruningHead(hidden_size=32, intermediate_size=16, dropout=0.0)
        head.eval()
        hidden = torch.randn(3, 7, 32)
        with torch.no_grad():
            out = head(hidden)
        assert out.shape == (3, 7)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_attention_mask_zeros_out_padding(self) -> None:
        head = PruningHead(hidden_size=8, intermediate_size=4, dropout=0.0)
        head.eval()
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        with torch.no_grad():
            out = head(hidden, mask)
        assert float(out[0, 3:].abs().sum()) == 0.0
        assert float(out[0, :3].max()) <= 1.0
        assert float(out[1].sum()) > 0.0

    def test_default_dimensions_match_bge_reranker(self) -> None:
        head = PruningHead()
        first = head.classifier[0]
        last = head.classifier[-1]
        assert isinstance(first, torch.nn.Linear)
        assert isinstance(last, torch.nn.Linear)
        assert first.in_features == 1024
        assert first.out_features == 512
        assert last.in_features == 512
        assert last.out_features == 1


class TestAdapterWiring:
    def test_capabilities_score_and_json(self) -> None:
        adapter = StablebridgePrunerAdapter()
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert set(caps.outputs) == {"score", "json"}

    def test_dims_empty(self) -> None:
        adapter = StablebridgePrunerAdapter()
        d = adapter.dims
        assert d.dense is None
        assert d.sparse is None
        assert d.multivector is None

    def test_get_preprocessor_returns_char_count(self) -> None:
        adapter = StablebridgePrunerAdapter(model_name_or_path="some/model")
        pp = adapter.get_preprocessor()
        assert isinstance(pp, CharCountPreprocessor)

    def test_default_thresholds(self) -> None:
        adapter = StablebridgePrunerAdapter()
        assert adapter._prune_threshold == DEFAULT_PRUNE_THRESHOLD
        assert adapter._highlight_threshold == DEFAULT_HIGHLIGHT_THRESHOLD

    def test_extract_text_raises_when_missing(self) -> None:
        adapter = StablebridgePrunerAdapter()
        with pytest.raises(ValueError, match="requires text"):
            adapter._extract_text(Item(text=None))

    def test_score_raises_when_not_loaded(self) -> None:
        adapter = StablebridgePrunerAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.score(Item(text="q"), [Item(text="d")])

    def test_extract_raises_when_not_loaded(self) -> None:
        adapter = StablebridgePrunerAdapter()
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.extract([Item(text="d")], instruction="q")


def _fake_tokenizer_with_offsets(text: str, offsets: list[tuple[int, int]]) -> Any:
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.sep_token_id = 2
    tok.cls_token_id = 1
    tok.bos_token_id = 1

    def call(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], str):
            return {"offset_mapping": offsets}
        msg = "unexpected tokenizer call shape"
        raise AssertionError(msg)

    tok.side_effect = call
    return tok


class TestTokensToCharSpans:
    def _adapter_with_tokenizer(self, offsets: list[tuple[int, int]]) -> StablebridgePrunerAdapter:
        adapter = StablebridgePrunerAdapter()
        adapter._tokenizer = _fake_tokenizer_with_offsets("ignored", offsets)
        return adapter

    def test_groups_consecutive_tokens_into_one_span(self) -> None:
        offsets = [(0, 5), (5, 6), (6, 11)]
        adapter = self._adapter_with_tokenizer(offsets)
        token_ids = [10, 11, 12]
        probs = np.array([0.9, 0.8, 0.7])
        mask = np.array([True, True, True])

        # All-same-label thresholds so adjacent tokens still merge into one span.
        spans = adapter._tokens_to_char_spans(
            token_ids,
            probs,
            mask,
            "hello world",
            prune_threshold=0.0,
            highlight_threshold=2.0,
        )

        assert len(spans) == 1
        text, start, end, avg, label = spans[0]
        assert (start, end) == (0, 11)
        assert text == "hello world"
        assert pytest.approx(avg, rel=1e-3) == 0.8
        assert label == "kept"

    def test_splits_on_gap(self) -> None:
        offsets = [(0, 5), (10, 15)]
        adapter = self._adapter_with_tokenizer(offsets)
        token_ids = [10, 11]
        probs = np.array([0.9, 0.2])
        mask = np.array([True, True])

        spans = adapter._tokens_to_char_spans(
            token_ids,
            probs,
            mask,
            "hello     world",
            prune_threshold=0.0,
            highlight_threshold=2.0,
        )

        assert len(spans) == 2
        assert spans[0][1:3] == (0, 5)
        assert spans[1][1:3] == (10, 15)

    def test_splits_adjacent_tokens_with_different_labels(self) -> None:
        # Two adjacent tokens (no character gap) should NOT be merged when
        # they fall into different label buckets.
        offsets = [(0, 5), (5, 11)]
        adapter = self._adapter_with_tokenizer(offsets)
        token_ids = [10, 11]
        probs = np.array([0.95, 0.30])  # highlight, pruned under defaults
        mask = np.array([True, True])

        spans = adapter._tokens_to_char_spans(
            token_ids,
            probs,
            mask,
            "hello world",
            prune_threshold=DEFAULT_PRUNE_THRESHOLD,
            highlight_threshold=DEFAULT_HIGHLIGHT_THRESHOLD,
        )

        assert len(spans) == 2
        assert spans[0][1:3] == (0, 5)
        assert spans[0][4] == "highlight"
        assert spans[1][1:3] == (5, 11)
        assert spans[1][4] == "pruned"

    def test_skips_special_tokens_and_padding(self) -> None:
        offsets = [(0, 0), (0, 5), (5, 10)]
        adapter = self._adapter_with_tokenizer(offsets)
        token_ids = [1, 42, 43]
        probs = np.array([0.5, 0.9, 0.5])
        mask = np.array([True, True, False])

        spans = adapter._tokens_to_char_spans(
            token_ids,
            probs,
            mask,
            "hello world",
            prune_threshold=0.0,
            highlight_threshold=2.0,
        )

        assert len(spans) == 1
        assert spans[0][1:3] == (0, 5)

    def test_returns_empty_for_empty_input(self) -> None:
        adapter = StablebridgePrunerAdapter()
        spans = adapter._tokens_to_char_spans(
            [],
            np.array([]),
            np.array([], dtype=bool),
            "x",
            prune_threshold=DEFAULT_PRUNE_THRESHOLD,
            highlight_threshold=DEFAULT_HIGHLIGHT_THRESHOLD,
        )
        assert spans == []


class TestExtractWithMocks:
    def test_extract_emits_summary_and_label_buckets(self) -> None:
        adapter = StablebridgePrunerAdapter()
        adapter._device = "cpu"

        seq_len = 6
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        input_ids = torch.tensor([[1, 100, 2, 2, 200, 201]])

        def tok_fn(*args: Any, **kwargs: Any) -> Any:
            if "padding" in kwargs:

                class _Out(dict):
                    def to(self, _: Any) -> _Out:
                        return self

                out = _Out()
                out["input_ids"] = input_ids
                out["attention_mask"] = attention_mask
                return out
            return {"offset_mapping": [(0, 5), (10, 15)]}

        tokenizer = MagicMock(side_effect=tok_fn)
        tokenizer.pad_token_id = 0
        tokenizer.sep_token_id = 2
        tokenizer.cls_token_id = 1
        tokenizer.bos_token_id = 1
        adapter._tokenizer = tokenizer

        class _FakeOutput:
            def __init__(self, logits: torch.Tensor, hidden: torch.Tensor) -> None:
                self.logits = logits
                self.hidden_states = (hidden,)

        hidden = torch.randn(1, seq_len, 8)
        logits = torch.tensor([[2.0]])

        def model_call(**kwargs: Any) -> _FakeOutput:
            return _FakeOutput(logits, hidden)

        adapter._model = MagicMock(side_effect=model_call)

        def head_call(_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            probs = torch.zeros((1, seq_len))
            probs[0, 4] = 0.95
            probs[0, 5] = 0.30
            return probs * mask.float()

        adapter._pruning_head = MagicMock(side_effect=head_call)

        out = adapter.extract([Item(text="hello     world")], instruction="q")

        assert out.batch_size == 1
        entities = out.entities[0]
        assert entities[0]["label"] == "summary"
        assert len(entities) == 3
        labels = {e["label"] for e in entities[1:]}
        assert labels == {"highlight", "pruned"}

    def test_concurrent_forwards_serialize(self) -> None:
        """extract()'s output_hidden_states=True forward (XLM-RoBERTa, a
        transformers recorder class) must be serialized by _forward_lock so
        concurrent forwards cannot race the recorder patch/restore (#2144/#2204).
        """
        adapter = StablebridgePrunerAdapter()
        adapter._device = "cpu"
        seq_len = 6
        input_ids = torch.tensor([[1, 100, 2, 2, 200, 201]])
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)

        def tok_fn(*args: Any, **kwargs: Any) -> Any:
            if "padding" in kwargs:

                class _Out(dict):
                    def to(self, _: Any) -> _Out:
                        return self

                out = _Out()
                out["input_ids"] = input_ids
                out["attention_mask"] = attention_mask
                return out
            return {"offset_mapping": [(0, 5), (10, 15)]}

        tokenizer = MagicMock(side_effect=tok_fn)
        tokenizer.pad_token_id, tokenizer.sep_token_id = 0, 2
        tokenizer.cls_token_id, tokenizer.bos_token_id = 1, 1
        adapter._tokenizer = tokenizer

        class _FakeOutput:
            def __init__(self, logits: torch.Tensor, hidden: torch.Tensor) -> None:
                self.logits = logits
                self.hidden_states = (hidden,)

        counter_guard = threading.Lock()
        active = 0
        max_active = 0

        def model_call(**kwargs: Any) -> _FakeOutput:
            nonlocal active, max_active
            with counter_guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with counter_guard:
                active -= 1
            return _FakeOutput(torch.tensor([[2.0]]), torch.randn(1, seq_len, 8))

        adapter._model = MagicMock(side_effect=model_call)
        adapter._pruning_head = MagicMock(side_effect=lambda _h, mask: torch.zeros((1, seq_len)) * mask.float())

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: adapter.extract([Item(text="hi")], instruction="q"), range(6)))

        assert max_active == 1
