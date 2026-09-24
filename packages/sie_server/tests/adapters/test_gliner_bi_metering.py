"""The GLiNER bi-encoder adapter reports the document tokens it encodes."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import torch
from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.types.inputs import Item


class _BiProcessor:
    def __init__(self) -> None:
        self.tokenized: list[list[list[str]]] = []

    def collate_raw_batch(self, batch: list[dict[str, Any]], *, entity_types: list[str]) -> dict[str, Any]:
        return {"tokens": [item["tokenized_text"][:4] for item in batch]}

    def tokenize_inputs(self, texts: list[list[str]], entities: Any = None) -> dict[str, Any]:
        assert entities is None
        self.tokenized.append(texts)
        longest = max(len(words) for words in texts) + 2
        return {"attention_mask": torch.tensor([[1] * (len(w) + 2) + [0] * (longest - len(w) - 2) for w in texts])}


def test_bi_encoder_reports_document_token_counts() -> None:
    processor = _BiProcessor()
    adapter = GLiNERBiAdapter("test-model", precompute_labels=False)
    model = MagicMock()
    model.data_processor = processor
    model.prepare_inputs.side_effect = lambda texts: ([t.split() for t in texts], [], [])
    model.prepare_base_input.side_effect = lambda words: [{"tokenized_text": w, "ner": None} for w in words]
    model.inference.return_value = [[], []]
    adapter._model = model
    adapter._device = "cpu"

    output = adapter.extract([Item(text="Ada lives in Paris today"), Item(text="Hi")], labels=["person"])

    # Word window of 4 plus CLS/SEP, and 1 word plus CLS/SEP.
    assert output.input_token_counts == [6, 3]


def test_bi_encoder_counts_whitespace_only_documents_as_zero() -> None:
    processor = _BiProcessor()
    adapter = GLiNERBiAdapter("test-model", precompute_labels=False)
    model = MagicMock()
    model.data_processor = processor
    model.prepare_inputs.side_effect = lambda texts: ([t.split() for t in texts], [], [])
    model.prepare_base_input.side_effect = lambda words: [{"tokenized_text": w, "ner": None} for w in words]
    model.inference.return_value = [[], [], []]
    adapter._model = model
    adapter._device = "cpu"

    output = adapter.extract([Item(text="  "), Item(text="Hi"), Item(text="\n")], labels=["person"])

    # GLiNER skips whitespace-only documents, so only "Hi" is encoded.
    assert output.input_token_counts == [0, 3, 0]
    assert processor.tokenized == [[["Hi"]]]


def test_bi_encoder_counts_an_all_whitespace_batch_as_zero() -> None:
    adapter = GLiNERBiAdapter("test-model", precompute_labels=False)
    model = MagicMock()
    model.data_processor = _BiProcessor()
    model.inference.return_value = [[]]
    adapter._model = model
    adapter._device = "cpu"

    output = adapter.extract([Item(text=" ")], labels=["person"])

    assert output.input_token_counts == [0]
    model.prepare_inputs.assert_not_called()
