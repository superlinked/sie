"""The GLiNER bi-encoder label-embedding cache must keep labels and embeddings aligned."""

from __future__ import annotations

from typing import Any

import pytest
from sie_server.adapters._prompt_limit import MAX_LABEL_CHARS
from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.types.inputs import InvalidInputError, Item


class _FakeBiEncoder:
    def __init__(self) -> None:
        self.encoded: list[list[str]] = []
        self.predicted: list[tuple[list[str], list[str]]] = []

    def encode_labels(self, labels: list[str], batch_size: int = 8) -> list[str]:
        self.encoded.append(list(labels))
        return [f"embedding of {label}" for label in labels]

    def batch_predict_with_embeds(
        self, texts: list[str], embeds: list[str], labels: list[str], **_: Any
    ) -> list[list[dict[str, Any]]]:
        self.predicted.append((list(embeds), list(labels)))
        return [[] for _ in texts]


def test_reordered_labels_get_their_own_embeddings() -> None:
    adapter = GLiNERBiAdapter("test-model", precompute_labels=True)
    model = _FakeBiEncoder()
    adapter._model = model
    adapter._device = "cpu"

    for labels in (["person", "city"], ["city", "person"], ["person", "city"]):
        adapter.extract([Item(text="Ada lives in Paris")], labels=labels)

    for embeds, labels in model.predicted:
        assert embeds == [f"embedding of {label}" for label in labels]
    # The third request repeats the first label order and reuses its cached embeddings.
    assert model.encoded == [["person", "city"], ["city", "person"]]


def test_bi_encoder_rejects_a_label_over_128_characters() -> None:
    adapter = GLiNERBiAdapter("test-model", precompute_labels=True)
    model = _FakeBiEncoder()
    adapter._model = model
    adapter._device = "cpu"
    item = Item(text="Ada lives in Paris")

    with pytest.raises(
        InvalidInputError, match=f"GLiNER bi-encoder labels may have at most {MAX_LABEL_CHARS} characters"
    ):
        adapter.extract([item], labels=["x" * (MAX_LABEL_CHARS + 1)])
    assert model.encoded == []
    assert model.predicted == []

    accepted = "y" * MAX_LABEL_CHARS
    output = adapter.extract([item], labels=[accepted])
    assert output.errors is None
    assert model.encoded == [[accepted]]
    assert model.predicted[0][1] == [accepted]
