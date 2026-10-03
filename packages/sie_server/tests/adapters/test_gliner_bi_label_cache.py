"""The GLiNER bi-encoder label-embedding cache must keep labels and embeddings aligned."""

from __future__ import annotations

from typing import Any

from sie_server.adapters.gliner_bi import GLiNERBiAdapter
from sie_server.types.inputs import Item


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
