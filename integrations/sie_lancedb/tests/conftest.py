"""LanceDB-specific pytest fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest

EMBEDDING_DIM = 384
MULTIVECTOR_TOKEN_DIM = 128


def _get_text(item: Any) -> str:
    """Extract text from an item."""
    if isinstance(item, dict):
        return item.get("text", str(item))
    if hasattr(item, "text"):
        return item.text
    return str(item)


def _create_mock_encode_result(
    text: str,
    *,
    include_dense: bool = True,
    include_multivector: bool = False,
) -> dict[str, Any]:
    """Create a mock encode result with deterministic embeddings."""
    result: dict[str, Any] = {}
    rng = np.random.default_rng(hash(text) % (2**32))

    if include_dense:
        embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        result["dense"] = embedding

    if include_multivector:
        num_tokens = len(text.split()) + 2
        result["multivector"] = rng.standard_normal((num_tokens, MULTIVECTOR_TOKEN_DIM)).astype(np.float32)

    return result


def _create_mock_score_result(
    query: str,
    items: list[dict],
) -> list[dict[str, Any]]:
    """Create mock score results with deterministic scores."""
    rng = np.random.default_rng(hash(query) % (2**32))
    scores = rng.uniform(0, 1, len(items))
    sorted_indices = np.argsort(scores)[::-1]

    results = []
    for rank, idx in enumerate(sorted_indices):
        results.append(
            {
                "item_id": idx,
                "score": float(scores[idx]),
                "rank": rank,
            }
        )
    return results


def _create_mock_extract_result(text: str, labels: list[str]) -> dict[str, Any]:
    """Create mock extract results (NER entities)."""
    rng = np.random.default_rng(hash(text) % (2**32))
    entities = []

    words = text.split()
    for i, word in enumerate(words):
        if rng.random() > 0.7 and labels:
            start = sum(len(w) + 1 for w in words[:i])
            entities.append(
                {
                    "text": word,
                    "label": labels[rng.integers(len(labels))],
                    "score": float(rng.uniform(0.7, 1.0)),
                    "start": start,
                    "end": start + len(word),
                    "bbox": None,
                }
            )

    return {"entities": entities}


@pytest.fixture
def mock_sie_client() -> MagicMock:
    """Create a mocked SIEClient for unit testing."""
    client = MagicMock()

    def mock_encode(
        _model: str,
        items: Any,
        output_types: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        output_types = output_types or ["dense"]
        include_dense = "dense" in output_types
        include_multivector = "multivector" in output_types

        if not isinstance(items, list):
            return _create_mock_encode_result(
                _get_text(items),
                include_dense=include_dense,
                include_multivector=include_multivector,
            )

        return [
            _create_mock_encode_result(
                _get_text(item),
                include_dense=include_dense,
                include_multivector=include_multivector,
            )
            for item in items
        ]

    def mock_score(
        _model: str,
        query: Any,
        items: list[Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        query_text = _get_text(query)
        item_dicts = [{"text": _get_text(i)} for i in items]
        return {
            "model": _model,
            "scores": _create_mock_score_result(query_text, item_dicts),
        }

    def mock_extract(
        _model: str,
        items: Any,
        labels: list[str],
        **_kwargs: Any,
    ) -> list[dict] | dict:
        if not isinstance(items, list):
            return _create_mock_extract_result(_get_text(items), labels)
        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    _model_registry = {
        "test-model": {
            "name": "test-model",
            "loaded": True,
            "inputs": ["text"],
            "outputs": ["dense", "multivector"],
            "dims": {"dense": EMBEDDING_DIM, "multivector": MULTIVECTOR_TOKEN_DIM},
            "max_sequence_length": 512,
        },
        "custom/model": {
            "name": "custom/model",
            "loaded": False,
            "inputs": ["text"],
            "outputs": ["dense"],
            "dims": {"dense": EMBEDDING_DIM},
            "max_sequence_length": 8192,
        },
    }

    def mock_get_model(model_name: str) -> dict:
        """Mock get_model returning metadata for a specific model."""
        from sie_sdk import RequestError

        if model_name not in _model_registry:
            raise RequestError(404, f"Model '{model_name}' not found")
        return _model_registry[model_name]

    client.encode = MagicMock(side_effect=mock_encode)
    client.score = MagicMock(side_effect=mock_score)
    client.extract = MagicMock(side_effect=mock_extract)
    client.get_model = MagicMock(side_effect=mock_get_model)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing embeddings."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing analyzes human language.",
    ]


@pytest.fixture
def sample_table() -> pa.Table:
    """Sample Arrow table for testing rerankers and extractors."""
    return pa.table(
        {
            "id": [0, 1, 2, 3, 4],
            "text": [
                "Vector similarity search finds items with similar embeddings.",
                "The weather today is sunny with clear skies.",
                "Embedding models convert text to dense vectors.",
                "Python is a popular programming language.",
                "Nearest neighbor search uses distance metrics.",
            ],
            "_rowid": [0, 1, 2, 3, 4],
        }
    )
