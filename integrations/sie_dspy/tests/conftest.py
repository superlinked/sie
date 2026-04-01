"""DSPy-specific pytest fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# Default test configuration
EMBEDDING_DIM = 384
ENTITY_PROBABILITY_THRESHOLD = 0.7


def _create_mock_encode_result(
    items: list[dict],
    *,
    is_query: bool = False,
    output_types: list[str] | None = None,
) -> list[dict]:
    """Create mock encode results with deterministic embeddings.

    Real SDK returns {"dense": np.ndarray, "sparse": {...}, ...} - numpy arrays directly.
    """
    output_types = output_types or ["dense"]
    include_dense = "dense" in output_types
    include_sparse = "sparse" in output_types

    results = []
    for idx, item in enumerate(items):
        # Create deterministic embedding based on text hash
        text = item.get("text", str(idx))
        rng = np.random.default_rng(hash(text) % (2**32))

        result = {}

        if include_dense:
            embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            result["dense"] = embedding

        if include_sparse:
            # Generate sparse representation (simulate token IDs and weights)
            num_tokens = rng.integers(5, 20)
            indices = np.sort(rng.choice(30000, size=num_tokens, replace=False))
            values = rng.uniform(0.1, 1.0, size=num_tokens).astype(np.float32)
            result["sparse"] = {
                "indices": indices,
                "values": values,
            }

        results.append(result)

    return results


def _create_mock_score_result(query: str, items: list[dict]) -> list[dict[str, Any]]:
    """Create mock score results."""
    rng = np.random.default_rng(hash(query) % (2**32))
    scores = rng.uniform(0, 1, len(items))

    results = []
    for idx, score in enumerate(scores):
        results.append(
            {
                "item_id": str(idx),
                "score": float(score),
                "rank": idx,
            }
        )
    return results


def _create_mock_extract_result(text: str, labels: list[str]) -> list[dict[str, Any]]:
    """Create mock extract results (NER entities)."""
    rng = np.random.default_rng(hash(text) % (2**32))
    entities = []

    words = text.split()
    char_offset = 0
    for word in words:
        if rng.random() > ENTITY_PROBABILITY_THRESHOLD and labels:
            entities.append(
                {
                    "text": word,
                    "label": labels[rng.integers(len(labels))],
                    "score": float(rng.uniform(0.7, 1.0)),
                    "start": char_offset,
                    "end": char_offset + len(word),
                }
            )
        char_offset += len(word) + 1  # +1 for space

    return entities


def _get_text(item: Any) -> str:
    """Extract text from an item."""
    if isinstance(item, dict):
        return item.get("text", str(item))
    if hasattr(item, "text"):
        return item.text
    return str(item)


@pytest.fixture
def mock_sie_client() -> MagicMock:
    """Create a mocked SIEClient for unit testing."""
    client = MagicMock()

    def mock_encode(
        _model: str,
        items: Any,
        output_types: list[str] | None = None,
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        # Extract is_query from options
        is_query = bool(options.get("is_query", False)) if options else False
        # Handle single item
        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_encode_result(item_dicts, is_query=is_query, output_types=output_types)
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_encode_result(item_dicts, is_query=is_query, output_types=output_types)

    def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> list[dict]:
        query_text = _get_text(query)
        item_dicts = [{"id": str(idx), "text": _get_text(i)} for idx, i in enumerate(items)]
        return _create_mock_score_result(query_text, item_dicts)

    def mock_extract(_model: str, item: Any, labels: list[str], **_kwargs: Any) -> list[dict]:
        return _create_mock_extract_result(_get_text(item), labels)

    client.encode = MagicMock(side_effect=mock_encode)
    client.score = MagicMock(side_effect=mock_score)
    client.extract = MagicMock(side_effect=mock_extract)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def ml_corpus() -> list[str]:
    """Sample corpus for ML/AI domain - typical DSPy use case."""
    return [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "The weather forecast predicts sunny skies for the weekend.",
        "Deep learning uses neural networks with multiple layers to extract features from data.",
        "Python is a popular programming language for data science and machine learning.",
        "Natural language processing allows computers to understand and generate human language.",
        "Stock prices fluctuated today due to market uncertainty.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
    ]


@pytest.fixture
def research_text() -> str:
    """Sample text for entity extraction - research/academic use case."""
    return (
        "Dr. Sarah Chen at Stanford University published a paper on transformer models. "
        "The research was funded by a $2M grant from the National Science Foundation. "
        "Co-authors include researchers from Google DeepMind and MIT."
    )
