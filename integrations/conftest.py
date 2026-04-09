"""Shared pytest fixtures for SIE framework integrations.

This module provides common fixtures for testing all integration packages.
Fixtures are automatically available to all tests under integrations/.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Default test configuration
DEFAULT_EMBEDDING_DIM = 384
# Probability threshold for entity extraction (30% chance of being an entity)
ENTITY_PROBABILITY_THRESHOLD = 0.7
DEFAULT_SPARSE_DIM = 30522
DEFAULT_MULTIVECTOR_TOKEN_DIM = 128


def _get_text(item: Any) -> str:
    """Extract text from an item (dict or object with text attribute).

    For image-only items (no text but has images), returns a deterministic
    placeholder so the mock can still produce consistent embeddings.
    """
    if isinstance(item, dict):
        text = item.get("text")
        if text:
            return text
        images = item.get("images")
        if images:
            return f"<image:{len(images)}>"
        return str(item)
    if hasattr(item, "text"):
        return item.text
    return str(item)


def _is_single_item(items: Any) -> bool:
    """Check if items is a single item (dict) vs a list of items."""
    # TypedDict/dict = single item, list = multiple items
    return isinstance(items, dict) or (hasattr(items, "text") and not isinstance(items, list))


def _create_mock_encode_result(
    text: str,
    *,
    include_dense: bool = True,
    include_sparse: bool = False,
    include_multivector: bool = False,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> dict[str, Any]:
    """Create a mock encode result matching real SDK EncodeResult structure.

    Real SDK returns:
    - dense: np.ndarray (shape [dims])
    - sparse: {"indices": np.ndarray, "values": np.ndarray} (SparseResult TypedDict)
    - multivector: np.ndarray (shape [num_tokens, token_dims])
    """
    result: dict[str, Any] = {}

    if include_dense:
        # Create deterministic embedding based on text hash
        rng = np.random.default_rng(hash(text) % (2**32))
        # Real SDK returns numpy array directly, not nested dict
        result["dense"] = rng.standard_normal(embedding_dim).astype(np.float32)

    if include_sparse:
        rng = np.random.default_rng(hash(text) % (2**32))
        num_nonzero = min(100, DEFAULT_SPARSE_DIM)
        # Real SDK returns SparseResult TypedDict with indices and values only
        result["sparse"] = {
            "indices": np.sort(rng.choice(DEFAULT_SPARSE_DIM, num_nonzero, replace=False)).astype(np.int32),
            "values": rng.uniform(0, 1, num_nonzero).astype(np.float32),
        }

    if include_multivector:
        rng = np.random.default_rng(hash(text) % (2**32))
        num_tokens = len(text.split()) + 2  # Approximate token count
        # Real SDK returns numpy array directly, not nested dict
        result["multivector"] = rng.standard_normal((num_tokens, DEFAULT_MULTIVECTOR_TOKEN_DIM)).astype(np.float32)

    return result


def _create_mock_score_result(query: str, items: list[dict], top_k: int | None = None) -> list[dict[str, Any]]:
    """Create mock score results."""
    rng = np.random.default_rng(hash(query) % (2**32))
    scores = rng.uniform(0, 1, len(items))

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]
    if top_k:
        sorted_indices = sorted_indices[:top_k]

    results = []
    for rank, idx in enumerate(sorted_indices):
        results.append(
            {
                "item_id": items[idx].get("id"),
                "score": float(scores[idx]),
                "rank": rank,
            }
        )
    return results


def _create_mock_extract_result(text: str, labels: list[str]) -> dict[str, Any]:
    """Create mock extract results with all extraction types."""
    # Generate deterministic mock entities
    rng = np.random.default_rng(hash(text) % (2**32))
    entities = []

    # Simple mock: find words and assign random labels
    words = text.split()
    for i, word in enumerate(words):
        if rng.random() > ENTITY_PROBABILITY_THRESHOLD and labels:
            entities.append(
                {
                    "text": word,
                    "label": labels[rng.integers(len(labels))],
                    "score": float(rng.uniform(0.7, 1.0)),
                    "start": sum(len(w) + 1 for w in words[:i]),
                    "end": sum(len(w) + 1 for w in words[:i]) + len(word),
                }
            )

    return {
        "entities": entities,
        "relations": [],
        "classifications": [],
        "objects": [],
    }


@pytest.fixture
def mock_sie_client() -> MagicMock:
    """Create a mocked SIEClient for unit testing.

    Returns:
        MagicMock that behaves like SIEClient with encode/score/extract methods.

    Example:
        def test_embeddings(mock_sie_client):
            embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")
            result = embeddings.embed_query("Hello")
            assert len(result) == 384
    """
    client = MagicMock()

    def mock_encode(_model: str, items: Any, **kwargs: Any) -> list[dict] | dict:
        """Mock encode that returns embeddings for each item."""
        # Determine what to include based on output_types
        output_types = kwargs.get("output_types", ["dense"])
        include_dense = "dense" in output_types
        include_sparse = "sparse" in output_types
        include_multivector = "multivector" in output_types

        # Handle single item vs list
        if _is_single_item(items):
            return _create_mock_encode_result(
                _get_text(items),
                include_dense=include_dense,
                include_sparse=include_sparse,
                include_multivector=include_multivector,
            )

        # List of items
        return [
            _create_mock_encode_result(
                _get_text(item),
                include_dense=include_dense,
                include_sparse=include_sparse,
                include_multivector=include_multivector,
            )
            for item in items
        ]

    def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> list[dict]:
        """Mock score that returns relevance scores."""
        query_text = _get_text(query)
        item_dicts = [
            {"id": i.get("id", str(idx)) if isinstance(i, dict) else str(idx), "text": _get_text(i)}
            for idx, i in enumerate(items)
        ]
        return _create_mock_score_result(query_text, item_dicts, kwargs.get("top_k"))

    def mock_extract(_model: str, items: Any, labels: list[str], **_kwargs: Any) -> list[dict] | dict:
        """Mock extract that returns NER entities."""
        if _is_single_item(items):
            return _create_mock_extract_result(_get_text(items), labels)

        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    client.encode = MagicMock(side_effect=mock_encode)
    client.score = MagicMock(side_effect=mock_score)
    client.extract = MagicMock(side_effect=mock_extract)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def mock_sie_async_client() -> MagicMock:
    """Create a mocked SIEAsyncClient for async unit testing.

    Returns:
        MagicMock that behaves like SIEAsyncClient with async encode/score/extract.
    """
    client = MagicMock()

    async def mock_encode(_model: str, items: Any, **kwargs: Any) -> list[dict] | dict:
        # Determine what to include based on output_types
        output_types = kwargs.get("output_types", ["dense"])
        include_dense = "dense" in output_types
        include_sparse = "sparse" in output_types
        include_multivector = "multivector" in output_types

        if _is_single_item(items):
            return _create_mock_encode_result(
                _get_text(items),
                include_dense=include_dense,
                include_sparse=include_sparse,
                include_multivector=include_multivector,
            )
        return [
            _create_mock_encode_result(
                _get_text(item),
                include_dense=include_dense,
                include_sparse=include_sparse,
                include_multivector=include_multivector,
            )
            for item in items
        ]

    async def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> list[dict]:
        query_text = _get_text(query)
        item_dicts = [
            {"id": i.get("id", str(idx)) if isinstance(i, dict) else str(idx), "text": _get_text(i)}
            for idx, i in enumerate(items)
        ]
        return _create_mock_score_result(query_text, item_dicts, kwargs.get("top_k"))

    async def mock_extract(_model: str, items: Any, labels: list[str], **_kwargs: Any) -> list[dict] | dict:
        if _is_single_item(items):
            return _create_mock_extract_result(_get_text(items), labels)
        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    client.encode = AsyncMock(side_effect=mock_encode)
    client.score = AsyncMock(side_effect=mock_score)
    client.extract = AsyncMock(side_effect=mock_extract)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def sie_server_url() -> str:
    """Get the URL of a running SIE server for integration tests.

    Uses SIE_SERVER_URL environment variable, defaults to localhost:8080.

    Returns:
        Server URL string.

    Note:
        Integration tests using this fixture should be marked with @pytest.mark.integration
    """
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


@pytest.fixture
def embedding_dim() -> int:
    """Default embedding dimension for tests."""
    return DEFAULT_EMBEDDING_DIM


@pytest.fixture
def test_texts() -> list[str]:
    """Sample texts for testing embeddings."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can understand natural language.",
        "Vector databases store embeddings for similarity search.",
        "SIE provides fast inference for embedding models.",
    ]


@pytest.fixture
def test_query() -> str:
    """Sample query for testing reranking."""
    return "What is vector similarity search?"


@pytest.fixture
def test_documents() -> list[str]:
    """Sample documents for testing reranking."""
    return [
        "Vector similarity search finds items with similar embeddings.",
        "The weather today is sunny with clear skies.",
        "Embedding models convert text to dense vectors.",
        "Python is a popular programming language.",
        "Nearest neighbor search uses distance metrics.",
    ]


@pytest.fixture
def test_ner_text() -> str:
    """Sample text for NER extraction testing."""
    return "John Smith works at Apple Inc. in California."


@pytest.fixture
def test_ner_labels() -> list[str]:
    """Sample NER labels for extraction testing."""
    return ["PERSON", "ORGANIZATION", "LOCATION"]


@pytest.fixture
def test_image_paths() -> list[str]:
    """Sample image paths for testing multimodal embeddings.

    These are placeholder paths — the mock client doesn't read files.
    """
    return ["photo_of_cat.jpg", "diagram.png"]


@pytest.fixture
def test_image_bytes() -> list[bytes]:
    """Sample image bytes for testing multimodal embeddings.

    Minimal JPEG-like headers — the mock client doesn't decode images.
    """
    return [b"\xff\xd8\xff\xe0" + b"\x00" * 100, b"\xff\xd8\xff\xe0" + b"\x00" * 200]
