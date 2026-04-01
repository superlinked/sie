"""Haystack-specific pytest fixtures.

Extends the shared fixtures from integrations/conftest.py with
Haystack-specific helpers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from haystack import Document

# Default test configuration (matches shared conftest)
DEFAULT_EMBEDDING_DIM = 384
ENTITY_PROBABILITY_THRESHOLD = 0.7
DEFAULT_SPARSE_DIM = 30522
DEFAULT_MULTIVECTOR_TOKEN_DIM = 128


def _get_text(item: Any) -> str:
    """Extract text from an item (dict or object with text attribute)."""
    if isinstance(item, dict):
        return item.get("text", str(item))
    if hasattr(item, "text"):
        return item.text
    return str(item)


def _is_single_item(items: Any) -> bool:
    """Check if items is a single item (dict) vs a list of items."""
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
        num_tokens = len(text.split()) + 2
        # Real SDK returns numpy array directly, not nested dict
        result["multivector"] = rng.standard_normal((num_tokens, DEFAULT_MULTIVECTOR_TOKEN_DIM)).astype(np.float32)

    return result


def _create_mock_score_result(query: str, items: list[dict], top_k: int | None = None) -> list[dict[str, Any]]:
    """Create mock score results."""
    rng = np.random.default_rng(hash(query) % (2**32))
    scores = rng.uniform(0, 1, len(items))

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


def _create_mock_extract_result(text: str, labels: list[str]) -> list[dict[str, Any]]:
    """Create mock extract results (NER entities)."""
    rng = np.random.default_rng(hash(text) % (2**32))
    entities = []

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

    return entities


@pytest.fixture
def mock_sie_client() -> MagicMock:
    """Create a mocked SIEClient for unit testing."""
    client = MagicMock()

    def mock_encode(_model: str, items: Any, **kwargs: Any) -> list[dict] | dict:
        # Check output_types to determine what to include
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

    def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> list[dict]:
        query_text = _get_text(query)
        item_dicts = [
            {"id": i.get("id", str(idx)) if isinstance(i, dict) else str(idx), "text": _get_text(i)}
            for idx, i in enumerate(items)
        ]
        return _create_mock_score_result(query_text, item_dicts, kwargs.get("top_k"))

    def mock_extract(_model: str, items: Any, labels: list[str], **_kwargs: Any) -> list[dict]:
        # Extract always returns a list of entities for Haystack
        if isinstance(items, str):
            return _create_mock_extract_result(items, labels)
        if _is_single_item(items):
            return _create_mock_extract_result(_get_text(items), labels)
        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    client.encode = MagicMock(side_effect=mock_encode)
    client.score = MagicMock(side_effect=mock_score)
    client.extract = MagicMock(side_effect=mock_extract)
    client.base_url = "http://localhost:8080"

    return client


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
def test_query() -> str:
    """Sample query for testing reranking."""
    return "What is vector similarity search?"


@pytest.fixture
def test_ner_text() -> str:
    """Sample text for NER extraction testing."""
    return "John Smith works at Apple Inc. in California."


@pytest.fixture
def test_ner_labels() -> list[str]:
    """Sample NER labels for extraction testing."""
    return ["PERSON", "ORGANIZATION", "LOCATION"]


@pytest.fixture
def haystack_documents(test_documents: list[str]) -> list[Document]:
    """Convert test documents to Haystack Document objects."""
    return [Document(content=text) for text in test_documents]


@pytest.fixture
def haystack_documents_with_meta() -> list[Document]:
    """Haystack documents with metadata for testing."""
    return [
        Document(content="Python is a programming language.", meta={"source": "docs", "category": "programming"}),
        Document(content="JavaScript runs in web browsers.", meta={"source": "docs", "category": "programming"}),
        Document(content="Machine learning uses statistical models.", meta={"source": "tutorial", "category": "ml"}),
        Document(content="Vector databases store embeddings.", meta={"source": "blog", "category": "databases"}),
    ]
