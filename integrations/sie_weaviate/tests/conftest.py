"""Weaviate-specific pytest fixtures.

Note: The mock SIEClient and embedding generation logic follows the same
pattern as sie_chroma/tests/conftest.py.  If more integrations are added,
consider extracting shared fixtures into a common test utilities package.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

EMBEDDING_DIM = 384
SPARSE_DIM = 30522  # BERT vocab size


def _create_mock_encode_result(
    items: list[dict],
    *,
    include_dense: bool = True,
    include_sparse: bool = False,
) -> list[dict]:
    """Create mock encode results with deterministic embeddings."""
    results = []
    for idx, item in enumerate(items):
        text = item.get("text", str(idx))
        rng = np.random.default_rng(hash(text) % (2**32))

        result: dict[str, Any] = {}

        if include_dense:
            embedding = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            result["dense"] = embedding

        if include_sparse:
            num_nonzero = min(100, SPARSE_DIM)
            result["sparse"] = {
                "indices": np.sort(rng.choice(SPARSE_DIM, num_nonzero, replace=False)).astype(np.int32),
                "values": rng.uniform(0, 1, num_nonzero).astype(np.float32),
            }

        results.append(result)

    return results


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
        output_types = output_types or ["dense"]
        include_dense = "dense" in output_types
        include_sparse = "sparse" in output_types

        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_encode_result(item_dicts, include_dense=include_dense, include_sparse=include_sparse)
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_encode_result(item_dicts, include_dense=include_dense, include_sparse=include_sparse)

    client.encode = MagicMock(side_effect=mock_encode)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def sample_documents() -> list[str]:
    """Sample documents for vector store testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing analyzes human language.",
        "Computer vision enables machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
    ]
