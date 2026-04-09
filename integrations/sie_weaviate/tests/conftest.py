"""Weaviate-specific pytest fixtures.

Note: The mock SIEClient and embedding generation logic follows the same
pattern as sie_chroma/tests/conftest.py.  If more integrations are added,
consider extracting shared fixtures into a common test utilities package.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

EMBEDDING_DIM = 384
MULTIVECTOR_TOKENS = 8
MULTIVECTOR_DIM = 128


def _create_mock_encode_result(
    items: list[dict],
    *,
    include_dense: bool = True,
    include_multivector: bool = False,
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

        if include_multivector:
            mv = rng.standard_normal((MULTIVECTOR_TOKENS, MULTIVECTOR_DIM)).astype(np.float32)
            result["multivector"] = mv

        results.append(result)

    return results


def _create_mock_extract_result(
    items: list[dict],
    *,
    labels: list[str] | None = None,
) -> list[dict]:
    """Create mock extract results with deterministic entities."""
    labels = labels or ["person", "organization", "location"]
    results = []
    for idx, _item in enumerate(items):
        entities = []
        # Generate one entity per label for testing
        for i, label in enumerate(labels):
            entities.append(
                {
                    "text": f"{label}_{idx}",
                    "label": label,
                    "score": 0.95 - i * 0.1,
                    "start": i * 10,
                    "end": i * 10 + 5,
                }
            )
        results.append({"entities": entities})
    return results


def _create_mock_classify_result(
    items: list[dict],
    *,
    labels: list[str] | None = None,
) -> list[dict]:
    """Create mock classification results."""
    labels = labels or ["technical", "business", "legal"]
    results = []
    for _idx, _item in enumerate(items):
        classifications = [{"label": label, "score": 0.9 - i * 0.2} for i, label in enumerate(labels)]
        results.append({"classifications": classifications})
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
        include_multivector = "multivector" in output_types

        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_encode_result(
                item_dicts,
                include_dense=include_dense,
                include_multivector=include_multivector,
            )
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_encode_result(
            item_dicts,
            include_dense=include_dense,
            include_multivector=include_multivector,
        )

    def mock_extract(
        _model: str,
        items: Any,
        *,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            # Distinguish NER vs classification by model name heuristic,
            # but for mocks just check if labels look like classifications
            results = _create_mock_extract_result(item_dicts, labels=labels)
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_extract_result(item_dicts, labels=labels)

    def mock_extract_classify(
        _model: str,
        items: Any,
        *,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_classify_result(item_dicts, labels=labels)
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_classify_result(item_dicts, labels=labels)

    client.encode = MagicMock(side_effect=mock_encode)
    # Default extract returns entities; tests that need classification
    # swap the side_effect on the second call.
    client.extract = MagicMock(side_effect=mock_extract)
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def mock_classify_client(mock_sie_client: MagicMock) -> MagicMock:
    """Mock client where extract returns classifications (for GLiClass)."""
    original_extract = mock_sie_client.extract.side_effect

    call_count = 0

    def extract_dispatch(_model: str, items: Any, *, labels: list[str] | None = None, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        # First call = NER, second call = classification
        if call_count % 2 == 0:
            if not isinstance(items, list):
                items = [items]
                item_dicts = [{"text": _get_text(items[0])}]
                return _create_mock_classify_result(item_dicts, labels=labels)[0]
            item_dicts = [{"text": _get_text(i)} for i in items]
            return _create_mock_classify_result(item_dicts, labels=labels)
        return original_extract(_model, items, labels=labels, **kwargs)

    mock_sie_client.extract = MagicMock(side_effect=extract_dispatch)
    return mock_sie_client


@pytest.fixture
def mock_sie_async_client() -> AsyncMock:
    """Create a mocked SIEAsyncClient for async unit testing."""
    client = AsyncMock()

    async def mock_encode(
        _model: str,
        items: Any,
        output_types: list[str] | None = None,
        *,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        output_types = output_types or ["dense"]
        include_dense = "dense" in output_types
        include_multivector = "multivector" in output_types

        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_encode_result(
                item_dicts,
                include_dense=include_dense,
                include_multivector=include_multivector,
            )
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_encode_result(
            item_dicts,
            include_dense=include_dense,
            include_multivector=include_multivector,
        )

    async def mock_extract(
        _model: str,
        items: Any,
        *,
        labels: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict] | dict:
        if not isinstance(items, list):
            items = [items]
            item_dicts = [{"text": _get_text(items[0])}]
            results = _create_mock_extract_result(item_dicts, labels=labels)
            return results[0]

        item_dicts = [{"text": _get_text(i)} for i in items]
        return _create_mock_extract_result(item_dicts, labels=labels)

    client.encode = AsyncMock(side_effect=mock_encode)
    client.extract = AsyncMock(side_effect=mock_extract)
    client.close = AsyncMock()
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def mock_async_classify_client(mock_sie_async_client: AsyncMock) -> AsyncMock:
    """Async mock client where extract alternates NER and classification."""
    original_extract = mock_sie_async_client.extract.side_effect

    call_count = 0

    async def extract_dispatch(_model: str, items: Any, *, labels: list[str] | None = None, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            if not isinstance(items, list):
                items = [items]
                item_dicts = [{"text": _get_text(items[0])}]
                return _create_mock_classify_result(item_dicts, labels=labels)[0]
            item_dicts = [{"text": _get_text(i)} for i in items]
            return _create_mock_classify_result(item_dicts, labels=labels)
        return await original_extract(_model, items, labels=labels, **kwargs)

    mock_sie_async_client.extract = AsyncMock(side_effect=extract_dispatch)
    return mock_sie_async_client


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
