"""Shared pytest fixtures for SIE framework integrations.

This module provides common fixtures for testing all integration packages.
Fixtures are automatically available to all tests under integrations/.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import NonCallableMagicMock, create_autospec

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient

# Default test configuration
DEFAULT_EMBEDDING_DIM = 384
# Probability threshold for entity extraction (30% chance of being an entity)
ENTITY_PROBABILITY_THRESHOLD = 0.7
DEFAULT_SPARSE_DIM = 30522
DEFAULT_MULTIVECTOR_TOKEN_DIM = 128
EXTRACT_ERROR_TEXT = "This item fails extraction in the mocked SIE client."
EXTRACT_ITEM_ERROR = {"code": "INPUT_TOO_LONG", "message": "Item exceeds the model input window."}


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


def _server_item_id(item: Any, index: int) -> str:
    """Return the ``item_id`` the SIE server reports for a scored item.

    The server echoes an item's ``id`` and reports ``item-<index>`` for an item
    sent without one.
    """
    item_id = item.get("id") if isinstance(item, dict) else None
    return item_id if item_id is not None else f"item-{index}"


def _create_mock_score_result(query: str, items: list[dict]) -> list[dict[str, Any]]:
    """Create mock score results."""
    rng = np.random.default_rng(hash(query) % (2**32))
    scores = rng.uniform(0, 1, len(items))

    # Sort by score descending
    sorted_indices = np.argsort(scores)[::-1]

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


def _create_mock_extract_result(text: str, labels: list[str] | None) -> dict[str, Any]:
    """Create mock extract results with all extraction types.

    An item whose text is ``EXTRACT_ERROR_TEXT`` gets the per-item ``error``
    the real SDK returns when extraction did not complete for that item.
    """
    if text == EXTRACT_ERROR_TEXT:
        return {
            "entities": [],
            "relations": [],
            "classifications": [],
            "objects": [],
            "error": dict(EXTRACT_ITEM_ERROR),
        }

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
def mock_sie_client() -> NonCallableMagicMock:
    """Create a mocked SIEClient for unit testing.

    The mock is autospecced from ``sie_sdk.SIEClient``: a call with an argument
    the SDK does not accept raises ``TypeError``, and an attribute the SDK does
    not define raises ``AttributeError``. Change a method's behavior through its
    ``side_effect`` or ``return_value``; assigning a new mock to the attribute
    discards the signature check.

    Returns:
        Autospecced SIEClient with deterministic encode/score/extract behavior.

    Example:
        def test_embeddings(mock_sie_client):
            embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")
            result = embeddings.embed_query("Hello")
            assert len(result) == 384
    """
    client = create_autospec(SIEClient, instance=True)

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

    def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> dict[str, Any]:
        """Mock score that returns a ScoreResult envelope.

        Mirrors the real SDK ``SIEClient.score()`` shape: a ``{model, scores,
        ...}`` envelope with ranked entries under ``scores`` (not a bare list).
        """
        query_text = _get_text(query)
        item_dicts = [{"id": _server_item_id(i, idx), "text": _get_text(i)} for idx, i in enumerate(items)]
        return {
            "model": _model,
            "scores": _create_mock_score_result(query_text, item_dicts),
        }

    def mock_extract(_model: str, items: Any, *, labels: list[str] | None = None, **_kwargs: Any) -> list[dict] | dict:
        """Mock extract that returns NER entities."""
        if _is_single_item(items):
            return _create_mock_extract_result(_get_text(items), labels)

        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    client.encode.side_effect = mock_encode
    client.score.side_effect = mock_score
    client.extract.side_effect = mock_extract
    client.base_url = "http://localhost:8080"

    return client


@pytest.fixture
def mock_sie_async_client() -> NonCallableMagicMock:
    """Create a mocked SIEAsyncClient for async unit testing.

    Autospecced from ``sie_sdk.SIEAsyncClient`` with the same contract as
    ``mock_sie_client``.

    Returns:
        Autospecced SIEAsyncClient with async encode/score/extract behavior.
    """
    client = create_autospec(SIEAsyncClient, instance=True)

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

    async def mock_score(_model: str, query: Any, items: list[Any], **kwargs: Any) -> dict[str, Any]:
        query_text = _get_text(query)
        item_dicts = [{"id": _server_item_id(i, idx), "text": _get_text(i)} for idx, i in enumerate(items)]
        return {
            "model": _model,
            "scores": _create_mock_score_result(query_text, item_dicts),
        }

    async def mock_extract(
        _model: str, items: Any, *, labels: list[str] | None = None, **_kwargs: Any
    ) -> list[dict] | dict:
        if _is_single_item(items):
            return _create_mock_extract_result(_get_text(items), labels)
        return [_create_mock_extract_result(_get_text(item), labels) for item in items]

    client.encode.side_effect = mock_encode
    client.score.side_effect = mock_score
    client.extract.side_effect = mock_extract
    client.base_url = "http://localhost:8080"

    return client


def _query_word_overlap(query: str, text: str) -> float:
    """Count the distinct query words that appear in ``text``."""
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    return float(len(query_words & set(re.findall(r"[a-z0-9]+", text.lower()))))


class _ScoreStubServer:
    """Local HTTP server that answers ``POST /v1/score/{model}`` in the SIE server's wire format.

    An item's score is the number of distinct query words it contains. Entries
    are sorted by descending score, ranked from 0, and carry the ``item_id`` the
    SIE server reports (see ``_server_item_id``). Decoded request bodies are
    recorded in ``requests``.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                body = msgpack.unpackb(self.rfile.read(int(self.headers.get("Content-Length") or 0)), raw=False)
                stub.requests.append(body)
                reply = msgpack.packb(
                    {"model": self.path.removeprefix("/v1/score/"), "scores": _stub_score_entries(body)},
                    use_bin_type=True,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/msgpack")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *_args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _stub_score_entries(body: dict[str, Any]) -> list[dict[str, Any]]:
    query = body["query"].get("text") or ""
    scored = sorted(
        (
            (_server_item_id(item, index), _query_word_overlap(query, item.get("text") or ""))
            for index, item in enumerate(body["items"])
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [{"item_id": item_id, "score": score, "rank": rank} for rank, (item_id, score) in enumerate(scored)]


@pytest.fixture
def score_stub_server() -> Iterator[_ScoreStubServer]:
    """A running ``_ScoreStubServer``; point a real ``SIEClient`` at its ``url``."""
    server = _ScoreStubServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def extract_error_text() -> str:
    """Item text for which the shared mock clients return a per-item extract ``error``."""
    return EXTRACT_ERROR_TEXT


@pytest.fixture
def extract_item_error() -> dict[str, str]:
    """The per-item extract ``error`` returned for ``extract_error_text``."""
    return dict(EXTRACT_ITEM_ERROR)


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
