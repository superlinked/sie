"""Tests for score endpoint."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import msgpack_numpy as m
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api.score import router as score_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    ScoreTask,
    Tasks,
)
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult

# Patch msgpack for numpy support
m.patch()

# Header for JSON responses (msgpack is default)
JSON_HEADERS = {"Accept": "application/json"}


def _mock_score_impl(
    query: Any,
    items: list[Any],
    *,
    instruction: str | None = None,
) -> list[float]:
    """Implementation for mock score.

    Returns decreasing scores based on item index (first item is most relevant).
    """
    return [1.0 - (i * 0.1) for i in range(len(items))]


def _create_mock_worker(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock worker for score tests."""
    worker = MagicMock()

    # Mock submit_score to return a future that resolves to WorkerResult
    async def mock_submit_score(
        prepared_items, query, items, *, instruction=None, options=None, request_id=None, timing=None
    ):
        # Create a real future
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Call the mock adapter's score function to get results
        # This allows tests to override mock_adapter.score.return_value
        scores = mock_adapter.score(query, items, instruction=instruction)

        # Build ScoreOutput
        score_output = ScoreOutput(
            scores=np.array(scores, dtype=np.float32),
            batch_size=len(scores),
        )

        # Create timing if not provided
        result_timing = timing or RequestTiming()

        # Create WorkerResult with typed output
        worker_result = WorkerResult(output=score_output, timing=result_timing)
        future.set_result(worker_result)
        return future

    worker.submit_score = mock_submit_score
    return worker


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter that supports scoring."""
    adapter = MagicMock()
    adapter.score = MagicMock(side_effect=_mock_score_impl)
    adapter.score_pairs = MagicMock(
        side_effect=lambda q, d, **kw: ScoreOutput(scores=np.array(_mock_score_impl(q[0], d), dtype=np.float32))
    )
    adapter.capabilities = ModelCapabilities(
        inputs=["text"],
        outputs=[],  # Cross-encoders don't produce embeddings
    )
    adapter.dims = ModelDims()
    return adapter


@pytest.fixture
def mock_encoder_adapter() -> MagicMock:
    """Create a mock adapter that does NOT support scoring (encoder-only)."""
    adapter = MagicMock()
    adapter.capabilities = ModelCapabilities(
        inputs=["text"],
        outputs=["dense"],
    )
    adapter.dims = ModelDims(dense=1024)
    return adapter


@pytest.fixture
def mock_registry(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock registry with a scoring model."""
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.get.return_value = mock_adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-reranker",
        hf_id="org/test-reranker",
        tasks=Tasks(encode=EncodeTask(), score=ScoreTask()),
        profiles={"default": ProfileConfig(adapter_path="test:TestCrossEncoderAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-reranker"]
    registry.device = "cpu"

    # Mock start_worker to return an async function that returns a mock worker
    mock_worker = _create_mock_worker(mock_adapter)
    registry.start_worker = AsyncMock(return_value=mock_worker)

    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(score_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestScoreEndpoint:
    """Tests for POST /v1/score/{model}."""

    def test_score_basic_json(self, client: TestClient) -> None:
        """Basic score request returns JSON when Accept header set."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "What is machine learning?"},
                "items": [
                    {"text": "Machine learning is a branch of AI."},
                    {"text": "The weather is nice today."},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-reranker"
        assert len(data["scores"]) == 2
        # Scores should be sorted by relevance (descending)
        assert data["scores"][0]["rank"] == 0
        assert data["scores"][1]["rank"] == 1
        # First item should have higher score
        assert data["scores"][0]["score"] >= data["scores"][1]["score"]

    def test_score_basic_msgpack(self, client: TestClient) -> None:
        """Basic score request returns msgpack by default."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "What is ML?"},
                "items": [{"text": "ML is AI."}],
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Deserialize msgpack
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-reranker"
        assert len(data["scores"]) == 1

    def test_score_with_query_id(self, client: TestClient) -> None:
        """Query ID is preserved in response."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"id": "q-123", "text": "Query text"},
                "items": [{"text": "Document text"}],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "q-123"

    def test_score_with_item_ids(self, client: TestClient) -> None:
        """Item IDs are preserved in response."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [
                    {"id": "doc-1", "text": "First document"},
                    {"id": "doc-2", "text": "Second document"},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # IDs should be present in scores
        item_ids = {score["item_id"] for score in data["scores"]}
        assert item_ids == {"doc-1", "doc-2"}

    def test_score_multiple_items(self, client: TestClient) -> None:
        """Can score multiple items at once."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [
                    {"text": "Doc 1"},
                    {"text": "Doc 2"},
                    {"text": "Doc 3"},
                    {"text": "Doc 4"},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) == 4
        # Verify ranks are 0, 1, 2, 3
        ranks = [score["rank"] for score in data["scores"]]
        assert sorted(ranks) == [0, 1, 2, 3]

    def test_score_with_instruction(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """Instruction parameter is passed to adapter."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Doc"}],
                "instruction": "Rank documents by relevance",
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        # Verify instruction was passed
        mock_adapter.score.assert_called_once()
        call_kwargs = mock_adapter.score.call_args
        assert call_kwargs.kwargs["instruction"] == "Rank documents by relevance"

    def test_score_model_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 404 for unknown model."""
        mock_registry.has_model.return_value = False
        response = client.post(
            "/v1/score/unknown-model",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Doc"}],
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "MODEL_NOT_FOUND"

    def test_score_model_load_failure(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 503 MODEL_LOADING when model is not loaded (non-blocking load)."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False
        mock_registry.start_load_async = AsyncMock(return_value=True)
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Doc"}],
            },
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["code"] == "MODEL_LOADING"
        assert "loading" in data["detail"]["message"].lower()
        mock_registry.start_load_async.assert_called_once()

    def test_score_lazy_loads_model(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Model triggers background load on first request if not loaded."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False
        mock_registry.start_load_async = AsyncMock(return_value=True)
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Doc"}],
            },
            headers=JSON_HEADERS,
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        mock_registry.start_load_async.assert_called_once_with("test-reranker", device="cpu")

    def test_score_model_does_not_support_scoring(
        self, client: TestClient, mock_registry: MagicMock, mock_encoder_adapter: MagicMock
    ) -> None:
        """Returns 400 when model doesn't support scoring."""
        # Mock config to return an encoder model (no "score" in outputs)
        mock_registry.get_config.return_value = ModelConfig(
            sie_id="test-encoder",
            hf_id="org/test-encoder",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
            profiles={"default": ProfileConfig(adapter_path="test:TestEncoderAdapter", max_batch_tokens=8192)},
        )
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Doc"}],
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "does not support scoring" in data["detail"]["message"]

    def test_score_empty_items_rejected(self, client: TestClient) -> None:
        """Empty items list is rejected."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [],
            },
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_score_non_dict_items_rejected(self, client: TestClient) -> None:
        """Non-dict items return 400, not 500."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": ["just a string", 123],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `object`, got `str` - at `$.items[0]`"

    def test_score_non_string_text_in_item_rejected(self, client: TestClient) -> None:
        """Item with non-string 'text' returns 400, not 500."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": 123}],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `str | null`, got `int` - at `$.items[0].text`"

    def test_score_non_string_text_in_query_rejected(self, client: TestClient) -> None:
        """Query with non-string 'text' returns 400, not 500."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": 456},
                "items": [{"text": "valid"}],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `str | null`, got `int` - at `$.query.text`"

    def test_score_missing_query_rejected(self, client: TestClient) -> None:
        """Missing query is rejected."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "items": [{"text": "Doc"}],
            },
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_score_generates_item_ids(self, client: TestClient) -> None:
        """Items without IDs get generated IDs."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [
                    {"text": "Doc 1"},
                    {"text": "Doc 2"},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Generated IDs should be "item-0", "item-1", etc.
        item_ids = {score["item_id"] for score in data["scores"]}
        assert "item-0" in item_ids
        assert "item-1" in item_ids


class TestMsgpackScoreRequests:
    """Tests for msgpack request body handling for score endpoint."""

    def test_msgpack_request_basic(self, client: TestClient) -> None:
        """Msgpack request body is parsed correctly."""
        request_data = {
            "query": {"text": "What is ML?"},
            "items": [{"text": "ML is AI."}],
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/score/test-reranker",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200
        # Response is also msgpack by default
        assert response.headers["content-type"] == "application/msgpack"
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-reranker"
        assert len(data["scores"]) == 1

    def test_msgpack_request_with_json_response(self, client: TestClient) -> None:
        """Msgpack request can get JSON response with Accept header."""
        request_data = {
            "query": {"text": "Query"},
            "items": [{"text": "Doc"}],
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/score/test-reranker",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["model"] == "test-reranker"

    def test_msgpack_request_with_instruction(self, client: TestClient) -> None:
        """Msgpack request with instruction is parsed correctly."""
        request_data = {
            "query": {"id": "q-1", "text": "Query"},
            "items": [{"id": "doc-1", "text": "Doc"}],
            "instruction": "Rank by relevance",
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/score/test-reranker",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "q-1"
        assert data["scores"][0]["item_id"] == "doc-1"

    def test_msgpack_request_invalid_body(self, client: TestClient) -> None:
        """Invalid msgpack body returns 400."""
        response = client.post(
            "/v1/score/test-reranker",
            content=b"not valid msgpack",
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400

    def test_msgpack_request_validation_error(self, client: TestClient) -> None:
        """Msgpack request with invalid schema returns 422."""
        request_data = {"items": []}  # Missing query, empty items
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/score/test-reranker",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_x_msgpack_content_type(self, client: TestClient) -> None:
        """Alternative x-msgpack content type is also accepted."""
        request_data = {
            "query": {"text": "Query"},
            "items": [{"text": "Doc"}],
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/score/test-reranker",
            content=msgpack_body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200


class TestScoreResponseOrdering:
    """Tests for score response ordering and ranking."""

    def test_scores_sorted_by_relevance_descending(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """Scores are sorted by relevance (descending)."""
        # Clear side_effect and set specific return value
        mock_adapter.score.side_effect = None
        mock_adapter.score.return_value = [0.9, 0.1, 0.5, 0.3]

        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [
                    {"id": "doc-0", "text": "Doc 0"},
                    {"id": "doc-1", "text": "Doc 1"},
                    {"id": "doc-2", "text": "Doc 2"},
                    {"id": "doc-3", "text": "Doc 3"},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        # Verify ordering: doc-0 (0.9), doc-2 (0.5), doc-3 (0.3), doc-1 (0.1)
        assert data["scores"][0]["item_id"] == "doc-0"
        assert abs(data["scores"][0]["score"] - 0.9) < 1e-6
        assert data["scores"][0]["rank"] == 0

        assert data["scores"][1]["item_id"] == "doc-2"
        assert abs(data["scores"][1]["score"] - 0.5) < 1e-6
        assert data["scores"][1]["rank"] == 1

        assert data["scores"][2]["item_id"] == "doc-3"
        assert abs(data["scores"][2]["score"] - 0.3) < 1e-6
        assert data["scores"][2]["rank"] == 2

        assert data["scores"][3]["item_id"] == "doc-1"
        assert abs(data["scores"][3]["score"] - 0.1) < 1e-6
        assert data["scores"][3]["rank"] == 3

    def test_scores_include_all_items(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """All items are included in scores, even with same score."""
        mock_adapter.score.return_value = [0.5, 0.5, 0.5]

        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [
                    {"id": "a", "text": "Doc A"},
                    {"id": "b", "text": "Doc B"},
                    {"id": "c", "text": "Doc C"},
                ],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["scores"]) == 3
        item_ids = {score["item_id"] for score in data["scores"]}
        assert item_ids == {"a", "b", "c"}


class TestScoreProfileResolution:
    """Tests for profile resolution error handling in score endpoint."""

    def test_invalid_profile_returns_400(self, client: TestClient) -> None:
        """Invalid profile name returns 400, not 500.

        Regression test: profile resolution was previously inside the inference
        try/except block, so ValueError → HTTPException(400) was caught by
        the outer except Exception → HTTPException(500).
        """
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Document"}],
                "options": {"profile": "nonexistent_profile"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "nonexistent_profile" in data["detail"]["message"]

    def test_no_profile_works(self, client: TestClient) -> None:
        """Request without profile still works normally."""
        response = client.post(
            "/v1/score/test-reranker",
            json={
                "query": {"text": "Query"},
                "items": [{"text": "Document"}],
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
