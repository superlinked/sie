"""Tests for extract endpoint."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import msgpack_numpy as m
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api.extract import router as extract_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult
from sie_server.types.responses import Entity

# Patch msgpack for numpy support
m.patch()

# Header for JSON responses (msgpack is default)
JSON_HEADERS = {"Accept": "application/json"}


def _mock_extract_impl(
    items: list[Any],
    *,
    labels: list[str] | None = None,
    output_schema: dict[str, Any] | None = None,
    instruction: str | None = None,
    options: dict[str, Any] | None = None,
) -> ExtractOutput:
    """Implementation for mock extract.

    Returns ExtractOutput with mock entities based on item text and provided labels.
    """
    all_entities: list[list[Entity]] = []
    for _item in items:
        entities: list[Entity] = []
        # Simulate finding entities for each label
        if labels:
            for i, label in enumerate(labels):
                # Create a mock entity if the label is mentioned in text
                entities.append(
                    Entity(
                        text=f"Mock {label}",
                        label=label,
                        score=0.9 - (i * 0.1),
                        start=0,
                        end=len(f"Mock {label}"),
                    )
                )
        all_entities.append(entities)
    return ExtractOutput(entities=all_entities)


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter that supports extraction."""
    adapter = MagicMock()
    adapter.extract = MagicMock(side_effect=_mock_extract_impl)
    adapter.capabilities = ModelCapabilities(
        inputs=["text"],
        outputs=[],  # Extractors don't produce embeddings
    )
    adapter.dims = ModelDims()
    return adapter


@pytest.fixture
def mock_encoder_adapter() -> MagicMock:
    """Create a mock adapter that does NOT support extraction (encoder-only)."""
    adapter = MagicMock()
    adapter.capabilities = ModelCapabilities(
        inputs=["text"],
        outputs=["dense"],
    )
    adapter.dims = ModelDims(dense=1024)
    return adapter


def _create_mock_worker(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock worker that uses the mock adapter."""
    worker = MagicMock()

    # Mock submit_extract to return a future that resolves to WorkerResult
    async def mock_submit_extract(
        prepared_items, items, *, labels=None, output_schema=None, instruction=None, options=None, timing=None
    ):
        # Create a real future
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Call the adapter to get ExtractOutput
        extract_output = mock_adapter.extract(
            items, labels=labels, output_schema=output_schema, instruction=instruction, options=options
        )

        # Create timing if not provided
        request_timing = timing or RequestTiming()
        # End timing phases that would normally be set
        if request_timing._queue_start is not None and request_timing._queue_end is None:
            request_timing._queue_end = request_timing._queue_start
        if request_timing._inference_start is None:
            request_timing._inference_start = request_timing._queue_start or 0
            request_timing._inference_end = request_timing._inference_start

        # Set the result with typed output
        worker_result = WorkerResult(output=extract_output, timing=request_timing)
        future.set_result(worker_result)
        return future

    worker.submit_extract = mock_submit_extract
    return worker


@pytest.fixture
def mock_registry(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock registry with an extraction model."""
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.get.return_value = mock_adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-extractor",
        hf_id="org/test-extractor",
        tasks=Tasks(encode=EncodeTask(), extract=ExtractTask()),
        profiles={"default": ProfileConfig(adapter_path="test:TestGLiNERAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-extractor"]
    registry.device = "cpu"

    # Mock start_worker to return an async function that returns a mock worker
    mock_worker = _create_mock_worker(mock_adapter)
    registry.start_worker = AsyncMock(return_value=mock_worker)

    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(extract_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestExtractEndpoint:
    """Tests for POST /v1/extract/{model}."""

    def test_extract_basic_json(self, client: TestClient) -> None:
        """Basic extract request returns JSON when Accept header set."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc. was founded by Steve Jobs."}],
                "params": {"labels": ["person", "organization"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-extractor"
        assert len(data["items"]) == 1
        assert "entities" in data["items"][0]
        assert "data" in data["items"][0]

    def test_extract_basic_msgpack(self, client: TestClient) -> None:
        """Basic extract request returns msgpack by default."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Hello world"}],
                "params": {"labels": ["greeting"]},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Deserialize msgpack
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-extractor"
        assert len(data["items"]) == 1

    def test_extract_with_item_id(self, client: TestClient) -> None:
        """Item ID is preserved in response."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"id": "doc-123", "text": "Some text"}],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-123"

    def test_extract_generates_item_ids(self, client: TestClient) -> None:
        """Items without IDs get generated IDs."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [
                    {"text": "First doc"},
                    {"text": "Second doc"},
                ],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Generated IDs should be "item-0", "item-1", etc.
        assert data["items"][0]["id"] == "item-0"
        assert data["items"][1]["id"] == "item-1"

    def test_extract_multiple_items(self, client: TestClient) -> None:
        """Can extract from multiple items at once."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [
                    {"text": "Doc 1"},
                    {"text": "Doc 2"},
                    {"text": "Doc 3"},
                ],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3

    def test_extract_with_labels(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """Labels parameter is passed to adapter."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["person", "organization", "location"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        # Verify labels were passed
        mock_adapter.extract.assert_called_once()
        call_kwargs = mock_adapter.extract.call_args
        assert call_kwargs.kwargs["labels"] == ["person", "organization", "location"]

    def test_extract_model_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 404 for unknown model."""
        mock_registry.has_model.return_value = False
        response = client.post(
            "/v1/extract/unknown-model",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["entity"]},
            },
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "MODEL_NOT_FOUND"

    def test_extract_model_load_failure(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 503 MODEL_LOADING when model is not loaded (non-blocking load)."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False
        mock_registry.start_load_async = AsyncMock(return_value=True)
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["entity"]},
            },
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["code"] == "MODEL_LOADING"
        assert "loading" in data["detail"]["message"].lower()
        mock_registry.start_load_async.assert_called_once()

    def test_extract_lazy_loads_model(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Model triggers background load on first request if not loaded."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False
        mock_registry.start_load_async = AsyncMock(return_value=True)
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        mock_registry.start_load_async.assert_called_once_with("test-extractor", device="cpu")

    def test_extract_model_does_not_support_extraction(
        self, client: TestClient, mock_registry: MagicMock, mock_encoder_adapter: MagicMock
    ) -> None:
        """Returns 400 when model doesn't support extraction (no json in outputs)."""
        # Mock config to return an encoder model (no "json" in outputs)
        mock_registry.get_config.return_value = ModelConfig(
            sie_id="test-encoder",
            hf_id="org/test-encoder",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=1024))),
            profiles={"default": ProfileConfig(adapter_path="test:TestEncoderAdapter", max_batch_tokens=8192)},
        )

        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["entity"]},
            },
        )
        # Config check returns 400
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "does not support extraction" in data["detail"]["message"]

    def test_extract_empty_items_rejected(self, client: TestClient) -> None:
        """Empty items list is rejected."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [],
                "params": {"labels": ["entity"]},
            },
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_extract_non_dict_items_rejected(self, client: TestClient) -> None:
        """Non-dict items return 400, not 500."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": ["just a string"],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `object`, got `str` - at `$.items[0]`"

    def test_extract_non_string_text_rejected(self, client: TestClient) -> None:
        """Item with non-string 'text' returns 400, not 500."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": 123}],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `str | null`, got `int` - at `$.items[0].text`"

    def test_extract_non_list_images_rejected(self, client: TestClient) -> None:
        """Item with non-list 'images' returns 400, not 500."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"images": "not-a-list"}],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `array | null`, got `str` - at `$.items[0].images`"


class TestExtractEntityResults:
    """Tests for entity extraction result format."""

    def test_entities_have_required_fields(self, client: TestClient) -> None:
        """Entities have text, label, score, start, end fields."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc. was founded by Steve Jobs."}],
                "params": {"labels": ["person", "organization"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        entities = data["items"][0]["entities"]
        assert len(entities) > 0
        for entity in entities:
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity
            assert "start" in entity
            assert "end" in entity

    def test_entities_have_correct_types(self, client: TestClient) -> None:
        """Entity fields have correct types."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Test text"}],
                "params": {"labels": ["entity"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        entities = data["items"][0]["entities"]
        if entities:
            entity = entities[0]
            assert isinstance(entity["text"], str)
            assert isinstance(entity["label"], str)
            assert isinstance(entity["score"], int | float)
            assert isinstance(entity["start"], int)
            assert isinstance(entity["end"], int)


class TestMsgpackExtractRequests:
    """Tests for msgpack request body handling for extract endpoint."""

    def test_msgpack_request_basic(self, client: TestClient) -> None:
        """Msgpack request body is parsed correctly."""
        request_data = {
            "items": [{"text": "Extract from this"}],
            "params": {"labels": ["entity"]},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/extract/test-extractor",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200
        # Response is also msgpack by default
        assert response.headers["content-type"] == "application/msgpack"
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-extractor"
        assert len(data["items"]) == 1

    def test_msgpack_request_with_json_response(self, client: TestClient) -> None:
        """Msgpack request can get JSON response with Accept header."""
        request_data = {
            "items": [{"text": "Text"}],
            "params": {"labels": ["entity"]},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/extract/test-extractor",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["model"] == "test-extractor"

    def test_msgpack_request_with_item_ids(self, client: TestClient) -> None:
        """Msgpack request with item IDs is parsed correctly."""
        request_data = {
            "items": [
                {"id": "doc-1", "text": "First document"},
                {"id": "doc-2", "text": "Second document"},
            ],
            "params": {"labels": ["entity"]},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/extract/test-extractor",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-1"
        assert data["items"][1]["id"] == "doc-2"

    def test_msgpack_request_invalid_body(self, client: TestClient) -> None:
        """Invalid msgpack body returns 400."""
        response = client.post(
            "/v1/extract/test-extractor",
            content=b"not valid msgpack",
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400

    def test_msgpack_request_validation_error(self, client: TestClient) -> None:
        """Msgpack request with invalid schema returns 422."""
        request_data = {"items": []}  # Empty items not allowed
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/extract/test-extractor",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_x_msgpack_content_type(self, client: TestClient) -> None:
        """Alternative x-msgpack content type is also accepted."""
        request_data = {
            "items": [{"text": "Text"}],
            "params": {"labels": ["entity"]},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/extract/test-extractor",
            content=msgpack_body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200


class TestExtractErrorHandling:
    """Tests for extract endpoint error handling."""

    def test_extract_adapter_value_error(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """ValueError from adapter returns 400."""
        mock_adapter.extract.side_effect = ValueError("Labels are required")
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {},  # No labels
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "Labels are required" in data["detail"]["message"]

    def test_invalid_profile_returns_400(self, client: TestClient) -> None:
        """Invalid profile name returns 400, not 500.

        Regression test: profile resolution was previously inside the inference
        try/except block, so ValueError → HTTPException(400) was caught by
        the outer except Exception → HTTPException(500).
        """
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc. was founded by Steve Jobs."}],
                "params": {
                    "labels": ["person", "organization"],
                    "options": {"profile": "nonexistent_profile"},
                },
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "nonexistent_profile" in data["detail"]["message"]

    def test_extract_adapter_runtime_error(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """RuntimeError from adapter returns 500."""
        mock_adapter.extract.side_effect = RuntimeError("Inference failed")
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Text"}],
                "params": {"labels": ["entity"]},
            },
        )
        assert response.status_code == 500
        data = response.json()
        assert data["detail"]["code"] == "INFERENCE_ERROR"


class TestFormatOutput:
    """Tests for ExtractHandler.format_output classification behavior."""

    def test_format_output_classifications_none_produces_empty_list(self) -> None:
        """When classifications is None, format_output still includes an empty list per item."""
        from sie_server.core.worker.handlers.extract import ExtractHandler

        output = ExtractOutput(
            entities=[[Entity(text="Apple", label="ORG", score=0.95, start=0, end=5)]],
            classifications=None,
        )
        results = ExtractHandler.format_output(output)
        assert len(results) == 1
        assert results[0]["classifications"] == []
        assert len(results[0]["entities"]) == 1

    def test_format_output_with_populated_classifications(self) -> None:
        """When classifications are populated, format_output serializes them correctly."""
        from sie_server.core.worker.handlers.extract import ExtractHandler
        from sie_server.types.responses import Classification

        output = ExtractOutput(
            entities=[[]],
            classifications=[
                [
                    Classification(label="positive", score=0.9),
                    Classification(label="negative", score=0.1),
                ]
            ],
        )
        results = ExtractHandler.format_output(output)
        assert len(results) == 1
        assert results[0]["entities"] == []
        assert len(results[0]["classifications"]) == 2
        assert results[0]["classifications"][0]["label"] == "positive"
        assert results[0]["classifications"][0]["score"] == 0.9
        assert results[0]["classifications"][1]["label"] == "negative"
        assert results[0]["classifications"][1]["score"] == 0.1

    def test_format_output_multiple_items_mixed(self) -> None:
        """format_output handles multiple items with classifications correctly."""
        from sie_server.core.worker.handlers.extract import ExtractHandler
        from sie_server.types.responses import Classification

        output = ExtractOutput(
            entities=[
                [Entity(text="Apple", label="ORG", score=0.95, start=0, end=5)],
                [],
            ],
            classifications=[
                [Classification(label="tech", score=0.8)],
                [Classification(label="finance", score=0.7)],
            ],
        )
        results = ExtractHandler.format_output(output)
        assert len(results) == 2
        assert len(results[0]["entities"]) == 1
        assert results[0]["classifications"][0]["label"] == "tech"
        assert results[1]["entities"] == []
        assert results[1]["classifications"][0]["label"] == "finance"
