"""Tests for extract endpoint."""

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import msgpack
import msgpack_numpy as m
import msgspec
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api.extract import _extract_via_worker
from sie_server.api.extract import router as extract_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ExtractTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.extract_cost import (
    MAX_EXTRACT_LABELS,
    MAX_OUTPUT_SCHEMA_DEPTH,
    MAX_OUTPUT_SCHEMA_VALUES,
    build_extract_prepared_items,
    extract_item_cost,
    output_schema_shape_error,
)
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import MAX_ITEM_TEXT_BYTES, Item
from sie_server.types.responses import Classification, Entity

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
    for _ in items:
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
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
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

    @pytest.mark.parametrize("instruction", [["x"], {"a": 1}, 3])
    def test_extract_rejects_a_non_string_options_instruction(
        self, client: TestClient, mock_adapter: MagicMock, instruction: object
    ) -> None:
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"text": "x"}], "params": {"labels": ["a"], "options": {"instruction": instruction}}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        mock_adapter.extract.assert_not_called()

    def test_extract_item_error_is_preserved_in_local_response(
        self,
        client: TestClient,
        mock_adapter: MagicMock,
    ) -> None:
        mock_adapter.extract.side_effect = None
        mock_adapter.extract.return_value = ExtractOutput(
            entities=[[]],
            data=[{"partial": True}],
            errors=[ExtractItemError(code="INFERENCE_ERROR", message="Document export failed")],
            pages=[3],
        )

        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"text": "document placeholder"}], "params": {}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 200
        assert response.json()["items"][0]["error"] == {
            "code": "INFERENCE_ERROR",
            "message": "Document export failed",
        }
        assert response.json()["items"][0]["data"] == {"partial": True}

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
        # Real registry raises for unknown models: guards the KeyError-500 regression.
        mock_registry.get_worker.side_effect = KeyError("Model 'unknown-model' not found in registry")
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

    def test_extract_over_cap_labels_rejected(self, client: TestClient) -> None:
        """A labels list past the cap is a typed 400 naming the limit.

        GLiNER-family adapters run one forward pass per label, so an unbounded
        labels list is an uncapped compute vector — mirror the score/rerank
        candidate cap with a named 400 rather than letting it through.
        """
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc."}],
                "params": {"labels": [f"label-{i}" for i in range(MAX_EXTRACT_LABELS + 1)]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert str(MAX_EXTRACT_LABELS) in data["detail"]["message"]
        assert "labels" in data["detail"]["message"]

    def test_extract_overly_deep_output_schema_rejected(self, client: TestClient) -> None:
        """A schema nested past the bound is a 400 at ingress.

        The worker's batching key and adapter compilers walk the schema
        recursively; hundreds of levels would otherwise surface as a
        RecursionError inside the worker instead of a 400.
        """
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(400):
            schema = {"type": "object", "properties": {"a": schema}}
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"text": "Apple Inc."}], "params": {"output_schema": schema}},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert str(MAX_OUTPUT_SCHEMA_DEPTH) in data["detail"]["message"]

    def test_output_schema_shape_limits(self) -> None:
        def nested(levels: int) -> dict[str, Any]:
            node: dict[str, Any] = {}
            for _ in range(levels - 1):
                node = {"a": node}
            return node

        assert output_schema_shape_error(nested(MAX_OUTPUT_SCHEMA_DEPTH)) is None
        assert "nest" in (output_schema_shape_error(nested(MAX_OUTPUT_SCHEMA_DEPTH + 1)) or "")
        assert output_schema_shape_error(nested(5000)) is not None  # iterative: no RecursionError
        wide = {"properties": {f"p{i}": {"type": "string"} for i in range(MAX_OUTPUT_SCHEMA_VALUES)}}
        assert "at most" in (output_schema_shape_error(wide) or "")
        assert output_schema_shape_error({"type": "object", "properties": {"n": {"type": "string"}}}) is None

    def test_extract_at_cap_labels_accepted(self, client: TestClient) -> None:
        """A labels list exactly at the cap is accepted."""
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc."}],
                "params": {"labels": [f"label-{i}" for i in range(MAX_EXTRACT_LABELS)]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

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

    def test_image_preprocessor_model_rejects_text_only_request(
        self,
        client: TestClient,
        mock_registry: MagicMock,
    ) -> None:
        """Image-only preprocessors reject text-only extract requests with 400."""
        preprocessor_registry = MagicMock()
        preprocessor_registry.has_preprocessor.side_effect = lambda _model, modality: modality == "image"
        mock_registry.preprocessor_registry = preprocessor_registry

        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "receipt total"}],
                "params": {},
            },
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "requires image input" in data["detail"]["message"]
        mock_registry.start_worker.assert_not_called()


# msgpack framing of ``{"state": <str of 65,536 bytes or more>}``.
_STATE_FRAMING = len(msgspec.msgpack.encode({"state": "x" * 70_000})) - 70_000


class TestExtractItemTextSize:
    """Each item's text and metadata are bounded at ingress, before any work."""

    def test_text_at_the_cap_is_extracted(self, client: TestClient, mock_adapter: MagicMock) -> None:
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"text": "x" * MAX_ITEM_TEXT_BYTES}], "params": {"labels": ["person"]}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 200
        (items,) = mock_adapter.extract.call_args.args
        assert len(items[0].text) == MAX_ITEM_TEXT_BYTES

    def test_text_over_the_cap_rejects_the_request_before_any_work(
        self, client: TestClient, mock_adapter: MagicMock, mock_registry: MagicMock
    ) -> None:
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [{"text": "Apple Inc."}, {"text": "x" * (MAX_ITEM_TEXT_BYTES + 1)}],
                "params": {"labels": ["organization"]},
            },
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "INVALID_INPUT",
            "message": f"Field 'items[1]' must hold at most {MAX_ITEM_TEXT_BYTES} bytes of UTF-8 text",
        }
        mock_registry.start_worker.assert_not_called()
        mock_adapter.extract_item_costs.assert_not_called()
        mock_adapter.extract.assert_not_called()

    def test_multibyte_text_is_measured_in_utf8_bytes(self, client: TestClient, mock_adapter: MagicMock) -> None:
        # Two bytes per character: over the cap in bytes, about half of it in characters.
        text = "é" * (MAX_ITEM_TEXT_BYTES // 2 + 1)
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"text": text}], "params": {"labels": ["person"]}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        assert str(MAX_ITEM_TEXT_BYTES) in response.json()["detail"]["message"]
        mock_adapter.extract.assert_not_called()

    def test_msgpack_body_over_the_cap_is_rejected(self, client: TestClient, mock_adapter: MagicMock) -> None:
        body = msgpack.packb(
            {"items": [{"text": "x" * (MAX_ITEM_TEXT_BYTES + 1)}], "params": {"labels": ["person"]}},
            use_bin_type=True,
        )
        response = client.post(
            "/v1/extract/test-extractor",
            content=body,
            headers={"Content-Type": "application/msgpack", **JSON_HEADERS},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "INVALID_INPUT"
        mock_adapter.extract.assert_not_called()

    def test_metadata_state_at_the_cap_is_extracted(self, client: TestClient, mock_adapter: MagicMock) -> None:
        state = "x" * (MAX_ITEM_TEXT_BYTES - _STATE_FRAMING)
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"metadata": {"state": state}}], "params": {"labels": ["approve", "deny"]}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 200
        (items,) = mock_adapter.extract.call_args.args
        assert items[0].metadata == {"state": state}

    def test_metadata_state_over_the_cap_is_rejected(self, client: TestClient, mock_adapter: MagicMock) -> None:
        state = {"ticket": "x" * (MAX_ITEM_TEXT_BYTES - _STATE_FRAMING)}
        response = client.post(
            "/v1/extract/test-extractor",
            json={"items": [{"metadata": {"state": state}}], "params": {"labels": ["approve", "deny"]}},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        assert response.json()["detail"]["message"] == (
            f"Field 'items[0]' must hold at most {MAX_ITEM_TEXT_BYTES} bytes of UTF-8 text and metadata"
        )
        mock_adapter.extract.assert_not_called()

    def test_ordinary_items_with_metadata_are_unaffected(self, client: TestClient, mock_adapter: MagicMock) -> None:
        response = client.post(
            "/v1/extract/test-extractor",
            json={
                "items": [
                    {"text": "Steve Jobs founded Apple.", "metadata": {"entities": [{"text": "Apple"}]}},
                    {"metadata": {"state": {"turns": ["refund please", "approved"]}}},
                ],
                "params": {"labels": ["person"]},
            },
            headers=JSON_HEADERS,
        )

        assert response.status_code == 200
        assert len(response.json()["items"]) == 2
        mock_adapter.extract.assert_called_once()


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


class TestExtractHandlerSchemaForwarding:
    def _metadata(self, output_schema: dict[str, Any]) -> MagicMock:
        metadata = MagicMock()
        metadata.labels = None
        metadata.output_schema = output_schema
        metadata.instruction = None
        metadata.options = None
        return metadata

    def test_output_schema_partitions_batches(self) -> None:
        handler = ExtractHandler()
        string_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        list_schema = {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
        }
        assert handler.make_config_key(self._metadata(string_schema)) != handler.make_config_key(
            self._metadata(list_schema)
        )

    def test_empty_output_schema_does_not_batch_with_absent_schema(self) -> None:
        handler = ExtractHandler()
        absent = self._metadata({})
        absent.output_schema = None

        assert handler.make_config_key(self._metadata({})) != handler.make_config_key(absent)

    def test_run_inference_forwards_original_output_schema(self) -> None:
        handler = ExtractHandler()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        metadata = self._metadata(schema)
        adapter = MagicMock()
        adapter.extract.return_value = ExtractOutput(entities=[[]], data=[{"name": "Ada"}])
        items = [Item(text="Ada founded Acme")]

        output = handler.run_inference(
            adapter,
            items,
            handler.make_config_key(metadata),
            None,
            [metadata],
        )

        assert output.data == [{"name": "Ada"}]
        assert adapter.extract.call_args.kwargs["output_schema"] is schema

    def test_run_inference_forwards_original_nested_options(self) -> None:
        handler = ExtractHandler()
        options = {
            "label_groups": {"urgency": ["low", "high"], "topic": ["billing", "bug"]},
            "examples": [{"text": "Refund me", "labels": ["topic.billing"]}],
            "threshold": 0.2,
        }
        metadata = self._metadata({})
        metadata.output_schema = None
        metadata.options = options
        adapter = MagicMock()
        adapter.extract.return_value = ExtractOutput(entities=[[]])

        handler.run_inference(
            adapter, [Item(text="Charged twice")], handler.make_config_key(metadata), None, [metadata]
        )

        forwarded = adapter.extract.call_args.kwargs["options"]
        assert forwarded == options
        assert list(forwarded["label_groups"]) == ["urgency", "topic"]
        assert isinstance(forwarded["examples"][0], dict)

    def test_options_that_differ_only_in_key_order_do_not_share_a_batch(self) -> None:
        handler = ExtractHandler()

        def key(options: dict[str, Any]) -> tuple[Any, ...]:
            metadata = self._metadata({})
            metadata.output_schema = None
            metadata.options = options
            return handler.make_config_key(metadata)

        urgency_first = {"label_groups": {"urgency": ["low", "high"], "topic": ["billing", "bug"]}}
        topic_first = {"label_groups": {"topic": ["billing", "bug"], "urgency": ["low", "high"]}}
        example_a = {"examples": [{"text": "x", "labels": {"urgency": "low", "topic": "bug"}}]}
        example_b = {"examples": [{"text": "x", "labels": {"topic": "bug", "urgency": "low"}}]}

        assert key(urgency_first) != key(topic_first)
        assert key(example_a) != key(example_b)
        assert key(dict(urgency_first)) == key(urgency_first)
        assert len({key(urgency_first), key(topic_first), key(dict(topic_first))}) == 2

    def test_schemas_that_differ_only_in_property_order_do_not_share_a_batch(self) -> None:
        handler = ExtractHandler()
        first = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
        second = {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "string"}}}

        assert handler.make_config_key(self._metadata(first)) != handler.make_config_key(self._metadata(second))

    def test_run_inference_keeps_whisper_timestamp_granularities_a_list(self) -> None:
        # Regression: rebuilding options from the batching key turned this list
        # into a tuple, which Whisper's option parser rejects.
        from sie_server.adapters.whisper.adapter import _parse_options

        handler = ExtractHandler()
        metadata = self._metadata({})
        metadata.output_schema = None
        metadata.options = {"timestamp_granularities": ["word", "segment"], "language": "en"}
        adapter = MagicMock()
        parsed: list[Any] = []

        def extract(items: list[Item], **kwargs: Any) -> ExtractOutput:
            parsed.append(_parse_options(kwargs["options"]))
            return ExtractOutput(entities=[[] for _ in items])

        adapter.extract.side_effect = extract

        handler.run_inference(adapter, [Item(text="x")], handler.make_config_key(metadata), None, [metadata])

        assert parsed == [("en", None, frozenset({"word", "segment"}))]

    def test_run_inference_passes_none_for_absent_options(self) -> None:
        handler = ExtractHandler()
        metadata = self._metadata({})
        metadata.output_schema = None
        adapter = MagicMock()
        adapter.extract.return_value = ExtractOutput(entities=[[]])

        handler.run_inference(adapter, [Item(text="x")], handler.make_config_key(metadata), None, [metadata])

        assert adapter.extract.call_args.kwargs["options"] is None


class TestFormatOutput:
    """Tests for ExtractHandler.format_output classification behavior."""

    def test_format_output_classifications_none_produces_empty_list(self) -> None:
        """When classifications is None, format_output still includes an empty list per item."""
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

    def test_format_output_data_none_produces_empty_dict(self) -> None:
        """When data is None, format_output emits an empty dict per item."""
        output = ExtractOutput(entities=[[]], data=None)
        results = ExtractHandler.format_output(output)
        assert results[0]["data"] == {}

    def test_format_output_with_populated_data(self) -> None:
        """When data is populated, format_output passes it through verbatim."""
        payload = {"document": {"pages": [{"text": "hello"}]}}
        output = ExtractOutput(entities=[[], []], data=[payload, {}])
        results = ExtractHandler.format_output(output)
        assert results[0]["data"] == payload
        assert results[1]["data"] == {}


class TestExtractOutputData:
    """Validation around aligned ExtractOutput data and error fields."""

    def test_data_length_must_match_batch_size(self) -> None:
        with pytest.raises(ValueError, match="data list length"):
            ExtractOutput(entities=[[], []], data=[{"a": 1}])

    def test_error_length_must_match_batch_size(self) -> None:
        with pytest.raises(ValueError, match="errors list length"):
            ExtractOutput(
                entities=[[], []],
                errors=[ExtractItemError(code="INFERENCE_ERROR", message="failed")],
            )

    def test_slice_output_threads_data(self) -> None:
        output = ExtractOutput(
            entities=[[], []],
            data=[{"page": 0}, {"page": 1}],
        )
        sliced = ExtractHandler().slice_output(output, 1)
        assert sliced.data == [{"page": 1}]

    def test_slice_assemble_and_format_threads_item_errors(self) -> None:
        error = ExtractItemError(code="INFERENCE_ERROR", message="Document export failed")
        output = ExtractOutput(entities=[[], []], data=[{"ok": True}, {}], errors=[None, error])

        handler = ExtractHandler()
        partials = {i: handler.slice_output(output, i) for i in range(2)}
        assembled = handler.assemble_output(partials, batch_size=2)
        formatted = handler.format_output(assembled)

        assert assembled.errors == [None, error]
        assert "error" not in formatted[0]
        assert formatted[1]["error"] == {
            "code": "INFERENCE_ERROR",
            "message": "Document export failed",
        }

    def test_assemble_output_reassembles_data(self) -> None:
        partials = {
            0: ExtractOutput(entities=[[]], data=[{"page": 0}]),
            1: ExtractOutput(entities=[[]], data=[{"page": 1}]),
        }
        assembled = ExtractHandler().assemble_output(partials, batch_size=2)
        assert assembled.data == [{"page": 0}, {"page": 1}]

    def test_assemble_output_data_partial_coverage(self) -> None:
        """When only one partial has data, the missing slots default to {}."""
        partials = {
            0: ExtractOutput(entities=[[]], data=[{"page": 0}]),
            1: ExtractOutput(entities=[[]], data=None),
        }
        assembled = ExtractHandler().assemble_output(partials, batch_size=2)
        assert assembled.data == [{"page": 0}, {}]


class TestExtractCost:
    """Cost calculation for prepared extract items."""

    def test_text_item_uses_character_count(self) -> None:
        assert extract_item_cost(Item(text="hello world")) == 11

    def test_document_item_uses_byte_size(self) -> None:
        document = {"data": b"%PDF-1.4 fake content", "format": "pdf"}
        assert extract_item_cost(Item(document=document)) == len(document["data"])

    def test_document_takes_priority_over_text(self) -> None:
        document = {"data": b"AB", "format": "pdf"}
        item = Item(text="ignored-since-document-present", document=document)
        assert extract_item_cost(item) == 2

    def test_empty_item_has_zero_cost(self) -> None:
        assert extract_item_cost(Item()) == 0

    def test_build_extract_prepared_items_assigns_indices(self) -> None:
        items = [
            Item(text="abc"),
            Item(document={"data": b"hello world", "format": "pdf"}),
        ]
        prepared = build_extract_prepared_items(items)
        assert [(p.cost, p.original_index) for p in prepared] == [(3, 0), (11, 1)]


class _PlainExtractAdapter(BaseAdapter):
    """An extract adapter that keeps ModelAdapter's default (no per-item cost hook)."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(inputs=("text",), outputs=("json",))

    def load(self, device: str) -> None:
        pass

    def extract(self, items: list[Item], **kwargs: Any) -> ExtractOutput:
        return ExtractOutput(entities=[[] for _ in items])


class _RowCostAdapter(_PlainExtractAdapter):
    """Reports its own per-item costs, like an adapter that runs several model rows per item."""

    def __init__(self) -> None:
        self.hook_calls: list[dict[str, Any]] = []

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        self.hook_calls.append(
            {"labels": labels, "output_schema": output_schema, "instruction": instruction, "options": options}
        )
        return [1000 + 10 * i for i in range(len(items))]


def _cost_capture_registry(adapter: BaseAdapter) -> tuple[MagicMock, MagicMock]:
    registry = MagicMock(spec=["get", "get_config", "start_worker"])
    registry.get.return_value = adapter
    worker = MagicMock()

    async def submit_extract(*, prepared_items: list[Any], items: list[Item], **kwargs: Any) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        future.set_result(WorkerResult(output=ExtractOutput(entities=[[] for _ in items]), timing=RequestTiming()))
        return future

    worker.submit_extract = AsyncMock(side_effect=submit_extract)
    registry.start_worker = AsyncMock(return_value=worker)
    return registry, worker


class TestExtractItemCostHook:
    """``_extract_via_worker`` sizes batches from the adapter's per-item cost hook when it has one."""

    @pytest.mark.asyncio
    async def test_hook_costs_reach_the_worker(self) -> None:
        adapter = _RowCostAdapter()
        registry, worker = _cost_capture_registry(adapter)
        items = [Item(text="a"), Item(text="bb"), Item(metadata={"state": {"k": "v"}})]
        schema = {"q": {"type": "noul", "instructions": "x"}}

        await _extract_via_worker(registry, "m", items, output_schema=schema, instruction=None, options={"max_len": 64})

        prepared = worker.submit_extract.await_args.kwargs["prepared_items"]
        assert [(p.cost, p.original_index) for p in prepared] == [(1000, 0), (1010, 1), (1020, 2)]
        assert adapter.hook_calls == [
            {"labels": None, "output_schema": schema, "instruction": None, "options": {"max_len": 64}}
        ]

    @pytest.mark.asyncio
    async def test_default_hook_keeps_character_and_byte_costs(self) -> None:
        registry, worker = _cost_capture_registry(_PlainExtractAdapter())
        document = {"data": b"%PDF-1.4 fake", "format": "pdf"}
        items = [Item(text="hello"), Item(document=document)]

        await _extract_via_worker(registry, "m", items, labels=["person"])

        prepared = worker.submit_extract.await_args.kwargs["prepared_items"]
        assert [p.cost for p in prepared] == [5, len(document["data"])]
