from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import msgpack_numpy as m
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.encode import router as encode_router
from sie_server.config.model import (
    AdapterOptions,
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry

# Patch msgpack for numpy support
m.patch()

# Header for JSON responses (msgpack is default)
JSON_HEADERS = {"Accept": "application/json"}


def _mock_encode_impl(items: list[Any], output_types: list[str], **kwargs: Any) -> Any:
    """Implementation for mock encode - returns EncodeOutput."""
    from sie_server.core.inference_output import EncodeOutput, SparseVector

    batch_size = len(items)

    # Build dense embeddings if requested
    dense = None
    if "dense" in output_types:
        dense = np.array([[0.1, 0.2, 0.3]] * batch_size, dtype=np.float32)

    # Build sparse embeddings if requested
    sparse = None
    if "sparse" in output_types:
        sparse = [
            SparseVector(
                indices=np.array([1, 5, 10]),
                values=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            )
            for _ in range(batch_size)
        ]

    # Build multivector if requested
    multivector = None
    if "multivector" in output_types:
        rng = np.random.default_rng(42)
        multivector = [rng.standard_normal((5, 128)).astype(np.float32) for _ in range(batch_size)]

    return EncodeOutput(
        dense=dense,
        sparse=sparse,
        multivector=multivector,
        batch_size=batch_size,
        dense_dim=3 if dense is not None else None,
        multivector_token_dim=128 if multivector is not None else None,
    )


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Create a mock adapter that returns test embeddings."""
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=_mock_encode_impl)
    return adapter


@pytest.fixture
def mock_registry(mock_adapter: MagicMock) -> MagicMock:
    """Create a mock registry."""
    from concurrent.futures import ThreadPoolExecutor

    from sie_server.core.postprocessor_registry import PostprocessorRegistry

    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.get.return_value = mock_adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test",
        tasks=Tasks(
            encode=EncodeTask(
                dense=EmbeddingDim(dim=3),
                sparse=EmbeddingDim(dim=30522),
                multivector=EmbeddingDim(dim=128),
            ),
        ),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"
    # Mock preprocessor_registry to NOT have a tokenizer (use direct adapter path)
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry
    # Use real postprocessor_registry for quantization
    cpu_pool = ThreadPoolExecutor(max_workers=1)
    registry.postprocessor_registry = PostprocessorRegistry(cpu_pool)
    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(encode_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestEncodeEndpoint:
    """Tests for POST /v1/encode/{model}."""

    def test_encode_basic_json(self, client: TestClient) -> None:
        """Basic encode request returns JSON when Accept header set."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["dense"]["dims"] == 3
        assert len(data["items"][0]["dense"]["values"]) == 3

    def test_encode_basic_msgpack(self, client: TestClient) -> None:
        """Basic encode request returns msgpack by default."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Deserialize msgpack
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1
        # Values come back as numpy arrays with msgpack-numpy
        assert isinstance(data["items"][0]["dense"]["values"], np.ndarray)

    def test_encode_with_id(self, client: TestClient) -> None:
        """Item IDs are preserved in response."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"id": "doc-1", "text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-1"

    def test_encode_multiple_items(self, client: TestClient) -> None:
        """Can encode multiple items at once."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [
                    {"text": "Hello"},
                    {"text": "World"},
                    {"text": "Test"},
                ]
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3

    def test_encode_sparse_output(self, client: TestClient) -> None:
        """Can request sparse output type."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["sparse"] is not None
        assert "indices" in data["items"][0]["sparse"]
        assert "values" in data["items"][0]["sparse"]

    def test_encode_multiple_output_types(self, client: TestClient) -> None:
        """Can request multiple output types."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["sparse"] is not None

    def test_encode_model_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 404 for unknown model."""
        mock_registry.has_model.return_value = False
        response = client.post(
            "/v1/encode/unknown-model",
            json={"items": [{"text": "Hello"}]},
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "MODEL_NOT_FOUND"

    def test_encode_model_load_failure(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 503 MODEL_LOADING when model is not loaded (non-blocking load)."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False

        async def start_load_async_success(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_registry.start_load_async = MagicMock(side_effect=start_load_async_success)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["code"] == "MODEL_LOADING"
        assert "loading" in data["detail"]["message"].lower()
        mock_registry.start_load_async.assert_called_once()

    def test_encode_lazy_loads_model(
        self, client: TestClient, mock_registry: MagicMock, mock_adapter: MagicMock
    ) -> None:
        """Model triggers background load on first request if not loaded."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False

        async def start_load_async_success(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_registry.start_load_async = MagicMock(side_effect=start_load_async_success)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503
        mock_registry.start_load_async.assert_called_once_with("test-model", device="cpu")

    def test_encode_unsupported_output_type(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Returns 400 for unsupported output type."""
        mock_registry.get_config.return_value = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
            profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
        )
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"]},  # Request unsupported type
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert "sparse" in data["detail"]["message"]

    def test_encode_with_instruction(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """Instruction parameter is passed to adapter."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"instruction": "Search for documents"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        # Verify instruction was passed
        mock_adapter.encode.assert_called_once()
        call_kwargs = mock_adapter.encode.call_args
        assert call_kwargs.kwargs["instruction"] == "Search for documents"

    def test_encode_is_query_param(self, client: TestClient, mock_adapter: MagicMock) -> None:
        """is_query option is passed to adapter via options dict."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"is_query": True}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        call_kwargs = mock_adapter.encode.call_args
        assert call_kwargs.kwargs["is_query"] is True

    def test_encode_empty_items_rejected(self, client: TestClient) -> None:
        """Empty items list is rejected."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": []},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_encode_non_dict_items_rejected(self, client: TestClient) -> None:
        """Non-dict items return 400, not 500."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": ["just a string"]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `object`, got `str` - at `$.items[0]`"

    def test_encode_non_string_text_rejected(self, client: TestClient) -> None:
        """Item with non-string 'text' returns 400, not 500."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": 123}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["code"] == "INVALID_INPUT"
        assert data["detail"]["message"] == "Expected `str | null`, got `int` - at `$.items[0].text`"


class TestMsgpackRequests:
    """Tests for msgpack request body handling (DESIGN.md Section 4.3)."""

    def test_msgpack_request_basic(self, client: TestClient) -> None:
        """Msgpack request body is parsed correctly."""
        request_data = {"items": [{"text": "Hello world"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200
        # Response is also msgpack by default
        assert response.headers["content-type"] == "application/msgpack"
        data = msgpack.unpackb(response.content, raw=False)
        assert data["model"] == "test-model"
        assert len(data["items"]) == 1

    def test_msgpack_request_with_json_response(self, client: TestClient) -> None:
        """Msgpack request can get JSON response with Accept header."""
        request_data = {"items": [{"text": "Hello"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["model"] == "test-model"

    def test_msgpack_request_with_params(self, client: TestClient) -> None:
        """Msgpack request with params is parsed correctly."""
        request_data = {
            "items": [{"id": "doc-1", "text": "Hello"}],
            "params": {"output_types": ["dense", "sparse"], "options": {"is_query": True}},
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["id"] == "doc-1"
        assert data["items"][0]["dense"] is not None
        assert data["items"][0]["sparse"] is not None

    def test_msgpack_request_with_binary_image(self, client: TestClient) -> None:
        """Msgpack request can include binary image data."""
        # Simulate image bytes (would be actual image in real usage)
        fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        request_data = {
            "items": [
                {
                    "text": "Describe this image",
                    "images": [{"data": fake_image_bytes, "format": "png"}],
                }
            ]
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/json",
            },
        )
        # Should parse successfully (adapter will handle the image)
        assert response.status_code == 200

    def test_msgpack_request_roundtrip(self, client: TestClient) -> None:
        """Full msgpack request/response roundtrip preserves numpy arrays."""
        request_data = {"items": [{"text": "Test embedding"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 200

        # Deserialize response
        data = msgpack.unpackb(response.content, raw=False)
        dense_values = data["items"][0]["dense"]["values"]

        # Values should be numpy array (not list)
        assert isinstance(dense_values, np.ndarray)
        assert dense_values.dtype == np.float32

    def test_msgpack_request_invalid_body(self, client: TestClient) -> None:
        """Invalid msgpack body returns 400."""
        response = client.post(
            "/v1/encode/test-model",
            content=b"not valid msgpack",
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400

    def test_msgpack_request_validation_error(self, client: TestClient) -> None:
        """Msgpack request with invalid schema returns 400."""
        request_data = {"items": []}  # Empty items should fail validation
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack"},
        )
        assert response.status_code == 400  # Custom validation error (not Pydantic)

    def test_x_msgpack_content_type(self, client: TestClient) -> None:
        """Alternative x-msgpack content type is also accepted."""
        request_data = {"items": [{"text": "Hello"}]}
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/x-msgpack"},
        )
        assert response.status_code == 200


class TestMsgspecDecodeEndToEnd:
    """Verify msgspec decode works end-to-end for JSON and msgpack with rich payloads."""

    def test_json_decode_with_all_item_fields(self, client: TestClient) -> None:
        """JSON request with text + id + metadata decodes through msgspec correctly."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [
                    {"id": "doc-1", "text": "Hello world", "metadata": {"source": "test"}},
                    {"id": "doc-2", "text": "Another doc"},
                    {"text": "No id or metadata"},
                ],
                "params": {
                    "output_types": ["dense"],
                    "instruction": "Represent this document",
                },
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 3
        assert data["items"][0]["id"] == "doc-1"
        assert data["items"][1]["id"] == "doc-2"

    def test_msgpack_decode_with_all_item_fields(self, client: TestClient) -> None:
        """Msgpack request with text + id + metadata decodes through msgspec correctly."""
        request_data = {
            "items": [
                {"id": "doc-1", "text": "Hello world", "metadata": {"source": "test"}},
                {"id": "doc-2", "text": "Another doc"},
            ],
            "params": {
                "output_types": ["dense"],
                "instruction": "Represent this document",
            },
        }
        msgpack_body = msgpack.packb(request_data, use_bin_type=True)

        response = client.post(
            "/v1/encode/test-model",
            content=msgpack_body,
            headers={"Content-Type": "application/msgpack", "Accept": "application/json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["id"] == "doc-1"

    def test_unknown_fields_ignored(self, client: TestClient) -> None:
        """Extra fields in request body are silently ignored (forward compatibility)."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello", "future_field": 42}],
                "unknown_top_level": True,
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200


class TestEncodeLoraRouting:
    """Regression tests for issue #94: map options['lora_id'] -> options['lora'].

    The encode handler resolves the LoRA from ``options["lora_id"]`` (which is
    populated by the profile config's ``adapter_options.runtime``), but the
    worker's batcher routes LoRA-specific requests on ``options["lora"]``.
    Without the mapping, named-profile LoRA requests load the adapter into the
    registry but every inference still runs against the base model weights —
    a silent quality regression.
    """

    @staticmethod
    def _make_lora_registry(
        mock_adapter: MagicMock,
        *,
        lora_state: tuple[bool, bool] = (True, False),
    ) -> MagicMock:
        """Build a mock registry whose 'us-regulatory' profile carries a lora_id.

        Mirrors the ``mock_registry`` fixture but adds a second profile and an
        async ``ensure_lora_loaded_async``. ``lora_state`` is the
        ``(is_ready, is_loading)`` tuple returned by that mock.
        """
        from concurrent.futures import ThreadPoolExecutor

        from sie_server.core.postprocessor_registry import PostprocessorRegistry

        registry = MagicMock(spec=ModelRegistry)
        registry.has_model.return_value = True
        registry.is_loaded.return_value = True
        registry.is_loading.return_value = False
        registry.is_unloading.return_value = False
        registry.get.return_value = mock_adapter
        registry.get_config.return_value = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="test:TestAdapter",
                    max_batch_tokens=8192,
                ),
                "us-regulatory": ProfileConfig(
                    adapter_path="test:TestAdapter",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(
                        runtime={"lora_id": "org/test-lora"},
                    ),
                ),
            },
        )
        registry.model_names = ["test-model"]
        registry.device = "cpu"
        registry.ensure_lora_loaded_async = AsyncMock(return_value=lora_state)

        preprocessor_registry = MagicMock()
        preprocessor_registry.has_tokenizer.return_value = False
        preprocessor_registry.has_preprocessor.return_value = False
        registry.preprocessor_registry = preprocessor_registry
        cpu_pool = ThreadPoolExecutor(max_workers=1)
        registry.postprocessor_registry = PostprocessorRegistry(cpu_pool)
        return registry

    @pytest.fixture
    def lora_client(self, mock_adapter: MagicMock) -> tuple[TestClient, MagicMock, MagicMock]:
        registry = self._make_lora_registry(mock_adapter)
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = registry
        return TestClient(app), registry, mock_adapter

    def test_profile_lora_id_is_mapped_to_lora_for_worker_routing(
        self, lora_client: tuple[TestClient, MagicMock, MagicMock]
    ) -> None:
        """Profile-resolved lora_id must surface as options['lora'] downstream."""
        client, registry, adapter = lora_client

        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"profile": "us-regulatory"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200, response.text

        # The LoRA ensure-loaded path must have been invoked with the resolved id.
        registry.ensure_lora_loaded_async.assert_called_once_with("test-model", "org/test-lora")

        # The adapter must receive options carrying both lora_id (legacy / profile
        # surface) and lora (worker-batcher routing key) with matching values.
        assert adapter.encode.call_count == 1
        forwarded_options = adapter.encode.call_args.kwargs["options"]
        assert forwarded_options.get("lora_id") == "org/test-lora"
        assert forwarded_options.get("lora") == "org/test-lora", (
            "options['lora'] must be set so the worker batcher routes to the "
            "LoRA-specific batch instead of base model weights (issue #94)."
        )

    def test_request_level_lora_id_is_mapped_to_lora(
        self, lora_client: tuple[TestClient, MagicMock, MagicMock]
    ) -> None:
        """A request-level options.lora_id override is also mapped to lora."""
        client, registry, adapter = lora_client

        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"lora_id": "org/override-lora"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200, response.text

        registry.ensure_lora_loaded_async.assert_called_once_with("test-model", "org/override-lora")
        forwarded_options = adapter.encode.call_args.kwargs["options"]
        assert forwarded_options.get("lora") == "org/override-lora"

    def test_no_lora_id_does_not_set_lora_key(self, lora_client: tuple[TestClient, MagicMock, MagicMock]) -> None:
        """Without a LoRA, options['lora'] must remain unset (no spurious routing)."""
        client, registry, adapter = lora_client

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200, response.text

        registry.ensure_lora_loaded_async.assert_not_called()
        forwarded_options = adapter.encode.call_args.kwargs["options"]
        assert "lora" not in forwarded_options
        assert "lora_id" not in forwarded_options

    def test_lora_loading_returns_503_and_does_not_invoke_adapter(self, mock_adapter: MagicMock) -> None:
        """When the LoRA is still loading, the request must short-circuit with 503.

        Locks in the invariant that the lora_id -> lora mapping is reachable
        only on the success path: a future refactor that moves the assignment
        above the early-return must not silently start dispatching requests
        whose LoRA is not ready.
        """
        registry = self._make_lora_registry(mock_adapter, lora_state=(False, True))
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = registry
        client = TestClient(app)

        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"profile": "us-regulatory"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "LORA_LOADING"
        assert response.headers.get("Retry-After") == "1"
        registry.ensure_lora_loaded_async.assert_called_once_with("test-model", "org/test-lora")
        mock_adapter.encode.assert_not_called()

    def test_lora_load_failure_returns_500_and_does_not_invoke_adapter(self, mock_adapter: MagicMock) -> None:
        """When the LoRA load has failed (not ready, not loading), respond 500 INFERENCE_ERROR.

        Complements ``test_lora_loading_returns_503_and_does_not_invoke_adapter``
        by exercising the second early-return branch in the endpoint: the
        adapter must not be invoked on a failed load, and no Retry-After hint
        is offered (this is a terminal failure, not a transient one).
        """
        registry = self._make_lora_registry(mock_adapter, lora_state=(False, False))
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = registry
        client = TestClient(app)

        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"profile": "us-regulatory"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 500
        assert response.json()["detail"]["code"] == "INFERENCE_ERROR"
        assert response.headers.get("Retry-After") != "1"
        registry.ensure_lora_loaded_async.assert_called_once_with("test-model", "org/test-lora")
        mock_adapter.encode.assert_not_called()
