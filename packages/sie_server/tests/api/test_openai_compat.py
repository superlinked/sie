"""Tests for OpenAI-compatible embeddings endpoint."""

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.openai_compat import router as openai_router
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import LoadErrorClass, LoadFailure
from sie_server.core.oom import ResourceExhausted, ResourceExhaustedError
from sie_server.core.registry import ModelRegistry


def _mock_encode_impl(items: list[Any], output_types: list[str], **kwargs: Any) -> Any:
    """Implementation for mock encode - returns EncodeOutput."""
    from sie_server.core.inference_output import EncodeOutput

    batch_size = len(items)

    # Always return dense for OpenAI compat
    dense = np.array([[0.1, 0.2, 0.3]] * batch_size, dtype=np.float32)

    return EncodeOutput(
        dense=dense,
        sparse=None,
        multivector=None,
        batch_size=batch_size,
        dense_dim=3,
        multivector_token_dim=None,
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
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = mock_adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="text-embedding-3-small",
        hf_id="org/test",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["text-embedding-3-small"]
    # Mock preprocessor_registry to NOT have a tokenizer (use direct adapter path)
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry

    postprocessor_registry = MagicMock()
    postprocessor_registry.transform_sync.return_value = 0
    registry.postprocessor_registry = postprocessor_registry

    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(openai_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestOpenAIEmbeddings:
    """Test OpenAI-compatible /v1/embeddings endpoint."""

    def test_single_text_input(self, client: TestClient) -> None:
        """Test embedding a single text string."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello world",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["object"] == "list"
        assert data["model"] == "text-embedding-3-small"
        assert len(data["data"]) == 1
        assert data["data"][0]["object"] == "embedding"
        assert data["data"][0]["index"] == 0
        assert isinstance(data["data"][0]["embedding"], list)
        assert len(data["data"][0]["embedding"]) == 3
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]

    def test_multiple_text_inputs(self, client: TestClient) -> None:
        """Test embedding multiple texts in one request."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": ["Hello", "World", "Test"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["data"]) == 3
        for i, item in enumerate(data["data"]):
            assert item["index"] == i
            assert item["object"] == "embedding"
            assert len(item["embedding"]) == 3

    def test_base64_encoding_format(self, client: TestClient) -> None:
        """Test base64 encoding format."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello world",
                "encoding_format": "base64",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # base64 encoding returns string
        assert isinstance(data["data"][0]["embedding"], str)
        # Can decode base64
        import base64

        decoded = base64.b64decode(data["data"][0]["embedding"])
        # 3 floats * 4 bytes = 12 bytes
        assert len(decoded) == 12

    def test_float_encoding_format_explicit(self, client: TestClient) -> None:
        """Test explicit float encoding format."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello world",
                "encoding_format": "float",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data["data"][0]["embedding"], list)

    def test_model_not_found(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Test 404 when model doesn't exist."""
        mock_registry.has_model.return_value = False
        # Real registry raises for unknown models: guards the KeyError-500 regression.
        mock_registry.get_worker.side_effect = KeyError("Model 'nonexistent-model' not found in registry")

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent-model",
                "input": "Hello",
            },
        )

        assert response.status_code == 404
        data = response.json()
        # Top-level OpenAI envelope — no FastAPI {"detail": ...} wrapper.
        assert "detail" not in data
        error = data["error"]
        assert error["code"] == "model_not_found"
        assert error["type"] == "invalid_request_error"
        assert "message" in error

    def test_model_not_found_suggests_a_near_match(self, client: TestClient, mock_registry: MagicMock) -> None:
        """A near miss must name the id the caller probably meant."""
        mock_registry.has_model.return_value = False
        mock_registry.model_names = ["text-embedding-3-small", "BAAI/bge-m3"]

        response = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "Hello"})

        assert response.status_code == 404
        assert "Did you mean 'BAAI/bge-m3'?" in response.json()["error"]["message"]

    def test_model_not_found_stays_terse_with_no_near_match(self, client: TestClient, mock_registry: MagicMock) -> None:
        """No confident suggestion must add nothing to the message."""
        mock_registry.has_model.return_value = False
        mock_registry.model_names = ["text-embedding-3-small"]

        response = client.post("/v1/embeddings", json={"model": "xyzzy-nothing-alike", "input": "Hello"})

        assert response.status_code == 404
        assert "Did you mean" not in response.json()["error"]["message"]

    def test_empty_input_rejected(self, client: TestClient) -> None:
        """Test 400 when input is empty."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": [],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" not in data
        error = data["error"]
        assert error["code"] == "invalid_request"
        assert error["type"] == "invalid_request_error"
        assert error["param"] == "input"
        assert error["message"] == "Input cannot be empty"

    def test_unhonourable_dimensions_rejected(self, client: TestClient) -> None:
        """A width SIE cannot produce must fail, not be silently ignored.

        This endpoint used to accept ``dimensions`` and drop it, so a client
        migrating from ``text-embedding-3-*`` asked for 256 and received the
        model's native width with a 200 — vectors of the wrong width, written
        into a vector store with nothing to signal the mismatch.
        """
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
                "dimensions": 256,
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "unsupported_field"
        assert error["param"] == "dimensions"
        assert error["type"] == "invalid_request_error"
        # The message must name the real width so the caller can act on it.
        # Asserted as the full phrase: a bare "3" also matches the model name
        # text-embedding-3-small, so it would pass on a wrong width too.
        assert "3-dimensional embeddings" in error["message"]
        assert "set it to 3" in error["message"]

    def test_matching_dimensions_accepted(self, client: TestClient) -> None:
        """Pinning the model's real width is a no-op, not an error."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
                "dimensions": 3,
            },
        )

        assert response.status_code == 200
        assert len(response.json()["data"][0]["embedding"]) == 3

    def test_omitted_dimensions_still_works(self, client: TestClient) -> None:
        """The overwhelmingly common case stays untouched."""
        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "Hello"},
        )

        assert response.status_code == 200
        assert len(response.json()["data"][0]["embedding"]) == 3

    def test_dimensions_rejected_before_any_model_load(self, client: TestClient, mock_registry: MagicMock) -> None:
        """A bad 'dimensions' must not cost the caller a cold load first."""
        mock_registry.is_loaded.return_value = False
        mock_registry.start_load_async = AsyncMock()

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
                "dimensions": 1536,
            },
        )

        assert response.status_code == 400
        mock_registry.start_load_async.assert_not_awaited()

    def test_user_field_ignored(self, client: TestClient) -> None:
        """Test that user field is accepted but ignored."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
                "user": "test-user-123",
            },
        )

        assert response.status_code == 200

    def test_model_unloading_returns_503(self, client: TestClient, mock_registry: MagicMock) -> None:
        """Test 503 when model is being unloaded."""
        mock_registry.is_unloading.return_value = True

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
            },
        )

        assert response.status_code == 503
        data = response.json()
        assert "detail" not in data
        error = data["error"]
        assert error["code"] == "model_not_available"
        assert error["type"] == "server_error"
        assert "message" in error

    def test_inference_error_returns_top_level_openai_500(self, client: TestClient, mock_registry: MagicMock) -> None:
        """A generic inference failure emits a top-level OpenAI 500 envelope."""
        with patch(
            "sie_server.api.openai_compat.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=RuntimeError("sensitive embeddings detail"),
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "text-embedding-3-small", "input": "hello"},
            )

        assert response.status_code == 500
        data = response.json()
        assert "detail" not in data
        error = data["error"]
        assert error["code"] == "inference_error"
        assert error["type"] == "server_error"
        assert error["message"] == "internal error during embeddings"
        assert "sensitive embeddings detail" not in response.text


class TestOpenAIEmbeddingsModelLoadStates:
    """/v1/embeddings must mirror the native routes' non-blocking load contract.

    Previously a cold model BLOCKED the request on ``load_async`` (observed
    77s on large checkpoints) and a failed-load model re-ran the doomed
    blocking load on every request, returning a generic 503. The endpoint now
    uses ``ModelStateChecker``: instant 503 MODEL_LOADING + Retry-After for a
    cold model, terminal 502 MODEL_LOAD_FAILED (no Retry-After, no re-load)
    for a recorded failure — in the OpenAI-shaped error envelope, matching
    the gateway's /v1/embeddings so the same URL behaves identically
    single-node and clustered.
    """

    def test_cold_model_returns_503_model_loading(self, client: TestClient, mock_registry: MagicMock) -> None:
        """A cold model starts a background load and returns 503 immediately."""
        mock_registry.is_loaded.return_value = False
        mock_registry.is_loading.return_value = False
        mock_registry.device = "cpu"

        async def start_load_async_success(*args: Any, **kwargs: Any) -> bool:
            return True

        mock_registry.start_load_async = MagicMock(side_effect=start_load_async_success)
        mock_registry.load_async = MagicMock()

        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "Hello"},
        )

        # Non-blocking loading returns 503 + MODEL_LOADING immediately
        assert response.status_code == 503, response.text
        assert response.headers.get("Retry-After") == "5"
        error = response.json()["error"]
        assert error["code"] == "MODEL_LOADING"
        assert error["type"] == "server_error"
        assert "loading" in error["message"].lower()
        mock_registry.start_load_async.assert_called_once_with("text-embedding-3-small", device="cpu")
        # The old blocking path must be gone.
        mock_registry.load_async.assert_not_called()

    def test_failed_model_returns_terminal_502_without_reload(
        self, client: TestClient, mock_registry: MagicMock
    ) -> None:
        """A registry-recorded terminal failure returns 502 MODEL_LOAD_FAILED.

        Mirrors the native-route terminal-failure contract: no Retry-After header
        (the SDK uses its absence to short-circuit the MODEL_LOADING retry
        budget) and crucially NO repeated load attempt for a known-bad model.
        """
        failure = LoadFailure(
            error_class=LoadErrorClass.GATED,
            message="GatedModelError: HF_TOKEN missing or invalid for org/test",
            attempts=1,
            last_attempt_ts=time.monotonic(),
            cooldown_s=None,
        )
        mock_registry.is_loaded.return_value = False
        mock_registry.is_failed.return_value = True
        mock_registry.get_failure.return_value = failure

        # Should NOT trigger any load attempt, background or blocking.
        mock_registry.start_load_async = MagicMock()
        mock_registry.load_async = MagicMock()

        response = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "Hello"},
        )

        assert response.status_code == 502, response.text
        error = response.json()["error"]
        assert error["code"] == "MODEL_LOAD_FAILED"
        assert error["type"] == "server_error"
        assert error["error_class"] == "GATED"
        assert error["permanent"] is True
        assert error["attempts"] == 1
        # Critical: no Retry-After header so clients do not loop.
        assert "retry-after" not in {k.lower() for k in response.headers}
        # And no load of any kind was kicked off for the known-bad model.
        mock_registry.start_load_async.assert_not_called()
        mock_registry.load_async.assert_not_called()


class TestOpenAIResponseFormat:
    """Test that response matches OpenAI's exact format."""

    def test_response_structure(self, client: TestClient) -> None:
        """Verify response matches OpenAI's structure exactly."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Test",
            },
        )

        data = response.json()

        # Top-level fields
        assert set(data.keys()) == {"object", "data", "model", "usage"}
        assert data["object"] == "list"

        # Data item fields
        item = data["data"][0]
        assert set(item.keys()) == {"object", "embedding", "index"}
        assert item["object"] == "embedding"

        # Usage fields
        assert set(data["usage"].keys()) == {"prompt_tokens", "total_tokens"}

    def test_embedding_indices_sequential(self, client: TestClient) -> None:
        """Verify embedding indices are sequential starting from 0."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": ["a", "b", "c", "d"],
            },
        )

        data = response.json()
        indices = [item["index"] for item in data["data"]]
        assert indices == [0, 1, 2, 3]


class TestOpenAIEmbeddingsOom:
    """/v1/embeddings must map OOM to 503 RESOURCE_EXHAUSTED + Retry-After
    (OpenAI envelope), matching the native endpoints so the SDK auto-retries
    instead of treating it as a terminal 500. See #1604.
    """

    @pytest.mark.parametrize(
        "failure",
        [
            RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"),
            ResourceExhaustedError(
                "Resource exhausted: CUDA out of memory",
                marker=ResourceExhausted(operation="encode", attempts=4, original_message="CUDA out of memory"),
            ),
        ],
        ids=["raw-cuda-oom", "wrapped-resource-exhausted"],
    )
    def test_embeddings_oom_maps_to_503_resource_exhausted(
        self, client: TestClient, mock_registry: MagicMock, failure: BaseException
    ) -> None:
        # engine_config=None → oom_retry_after_from_registry falls back to the
        # module default (5), same as the native OOM test.
        mock_registry.engine_config = None

        with patch(
            "sie_server.api.openai_compat.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=failure,
        ):
            response = client.post(
                "/v1/embeddings",
                json={"model": "text-embedding-3-small", "input": "hello"},
            )

        assert response.status_code == 503, response.text
        assert response.headers.get("Retry-After") == "5"
        data = response.json()
        assert "detail" not in data
        error = data["error"]
        assert error["code"] == "RESOURCE_EXHAUSTED"
        assert error["type"] == "server_error"
