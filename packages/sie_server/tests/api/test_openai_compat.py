"""Tests for OpenAI-compatible embeddings endpoint."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.openai_compat import router as openai_router
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
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

        response = client.post(
            "/v1/embeddings",
            json={
                "model": "nonexistent-model",
                "input": "Hello",
            },
        )

        assert response.status_code == 404
        data = response.json()
        assert "error" in data["detail"]
        assert data["detail"]["error"]["code"] == "model_not_found"

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

    def test_dimensions_ignored(self, client: TestClient) -> None:
        """Test that dimensions parameter is ignored (not supported)."""
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "text-embedding-3-small",
                "input": "Hello",
                "dimensions": 256,  # Should be ignored
            },
        )

        assert response.status_code == 200
        # Still returns original dimensions (3)
        assert len(response.json()["data"][0]["embedding"]) == 3

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
