from typing import Any
from unittest.mock import MagicMock

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

    dense = None
    if "dense" in output_types:
        dense = np.array([[0.1, 0.2, 0.3]] * batch_size, dtype=np.float32)

    sparse = None
    if "sparse" in output_types:
        sparse = [
            SparseVector(
                indices=np.array([1, 5, 10]),
                values=np.array([0.5, 0.3, 0.2], dtype=np.float32),
            )
            for _ in range(batch_size)
        ]

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
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
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
    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry
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


class TestOutputDtype:
    """Tests for output_dtype parameter (DESIGN.md Section 5.9)."""

    def test_default_dtype_is_float32(self, client: TestClient) -> None:
        """Default output dtype is float32."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_explicit_float32_dtype(self, client: TestClient) -> None:
        """Explicit float32 dtype works."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float32"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_float16_dtype(self, client: TestClient) -> None:
        """float16 dtype casting works."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float16"

    def test_int8_dtype(self, client: TestClient) -> None:
        """int8 dtype casting works for dense embeddings."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "int8"

    def test_binary_dtype(self, client: TestClient) -> None:
        """Binary dtype casting works for dense embeddings."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "binary"
        # Binary packing: 3 dims -> ceil(3/8) = 1 byte
        # But dims should still report original dimension
        assert data["items"][0]["dense"]["dims"] == 3

    def test_sparse_with_float16_dtype(self, client: TestClient) -> None:
        """Sparse embeddings support float16 dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["sparse"]["dtype"] == "float16"

    def test_sparse_int8_falls_back_to_float32(self, client: TestClient) -> None:
        """Sparse embeddings fall back to float32 for int8/binary."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Sparse falls back to float32 for int8
        assert data["items"][0]["sparse"]["dtype"] == "float32"

    def test_sparse_binary_falls_back_to_float32(self, client: TestClient) -> None:
        """Sparse embeddings fall back to float32 for binary."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["sparse"], "output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Sparse falls back to float32 for binary
        assert data["items"][0]["sparse"]["dtype"] == "float32"

    def test_multivector_with_float16_dtype(self, client: TestClient) -> None:
        """Multivector embeddings support float16 dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["multivector"]["dtype"] == "float16"

    def test_multivector_int8_quantization(self, client: TestClient) -> None:
        """Multivector embeddings support int8 quantization."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Multivector supports int8 quantization (per-token, like ColBERTv2)
        assert data["items"][0]["multivector"]["dtype"] == "int8"

    def test_multivector_binary_quantization(self, client: TestClient) -> None:
        """Multivector embeddings support binary quantization."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["multivector"], "output_dtype": "binary"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Multivector supports binary quantization (per-token)
        assert data["items"][0]["multivector"]["dtype"] == "binary"

    def test_multiple_output_types_with_dtype(self, client: TestClient) -> None:
        """Multiple output types all respect output_dtype."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"], "output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Both should be float16
        assert data["items"][0]["dense"]["dtype"] == "float16"
        assert data["items"][0]["sparse"]["dtype"] == "float16"

    def test_dtype_with_msgpack_response(self, client: TestClient) -> None:
        """Output dtype works with msgpack responses."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "float16"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        assert data["items"][0]["dense"]["dtype"] == "float16"
        # Values should be numpy array with float16 dtype
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float16

    def test_int8_values_are_integers(self, client: TestClient) -> None:
        """int8 dtype returns integer values."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "int8"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.int8

    def test_binary_values_are_packed(self, client: TestClient) -> None:
        """Binary dtype returns packed uint8 values."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"output_dtype": "binary"},
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        data = msgpack.unpackb(response.content, raw=False)
        values = data["items"][0]["dense"]["values"]
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.uint8
        # Original dims = 3, packed into ceil(3/8) = 1 byte
        assert len(values) == 1


class TestProfileOutputDtype:
    """Tests for profile-based output_dtype (profile > request > default)."""

    @pytest.fixture
    def mock_registry_with_profile(self, mock_adapter: MagicMock) -> MagicMock:
        """Registry with a model that has a quantized profile."""
        from concurrent.futures import ThreadPoolExecutor

        from sie_server.core.postprocessor_registry import PostprocessorRegistry

        registry = MagicMock(spec=ModelRegistry)
        registry.has_model.return_value = True
        registry.is_loaded.return_value = True
        registry.is_loading.return_value = False
        registry.is_unloading.return_value = False
        registry.is_failed.return_value = False
        registry.get_failure.return_value = None
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
            profiles={
                "default": ProfileConfig(adapter_path="test:TestAdapter", max_batch_tokens=8192),
                "quantized": ProfileConfig(
                    adapter_path="test:TestAdapter",
                    max_batch_tokens=8192,
                    adapter_options=AdapterOptions(runtime={"output_dtype": "int8"}),
                ),
            },
        )
        registry.model_names = ["test-model"]
        registry.device = "cpu"
        preprocessor_registry = MagicMock()
        preprocessor_registry.has_tokenizer.return_value = False
        preprocessor_registry.has_preprocessor.return_value = False
        registry.preprocessor_registry = preprocessor_registry
        # Use real postprocessor_registry for quantization
        cpu_pool = ThreadPoolExecutor(max_workers=1)
        registry.postprocessor_registry = PostprocessorRegistry(cpu_pool)
        return registry

    @pytest.fixture
    def client_with_profile(self, mock_registry_with_profile: MagicMock) -> TestClient:
        """Client with profile-aware registry."""
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = mock_registry_with_profile
        return TestClient(app)

    def test_default_profile_uses_float32(self, client_with_profile: TestClient) -> None:
        """Default profile with no output_dtype uses float32."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "float32"

    def test_quantized_profile_uses_int8(self, client_with_profile: TestClient) -> None:
        """Quantized profile with output_dtype=int8 returns int8."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {"options": {"profile": "quantized"}},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"][0]["dense"]["dtype"] == "int8"

    def test_request_overrides_profile(self, client_with_profile: TestClient) -> None:
        """Request output_dtype overrides profile output_dtype."""
        response = client_with_profile.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello"}],
                "params": {
                    "options": {"profile": "quantized"},
                    "output_dtype": "float16",
                },
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()
        # Request float16 overrides profile int8
        assert data["items"][0]["dense"]["dtype"] == "float16"
