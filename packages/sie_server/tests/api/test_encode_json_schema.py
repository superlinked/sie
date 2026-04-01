import json
import math
from typing import Any
from unittest.mock import MagicMock

import msgpack_numpy as m
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.encode import router as encode_router
from sie_server.config.model import (
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


class TestJsonResponseSchema:
    """Tests for JSON response schema validation and human-readability.

    Verifies that:
    - JSON responses match documented OpenAPI schemas
    - Embeddings are human-readable float lists (not binary blobs)
    - All documented fields are present
    """

    def test_json_encode_response_matches_openapi_schema(self, client: TestClient) -> None:
        """JSON response structure matches EncodeResponseModel schema."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"id": "doc-1", "text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        # Top-level required fields per EncodeResponseModel
        assert "model" in data
        assert "items" in data
        assert isinstance(data["model"], str)
        assert isinstance(data["items"], list)

        # Per-item structure per EncodeResultModel
        item = data["items"][0]
        assert "id" in item or item.get("id") is None  # Optional field
        # Dense embedding per DenseVectorModel
        assert "dense" in item
        dense = item["dense"]
        assert "dims" in dense
        assert "dtype" in dense
        assert "values" in dense
        assert isinstance(dense["dims"], int)
        assert isinstance(dense["dtype"], str)
        assert dense["dtype"] == "float32"

    def test_json_dense_values_are_human_readable_floats(self, client: TestClient) -> None:
        """Dense embedding values in JSON are human-readable float lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        values = data["items"][0]["dense"]["values"]
        # Must be a plain list (not binary blob, not base64)
        assert isinstance(values, list)
        # Each element must be a number
        for val in values:
            assert isinstance(val, int | float)
            # Should be readable (not NaN, not inf)
            assert not math.isnan(val)
            assert not math.isinf(val)

    def test_json_sparse_values_are_human_readable(self, client: TestClient) -> None:
        """Sparse embedding indices/values in JSON are human-readable lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello world"}],
                "params": {"output_types": ["sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        sparse = data["items"][0]["sparse"]
        # Check SparseVectorModel schema
        assert "indices" in sparse
        assert "values" in sparse
        assert "dtype" in sparse

        # Indices must be a list of integers
        assert isinstance(sparse["indices"], list)
        for idx in sparse["indices"]:
            assert isinstance(idx, int)
            assert idx >= 0

        # Values must be a list of floats
        assert isinstance(sparse["values"], list)
        for val in sparse["values"]:
            assert isinstance(val, int | float)

    def test_json_multivector_values_are_human_readable(self, client: TestClient) -> None:
        """Multivector embedding values in JSON are human-readable nested lists."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Hello world"}],
                "params": {"output_types": ["multivector"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        mv = data["items"][0]["multivector"]
        # Check MultiVectorModel schema
        assert "token_dims" in mv
        assert "num_tokens" in mv
        assert "dtype" in mv
        assert "values" in mv

        assert isinstance(mv["token_dims"], int)
        assert isinstance(mv["num_tokens"], int)
        assert isinstance(mv["values"], list)

        # Values is list[list[float]] - nested structure
        for token_embedding in mv["values"]:
            assert isinstance(token_embedding, list)
            for val in token_embedding:
                assert isinstance(val, int | float)

    def test_json_response_is_valid_json_string(self, client: TestClient) -> None:
        """Response can be parsed as JSON and re-serialized."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Test"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Should not raise - content is valid JSON
        data = json.loads(response.text)
        # Re-serialization should work
        json_str = json.dumps(data, indent=2)
        assert len(json_str) > 0
        # Round-trip should preserve data
        assert json.loads(json_str) == data

    def test_json_float16_values_serialized_as_floats(self, client: TestClient) -> None:
        """float16 dtype values are serialized as readable floats in JSON."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Test"}],
                "params": {"output_dtype": "float16"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["items"][0]["dense"]["dtype"] == "float16"
        values = data["items"][0]["dense"]["values"]
        # Values should still be readable numbers in JSON (not binary)
        assert isinstance(values, list)
        for val in values:
            assert isinstance(val, int | float)

    def test_json_int8_values_serialized_as_integers(self, client: TestClient) -> None:
        """int8 dtype values are serialized as readable integers in JSON."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"text": "Test"}],
                "params": {"output_dtype": "int8"},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        assert data["items"][0]["dense"]["dtype"] == "int8"
        values = data["items"][0]["dense"]["values"]
        # Values should be integers
        assert isinstance(values, list)
        for val in values:
            assert isinstance(val, int)
            assert -128 <= val <= 127

    def test_json_contains_all_documented_fields(self, client: TestClient) -> None:
        """JSON response contains all fields documented in OpenAPI schema."""
        response = client.post(
            "/v1/encode/test-model",
            json={
                "items": [{"id": "doc-1", "text": "Hello"}],
                "params": {"output_types": ["dense", "sparse"]},
            },
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200
        data = response.json()

        # Response level - EncodeResponseModel
        assert "model" in data
        assert "items" in data
        # timing is optional

        # Item level - EncodeResultModel
        item = data["items"][0]
        assert item.get("id") == "doc-1"  # ID preserved from request

        # Dense - DenseVectorModel
        dense = item["dense"]
        assert set(dense.keys()) >= {"dims", "dtype", "values"}

        # Sparse - SparseVectorModel
        sparse = item["sparse"]
        assert set(sparse.keys()) >= {"dtype", "indices", "values"}
