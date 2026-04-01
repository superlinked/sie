from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


class TestTimingHeaders:
    """Tests for timing headers in encode responses (DESIGN.md Section 5.7)."""

    @pytest.fixture
    def mock_adapter_with_timing(self) -> MagicMock:
        """Create a mock adapter that returns test embeddings."""
        adapter = MagicMock()
        adapter.encode = MagicMock(side_effect=_mock_encode_impl)
        return adapter

    @pytest.fixture
    def mock_registry_with_worker(self, mock_adapter_with_timing: MagicMock) -> MagicMock:
        """Create a mock registry that uses the worker path with timing."""
        from sie_server.core.timing import RequestTiming
        from sie_server.core.worker import WorkerResult

        registry = MagicMock(spec=ModelRegistry)
        registry.has_model.return_value = True
        registry.is_loaded.return_value = True
        registry.is_loading.return_value = False
        registry.is_unloading.return_value = False
        registry.get.return_value = mock_adapter_with_timing
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

        # Set up preprocessor_registry to trigger worker path
        prepared_batch = MagicMock()
        prepared_item = MagicMock()
        prepared_item.cost = 5
        prepared_item.original_index = 0
        prepared_batch.items = [prepared_item]

        preprocessor_registry = MagicMock()
        # has_preprocessor returns True for "text", False for "image"
        preprocessor_registry.has_preprocessor.side_effect = lambda model, modality: modality == "text"
        preprocessor_registry.prepare = AsyncMock(return_value=prepared_batch)
        registry.preprocessor_registry = preprocessor_registry

        # Mock worker with timing
        timing = RequestTiming()
        timing.start_tokenization()
        timing.end_tokenization()
        timing.start_queue()
        timing.start_inference()
        timing.end_inference()
        timing.finish()

        from sie_server.core.inference_output import EncodeOutput

        worker_result = WorkerResult(
            output=EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                batch_size=1,
                dense_dim=3,
            ),
            timing=timing,
        )

        worker = MagicMock()
        worker.submit = AsyncMock(return_value=AsyncMock(return_value=worker_result)())
        registry.start_worker = AsyncMock(return_value=worker)

        return registry

    @pytest.fixture
    def client_with_worker(self, mock_registry_with_worker: MagicMock) -> TestClient:
        """Create test client with worker path enabled."""
        app = FastAPI()
        app.include_router(encode_router)
        app.state.registry = mock_registry_with_worker
        return TestClient(app)

    def test_timing_headers_present_json(self, client_with_worker: TestClient) -> None:
        """Timing headers are present in JSON responses."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Check timing headers present
        assert "x-queue-time" in response.headers
        assert "x-tokenization-time" in response.headers
        assert "x-inference-time" in response.headers
        assert "x-total-time" in response.headers

    def test_timing_headers_present_msgpack(self, client_with_worker: TestClient) -> None:
        """Timing headers are present in msgpack responses."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/msgpack"

        # Check timing headers present
        assert "x-queue-time" in response.headers
        assert "x-tokenization-time" in response.headers
        assert "x-inference-time" in response.headers
        assert "x-total-time" in response.headers

    def test_timing_header_values_format(self, client_with_worker: TestClient) -> None:
        """Timing header values are formatted with 2 decimal places."""
        response = client_with_worker.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # All timing values should be floats with 2 decimal places
        for header in ["x-queue-time", "x-tokenization-time", "x-inference-time", "x-total-time"]:
            value = response.headers[header]
            # Should be parseable as float
            float_val = float(value)
            assert float_val >= 0.0
            # Should have format X.XX (2 decimal places)
            parts = value.split(".")
            assert len(parts) == 2
            assert len(parts[1]) == 2

    def test_no_timing_headers_without_worker(self, client: TestClient) -> None:
        """No timing headers when using direct adapter path (no worker)."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

        # Timing headers should NOT be present (direct adapter path has no timing)
        # Note: Headers may be absent or have value "0.00" depending on implementation
        # The current implementation doesn't add headers when timing is None
        assert "x-queue-time" not in response.headers or response.headers.get("x-queue-time") == "0.00"
