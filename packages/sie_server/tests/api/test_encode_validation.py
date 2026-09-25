import base64
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
from sie_server.types.inputs import MAX_ITEM_TEXT_BYTES

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


class TestMachineProfileValidation:
    """Tests for X-SIE-MACHINE-PROFILE header validation."""

    def test_no_profile_header_succeeds(self, client: TestClient) -> None:
        """Request without X-SIE-MACHINE-PROFILE header proceeds normally."""
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers=JSON_HEADERS,
        )
        assert response.status_code == 200

    def test_profile_header_with_no_gpu_fails(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with X-SIE-MACHINE-PROFILE when worker has no identity returns 400."""
        # Mock get_worker_gpu_type to return None (no GPU)
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: None,
        )
        # No SIE_MACHINE_PROFILE env var
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "no GPU" in data["detail"]["message"]
        assert data["detail"]["requested_profile"] == "l4"
        assert data["detail"]["worker_identity"] is None

    def test_profile_header_mismatch_fails(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with mismatched X-SIE-MACHINE-PROFILE returns 400."""
        # Mock get_worker_gpu_type to return different GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "a100-80gb",
        )
        # No SIE_MACHINE_PROFILE env var, so identity comes from detected GPU
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400
        data = response.json()
        assert "routing error" in data["detail"]["message"]
        assert data["detail"]["requested_profile"] == "l4"
        assert data["detail"]["worker_identity"] == "a100-80gb"

    def test_profile_header_matches_succeeds(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request with matching X-SIE-MACHINE-PROFILE proceeds normally."""
        # Mock get_worker_gpu_type to return matching GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "l4",
        )
        # No SIE_MACHINE_PROFILE env var, so identity comes from detected GPU
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 200

    def test_profile_header_case_insensitive(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Machine profile header comparison is case-insensitive."""
        # Mock get_worker_gpu_type to return lowercase GPU
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "a100-80gb",
        )
        # No SIE_MACHINE_PROFILE env var
        monkeypatch.delenv("SIE_MACHINE_PROFILE", raising=False)

        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "A100-80GB"},
        )
        assert response.status_code == 200

    def test_env_var_overrides_detected_gpu(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_MACHINE_PROFILE env var takes precedence over detected GPU."""
        # Mock get_worker_gpu_type to return l4
        monkeypatch.setattr(
            "sie_server.api.validation.get_worker_gpu_type",
            lambda: "l4",
        )
        # Set SIE_MACHINE_PROFILE to l4-spot (K8s scenario)
        monkeypatch.setenv("SIE_MACHINE_PROFILE", "l4-spot")

        # Request for l4 should fail (worker identity is l4-spot, not l4)
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4"},
        )
        assert response.status_code == 400

        # Request for l4-spot should succeed
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}]},
            headers={**JSON_HEADERS, "X-SIE-MACHINE-PROFILE": "l4-spot"},
        )
        assert response.status_code == 200


class TestUndecodableImageValidation:
    """Undecodable image bytes are a 400 INVALID_INPUT, never a 500.

    Valid base64 that decodes to non-image bytes used to raise PIL's
    ``UnidentifiedImageError`` (an ``OSError``) inside the adapter, missing the
    ``ValueError`` -> 400 mapping and surfacing as a 500 ``INFERENCE_ERROR``
    with a leaked ``BytesIO`` repr. Adapters now decode through
    ``sie_server.types.inputs.decode_image``, whose ``InvalidMediaError``
    reaches the client as 400 INVALID_INPUT with the same JSON-path message
    style as the corrupt-base64 ingress error.
    """

    def test_undecodable_image_bytes_return_400_with_json_path(
        self, client: TestClient, mock_adapter: MagicMock
    ) -> None:
        from sie_server.types.inputs import decode_image

        def _decode_like_a_real_adapter(items: list[Any], output_types: list[str], **kwargs: Any) -> Any:
            # Mirrors every image adapter/preprocessor: user bytes are opened
            # via the shared decode_image seam, which raises here.
            for i, item in enumerate(items):
                for j, img in enumerate(item.images or []):
                    decode_image(img, item_index=i, image_index=j)
            raise AssertionError("expected decode_image to reject the payload")

        mock_adapter.encode.side_effect = _decode_like_a_real_adapter

        not_an_image = base64.b64encode(b"valid base64, but not an image").decode()
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"images": [{"data": not_an_image}]}]},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["code"] == "INVALID_INPUT"
        assert "image data is not a decodable image" in detail["message"]
        assert "$.items[0].images[0].data" in detail["message"]
        assert "BytesIO" not in detail["message"]


class TestItemTextSize:
    """Each encode item's text and metadata are bounded at ingress, before any tokenization."""

    def test_text_at_the_cap_is_encoded(self, client: TestClient, mock_adapter: MagicMock) -> None:
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "x" * MAX_ITEM_TEXT_BYTES}]},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 200
        mock_adapter.encode.assert_called_once()

    def test_text_over_the_cap_is_rejected_before_encoding(
        self, client: TestClient, mock_adapter: MagicMock, mock_registry: MagicMock
    ) -> None:
        response = client.post(
            "/v1/encode/test-model",
            json={"items": [{"text": "Hello world"}, {"text": "中" * (MAX_ITEM_TEXT_BYTES // 3 + 1)}]},
            headers=JSON_HEADERS,
        )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "INVALID_INPUT",
            "message": f"Field 'items[1]' must hold at most {MAX_ITEM_TEXT_BYTES} bytes of UTF-8 text",
        }
        mock_adapter.encode.assert_not_called()
        mock_registry.preprocessor_registry.prepare.assert_not_called()
