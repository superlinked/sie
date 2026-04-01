from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.models import router as models_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry


def _make_config(
    sie_id: str,
    hf_id: str,
    *,
    dense_dim: int | None = None,
    sparse_dim: int | None = None,
    multivector_dim: int | None = None,
    max_sequence_length: int | None = None,
    adapter_path: str = "test:Adapter",
) -> ModelConfig:
    return ModelConfig(
        sie_id=sie_id,
        hf_id=hf_id,
        tasks=Tasks(
            encode=EncodeTask(
                dense=EmbeddingDim(dim=dense_dim) if dense_dim else None,
                sparse=EmbeddingDim(dim=sparse_dim) if sparse_dim else None,
                multivector=EmbeddingDim(dim=multivector_dim) if multivector_dim else None,
            ),
        ),
        max_sequence_length=max_sequence_length,
        profiles={"default": ProfileConfig(adapter_path=adapter_path, max_batch_tokens=8192)},
    )


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock registry with test models."""
    registry = MagicMock(spec=ModelRegistry)
    registry.model_names = ["model-a", "model-b"]

    configs = {
        "model-a": _make_config(
            "model-a",
            "org/model-a",
            dense_dim=768,
            max_sequence_length=512,
            adapter_path="test:DenseAdapter",
        ),
        "model-b": _make_config(
            "model-b",
            "org/model-b",
            dense_dim=1024,
            sparse_dim=30522,
            multivector_dim=128,
            max_sequence_length=8192,
            adapter_path="test:MultiAdapter",
        ),
    }

    def get_config(name: str) -> ModelConfig:
        return configs[name]

    def has_model(name: str) -> bool:
        return name in configs

    def is_loaded(name: str) -> bool:
        return name == "model-a"  # Only model-a is loaded

    registry.get_config = get_config
    registry.has_model = has_model
    registry.is_loaded = is_loaded

    return registry


@pytest.fixture
def client(mock_registry: MagicMock) -> TestClient:
    """Create test client with mocked registry."""
    app = FastAPI()
    app.include_router(models_router)
    app.state.registry = mock_registry
    return TestClient(app)


class TestListModels:
    """Tests for GET /v1/models."""

    def test_list_models(self, client: TestClient) -> None:
        """Returns list of all models."""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 2

    def test_list_models_includes_info(self, client: TestClient) -> None:
        """Each model includes expected info."""
        response = client.get("/v1/models")
        data = response.json()

        # Find model-a
        model_a = next(m for m in data["models"] if m["name"] == "model-a")
        assert model_a["inputs"] == ["text"]
        assert model_a["outputs"] == ["dense"]
        assert model_a["dims"]["dense"] == 768
        assert model_a["loaded"] is True
        assert model_a["max_sequence_length"] == 512

    def test_list_models_shows_loaded_state(self, client: TestClient) -> None:
        """Shows which models are loaded."""
        response = client.get("/v1/models")
        data = response.json()

        model_a = next(m for m in data["models"] if m["name"] == "model-a")
        model_b = next(m for m in data["models"] if m["name"] == "model-b")

        assert model_a["loaded"] is True
        assert model_b["loaded"] is False


class TestGetModel:
    """Tests for GET /v1/models/{model}."""

    def test_get_model(self, client: TestClient) -> None:
        """Returns info for a specific model."""
        response = client.get("/v1/models/model-a")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "model-a"
        assert data["outputs"] == ["dense"]
        assert data["loaded"] is True

    def test_get_model_with_multiple_outputs(self, client: TestClient) -> None:
        """Returns all output types for multi-output model."""
        response = client.get("/v1/models/model-b")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "model-b"
        assert set(data["outputs"]) == {"dense", "sparse", "multivector"}
        assert data["dims"]["dense"] == 1024
        assert data["dims"]["sparse"] == 30522
        assert data["dims"]["multivector"] == 128

    def test_get_model_not_found(self, client: TestClient) -> None:
        """Returns 404 for unknown model."""
        response = client.get("/v1/models/unknown-model")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["code"] == "MODEL_NOT_FOUND"
