import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.models import router as models_router
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.load_errors import LoadErrorClass, LoadFailure
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


def _make_generate_config(
    sie_id: str,
    hf_id: str,
    *,
    capabilities: GenerateCapabilities,
) -> ModelConfig:
    return ModelConfig(
        sie_id=sie_id,
        hf_id=hf_id,
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                capabilities=capabilities,
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
            ),
        },
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
    registry.is_loading = lambda _name: False
    registry.is_unloading = lambda _name: False
    registry.is_failed = lambda _name: False
    registry.get_failure = lambda _name: None

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


class TestModelStateField:
    """Coverage for the ``state`` and ``last_error`` fields on ``ModelInfo``."""

    def test_loaded_model_reports_loaded_state(self, client: TestClient) -> None:
        response = client.get("/v1/models/model-a")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "loaded"
        assert data["last_error"] is None

    def test_available_model_reports_available_state(self, client: TestClient) -> None:
        response = client.get("/v1/models/model-b")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "available"
        assert data["last_error"] is None

    def test_failed_model_reports_failed_state_with_error(self, mock_registry: MagicMock) -> None:
        """A registry-recorded failure surfaces via ``state`` + ``last_error``."""
        failure = LoadFailure(
            error_class=LoadErrorClass.GATED,
            message="GatedModelError: model is gated, set HF_TOKEN",
            attempts=2,
            last_attempt_ts=time.monotonic(),
            cooldown_s=None,
        )

        mock_registry.is_loaded = lambda name: False
        mock_registry.is_failed = lambda name: name == "model-b"
        mock_registry.get_failure = lambda name: failure if name == "model-b" else None

        app = FastAPI()
        app.include_router(models_router)
        app.state.registry = mock_registry
        client = TestClient(app)

        response = client.get("/v1/models/model-b")
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "failed"
        assert data["loaded"] is False
        assert data["last_error"] is not None
        assert data["last_error"]["code"] == "GATED"
        assert data["last_error"]["attempts"] == 2
        assert data["last_error"]["permanent"] is True
        assert "HF_TOKEN" in data["last_error"]["message"]

    def test_list_models_includes_failed_entries(self, mock_registry: MagicMock) -> None:
        """``GET /v1/models`` carries the per-model state for clients."""
        failure = LoadFailure(
            error_class=LoadErrorClass.OOM,
            message="RuntimeError: CUDA out of memory",
            attempts=1,
            last_attempt_ts=time.monotonic(),
            cooldown_s=60.0,
        )

        mock_registry.is_loaded = lambda name: False
        mock_registry.is_failed = lambda name: name == "model-a"
        mock_registry.get_failure = lambda name: failure if name == "model-a" else None

        app = FastAPI()
        app.include_router(models_router)
        app.state.registry = mock_registry
        client = TestClient(app)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()

        by_name = {m["name"]: m for m in data["models"]}
        assert by_name["model-a"]["state"] == "failed"
        assert by_name["model-a"]["last_error"]["code"] == "OOM"
        assert by_name["model-a"]["last_error"]["permanent"] is False
        assert by_name["model-b"]["state"] == "available"
        # Healthy models must not leak a stale error payload alongside
        # a failed sibling — guard against accidental cross-talk in the
        # list serializer.
        assert by_name["model-b"].get("last_error") is None


class TestModelCapabilitiesField:
    """Coverage for the ``capabilities`` field on ``ModelInfo``."""

    @pytest.fixture
    def caps_client(self) -> TestClient:
        """Client exposing a generate model and an encode-only model."""
        registry = MagicMock(spec=ModelRegistry)
        registry.model_names = ["gen-model", "enc-model"]

        configs = {
            "gen-model": _make_generate_config(
                "gen-model",
                "org/gen-model",
                capabilities=GenerateCapabilities(
                    streaming=False,
                    grammar=["json_schema", "regex"],
                    tools=True,
                    code=True,
                    sql=True,
                    guard=False,
                ),
            ),
            "enc-model": _make_config(
                "enc-model",
                "org/enc-model",
                dense_dim=768,
            ),
        }

        registry.get_config = lambda name: configs[name]
        registry.has_model = lambda name: name in configs
        registry.is_loaded = lambda _name: False
        registry.is_loading = lambda _name: False
        registry.is_unloading = lambda _name: False
        registry.is_failed = lambda _name: False
        registry.get_failure = lambda _name: None

        app = FastAPI()
        app.include_router(models_router)
        app.state.registry = registry
        return TestClient(app)

    def test_generate_model_surfaces_capabilities(self, caps_client: TestClient) -> None:
        """``code``/``sql``/``guard`` (and grammar/tools) surface for a generate model."""
        response = caps_client.get("/v1/models/gen-model")
        assert response.status_code == 200
        caps = response.json()["capabilities"]
        assert caps is not None
        assert caps["streaming"] is False
        assert caps["grammar"] == ["json_schema", "regex"]
        assert caps["tools"] is True
        assert caps["code"] is True
        assert caps["sql"] is True
        assert caps["guard"] is False

    def test_encode_model_has_no_capabilities(self, caps_client: TestClient) -> None:
        """Non-generate models omit generation capabilities (``None``)."""
        response = caps_client.get("/v1/models/enc-model")
        assert response.status_code == 200
        assert response.json()["capabilities"] is None

    def test_list_models_includes_capabilities(self, caps_client: TestClient) -> None:
        """``GET /v1/models`` carries per-model capabilities."""
        response = caps_client.get("/v1/models")
        assert response.status_code == 200
        by_name = {m["name"]: m for m in response.json()["models"]}
        assert by_name["gen-model"]["capabilities"]["streaming"] is False
        assert by_name["gen-model"]["capabilities"]["code"] is True
        assert by_name["enc-model"]["capabilities"] is None


class TestModelRevisionField:
    """Coverage for the additive ``revision`` field on ``ModelInfo`` (design §6.6)."""

    _SHA = "5617a9f61b028005a4858fdac845db406aefb181"

    @pytest.fixture
    def rev_client(self) -> TestClient:
        registry = MagicMock(spec=ModelRegistry)
        registry.model_names = ["pinned", "unpinned"]
        configs = {
            "pinned": ModelConfig(
                sie_id="pinned",
                hf_id="org/pinned",
                hf_revision=self._SHA,
                tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
                profiles={"default": ProfileConfig(adapter_path="test:Adapter", max_batch_tokens=8192)},
            ),
            "unpinned": _make_config("unpinned", "org/unpinned", dense_dim=768),
        }
        registry.get_config = lambda name: configs[name]
        registry.has_model = lambda name: name in configs
        registry.is_loaded = lambda _name: False
        registry.is_loading = lambda _name: False
        registry.is_unloading = lambda _name: False
        registry.is_failed = lambda _name: False
        registry.get_failure = lambda _name: None

        app = FastAPI()
        app.include_router(models_router)
        app.state.registry = registry
        return TestClient(app)

    def test_get_model_surfaces_pinned_revision(self, rev_client: TestClient) -> None:
        data = rev_client.get("/v1/models/pinned").json()
        assert data["revision"] == self._SHA

    def test_unpinned_model_revision_is_null(self, rev_client: TestClient) -> None:
        data = rev_client.get("/v1/models/unpinned").json()
        assert data["revision"] is None

    def test_list_models_carries_revision(self, rev_client: TestClient) -> None:
        by_name = {m["name"]: m for m in rev_client.get("/v1/models").json()["models"]}
        assert by_name["pinned"]["revision"] == self._SHA
        assert by_name["unpinned"]["revision"] is None
