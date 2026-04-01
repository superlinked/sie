"""Tests for WebSocket status endpoint."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sie_server.api.ws import get_model_status, get_server_info
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.observability.gpu import normalize_gpu_type


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the SIE server with lifespan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        models_dir.mkdir()
        app = AppFactory.create_app(AppStateConfig(models_dir=models_dir))
        with TestClient(app) as client:
            yield client


class TestServerInfo:
    """Tests for server info collection."""

    def test_get_server_info_returns_expected_fields(self) -> None:
        """Server info contains all required fields."""
        info = get_server_info()

        assert "version" in info
        assert "uptime_seconds" in info
        assert "user" in info
        assert "working_dir" in info
        assert "pid" in info

    def test_server_info_types(self) -> None:
        """Server info fields have correct types."""
        info = get_server_info()

        assert isinstance(info["version"], str)
        assert isinstance(info["uptime_seconds"], int)
        assert isinstance(info["user"], str)
        assert isinstance(info["working_dir"], str)
        assert isinstance(info["pid"], int)

    def test_uptime_is_non_negative(self) -> None:
        """Uptime should be non-negative."""
        info = get_server_info()
        assert info["uptime_seconds"] >= 0


class TestModelStatus:
    """Tests for model status collection."""

    def test_empty_registry_returns_empty_list(self) -> None:
        """Empty registry returns empty model list."""
        # Create an empty registry directly
        registry = ModelRegistry()
        models = get_model_status(registry)
        assert models == []

    def test_model_status_shows_available_state(self) -> None:
        """Unloaded model shows as 'available'."""
        registry = ModelRegistry()
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )
        registry.add_config(config)

        models = get_model_status(registry)
        assert len(models) == 1
        assert models[0]["name"] == "test-model"
        assert models[0]["state"] == "available"

    def test_model_status_shows_loading_state(self) -> None:
        """Model being loaded shows as 'loading'."""
        registry = ModelRegistry()
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )
        registry.add_config(config)

        # Manually set loading state
        registry._loading.add("test-model")

        models = get_model_status(registry)
        assert len(models) == 1
        assert models[0]["state"] == "loading"

    def test_model_status_shows_unloading_state(self) -> None:
        """Model being unloaded shows as 'unloading'."""
        registry = ModelRegistry()
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )
        registry.add_config(config)

        # Manually set unloading state
        registry._unloading.add("test-model")

        models = get_model_status(registry)
        assert len(models) == 1
        assert models[0]["state"] == "unloading"

    def test_model_status_loading_takes_precedence(self) -> None:
        """Loading state takes precedence over unloading."""
        registry = ModelRegistry()
        config = ModelConfig(
            sie_id="test-model",
            hf_id="org/test",
            tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
            profiles={
                "default": ProfileConfig(
                    adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                    max_batch_tokens=8192,
                )
            },
        )
        registry.add_config(config)

        # Set both loading and unloading (shouldn't happen normally, but test precedence)
        registry._loading.add("test-model")
        registry._unloading.add("test-model")

        models = get_model_status(registry)
        assert len(models) == 1
        assert models[0]["state"] == "loading"


class TestWebSocketEndpoint:
    """Tests for the WebSocket status endpoint."""

    def test_websocket_connects_and_receives_status(self, client: TestClient) -> None:
        """WebSocket endpoint accepts connection and sends status."""
        with client.websocket_connect("/ws/status") as websocket:
            # Receive first status message
            data = websocket.receive_json()

            # Verify structure
            assert "timestamp" in data
            assert "server" in data
            assert "gpus" in data
            assert "models" in data
            assert "counters" in data
            assert "histograms" in data

            # Verify server info
            server = data["server"]
            assert "version" in server
            assert "uptime_seconds" in server
            assert "user" in server
            assert "working_dir" in server
            assert "pid" in server

    @patch("sie_server.api.ws.asyncio.sleep", new_callable=AsyncMock)
    def test_websocket_sends_multiple_updates(self, mock_sleep: AsyncMock, client: TestClient) -> None:
        """WebSocket sends multiple status updates."""
        with client.websocket_connect("/ws/status") as websocket:
            # Receive two messages
            data1 = websocket.receive_json()
            data2 = websocket.receive_json()

            # Both should have timestamps
            assert "timestamp" in data1
            assert "timestamp" in data2

            # Second timestamp should be >= first
            assert data2["timestamp"] >= data1["timestamp"]
        mock_sleep.assert_called()

    def test_websocket_gpus_is_list(self, client: TestClient) -> None:
        """GPUs field is always a list."""
        with client.websocket_connect("/ws/status") as websocket:
            data = websocket.receive_json()
            assert isinstance(data["gpus"], list)

    def test_websocket_models_is_list(self, client: TestClient) -> None:
        """Models field is always a list."""
        with client.websocket_connect("/ws/status") as websocket:
            data = websocket.receive_json()
            assert isinstance(data["models"], list)

    def test_websocket_has_router_fields(self, client: TestClient) -> None:
        """Status includes router-friendly summary fields."""
        with client.websocket_connect("/ws/status") as websocket:
            data = websocket.receive_json()

            # Router-friendly fields at top level
            assert "machine_profile" in data
            assert "gpu_count" in data
            assert "loaded_models" in data

            # Types
            assert isinstance(data["machine_profile"], str)
            assert isinstance(data["gpu_count"], int)
            assert isinstance(data["loaded_models"], list)

            # gpu_count should be non-negative
            assert data["gpu_count"] >= 0

            # Per-model queue_depth is in models array, not aggregated at top level
            assert "queue_depth" not in data
            for model in data.get("models", []):
                assert "queue_depth" in model


class TestGpuTypeNormalization:
    """Tests for GPU type normalization."""

    def test_normalize_l4(self) -> None:
        """L4 GPU variants normalize correctly."""
        assert normalize_gpu_type("NVIDIA L4") == "l4"
        assert normalize_gpu_type("nvidia l4") == "l4"

    def test_normalize_t4(self) -> None:
        """T4 GPU variants normalize correctly."""
        assert normalize_gpu_type("NVIDIA T4") == "t4"
        assert normalize_gpu_type("Tesla T4") == "t4"

    def test_normalize_a10g(self) -> None:
        """A10G GPU normalizes correctly."""
        assert normalize_gpu_type("NVIDIA A10G") == "a10g"

    def test_normalize_a100_40gb(self) -> None:
        """A100 40GB variants normalize correctly."""
        assert normalize_gpu_type("NVIDIA A100-SXM4-40GB") == "a100-40gb"
        assert normalize_gpu_type("NVIDIA A100-PCIE-40GB") == "a100-40gb"

    def test_normalize_a100_80gb(self) -> None:
        """A100 80GB variants normalize correctly."""
        assert normalize_gpu_type("NVIDIA A100-SXM4-80GB") == "a100-80gb"
        assert normalize_gpu_type("NVIDIA A100-PCIE-80GB") == "a100-80gb"

    def test_normalize_h100(self) -> None:
        """H100 GPU variants normalize correctly."""
        assert normalize_gpu_type("NVIDIA H100") == "h100"
        assert normalize_gpu_type("NVIDIA H100-SXM5-80GB") == "h100"

    def test_normalize_unknown_gpu(self) -> None:
        """Unknown GPU returns 'unknown'."""
        assert normalize_gpu_type("Some Future GPU") == "unknown"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIE_GPU_TYPE env var overrides auto-detection."""
        monkeypatch.setenv("SIE_GPU_TYPE", "custom-gpu")
        assert normalize_gpu_type("NVIDIA L4") == "custom-gpu"


class TestMultipleWebSocketClients:
    """Tests for multiple WebSocket client support."""

    def test_multiple_clients_connect(self, client: TestClient) -> None:
        """Multiple WebSocket clients can connect simultaneously."""
        with client.websocket_connect("/ws/status") as ws1:
            with client.websocket_connect("/ws/status") as ws2:
                # Both should receive status
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()

                assert "timestamp" in data1
                assert "timestamp" in data2
                assert "machine_profile" in data1
                assert "machine_profile" in data2
