"""Tests for the FastAPI app factory."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


class TestAppFactory:
    """Tests for the FastAPI app factory."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """App factory returns a FastAPI application."""
        app = AppFactory.create_app(AppStateConfig())
        assert isinstance(app, FastAPI)

    def test_app_has_correct_metadata(self) -> None:
        """App has correct title and version."""
        app = AppFactory.create_app(AppStateConfig())
        assert app.title == "SIE Server"
        assert app.version == "0.1.0"

    def test_health_routes_registered(self, client: TestClient) -> None:
        """Health routes are registered in the app."""
        # Get OpenAPI schema to verify routes exist
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/healthz" in paths
        assert "/readyz" in paths


class TestNatsPullLoopGuard:
    """Tests for _nats_pull_loop RuntimeError when NATS is None."""

    @pytest.mark.asyncio
    async def test_nats_pull_loop_raises_when_nats_is_none(self, monkeypatch) -> None:
        """SIE_CLUSTER_ROUTING=queue with no NATS subscriber raises RuntimeError."""
        monkeypatch.setenv("SIE_CLUSTER_ROUTING", "queue")

        registry = MagicMock()

        with pytest.raises(RuntimeError, match="no NATS subscriber available"):
            async with AppFactory._nats_pull_loop(registry, None):
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_nats_pull_loop_yields_none_when_not_queue(self, monkeypatch) -> None:
        """When SIE_CLUSTER_ROUTING != queue, _nats_pull_loop yields None."""
        monkeypatch.delenv("SIE_CLUSTER_ROUTING", raising=False)

        registry = MagicMock()

        async with AppFactory._nats_pull_loop(registry, None) as pull_loop:
            assert pull_loop is None


class TestPreloadModels:
    """Tests for the _preload_models startup behavior."""

    @pytest.mark.asyncio
    async def test_preload_loads_models(self) -> None:
        """_preload_models calls load_async for each model."""
        registry = AsyncMock()
        config = AppStateConfig(device="cpu", preload_models=["model-a", "model-b"])

        await AppFactory._preload_models(registry, config)

        assert registry.load_async.call_count == 2
        registry.load_async.assert_any_call("model-a", "cpu")
        registry.load_async.assert_any_call("model-b", "cpu")

    @pytest.mark.asyncio
    async def test_preload_skips_when_none(self) -> None:
        """_preload_models is a no-op when preload_models is None."""
        registry = AsyncMock()
        config = AppStateConfig(device="cpu", preload_models=None)

        await AppFactory._preload_models(registry, config)

        registry.load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_preload_continues_on_failure(self) -> None:
        """Failed preload doesn't prevent other models from loading."""
        registry = AsyncMock()
        registry.load_async.side_effect = [RuntimeError("OOM"), AsyncMock()]
        config = AppStateConfig(device="cpu", preload_models=["model-a", "model-b"])

        await AppFactory._preload_models(registry, config)  # Should not raise

        assert registry.load_async.call_count == 2


class TestPreloadModelsEnvRoundTrip:
    """Tests for preload_models env var serialization."""

    def test_preload_models_env_round_trip(self, monkeypatch) -> None:
        """preload_models survives save_to_env_vars / from_env_vars cycle."""
        # Clean env first
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)

        config = AppStateConfig(preload_models=["model-a", "model-b"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models == ["model-a", "model-b"]

    def test_preload_models_none_round_trip(self, monkeypatch) -> None:
        """preload_models=None survives env round-trip."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)

        config = AppStateConfig(preload_models=None)
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models is None
