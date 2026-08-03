"""Tests for the FastAPI app factory."""

import logging
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core import readiness
from sie_server.core.postprocessor_registry import PostprocessorRegistry


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


@asynccontextmanager
async def _async_value(value: Any = None) -> AsyncGenerator[Any, None]:
    yield value


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
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/healthz" in paths
        assert "/livez" in paths
        assert "/readyz" in paths


class TestAppLifespanState:
    @pytest.mark.asyncio
    async def test_lifespan_wires_ipc_server_as_queue_runtime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = MagicMock()
        ipc_server = MagicMock()

        monkeypatch.setattr(AppFactory, "_configure_torch_threads", MagicMock())
        monkeypatch.setattr(AppFactory, "_configure_cuda_defaults", MagicMock())
        monkeypatch.setattr(AppFactory, "_nvml", MagicMock(return_value=_async_value()))
        monkeypatch.setattr(AppFactory, "_model_registry", MagicMock(return_value=_async_value(registry)))
        monkeypatch.setattr(AppFactory, "_ipc_server", MagicMock(return_value=_async_value(ipc_server)))
        monkeypatch.setattr(AppFactory, "_graceful_shutdown", MagicMock(return_value=_async_value()))
        monkeypatch.setattr(AppFactory, "_readiness_handling", MagicMock(return_value=_async_value()))

        app = AppFactory.create_app(AppStateConfig())

        async with app.router.lifespan_context(app):
            assert app.state.registry is registry
            assert app.state.ipc_server is ipc_server
            assert app.state.queue_runtime is ipc_server


class TestAllRoutersAlwaysMounted:
    """Queue routing is the only supported mode, so routers are mounted
    unconditionally.

    /ws/status MUST be mounted because it is the gateway's
    worker-registration channel: the gateway opens
    `ws://<worker>/ws/status` to learn pool_name, bundle,
    machine_profile, and readiness. Without it, the gateway's
    WorkerRegistry stays empty and every request falls through to the
    202 "no queue worker available" branch. This test suite locks that
    contract in.
    """

    @staticmethod
    def _route_paths(app: FastAPI) -> set[str]:
        return {getattr(r, "path", "") for r in app.router.routes}

    def test_ws_status_is_always_mounted(self) -> None:
        app = AppFactory.create_app(AppStateConfig())
        assert "/ws/status" in self._route_paths(app)

    def test_probe_routes_are_mounted_and_application_metrics_route_is_absent(self) -> None:
        app = AppFactory.create_app(AppStateConfig())
        paths = self._route_paths(app)
        for path in ("/healthz", "/readyz"):
            assert path in paths, f"{path} missing — breaks K8s probes"
        assert "/metrics" not in paths

    def test_inference_routers_are_mounted(self) -> None:
        """Inference routers stay mounted unconditionally.

        In production these endpoints are not a real ingress (the Rust
        gateway is queue-only and publishes to JetStream, not HTTP),
        but the routes exist on every pod for local debugging and to
        keep the existing test/OpenAPI surface intact. If a future
        refactor deletes them, update this test along with the router
        imports in ``app_factory.create_app``.
        """
        app = AppFactory.create_app(AppStateConfig())
        paths = self._route_paths(app)
        for path in (
            "/v1/encode/{model:path}",
            "/v1/score/{model:path}",
            "/v1/extract/{model:path}",
            "/v1/embeddings",
            "/v1/models",
        ):
            assert path in paths, f"{path} router missing"


class TestIpcHeartbeatReadiness:
    """IPC heartbeat readiness is opt-in so direct HTTP stays usable."""

    @staticmethod
    def _short_socket_path() -> Path:
        # macOS caps AF_UNIX paths at 104 bytes; pytest temp roots can exceed it.
        return Path("/tmp") / f"sie-ipc-{uuid.uuid4().hex[:12]}.sock"  # noqa: S108

    @pytest.mark.asyncio
    async def test_direct_server_readiness_does_not_require_sidecar_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        monkeypatch.setenv("SIE_WORKER_ID", "worker-from-env")
        socket_path = self._short_socket_path()
        monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(socket_path))
        readiness.register_liveness_probe(None)
        readiness.mark_ready()

        try:
            async with AppFactory._ipc_server(MagicMock()) as server:
                assert server.worker_id == "worker-from-env"
                assert readiness.is_ready() is True
        finally:
            readiness.mark_not_ready()
            readiness.register_liveness_probe(None)
            socket_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_sidecar_deployments_gate_readiness_on_ipc_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_IPC_REQUIRE_HEARTBEAT", "true")
        socket_path = self._short_socket_path()
        monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(socket_path))
        readiness.register_liveness_probe(None)
        readiness.mark_ready()

        try:
            async with AppFactory._ipc_server(MagicMock()):
                assert readiness.is_ready() is False
        finally:
            readiness.mark_not_ready()
            readiness.register_liveness_probe(None)
            socket_path.unlink(missing_ok=True)


def _direct_encode_adapter_output(items: list[Any], output_types: list[str], **_kwargs: Any) -> Any:
    from sie_server.core.inference_output import EncodeOutput

    dense = None
    if "dense" in output_types:
        dense = np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32)

    return EncodeOutput(
        dense=dense,
        sparse=None,
        multivector=None,
        batch_size=len(items),
        dense_dim=3 if dense is not None else None,
        multivector_token_dim=None,
    )


def _direct_http_registry() -> tuple[MagicMock, ThreadPoolExecutor]:
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=_direct_encode_adapter_output)

    registry = MagicMock()
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test-model",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:Adapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"

    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry

    executor = ThreadPoolExecutor(max_workers=1)
    registry.postprocessor_registry = PostprocessorRegistry(executor)
    return registry, executor


class TestDirectServerHttpPath:
    """Direct Python HTTP inference remains usable without the worker-sidecar."""

    def test_app_factory_mounts_direct_generate_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        app = AppFactory.create_app(AppStateConfig())
        paths = {getattr(route, "path", "") for route in app.routes}
        assert "/v1/generate/{model:path}" in paths

        operation = app.openapi()["paths"]["/v1/generate/{model}"]["post"]
        assert operation["requestBody"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/GenerateRequestModel"
        }
        event_stream = operation["responses"]["200"]["content"]["text/event-stream"]
        assert event_stream["x-sie-event-schema"] == {"$ref": "#/components/schemas/GenerateChunk"}
        assert {"413", "502"} <= operation["responses"].keys()

    def test_app_factory_direct_encode_path_without_sidecar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        app = AppFactory.create_app(AppStateConfig())
        registry, executor = _direct_http_registry()
        app.state.registry = registry

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/encode/test-model",
                json={"items": [{"text": "direct python path"}]},
                headers={"Accept": "application/json"},
            )
            models_response = client.get("/v1/models")
        finally:
            executor.shutdown(wait=True)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert data["items"][0]["dense"]["dims"] == 3
        assert data["items"][0]["dense"]["values"] == pytest.approx([0.1, 0.2, 0.3])

        assert models_response.status_code == 200
        assert models_response.json()["models"][0]["name"] == "test-model"


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


class TestPinnedModels:
    """Tests for pinned model startup behavior."""

    @pytest.mark.asyncio
    async def test_pinned_models_are_loaded_at_startup(self) -> None:
        """_load_pinned_models calls load_async for each pinned model."""
        registry = AsyncMock()
        # is_loaded and resolve_model_id are sync; sync mocks avoid truthy coroutines.
        registry.is_loaded = MagicMock(return_value=False)
        registry.resolve_model_id = MagicMock(side_effect=lambda raw: raw)
        config = AppStateConfig(device="cpu", pinned_models=["model-a", "model-b"])

        await AppFactory._load_pinned_models(registry, config)

        assert registry.load_async.call_count == 2
        registry.load_async.assert_any_call("model-a", "cpu")
        registry.load_async.assert_any_call("model-b", "cpu")

    @pytest.mark.asyncio
    async def test_pinned_models_skips_when_none(self) -> None:
        """_load_pinned_models is a no-op when pinned_models is None."""
        registry = AsyncMock()
        config = AppStateConfig(device="cpu", pinned_models=None)

        await AppFactory._load_pinned_models(registry, config)

        registry.load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_models_continues_on_failure(self) -> None:
        """A failed pinned model load does not crash startup."""
        registry = AsyncMock()
        registry.is_loaded = MagicMock(return_value=False)
        registry.resolve_model_id = MagicMock(side_effect=lambda raw: raw)
        registry.load_async.side_effect = [RuntimeError("OOM"), AsyncMock()]
        config = AppStateConfig(device="cpu", pinned_models=["model-a", "model-b"])

        await AppFactory._load_pinned_models(registry, config)  # must not raise

        assert registry.load_async.call_count == 2

    @pytest.mark.asyncio
    async def test_pinned_model_in_preload_loaded_only_once(self) -> None:
        """A model in both preload and pinned is loaded exactly once."""
        registry = AsyncMock()
        # Simulate already loaded after preload (sync mocks: not async methods)
        registry.is_loaded = MagicMock(return_value=True)
        registry.resolve_model_id = MagicMock(side_effect=lambda raw: raw)
        config = AppStateConfig(device="cpu", pinned_models=["model-a"])

        await AppFactory._load_pinned_models(registry, config)

        # Already loaded, so load_async must NOT be called again
        registry.load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_pinned_profile_qualified_id_loads_bare_sie_id(self) -> None:
        """A profile-qualified pin (``sie_id:profile``) eager-loads the bare sie_id."""
        registry = AsyncMock()
        registry.is_loaded = MagicMock(return_value=False)
        # resolve_model_id strips the :profile suffix to the config's bare id.
        registry.resolve_model_id = MagicMock(return_value="org/model")
        config = AppStateConfig(device="cpu", pinned_models=["org/model:fast"])

        await AppFactory._load_pinned_models(registry, config)

        registry.resolve_model_id.assert_called_once_with("org/model:fast")
        registry.load_async.assert_called_once_with("org/model", "cpu")

    @pytest.mark.asyncio
    async def test_pinned_unresolved_id_skipped(self) -> None:
        """A pin with no matching config is skipped without crashing startup."""
        registry = AsyncMock()
        registry.resolve_model_id = MagicMock(return_value=None)
        config = AppStateConfig(device="cpu", pinned_models=["does/not-exist"])

        await AppFactory._load_pinned_models(registry, config)  # must not raise

        registry.load_async.assert_not_called()


class TestModelFilterEnvRoundTrip:
    """model_filter env serialization must distinguish [] (zero models) from None."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch) -> None:
        for var in (
            "SIE_PRELOAD_MODELS",
            "SIE_MODELS_DIR",
            "SIE_MODEL_FILTER",
            "SIE_DEVICE",
            "SIE_DEVICES",
            "SIE_POOL",
            "SIE_PINNED_MODELS",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_empty_model_filter_env_round_trip(self) -> None:
        """[] survives the uvicorn-factory env rebuild instead of collapsing to None.

        A zero-match bundle (transformers514 before its pilot model lands)
        produces model_filter=[]; losing it to None would re-advertise the
        full catalog from a worker that can serve none of it.
        """
        config = AppStateConfig(model_filter=[])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.model_filter == []

    def test_model_filter_env_round_trip(self) -> None:
        config = AppStateConfig(model_filter=["org/model-a", "org/model-b"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.model_filter == ["org/model-a", "org/model-b"]

    def test_absent_model_filter_stays_none(self, monkeypatch) -> None:
        """save_to_env_vars must also CLEAR a stale filter left in the env."""
        monkeypatch.setenv("SIE_MODEL_FILTER", "org/stale-model")
        config = AppStateConfig(model_filter=None)
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.model_filter is None


class TestPreloadModelsEnvRoundTrip:
    """Tests for preload_models env var serialization."""

    def test_preload_models_env_round_trip(self, monkeypatch) -> None:
        """preload_models survives save_to_env_vars / from_env_vars cycle."""
        # Clean env first
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_DEVICES", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)
        monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)

        config = AppStateConfig(preload_models=["model-a", "model-b"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models == ["model-a", "model-b"]
        assert restored.devices is None

    def test_devices_env_round_trip(self, monkeypatch) -> None:
        """Devices survives save_to_env_vars / from_env_vars cycle."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_DEVICES", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)

        config = AppStateConfig(device="cuda", devices=["cuda:0", "cuda:1"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.device == "cuda"
        assert restored.devices == ["cuda:0", "cuda:1"]
        assert restored.pool_name is None

    def test_devices_env_derives_default_device_family(self, monkeypatch) -> None:
        """SIE_DEVICES alone turns the default CPU family into the concrete device family."""
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.setenv("SIE_DEVICES", "cuda:0,cuda:1")

        restored = AppStateConfig.from_env_vars()

        assert restored.device == "cuda"
        assert restored.devices == ["cuda:0", "cuda:1"]

    def test_devices_reject_mismatched_device_family(self) -> None:
        with pytest.raises(ValueError, match="must match SIE_DEVICES"):
            AppStateConfig(device="mps", devices=["cuda:0"])

    def test_devices_reject_mixed_families_with_default_device(self) -> None:
        with pytest.raises(ValueError, match="must match SIE_DEVICES"):
            AppStateConfig(devices=["cuda:0", "mps:0"])

    def test_pool_name_env_round_trip(self, monkeypatch) -> None:
        """SIE_POOL survives save_to_env_vars / from_env_vars cycle."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_DEVICES", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)

        config = AppStateConfig(pool_name="customer-a")
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.pool_name == "customer-a"

    def test_preload_models_none_round_trip(self, monkeypatch) -> None:
        """preload_models=None survives env round-trip."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_DEVICES", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)
        monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)

        config = AppStateConfig(preload_models=None)
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models is None
        assert restored.pool_name is None
        assert restored.devices is None

    def test_pinned_models_env_round_trip(self, monkeypatch) -> None:
        """SIE_PINNED_MODELS='a,b' round-trips to pinned_models == ['a', 'b']."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)
        monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)

        config = AppStateConfig(pinned_models=["a", "b"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.pinned_models == ["a", "b"]

    def test_pinned_models_none_round_trip(self, monkeypatch) -> None:
        """pinned_models=None survives env round-trip (unset => None)."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)
        monkeypatch.delenv("SIE_PINNED_MODELS", raising=False)

        config = AppStateConfig(pinned_models=None)
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.pinned_models is None


class TestModelRegistryConfig:
    class _FakeRegistry:
        def get_configs_snapshot(self) -> dict[str, ModelConfig]:
            return {}

        async def start_memory_monitor(self) -> None:
            pass

        async def start_idle_evictor(self) -> None:
            pass

        async def start_hot_reload(self) -> None:
            pass

        async def stop_memory_monitor(self) -> None:
            pass

        async def stop_idle_evictor(self) -> None:
            pass

        async def stop_hot_reload(self) -> None:
            pass

        async def unload_all_async(self) -> None:
            pass

        def is_loaded(self, name: str) -> bool:
            return False

        def resolve_model_id(self, raw_id: str) -> str | None:
            return raw_id

        async def load_async(self, name: str, device: str) -> None:
            pass

    @pytest.mark.asyncio
    async def test_model_registry_receives_pool_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created_kwargs: dict[str, Any] = {}

        def fake_model_registry(**kwargs: Any) -> TestModelRegistryConfig._FakeRegistry:
            created_kwargs.update(kwargs)
            return TestModelRegistryConfig._FakeRegistry()

        monkeypatch.setattr("sie_server.app.app_factory.ModelRegistry", fake_model_registry)

        async with AppFactory._model_registry(AppStateConfig(pool_name="pool-a")):
            pass

        assert created_kwargs["pool_name"] == "pool-a"

    @pytest.mark.asyncio
    async def test_model_registry_receives_pinned_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ModelRegistry constructor receives pinned_models from AppStateConfig."""
        created_kwargs: dict[str, Any] = {}

        def fake_model_registry(**kwargs: Any) -> TestModelRegistryConfig._FakeRegistry:
            created_kwargs.update(kwargs)
            return TestModelRegistryConfig._FakeRegistry()

        monkeypatch.setattr("sie_server.app.app_factory.ModelRegistry", fake_model_registry)

        async with AppFactory._model_registry(AppStateConfig(pinned_models=["model-x", "model-y"])):
            pass

        assert created_kwargs["pinned_models"] == ["model-x", "model-y"]


class TestConfigureTorchThreads:
    def test_default_uses_half_cpu_count(self, monkeypatch) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        set_interop = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", set_interop)
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        set_interop.assert_called_once_with(1)

    def test_default_floor_when_cpu_count_is_none(self, monkeypatch) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: None)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(2)

    def test_env_override_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "3")
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(3)

    def test_invalid_env_override_falls_back(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "abc")
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        assert any("not a positive integer" in r.message for r in caplog.records)

    def test_zero_env_override_falls_back(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "0")
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        assert any("not a positive integer" in r.message for r in caplog.records)

    def test_interop_runtime_error_is_swallowed(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", MagicMock())

        def _raise(_n: int) -> None:
            raise RuntimeError("parallel runtime already started")

        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", _raise)
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()  # must not raise
        assert any("set_num_interop_threads" in r.message for r in caplog.records)
