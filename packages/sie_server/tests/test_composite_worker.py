"""Component processes of a composite bundle: shared hash, split capability."""

from __future__ import annotations

import pytest
from sie_server.api import ws as ws_module
from sie_server.composite_bundles import COMPONENT_ENV
from sie_server.config.model import (
    EmbeddingDim,
    EncodeTask,
    GenerateCapabilities,
    GenerateTask,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry
from sie_server.ipc_server import IpcServer
from sie_server.ipc_types import EnsureModelReadyRequest, SetPinnedModelsRequest, WorkerCapabilitiesRequest
from sie_server.queue_executor import QueueExecutor

BUNDLE = "default-sglang"
LLM = "Qwen/Qwen3-4B-Instruct-2507"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def _llm() -> ModelConfig:
    return ModelConfig(
        sie_id=LLM,
        hf_id=LLM,
        tasks=Tasks(
            generate=GenerateTask(context_length=32768, max_output_tokens=4096, capabilities=GenerateCapabilities())
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
            )
        },
    )


def _encoder() -> ModelConfig:
    return ModelConfig(
        sie_id=ENCODER,
        hf_id=ENCODER,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=384))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerAdapter",
                max_batch_tokens=4096,
            )
        },
    )


def _registry() -> ModelRegistry:
    registry = ModelRegistry(enable_hot_reload=False, pool_name="default")
    registry.add_config(_llm())
    registry.add_config(_encoder())
    return registry


def _view(monkeypatch: pytest.MonkeyPatch, component: str | None) -> ws_module.BundleConfigView:
    if component is None:
        monkeypatch.delenv(COMPONENT_ENV, raising=False)
    else:
        monkeypatch.setenv(COMPONENT_ENV, component)
    return ws_module.compute_bundle_config_view(_registry(), BUNDLE)


def test_component_processes_share_the_composite_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    whole = _view(monkeypatch, None)
    default = _view(monkeypatch, "default")
    sglang = _view(monkeypatch, "sglang")

    assert whole.bundle_config_hash
    assert default.bundle_config_hash == sglang.bundle_config_hash == whole.bundle_config_hash
    assert whole.unsupported_models == []
    assert default.unsupported_models == [LLM]
    assert sglang.unsupported_models == [ENCODER]


def test_control_plane_scope_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPONENT_ENV, "sglang")
    adapters = ws_module._bundle_adapter_modules(BUNDLE)

    view = ws_module.compute_bundle_config_view(_registry(), BUNDLE, control_plane_adapters=adapters)

    assert view.unsupported_models == [ENCODER]
    assert view.bundle_config_hash == ws_module.compute_bundle_config_view(_registry(), BUNDLE).bundle_config_hash


def test_served_models_follow_the_component(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()
    monkeypatch.setenv(COMPONENT_ENV, "default")
    assert ws_module.served_model_ids(registry, BUNDLE) == [ENCODER]
    assert ws_module.process_serves_model(registry, BUNDLE, ENCODER)
    assert not ws_module.process_serves_model(registry, BUNDLE, LLM)
    assert not ws_module.process_serves_model(registry, BUNDLE, "unknown/model")
    monkeypatch.delenv(COMPONENT_ENV)
    assert ws_module.served_model_ids(registry, BUNDLE) == sorted([ENCODER, LLM])


def _server(registry: ModelRegistry) -> IpcServer:
    # Handlers are called directly; the server never binds this socket.
    return IpcServer("unused.sock", QueueExecutor(registry), worker_id="w", bundle_id=BUNDLE)


@pytest.mark.asyncio
async def test_component_refuses_to_load_a_model_of_another_component(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPONENT_ENV, "sglang")
    registry = _registry()
    server = _server(registry)

    response = await server._handle_ensure_ready(EnsureModelReadyRequest(model_id=ENCODER))

    assert response.state == "retry_later"
    assert not registry.is_loading(ENCODER)
    assert not registry.is_loaded(ENCODER)


def test_worker_capabilities_list_only_the_models_this_process_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPONENT_ENV, "sglang")
    server = _server(_registry())

    capabilities = server._handle_worker_capabilities(WorkerCapabilitiesRequest())

    assert capabilities.supported_models == [LLM]
    assert capabilities.generation_models == [LLM]

    monkeypatch.setenv(COMPONENT_ENV, "default")
    capabilities = server._handle_worker_capabilities(WorkerCapabilitiesRequest())
    assert capabilities.supported_models == [ENCODER]
    assert capabilities.generation_models == []
    assert not capabilities.has_generation_models


def test_outside_a_composite_every_model_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(COMPONENT_ENV, raising=False)
    server = _server(_registry())

    assert not server._component_refuses(ENCODER)
    assert not server._component_refuses(LLM)


@pytest.mark.asyncio
async def test_component_pins_only_the_models_it_serves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPONENT_ENV, "default")
    registry = _registry()
    server = _server(registry)
    loads: list[str] = []

    async def fake_start_load(name: str, _device: str) -> bool:
        loads.append(name)
        return True

    monkeypatch.setattr(registry, "start_load_async", fake_start_load)

    response = await server._handle_set_pinned_models(SetPinnedModelsRequest(models=[LLM.lower(), ENCODER]))

    assert response.pinned_count == 1
    assert loads == [ENCODER]
