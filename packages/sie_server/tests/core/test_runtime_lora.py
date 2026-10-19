"""Registry residency contracts for engine-owned runtime LoRAs."""

import asyncio
import threading
from unittest.mock import Mock

import pytest
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.core.model_loader import LoadedLora, LoadedModel
from sie_server.core.registry import ModelRegistry


def registry_with_loras(monkeypatch, *, capacity=1, preload=None):
    adapter = SGLangGenerationAdapter(
        "base",
        mem_fraction_static=0.5,
        lora_paths={"a": "org/a", "b": "org/b", "c": "org/c"},
        preload_loras=preload or [],
        max_lora_rank=8,
        lora_target_modules=["q_proj"],
        max_loras_per_batch=1,
        max_loaded_loras=capacity,
    )
    adapter._lora.started()
    monkeypatch.setattr(adapter, "fetch_lora", Mock(return_value="/cache/adapter"))
    monkeypatch.setattr(adapter, "discard_lora_source", Mock(wraps=adapter.discard_lora_source))
    monkeypatch.setattr(adapter._lora, "_mutate", Mock())
    config = Mock()
    config.lora_revisions.return_value = dict.fromkeys(["org/a", "org/b", "org/c"])
    runtime = adapter.lora_runtime_config()
    loaded = LoadedModel(config, adapter, "cpu", max_loras=capacity, lora_runtime=runtime)
    for source in runtime.preloaded:
        loaded.loras[source] = LoadedLora(source, pinned=True)
    registry = ModelRegistry()
    registry._configs["base"] = config
    registry._loaded["base"] = loaded
    return registry, loaded, adapter


async def finish_loads(registry):
    await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 2)


@pytest.mark.asyncio
async def test_first_request_returns_loading_then_alias_and_source_are_ready(monkeypatch):
    registry, _, adapter = registry_with_loras(monkeypatch)
    assert await registry.ensure_lora_loaded_async("base", "a") == (False, True)
    assert await registry.ensure_lora_loaded_async("base", "org/a") == (False, True)
    await finish_loads(registry)
    assert await registry.ensure_lora_loaded_async("base", "a") == (True, False)
    adapter.fetch_lora.assert_called_once_with("org/a", None)
    assert registry.loaded_lora_names("base") == ["a"]


@pytest.mark.asyncio
async def test_loading_slot_cannot_be_evicted_or_overcommitted(monkeypatch):
    registry, loaded, adapter = registry_with_loras(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def fetch(*args):
        entered.set()
        assert release.wait(2)
        return "/cache/a"

    monkeypatch.setattr(adapter, "fetch_lora", Mock(side_effect=fetch))
    assert await registry.ensure_lora_loaded_async("base", "a") == (False, True)
    await asyncio.to_thread(entered.wait, 1)
    try:
        assert await registry.ensure_lora_loaded_async("base", "b") == (False, True)
        assert list(loaded.loras) == ["org/a"]
        assert len(registry._background_tasks) == 1
    finally:
        release.set()
        await finish_loads(registry)
    assert registry.is_lora_loaded("base", "a")


@pytest.mark.asyncio
async def test_lru_removes_confirmed_victim_before_replacement_and_preserves_pins(monkeypatch):
    registry, loaded, adapter = registry_with_loras(monkeypatch, capacity=2, preload=["a"])
    await registry.ensure_lora_loaded_async("base", "b")
    await finish_loads(registry)
    adapter._lora._mutate.reset_mock()
    await registry.ensure_lora_loaded_async("base", "c")
    assert not registry.is_lora_loaded("base", "b")
    await finish_loads(registry)
    assert list(loaded.loras) == ["org/a", "org/c"]
    assert [call.args[0] for call in adapter._lora._mutate.call_args_list] == [
        "unload_lora_adapter",
        "load_lora_adapter",
    ]
    assert adapter._lora._mutate.call_args_list[0].args[1] == {"lora_name": "b"}


@pytest.mark.asyncio
async def test_all_pinned_capacity_returns_retry_without_fetch(monkeypatch):
    registry, loaded, adapter = registry_with_loras(monkeypatch, capacity=2, preload=["a"])
    loaded.max_loras = 1  # Defensive runtime state; constructor rejects this declaration.
    assert await registry.ensure_lora_loaded_async("base", "b") == (False, True)
    adapter.fetch_lora.assert_not_called()
    assert list(loaded.loras) == ["org/a"]


@pytest.mark.asyncio
async def test_failed_unload_retains_capacity_and_does_not_attach_replacement(monkeypatch):
    registry, loaded, adapter = registry_with_loras(monkeypatch)
    await registry.ensure_lora_loaded_async("base", "a")
    await finish_loads(registry)
    adapter._lora._mutate.reset_mock()
    adapter._lora._mutate.side_effect = TimeoutError("uncertain removal")
    await registry.ensure_lora_loaded_async("base", "b")
    await finish_loads(registry)
    assert list(loaded.loras) == ["org/a"]
    assert loaded.loras["org/a"].loading
    assert await registry.ensure_lora_loaded_async("base", "c") == (False, True)
    adapter._lora._mutate.assert_called_once_with("unload_lora_adapter", {"lora_name": "a"})
    adapter.discard_lora_source.assert_called_once_with("org/b")
    assert registry.loaded_lora_names("base") == []


@pytest.mark.asyncio
async def test_failed_fetch_restores_unmodified_victim(monkeypatch):
    registry, loaded, adapter = registry_with_loras(monkeypatch)
    await registry.ensure_lora_loaded_async("base", "a")
    await finish_loads(registry)
    adapter._lora._mutate.reset_mock()
    adapter.fetch_lora.side_effect = OSError("download failed")
    await registry.ensure_lora_loaded_async("base", "b")
    await finish_loads(registry)
    assert list(loaded.loras) == ["org/a"]
    assert registry.is_lora_loaded("base", "a")
    adapter._lora._mutate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fetch", [False, True])
async def test_delayed_old_fetch_never_attaches_or_changes_replacement_model(monkeypatch, fail_fetch):
    registry, old, adapter = registry_with_loras(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def fetch(*args):
        entered.set()
        assert release.wait(2)
        if fail_fetch:
            raise OSError("old fetch failed")
        return "/cache/a"

    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    await registry.ensure_lora_loaded_async("base", "a")
    await asyncio.to_thread(entered.wait, 1)
    replacement = LoadedModel(old.config, adapter, "cpu", lora_runtime=old.lora_runtime)
    replacement.loras["org/a"] = LoadedLora("org/a")
    registry._loaded["base"] = replacement
    release.set()
    await finish_loads(registry)
    assert list(replacement.loras) == ["org/a"]
    assert not replacement.lora_failures
    adapter._lora._mutate.assert_not_called()
    adapter.discard_lora_source.assert_called_once_with("org/a")


@pytest.mark.asyncio
async def test_waiting_victim_unload_rechecks_model_identity_inside_mutation(monkeypatch):
    registry, old, adapter = registry_with_loras(monkeypatch)
    await registry.ensure_lora_loaded_async("base", "a")
    await finish_loads(registry)
    adapter._lora._mutate.reset_mock()
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_mutation(mutation):
        entered.set()
        await release.wait()
        return mutation()

    worker = Mock()
    worker.run_adapter_mutation = delayed_mutation
    old.worker = worker
    await registry.ensure_lora_loaded_async("base", "b")
    await asyncio.wait_for(entered.wait(), 1)
    replacement = LoadedModel(old.config, adapter, "cpu", lora_runtime=old.lora_runtime)
    replacement.loras["org/a"] = LoadedLora("org/a")
    registry._loaded["base"] = replacement
    release.set()
    await finish_loads(registry)
    assert list(replacement.loras) == ["org/a"]
    assert not replacement.lora_failures
    adapter._lora._mutate.assert_not_called()
