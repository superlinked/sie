# Tests for the proactive idle-eviction loop in ModelRegistry.
#
# Cold-model unload runs alongside the existing 85%-pressure monitor and is
# disabled by default. These tests cover:
#
# - ``MemoryManager.get_idle_models`` (already covered in test_memory.py).
# - Registry's ``_idle_evict_loop``: it unloads stale models and respects
#   the load-lock + freshness recheck.
# - Lifecycle: ``start_idle_evictor`` is a no-op when ``idle_evict_s`` is
#   None.
#
# The loop is exercised by hand-rolling the time advance: rather than
# sleeping for real, we override ``last_used_at`` to an epoch in the past
# and let the loop's snapshot pick it up on the next tick.

from __future__ import annotations

import asyncio
import time
from collections.abc import Container
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.config.engine import EngineConfig
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.memory import MemoryConfig
from sie_server.core.registry import ModelRegistry
from sie_server.core.residency import EvictionResult


def _make_config(name: str = "test", hf_id: str | None = "org/test") -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


def _build_registry(
    *,
    idle_evict_s: int | None,
    check_interval_s: float = 0.01,
    pinned_models: list[str] | None = None,
) -> ModelRegistry:
    """Construct a registry wired with ``EngineConfig.idle_evict_s`` set.

    Tests bypass ``EngineConfig``'s production-safety lower bound (``ge=10``)
    by setting the field directly on a model_construct-created instance —
    this lets the loop tick fast enough for ``await asyncio.sleep`` based
    waits to complete in test time.
    """
    if idle_evict_s is None:
        engine_config = EngineConfig()
    else:
        # ``model_construct`` skips validators (we want sub-second values
        # in tests; the production validator caps minimum at 10s).
        engine_config = EngineConfig.model_construct(idle_evict_s=idle_evict_s)
    return ModelRegistry(
        memory_config=MemoryConfig(memory_check_interval_s=check_interval_s),
        engine_config=engine_config,
        pinned_models=pinned_models,
    )


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_idle_evictor_noop_when_disabled() -> None:
    """idle_evict_s=None must not start a background task."""
    reg = _build_registry(idle_evict_s=None)
    await reg.start_idle_evictor()
    assert reg._idle_evict_task is None
    await reg.stop_idle_evictor()  # idempotent


@pytest.mark.asyncio
async def test_start_stop_idempotent_when_enabled() -> None:
    reg = _build_registry(idle_evict_s=10)
    await reg.start_idle_evictor()
    assert reg._idle_evict_task is not None
    # Second start is a no-op.
    first_task = reg._idle_evict_task
    await reg.start_idle_evictor()
    assert reg._idle_evict_task is first_task
    await reg.stop_idle_evictor()
    assert reg._idle_evict_task is None


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_start_worker_refreshes_idle_timestamp(mock_load_adapter: MagicMock) -> None:
    """Score/extract worker use must count as model activity for idle eviction."""
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    mock_load_adapter.return_value = adapter

    reg = _build_registry(idle_evict_s=10)
    reg.add_config(_make_config(name="model-score"))
    await reg.load_async("model-score", device="cpu")

    worker = reg.get_worker("model-score")
    assert worker is not None
    worker.start = AsyncMock()

    info = reg._memory_manager.get_model_info("model-score")
    assert info is not None
    stale_time = time.monotonic() - 100.0
    info.last_used_at = stale_time

    await reg.start_worker("model-score")

    assert info.last_used_at > stale_time
    assert "model-score" not in reg._memory_manager.get_idle_models(idle_threshold_s=10)


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_start_worker_rejects_model_being_unloaded(mock_load_adapter: MagicMock) -> None:
    """Requests must not revive or use a model after unload has started."""
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    mock_load_adapter.return_value = adapter

    reg = _build_registry(idle_evict_s=10)
    reg.add_config(_make_config(name="model-score"))
    await reg.load_async("model-score", device="cpu")
    reg._unloading.add("model-score")

    with pytest.raises(RuntimeError, match="currently being unloaded"):
        await reg.start_worker("model-score")


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_idle_eviction_unloads_stale_model(mock_load_adapter: MagicMock) -> None:
    """A model whose last_used_at exceeds idle_evict_s gets unloaded."""

    def make_adapter() -> MagicMock:
        m = MagicMock()
        m.capabilities.outputs = ["dense"]
        return m

    adapters = [make_adapter() for _ in range(2)]
    mock_load_adapter.side_effect = adapters

    reg = _build_registry(idle_evict_s=1)
    for name in ["model-a", "model-b"]:
        reg.add_config(_make_config(name=name, hf_id=f"org/{name}"))
        await reg.load_async(name, device="cpu")

    # Force model-a stale, leave model-b fresh.
    info_a = reg._memory_manager.get_model_info("model-a")
    assert info_a is not None
    info_a.last_used_at = time.monotonic() - 100.0

    await reg.start_idle_evictor()
    # Wait for at least one full tick of the loop.
    await asyncio.sleep(0.05)
    await reg.stop_idle_evictor()

    assert not reg.is_loaded("model-a")
    assert reg.is_loaded("model-b")


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_touch_postpones_eviction(mock_load_adapter: MagicMock) -> None:
    """A touch between snapshot and unload must save the model.

    We exercise the recheck path: the snapshot picks up the stale model
    name, but before the loop reaches the unload step, the model is
    touched and its age drops below the threshold. The unload should NOT
    happen.
    """

    def make_adapter() -> MagicMock:
        m = MagicMock()
        m.capabilities.outputs = ["dense"]
        return m

    mock_load_adapter.return_value = make_adapter()

    reg = _build_registry(idle_evict_s=1)
    reg.add_config(_make_config(name="model-x"))
    await reg.load_async("model-x", device="cpu")

    info = reg._memory_manager.get_model_info("model-x")
    assert info is not None

    # Set stale, then touch right before the loop's recheck. We control
    # this by stubbing get_idle_models to return the name even though we
    # then rebump last_used_at to "now" before the lock-recheck runs.
    info.last_used_at = time.monotonic() - 100.0

    original_get_idle = reg._memory_manager.get_idle_models

    def stub_get_idle(*, idle_threshold_s: float, now: float | None = None) -> list[str]:
        # First call returns the stale name; the side-effect is to refresh
        # the model's last_used_at so the in-loop recheck rejects it.
        result = original_get_idle(idle_threshold_s=idle_threshold_s, now=now)
        info.last_used_at = time.monotonic()
        return result

    with patch.object(reg._memory_manager, "get_idle_models", side_effect=stub_get_idle):
        await reg.start_idle_evictor()
        await asyncio.sleep(0.05)
        await reg.stop_idle_evictor()

    # Model should still be loaded — recheck saved it.
    assert reg.is_loaded("model-x")


@pytest.mark.asyncio
async def test_idle_eviction_loop_handles_no_stale_models() -> None:
    """An empty stale-list must not crash the loop or unload anything."""
    reg = _build_registry(idle_evict_s=3600)  # very long threshold
    await reg.start_idle_evictor()
    await asyncio.sleep(0.05)
    # Loop should still be running (no exceptions inside).
    assert reg._idle_evict_task is not None
    assert not reg._idle_evict_task.done()
    await reg.stop_idle_evictor()


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_idle_eviction_skips_already_unloaded(mock_load_adapter: MagicMock) -> None:
    """The loop's `name not in self._loaded` recheck handles a stale snapshot.

    Setup: stub ``get_idle_models`` to return a name that has *just* been
    unloaded by another path between the snapshot and the lock-acquired
    recheck. The stub fires on the loop's first invocation; we trigger
    the unload in the stub itself so the recheck on registry.py:1052
    (`if name not in self._loaded: continue`) actually runs against a
    racing unload, not a pre-emptied registry.
    """

    def make_adapter() -> MagicMock:
        m = MagicMock()
        m.capabilities.outputs = ["dense"]
        return m

    mock_load_adapter.return_value = make_adapter()

    reg = _build_registry(idle_evict_s=1)
    reg.add_config(_make_config(name="model-y"))
    await reg.load_async("model-y", device="cpu")

    info = reg._memory_manager.get_model_info("model-y")
    assert info is not None
    info.last_used_at = time.monotonic() - 100.0

    # Stub ``get_idle_models`` to (a) return the stale name, (b) unload it
    # immediately as a side effect — so by the time the loop acquires the
    # lock and runs ``if name not in self._loaded: continue``, the model
    # is genuinely gone. Exercises the snapshot-vs-recheck race path.
    original_get_idle = reg._memory_manager.get_idle_models

    def racing_get_idle(
        *, idle_threshold_s: float, now: float | None = None, exclude: Container[str] = frozenset()
    ) -> list[str]:
        result = original_get_idle(idle_threshold_s=idle_threshold_s, now=now, exclude=exclude)
        # Synchronous unload: drop from the loaded dict directly to avoid
        # awaiting an async call from a sync side-effect. This mimics what
        # any other sync code path that mutated `_loaded` would do.
        reg._loaded.pop("model-y", None)
        reg._memory_manager.unregister_model("model-y")
        return result

    with patch.object(reg._memory_manager, "get_idle_models", side_effect=racing_get_idle):
        await reg.start_idle_evictor()
        await asyncio.sleep(0.05)
        await reg.stop_idle_evictor()

    # No crash, model stays unloaded, and crucially the loop did not try
    # to call ``_do_unload`` on a model that was already gone.
    assert not reg.is_loaded("model-y")


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_idle_evictor_skips_pinned_model(mock_load_adapter: MagicMock) -> None:
    """Idle evictor must not unload a pinned model even when it is stale.

    Both models are forced stale; only the non-pinned one should be evicted.
    """

    def make_adapter() -> MagicMock:
        m = MagicMock()
        m.capabilities.outputs = ["dense"]
        return m

    mock_load_adapter.side_effect = [make_adapter(), make_adapter()]

    reg = _build_registry(idle_evict_s=1, pinned_models=["model-pinned"])
    for name in ["model-pinned", "model-evictable"]:
        reg.add_config(_make_config(name=name, hf_id=f"org/{name}"))
        await reg.load_async(name, device="cpu")

    # Force both models stale
    now = time.monotonic()
    for name in ["model-pinned", "model-evictable"]:
        info = reg._memory_manager.get_model_info(name)
        assert info is not None
        info.last_used_at = now - 200.0

    await reg.start_idle_evictor()
    await asyncio.sleep(0.05)
    await reg.stop_idle_evictor()

    # Non-pinned model evicted; pinned model kept
    assert not reg.is_loaded("model-evictable")
    assert reg.is_loaded("model-pinned")


# --------------------------------------------------------------------------
# _do_unload must not leave a MemoryManager "ghost" on adapter failure (#1600)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_do_unload_raising_adapter_leaves_no_ghost(mock_load_adapter: MagicMock) -> None:
    """A raising ``adapter.unload()`` must still unregister the model from the
    MemoryManager.

    Otherwise the model becomes a ghost — absent from ``_loaded`` but still in
    ``_models`` — that ``get_lru_model`` keeps re-selecting while ``_do_unload``
    no-ops on it, a no-yield spin under ``_load_lock`` that hangs the server.
    See #1600.
    """
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    adapter.aclose_client = None  # not an SGLang-style client-holding adapter
    adapter.unload.side_effect = RuntimeError("unload boom")
    mock_load_adapter.return_value = adapter

    reg = _build_registry(idle_evict_s=None)
    reg.add_config(_make_config(name="doomed"))
    await reg.load_async("doomed", device="cpu")
    assert reg._memory_manager.get_model_info("doomed") is not None

    # _do_unload re-raises the adapter error (try/finally, not try/except)...
    with pytest.raises(RuntimeError, match="unload boom"):
        await reg._do_unload("doomed")

    # ...but the model is gone from BOTH _loaded and the MemoryManager — the
    # finally ran unregister_model, so there is no ghost to spin on.
    assert "doomed" not in reg._loaded
    assert reg._memory_manager.get_model_info("doomed") is None
    assert reg._memory_manager.loaded_model_count == 0


# --------------------------------------------------------------------------
# evict_lru_excluding — typed EvictionResult (issue #1569)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_evict_lru_excluding_no_candidate(mock_load_adapter: MagicMock) -> None:
    """Only the excluded model is resident → NO_CANDIDATE (not a bare False)."""
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    mock_load_adapter.return_value = adapter

    reg = _build_registry(idle_evict_s=None)
    reg.add_config(_make_config(name="only-self"))
    await reg.load_async("only-self", device="cpu")

    assert await reg.evict_lru_excluding("only-self") is EvictionResult.NO_CANDIDATE


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_evict_lru_excluding_evicts_sibling(mock_load_adapter: MagicMock) -> None:
    """A cold sibling is unloaded → EVICTED, targeting that sibling."""
    adapters = [MagicMock(), MagicMock()]
    for a in adapters:
        a.capabilities.outputs = ["dense"]
    mock_load_adapter.side_effect = adapters

    reg = _build_registry(idle_evict_s=None)
    reg.add_config(_make_config(name="keep-me"))
    reg.add_config(_make_config(name="evict-me"))
    await reg.load_async("keep-me", device="cpu")
    await reg.load_async("evict-me", device="cpu")

    reg._do_unload = AsyncMock()
    assert await reg.evict_lru_excluding("keep-me") is EvictionResult.EVICTED
    reg._do_unload.assert_awaited_once_with("evict-me", reason="oom_recovery")


@pytest.mark.asyncio
@patch("sie_server.core.model_loader.load_adapter")
async def test_evict_lru_excluding_unload_failure(mock_load_adapter: MagicMock) -> None:
    """If the chosen candidate's unload raises → UNLOAD_FAILED."""
    adapters = [MagicMock(), MagicMock()]
    for a in adapters:
        a.capabilities.outputs = ["dense"]
    mock_load_adapter.side_effect = adapters

    reg = _build_registry(idle_evict_s=None)
    reg.add_config(_make_config(name="keep-me"))
    reg.add_config(_make_config(name="evict-me"))
    await reg.load_async("keep-me", device="cpu")
    await reg.load_async("evict-me", device="cpu")

    reg._do_unload = AsyncMock(side_effect=RuntimeError("boom"))
    assert await reg.evict_lru_excluding("keep-me") is EvictionResult.UNLOAD_FAILED
