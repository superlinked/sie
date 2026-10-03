"""Eviction must not stop the event loop.

Two blocking calls used to run directly on the loop while the global load
lock was held: ``ThreadPoolExecutor.shutdown(wait=True)`` in
``ModelWorker.stop()`` and ``adapter.unload()`` (``gc.collect`` + a CUDA
cache flush, and for subprocess-backed adapters a SIGTERM/SIGKILL wait) in
``RegistryManager._do_unload``. Because they were synchronous, *no* other
coroutine ran for their duration — health probes and in-flight responses on
models that were not being evicted included. This is the first stage of the
eviction-drain contract.

The assertion in each test is the same: a concurrent coroutine must make
progress while the eviction is happening. Both fail against the pre-fix code
rather than hanging — the blocking work is a bounded ``time.sleep``, so a
regression reports a starved loop instead of stalling the suite.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import ModelWorker, WorkerConfig

# How long the blocking teardown takes. Long enough that a starved loop is
# unambiguous, short enough to keep the suite quick.
_BLOCKING_S = 0.6

# Ticker period. ~60 ticks are expected over ``_BLOCKING_S`` on a healthy
# loop; a loop blocked by the teardown manages zero. The floor is an order
# of magnitude below the healthy value so a loaded CI machine cannot flake
# it — a busy runner was measured stalling a *healthy* loop for 0.61 s in
# one shot, so the margin here is deliberately generous.
#
# Note this counts ticks rather than measuring the largest gap between
# them: a max over a noisy sample is exactly the statistic that runner
# jitter corrupts, whereas a cumulative count barely moves for one hiccup.
_TICK_S = 0.01
_MIN_TICKS = 5

_AWAIT_TIMEOUT_S = 5.0


class _Ticker:
    """Counts event-loop turns, standing in for a health probe."""

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = False

    async def run(self) -> None:
        while not self._stop:
            self.ticks += 1
            await asyncio.sleep(_TICK_S)

    def stop(self) -> None:
        self._stop = True


async def _start_ticker() -> tuple[_Ticker, asyncio.Task[None]]:
    """Start a ticker and reset its count once it is actually running."""
    ticker = _Ticker()
    task = asyncio.create_task(ticker.run())
    await asyncio.sleep(0)
    ticker.ticks = 0
    return ticker, task


async def _stop_ticker(ticker: _Ticker, task: asyncio.Task[None]) -> int:
    ticker.stop()
    await asyncio.wait_for(task, timeout=_AWAIT_TIMEOUT_S)
    return ticker.ticks


def _make_config(name: str = "test-model") -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id="org/test",
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


class TestAdapterUnloadDoesNotStarveTheLoop:
    @pytest.mark.asyncio
    async def test_concurrent_coroutine_progresses_during_a_slow_unload(self) -> None:
        """``adapter.unload()`` runs off-loop, so probes keep being served."""
        registry = ModelRegistry()
        registry.add_config(_make_config())

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            adapter = MagicMock()
            adapter.memory_footprint.return_value = 1000
            del adapter.aclose_client  # embedding-style adapter
            adapter.unload = MagicMock(side_effect=lambda: time.sleep(_BLOCKING_S))
            mock_load.return_value = adapter

            await registry.load_async("test-model", "cpu")

            ticker, ticker_task = await _start_ticker()
            await asyncio.wait_for(registry.unload_async("test-model"), timeout=_AWAIT_TIMEOUT_S)
            ticks = await _stop_ticker(ticker, ticker_task)

        adapter.unload.assert_called_once()
        assert not registry.is_loaded("test-model")
        assert ticks >= _MIN_TICKS, f"event loop was starved during unload ({ticks} ticks)"

    @pytest.mark.asyncio
    async def test_unload_still_completes_before_the_lock_is_released(self) -> None:
        """Ordering is unchanged: teardown finishes before accounting drops.

        Moving the call to a thread must not let ``unload_async`` return
        while VRAM is still held — a load admitted against the freed
        accounting would size itself against memory the dying adapter has
        not released. The load lock stays held across the whole teardown on
        purpose.
        """
        registry = ModelRegistry()
        registry.add_config(_make_config())
        order: list[str] = []

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            adapter = MagicMock()
            adapter.memory_footprint.return_value = 1000
            del adapter.aclose_client

            def _slow_unload() -> None:
                time.sleep(_BLOCKING_S)
                order.append("unload-finished")

            adapter.unload = MagicMock(side_effect=_slow_unload)
            mock_load.return_value = adapter

            await registry.load_async("test-model", "cpu")
            await asyncio.wait_for(registry.unload_async("test-model"), timeout=_AWAIT_TIMEOUT_S)
            order.append("unload_async-returned")

        assert order == ["unload-finished", "unload_async-returned"]


class TestExecutorJoinDoesNotStarveTheLoop:
    @pytest.mark.asyncio
    async def test_concurrent_coroutine_progresses_while_the_inference_thread_drains(self) -> None:
        """``stop()``'s executor join is a thread join; it belongs off-loop."""
        adapter = MagicMock()
        worker = ModelWorker(adapter, WorkerConfig(), model_name="test-model")
        await worker.start()

        # Occupy the single inference thread so the join has something real
        # to wait for, exactly as an in-progress forward pass would.
        worker._inference_executor.submit(time.sleep, _BLOCKING_S)

        ticker, ticker_task = await _start_ticker()
        await asyncio.wait_for(worker.stop(), timeout=_AWAIT_TIMEOUT_S)
        ticks = await _stop_ticker(ticker, ticker_task)

        assert ticks >= _MIN_TICKS, f"event loop was starved during the executor join ({ticks} ticks)"

    @pytest.mark.asyncio
    async def test_stop_waits_for_the_inference_thread(self) -> None:
        """Off-loop is not fire-and-forget: the thread is still joined."""
        adapter = MagicMock()
        worker = ModelWorker(adapter, WorkerConfig(), model_name="test-model")
        await worker.start()

        finished: list[str] = []
        worker._inference_executor.submit(lambda: (time.sleep(_BLOCKING_S), finished.append("thread"))[1])

        await asyncio.wait_for(worker.stop(), timeout=_AWAIT_TIMEOUT_S)

        assert finished == ["thread"]
