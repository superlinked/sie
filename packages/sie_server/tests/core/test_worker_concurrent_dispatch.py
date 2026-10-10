"""Concurrent batch dispatch for adapters that front a continuously batching engine."""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.prepared import ExtractPreparedItem
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.model_worker import _dispatch_width
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity


class _GatedAdapter:
    """An extract adapter whose calls block until released, counting the peak overlap."""

    def __init__(self, width: int, *, lora: bool = False) -> None:
        self._width = width
        self._lora = lora
        self.release = threading.Event()
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.set_active_lora = MagicMock()

    def max_concurrent_dispatch(self) -> int:
        return self._width

    def supports_lora(self) -> bool:
        return self._lora

    def extract(self, items: list[Item], **_: object) -> ExtractOutput:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            self.release.wait(timeout=5)
        finally:
            with self._lock:
                self.active -= 1
        return ExtractOutput(entities=[[Entity(text="x", label="text", score=1.0)] for _ in items])


def _config() -> WorkerConfig:
    return WorkerConfig(max_batch_tokens=100, max_batch_requests=1, max_batch_wait_ms=1)


async def _submit(worker: ModelWorker, n: int) -> list[asyncio.Future]:
    return [
        await worker.submit_extract([ExtractPreparedItem(cost=1, original_index=0)], [Item(text=f"t{i}")])
        for i in range(n)
    ]


async def _wait_for_peak(adapter: _GatedAdapter, peak: int) -> None:
    for _ in range(200):
        if adapter.peak >= peak:
            return
        await asyncio.sleep(0.01)


def test_dispatch_width_defaults_to_one() -> None:
    assert _dispatch_width(object()) == 1
    assert _dispatch_width(_GatedAdapter(8)) == 8
    assert _dispatch_width(_GatedAdapter(0)) == 1


def test_dispatch_width_is_one_for_lora_adapters() -> None:
    assert _dispatch_width(_GatedAdapter(8, lora=True)) == 1


@pytest.mark.asyncio
async def test_batches_overlap_up_to_the_declared_width() -> None:
    adapter = _GatedAdapter(3)
    worker = ModelWorker(adapter, _config())
    await worker.start()
    try:
        futures = await _submit(worker, 5)
        await _wait_for_peak(adapter, 3)
        await asyncio.sleep(0.05)
        assert adapter.peak == 3
        adapter.release.set()
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
        assert len(results) == 5
        adapter.set_active_lora.assert_not_called()
    finally:
        adapter.release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_width_one_keeps_single_batch_dispatch() -> None:
    adapter = _GatedAdapter(1)
    worker = ModelWorker(adapter, _config())
    await worker.start()
    try:
        futures = await _submit(worker, 3)
        await _wait_for_peak(adapter, 1)
        await asyncio.sleep(0.05)
        assert adapter.peak == 1
        adapter.release.set()
        await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
    finally:
        adapter.release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_stop_fails_concurrent_batches_instead_of_hanging() -> None:
    adapter = _GatedAdapter(2)
    worker = ModelWorker(adapter, _config())
    await worker.start()
    futures = await _submit(worker, 4)
    await _wait_for_peak(adapter, 2)
    stopping = asyncio.create_task(worker.stop())
    await asyncio.sleep(0.05)
    adapter.release.set()
    await asyncio.wait_for(stopping, timeout=10)
    for future in futures:
        assert future.done()


@pytest.mark.asyncio
async def test_an_idle_direct_queue_holds_no_slot_from_preformed_batches() -> None:
    from sie_server.core.worker.model_worker import PreformedExtractRequest

    adapter = _GatedAdapter(2)
    worker = ModelWorker(adapter, _config())
    await worker.start()
    try:
        # One direct request starts the loop and finishes, leaving it idle.
        adapter.release.set()
        first = await _submit(worker, 1)
        await asyncio.wait_for(asyncio.gather(*first), timeout=5)
        adapter.release.clear()
        adapter.peak = 0

        def request() -> PreformedExtractRequest:
            return PreformedExtractRequest(
                prepared_items=[ExtractPreparedItem(cost=1, original_index=0)], items=[Item(text="p")]
            )

        batches = [asyncio.create_task(worker.submit_extract_preformed_batch([request()], lora=None)) for _ in range(2)]
        await _wait_for_peak(adapter, 2)
        assert adapter.peak == 2
        adapter.release.set()
        await asyncio.wait_for(asyncio.gather(*batches), timeout=5)
    finally:
        adapter.release.set()
        await worker.stop()


@pytest.mark.asyncio
async def test_concurrent_dispatch_steps_the_adaptive_controller() -> None:
    from sie_server.core.worker.types import AdaptiveBatchingParams

    adapter = _GatedAdapter(2)
    adapter.release.set()
    config = WorkerConfig(
        max_batch_tokens=100,
        max_batch_requests=1,
        max_batch_wait_ms=1,
        adaptive_batching=AdaptiveBatchingParams(enabled=True),
    )
    worker = ModelWorker(adapter, config)
    await worker.start()
    try:
        assert worker._adaptive_controller is not None
        steps: list[int] = []
        original = worker._step_adaptive_controller

        def counted(batch, telemetry) -> None:
            steps.append(batch.size)
            original(batch, telemetry)

        worker._step_adaptive_controller = counted  # type: ignore[method-assign]
        futures = await _submit(worker, 3)
        await asyncio.wait_for(asyncio.gather(*futures), timeout=5)
        await asyncio.sleep(0.05)
        assert sum(steps) == 3
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_without_a_release_hook_builds_and_releases_nothing() -> None:
    """Duck-typed adapters with no release hook still build a worker.

    ``_GatedAdapter`` has neither hook. ``_ClaimsMemory`` says it holds
    memory but still has no hook, so the invoke path must return 0 instead
    of calling a missing method.
    """

    class _ClaimsMemory:
        def has_releasable_memory(self) -> bool:
            return True

    for adapter in (_GatedAdapter(1), _ClaimsMemory()):
        worker = ModelWorker(adapter, _config())
        try:
            assert worker._batch_executor._release_optional_memory is None
            assert await asyncio.wait_for(worker.release_optional_memory(), 0.5) == 0
        finally:
            worker._inference_executor.shutdown(wait=False, cancel_futures=True)
