"""Event wakeup must preserve admission, shutdown and batch-window behavior."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core import batcher as batcher_module
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig, WorkerResult
from sie_server.core.worker.types import WorkerDrainedError
from sie_server.types.inputs import Item

_TIMEOUT = 2.0


@pytest.fixture
async def worker() -> AsyncIterator[ModelWorker]:
    adapter = MagicMock()
    adapter.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
        dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
        batch_size=len(items),
    )
    instance = ModelWorker(adapter, WorkerConfig(idle_coalesce_ms=0))
    await instance.start()
    try:
        yield instance
    finally:
        await asyncio.wait_for(instance.stop(), _TIMEOUT)


async def _submit(worker: ModelWorker, name: str, lora: str | None = None) -> asyncio.Future[WorkerResult]:
    return await worker.submit(
        [make_text_item([1, 2], 0)],
        [Item(text=name)],
        ["dense"],
        options={"lora": lora} if lora is not None else None,
        request_id=name,
    )


@pytest.mark.parametrize("lora", [None, "adapter-a"])
@pytest.mark.parametrize("before_wait", [False, True])
async def test_admission_at_idle_wait_is_not_lost(worker, monkeypatch, lora, before_wait):
    """Both a latched signal and an already-blocked waiter deliver real work."""
    entered = asyncio.Queue()
    original_wait = worker._work_available.wait
    injected = []

    async def observed_wait():
        if before_wait and not injected:
            # Empty scan has finished, but Event.wait has not started yet.
            injected.append(await _submit(worker, "first", lora))
        entered.put_nowait(None)
        return await original_wait()

    monkeypatch.setattr(worker._work_available, "wait", observed_wait)
    worker._ensure_process_loop_started()
    await asyncio.wait_for(entered.get(), _TIMEOUT)
    first = injected[0] if before_wait else await _submit(worker, "first", lora)
    result = await asyncio.wait_for(first, _TIMEOUT)
    assert result.output.batch_size == 1

    # Return to idle, then admit again. Clearing the first signal must not
    # strand the next request, including a newly-created LoRA batcher.
    await asyncio.wait_for(entered.get(), _TIMEOUT)
    second = await _submit(worker, "second", lora)
    assert (await asyncio.wait_for(second, _TIMEOUT)).output.batch_size == 1
    await asyncio.wait_for(entered.get(), _TIMEOUT)
    assert [call.args[0][0].text for call in worker.adapter.encode.call_args_list] == ["first", "second"]
    assert worker.pending_count == 0


async def test_empty_worker_stays_parked_without_polling(worker, monkeypatch):
    entered = asyncio.Event()
    calls = 0
    original_wait = worker._work_available.wait

    async def observed_wait():
        nonlocal calls
        calls += 1
        entered.set()
        return await original_wait()

    monkeypatch.setattr(worker._work_available, "wait", observed_wait)
    worker._ensure_process_loop_started()
    await asyncio.wait_for(entered.wait(), _TIMEOUT)
    # An unrelated event-loop timer can run without waking this worker.
    await asyncio.sleep(0.02)
    assert calls == 1
    assert not worker._process_task.done()
    worker.adapter.encode.assert_not_called()


async def test_cancelled_admission_does_not_signal_or_lose_next_work(worker):
    batcher = worker._batchers[None]
    await batcher._lock.acquire()
    submitting = asyncio.create_task(_submit(worker, "cancelled"))
    try:
        await asyncio.sleep(0)
        assert not submitting.done()
        assert not worker._work_available.is_set()
        assert worker._process_task is None
        submitting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submitting
        assert worker.pending_count == 0
    finally:
        batcher._lock.release()
        submitting.cancel()
        await asyncio.gather(submitting, return_exceptions=True)

    future = await _submit(worker, "accepted")
    assert (await asyncio.wait_for(future, _TIMEOUT)).output.batch_size == 1
    assert [call.args[0][0].text for call in worker.adapter.encode.call_args_list] == ["accepted"]


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_stop_reaps_an_idle_waiter(worker, monkeypatch, cancel_first):
    entered = asyncio.Event()
    left = asyncio.Event()
    original_wait = worker._work_available.wait

    async def observed_wait():
        entered.set()
        try:
            return await original_wait()
        finally:
            left.set()

    monkeypatch.setattr(worker._work_available, "wait", observed_wait)
    worker._ensure_process_loop_started()
    task = worker._process_task
    await asyncio.wait_for(entered.wait(), _TIMEOUT)
    if cancel_first:
        task.cancel()
    await asyncio.wait_for(worker.stop(), _TIMEOUT)
    assert left.is_set()
    assert task.done()
    assert worker._process_task is None
    assert not worker.is_running
    await worker.stop()  # Remains idempotent after cancellation.


async def test_stop_during_coalescing_fails_admitted_future(worker, monkeypatch):
    worker.config.idle_coalesce_ms = 60_000
    worker._batch_config.coalesce_ms = 60_000
    worker._batch_config.max_batch_wait_ms = 120_000
    waiting = asyncio.Event()
    batcher = worker._batchers[None]
    original_wait = batcher._batch_ready.wait

    async def observed_wait():
        waiting.set()
        return await original_wait()

    monkeypatch.setattr(batcher._batch_ready, "wait", observed_wait)
    future = await _submit(worker, "queued")
    await asyncio.wait_for(waiting.wait(), _TIMEOUT)
    await asyncio.wait_for(worker.stop(), _TIMEOUT)
    with pytest.raises(WorkerDrainedError):
        await future
    assert worker.pending_count == 0
    assert worker._process_task is None
    worker.adapter.encode.assert_not_called()


@pytest.mark.parametrize(
    ("case", "steps", "expected_time", "expected_count"),
    [
        ("lone", [(2.0, False), (3.1, False)], 3.1, 1),
        ("burst", [(2.0, True), (4.0, False), (5.1, False)], 5.1, 2),
        ("busy", [(3.1, False), (7.6, False)], 7.6, 1),
        ("full", [(2.0, True)], 2.0, 2),
        ("timeout", [(2.0 * i, True) for i in range(1, 8)] + [(15.1, False)], 15.1, 8),
        ("zero", [], 0.0, 1),
    ],
)
async def test_original_coalescing_deadlines_and_membership(
    worker, monkeypatch, case, steps, expected_time, expected_count
):
    """Use a controlled batcher clock, not fragile wall-time upper bounds.

    The real selector and BatchFormer must retain the default 3ms idle quiet
    period, its reset on arrival, the 15ms first-item deadline, fullness and
    the uncapped 7.5ms busy quiet period. No model or thread timing is measured.
    """
    worker.config.idle_coalesce_ms = 0.0 if case == "zero" else 3.0
    if case == "full":
        worker._batch_config.max_batch_requests = 2
    now = 0.0
    monkeypatch.setattr(batcher_module, "time", SimpleNamespace(monotonic=lambda: now / 1000))
    monkeypatch.setattr(worker, "_ensure_process_loop_started", lambda: None)
    futures = [await _submit(worker, "first")]
    remaining = iter(steps)
    observed = []

    async def advance_clock():
        nonlocal now
        now, add_item = next(remaining)
        observed.append(now)
        if add_item:
            futures.append(await _submit(worker, f"item-{len(futures)}"))
        return True

    monkeypatch.setattr(worker._batchers[None]._batch_ready, "wait", advance_clock)
    try:
        lora, batch, idle = await asyncio.wait_for(worker._get_next_batch_fcfs(was_idle=case != "busy"), _TIMEOUT)
        assert lora is None
        assert idle is (case != "busy")
        assert now == expected_time
        assert observed == [step[0] for step in steps]
        assert batch.size == expected_count
        assert [metadata.request_id for metadata in batch.metadata] == ["first"] + [
            f"item-{index}" for index in range(1, expected_count)
        ]
        assert worker.pending_count == 0
    finally:
        # This selector test deliberately does not dispatch the extracted
        # batch. Dispose of its test-owned futures explicitly.
        for future in futures:
            future.cancel()
