"""Eviction drain: interrupted requests must fail fast, not hang.

``ModelWorker.stop()`` cancels the batch loop, so anything still sitting in a
``BatchFormer`` will never run. Before the drain, none of the three paths that
complete a queued future with an exception (malformed preformed request, the
per-batch error fan-out, OOM recovery) ran on the eviction path, so an awaiter
blocked until some outer timeout fired. This is the second stage of the
eviction-drain contract.

Two populations, covered in that order below: work still *queued* in a
batcher, and a batch already extracted and *in flight* when the cancellation
landed. The second has its metadata only in ``_process_batch``'s locals, and
``CancelledError`` unwinds past every ``except Exception`` on the way out, so
it needs the worker to hand the registration to ``stop()``.

Every wait here is bounded by a short ``asyncio.wait_for`` so a regression
fails the suite fast instead of stalling it.
"""

import asyncio
import contextlib
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import status
from sie_server.api.helpers import WORKER_DRAINED_RETRY_AFTER_S, InferenceErrorHandler
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.types import WorkerDrainedError, WorkerResult
from sie_server.ipc_types import EncodeBatchItem
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.inputs import Item
from sie_server.types.responses import ErrorCode

# Bounded so a hang regression fails rather than stalls.
_AWAIT_TIMEOUT_S = 2.0


def _parked_config() -> WorkerConfig:
    """Config whose batcher never yields a batch during a test.

    Every dispatch trigger (cost limit, request limit, first-request timeout,
    coalesce window, idle accumulation window) is pushed far out of reach, so
    submitted items stay queued in the ``BatchFormer`` for the whole test and
    the drain is the only thing that can complete them. This is the state a
    real worker is in whenever it is evicted with a backlog.
    """
    return WorkerConfig(
        max_batch_tokens=10**6,
        max_batch_requests=10**6,
        max_batch_wait_ms=600_000,
        coalesce_ms=600_000,
        idle_coalesce_ms=600_000,
        max_queue_size=0,
    )


def _dispatching_config() -> WorkerConfig:
    """Opposite of ``_parked_config``: form and dispatch a batch at once.

    Used by the in-flight tests, which need the batch *out* of the batcher
    (so ``drain_pending`` can no longer see it) and sitting in the inference
    executor when the cancellation arrives.
    """
    return WorkerConfig(
        max_batch_tokens=10**6,
        max_batch_requests=10**6,
        max_batch_wait_ms=1,
        coalesce_ms=0,
        idle_coalesce_ms=0,
        max_queue_size=0,
    )


@pytest.fixture
def mock_adapter() -> MagicMock:
    mock = MagicMock()
    mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
        dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
        batch_size=len(items),
    )
    return mock


class _BlockingEncode:
    """An ``encode`` that parks the inference thread until released.

    ``started`` fires once the forward pass is genuinely running on the
    executor thread, which is the only reliable signal that the batch has
    left its batcher. ``release`` lets the thread finish so the test does
    not leak it.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, items: list[Any], *args: Any, **kwargs: Any) -> EncodeOutput:
        self.started.set()
        self.release.wait(timeout=_AWAIT_TIMEOUT_S * 2)
        return EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
            batch_size=len(items),
        )


async def _await_started(blocking: _BlockingEncode) -> None:
    """Yield to the loop until the forward pass has begun. Bounded."""
    deadline = time.monotonic() + _AWAIT_TIMEOUT_S
    while not blocking.started.is_set():
        assert time.monotonic() < deadline, "inference thread never started"
        await asyncio.sleep(0.005)


def _item(index: int = 0) -> TextPreparedItem:
    return make_text_item([1, 2, 3, 4, 5], index)


async def _submit(
    worker: ModelWorker,
    *,
    count: int = 1,
    options: dict[str, Any] | None = None,
) -> asyncio.Future[WorkerResult]:
    """Queue one request of ``count`` items and let the process loop park."""
    future = await worker.submit(
        [_item(i) for i in range(count)],
        [Item(text=f"hello-{i}") for i in range(count)],
        ["dense"],
        options=options,
    )
    # Give the process loop a chance to start and block on the batcher, so
    # the drain is exercised against a live (not merely un-started) loop.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return future


class TestStopDrainsQueuedRequests:
    @pytest.mark.asyncio
    async def test_queued_request_fails_with_retryable_error(self, mock_adapter: MagicMock) -> None:
        """A queued request on an evicted model fails instead of hanging."""
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker)
        assert worker.pending_count == 1
        assert not future.done()

        await worker.stop()

        with pytest.raises(WorkerDrainedError, match="test-model"):
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)

    @pytest.mark.asyncio
    async def test_no_item_is_left_pending_in_any_batcher(self, mock_adapter: MagicMock) -> None:
        """Every batcher is drained, not just the base-model one.

        Each LoRA gets its own ``BatchFormer``; a drain that only walked the
        base-model queue would leave every LoRA request hanging.
        """
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        base = await _submit(worker)
        lora_a = await _submit(worker, options={"lora": "adapter-a"})
        lora_b = await _submit(worker, options={"lora": "adapter-b"})
        assert set(worker._batchers) == {None, "adapter-a", "adapter-b"}
        assert worker.pending_count == 3

        await worker.stop()

        for future in (base, lora_a, lora_b):
            with pytest.raises(WorkerDrainedError):
                await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)
        assert worker.pending_count == 0
        assert worker.pending_tokens == 0
        assert all(batcher.pending_count == 0 for batcher in worker._batchers.values())

    @pytest.mark.asyncio
    async def test_multi_item_request_fails_its_single_future_once(self, mock_adapter: MagicMock) -> None:
        """One request occupying N queue slots shares one future.

        Metadata is deduplicated by identity, so the drain must not try to
        set an exception on the same future once per item.
        """
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker, count=4)
        assert worker.pending_count == 4

        await worker.stop()

        with pytest.raises(WorkerDrainedError):
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)
        assert worker.pending_count == 0

    @pytest.mark.asyncio
    async def test_already_completed_future_is_not_touched(self, mock_adapter: MagicMock) -> None:
        """The drain is idempotent against a future that already resolved."""
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker)

        already_resolved = WorkerResult(
            output=EncodeOutput(dense=np.array([[0.1, 0.2, 0.3]]), batch_size=1),
            timing=RequestTiming(),
        )
        future.set_result(already_resolved)

        await worker.stop()

        assert await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S) is already_resolved

    @pytest.mark.asyncio
    async def test_drain_is_idempotent_across_repeated_stops(self, mock_adapter: MagicMock) -> None:
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker)

        await worker.stop()
        await worker.stop()

        with pytest.raises(WorkerDrainedError):
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)


class TestDrainIsCancellationSafe:
    @pytest.mark.asyncio
    async def test_drain_runs_when_process_task_already_cancelled(self, mock_adapter: MagicMock) -> None:
        """A loop cancelled before ``stop()`` still gets its queue drained.

        ``CancelledError`` is a ``BaseException``, so the ``except Exception``
        fan-out in ``oom_recovery`` never covered this; only an explicit
        drain does.
        """
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker)

        assert worker._process_task is not None
        worker._process_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker._process_task
        assert not future.done()

        await worker.stop()

        with pytest.raises(WorkerDrainedError):
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)

    @pytest.mark.asyncio
    async def test_drain_runs_when_stop_itself_is_cancelled(self, mock_adapter: MagicMock) -> None:
        """``_do_unload`` runs ``stop()`` under a drain timeout.

        When that timeout fires, the cancellation lands on ``stop()``'s own
        await. The drain lives in a ``finally`` so it still runs.
        """
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()
        future = await _submit(worker)

        stop_task = asyncio.create_task(worker.stop())
        await asyncio.sleep(0)  # let stop() reach its await on the process task
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task

        with pytest.raises(WorkerDrainedError):
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)


class TestStopDrainsInFlightBatch:
    """A batch already extracted from its batcher when the loop is cancelled.

    ``BatchFormer.drain_pending()`` cannot reach it — the metadata lives only
    in ``_process_batch``'s locals — and ``CancelledError`` unwinds past
    ``BatchExecutor``'s ``except Exception`` fan-out without failing a single
    future. The worker therefore registers the batch while it is in flight
    and hands that registration to ``stop()`` on cancellation.
    """

    @pytest.mark.asyncio
    async def test_in_flight_request_fails_with_retryable_error(self, mock_adapter: MagicMock) -> None:
        blocking = _BlockingEncode()
        mock_adapter.encode.side_effect = blocking
        worker = ModelWorker(mock_adapter, _dispatching_config(), model_name="test-model")
        await worker.start()
        try:
            future = await worker.submit([_item()], [Item(text="hello")], ["dense"])
            await _await_started(blocking)

            # The batch is out of the batcher: nothing the part-two drain
            # walks can see it any more. This is what makes the test
            # discriminate against the queued-only drain.
            assert worker.pending_count == 0
            assert worker._in_flight, "in-flight batch was not tracked"

            stop_task = asyncio.create_task(worker.stop())
            with pytest.raises(WorkerDrainedError, match="test-model"):
                await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)
        finally:
            blocking.release.set()

        await asyncio.wait_for(stop_task, timeout=_AWAIT_TIMEOUT_S)

    @pytest.mark.asyncio
    async def test_registration_is_released_after_a_normal_batch(self, mock_adapter: MagicMock) -> None:
        """A live worker accumulates nothing: only cancellation retains."""
        worker = ModelWorker(mock_adapter, _dispatching_config(), model_name="test-model")
        await worker.start()
        try:
            future = await worker.submit([_item()], [Item(text="hello")], ["dense"])
            await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)

            assert worker._in_flight == {}
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_cancelling_a_live_preformed_request_retains_nothing(self, mock_adapter: MagicMock) -> None:
        """A cancellation with no ``stop()`` behind it must not accumulate.

        The ownership handoff is only correct when someone is coming to
        collect. ``IpcServer`` cancels its in-flight request tasks when its
        drain deadline expires while the worker is still running, and each
        retained registration would pin an ``_InFlightBatch``, its metadata,
        the prepared items and the future for the life of the process.
        """
        blocking = _BlockingEncode()
        mock_adapter.encode.side_effect = blocking
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()

        for _ in range(3):
            request = asyncio.create_task(worker.submit_preformed([_item()], [Item(text="hello")], ["dense"]))
            await _await_started(blocking)
            assert worker._in_flight, "pre-formed batch was not tracked"

            request.cancel()
            blocking.release.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(request, timeout=_AWAIT_TIMEOUT_S)

            # The worker is still running, so nothing will ever drain this.
            assert worker.is_running
            assert worker._in_flight == {}, "a cancelled live request retained its registration"
            blocking.started.clear()
            blocking.release.clear()

        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_leaves_a_live_preformed_batch_alone(self, mock_adapter: MagicMock) -> None:
        """The sidecar's pre-formed path dispatches from its own task.

        It bypasses the batchers and is not cancelled by ``stop()``, so its
        forward pass is still going to produce a result. Failing its futures
        would turn a request that is about to succeed into a spurious retry,
        which is why the drain checks the owning task rather than simply
        failing everything registered.
        """
        blocking = _BlockingEncode()
        mock_adapter.encode.side_effect = blocking
        worker = ModelWorker(mock_adapter, _parked_config(), model_name="test-model")
        await worker.start()

        preformed = asyncio.create_task(worker.submit_preformed([_item()], [Item(text="hello")], ["dense"]))
        await _await_started(blocking)
        assert worker._in_flight, "pre-formed batch was not tracked"

        # Drain directly: ``stop()`` would join the executor and block until
        # the forward pass finishes, hiding the ordering under test.
        assert worker._fail_queued_requests() == 0
        assert worker._in_flight, "a live pre-formed batch must not be reaped"

        blocking.release.set()
        future = await asyncio.wait_for(preformed, timeout=_AWAIT_TIMEOUT_S)
        result = await asyncio.wait_for(future, timeout=_AWAIT_TIMEOUT_S)
        assert result.output.batch_size == 1
        await worker.stop()


class TestDrainedErrorSurfaces:
    """A drained request must read as retryable on both ingress paths."""

    def test_http_path_maps_to_retryable_model_loading(self) -> None:
        handler = InferenceErrorHandler("test-model", "encode", MagicMock())

        exc = handler.handle_inference_error(WorkerDrainedError("stopped before this request ran"))

        assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert isinstance(exc.detail, dict)
        assert exc.detail["code"] == ErrorCode.MODEL_LOADING.value
        assert exc.headers == {"Retry-After": str(WORKER_DRAINED_RETRY_AFTER_S)}

    def test_queue_path_maps_to_nak_retry(self) -> None:
        batch_item = EncodeBatchItem(
            work_item_id="req-1.0",
            request_id="req-1",
            item_index=0,
            total_items=1,
            timestamp=time.time(),
            item={"text": "hi"},
        )

        outcome = _inference_exception_outcome(batch_item, WorkerDrainedError("evicted"))

        assert outcome.disposition == "nak_retry"
        assert outcome.error_code is None
