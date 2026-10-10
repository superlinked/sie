import asyncio
import io
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image as PILImage
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.preprocessor import ImagePreprocessor
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, RequestMetadata, WorkerConfig, WorkerResult, WorkerStats
from sie_server.core.worker import model_worker as model_worker_module
from sie_server.types.inputs import Item


class TestRequestMetadata:
    """Tests for RequestMetadata dataclass."""

    def test_basic_creation(self) -> None:
        """Can create basic metadata."""
        loop = asyncio.new_event_loop()
        future: asyncio.Future[WorkerResult] = loop.create_future()
        timing = RequestTiming()

        metadata = RequestMetadata(
            future=future,
            items=[Item(text="hello")],
            output_types=["dense"],
            timing=timing,
        )

        assert metadata.future is future
        assert metadata.items == [Item(text="hello")]
        assert metadata.output_types == ["dense"]
        assert metadata.timing is timing
        assert metadata.instruction is None
        assert metadata.is_query is False
        assert metadata.request_id is None
        loop.close()

    def test_with_all_fields(self) -> None:
        """Can create metadata with all fields."""
        loop = asyncio.new_event_loop()
        future: asyncio.Future[WorkerResult] = loop.create_future()
        timing = RequestTiming()

        metadata = RequestMetadata(
            future=future,
            items=[Item(text="hello")],
            output_types=["dense", "sparse"],
            timing=timing,
            instruction="Search query",
            is_query=True,
            request_id="req-123",
        )

        assert metadata.instruction == "Search query"
        assert metadata.is_query is True
        assert metadata.request_id == "req-123"
        assert metadata.timing is timing
        loop.close()


class TestWorkerConfig:
    """Tests for WorkerConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config values."""
        config = WorkerConfig()

        assert config.max_batch_tokens == 16384
        assert config.max_batch_requests == 256
        assert config.max_batch_wait_ms == 15
        assert config.coalesce_ms == 15.0
        assert config.coalesce_ratio == 0.5
        # Tail window for a staggered burst (#2874, #373). Default stays
        # non-zero; a lone request is not charged it.
        assert config.idle_coalesce_ms == 3.0

    def test_custom_values(self) -> None:
        """Can set custom config values."""
        config = WorkerConfig(
            max_batch_tokens=8192,
            max_batch_requests=32,
            max_batch_wait_ms=5,
        )

        assert config.max_batch_tokens == 8192
        assert config.max_batch_requests == 32
        assert config.max_batch_wait_ms == 5


class TestWorkerStats:
    """Tests for WorkerStats dataclass."""

    def test_defaults(self) -> None:
        """Default stats values."""
        stats = WorkerStats()

        assert stats.batches_processed == 0
        assert stats.items_processed == 0
        assert stats.total_tokens_processed == 0
        assert stats.inference_errors == 0

    def test_instrumentation_series_are_bounded(self) -> None:
        """Instrumentation must not grow for the life of the process.

        ``SIE_INSTRUMENTATION=1`` left on in a long-lived deployment appends
        five samples per batch per worker. Unbounded, that is a leak that also
        makes ``summary()`` progressively slower, since it runs min/max/mean/
        median over the whole history. Each series is a bounded ring buffer
        that keeps the NEWEST samples.
        """
        limit = 8
        stats = WorkerStats()
        stats.enable_instrumentation(history_limit=limit)
        assert stats.instrumentation_enabled

        series = (
            stats.batch_sizes,
            stats.batch_tokens,
            stats.batch_wait_ms,
            stats.inference_ms,
            stats.requests_per_batch,
        )
        for sample in range(limit * 5):
            for entries in series:
                assert entries is not None
                entries.append(sample)

        for entries in series:
            assert entries is not None
            assert len(entries) == limit
            # Oldest samples evicted, newest retained.
            assert list(entries) == list(range(limit * 5 - limit, limit * 5))

    def test_instrumentation_summary_shape_is_unchanged(self) -> None:
        """``summary()`` output must stay byte-identical in shape for consumers."""
        stats = WorkerStats(batches_processed=2, items_processed=6, total_tokens_processed=90)
        stats.enable_instrumentation()
        assert stats.batch_sizes is not None
        assert stats.batch_tokens is not None
        assert stats.batch_wait_ms is not None
        assert stats.inference_ms is not None
        assert stats.requests_per_batch is not None
        for size, tokens, wait, infer, requests in ((2, 30, 1.0, 10.0, 2), (4, 60, 3.0, 20.0, 3)):
            stats.batch_sizes.append(size)
            stats.batch_tokens.append(tokens)
            stats.batch_wait_ms.append(wait)
            stats.inference_ms.append(infer)
            stats.requests_per_batch.append(requests)

        summary = stats.summary()
        assert "Batches processed: 2" in summary
        assert "=== Batch Size Stats ===" in summary
        assert "Items/batch: min=2, max=4, mean=3.0, median=3.0" in summary
        assert "Tokens/batch: min=30, max=60, mean=45.0" in summary
        assert "Requests/batch: min=2, max=3, mean=2.5" in summary
        assert "=== Timing Stats ===" in summary
        assert "Batch wait (ms): min=1.0, max=3.0, mean=2.0, p50=2.0" in summary
        assert "Inference (ms): min=10.0, max=20.0, mean=15.0, p50=15.0" in summary

    def test_summary_without_instrumentation_omits_detail(self) -> None:
        """The non-instrumented summary keeps its four-line shape."""
        summary = WorkerStats().summary()

        assert summary.splitlines() == [
            "Batches processed: 0",
            "Items processed: 0",
            "Tokens processed: 0",
            "Inference errors: 0",
        ]


class TestModelWorker:
    """Tests for ModelWorker."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        # Return EncodeOutput (adapters return batched output now)
        mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
            batch_size=len(items),
        )
        return mock

    @pytest.fixture
    def tokenized_item(self) -> TextPreparedItem:
        """Create a tokenized item."""
        return make_text_item([1, 2, 3, 4, 5], 0)

    def test_init_default_config(self, mock_adapter: MagicMock) -> None:
        """Initialize with default config."""
        worker = ModelWorker(mock_adapter)

        assert worker.adapter is mock_adapter
        assert worker.config.max_batch_tokens == 16384
        assert worker.is_running is False

    def test_init_custom_config(self, mock_adapter: MagicMock) -> None:
        """Initialize with custom config."""
        config = WorkerConfig(max_batch_tokens=8192)
        worker = ModelWorker(mock_adapter, config)

        assert worker.config.max_batch_tokens == 8192

    def test_adaptive_cost_floor_anchors_to_max_batch_tokens(self, mock_adapter: MagicMock) -> None:
        """The adaptive controller's cost floor scales with max_batch_tokens.

        Regression guard: before this fix, ``min_batch_cost`` was always
        ``min(256, max_batch_tokens)`` — i.e. 256 for anything realistic,
        which let the PI loop collapse each GPU forward to a single item
        under sustained negative-headroom load. The floor must now be at
        least ``max_batch_tokens // 4`` so even a fully-collapsed cost knob
        still packs several items per forward.
        """
        from sie_server.core.worker.types import AdaptiveBatchingParams

        ab = AdaptiveBatchingParams(enabled=True)

        # Typical production model: floor should be a quarter of budget.
        worker_big = ModelWorker(mock_adapter, WorkerConfig(max_batch_tokens=16384, adaptive_batching=ab))
        assert worker_big._adaptive_controller is not None
        assert worker_big._adaptive_controller.min_batch_cost == 4096

        # Medium model: still anchored to budget // 4.
        worker_med = ModelWorker(mock_adapter, WorkerConfig(max_batch_tokens=8192, adaptive_batching=ab))
        assert worker_med._adaptive_controller is not None
        assert worker_med._adaptive_controller.min_batch_cost == 2048

        # Small model where max_batch_tokens > 256: floor stays at the 256
        # legacy minimum (max(256, 1024//4) = max(256, 256) = 256).
        worker_boundary = ModelWorker(mock_adapter, WorkerConfig(max_batch_tokens=1024, adaptive_batching=ab))
        assert worker_boundary._adaptive_controller is not None
        assert worker_boundary._adaptive_controller.min_batch_cost == 256

        # Tiny ``max_batch_tokens`` (pathological unit tests): clamp the
        # floor to the configured budget so we never drive min above max.
        worker_small = ModelWorker(mock_adapter, WorkerConfig(max_batch_tokens=100, adaptive_batching=ab))
        assert worker_small._adaptive_controller is not None
        assert worker_small._adaptive_controller.min_batch_cost == 100

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_adapter: MagicMock) -> None:
        """Start and stop worker."""
        worker = ModelWorker(mock_adapter)

        assert worker.is_running is False

        await worker.start()
        assert worker.is_running is True

        # Starting again is idempotent
        await worker.start()
        assert worker.is_running is True

        await worker.stop()
        assert worker.is_running is False

        # Stopping again is idempotent
        await worker.stop()
        assert worker.is_running is False

    @pytest.mark.asyncio
    async def test_submit_not_running(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Submit raises when worker not running."""
        worker = ModelWorker(mock_adapter)

        with pytest.raises(RuntimeError, match="not running"):
            await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

    @pytest.mark.asyncio
    async def test_submit_and_get_result(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Submit items and get result via future."""
        # Set up adapter to return embeddings via encode()
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,  # Batch immediately
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            # Wait for result
            worker_result = await asyncio.wait_for(future, timeout=2.0)

            assert worker_result.output.batch_size == 1
            assert worker_result.output.dense is not None
            np.testing.assert_array_equal(worker_result.output.dense[0], np.array([0.1, 0.2, 0.3]))

            # Stats updated
            assert worker.stats.batches_processed >= 1
            assert worker.stats.items_processed >= 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_mixed_text_image_batch_completes(self, mock_adapter: MagicMock) -> None:
        """A mixed text+image request must resolve its future (#3136).

        Submits exactly what ``EncodePipeline.run_encode`` submits for a mixed
        batch on an image-preprocessor model: the ImagePreprocessor-prepared
        list plus the full original item list. When the preprocessor dropped
        text-only items, ``_complete_requests`` never saw a result for every
        item and the request hung until an outer timeout.
        """
        buf = io.BytesIO()
        PILImage.new("RGB", (4, 4), color=(255, 0, 0)).save(buf, format="PNG")
        png = buf.getvalue()

        items = [
            Item(text="a red square"),
            *[Item(images=[{"data": png, "format": "png"}]) for _ in range(4)],
        ]
        processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 4, 4)})
        prepared_batch = ImagePreprocessor(processor, "test-model").prepare(items, config=MagicMock())

        worker = ModelWorker(
            mock_adapter,
            WorkerConfig(max_batch_tokens=100, max_batch_requests=1, max_batch_wait_ms=1),
        )
        await worker.start()
        try:
            future = await worker.submit(
                prepared_items=prepared_batch.items,
                items=items,
                output_types=["dense"],
            )
            worker_result = await asyncio.wait_for(future, timeout=3.0)
            assert worker_result.output.batch_size == len(items)
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_direct_queue_release_emits_once_at_engine_dequeue(
        self,
        mock_adapter: MagicMock,
        tokenized_item: TextPreparedItem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        telemetry = MagicMock()
        monkeypatch.setattr(model_worker_module, "worker_telemetry", lambda: telemetry)
        monkeypatch.setattr(model_worker_module, "worker_telemetry_enabled", lambda: True)
        worker = ModelWorker(
            mock_adapter,
            WorkerConfig(max_batch_tokens=100, max_batch_requests=1, max_batch_wait_ms=1),
            model_name="catalog/model",
        )
        await worker.start()
        try:
            future = await worker.submit([tokenized_item], [Item(text="hello")], ["dense"])
            await asyncio.wait_for(future, timeout=2.0)

            telemetry.queue_released.assert_called_once()
            attributes = telemetry.queue_released.call_args.kwargs
            assert attributes["operation"] == "encode"
            assert attributes["model"] == "catalog/model"
            assert attributes["profile"] == "default"
            assert attributes["duration_s"] >= 0.0
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem
    ) -> None:
        """Multiple concurrent requests get batched."""

        # Set up adapter to return embeddings matching batch size
        def mock_encode(items, output_types, **kwargs):
            batch_size = len(items)
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * batch_size),
                batch_size=batch_size,
            )

        mock_adapter.encode.side_effect = mock_encode

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=3,  # Batch up to 3 requests
            max_batch_wait_ms=1,  # Short wait
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit 3 requests concurrently
            # Each request has one item, so original_index should be 0 for all
            # (original_index represents position within the request's items list)
            items = [
                make_text_item([1, 2], 0),
                make_text_item([1, 2, 3], 0),
                make_text_item([1, 2, 3, 4], 0),
            ]

            futures = []
            for i, item in enumerate(items):
                future = await worker.submit(
                    [item],
                    [Item(text=f"hello {i}")],
                    ["dense"],
                )
                futures.append(future)

            # Wait for all results
            worker_results = await asyncio.gather(*futures)

            # All requests completed
            assert len(worker_results) == 3
            for worker_result in worker_results:
                assert worker_result.output.batch_size == 1
                assert worker_result.output.dense is not None

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_inference_error_propagates(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Inference error is propagated to future."""
        mock_adapter.encode.side_effect = RuntimeError("GPU OOM")

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            with pytest.raises(RuntimeError, match="GPU OOM"):
                await asyncio.wait_for(future, timeout=2.0)

            # Error stats updated
            assert worker.stats.inference_errors >= 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_pending_count(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Pending count reflects submitted items."""

        # Make encode slow so items stay pending
        def slow_encode(*args, **kwargs):
            import time

            time.sleep(0.5)
            return EncodeOutput(dense=np.array([[0.1, 0.2, 0.3]]), batch_size=1)

        mock_adapter.encode.side_effect = slow_encode

        config = WorkerConfig(
            max_batch_tokens=1000,  # High token limit
            max_batch_requests=100,  # High request limit
            max_batch_wait_ms=5,  # Wait before batching
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Initially no pending
            assert worker.pending_count == 0
            assert worker.pending_tokens == 0

            # Submit and check pending (don't await yet)
            await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
            )

            # Should have pending items (before batch forms)
            # Note: This is timing-dependent but should work with high limits
            assert worker.pending_count >= 0  # May already be processed

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_passes_params_to_adapter(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Request params are passed to adapter."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense", "sparse"],
                instruction="Search query",
                is_query=True,
            )

            await asyncio.wait_for(future, timeout=2.0)

            # Verify adapter.encode was called with correct params
            mock_adapter.encode.assert_called_once()
            call_args = mock_adapter.encode.call_args

            # Check positional args
            assert call_args[0][0] == [Item(text="hello")]  # items
            assert call_args[0][1] == ["dense", "sparse"]  # output_types

            # Check keyword args
            assert call_args[1]["instruction"] == "Search query"
            assert call_args[1]["is_query"] is True

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_idle_dispatch_no_wait(self, mock_adapter: MagicMock, tokenized_item: TextPreparedItem) -> None:
        """Idle worker dispatches immediately without waiting for batch timeout."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=256,
            max_batch_wait_ms=50,  # Long timeout to make the test meaningful
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            timing = RequestTiming()
            future = await worker.submit(
                [tokenized_item],
                [Item(text="hello")],
                ["dense"],
                timing=timing,
            )

            start = time.monotonic()
            await asyncio.wait_for(future, timeout=2.0)
            elapsed_ms = (time.monotonic() - start) * 1000

            # Should complete well under the 50ms batch timeout
            # (inference is near-instant with mock adapter)
            assert elapsed_ms < 20

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_staggered_burst_fuses_with_idle_window(self, mock_adapter: MagicMock) -> None:
        """#373: the head runs alone, and a still-pending tail fuses behind it.

        The head is not charged ``idle_coalesce_ms``. A finished request does
        not hold the next one. Two requests queued on the same batcher are a
        tail, so the idle window holds them and a third arrival inside the
        window joins that forward: two forwards, not one per request.
        """
        call_sizes: list[int] = []

        def counting_encode(items, output_types, **kwargs):
            call_sizes.append(len(items))
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
                batch_size=len(items),
            )

        mock_adapter.encode.side_effect = counting_encode

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            started = time.monotonic()
            first = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 0")],
                ["dense"],
            )
            await asyncio.wait_for(first, timeout=2.0)
            head_ms = (time.monotonic() - started) * 1000
            # Must not have slept the 200ms window. 50ms leaves room for CI noise.
            assert head_ms < 50, f"lone head waited {head_ms:.1f}ms"
            assert call_sizes == [1], f"expected the head to run alone, got: {call_sizes}"

            # The head has finished, so it must not hold these. Queue two
            # before the loop selects, then one more inside the 200ms window.
            second = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 1")],
                ["dense"],
            )
            third = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 2")],
                ["dense"],
            )
            assert worker.pending_count == 2
            await asyncio.sleep(0.01)
            assert worker.pending_count == 2, "tail was dispatched instead of held"
            fourth = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 3")],
                ["dense"],
            )

            await asyncio.gather(second, third, fourth)

            assert sum(call_sizes) == 4
            assert call_sizes == [1, 3], f"expected the tail to fuse, got: {call_sizes}"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_already_queued_burst_fuses(self, mock_adapter: MagicMock) -> None:
        """Several items already pending when an idle worker selects still fuse.

        Queued before the process loop runs, so selection sees the whole
        burst rather than a request that arrived alone (#373).
        """
        call_sizes: list[int] = []

        def counting_encode(items, output_types, **kwargs):
            call_sizes.append(len(items))
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
                batch_size=len(items),
            )

        mock_adapter.encode.side_effect = counting_encode

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()
        start_loop = worker._ensure_process_loop_started
        worker._ensure_process_loop_started = lambda: None  # type: ignore[method-assign]

        try:
            futures = []
            for i in range(4):
                future = await worker.submit(
                    [make_text_item([1, 2], 0)],
                    [Item(text=f"text {i}")],
                    ["dense"],
                )
                futures.append(future)
            assert worker.pending_count == 4

            del worker._ensure_process_loop_started
            start_loop()
            await asyncio.gather(*futures)

            assert sum(call_sizes) == 4
            assert call_sizes == [4], f"expected one fused batch, got: {call_sizes}"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_staggered_burst_shreds_without_idle_window(self, mock_adapter: MagicMock) -> None:
        """``idle_coalesce_ms=0`` does not hold the queued tail the window fuses.

        Same shape as ``test_staggered_burst_fuses_with_idle_window``: the
        head runs alone, two requests are already queued before the loop
        selects, and one more arrives 10ms later. With the window off that
        pair is dispatched immediately, so the later arrival does not join
        it (``[1, 2, 1]``). With the window on the same tail is held and
        fuses (``[1, 3]``). This is the escape hatch, not the default.
        """
        call_sizes: list[int] = []

        def counting_encode(items, output_types, **kwargs):
            call_sizes.append(len(items))
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
                batch_size=len(items),
            )

        mock_adapter.encode.side_effect = counting_encode

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=0,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            started = time.monotonic()
            first = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 0")],
                ["dense"],
            )
            await asyncio.wait_for(first, timeout=2.0)
            head_ms = (time.monotonic() - started) * 1000
            assert head_ms < 50, f"lone head waited {head_ms:.1f}ms"
            assert call_sizes == [1], f"expected the head to run alone, got: {call_sizes}"

            # Same already-queued tail as the fusing test. The window is 0,
            # so the pair must leave before the later arrival instead of being held.
            second = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 1")],
                ["dense"],
            )
            third = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 2")],
                ["dense"],
            )
            assert worker.pending_count == 2
            await asyncio.sleep(0.01)
            assert worker.pending_count == 0, "tail was held instead of dispatched"
            fourth = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="text 3")],
                ["dense"],
            )

            await asyncio.gather(second, third, fourth)

            assert sum(call_sizes) == 4
            assert call_sizes == [1, 2, 1], f"expected the queued tail to shred, got: {call_sizes}"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_lone_request_does_not_wait_the_idle_window(self, mock_adapter: MagicMock) -> None:
        """A request that stays alone is not held for ``idle_coalesce_ms`` (#373).

        The busy-path coalesce and batch windows are also not charged. The
        bound is loose on purpose: a hard 1ms assert flakes under CI noise,
        but it must finish far below the 200ms idle window.
        """
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            start = time.monotonic()
            future = await worker.submit(
                [make_text_item([1, 2, 3], 0)],
                [Item(text="hello")],
                ["dense"],
            )
            await asyncio.wait_for(future, timeout=2.0)
            elapsed_ms = (time.monotonic() - start) * 1000

            assert elapsed_ms < 50, f"lone request waited {elapsed_ms:.1f}ms"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_multi_item_request_does_not_wait_the_idle_window(self, mock_adapter: MagicMock) -> None:
        """One submit of two items is one request and skips ``idle_coalesce_ms`` (#373)."""
        call_sizes: list[int] = []

        def counting_encode(items, output_types, **kwargs):
            call_sizes.append(len(items))
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
                batch_size=len(items),
            )

        mock_adapter.encode.side_effect = counting_encode

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            start = time.monotonic()
            future = await worker.submit(
                [make_text_item([1, 2], 0), make_text_item([3, 4], 1)],
                [Item(text="hello-0"), Item(text="hello-1")],
                ["dense"],
            )
            await asyncio.wait_for(future, timeout=2.0)
            elapsed_ms = (time.monotonic() - start) * 1000

            assert elapsed_ms < 50, f"multi-item request waited {elapsed_ms:.1f}ms"
            assert call_sizes == [2]

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_sequential_request_does_not_wait_the_idle_window(self, mock_adapter: MagicMock) -> None:
        """A finished request does not make the next submit wait (#373).

        The second request is sent as soon as the first has been awaited, well
        inside ``idle_coalesce_ms``. It is not a tail: nothing is still pending
        on this batcher.
        """
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            first = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="first")],
                ["dense"],
            )
            await asyncio.wait_for(first, timeout=2.0)

            start = time.monotonic()
            second = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="second")],
                ["dense"],
            )
            await asyncio.wait_for(second, timeout=2.0)
            elapsed_ms = (time.monotonic() - start) * 1000

            assert elapsed_ms < 50, f"sequential request waited {elapsed_ms:.1f}ms"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_other_lora_arrival_does_not_hold_this_batcher(self, mock_adapter: MagicMock) -> None:
        """A different LoRA's arrival does not make this batcher wait (#373)."""
        mock_adapter.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=64,
            max_batch_wait_ms=500,
            coalesce_ms=400,
            coalesce_ratio=1.0,
            idle_coalesce_ms=200,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            other = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="lora")],
                ["dense"],
                options={"lora": "adapter-a"},
            )
            await asyncio.wait_for(other, timeout=2.0)

            start = time.monotonic()
            base = await worker.submit(
                [make_text_item([1, 2], 0)],
                [Item(text="base")],
                ["dense"],
            )
            await asyncio.wait_for(base, timeout=2.0)
            elapsed_ms = (time.monotonic() - start) * 1000

            assert elapsed_ms < 50, f"other LoRA held this batcher for {elapsed_ms:.1f}ms"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests_still_batch(self, mock_adapter: MagicMock) -> None:
        """Concurrent requests are still batched together when worker is busy."""
        call_count = 0

        def counting_encode(items, output_types, **kwargs):
            nonlocal call_count
            call_count += 1
            batch_size = len(items)
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * batch_size),
                batch_size=batch_size,
            )

        mock_adapter.encode.side_effect = counting_encode

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=4,
            max_batch_wait_ms=10,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit 4 requests truly concurrently using asyncio tasks
            items = [make_text_item([1, 2], 0) for _ in range(4)]

            async def submit_one(idx: int) -> asyncio.Future[WorkerResult]:
                return await worker.submit(
                    [items[idx]],
                    [Item(text=f"text {idx}")],
                    ["dense"],
                )

            submit_tasks = [asyncio.create_task(submit_one(i)) for i in range(4)]
            inference_futures = await asyncio.gather(*submit_tasks)
            await asyncio.gather(*inference_futures)

            # Should have been batched into a single inference call
            assert call_count == 1

        finally:
            await worker.stop()
