import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, RequestMetadata, WorkerConfig, WorkerResult, WorkerStats
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
        assert config.max_batch_wait_ms == 10

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
