import asyncio
from unittest.mock import MagicMock

import pytest
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity


class TestModelWorkerExtract:
    """Tests for submit_extract method."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that supports extraction."""
        mock = MagicMock()
        # Return ExtractOutput for each batch
        mock.extract.side_effect = lambda items, **kwargs: ExtractOutput(
            entities=[[Entity(text="Mock", label="test", score=0.9, start=0, end=4)] for _ in items]
        )
        return mock

    @pytest.fixture
    def prepared_item(self) -> "ExtractPreparedItem":
        """Create a prepared item for extract (uses character cost, not tokens)."""
        from sie_server.core.prepared import ExtractPreparedItem

        return ExtractPreparedItem(
            cost=11,  # Character count
            original_index=0,
        )

    @pytest.mark.asyncio
    async def test_submit_extract_basic(self, mock_adapter: MagicMock, prepared_item: "ExtractPreparedItem") -> None:
        """Submit extract returns results."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit_extract(
                [prepared_item],
                [Item(text="Hello world")],
                labels=["person", "organization"],
            )

            result = await asyncio.wait_for(future, timeout=2.0)

            assert result.output.batch_size == 1
            assert len(result.output.entities) == 1
            assert result.output.entities[0][0]["label"] == "test"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_passes_labels(
        self, mock_adapter: MagicMock, prepared_item: "ExtractPreparedItem"
    ) -> None:
        """Labels are passed to adapter.extract."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            future = await worker.submit_extract(
                [prepared_item],
                [Item(text="Hello world")],
                labels=["person", "organization", "location"],
            )

            await asyncio.wait_for(future, timeout=2.0)

            mock_adapter.extract.assert_called_once()
            call_kwargs = mock_adapter.extract.call_args.kwargs
            assert call_kwargs["labels"] == ["person", "organization", "location"]

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_concurrent_batching(self, mock_adapter: MagicMock) -> None:
        """Multiple concurrent extract requests get batched together."""
        from sie_server.core.prepared import ExtractPreparedItem

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=3,  # Batch up to 3 requests
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Create 3 prepared items for concurrent requests
            prepared_items = [ExtractPreparedItem(cost=6, original_index=0) for i in range(3)]

            # Submit 3 extract requests concurrently
            futures = []
            for i, item in enumerate(prepared_items):
                future = await worker.submit_extract(
                    [item],
                    [Item(text=f"Text {i}")],
                    labels=["entity"],
                )
                futures.append(future)

            # Wait for all results
            results = await asyncio.gather(*futures)

            # All requests completed
            assert len(results) == 3
            for result in results:
                assert result.output.batch_size == 1
                assert len(result.output.entities) == 1

            # Verify batching happened (adapter called with 3 items)
            # Note: Due to timing, might be 1 call with 3 items or multiple calls
            total_items = sum(len(call.args[0]) for call in mock_adapter.extract.call_args_list)
            assert total_items == 3

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_extract_groups_by_ordered_labels(self, mock_adapter: MagicMock) -> None:
        """Requests with different ordered labels are grouped separately."""
        from sie_server.core.prepared import ExtractPreparedItem

        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit requests with the same labels in different orders. Some
            # extractors condition on label order, so the worker must not
            # normalize it while grouping.
            item1 = ExtractPreparedItem(cost=6, original_index=0)
            item2 = ExtractPreparedItem(cost=6, original_index=0)

            future1 = await worker.submit_extract(
                [item1],
                [Item(text="Text 1")],
                labels=["person", "organization"],
            )
            future2 = await worker.submit_extract(
                [item2],
                [Item(text="Text 2")],
                labels=["organization", "person"],
            )

            await asyncio.gather(future1, future2)

            # Should have been 2 separate calls because label order differs.
            assert mock_adapter.extract.call_count >= 2

            label_orders = [tuple(call.kwargs["labels"]) for call in mock_adapter.extract.call_args_list]
            assert ("person", "organization") in label_orders
            assert ("organization", "person") in label_orders

        finally:
            await worker.stop()


class TestModelWorkerExtractIsolation:
    @pytest.mark.asyncio
    async def test_a_request_that_cannot_be_batched_fails_alone(self) -> None:
        from sie_server.core.prepared import ExtractPreparedItem
        from sie_server.types.inputs import InvalidInputError

        adapter = MagicMock()
        adapter.extract.side_effect = lambda items, **kwargs: ExtractOutput(entities=[[] for _ in items])
        worker = ModelWorker(adapter, WorkerConfig(max_batch_tokens=100, max_batch_requests=10, max_batch_wait_ms=20))
        await worker.start()
        try:
            bad = await worker.submit_extract(
                [ExtractPreparedItem(cost=6, original_index=0)],
                [Item(text="Text 1")],
                labels=["person"],
                instruction=["not", "a", "string"],  # ty:ignore[invalid-argument-type]
            )
            good = await worker.submit_extract(
                [ExtractPreparedItem(cost=6, original_index=0)], [Item(text="Text 2")], labels=["person"]
            )

            with pytest.raises(InvalidInputError):
                await asyncio.wait_for(bad, timeout=2.0)
            result = await asyncio.wait_for(good, timeout=2.0)
            assert result.output.entities == [[]]
        finally:
            await worker.stop()


class TestModelWorkerExtractBackpressure:
    """Tests for extract backpressure."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for extraction."""
        mock = MagicMock()
        mock.extract.return_value = ExtractOutput(entities=[[]])
        return mock

    def test_extract_queue_full_error(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised for extract when queue exceeds limit."""
        from sie_server.core.prepared import ExtractPreparedItem
        from sie_server.core.worker import QueueFullError

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 5 items (at limit)
                for i in range(5):
                    item = ExtractPreparedItem(cost=6, original_index=0)
                    await worker.submit_extract(
                        [item],
                        [Item(text=f"Text {i}")],
                        labels=["entity"],
                    )

                # Try to submit one more (should fail)
                extra_item = ExtractPreparedItem(cost=5, original_index=0)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit_extract(
                        [extra_item],
                        [Item(text="should fail")],
                        labels=["entity"],
                    )
            finally:
                await worker.stop()

        asyncio.new_event_loop().run_until_complete(test())
