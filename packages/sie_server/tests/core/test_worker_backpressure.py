import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item


class TestModelWorkerBackpressure:
    """Tests for backpressure (bounded queue) functionality."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.encode.return_value = EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]]),
            batch_size=1,
        )
        return mock

    def test_queue_full_error_raised(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised when queue exceeds max_queue_size."""
        from sie_server.core.worker import QueueFullError

        # Very small queue limit
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=5,  # Only allow 5 items
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit 5 items (should succeed - at limit)
                items = [make_text_item([1, 2], i) for i in range(5)]
                for item in items:
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {item.original_index}")],
                        ["dense"],
                    )

                # Try to submit one more (should fail)
                extra_item = make_text_item([3, 4], 5)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit(
                        [extra_item],
                        [Item(text="should fail")],
                        ["dense"],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_unlimited_queue_with_zero(self, mock_adapter: MagicMock) -> None:
        """max_queue_size=0 means unlimited queue."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=100,
            max_batch_wait_ms=1,
            max_queue_size=0,  # Unlimited
        )
        worker = ModelWorker(mock_adapter, config)

        async def test() -> None:
            await worker.start()
            try:
                # Submit many items (should all succeed)
                for i in range(100):
                    item = make_text_item([1, 2], i)
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {i}")],
                        ["dense"],
                    )
                # No QueueFullError raised
                assert worker.pending_count > 0
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_queue_rejects_batch_that_would_exceed(self, mock_adapter: MagicMock) -> None:
        """Queue rejects a batch if adding it would exceed the limit."""
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
                # Submit 3 items (leaves room for 2 more)
                items = [make_text_item([1, 2], i) for i in range(3)]
                for item in items:
                    await worker.submit(
                        [item],
                        [Item(text=f"hello {item.original_index}")],
                        ["dense"],
                    )

                # Try to submit 3 more items in one request (would make 6 total)
                batch_items = [make_text_item([3, 4], i) for i in range(3)]
                with pytest.raises(QueueFullError):
                    await worker.submit(
                        batch_items,
                        [Item(text=f"batch {i}") for i in range(3)],
                        ["dense"],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())

    def test_default_max_queue_size(self, mock_adapter: MagicMock) -> None:
        """Default max_queue_size is 1000."""
        config = WorkerConfig()
        assert config.max_queue_size == 1000
