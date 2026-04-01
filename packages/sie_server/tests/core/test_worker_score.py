import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item


class TestModelWorkerScore:
    """Tests for score (reranking) via worker."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for scoring."""
        mock = MagicMock()
        # Return ScoreOutput based on item count
        mock.score_pairs.side_effect = lambda q, d, **kw: ScoreOutput(
            scores=np.array([0.9 - (i * 0.1) for i in range(len(d))], dtype=np.float32)
        )
        return mock

    @pytest.fixture
    def prepared_item(self) -> "ScorePreparedItem":
        """Create a prepared item for tests."""
        from sie_server.core.prepared import ScorePreparedItem

        return ScorePreparedItem(cost=50, original_index=0)

    @pytest.mark.asyncio
    async def test_submit_score_basic(self, mock_adapter: MagicMock, prepared_item) -> None:
        """Submit score returns results."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            query = Item(text="What is machine learning?")
            items = [Item(text="ML is a branch of AI.")]

            future = await worker.submit_score(
                [prepared_item],
                query,
                items,
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1
            assert len(result.output.scores) == 1

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_passes_instruction(self, mock_adapter: MagicMock, prepared_item) -> None:
        """Instruction is passed to adapter.score_pairs."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            query = Item(text="Query")
            items = [Item(text="Document")]

            future = await worker.submit_score(
                [prepared_item],
                query,
                items,
                instruction="Rank by relevance",
            )

            await asyncio.wait_for(future, timeout=2.0)

            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["instruction"] == "Rank by relevance"

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_concurrent_batching(self, mock_adapter: MagicMock) -> None:
        """Multiple concurrent score requests get batched together."""
        from sie_server.core.prepared import ScorePreparedItem

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=3,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            # Submit 3 score requests concurrently
            futures = []
            for i in range(3):
                item = ScorePreparedItem(cost=20, original_index=0)
                query = Item(text=f"Query {i}")
                docs = [Item(text=f"Doc {i}")]
                future = await worker.submit_score(
                    [item],
                    query,
                    docs,
                )
                futures.append(future)

            # Wait for all results
            results = await asyncio.gather(*futures)

            # All requests completed
            assert len(results) == 3
            for result in results:
                assert result.output.batch_size == 1
                assert len(result.output.scores) == 1

            # Verify batching happened
            total_items = sum(len(call.args[1]) for call in mock_adapter.score_pairs.call_args_list)
            assert total_items == 3

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_submit_score_groups_by_instruction(self, mock_adapter: MagicMock) -> None:
        """Requests with different instructions are grouped separately."""
        from sie_server.core.prepared import ScorePreparedItem

        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = ScorePreparedItem(cost=20, original_index=0)
            item2 = ScorePreparedItem(cost=20, original_index=0)

            future1 = await worker.submit_score(
                [item1],
                Item(text="Query 1"),
                [Item(text="Doc 1")],
                instruction="instruction A",
            )
            future2 = await worker.submit_score(
                [item2],
                Item(text="Query 2"),
                [Item(text="Doc 2")],
                instruction="instruction B",
            )

            await asyncio.gather(future1, future2)

            # Should have been 2 separate calls (different instructions)
            assert mock_adapter.score_pairs.call_count >= 2

            # Verify different instructions were passed
            instructions = [call.kwargs["instruction"] for call in mock_adapter.score_pairs.call_args_list]
            assert "instruction A" in instructions
            assert "instruction B" in instructions

        finally:
            await worker.stop()


class TestModelWorkerScoreBackpressure:
    """Tests for score backpressure."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter for scoring."""
        mock = MagicMock()
        mock.score_pairs.return_value = ScoreOutput(scores=np.array([0.5], dtype=np.float32))
        return mock

    def test_score_queue_full_error(self, mock_adapter: MagicMock) -> None:
        """QueueFullError raised for score when queue exceeds limit."""
        from sie_server.core.prepared import ScorePreparedItem
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
                    item = ScorePreparedItem(cost=20, original_index=0)
                    await worker.submit_score(
                        [item],
                        Item(text="Query"),
                        [Item(text=f"Doc {i}")],
                    )

                # Try to submit one more (should fail)
                extra_item = ScorePreparedItem(cost=20, original_index=0)
                with pytest.raises(QueueFullError, match="Queue full"):
                    await worker.submit_score(
                        [extra_item],
                        Item(text="Query"),
                        [Item(text="should fail")],
                    )
            finally:
                await worker.stop()

        asyncio.get_event_loop().run_until_complete(test())
