import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput, ScoreOutput
from sie_server.core.prepared import ScorePreparedItem, make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.types.inputs import Item


class TestModelWorkerOptionsThreading:
    """Tests that runtime options are passed through to adapter.encode()."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that records kwargs."""
        mock = MagicMock()
        mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
            batch_size=len(items),
        )
        return mock

    @pytest.mark.asyncio
    async def test_options_forwarded_to_adapter_encode(self, mock_adapter: MagicMock) -> None:
        """Runtime options are passed through worker pipeline to adapter.encode()."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)
            runtime_options = {"query_template": "Represent this: {text}", "doc_template": "{text}"}

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                options=runtime_options,
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # Verify adapter.encode() was called with options
            mock_adapter.encode.assert_called_once()
            call_kwargs = mock_adapter.encode.call_args
            assert call_kwargs.kwargs["options"] == runtime_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_none_options_forwarded_to_adapter_encode(self, mock_adapter: MagicMock) -> None:
        """When no options provided, None is passed through to adapter.encode()."""
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                # No options
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # Verify adapter.encode() was called with options=None
            mock_adapter.encode.assert_called_once()
            call_kwargs = mock_adapter.encode.call_args
            assert call_kwargs.kwargs["options"] == {}

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_different_options_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Requests with different options are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item1 = make_text_item([1, 2, 3], 0)
            item2 = make_text_item([4, 5, 6], 0)

            future1 = await worker.submit(
                [item1],
                [Item(text="hello")],
                ["dense"],
                options={"query_template": "template_a"},
            )
            future2 = await worker.submit(
                [item2],
                [Item(text="world")],
                ["dense"],
                options={"query_template": "template_b"},
            )

            await asyncio.gather(future1, future2)

            # Different options should produce separate batches
            assert mock_adapter.encode.call_count == 2

            # Verify each call got the right options
            call_options = [call.kwargs.get("options") for call in mock_adapter.encode.call_args_list]
            assert {"query_template": "template_a"} in call_options
            assert {"query_template": "template_b"} in call_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_nested_options_preserved_through_pipeline(self, mock_adapter: MagicMock) -> None:
        """Nested dicts/lists in options survive the worker pipeline intact.

        options go through _make_hashable() for config_key grouping, but
        adapter.encode() should receive the original dict (from metadata),
        not a reconstructed version.  dict(options_tuple) would corrupt
        nested structures (dicts→tuples-of-pairs, lists→tuples).
        """
        config = WorkerConfig(
            max_batch_tokens=100,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            item = make_text_item([1, 2, 3], 0)
            nested_options = {
                "muvera": {"num_repetitions": 4, "rescale_factor": 0.5},
                "output_types": ["dense", "sparse"],
                "query_template": "Represent: {text}",
            }

            future = await worker.submit(
                [item],
                [Item(text="hello")],
                ["dense"],
                options=nested_options,
            )

            worker_result = await asyncio.wait_for(future, timeout=2.0)
            assert worker_result.output.batch_size == 1

            # The adapter must receive the original dict — not a
            # shallow reconstruction where nested dicts become tuples.
            mock_adapter.encode.assert_called_once()
            received_options = mock_adapter.encode.call_args.kwargs["options"]

            # Nested dict must still be a dict
            assert isinstance(received_options["muvera"], dict)
            assert received_options["muvera"] == {"num_repetitions": 4, "rescale_factor": 0.5}

            # List must still be a list
            assert isinstance(received_options["output_types"], list)
            assert received_options["output_types"] == ["dense", "sparse"]

            # Flat string preserved
            assert received_options["query_template"] == "Represent: {text}"

        finally:
            await worker.stop()


class TestModelWorkerScoreOptionsThreading:
    """Tests that runtime options are passed through to adapter.score_pairs()."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter that records kwargs."""
        mock = MagicMock()
        mock.score_pairs.side_effect = lambda q, d, **kw: ScoreOutput(scores=np.array([0.5] * len(d), dtype=np.float32))
        return mock

    @pytest.mark.asyncio
    async def test_score_options_forwarded_to_adapter(self, mock_adapter: MagicMock) -> None:
        """Runtime options are passed through worker pipeline to adapter.score_pairs()."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared = ScorePreparedItem(cost=50, original_index=0)
            runtime_options = {"max_seq_length": 256}

            future = await worker.submit_score(
                [prepared],
                Item(text="query"),
                [Item(text="document")],
                options=runtime_options,
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1

            # Verify adapter.score_pairs() was called with options
            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["options"] == runtime_options

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_score_none_options_forwarded(self, mock_adapter: MagicMock) -> None:
        """When no options provided, None is passed to adapter.score_pairs()."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=1,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared = ScorePreparedItem(cost=50, original_index=0)

            future = await worker.submit_score(
                [prepared],
                Item(text="query"),
                [Item(text="document")],
                # No options
            )

            result = await asyncio.wait_for(future, timeout=2.0)
            assert result.output.batch_size == 1

            # Verify adapter.score_pairs() was called with options=None
            mock_adapter.score_pairs.assert_called_once()
            call_kwargs = mock_adapter.score_pairs.call_args.kwargs
            assert call_kwargs["options"] is None

        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_score_different_options_batched_separately(self, mock_adapter: MagicMock) -> None:
        """Score requests with different options are batched separately."""
        config = WorkerConfig(
            max_batch_tokens=1000,
            max_batch_requests=10,
            max_batch_wait_ms=1,
        )
        worker = ModelWorker(mock_adapter, config)
        await worker.start()

        try:
            prepared1 = ScorePreparedItem(cost=50, original_index=0)
            prepared2 = ScorePreparedItem(cost=50, original_index=0)

            future1 = await worker.submit_score(
                [prepared1],
                Item(text="query 1"),
                [Item(text="doc 1")],
                options={"max_seq_length": 256},
            )
            future2 = await worker.submit_score(
                [prepared2],
                Item(text="query 2"),
                [Item(text="doc 2")],
                options={"max_seq_length": 512},
            )

            await asyncio.gather(future1, future2)

            # Different options should produce separate batches
            assert mock_adapter.score_pairs.call_count == 2

            # Verify each call got the right options
            call_options = [call.kwargs.get("options") for call in mock_adapter.score_pairs.call_args_list]
            assert {"max_seq_length": 256} in call_options
            assert {"max_seq_length": 512} in call_options

        finally:
            await worker.stop()
