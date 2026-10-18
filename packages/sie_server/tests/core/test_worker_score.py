import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.prepared import ScorePreparedItem
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.model_worker import PreformedScoreRequest
from sie_server.core.worker.types import LoraNotReadyError, WorkerResult
from sie_server.types.inputs import InvalidInputError, Item


class LoraScoreAdapter:
    """Scores expose the active weights so selection bugs change the result."""

    def __init__(self, *, supports_lora: bool = True) -> None:
        self._supports_lora = supports_lora
        self.active_lora: str | None = None
        self.calls: list[tuple[str | None, int]] = []

    def supports_lora(self) -> bool:
        return self._supports_lora

    def max_concurrent_dispatch(self) -> int:
        return 4

    def set_active_lora(self, lora: str | None) -> None:
        if lora not in (None, "legal", "medical"):
            raise ValueError("LoRA not loaded; private adapter inventory")
        self.active_lora = lora

    def score_pairs(self, queries, items, **kwargs) -> ScoreOutput:
        self.calls.append((self.active_lora, len(items)))
        score = {None: 0.0, "legal": 1.0, "medical": 2.0}[self.active_lora]
        return ScoreOutput(scores=np.full(len(items), score, dtype=np.float32))


def _lora_score_request(options: dict[str, Any] | None = None, *, count: int = 1) -> PreformedScoreRequest:
    return PreformedScoreRequest(
        prepared_items=[ScorePreparedItem(cost=20, original_index=i) for i in range(count)],
        query=Item(text="query"),
        items=[Item(text=f"document {i}") for i in range(count)],
        options=options,
    )


async def _submit_lora_score(
    worker: ModelWorker, entrypoint: str, options: dict[str, Any] | None
) -> asyncio.Future[WorkerResult]:
    request = _lora_score_request(options)
    if entrypoint == "submit_score_preformed_batch":
        return (await worker.submit_score_preformed_batch([request]))[0]
    return await getattr(worker, entrypoint)(
        request.prepared_items, request.query, request.items, options=request.options
    )


class TestModelWorkerScoreLoRA:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["submit_score", "submit_score_preformed"])
    async def test_concurrent_selections_and_following_base_request(self, entrypoint: str) -> None:
        adapter = LoraScoreAdapter()
        worker = ModelWorker(adapter, WorkerConfig(max_batch_wait_ms=1))
        await worker.start()
        try:
            futures = await asyncio.gather(
                *(
                    _submit_lora_score(worker, entrypoint, {"lora": lora})
                    for lora in ("legal", "medical", "legal", None)
                )
            )
            results = await asyncio.wait_for(asyncio.gather(*futures), 2)
            assert [result.output.scores.tolist() for result in results] == [[1.0], [2.0], [1.0], [0.0]]

            selected = await _submit_lora_score(worker, entrypoint, {"lora": "medical"})
            await asyncio.wait_for(selected, 2)
            base = await _submit_lora_score(worker, entrypoint, None)
            assert (await asyncio.wait_for(base, 2)).output.scores.tolist() == [0.0]
            assert adapter.calls[-2:] == [("medical", 1), (None, 1)]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_preformed_batch_groups_same_lora_and_preserves_request_order(self) -> None:
        adapter = LoraScoreAdapter()
        worker = ModelWorker(adapter)
        await worker.start()
        try:
            futures = await worker.submit_score_preformed_batch(
                [
                    _lora_score_request({"lora": "legal"}, count=2),
                    _lora_score_request({"lora": "medical"}),
                    _lora_score_request({"lora": "legal"}),
                    _lora_score_request(),
                ]
            )
            results = await asyncio.wait_for(asyncio.gather(*futures), 2)
            assert [result.output.scores.tolist() for result in results] == [[1.0, 1.0], [2.0], [1.0], [0.0]]
            assert adapter.calls == [("legal", 3), ("medical", 1), (None, 1)]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["submit_score", "submit_score_preformed", "submit_score_preformed_batch"])
    @pytest.mark.parametrize(
        ("options", "expected"),
        [
            ({"lora_id": "legal"}, 1.0),
            ({"lora_id": "legal", "lora": "medical"}, 2.0),
            ({"lora_id": "legal", "lora": None}, 0.0),
        ],
    )
    async def test_legacy_lora_id_and_explicit_selection_precedence(
        self, entrypoint: str, options: dict[str, Any], expected: float
    ) -> None:
        adapter = LoraScoreAdapter()
        worker = ModelWorker(adapter, WorkerConfig(max_batch_wait_ms=1))
        await worker.start()
        try:
            future = await _submit_lora_score(worker, entrypoint, options)
            assert (await asyncio.wait_for(future, 2)).output.scores.tolist() == [expected]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["submit_score", "submit_score_preformed", "submit_score_preformed_batch"])
    async def test_unsupported_selection_rejects_without_scoring_base(self, entrypoint: str) -> None:
        adapter = LoraScoreAdapter(supports_lora=False)
        worker = ModelWorker(adapter, WorkerConfig(max_batch_wait_ms=1))
        await worker.start()
        try:
            with pytest.raises(InvalidInputError, match="does not support LoRA scoring"):
                future = await _submit_lora_score(worker, entrypoint, {"lora": "legal"})
                await asyncio.wait_for(future, 2)
            assert adapter.calls == []
            base = await _submit_lora_score(worker, entrypoint, None)
            assert (await asyncio.wait_for(base, 2)).output.scores.tolist() == [0.0]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("selection", ["", " ", 3, ["legal"]])
    async def test_invalid_preformed_selection_does_not_block_sibling(self, selection: object) -> None:
        adapter = LoraScoreAdapter()
        worker = ModelWorker(adapter)
        await worker.start()
        try:
            futures = await worker.submit_score_preformed_batch(
                [_lora_score_request({"lora": selection}), _lora_score_request()]
            )
            results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), 2)
            assert isinstance(results[0], InvalidInputError)
            assert results[1].output.scores.tolist() == [0.0]
            assert adapter.calls == [(None, 1)]
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("entrypoint", ["submit_score", "submit_score_preformed", "submit_score_preformed_batch"])
    async def test_evicted_selection_requests_reload_and_never_scores_base(self, entrypoint: str) -> None:
        adapter = LoraScoreAdapter()
        callbacks = MagicMock()
        callbacks.request_lora_load.return_value = 7
        worker = ModelWorker(
            adapter, WorkerConfig(max_batch_wait_ms=1), model_name="org/model", registry_callbacks=callbacks
        )
        await worker.start()
        try:
            future = await _submit_lora_score(worker, entrypoint, {"lora": "evicted"})
            with pytest.raises(LoraNotReadyError) as raised:
                await asyncio.wait_for(future, 2)
            assert raised.value.retry_after_s == 7
            assert "private" not in str(raised.value)
            callbacks.request_lora_load.assert_called_once_with("org/model", "evicted")
            assert adapter.calls == []
            base = await _submit_lora_score(worker, entrypoint, None)
            assert (await asyncio.wait_for(base, 2)).output.scores.tolist() == [0.0]
        finally:
            await worker.stop()


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

        asyncio.new_event_loop().run_until_complete(test())
