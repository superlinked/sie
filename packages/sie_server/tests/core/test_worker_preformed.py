"""Tests for ModelWorker pre-formed sidecar batches.

The worker-sidecar owns queue pull, scheduling, batching, and adaptive
control. When Python receives an IPC batch from the sidecar, ModelWorker must
execute that caller-formed batch directly instead of submitting it to the
Python BatchFormer again. Direct ``submit*`` calls still use Python batching for
single-instance HTTP serving.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker import model_worker as model_worker_module
from sie_server.core.worker.model_worker import PreformedExtractRequest, PreformedScoreRequest
from sie_server.types.inputs import Item


class RecordingAdapter:
    def __init__(self) -> None:
        self.encode_calls: list[tuple[int, str | None]] = []
        self.extract_calls: list[tuple[int, str | None]] = []
        self.score_calls: list[int] = []
        self.set_lora_calls: list[str | None] = []
        self._current_lora: str | None = None

    def set_active_lora(self, lora: str | None) -> None:
        self._current_lora = lora
        self.set_lora_calls.append(lora)

    def encode(self, items, *args, **kwargs) -> EncodeOutput:
        self.encode_calls.append((len(items), self._current_lora))
        return EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32),
            batch_size=len(items),
        )

    def score_pairs(self, queries, items, *args, **kwargs) -> ScoreOutput:
        self.score_calls.append(len(items))
        return ScoreOutput(scores=np.arange(len(items), dtype=np.float32))

    def extract(self, items, *args, **kwargs) -> ExtractOutput:
        self.extract_calls.append((len(items), self._current_lora))
        return ExtractOutput(entities=[[] for _ in items])


@pytest.fixture
def adapter() -> RecordingAdapter:
    return RecordingAdapter()


@pytest.fixture
def prepared_item() -> TextPreparedItem:
    return make_text_item([1, 2, 3, 4, 5], 0)


@pytest.mark.asyncio
async def test_preformed_submit_dispatches_immediately(
    adapter: RecordingAdapter,
    prepared_item: TextPreparedItem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = MagicMock()
    monkeypatch.setattr(model_worker_module, "worker_telemetry", lambda: telemetry)
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        assert worker._process_task is None

        future = await worker.submit_preformed(
            [prepared_item],
            [Item(text="hello")],
            ["dense"],
        )

        assert future.done()
        worker_result = future.result()
        assert worker_result.output.batch_size == 1
        assert adapter.encode_calls == [(1, None)]
        assert worker.pending_count == 0
        assert worker._process_task is None
        telemetry.queue_released.assert_not_called()
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_submit_preserves_caller_batch(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        future = await worker.submit_preformed(
            [prepared_item, make_text_item([6, 7], 1)],
            [Item(text="a"), Item(text="b")],
            ["dense"],
        )

        assert future.done()
        assert adapter.encode_calls == [(2, None)]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_score_batch_preserves_sidecar_grouping(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        futures = await worker.submit_score_preformed_batch(
            [
                PreformedScoreRequest(
                    prepared_items=[prepared_item],
                    query=Item(text="q1"),
                    items=[Item(text="a")],
                ),
                PreformedScoreRequest(
                    prepared_items=[prepared_item, make_text_item([6, 7], 1)],
                    query=Item(text="q2"),
                    items=[Item(text="b"), Item(text="c")],
                ),
            ]
        )

        assert len(futures) == 2
        assert [f.result().output.batch_size for f in futures] == [1, 2]
        assert adapter.score_calls == [3]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_extract_batch_preserves_sidecar_grouping(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        futures = await worker.submit_extract_preformed_batch(
            [
                PreformedExtractRequest(
                    prepared_items=[prepared_item],
                    items=[Item(text="a")],
                    options={"lora": "lora-x"},
                ),
                PreformedExtractRequest(
                    prepared_items=[prepared_item],
                    items=[Item(text="b")],
                    options={"lora": "lora-x"},
                ),
            ],
            lora="lora-x",
        )

        assert len(futures) == 2
        assert [f.result().output.batch_size for f in futures] == [1, 1]
        assert adapter.extract_calls == [(2, "lora-x")]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_batch_rejects_empty_request_without_dispatch(adapter: RecordingAdapter) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        futures = await worker.submit_extract_preformed_batch(
            [
                PreformedExtractRequest(
                    prepared_items=[],
                    items=[],
                    options={"lora": "lora-x"},
                )
            ],
            lora="lora-x",
        )

        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=0.5)

        assert len(results) == 1
        assert isinstance(results[0], ValueError)
        assert str(results[0]) == "preformed request contains no prepared items"
        assert adapter.extract_calls == []
        assert adapter.set_lora_calls == []
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_score_batch_empty_request_does_not_block_siblings(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        futures = await worker.submit_score_preformed_batch(
            [
                PreformedScoreRequest(
                    prepared_items=[],
                    query=Item(text="q-empty"),
                    items=[],
                ),
                PreformedScoreRequest(
                    prepared_items=[prepared_item],
                    query=Item(text="q"),
                    items=[Item(text="a")],
                ),
            ]
        )

        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=0.5)

        assert len(results) == 2
        assert isinstance(results[0], ValueError)
        assert results[1].output.batch_size == 1
        assert adapter.score_calls == [1]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_submit_does_not_batch_across_calls(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        results = await asyncio.gather(
            worker.submit_preformed([prepared_item], [Item(text="a")], ["dense"]),
            worker.submit_preformed([prepared_item], [Item(text="b")], ["dense"]),
            worker.submit_preformed([prepared_item], [Item(text="c")], ["dense"]),
        )

        assert len(results) == 3
        assert adapter.encode_calls == [(1, None), (1, None), (1, None)]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_preformed_submit_serializes_lora_switch(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig())
    await worker.start()
    try:
        await asyncio.gather(
            worker.submit_preformed([prepared_item], [Item(text="a")], ["dense"], options={"lora": "lora-x"}),
            worker.submit_preformed(
                [prepared_item, make_text_item([6, 7], 1)],
                [Item(text="b"), Item(text="b2")],
                ["dense"],
                options={"lora": "lora-y"},
            ),
            worker.submit_preformed(
                [prepared_item, make_text_item([8, 9], 1), make_text_item([10], 2)],
                [Item(text="c"), Item(text="c2"), Item(text="c3")],
                ["dense"],
                options={"lora": "lora-x"},
            ),
        )

        assert sorted(adapter.encode_calls) == [(1, "lora-x"), (2, "lora-y"), (3, "lora-x")]
        assert sorted(lora or "" for lora in adapter.set_lora_calls) == ["lora-x", "lora-x", "lora-y"]
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_normal_submit_still_uses_python_batcher(
    adapter: RecordingAdapter, prepared_item: TextPreparedItem
) -> None:
    worker = ModelWorker(adapter, WorkerConfig(max_batch_wait_ms=20, coalesce_ms=5))
    await worker.start()
    try:
        assert worker._process_task is None

        futures = await asyncio.gather(
            worker.submit([prepared_item], [Item(text="a")], ["dense"]),
            worker.submit([make_text_item([6, 7], 0)], [Item(text="b")], ["dense"]),
        )
        await asyncio.gather(*futures)

        assert worker._process_task is not None
        assert adapter.encode_calls == [(2, None)]
    finally:
        await worker.stop()
