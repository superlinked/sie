"""Opt-in progress and exact default-worker selection controls."""

import asyncio
import random
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from sie_server.core.batcher import BatchConfig, BatchFormer
from sie_server.core.prepared import ExtractPreparedItem
from sie_server.core.progress_batcher import ProgressBatchFormer
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import ModelWorker, RequestMetadata, WorkerConfig
from sie_server.core.worker.types import WorkerDrainedError
from sie_server.types.inputs import Item


@dataclass
class _Audio:
    """Audio item stub: cost is duration in ms, as ``AudioPreprocessor`` sets it."""

    cost: int
    original_index: int = 0
    runs_alone: bool = False


def _greedy_reference(pending: list[tuple[int, int]], config: BatchConfig, max_items: int | None) -> list[int]:
    """Cost-sorted greedy packing as it was before the runs_alone lane.

    ``pending`` holds ``(cost, id)`` in queue order and is updated in place,
    as ``BatchFormer`` updates its own queue.
    """
    pending.sort(key=lambda entry: entry[0])
    taken: list[int] = []
    batch_cost = 0
    for cost, request_id in pending:
        if max_items is not None and len(taken) >= max_items:
            break
        if batch_cost + cost > config.max_batch_cost and taken:
            break
        taken.append(request_id)
        batch_cost += cost
        if len(taken) >= config.max_batch_requests:
            break
    del pending[: len(taken)]
    return taken


class TestBatchableProgress:
    """Replenishing cheaper work cannot indefinitely bypass an older valid item."""

    @pytest.mark.parametrize(("budget", "heavy_cost"), [(4096, 12_000), (16_384, 12_000), (16_384, 24_000)])
    @pytest.mark.asyncio
    async def test_replenishing_cheap_items_cannot_starve_oldest(self, budget: int, heavy_cost: int) -> None:
        batcher: ProgressBatchFormer[_Audio, str] = ProgressBatchFormer(BatchConfig(max_batch_cost=budget))
        await batcher.submit(_Audio(heavy_cost), "heavy")

        served: list[list[str]] = []
        for turn in range(4):
            await batcher.submit_many([(_Audio(budget // 64), f"cheap-{turn}-{index}") for index in range(64)])
            batch = await batcher.try_drain()
            assert batch is not None
            served.append(batch.metadata)
            assert batch.total_cost <= budget or batch.size == 1

        assert "heavy" not in served[0]
        assert "heavy" in served[1]
        assert all(name.startswith("cheap-") for name in served[2])

    @pytest.mark.parametrize("cap", [None, 1, 3])
    @pytest.mark.asyncio
    async def test_two_heavy_documents_make_bounded_progress(self, cap: int | None) -> None:
        """All 48 individually oversized rows progress under continuous cheap arrivals."""
        batcher: ProgressBatchFormer[ExtractPreparedItem, str] = ProgressBatchFormer(BatchConfig(max_batch_cost=16_384))
        heavy = [
            (ExtractPreparedItem(cost, index), f"long{labels}/{index}")
            for labels, cost in ((7, 22_355), (8, 25_554))
            for index in range(24)
        ]
        await batcher.submit_many(heavy)
        served: list[str] = []
        for turn in range(96):
            await batcher.submit_many(
                [(ExtractPreparedItem(83, index), f"cheap/{turn}/{index}") for index in range(64)]
            )
            batch = await batcher.try_drain(max_items=cap)
            assert batch is not None
            if cap is not None:
                assert batch.size <= cap
            assert batch.total_cost <= 16_384 or batch.size == 1
            served.extend(batch.metadata)
            for position, (_, name) in enumerate(heavy):
                if turn + 1 >= 2 * (position + 1):
                    assert name in served
            assert batcher.pending_cost == sum(request.item.cost for request in batcher._pending)
        assert all(served.count(name) == 1 for _, name in heavy)
        assert any(name.startswith("cheap/") for name in served)
        remaining = batcher.drain_pending()
        assert len(served + remaining) == len(set(served + remaining)) == 48 + 96 * 64

    def test_reserved_suffix_packs_peers_without_wrapping_to_cheaper_items(self) -> None:
        batcher: ProgressBatchFormer[_Audio, str] = ProgressBatchFormer(BatchConfig(max_batch_cost=10))
        batcher._append_item(_Audio(4), "old")
        batcher._append_item(_Audio(4), "peer")
        for index in range(3):
            batcher._append_item(_Audio(3), f"cheap-{index}")
        assert batcher._extract_batch().metadata == ["cheap-0", "cheap-1", "cheap-2"]
        batcher._append_item(_Audio(1), "new-cheap")

        reserved = batcher._extract_batch()

        assert reserved.metadata == ["old", "peer"]
        assert reserved.total_cost == 8
        assert batcher._extract_batch().metadata == ["new-cheap"]

    def test_same_timestamp_uses_arrival_identity_not_metadata_equality(self) -> None:
        class Metadata:
            __hash__ = object.__hash__

            def __eq__(self, other: object) -> bool:
                raise AssertionError("batch selection must never compare metadata")

        batcher: ProgressBatchFormer[_Audio, Metadata] = ProgressBatchFormer(BatchConfig(max_batch_cost=10))
        first, second = Metadata(), Metadata()
        for item, metadata in [(_Audio(20), first), (_Audio(15), second)]:
            batcher._append_item(item, metadata)
            batcher._pending[-1].arrival_time = 1.0

        served = []
        for _ in range(4):
            batcher._append_item(_Audio(10), Metadata())
            batcher._pending[-1].arrival_time = 1.0
            served.append(batcher._extract_batch().metadata)
        assert served[1][0] is first
        assert served[3][0] is second

    @pytest.mark.parametrize("new_budget", [5, 12, 100])
    def test_reservation_survives_adaptive_budget_changes(self, new_budget: int) -> None:
        config = BatchConfig(max_batch_cost=10)
        batcher: ProgressBatchFormer[_Audio, str] = ProgressBatchFormer(config)
        batcher._append_item(_Audio(12), "old")
        batcher._append_item(_Audio(10), "cheap")
        assert batcher._extract_batch().metadata == ["cheap"]
        config.max_batch_cost = new_budget
        batcher._append_item(_Audio(1), "new-cheap")

        batch = batcher._extract_batch()

        assert batch.metadata == ["old"]
        assert batch.total_cost <= new_budget or batch.size == 1
        assert batcher._extract_batch().metadata == ["new-cheap"]

    @pytest.mark.asyncio
    async def test_flagged_and_zero_item_turns_do_not_consume_reservation(self) -> None:
        batcher: ProgressBatchFormer[_Audio, str] = ProgressBatchFormer(BatchConfig(max_batch_cost=10))
        await batcher.submit_many([(_Audio(20), "old"), (_Audio(10), "cheap"), (_Audio(30, runs_alone=True), "alone")])
        first = await batcher.try_drain()
        assert first is not None
        assert first.metadata == ["cheap"]
        reserved = batcher._reserved_request
        alone = await batcher.try_drain()
        assert alone is not None
        assert alone.metadata == ["alone"]
        assert batcher._reserved_request is reserved
        zero = await batcher.try_drain(max_items=0)
        assert zero is not None
        assert zero.size == 0
        assert batcher._reserved_request is reserved
        await batcher.submit_many([(_Audio(1), "new-cheap"), (_Audio(40, runs_alone=True), "next-alone")])

        served = []
        while batcher.pending_count:
            batch = await batcher.try_drain()
            assert batch is not None
            served.append(batch.metadata)
        assert served == [["old"], ["next-alone"], ["new-cheap"]]

    def test_shutdown_clears_reservation_without_losing_metadata(self) -> None:
        batcher: ProgressBatchFormer[_Audio, str] = ProgressBatchFormer(BatchConfig(max_batch_cost=10))
        batcher._append_item(_Audio(20), "old")
        batcher._append_item(_Audio(10), "cheap")
        assert batcher._extract_batch().metadata == ["cheap"]
        assert batcher._reserved_request is not None

        assert batcher.drain_pending() == ["old"]
        assert batcher._reserved_request is None
        assert batcher.pending_count == batcher.pending_cost == 0
        batcher._append_item(_Audio(8), "new-large")
        batcher._append_item(_Audio(1), "new-small")
        assert batcher._extract_batch().metadata == ["new-small", "new-large"]

    @pytest.mark.parametrize("seed", range(10))
    def test_replenishment_respects_arrival_bound_and_conserves_every_item(self, seed: int) -> None:
        rng = random.Random(seed)  # noqa: S311 -- deterministic adversarial arrivals
        batcher: ProgressBatchFormer[_Audio, int] = ProgressBatchFormer(
            BatchConfig(max_batch_cost=100, max_batch_requests=8)
        )
        waits: dict[int, list[int]] = {}
        served: list[int] = []
        next_id = 0

        def extract() -> None:
            batch = batcher._extract_batch(max_items=rng.choice([None, 0, 1, 3]))
            if not batch.size:
                return
            for wait in waits.values():
                wait[1] += 1
            for name in batch.metadata:
                older, turns = waits.pop(name)
                assert turns <= 2 * (older + 1)
            assert [item.cost for item in batch.items] == sorted(item.cost for item in batch.items)
            assert batch.total_cost <= batcher.config.max_batch_cost or batch.size == 1
            served.extend(batch.metadata)
            assert batcher.pending_count == len(waits)
            assert batcher.pending_cost == sum(request.item.cost for request in batcher._pending)

        for _ in range(100):
            for _ in range(rng.randint(1, 5)):
                waits[next_id] = [len(waits), 0]
                batcher._append_item(_Audio(rng.randint(1, 200)), next_id)
                batcher._pending[-1].arrival_time = 1.0
                next_id += 1
            batcher.config.max_batch_cost = rng.choice([50, 100, 300])
            extract()
        while batcher.pending_count:
            extract()
        assert sorted(served) == list(range(next_id))

    @pytest.mark.parametrize("max_items", [None, 1, 7])
    def test_homogeneous_memberships_and_batch_counts_match_original(self, max_items: int | None) -> None:
        config = BatchConfig(max_batch_cost=80, max_batch_requests=16)
        batcher: ProgressBatchFormer[_Audio, int] = ProgressBatchFormer(config)
        reference: list[tuple[int, int]] = []
        for turn in range(20):
            for index in range(32):
                name = turn * 32 + index
                batcher._append_item(_Audio(5), name)
                reference.append((5, name))
            assert batcher._extract_batch(max_items=max_items).metadata == _greedy_reference(
                reference, config, max_items
            )
            assert batcher._reserved_request is None
        while reference:
            assert batcher._extract_batch(max_items=max_items).metadata == _greedy_reference(
                reference, config, max_items
            )

    @pytest.mark.parametrize("seed", range(10))
    def test_fitting_mixed_cost_batches_match_original(self, seed: int) -> None:
        rng = random.Random(seed)  # noqa: S311 -- deterministic fitting batches
        config = BatchConfig(max_batch_cost=128, max_batch_requests=16)
        batcher: ProgressBatchFormer[_Audio, int] = ProgressBatchFormer(config)
        for _ in range(20):
            pending = [(rng.randint(1, 8), index) for index in range(16)]
            for cost, name in pending:
                batcher._append_item(_Audio(cost), name)
            assert batcher._extract_batch().metadata == _greedy_reference(pending, config, None)
            assert batcher.pending_count == 0
            assert batcher._reserved_request is None


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("enabled", [None, False])
def test_default_worker_traces_match_original_batcher(seed: int, enabled: bool | None) -> None:
    rng = random.Random(seed)  # noqa: S311 -- deterministic scheduling witness
    kwargs = {} if enabled is None else {"bounded_batch_progress": enabled}
    config = WorkerConfig(max_batch_tokens=100, max_batch_requests=8, **kwargs)
    adapter = MagicMock()
    adapter.max_concurrent_dispatch = 1
    adapter.lora_capability.return_value = None
    worker = ModelWorker(adapter, config)
    reference = BatchFormer(BatchConfig(max_batch_cost=100, max_batch_requests=8))
    actual = worker._get_batcher("selected" if seed % 2 else None)
    try:
        assert type(actual) is BatchFormer
        assert not hasattr(actual, "_arrivals")
        assert not hasattr(actual, "_reserved_request")
        next_id = 0
        for _ in range(80):
            for _ in range(rng.randint(0, 6)):
                cost, alone = rng.randint(1, 250), rng.random() < 0.2
                for batcher in (reference, actual):
                    batcher._append_item(_Audio(cost, runs_alone=alone), next_id)
                    batcher._pending[-1].arrival_time = float(next_id)
                next_id += 1
            budget = rng.choice([30, 100, 300])
            for batcher in (reference, actual):
                batcher.config.max_batch_cost = budget
            cap = rng.choice([None, 0, 1, 3])
            expected, observed = [batcher._extract_batch(max_items=cap) for batcher in (reference, actual)]
            assert observed.metadata == expected.metadata
            assert observed.total_cost == expected.total_cost
            assert [item.cost for item in observed.items] == [item.cost for item in expected.items]
            assert actual.pending_count == reference.pending_count
            assert actual.pending_cost == reference.pending_cost
        assert actual.drain_pending() == reference.drain_pending()
    finally:
        worker._inference_executor.shutdown(wait=True)


def test_progress_arrival_references_are_released_after_extract_and_drain() -> None:
    batcher = ProgressBatchFormer(BatchConfig(max_batch_cost=10))
    batcher._append_item(_Audio(20), "old")
    batcher._append_item(_Audio(10), "cheap")
    assert batcher._extract_batch().metadata == ["cheap"]
    assert [request.metadata for request in batcher._arrivals.values()] == ["old"]
    assert batcher.drain_pending() == ["old"]
    assert batcher._arrivals == {}
    assert batcher._reserved_request is None
    batcher._append_item(_Audio(1), "new")
    assert batcher._extract_batch().metadata == ["new"]
    assert batcher._arrivals == {}


@pytest.mark.parametrize("selection", [None, "selected"])
@pytest.mark.asyncio
async def test_worker_stop_clears_reserved_requests_and_cancelled_futures(selection: str | None) -> None:
    adapter = MagicMock()
    adapter.max_concurrent_dispatch = 1
    adapter.lora_capability.return_value = None
    worker = ModelWorker(adapter, WorkerConfig(max_batch_tokens=10, bounded_batch_progress=True))
    worker._running = True
    batcher = worker._get_batcher(selection)
    loop = asyncio.get_running_loop()
    futures = [loop.create_future() for _ in range(3)]
    metadata = [
        RequestMetadata(future=future, items=[Item(text="queued")], output_types=[], timing=RequestTiming())
        for future in futures
    ]
    batcher._append_item(_Audio(20), metadata[0])
    batcher._append_item(_Audio(10), metadata[1])
    assert batcher._extract_batch().metadata == [metadata[1]]
    # The extracted fixture item is already complete; the reserved item's
    # caller cancels before the worker drains every remaining queue on stop.
    futures[1].cancel()
    futures[0].cancel()
    batcher._append_item(_Audio(1), metadata[2])
    try:
        await worker.stop()
        assert futures[0].cancelled()
        with pytest.raises(WorkerDrainedError):
            await futures[2]
        assert batcher.pending_count == batcher.pending_cost == 0
        assert batcher._arrivals == {}
        assert batcher._reserved_request is None
        await worker.stop()
    finally:
        worker._inference_executor.shutdown(wait=True)
