"""Tests for batch formation module."""

import asyncio
import time

import pytest
from sie_server.core.batcher import (
    BatchConfig,
    BatchFormer,
    FormattedBatch,
    PendingRequest,
    collate_batch,
)
from sie_server.core.prepared import TextPayload, TextPreparedItem, make_text_item


class TestPendingRequest:
    """Tests for PendingRequest dataclass."""

    def test_create_request(self) -> None:
        """Can create a pending request."""
        item = make_text_item([1, 2, 3])
        request = PendingRequest(item=item, metadata={"id": "req-1"})

        assert request.item == item
        assert request.metadata == {"id": "req-1"}
        assert request.arrival_time > 0

    def test_arrival_time_auto_set(self) -> None:
        """Arrival time is automatically set."""
        item = make_text_item([1])
        before = time.monotonic()
        request = PendingRequest(item=item, metadata=None)
        after = time.monotonic()

        assert before <= request.arrival_time <= after


class TestFormattedBatch:
    """Tests for FormattedBatch dataclass."""

    def test_create_batch(self) -> None:
        """Can create a formatted batch."""
        items = [
            make_text_item([1, 2], 0),
            make_text_item([3, 4, 5], 1),
        ]
        batch = FormattedBatch(
            items=items,
            metadata=["meta-0", "meta-1"],
            total_cost=5,
        )

        assert batch.size == 2
        assert batch.total_tokens == 5  # Backward compat alias
        assert batch.total_cost == 5
        assert batch.metadata == ["meta-0", "meta-1"]

    def test_size_property(self) -> None:
        """Size returns number of items."""
        items = [make_text_item([1], i) for i in range(5)]
        batch = FormattedBatch(items=items, metadata=list(range(5)), total_cost=5)

        assert batch.size == 5

    def test_sorted_by_length(self) -> None:
        """Sorted by length returns items in ascending cost order."""
        items = [
            make_text_item([1, 2, 3, 4, 5], 0),  # longest
            make_text_item([1], 1),  # shortest
            make_text_item([1, 2, 3], 2),  # middle
        ]
        batch = FormattedBatch(
            items=items,
            metadata=["longest", "shortest", "middle"],
            total_cost=9,
        )

        sorted_batch = batch.sorted_by_length()

        # Items sorted ascending by cost
        assert sorted_batch.items[0].cost == 1
        assert sorted_batch.items[1].cost == 3
        assert sorted_batch.items[2].cost == 5

        # Metadata follows items
        assert sorted_batch.metadata[0] == "shortest"
        assert sorted_batch.metadata[1] == "middle"
        assert sorted_batch.metadata[2] == "longest"

        # Total tokens unchanged
        assert sorted_batch.total_tokens == 9

    def test_sorted_by_length_preserves_original(self) -> None:
        """Sorting creates new batch, doesn't modify original."""
        items = [
            make_text_item([1, 2], 0),
            make_text_item([1], 1),
        ]
        original = FormattedBatch(items=items, metadata=["a", "b"], total_cost=3)

        sorted_batch = original.sorted_by_length()

        # Original unchanged
        assert original.items[0].cost == 2
        assert original.metadata[0] == "a"

        # Sorted is different
        assert sorted_batch.items[0].cost == 1
        assert sorted_batch.metadata[0] == "b"


class TestBatchConfig:
    """Tests for BatchConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = BatchConfig()

        assert config.max_batch_tokens == 16384
        assert config.max_batch_requests == 64
        assert config.max_batch_wait_ms == 10

    def test_custom_values(self) -> None:
        """Can create config with custom values."""
        config = BatchConfig(
            max_batch_tokens=8192,
            max_batch_requests=32,
            max_batch_wait_ms=5,
        )

        assert config.max_batch_tokens == 8192
        assert config.max_batch_requests == 32
        assert config.max_batch_wait_ms == 5


class TestBatchFormer:
    """Tests for BatchFormer."""

    @pytest.fixture
    def config(self) -> BatchConfig:
        """Create a test config with short timeout."""
        return BatchConfig(
            max_batch_tokens=100,
            max_batch_requests=4,
            max_batch_wait_ms=50,
            coalesce_ms=10.0,
        )

    @pytest.fixture
    def batcher(self, config: BatchConfig) -> BatchFormer[TextPreparedItem, str]:
        """Create a BatchFormer with test config."""
        return BatchFormer(config)

    def test_init_default_config(self) -> None:
        """BatchFormer uses default config if none provided."""
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer()

        assert batcher.config.max_batch_tokens == 16384
        assert batcher.config.max_batch_requests == 64

    def test_init_custom_config(self, config: BatchConfig) -> None:
        """BatchFormer uses provided config."""
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        assert batcher.config.max_batch_tokens == 100
        assert batcher.config.max_batch_requests == 4

    def test_initial_state(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """BatchFormer starts empty."""
        assert batcher.pending_count == 0
        assert batcher.pending_tokens == 0

    @pytest.mark.asyncio
    async def test_submit_increases_counts(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Submit increases pending count and tokens."""
        item = make_text_item([1, 2, 3])

        await batcher.submit(item, "req-1")

        assert batcher.pending_count == 1
        assert batcher.pending_tokens == 3

    @pytest.mark.asyncio
    async def test_submit_multiple(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Can submit multiple requests."""
        item1 = make_text_item([1, 2], 0)
        item2 = make_text_item([3, 4, 5], 1)

        await batcher.submit(item1, "req-1")
        await batcher.submit(item2, "req-2")

        assert batcher.pending_count == 2
        assert batcher.pending_tokens == 5

    @pytest.mark.asyncio
    async def test_get_batch_returns_on_coalesce(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch returns after coalescing window when no more items arrive."""
        item = make_text_item([1, 2, 3])
        await batcher.submit(item, "req-1")

        start = time.monotonic()
        batch = await batcher.get_batch()
        elapsed = time.monotonic() - start

        # Should wait roughly the coalesce period (~10ms), not the full timeout (50ms)
        assert elapsed >= 0.005  # At least ~5ms
        assert elapsed < 0.10  # Well under the 50ms batch timeout
        assert batch.size == 1

    @pytest.mark.asyncio
    async def test_get_batch_returns_on_request_limit(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch returns immediately when request limit reached."""
        # Config has max_batch_requests=4
        items = [make_text_item([i], i) for i in range(4)]

        for i, item in enumerate(items):
            await batcher.submit(item, f"req-{i}")

        start = time.monotonic()
        batch = await batcher.get_batch()
        elapsed = time.monotonic() - start

        # Should return immediately (not wait for timeout)
        assert elapsed < 0.02
        assert batch.size == 4

    @pytest.mark.asyncio
    async def test_get_batch_returns_on_token_limit(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch returns immediately when token limit reached."""
        # Config has max_batch_tokens=100
        # Create one big item that hits the limit
        item = make_text_item(list(range(100)))

        await batcher.submit(item, "big-req")

        start = time.monotonic()
        batch = await batcher.get_batch()
        elapsed = time.monotonic() - start

        # Should return immediately
        assert elapsed < 0.02
        assert batch.size == 1
        assert batch.total_tokens == 100

    @pytest.mark.asyncio
    async def test_get_batch_returns_sorted(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch returns items sorted by cost."""
        items = [
            make_text_item([1, 2, 3, 4, 5], 0),
            make_text_item([1], 1),
            make_text_item([1, 2, 3], 2),
            make_text_item([1, 2], 3),  # triggers limit
        ]

        for i, item in enumerate(items):
            await batcher.submit(item, f"req-{i}")

        batch = await batcher.get_batch()

        # Should be sorted ascending by cost
        assert batch.items[0].cost == 1
        assert batch.items[1].cost == 2
        assert batch.items[2].cost == 3
        assert batch.items[3].cost == 5

    @pytest.mark.asyncio
    async def test_get_batch_clears_pending(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch clears pending requests."""
        items = [make_text_item([i], i) for i in range(4)]
        for i, item in enumerate(items):
            await batcher.submit(item, f"req-{i}")

        await batcher.get_batch()

        assert batcher.pending_count == 0
        assert batcher.pending_tokens == 0

    @pytest.mark.asyncio
    async def test_get_batch_preserves_metadata(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """get_batch preserves metadata correspondence with items."""
        items = [
            make_text_item([1, 2], 0),
            make_text_item([3], 1),
        ]
        await batcher.submit(items[0], "longer")
        await batcher.submit(items[1], "shorter")

        # Trigger batch
        await batcher.submit(make_text_item([4, 5], 2), "trigger")
        await batcher.submit(make_text_item([6], 3), "last")

        batch = await batcher.get_batch()

        # After sorting: shortest items first
        # Original order: 2 tokens, 1 token, 2 tokens, 1 token
        # Sorted: 1, 1, 2, 2
        assert batch.items[0].cost == 1
        assert batch.items[1].cost == 1
        assert batch.items[2].cost == 2
        assert batch.items[3].cost == 2

    def test_try_get_batch_returns_none_when_empty(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_get_batch returns None when no pending requests."""
        result = batcher.try_get_batch()

        assert result is None

    @pytest.mark.asyncio
    async def test_try_get_batch_returns_none_when_not_ready(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_get_batch returns None when batch not ready."""
        item = make_text_item([1, 2, 3])
        await batcher.submit(item, "req-1")

        # Batch not full and timeout not expired
        result = batcher.try_get_batch()

        assert result is None

    @pytest.mark.asyncio
    async def test_try_get_batch_returns_when_full(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_get_batch returns batch when full."""
        items = [make_text_item([i], i) for i in range(4)]
        for i, item in enumerate(items):
            await batcher.submit(item, f"req-{i}")

        result = batcher.try_get_batch()

        assert result is not None
        assert result.size == 4


class TestBatchFormerConcurrency:
    """Tests for BatchFormer concurrent behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_submits(self) -> None:
        """Multiple coroutines can submit concurrently."""
        config = BatchConfig(max_batch_requests=10, max_batch_wait_ms=1, coalesce_ms=0.5)
        batcher: BatchFormer[TextPreparedItem, int] = BatchFormer(config)

        async def submit_one(idx: int) -> None:
            item = make_text_item([idx], idx)
            await batcher.submit(item, idx)

        # Submit 10 items concurrently
        await asyncio.gather(*[submit_one(i) for i in range(10)])

        assert batcher.pending_count == 10
        assert batcher.pending_tokens == 10

    @pytest.mark.asyncio
    async def test_producer_consumer_pattern(self) -> None:
        """Producer submits while consumer waits for batch."""
        config = BatchConfig(
            max_batch_requests=4,
            max_batch_tokens=1000,
            max_batch_wait_ms=50,
            coalesce_ms=10.0,
        )
        batcher: BatchFormer[TextPreparedItem, int] = BatchFormer(config)
        batches: list[FormattedBatch[TextPreparedItem, int]] = []

        async def producer() -> None:
            for i in range(8):
                item = make_text_item([i], i)
                await batcher.submit(item, i)
                await asyncio.sleep(0)

        async def consumer() -> None:
            # Get two batches (8 items, 4 per batch)
            for _ in range(2):
                batch = await batcher.get_batch()
                batches.append(batch)

        await asyncio.gather(producer(), consumer())

        assert len(batches) == 2
        assert batches[0].size == 4
        assert batches[1].size == 4

    @pytest.mark.asyncio
    async def test_get_batch_waits_for_first_request(self) -> None:
        """get_batch returns after first request plus coalescing window."""
        # Use a very short timeout so test runs quickly
        config = BatchConfig(max_batch_wait_ms=200, max_batch_requests=10, coalesce_ms=20.0)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        async def delayed_submit() -> None:
            await asyncio.sleep(0.010)  # Small delay before submit
            item = make_text_item([1, 2, 3])
            await batcher.submit(item, "delayed")

        # Start submit in background
        submit_task = asyncio.create_task(delayed_submit())

        # get_batch will wait for first request, then coalesce window
        start = time.monotonic()
        batch = await asyncio.wait_for(batcher.get_batch(), timeout=1.0)
        elapsed = time.monotonic() - start

        await submit_task

        # Should have gotten the batch (submit at ~10ms + coalesce ~20ms = ~30ms)
        assert batch is not None
        assert batch.size == 1
        # Should take at least the coalesce period after first request
        assert elapsed >= 0.020  # At least submit delay + coalesce


class TestImmediateDispatch:
    """Tests for immediate dispatch mode in get_batch()."""

    @pytest.mark.asyncio
    async def test_immediate_returns_without_timeout(self) -> None:
        """get_batch(immediate=True) returns immediately when items are pending."""
        config = BatchConfig(max_batch_wait_ms=50, max_batch_requests=10)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        item = make_text_item([1, 2, 3])
        await batcher.submit(item, "req-1")

        start = time.monotonic()
        batch = await batcher.get_batch(immediate=True)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert batch.size == 1
        # Should return in well under the 50ms timeout
        assert elapsed_ms < 5

    @pytest.mark.asyncio
    async def test_immediate_false_waits_for_timeout(self) -> None:
        """get_batch(immediate=False) waits for timeout as usual."""
        config = BatchConfig(max_batch_wait_ms=50, max_batch_requests=10, coalesce_ms=100)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        item = make_text_item([1, 2, 3])
        await batcher.submit(item, "req-1")

        start = time.monotonic()
        batch = await batcher.get_batch(immediate=False)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert batch.size == 1
        # Should wait at least close to the timeout (coalesce_ms > max_batch_wait_ms
        # so the batch timeout is the binding constraint, not coalescing)
        assert elapsed_ms >= 40

    @pytest.mark.asyncio
    async def test_immediate_empty_waits_for_first_request(self) -> None:
        """get_batch(immediate=True) still waits when no items are pending."""
        config = BatchConfig(max_batch_wait_ms=50, max_batch_requests=10)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        async def delayed_submit() -> None:
            await asyncio.sleep(0.005)  # 5ms delay
            item = make_text_item([1, 2, 3])
            await batcher.submit(item, "delayed")

        submit_task = asyncio.create_task(delayed_submit())

        start = time.monotonic()
        batch = await asyncio.wait_for(batcher.get_batch(immediate=True), timeout=1.0)
        elapsed_ms = (time.monotonic() - start) * 1000

        await submit_task

        assert batch.size == 1
        # Should have waited for the submit (~5ms) but not the full timeout (~50ms)
        assert elapsed_ms >= 4
        assert elapsed_ms < 30

    @pytest.mark.asyncio
    async def test_immediate_still_sorts_by_cost(self) -> None:
        """Immediate mode still returns items sorted by cost."""
        config = BatchConfig(max_batch_wait_ms=50, max_batch_requests=10)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        await batcher.submit(make_text_item([1, 2, 3, 4, 5], 0), "big")
        await batcher.submit(make_text_item([1], 1), "small")
        await batcher.submit(make_text_item([1, 2, 3], 2), "mid")

        batch = await batcher.get_batch(immediate=True)

        assert batch.size == 3
        assert batch.items[0].cost == 1
        assert batch.items[1].cost == 3
        assert batch.items[2].cost == 5


class TestSubBatching:
    """Tests for sub-batching behavior (splitting large batches)."""

    @pytest.fixture
    def sub_batch_config(self) -> BatchConfig:
        """Config with small token limit to trigger sub-batching."""
        return BatchConfig(
            max_batch_tokens=50,  # Small limit to trigger splits
            max_batch_requests=10,
            max_batch_wait_ms=50,
            coalesce_ms=10.0,
        )

    @pytest.fixture
    def sub_batcher(self, sub_batch_config: BatchConfig) -> BatchFormer[TextPreparedItem, str]:
        """Create a BatchFormer configured for sub-batching tests."""
        return BatchFormer(sub_batch_config)

    @pytest.mark.asyncio
    async def test_splits_batch_on_token_limit(self, sub_batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Batch is split when items exceed token limit."""
        # Submit items totaling 80 tokens (30 + 10 + 25 + 15)
        items = [
            make_text_item(list(range(30)), 0),
            make_text_item(list(range(10)), 1),
            make_text_item(list(range(25)), 2),
            make_text_item(list(range(15)), 3),
        ]

        for i, item in enumerate(items):
            await sub_batcher.submit(item, f"req-{i}")

        # First batch should respect 50 token limit
        batch1 = await sub_batcher.get_batch()
        assert batch1.total_tokens <= 50

        # Should have remaining items
        assert sub_batcher.pending_count > 0

        # Second batch gets remaining items
        batch2 = await sub_batcher.get_batch()
        assert batch2.size >= 1

        # All items processed
        total_items = batch1.size + batch2.size
        assert total_items == 4

    @pytest.mark.asyncio
    async def test_sub_batches_are_sorted(self, sub_batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Each sub-batch is sorted by cost (ascending)."""
        items = [
            make_text_item(list(range(30)), 0),
            make_text_item(list(range(10)), 1),
            make_text_item(list(range(25)), 2),
            make_text_item(list(range(15)), 3),
        ]

        for i, item in enumerate(items):
            await sub_batcher.submit(item, f"req-{i}")

        batch1 = await sub_batcher.get_batch()
        costs1 = [item.cost for item in batch1.items]
        assert costs1 == sorted(costs1), "Batch 1 not sorted"

        if sub_batcher.pending_count > 0:
            batch2 = await sub_batcher.get_batch()
            costs2 = [item.cost for item in batch2.items]
            assert costs2 == sorted(costs2), "Batch 2 not sorted"

    @pytest.mark.asyncio
    async def test_metadata_preserved_through_split(self, sub_batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Metadata stays with correct items through split and sort."""
        items = [
            make_text_item(list(range(30)), 0),
            make_text_item(list(range(10)), 1),
        ]

        await sub_batcher.submit(items[0], "large")
        await sub_batcher.submit(items[1], "small")

        batch = await sub_batcher.get_batch()

        # After sorting, smallest should be first
        for item, meta in zip(batch.items, batch.metadata, strict=False):
            if item.cost == 10:
                assert meta == "small"
            elif item.cost == 30:
                assert meta == "large"

    @pytest.mark.asyncio
    async def test_single_large_item_exceeds_limit(self) -> None:
        """Single item larger than limit is still returned."""
        config = BatchConfig(max_batch_tokens=10, max_batch_requests=10, max_batch_wait_ms=10)
        batcher: BatchFormer[TextPreparedItem, str] = BatchFormer(config)

        # Single item with 50 tokens (exceeds 10 token limit)
        big_item = make_text_item(list(range(50)))
        await batcher.submit(big_item, "big")

        batch = await batcher.get_batch()

        # Should still get the item even though it exceeds limit
        assert batch.size == 1
        assert batch.total_tokens == 50

    @pytest.mark.asyncio
    async def test_greedy_packing(self, sub_batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """Greedy packing takes items until limit reached."""
        # Submit items: 10, 15, 20, 25 tokens
        # After sorting: 10, 15, 20, 25
        # Greedy packing with 50 limit: 10+15+20 = 45 (fits), 45+25 = 70 (doesn't fit)
        # So batch 1 should have 3 items (45 tokens), batch 2 should have 1 item (25 tokens)
        items = [
            make_text_item(list(range(25)), 0),
            make_text_item(list(range(10)), 1),
            make_text_item(list(range(20)), 2),
            make_text_item(list(range(15)), 3),
        ]

        for i, item in enumerate(items):
            await sub_batcher.submit(item, f"req-{i}")

        batch1 = await sub_batcher.get_batch()
        assert batch1.size == 3
        assert batch1.total_tokens == 45  # 10 + 15 + 20

        batch2 = await sub_batcher.get_batch()
        assert batch2.size == 1
        assert batch2.total_tokens == 25


class TestBatchFormerDrain:
    """Tests for BatchFormer try_drain (continuous batching)."""

    @pytest.fixture
    def config(self) -> BatchConfig:
        return BatchConfig(max_batch_tokens=100, max_batch_requests=4, max_batch_wait_ms=50)

    @pytest.fixture
    def batcher(self, config: BatchConfig) -> BatchFormer[TextPreparedItem, str]:
        return BatchFormer(config)

    @pytest.mark.asyncio
    async def test_try_drain_returns_none_when_empty(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_drain returns None when no pending requests."""
        result = await batcher.try_drain()
        assert result is None

    @pytest.mark.asyncio
    async def test_try_drain_returns_pending_items(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_drain returns pending items immediately without waiting for timeout."""
        item = make_text_item([1, 2, 3])
        await batcher.submit(item, "req-1")

        start = time.monotonic()
        result = await batcher.try_drain()
        elapsed = time.monotonic() - start

        # Should return immediately (not wait for 50ms timeout)
        assert elapsed < 0.005
        assert result is not None
        assert result.size == 1
        assert batcher.pending_count == 0

    @pytest.mark.asyncio
    async def test_try_drain_returns_all_accumulated(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """try_drain returns all accumulated items as a batch."""
        for i in range(3):
            await batcher.submit(make_text_item([i], i), f"req-{i}")

        result = await batcher.try_drain()
        assert result is not None
        assert result.size == 3
        assert batcher.pending_count == 0


class TestBatchFormerSubmitMany:
    """Tests for BatchFormer submit_many (bulk submit)."""

    @pytest.fixture
    def batcher(self) -> BatchFormer[TextPreparedItem, str]:
        return BatchFormer(BatchConfig(max_batch_tokens=100, max_batch_requests=10, max_batch_wait_ms=5))

    @pytest.mark.asyncio
    async def test_submit_many_adds_all_items(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """submit_many adds all items with single lock acquisition."""
        items = [(make_text_item([i], i), f"req-{i}") for i in range(5)]
        await batcher.submit_many(items)

        assert batcher.pending_count == 5
        assert batcher.pending_tokens == 5

    @pytest.mark.asyncio
    async def test_submit_many_empty_list(self, batcher: BatchFormer[TextPreparedItem, str]) -> None:
        """submit_many with empty list is a no-op."""
        await batcher.submit_many([])
        assert batcher.pending_count == 0


class TestCollateBatch:
    """Tests for collate_batch function."""

    def test_empty_items(self) -> None:
        """Collate returns empty for empty input."""
        result = collate_batch([])

        assert result == {"input_ids": [], "attention_mask": []}

    def test_single_item(self) -> None:
        """Collate handles single item."""
        payloads = [TextPayload([1, 2, 3], [1, 1, 1])]

        result = collate_batch(payloads)

        assert result["input_ids"] == [[1, 2, 3]]
        assert result["attention_mask"] == [[1, 1, 1]]

    def test_same_length_items(self) -> None:
        """Collate handles items of same length."""
        payloads = [
            TextPayload([1, 2, 3], [1, 1, 1]),
            TextPayload([4, 5, 6], [1, 1, 1]),
        ]

        result = collate_batch(payloads)

        assert result["input_ids"] == [[1, 2, 3], [4, 5, 6]]
        assert result["attention_mask"] == [[1, 1, 1], [1, 1, 1]]

    def test_pads_shorter_items(self) -> None:
        """Collate pads shorter items to max length."""
        payloads = [
            TextPayload([1, 2, 3], [1, 1, 1]),
            TextPayload([4, 5], [1, 1]),
            TextPayload([6], [1]),
        ]

        result = collate_batch(payloads)

        # All padded to length 3
        assert result["input_ids"] == [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
        assert result["attention_mask"] == [[1, 1, 1], [1, 1, 0], [1, 0, 0]]

    def test_custom_pad_token(self) -> None:
        """Collate uses custom pad token."""
        payloads = [
            TextPayload([1, 2, 3], [1, 1, 1]),
            TextPayload([4], [1]),
        ]

        result = collate_batch(payloads, pad_token_id=99)

        assert result["input_ids"] == [[1, 2, 3], [4, 99, 99]]
        assert result["attention_mask"] == [[1, 1, 1], [1, 0, 0]]

    def test_preserves_order(self) -> None:
        """Collate preserves item order (caller should pre-sort)."""
        payloads = [
            TextPayload([1], [1]),  # shortest
            TextPayload([2, 3], [1, 1]),  # middle
            TextPayload([4, 5, 6], [1, 1, 1]),  # longest
        ]

        result = collate_batch(payloads)

        # Order preserved, all padded to length 3
        assert result["input_ids"][0] == [1, 0, 0]
        assert result["input_ids"][1] == [2, 3, 0]
        assert result["input_ids"][2] == [4, 5, 6]
