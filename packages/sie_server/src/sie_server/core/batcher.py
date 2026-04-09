"""Batch formation for dynamic batching.

Accumulates requests until batch limits are reached, then returns a batch
ready for inference. See DESIGN.md Section 5.1.

Batch formation loop:
1. Wait for first request
2. Start batch timer
3. Accumulate while: total_cost < max_batch_cost AND
                     num_requests < max_batch_requests AND
                     elapsed < max_batch_wait_ms
4. Sort by cost (reduces padding waste for text)
5. Split into optimal sub-batches respecting max_batch_cost
6. Return one sub-batch per get_batch() call

The key optimization is cost-sorted sub-batching:
- Items are sorted by cost (ascending)
- Each sub-batch contains similar-cost items
- This minimizes padding waste within each sub-batch

Cost semantics vary by modality:
- Text: cost = token count
- Images: cost = 1 per image (fixed dimensions)
- Audio: cost = sample_count / chunk_size
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sie_server.core.prepared import TextPayload

logger = logging.getLogger(__name__)


@runtime_checkable
class HasCost(Protocol):
    """Protocol for items that have a batching cost.

    PreparedItem and other prepared item types satisfy this protocol.
    """

    @property
    def cost(self) -> int:
        """Return batching cost for this item."""
        ...

    @property
    def original_index(self) -> int:
        """Return original position in request."""
        ...


@dataclass(slots=True)
class PendingRequest[I: HasCost, T]:
    """A request waiting to be batched.

    Holds the prepared item plus metadata for routing results.
    """

    item: I
    metadata: T
    arrival_time: float = field(default_factory=time.monotonic)


@dataclass(slots=True)
class FormattedBatch[I: HasCost, T]:
    """A batch ready for inference.

    Items are sorted by cost (ascending) to minimize padding.
    Metadata is preserved for routing results back to requesters.
    """

    items: list[I]
    metadata: list[T]
    total_cost: int

    @property
    def size(self) -> int:
        """Number of items in the batch."""
        return len(self.items)

    @property
    def total_tokens(self) -> int:
        """Return total tokens (alias for total_cost for backward compatibility)."""
        return self.total_cost

    def sorted_by_cost(self) -> FormattedBatch[I, T]:
        """Return a new batch with items sorted by cost (ascending).

        Sorting by cost reduces padding waste when collating to tensors.
        Preserves the correspondence between items and metadata.
        """
        pairs = list(zip(self.items, self.metadata, strict=True))
        pairs.sort(key=lambda p: p[0].cost)

        sorted_items = [p[0] for p in pairs]
        sorted_metadata = [p[1] for p in pairs]

        return FormattedBatch(
            items=sorted_items,
            metadata=sorted_metadata,
            total_cost=self.total_cost,
        )

    def sorted_by_length(self) -> FormattedBatch[I, T]:
        """Alias for sorted_by_cost (backward compatibility)."""
        return self.sorted_by_cost()


class BatchConfig:
    """Configuration for batch formation.

    Cost semantics:
    - For text: cost = token count
    - For images: cost = 1 per image
    - For mixed: sum of component costs
    """

    def __init__(
        self,
        max_batch_cost: int = 16384,
        max_batch_requests: int = 64,
        max_batch_wait_ms: float = 10.0,
        *,
        max_batch_tokens: int | None = None,  # Backward compatibility alias
        coalesce_ms: float = 2.0,
        coalesce_ratio: float = 0.2,
    ) -> None:
        """Initialize batch configuration.

        Args:
            max_batch_cost: Maximum total cost per batch.
            max_batch_requests: Maximum requests per batch.
            max_batch_wait_ms: Maximum wait time before yielding batch.
            max_batch_tokens: Alias for max_batch_cost (backward compatibility).
            coalesce_ms: Idle coalescing window ceiling. If no new items arrive
                within this period, yield the batch even if max_batch_wait_ms
                hasn't expired. The actual coalescing window may be smaller;
                see effective_coalesce_ms.
            coalesce_ratio: Coalesce window as a fraction of max_batch_wait_ms.
                The effective coalesce window is
                ``min(coalesce_ms, max_batch_wait_ms * coalesce_ratio)``.
                This keeps coalesce proportional when the adaptive controller
                changes max_batch_wait_ms. Default 0.2 (20%).
        """
        # Allow max_batch_tokens as an alias for max_batch_cost
        if max_batch_tokens is not None:
            self.max_batch_cost = max_batch_tokens
        else:
            self.max_batch_cost = max_batch_cost
        self.max_batch_requests = max_batch_requests
        self.max_batch_wait_ms = max_batch_wait_ms
        self.coalesce_ms = coalesce_ms
        self.coalesce_ratio = coalesce_ratio

    @property
    def effective_coalesce_ms(self) -> float:
        """Coalesce window, scaled proportionally to max_batch_wait_ms.

        Returns ``min(coalesce_ms, max_batch_wait_ms * coalesce_ratio)``
        so the window stays proportional when the adaptive controller
        changes max_batch_wait_ms. The base ``coalesce_ms`` acts as a ceiling.
        """
        return min(self.coalesce_ms, self.max_batch_wait_ms * self.coalesce_ratio)

    @property
    def max_batch_tokens(self) -> int:
        """Alias for max_batch_cost (backward compatibility)."""
        return self.max_batch_cost


class BatchFormer[I: HasCost, T]:
    """Accumulates requests into batches based on cost/request limits and timeout.

    Thread-safe for async use. Multiple coroutines can submit requests concurrently.
    Generic over item type I (any HasCost implementation) and metadata type T.

    Usage:
        batcher = BatchFormer(config)

        # In request handler:
        await batcher.submit(item, metadata)

        # In batch processor:
        batch = await batcher.get_batch()
        # process batch...
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        """Initialize the batch former.

        Args:
            config: Batch configuration. Uses defaults if not provided.
        """
        self._config = config or BatchConfig()
        self._pending: list[PendingRequest[I, T]] = []
        self._total_cost = 0
        self._lock = asyncio.Lock()
        self._batch_ready = asyncio.Event()
        self._first_request_time: float | None = None
        self._last_submit_time: float | None = None

    @property
    def config(self) -> BatchConfig:
        """Return the batch configuration."""
        return self._config

    @property
    def pending_count(self) -> int:
        """Return number of pending requests."""
        return len(self._pending)

    @property
    def pending_cost(self) -> int:
        """Return total cost in pending requests."""
        return self._total_cost

    @property
    def pending_tokens(self) -> int:
        """Alias for pending_cost (backward compatibility)."""
        return self._total_cost

    def _batch_is_full(self) -> bool:
        """Check if batch limits have been reached."""
        return self._total_cost >= self._config.max_batch_cost or len(self._pending) >= self._config.max_batch_requests

    def _batch_timeout_expired(self) -> bool:
        """Check if batch timeout has expired."""
        if self._first_request_time is None:
            return False
        elapsed_ms = (time.monotonic() - self._first_request_time) * 1000
        return elapsed_ms >= self._config.max_batch_wait_ms

    def _coalesce_expired(self) -> bool:
        """Check if the coalescing window has expired (no new items recently).

        Returns True when items are pending and no new items have arrived
        within the effective coalescing period. This detects the end of a
        burst — all concurrent requests have been submitted, so waiting
        longer won't add more items to the batch.

        Uses effective_coalesce_ms which scales proportionally with
        max_batch_wait_ms to stay relevant across the adaptive range.
        """
        if self._last_submit_time is None or not self._pending:
            return False
        elapsed_ms = (time.monotonic() - self._last_submit_time) * 1000
        return elapsed_ms >= self._config.effective_coalesce_ms

    def _should_yield_batch(self) -> bool:
        """Check if batch should be yielded (full, timeout, or coalesced)."""
        if len(self._pending) == 0:
            return False
        return self._batch_is_full() or self._batch_timeout_expired() or self._coalesce_expired()

    async def submit(self, item: I, metadata: T) -> None:
        """Submit an item for batching.

        Args:
            item: Item with cost property (any HasCost implementation).
            metadata: Request metadata (e.g., request ID, params, future).
        """
        async with self._lock:
            self._append_item(item, metadata)

    async def submit_many(self, items: list[tuple[I, T]]) -> None:
        """Submit multiple items for batching with a single lock acquisition.

        More efficient than calling submit() per item for multi-item requests.

        Args:
            items: List of (item, metadata) tuples to submit.
        """
        async with self._lock:
            for item, metadata in items:
                self._append_item(item, metadata)

    def _append_item(self, item: I, metadata: T) -> None:
        """Append item to pending list (caller must hold lock)."""
        request = PendingRequest(item=item, metadata=metadata)
        self._pending.append(request)
        self._total_cost += item.cost

        now = time.monotonic()
        self._last_submit_time = now

        # Track first request time for timeout
        is_first_request = self._first_request_time is None
        if is_first_request:
            self._first_request_time = now

        # Signal if batch is ready OR if this is the first request
        # (first request wakes up get_batch so it can start the timeout)
        if self._should_yield_batch() or is_first_request:
            self._batch_ready.set()

    async def get_batch(self, *, immediate: bool = False) -> FormattedBatch[I, T]:
        """Wait for and return the next batch.

        Blocks until:
        - Batch is full (cost or request limit reached), OR
        - Timeout expires after first request, OR
        - immediate=True and any items are pending

        Args:
            immediate: If True, yield immediately when any items are pending
                without waiting for timeout. Used when the worker was idle
                to eliminate unnecessary batch wait at low concurrency.

        Returns:
            FormattedBatch with items sorted by cost (ascending).
            The batch respects max_batch_cost limit for optimal GPU utilization.
        """
        while True:
            async with self._lock:
                if self._should_yield_batch() or (immediate and len(self._pending) > 0):
                    # _extract_batch already sorts by cost and respects limits
                    return self._extract_batch()

            # Wait for batch ready signal or timeout
            timeout_s = self._get_wait_timeout()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._batch_ready.wait(),
                    timeout=timeout_s,
                )

            # Clear event for next batch
            self._batch_ready.clear()

    def _get_wait_timeout(self) -> float | None:
        """Calculate how long to wait for more requests.

        Returns the shorter of:
        - Time remaining in the batch formation window (max_batch_wait_ms)
        - Time remaining until the coalesce window expires (based on when the
          last item was submitted, not a fixed polling interval)

        This ensures precise coalesce detection: the sleep duration is exactly
        the time until coalesce fires, rather than polling every coalesce_ms.
        """
        if self._first_request_time is None:
            return None  # Wait indefinitely for first request

        now = time.monotonic()
        elapsed_ms = (now - self._first_request_time) * 1000
        batch_remaining_ms = max(0, self._config.max_batch_wait_ms - elapsed_ms)

        # Compute actual time remaining until coalesce fires
        coalesce_ms = self._config.effective_coalesce_ms
        if self._last_submit_time is not None:
            since_last_submit_ms = (now - self._last_submit_time) * 1000
            coalesce_remaining_ms = max(0, coalesce_ms - since_last_submit_ms)
        else:
            coalesce_remaining_ms = coalesce_ms

        effective_ms = min(batch_remaining_ms, coalesce_remaining_ms)
        return effective_ms / 1000  # Convert to seconds

    def _extract_batch(self) -> FormattedBatch[I, T]:
        """Extract an optimal sub-batch from pending requests.

        Algorithm:
        1. Sort pending requests by cost (ascending)
        2. Take requests until max_batch_cost would be exceeded
        3. Keep remaining requests for next get_batch() call

        This minimizes padding waste by grouping similar-cost sequences.
        """
        # Track why batch was triggered (must check BEFORE extracting items)
        was_full = self._batch_is_full()
        was_timeout = self._batch_timeout_expired()
        was_coalesced = self._coalesce_expired()
        pending_before = len(self._pending)
        cost_before = self._total_cost

        # Sort pending by cost (ascending) for optimal batching
        self._pending.sort(key=lambda r: r.item.cost)

        # Greedily take items until we exceed max_batch_cost
        batch_items: list[I] = []
        batch_metadata: list[T] = []
        batch_cost = 0
        take_count = 0
        hit_cost_limit = False
        hit_request_limit = False

        for request in self._pending:
            # Would this item push us over the cost limit?
            # Always take at least one item (handles single large requests)
            if batch_cost + request.item.cost > self._config.max_batch_cost:
                if take_count > 0:
                    hit_cost_limit = True
                    break  # Stop here, we have a batch
                # Otherwise, take this single item even if it exceeds the limit

            batch_items.append(request.item)
            batch_metadata.append(request.metadata)
            batch_cost += request.item.cost
            take_count += 1

            # Also respect max_batch_requests
            if take_count >= self._config.max_batch_requests:
                hit_request_limit = True
                break

        # Remove taken items from pending
        self._pending = self._pending[take_count:]
        self._total_cost -= batch_cost

        # Log batch formation details (was_coalesced evaluated before extraction above)
        trigger = (
            "timeout" if was_timeout else ("full" if was_full else ("coalesced" if was_coalesced else "immediate"))
        )
        limit_hit = "cost_limit" if hit_cost_limit else ("request_limit" if hit_request_limit else "none")
        logger.debug(
            "Batch formed: trigger=%s, limit_hit=%s, items=%d/%d taken, cost=%d/%d, remaining=%d items",
            trigger,
            limit_hit,
            take_count,
            pending_before,
            batch_cost,
            cost_before,
            len(self._pending),
        )

        # Reset timers if we've emptied the queue
        if not self._pending:
            self._first_request_time = None
            self._last_submit_time = None

        return FormattedBatch(
            items=batch_items,
            metadata=batch_metadata,
            total_cost=batch_cost,
        )

    def try_get_batch(self) -> FormattedBatch[I, T] | None:
        """Try to get a batch without waiting.

        Returns:
            FormattedBatch if batch is ready (sorted by cost), None otherwise.
        """
        if self._should_yield_batch():
            return self._extract_batch()
        return None

    async def try_drain(self) -> FormattedBatch[I, T] | None:
        """Drain accumulated items immediately, bypassing batch timeout.

        Used by the process loop after inference completes to immediately
        grab items that accumulated while the GPU was busy. This is the
        "continuous batching" optimization — items that waited during
        inference don't need to wait an additional batch formation window.

        Returns:
            FormattedBatch if items were pending, None otherwise.
        """
        async with self._lock:
            if not self._pending:
                return None
            return self._extract_batch()


def collate_batch(
    payloads: list[TextPayload],
    pad_token_id: int = 0,
) -> dict[str, Any]:
    """Collate text payloads into padded tensors.

    Args:
        payloads: List of TextPayload instances (should be sorted by length).
        pad_token_id: Token ID to use for padding.

    Returns:
        Dictionary with 'input_ids' and 'attention_mask' as lists of lists
        (ready to be converted to tensors).
    """
    if not payloads:
        return {"input_ids": [], "attention_mask": []}

    # Find max length in batch
    max_length = max(p.token_count for p in payloads)

    input_ids_batch = []
    attention_mask_batch = []

    for payload in payloads:
        # Pad to max length
        padding_length = max_length - payload.token_count
        padded_ids = payload.input_ids + [pad_token_id] * padding_length
        padded_mask = payload.attention_mask + [0] * padding_length

        input_ids_batch.append(padded_ids)
        attention_mask_batch.append(padded_mask)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
    }
