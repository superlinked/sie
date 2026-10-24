"""Opt-in dispatch-turn progress without changing the default batcher.

The reservation policy follows PR798. It can change native batch composition
and therefore model outputs; it does not provide batch-shape invariance.
"""

import logging

from sie_server.core.batcher import BatchConfig, BatchFormer, FormattedBatch, HasCost, PendingRequest

logger = logging.getLogger(__name__)


class ProgressBatchFormer[I: HasCost, T](BatchFormer[I, T]):
    """Reserve a bypassed oldest item for the next batchable turn.

    The insertion-ordered identity map contains only pending requests. Neither
    this map nor the reservation exists on an ordinary ``BatchFormer``.
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        super().__init__(config)
        self._arrivals: dict[int, PendingRequest[I, T]] = {}
        self._reserved_request: PendingRequest[I, T] | None = None

    def _append_item(self, item: I, metadata: T) -> None:
        super()._append_item(item, metadata)
        request = self._pending[-1]
        self._arrivals[id(request)] = request

    def _extract_batch(
        self,
        max_items: int | None = None,
        coalesce_cap_ms: float | None = None,
    ) -> FormattedBatch[I, T]:
        """Pack greedily, alternating a bypassed oldest item's suffix when needed.

        ``max_items`` caps the number selected from the current pending set;
        it does not freeze arrival identities. Flagged ``runs_alone`` turns
        and zero-item drains do not consume a batchable reservation.
        """
        was_full = self._batch_is_full()
        was_timeout = self._batch_timeout_expired()
        was_coalesced = self._coalesce_expired(coalesce_cap_ms)
        pending_before = len(self._pending)
        cost_before = self._total_cost

        if any(request.runs_alone for request in self._pending):
            self._order_runs_alone_lane()
        else:
            self._pending.sort(key=lambda request: request.item.cost)

        reserved_turn = self._reserved_request is not None and bool(self._pending) and not self._pending[0].runs_alone
        reserved_limit = len(self._pending)
        if reserved_turn:
            reserved_index = next(
                index for index, request in enumerate(self._pending) if request is self._reserved_request
            )
            batchable_end = next(
                (index for index, request in enumerate(self._pending) if request.runs_alone), len(self._pending)
            )
            self._pending[:batchable_end] = self._pending[reserved_index:batchable_end] + self._pending[:reserved_index]
            reserved_limit = batchable_end - reserved_index

        batch_items: list[I] = []
        batch_metadata: list[T] = []
        batch_cost = 0
        take_count = 0
        hit_cost_limit = False
        hit_request_limit = False

        for request in self._pending:
            if take_count >= reserved_limit:
                break
            if max_items is not None and take_count >= max_items:
                break
            if request.runs_alone and take_count > 0:
                break
            if batch_cost + request.item.cost > self._config.max_batch_cost and take_count > 0:
                hit_cost_limit = True
                break

            batch_items.append(request.item)
            batch_metadata.append(request.metadata)
            batch_cost += request.item.cost
            take_count += 1
            if request.runs_alone:
                break
            if take_count >= self._config.max_batch_requests:
                hit_request_limit = True
                break

        lane = "alone" if take_count > 0 and self._pending[0].runs_alone else "batched"
        if take_count > 0:
            self._last_batch_alone = lane == "alone"
            if lane == "batched":
                if reserved_turn:
                    self._reserved_request = None
                elif take_count < len(self._pending):
                    oldest = next(request for request in self._arrivals.values() if not request.runs_alone)
                    if all(request is not oldest for request in self._pending[:take_count]):
                        self._reserved_request = oldest

        for request in self._pending[:take_count]:
            del self._arrivals[id(request)]
        self._pending = self._pending[take_count:]
        self._total_cost -= batch_cost

        trigger = (
            "timeout" if was_timeout else ("full" if was_full else ("coalesced" if was_coalesced else "immediate"))
        )
        limit_hit = "cost_limit" if hit_cost_limit else ("request_limit" if hit_request_limit else "none")
        logger.debug(
            "Batch formed: trigger=%s, lane=%s, limit_hit=%s, items=%d/%d taken, cost=%d/%d, remaining=%d items",
            trigger,
            lane,
            limit_hit,
            take_count,
            pending_before,
            batch_cost,
            cost_before,
            len(self._pending),
        )
        if not self._pending:
            self._first_request_time = None
            self._last_submit_time = None
            self._reserved_request = None
        return FormattedBatch(items=batch_items, metadata=batch_metadata, total_cost=batch_cost)

    def drain_pending(self) -> list[T]:
        metadata = super().drain_pending()
        self._arrivals.clear()
        self._reserved_request = None
        return metadata
