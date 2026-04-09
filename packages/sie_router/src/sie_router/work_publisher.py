"""Publishes work items to NATS JetStream and collects results.

This module replaces ``_forward_request()`` in ``proxy.py`` for cluster mode
when ``SIE_CLUSTER_ROUTING=queue``. It decomposes client requests into
individual work items, publishes them to JetStream subjects, and waits for
results from workers.

Result payloads are treated as opaque msgpack blobs — the router does NOT
deserialize numpy arrays. This keeps ``numpy`` out of the router's dependency
graph.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import msgpack
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from sie_sdk.queue_types import (
    INLINE_THRESHOLD_BYTES,
    WorkItem,
    WorkResult,
    work_subject,
)

from sie_router.jetstream_manager import JetStreamManager
from sie_router.payload_store import PayloadStore

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram

    QUEUE_PUBLISH_SECONDS = Histogram(
        "sie_router_queue_publish_seconds",
        "Time to publish all work items for a request (seconds)",
        ["operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    QUEUE_ITEMS_PUBLISHED = Histogram(
        "sie_router_queue_items_published",
        "Number of work items published per request",
        ["operation"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    QUEUE_RESULT_WAIT_SECONDS = Histogram(
        "sie_router_queue_result_wait_seconds",
        "Time waiting for all results from workers (seconds)",
        ["operation"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    )
    QUEUE_PAYLOAD_OFFLOADS = Counter(
        "sie_router_queue_payload_offloads_total",
        "Number of payloads offloaded to payload store",
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


class NoConsumersError(RuntimeError):
    """Raised when no workers are subscribed to a model's stream."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"No consumers for model {model_id} — no workers subscribed")


# Default timeout for waiting for all results
_DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("SIE_ROUTER_REQUEST_TIMEOUT", "30"))

# Backpressure: reject if stream has more than this many pending messages
_MAX_STREAM_PENDING = int(os.environ.get("SIE_ROUTER_MAX_STREAM_PENDING", "50000"))

# Cache TTL for backpressure check (seconds).  Avoids a full NATS
# ``stream_info`` RPC on every request.  100ms is short enough to detect
# consumer loss or backpressure within a few requests.
_BP_CACHE_TTL_S = 1.0


@dataclass
class PendingRequest:
    """Tracks the state of a pending client request."""

    request_id: str
    total_items: int
    results: dict[int, WorkResult]  # item_index → full WorkResult (preserves timing metadata)
    errors: dict[int, tuple[str, str]]  # item_index → (error, error_code)
    operation: str = ""
    completed: asyncio.Event = field(default_factory=asyncio.Event)
    start_time: float = field(default_factory=time.monotonic)


class WorkPublisher:
    """Publishes work items to NATS JetStream and collects results.

    Thread-safe for concurrent use by multiple proxy request handlers.
    """

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        jsm: JetStreamManager,
        router_id: str,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._nc = nc
        self._js = js
        self._jsm = jsm
        self._router_id = router_id
        self._payload_store = payload_store
        self._pending: dict[str, PendingRequest] = {}
        # itertools.count is thread-safe in CPython (GIL protects __next__).
        # For pure async use this is always safe; the GIL note is for mixed
        # threaded access from sync middleware.
        self._counter = itertools.count(1)
        self._inbox_sub: object | None = None
        self._running = False
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Backpressure cache: {pool_or_model: (timestamp, consumer_count, pending)}
        self._bp_cache: dict[str, tuple[float, int, int]] = {}

    async def start(self) -> None:
        """Start the result listener.

        Subscribes to the router's inbox subject for receiving work results.
        Must be called before ``submit_encode`` / ``submit_score`` / ``submit_extract``.
        """
        await self._subscribe_inbox()
        self._running = True
        logger.info("WorkPublisher started (inbox=_INBOX.%s.>)", self._router_id)

    async def _subscribe_inbox(self) -> None:
        """Create (or re-create) the inbox subscription.

        The new subscription is created BEFORE the old one is removed so
        there is no window where incoming results could be dropped.
        """
        old_sub = self._inbox_sub
        inbox_subject = f"_INBOX.{self._router_id}.>"
        self._inbox_sub = await self._nc.subscribe(inbox_subject, cb=self._on_result)
        if old_sub is not None:
            try:
                await old_sub.unsubscribe()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe old inbox during re-subscribe")

    async def handle_reconnect(self) -> None:
        """Re-create inbox subscription and clear JSM caches after NATS reconnect.

        After a NATS server restart the subscription and cached stream/consumer
        metadata may be stale. This method is called from the reconnect handler
        registered in ``app_factory._work_publisher``.
        """
        self._jsm.clear_caches()
        self._bp_cache.clear()
        try:
            await self._subscribe_inbox()
            logger.info("WorkPublisher reconnected (inbox re-subscribed, JSM caches cleared)")
        except Exception:
            logger.exception("WorkPublisher failed to re-subscribe inbox on reconnect")

    async def stop(self) -> None:
        """Stop the result listener and cancel pending requests."""
        self._running = False
        if self._inbox_sub:
            await self._inbox_sub.unsubscribe()  # type: ignore[union-attr]
            self._inbox_sub = None

        # Cancel all pending requests
        for pending in self._pending.values():
            pending.completed.set()
        self._pending.clear()
        logger.info("WorkPublisher stopped")

    def _next_request_id(self) -> str:
        return f"{self._router_id}-{next(self._counter)}"

    async def submit_encode(
        self,
        model_id: str,
        profile_id: str,
        pool_name: str,
        machine_profile: str,
        items: list[dict[str, Any]],
        output_types: list[str] | None = None,
        instruction: str | None = None,
        is_query: bool = False,
        options: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,  # noqa: ASYNC109
        bundle_config_hash: str = "",
    ) -> list[WorkResult]:
        """Publish encode work items and wait for all results.

        Decomposes the request into one work item per item.
        """
        total_items = len(items)

        async def _publish(request_id: str, reply_subject: str, subject: str) -> None:
            published = 0
            publish_start = time.monotonic()
            publish_futures: list[asyncio.Future] = []
            try:
                # Build and serialize all work items in a single thread call
                # (1 thread hop for N items instead of N thread hops).
                built = await self._build_work_items_batch(
                    request_id=request_id,
                    total_items=total_items,
                    operation="encode",
                    model_id=model_id,
                    profile_id=profile_id,
                    pool_name=pool_name,
                    machine_profile=machine_profile,
                    reply_subject=reply_subject,
                    items=items,
                    output_types=output_types,
                    instruction=instruction,
                    is_query=is_query,
                    options=options,
                    bundle_config_hash=bundle_config_hash,
                )
                for _wi, payload in built:
                    fut = await self._js.publish_async(subject, payload)
                    publish_futures.append(fut)
                    published += 1

                await self._js.publish_async_completed()

                for fut in publish_futures:
                    exc = fut.exception() if fut.done() else None
                    if exc is not None:
                        raise exc
            except Exception:
                logger.error(
                    "Partial publish failure for request %s: %d/%d items published "
                    "(orphaned items will expire via stream max_age=60s)",
                    request_id,
                    published,
                    total_items,
                )
                raise
            finally:
                if _HAS_METRICS:
                    QUEUE_PUBLISH_SECONDS.labels(operation="encode").observe(time.monotonic() - publish_start)
                    QUEUE_ITEMS_PUBLISHED.labels(operation="encode").observe(published)

        return await self._submit_items(
            model_id=model_id,
            pool_name=pool_name,
            operation="encode",
            total_items=total_items,
            timeout=timeout,
            publish_fn=_publish,
        )

    async def submit_score(
        self,
        model_id: str,
        profile_id: str,
        pool_name: str,
        machine_profile: str,
        query: dict[str, Any],
        items: list[dict[str, Any]],
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,  # noqa: ASYNC109
        bundle_config_hash: str = "",
    ) -> list[WorkResult]:
        """Publish a score work item and wait for result.

        Score operations are NOT decomposed — query + all items go to one worker
        because cross-attention requires co-location.
        """

        async def _publish(request_id: str, reply_subject: str, subject: str) -> None:
            work_item: WorkItem = {
                "work_item_id": f"{request_id}.0",
                "request_id": request_id,
                "item_index": 0,
                "total_items": 1,
                "operation": "score",
                "model_id": model_id,
                "profile_id": profile_id,
                "pool_name": pool_name,
                "machine_profile": machine_profile,
                "query_item": query,
                "score_items": items,
                "instruction": instruction,
                "options": options,
                "bundle_config_hash": bundle_config_hash,
                "router_id": self._router_id,
                "reply_subject": reply_subject,
                "timestamp": time.time(),
            }

            # Consolidate all serialization into a single to_thread call
            # (avoids up to 3 thread hops for score requests).
            def _serialize_score(
                wi: WorkItem,
                threshold: int,
                q: dict[str, Any],
                it: list[dict[str, Any]],
                req_id: str,
            ) -> tuple[bytes, str | None, bytes | None]:
                packb = msgpack.packb
                payload = packb(wi, use_bin_type=True)
                if len(payload) > threshold:
                    score_bytes = packb({"query": q, "items": it}, use_bin_type=True)
                    wi["query_item"] = None
                    wi["score_items"] = None
                    wi["query_payload_ref"] = f"payloads/{req_id}/score"
                    payload = packb(wi, use_bin_type=True)
                    return payload, wi["query_payload_ref"], score_bytes
                return payload, None, None

            payload, ref_key, score_bytes = await asyncio.to_thread(
                _serialize_score,
                work_item,
                INLINE_THRESHOLD_BYTES,
                query,
                items,
                request_id,
            )
            if ref_key and self._payload_store and score_bytes:
                await self._payload_store.put(ref_key, score_bytes)
                if _HAS_METRICS:
                    QUEUE_PAYLOAD_OFFLOADS.inc()

            publish_start = time.monotonic()
            try:
                await self._js.publish(subject, payload)
            except Exception:
                logger.error(
                    "Publish failure for score request %s (orphaned item will expire via stream max_age=60s)",
                    request_id,
                )
                raise
            finally:
                if _HAS_METRICS:
                    QUEUE_PUBLISH_SECONDS.labels(operation="score").observe(time.monotonic() - publish_start)
                    QUEUE_ITEMS_PUBLISHED.labels(operation="score").observe(1)

        return await self._submit_items(
            model_id=model_id,
            pool_name=pool_name,
            operation="score",
            total_items=1,
            timeout=timeout,
            publish_fn=_publish,
        )

    async def submit_extract(
        self,
        model_id: str,
        profile_id: str,
        pool_name: str,
        machine_profile: str,
        items: list[dict[str, Any]],
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_REQUEST_TIMEOUT,  # noqa: ASYNC109
        bundle_config_hash: str = "",
    ) -> list[WorkResult]:
        """Publish extract work items and wait for all results.

        Decomposes the request into one work item per item (same as encode).
        """
        total_items = len(items)

        async def _publish(request_id: str, reply_subject: str, subject: str) -> None:
            published = 0
            publish_start = time.monotonic()
            publish_futures: list[asyncio.Future] = []
            try:
                # Build and serialize all work items in a single thread call
                built = await self._build_work_items_batch(
                    request_id=request_id,
                    total_items=total_items,
                    operation="extract",
                    model_id=model_id,
                    profile_id=profile_id,
                    pool_name=pool_name,
                    machine_profile=machine_profile,
                    reply_subject=reply_subject,
                    items=items,
                    labels=labels,
                    output_schema=output_schema,
                    instruction=instruction,
                    options=options,
                    bundle_config_hash=bundle_config_hash,
                )
                for _wi, payload in built:
                    fut = await self._js.publish_async(subject, payload)
                    publish_futures.append(fut)
                    published += 1

                await self._js.publish_async_completed()

                for fut in publish_futures:
                    exc = fut.exception() if fut.done() else None
                    if exc is not None:
                        raise exc
            except Exception:
                logger.error(
                    "Partial publish failure for request %s: %d/%d items published "
                    "(orphaned items will expire via stream max_age=60s)",
                    request_id,
                    published,
                    total_items,
                )
                raise
            finally:
                if _HAS_METRICS:
                    QUEUE_PUBLISH_SECONDS.labels(operation="extract").observe(time.monotonic() - publish_start)
                    QUEUE_ITEMS_PUBLISHED.labels(operation="extract").observe(published)

        return await self._submit_items(
            model_id=model_id,
            pool_name=pool_name,
            operation="extract",
            total_items=total_items,
            timeout=timeout,
            publish_fn=_publish,
        )

    # -- Internal helpers ----------------------------------------------------

    async def _submit_items(
        self,
        *,
        model_id: str,
        pool_name: str,
        operation: str,
        total_items: int,
        timeout: float,  # noqa: ASYNC109
        publish_fn: Callable[..., Awaitable[None]],
    ) -> list[WorkResult]:
        """Common scaffolding for submit_encode/score/extract.

        Handles stream creation, backpressure check, pending request
        registration, result waiting, and payload cleanup. The caller
        provides ``publish_fn(request_id, reply_subject, subject)`` to
        handle operation-specific work item construction and publishing.
        """
        request_id = self._next_request_id()
        reply_subject = f"_INBOX.{self._router_id}.{request_id}"

        await self._jsm.ensure_stream(model_id, pool_name=pool_name)
        await self._check_backpressure(model_id, pool_name=pool_name)

        pending = PendingRequest(
            request_id=request_id,
            total_items=total_items,
            results={},
            errors={},
            operation=operation,
        )
        self._pending[request_id] = pending

        try:
            subject = work_subject(model_id, pool_name)
            await publish_fn(request_id, reply_subject, subject)
            return await self._wait_for_results(pending, timeout)
        finally:
            self._pending.pop(request_id, None)
            if self._payload_store:
                task = asyncio.create_task(self._payload_store.delete_prefix(f"payloads/{request_id}/"))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    async def _build_work_items_batch(
        self,
        *,
        request_id: str,
        total_items: int,
        operation: str,
        model_id: str,
        profile_id: str,
        pool_name: str,
        machine_profile: str,
        reply_subject: str,
        items: list[dict[str, Any]],
        output_types: list[str] | None = None,
        instruction: str | None = None,
        is_query: bool = False,
        options: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        bundle_config_hash: str = "",
    ) -> list[tuple[WorkItem, bytes]]:
        """Build and serialize multiple WorkItems in a single thread call.

        Constructs all work item dicts on the event loop (cheap dict
        creation), then serializes them all in one ``to_thread`` call to
        avoid N thread roundtrips for N items.  Items that exceed the
        inline threshold are offloaded to the payload store.

        Returns:
            List of (work_item_dict, serialized_bytes) tuples.
        """
        now = time.time()
        work_items: list[WorkItem] = []
        for i, item in enumerate(items):
            wi: WorkItem = {
                "work_item_id": f"{request_id}.{i}",
                "request_id": request_id,
                "item_index": i,
                "total_items": total_items,
                "operation": operation,
                "model_id": model_id,
                "profile_id": profile_id,
                "pool_name": pool_name,
                "machine_profile": machine_profile,
                "item": item,
                "output_types": output_types,
                "instruction": instruction,
                "is_query": is_query,
                "options": options,
                "labels": labels,
                "output_schema": output_schema,
                "bundle_config_hash": bundle_config_hash,
                "router_id": self._router_id,
                "reply_subject": reply_subject,
                "timestamp": now,
            }
            work_items.append(wi)

        # For small batches (1-2 items), serialize inline on the event loop.
        # msgpack.packb on a single work item takes ~5-10μs — well below the
        # ~50μs overhead of asyncio.to_thread.  Only offload for larger batches
        # where the cumulative serialization cost justifies the thread hop.
        _INLINE_SERIALIZE_THRESHOLD = 2
        if len(work_items) <= _INLINE_SERIALIZE_THRESHOLD:
            serialized_list = [msgpack.packb(wi, use_bin_type=True) for wi in work_items]
        else:

            def _serialize_batch(wis: list[WorkItem]) -> list[bytes]:
                return [msgpack.packb(wi, use_bin_type=True) for wi in wis]

            serialized_list = await asyncio.to_thread(_serialize_batch, work_items)

        # Check for payload offloading (rare — only for very large items)
        results: list[tuple[WorkItem, bytes]] = []
        for i, (wi, serialized) in enumerate(zip(work_items, serialized_list)):
            if len(serialized) > INLINE_THRESHOLD_BYTES and self._payload_store and wi.get("item"):
                item_bytes = await asyncio.to_thread(msgpack.packb, wi["item"], use_bin_type=True)
                ref_key = f"payloads/{request_id}/{i}"
                await self._payload_store.put(ref_key, item_bytes)
                wi["item"] = None
                wi["payload_ref"] = ref_key
                serialized = await asyncio.to_thread(msgpack.packb, wi, use_bin_type=True)
                if _HAS_METRICS:
                    QUEUE_PAYLOAD_OFFLOADS.inc()
            results.append((wi, serialized))

        return results

    async def _build_work_item(
        self,
        *,
        request_id: str,
        item_index: int,
        total_items: int,
        operation: str,
        model_id: str,
        profile_id: str,
        pool_name: str,
        machine_profile: str,
        reply_subject: str,
        item: dict[str, Any] | None = None,
        output_types: list[str] | None = None,
        instruction: str | None = None,
        is_query: bool = False,
        options: dict[str, Any] | None = None,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        bundle_config_hash: str = "",
    ) -> tuple[WorkItem, bytes]:
        """Build a WorkItem and serialize it, offloading payload if needed.

        Returns:
            Tuple of (work_item_dict, serialized_bytes).
        """
        work_item_id = f"{request_id}.{item_index}"

        wi: WorkItem = {
            "work_item_id": work_item_id,
            "request_id": request_id,
            "item_index": item_index,
            "total_items": total_items,
            "operation": operation,
            "model_id": model_id,
            "profile_id": profile_id,
            "pool_name": pool_name,
            "machine_profile": machine_profile,
            "item": item,
            "output_types": output_types,
            "instruction": instruction,
            "is_query": is_query,
            "options": options,
            "labels": labels,
            "output_schema": output_schema,
            "bundle_config_hash": bundle_config_hash,
            "router_id": self._router_id,
            "reply_subject": reply_subject,
            "timestamp": time.time(),
        }

        # Check if payload needs offloading — offload msgpack serialization to
        # avoid blocking the event loop (called per-item in batch publish loops).
        _packb = lambda o: msgpack.packb(o, use_bin_type=True)  # noqa: E731
        serialized = await asyncio.to_thread(_packb, wi)
        if len(serialized) > INLINE_THRESHOLD_BYTES and self._payload_store and item:
            # Offload item payload to store
            item_bytes = await asyncio.to_thread(_packb, item)
            ref_key = f"payloads/{request_id}/{item_index}"
            await self._payload_store.put(ref_key, item_bytes)
            wi["item"] = None
            wi["payload_ref"] = ref_key
            serialized = await asyncio.to_thread(_packb, wi)
            if _HAS_METRICS:
                QUEUE_PAYLOAD_OFFLOADS.inc()

        return wi, serialized

    async def _on_result(self, msg: Any) -> None:
        """Handle incoming work result from NATS subscription.

        Result messages contain a small envelope (~200-500 bytes of
        strings/ints) plus an opaque ``result_msgpack`` blob that stays
        as ``bytes``.  Deserialization takes <5μs — well below the ~50μs
        overhead of ``asyncio.to_thread`` — so we unpack directly on the
        event loop.  The ``_extract_request_id_fast`` early-exit skips
        even that cost for stale/duplicate messages.
        """
        data = msg.data

        # Fast path: look up request_id from raw bytes before full
        # deserialization.  If the request is unknown (already completed
        # or a duplicate), skip the unpackb entirely.
        _req_id = self._extract_request_id_fast(data)
        if _req_id is not None and _req_id not in self._pending:
            return

        try:
            result: WorkResult = msgpack.unpackb(data, raw=False)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to deserialize work result", exc_info=True)
            return

        request_id = result.get("request_id", "")
        pending = self._pending.get(request_id)
        if not pending:
            # Result for unknown or already-completed request (duplicate or late)
            return

        item_index = result.get("item_index", 0)

        # Deduplicate: first result wins
        if item_index in pending.results or item_index in pending.errors:
            return

        if result.get("success", False):
            pending.results[item_index] = result
        else:
            pending.errors[item_index] = (
                result.get("error", "unknown error"),
                result.get("error_code", "internal_error"),
            )

        # Check if all items are done
        total_received = len(pending.results) + len(pending.errors)
        if total_received >= pending.total_items:
            pending.completed.set()

    @staticmethod
    def _extract_request_id_fast(data: bytes) -> str | None:
        r"""Try to extract request_id from raw msgpack bytes without full deserialization.

        Returns the request_id string if found, or None if the fast path
        cannot determine it (falls through to full deserialization).
        The ``request_id`` key is always encoded as the msgpack fixstr
        ``\xaa request_id`` (10-char fixstr).  We search for this marker
        and read the subsequent string value.
        """
        marker = b"\xaarequest_id"
        pos = data.find(marker)
        if pos == -1:
            return None
        # The value starts right after the key
        val_start = pos + len(marker)
        if val_start >= len(data):
            return None
        # Read the msgpack string value
        first_byte = data[val_start]
        if 0xA0 <= first_byte <= 0xBF:
            # fixstr: length in lower 5 bits
            str_len = first_byte & 0x1F
            str_start = val_start + 1
        elif first_byte == 0xD9:
            # str 8: next byte is length
            if val_start + 1 >= len(data):
                return None
            str_len = data[val_start + 1]
            str_start = val_start + 2
        elif first_byte == 0xDA:
            # str 16: next 2 bytes are length (big-endian)
            if val_start + 2 >= len(data):
                return None
            str_len = (data[val_start + 1] << 8) | data[val_start + 2]
            str_start = val_start + 3
        else:
            return None
        if str_start + str_len > len(data):
            return None
        try:
            return data[str_start : str_start + str_len].decode("utf-8")
        except UnicodeDecodeError:
            return None

    async def _wait_for_results(
        self,
        pending: PendingRequest,
        timeout: float,  # noqa: ASYNC109
    ) -> list[WorkResult]:
        """Wait for all results to arrive, then return ordered results."""
        wait_start = time.monotonic()
        try:
            await asyncio.wait_for(pending.completed.wait(), timeout=timeout)
        except TimeoutError:
            received = len(pending.results) + len(pending.errors)
            logger.warning(
                "Request %s timed out: %d/%d results received",
                pending.request_id,
                received,
                pending.total_items,
            )
            raise
        finally:
            if _HAS_METRICS and pending.operation:
                QUEUE_RESULT_WAIT_SECONDS.labels(operation=pending.operation).observe(time.monotonic() - wait_start)

        # Build ordered result list — successful results are stored as full
        # WorkResult dicts so timing metadata (inference_ms, queue_ms, etc.)
        # flows through to the caller.
        results: list[WorkResult] = []
        for i in range(pending.total_items):
            if i in pending.results:
                results.append(pending.results[i])
            elif i in pending.errors:
                error_msg, error_code = pending.errors[i]
                results.append(
                    WorkResult(
                        work_item_id=f"{pending.request_id}.{i}",
                        request_id=pending.request_id,
                        item_index=i,
                        success=False,
                        error=error_msg,
                        error_code=error_code,
                    )
                )
            else:
                # Should not happen if completed event is set correctly
                results.append(
                    WorkResult(
                        work_item_id=f"{pending.request_id}.{i}",
                        request_id=pending.request_id,
                        item_index=i,
                        success=False,
                        error="Result missing",
                        error_code="result_missing",
                    )
                )

        return results

    async def _check_backpressure(self, model_id: str, pool_name: str = "") -> None:
        """Check stream health and reject if no consumers or over backpressure limit.

        Results are cached for ``_BP_CACHE_TTL_S`` to avoid a full NATS
        ``stream_info`` RPC on every request (~3ms savings at c=1).
        The cache is keyed by pool_name (one pool stream per pool).
        """
        now = time.monotonic()
        cache_key = pool_name or model_id
        cached = self._bp_cache.get(cache_key)
        if cached is not None:
            ts, consumer_count, pending = cached
            if (now - ts) < _BP_CACHE_TTL_S:
                if consumer_count == 0:
                    raise NoConsumersError(model_id)
                if pending > _MAX_STREAM_PENDING:
                    raise RuntimeError(
                        f"Stream backpressure: {pending} pending for {model_id} (limit: {_MAX_STREAM_PENDING})"
                    )
                return

        consumer_count, pending = await self._jsm.get_stream_health(model_id, pool_name=pool_name)
        self._bp_cache[cache_key] = (now, consumer_count, pending)

        if consumer_count == 0:
            raise NoConsumersError(model_id)

        if pending > _MAX_STREAM_PENDING:
            raise RuntimeError(
                f"Stream backpressure: {pending} pending messages for model {model_id} (limit: {_MAX_STREAM_PENDING})"
            )
