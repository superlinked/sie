from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from sie_router.jetstream_manager import JetStreamManager
from sie_router.payload_store import LocalPayloadStore, PayloadStore
from sie_router.work_publisher import WorkPublisher
from sie_sdk.queue_types import (
    WorkItem,
    WorkResult,
    work_subject,
)

# ---------------------------------------------------------------------------
# Simulation infrastructure
# ---------------------------------------------------------------------------


class SimMsg:
    """Simulated NATS JetStream message."""

    def __init__(self, data: bytes, subject: str, bus: QueueBus) -> None:
        self.data = data
        self.subject = subject
        self._bus = bus
        self._acked = False
        self._naked = False

    async def ack(self) -> None:
        self._acked = True

    async def nak(self, delay: float | None = None) -> None:
        self._naked = True
        if delay and delay > 0:
            # Schedule redelivery after a short delay (10ms in tests)
            loop = asyncio.get_running_loop()
            loop.call_later(
                0.01,
                lambda: asyncio.ensure_future(self._bus._redeliver(self)),
            )


class QueueBus:
    """In-memory NATS JetStream simulation.

    Provides work queue semantics: messages published to a subject are
    delivered to exactly one consumer pulling from that subject. ACK removes
    the message. NAK redelivers after delay. Inbox subjects use plain
    pub/sub (all subscribers get the message).
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[SimMsg]] = {}
        self._inbox_callbacks: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._published_count: dict[str, int] = {}

    async def publish_work(self, subject: str, data: bytes) -> None:
        """Publish a work item to a stream subject."""
        async with self._lock:
            if subject not in self._queues:
                self._queues[subject] = asyncio.Queue()
            msg = SimMsg(data, subject, self)
            await self._queues[subject].put(msg)
            self._published_count[subject] = self._published_count.get(subject, 0) + 1

    async def pull(self, subject: str, batch: int, timeout: float = 0.1) -> list[SimMsg]:  # noqa: ASYNC109
        """Pull up to *batch* messages from a subject."""
        if subject not in self._queues:
            return []
        q = self._queues[subject]
        msgs: list[SimMsg] = []
        try:
            for _ in range(batch):
                msg = q.get_nowait()
                msgs.append(msg)
        except asyncio.QueueEmpty:
            if not msgs:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=timeout)
                    msgs.append(msg)
                except TimeoutError:
                    pass
        return msgs

    async def publish_inbox(self, subject: str, data: bytes) -> None:
        """Publish to inbox (plain pub/sub — matches wildcard subscribers)."""
        for pattern, cb in list(self._inbox_callbacks.items()):
            if self._matches(pattern, subject):
                msg = MagicMock()
                msg.data = data
                msg.subject = subject
                await cb(msg)

    def subscribe_inbox(self, pattern: str, cb: Any) -> None:
        """Subscribe to inbox with wildcard pattern."""
        self._inbox_callbacks[pattern] = cb

    async def _redeliver(self, msg: SimMsg) -> None:
        """Redeliver a NAKed message."""
        async with self._lock:
            if msg.subject not in self._queues:
                self._queues[msg.subject] = asyncio.Queue()
            new_msg = SimMsg(msg.data, msg.subject, self)
            await self._queues[msg.subject].put(new_msg)

    @staticmethod
    def _matches(pattern: str, subject: str) -> bool:
        """Simple wildcard matching for NATS subjects."""
        if pattern.endswith(".>"):
            prefix = pattern[:-2]
            return subject.startswith(prefix + ".") or subject == prefix
        return pattern == subject

    def get_published_count(self, subject: str) -> int:
        return self._published_count.get(subject, 0)


class SimulatedWorker:
    """Simulates a worker that pulls from QueueBus, processes items, publishes results."""

    def __init__(
        self,
        bus: QueueBus,
        worker_id: str,
        model_ids: list[str],
        pool_name: str = "_default",
        batch_budget: int = 64,
        process_time: float = 0.001,
        loaded_models: set[str] | None = None,
    ) -> None:
        self.bus = bus
        self.worker_id = worker_id
        self.model_ids = model_ids
        self.pool_name = pool_name
        self.batch_budget = batch_budget
        self.process_time = process_time
        self.loaded_models = loaded_models if loaded_models is not None else set(model_ids)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self.processed_items: list[WorkItem] = []
        self.naked_items: list[WorkItem] = []

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while self._running:
            pulled_any = False
            for model_id in self.model_ids:
                subject = work_subject(model_id, self.pool_name)
                msgs = await self.bus.pull(subject, self.batch_budget, timeout=0.01)
                if not msgs:
                    continue
                pulled_any = True
                for msg in msgs:
                    wi: WorkItem = msgpack.unpackb(msg.data, raw=False)

                    # Simulate unloaded model behaviour
                    if model_id not in self.loaded_models:
                        self.naked_items.append(wi)
                        await msg.nak(delay=0.01)
                        continue

                    self.processed_items.append(wi)

                    # Simulate processing time
                    await asyncio.sleep(self.process_time)

                    # Build and publish result
                    result: WorkResult = {
                        "work_item_id": wi.get("work_item_id", ""),
                        "request_id": wi.get("request_id", ""),
                        "item_index": wi.get("item_index", 0),
                        "success": True,
                        "result_msgpack": msgpack.packb({"dense": [0.1, 0.2, 0.3]}, use_bin_type=True),
                        "queue_ms": 1.0,
                        "processing_ms": self.process_time * 1000,
                        "worker_id": self.worker_id,
                    }
                    reply = wi.get("reply_subject", "")
                    if reply:
                        await self.bus.publish_inbox(
                            reply,
                            msgpack.packb(result, use_bin_type=True),
                        )
                    await msg.ack()

            if not pulled_any:
                await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Helper: create a WorkPublisher wired to a QueueBus
# ---------------------------------------------------------------------------


def _make_bus_publisher(
    bus: QueueBus,
    router_id: str = "router-1",
    payload_store: PayloadStore | None = None,
) -> WorkPublisher:
    """Create a WorkPublisher wired to a QueueBus instead of real NATS."""
    nc = AsyncMock()
    js = AsyncMock()
    jsm = AsyncMock(spec=JetStreamManager)

    jsm.ensure_stream = AsyncMock()
    jsm.get_stream_health = AsyncMock(return_value=(2, 0))

    # Wire publish_async to bus
    async def _fake_publish_async(subject: str, payload: bytes, **kw: Any) -> asyncio.Future:
        await bus.publish_work(subject, payload)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    js.publish_async = AsyncMock(side_effect=_fake_publish_async)
    js.publish_async_completed = AsyncMock()

    # Wire publish (for score) to bus
    async def _fake_publish(subject: str, payload: bytes, **kw: Any) -> MagicMock:
        await bus.publish_work(subject, payload)
        ack = MagicMock()
        ack.stream = "test"
        return ack

    js.publish = AsyncMock(side_effect=_fake_publish)

    # Wire inbox subscription to bus
    async def _fake_subscribe(subject: str, cb: Any = None, **kw: Any) -> MagicMock:
        bus.subscribe_inbox(subject, cb)
        sub = MagicMock()
        sub.unsubscribe = AsyncMock()
        return sub

    nc.subscribe = AsyncMock(side_effect=_fake_subscribe)

    publisher = WorkPublisher(nc, js, jsm, router_id, payload_store=payload_store)
    return publisher


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestMultiModelIsolation:
    """Items for different models route to the correct workers."""

    @pytest.mark.asyncio
    async def test_two_models_separate_workers(self) -> None:
        """Items for model-A go to worker-A, items for model-B go to worker-B."""
        bus = QueueBus()
        worker_a = SimulatedWorker(bus, "w-a", ["test/model-a"])
        worker_b = SimulatedWorker(bus, "w-b", ["test/model-b"])
        pub = _make_bus_publisher(bus)
        await pub.start()
        await worker_a.start()
        await worker_b.start()

        try:
            results_a = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model-a",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": f"a-{i}"} for i in range(5)],
                ),
                timeout=5.0,
            )
            results_b = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model-b",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": f"b-{i}"} for i in range(5)],
                ),
                timeout=5.0,
            )

            assert len(results_a) == 5
            assert len(results_b) == 5

            # Worker A only processed model-A items
            a_models = {wi["model_id"] for wi in worker_a.processed_items}
            assert a_models == {"test/model-a"}

            # Worker B only processed model-B items
            b_models = {wi["model_id"] for wi in worker_b.processed_items}
            assert b_models == {"test/model-b"}
        finally:
            await worker_a.stop()
            await worker_b.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_concurrent_models_no_interference(self) -> None:
        """Publishing to two models concurrently produces correct results for each."""
        bus = QueueBus()
        worker_a = SimulatedWorker(bus, "w-a", ["test/model-a"])
        worker_b = SimulatedWorker(bus, "w-b", ["test/model-b"])
        pub = _make_bus_publisher(bus)
        await pub.start()
        await worker_a.start()
        await worker_b.start()

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    pub.submit_encode(
                        model_id="test/model-a",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"a-{i}"} for i in range(10)],
                    ),
                    pub.submit_encode(
                        model_id="test/model-b",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"b-{i}"} for i in range(10)],
                    ),
                ),
                timeout=5.0,
            )
            assert len(results[0]) == 10
            assert len(results[1]) == 10
            assert len(worker_a.processed_items) == 10
            assert len(worker_b.processed_items) == 10
        finally:
            await worker_a.stop()
            await worker_b.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_pool_isolation(self) -> None:
        """Items for pool-eval only go to eval workers, not default workers."""
        bus = QueueBus()
        worker_default = SimulatedWorker(bus, "w-default", ["test/model-a"], pool_name="_default")
        worker_eval = SimulatedWorker(bus, "w-eval", ["test/model-a"], pool_name="eval")
        pub = _make_bus_publisher(bus)
        await pub.start()
        await worker_default.start()
        await worker_eval.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model-a",
                    profile_id="default",
                    pool_name="eval",
                    machine_profile="l4",
                    items=[{"text": f"eval-{i}"} for i in range(5)],
                ),
                timeout=5.0,
            )
            assert len(results) == 5
            assert len(worker_eval.processed_items) == 5
            assert len(worker_default.processed_items) == 0
        finally:
            await worker_default.stop()
            await worker_eval.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_model_not_in_worker_ignored(self) -> None:
        """Worker subscribed to model-A ignores items for model-B."""
        bus = QueueBus()
        worker = SimulatedWorker(bus, "w-a", ["test/model-a"])
        await worker.start()

        # Manually publish to model-B — worker shouldn't see it
        subject = work_subject("test/model-b", "_default")
        wi = msgpack.packb(
            {"work_item_id": "orphan", "model_id": "test/model-b"},
            use_bin_type=True,
        )
        await bus.publish_work(subject, wi)

        await asyncio.sleep(0.05)
        assert len(worker.processed_items) == 0
        await worker.stop()

    @pytest.mark.asyncio
    async def test_score_not_decomposed(self) -> None:
        """Score publishes single work item with query + items."""
        bus = QueueBus()
        worker = SimulatedWorker(bus, "w-1", ["test/reranker"])
        pub = _make_bus_publisher(bus)
        await pub.start()
        await worker.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_score(
                    model_id="test/reranker",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    query={"text": "query"},
                    items=[{"text": f"doc-{i}"} for i in range(10)],
                ),
                timeout=5.0,
            )
            assert len(results) == 1
            # Only 1 work item published (not 10)
            assert len(worker.processed_items) == 1
            wi = worker.processed_items[0]
            assert wi["operation"] == "score"
        finally:
            await worker.stop()
            await pub.stop()


class TestMultiWorkerDistribution:
    """Multiple workers sharing the same model distribute items fairly."""

    @pytest.mark.asyncio
    async def test_items_distributed_across_workers(self) -> None:
        """20 items published, 2 workers — both get items, no duplicates."""
        bus = QueueBus()
        w1 = SimulatedWorker(bus, "w-1", ["test/model"], batch_budget=5)
        w2 = SimulatedWorker(bus, "w-2", ["test/model"], batch_budget=5)
        pub = _make_bus_publisher(bus)
        await pub.start()
        await w1.start()
        await w2.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": f"item-{i}"} for i in range(20)],
                ),
                timeout=5.0,
            )
            assert len(results) == 20

            # Both workers processed items
            total = len(w1.processed_items) + len(w2.processed_items)
            assert total == 20
            assert len(w1.processed_items) > 0
            assert len(w2.processed_items) > 0

            # No duplicates
            all_ids = [wi["work_item_id"] for wi in w1.processed_items + w2.processed_items]
            assert len(set(all_ids)) == 20
        finally:
            await w1.stop()
            await w2.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_worker_crash_items_redelivered(self) -> None:
        """Worker crashes (NAKs items) -> items redelivered to other worker."""
        bus = QueueBus()

        class CrashingWorker(SimulatedWorker):
            """Worker that NAKs all items (simulates crash)."""

            async def _run(self) -> None:
                while self._running:
                    for model_id in self.model_ids:
                        subject = work_subject(model_id, self.pool_name)
                        msgs = await self.bus.pull(subject, 1, timeout=0.01)
                        for msg in msgs:
                            wi = msgpack.unpackb(msg.data, raw=False)
                            self.naked_items.append(wi)
                            await msg.nak(delay=0.01)
                    await asyncio.sleep(0.005)

        crasher = CrashingWorker(bus, "w-crash", ["test/model"])
        healthy = SimulatedWorker(bus, "w-healthy", ["test/model"])
        pub = _make_bus_publisher(bus)
        await pub.start()
        await crasher.start()
        await healthy.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "hello"}],
                ),
                timeout=5.0,
            )
            assert len(results) == 1
            assert results[0]["success"] is True
            # Healthy worker eventually processed it
            assert len(healthy.processed_items) >= 1
        finally:
            await crasher.stop()
            await healthy.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_all_results_from_correct_request(self) -> None:
        """Results from worker include correct request_id and item_index."""
        bus = QueueBus()
        w = SimulatedWorker(bus, "w-1", ["test/model"])
        pub = _make_bus_publisher(bus)
        await pub.start()
        await w.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": f"item-{i}"} for i in range(5)],
                ),
                timeout=5.0,
            )
            assert len(results) == 5
            # All results must share the same request_id
            request_ids = {r["request_id"] for r in results}
            assert len(request_ids) == 1, f"Expected 1 request_id, got {request_ids}"
            for i, r in enumerate(results):
                assert r["item_index"] == i
                assert r["success"] is True
        finally:
            await w.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_large_batch_distributed(self) -> None:
        """100 items across 3 workers — all processed, no duplicates."""
        bus = QueueBus()
        workers = [SimulatedWorker(bus, f"w-{i}", ["test/model"], batch_budget=20) for i in range(3)]
        pub = _make_bus_publisher(bus)
        await pub.start()
        for w in workers:
            await w.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": f"item-{i}"} for i in range(100)],
                ),
                timeout=10.0,
            )
            assert len(results) == 100
            total = sum(len(w.processed_items) for w in workers)
            assert total == 100
            all_ids: list[str] = []
            for w in workers:
                all_ids.extend(wi["work_item_id"] for wi in w.processed_items)
            assert len(set(all_ids)) == 100
        finally:
            for w in workers:
                await w.stop()
            await pub.stop()


class TestMultiRouterConcurrency:
    """Multiple routers publishing concurrently to the same stream."""

    @pytest.mark.asyncio
    async def test_two_routers_no_cross_contamination(self) -> None:
        """Router-A and Router-B publish concurrently — each gets only its own results."""
        bus = QueueBus()
        w = SimulatedWorker(bus, "w-1", ["test/model"])
        pub_a = _make_bus_publisher(bus, router_id="router-a")
        pub_b = _make_bus_publisher(bus, router_id="router-b")
        await pub_a.start()
        await pub_b.start()
        await w.start()

        try:
            results_a, results_b = await asyncio.wait_for(
                asyncio.gather(
                    pub_a.submit_encode(
                        model_id="test/model",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"a-{i}"} for i in range(5)],
                    ),
                    pub_b.submit_encode(
                        model_id="test/model",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"b-{i}"} for i in range(5)],
                    ),
                ),
                timeout=5.0,
            )
            assert len(results_a) == 5
            assert len(results_b) == 5

            # Verify request IDs are different
            req_ids_a = {r["request_id"] for r in results_a}
            req_ids_b = {r["request_id"] for r in results_b}
            assert req_ids_a.isdisjoint(req_ids_b)
        finally:
            await w.stop()
            await pub_a.stop()
            await pub_b.stop()

    @pytest.mark.asyncio
    async def test_router_crash_doesnt_affect_other(self) -> None:
        """Router-A's request times out — Router-B's request succeeds."""
        bus = QueueBus()
        w = SimulatedWorker(bus, "w-1", ["test/model"])
        pub_a = _make_bus_publisher(bus, router_id="router-a")
        pub_b = _make_bus_publisher(bus, router_id="router-b")
        await pub_a.start()
        await pub_b.start()
        await w.start()

        try:
            # Simulate Router-A crashing: start a request but cancel it quickly
            task_a = asyncio.create_task(
                pub_a.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "a-crash"}],
                    timeout=0.5,  # Very short timeout to simulate crash
                )
            )

            # Router-B's request should succeed despite Router-A's failure
            results_b = await asyncio.wait_for(
                pub_b.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "b-1"}],
                ),
                timeout=5.0,
            )

            # Clean up Router-A's failed task
            try:
                await task_a
            except (TimeoutError, Exception):  # noqa: BLE001, S110
                pass  # Expected — Router-A "crashed"

            assert len(results_b) == 1
            assert results_b[0]["success"] is True
        finally:
            await w.stop()
            await pub_a.stop()
            await pub_b.stop()

    @pytest.mark.asyncio
    async def test_many_concurrent_requests(self) -> None:
        """5 concurrent requests from 2 routers — all succeed."""
        bus = QueueBus()
        workers = [SimulatedWorker(bus, f"w-{i}", ["test/model"]) for i in range(2)]
        pub_a = _make_bus_publisher(bus, router_id="router-a")
        pub_b = _make_bus_publisher(bus, router_id="router-b")
        await pub_a.start()
        await pub_b.start()
        for w in workers:
            await w.start()

        try:
            tasks = []
            for i in range(3):
                tasks.append(
                    pub_a.submit_encode(
                        model_id="test/model",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"a-{i}-{j}"} for j in range(3)],
                    )
                )
            for i in range(2):
                tasks.append(
                    pub_b.submit_encode(
                        model_id="test/model",
                        profile_id="default",
                        pool_name="_default",
                        machine_profile="l4",
                        items=[{"text": f"b-{i}-{j}"} for j in range(3)],
                    )
                )

            all_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10.0)
            for results in all_results:
                assert len(results) == 3
                assert all(r["success"] for r in results)
        finally:
            for w in workers:
                await w.stop()
            await pub_a.stop()
            await pub_b.stop()


class TestReactiveModelLoading:
    """Worker handles items for unloaded models via NAK and background load."""

    @pytest.mark.asyncio
    async def test_unloaded_model_items_naked(self) -> None:
        """Items for an unloaded model are NAKed, not processed."""
        bus = QueueBus()
        # Worker has model-a loaded but model-b configured but not loaded
        w = SimulatedWorker(
            bus,
            "w-1",
            ["test/model-a", "test/model-b"],
            loaded_models={"test/model-a"},
        )
        await w.start()

        # Publish item for unloaded model
        subject = work_subject("test/model-b", "_default")
        wi: WorkItem = {
            "work_item_id": "req-1.0",
            "request_id": "req-1",
            "item_index": 0,
            "total_items": 1,
            "operation": "encode",
            "model_id": "test/model-b",
            "profile_id": "default",
            "pool_name": "_default",
            "reply_subject": "_INBOX.test.req-1",
            "router_id": "r1",
            "timestamp": time.time(),
        }
        await bus.publish_work(subject, msgpack.packb(wi, use_bin_type=True))

        await asyncio.sleep(0.1)
        assert len(w.naked_items) >= 1
        assert len(w.processed_items) == 0
        await w.stop()

    @pytest.mark.asyncio
    async def test_loaded_model_not_starved_during_unloaded_nak(self) -> None:
        """While model-B items are NAKed, model-A items still process normally."""
        bus = QueueBus()
        w = SimulatedWorker(
            bus,
            "w-1",
            ["test/model-a", "test/model-b"],
            loaded_models={"test/model-a"},
        )
        pub = _make_bus_publisher(bus)
        await pub.start()
        await w.start()

        try:
            # Model-A should succeed even though model-B items would be NAKed
            results_a = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model-a",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "a-item"}],
                ),
                timeout=5.0,
            )
            assert len(results_a) == 1
            assert results_a[0]["success"] is True
        finally:
            await w.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_naked_items_redeliver_to_loaded_worker(self) -> None:
        """NAKed items are redelivered and eventually processed by a worker that has the model loaded."""
        bus = QueueBus()

        class NakOnceWorker(SimulatedWorker):
            """Worker that NAKs once then stops pulling (simulates unloaded worker stepping aside)."""

            async def _run(self) -> None:
                while self._running:
                    for model_id in self.model_ids:
                        subject = work_subject(model_id, self.pool_name)
                        msgs = await self.bus.pull(subject, 1, timeout=0.01)
                        for msg in msgs:
                            wi = msgpack.unpackb(msg.data, raw=False)
                            self.naked_items.append(wi)
                            await msg.nak(delay=0.01)
                            # Stop after first NAK so redelivered msg goes to w2
                            self._running = False
                            return
                    await asyncio.sleep(0.005)

        # Worker-1 doesn't have model-b loaded — will NAK once, then stop
        w1 = NakOnceWorker(bus, "w-1", ["test/model-b"], loaded_models=set())
        # Worker-2 has model-b loaded — will process
        w2 = SimulatedWorker(bus, "w-2", ["test/model-b"], loaded_models={"test/model-b"})
        pub = _make_bus_publisher(bus)
        await pub.start()
        await w1.start()
        await w2.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model-b",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "b-item"}],
                ),
                timeout=5.0,
            )
            assert len(results) == 1
            assert results[0]["success"] is True
            # w1 NAKed at least once
            assert len(w1.naked_items) >= 1
            # Worker-2 processed it after redelivery
            assert len(w2.processed_items) >= 1
        finally:
            await w1.stop()
            await w2.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_model_loaded_after_nak_processes_items(self) -> None:
        """Worker initially NAKs, then 'loads' the model, processes redelivered items."""
        bus = QueueBus()
        w = SimulatedWorker(bus, "w-1", ["test/model"], loaded_models=set())
        pub = _make_bus_publisher(bus)
        await pub.start()
        await w.start()

        # Simulate model being loaded after a short delay
        async def load_model_later() -> None:
            await asyncio.sleep(0.05)
            w.loaded_models.add("test/model")

        try:
            load_task = asyncio.create_task(load_model_later())
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "delayed-item"}],
                ),
                timeout=5.0,
            )
            await load_task
            assert len(results) == 1
            assert results[0]["success"] is True
        finally:
            await w.stop()
            await pub.stop()


class TestPayloadOffloadEndToEnd:
    """Router offloads large payload -> worker fetches from same store."""

    @pytest.mark.asyncio
    async def test_large_payload_roundtrip(self, tmp_path: Any) -> None:
        """Large item offloaded to payload store, worker fetches and processes."""
        store = LocalPayloadStore(base_dir=str(tmp_path / "payloads"))
        bus = QueueBus()

        class PayloadAwareWorker(SimulatedWorker):
            """Worker that checks for payload_ref."""

            async def _run(self) -> None:
                while self._running:
                    for model_id in self.model_ids:
                        subject = work_subject(model_id, self.pool_name)
                        msgs = await self.bus.pull(subject, self.batch_budget, timeout=0.01)
                        for msg in msgs:
                            wi = msgpack.unpackb(msg.data, raw=False)
                            self.processed_items.append(wi)

                            # Verify payload was offloaded
                            if wi.get("payload_ref"):
                                payload = await store.get(wi["payload_ref"])
                                assert len(payload) > 0

                            result: WorkResult = {
                                "work_item_id": wi.get("work_item_id", ""),
                                "request_id": wi.get("request_id", ""),
                                "item_index": wi.get("item_index", 0),
                                "success": True,
                                "result_msgpack": msgpack.packb({"dense": [0.1]}, use_bin_type=True),
                                "queue_ms": 0.0,
                                "processing_ms": 0.0,
                                "worker_id": self.worker_id,
                            }
                            reply = wi.get("reply_subject", "")
                            if reply:
                                await self.bus.publish_inbox(
                                    reply,
                                    msgpack.packb(result, use_bin_type=True),
                                )
                            await msg.ack()
                    await asyncio.sleep(0.005)

        w = PayloadAwareWorker(bus, "w-1", ["test/model"])
        pub = _make_bus_publisher(bus, payload_store=store)
        await pub.start()
        await w.start()

        try:
            # Create a large item that exceeds inline threshold
            large_text = "x" * (1024 * 1024 + 100)  # > 1MB
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": large_text}],
                ),
                timeout=5.0,
            )
            assert len(results) == 1
            assert results[0]["success"] is True
            # Verify the worker received a payload_ref (not inline)
            assert w.processed_items[0].get("payload_ref") is not None
        finally:
            await w.stop()
            await pub.stop()

    @pytest.mark.asyncio
    async def test_small_payload_stays_inline(self, tmp_path: Any) -> None:
        """Small items are published inline, not offloaded."""
        store = LocalPayloadStore(base_dir=str(tmp_path / "payloads"))
        bus = QueueBus()
        w = SimulatedWorker(bus, "w-1", ["test/model"])
        pub = _make_bus_publisher(bus, payload_store=store)
        await pub.start()
        await w.start()

        try:
            results = await asyncio.wait_for(
                pub.submit_encode(
                    model_id="test/model",
                    profile_id="default",
                    pool_name="_default",
                    machine_profile="l4",
                    items=[{"text": "small text"}],
                ),
                timeout=5.0,
            )
            assert len(results) == 1
            # Item should be inline (no payload_ref)
            assert w.processed_items[0].get("payload_ref") is None
            assert w.processed_items[0].get("item") is not None
        finally:
            await w.stop()
            await pub.stop()
