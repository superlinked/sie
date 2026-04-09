import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from sie_router.jetstream_manager import JetStreamManager
from sie_router.work_publisher import WorkPublisher
from sie_sdk.queue_types import WorkResult

ROUTER_ID = "int-test-router"
MODEL_ID = "BAAI/bge-m3"
PROFILE_ID = "default"
POOL_NAME = "_default"
MACHINE_PROFILE = "l4"


def _make_work_result(
    request_id: str,
    item_index: int,
    *,
    success: bool = True,
    error: str | None = None,
    result_data: bytes | None = None,
) -> WorkResult:
    result: WorkResult = {
        "work_item_id": f"{request_id}.{item_index}",
        "request_id": request_id,
        "item_index": item_index,
        "success": success,
    }
    if result_data is not None:
        result["result_msgpack"] = result_data
    if not success and error:
        result["error"] = error
        result["error_code"] = "worker_error"
    return result


class _InProcessNATS:
    """Simulates NATS pub/sub in-process for integration tests.

    Tracks subscriptions and routes published messages to callbacks.
    JetStream publish goes through the same mechanism with subject matching.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list] = {}  # subject -> list of callbacks
        self._published: list[tuple[str, bytes]] = []

    async def subscribe(self, subject: str, cb: object = None) -> MagicMock:
        """Register a subscription. Supports '>' wildcard at end."""
        if subject not in self._subscriptions:
            self._subscriptions[subject] = []
        self._subscriptions[subject].append(cb)

        sub = MagicMock()
        sub.unsubscribe = AsyncMock()
        return sub

    async def publish(self, subject: str, payload: bytes) -> None:
        """Publish a message (core NATS publish for results)."""
        self._published.append((subject, payload))
        await self._deliver(subject, payload)

    async def js_publish(self, subject: str, payload: bytes) -> None:
        """JetStream publish (for work items). Records but doesn't auto-deliver."""
        self._published.append((subject, payload))

    async def _deliver(self, subject: str, payload: bytes) -> None:
        """Deliver message to matching subscribers."""
        for pattern, callbacks in self._subscriptions.items():
            if self._matches(pattern, subject):
                for cb in callbacks:
                    msg = MagicMock()
                    msg.data = payload
                    msg.subject = subject
                    await cb(msg)

    @staticmethod
    def _matches(pattern: str, subject: str) -> bool:
        """Simple NATS subject matching with '>' wildcard."""
        if pattern == subject:
            return True
        if pattern.endswith(".>"):
            prefix = pattern[:-2]
            return subject.startswith(prefix + ".") or subject == prefix
        return False


def _wrap_as_publish_async(
    fn: Any,
) -> Any:
    """Wrap an async worker function so it returns a Future (like js.publish_async)."""

    async def _wrapped(subject: str, payload: bytes, **kwargs: Any) -> asyncio.Future:
        await fn(subject, payload)
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    return _wrapped


def _make_integration_publisher() -> tuple[WorkPublisher, _InProcessNATS, AsyncMock]:
    """Create a WorkPublisher wired to an in-process NATS simulation."""
    sim = _InProcessNATS()

    nc = MagicMock()
    nc.subscribe = sim.subscribe

    js = AsyncMock()
    js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(sim.js_publish))
    js.publish_async_completed = AsyncMock()

    jsm = AsyncMock(spec=JetStreamManager)
    jsm.get_stream_health = AsyncMock(return_value=(2, 0))
    jsm.ensure_stream = AsyncMock(return_value="WORK_BAAI__bge-m3")

    publisher = WorkPublisher(nc, js, jsm, ROUTER_ID)
    return publisher, sim, js


class TestEncodeFlowIntegration:
    """Integration test: publish encode items -> simulate worker results -> verify ordering."""

    @pytest.mark.asyncio
    async def test_full_encode_flow(self) -> None:
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        items = [{"text": "hello"}, {"text": "world"}, {"text": "test"}]

        # Intercept JetStream publishes and respond with results (simulating a worker)
        published_work_items: list[dict] = []

        async def worker_sim(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            published_work_items.append(work_item)

            # Simulate worker processing and publishing result to reply_subject
            reply_subject = work_item["reply_subject"]
            result = _make_work_result(
                work_item["request_id"],
                work_item["item_index"],
                result_data=msgpack.packb({"embedding": [0.1, 0.2]}, use_bin_type=True),
            )
            result_payload = msgpack.packb(result, use_bin_type=True)
            # Deliver result via the simulated NATS (hits the inbox subscription)
            await sim.publish(reply_subject, result_payload)

        # Replace js.publish_async to call our worker simulator
        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(worker_sim))

        results = await publisher.submit_encode(
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            items=items,
            timeout=5.0,
        )

        await publisher.stop()

        # Verify: 3 work items published
        assert len(published_work_items) == 3

        # Verify: results are ordered by item_index
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["item_index"] == i
            assert result["success"] is True

        # Verify: work items have correct indices
        indices = sorted(wi["item_index"] for wi in published_work_items)
        assert indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_out_of_order_results_are_reordered(self) -> None:
        """Results arriving in reverse order are still returned in correct order."""
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        collected_work_items: list[dict] = []

        async def reverse_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            collected_work_items.append(work_item)

        # Collect work items but don't respond yet
        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(reverse_worker))

        # Start the submit in a background task
        async def do_submit():
            return await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
                timeout=5.0,
            )

        task = asyncio.create_task(do_submit())

        # Wait for work items to be published (poll instead of fixed sleep)
        for _ in range(100):  # up to 1s
            await asyncio.sleep(0.01)
            if len(collected_work_items) >= len(items):
                break
        else:
            pytest.fail(
                f"Timed out waiting for work items to be published (got {len(collected_work_items)}/{len(items)})"
            )

        # Deliver results in reverse order (2, 1, 0)
        for wi in reversed(collected_work_items):
            reply_subject = wi["reply_subject"]
            result = _make_work_result(
                wi["request_id"],
                wi["item_index"],
                result_data=msgpack.packb({"idx": wi["item_index"]}, use_bin_type=True),
            )
            await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        results = await task
        await publisher.stop()

        # Despite reverse delivery, results should be ordered
        assert [r["item_index"] for r in results] == [0, 1, 2]


class TestScoreFlowIntegration:
    """Integration test for score flow."""

    @pytest.mark.asyncio
    async def test_score_single_work_item(self) -> None:
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        async def score_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            assert work_item["operation"] == "score"
            assert work_item["query_item"] == {"text": "query"}
            assert work_item["score_items"] == [{"text": "doc1"}, {"text": "doc2"}]

            reply_subject = work_item["reply_subject"]
            result = _make_work_result(
                work_item["request_id"],
                0,
                result_data=msgpack.packb({"scores": [0.9, 0.3]}, use_bin_type=True),
            )
            await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        js.publish = score_worker

        results = await publisher.submit_score(
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            query={"text": "query"},
            items=[{"text": "doc1"}, {"text": "doc2"}],
            timeout=5.0,
        )

        await publisher.stop()

        assert len(results) == 1
        assert results[0]["success"] is True


class TestTimeoutIntegration:
    """Integration test: no results arrive -> TimeoutError."""

    @pytest.mark.asyncio
    async def test_timeout_when_no_results(self) -> None:
        publisher, _sim, js = _make_integration_publisher()
        await publisher.start()

        # Worker never responds (js.publish_async is a no-op that returns Futures)
        async def _noop_publish(subject: str, payload: bytes, **kwargs: Any) -> asyncio.Future:
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            fut.set_result(None)
            return fut

        js.publish_async = AsyncMock(side_effect=_noop_publish)

        with pytest.raises(TimeoutError):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "a"}],
                timeout=0.05,
            )

        await publisher.stop()

        # Pending request should be cleaned up even on timeout
        assert len(publisher._pending) == 0

    @pytest.mark.asyncio
    async def test_timeout_with_partial_results(self) -> None:
        """Some results arrive but not all -> TimeoutError."""
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        async def partial_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            # Only respond to item_index 0, skip item_index 1
            if work_item["item_index"] == 0:
                reply_subject = work_item["reply_subject"]
                result = _make_work_result(work_item["request_id"], 0)
                await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(partial_worker))

        with pytest.raises(TimeoutError):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "a"}, {"text": "b"}],
                timeout=0.1,
            )

        await publisher.stop()


class TestPartialFailureIntegration:
    """Integration test: some items succeed, some fail."""

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self) -> None:
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        async def flaky_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            reply_subject = work_item["reply_subject"]
            idx = work_item["item_index"]

            if idx % 2 == 0:
                # Even indices succeed
                result = _make_work_result(
                    work_item["request_id"],
                    idx,
                    result_data=msgpack.packb({"ok": True}, use_bin_type=True),
                )
            else:
                # Odd indices fail
                result = _make_work_result(
                    work_item["request_id"],
                    idx,
                    success=False,
                    error=f"item {idx} failed",
                )

            await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(flaky_worker))

        results = await publisher.submit_encode(
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            items=[{"text": str(i)} for i in range(4)],
            timeout=5.0,
        )

        await publisher.stop()

        assert len(results) == 4

        # Even indices succeeded
        assert results[0]["success"] is True
        assert results[2]["success"] is True

        # Odd indices failed
        assert results[1]["success"] is False
        assert results[1]["error"] == "item 1 failed"
        assert results[3]["success"] is False
        assert results[3]["error"] == "item 3 failed"

        # All ordered by item_index
        assert [r["item_index"] for r in results] == [0, 1, 2, 3]

    @pytest.mark.asyncio
    async def test_all_items_fail(self) -> None:
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        async def failing_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            reply_subject = work_item["reply_subject"]
            result = _make_work_result(
                work_item["request_id"],
                work_item["item_index"],
                success=False,
                error="GPU out of memory",
            )
            await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(failing_worker))

        results = await publisher.submit_encode(
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            items=[{"text": "a"}, {"text": "b"}],
            timeout=5.0,
        )

        await publisher.stop()

        assert len(results) == 2
        assert all(not r["success"] for r in results)
        assert all(r["error"] == "GPU out of memory" for r in results)


class TestExtractFlowIntegration:
    """Integration test for extract flow."""

    @pytest.mark.asyncio
    async def test_extract_decomposes_like_encode(self) -> None:
        publisher, sim, js = _make_integration_publisher()
        await publisher.start()

        published_operations: list[str] = []

        async def extract_worker(subject: str, payload: bytes) -> None:
            work_item = msgpack.unpackb(payload, raw=False)
            published_operations.append(work_item["operation"])

            reply_subject = work_item["reply_subject"]
            result = _make_work_result(
                work_item["request_id"],
                work_item["item_index"],
                result_data=msgpack.packb({"entities": []}, use_bin_type=True),
            )
            await sim.publish(reply_subject, msgpack.packb(result, use_bin_type=True))

        js.publish_async = AsyncMock(side_effect=_wrap_as_publish_async(extract_worker))

        results = await publisher.submit_extract(
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            items=[{"text": "Alice works at Acme"}, {"text": "Bob went to Paris"}],
            labels=["PER", "ORG", "LOC"],
            timeout=5.0,
        )

        await publisher.stop()

        # Extract decomposes into one work item per input
        assert len(published_operations) == 2
        assert all(op == "extract" for op in published_operations)
        assert len(results) == 2
        assert all(r["success"] for r in results)
