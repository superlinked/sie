import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from sie_router.jetstream_manager import JetStreamManager
from sie_router.payload_store import PayloadStore
from sie_router.work_publisher import (
    NoConsumersError,
    PendingRequest,
    WorkPublisher,
)
from sie_sdk.queue_types import INLINE_THRESHOLD_BYTES, WorkResult

ROUTER_ID = "test-router"
MODEL_ID = "BAAI/bge-m3"
PROFILE_ID = "default"
POOL_NAME = "_default"
MACHINE_PROFILE = "l4"


def _make_publisher(
    *,
    payload_store: PayloadStore | None = None,
) -> tuple[WorkPublisher, AsyncMock, AsyncMock, AsyncMock]:
    """Create a WorkPublisher with mocked NATS/JetStream dependencies."""
    nc = AsyncMock()
    js = AsyncMock()

    # publish_async must return real asyncio.Future objects (not AsyncMock)
    # because submit_encode inspects .done() and .exception() on them.
    async def _fake_publish_async(subject: str, payload: bytes, **kwargs: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        fut.set_result(None)  # Simulate successful PubAck
        return fut

    js.publish_async = AsyncMock(side_effect=_fake_publish_async)
    js.publish_async_completed = AsyncMock()

    jsm = AsyncMock(spec=JetStreamManager)
    # Default: stream has consumers and is below backpressure limit
    jsm.get_stream_health = AsyncMock(return_value=(2, 0))
    jsm.ensure_stream = AsyncMock(return_value="WORK_BAAI__bge-m3")
    publisher = WorkPublisher(nc, js, jsm, ROUTER_ID, payload_store=payload_store)
    return publisher, nc, js, jsm


def _make_work_result(
    request_id: str,
    item_index: int,
    *,
    success: bool = True,
    error: str | None = None,
) -> WorkResult:
    """Build a minimal WorkResult dict."""
    result: WorkResult = {
        "work_item_id": f"{request_id}.{item_index}",
        "request_id": request_id,
        "item_index": item_index,
        "success": success,
    }
    if not success and error:
        result["error"] = error
        result["error_code"] = "test_error"
    return result


class TestNextRequestId:
    """Tests for WorkPublisher._next_request_id."""

    def test_generates_unique_sequential_ids(self) -> None:
        publisher, *_ = _make_publisher()

        id1 = publisher._next_request_id()
        id2 = publisher._next_request_id()
        id3 = publisher._next_request_id()

        assert id1 == f"{ROUTER_ID}-1"
        assert id2 == f"{ROUTER_ID}-2"
        assert id3 == f"{ROUTER_ID}-3"

    def test_ids_include_router_id(self) -> None:
        nc = AsyncMock()
        js = AsyncMock()
        jsm = AsyncMock(spec=JetStreamManager)
        publisher = WorkPublisher(nc, js, jsm, "my-router-42")

        rid = publisher._next_request_id()
        assert rid.startswith("my-router-42-")

    def test_no_duplicates_across_many_calls(self) -> None:
        publisher, *_ = _make_publisher()
        ids = {publisher._next_request_id() for _ in range(1000)}
        assert len(ids) == 1000


class TestOnResult:
    """Tests for WorkPublisher._on_result."""

    @pytest.mark.asyncio
    async def test_stores_result_and_sets_completed(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=2,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        # Deliver result for item 0
        msg0 = MagicMock()
        msg0.data = msgpack.packb(_make_work_result(request_id, 0), use_bin_type=True)
        await publisher._on_result(msg0)

        assert 0 in pending.results
        assert not pending.completed.is_set()

        # Deliver result for item 1 — should complete
        msg1 = MagicMock()
        msg1.data = msgpack.packb(_make_work_result(request_id, 1), use_bin_type=True)
        await publisher._on_result(msg1)

        assert 1 in pending.results
        assert pending.completed.is_set()

    @pytest.mark.asyncio
    async def test_stores_error_result(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=1,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        msg = MagicMock()
        msg.data = msgpack.packb(
            _make_work_result(request_id, 0, success=False, error="model OOM"),
            use_bin_type=True,
        )
        await publisher._on_result(msg)

        assert 0 in pending.errors
        assert pending.errors[0] == ("model OOM", "test_error")
        assert pending.completed.is_set()

    @pytest.mark.asyncio
    async def test_deduplicates_same_item_index(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=1,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        result = _make_work_result(request_id, 0)
        msg = MagicMock()
        msg.data = msgpack.packb(result, use_bin_type=True)

        await publisher._on_result(msg)
        assert pending.completed.is_set()

        # Second delivery of same item_index — should be silently dropped
        dup_result = _make_work_result(request_id, 0)
        dup_result["worker_id"] = "duplicate-worker"
        dup_msg = MagicMock()
        dup_msg.data = msgpack.packb(dup_result, use_bin_type=True)
        await publisher._on_result(dup_msg)

        # Original result preserved (no "worker_id" key)
        assert pending.results[0].get("worker_id") is None

    @pytest.mark.asyncio
    async def test_deduplicates_error_then_success(self) -> None:
        """First error result wins; subsequent success for same index is dropped."""
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=1,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        # First: error
        err_msg = MagicMock()
        err_msg.data = msgpack.packb(
            _make_work_result(request_id, 0, success=False, error="fail"),
            use_bin_type=True,
        )
        await publisher._on_result(err_msg)
        assert 0 in pending.errors

        # Second: success for same index — should be dropped
        ok_msg = MagicMock()
        ok_msg.data = msgpack.packb(
            _make_work_result(request_id, 0, success=True),
            use_bin_type=True,
        )
        await publisher._on_result(ok_msg)

        # Error still there, success not added
        assert 0 in pending.errors
        assert 0 not in pending.results

    @pytest.mark.asyncio
    async def test_unknown_request_id_dropped(self) -> None:
        publisher, *_ = _make_publisher()

        msg = MagicMock()
        msg.data = msgpack.packb(
            _make_work_result("unknown-request-999", 0),
            use_bin_type=True,
        )

        # Should not raise and not modify state
        await publisher._on_result(msg)
        assert len(publisher._pending) == 0

    @pytest.mark.asyncio
    async def test_malformed_msgpack_dropped(self) -> None:
        publisher, *_ = _make_publisher()

        msg = MagicMock()
        msg.data = b"not valid msgpack \xff\xfe"

        # Should not raise
        await publisher._on_result(msg)

    @pytest.mark.asyncio
    async def test_mixed_successes_and_errors_complete(self) -> None:
        """Completed event fires when total_received == total_items (mix of ok/err)."""
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=3,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        # Item 0: success
        msg0 = MagicMock()
        msg0.data = msgpack.packb(_make_work_result(request_id, 0), use_bin_type=True)
        await publisher._on_result(msg0)

        # Item 1: error
        msg1 = MagicMock()
        msg1.data = msgpack.packb(
            _make_work_result(request_id, 1, success=False, error="err"),
            use_bin_type=True,
        )
        await publisher._on_result(msg1)

        assert not pending.completed.is_set()

        # Item 2: success — should complete
        msg2 = MagicMock()
        msg2.data = msgpack.packb(_make_work_result(request_id, 2), use_bin_type=True)
        await publisher._on_result(msg2)

        assert pending.completed.is_set()


class TestWaitForResults:
    """Tests for WorkPublisher._wait_for_results."""

    @pytest.mark.asyncio
    async def test_returns_ordered_results(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=3,
            results={
                2: _make_work_result(request_id, 2),
                0: _make_work_result(request_id, 0),
                1: _make_work_result(request_id, 1),
            },
            errors={},
        )
        pending.completed.set()

        results = await publisher._wait_for_results(pending, timeout=5.0)

        assert len(results) == 3
        assert results[0]["item_index"] == 0
        assert results[1]["item_index"] == 1
        assert results[2]["item_index"] == 2

    @pytest.mark.asyncio
    async def test_returns_error_results_in_order(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=2,
            results={0: _make_work_result(request_id, 0)},
            errors={1: ("oops", "internal_error")},
        )
        pending.completed.set()

        results = await publisher._wait_for_results(pending, timeout=5.0)

        assert len(results) == 2
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[1]["error"] == "oops"
        assert results[1]["error_code"] == "internal_error"

    @pytest.mark.asyncio
    async def test_raises_timeout_error(self) -> None:
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=2,
            results={},
            errors={},
        )
        # Do NOT set completed — simulates timeout

        with pytest.raises(TimeoutError):
            await publisher._wait_for_results(pending, timeout=0.01)

    @pytest.mark.asyncio
    async def test_missing_result_index_returns_error_placeholder(self) -> None:
        """If completed is set but an index is missing, a placeholder error is returned."""
        publisher, *_ = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=2,
            results={0: _make_work_result(request_id, 0)},
            errors={},
            # Missing index 1 — but we force completed
        )
        pending.completed.set()

        results = await publisher._wait_for_results(pending, timeout=5.0)

        assert results[1]["success"] is False
        assert results[1]["error"] == "Result missing"
        assert results[1]["error_code"] == "result_missing"


class TestCheckBackpressure:
    """Tests for WorkPublisher._check_backpressure."""

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_over_limit(self) -> None:
        publisher, _, _, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(2, 999_999))

        with pytest.raises(RuntimeError, match="Stream backpressure"):
            await publisher._check_backpressure(MODEL_ID)

    @pytest.mark.asyncio
    async def test_raises_no_consumers_error_when_zero(self) -> None:
        publisher, _, _, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(0, 0))

        with pytest.raises(NoConsumersError) as exc_info:
            await publisher._check_backpressure(MODEL_ID)

        assert exc_info.value.model_id == MODEL_ID

    @pytest.mark.asyncio
    async def test_passes_when_within_limit(self) -> None:
        publisher, _, _, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(5, 100))

        # Should not raise
        await publisher._check_backpressure(MODEL_ID)

    @pytest.mark.asyncio
    async def test_no_consumers_error_has_model_id(self) -> None:
        publisher, _, _, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(0, 0))

        with pytest.raises(NoConsumersError, match="custom/model") as exc_info:
            await publisher._check_backpressure("custom/model")

        assert exc_info.value.model_id == "custom/model"


class TestBuildWorkItem:
    """Tests for WorkPublisher._build_work_item."""

    @pytest.mark.asyncio
    async def test_includes_all_required_fields(self) -> None:
        publisher, *_ = _make_publisher()
        item = {"text": "hello world"}

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=3,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item=item,
        )

        # Required fields
        assert wi["work_item_id"] == "r1.0"
        assert wi["request_id"] == "r1"
        assert wi["item_index"] == 0
        assert wi["total_items"] == 3
        assert wi["operation"] == "encode"
        assert wi["model_id"] == MODEL_ID
        assert wi["profile_id"] == PROFILE_ID
        assert wi["pool_name"] == POOL_NAME
        assert wi["machine_profile"] == MACHINE_PROFILE
        assert wi["router_id"] == ROUTER_ID
        assert wi["reply_subject"] == "_INBOX.test-router.r1"
        assert wi["timestamp"] > 0

        # Item is inline
        assert wi["item"] == item
        assert wi.get("payload_ref") is None

    @pytest.mark.asyncio
    async def test_includes_optional_fields(self) -> None:
        publisher, *_ = _make_publisher()

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=1,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item={"text": "x"},
            output_types=["dense", "sparse"],
            instruction="Retrieve passages",
            is_query=True,
            options={"pooling": "mean"},
        )

        assert wi["output_types"] == ["dense", "sparse"]
        assert wi["instruction"] == "Retrieve passages"
        assert wi["is_query"] is True
        assert wi["options"] == {"pooling": "mean"}

    @pytest.mark.asyncio
    async def test_extract_fields(self) -> None:
        publisher, *_ = _make_publisher()

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=1,
            operation="extract",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item={"text": "x"},
            labels=["PER", "ORG"],
            output_schema={"type": "array"},
        )

        assert wi["labels"] == ["PER", "ORG"]
        assert wi["output_schema"] == {"type": "array"}

    @pytest.mark.asyncio
    async def test_work_item_id_format(self) -> None:
        publisher, *_ = _make_publisher()

        wi, _payload = await publisher._build_work_item(
            request_id="router-1-42",
            item_index=7,
            total_items=10,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item={"text": "x"},
        )

        assert wi["work_item_id"] == "router-1-42.7"


class TestPayloadOffloading:
    """Tests for payload offloading when item exceeds inline threshold."""

    @pytest.mark.asyncio
    async def test_offloads_when_exceeds_threshold(self) -> None:
        store = AsyncMock(spec=PayloadStore)
        publisher, *_ = _make_publisher(payload_store=store)

        # Create an item large enough that the serialized WorkItem exceeds 1MB
        large_item = {"text": "x" * (INLINE_THRESHOLD_BYTES + 1000)}

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=3,
            total_items=5,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item=large_item,
        )

        # Item should be cleared and replaced with a reference
        assert wi["item"] is None
        assert wi["payload_ref"] == "payloads/r1/3"

        # Store.put should have been called
        store.put.assert_called_once()
        call_args = store.put.call_args
        assert call_args[0][0] == "payloads/r1/3"

    @pytest.mark.asyncio
    async def test_no_offload_when_below_threshold(self) -> None:
        store = AsyncMock(spec=PayloadStore)
        publisher, *_ = _make_publisher(payload_store=store)

        small_item = {"text": "small"}

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=1,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item=small_item,
        )

        # Item should remain inline
        assert wi["item"] == small_item
        assert wi.get("payload_ref") is None
        store.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_offload_without_store(self) -> None:
        publisher, *_ = _make_publisher(payload_store=None)

        large_item = {"text": "x" * (INLINE_THRESHOLD_BYTES + 1000)}

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=1,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item=large_item,
        )

        # Item stays inline when no store is configured
        assert wi["item"] == large_item

    @pytest.mark.asyncio
    async def test_no_offload_when_item_is_none(self) -> None:
        store = AsyncMock(spec=PayloadStore)
        publisher, *_ = _make_publisher(payload_store=store)

        wi, _payload = await publisher._build_work_item(
            request_id="r1",
            item_index=0,
            total_items=1,
            operation="encode",
            model_id=MODEL_ID,
            profile_id=PROFILE_ID,
            pool_name=POOL_NAME,
            machine_profile=MACHINE_PROFILE,
            reply_subject="_INBOX.test-router.r1",
            item=None,
        )

        assert wi["item"] is None
        store.put.assert_not_called()


class TestSubmitEncode:
    """Tests for submit_encode end-to-end with mocked NATS."""

    @pytest.mark.asyncio
    async def test_publishes_one_item_per_input(self) -> None:
        publisher, _nc, js, _jsm = _make_publisher()
        items = [{"text": "a"}, {"text": "b"}, {"text": "c"}]

        # We need to simulate results arriving. Do it by manipulating _pending
        # after the publish calls are made.
        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            """Fake _wait_for_results that returns immediately."""
            for i in range(pending.total_items):
                pending.results[i] = _make_work_result(pending.request_id, i)
            pending.completed.set()
            return [pending.results[i] for i in range(pending.total_items)]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            results = await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
            )

        assert js.publish_async.call_count == 3
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_cleanup_payload_store_on_success(self) -> None:
        import asyncio

        store = AsyncMock(spec=PayloadStore)
        publisher, _nc, _js, _jsm = _make_publisher(payload_store=store)

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            for i in range(pending.total_items):
                pending.results[i] = _make_work_result(pending.request_id, i)
            pending.completed.set()
            return [pending.results[i] for i in range(pending.total_items)]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "a"}],
            )

        # Cleanup is fire-and-forget; yield to let the background task run
        await asyncio.sleep(0)

        # delete_prefix should be called for cleanup
        store.delete_prefix.assert_called_once()
        call_arg = store.delete_prefix.call_args[0][0]
        assert call_arg.startswith("payloads/")

    @pytest.mark.asyncio
    async def test_pending_request_cleaned_up(self) -> None:
        publisher, _nc, _js, _jsm = _make_publisher()

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            pending.results[0] = _make_work_result(pending.request_id, 0)
            pending.completed.set()
            return [pending.results[0]]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "a"}],
            )

        # Pending should be cleaned up after completion
        assert len(publisher._pending) == 0


class TestSubmitScore:
    """Tests for submit_score with mocked NATS."""

    @pytest.mark.asyncio
    async def test_publishes_single_work_item(self) -> None:
        publisher, _nc, js, _jsm = _make_publisher()

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            pending.results[0] = _make_work_result(pending.request_id, 0)
            pending.completed.set()
            return [pending.results[0]]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            results = await publisher.submit_score(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                query={"text": "query"},
                items=[{"text": "a"}, {"text": "b"}],
            )

        # Score sends exactly 1 work item (not decomposed)
        assert js.publish.call_count == 1
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_score_work_item_contains_query_and_items(self) -> None:
        publisher, _nc, js, _jsm = _make_publisher()

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            pending.results[0] = _make_work_result(pending.request_id, 0)
            pending.completed.set()
            return [pending.results[0]]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            await publisher.submit_score(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                query={"text": "query"},
                items=[{"text": "a"}, {"text": "b"}],
            )

        # Verify the published payload
        publish_call = js.publish.call_args
        payload = msgpack.unpackb(publish_call[0][1], raw=False)
        assert payload["operation"] == "score"
        assert payload["query_item"] == {"text": "query"}
        assert payload["score_items"] == [{"text": "a"}, {"text": "b"}]
        assert payload["total_items"] == 1


class TestStartStop:
    """Tests for WorkPublisher lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_inbox(self) -> None:
        publisher, nc, _js, _jsm = _make_publisher()

        await publisher.start()

        nc.subscribe.assert_called_once()
        call_args = nc.subscribe.call_args
        subject = call_args[0][0] if call_args[0] else call_args[1].get("subject")
        assert subject == f"_INBOX.{ROUTER_ID}.>"
        assert publisher._running is True

    @pytest.mark.asyncio
    async def test_stop_unsubscribes_and_cancels_pending(self) -> None:
        publisher, nc, _js, _jsm = _make_publisher()

        # Start first
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await publisher.start()

        # Add a pending request
        pending = PendingRequest(
            request_id="test-1",
            total_items=1,
            results={},
            errors={},
        )
        publisher._pending["test-1"] = pending

        await publisher.stop()

        mock_sub.unsubscribe.assert_called_once()
        assert publisher._running is False
        assert len(publisher._pending) == 0
        # Completed event should be set (to unblock waiters)
        assert pending.completed.is_set()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_safe(self) -> None:
        publisher, *_ = _make_publisher()
        await publisher.stop()  # Should not raise
