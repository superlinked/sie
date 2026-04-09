from __future__ import annotations

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

    async def _fake_publish_async(subject: str, payload: bytes, **kwargs: Any) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

    js.publish_async = AsyncMock(side_effect=_fake_publish_async)
    js.publish_async_completed = AsyncMock()

    jsm = AsyncMock(spec=JetStreamManager)
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
    result_data: dict | None = None,
) -> WorkResult:
    """Build a minimal WorkResult dict with optional result_msgpack."""
    result: WorkResult = {
        "work_item_id": f"{request_id}.{item_index}",
        "request_id": request_id,
        "item_index": item_index,
        "success": success,
    }
    if success and result_data is not None:
        result["result_msgpack"] = msgpack.packb(result_data, use_bin_type=True)
    if not success and error:
        result["error"] = error
        result["error_code"] = "test_error"
    return result


class TestEncodeRoundtrip:
    """End-to-end encode: submit → publish → results → response."""

    @pytest.mark.asyncio
    async def test_encode_roundtrip(self) -> None:
        """Submit encode request, simulate worker results, verify ordered response."""
        publisher, _nc, js, _jsm = _make_publisher()

        items = [{"text": "hello"}, {"text": "world"}, {"text": "foo"}]

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            # Simulate results arriving out of order
            for i in [2, 0, 1]:
                pending.results[i] = _make_work_result(
                    pending.request_id,
                    i,
                    result_data={"dense": [float(i)]},
                )
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

        # Results should be ordered by item_index
        assert len(results) == 3
        assert results[0]["item_index"] == 0
        assert results[1]["item_index"] == 1
        assert results[2]["item_index"] == 2

        # Verify 3 work items were published
        assert js.publish_async.call_count == 3


class TestScoreRoundtrip:
    """End-to-end score: single work item with query + items."""

    @pytest.mark.asyncio
    async def test_score_roundtrip(self) -> None:
        """Submit score, verify single work item published with query + items."""
        publisher, _nc, js, _jsm = _make_publisher()

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            pending.results[0] = _make_work_result(
                pending.request_id,
                0,
                result_data=[0.95, 0.42],
            )
            pending.completed.set()
            return [pending.results[0]]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            results = await publisher.submit_score(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                query={"text": "What is ML?"},
                items=[{"text": "ML is AI."}, {"text": "Cooking recipe."}],
            )

        # Score sends exactly 1 work item (score uses js.publish, not publish_async)
        assert js.publish.call_count == 1
        assert len(results) == 1
        assert results[0]["success"] is True

        # Verify published payload contains query + items
        payload = msgpack.unpackb(js.publish.call_args[0][1], raw=False)
        assert payload["operation"] == "score"
        assert payload["query_item"] == {"text": "What is ML?"}
        assert payload["score_items"] == [{"text": "ML is AI."}, {"text": "Cooking recipe."}]


class TestExtractRoundtrip:
    """End-to-end extract: decomposed into per-item work items."""

    @pytest.mark.asyncio
    async def test_extract_roundtrip(self) -> None:
        """Submit extract, verify decomposed work items and results."""
        publisher, _nc, js, _jsm = _make_publisher()

        items = [{"text": "Alice works at Acme."}, {"text": "Bob is from Paris."}]

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            for i in range(pending.total_items):
                pending.results[i] = _make_work_result(
                    pending.request_id,
                    i,
                    result_data={"entities": [{"text": f"entity-{i}", "label": "PER"}]},
                )
            pending.completed.set()
            return [pending.results[i] for i in range(pending.total_items)]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            results = await publisher.submit_extract(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
                labels=["PER", "ORG"],
            )

        # Extract decomposes: 2 work items published
        assert js.publish_async.call_count == 2
        assert len(results) == 2
        assert all(r["success"] for r in results)


class TestPartialFailure:
    """Verify mixed success/failure responses."""

    @pytest.mark.asyncio
    async def test_partial_failure(self) -> None:
        """5 items: 3 succeed and 2 fail. Response contains both."""
        publisher, _nc, _js, _jsm = _make_publisher()

        items = [{"text": f"item-{i}"} for i in range(5)]

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            # Items 0, 2, 4 succeed; items 1, 3 fail
            for i in range(5):
                if i % 2 == 0:
                    pending.results[i] = _make_work_result(
                        pending.request_id,
                        i,
                        result_data={"dense": [float(i)]},
                    )
                else:
                    pending.errors[i] = (f"error for item {i}", "test_error")
            pending.completed.set()
            # Build ordered results (same as _wait_for_results)
            ordered: list[WorkResult] = []
            for i in range(5):
                if i in pending.results:
                    ordered.append(pending.results[i])
                else:
                    error_msg, error_code = pending.errors[i]
                    ordered.append(
                        WorkResult(
                            work_item_id=f"{pending.request_id}.{i}",
                            request_id=pending.request_id,
                            item_index=i,
                            success=False,
                            error=error_msg,
                            error_code=error_code,
                        )
                    )
            return ordered

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            results = await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
            )

        assert len(results) == 5
        # Items 0, 2, 4 succeeded
        assert results[0]["success"] is True
        assert results[2]["success"] is True
        assert results[4]["success"] is True
        # Items 1, 3 failed
        assert results[1]["success"] is False
        assert results[1]["error"] == "error for item 1"
        assert results[3]["success"] is False
        assert results[3]["error"] == "error for item 3"


class TestTimeoutError:
    """Verify timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self) -> None:
        """No results arrive within timeout — TimeoutError raised."""
        publisher, _nc, _js, _jsm = _make_publisher()

        items = [{"text": "hello"}]

        # Don't patch _wait_for_results — let the actual timeout fire
        with pytest.raises(TimeoutError):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
                timeout=0.01,
            )


class TestNoConsumers:
    """Verify NoConsumersError when no workers are subscribed."""

    @pytest.mark.asyncio
    async def test_no_consumers_returns_error(self) -> None:
        """get_stream_health returns (0, 0) — NoConsumersError raised."""
        publisher, _nc, _js, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(0, 0))

        with pytest.raises(NoConsumersError) as exc_info:
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "hello"}],
            )

        assert exc_info.value.model_id == MODEL_ID


class TestBackpressure:
    """Verify backpressure detection."""

    @pytest.mark.asyncio
    async def test_backpressure_returns_error(self) -> None:
        """Pending count exceeds limit — RuntimeError with 'backpressure'."""
        publisher, _nc, _js, jsm = _make_publisher()
        jsm.get_stream_health = AsyncMock(return_value=(2, 999_999))

        with pytest.raises(RuntimeError, match=r"[Bb]ackpressure"):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=[{"text": "hello"}],
            )


class TestDuplicateResults:
    """Verify deduplication of result messages."""

    @pytest.mark.asyncio
    async def test_duplicate_results_deduplicated(self) -> None:
        """Same item_index arrives twice. Only the first result is used."""
        publisher, _nc, _js, _jsm = _make_publisher()
        request_id = "test-router-1"

        pending = PendingRequest(
            request_id=request_id,
            total_items=1,
            results={},
            errors={},
        )
        publisher._pending[request_id] = pending

        # First result
        first_result = _make_work_result(request_id, 0, result_data={"dense": [1.0]})
        first_result["worker_id"] = "worker-1"
        msg1 = MagicMock()
        msg1.data = msgpack.packb(first_result, use_bin_type=True)
        await publisher._on_result(msg1)

        assert 0 in pending.results
        assert pending.completed.is_set()

        # Duplicate result from different worker
        dup_result = _make_work_result(request_id, 0, result_data={"dense": [2.0]})
        dup_result["worker_id"] = "worker-2"
        msg2 = MagicMock()
        msg2.data = msgpack.packb(dup_result, use_bin_type=True)
        await publisher._on_result(msg2)

        # Original result preserved
        assert pending.results[0].get("worker_id") == "worker-1"


class TestPayloadOffloading:
    """Verify large payload offloading to payload store."""

    @pytest.mark.asyncio
    async def test_large_payload_offloaded(self) -> None:
        """Item exceeding inline threshold is offloaded to payload store."""
        store = AsyncMock(spec=PayloadStore)
        publisher, _nc, js, _jsm = _make_publisher(payload_store=store)

        large_item = {"text": "x" * (INLINE_THRESHOLD_BYTES + 1000)}
        items = [large_item]

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            pending.results[0] = _make_work_result(
                pending.request_id,
                0,
                result_data={"dense": [0.1]},
            )
            pending.completed.set()
            return [pending.results[0]]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            await publisher.submit_encode(
                model_id=MODEL_ID,
                profile_id=PROFILE_ID,
                pool_name=POOL_NAME,
                machine_profile=MACHINE_PROFILE,
                items=items,
            )

        # Verify store.put was called for offloading
        store.put.assert_called_once()
        call_args = store.put.call_args
        ref_key = call_args[0][0]
        assert ref_key.startswith("payloads/")

        # Verify the published work item has payload_ref set and item cleared
        published_payload = msgpack.unpackb(js.publish_async.call_args[0][1], raw=False)
        assert published_payload["item"] is None
        assert published_payload["payload_ref"] == ref_key


class TestBundleConfigHash:
    """Verify bundle_config_hash is included in work items."""

    @pytest.mark.asyncio
    async def test_bundle_config_hash_included(self) -> None:
        """Submitted work items include the bundle_config_hash field."""
        publisher, _nc, js, _jsm = _make_publisher()

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
                items=[{"text": "hello"}],
                bundle_config_hash="abc123hash",
            )

        published_payload = msgpack.unpackb(js.publish_async.call_args[0][1], raw=False)
        assert published_payload["bundle_config_hash"] == "abc123hash"

    @pytest.mark.asyncio
    async def test_bundle_config_hash_empty_by_default(self) -> None:
        """When no hash is provided, field defaults to empty string."""
        publisher, _nc, js, _jsm = _make_publisher()

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
                items=[{"text": "hello"}],
            )

        published_payload = msgpack.unpackb(js.publish_async.call_args[0][1], raw=False)
        assert published_payload["bundle_config_hash"] == ""


class TestConcurrentRequests:
    """Verify concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """5 encode requests submitted concurrently complete without interference."""
        publisher, _nc, js, _jsm = _make_publisher()

        async def fake_wait(pending, timeout):  # noqa: ASYNC109
            # Simulate brief processing delay
            await asyncio.sleep(0.001)
            for i in range(pending.total_items):
                pending.results[i] = _make_work_result(
                    pending.request_id,
                    i,
                    result_data={"dense": [float(i)]},
                )
            pending.completed.set()
            return [pending.results[i] for i in range(pending.total_items)]

        with patch.object(publisher, "_wait_for_results", side_effect=fake_wait):
            # Launch 5 concurrent requests
            tasks = [
                publisher.submit_encode(
                    model_id=MODEL_ID,
                    profile_id=PROFILE_ID,
                    pool_name=POOL_NAME,
                    machine_profile=MACHINE_PROFILE,
                    items=[{"text": f"item-{req_idx}"}],
                )
                for req_idx in range(5)
            ]
            all_results = await asyncio.gather(*tasks)

        # All 5 requests should succeed
        assert len(all_results) == 5
        for results in all_results:
            assert len(results) == 1
            assert results[0]["success"] is True

        # All 5 requests should have published work items
        assert js.publish_async.call_count == 5

        # No pending requests should remain
        assert len(publisher._pending) == 0


class TestPayloadCleanup:
    """Verify payload store cleanup fires after results."""

    @pytest.mark.asyncio
    async def test_payload_cleanup_fires(self) -> None:
        """After results are received, payload store cleanup happens as a background task."""
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

        # Yield to let the fire-and-forget background task run
        await asyncio.sleep(0)

        # delete_prefix should be called for cleanup
        store.delete_prefix.assert_called_once()
        call_arg = store.delete_prefix.call_args[0][0]
        assert call_arg.startswith("payloads/")
