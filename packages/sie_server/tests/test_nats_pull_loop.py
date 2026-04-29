from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.nats_pull_loop import _OOM_NAK_DELAY_S, NatsPullLoop
from sie_server.types.inputs import Item


def _make_work_item(**overrides) -> dict:
    """Build a minimal WorkItem dict with required fields."""
    wi = {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "encode",
        "model_id": "test/model",
        "profile_id": "default",
        "pool_name": "_default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": time.time(),
    }
    wi.update(overrides)
    return wi


def _make_loop(
    *,
    registry: MagicMock | None = None,
    nc: AsyncMock | None = None,
    js: AsyncMock | None = None,
    payload_store_url: str | None = None,
) -> NatsPullLoop:
    """Create a NatsPullLoop with mocked dependencies."""
    if nc is None:
        nc = AsyncMock()
    if js is None:
        js = AsyncMock()
    if registry is None:
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
    return NatsPullLoop(
        nc=nc,
        js=js,
        registry=registry,
        bundle_id="default",
        pool_name="_default",
        payload_store_url=payload_store_url,
    )


def _make_msg(wi: dict) -> AsyncMock:
    """Wrap a WorkItem dict into a mock NATS message with serialized data."""
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    return msg


def _published_result(nc_mock: AsyncMock) -> dict:
    """Extract the first published WorkResult from the mock nc.publish calls."""
    nc_mock.publish.assert_called_once()
    _subject, data = nc_mock.publish.call_args.args
    return msgpack.unpackb(data, raw=False)


class TestProcessEncodeItem:
    """Verify encode item goes through EncodePipeline.run_encode, result published."""

    @pytest.mark.asyncio
    async def test_process_encode_item(self) -> None:
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="encode",
            item={"text": "hello world"},
            output_types=["dense"],
        )
        msg = _make_msg(wi)

        fake_output = [{"dense": [0.1, 0.2]}]
        fake_timing = RequestTiming()

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_output, fake_timing),
        ) as mock_encode:
            await loop._process_messages("test/model", [msg])

            mock_encode.assert_called_once()
            call_kwargs = mock_encode.call_args.kwargs
            assert call_kwargs["model"] == "test/model"
            assert call_kwargs["output_types"] == ["dense"]
            assert call_kwargs["registry"] is registry
            # Batch API passes a list of Item objects
            assert len(call_kwargs["items"]) == 1
            assert call_kwargs["items"][0].text == "hello world"

        result = _published_result(nc)
        assert result["success"] is True
        assert result["work_item_id"] == "req-1.0"
        # The inner result_msgpack contains the formatted_output[0]
        inner = msgpack.unpackb(result["result_msgpack"], raw=False)
        assert inner == {"dense": [0.1, 0.2]}
        msg.ack.assert_awaited_once()


class TestProcessScoreItem:
    """Verify score item skips _resolve_item, calls worker.submit_score."""

    @pytest.mark.asyncio
    async def test_process_score_item(self) -> None:
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        # Create mock worker with submit_score returning an awaitable future
        mock_worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.95, 0.42], dtype=np.float32))
        worker_result = WorkerResult(output=score_output, timing=RequestTiming())
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(worker_result)
        mock_worker.submit_score = AsyncMock(return_value=future)
        registry.start_worker = AsyncMock(return_value=mock_worker)

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="score",
            query_item={"text": "What is ML?"},
            score_items=[{"text": "ML is AI."}, {"text": "Cooking recipe."}],
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        # Verify submit_score was called
        mock_worker.submit_score.assert_awaited_once()
        call_kwargs = mock_worker.submit_score.call_args.kwargs
        assert isinstance(call_kwargs["query"], Item)
        assert call_kwargs["query"].text == "What is ML?"
        assert len(call_kwargs["items"]) == 2

        # Verify result published
        result = _published_result(nc)
        assert result["success"] is True
        scores = msgpack.unpackb(result["result_msgpack"], raw=False)
        assert len(scores) == 2
        assert abs(scores[0]["score"] - 0.95) < 0.01
        assert abs(scores[1]["score"] - 0.42) < 0.01
        msg.ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_score_item_no_resolve_item(self) -> None:
        """Score work items with query_item/score_items DON'T go through _resolve_item."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        mock_worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.5], dtype=np.float32))
        worker_result = WorkerResult(output=score_output, timing=RequestTiming())
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(worker_result)
        mock_worker.submit_score = AsyncMock(return_value=future)
        registry.start_worker = AsyncMock(return_value=mock_worker)

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="score",
            query_item={"text": "query"},
            score_items=[{"text": "doc"}],
            # Note: no "item" field — _resolve_item would fail for score
            # if it were called, since there's no item/payload_ref.
        )
        msg = _make_msg(wi)

        with patch.object(loop, "_resolve_item", new_callable=AsyncMock) as mock_resolve:
            await loop._process_messages("test/model", [msg])
            mock_resolve.assert_not_called()

        # Score should still succeed
        result = _published_result(nc)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_score_with_payload_ref(self) -> None:
        """Score with offloaded payload fetches from store."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        mock_worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.8], dtype=np.float32))
        worker_result = WorkerResult(output=score_output, timing=RequestTiming())
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(worker_result)
        mock_worker.submit_score = AsyncMock(return_value=future)
        registry.start_worker = AsyncMock(return_value=mock_worker)

        loop = _make_loop(nc=nc, registry=registry)

        # Set up payload store mock
        payload_data = msgpack.packb(
            {
                "query": {"text": "offloaded query"},
                "items": [{"text": "offloaded doc"}],
            },
            use_bin_type=True,
        )
        mock_store = AsyncMock()
        mock_store.get = AsyncMock(return_value=payload_data)
        loop._payload_store = mock_store

        wi = _make_work_item(
            operation="score",
            # No inline query_item/score_items — use payload ref
            query_payload_ref="payloads/abc123",
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        # Verify payload was fetched
        mock_store.get.assert_awaited_once_with("payloads/abc123")

        # Verify submit_score received resolved items
        call_kwargs = mock_worker.submit_score.call_args.kwargs
        assert call_kwargs["query"].text == "offloaded query"
        assert len(call_kwargs["items"]) == 1
        assert call_kwargs["items"][0].text == "offloaded doc"

        result = _published_result(nc)
        assert result["success"] is True


class TestProcessExtractItem:
    """Verify extract item calls worker.submit_extract, result published."""

    @pytest.mark.asyncio
    async def test_process_extract_item(self) -> None:
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        mock_worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        worker_result = WorkerResult(output=extract_output, timing=RequestTiming())
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(worker_result)
        mock_worker.submit_extract = AsyncMock(return_value=future)
        registry.start_worker = AsyncMock(return_value=mock_worker)

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="extract",
            item={"text": "Alice works at Acme."},
            labels=["person", "organization"],
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        # Verify submit_extract was called
        mock_worker.submit_extract.assert_awaited_once()
        call_kwargs = mock_worker.submit_extract.call_args.kwargs
        assert len(call_kwargs["items"]) == 1
        assert call_kwargs["items"][0].text == "Alice works at Acme."
        assert call_kwargs["labels"] == ["person", "organization"]

        # Verify result published
        result = _published_result(nc)
        assert result["success"] is True
        inner = msgpack.unpackb(result["result_msgpack"], raw=False)
        assert "entities" in inner
        msg.ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_extract_item_with_document(self) -> None:
        """Document items round-trip through the queue and surface structured data."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()

        mock_worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[]],
            data=[{"document": {"pages": [{"text": "hello"}]}}],
        )
        worker_result = WorkerResult(output=extract_output, timing=RequestTiming())
        future: asyncio.Future[WorkerResult] = asyncio.Future()
        future.set_result(worker_result)
        mock_worker.submit_extract = AsyncMock(return_value=future)
        registry.start_worker = AsyncMock(return_value=mock_worker)

        loop = _make_loop(nc=nc, registry=registry)

        document_bytes = b"%PDF-1.4 fake content"
        wi = _make_work_item(
            operation="extract",
            item={"document": {"data": document_bytes, "format": "pdf"}},
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        mock_worker.submit_extract.assert_awaited_once()
        call_kwargs = mock_worker.submit_extract.call_args.kwargs
        assert call_kwargs["items"][0].document == {"data": document_bytes, "format": "pdf"}
        prepared = call_kwargs["prepared_items"]
        assert len(prepared) == 1
        assert prepared[0].cost == len(document_bytes)

        result = _published_result(nc)
        assert result["success"] is True
        inner = msgpack.unpackb(result["result_msgpack"], raw=False)
        assert inner["data"] == {"document": {"pages": [{"text": "hello"}]}}
        msg.ack.assert_awaited_once()


class TestErrorPaths:
    """Verify error conditions publish error results."""

    @pytest.mark.asyncio
    async def test_unknown_operation_publishes_error(self) -> None:
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="summarize",
            item={"text": "some text"},
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        result = _published_result(nc)
        assert result["success"] is False
        assert result["error_code"] == "unknown_operation"
        assert "summarize" in result["error"]
        msg.ack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_oom_during_encode_naks_without_publishing(self) -> None:
        """OOM in the encode pipeline must NAK and skip the error publish.

        Contract:

        1. **No** ``WorkResult`` is published to the reply subject for
           ``RESOURCE_EXHAUSTED``. The gateway aggregates per-item replies
           into a per-request collector; once an item's slot is filled,
           the request_id is removed from ``pending_results`` and any
           later reply (e.g. from the redelivered work) is silently
           dropped at the inbox. Publishing the error here would defeat
           the redelivery — the gateway would already have responded to
           the client by the time JetStream redelivers, so the work
           done by the second worker is wasted compute.
        2. The JetStream message is NAKed, not ACKed. NAK with
           ``_OOM_NAK_DELAY_S`` lets JetStream redeliver the work item —
           potentially to a sibling worker, or to this worker after its
           memory pressure clears. The gateway's existing result-await
           timeout still synthesises a 503 ``MODEL_LOADING`` /
           ``RESOURCE_EXHAUSTED`` response for the original client, and
           the SDK auto-retries; the NAK gives non-retrying clients
           (``max_oom_retries=0``) a real second attempt.
        """
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="encode",
            item={"text": "hello world"},
            output_types=["dense"],
        )
        msg = _make_msg(wi)

        oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=oom,
        ):
            await loop._process_messages("test/model", [msg])

        # No reply published — the redelivery via NAK is the recovery path.
        nc.publish.assert_not_called()
        # Critical: delayed NAK for redelivery, not ACK. The delay must
        # match _OOM_NAK_DELAY_S so JetStream redelivers after the
        # configured backoff (gives the worker time to recover memory
        # or another worker time to take over).
        msg.nak.assert_awaited_once_with(delay=_OOM_NAK_DELAY_S)
        msg.ack.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_oom_during_encode_multi_item_batch_silently_naks_all(self) -> None:
        """A unanimous-OOM multi-item batch must NAK every item with no replies.

        Regression guard for the gateway-awaiter contract: a sub-batch of N
        items hits OOM and the worker must skip ``_publish_error`` for
        every item AND NAK every JetStream message. Even one stray
        ``_publish_error`` would fill the gateway's per-item slot and let
        the request collector complete on the partial set, dropping the
        redelivered work for the remaining items.
        """
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        msgs = []
        for idx in range(3):
            wi = _make_work_item(
                work_item_id=f"req-1.{idx}",
                item_index=idx,
                total_items=3,
                operation="encode",
                item={"text": f"hello {idx}"},
                output_types=["dense"],
            )
            msgs.append(_make_msg(wi))

        oom = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=oom,
        ):
            await loop._process_messages("test/model", msgs)

        # No reply published for any item — the gateway will synthesize a
        # 503 from its own result-await timeout once the collector ages out.
        nc.publish.assert_not_called()
        # Every JetStream message NAKed exactly once for redelivery,
        # with the OOM-specific delay so JetStream waits before
        # redelivering (sibling workers or this worker after recovery).
        for msg in msgs:
            msg.nak.assert_awaited_once_with(delay=_OOM_NAK_DELAY_S)
            msg.ack.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_oom_inference_error_keeps_legacy_code_and_acks(self) -> None:
        """Non-OOM inference exceptions still emit the legacy literal AND ACK.

        Regression guard: the OOM classifier must not swallow other
        failure modes (shape errors, adapter bugs, …). And — crucially —
        non-retryable errors are ACKed so JetStream doesn't redeliver
        the same broken request indefinitely.
        """
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="encode",
            item={"text": "hello"},
            output_types=["dense"],
        )
        msg = _make_msg(wi)

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            side_effect=ValueError("bad shape"),
        ):
            await loop._process_messages("test/model", [msg])

        result = _published_result(nc)
        assert result["success"] is False
        assert result["error_code"] == "inference_error"
        # Non-retryable: ACK so the broken request doesn't redeliver.
        msg.ack.assert_awaited_once()
        msg.nak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_model_not_found_naks_for_redelivery(self) -> None:
        """When start_worker raises KeyError (model evicted), items are NAKed."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(side_effect=KeyError("test/model"))

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="score",
            query_item={"text": "hello"},
            score_items=[{"text": "world"}],
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        # Item should be NAKed (not ACKed with error) for redelivery
        msg.nak.assert_awaited_once()
        msg.ack.assert_not_awaited()
        # No result published — item goes back to queue
        nc.publish.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_config_key_error_naks_for_redelivery(self) -> None:
        """When get_config raises KeyError (model evicted), items are NAKed."""
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.side_effect = KeyError("test/model")
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        wi = _make_work_item(
            operation="encode",
            item={"text": "hello"},
        )
        msg = _make_msg(wi)

        await loop._process_messages("test/model", [msg])

        # Item should be NAKed (not ACKed with error) for redelivery
        msg.nak.assert_awaited_once()
        msg.ack.assert_not_awaited()
        nc.publish.assert_not_awaited()


class TestBatchProcessing:
    """Verify _process_messages processes all messages in a batch."""

    @pytest.mark.asyncio
    async def test_batch_processes_all_messages(self) -> None:
        nc = AsyncMock()
        registry = MagicMock()
        registry.model_names = ["test/model"]
        registry.get_config.return_value = MagicMock()
        registry.start_worker = AsyncMock(return_value=MagicMock())

        loop = _make_loop(nc=nc, registry=registry)

        fake_timing = RequestTiming()

        # Create 5 encode messages
        messages = []
        fake_outputs = []
        for i in range(5):
            wi = _make_work_item(
                work_item_id=f"req-{i}.0",
                item={"text": f"text {i}"},
                output_types=["dense"],
            )
            messages.append(_make_msg(wi))
            fake_outputs.append({"dense": [float(i)]})

        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=(fake_outputs, fake_timing),
        ):
            await loop._process_messages("test/model", messages)

        # All 5 messages should be acked
        assert sum(1 for m in messages if m.ack.await_count > 0) == 5
        # All 5 results should be published
        assert nc.publish.await_count == 5
