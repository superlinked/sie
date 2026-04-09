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
from sie_server.nats_pull_loop import NatsPullLoop
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
