from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_router.dlq_listener import (
    _DLQ_STREAM_NAME,
    _DLQ_SUBJECTS,
    _MAX_DELIVERIES_ADVISORY,
    DlqListener,
)
from sie_sdk.queue_types import DEAD_LETTER_PREFIX


def _make_listener() -> tuple[DlqListener, AsyncMock, AsyncMock]:
    """Create a DlqListener with mocked NATS/JetStream dependencies."""
    nc = AsyncMock()
    js = AsyncMock()
    listener = DlqListener(nc=nc, js=js)
    return listener, nc, js


def _make_advisory(
    *,
    stream: str = "WORK_BAAI__bge-m3",
    consumer: str = "default__default",
    stream_seq: int = 42,
    deliveries: int = 5,
) -> bytes:
    """Build a simulated max-delivery advisory payload."""
    return json.dumps(
        {
            "stream": stream,
            "consumer": consumer,
            "stream_seq": stream_seq,
            "deliveries": deliveries,
        }
    ).encode()


def _make_msg(data: bytes) -> MagicMock:
    """Build a minimal mock NATS message."""
    msg = MagicMock()
    msg.data = data
    return msg


class TestDlqStreamCreation:
    """Tests for DLQ stream creation on start."""

    @pytest.mark.asyncio
    async def test_dlq_stream_created_on_start(self) -> None:
        listener, nc, js = _make_listener()

        await listener.start()

        js.add_stream.assert_called_once()
        config = js.add_stream.call_args[0][0]
        assert config.name == _DLQ_STREAM_NAME
        assert config.subjects == _DLQ_SUBJECTS
        nc.subscribe.assert_called_once()
        assert nc.subscribe.call_args[0][0] == _MAX_DELIVERIES_ADVISORY

    @pytest.mark.asyncio
    async def test_dlq_stream_creation_failure_propagates(self) -> None:
        """Non-BadRequestError failures in stream creation now propagate."""
        listener, _nc, js = _make_listener()
        js.add_stream = AsyncMock(side_effect=RuntimeError("stream error"))

        with pytest.raises(RuntimeError, match="stream error"):
            await listener.start()

    @pytest.mark.asyncio
    async def test_dlq_stream_already_exists_is_tolerated(self) -> None:
        """BadRequestError with err_code 10058 (stream exists) is not fatal."""
        from nats.js.errors import BadRequestError

        listener, nc, js = _make_listener()
        err = BadRequestError()
        err.err_code = 10058
        js.add_stream = AsyncMock(side_effect=err)

        await listener.start()
        nc.subscribe.assert_called_once()


class TestAdvisoryHandling:
    """Tests for advisory message processing."""

    @pytest.mark.asyncio
    async def test_advisory_triggers_dlq_publish(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        # Mock get_msg to return original message data
        original_data = b'{"work_item_id": "r1.0"}'
        original_msg = MagicMock()
        original_msg.data = original_data
        original_msg.subject = "sie.work.BAAI__bge-m3._default"
        js.get_msg = AsyncMock(return_value=original_msg)

        advisory_data = _make_advisory(stream="WORK_BAAI__bge-m3", stream_seq=42)
        msg = _make_msg(advisory_data)

        await listener._on_advisory(msg)

        expected_subject = f"{DEAD_LETTER_PREFIX}.BAAI__bge-m3"
        js.get_msg.assert_called_once_with("WORK_BAAI__bge-m3", seq=42)
        js.publish.assert_called_once_with(expected_subject, original_data)

    @pytest.mark.asyncio
    async def test_advisory_with_expired_message(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        # get_msg raises (message expired/purged)
        js.get_msg = AsyncMock(side_effect=RuntimeError("not found"))

        advisory_data = _make_advisory(stream="WORK_BAAI__bge-m3", stream_seq=42)
        msg = _make_msg(advisory_data)

        await listener._on_advisory(msg)

        # Should fall back to publishing advisory metadata
        expected_subject = f"{DEAD_LETTER_PREFIX}.BAAI__bge-m3"
        js.publish.assert_called_once()
        call_args = js.publish.call_args
        assert call_args[0][0] == expected_subject
        # The published data should be the advisory JSON
        published = json.loads(call_args[0][1])
        assert published["stream"] == "WORK_BAAI__bge-m3"
        assert published["stream_seq"] == 42

    @pytest.mark.asyncio
    async def test_advisory_without_stream_seq(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        advisory = json.dumps(
            {
                "stream": "WORK_BAAI__bge-m3",
                "consumer": "default__default",
                "deliveries": 5,
            }
        ).encode()
        msg = _make_msg(advisory)

        await listener._on_advisory(msg)

        # Should skip get_msg and publish advisory metadata directly
        js.get_msg.assert_not_called()
        js.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_malformed_advisory_ignored(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        msg = _make_msg(b"not valid json {{{")

        # Should not raise
        await listener._on_advisory(msg)

        js.get_msg.assert_not_called()
        js.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_work_stream_uses_subject_model_id(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        original_msg = MagicMock()
        original_msg.data = b"payload"
        original_msg.subject = "sie.work.custom_model._default"
        js.get_msg = AsyncMock(return_value=original_msg)

        advisory_data = _make_advisory(stream="CUSTOM_STREAM", stream_seq=1)
        msg = _make_msg(advisory_data)

        await listener._on_advisory(msg)

        expected_subject = f"{DEAD_LETTER_PREFIX}.custom_model"
        js.publish.assert_called_once_with(expected_subject, b"payload")

    @pytest.mark.asyncio
    async def test_pool_level_stream_extracts_model_from_subject(self) -> None:
        """Pool-level streams (WORK_POOL_l4) should derive model_id from the message subject."""
        listener, _nc, js = _make_listener()
        await listener.start()

        original_msg = MagicMock()
        original_msg.data = b"payload"
        original_msg.subject = "sie.work.BAAI__bge-m3.l4"
        js.get_msg = AsyncMock(return_value=original_msg)

        advisory_data = _make_advisory(stream="WORK_POOL_l4", stream_seq=10)
        msg = _make_msg(advisory_data)

        await listener._on_advisory(msg)

        expected_subject = f"{DEAD_LETTER_PREFIX}.BAAI__bge-m3"
        js.get_msg.assert_called_once_with("WORK_POOL_l4", seq=10)
        js.publish.assert_called_once_with(expected_subject, b"payload")


class TestMetrics:
    """Tests for Prometheus metrics."""

    @pytest.mark.asyncio
    async def test_metrics_incremented(self) -> None:
        listener, _nc, js = _make_listener()
        await listener.start()

        original_msg = MagicMock()
        original_msg.data = b"payload"
        original_msg.subject = "sie.work.test._default"
        js.get_msg = AsyncMock(return_value=original_msg)

        with (
            patch("sie_router.dlq_listener._HAS_METRICS", True),
            patch("sie_router.dlq_listener.DLQ_EVENTS_TOTAL") as mock_counter,
        ):
            mock_labels = MagicMock()
            mock_counter.labels.return_value = mock_labels

            advisory_data = _make_advisory(stream="WORK_test", consumer="cons1")
            msg = _make_msg(advisory_data)

            await listener._on_advisory(msg)

            mock_counter.labels.assert_called_once_with(stream="WORK_test", consumer="cons1")
            mock_labels.inc.assert_called_once()


class TestStopListener:
    """Tests for listener lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        listener, nc, _js = _make_listener()
        mock_sub = AsyncMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)

        await listener.start()
        await listener.stop()

        mock_sub.unsubscribe.assert_called_once()
        assert listener._sub is None

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self) -> None:
        listener, _nc, _js = _make_listener()

        # Should not raise
        await listener.stop()
