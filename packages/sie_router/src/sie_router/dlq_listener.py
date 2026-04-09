from __future__ import annotations

import logging
from typing import Any

import orjson
from nats.aio.client import Client as NATSClient
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import RetentionPolicy, StorageType, StreamConfig
from nats.js.errors import BadRequestError
from sie_sdk.queue_types import DEAD_LETTER_PREFIX

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter

    DLQ_EVENTS_TOTAL = Counter(
        "sie_router_dlq_events_total",
        "Number of work items that exhausted max deliveries",
        ["stream", "consumer"],
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False

# NATS advisory subject for max-delivery events across all consumers
_MAX_DELIVERIES_ADVISORY = "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>"

# Stream config for dead-letter items
_DLQ_STREAM_NAME = "DEAD_LETTERS"
_DLQ_SUBJECTS = [f"{DEAD_LETTER_PREFIX}.>"]
_DLQ_MAX_AGE_S = 86400  # 24 hours in seconds (nats-py converts to nanoseconds)


class DlqListener:
    """Listens for JetStream max-delivery advisories and republishes to a dead-letter stream.

    When a work item exhausts its max_deliver retries, JetStream publishes an
    advisory. This listener captures those advisories, fetches the original
    message from the stream (if still available), and republishes it to
    ``sie.dlq.{model_id_normalized}`` for post-mortem analysis.

    If the original message has already been purged (expired via max_age), the
    advisory metadata is published instead so operators still have visibility.
    """

    def __init__(self, nc: NATSClient, js: JetStreamContext) -> None:
        self._nc = nc
        self._js = js
        self._sub: Any = None
        self._dlq_stream_ensured = False

    async def start(self) -> None:
        """Subscribe to max-delivery advisories."""
        await self._ensure_dlq_stream()
        self._sub = await self._nc.subscribe(
            _MAX_DELIVERIES_ADVISORY,
            cb=self._on_advisory,
        )
        logger.info("DLQ listener started on %s", _MAX_DELIVERIES_ADVISORY)

    async def stop(self) -> None:
        """Unsubscribe from advisories."""
        if self._sub is not None:
            await self._sub.unsubscribe()
            self._sub = None

    async def _ensure_dlq_stream(self) -> None:
        """Create the dead-letter stream if it doesn't exist."""
        if self._dlq_stream_ensured:
            return
        config = StreamConfig(
            name=_DLQ_STREAM_NAME,
            subjects=_DLQ_SUBJECTS,
            retention=RetentionPolicy.LIMITS,
            max_age=_DLQ_MAX_AGE_S,
            storage=StorageType.MEMORY,
            num_replicas=1,
        )
        try:
            await self._js.add_stream(config)
        except BadRequestError as e:
            if e.err_code == 10058:  # stream already exists
                pass
            else:
                raise
        self._dlq_stream_ensured = True
        logger.info("DLQ stream '%s' ensured", _DLQ_STREAM_NAME)

    async def _on_advisory(self, msg: Msg) -> None:
        """Handle a max-delivery advisory event."""
        try:
            advisory = orjson.loads(msg.data)
        except (orjson.JSONDecodeError, ValueError):
            logger.warning("Malformed DLQ advisory: %s", msg.data[:200])
            return

        stream = advisory.get("stream", "unknown")
        consumer = advisory.get("consumer", "unknown")
        stream_seq = advisory.get("stream_seq")
        deliveries = advisory.get("deliveries", 0)

        logger.warning(
            "Work item exhausted max deliveries: stream=%s consumer=%s seq=%s deliveries=%d",
            stream,
            consumer,
            stream_seq,
            deliveries,
        )

        if _HAS_METRICS:
            DLQ_EVENTS_TOTAL.labels(stream=stream, consumer=consumer).inc()

        # Try to fetch the original message and republish to DLQ
        await self._republish_to_dlq(stream, stream_seq, advisory)

    async def _republish_to_dlq(
        self,
        stream: str,
        stream_seq: int | None,
        advisory: dict[str, Any],
    ) -> None:
        """Fetch the original message from the stream and republish to DLQ."""
        if stream_seq is not None:
            try:
                # Attempt to fetch the original message by sequence number
                original = await self._js.get_msg(stream, seq=stream_seq)
                # Derive model_id from the message subject (sie.work.{model}.{pool})
                model_id_normalized = self._model_id_from_subject(original.subject)
                dlq_subject = f"{DEAD_LETTER_PREFIX}.{model_id_normalized}"
                await self._js.publish(dlq_subject, original.data or b"")
                logger.info("Republished dead-letter item to %s (seq=%d)", dlq_subject, stream_seq)
                return
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Could not fetch original message seq=%d from %s (likely expired)",
                    stream_seq,
                    stream,
                )

        # Fallback: publish advisory metadata as the DLQ message.
        # Try to derive model ID from the stream name (e.g. WORK_BAAI__bge-m3
        # → BAAI__bge-m3).  Fall back to "unknown" only if the prefix is absent.
        if stream.startswith("WORK_"):
            model_id_normalized = stream[len("WORK_") :]
        else:
            model_id_normalized = "unknown"
        dlq_subject = f"{DEAD_LETTER_PREFIX}.{model_id_normalized}"
        advisory_bytes = orjson.dumps(advisory)
        try:
            await self._js.publish(dlq_subject, advisory_bytes)
            logger.info("Published advisory metadata to %s (original expired)", dlq_subject)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish to DLQ subject %s", dlq_subject, exc_info=True)

    @staticmethod
    def _model_id_from_subject(subject: str | None) -> str:
        """Extract the normalized model ID from a work subject.

        Work subjects follow the pattern ``sie.work.{model_id}.{pool}``.
        Returns the model_id token, or ``"unknown"`` when the subject is
        missing or malformed.
        """
        if not subject:
            return "unknown"
        parts = subject.split(".")
        if len(parts) >= 3:
            return parts[2]
        return "unknown"
