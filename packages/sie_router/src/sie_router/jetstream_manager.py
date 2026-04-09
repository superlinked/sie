"""NATS JetStream stream and consumer lifecycle management.

Manages creation and lookup of JetStream streams (one per pool) and
consumers (one per bundle+pool combination). Streams and consumers are
created lazily on first use and cached.

This module is only active when ``SIE_CLUSTER_ROUTING=queue``.
"""

from __future__ import annotations

import logging

from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from nats.js.errors import BadRequestError
from sie_sdk.queue_types import (
    WORK_SUBJECT_PREFIX,
    work_consumer_name,
    work_pool_stream_name,
    work_pool_stream_subjects,
    work_stream_name,
)

logger = logging.getLogger(__name__)

# Stream defaults (nats-py expects seconds, converts to nanoseconds internally)
_DEFAULT_MAX_AGE_S = 60  # 60 seconds
_DEFAULT_MAX_MSGS = 100_000
_DEFAULT_REPLICAS = 1  # Override to 3 for production
_DEFAULT_ACK_WAIT_S = 30.0  # 30 seconds
_DEFAULT_MAX_DELIVER = 3
_DEFAULT_MAX_ACK_PENDING = 1000


class JetStreamManager:
    """Manages JetStream streams and consumers for the work queue.

    Streams are created per pool (lazily on first publish/subscribe).
    Consumers are created per (bundle, pool) combination.
    All operations are idempotent — calling create on an existing
    stream/consumer with the same config is a no-op.
    """

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        *,
        replicas: int = _DEFAULT_REPLICAS,
    ) -> None:
        self._nc = nc
        self._js = js
        self._replicas = replicas
        self._known_streams: set[str] = set()
        self._known_consumers: set[str] = set()

    async def ensure_stream(self, model_id: str, pool_name: str = "") -> str:
        """Ensure the pool-level JetStream stream exists.

        Creates ``WORK_POOL_{pool}`` with subject ``sie.work.*.{pool}``
        which captures all models for the pool. If a legacy per-model
        stream overlaps, it is deleted first.

        Args:
            model_id: Model identifier (used for legacy stream cleanup).
            pool_name: Pool name (e.g., ``"l4"``). Required for
                pool-level stream creation.

        Returns:
            Pool stream name (e.g., ``"WORK_POOL_l4"``).
        """
        if not pool_name:
            pool_name = "_default"

        pool_stream = work_pool_stream_name(pool_name)

        if pool_stream in self._known_streams:
            return pool_stream

        # Delete any legacy per-model stream whose subjects overlap
        legacy_stream = work_stream_name(model_id)
        try:
            await self._js.delete_stream(legacy_stream)
            logger.info("Deleted legacy per-model stream: %s", legacy_stream)
            self._known_streams.discard(legacy_stream)
        except Exception:  # noqa: BLE001, S110
            pass  # Stream may not exist

        subjects = work_pool_stream_subjects(pool_name)

        config = StreamConfig(
            name=pool_stream,
            subjects=subjects,
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=_DEFAULT_MAX_AGE_S,
            max_msgs=_DEFAULT_MAX_MSGS,
            storage=StorageType.MEMORY,
            num_replicas=self._replicas,
            discard=DiscardPolicy.NEW,
        )

        try:
            await self._js.add_stream(config)
            logger.info("Pool stream created/verified: %s (subjects=%s)", pool_stream, subjects)
        except BadRequestError as e:
            if e.err_code == 10058:
                logger.info("Pool stream %s already exists (created by worker), reusing", pool_stream)
            else:
                raise
        except Exception:
            logger.exception("Failed to create pool stream: %s", pool_stream)
            raise

        self._known_streams.add(pool_stream)

        return pool_stream

    async def ensure_consumer(
        self,
        model_id: str,
        bundle_id: str,
        pool_name: str,
    ) -> str:
        """Ensure a JetStream consumer exists for the given (bundle, pool).

        Creates a durable pull consumer on the pool-level stream with a
        wildcard filter covering all models for the pool. Idempotent.

        Args:
            model_id: Model identifier (used for stream creation, not for
                consumer filtering).
            bundle_id: Bundle identifier (e.g., ``"default"``).
            pool_name: Pool name (e.g., ``"_default"``).

        Returns:
            Consumer durable name.
        """
        stream_name = await self.ensure_stream(model_id, pool_name=pool_name)
        consumer_name = work_consumer_name(bundle_id, pool_name)
        cache_key = f"{stream_name}/{consumer_name}"

        if cache_key in self._known_consumers:
            return consumer_name

        filter_subject = f"{WORK_SUBJECT_PREFIX}.*.{pool_name}"

        config = ConsumerConfig(
            durable_name=consumer_name,
            filter_subject=filter_subject,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=_DEFAULT_ACK_WAIT_S,
            max_deliver=_DEFAULT_MAX_DELIVER,
            max_ack_pending=_DEFAULT_MAX_ACK_PENDING,
        )

        try:
            await self._js.add_consumer(stream_name, config)
            self._known_consumers.add(cache_key)
            logger.info(
                "JetStream consumer created/verified: %s on stream %s (filter=%s)",
                consumer_name,
                stream_name,
                filter_subject,
            )
        except Exception:
            logger.exception("Failed to create JetStream consumer: %s", consumer_name)
            raise

        return consumer_name

    async def get_pending_count(self, model_id: str, bundle_id: str, pool_name: str) -> int:
        """Get the number of pending (undelivered) messages for a consumer.

        Used for backpressure checks before publishing.

        Args:
            model_id: Model identifier (unused in pool-level design, kept for API compat).
            bundle_id: Bundle identifier.
            pool_name: Pool name.

        Returns:
            Number of pending messages, or 0 if consumer doesn't exist.
        """
        stream_name = work_pool_stream_name(pool_name) if pool_name else work_stream_name(model_id)
        consumer_name = work_consumer_name(bundle_id, pool_name)

        try:
            info = await self._js.consumer_info(stream_name, consumer_name)
            return info.num_pending
        except Exception:  # noqa: BLE001
            return 0

    async def get_stream_pending_count(self, model_id: str, pool_name: str = "") -> int:
        """Get total pending messages across all consumers for a pool's stream.

        Args:
            model_id: Model identifier (unused in pool design, kept for compat).
            pool_name: Pool name for pool-level stream lookup.

        Returns:
            Total message count in the stream.
        """
        stream_name = work_pool_stream_name(pool_name) if pool_name else work_stream_name(model_id)

        try:
            info = await self._js.stream_info(stream_name)
            return info.state.messages
        except Exception:  # noqa: BLE001
            return 0

    async def get_consumer_count(self, model_id: str, pool_name: str = "") -> int:
        """Get the number of consumers on the pool stream.

        Used for zero-consumer fast-fail: if no workers are subscribed,
        publishing would just queue messages with nobody to process them.

        Returns -1 when the stream doesn't exist yet (e.g., first request
        for a model before any worker has started). Callers should treat
        -1 as "unknown" rather than "no consumers" to avoid rejecting
        requests before the stream is lazily created.

        Args:
            model_id: Model identifier (unused in pool design, kept for compat).
            pool_name: Pool name for pool-level stream lookup.

        Returns:
            Number of consumers, or -1 if the stream doesn't exist yet.
        """
        stream_name = work_pool_stream_name(pool_name) if pool_name else work_stream_name(model_id)

        try:
            info = await self._js.stream_info(stream_name)
            return info.state.consumer_count
        except Exception:  # noqa: BLE001
            return -1

    async def get_stream_health(self, model_id: str, pool_name: str = "") -> tuple[int, int]:
        """Get consumer count and pending message count in one RPC.

        Returns:
            Tuple of (consumer_count, pending_messages).
            Returns (-1, 0) if the stream doesn't exist.
        """
        stream_name = work_pool_stream_name(pool_name) if pool_name else work_stream_name(model_id)

        try:
            info = await self._js.stream_info(stream_name)
            return info.state.consumer_count, info.state.messages
        except Exception:  # noqa: BLE001
            return -1, 0

    def clear_caches(self) -> None:
        """Clear stream and consumer caches.

        Call from the NATS reconnect handler. After a NATS restart the
        server-side streams/consumers may no longer exist, so the next
        operation should re-create them. ``add_stream`` / ``add_consumer``
        are idempotent, so re-creating an existing object is harmless.
        """
        self._known_streams.clear()
        self._known_consumers.clear()
        logger.info("JetStream caches cleared (reconnect)")
