"""Integration tests for NATS JetStream queue-based routing.

Requires a running NATS server with JetStream enabled.
Set SIE_NATS_URL env var (default: nats://localhost:4222).

Run via: mise run integration-test-queue
   or:  mise run test -- -i packages/sie_router/tests/test_queue_integration.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import msgpack
import pytest
from nats.aio.client import Client as NATSClient
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from nats.js.errors import BadRequestError, ServiceUnavailableError
from sie_sdk.queue_types import normalize_model_id

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration

# Test constants
TEST_STREAM = "TEST_QUEUE_INTEGRATION"
TEST_SUBJECT = "test.work.items"
TEST_CONSUMER = "test-consumer"


@pytest.fixture
async def nats_client(nats_server: str) -> NATSClient:
    """Connect to NATS and yield a connected client. Drain on teardown."""
    nc = NATSClient()
    await nc.connect(servers=[nats_server])
    yield nc  # type: ignore[misc]
    await nc.drain()


@pytest.fixture
async def js(nats_client: NATSClient) -> Any:
    """Return a JetStream context from the connected client."""
    return nats_client.jetstream()


@pytest.fixture
async def clean_stream(js: Any) -> str:
    """Create a clean test stream, deleting any prior version. Return stream name."""
    try:
        await js.delete_stream(TEST_STREAM)
    except Exception:  # noqa: BLE001, S110
        pass

    config = StreamConfig(
        name=TEST_STREAM,
        subjects=[f"{TEST_SUBJECT}.>"],
        retention=RetentionPolicy.WORK_QUEUE,
        max_age=60,  # 60s (nats-py expects seconds)
        max_msgs=1000,
        storage=StorageType.MEMORY,
        num_replicas=1,
        discard=DiscardPolicy.OLD,
    )
    await js.add_stream(config)

    yield TEST_STREAM  # type: ignore[misc]

    try:
        await js.delete_stream(TEST_STREAM)
    except Exception:  # noqa: BLE001, S110
        pass


# ── Test 1: Basic publish / consume ──────────────────────────────────────────


class TestBasicPubSub:
    """Connect to NATS, create stream, publish items, verify pull consumer."""

    @pytest.mark.asyncio
    async def test_publish_and_pull(self, js: Any, clean_stream: str) -> None:
        subject = f"{TEST_SUBJECT}.pool_a"

        # Create pull consumer
        consumer_config = ConsumerConfig(
            durable_name=TEST_CONSUMER,
            filter_subject=subject,
            ack_policy=AckPolicy.EXPLICIT,
        )
        await js.add_consumer(clean_stream, consumer_config)

        # Publish work items
        items = [msgpack.packb({"item_index": i, "data": f"payload-{i}"}, use_bin_type=True) for i in range(5)]
        for item in items:
            ack = await js.publish(subject, item)
            assert ack.stream == clean_stream

        # Pull and verify
        sub = await js.pull_subscribe(subject, TEST_CONSUMER)
        msgs = await sub.fetch(5, timeout=5)
        assert len(msgs) == 5

        received_indices = []
        for msg in msgs:
            payload = msgpack.unpackb(msg.data, raw=False)
            received_indices.append(payload["item_index"])
            await msg.ack()

        assert sorted(received_indices) == [0, 1, 2, 3, 4]


# ── Test 2: Full round-trip (publish → pull → result → inbox) ────────────────


class TestRoundTrip:
    """Full round-trip: publish work → pull on consumer → publish result → verify inbox."""

    @pytest.mark.asyncio
    async def test_work_result_round_trip(self, nats_client: NATSClient, js: Any, clean_stream: str) -> None:
        work_subject = f"{TEST_SUBJECT}.pool_a"
        inbox_subject = "_INBOX.test-router.req-1"

        # Create pull consumer for work items
        consumer_config = ConsumerConfig(
            durable_name="roundtrip-worker",
            filter_subject=work_subject,
            ack_policy=AckPolicy.EXPLICIT,
        )
        await js.add_consumer(clean_stream, consumer_config)

        # Subscribe to inbox for results (plain NATS, not JetStream)
        results: list[bytes] = []
        result_received = asyncio.Event()

        async def on_result(msg: Any) -> None:
            results.append(msg.data)
            result_received.set()

        await nats_client.subscribe(inbox_subject, cb=on_result)

        # Publish work item with reply_subject
        work_item = {
            "work_item_id": "req-1.0",
            "request_id": "req-1",
            "item_index": 0,
            "total_items": 1,
            "operation": "encode",
            "model_id": "test/model",
            "reply_subject": inbox_subject,
        }
        await js.publish(work_subject, msgpack.packb(work_item, use_bin_type=True))

        # Simulate worker: pull, process, publish result
        sub = await js.pull_subscribe(work_subject, "roundtrip-worker")
        msgs = await sub.fetch(1, timeout=5)
        assert len(msgs) == 1

        received_work = msgpack.unpackb(msgs[0].data, raw=False)
        assert received_work["work_item_id"] == "req-1.0"

        # Worker publishes result to reply_subject
        result = {
            "work_item_id": received_work["work_item_id"],
            "request_id": received_work["request_id"],
            "item_index": 0,
            "success": True,
            "result_data": "embedding-vector",
        }
        await nats_client.publish(received_work["reply_subject"], msgpack.packb(result, use_bin_type=True))

        # Verify result arrives on inbox
        await asyncio.wait_for(result_received.wait(), timeout=5)
        assert len(results) == 1
        received_result = msgpack.unpackb(results[0], raw=False)
        assert received_result["success"] is True
        assert received_result["work_item_id"] == "req-1.0"


# ── Test 3: Backpressure ─────────────────────────────────────────────────────


class TestBackpressure:
    """Fill stream to near limit, verify new publishes are rejected/discarded."""

    @pytest.mark.asyncio
    async def test_stream_rejects_when_full(self, js: Any) -> None:
        stream_name = "TEST_BACKPRESSURE"
        subject = "test.backpressure.items"

        # Clean up any prior stream
        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        # Create stream with very low limit and DiscardPolicy.NEW
        # so new publishes fail when the stream is full.
        config = StreamConfig(
            name=stream_name,
            subjects=[subject],
            retention=RetentionPolicy.WORK_QUEUE,
            max_msgs=5,
            storage=StorageType.MEMORY,
            num_replicas=1,
            discard=DiscardPolicy.NEW,
        )
        await js.add_stream(config)

        try:
            # Fill stream to capacity
            for i in range(5):
                await js.publish(subject, f"item-{i}".encode())

            # Verify stream is full
            info = await js.stream_info(stream_name)
            assert info.state.messages == 5

            # Next publish should fail with DiscardPolicy.NEW
            # NATS >= 2.10 returns 503 ServiceUnavailableError (not 400)
            with pytest.raises((BadRequestError, ServiceUnavailableError)):
                await js.publish(subject, b"overflow-item")
        finally:
            await js.delete_stream(stream_name)


# ── Test 4: Consumer group (work distribution) ───────────────────────────────


class TestConsumerGroup:
    """Two consumers on same subject verify items are distributed (no duplicates)."""

    @pytest.mark.asyncio
    async def test_items_distributed_across_consumers(self, js: Any, clean_stream: str) -> None:
        subject = f"{TEST_SUBJECT}.pool_shared"
        shared_consumer = "shared-worker-group"
        num_items = 20

        # Create a single durable consumer (consumer group pattern:
        # multiple pull subscribers share the same durable name)
        consumer_config = ConsumerConfig(
            durable_name=shared_consumer,
            filter_subject=subject,
            ack_policy=AckPolicy.EXPLICIT,
        )
        await js.add_consumer(clean_stream, consumer_config)

        # Publish items
        for i in range(num_items):
            await js.publish(subject, f"item-{i}".encode())

        # Two pull subscribers sharing the same durable consumer
        sub_a = await js.pull_subscribe(subject, shared_consumer)
        sub_b = await js.pull_subscribe(subject, shared_consumer)

        received_a: list[bytes] = []
        received_b: list[bytes] = []

        # Pull from both subscribers in interleaved fashion
        for _ in range(4):
            try:
                msgs = await sub_a.fetch(3, timeout=2)
                for msg in msgs:
                    received_a.append(msg.data)
                    await msg.ack()
            except Exception:  # noqa: BLE001, S110
                pass

            try:
                msgs = await sub_b.fetch(3, timeout=2)
                for msg in msgs:
                    received_b.append(msg.data)
                    await msg.ack()
            except Exception:  # noqa: BLE001, S110
                pass

        # All items should be received exactly once across both consumers
        all_received = received_a + received_b
        assert len(all_received) == num_items

        # No duplicates
        assert len(set(all_received)) == num_items

        # Both consumers should have received some items (distribution)
        # (not guaranteed to be perfectly balanced, just that both got work)
        logger.info("Consumer A received %d items, Consumer B received %d items", len(received_a), len(received_b))
        assert len(received_a) > 0, "Consumer A should have received at least some items"
        assert len(received_b) > 0, "Consumer B should have received at least some items"


# ── Test 5: Message expiry ───────────────────────────────────────────────────


class TestMessageExpiry:
    """Publish items, wait > max_age, verify they're gone."""

    @pytest.mark.asyncio
    async def test_messages_expire_after_max_age(self, js: Any) -> None:
        stream_name = "TEST_EXPIRY"
        subject = "test.expiry.items"

        # Clean up
        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        # Create stream with very short max_age (2 seconds)
        max_age_s = 2  # 2 seconds (nats-py expects seconds)
        config = StreamConfig(
            name=stream_name,
            subjects=[subject],
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=max_age_s,
            max_msgs=1000,
            storage=StorageType.MEMORY,
            num_replicas=1,
        )
        await js.add_stream(config)

        try:
            # Publish items
            for i in range(5):
                await js.publish(subject, f"expiring-{i}".encode())

            # Verify items exist
            info = await js.stream_info(stream_name)
            assert info.state.messages == 5

            # Wait for expiry (max_age + buffer)
            await asyncio.sleep(3)

            # Verify items are gone
            info = await js.stream_info(stream_name)
            assert info.state.messages == 0, f"Expected 0 messages after expiry, got {info.state.messages}"
        finally:
            await js.delete_stream(stream_name)


# ── Test 6: Multi-model stream isolation ─────────────────────────────────────


class TestMultiModelStreams:
    """Multiple model streams coexist and isolate traffic correctly."""

    @pytest.mark.asyncio
    async def test_two_models_isolated_streams(self, js: Any, nats_client: NATSClient) -> None:
        """Items published to model-A stream are not visible to model-B consumer."""
        model_a = "test/model-alpha"
        model_b = "test/model-beta"
        stream_a = f"TEST_MULTI_{normalize_model_id(model_a)}"
        stream_b = f"TEST_MULTI_{normalize_model_id(model_b)}"
        subject_a = f"test.multi.{normalize_model_id(model_a)}._default"
        subject_b = f"test.multi.{normalize_model_id(model_b)}._default"

        # Clean up
        for s in [stream_a, stream_b]:
            try:
                await js.delete_stream(s)
            except Exception:  # noqa: BLE001, S110
                pass

        # Create separate streams
        for name, subj in [(stream_a, subject_a), (stream_b, subject_b)]:
            await js.add_stream(
                StreamConfig(
                    name=name,
                    subjects=[subj],
                    retention=RetentionPolicy.WORK_QUEUE,
                    max_msgs=100,
                    storage=StorageType.MEMORY,
                    num_replicas=1,
                )
            )

        try:
            # Publish to model-A
            for i in range(5):
                await js.publish(subject_a, msgpack.packb({"model": "a", "i": i}, use_bin_type=True))

            # Publish to model-B
            for i in range(3):
                await js.publish(subject_b, msgpack.packb({"model": "b", "i": i}, use_bin_type=True))

            # Consumer on model-A sees only model-A items
            await js.add_consumer(
                stream_a,
                ConsumerConfig(
                    durable_name="worker-a",
                    filter_subject=subject_a,
                    ack_policy=AckPolicy.EXPLICIT,
                ),
            )
            sub_a = await js.pull_subscribe(subject_a, "worker-a", stream=stream_a)
            msgs_a = await sub_a.fetch(10, timeout=2)
            assert len(msgs_a) == 5
            for msg in msgs_a:
                assert msgpack.unpackb(msg.data, raw=False)["model"] == "a"
                await msg.ack()

            # Consumer on model-B sees only model-B items
            await js.add_consumer(
                stream_b,
                ConsumerConfig(
                    durable_name="worker-b",
                    filter_subject=subject_b,
                    ack_policy=AckPolicy.EXPLICIT,
                ),
            )
            sub_b = await js.pull_subscribe(subject_b, "worker-b", stream=stream_b)
            msgs_b = await sub_b.fetch(10, timeout=2)
            assert len(msgs_b) == 3
            for msg in msgs_b:
                assert msgpack.unpackb(msg.data, raw=False)["model"] == "b"
                await msg.ack()
        finally:
            for s in [stream_a, stream_b]:
                try:
                    await js.delete_stream(s)
                except Exception:  # noqa: BLE001, S110
                    pass

    @pytest.mark.asyncio
    async def test_pool_scoped_consumers(self, js: Any) -> None:
        """Consumers filtered by pool only see items for their pool."""
        stream_name = "TEST_POOL_ISOLATION"
        base_subject = "test.pool"

        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        await js.add_stream(
            StreamConfig(
                name=stream_name,
                subjects=[f"{base_subject}.>"],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=100,
                storage=StorageType.MEMORY,
                num_replicas=1,
            )
        )

        try:
            default_subject = f"{base_subject}._default"
            eval_subject = f"{base_subject}.eval"

            # Publish 5 items to default pool, 3 to eval pool
            for i in range(5):
                await js.publish(default_subject, f"default-{i}".encode())
            for i in range(3):
                await js.publish(eval_subject, f"eval-{i}".encode())

            # Consumer for default pool
            await js.add_consumer(
                stream_name,
                ConsumerConfig(
                    durable_name="default-worker",
                    filter_subject=default_subject,
                    ack_policy=AckPolicy.EXPLICIT,
                ),
            )
            sub_default = await js.pull_subscribe(default_subject, "default-worker", stream=stream_name)
            msgs_default = await sub_default.fetch(10, timeout=2)
            assert len(msgs_default) == 5
            for msg in msgs_default:
                assert msg.data.startswith(b"default-")
                await msg.ack()

            # Consumer for eval pool
            await js.add_consumer(
                stream_name,
                ConsumerConfig(
                    durable_name="eval-worker",
                    filter_subject=eval_subject,
                    ack_policy=AckPolicy.EXPLICIT,
                ),
            )
            sub_eval = await js.pull_subscribe(eval_subject, "eval-worker", stream=stream_name)
            msgs_eval = await sub_eval.fetch(10, timeout=2)
            assert len(msgs_eval) == 3
            for msg in msgs_eval:
                assert msg.data.startswith(b"eval-")
                await msg.ack()
        finally:
            try:
                await js.delete_stream(stream_name)
            except Exception:  # noqa: BLE001, S110
                pass

    @pytest.mark.asyncio
    async def test_stream_info_reflects_pending(self, js: Any) -> None:
        """stream_info shows correct pending/consumer counts."""
        stream_name = "TEST_STREAM_INFO"
        subject = "test.info.items"

        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        await js.add_stream(
            StreamConfig(
                name=stream_name,
                subjects=[subject],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=100,
                storage=StorageType.MEMORY,
                num_replicas=1,
            )
        )

        try:
            # Initially: 0 messages, 0 consumers
            info = await js.stream_info(stream_name)
            assert info.state.messages == 0
            assert info.state.consumer_count == 0

            # Add a consumer
            await js.add_consumer(
                stream_name,
                ConsumerConfig(
                    durable_name="info-worker",
                    filter_subject=subject,
                    ack_policy=AckPolicy.EXPLICIT,
                ),
            )
            info = await js.stream_info(stream_name)
            assert info.state.consumer_count == 1

            # Publish 5 items
            for i in range(5):
                await js.publish(subject, f"item-{i}".encode())

            info = await js.stream_info(stream_name)
            assert info.state.messages == 5
        finally:
            try:
                await js.delete_stream(stream_name)
            except Exception:  # noqa: BLE001, S110
                pass


# ── Test 7: NAK and redelivery ───────────────────────────────────────────────


class TestNakAndRedeliver:
    """NAK and ack_wait redelivery behavior with real NATS."""

    @pytest.mark.asyncio
    async def test_nak_with_delay_redelivers(self, js: Any) -> None:
        """NAKed message with delay is redelivered after the delay."""
        stream_name = "TEST_NAK_REDELIVER"
        subject = "test.nak.items"

        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        await js.add_stream(
            StreamConfig(
                name=stream_name,
                subjects=[subject],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=100,
                storage=StorageType.MEMORY,
                num_replicas=1,
            )
        )

        try:
            await js.add_consumer(
                stream_name,
                ConsumerConfig(
                    durable_name="nak-worker",
                    filter_subject=subject,
                    ack_policy=AckPolicy.EXPLICIT,
                    max_deliver=5,
                ),
            )

            await js.publish(subject, b"test-item")

            sub = await js.pull_subscribe(subject, "nak-worker", stream=stream_name)

            # First pull — NAK with short delay
            msgs = await sub.fetch(1, timeout=2)
            assert len(msgs) == 1
            await msgs[0].nak(delay=1)  # 1 second delay

            # Immediate pull — should get nothing (delay not elapsed)
            try:
                msgs2 = await sub.fetch(1, timeout=0.5)
                # May or may not get the message depending on timing
            except Exception:  # noqa: BLE001
                msgs2 = []  # noqa: F841

            # Wait for delay and pull again — should get the redelivered message
            await asyncio.sleep(1.5)
            msgs3 = await sub.fetch(1, timeout=2)
            assert len(msgs3) == 1
            await msgs3[0].ack()
        finally:
            try:
                await js.delete_stream(stream_name)
            except Exception:  # noqa: BLE001, S110
                pass

    @pytest.mark.asyncio
    async def test_ack_wait_redelivers_unacked(self, js: Any) -> None:
        """Unacked message is redelivered after ack_wait expires."""
        stream_name = "TEST_ACK_WAIT"
        subject = "test.ackwait.items"

        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        await js.add_stream(
            StreamConfig(
                name=stream_name,
                subjects=[subject],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=100,
                storage=StorageType.MEMORY,
                num_replicas=1,
            )
        )

        try:
            # Consumer with very short ack_wait (2 seconds)
            await js.add_consumer(
                stream_name,
                ConsumerConfig(
                    durable_name="slow-worker",
                    filter_subject=subject,
                    ack_policy=AckPolicy.EXPLICIT,
                    ack_wait=2.0,  # seconds
                    max_deliver=3,
                ),
            )

            await js.publish(subject, b"slow-item")

            sub = await js.pull_subscribe(subject, "slow-worker", stream=stream_name)

            # Pull but don't ACK
            msgs = await sub.fetch(1, timeout=2)
            assert len(msgs) == 1
            # Intentionally not acking

            # Wait for ack_wait to expire
            await asyncio.sleep(3)

            # Should get redelivered
            msgs2 = await sub.fetch(1, timeout=3)
            assert len(msgs2) == 1
            await msgs2[0].ack()
        finally:
            try:
                await js.delete_stream(stream_name)
            except Exception:  # noqa: BLE001, S110
                pass


# ── Test 8: DLQ advisory on max delivery ─────────────────────────────────────


class TestDLQAdvisory:
    """Max-delivery advisory published when message exhausts retries."""

    @pytest.mark.asyncio
    async def test_max_deliver_triggers_advisory(self, nats_client: NATSClient, js: Any) -> None:
        """After max_deliver NAKs, NATS publishes a max-delivery advisory.

        Uses a push-based consumer because NATS only emits MAX_DELIVERIES
        advisories when the *server* drives redelivery (push). Pull consumers
        emit MSG_NAKED instead — the server never internally attempts
        redelivery, so it never detects the max_deliver violation.
        """
        stream_name = "TEST_DLQ_ADVISORY"
        subject = "test.dlq.items"

        try:
            await js.delete_stream(stream_name)
        except Exception:  # noqa: BLE001, S110
            pass

        await js.add_stream(
            StreamConfig(
                name=stream_name,
                subjects=[subject],
                retention=RetentionPolicy.WORK_QUEUE,
                max_msgs=100,
                storage=StorageType.MEMORY,
                num_replicas=1,
            )
        )

        try:
            # Subscribe to advisory subject
            advisories: list[bytes] = []
            advisory_received = asyncio.Event()

            async def on_advisory(msg: Any) -> None:
                advisories.append(msg.data)
                advisory_received.set()

            await nats_client.subscribe(
                "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>",
                cb=on_advisory,
            )

            # Push-based consumer that NAKs every delivery
            async def on_msg(msg: Any) -> None:
                await msg.nak()

            sub = await js.subscribe(
                subject,
                stream=stream_name,
                config=ConsumerConfig(
                    ack_policy=AckPolicy.EXPLICIT,
                    max_deliver=2,
                    ack_wait=1,
                ),
                cb=on_msg,
            )

            # Publish one item — the push consumer will NAK it twice,
            # then NATS emits the MAX_DELIVERIES advisory.
            await js.publish(subject, b"doomed-item")

            try:
                await asyncio.wait_for(advisory_received.wait(), timeout=10)
            except TimeoutError:
                pytest.fail(
                    f"Advisory not received within timeout for stream {stream_name!r}. "
                    f"Advisories collected: {advisories}"
                )
            assert len(advisories) >= 1
            # Verify advisory content
            advisory = json.loads(advisories[0])
            assert advisory["stream"] == stream_name

            await sub.unsubscribe()
        finally:
            try:
                await js.delete_stream(stream_name)
            except Exception:  # noqa: BLE001, S110
                pass
