from unittest.mock import AsyncMock, MagicMock

import pytest
from sie_router.jetstream_manager import JetStreamManager
from sie_sdk.queue_types import work_consumer_name, work_pool_stream_name, work_stream_name

MODEL_ID = "BAAI/bge-m3"
BUNDLE_ID = "default"
POOL_NAME = "_default"


def _make_manager() -> tuple[JetStreamManager, AsyncMock, AsyncMock]:
    """Create a JetStreamManager with mocked NATS client and JetStream context."""
    nc = AsyncMock()
    js = AsyncMock()
    manager = JetStreamManager(nc, js)
    return manager, nc, js


class TestEnsureStream:
    """Tests for JetStreamManager.ensure_stream (pool-level streams)."""

    @pytest.mark.asyncio
    async def test_creates_pool_stream_on_first_call(self) -> None:
        manager, _nc, js = _make_manager()

        stream_name = await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)

        assert stream_name == work_pool_stream_name(POOL_NAME)
        # delete_stream for legacy per-model stream + add_stream for pool stream
        js.add_stream.assert_called_once()
        config = js.add_stream.call_args[0][0]
        assert config.name == work_pool_stream_name(POOL_NAME)

    @pytest.mark.asyncio
    async def test_caches_on_second_call(self) -> None:
        manager, _nc, js = _make_manager()

        name1 = await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)
        name2 = await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)

        assert name1 == name2
        # add_stream should only be called once (cached on second call)
        js.add_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_pools_create_different_streams(self) -> None:
        manager, _nc, js = _make_manager()

        name1 = await manager.ensure_stream("BAAI/bge-m3", pool_name="l4")
        name2 = await manager.ensure_stream("intfloat/e5-base-v2", pool_name="cpu")

        assert name1 != name2
        assert js.add_stream.call_count == 2

    @pytest.mark.asyncio
    async def test_same_pool_different_models_same_stream(self) -> None:
        """Multiple models in the same pool share a single stream."""
        manager, _nc, js = _make_manager()

        name1 = await manager.ensure_stream("BAAI/bge-m3", pool_name="l4")
        name2 = await manager.ensure_stream("intfloat/e5-base-v2", pool_name="l4")

        assert name1 == name2
        # Only one add_stream call (second is cached)
        js.add_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_propagates_add_stream_exception(self) -> None:
        manager, _nc, js = _make_manager()
        js.add_stream = AsyncMock(side_effect=RuntimeError("NATS down"))

        with pytest.raises(RuntimeError, match="NATS down"):
            await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)

        # Stream should NOT be cached on failure
        assert work_pool_stream_name(POOL_NAME) not in manager._known_streams

    @pytest.mark.asyncio
    async def test_stream_config_uses_work_queue_retention(self) -> None:
        manager, _nc, js = _make_manager()

        await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)

        config = js.add_stream.call_args[0][0]
        from nats.js.api import RetentionPolicy, StorageType

        assert config.retention == RetentionPolicy.WORK_QUEUE
        assert config.storage == StorageType.MEMORY

    @pytest.mark.asyncio
    async def test_deletes_legacy_per_model_stream(self) -> None:
        """ensure_stream deletes overlapping per-model stream before creating pool stream."""
        manager, _nc, js = _make_manager()

        await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)

        # Should attempt to delete the legacy per-model stream
        js.delete_stream.assert_called_once_with(work_stream_name(MODEL_ID))


class TestEnsureConsumer:
    """Tests for JetStreamManager.ensure_consumer."""

    @pytest.mark.asyncio
    async def test_creates_consumer_and_caches(self) -> None:
        manager, _nc, js = _make_manager()

        consumer_name = await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        assert consumer_name == work_consumer_name(BUNDLE_ID, POOL_NAME)
        # ensure_stream should be called first
        js.add_stream.assert_called_once()
        # Then add_consumer
        js.add_consumer.assert_called_once()

    @pytest.mark.asyncio
    async def test_caches_consumer_on_second_call(self) -> None:
        manager, _nc, js = _make_manager()

        name1 = await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)
        name2 = await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        assert name1 == name2
        js.add_consumer.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_pools_create_different_consumers(self) -> None:
        manager, _nc, js = _make_manager()

        name1 = await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, "_default")
        name2 = await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, "eval-pool")

        assert name1 != name2
        assert js.add_consumer.call_count == 2

    @pytest.mark.asyncio
    async def test_consumer_config_has_explicit_ack(self) -> None:
        manager, _nc, js = _make_manager()

        await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        stream_name = js.add_consumer.call_args[0][0]
        config = js.add_consumer.call_args[0][1]
        assert stream_name == work_pool_stream_name(POOL_NAME)
        assert config.durable_name == work_consumer_name(BUNDLE_ID, POOL_NAME)

        from nats.js.api import AckPolicy

        assert config.ack_policy == AckPolicy.EXPLICIT

    @pytest.mark.asyncio
    async def test_consumer_filter_subject_uses_wildcard(self) -> None:
        """Consumer filter_subject uses pool-level wildcard, not model-specific subject."""
        manager, _nc, js = _make_manager()

        await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        config = js.add_consumer.call_args[0][1]
        # Must use wildcard (*.pool) — NOT model-specific (BAAI__bge-m3.pool)
        expected_filter = f"sie.work.*.{POOL_NAME}"
        assert config.filter_subject == expected_filter
        # Verify it does NOT contain the normalized model id
        assert "BAAI__bge-m3" not in config.filter_subject

    @pytest.mark.asyncio
    async def test_propagates_add_consumer_exception(self) -> None:
        manager, _nc, js = _make_manager()
        js.add_consumer = AsyncMock(side_effect=RuntimeError("consumer error"))

        with pytest.raises(RuntimeError, match="consumer error"):
            await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        # Consumer should NOT be cached on failure
        cache_key = f"{work_pool_stream_name(POOL_NAME)}/{work_consumer_name(BUNDLE_ID, POOL_NAME)}"
        assert cache_key not in manager._known_consumers


class TestGetPendingCount:
    """Tests for JetStreamManager.get_pending_count."""

    @pytest.mark.asyncio
    async def test_returns_count_from_consumer_info(self) -> None:
        manager, _nc, js = _make_manager()

        mock_info = MagicMock()
        mock_info.num_pending = 42
        js.consumer_info = AsyncMock(return_value=mock_info)

        count = await manager.get_pending_count(MODEL_ID, BUNDLE_ID, POOL_NAME)

        assert count == 42
        js.consumer_info.assert_called_once_with(
            work_pool_stream_name(POOL_NAME),
            work_consumer_name(BUNDLE_ID, POOL_NAME),
        )

    @pytest.mark.asyncio
    async def test_returns_zero_when_consumer_not_found(self) -> None:
        manager, _nc, js = _make_manager()
        js.consumer_info = AsyncMock(side_effect=Exception("consumer not found"))

        count = await manager.get_pending_count(MODEL_ID, BUNDLE_ID, POOL_NAME)

        assert count == 0


class TestGetConsumerCount:
    """Tests for JetStreamManager.get_consumer_count."""

    @pytest.mark.asyncio
    async def test_returns_count_from_stream_info(self) -> None:
        manager, _nc, js = _make_manager()

        mock_info = MagicMock()
        mock_info.state.consumer_count = 5
        js.stream_info = AsyncMock(return_value=mock_info)

        count = await manager.get_consumer_count(MODEL_ID, pool_name=POOL_NAME)

        assert count == 5
        js.stream_info.assert_called_once_with(work_pool_stream_name(POOL_NAME))

    @pytest.mark.asyncio
    async def test_returns_negative_one_when_stream_doesnt_exist(self) -> None:
        manager, _nc, js = _make_manager()
        js.stream_info = AsyncMock(side_effect=Exception("stream not found"))

        count = await manager.get_consumer_count(MODEL_ID)

        assert count == -1

    @pytest.mark.asyncio
    async def test_falls_back_to_per_model_when_no_pool(self) -> None:
        """Without pool_name, falls back to legacy per-model stream lookup."""
        manager, _nc, js = _make_manager()

        mock_info = MagicMock()
        mock_info.state.consumer_count = 2
        js.stream_info = AsyncMock(return_value=mock_info)

        count = await manager.get_consumer_count(MODEL_ID)

        assert count == 2
        js.stream_info.assert_called_once_with(work_stream_name(MODEL_ID))


class TestGetStreamPendingCount:
    """Tests for JetStreamManager.get_stream_pending_count."""

    @pytest.mark.asyncio
    async def test_returns_message_count_from_stream_info(self) -> None:
        manager, _nc, js = _make_manager()

        mock_info = MagicMock()
        mock_info.state.messages = 1234
        js.stream_info = AsyncMock(return_value=mock_info)

        count = await manager.get_stream_pending_count(MODEL_ID)

        assert count == 1234

    @pytest.mark.asyncio
    async def test_returns_zero_when_stream_doesnt_exist(self) -> None:
        manager, _nc, js = _make_manager()
        js.stream_info = AsyncMock(side_effect=Exception("stream not found"))

        count = await manager.get_stream_pending_count(MODEL_ID)

        assert count == 0


class TestGetStreamHealth:
    """Tests for JetStreamManager.get_stream_health."""

    @pytest.mark.asyncio
    async def test_returns_consumer_count_and_messages(self) -> None:
        manager, _nc, js = _make_manager()

        mock_info = MagicMock()
        mock_info.state.consumer_count = 3
        mock_info.state.messages = 42
        js.stream_info = AsyncMock(return_value=mock_info)

        consumer_count, pending = await manager.get_stream_health(MODEL_ID, pool_name=POOL_NAME)

        assert consumer_count == 3
        assert pending == 42
        js.stream_info.assert_called_once_with(work_pool_stream_name(POOL_NAME))

    @pytest.mark.asyncio
    async def test_returns_negative_one_zero_when_stream_doesnt_exist(self) -> None:
        manager, _nc, js = _make_manager()
        js.stream_info = AsyncMock(side_effect=Exception("stream not found"))

        consumer_count, pending = await manager.get_stream_health(MODEL_ID)

        assert consumer_count == -1
        assert pending == 0


class TestClearCaches:
    """Tests for JetStreamManager.clear_caches."""

    @pytest.mark.asyncio
    async def test_clears_both_caches(self) -> None:
        manager, _nc, _js = _make_manager()

        # Populate caches
        await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)
        await manager.ensure_consumer(MODEL_ID, BUNDLE_ID, POOL_NAME)

        assert len(manager._known_streams) > 0
        assert len(manager._known_consumers) > 0

        manager.clear_caches()

        assert len(manager._known_streams) == 0
        assert len(manager._known_consumers) == 0

    @pytest.mark.asyncio
    async def test_after_clear_recreates_on_next_call(self) -> None:
        manager, _nc, js = _make_manager()

        # First call populates cache
        await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)
        assert js.add_stream.call_count == 1

        # Clear caches
        manager.clear_caches()

        # Second call should re-create (cache was cleared)
        await manager.ensure_stream(MODEL_ID, pool_name=POOL_NAME)
        assert js.add_stream.call_count == 2

    def test_clear_on_empty_caches_is_safe(self) -> None:
        manager, _nc, _js = _make_manager()
        manager.clear_caches()  # Should not raise
        assert len(manager._known_streams) == 0
        assert len(manager._known_consumers) == 0
