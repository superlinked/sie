import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_router.nats_manager import _ALL_SUBJECT, NatsManager


class TestNatsManagerConnect:
    """Tests for NATS connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_failure_is_graceful(self) -> None:
        """NATS connect failure doesn't crash — sets connected=False."""
        manager = NatsManager(nats_url="nats://nonexistent:4222")
        # Patch nats.connect to raise
        with patch("nats.connect", side_effect=ConnectionRefusedError("refused")):
            await manager.connect()
        assert not manager.connected

    @pytest.mark.asyncio
    async def test_connected_false_before_connect(self) -> None:
        """Manager starts disconnected."""
        manager = NatsManager()
        assert not manager.connected

    @pytest.mark.asyncio
    async def test_router_id_from_hostname(self) -> None:
        """router_id defaults to hostname."""
        manager = NatsManager()
        assert manager.router_id  # Non-empty string

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        """Disconnect on never-connected manager is a no-op."""
        manager = NatsManager()
        await manager.disconnect()  # Should not raise


class TestNatsManagerPublish:
    """Tests for publishing config notifications."""

    @pytest.mark.asyncio
    async def test_publish_raises_when_not_connected(self) -> None:
        """publish_config_notification raises RuntimeError if not connected."""
        manager = NatsManager()
        with pytest.raises(RuntimeError, match="NATS not connected"):
            await manager.publish_config_notification(
                model_id="test/model",
                profiles_added=["default"],
                affected_bundles=["default"],
                bundle_config_hashes={"default": "abc123"},
                epoch=1,
                model_config_yaml="sie_id: test/model\n",
            )

    @pytest.mark.asyncio
    async def test_publish_sends_to_bundle_subjects(self) -> None:
        """Publishes to sie.config.models.<bundle_id> for each affected bundle."""
        manager = NatsManager()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        manager._nc = mock_nc
        manager._connected = True

        await manager.publish_config_notification(
            model_id="test/model",
            profiles_added=["default"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={"default": "hash1", "sglang": "hash2"},
            epoch=5,
            model_config_yaml="sie_id: test/model\n",
        )

        # Should publish to 3 subjects: default, sglang, _all
        assert mock_nc.publish.call_count == 3

        # Check per-bundle subject calls
        calls = mock_nc.publish.call_args_list
        subjects = [call.args[0] for call in calls]
        assert "sie.config.models.default" in subjects
        assert "sie.config.models.sglang" in subjects
        assert "sie.config.models._all" in subjects

    @pytest.mark.asyncio
    async def test_publish_bundle_payload_structure(self) -> None:
        """Per-bundle payload contains correct fields."""
        manager = NatsManager()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        manager._nc = mock_nc
        manager._connected = True

        await manager.publish_config_notification(
            model_id="org/model",
            profiles_added=["default", "custom"],
            affected_bundles=["default"],
            bundle_config_hashes={"default": "abc"},
            epoch=42,
            model_config_yaml="full yaml content",
        )

        # Find the per-bundle call
        for call in mock_nc.publish.call_args_list:
            if call.args[0] == "sie.config.models.default":
                payload = json.loads(call.args[1].decode())
                assert payload["model_id"] == "org/model"
                assert payload["profiles_added"] == ["default", "custom"]
                assert payload["bundle_config_hash"] == "abc"
                assert payload["epoch"] == 42
                assert payload["router_id"] == manager.router_id
                assert payload["model_config"] == "full yaml content"
                break
        else:
            pytest.fail("Per-bundle publish not found")

    @pytest.mark.asyncio
    async def test_publish_all_payload_has_affected_bundles(self) -> None:
        """Global _all payload includes affected_bundles list."""
        manager = NatsManager()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        manager._nc = mock_nc
        manager._connected = True

        await manager.publish_config_notification(
            model_id="test/model",
            profiles_added=["default"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={},
            epoch=1,
            model_config_yaml="yaml",
        )

        for call in mock_nc.publish.call_args_list:
            if call.args[0] == _ALL_SUBJECT:
                payload = json.loads(call.args[1].decode())
                assert payload["affected_bundles"] == ["default", "sglang"]
                assert payload["model_id"] == "test/model"
                break
        else:
            pytest.fail("_all publish not found")


class TestNatsManagerCallbacks:
    """Tests for _handle_disconnect, _handle_error, _handle_reconnect callbacks."""

    @pytest.mark.asyncio
    async def test_handle_disconnect_sets_connected_false(self) -> None:
        """_handle_disconnect sets _connected to False."""
        manager = NatsManager()
        manager._connected = True

        await manager._handle_disconnect()

        assert manager._connected is False

    @pytest.mark.asyncio
    async def test_handle_error_does_not_crash(self) -> None:
        """_handle_error logs the error without raising."""
        manager = NatsManager()
        # Should not raise for any exception type
        await manager._handle_error(RuntimeError("test nats error"))
        await manager._handle_error(ConnectionError("connection lost"))
        await manager._handle_error(Exception("generic"))

    @pytest.mark.asyncio
    async def test_handle_reconnect_sets_connected_true(self) -> None:
        """_handle_reconnect sets _connected back to True."""
        manager = NatsManager()
        manager._connected = False

        await manager._handle_reconnect()

        assert manager._connected is True

    @pytest.mark.asyncio
    async def test_handle_reconnect_calls_on_reconnect_callback(self) -> None:
        """_handle_reconnect invokes the on_reconnect callback."""
        reconnect_cb = AsyncMock()
        manager = NatsManager(on_reconnect=reconnect_cb)
        manager._connected = False

        await manager._handle_reconnect()

        reconnect_cb.assert_called_once()
        assert manager._connected is True

    @pytest.mark.asyncio
    async def test_handle_reconnect_without_callback(self) -> None:
        """_handle_reconnect works when no on_reconnect callback is set."""
        manager = NatsManager(on_reconnect=None)
        manager._connected = False

        await manager._handle_reconnect()  # Should not raise

        assert manager._connected is True

    @pytest.mark.asyncio
    async def test_handle_reconnect_callback_exception_does_not_crash(self) -> None:
        """_handle_reconnect catches exceptions from the callback."""
        failing_cb = AsyncMock(side_effect=RuntimeError("callback boom"))
        manager = NatsManager(on_reconnect=failing_cb)
        manager._connected = False

        # Should not raise despite callback failure
        await manager._handle_reconnect()

        assert manager._connected is True
        failing_cb.assert_called_once()


class TestNatsManagerSubscription:
    """Tests for receiving notifications from other routers."""

    @pytest.mark.asyncio
    async def test_self_notification_skipped(self) -> None:
        """Notifications from this router (same router_id) are ignored."""
        callback = AsyncMock()
        manager = NatsManager(on_config_notification=callback)

        # Simulate receiving a message from ourselves
        msg = MagicMock()
        msg.data = json.dumps(
            {
                "router_id": manager.router_id,
                "model_id": "test/model",
                "epoch": 1,
            }
        ).encode()

        await manager._handle_all_notification(msg)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_router_notification_calls_callback(self) -> None:
        """Notifications from other routers trigger the callback."""
        callback = AsyncMock()
        manager = NatsManager(on_config_notification=callback)

        msg = MagicMock()
        msg.data = json.dumps(
            {
                "router_id": "other-router",
                "model_id": "test/model",
                "epoch": 5,
                "model_config": "sie_id: test/model\n",
            }
        ).encode()

        await manager._handle_all_notification(msg)
        callback.assert_called_once()
        call_data = callback.call_args[0][0]
        assert call_data["model_id"] == "test/model"
        assert call_data["router_id"] == "other-router"

    @pytest.mark.asyncio
    async def test_malformed_notification_does_not_crash(self) -> None:
        """Malformed NATS message is logged, not raised."""
        callback = AsyncMock()
        manager = NatsManager(on_config_notification=callback)

        msg = MagicMock()
        msg.data = b"not json at all"

        # Should not raise
        await manager._handle_all_notification(msg)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnect_triggers_callback(self) -> None:
        """NATS reconnect fires the on_reconnect callback."""
        reconnect_cb = AsyncMock()
        manager = NatsManager(on_reconnect=reconnect_cb)

        await manager._handle_reconnect()
        reconnect_cb.assert_called_once()
        assert manager._connected is True


class TestMultiRouterCAS:
    """Tests for multi-router write coordination via config store CAS."""

    def test_concurrent_cas_second_writer_retries(self) -> None:
        """Simulates two routers racing on CAS — loser retries successfully."""
        import tempfile

        from sie_router.config_store import ConfigStore, EpochCASError

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(tmpdir)

            # Both routers read epoch=0
            epoch_r1 = store.read_epoch()
            epoch_r2 = store.read_epoch()

            # Router 1 wins CAS
            store.cas_epoch(epoch_r1)

            # Router 2 fails CAS
            with pytest.raises(EpochCASError):
                store.cas_epoch(epoch_r2)

            # Router 2 retries with fresh epoch
            fresh_epoch = store.read_epoch()
            assert fresh_epoch == 1
            store.cas_epoch(fresh_epoch)
            assert store.read_epoch() == 2

    def test_multi_router_restore_sees_all_models(self) -> None:
        """Multiple routers writing to same store — all models visible on restore."""
        import tempfile

        import yaml as _yaml
        from sie_router.config_store import ConfigStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ConfigStore(tmpdir)

            # Router A adds model
            store.cas_epoch(0)
            store.write_model("router-a/model", _yaml.dump({"sie_id": "router-a/model"}))

            # Router B adds different model
            store.cas_epoch(1)
            store.write_model("router-b/model", _yaml.dump({"sie_id": "router-b/model"}))

            # New router restores — sees both
            all_models = store.load_all_models()
            assert "router-a/model" in all_models
            assert "router-b/model" in all_models
