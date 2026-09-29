import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from sie_config.nats_publisher import _ALL_SUBJECT, NatsPublisher


class TestNatsPublisherConnect:
    @pytest.mark.asyncio
    async def test_kickoff_connect_schedules_background_task(self) -> None:
        publisher = NatsPublisher()
        started = asyncio.Event()

        async def fake_connect() -> None:
            started.set()
            await asyncio.Event().wait()

        with patch.object(publisher, "connect", side_effect=fake_connect) as connect:
            publisher.kickoff_connect()
            await asyncio.wait_for(started.wait(), timeout=1.0)
            assert connect.call_count == 1

            publisher.kickoff_connect()
            assert connect.call_count == 1

            await publisher.disconnect()
            assert publisher._boot_connect_task is None

    @pytest.mark.asyncio
    async def test_connect_failure_is_graceful(self) -> None:
        publisher = NatsPublisher(nats_url="nats://nonexistent:4222")
        with patch("nats.connect", side_effect=ConnectionRefusedError("refused")):
            await publisher.connect()
        assert not publisher.connected
        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_connect_invalid_startup_timeout_uses_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("SIE_NATS_STARTUP_CONNECT_TIMEOUT_SEC", "not-a-number")
        publisher = NatsPublisher(nats_url="nats://nonexistent:4222")

        with caplog.at_level(logging.WARNING):
            with patch("nats.connect", side_effect=ConnectionRefusedError("refused")):
                await publisher.connect()

        assert "Invalid SIE_NATS_STARTUP_CONNECT_TIMEOUT_SEC" in caplog.text
        assert not publisher.connected
        await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_connect_timeout_defers_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_NATS_STARTUP_CONNECT_TIMEOUT_SEC", "0.01")
        publisher = NatsPublisher(nats_url="nats://slow:4222")

        async def never_connect(*_args: object, **_kwargs: object) -> None:
            await asyncio.Event().wait()

        with patch("nats.connect", side_effect=never_connect):
            await publisher.connect()
            assert not publisher.connected
            assert publisher._deferred_connect_task is not None
            assert not publisher._deferred_connect_task.done()
            await publisher.disconnect()

    @pytest.mark.asyncio
    async def test_connection_refused_error_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        publisher = NatsPublisher()

        with caplog.at_level(logging.DEBUG):
            await publisher._handle_error(ConnectionRefusedError("refused"))

        refused_records = [record for record in caplog.records if "NATS connection refused" in record.message]
        assert refused_records
        assert all(record.levelno == logging.DEBUG for record in refused_records)

    @pytest.mark.asyncio
    async def test_connected_false_before_connect(self) -> None:
        publisher = NatsPublisher()
        assert not publisher.connected

    @pytest.mark.asyncio
    async def test_router_id_from_hostname(self) -> None:
        publisher = NatsPublisher()
        assert publisher.router_id

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self) -> None:
        publisher = NatsPublisher()
        await publisher.disconnect()


class TestNatsPublisherPublish:
    @pytest.mark.asyncio
    async def test_publish_raises_when_not_connected(self) -> None:
        publisher = NatsPublisher()
        with pytest.raises(RuntimeError, match="NATS not connected"):
            await publisher.publish_config_notification(
                model_id="test/model",
                profiles_added=["default"],
                affected_bundles=["default"],
                bundle_config_hashes={"default": "abc123"},
                epoch=1,
                model_config_yaml="sie_id: test/model\n",
            )

    @pytest.mark.asyncio
    async def test_publish_sends_to_bundle_and_all_subjects(self) -> None:
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="test/model",
            profiles_added=["default"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={"default": "hash1", "sglang": "hash2"},
            epoch=5,
            model_config_yaml="sie_id: test/model\n",
        )
        # One publish per affected bundle to its bundle subject + one to _all.
        assert mock_nc.publish.call_count == 4
        subjects = [call.args[0] for call in mock_nc.publish.call_args_list]
        assert subjects.count("sie.config.models.default") == 1
        assert subjects.count("sie.config.models.sglang") == 1
        assert subjects.count(_ALL_SUBJECT) == 2

    @pytest.mark.asyncio
    async def test_publish_payload_matches_rust_contract(self) -> None:
        """Each publish carries the ConfigNotification superset consumed by
        the gateway and worker sidecars.
        """
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="org/model",
            profiles_added=["default", "custom"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={"default": "abc", "sglang": "def"},
            epoch=42,
            model_config_yaml="full yaml content",
        )
        expected_fields = {
            "router_id",
            "bundle_id",
            "epoch",
            "bundle_config_hash",
            "bundle_pool_config_hashes",
            "bundle_adapters",
            "model_id",
            "profiles_added",
            "model_config",
            "affected_bundles",
            "pool",
        }
        for call in mock_nc.publish.call_args_list:
            payload = json.loads(call.args[1].decode())
            assert set(payload.keys()) == expected_fields
            assert payload["model_id"] == "org/model"
            assert payload["profiles_added"] == ["default", "custom"]
            assert payload["epoch"] == 42
            assert payload["router_id"] == publisher.router_id
            assert payload["model_config"] == "full yaml content"
            assert payload["affected_bundles"] == ["default", "sglang"]
            assert payload["pool"] == "default"
            assert payload["bundle_pool_config_hashes"] == {}
            assert payload["bundle_adapters"] == {}

    @pytest.mark.asyncio
    async def test_publish_bundle_subject_carries_its_own_hash(self) -> None:
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="org/model",
            profiles_added=["default"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={"default": "hash1", "sglang": "hash2"},
            epoch=7,
            model_config_yaml="yaml",
        )
        bundle_payloads: dict[str, dict] = {}
        for call in mock_nc.publish.call_args_list:
            subject = call.args[0]
            if subject.startswith("sie.config.models.") and subject != _ALL_SUBJECT:
                bundle_payloads[subject] = json.loads(call.args[1].decode())

        assert bundle_payloads["sie.config.models.default"]["bundle_id"] == "default"
        assert bundle_payloads["sie.config.models.default"]["bundle_config_hash"] == "hash1"
        assert bundle_payloads["sie.config.models.sglang"]["bundle_id"] == "sglang"
        assert bundle_payloads["sie.config.models.sglang"]["bundle_config_hash"] == "hash2"

    @pytest.mark.asyncio
    async def test_publish_payload_carries_pool_hashes_for_workers(self) -> None:
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="org/model",
            profiles_added=["candle"],
            affected_bundles=["candle"],
            bundle_config_hashes={"candle": "global-hash"},
            epoch=8,
            model_config_yaml="yaml",
            model_pool="Customer-A",
            bundle_pool_config_hashes={
                "candle": {
                    "default": "default-hash",
                    "customer-a": "tenant-hash",
                }
            },
        )

        payload = json.loads(mock_nc.publish.call_args_list[0].args[1].decode())
        assert payload["pool"] == "customer-a"
        assert payload["bundle_config_hash"] == "global-hash"
        assert payload["bundle_pool_config_hashes"]["candle"]["customer-a"] == "tenant-hash"

    @pytest.mark.asyncio
    async def test_publish_payload_carries_the_bundle_adapters_the_hashes_used(self) -> None:
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        adapters = {"default": ["sie_server.adapters.bert_flash", "sie_server.adapters.laya.adapter"]}
        await publisher.publish_config_notification(
            model_id="org/model",
            profiles_added=["default"],
            affected_bundles=["default"],
            bundle_config_hashes={"default": "hash"},
            epoch=9,
            model_config_yaml="yaml",
            bundle_adapters=adapters,
        )

        for call in mock_nc.publish.call_args_list:
            assert json.loads(call.args[1].decode())["bundle_adapters"] == adapters

    @pytest.mark.asyncio
    async def test_publish_all_payload_matches_paired_bundle_payload(self) -> None:
        """The payload sent to _all for a given bundle is byte-identical to the
        payload sent to that bundle's dedicated subject. This preserves
        self-filtering by ``router_id`` and bundle-aware logging on the Rust
        consumer.
        """
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="test/model",
            profiles_added=["default"],
            affected_bundles=["default", "sglang"],
            bundle_config_hashes={"default": "h1", "sglang": "h2"},
            epoch=1,
            model_config_yaml="yaml",
        )
        # Publishes are ordered (bundle, _all) per bundle; pair them up.
        calls = mock_nc.publish.call_args_list
        assert len(calls) == 4
        for i in range(0, len(calls), 2):
            bundle_call, all_call = calls[i], calls[i + 1]
            assert all_call.args[0] == _ALL_SUBJECT
            assert bundle_call.args[1] == all_call.args[1]

    @pytest.mark.asyncio
    async def test_publish_continues_and_raises_on_partial_failure(self) -> None:
        # Fix #7 regression: if a single bundle's NATS publish raises,
        # the publisher must (a) keep going so workers on healthy
        # bundles still receive the delta, (b) collect the failing
        # bundles, and (c) raise `PartialPublishError` at the end with
        # the list of failures. Previously it aborted on the first
        # exception, hiding which bundles actually got the delta.
        from sie_config.nats_publisher import PartialPublishError

        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True

        # Fail on the second bundle's main-subject publish.
        call_count = {"n": 0}

        async def maybe_fail(subject, encoded):
            call_count["n"] += 1
            if subject == "sie.config.models.sglang":
                raise RuntimeError("simulated NATS disconnect")

        mock_nc.publish = AsyncMock(side_effect=maybe_fail)

        with pytest.raises(PartialPublishError) as excinfo:
            await publisher.publish_config_notification(
                model_id="test/model",
                profiles_added=["default"],
                affected_bundles=["default", "sglang", "other"],
                bundle_config_hashes={"default": "h1", "sglang": "h2", "other": "h3"},
                epoch=9,
                model_config_yaml="yaml",
            )
        err = excinfo.value
        assert err.failed_bundles == ["sglang"]
        assert err.total == 3
        assert err.model_id == "test/model"
        assert err.epoch == 9
        # 'default' got 2 publishes, 'sglang' attempted 1 (failed before _all),
        # 'other' got 2 publishes. Total calls = 5.
        assert call_count["n"] == 5

    @pytest.mark.asyncio
    async def test_publish_with_no_affected_bundles_is_noop(self) -> None:
        publisher = NatsPublisher()
        mock_nc = AsyncMock()
        mock_nc.is_connected = True
        publisher._nc = mock_nc
        publisher._connected = True
        await publisher.publish_config_notification(
            model_id="test/model",
            profiles_added=["default"],
            affected_bundles=[],
            bundle_config_hashes={},
            epoch=1,
            model_config_yaml="yaml",
        )
        assert mock_nc.publish.call_count == 0


class TestNatsPublisherCallbacks:
    @pytest.mark.asyncio
    async def test_handle_disconnect_sets_connected_false(self) -> None:
        publisher = NatsPublisher()
        publisher._connected = True
        await publisher._handle_disconnect()
        assert publisher._connected is False

    @pytest.mark.asyncio
    async def test_handle_error_does_not_crash(self) -> None:
        publisher = NatsPublisher()
        await publisher._handle_error(RuntimeError("test nats error"))
        await publisher._handle_error(ConnectionError("connection lost"))
        await publisher._handle_error(Exception("generic"))

    @pytest.mark.asyncio
    async def test_handle_reconnect_sets_connected_true(self) -> None:
        publisher = NatsPublisher()
        publisher._connected = False
        await publisher._handle_reconnect()
        assert publisher._connected is True
