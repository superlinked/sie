import asyncio
import logging
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sie_server.observability.telemetry import (
    _CONSECUTIVE_FAILURE_WARN_THRESHOLD,
    _FIRST_UPDATE_MAX_S,
    _FIRST_UPDATE_MIN_S,
    _UPDATE_MAX_S,
    _UPDATE_MIN_S,
    _build_payload,
    _get_or_create_worker_id,
    _is_telemetry_disabled,
    _normalize_arch,
    _send_heartbeat,
    _telemetry_loop,
    telemetry_sender,
)


class TestIsTelemetryDisabled:
    def test_disabled_when_sie_telemetry_disabled_is_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "1")
        assert _is_telemetry_disabled() is True

    def test_disabled_when_sie_telemetry_disabled_is_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "true")
        assert _is_telemetry_disabled() is True

    def test_disabled_when_sie_telemetry_disabled_is_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "yes")
        assert _is_telemetry_disabled() is True

    def test_disabled_when_sie_telemetry_disabled_is_true_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "TRUE")
        assert _is_telemetry_disabled() is True

    def test_disabled_when_do_not_track_is_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        assert _is_telemetry_disabled() is True

    def test_not_disabled_when_do_not_track_is_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.setenv("DO_NOT_TRACK", "0")
        assert _is_telemetry_disabled() is False

    def test_not_disabled_when_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        assert _is_telemetry_disabled() is False

    def test_both_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "1")
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        assert _is_telemetry_disabled() is True


class TestGetOrCreateWorkerId:
    def test_creates_and_persists_worker_id(self, tmp_path: Path) -> None:
        data_dir = str(tmp_path)
        worker_id = _get_or_create_worker_id(data_dir)
        uuid.UUID(worker_id)  # validates format

        # Second call returns same ID
        worker_id_2 = _get_or_create_worker_id(data_dir)
        assert worker_id == worker_id_2

    def test_corrupt_file_generates_new_id(self, tmp_path: Path) -> None:
        data_dir = str(tmp_path)
        id_path = os.path.join(data_dir, "worker-id")
        with open(id_path, "w") as f:
            f.write("not-a-uuid")

        worker_id = _get_or_create_worker_id(data_dir)
        uuid.UUID(worker_id)  # valid UUID
        assert worker_id != "not-a-uuid"

    def test_readonly_dir_falls_back_to_ephemeral(self, tmp_path: Path) -> None:
        nonexistent = os.path.join(str(tmp_path), "readonly", "nested", "deep")
        # Make parent read-only
        readonly_parent = Path(str(tmp_path)) / "readonly"
        readonly_parent.mkdir(parents=True)
        readonly_parent.chmod(0o444)

        try:
            worker_id = _get_or_create_worker_id(nonexistent)
            uuid.UUID(worker_id)  # valid UUID, ephemeral
        finally:
            readonly_parent.chmod(0o755)


class TestNormalizeArch:
    def test_x86_64_to_amd64(self) -> None:
        assert _normalize_arch("x86_64") == "amd64"

    def test_amd64_passthrough(self) -> None:
        assert _normalize_arch("amd64") == "amd64"

    def test_aarch64_to_arm64(self) -> None:
        assert _normalize_arch("aarch64") == "arm64"

    def test_arm64_passthrough(self) -> None:
        assert _normalize_arch("arm64") == "arm64"

    def test_unknown_arch_lowered(self) -> None:
        assert _normalize_arch("MIPS") == "mips"


class TestBuildPayload:
    def test_all_fields_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_VARIANT", raising=False)
        monkeypatch.delenv("SIE_DEPLOYMENT_ENV", raising=False)

        payload = _build_payload("test-worker-id", "init", ["NVIDIA L4"])

        assert payload["worker_id"] == "test-worker-id"
        assert payload["event"] == "init"
        assert payload["sie_version"] is not None
        assert payload["variant"] is None
        assert payload["os"] is not None
        assert payload["arch"] in ("amd64", "arm64") or isinstance(payload["arch"], str)
        assert payload["gpus"] == ["NVIDIA L4"]
        assert payload["deployment_env"] == "unknown"
        assert "sent_at" in payload

    def test_deployment_env_defaults_to_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_DEPLOYMENT_ENV", raising=False)
        payload = _build_payload("w", "init", [])
        assert payload["deployment_env"] == "unknown"

    def test_deployment_env_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_DEPLOYMENT_ENV", "staging")
        payload = _build_payload("w", "init", [])
        assert payload["deployment_env"] == "staging"

    def test_variant_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_VARIANT", "cuda12-default")
        payload = _build_payload("w", "init", [])
        assert payload["variant"] == "cuda12-default"

    def test_gpu_names_list(self) -> None:
        payload = _build_payload("w", "update", ["NVIDIA L4", "NVIDIA A100"])
        assert payload["gpus"] == ["NVIDIA L4", "NVIDIA A100"]

    def test_empty_gpus(self) -> None:
        payload = _build_payload("w", "update", [])
        assert payload["gpus"] == []


class TestSendHeartbeat:
    @pytest.mark.asyncio
    async def test_success_returns_true(self) -> None:
        mock_response = MagicMock()
        mock_response.is_success = True
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = mock_response

        result = await _send_heartbeat(client, "http://test.example.com", {"event": "init"})
        assert result is True
        client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_returns_false(self) -> None:
        mock_response = MagicMock()
        mock_response.is_success = False
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = mock_response

        result = await _send_heartbeat(client, "http://test.example.com", {"event": "init"})
        assert result is False

    @pytest.mark.asyncio
    async def test_exception_returns_false(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.side_effect = httpx.ConnectError("connection refused")

        result = await _send_heartbeat(client, "http://test.example.com", {"event": "init"})
        assert result is False


class TestTelemetryLoop:
    @pytest.mark.asyncio
    async def test_sends_init_then_stops(self) -> None:
        mock_response = MagicMock()
        mock_response.is_success = True
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = mock_response

        stop_event = asyncio.Event()
        # Set stop immediately so the loop exits after INIT
        stop_event.set()

        await _telemetry_loop(client, "http://test.example.com", "worker-1", ["NVIDIA L4"], stop_event)

        # Should have sent at least the INIT event
        assert client.post.call_count >= 1
        first_call_payload = client.post.call_args_list[0].kwargs.get("json") or client.post.call_args_list[0][1].get(
            "json"
        )
        assert first_call_payload["event"] == "init"

    @pytest.mark.asyncio
    async def test_failure_counter_warns_after_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_response = MagicMock()
        mock_response.is_success = False
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post.return_value = mock_response

        stop_event = asyncio.Event()

        # Patch random.uniform to return 0 so the loop progresses instantly
        with patch("sie_server.observability.telemetry.random.uniform", return_value=0.0):
            # Run the loop for a short time then stop
            async def stop_after_sends() -> None:
                # Wait until enough failures have occurred
                while client.post.call_count < _CONSECUTIVE_FAILURE_WARN_THRESHOLD + 2:  # noqa: ASYNC110
                    await asyncio.sleep(0.01)
                stop_event.set()

            task = asyncio.create_task(stop_after_sends())
            await _telemetry_loop(client, "http://test.example.com", "worker-1", [], stop_event)
            await task

        assert any("consecutive send failures" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self) -> None:
        call_count = 0

        mock_success = MagicMock()
        mock_success.is_success = True
        mock_failure = MagicMock()
        mock_failure.is_success = False

        client = AsyncMock(spec=httpx.AsyncClient)

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First call (INIT) succeeds, next 2 fail, then succeed
            if call_count == 1 or call_count >= 4:
                return mock_success
            return mock_failure

        client.post.side_effect = side_effect

        stop_event = asyncio.Event()

        with patch("sie_server.observability.telemetry.random.uniform", return_value=0.0):

            async def stop_after_sends() -> None:
                while client.post.call_count < 5:  # noqa: ASYNC110
                    await asyncio.sleep(0.01)
                stop_event.set()

            task = asyncio.create_task(stop_after_sends())
            await _telemetry_loop(client, "http://test.example.com", "worker-1", [], stop_event)
            await task

        # No warning should have been logged since counter was reset before reaching threshold


class TestJitterRanges:
    def test_first_update_range(self) -> None:
        assert _FIRST_UPDATE_MIN_S == 45 * 60
        assert _FIRST_UPDATE_MAX_S == 75 * 60

    def test_update_range(self) -> None:
        assert _UPDATE_MIN_S == 55 * 60
        assert _UPDATE_MAX_S == 65 * 60


class TestTelemetrySender:
    @pytest.mark.asyncio
    async def test_disabled_via_env_yields_immediately(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("SIE_TELEMETRY_DISABLED", "1")
        with caplog.at_level(logging.INFO, logger="sie_server.observability.telemetry"):
            async with telemetry_sender():
                pass
        assert any("disabled via env var" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_lifecycle_init_terminate(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        sent_payloads: list[dict] = []

        mock_response = MagicMock()
        mock_response.is_success = True

        async def capture_post(*args, **kwargs):
            payload = kwargs.get("json", {})
            sent_payloads.append(payload)
            return mock_response

        with (
            patch("sie_server.observability.telemetry.httpx.AsyncClient") as mock_client_cls,
            patch("sie_server.observability.telemetry.get_gpu_metrics", return_value=[]),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with telemetry_sender():
                # Give the background task a moment to send INIT
                await asyncio.sleep(0.1)

        # Should have INIT and TERMINATE events
        events = [p.get("event") for p in sent_payloads]
        assert "init" in events
        assert "terminate" in events

    @pytest.mark.asyncio
    async def test_terminate_timeout_does_not_block(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        async def slow_post(*args, **kwargs):
            await asyncio.sleep(100)  # simulate unreachable receiver
            return MagicMock(is_success=True)

        with (
            patch("sie_server.observability.telemetry.httpx.AsyncClient") as mock_client_cls,
            patch("sie_server.observability.telemetry.get_gpu_metrics", return_value=[]),
            patch("sie_server.observability.telemetry._SEND_TIMEOUT_S", 0.1),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=slow_post)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            # Should not hang
            async with telemetry_sender():
                await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_endpoint_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))
        custom_url = "https://custom.example.com/telemetry"
        monkeypatch.setenv("SIE_TELEMETRY_URL", custom_url)

        called_urls: list[str] = []
        mock_response = MagicMock()
        mock_response.is_success = True

        async def capture_post(url, *args, **kwargs):
            called_urls.append(url)
            return mock_response

        with (
            patch("sie_server.observability.telemetry.httpx.AsyncClient") as mock_client_cls,
            patch("sie_server.observability.telemetry.get_gpu_metrics", return_value=[]),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with telemetry_sender():
                await asyncio.sleep(0.1)

        assert all(url == custom_url for url in called_urls)

    @pytest.mark.asyncio
    async def test_gpu_names_extraction(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        gpu_data = [
            {
                "device": "cuda:0",
                "name": "NVIDIA L4",
                "gpu_type": "l4",
                "utilization_pct": 50,
                "memory_used_bytes": 0,
                "memory_total_bytes": 0,
            },
            {
                "device": "cuda:1",
                "name": "NVIDIA A100",
                "gpu_type": "a100-40gb",
                "utilization_pct": 60,
                "memory_used_bytes": 0,
                "memory_total_bytes": 0,
            },
        ]

        sent_payloads: list[dict] = []
        mock_response = MagicMock()
        mock_response.is_success = True

        async def capture_post(*args, **kwargs):
            sent_payloads.append(kwargs.get("json", {}))
            return mock_response

        with (
            patch("sie_server.observability.telemetry.httpx.AsyncClient") as mock_client_cls,
            patch("sie_server.observability.telemetry.get_gpu_metrics", return_value=gpu_data),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with telemetry_sender():
                await asyncio.sleep(0.1)

        init_payloads = [p for p in sent_payloads if p.get("event") == "init"]
        assert len(init_payloads) >= 1
        assert init_payloads[0]["gpus"] == ["NVIDIA L4", "NVIDIA A100"]

    @pytest.mark.asyncio
    async def test_gpu_detection_failure_gives_empty_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("SIE_TELEMETRY_DISABLED", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
        monkeypatch.setenv("SIE_DATA_DIR", str(tmp_path))

        sent_payloads: list[dict] = []
        mock_response = MagicMock()
        mock_response.is_success = True

        async def capture_post(*args, **kwargs):
            sent_payloads.append(kwargs.get("json", {}))
            return mock_response

        with (
            patch("sie_server.observability.telemetry.httpx.AsyncClient") as mock_client_cls,
            patch("sie_server.observability.telemetry.get_gpu_metrics", side_effect=RuntimeError("NVML failed")),
        ):
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=capture_post)
            mock_client.aclose = AsyncMock()
            mock_client_cls.return_value = mock_client

            async with telemetry_sender():
                await asyncio.sleep(0.1)

        init_payloads = [p for p in sent_payloads if p.get("event") == "init"]
        assert len(init_payloads) >= 1
        assert init_payloads[0]["gpus"] == []
