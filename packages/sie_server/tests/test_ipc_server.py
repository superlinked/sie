from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import struct
import tempfile
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import msgspec
import numpy as np
import pytest
import sie_server.ipc_server as ipc_server_module
import yaml
from sie_config.model_registry import ModelRegistry as ConfigModelRegistry
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.api.ws import compute_bundle_config_hash_cached
from sie_server.config.model import ModelConfig
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.readiness import mark_not_ready, mark_ready, register_liveness_probe
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_server import IpcServer, IpcServerError
from sie_server.ipc_types import (
    IPC_VERSION,
    ApplyModelConfigRequest,
    IpcResponseChunkV1,
    ProcessGenerateRequest,
    ReplaceModelConfigEntry,
    ReplaceModelConfigsRequest,
    ResponseEnvelope,
    SetPinnedModelsRequest,
    SignalGenerateCancelRequest,
)
from sie_server.observability import worker_telemetry
from sie_server.queue_executor import QueueExecutor

_LEN_STRUCT = struct.Struct("!I")


def _short_sock_path() -> Path:
    """Return a UDS path short enough for AF_UNIX on macOS (~104 char limit).

    pytest's ``tmp_path`` ends up under ``/private/var/folders/.../`` which
    overshoots the limit — put sockets in ``/tmp`` with a short random name.
    """
    base = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    # Fall back to /tmp if TMPDIR is itself too long (macOS default TMPDIR is).
    if len(str(base)) > 20:
        base = Path("/tmp")  # noqa: S108
    return base / f"sie-{uuid.uuid4().hex[:12]}.sock"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _Client:
    """Minimal UDS IPC client used only by these tests."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    @classmethod
    async def connect(cls, path: Path) -> _Client:
        r, w = await asyncio.open_unix_connection(str(path))
        return cls(r, w)

    async def close(self) -> None:
        self.writer.close()
        with contextlib.suppress(Exception):
            await self.writer.wait_closed()

    async def rpc(self, method: str, body: dict, *, request_id: str = "r1") -> dict:
        envelope = {
            "version": IPC_VERSION,
            "method": method,
            "request_id": request_id,
            "body": body,
        }
        payload = msgpack.packb(envelope, use_bin_type=True)
        self.writer.write(_LEN_STRUCT.pack(len(payload)) + payload)
        await self.writer.drain()

        header = await self.reader.readexactly(_LEN_STRUCT.size)
        (length,) = _LEN_STRUCT.unpack(header)
        resp_bytes = await self.reader.readexactly(length)
        return msgpack.unpackb(resp_bytes, raw=False)


class _ClosedWriter:
    def write(self, _payload: bytes) -> None:
        msg = "unable to perform operation on <UnixTransport closed=True>; the handler is closed"
        raise RuntimeError(msg)

    async def drain(self) -> None:
        return None


class _CapturingWriter:
    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, payload: bytes) -> None:
        self.buf.extend(payload)

    async def drain(self) -> None:
        return None


def _decode_written_frames(writer: _CapturingWriter) -> list[dict]:
    frames: list[dict] = []
    offset = 0
    data = bytes(writer.buf)
    while offset < len(data):
        (length,) = _LEN_STRUCT.unpack(data[offset : offset + _LEN_STRUCT.size])
        offset += _LEN_STRUCT.size
        payload = data[offset : offset + length]
        offset += length
        frames.append(msgpack.unpackb(payload, raw=False))
    return frames


def _written_payloads(writer: _CapturingWriter) -> list[bytes]:
    payloads: list[bytes] = []
    offset = 0
    data = bytes(writer.buf)
    while offset < len(data):
        (length,) = _LEN_STRUCT.unpack(data[offset : offset + _LEN_STRUCT.size])
        offset += _LEN_STRUCT.size
        payloads.append(data[offset : offset + length])
        offset += length
    return payloads


def _make_executor() -> tuple[QueueExecutor, MagicMock]:
    reg = MagicMock()
    reg.model_names = ["test/model"]
    reg.device = "cpu"
    reg.loaded_model_names = ["test/model"]
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    config = MagicMock()
    config.sie_id = "test/model"
    config.outputs = ["dense"]
    config.resolve_profile.return_value.runtime = {}
    reg.get_config.return_value = config
    reg.get_configs_snapshot.return_value = {}
    return QueueExecutor(reg), reg


@pytest.fixture(autouse=True)
def _reset_readiness_state() -> Iterator[None]:
    register_liveness_probe(None)
    mark_not_ready()
    yield
    register_liveness_probe(None)
    mark_not_ready()


def _qwen_profile_variant_yaml() -> str:
    return """
sie_id: Qwen/Qwen3.6-27B
hf_id: Qwen/Qwen3.6-27B
tasks:
  generate:
    context_length: 4096
    max_output_tokens: 4096
max_sequence_length: 4096
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 16384
    kv_budget_tokens: 8192
  rtx-pro-6000:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 32768
    kv_budget_tokens: 65536
    adapter_options:
      loadtime:
        max_seq_length: 32768
"""


def _qwen_default_only_yaml() -> str:
    return """
sie_id: Qwen/Qwen3.6-27B
hf_id: Qwen/Qwen3.6-27B
tasks:
  generate:
    context_length: 4096
    max_output_tokens: 4096
max_sequence_length: 4096
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 16384
    kv_budget_tokens: 8192
"""


def _qwen_real_colon_config_yaml() -> str:
    return """
sie_id: Qwen/Qwen3.6-27B:rtx-pro-6000
hf_id: Qwen/Qwen3.6-27B
tasks:
  generate:
    context_length: 32768
    max_output_tokens: 4096
max_sequence_length: 32768
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 32768
    kv_budget_tokens: 65536
    adapter_options:
      loadtime:
        max_seq_length: 32768
"""


def _qwen_invalid_legacy_lora_yaml() -> str:
    return """
sie_id: Qwen/Qwen3.6-27B
hf_id: Qwen/Qwen3.6-27B
tasks:
  generate:
    context_length: 4096
    max_output_tokens: 4096
max_sequence_length: 4096
profiles:
  default:
    adapter_path: sie_server.adapters.sglang.generation:SGLangGenerationAdapter
    max_batch_tokens: 16384
    kv_budget_tokens: 8192
    adapter_options:
      runtime:
        lora_id: legacy-lora
"""


def _worker_telemetry_model_yaml(
    model_id: str,
    *,
    default_adapter: str = "sie_server.adapters.sentence_transformer:Adapter",
    accelerated_adapter: str | None = None,
) -> str:
    accelerated_profile = ""
    if accelerated_adapter is not None:
        accelerated_profile = f"""
  accelerated:
    adapter_path: {accelerated_adapter}
    max_batch_tokens: 8192
"""
    return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: {default_adapter}
    max_batch_tokens: 4096
{accelerated_profile}
"""


@pytest.fixture
async def server_and_path():
    executor, _reg = _make_executor()
    sock = _short_sock_path()
    srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
    await srv.start()
    try:
        yield srv, sock
    finally:
        await srv.stop(drain_timeout_s=1.0)


# -----------------------------------------------------------------------------
# Transport / framing
# -----------------------------------------------------------------------------


class TestFraming:
    @pytest.mark.asyncio
    async def test_small_negotiated_response_preserves_legacy_bytes(self) -> None:
        executor, _reg = _make_executor()
        server = IpcServer(_short_sock_path(), executor, worker_id="w")
        writer = _CapturingWriter()
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id="r-small",
            ok=True,
            body={"value": [1, 2, 3]},
        )
        expected = msgpack.packb(
            {
                "version": IPC_VERSION,
                "request_id": "r-small",
                "ok": True,
                "body": {"value": [1, 2, 3]},
                "error": None,
            },
            use_bin_type=True,
        )

        await server._write_response_envelope(
            writer,  # type: ignore[arg-type]
            envelope,
            accepts_response_chunks_v1=True,
        )

        assert _written_payloads(writer) == [expected]

    @pytest.mark.asyncio
    async def test_oversized_negotiated_response_chunks_exact_serialized_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ipc_server_module, "_MAX_FRAME_BYTES", 512)
        monkeypatch.setattr(ipc_server_module, "_IPC_RESPONSE_CHUNK_PAYLOAD_BYTES", 128)
        monkeypatch.setattr(ipc_server_module, "_MAX_CHUNKED_IPC_RESPONSE_BYTES", 4096)
        executor, _reg = _make_executor()
        server = IpcServer(_short_sock_path(), executor, worker_id="w")
        writer = _CapturingWriter()
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id="r-large",
            ok=True,
            body={"blob": b"x" * 1200},
        )
        expected = msgpack.packb(
            {
                "version": IPC_VERSION,
                "request_id": "r-large",
                "ok": True,
                "body": {"blob": b"x" * 1200},
                "error": None,
            },
            use_bin_type=True,
        )

        await server._write_response_envelope(
            writer,  # type: ignore[arg-type]
            envelope,
            accepts_response_chunks_v1=True,
        )

        frames = _decode_written_frames(writer)
        assert len(frames) > 1
        assert all(frame["kind"] == "ipc_response_chunk_v1" for frame in frames)
        assert [frame["chunk_index"] for frame in frames] == list(range(len(frames)))
        assert all(frame["chunk_count"] == len(frames) for frame in frames)
        assert all(frame["total_bytes"] == len(expected) for frame in frames)
        assert all(frame["transfer_digest"] == hashlib.sha256(expected).digest() for frame in frames)
        assert b"".join(frame["payload"] for frame in frames) == expected
        assert all(len(payload) <= 512 for payload in _written_payloads(writer))

    def test_chunk_builtins_preserve_memoryview_as_msgpack_binary(self) -> None:
        payload = b"abcdef"
        chunk = IpcResponseChunkV1(
            version=IPC_VERSION,
            request_id="r-view",
            transfer_digest=b"0" * 32,
            chunk_index=0,
            chunk_count=1,
            total_bytes=len(payload),
            payload=cast("bytes", memoryview(payload)[1:4]),
        )

        builtins = ipc_server_module._to_builtins(chunk)
        assert isinstance(builtins["payload"], memoryview)
        encoded = msgpack.packb(builtins, use_bin_type=True)
        assert msgpack.unpackb(encoded, raw=False)["payload"] == b"bcd"
        assert msgspec.msgpack.Decoder(IpcResponseChunkV1).decode(encoded).payload == b"bcd"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capability", [False, 1, "true", None])
    async def test_oversized_response_without_exact_v1_capability_returns_compact_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capability: object,
    ) -> None:
        monkeypatch.setattr(ipc_server_module, "_MAX_FRAME_BYTES", 512)
        executor, _reg = _make_executor()
        server = IpcServer(_short_sock_path(), executor, worker_id="w")
        writer = _CapturingWriter()
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id="r-large",
            ok=True,
            body={"blob": b"x" * 1200},
        )

        await server._write_response_envelope(
            writer,  # type: ignore[arg-type]
            envelope,
            accepts_response_chunks_v1=capability is True,
        )

        frames = _decode_written_frames(writer)
        assert len(frames) == 1
        assert frames[0]["ok"] is False
        assert frames[0]["request_id"] == "r-large"
        assert "not negotiated" in frames[0]["error"]

    @pytest.mark.asyncio
    async def test_negotiated_response_over_logical_bound_returns_compact_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ipc_server_module, "_MAX_FRAME_BYTES", 512)
        monkeypatch.setattr(ipc_server_module, "_IPC_RESPONSE_CHUNK_PAYLOAD_BYTES", 128)
        monkeypatch.setattr(ipc_server_module, "_MAX_CHUNKED_IPC_RESPONSE_BYTES", 800)
        executor, _reg = _make_executor()
        server = IpcServer(_short_sock_path(), executor, worker_id="w")
        writer = _CapturingWriter()
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id="r-overbound",
            ok=True,
            body={"blob": b"x" * 1200},
        )

        await server._write_response_envelope(
            writer,  # type: ignore[arg-type]
            envelope,
            accepts_response_chunks_v1=True,
        )

        frames = _decode_written_frames(writer)
        assert len(frames) == 1
        assert frames[0]["ok"] is False
        assert "bounds" in frames[0]["error"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("wire_value", "expected"),
        [(True, True), (False, False), (1, False), ("true", False), (None, False)],
    )
    async def test_request_response_chunk_capability_is_exact_v1_boolean(
        self,
        wire_value: object,
        expected: bool,
    ) -> None:
        executor, _reg = _make_executor()
        server = IpcServer(_short_sock_path(), executor, worker_id="w")
        writer = _CapturingWriter()
        frame = msgpack.packb(
            {
                "version": IPC_VERSION,
                "method": "Ping",
                "request_id": "r-cap",
                "body": {"timestamp_ms": 1.0},
                "accepts_ipc_response_chunks_v1": wire_value,
            },
            use_bin_type=True,
        )

        with patch.object(server, "_run_method", new=AsyncMock()) as run_method:
            await server._dispatch_frame(frame, writer)  # type: ignore[arg-type]

        run_method.assert_awaited_once_with(
            "Ping",
            "r-cap",
            {"timestamp_ms": 1.0},
            writer,
            accepts_response_chunks_v1=expected,
        )

    @pytest.mark.asyncio
    async def test_ping_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Ping", {"timestamp_ms": 123.0})
            assert resp["ok"] is True
            assert resp["version"] == IPC_VERSION
            assert resp["request_id"] == "r1"
            assert resp["body"]["timestamp_ms"] == 123.0
            assert resp["body"]["worker_id"] == "worker-test"
            assert resp["body"]["ready"] is False
            assert resp["body"]["bundle_config_hash"] == ""
            assert resp["body"]["loaded_models"] == ["test/model"]
            assert srv.is_heartbeat_fresh()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_ping_reports_ready_after_worker_ready(self, server_and_path) -> None:
        _srv, sock = server_and_path
        mark_ready()
        client = await _Client.connect(sock)
        try:
            with patch(
                "sie_server.ipc_server.gpu_is_healthy_async",
                new=AsyncMock(return_value=True),
            ) as health:
                resp = await client.rpc("Ping", {"timestamp_ms": 123.0})
            assert resp["ok"] is True
            assert resp["body"]["ready"] is True
            health.assert_awaited_once_with()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_ping_reports_not_ready_when_gpu_health_fails(self, server_and_path) -> None:
        _srv, sock = server_and_path
        mark_ready()
        client = await _Client.connect(sock)
        try:
            with patch(
                "sie_server.ipc_server.gpu_is_healthy_async",
                new=AsyncMock(return_value=False),
            ) as health:
                resp = await client.rpc("Ping", {"timestamp_ms": 123.0})
            assert resp["ok"] is True
            assert resp["body"]["ready"] is False
            health.assert_awaited_once_with()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_ping_reports_not_ready_when_gpu_health_raises(self, server_and_path) -> None:
        _srv, sock = server_and_path
        mark_ready()
        client = await _Client.connect(sock)
        try:
            with patch(
                "sie_server.ipc_server.gpu_is_healthy_async",
                new=AsyncMock(side_effect=RuntimeError("nvml unavailable")),
            ) as health:
                resp = await client.rpc("Ping", {"timestamp_ms": 123.0})
            assert resp["ok"] is True
            assert resp["body"]["ready"] is False
            assert resp["body"]["loaded_models"] == ["test/model"]
            health.assert_awaited_once_with()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multiple_requests_on_one_connection(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            r1 = await client.rpc("Ping", {"timestamp_ms": 1.0}, request_id="a")
            r2 = await client.rpc("Ping", {"timestamp_ms": 2.0}, request_id="b")
            assert r1["request_id"] == "a"
            assert r2["request_id"] == "b"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("NoSuchMethod", {})
            assert resp["ok"] is False
            assert "unknown method" in (resp["error"] or "")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bad_envelope_version_returns_error(self) -> None:
        executor, _reg = _make_executor()
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w"):
            client = await _Client.connect(sock)
            try:
                envelope = {"version": 999, "method": "Ping", "request_id": "r1", "body": {}}
                payload = msgpack.packb(envelope, use_bin_type=True)
                client.writer.write(_LEN_STRUCT.pack(len(payload)) + payload)
                await client.writer.drain()
                header = await client.reader.readexactly(_LEN_STRUCT.size)
                (length,) = _LEN_STRUCT.unpack(header)
                resp_bytes = await client.reader.readexactly(length)
                resp = msgpack.unpackb(resp_bytes, raw=False)
                assert resp["ok"] is False
                assert "unsupported version" in resp["error"]
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_malformed_msgpack_closes_connection(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            client.writer.write(_LEN_STRUCT.pack(3) + b"\xff\xff\xff")
            await client.writer.drain()
            data = await client.reader.read(1)
            assert data == b""  # connection closed on protocol error
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_background_response_to_closed_peer_is_not_method_crash(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        executor, _reg = _make_executor()
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="w")

        caplog.set_level("INFO", logger="sie_server.ipc_server")
        await srv._run_method_task("Ping", "r1", {"timestamp_ms": 1.0}, _ClosedWriter())  # type: ignore[arg-type]

        messages = [record.getMessage() for record in caplog.records]
        assert any("IPC client disconnected before Ping response" in message for message in messages)
        assert not any("IPC method Ping crashed" in message for message in messages)


# -----------------------------------------------------------------------------
# EnsureModelReady
# -----------------------------------------------------------------------------


class TestEnsureModelReady:
    @pytest.mark.asyncio
    async def test_ready_model(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("EnsureModelReady", {"model_id": "test/model"})
            assert resp["ok"] is True
            assert resp["body"]["state"] == "ready"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_batch_budget_included_on_ready(self) -> None:
        """Parity #4: EnsureModelReady carries per-model batch budget for the
        Rust consumer's fair-dispatch cap.
        """
        reg = MagicMock()
        reg.model_names = ["test/model"]
        reg.device = "cpu"
        reg.is_loaded.return_value = True
        reg.is_loading.return_value = False
        reg.loaded_model_names = []
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=42)
        reg.get_worker.return_value = worker
        executor = QueueExecutor(reg)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("EnsureModelReady", {"model_id": "test/model"})
                assert resp["ok"] is True
                assert resp["body"]["state"] == "ready"
                assert resp["body"]["batch_budget"] == 42
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)


# -----------------------------------------------------------------------------
# ApplyModelConfig
# -----------------------------------------------------------------------------


class TestApplyModelConfig:
    def test_python_bundle_hash_matches_config_registry_and_skips_rust_profiles(self) -> None:
        root = _repo_root()
        bundles_dir = root / "packages/sie_server/bundles"
        models_dir = root / "packages/sie_server/models"
        config_registry = ConfigModelRegistry(bundles_dir, models_dir)

        default_filter = match_bundle_models(bundles_dir / "default.yaml", models_dir)
        default_registry = ModelRegistry(
            models_dir=models_dir,
            model_filter=default_filter,
            enable_hot_reload=False,
        )
        assert any(config.hf_revision for config in default_registry.get_configs_snapshot("default").values())
        assert compute_bundle_config_hash_cached(
            default_registry,
            "default",
        ) == config_registry.compute_bundle_config_hash("default")

        candle_filter = match_bundle_models(bundles_dir / "candle.yaml", models_dir)
        candle_registry = ModelRegistry(
            models_dir=models_dir,
            model_filter=candle_filter,
            enable_hot_reload=False,
        )
        assert compute_bundle_config_hash_cached(candle_registry, "candle") == ""
        assert config_registry.compute_bundle_config_hash("candle")

    @pytest.mark.asyncio
    async def test_apply_model_config_hash_cache_is_per_registry(self) -> None:
        stale_registry = ModelRegistry(models_dir=None)
        stale_registry._config_version = 1
        assert compute_bundle_config_hash_cached(stale_registry, "default") == ""

        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)

        resp = await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="default",
                model_id="test/model",
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default"],
                model_config="""
sie_id: test/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
            )
        )

        assert resp.applied is True
        assert resp.bundle_config_hash
        assert resp.config_version == 1

    @pytest.mark.asyncio
    async def test_apply_model_config_adds_registry_config_and_returns_hash(self) -> None:
        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ApplyModelConfig",
                    {
                        "bundle_id": "default",
                        "model_id": "test/model",
                        "epoch": 7,
                        "bundle_config_hash": "from-sie-config",
                        "profiles_added": ["default"],
                        "model_config": """
sie_id: test/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
                    },
                )
                assert resp["ok"] is True
                assert resp["body"]["applied"] is True
                assert resp["body"]["bundle_config_hash"]
                assert resp["body"]["config_version"] == 1
                assert registry.get_config("test/model").sie_id == "test/model"
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_apply_model_config_refreshes_worker_telemetry_catalog_for_added_model(self) -> None:
        lane = "realtime|l4|default"
        model_id = "telemetry/added-model"
        worker_telemetry.configure_worker_metric_context(lane=lane, configs={})
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="default",
                model_id=model_id,
                epoch=7,
                bundle_config_hash="",
                profiles_added=["default"],
                model_config=_worker_telemetry_model_yaml(model_id),
            )
        )

        assert worker_telemetry._dimensions(model_id, "default") == (
            lane,
            model_id,
            "default",
            "python",
        )

    @pytest.mark.asyncio
    async def test_apply_model_config_refreshes_worker_telemetry_profiles_and_backend(self) -> None:
        lane = "realtime|l4|default"
        model_id = "telemetry/updated-model"
        worker_telemetry.configure_worker_metric_context(lane=lane, configs={})
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="default",
                model_id=model_id,
                epoch=7,
                bundle_config_hash="",
                profiles_added=["default"],
                model_config=_worker_telemetry_model_yaml(model_id),
            )
        )
        assert worker_telemetry._dimensions(model_id, "accelerated") == (
            lane,
            "other",
            "other",
            "other",
        )

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="default",
                model_id=model_id,
                epoch=8,
                bundle_config_hash="",
                profiles_added=["default", "accelerated"],
                model_config=_worker_telemetry_model_yaml(
                    model_id,
                    default_adapter="sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                    accelerated_adapter="sie_server.adapters.sentence_transformer:Adapter",
                ),
            )
        )

        assert worker_telemetry._dimensions(model_id, "default") == (
            lane,
            model_id,
            "default",
            "sglang",
        )
        assert worker_telemetry._dimensions(model_id, "accelerated") == (
            lane,
            model_id,
            "accelerated",
            "python",
        )
        variant_id = f"{model_id}:accelerated"
        assert worker_telemetry._dimensions(variant_id, "default") == (
            lane,
            model_id,
            "accelerated",
            "python",
        )

    @pytest.mark.asyncio
    async def test_apply_model_config_expands_profile_variants(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)
        executor._descriptor_cache["Qwen/Qwen3.6-27B:rtx-pro-6000"] = MagicMock()

        resp = await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id="Qwen/Qwen3.6-27B",
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default", "rtx-pro-6000"],
                model_config=_qwen_profile_variant_yaml(),
            )
        )

        assert resp.applied is True
        assert registry.has_model("Qwen/Qwen3.6-27B")
        assert registry.has_model("Qwen/Qwen3.6-27B:rtx-pro-6000")
        variant = registry.get_config("Qwen/Qwen3.6-27B:rtx-pro-6000")
        assert variant.tasks.generate is not None
        assert variant.tasks.generate.context_length == 32768
        assert variant.max_sequence_length == 32768
        assert "Qwen/Qwen3.6-27B:rtx-pro-6000" not in executor._descriptor_cache

    @pytest.mark.asyncio
    async def test_bundle_hash_ignores_expanded_profile_variant_aliases(self) -> None:
        base_config = ModelConfig(**yaml.safe_load(_qwen_profile_variant_yaml()))
        base_registry = ModelRegistry(models_dir=None, model_filter=["Qwen/Qwen3.6-27B"])
        base_registry._configs[base_config.sie_id] = base_config
        base_registry._config_version += 1
        base_hash = compute_bundle_config_hash_cached(base_registry, "sglang")
        assert base_hash

        expanded_registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(expanded_registry)
        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id="Qwen/Qwen3.6-27B",
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default", "rtx-pro-6000"],
                model_config=_qwen_profile_variant_yaml(),
            )
        )

        assert expanded_registry.has_model("Qwen/Qwen3.6-27B:rtx-pro-6000")
        assert compute_bundle_config_hash_cached(expanded_registry, "sglang") == base_hash

    def test_bundle_hash_keeps_real_colon_config(self) -> None:
        base_config = ModelConfig(**yaml.safe_load(_qwen_default_only_yaml()))
        real_colon_config = ModelConfig(**yaml.safe_load(_qwen_real_colon_config_yaml()))
        base_registry = ModelRegistry(models_dir=None)
        base_registry.add_config(base_config)
        base_hash = compute_bundle_config_hash_cached(base_registry, "sglang")
        assert base_hash

        real_colon_registry = ModelRegistry(models_dir=None)
        real_colon_registry.add_config(base_config)
        real_colon_registry.add_config(real_colon_config)

        assert real_colon_config.synthetic_profile_variant_source is None
        assert real_colon_registry.has_model("Qwen/Qwen3.6-27B:rtx-pro-6000")
        assert compute_bundle_config_hash_cached(real_colon_registry, "sglang") != base_hash

    @pytest.mark.asyncio
    async def test_apply_model_config_removes_dropped_profile_variant(self) -> None:
        base_id = "Qwen/Qwen3.6-27B"
        variant_id = "Qwen/Qwen3.6-27B:rtx-pro-6000"
        registry = ModelRegistry(models_dir=None, model_filter=[base_id, variant_id])
        executor = QueueExecutor(registry)

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id=base_id,
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default", "rtx-pro-6000"],
                model_config=_qwen_profile_variant_yaml(),
            )
        )
        assert registry.has_model(variant_id)

        executor._descriptor_cache[variant_id] = MagicMock()
        loaded = MagicMock()
        loaded.worker = None
        loaded.device = "cpu"
        registry._loaded[variant_id] = loaded

        resp = await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id=base_id,
                epoch=8,
                bundle_config_hash="from-sie-config",
                profiles_added=["default"],
                model_config=_qwen_default_only_yaml(),
            )
        )

        assert resp.applied is True
        assert registry.has_model(base_id)
        assert not registry.has_model(variant_id)
        assert variant_id not in registry._loaded
        assert registry._model_filter == {base_id}
        loaded.adapter.unload.assert_called_once()
        assert variant_id not in executor._descriptor_cache

    @pytest.mark.asyncio
    async def test_apply_model_config_invalid_replacement_keeps_dropped_profile_variant(self) -> None:
        base_id = "Qwen/Qwen3.6-27B"
        variant_id = "Qwen/Qwen3.6-27B:rtx-pro-6000"
        lane = "realtime|l4|sglang"
        worker_telemetry.configure_worker_metric_context(lane=lane, configs={})
        registry = ModelRegistry(models_dir=None, model_filter=[base_id, variant_id])
        executor = QueueExecutor(registry)

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id=base_id,
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default", "rtx-pro-6000"],
                model_config=_qwen_profile_variant_yaml(),
            )
        )
        assert registry.has_model(variant_id)
        expected_dimensions = (lane, base_id, "rtx-pro-6000", "sglang")
        assert worker_telemetry._dimensions(base_id, "rtx-pro-6000") == expected_dimensions
        executor._descriptor_cache[variant_id] = MagicMock()

        with pytest.raises(ValueError, match="legacy scalar"):
            await executor.apply_model_config(
                ApplyModelConfigRequest(
                    bundle_id="sglang",
                    model_id=base_id,
                    epoch=8,
                    bundle_config_hash="from-sie-config",
                    profiles_added=["default"],
                    model_config=_qwen_invalid_legacy_lora_yaml(),
                )
            )

        assert registry.has_model(base_id)
        assert registry.has_model(variant_id)
        assert registry._model_filter == {base_id, variant_id}
        assert variant_id in executor._descriptor_cache
        assert worker_telemetry._dimensions(base_id, "rtx-pro-6000") == expected_dimensions

    @pytest.mark.asyncio
    async def test_apply_model_config_serializes_stale_variant_removal(self) -> None:
        base_id = "Qwen/Qwen3.6-27B"
        variant_id = "Qwen/Qwen3.6-27B:rtx-pro-6000"
        registry = ModelRegistry(models_dir=None, model_filter=[base_id, variant_id])
        executor = QueueExecutor(registry)

        await executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="sglang",
                model_id=base_id,
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default", "rtx-pro-6000"],
                model_config=_qwen_profile_variant_yaml(),
            )
        )
        assert registry.has_model(variant_id)

        load_lock = registry._get_load_lock()
        await load_lock.acquire()
        remove_task = asyncio.create_task(
            executor.apply_model_config(
                ApplyModelConfigRequest(
                    bundle_id="sglang",
                    model_id=base_id,
                    epoch=8,
                    bundle_config_hash="from-sie-config",
                    profiles_added=["default"],
                    model_config=_qwen_default_only_yaml(),
                )
            )
        )
        await asyncio.sleep(0)
        assert not remove_task.done()

        readd_task = asyncio.create_task(
            executor.apply_model_config(
                ApplyModelConfigRequest(
                    bundle_id="sglang",
                    model_id=base_id,
                    epoch=9,
                    bundle_config_hash="from-sie-config",
                    profiles_added=["default", "rtx-pro-6000"],
                    model_config=_qwen_profile_variant_yaml(),
                )
            )
        )

        try:
            await asyncio.sleep(0)
            load_lock.release()
            await asyncio.gather(remove_task, readd_task)
        finally:
            if load_lock.locked():
                load_lock.release()
            for task in (remove_task, readd_task):
                if not task.done():
                    task.cancel()

        assert registry.has_model(base_id)
        assert registry.has_model(variant_id)
        assert registry._model_filter == {base_id, variant_id}

    @pytest.mark.asyncio
    async def test_apply_model_config_rejects_model_id_mismatch(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ApplyModelConfig",
                    {
                        "bundle_id": "default",
                        "model_id": "notification/model",
                        "epoch": 7,
                        "bundle_config_hash": "from-sie-config",
                        "profiles_added": ["default"],
                        "model_config": """
sie_id: yaml/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
                    },
                )
                assert resp["ok"] is False
                assert "model_id mismatch" in (resp["error"] or "")
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_replace_model_configs_requires_models_field(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        model_config = """
sie_id: kept/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""
        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_config,
                    )
                ],
            )
        )
        assert registry.has_model("kept/model")

        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ReplaceModelConfigs",
                    {
                        "bundle_id": "default",
                        "epoch": 8,
                        "bundle_config_hash": "",
                    },
                )
                assert resp["ok"] is False
                assert "Object missing required field `models`" in (resp["error"] or "")
                assert registry.has_model("kept/model")
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_replace_model_configs_drops_removed_registry_config(self) -> None:
        lane = "realtime|l4|default"
        worker_telemetry.configure_worker_metric_context(lane=lane, configs={})
        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="stale/model",
                        model_config=model_yaml("stale/model"),
                    ),
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    ),
                ],
            )
        )
        assert registry.has_model("stale/model")
        assert registry.has_model("kept/model")
        assert worker_telemetry._dimensions("stale/model", "default") == (
            lane,
            "stale/model",
            "default",
            "python",
        )
        registry._loaded["stale/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        assert resp.applied_profiles == ["default"]
        registry._do_unload.assert_awaited_once_with("stale/model", reason="config_change")
        assert not registry.has_model("stale/model")
        assert registry.has_model("kept/model")
        assert worker_telemetry._dimensions("stale/model", "default") == (
            lane,
            "other",
            "other",
            "other",
        )
        assert worker_telemetry._dimensions("kept/model", "default") == (
            lane,
            "kept/model",
            "default",
            "python",
        )

    @pytest.mark.asyncio
    async def test_replace_model_configs_expands_profile_variants_from_export(self) -> None:
        registry = ModelRegistry(models_dir=None, model_filter=["Qwen/Qwen3.6-27B:rtx-pro-6000"])
        executor = QueueExecutor(registry)

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="sglang",
                epoch=8,
                bundle_config_hash="from-sie-config",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="Qwen/Qwen3.6-27B",
                        model_config=_qwen_profile_variant_yaml(),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B:rtx-pro-6000"]
        assert resp.applied_profiles == ["default", "rtx-pro-6000"]
        assert registry.has_model("Qwen/Qwen3.6-27B")
        assert registry.has_model("Qwen/Qwen3.6-27B:rtx-pro-6000")
        assert registry._model_filter == {"Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B:rtx-pro-6000"}

    @pytest.mark.asyncio
    async def test_replace_model_configs_rejects_duplicate_export_entries(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        with pytest.raises(ValueError, match="duplicate model config"):
            await executor.replace_model_configs(
                ReplaceModelConfigsRequest(
                    bundle_id="sglang",
                    epoch=8,
                    bundle_config_hash="from-sie-config",
                    models=[
                        ReplaceModelConfigEntry(
                            model_id="Qwen/Qwen3.6-27B",
                            model_config=_qwen_profile_variant_yaml(),
                        ),
                        ReplaceModelConfigEntry(
                            model_id="Qwen/Qwen3.6-27B",
                            model_config=_qwen_profile_variant_yaml(),
                        ),
                    ],
                )
            )

    @pytest.mark.asyncio
    async def test_replace_model_configs_keeps_loaded_model_for_identical_config(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )
        config_version = registry._config_version
        registry._loaded["kept/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        registry._do_unload.assert_not_awaited()
        assert registry._config_version == config_version
        assert registry.has_model("kept/model")

    @pytest.mark.asyncio
    async def test_replace_model_configs_unloads_loaded_model_for_changed_config(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str, *, max_batch_tokens: int) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: {max_batch_tokens}
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model", max_batch_tokens=4096),
                    )
                ],
            )
        )
        registry._loaded["kept/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model", max_batch_tokens=8192),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        assert resp.applied_profiles == ["default"]
        registry._do_unload.assert_awaited_once_with("kept/model", reason="config_change")
        assert registry.get_config("kept/model").profiles["default"].max_batch_tokens == 8192

    @pytest.mark.asyncio
    async def test_replace_model_configs_reports_pool_filtered_models(self) -> None:
        registry = ModelRegistry(models_dir=None, pool_name="customer-a")
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str, *, pool: str | None = None) -> str:
            pool_line = f"pool: {pool}\n" if pool is not None else ""
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
{pool_line}tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="default/model",
                        model_config=model_yaml("default/model"),
                    ),
                    ReplaceModelConfigEntry(
                        model_id="tenant/model",
                        model_config=model_yaml("tenant/model", pool="customer-a"),
                    ),
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["tenant/model"]
        assert not registry.has_model("default/model")
        assert registry.has_model("tenant/model")


# -----------------------------------------------------------------------------
# SetPinnedModels
# -----------------------------------------------------------------------------


class TestSetPinnedModels:
    @pytest.mark.asyncio
    async def test_executor_batch_budget_returns_count(self) -> None:
        registry = ModelRegistry(models_dir=None)
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]
        executor = QueueExecutor(registry)

        resp = await executor.set_pinned_models(SetPinnedModelsRequest(models=["org/a", "org/b"]))

        assert resp.applied is True
        assert resp.pinned_count == 2
        assert registry._pinned_models == frozenset({"org/a", "org/b"})

    @pytest.mark.asyncio
    async def test_set_pinned_models_roundtrip_over_ipc(self) -> None:
        registry = ModelRegistry(models_dir=None)
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]
        executor = QueueExecutor(registry)

        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("SetPinnedModels", {"models": ["org/a:fast", "Org/B"]})
                assert resp["ok"] is True
                assert resp["body"]["applied"] is True
                assert resp["body"]["pinned_count"] == 2
                # Lowercased; non-default profile preserved, base kept as-is.
                assert registry._pinned_models == frozenset({"org/a:fast", "org/b"})
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)


# -----------------------------------------------------------------------------
# ProcessEncodeBatch
# -----------------------------------------------------------------------------


class TestProcessEncodeBatch:
    @pytest.mark.asyncio
    async def test_encode_batch_roundtrip(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            fake_outputs = [{"dense": [0.25, 0.75]}]
            with patch(
                "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
                new_callable=AsyncMock,
                return_value=(fake_outputs, RequestTiming()),
            ):
                resp = await client.rpc(
                    "ProcessEncodeBatch",
                    {
                        "model_id": "test/model",
                        "items": [
                            {
                                "work_item_id": "req-1.0",
                                "request_id": "req-1",
                                "item_index": 0,
                                "total_items": 1,
                                "timestamp": time.time(),
                                "item": {"text": "hello"},
                                "output_types": ["dense"],
                                "is_query": False,
                                "payload_fetch_ms": 0.0,
                            }
                        ],
                    },
                )

            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert len(outcomes) == 1
            outcome = outcomes[0]
            assert outcome["work_item_id"] == "req-1.0"
            assert outcome["disposition"] == "publish_and_ack"
            inner = msgpack.unpackb(outcome["result_msgpack"], raw=False)
            assert inner == {"dense": [0.25, 0.75]}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_encode_batch_during_drain_rejected(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            drain_resp = await client.rpc("Drain", {"deadline_ms": 1000})
            assert drain_resp["ok"] is True
            assert srv._drain_event.is_set()

            resp = await client.rpc(
                "ProcessEncodeBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-2.0",
                            "request_id": "req-2",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "item": {"text": "x"},
                            "output_types": ["dense"],
                            "is_query": False,
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is False
            assert resp["error"] == "draining"
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# ProcessScoreBatch / ProcessExtractBatch smoke
# -----------------------------------------------------------------------------


class TestProcessScoreAndExtract:
    @pytest.mark.asyncio
    async def test_score_batch_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        srv._executor._registry.start_worker = AsyncMock(return_value=worker)

        client = await _Client.connect(sock)
        try:
            resp = await client.rpc(
                "ProcessScoreBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-s.0",
                            "request_id": "req-s",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "query_item": {"text": "q"},
                            "score_items": [
                                {"text": "a", "id": "doc-a"},
                                {"text": "b", "id": "doc-b"},
                            ],
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert outcomes[0]["disposition"] == "publish_and_ack"
            # Score output is Rust-frameable. The legacy ``result_msgpack``
            # is None; the parallel arrays land on ``raw_output.score`` and
            # the Rust publisher does the descending stable sort + rank
            # assignment.
            assert outcomes[0]["result_msgpack"] is None
            score = outcomes[0]["raw_output"]["score"]
            assert score["item_ids"] == ["doc-a", "doc-b"]
            assert score["scores"] == pytest.approx([0.9, 0.1])
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_batch_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        wr = WorkerResult(output=extract_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_extract_preformed_batch = AsyncMock(return_value=[fut])
        srv._executor._registry.start_worker = AsyncMock(return_value=worker)

        client = await _Client.connect(sock)
        try:
            resp = await client.rpc(
                "ProcessExtractBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-e.0",
                            "request_id": "req-e",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "item": {"text": "Alice works at Acme."},
                            "labels": ["person"],
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert outcomes[0]["disposition"] == "publish_and_ack"
            inner = msgpack.unpackb(outcomes[0]["result_msgpack"], raw=False)
            assert "entities" in inner
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# Heartbeat / readiness
# -----------------------------------------------------------------------------


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_stale_when_never_pinged(self) -> None:
        executor, _reg = _make_executor()
        srv = IpcServer(_short_sock_path(), executor, worker_id="w", stale_after_ms=100)
        assert srv.is_heartbeat_fresh() is False

    @pytest.mark.asyncio
    async def test_heartbeat_expires(self) -> None:
        executor, _reg = _make_executor()
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w", stale_after_ms=50) as srv:
            client = await _Client.connect(sock)
            try:
                await client.rpc("Ping", {"timestamp_ms": 0.0})
                assert srv.is_heartbeat_fresh() is True
                # Simulate stale heartbeat by asking about a future monotonic time.
                future = (srv.last_ping_monotonic or 0.0) + 10.0
                assert srv.is_heartbeat_fresh(now_monotonic=future) is False
            finally:
                await client.close()


# -----------------------------------------------------------------------------
# Generation sidecar IPC
# -----------------------------------------------------------------------------


class TestGenerationSidecarIpc:
    @pytest.mark.asyncio
    async def test_worker_capabilities_reports_generation_models(self) -> None:
        executor, reg = _make_executor()
        gen_cfg = MagicMock()
        gen_cfg.tasks.generate = object()
        encode_cfg = MagicMock()
        encode_cfg.tasks.generate = None
        reg.get_configs_snapshot.return_value = {
            "encode/model": encode_cfg,
            "z-generate/model": gen_cfg,
        }
        reg.loaded_model_names = ["encode/model"]
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w"):
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("WorkerCapabilities", {})
            finally:
                await client.close()

        assert resp["ok"] is True
        assert resp["body"] == {
            "has_generation_models": True,
            "generation_models": ["z-generate/model"],
            "supported_models": ["encode/model", "z-generate/model"],
            "loaded_models": ["encode/model"],
        }

    @pytest.mark.asyncio
    async def test_signal_generate_cancel_forwards_to_streaming_processor(self) -> None:
        executor, _reg = _make_executor()
        processor = MagicMock()
        processor.signal_cancel.return_value = True
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w") as srv:
            srv._get_streaming_processor = AsyncMock(return_value=processor)  # type: ignore[method-assign]
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("SignalGenerateCancel", {"request_id": "req-123"})
            finally:
                await client.close()

        assert resp["ok"] is True
        assert resp["body"] == {"matched": True}
        srv._get_streaming_processor.assert_awaited_once_with(prewarm=False)  # type: ignore[attr-defined]
        processor.signal_cancel.assert_called_once_with("req-123")

    @pytest.mark.asyncio
    async def test_signal_generate_cancel_does_not_prewarm(self) -> None:
        executor, _reg = _make_executor()
        processor = MagicMock()
        processor.signal_cancel.return_value = False
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="w")
        srv._streaming_processor = processor
        srv._prewarm_generation_grammars = AsyncMock()  # type: ignore[method-assign]

        resp = await srv._handle_signal_generate_cancel(
            SignalGenerateCancelRequest(request_id="req-123"),
        )

        assert resp.matched is False
        srv._prewarm_generation_grammars.assert_not_awaited()  # type: ignore[attr-defined]
        processor.signal_cancel.assert_called_once_with("req-123")

    @pytest.mark.asyncio
    async def test_cancel_created_processor_still_prewarms_on_generate(self) -> None:
        executor, _reg = _make_executor()
        processor = MagicMock()
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="w")
        srv._prewarm_generation_grammars = AsyncMock()  # type: ignore[method-assign]

        with patch("sie_server.processors.streaming.StreamingProcessor", return_value=processor):
            cancel_processor = await srv._get_streaming_processor(prewarm=False)
            generate_processor = await srv._get_streaming_processor()

        assert cancel_processor is processor
        assert generate_processor is processor
        srv._prewarm_generation_grammars.assert_awaited_once_with(processor)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_process_generate_sends_progress_before_lazy_processor(self) -> None:
        executor, _reg = _make_executor()
        processor = MagicMock()
        processor.process = AsyncMock()
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="w")
        writer = _CapturingWriter()
        work_item_msgpack = msgpack.packb({"request_id": "req-1"}, use_bin_type=True)

        async def get_processor() -> MagicMock:
            frames = _decode_written_frames(writer)
            assert [frame["body"]["kind"] for frame in frames] == ["in_progress"]
            return processor

        get_processor_mock = AsyncMock(side_effect=get_processor)
        srv._get_streaming_processor = get_processor_mock  # type: ignore[method-assign]

        await srv._handle_process_generate(
            ProcessGenerateRequest(model_id="test/model", work_item_msgpack=work_item_msgpack),
            request_id="ipc-req-1",
            writer=writer,  # type: ignore[arg-type]
        )

        frames = _decode_written_frames(writer)
        assert [frame["body"]["kind"] for frame in frames] == ["in_progress", "done"]
        assert [frame["request_id"] for frame in frames] == ["ipc-req-1", "ipc-req-1"]
        get_processor_mock.assert_awaited_once()
        processor.process.assert_awaited_once()


# -----------------------------------------------------------------------------
# Socket cleanup
# -----------------------------------------------------------------------------


class TestSocketLifecycle:
    @pytest.mark.asyncio
    async def test_stale_socket_is_unlinked_on_start(self) -> None:
        sock = _short_sock_path()
        sock.touch()
        assert sock.exists()
        executor, _reg = _make_executor()
        async with IpcServer(sock, executor, worker_id="w"):
            # Socket should exist but be of type socket now.
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("Ping", {"timestamp_ms": 0.0})
                assert resp["ok"] is True
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_socket_unlinked_on_stop(self) -> None:
        sock = _short_sock_path()
        executor, _reg = _make_executor()
        srv = IpcServer(sock, executor, worker_id="w")
        await srv.start()
        assert sock.exists()
        await srv.stop(drain_timeout_s=0.5)
        assert not sock.exists()


# -----------------------------------------------------------------------------
# Drain deadline propagation
# -----------------------------------------------------------------------------


class TestDrainDeadline:
    @pytest.mark.asyncio
    async def test_deadline_ms_sets_drain_timeout(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {"deadline_ms": 2500})
            assert resp["ok"] is True
            assert resp["body"]["acknowledged"] is True
            assert srv._drain_deadline_s == pytest.approx(2.5)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_missing_deadline_leaves_drain_unset(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {})
            assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_non_positive_deadline_is_ignored(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            for bad in (0, -1, -1000.5):
                resp = await client.rpc("Drain", {"deadline_ms": bad})
                assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bool_deadline_is_ignored(self, server_and_path) -> None:
        """`bool` is a subclass of `int` in Python, so `deadline_ms=True`
        would otherwise set a 1ms timeout and break shutdowns.
        """
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {"deadline_ms": True})
            assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_repeated_drain_tightens_deadline(self, server_and_path) -> None:
        """Second Drain with a smaller deadline must win; a later drain
        with a LARGER deadline must not loosen the stored budget — Rust
        may retry a Drain RPC and we can't let a retransmit extend the
        SIGTERM countdown.
        """
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            await client.rpc("Drain", {"deadline_ms": 5000}, request_id="a")
            assert srv._drain_deadline_s == pytest.approx(5.0)
            await client.rpc("Drain", {"deadline_ms": 1000}, request_id="b")
            assert srv._drain_deadline_s == pytest.approx(1.0)
            # Loose retry must not widen the window.
            await client.rpc("Drain", {"deadline_ms": 10_000}, request_id="c")
            assert srv._drain_deadline_s == pytest.approx(1.0)
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# Silence unused-import warning for IpcServerError; kept exported for future tests.
# -----------------------------------------------------------------------------

_ = IpcServerError
