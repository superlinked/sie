from __future__ import annotations

import asyncio
import contextvars
import hashlib
import logging
import os
import struct
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

import msgpack
import msgspec

from sie_server.adapter_call_loop import handle_run_batch
from sie_server.core.gpu_health import gpu_is_healthy_async
from sie_server.core.readiness import is_ready
from sie_server.ipc_types import (
    IPC_VERSION,
    METHOD_APPLY_MODEL_CONFIG,
    METHOD_DRAIN,
    METHOD_ENSURE_MODEL_READY,
    METHOD_PING,
    METHOD_PROCESS_ENCODE_BATCH,
    METHOD_PROCESS_EXTRACT_BATCH,
    METHOD_PROCESS_GENERATE,
    METHOD_PROCESS_SCORE_BATCH,
    METHOD_REPLACE_MODEL_CONFIGS,
    METHOD_RUN_BATCH,
    METHOD_SET_PINNED_MODELS,
    METHOD_SIGNAL_GENERATE_CANCEL,
    METHOD_WORKER_CAPABILITIES,
    ApplyModelConfigRequest,
    ApplyModelConfigResponse,
    BatchOutcome,
    DrainResponse,
    EnsureModelReadyRequest,
    EnsureModelReadyResponse,
    GenerateEvent,
    IpcResponseChunkV1,
    PingRequest,
    PingResponse,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessGenerateRequest,
    ProcessScoreBatchRequest,
    ReplaceModelConfigsRequest,
    ReplaceModelConfigsResponse,
    ResponseEnvelope,
    RunBatchRequest,
    SetPinnedModelsRequest,
    SetPinnedModelsResponse,
    SignalGenerateCancelRequest,
    SignalGenerateCancelResponse,
    WorkerCapabilitiesRequest,
    WorkerCapabilitiesResponse,
)
from sie_server.processors.admission import resolve_admission_enabled
from sie_server.queue_executor import QueueExecutor

logger = logging.getLogger(__name__)


# Frame layout: [4-byte big-endian unsigned length][msgpack body]
_LEN_STRUCT = struct.Struct("!I")
_LEN_BYTES = _LEN_STRUCT.size

# Upper bound on a single request/response frame body. 32 MiB is enough for
# any decoded WorkItem batch we would send in-band — large payloads arrive
# via the payload store, not via IPC.
_MAX_FRAME_BYTES = 32 * 1024 * 1024

# A negotiated response may exceed one legacy IPC frame, but remains tightly
# bounded so a malformed or unexpectedly large backend output cannot grow the
# colocated sidecar without limit. Four MiB leaves ample room for the named-map
# chunk envelope below the 32 MiB physical-frame ceiling.
_IPC_RESPONSE_CHUNK_PAYLOAD_BYTES = 4 * 1024 * 1024
_MAX_CHUNKED_IPC_RESPONSE_BYTES = 128 * 1024 * 1024
_MAX_IPC_RESPONSE_CHUNKS = 64

_GENERATE_SINK: contextvars.ContextVar[_IpcGenerateSink | None] = contextvars.ContextVar(
    "sie_generate_ipc_sink",
    default=None,
)


class IpcServerError(Exception):
    """Raised for protocol-level IPC failures (framing, encoding)."""


class IpcClientDisconnectedError(IpcServerError):
    """Raised when the IPC peer disconnects before a response is written."""


def _is_disconnected_write_error(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "closed" in message and ("transport" in message or "handler" in message)
    return False


class _IpcGenerateSink:
    """Serializes generation events back to the Rust sidecar over one IPC stream."""

    def __init__(self, server: IpcServer, writer: asyncio.StreamWriter, request_id: str) -> None:
        self._server = server
        self._writer = writer
        self._request_id = request_id
        self._lock = asyncio.Lock()

    async def send(self, event: GenerateEvent) -> None:
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id=self._request_id,
            ok=True,
            body=_to_builtins(event),
        )
        payload = msgpack.packb(_envelope_to_dict(envelope), use_bin_type=True)
        async with self._lock:
            await self._server._write_frame(self._writer, payload)

    async def publish(self, reply_subject: str, payload: bytes) -> None:
        await self.send(
            GenerateEvent(
                kind="publish",
                reply_subject=reply_subject,
                payload=payload,
            )
        )


class _IpcGenerateNatsShim:
    """NATS-like publisher used by ``StreamingProcessor`` in sidecar mode.

    The sidecar owns the real NATS client. Python still owns generation
    adapter execution and chunk encoding, so ``publish`` emits a typed IPC
    event and the Rust sidecar publishes the bytes to the gateway inbox.
    """

    async def publish(self, reply_subject: str, payload: bytes) -> None:
        sink = _GENERATE_SINK.get()
        if sink is None:
            raise RuntimeError("ProcessGenerate publish called without an active IPC sink")
        await sink.publish(reply_subject, payload)


class _IpcGenerateMessage:
    """Minimal JetStream-message facade for ``StreamingProcessor``.

    Python decides *what* settlement it wants; the Rust sidecar performs
    the actual JetStream ACK/NAK/progress on the consumed message.
    """

    def __init__(self, sink: _IpcGenerateSink, data: bytes) -> None:
        self.data = data
        self._sink = sink

    async def ack(self) -> None:
        await self._sink.send(GenerateEvent(kind="ack"))

    async def nak(self, delay: float | None = None) -> None:
        delay_ms: int | None = None
        if delay is not None:
            delay_ms = max(0, int(delay * 1000))
        await self._sink.send(GenerateEvent(kind="nak", delay_ms=delay_ms))

    async def in_progress(self) -> None:
        await self._sink.send(GenerateEvent(kind="in_progress"))


class IpcServer:
    """UDS msgpack RPC server fronting the QueueExecutor.

    Framing:
        [4-byte BE length][msgpack RequestEnvelope] -> [4-byte BE length][msgpack ResponseEnvelope]

    Lifecycle:
        async with IpcServer(path, executor) as server:
            await server.wait_closed()

    At most one Rust client is expected per worker pod, but the server accepts
    multiple concurrent connections so that restarting the worker-sidecar does
    not leave stale connections.
    """

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        executor: QueueExecutor,
        *,
        worker_id: str,
        stale_after_ms: float = 10_000.0,
        bundle_id: str | None = None,
    ) -> None:
        self._socket_path = Path(socket_path)
        self._executor = executor
        self._worker_id = worker_id
        self._stale_after_ms = stale_after_ms
        self._bundle_id = bundle_id or os.environ.get("SIE_BUNDLE", "")

        self._server: asyncio.AbstractServer | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._inflight: set[asyncio.Task[None]] = set()

        self._last_ping_monotonic: float | None = None
        self._drain_event = asyncio.Event()
        # Populated by the Rust caller on Drain via `deadline_ms`. Left as
        # `None` until then, in which case `stop()` falls back to its
        # caller-supplied default. Enforced in `stop()`.
        self._drain_deadline_s: float | None = None
        self._streaming_processor: Any | None = None
        self._streaming_processor_lock = asyncio.Lock()
        self._generation_prewarm_lock = asyncio.Lock()
        self._generation_prewarmed = False

    # -- Public surface ----------------------------------------------------

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def is_heartbeat_fresh(self, *, now_monotonic: float | None = None) -> bool:
        """Return True iff the Rust side has pinged within ``stale_after_ms``.

        Used by readiness: a stale heartbeat means the worker-sidecar is dead
        or wedged, and the pod should fail ``/readyz`` so Kubernetes reroutes.
        """
        if self._last_ping_monotonic is None:
            return False
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        elapsed_ms = (now_monotonic - self._last_ping_monotonic) * 1000
        return elapsed_ms <= self._stale_after_ms

    @property
    def last_ping_monotonic(self) -> float | None:
        return self._last_ping_monotonic

    async def start(self) -> None:
        """Bind the UDS and begin serving. Unlinks any stale socket file first."""
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError:
                logger.warning("Could not unlink existing socket at %s", self._socket_path)

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self._socket_path),
        )
        try:
            self._socket_path.chmod(0o600)
        except OSError:
            logger.debug("Could not chmod socket at %s", self._socket_path)
        logger.info("IPC server listening on %s", self._socket_path)

    async def stop(self, *, drain_timeout_s: float = 25.0) -> None:
        """Stop accepting new connections and drain in-flight requests."""
        self._drain_event.set()

        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5.0)
            except TimeoutError:
                logger.warning("IPC server wait_closed timed out")

        if self._inflight:
            # Prefer the Rust-supplied deadline when it is tighter than the
            # caller's default — the worker-sidecar sizes it from its own
            # shutdown budget so exceeding it guarantees a SIGKILL mid-IPC.
            effective_timeout = drain_timeout_s
            if self._drain_deadline_s is not None:
                effective_timeout = min(effective_timeout, self._drain_deadline_s)
            logger.info(
                "IPC server draining %d in-flight requests (timeout=%.1fs)",
                len(self._inflight),
                effective_timeout,
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._inflight, return_exceptions=True),
                    timeout=effective_timeout,
                )
            except TimeoutError:
                logger.warning("IPC drain timed out after %.1fs — cancelling", effective_timeout)
                for t in self._inflight:
                    t.cancel()

        for t in self._connections:
            t.cancel()

        try:
            if self._socket_path.exists():
                self._socket_path.unlink()
        except OSError:
            logger.debug("Could not unlink socket at %s on shutdown", self._socket_path)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop()

    # -- Connection / framing ----------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
            task.add_done_callback(self._connections.discard)

        peer = writer.get_extra_info("peername") or "unknown"
        logger.info("IPC connection accepted from %s", peer)

        try:
            while True:
                frame = await self._read_frame(reader)
                if frame is None:
                    return
                await self._dispatch_frame(frame, writer)
        except asyncio.CancelledError:
            raise
        except IpcClientDisconnectedError as e:
            logger.info("IPC client disconnected: %s", e)
        except IpcServerError as e:
            logger.warning("IPC protocol error: %s — closing connection", e)
        except Exception:
            logger.exception("IPC connection crashed")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing IPC writer", exc_info=True)

    async def _read_frame(self, reader: asyncio.StreamReader) -> bytes | None:
        """Read one length-prefixed frame. Returns None on clean EOF."""
        try:
            header = await reader.readexactly(_LEN_BYTES)
        except asyncio.IncompleteReadError as e:
            if not e.partial:
                return None
            raise IpcServerError("truncated frame header") from e

        (length,) = _LEN_STRUCT.unpack(header)
        if length == 0:
            return b""
        if length > _MAX_FRAME_BYTES:
            raise IpcServerError(f"frame too large: {length} > {_MAX_FRAME_BYTES}")

        try:
            return await reader.readexactly(length)
        except asyncio.IncompleteReadError as e:
            raise IpcServerError("truncated frame body") from e

    async def _write_frame(self, writer: asyncio.StreamWriter, payload: bytes) -> None:
        if len(payload) > _MAX_FRAME_BYTES:
            raise IpcServerError(f"frame too large: {len(payload)} > {_MAX_FRAME_BYTES}")
        try:
            writer.write(_LEN_STRUCT.pack(len(payload)) + payload)
            await writer.drain()
        except Exception as e:
            if _is_disconnected_write_error(e):
                raise IpcClientDisconnectedError("peer closed before IPC response write completed") from e
            raise

    # -- Dispatch ----------------------------------------------------------

    async def _dispatch_frame(self, frame: bytes, writer: asyncio.StreamWriter) -> None:
        try:
            raw = msgpack.unpackb(frame, raw=False)
        except Exception as e:
            raise IpcServerError(f"malformed msgpack: {e}") from e

        if not isinstance(raw, dict):
            raise IpcServerError("envelope must be a map")

        version = raw.get("version")
        if version != IPC_VERSION:
            await self._send_error(writer, raw.get("request_id", ""), f"unsupported version {version}")
            return

        method = raw.get("method")
        request_id = raw.get("request_id", "")
        body = raw.get("body") or {}
        # This is deliberately an exact, v1-only capability. Missing, false,
        # or non-boolean values retain the one-frame response contract, which
        # keeps new backends compatible with old sidecars during rolling
        # deploys.
        accepts_response_chunks_v1 = raw.get("accepts_ipc_response_chunks_v1") is True
        if not isinstance(method, str) or not isinstance(request_id, str) or not isinstance(body, dict):
            await self._send_error(writer, request_id, "envelope missing fields")
            return

        # Process methods (encode/score/extract) are dispatched in background
        # tasks so the Rust client can fire up to N in parallel if it ever
        # does. Ping/EnsureModelReady/Drain run inline — they are cheap.
        if method in (
            METHOD_PROCESS_ENCODE_BATCH,
            METHOD_PROCESS_SCORE_BATCH,
            METHOD_PROCESS_EXTRACT_BATCH,
            METHOD_RUN_BATCH,
            METHOD_PROCESS_GENERATE,
        ):
            if self._drain_event.is_set():
                await self._send_error(writer, request_id, "draining")
                return
            task = asyncio.create_task(
                self._run_method_task(
                    method,
                    request_id,
                    body,
                    writer,
                    accepts_response_chunks_v1=accepts_response_chunks_v1,
                )
            )
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)
            return

        await self._run_method(
            method,
            request_id,
            body,
            writer,
            accepts_response_chunks_v1=accepts_response_chunks_v1,
        )

    async def _run_method_task(
        self,
        method: str,
        request_id: str,
        body: dict[str, Any],
        writer: asyncio.StreamWriter,
        *,
        accepts_response_chunks_v1: bool = False,
    ) -> None:
        try:
            await self._run_method(
                method,
                request_id,
                body,
                writer,
                accepts_response_chunks_v1=accepts_response_chunks_v1,
            )
        except IpcClientDisconnectedError:
            logger.info("IPC client disconnected before %s response could be sent", method)
        except Exception:
            logger.exception("IPC method %s crashed", method)

    async def _run_method(
        self,
        method: str,
        request_id: str,
        body: dict[str, Any],
        writer: asyncio.StreamWriter,
        *,
        accepts_response_chunks_v1: bool = False,
    ) -> None:
        import msgspec  # noqa: PLC0415

        try:
            if method == METHOD_PROCESS_GENERATE:
                await self._handle_process_generate(
                    msgspec.convert(body, ProcessGenerateRequest),
                    request_id=request_id,
                    writer=writer,
                )
                return
            if method == METHOD_PING:
                resp_body = await self._handle_ping(msgspec.convert(body, PingRequest))
            elif method == METHOD_ENSURE_MODEL_READY:
                resp_body = await self._handle_ensure_ready(msgspec.convert(body, EnsureModelReadyRequest))
            elif method == METHOD_WORKER_CAPABILITIES:
                resp_body = self._handle_worker_capabilities(msgspec.convert(body, WorkerCapabilitiesRequest))
            elif method == METHOD_PROCESS_ENCODE_BATCH:
                resp_body = await self._handle_process_encode(msgspec.convert(body, ProcessEncodeBatchRequest))
            elif method == METHOD_PROCESS_SCORE_BATCH:
                resp_body = await self._handle_process_score(msgspec.convert(body, ProcessScoreBatchRequest))
            elif method == METHOD_PROCESS_EXTRACT_BATCH:
                resp_body = await self._handle_process_extract(msgspec.convert(body, ProcessExtractBatchRequest))
            elif method == METHOD_RUN_BATCH:
                # Rust scheduler drove the batch formation; we just fan
                # the items into the existing
                # per-op handlers. See adapter_call_loop.py for the
                # full fallback matrix.
                resp_body = await self._handle_run_batch(msgspec.convert(body, RunBatchRequest))
            elif method == METHOD_APPLY_MODEL_CONFIG:
                resp_body = await self._handle_apply_model_config(msgspec.convert(body, ApplyModelConfigRequest))
            elif method == METHOD_REPLACE_MODEL_CONFIGS:
                resp_body = await self._handle_replace_model_configs(msgspec.convert(body, ReplaceModelConfigsRequest))
            elif method == METHOD_SET_PINNED_MODELS:
                resp_body = await self._handle_set_pinned_models(msgspec.convert(body, SetPinnedModelsRequest))
            elif method == METHOD_SIGNAL_GENERATE_CANCEL:
                resp_body = await self._handle_signal_generate_cancel(
                    msgspec.convert(body, SignalGenerateCancelRequest)
                )
            elif method == METHOD_DRAIN:
                # The Rust caller may pass `deadline_ms` to indicate how
                # long it is willing to wait before SIGTERM. We don't
                # block here (that would hold the IPC response), but we
                # record the deadline so `stop()` can cap its in-flight
                # drain instead of using the hard-coded default.
                #
                # Repeated Drain calls (e.g. a retry after a transient
                # IPC glitch) must tighten, never loosen, the effective
                # shutdown budget — the worker-sidecar's own budget is the
                # wall-clock floor. `bool` is also an `int`, so reject
                # it explicitly or `deadline_ms=True` would set a
                # 1ms timeout.
                deadline_ms = body.get("deadline_ms")
                if not isinstance(deadline_ms, bool) and isinstance(deadline_ms, (int, float)) and deadline_ms > 0:
                    new_deadline_s = float(deadline_ms) / 1000.0
                    if self._drain_deadline_s is None:
                        self._drain_deadline_s = new_deadline_s
                    else:
                        self._drain_deadline_s = min(self._drain_deadline_s, new_deadline_s)
                self._drain_event.set()
                resp_body = DrainResponse(acknowledged=True)
            else:
                await self._send_error(writer, request_id, f"unknown method {method}")
                return
        except msgspec.ValidationError as e:
            await self._send_error(writer, request_id, f"bad body for {method}: {e}")
            return
        except Exception as e:
            logger.exception("IPC method %s failed", method)
            await self._send_error(writer, request_id, f"{type(e).__name__}: {e}")
            return

        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id=request_id,
            ok=True,
            body=_to_builtins(resp_body),
        )
        await self._write_response_envelope(
            writer,
            envelope,
            accepts_response_chunks_v1=accepts_response_chunks_v1,
        )

    async def _write_response_envelope(
        self,
        writer: asyncio.StreamWriter,
        envelope: ResponseEnvelope,
        *,
        accepts_response_chunks_v1: bool,
    ) -> None:
        """Write one response frame or a negotiated sequence of v1 chunks.

        Small responses take the legacy path byte-for-byte. For an oversized
        response, the chunk payloads are slices of the exact bytes that would
        otherwise have formed that single legacy frame.
        """
        payload = msgpack.packb(_envelope_to_dict(envelope), use_bin_type=True)
        if len(payload) <= _MAX_FRAME_BYTES:
            await self._write_frame(writer, payload)
            return

        if not accepts_response_chunks_v1:
            await self._send_error(
                writer,
                envelope.request_id,
                "IPC response exceeds the legacy frame limit; response chunking v1 was not negotiated",
            )
            return

        chunk_count = (len(payload) + _IPC_RESPONSE_CHUNK_PAYLOAD_BYTES - 1) // _IPC_RESPONSE_CHUNK_PAYLOAD_BYTES
        if len(payload) > _MAX_CHUNKED_IPC_RESPONSE_BYTES or chunk_count > _MAX_IPC_RESPONSE_CHUNKS:
            await self._send_error(
                writer,
                envelope.request_id,
                "IPC response exceeds the negotiated response chunking v1 bounds",
            )
            return

        digest = hashlib.sha256(payload).digest()
        payload_view = memoryview(payload)
        for chunk_index in range(chunk_count):
            start = chunk_index * _IPC_RESPONSE_CHUNK_PAYLOAD_BYTES
            chunk = IpcResponseChunkV1(
                version=IPC_VERSION,
                request_id=envelope.request_id,
                transfer_digest=digest,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                total_bytes=len(payload),
                # The wire field remains typed as bytes so msgspec can build
                # decoders/schemas for the shared contract. msgpack also
                # accepts this bytes-like view and copies it only into the
                # physical frame.
                payload=cast("bytes", payload_view[start : start + _IPC_RESPONSE_CHUNK_PAYLOAD_BYTES]),
            )
            chunk_payload = msgpack.packb(_to_builtins(chunk), use_bin_type=True)
            # Keep this check even though the configured payload target is far
            # below the frame ceiling: it makes future envelope growth fail
            # closed instead of silently violating the physical frame limit.
            if len(chunk_payload) > _MAX_FRAME_BYTES:
                raise IpcServerError(f"response chunk frame too large: {len(chunk_payload)} > {_MAX_FRAME_BYTES}")
            await self._write_frame(writer, chunk_payload)

    # -- Handlers ----------------------------------------------------------

    async def _handle_ping(self, req: PingRequest) -> PingResponse:
        self._last_ping_monotonic = time.monotonic()
        ready = False
        try:
            ready = is_ready() and await gpu_is_healthy_async()
        except Exception:  # noqa: BLE001
            logger.debug("Could not compute readiness for ping", exc_info=True)
        bundle_config_hash = ""
        if self._bundle_id:
            try:
                bundle_config_hash = self._executor.compute_bundle_config_hash(self._bundle_id)
            except Exception:  # noqa: BLE001
                logger.debug("Could not compute bundle_config_hash for ping", exc_info=True)
        loaded_models: list[str] = []
        try:
            loaded_models = self._executor.loaded_model_names()
        except Exception:  # noqa: BLE001
            logger.debug("Could not compute loaded_models for ping", exc_info=True)
        return PingResponse(
            timestamp_ms=req.timestamp_ms,
            worker_id=self._worker_id,
            ready=ready,
            bundle_config_hash=bundle_config_hash,
            loaded_models=loaded_models,
        )

    async def _handle_ensure_ready(self, req: EnsureModelReadyRequest) -> EnsureModelReadyResponse:
        state = await self._executor.ensure_model_ready(req.model_id)
        # Only populate batch_budget + descriptor on the "ready" path — the
        # worker isn't guaranteed to exist in other states and the Rust
        # side will re-query after the NAK delay anyway.
        batch_budget: int | None = None
        descriptor = None
        if state == "ready":
            batch_budget = self._executor.get_batch_budget(req.model_id)
            descriptor = self._executor.get_model_descriptor(req.model_id)
        return EnsureModelReadyResponse(
            state=state,
            batch_budget=batch_budget,
            descriptor=descriptor,
        )

    async def _handle_process_encode(self, req: ProcessEncodeBatchRequest) -> BatchOutcome:
        return await self._executor.process_encode_batch(req)

    async def _handle_process_score(self, req: ProcessScoreBatchRequest) -> BatchOutcome:
        return await self._executor.process_score_batch(req)

    async def _handle_process_extract(self, req: ProcessExtractBatchRequest) -> BatchOutcome:
        return await self._executor.process_extract_batch(req)

    async def _handle_run_batch(self, req: RunBatchRequest) -> BatchOutcome:
        return await handle_run_batch(self._executor, req)

    async def _handle_apply_model_config(self, req: ApplyModelConfigRequest) -> ApplyModelConfigResponse:
        return await self._executor.apply_model_config(req)

    async def _handle_replace_model_configs(
        self,
        req: ReplaceModelConfigsRequest,
    ) -> ReplaceModelConfigsResponse:
        return await self._executor.replace_model_configs(req)

    async def _handle_set_pinned_models(self, req: SetPinnedModelsRequest) -> SetPinnedModelsResponse:
        return await self._executor.set_pinned_models(req)

    def _handle_worker_capabilities(self, _req: WorkerCapabilitiesRequest) -> WorkerCapabilitiesResponse:
        generation_models: list[str] = []
        supported_models: list[str] = []
        loaded_models: list[str] = []
        try:
            configs = self._executor.registry.get_configs_snapshot()
        except Exception:  # noqa: BLE001
            logger.debug("Could not snapshot configs for worker capabilities", exc_info=True)
            return WorkerCapabilitiesResponse()

        supported_models = sorted(configs)
        try:
            loaded_models = sorted(self._executor.registry.loaded_model_names)
        except Exception:  # noqa: BLE001
            logger.debug("Could not snapshot loaded models for worker capabilities", exc_info=True)

        for model_id, config in configs.items():
            try:
                if config.tasks.generate is not None:
                    generation_models.append(model_id)
            except Exception:  # noqa: BLE001
                logger.debug("Could not inspect generation task for %s", model_id, exc_info=True)

        generation_models.sort()
        return WorkerCapabilitiesResponse(
            has_generation_models=bool(generation_models),
            generation_models=generation_models,
            supported_models=supported_models,
            loaded_models=loaded_models,
        )

    async def _handle_signal_generate_cancel(
        self,
        req: SignalGenerateCancelRequest,
    ) -> SignalGenerateCancelResponse:
        if not req.request_id:
            return SignalGenerateCancelResponse(matched=False)
        processor = await self._get_streaming_processor(prewarm=False)
        return SignalGenerateCancelResponse(matched=bool(processor.signal_cancel(req.request_id)))

    async def _handle_process_generate(
        self,
        req: ProcessGenerateRequest,
        *,
        request_id: str,
        writer: asyncio.StreamWriter,
    ) -> None:
        sink = _IpcGenerateSink(self, writer, request_id)
        msg = _IpcGenerateMessage(sink, req.work_item_msgpack)
        token = _GENERATE_SINK.set(sink)
        try:
            # Keep the JetStream lease alive while the first generation
            # request lazily constructs the streaming processor and prewarms
            # grammars. The processor's own heartbeat starts later, once
            # request processing reaches the decode path.
            await msg.in_progress()
            processor = await self._get_streaming_processor()
            await processor.process(msg, req.model_id)
            await sink.send(GenerateEvent(kind="done"))
        finally:
            _GENERATE_SINK.reset(token)

    async def _get_streaming_processor(self, *, prewarm: bool = True) -> Any:
        if self._streaming_processor is not None:
            processor = self._streaming_processor
            if prewarm:
                await self._prewarm_generation_grammars(processor)
            return processor
        async with self._streaming_processor_lock:
            if self._streaming_processor is None:
                from sie_server.processors.streaming import StreamingProcessor  # noqa: PLC0415

                self._streaming_processor = StreamingProcessor(
                    nc=_IpcGenerateNatsShim(),  # type: ignore[arg-type]
                    registry=self._executor.registry,
                    worker_id=self._worker_id,
                    admission_resolver=self._resolve_generation_admission,
                )
            processor = self._streaming_processor
        if prewarm:
            await self._prewarm_generation_grammars(processor)
        return processor

    def _resolve_generation_admission(self, model_id: str) -> tuple[int | None, bool | None]:
        try:
            config = self._executor.registry.get_config(model_id)
        except KeyError:
            return None, None
        try:
            if config.tasks.generate is None:
                return None, None
            resolved = config.resolve_profile("default")
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve generation admission config for %s", model_id, exc_info=True)
            return None, None

        budget = resolved.kv_budget_tokens
        if not isinstance(budget, int) or budget <= 0:
            budget = None
        profile_admission = resolved.admission_enabled
        if not isinstance(profile_admission, bool):
            profile_admission = None
        return budget, resolve_admission_enabled(profile_admission=profile_admission)

    async def _prewarm_generation_grammars(self, processor: Any) -> None:
        async with self._generation_prewarm_lock:
            if self._generation_prewarmed:
                return
            self._generation_prewarmed = True
            try:
                configs = self._executor.registry.get_configs_snapshot()
            except Exception:  # noqa: BLE001
                logger.debug("Could not snapshot configs for generation grammar prewarm", exc_info=True)
                return
            for model_id, config in configs.items():
                try:
                    if config.tasks.generate is None:
                        continue
                    await processor.prewarm_grammars_for_model(model_id)
                except Exception:  # noqa: BLE001
                    logger.warning("Generation grammar prewarm failed for %s", model_id, exc_info=True)

    # -- Error helper ------------------------------------------------------

    async def _send_error(self, writer: asyncio.StreamWriter, request_id: str, message: str) -> None:
        """Send a ``ok=False`` error response.

        The Rust client surfaces these as ``IpcError::Server(...)`` and
        treats them as logical (non-transport) failures, so no reconnect
        or retry is attempted.
        """
        envelope = ResponseEnvelope(
            version=IPC_VERSION,
            request_id=request_id,
            ok=False,
            body=None,
            error=message,
        )
        await self._write_frame(writer, msgpack.packb(_envelope_to_dict(envelope), use_bin_type=True))


# ---------------------------------------------------------------------------
# msgspec → plain dict helpers for msgpack wire format
# ---------------------------------------------------------------------------
#
# We don't use ``msgspec.msgpack.encode`` directly because the Rust side
# expects unadorned msgpack (no ``msgspec`` tag field on nested structs), so
# we convert via ``msgspec.to_builtins`` first.


def _to_builtins(obj: Any) -> Any:
    """Convert a msgspec Struct tree to plain builtins for msgpack.

    Preserving ``bytes`` and ``memoryview`` is critical: by default
    ``to_builtins`` base64-encodes binary values to ``str`` for JSON
    compatibility, but this wire is msgpack-native. Response chunk payloads
    deliberately use borrowed ``memoryview`` slices to avoid an intermediate
    full chunk copy before ``msgpack.packb`` writes the physical frame.
    """
    if obj is None:
        return None
    return msgspec.to_builtins(obj, builtin_types=(bytes, memoryview))


def _envelope_to_dict(envelope: ResponseEnvelope) -> dict[str, Any]:
    return {
        "version": envelope.version,
        "request_id": envelope.request_id,
        "ok": envelope.ok,
        "body": envelope.body,
        "error": envelope.error,
    }
