"""One physical POST, with complete response evidence before SDK interpretation."""

from __future__ import annotations

import base64
import json
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path

import httpx
import msgpack
from sie_sdk.client._shared import modal_continuation_path

from common import OPTIONS, canonical, read_piece, redact, save_new, sha256

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
SAFE_HEADERS = {
    "content-type",
    "content-encoding",
    "content-length",
    "retry-after",
    "x-sie-server-version",
    "x-sie-model-revision",
    "x-sie-request-id",
    "x-sie-execution-identity-sha256",
    "x-sie-execution-binding-sha256",
    "x-sie-units-input-tokens",
    "x-sie-units-output-tokens",
    "x-sie-units-audio-ms",
}


class CaseDeadline(RuntimeError):
    pass


class ClosureFailure(RuntimeError):
    pass


class DispatchStopped(RuntimeError):
    pass


class IncompleteResponse(RuntimeError):
    def __init__(self, raw: bytes, decoded: bytes, reason: str) -> None:
        super().__init__("Response body is incomplete")
        self.raw = raw
        self.decoded = decoded
        self.reason = reason


def bounded_body(response: httpx.Response, deadline: float) -> tuple[bytes | None, bytes]:
    if response.is_stream_consumed:
        decoded = response.content
        if len(decoded) > MAX_RESPONSE_BYTES:
            raise IncompleteResponse(b"", b"", "ResponseLimit")
        return None, decoded  # Eager fixture clients expose decoded bytes only.
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    raw, decoded = bytearray(), bytearray()
    decoder = None
    if encoding == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decoder = zlib.decompressobj()
    elif encoding not in {"", "identity"}:
        raise IncompleteResponse(b"", b"", "UnsupportedContentEncoding")
    try:
        for chunk in response.iter_raw():
            if time.monotonic() >= deadline:
                raise CaseDeadline("Case deadline exhausted while reading response")
            if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
                raise ValueError("Encoded response byte limit")
            raw.extend(chunk)
            if decoder is None:
                data = chunk
            else:
                try:
                    data = decoder.decompress(chunk, MAX_RESPONSE_BYTES - len(decoded) + 1)
                except zlib.error:
                    if encoding != "deflate" or len(raw) != len(chunk):
                        raise
                    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
                    data = decoder.decompress(chunk, MAX_RESPONSE_BYTES + 1)
                if decoder.unconsumed_tail or decoder.unused_data:
                    raise ValueError("Compressed response exceeds limit or has trailing data")
            if len(decoded) + len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Decoded response byte limit")
            decoded.extend(data)
        if decoder is not None and not decoder.eof:
            raise ValueError("Truncated compressed response")
        return bytes(raw), bytes(decoded)
    except Exception as exc:
        raise IncompleteResponse(bytes(raw), bytes(decoded), type(exc).__name__) from exc


def decode_payload(body: bytes, content_type: str) -> dict | None:
    try:
        value = json.loads(body) if "json" in content_type else msgpack.unpackb(body, raw=False)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def response_result(record: dict) -> dict:
    """Reconstruct scientific output from a complete terminal response, never an SDK result."""
    result = {"status": "failed", "text": "", "http_status": record["status"], "halt": False}
    if not record["complete"] or record["status"] == 303:
        return {**result, "status": "UNKNOWN", "halt": True}
    payload = record.get("payload")
    result["response_payload"] = payload
    result["reported_revision"] = record["headers"].get("x-sie-model-revision")
    if isinstance(payload, dict):
        result["reported_model"] = payload.get("model")
        result["usage"] = payload.get("usage")
        items = payload.get("items")
        if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict):
            item = items[0]
            result["error"] = item.get("error")
            data = item.get("data")
            if isinstance(data, dict):
                result["data"] = data
                for key in ("output_tokens", "finish_reason", "cap_state"):
                    result[key] = data.get(key, item.get(key))
                if record["status"] == 200 and item.get("error") is None and isinstance(data.get("text"), str):
                    result.update(status="ok", text=data["text"])
    return result


class OnePost(httpx.BaseTransport):
    def __init__(
        self, piece: dict, folder: Path, settings: dict, halt, inner: httpx.BaseTransport | None = None
    ) -> None:
        self.piece, self.folder, self.settings, self.halt = piece, folder, settings, halt
        self.base = httpx.URL(settings["url"])
        self.path = self.base.path.rstrip("/") + "/v1/extract/" + settings["model"]
        self.inner = inner if inner is not None else httpx.HTTPTransport(retries=0, trust_env=False)
        self.sends = 0
        self.wire: list[dict] = []
        self.pending: httpx.URL | None = None

    def remaining(self) -> float:
        remaining = self.settings["case_deadline_at"] - time.monotonic()
        if remaining <= 0:
            raise CaseDeadline("Case deadline exhausted")
        return remaining

    def evidence(self, name: str, value: dict) -> None:
        try:
            save_new(self.folder / name, canonical(value))
        except Exception:
            self.halt.set()
            raise

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.remaining()
        if (request.url.scheme, request.url.host, request.url.port) != (
            self.base.scheme,
            self.base.host,
            self.base.port,
        ):
            raise ValueError("Cross-origin request rejected")
        if request.method == "POST":
            if not self.sends and self.halt.is_set():
                raise DispatchStopped("Dispatch halted before physical handoff")
            if self.sends or request.url.path != self.path or request.url.query:
                raise ValueError("Each piece permits exactly one intended extract POST")
            raw = request.read()
            if request.headers.get("content-type", "").split(";")[0] != "application/msgpack":
                raise ValueError("The stock MessagePack content type is required")
            audio = read_piece(Path(self.settings["pieces_dir"]), self.piece)
            expected = {"items": [{"audio": {"data": audio, "format": "wav", "sample_rate": None}}]}
            expected["params"] = {"options": OPTIONS}
            if msgpack.unpackb(raw, raw=False) != expected:
                raise ValueError("Stock SDK request differs from the qualified piece and fixed options")
            self.evidence(
                "intent.json",
                {
                    "identity": {
                        key: self.piece[key] for key in ("call_id", "piece_index", "start_sample", "end_sample")
                    },
                    "audio_sha256": sha256(audio),
                    "request_sha256": sha256(raw),
                    "request_bytes": len(raw),
                    "utc_start": datetime.now(UTC).isoformat(),
                    "monotonic_start": time.monotonic(),
                    "case_deadline_at": self.settings["case_deadline_at"],
                },
            )
            self.remaining()
            if self.halt.is_set():
                raise DispatchStopped("Dispatch halted after intent persistence")
            self.sends = 1
        elif request.method != "GET" or self.pending is None or request.url != self.pending:
            raise ValueError("Only the recorded same-origin continuation GET is permitted")
        self.pending = None
        response = self.inner.handle_request(request)
        index = len(self.wire)
        secret = self.settings["api_key"]
        headers = {key: value[:8192] for key, value in response.headers.items() if key in SAFE_HEADERS}
        safe_headers = redact(headers, secret)
        record = {
            "method": request.method,
            "status": response.status_code,
            "headers": safe_headers,
            "headers_known_secret_redacted": safe_headers != headers,
            "complete": False,
            "body_representation": "decoded",
            "monotonic_headers": time.monotonic(),
        }
        try:
            self.evidence(f"wire-{index:03d}-headers.json", record)
            try:
                raw_body, body = bounded_body(response, self.settings["case_deadline_at"])
                record["complete"] = True
            except IncompleteResponse as exc:
                raw_body, body = exc.raw, exc.decoded
                record["reason"] = exc.reason
            stored = body.replace(secret.encode(), b"[REDACTED]") if secret else body
            record.update(
                original_encoding=response.headers.get("content-encoding", "identity"),
                encoded_bytes=len(raw_body) if raw_body is not None else None,
                encoded_sha256=sha256(raw_body) if raw_body is not None else None,
                decoded_bytes=len(body),
                decoded_sha256=sha256(body),
                body_base64=base64.b64encode(stored).decode(),
                stored_sha256=sha256(stored),
                known_secret_redacted=stored != body,
                payload=redact(decode_payload(body, response.headers.get("content-type", "")), secret),
                monotonic_body_end=time.monotonic(),
            )
            self.evidence(f"wire-{index:03d}.json", record)
            self.wire.append(record)
            if not record["complete"]:
                self.halt.set()
                raise IncompleteResponse(b"", b"", record["reason"])
            self.remaining()
            continuation = modal_continuation_path(self.settings["url"], response)
            if continuation is not None:
                # Match the pinned stock SDK's HTTPX base-path merge, including a continuation query.
                relative = httpx.URL(continuation)
                self.pending = self.base.copy_with(
                    raw_path=self.base.raw_path.rstrip(b"/") + b"/" + relative.raw_path.lstrip(b"/")
                )
            headers = response.headers.copy()
            for name in ("content-encoding", "content-length"):
                headers.pop(name, None)
            return httpx.Response(response.status_code, content=body, headers=headers)
        finally:
            try:
                response.close()
            except Exception as exc:
                self.halt.set()
                raise ClosureFailure("Response closure failed") from exc

    def close(self) -> None:
        try:
            self.inner.close()
        except Exception as exc:
            self.halt.set()
            raise ClosureFailure("Transport closure failed") from exc


def recover(folder: Path) -> dict:
    terminal = folder / "terminal.json"
    if terminal.exists():
        try:
            row = json.loads(terminal.read_bytes())
            failure = folder / "worker-failure.json"
            if failure.exists():
                row.update(halt=True, reason=json.loads(failure.read_bytes())["reason"])
            return row
        except (ValueError, OSError):
            pass
    records = []
    for path in sorted(folder.glob("wire-[0-9][0-9][0-9].json")):
        try:
            records.append(json.loads(path.read_bytes()))
        except (ValueError, OSError):
            continue
    result = {"status": "known_unattempted", "text": "", "halt": True}
    if (folder / "intent.json").exists():
        result["status"] = "UNKNOWN"
        terminal_records = [record for record in records if record["complete"] and record["status"] != 303]
        if terminal_records:
            result = {**response_result(terminal_records[-1]), "halt": True, "reconstructed_from_response": True}
    result["physical_posts"] = (
        1
        if any(record["method"] == "POST" for record in records)
        else (None if (folder / "intent.json").exists() else 0)
    )
    result["closure_verified"] = False
    result["wire"] = records
    result["reason"] = "WorkerExitWithoutTerminal"
    return result
