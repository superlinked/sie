"""HTTP rules shared by the remote adapters.

The upstream is outside this deployment, so its answer is untrusted. A body is
read uncompressed under a size cap and a deadline. A failed call becomes fixed
text: the status, a known error code and ``Retry-After`` are read only to
decide whether the same request may succeed later, and nothing else the
upstream sent is kept.

A failure the same request may survive raises :class:`UpstreamUnavailableError`:
the model is not ready upstream (``MODEL_LOADING``, ``PROVISIONING``,
``LORA_LOADING``), the upstream is busy (429, ``QUEUE_FULL``,
``RESOURCE_EXHAUSTED``, ``BILLING_CAPACITY_UNAVAILABLE``, or
``QUEUE_UNAVAILABLE`` with a ``Retry-After``), or it is unavailable (no
connection, a timeout, a dropped response, another 5xx). A 501, a 505 and the
codes ``MODEL_LOAD_FAILED``, ``ACCOUNT_STATE_UNAVAILABLE`` and
``COLD_START_RATE_LIMITED`` are final. A refused input raises
:class:`InputTooLongError` or :class:`InvalidInputError`. Everything else is a
:class:`RemoteUpstreamError`.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
from sie_sdk._msgpack import unpackb

from sie_server.adapters._generation_base import GenerationCapacityError, GenerationDrainingError, GenerationError
from sie_server.adapters.errors import (
    RETRY_AFTER_MAX_S,
    RETRY_AFTER_MIN_S,
    InputTooLongError,
    UpstreamUnavailableError,
)
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.config.upstreams import UpstreamCredentialError, upstream_for_serving
from sie_server.core.upstream_client import UpstreamRedirectRefusedError
from sie_server.types.inputs import InvalidInputError

DEFAULT_RETRY_AFTER_S = 5
_ERROR_BODY_MAX_BYTES = 64 << 10
_NOT_READY_CODES = frozenset({"MODEL_LOADING", "PROVISIONING", "LORA_LOADING"})
_BUSY_CODES = frozenset({"QUEUE_FULL", "RESOURCE_EXHAUSTED", "BILLING_CAPACITY_UNAVAILABLE"})
_BUSY_WITH_RETRY_AFTER_CODES = frozenset({"QUEUE_UNAVAILABLE", "transport_failure"})
_TERMINAL_CODES = frozenset({"MODEL_LOAD_FAILED", "ACCOUNT_STATE_UNAVAILABLE", "COLD_START_RATE_LIMITED"})
_TERMINAL_SERVER_STATUSES = frozenset({501, 505})
_KNOWN_ERROR_CODES = (
    _NOT_READY_CODES
    | _BUSY_CODES
    | _BUSY_WITH_RETRY_AFTER_CODES
    | _TERMINAL_CODES
    | frozenset(
        {
            "ACCOUNT_PENDING_REVIEW",
            "ACCOUNT_SUSPENDED",
            "GATEWAY_TIMEOUT",
            "INFERENCE_ERROR",
            "INPUT_TOO_LONG",
            "INSUFFICIENT_CREDITS",
            "INTERNAL_ERROR",
            "INVALID_INPUT",
            "KEY_SPEND_LIMIT_EXCEEDED",
            "MODEL_NOT_FOUND",
            "MODEL_NOT_LOADED",
            "RATE_LIMIT",
        }
    )
)


class RemoteUpstreamError(RuntimeError):
    """The upstream call failed in a way a retry would not fix, or its answer broke the profile's contract."""


def send_bounded(
    client: httpx.Client,
    request: httpx.Request,
    *,
    upstream: str,
    max_bytes: int,
    deadline_s: float,
) -> bytes:
    """Send ``request`` to ``upstream`` and return the body of a successful answer.

    The call goes through the upstream's limiter, which may refuse it without
    sending (see :mod:`sie_server.adapters.remote._limits`). Raises
    :class:`UpstreamUnavailableError` when the same request may succeed later,
    :class:`InputTooLongError` or :class:`InvalidInputError` when the upstream
    refused the input, and :class:`RemoteUpstreamError` otherwise.
    """
    with upstream_limiter(upstream).call():
        return _send(client, request, upstream=upstream, max_bytes=max_bytes, deadline=time.monotonic() + deadline_s)


def _send(client: httpx.Client, request: httpx.Request, *, upstream: str, max_bytes: int, deadline: float) -> bytes:
    try:
        response = client.send(request, stream=True)
    except UpstreamCredentialError:
        raise RemoteUpstreamError("the upstream credential is unavailable") from None
    except httpx.HTTPError as exc:
        raise failure_for_transport_error(exc, upstream=upstream) from None
    try:
        if response.status_code >= 400:
            raise failure_for_status(
                response.status_code,
                upstream_error_code(response.headers, _read_error_body(response, deadline)),
                parse_retry_after(response.headers.get("retry-after")),
                upstream=upstream,
            )
        if _is_compressed(response):
            raise RemoteUpstreamError("upstream sent a compressed body, which is refused")
        return _read_body(response, upstream=upstream, max_bytes=max_bytes, deadline=deadline)
    except httpx.HTTPError as exc:
        raise failure_for_transport_error(exc, upstream=upstream) from None
    finally:
        response.close()


def failure_for_status(status: int, code: str | None, retry_after_s: int | None, *, upstream: str) -> Exception:
    """The failure an upstream error status amounts to, given its known error code and wait."""
    answered = f"answered {status} {code}" if code else f"answered {status}"
    wait = retry_after_s if retry_after_s is not None else DEFAULT_RETRY_AFTER_S
    if code in _TERMINAL_CODES or status in _TERMINAL_SERVER_STATUSES:
        return RemoteUpstreamError(f"upstream {answered}")
    if status == 429:
        return UpstreamUnavailableError(upstream, "busy", retry_after_s=wait, reason=answered)
    if status >= 500:
        if code in _NOT_READY_CODES:
            return UpstreamUnavailableError(upstream, "not_ready", retry_after_s=wait, reason=answered)
        if code in _BUSY_CODES or (code in _BUSY_WITH_RETRY_AFTER_CODES and retry_after_s is not None):
            return UpstreamUnavailableError(upstream, "busy", retry_after_s=wait, reason=answered)
        return UpstreamUnavailableError(upstream, "unavailable", retry_after_s=wait, reason=answered)
    if code == "INPUT_TOO_LONG":
        return InputTooLongError(f"the upstream refused the input as too long ({status} INPUT_TOO_LONG)")
    if code == "INVALID_INPUT" and status in {400, 422}:
        return InvalidInputError(f"the upstream refused the input ({status} INVALID_INPUT)")
    return RemoteUpstreamError(f"upstream {answered}")


def failure_for_transport_error(exc: httpx.HTTPError, *, upstream: str) -> Exception:
    """The failure a client-side error amounts to: retryable unless the request itself was at fault."""
    if isinstance(exc, UpstreamRedirectRefusedError):
        return RemoteUpstreamError("upstream answered with a redirect, which is refused")
    failed = f"request failed ({type(exc).__name__})"
    if isinstance(exc, httpx.TransportError) and not isinstance(
        exc, httpx.UnsupportedProtocol | httpx.LocalProtocolError
    ):
        return UpstreamUnavailableError(
            upstream,
            "unavailable",
            retry_after_s=DEFAULT_RETRY_AFTER_S,
            reason=failed,
        )
    return RemoteUpstreamError(f"upstream {failed}")


@asynccontextmanager
async def open_stream(
    client: httpx.AsyncClient, request: httpx.Request, *, upstream: str, error_body_timeout_s: float
) -> AsyncIterator[httpx.Response]:
    """Send ``request`` to ``upstream`` and hold its streamed answer open for the block.

    The call goes through the upstream's limiter, whose slot is held until the
    block ends. The answer is a success with an uncompressed body, which the
    block reads. Failures raise as :func:`send_bounded` describes, including a
    connection lost while the block reads. Leaving the block closes the answer,
    which cancels work still running upstream.
    """
    upstream_for_serving(upstream)
    with upstream_limiter(upstream).call():
        try:
            response = await client.send(request, stream=True)
        except UpstreamCredentialError:
            raise RemoteUpstreamError("the upstream credential is unavailable") from None
        except httpx.HTTPError as exc:
            raise failure_for_transport_error(exc, upstream=upstream) from None
        try:
            if response.status_code >= 400:
                raise failure_for_status(
                    response.status_code,
                    upstream_error_code(
                        response.headers, await _read_error_body_async(response, timeout_s=error_body_timeout_s)
                    ),
                    parse_retry_after(response.headers.get("retry-after")),
                    upstream=upstream,
                )
            if _is_compressed(response):
                raise RemoteUpstreamError("upstream sent a compressed body, which is refused")
            yield response
        except httpx.HTTPError as exc:
            raise failure_for_transport_error(exc, upstream=upstream) from None
        finally:
            await response.aclose()


async def sse_data(response: httpx.Response, *, max_event_bytes: int, max_total_bytes: int) -> AsyncIterator[bytes]:
    """Read bounded SSE events, scanning new bytes once and dispatching CR immediately."""
    pending = bytearray()
    data: list[bytes] = []
    event_bytes = received = 0
    skip_lf = False
    async for chunk in response.aiter_raw():
        received += len(chunk)
        if received > max_total_bytes:
            raise RemoteUpstreamError("upstream stream exceeds the size limit")
        start = 0
        if skip_lf and chunk:
            start = int(chunk[0] == 10)
            skip_lf = False
        origin = start
        for ending in re.finditer(rb"\r\n?|\n", chunk[origin:]):
            end = origin + ending.start()
            # CRLF is one line ending, including when split across chunks.
            pending.extend(chunk[start:end])
            event_bytes += len(pending) + 1
            if event_bytes > max_event_bytes:
                raise RemoteUpstreamError("upstream sent an event over the size limit")
            line = bytes(pending)
            pending.clear()
            next_start = origin + ending.end()
            start = next_start
            skip_lf = next_start == len(chunk) and chunk[next_start - 1] == 13
            if not line:
                if data:
                    yield b"\n".join(data)
                data, event_bytes = [], 0
            else:
                field, _, value = line.partition(b":")
                if field == b"data":
                    data.append(value.removeprefix(b" "))
        pending.extend(chunk[start:])
        if event_bytes + len(pending) > max_event_bytes:
            raise RemoteUpstreamError("upstream sent an event over the size limit")


def generation_error(error: UpstreamUnavailableError) -> GenerationError:
    """Map an upstream failure before output to a retryable 503 with its wait."""
    if error.kind == "not_ready":
        return GenerationDrainingError("the upstream is not ready, please retry", retry_after_s=error.retry_after_s)
    return GenerationCapacityError(f"the upstream is {error.kind}, please retry", retry_after_s=error.retry_after_s)


def upstream_error_code(headers: httpx.Headers, body: bytes) -> str | None:
    """The upstream's error code when it is one this server knows, else ``None``.

    Read from ``X-SIE-Error-Code`` first, then from the ``error`` or ``detail``
    object of a JSON or msgpack body, as the SDKs read it.
    """
    from_header = _known_code(headers.get("x-sie-error-code"))
    if from_header is not None:
        return from_header
    if not body:
        return None
    try:
        if "msgpack" in headers.get("content-type", "").lower():
            decoded = unpackb(body, numeric_arrays=False)
        else:
            decoded = json.loads(body)
    except Exception:  # noqa: BLE001 - untrusted bytes; an unreadable body carries no code
        return None
    if not isinstance(decoded, dict):
        return None
    for key in ("error", "detail"):
        envelope = decoded.get(key)
        if isinstance(envelope, dict):
            return _known_code(envelope.get("code"))
    return None


def parse_retry_after(value: str | None) -> int | None:
    """Whole seconds to wait from a ``Retry-After`` value, clamped to the bounded range.

    Accepts delay seconds or an HTTP date; a date in the past means the
    shortest wait. ``None`` when the value is absent or unusable.
    """
    text = (value or "").strip()
    if not text:
        return None
    try:
        seconds: float | None = float(text)
    except ValueError:
        seconds = _seconds_until(text)
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return None
    return min(RETRY_AFTER_MAX_S, max(RETRY_AFTER_MIN_S, math.ceil(seconds)))


def _seconds_until(http_date: str) -> float | None:
    try:
        when = parsedate_to_datetime(http_date)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def _known_code(value: object) -> str | None:
    if value == "provisioning":
        return "PROVISIONING"
    return value if isinstance(value, str) and value in _KNOWN_ERROR_CODES else None


def _is_compressed(response: httpx.Response) -> bool:
    return response.headers.get("content-encoding", "identity").strip().lower() not in {"", "identity"}


def _read_body(response: httpx.Response, *, upstream: str, max_bytes: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_raw():
        size += len(chunk)
        if size > max_bytes:
            raise RemoteUpstreamError("upstream response exceeds the size limit")
        if time.monotonic() > deadline:
            raise UpstreamUnavailableError(
                upstream,
                "unavailable",
                retry_after_s=DEFAULT_RETRY_AFTER_S,
                reason="the response exceeded the deadline",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_error_body_async(response: httpx.Response, *, timeout_s: float) -> bytes:
    if _is_compressed(response):
        return b""
    chunks: list[bytes] = []
    size = 0
    try:
        async with asyncio.timeout(timeout_s):
            async for chunk in response.aiter_raw():
                size += len(chunk)
                if size > _ERROR_BODY_MAX_BYTES:
                    return b""
                chunks.append(chunk)
    except (httpx.HTTPError, TimeoutError):
        return b""
    return b"".join(chunks)


def _read_error_body(response: httpx.Response, deadline: float) -> bytes:
    if _is_compressed(response):
        return b""
    chunks: list[bytes] = []
    size = 0
    try:
        for chunk in response.iter_raw():
            size += len(chunk)
            if size > _ERROR_BODY_MAX_BYTES or time.monotonic() > deadline:
                return b""
            chunks.append(chunk)
    except httpx.HTTPError:
        return b""
    return b"".join(chunks)
