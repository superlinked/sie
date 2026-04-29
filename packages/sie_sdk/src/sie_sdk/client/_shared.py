from __future__ import annotations

import errno
import logging
import socket
import ssl
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any, Protocol

import msgpack
import numpy as np

_logger = logging.getLogger(__name__)


class _HttpResponse(Protocol):
    """Structural type for HTTP responses (httpx.Response or _AioResponse)."""

    status_code: int
    content: bytes
    headers: Any

    @property
    def text(self) -> str: ...
    def json(self) -> Any: ...


from sie_sdk.types import (
    Classification,
    DetectedObject,
    EncodeResult,
    EntityResult,
    ExtractResult,
    Relation,
    ScoreEntry,
    ScoreResult,
    SparseResult,
)

from .errors import ModelLoadFailedError, RequestError, ServerError

# Content types
MSGPACK_CONTENT_TYPE = "application/msgpack"
JSON_CONTENT_TYPE = "application/json"

# HTTP status code thresholds
HTTP_ACCEPTED = 202
HTTP_CLIENT_ERROR = 400
HTTP_SERVER_ERROR = 500
HTTP_GATEWAY_TIMEOUT = 504

# Default provisioning settings
DEFAULT_PROVISION_TIMEOUT_S = 900.0  # 15 minutes
DEFAULT_RETRY_DELAY_S = 5.0  # Retry every 5 seconds if no Retry-After header

# Pool settings
DEFAULT_LEASE_RENEWAL_INTERVAL_S = 60.0  # Renew lease every 60s (lease is 1200s)

# LoRA loading retry settings
LORA_LOADING_MAX_RETRIES = 10  # Max retries for LoRA loading (usually completes in 1-2s)
LORA_LOADING_DEFAULT_DELAY_S = 1.0  # Default retry delay if no Retry-After header
LORA_LOADING_ERROR_CODE = "LORA_LOADING"  # Error code from server

# Model loading retry settings
MODEL_LOADING_MAX_RETRIES = 60  # Max retries (60 * 5s = 5 min, matches provision timeout)
MODEL_LOADING_DEFAULT_DELAY_S = 5.0  # Default retry delay (model loads take longer than LoRA)
MODEL_LOADING_ERROR_CODE = "MODEL_LOADING"  # Error code from server

# Terminal load failure (non-retryable). Server returns this with HTTP 502
# and *no* Retry-After header so the SDK can short-circuit immediately
# instead of burning the MODEL_LOADING retry budget.
MODEL_LOAD_FAILED_ERROR_CODE = "MODEL_LOAD_FAILED"

# Resource-exhausted retry settings (server-side OOM recovery exhausted).
# Default backoff sequence: 5 -> 10 -> 20 s (capped at 30s). Three attempts
# is enough to cover the typical eviction + retry window without making
# pathological cases hang indefinitely. Callers can opt out with
# ``max_oom_retries=0``.
RESOURCE_EXHAUSTED_MAX_RETRIES = 3
RESOURCE_EXHAUSTED_DEFAULT_DELAY_S = 5.0
RESOURCE_EXHAUSTED_MAX_DELAY_S = 30.0
RESOURCE_EXHAUSTED_ERROR_CODE = "RESOURCE_EXHAUSTED"

# Version negotiation headers
SDK_VERSION_HEADER = "X-SIE-SDK-Version"
SERVER_VERSION_HEADER = "X-SIE-Server-Version"


def get_sdk_version() -> str:
    try:
        return pkg_version("sie-sdk")
    except PackageNotFoundError:
        return "unknown"


def check_version_skew(sdk_version: str, server_version: str) -> str | None:
    try:
        sdk_parts = sdk_version.split(".")
        server_parts = server_version.split(".")
        if len(sdk_parts) < 2 or len(server_parts) < 2:
            return None

        sdk_major, sdk_minor = int(sdk_parts[0]), int(sdk_parts[1])
        server_major, server_minor = int(server_parts[0]), int(server_parts[1])

        if sdk_major != server_major:
            return (
                f"SDK version {sdk_version} has different major version than server {server_version}. Please upgrade."
            )

        if abs(sdk_minor - server_minor) > 1:
            return (
                f"SDK version {sdk_version} is more than one minor version "
                f"{'behind' if sdk_minor < server_minor else 'ahead of'} "
                f"server {server_version}. Consider upgrading."
            )
    except (ValueError, IndexError):
        pass
    return None


def parse_gpu_param(gpu: str) -> tuple[str | None, str]:
    """Parse GPU parameter to extract pool name and GPU type.

    Args:
        gpu: GPU string, either "pool_name/gpu_type" or just "gpu_type".

    Returns:
        Tuple of (pool_name, gpu_type). pool_name is None if not specified.

    Examples:
        >>> parse_gpu_param("eval-bench/l4")
        ("eval-bench", "l4")
        >>> parse_gpu_param("l4")
        (None, "l4")
    """
    if "/" in gpu:
        parts = gpu.split("/", 1)
        return parts[0], parts[1]
    return None, gpu


# Errnos retried under `wait_for_capacity=True`; everything else (SSL,
# EAI_NONAME, EACCES, …) fails fast.
_TRANSIENT_CONNECT_ERRNOS: frozenset[int] = frozenset(
    n
    for n in (
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "EHOSTDOWN", None),
        getattr(socket, "EAI_AGAIN", None),
    )
    if n is not None
)


def is_transient_connect_error(exc: BaseException) -> bool:
    """True iff a connect-time exception is worth retrying.

    Walks ``__cause__`` / ``__context__`` and ``os_error`` to handle both
    ``aiohttp.ClientConnectorError`` and ``httpx.ConnectError``. Defaults
    to True when no errno/SSL marker is found (preserves bare-exception
    test cases and platforms that don't surface errno).
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLError):
            return False
        os_err = getattr(cur, "os_error", None)
        if isinstance(os_err, OSError) and os_err.errno is not None:
            return os_err.errno in _TRANSIENT_CONNECT_ERRNOS
        if isinstance(cur, OSError) and cur.errno is not None:
            return cur.errno in _TRANSIENT_CONNECT_ERRNOS
        cur = cur.__cause__ or cur.__context__
    return True


def compute_retry_delay(
    *,
    start_time: float,
    timeout: float,
    error_label: str,
    error: BaseException,
) -> float | None:
    """Sleep duration for the next transport-error retry, or ``None`` if
    the provision-timeout budget is exhausted (caller must re-raise).
    """
    elapsed = time.monotonic() - start_time
    if elapsed >= timeout:
        return None
    actual_delay = min(MODEL_LOADING_DEFAULT_DELAY_S, timeout - elapsed)
    _logger.info(
        "%s (%s), retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs): %s",
        error_label,
        type(error).__name__,
        actual_delay,
        elapsed,
        timeout,
        error,
    )
    return actual_delay


def get_retry_after(response: _HttpResponse) -> float | None:
    """Extract Retry-After header value from response.

    Args:
        response: HTTP response that may contain Retry-After header.

    Returns:
        Retry delay in seconds, or None if header not present.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            return None
    return None


def compute_oom_backoff(
    retry_after: float | None,
    attempt: int,
    *,
    base_delay: float = RESOURCE_EXHAUSTED_DEFAULT_DELAY_S,
    max_delay: float = RESOURCE_EXHAUSTED_MAX_DELAY_S,
) -> float:
    """Compute the next sleep interval for a RESOURCE_EXHAUSTED retry.

    Honours a server-supplied ``Retry-After`` (when present) for the first
    attempt, then applies bounded exponential backoff:
    ``base * 2**attempt`` capped at ``max_delay``. The cap exists because
    a misbehaving server that holds OOM forever shouldn't push the SDK
    into multi-minute sleeps; the floor at ``0.0`` defends against a
    negative or malformed header value being passed straight to
    ``time.sleep`` (which raises ``ValueError`` on negative input).

    Args:
        retry_after: Value parsed from the ``Retry-After`` header, or None.
        attempt: 0-indexed retry number (0 = first retry).
        base_delay: Base interval when no Retry-After is supplied.
        max_delay: Hard ceiling on the returned delay.

    Returns:
        Seconds to sleep before the next attempt. Always non-negative.
    """
    # Defensive floor: a negative ``Retry-After`` (malformed / malicious
    # upstream) would otherwise crash ``time.sleep``.
    safe_retry_after = max(retry_after, 0.0) if retry_after is not None else None
    if safe_retry_after is not None and attempt == 0:
        # Trust the first server hint (capped to ``max_delay`` so a buggy
        # header can't strand the caller).
        return min(safe_retry_after, max_delay)
    # On subsequent attempts, the exponential base is the larger of
    # ``base_delay`` and the server-supplied hint:
    #
    #   * Using the hint alone would collapse the backoff to zero when
    #     the server returns ``Retry-After: 0`` (``0 * 2**N == 0``).
    #   * Using ``base_delay`` alone would sleep *less* on attempt 1 than
    #     the server asked for on attempt 0 if the server's hint exceeds
    #     ``base_delay`` (e.g. ``Retry-After: 20`` then 5*2 = 10 s) —
    #     producing a non-monotonic schedule that contradicts the
    #     server's "wait at least N seconds" instruction.
    #
    # ``max(...)`` covers both: a zero hint falls back to ``base_delay``,
    # and a hint above ``base_delay`` keeps the schedule non-decreasing.
    base = max(base_delay, safe_retry_after) if safe_retry_after is not None else base_delay
    return max(0.0, min(base * (2**attempt), max_delay))


def get_error_code(response: _HttpResponse) -> str | None:
    """Extract error code from response body.

    Args:
        response: HTTP response to parse.

    Returns:
        Error code string, or None if not found.
    """
    detail = get_error_detail(response)
    if detail is None:
        return None
    code = detail.get("code")
    return code if isinstance(code, str) else None


def get_error_detail(response: _HttpResponse) -> dict[str, Any] | None:
    """Extract the full error-detail dict from a response body.

    Returns the nested ``error``/``detail`` object so callers can read
    auxiliary fields like ``error_class``, ``permanent``, ``attempts``
    (carried by ``MODEL_LOAD_FAILED`` responses) without re-parsing.

    Returns ``None`` if the body has no recognised error detail or it is
    not a dict.
    """
    try:
        if MSGPACK_CONTENT_TYPE in response.headers.get("content-type", ""):
            data = msgpack.unpackb(response.content, raw=False)
        else:
            data = response.json()

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                return error
            return None
        if "detail" in data:
            detail = data["detail"]
            if isinstance(detail, dict):
                return detail
    except (ValueError, KeyError, TypeError):
        pass
    return None


def raise_if_model_load_failed(response: _HttpResponse, model: str | None = None) -> None:
    """Raise :class:`ModelLoadFailedError` if the response is 502 ``MODEL_LOAD_FAILED``.

    Used by the SDK retry loops to short-circuit *before* checking the
    ``MODEL_LOADING`` retry budget. The server returns this on the very
    first request when it has a recorded terminal failure for the
    model, so the caller should fail fast instead of retrying for 5
    minutes.

    Args:
        response: HTTP response to inspect.
        model: Model name for inclusion in the raised error.

    Raises:
        ModelLoadFailedError: If the response is a 502 carrying the
            ``MODEL_LOAD_FAILED`` error code.
    """
    if response.status_code != 502:
        return
    detail = get_error_detail(response)
    if detail is None:
        return
    if detail.get("code") != MODEL_LOAD_FAILED_ERROR_CODE:
        return
    error_class = detail.get("error_class")
    permanent = bool(detail.get("permanent", True))
    attempts_raw = detail.get("attempts", 1)
    # Defensive: a malformed/buggy server payload (e.g. ``"attempts": "n/a"``,
    # ``"inf"``, ``-5``) must not crash the retry loop or expose nonsense
    # values upstream. Coerce best-effort, then clamp to a sane minimum of 1.
    # OverflowError (from float('inf')) and any other exception fall back
    # to 1 so malformed payloads always degrade safely.
    try:
        if isinstance(attempts_raw, int | float | str):
            attempts = int(attempts_raw)
        else:
            attempts = 1
    except (TypeError, ValueError, OverflowError):
        attempts = 1
    attempts = max(attempts, 1)
    message = str(detail.get("message") or f"Model '{model}' failed to load")
    raise ModelLoadFailedError(
        message,
        model=model,
        error_class=str(error_class) if error_class is not None else None,
        permanent=permanent,
        attempts=attempts,
    )


def handle_error(response: _HttpResponse) -> None:
    """Handle error response from server.

    Raises:
        RequestError: For 4xx responses.
        ServerError: For 5xx responses.
    """
    code = None
    message = f"HTTP {response.status_code}"

    try:
        # Try msgpack first
        if MSGPACK_CONTENT_TYPE in response.headers.get("content-type", ""):
            data = msgpack.unpackb(response.content, raw=False)
        else:
            data = response.json()

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message", message)
            else:
                # error is a string, use it as the message
                message = str(error)
        elif "detail" in data:
            detail = data["detail"]
            if isinstance(detail, dict):
                code = detail.get("code")
                message = detail.get("message", str(detail))
            else:
                message = str(detail)
    except (ValueError, KeyError, TypeError):
        # Fall back to raw text
        message = response.text or message

    if response.status_code >= HTTP_SERVER_ERROR:
        raise ServerError(message, code=code, status_code=response.status_code)
    raise RequestError(message, code=code, status_code=response.status_code)


def parse_encode_results(items: list[dict[str, Any]]) -> list[EncodeResult]:
    """Parse encode response items into EncodeResult TypedDicts.

    Extracts numpy arrays from the wire format. Arrays are expected to be
    numpy arrays from msgpack-numpy deserialization.
    """
    results: list[EncodeResult] = []

    for item in items:
        result: EncodeResult = {}

        # Copy id if present
        if "id" in item:
            result["id"] = item["id"]

        # Extract dense embedding (may be None if not requested)
        if "dense" in item and item["dense"] is not None:
            values = item["dense"]["values"]
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["dense"] = values

        # Extract sparse embedding (may be None if not requested)
        if "sparse" in item and item["sparse"] is not None:
            sparse = item["sparse"]
            indices = sparse["indices"]
            values = sparse["values"]
            assert isinstance(indices, np.ndarray), "Expected numpy array from msgpack-numpy"
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["sparse"] = SparseResult(indices=indices, values=values)

        # Extract multivector embedding (may be None if not requested)
        if "multivector" in item and item["multivector"] is not None:
            values = item["multivector"]["values"]
            assert isinstance(values, np.ndarray), "Expected numpy array from msgpack-numpy"
            result["multivector"] = values

        results.append(result)

    return results


def parse_score_result(data: dict[str, Any]) -> ScoreResult:
    """Parse score response into ScoreResult TypedDict."""
    result: ScoreResult = {
        "model": data["model"],
        "scores": [
            ScoreEntry(
                item_id=s["item_id"],
                score=s["score"],
                rank=s["rank"],
            )
            for s in data["scores"]
        ],
    }
    if data.get("query_id") is not None:
        result["query_id"] = data["query_id"]
    return result


def parse_extract_results(items: list[dict[str, Any]]) -> list[ExtractResult]:
    """Parse extract response items into ExtractResult TypedDicts."""
    results: list[ExtractResult] = []

    for item in items:
        result: ExtractResult = {
            "entities": [
                EntityResult(
                    text=e["text"],
                    label=e["label"],
                    score=e["score"],
                    start=e.get("start"),
                    end=e.get("end"),
                    bbox=e.get("bbox"),
                )
                for e in item.get("entities", [])
            ],
            "relations": [
                Relation(
                    head=r["head"],
                    tail=r["tail"],
                    relation=r["relation"],
                    score=r["score"],
                )
                for r in item.get("relations", [])
            ],
            "classifications": [
                Classification(label=c["label"], score=c["score"]) for c in item.get("classifications", [])
            ],
            "objects": [
                DetectedObject(label=o["label"], score=o["score"], bbox=o["bbox"]) for o in item.get("objects", [])
            ],
        }

        # Copy optional fields
        if item.get("id") is not None:
            result["id"] = item["id"]
        if item.get("data"):
            result["data"] = item["data"]

        results.append(result)

    return results
