from __future__ import annotations

import errno
import logging
import math
import random
import re
import socket
import ssl
import time
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any, NoReturn, Protocol, cast
from urllib.parse import parse_qsl, urljoin, urlsplit

import numpy as np
from msgpack.exceptions import UnpackException

from sie_sdk._msgpack import unpackb as unpack_msgpack
from sie_sdk.images import convert_item_images

_logger = logging.getLogger(__name__)


_HTTP_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_HTTP_MEDIA_TYPE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+/[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_CONTENT_SAFE_MEDIA_TYPES = frozenset(
    {"application/json", "application/msgpack", "application/problem+json", "text/html"}
)
_RESERVED_BASE_URL_HEADERS = frozenset(
    {
        "accept",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "idempotency-key",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "x-sie-sdk-version",
    }
)
MODAL_CONTINUATION_MAX_HOPS = 20
_MODAL_CONTINUATION_QUERY_PARAMETER = "__modal_attempt_token"


def copy_base_url_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Validate and detach additional gateway-origin request headers.

    These headers are intended for an HTTP edge in front of the configured SIE
    gateway (for example Modal's ``Modal-Key`` / ``Modal-Secret`` pair). They
    cannot replace SDK-owned authentication, representation, or hop-by-hop
    headers. A detached copy prevents a caller from changing credentials after
    client construction.
    """
    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        msg = "base_url_headers must be a mapping of string header names to string values"
        raise TypeError(msg)

    copied: dict[str, str] = {}
    normalized_names: set[str] = set()
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            msg = "base_url_headers must contain only string header names and values"
            raise TypeError(msg)
        normalized = name.lower()
        if not _HTTP_HEADER_NAME.fullmatch(name):
            msg = f"invalid base_url_headers name: {name!r}"
            raise ValueError(msg)
        if normalized in normalized_names:
            msg = f"duplicate base_url_headers name (case-insensitive): {name!r}"
            raise ValueError(msg)
        if normalized in _RESERVED_BASE_URL_HEADERS or normalized.startswith("sec-websocket-"):
            msg = f"base_url_headers cannot override SDK-owned header {name!r}"
            raise ValueError(msg)
        if any(ord(char) < 0x20 and char != "\t" for char in value) or "\x7f" in value:
            msg = f"invalid control character in base_url_headers value for {name!r}"
            raise ValueError(msg)
        normalized_names.add(normalized)
        copied[name] = value
    return copied


def _http_origin(url: str) -> tuple[str, str, int] | None:
    """Return a normalized HTTP origin, rejecting userinfo and malformed URLs."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if scheme not in {"http", "https"} or host is None or parsed.username is not None or parsed.password is not None:
        return None
    if port is None:
        port = 443 if scheme == "https" else 80
    return scheme, host.lower(), port


def base_url_accepts_origin_credentials(base_url: str) -> bool:
    """Whether gateway-origin credentials can be transported to ``base_url``."""
    origin = _http_origin(base_url)
    return origin is not None and origin[0] == "https"


def validate_base_url(base_url: str) -> None:
    """Reject a malformed ``base_url`` at construction.

    A scheme-less URL such as ``"localhost:8080"`` would otherwise surface
    only at request time as an opaque transport error
    (``httpx.UnsupportedProtocol`` on the sync client), or — worse — be
    silently retried as a "connect" failure for the whole provision budget.
    A hostless URL such as ``"http:///v1"`` parses with a valid scheme but an
    empty host, which cannot be connected to; reject it here too. A textual or
    out-of-range port (``"http://host:notaport"``, ``"http://host:99999"``)
    likewise fails here rather than at request time.
    """
    parts = urlsplit(base_url)
    msg = (
        "base_url must be an absolute http:// or https:// URL with a host "
        f"(e.g. 'http://localhost:8080'), got {base_url!r}"
    )
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(msg)
    try:
        # ``urlsplit`` validates the port lazily, on access.
        _ = parts.port
    except ValueError as exc:
        raise ValueError(msg) from exc


def url_origin_for_logging(url: str) -> str:
    """Return only the ``scheme://host[:port]`` origin of ``url``.

    Path, query, fragment, and any ``user:password@`` userinfo are dropped, so
    a ``base_url`` carrying embedded credentials or a token query parameter
    never reaches a log line. The value is
    for logging only — callers still issue requests against the real URL.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = f"{host}:{parts.port}" if parts.port is not None else host
    return f"{parts.scheme}://{netloc}" if parts.scheme else netloc


def request_matches_base_url_origin(base_url: str, request_url: str) -> bool:
    """Whether ``request_url`` resolves to the exact HTTP origin of ``base_url``."""
    base_origin = _http_origin(base_url)
    if base_origin is None:
        return False
    resolved = urljoin(f"{base_url.rstrip('/')}/", request_url)
    return _http_origin(resolved) == base_origin


def modal_continuation_path(base_url: str, response: _HttpResponse) -> str | None:
    """Return a safe edge-relative Modal result URL for an HTTP 303.

    Modal turns a web request that exceeds 150 seconds into a 303 whose
    ``Location`` identifies the *same in-flight request*. Following that
    result URL with GET consumes the original response; it does not replay the
    non-idempotent POST. Only the documented attempt-token shape on the exact
    configured origin is accepted so SDK credentials can never cross an
    origin boundary.
    """
    if response.status_code != 303:
        return None
    location = _header_value(response.headers, "Location")
    if not isinstance(location, str) or not location or len(location) > 8192:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in location):
        return None

    resolved = urljoin(f"{base_url.rstrip('/')}/", location)
    if not request_matches_base_url_origin(base_url, resolved):
        return None
    try:
        parsed = urlsplit(resolved)
        query = parse_qsl(parsed.query, keep_blank_values=True, max_num_fields=100)
    except ValueError:
        return None
    tokens = [value for key, value in query if key == _MODAL_CONTINUATION_QUERY_PARAMETER]
    if parsed.fragment or len(tokens) != 1 or not tokens[0]:
        return None

    path = parsed.path or "/"
    return f"{path}?{parsed.query}" if parsed.query else path


def websocket_matches_base_url_origin(base_url: str, websocket_url: str) -> bool:
    """Whether a ws(s) URL is the WebSocket counterpart of ``base_url``'s origin."""
    parsed = urlsplit(websocket_url)
    if parsed.scheme not in {"ws", "wss"}:
        return False
    http_scheme = "https" if parsed.scheme == "wss" else "http"
    http_url = parsed._replace(scheme=http_scheme).geturl()
    return request_matches_base_url_origin(base_url, http_url)


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
    ExtractItemErrorDetail,
    ExtractResult,
    GenerateGrammar,
    Relation,
    RequestMetadata,
    RequestUsage,
    ScoreEntry,
    ScoreResult,
    SparseResult,
)

from .errors import (
    AccountInactiveError,
    AccountStateUnavailableError,
    EstimateUnroutableError,
    IncompleteBatchError,
    InputTooLongError,
    InsufficientCreditsError,
    ModelLoadFailedError,
    ModelLoadingError,
    ProvisioningError,
    RateLimitError,
    RequestError,
    ResourceExhaustedError,
    ServerError,
    SpendLimitError,
)

# Content types
MSGPACK_CONTENT_TYPE = "application/msgpack"
JSON_CONTENT_TYPE = "application/json"

# HTTP status code thresholds
HTTP_CLIENT_ERROR = 400
HTTP_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

# A result ref can land on a gateway replica whose payload-store view has not
# converged yet. The gateway uses this exact 404 code to tell the SDK to fetch
# a fresh job view before trying the refs again. Keep the refresh bounded: a
# storage outage is a distinct 503 and must never be hidden by this path.
JOB_RESULT_NOT_FOUND_ERROR_CODE = "RESULT_NOT_FOUND"
JOB_RESULT_REF_MAX_REFRESHES = 3

# jobs.results() decodes chunk refs into per-item results. A non-terminal job
# has no stable result set: its ref list is still growing, so decoding one would
# return a partial subset indistinguishable from a partial-FAILURE subset. The
# SDK refuses with this code rather than return a misleading partial.
JOB_NOT_TERMINAL_ERROR_CODE = "job_not_terminal"


_GENERATE_GRAMMAR_VARIANTS = frozenset({"json_schema", "regex", "ebnf"})
_GENERATE_GRAMMAR_FIELDS = _GENERATE_GRAMMAR_VARIANTS | {"label", "strict"}


def validate_generate_grammar(grammar: GenerateGrammar | Mapping[str, Any]) -> GenerateGrammar:
    """Validate and detach the native structured-output grammar envelope.

    ``generate`` historically accepted a broad dictionary here. Keep that
    source-compatible input type while validating the exact native three-arm
    shape before issuing a request.
    """
    if not isinstance(grammar, Mapping):
        msg = "grammar must be a mapping"
        raise TypeError(msg)

    unknown = set(grammar) - _GENERATE_GRAMMAR_FIELDS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        msg = f"grammar contains unsupported field(s): {names}"
        raise ValueError(msg)

    grammar_dict = dict(grammar)
    variants = _GENERATE_GRAMMAR_VARIANTS.intersection(grammar_dict)
    if len(variants) != 1:
        msg = "grammar must contain exactly one of json_schema, regex, or ebnf"
        raise ValueError(msg)

    variant = next(iter(variants))
    value = grammar_dict[variant]
    if variant == "json_schema":
        if not isinstance(value, Mapping):
            msg = "grammar.json_schema must be a mapping"
            raise TypeError(msg)
        grammar_dict["json_schema"] = dict(value)
    elif not isinstance(value, str):
        msg = f"grammar.{variant} must be a string"
        raise TypeError(msg)

    if "label" in grammar and grammar["label"] is not None and not isinstance(grammar["label"], str):
        msg = "grammar.label must be a string"
        raise TypeError(msg)
    if "strict" in grammar and grammar["strict"] is not None and not isinstance(grammar["strict"], bool):
        msg = "grammar.strict must be a boolean"
        raise TypeError(msg)

    return cast("GenerateGrammar", grammar_dict)


def validate_generate_request_body(request_body: MutableMapping[str, Any]) -> None:
    """Validate and detach any grammar in a fully merged generate request."""
    grammar = request_body.get("grammar")
    if grammar is not None:
        request_body["grammar"] = validate_generate_grammar(grammar)


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
PROVISIONING_ERROR_CODE = "PROVISIONING"  # Gateway scale-from-zero / worker provisioning
SIE_ERROR_CODE_HEADER = "X-SIE-Error-Code"
_MALFORMED_EXTRACT_ERROR_CODE = "INTERNAL_ERROR"
_MALFORMED_EXTRACT_ERROR_MESSAGE = "Malformed extraction item error"

# Terminal load failure (non-retryable). Server returns this with HTTP 502
# and *no* Retry-After header so the SDK can short-circuit immediately
# instead of burning the MODEL_LOADING retry budget.
MODEL_LOAD_FAILED_ERROR_CODE = "MODEL_LOAD_FAILED"

# Terminal client-side error: request input exceeds the model's maximum
# token capacity. Server returns HTTP 400 + this code; the SDK surfaces
# a typed ``InputTooLongError`` so callers can react without parsing
# error codes by hand.
INPUT_TOO_LONG_ERROR_CODE = "INPUT_TOO_LONG"

# ── cost-estimate dry run (POST /v1/estimate, #2435) ──────────────────────
ESTIMATE_PATH = "/v1/estimate"
RECOMMEND_PATH = "/v1/recommend"
# The gateway answers an unpriced/unroutable identity with the SAME code the
# live billing path uses for "this request cannot be priced, so it will not be
# run". Both are mapped to ``EstimateUnroutableError``; a slab-capacity 503
# (``BILLING_CAPACITY_UNAVAILABLE``) is a retryable gateway condition and stays
# a plain ``ServerError``.
ESTIMATE_UNROUTABLE_ERROR_CODES = frozenset({"QUEUE_UNAVAILABLE", "PROVISIONING"})


def build_estimate_envelope(endpoint: str, request: Mapping[str, Any]) -> dict[str, Any]:
    """The ``POST /v1/estimate`` envelope: the target path plus its verbatim body.

    The gateway prices whatever ``request`` would have been sent to ``endpoint``,
    so the SDK does NOT reshape it. Callers pass the exact body they would pass
    to the real route; anything the SDK normalized on the way in would be
    something the quote priced and the request never sent.

    The returned mapping is a SHALLOW copy: rebinding a top-level key on the
    caller's dict after this returns cannot change what is priced, but nested
    containers are shared, so mutating one in place still can. Both clients
    serialize the envelope immediately, so the window is a single call — and a
    deep copy is deliberately not taken, because these bodies routinely carry
    megabytes of base64 audio or image payloads.

    Raises:
        ValueError: If ``endpoint`` is not a non-empty path or ``request`` is
            not a mapping. Both are caller mistakes about WHAT is being priced,
            so they fail here rather than as a server-side 400.
    """
    if not isinstance(endpoint, str) or not endpoint.startswith("/"):
        msg = f"estimate endpoint must be the exact target path (got {endpoint!r})"
        raise ValueError(msg)
    if not isinstance(request, Mapping):
        msg = f"estimate request must be the target request body mapping (got {type(request).__name__})"
        raise ValueError(msg)
    return {"endpoint": endpoint, "request": dict(request)}


def raise_if_estimate_unroutable(response: _HttpResponse) -> None:
    """Raise :class:`EstimateUnroutableError` for an unpriceable estimate.

    Short-circuits ``handle_error``'s generic 5xx dispatch so callers can catch
    "this request is not sellable" specifically, without string-matching a code.
    The message is the planner's own reason — it names the unpriced identity or
    the dimension it could not bound.
    """
    if response.status_code != HTTP_SERVICE_UNAVAILABLE:
        return
    code = get_error_code(response)
    if code not in ESTIMATE_UNROUTABLE_ERROR_CODES:
        return
    detail = get_error_detail(response)
    message = None
    if isinstance(detail, Mapping):
        message = detail.get("message")
    raise EstimateUnroutableError(
        str(message or "the active rate book cannot price this request"),
        code=code,
        param=get_error_param(response),
        request=parse_request_metadata(response.headers),
    )


# Resource-exhausted retry settings (server-side OOM recovery exhausted).
# Default backoff sequence: 5 -> 10 -> 20 s (capped at 30s). Three attempts
# is enough to cover the typical eviction + retry window without making
# pathological cases hang indefinitely. Callers can opt out with
# ``max_oom_retries=0``.
RESOURCE_EXHAUSTED_MAX_RETRIES = 3
RESOURCE_EXHAUSTED_DEFAULT_DELAY_S = 5.0
RESOURCE_EXHAUSTED_MAX_DELAY_S = 30.0
RESOURCE_EXHAUSTED_ERROR_CODE = "RESOURCE_EXHAUSTED"

# ── Pre-execution admission backpressure / billing signals (pass-2 audit) ──
# All of these are emitted BEFORE any work is published to the queue, so
# retrying them is idempotent even on the non-idempotent generate paths. They
# are honored on the SDK's admission ladder (:func:`admission_retry_delay`),
# bounded by the caller's ``provision_timeout_s`` budget and the server's
# ``Retry-After``. Kept deliberately separate from the 402/403 credit/account
# errors below, which are TERMINAL and must NEVER be retried.
#
# B1 — 429 rate limit (rate_limit.rs): per-key/per-account, default-on. The
# gateway always supplies a ``Retry-After``; the SDK honors it and gives up as
# a typed :class:`RateLimitError` when the budget is spent.
HTTP_TOO_MANY_REQUESTS = 429
RATE_LIMIT_ERROR_CODE = "RATE_LIMIT"
RATE_LIMIT_DEFAULT_DELAY_S = 1.0  # Fallback when the server omits Retry-After

# B2 — 503 BILLING_CAPACITY_UNAVAILABLE (slab_ledger.rs): a gateway-local
# family cap on billing admission, NOT customer credit exhaustion. Server sends
# Retry-After: 1. B7 — 503 QUEUE_FULL (self-hosted server, #3180): transient
# queue backpressure, Retry-After: 1. Both are retryable 503s the pre-existing
# ladder did not match, so they fell through and failed hard.
BILLING_CAPACITY_UNAVAILABLE_ERROR_CODE = "BILLING_CAPACITY_UNAVAILABLE"
QUEUE_FULL_ERROR_CODE = "QUEUE_FULL"
BACKPRESSURE_503_ERROR_CODES = frozenset({BILLING_CAPACITY_UNAVAILABLE_ERROR_CODE, QUEUE_FULL_ERROR_CODE})
BACKPRESSURE_503_DEFAULT_DELAY_S = 1.0  # Fallback when the server omits Retry-After

# ── Terminal credit / account errors (pass-2 audit B3) — NEVER retried ──
# 402/403 credit/account failures are mapped to typed exceptions in
# ``handle_error`` and, having no arm on any retry ladder, surface on the first
# response (single attempt). Retrying a credit/account failure would be wrong.
HTTP_PAYMENT_REQUIRED = 402
HTTP_FORBIDDEN = 403
INSUFFICIENT_CREDITS_ERROR_CODE = "INSUFFICIENT_CREDITS"
KEY_SPEND_LIMIT_EXCEEDED_ERROR_CODE = "KEY_SPEND_LIMIT_EXCEEDED"
ACCOUNT_SUSPENDED_ERROR_CODE = "ACCOUNT_SUSPENDED"
ACCOUNT_PENDING_REVIEW_ERROR_CODE = "ACCOUNT_PENDING_REVIEW"
ACCOUNT_STATE_UNAVAILABLE_ERROR_CODE = "ACCOUNT_STATE_UNAVAILABLE"

# Retry jitter. Fixed / pure-exponential backoff makes every client that
# lost a worker at the same instant (cluster cold start, rolling restart)
# wake up and retry in lockstep — a thundering herd that re-saturates the
# gateway. We apply *downward-only* "equal jitter": the returned delay is
# drawn uniformly from ``[delay * (1 - RETRY_JITTER_FRACTION), delay]``.
# Downward-only is deliberate so the jittered value never exceeds the
# caller's existing cap (``timeout - elapsed``, ``max_delay``) and stays
# non-negative — preserving every existing delay/budget bound.
RETRY_JITTER_FRACTION = 0.25

# Module-level RNG, seedable in tests for determinism. Not used for
# anything security-sensitive — only to de-correlate retry timing.
_retry_rng = random.Random()  # noqa: S311 — non-cryptographic jitter only


def convert_score_images_for_wire(query: Any, items: Sequence[Any]) -> tuple[Any, list[Any]]:
    """Convert image-bearing score query/items to the SDK image wire shape."""
    query_for_wire = convert_item_images({**query}) if "images" in query else query
    items_for_wire = [
        convert_item_images({**item}) if "images" in item else item  # ty: ignore[invalid-argument-type]
        for item in items
    ]
    return query_for_wire, items_for_wire


def apply_jitter(delay: float, *, rng: random.Random | None = None) -> float:
    """Apply bounded downward jitter to a backoff ``delay``.

    Returns a value drawn uniformly from
    ``[delay * (1 - RETRY_JITTER_FRACTION), delay]`` (clamped to
    ``>= 0``). Jittering *down only* guarantees the result never exceeds
    the input, so callers' existing caps and provision-timeout budgets
    remain valid. A non-positive ``delay`` is returned unchanged (no
    point jittering a zero/negative sleep).

    Args:
        delay: The pre-jitter backoff seconds.
        rng: Optional :class:`random.Random` for deterministic tests.
            Defaults to the module RNG.

    Returns:
        Jittered, non-negative delay in seconds.
    """
    if delay <= 0:
        return max(delay, 0.0)
    r = rng if rng is not None else _retry_rng
    low = delay * (1.0 - RETRY_JITTER_FRACTION)
    return max(0.0, r.uniform(low, delay))


# Version negotiation headers
SDK_VERSION_HEADER = "X-SIE-SDK-Version"
SERVER_VERSION_HEADER = "X-SIE-Server-Version"
MODEL_REVISION_HEADER = "X-SIE-Model-Revision"
EXECUTION_IDENTITY_SHA256_HEADER = "X-SIE-Execution-Identity-SHA256"
EXECUTION_BINDING_SHA256_HEADER = "X-SIE-Execution-Binding-SHA256"
REQUEST_ID_HEADER = "X-SIE-Request-ID"
CREDITS_DEBITED_HEADER = "X-SIE-Credits-Debited"
REQUEST_USAGE_HEADERS = {
    "input_tokens": "X-SIE-Units-Input-Tokens",
    "pairs": "X-SIE-Units-Pairs",
    "images": "X-SIE-Units-Images",
    "pages": "X-SIE-Units-Pages",
    "output_tokens": "X-SIE-Units-Output-Tokens",
    "audio_ms": "X-SIE-Units-Audio-Ms",
}
_MAX_REQUEST_ID_LENGTH = 256
_MAX_METER_VALUE = (1 << 64) - 1


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
        # Best-effort version compare; non-semver inputs silently skip the warning.
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
    rng: random.Random | None = None,
    attempt: int | None = None,
    target: str | None = None,
) -> float | None:
    """Sleep duration for the next transport-error retry, or ``None`` if
    the provision-timeout budget is exhausted (caller must re-raise).

    Bounded downward jitter (see :func:`apply_jitter`) is applied so a
    fleet of clients that lost connectivity simultaneously don't retry in
    lockstep. The jittered value never exceeds ``timeout - elapsed``, so
    the provision-timeout budget is still respected. ``rng`` is exposed
    for deterministic tests.

    When ``attempt`` is 0 the retry is logged at WARNING — matching the
    OOM-retry convention — so a user at the default log level can see
    "the SDK is retrying you" (naming the unreachable ``target`` and the
    total wait budget) instead of silently blocking for the whole
    provision timeout. Subsequent retries stay at INFO to avoid log spam.
    """
    elapsed = time.monotonic() - start_time
    if elapsed >= timeout:
        return None
    actual_delay = apply_jitter(min(MODEL_LOADING_DEFAULT_DELAY_S, timeout - elapsed), rng=rng)
    log_fn = _logger.warning if attempt == 0 else _logger.info
    # Log only the origin: drops path/query/fragment and any embedded
    # credentials before the URL reaches the log.
    safe_target = url_origin_for_logging(target) if target else None
    log_fn(
        "%s (%s)%s, retrying in %.1fs (elapsed: %.1fs, timeout: %.1fs): %s",
        error_label,
        type(error).__name__,
        f" contacting {safe_target}" if safe_target else "",
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
        Retry delay in seconds, or None if the header is absent OR carries
        an unusable value. A non-finite (``nan`` / ``inf`` / ``-inf``) or
        negative value is treated as "no usable hint" and returned as
        ``None`` so callers fall back to their default delay. Returning
        these verbatim would crash sync ``time.sleep`` (``ValueError`` on
        ``nan``) and busy-loop async ``asyncio.sleep`` (returns instantly
        for ``nan`` / negative input) — a retry-budget-burning DoS.
    """
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        value = float(retry_after)
    except ValueError:
        # RFC 7231 also permits an HTTP-date form ("Wed, 21 Oct 2025
        # 07:28:00 GMT"). Parse it and return the delta in seconds for
        # cross-SDK parity with the TS SDK. A past date / unparseable
        # value yields "no usable hint" (None).
        try:
            when = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        # RFC 7231 HTTP-dates are GMT, so ``parsedate_to_datetime`` normally
        # returns an aware datetime; tolerate a naive one (compare against a
        # naive UTC ``now``) so the subtraction never raises ``TypeError``.
        if when.tzinfo is not None:
            now = datetime.now(when.tzinfo)
        else:
            now = datetime.now(UTC).replace(tzinfo=None)
        delta = (when - now).total_seconds()
        return max(delta, 0.0)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def retry_after_or_default(retry_after: float | None, default: float) -> float:
    """Return a parsed ``Retry-After`` hint, preserving explicit zero."""
    return retry_after if retry_after is not None else default


def compute_oom_backoff(
    retry_after: float | None,
    attempt: int,
    *,
    base_delay: float = RESOURCE_EXHAUSTED_DEFAULT_DELAY_S,
    max_delay: float = RESOURCE_EXHAUSTED_MAX_DELAY_S,
    rng: random.Random | None = None,
) -> float:
    """Compute the next sleep interval for a RESOURCE_EXHAUSTED retry.

    Honours a server-supplied ``Retry-After`` (when present) for the first
    attempt, then applies bounded exponential backoff:
    ``base * 2**attempt`` capped at ``max_delay``. The cap exists because
    a misbehaving server that holds OOM forever shouldn't push the SDK
    into multi-minute sleeps; the floor at ``0.0`` defends against a
    negative or malformed header value being passed straight to
    ``time.sleep`` (which raises ``ValueError`` on negative input).

    Bounded downward jitter (see :func:`apply_jitter`) de-correlates a
    fleet of clients all evicted by the same OOM event. Jitter is applied
    *after* the cap, so the returned value is still ``<= max_delay`` and
    ``>= 0`` — preserving the documented bound. A first-attempt
    ``Retry-After`` hint is honoured verbatim (no jitter): when the
    server explicitly tells us "wait N seconds" we respect it, only
    de-correlating the SDK-derived exponential schedule.

    Args:
        retry_after: Value parsed from the ``Retry-After`` header, or None.
        attempt: 0-indexed retry number (0 = first retry).
        base_delay: Base interval when no Retry-After is supplied.
        max_delay: Hard ceiling on the returned delay.
        rng: Optional :class:`random.Random` for deterministic tests.

    Returns:
        Seconds to sleep before the next attempt. Always non-negative and
        never greater than ``max_delay``.
    """
    # Defensive floor: a negative ``Retry-After`` (malformed / malicious
    # upstream) would otherwise crash ``time.sleep``.
    safe_retry_after = max(retry_after, 0.0) if retry_after is not None else None
    if safe_retry_after is not None and attempt == 0:
        # Trust the first server hint (capped to ``max_delay`` so a buggy
        # header can't strand the caller). Honoured verbatim — no jitter,
        # because the server gave an explicit instruction.
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
    capped = max(0.0, min(base * (2**attempt), max_delay))
    # Jitter is applied after the cap and is downward-only, so the result
    # remains ``0 <= result <= max_delay``.
    return apply_jitter(capped, rng=rng)


def _header_value(headers: Any, name: str) -> Any:
    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower())
    return value


def _parse_nonnegative_meter_header(headers: Any, name: str) -> int | None:
    value = _header_value(headers, name)
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdigit()
        or (len(value) > 1 and value.startswith("0"))
    ):
        return None
    parsed = int(value)
    return parsed if parsed <= _MAX_METER_VALUE else None


def settled_charge_from_usage(usage: Any) -> tuple[int, str] | None:
    """The settled charge carried by one ``usage`` block (#2434).

    Returns ``(credits_charged, rate_book_version)`` only when the block
    carries BOTH — a charge with no book version cannot be reconciled, and a
    version with no charge describes nothing. Anything malformed is treated as
    absent.
    """
    if not isinstance(usage, Mapping):
        return None
    credits = usage.get("credits_charged")
    version = usage.get("rate_book_version")
    if not isinstance(credits, int) or isinstance(credits, bool) or credits < 0:
        return None
    if not isinstance(version, str) or not version:
        return None
    return credits, version


def _settled_charge_from_body(body: Any) -> tuple[int, str] | None:
    """The settled charge from a response envelope's ``usage`` block (#2434)."""
    if not isinstance(body, Mapping):
        return None
    return settled_charge_from_usage(body.get("usage"))


def valid_stream_request_id(value: Any) -> str | None:
    """Return ``value`` when it is a well-formed request id, else ``None``.

    Same rules as ``parse_request_metadata`` (non-empty visible ASCII without
    surrounding whitespace, bounded length). Stream error chunks carry the id
    in-band; a malformed value must be dropped BEFORE it is placed on a
    synthetic HTTP header, where non-ASCII would raise ``UnicodeEncodeError``
    inside HTTPX and bypass the retry path (#3136).
    """
    if (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_REQUEST_ID_LENGTH
        and value == value.strip()
        and value.isascii()
        and value.isprintable()
    ):
        return value
    return None


def parse_request_metadata(headers: Any, body: Any = None) -> RequestMetadata | None:
    """Parse optional metadata from one terminal response.

    Malformed fields are omitted independently. Integer meter headers must be
    canonical non-negative unsigned decimal values; request ids must be
    non-empty visible ASCII without surrounding whitespace. If no valid fields
    remain, no metadata is attached.

    ``body`` is the decoded response envelope, when the caller has one. The
    settled charge is read from its ``usage`` block FIRST and falls back to the
    ``x-sie-credits-debited`` header, which is the only source on a gateway
    that predates in-body surfacing (#2434). The two always agree when both are
    present; the body additionally names the rate book, which no header does.
    """
    metadata: RequestMetadata = {}

    request_id = _header_value(headers, REQUEST_ID_HEADER)
    if (
        isinstance(request_id, str)
        and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH
        and request_id == request_id.strip()
        and request_id.isascii()
        and request_id.isprintable()
    ):
        metadata["id"] = request_id

    usage: RequestUsage = {}
    input_tokens = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["input_tokens"])
    pairs = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["pairs"])
    images = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["images"])
    pages = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["pages"])
    output_tokens = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["output_tokens"])
    audio_ms = _parse_nonnegative_meter_header(headers, REQUEST_USAGE_HEADERS["audio_ms"])
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if pairs is not None:
        usage["pairs"] = pairs
    if images is not None:
        usage["images"] = images
    if pages is not None:
        usage["pages"] = pages
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if audio_ms is not None:
        usage["audio_ms"] = audio_ms

    settled = _settled_charge_from_body(body)
    if settled is not None:
        credits_charged, rate_book_version = settled
        usage["credits_charged"] = credits_charged
        usage["rate_book_version"] = rate_book_version
        metadata["credits_debited"] = credits_charged
        metadata["rate_book_version"] = rate_book_version
    else:
        credits_debited = _parse_nonnegative_meter_header(headers, CREDITS_DEBITED_HEADER)
        if credits_debited is not None:
            metadata["credits_debited"] = credits_debited

    if usage:
        metadata["usage"] = usage

    execution_identity_sha256 = _header_value(headers, EXECUTION_IDENTITY_SHA256_HEADER)
    if (
        isinstance(execution_identity_sha256, str)
        and len(execution_identity_sha256) == 64
        and all(char in "0123456789abcdef" for char in execution_identity_sha256)
    ):
        metadata["execution_identity_sha256"] = execution_identity_sha256

    execution_binding_sha256 = _header_value(headers, EXECUTION_BINDING_SHA256_HEADER)
    if (
        isinstance(execution_binding_sha256, str)
        and len(execution_binding_sha256) == 64
        and all(char in "0123456789abcdef" for char in execution_binding_sha256)
    ):
        metadata["execution_binding_sha256"] = execution_binding_sha256

    return metadata or None


def _terminal_response_diagnostic(response: _HttpResponse) -> str:
    """Describe a terminal response without copying attacker-controlled values."""
    raw_content_type = _header_value(response.headers, "content-type")
    content_type = "unknown"
    if isinstance(raw_content_type, str):
        candidate = raw_content_type.partition(";")[0].strip().lower()
        if candidate in _CONTENT_SAFE_MEDIA_TYPES:
            content_type = candidate
        elif _HTTP_MEDIA_TYPE.fullmatch(candidate):
            content_type = "other"
    return f"status={response.status_code}, content_type={content_type}, body_bytes={len(response.content)}"


def parse_terminal_json_object(response: _HttpResponse, *, owner: str) -> dict[str, Any]:
    """Parse one successful terminal JSON object with content-safe diagnostics."""
    request = parse_request_metadata(response.headers)
    diagnostic = _terminal_response_diagnostic(response)

    if not 200 <= response.status_code < 300:
        msg = f"Unexpected {owner} HTTP response ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)

    malformed = False
    try:
        data = response.json()
    except (TypeError, ValueError):
        # Raise only after the parser exception context has been cleared. JSON
        # decoder exceptions can retain the raw response body (for example in
        # JSONDecodeError.doc), which must never escape through chaining.
        malformed = True
        data = None
    if malformed:
        msg = f"Malformed {owner} JSON response ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)
    if not isinstance(data, dict):
        msg = f"Unexpected {owner} response shape: {type(data).__name__} ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)
    return data


def parse_terminal_msgpack_object(response: _HttpResponse, *, owner: str) -> dict[str, Any]:
    """Parse one successful terminal MessagePack object with content-safe diagnostics."""
    request = parse_request_metadata(response.headers)
    diagnostic = _terminal_response_diagnostic(response)

    if not 200 <= response.status_code < 300:
        msg = f"Unexpected {owner} HTTP response ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)

    malformed = False
    try:
        data = unpack_msgpack(response.content, numeric_arrays=True)
    except (TypeError, ValueError, UnpackException):
        # Raise only after the parser exception context has been cleared.
        # MessagePack exceptions can retain decoded or trailing body content,
        # which must never escape through exception chaining.
        malformed = True
        data = None
    if malformed:
        msg = f"Malformed {owner} MessagePack response ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)
    if not isinstance(data, dict):
        msg = f"Unexpected {owner} response shape: {type(data).__name__} ({diagnostic})"
        raise RequestError(msg, status_code=response.status_code, request=request)
    return data


def attach_request_metadata(results: Sequence[Any], headers: Any, body: Any = None) -> None:
    """Attach detached request-scoped metadata to one or more results.

    ``body`` is the decoded response envelope when the caller has one, so the
    settled charge comes from its ``usage`` block rather than the header
    (#2434).
    """
    metadata = parse_request_metadata(headers, body)
    if metadata is None:
        return
    for result in results:
        detached = dict(metadata)
        if "usage" in metadata:
            detached["usage"] = dict(metadata["usage"])
        result["request"] = detached


def normalize_error_code(code: str | None) -> str | None:
    if code == "provisioning":
        return PROVISIONING_ERROR_CODE
    return code


def get_error_code(response: _HttpResponse) -> str | None:
    """Extract error code from response body.

    Args:
        response: HTTP response to parse.

    Returns:
        Error code string, or None if not found.
    """
    header_code = _header_value(response.headers, SIE_ERROR_CODE_HEADER)
    if isinstance(header_code, str) and header_code:
        return header_code

    detail = get_error_detail(response)
    if detail is None:
        return None
    code = detail.get("code")
    return normalize_error_code(code) if isinstance(code, str) else None


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
            data = unpack_msgpack(response.content, numeric_arrays=False)
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
    except (ValueError, KeyError, TypeError, UnpackException):
        # Malformed error body — caller treats as "no detail" and falls back.
        pass
    return None


def get_error_param(response: _HttpResponse) -> str | None:
    """Return the exact field name from a typed error envelope, if present."""
    detail = get_error_detail(response)
    if detail is None:
        return None
    param = detail.get("param")
    return param if isinstance(param, str) else None


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
        param=get_error_param(response),
        request=parse_request_metadata(response.headers),
    )


def raise_if_input_too_long(response: _HttpResponse, model: str | None = None) -> None:
    """Raise :class:`InputTooLongError` if the response is 400 ``INPUT_TOO_LONG``.

    Used by the extract path to surface token-budget overruns as a
    typed exception (so callers can catch :class:`InputTooLongError`
    specifically) instead of relying on a generic
    :class:`RequestError` + string-matching the ``code``.

    Args:
        response: HTTP response to inspect.
        model: Model name for inclusion in the raised error.

    Raises:
        InputTooLongError: If the response is a 400 carrying the
            ``INPUT_TOO_LONG`` error code.
    """
    if response.status_code != HTTP_CLIENT_ERROR:
        return
    detail = get_error_detail(response)
    if detail is None:
        return
    if detail.get("code") != INPUT_TOO_LONG_ERROR_CODE:
        return
    message = str(detail.get("message") or "Input exceeds the model's maximum token capacity")
    raise InputTooLongError(
        message,
        model=model,
        param=get_error_param(response),
        request=parse_request_metadata(response.headers),
    )


def handle_error(response: _HttpResponse) -> NoReturn:
    """Handle error response from server.

    Always raises — every path terminates in a typed exception. Declared
    ``NoReturn`` so callers (and the type checker) know a call terminates
    control flow, e.g. the admission-ladder give-up paths.

    Raises:
        RateLimitError: For 429 responses (retry budget already spent by the
            caller, or a direct terminal path).
        InsufficientCreditsError / SpendLimitError: For 402 credit/spend
            failures (terminal — never retried).
        AccountInactiveError: For 403 account suspended / pending review
            (terminal — never retried).
        AccountStateUnavailableError: For 503 ACCOUNT_STATE_UNAVAILABLE.
        RequestError: For other 4xx responses.
        ServerError: For other 5xx responses.
    """
    code = None
    param = None
    message = f"HTTP {response.status_code}"

    try:
        # Try msgpack first
        if MSGPACK_CONTENT_TYPE in response.headers.get("content-type", ""):
            data = unpack_msgpack(response.content, numeric_arrays=False)
        else:
            data = response.json()

        if "error" in data:
            error = data["error"]
            if isinstance(error, dict):
                code = error.get("code")
                message = error.get("message", message)
                error_param = error.get("param")
                param = error_param if isinstance(error_param, str) else None
            else:
                # error is a string, use it as the message
                message = str(error)
        elif "detail" in data:
            detail = data["detail"]
            if isinstance(detail, dict):
                code = detail.get("code")
                message = detail.get("message", str(detail))
                detail_param = detail.get("param")
                param = detail_param if isinstance(detail_param, str) else None
            else:
                message = str(detail)
    except (ValueError, KeyError, TypeError, UnpackException):
        # Fall back to raw text
        message = response.text or message

    header_code = _header_value(response.headers, SIE_ERROR_CODE_HEADER)
    if isinstance(header_code, str) and header_code:
        code = header_code
    else:
        code = normalize_error_code(code)

    request = parse_request_metadata(response.headers)

    # Fallback dispatch — ``model`` is only attached by the helper-style
    # short-circuit (``raise_if_input_too_long``) on the extract path.
    if response.status_code == HTTP_SERVICE_UNAVAILABLE and code == PROVISIONING_ERROR_CODE:
        raise ProvisioningError(message, retry_after=get_retry_after(response), param=param)
    if response.status_code == HTTP_CLIENT_ERROR and code == INPUT_TOO_LONG_ERROR_CODE:
        raise InputTooLongError(message, param=param, request=request)
    # Rate limit (pass-2 audit B1). Retried on the admission ladder in the
    # buffered loops; a give-up there raises RateLimitError directly. This arm
    # covers the terminal paths (streaming, list_models, estimate, …) so a 429
    # is always a typed RateLimitError rather than a generic RequestError.
    if response.status_code == HTTP_TOO_MANY_REQUESTS:
        raise RateLimitError(message, code=code, retry_after=get_retry_after(response), param=param, request=request)
    # Terminal credit / account failures (pass-2 audit B3). 402/403 are NEVER
    # retried — they have no arm on any retry ladder, so they surface here on
    # the first response. Unrecognised 402/403 codes stay generic RequestError.
    if response.status_code == HTTP_PAYMENT_REQUIRED:
        if code == INSUFFICIENT_CREDITS_ERROR_CODE:
            raise InsufficientCreditsError(message, param=param, request=request)
        if code == KEY_SPEND_LIMIT_EXCEEDED_ERROR_CODE:
            raise SpendLimitError(message, param=param, request=request)
    if response.status_code == HTTP_FORBIDDEN and code in (
        ACCOUNT_SUSPENDED_ERROR_CODE,
        ACCOUNT_PENDING_REVIEW_ERROR_CODE,
    ):
        raise AccountInactiveError(message, code=code, param=param, request=request)
    if response.status_code == HTTP_SERVICE_UNAVAILABLE and code == ACCOUNT_STATE_UNAVAILABLE_ERROR_CODE:
        raise AccountStateUnavailableError(message, param=param, request=request)
    if response.status_code >= HTTP_SERVER_ERROR:
        raise ServerError(message, code=code, status_code=response.status_code, param=param, request=request)
    raise RequestError(message, code=code, status_code=response.status_code, param=param, request=request)


def provisioning_retry_delay(
    response: _HttpResponse,
    *,
    gpu: str | None,
    wait_for_capacity: bool,
    start_time: float,
    timeout: float,
) -> float:
    """Return a retry delay for ``503 PROVISIONING`` or raise ``ProvisioningError``."""
    retry_after = get_retry_after(response)
    param = get_error_param(response)
    if not wait_for_capacity:
        msg = f"No capacity available for GPU '{gpu}'. Server is provisioning."
        raise ProvisioningError(msg, gpu=gpu, retry_after=retry_after, param=param)
    elapsed = time.monotonic() - start_time
    if elapsed >= timeout:
        msg = f"Provisioning timeout after {elapsed:.1f}s waiting for GPU '{gpu}'"
        raise ProvisioningError(msg, gpu=gpu, retry_after=retry_after, param=param)
    remaining = timeout - elapsed
    if retry_after is not None:
        return min(retry_after, remaining)
    return apply_jitter(min(DEFAULT_RETRY_DELAY_S, remaining))


def admission_retry_delay(
    response: _HttpResponse,
    *,
    start_time: float,
    timeout: float,
) -> float | None:
    """Delay before retrying a pre-execution admission rejection, or ``None``
    when the response is not a retryable admission signal (the caller then
    falls through to its existing terminal handling).

    Handles the pass-2 audit backpressure/billing signals that are safe to
    retry because NO work has been published to the queue yet — they are
    admission decisions the gateway/self-hosted server makes *before*
    dispatch, so retrying is idempotent even on the non-idempotent generate
    paths:

    * ``429 TOO_MANY_REQUESTS`` (code ``RATE_LIMIT``, B1) — per-key/per-account
      rate limiting. On give-up (``provision_timeout_s`` budget spent) raises a
      typed :class:`RateLimitError` carrying the last ``Retry-After``.
    * ``503 BILLING_CAPACITY_UNAVAILABLE`` (B2 — a gateway-local billing-family
      cap, NOT customer credit exhaustion) and ``503 QUEUE_FULL`` (B7 —
      self-hosted queue backpressure, #3180). On give-up raises the server's
      terminal 503 verbatim via :func:`handle_error` (a :class:`ServerError`
      preserving the code).

    Retry timing mirrors :func:`provisioning_retry_delay`: the server-supplied
    ``Retry-After`` is honored verbatim, and only the SDK's own fallback default
    is jittered. Give-up mirrors :func:`_handle_oom_retry`: when the budget is
    spent OR the next wait would consume the rest of it (leaving no room for the
    retried request to run), the typed root cause is surfaced NOW instead of
    sleeping the budget away and letting the outer loop's ``remaining <= 0``
    guard mask it as a ``ProvisioningError``.

    IMPORTANT: 402/403 credit/account failures are deliberately NOT handled
    here. They are terminal and must never be retried; :func:`handle_error`
    maps them to typed exceptions on the first response.
    """
    status = response.status_code
    if status == HTTP_TOO_MANY_REQUESTS:
        retry_after = get_retry_after(response)
        elapsed = time.monotonic() - start_time
        remaining = timeout - elapsed
        delay = retry_after if retry_after is not None else apply_jitter(min(RATE_LIMIT_DEFAULT_DELAY_S, remaining))
        if remaining <= 0 or delay >= remaining:
            msg = f"Rate limited (429); retry budget ({timeout:.1f}s) exhausted after {elapsed:.1f}s"
            raise RateLimitError(
                msg,
                retry_after=retry_after,
                param=get_error_param(response),
                request=parse_request_metadata(response.headers),
            )
        return delay
    if status == HTTP_SERVICE_UNAVAILABLE and get_error_code(response) in BACKPRESSURE_503_ERROR_CODES:
        retry_after = get_retry_after(response)
        elapsed = time.monotonic() - start_time
        remaining = timeout - elapsed
        delay = (
            retry_after if retry_after is not None else apply_jitter(min(BACKPRESSURE_503_DEFAULT_DELAY_S, remaining))
        )
        if remaining <= 0 or delay >= remaining:
            # Budget spent (or the next wait would consume it): surface the
            # server's terminal 503 (ServerError, code preserved) now.
            # ``handle_error`` is NoReturn.
            handle_error(response)
        return delay
    return None


def next_stream_retry_delay(
    response: _HttpResponse,
    *,
    model: str,
    gpu: str | None,
    wait_for_capacity: bool,
    start_time: float,
    timeout: float,
    oom_retries: int,
    max_oom_retries: int,
) -> tuple[float, int]:
    """Decide whether to retry an opened streaming response with a non-2xx status.

    Shared by the sync and async streaming surfaces so the pre-stream
    provisioning rules stay identical to the buffered ``generate()``. The
    caller opens the stream, and on a non-200 status buffers the body
    (``response.read()`` / ``await response.aread()``) and calls this — the
    body must already be available so :func:`get_error_code` can inspect it.

    Returns ``(delay_seconds, new_oom_retries)`` to sleep-then-retry, or
    raises a terminal error. Only explicit pre-execution signals are retried
    (503 PROVISIONING / MODEL_LOADING / RESOURCE_EXHAUSTED); a 504 and any
    other error are terminal —
    streaming generation is non-idempotent, so a post-publish retry could
    double-bill.
    """
    elapsed = time.monotonic() - start_time
    status = response.status_code

    # Non-retryable load failure / oversized input short-circuits (these
    # read the buffered body and raise their own typed errors).
    raise_if_model_load_failed(response, model=model)
    raise_if_input_too_long(response, model=model)

    if status == HTTP_SERVICE_UNAVAILABLE:
        code = get_error_code(response)
        if code == PROVISIONING_ERROR_CODE:
            delay = provisioning_retry_delay(
                response,
                gpu=gpu,
                wait_for_capacity=wait_for_capacity,
                start_time=start_time,
                timeout=timeout,
            )
            return delay, oom_retries
        if code == MODEL_LOADING_ERROR_CODE:
            if elapsed >= timeout:
                msg = f"Model loading timeout after {elapsed:.1f}s for '{model}'"
                raise ModelLoadingError(msg, model=model)
            retry_after = get_retry_after(response)
            delay = retry_after_or_default(retry_after, MODEL_LOADING_DEFAULT_DELAY_S)
            return min(delay, timeout - elapsed), oom_retries
        if code == RESOURCE_EXHAUSTED_ERROR_CODE:
            if not wait_for_capacity or oom_retries >= max_oom_retries or elapsed >= timeout:
                msg = f"Server out of memory after {oom_retries} retries for '{model}'"
                raise ResourceExhaustedError(
                    msg,
                    model=model,
                    retries=oom_retries,
                    param=get_error_param(response),
                    request=parse_request_metadata(response.headers),
                    retry_after=get_retry_after(response),
                )
            delay = compute_oom_backoff(get_retry_after(response), oom_retries)
            return min(delay, timeout - elapsed), oom_retries + 1

    if status == HTTP_GATEWAY_TIMEOUT:
        msg = (
            "Gateway timed out (504) after the request was published to the queue; "
            "a worker may already be generating. Not retried because generation is "
            "non-idempotent (retrying could double-bill)."
        )
        raise ServerError(
            msg,
            code=get_error_code(response),
            status_code=status,
            param=get_error_param(response),
            request=parse_request_metadata(response.headers),
        )

    if 300 <= status < HTTP_CLIENT_ERROR:
        msg = f"Unexpected HTTP redirect ({_terminal_response_diagnostic(response)})"
        raise RequestError(
            msg,
            status_code=status,
            request=parse_request_metadata(response.headers),
        )

    if status >= HTTP_CLIENT_ERROR:
        handle_error(response)  # always raises

    # A non-200, non-error status with no retry rule — surface rather than
    # silently treat as streamable.
    msg = f"Unexpected status {status} opening stream"
    raise ServerError(
        msg,
        code=get_error_code(response),
        status_code=status,
        param=get_error_param(response),
        request=parse_request_metadata(response.headers),
    )


def build_chat_body(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    stop: str | list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: dict[str, Any] | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    n: int | None = None,
    best_of: int | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    seed: int | None = None,
    user: str | None = None,
    safety_identifier: str | None = None,
    lora_adapter: str | None = None,
    stream_options: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``/v1/chat/completions`` request body (snake_case wire shape).

    Only fields the caller set are included so the gateway applies its own
    defaults for the rest. ``extra_body`` is merged LAST so a caller can still
    override / supply any forward-compat field not yet named on the typed
    surface (the typed kwargs win unless the caller explicitly puts the same
    key into ``extra_body``). Shared by the sync and async clients.

    Field set mirrors ``packages/sie_gateway/src/handlers/proxy.rs::chat_params_from_json``:
    the gateway rejects unknown keys with 400 ``unsupported_field`` and
    validates ranges (e.g. ``top_logprobs`` in ``[0, 20]``,
    ``logit_bias`` values in ``[-100.0, 100.0]``, ``best_of`` in
    ``[1, 128]``); ``logprobs: true`` is required when ``top_logprobs > 0``.
    """
    body: dict[str, Any] = {"model": model, "messages": list(messages)}
    if stream:
        body["stream"] = True
    optional: dict[str, Any] = {
        "max_completion_tokens": max_completion_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "stop": stop,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "response_format": response_format,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "n": n,
        "best_of": best_of,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "logit_bias": logit_bias,
        "seed": seed,
        "user": user,
        "safety_identifier": safety_identifier,
        "lora_adapter": lora_adapter,
        "stream_options": stream_options,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    if extra_body:
        body.update(extra_body)
    return body


def build_responses_body(
    model: str,
    input: str | Sequence[Mapping[str, Any]],
    *,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Assemble the strict, non-streaming ``/v1/responses`` request body.

    The gateway's Responses MVP is stateless, text-only, and non-streaming.
    Keeping this builder typed to the complete public OpenAPI allow-list
    prevents SDK callers from accidentally relying on wider parser behavior or
    OpenAI fields that the gateway contract does not expose.
    """
    body: dict[str, Any] = {
        "model": model,
        "input": input if isinstance(input, str) else list(input),
    }
    optional: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def sse_headers(resolved_gpu: str | None, pool_name: str | None) -> dict[str, str]:
    """Headers for an SSE streaming request (``Accept: text/event-stream``)."""
    headers: dict[str, str] = {
        "content-type": JSON_CONTENT_TYPE,
        "accept": "text/event-stream",
    }
    if resolved_gpu:
        headers["X-SIE-MACHINE-PROFILE"] = resolved_gpu
    if pool_name:
        headers["X-SIE-Pool"] = pool_name
    return headers


def sse_chunk_error(chunk: dict[str, Any]) -> tuple[str, str, str | None, int | None] | None:
    """Return ``(code, message, param, retry_after_s)`` for a typed SSE error.

    Both the chat and SIE-native generate surfaces put the error object at the
    top level of the chunk (see ``send_error_chunk`` in
    ``packages/sie_gateway/src/handlers/sse.rs``). The additive retry hint is
    trusted only for ``RESOURCE_EXHAUSTED`` and only in the engine config's
    integer ``1..=60`` domain; booleans are integers in Python, so exclude
    them explicitly.
    """
    err = chunk.get("error")
    if isinstance(err, dict):
        code = str(err.get("code") or "error")
        param = err.get("param")
        retry_after_s = err.get("retry_after_s")
        validated_retry_after_s = (
            retry_after_s
            if code == RESOURCE_EXHAUSTED_ERROR_CODE
            and isinstance(retry_after_s, int)
            and not isinstance(retry_after_s, bool)
            and 1 <= retry_after_s <= 60
            else None
        )
        return (
            code,
            str(err.get("message") or "stream error"),
            param if isinstance(param, str) else None,
            validated_retry_after_s,
        )
    return None


def _coerce_token_count(value: Any) -> int:
    """Best-effort coerce a usage token count to a non-negative ``int``.

    The generate-result parser is tolerant of malformed *optional* usage
    fields (mirroring how it silently skips a non-numeric ``ttft_ms`` /
    ``tpot_ms``). A non-numeric token count (``None``, a string, a list,
    …) must NOT crash the parser with an un-wrapped ``ValueError`` /
    ``TypeError`` outside the parser's :class:`RequestError` contract, so
    it degrades to ``0`` instead. ``bool`` is accepted (it is an ``int``
    subclass) and coerces to 0/1. This deliberately does not loosen the
    strict ``model`` / ``text`` checks, which still raise.

    Args:
        value: Raw value pulled from the ``usage`` dict.

    Returns:
        The integer token count, or ``0`` for any non-numeric input.
    """
    # ``math.isfinite`` guards against a non-finite float (``nan`` / ``inf``)
    # which is an ``int``/``float`` instance but blows up ``int()``:
    # ``int(nan)`` -> ``ValueError``, ``int(inf)`` -> ``OverflowError``.
    # Both would escape the parser's ``RequestError``-only contract, so they
    # degrade to ``0`` like any other non-numeric value. ``bool`` is finite
    # and coerces to 0/1 as documented.
    if isinstance(value, (int, float)) and math.isfinite(value):
        return int(value)
    return 0


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


def _missing_result_ids(
    results: Sequence[Mapping[str, Any]],
    submitted: Sequence[Any],
) -> list[str] | None:
    """Best-effort ids of submitted items absent from a shortened response.

    Only computed when ids identify every position on both sides: every
    submitted item carries a string ``id`` and every returned item echoes
    one. Otherwise the set difference could mislabel a present-but-unnamed
    item as missing, so the diagnostic degrades to ``None`` (positional
    counts only).

    This runs only while building an exception, so it degrades rather than
    raises: a non-mapping item must not replace the actionable count error
    with an ``AttributeError`` from the diagnostic itself.
    """
    if not all(isinstance(item, Mapping) for item in submitted):
        return None
    submitted_ids = [item.get("id") for item in submitted]
    if not all(isinstance(item_id, str) for item_id in submitted_ids):
        return None
    returned_ids: set[str] = set()
    for result in results:
        result_id = result.get("id")
        if not isinstance(result_id, str):
            return None
        returned_ids.add(result_id)
    missing = [item_id for item_id in submitted_ids if isinstance(item_id, str) and item_id not in returned_ids]
    return missing or None


def validate_batch_result_count(
    results: Sequence[Mapping[str, Any]],
    # `Sequence[Any]`, not `Sequence[Mapping[...]]`: callers pass the
    # single-or-list item argument, whose `Item | list[Item]` union survives
    # the isinstance ternary, and only `len()` is actually required here. The
    # id diagnostic guards element shape at runtime instead.
    submitted: Sequence[Any],
    model: str,
    *,
    operation: str,
    request: RequestMetadata | None = None,
) -> None:
    """Guard the positional batch contract: exactly one result per input item.

    Encode and extract are positional — callers rely on a 1:1 input↔output
    correspondence: both return ``results[0]`` for a single-item request, and
    batch callers reassemble results by index. The contract can break with an
    HTTP 200 whose ``items`` list is *shorter* than the request: the gateway
    returns mixed-success
    batches as ``200`` carrying only the successful items (a per-item
    server-side failure — e.g. an input exceeding the model's
    ``max_sequence_length`` — is dropped from the body, not surfaced as an
    error envelope). Without this check, that short list flows into positional
    access: either a context-free ``IndexError`` (issue #1526) or — worse —
    a silently misaligned zip that stores results against the wrong inputs.

    Raising a typed, actionable :class:`IncompleteBatchError` keeps the
    failure legible: it names the model, the expected vs. returned counts,
    the gateway request id, and — when the submitted items carried ids — the
    ids the response dropped.

    Args:
        results: Parsed results from the server response.
        submitted: The input items submitted in this HTTP request.
        model: Model name, for the error message.
        operation: ``"encode"`` or ``"extract"`` — selects the error code
            and wording.
        request: Request metadata parsed from the terminal response headers.

    Raises:
        IncompleteBatchError: If ``len(results) != len(submitted)``.
    """
    if len(results) == len(submitted):
        return
    if operation == "encode":
        label, noun, code = "Encode", "embedding(s)", "ENCODE_RESULT_COUNT_MISMATCH"
    else:
        label, noun, code = "Extract", "extraction result(s)", "EXTRACT_RESULT_COUNT_MISMATCH"
    missing_ids = _missing_result_ids(results, submitted)
    msg = (
        f"{label} response desync for model {model!r}: server returned "
        f"{len(results)} {noun} for {len(submitted)} input item(s); expected "
        f"exactly one per input. An input may have failed server-side "
        f"(e.g. exceeding the model's max_sequence_length) and been dropped from the batch."
    )
    if missing_ids is not None:
        msg += f" Missing item id(s): {', '.join(missing_ids)}."
    raise IncompleteBatchError(
        msg,
        code=code,
        expected=len(submitted),
        received=len(results),
        model=model,
        missing_ids=missing_ids,
        request=request,
    )


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
    if data.get("usage") is not None:
        result["usage"] = data["usage"]
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
        error = item.get("error")
        if error is not None:
            code = error.get("code") if isinstance(error, Mapping) else None
            message = error.get("message") if isinstance(error, Mapping) else None
            if not isinstance(code, str) or not code.strip() or not isinstance(message, str) or not message.strip():
                code = _MALFORMED_EXTRACT_ERROR_CODE
                message = _MALFORMED_EXTRACT_ERROR_MESSAGE
            result["error"] = ExtractItemErrorDetail(code=code, message=message)

        results.append(result)

    return results
