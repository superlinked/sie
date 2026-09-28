from __future__ import annotations

import errno
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import botocore.exceptions
import httpx
import requests.exceptions
from huggingface_hub.errors import (
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from sie_sdk.exceptions import GatedModelError

from sie_server.core.oom import is_oom_error


class LoadErrorClass(StrEnum):
    """Classification of model-load failures.

    Drives the registry's failed state machine and the API's
    retry-vs-no-retry decision in ``ensure_loaded``.

    Permanent classes (``GATED``, ``NOT_FOUND``, ``DEPENDENCY``, ``CONFIG``)
    are not retried automatically because re-attempting them produces the
    same error every time; a config change or a restart clears them. Every
    other class is transient: the registry holds a cooldown that doubles with
    each failed attempt, and a failure that is still recurring after
    :data:`MAX_TRANSIENT_ATTEMPTS` attempts becomes permanent. ``UNKNOWN``
    is transient with a long base cooldown.
    """

    GATED = "GATED"
    NOT_FOUND = "NOT_FOUND"
    OOM = "OOM"
    NETWORK = "NETWORK"
    DEPENDENCY = "DEPENDENCY"
    TIMEOUT = "TIMEOUT"
    PLACEMENT = "PLACEMENT"
    CONFIG = "CONFIG"
    STORAGE = "STORAGE"
    ENGINE = "ENGINE"
    UNKNOWN = "UNKNOWN"


_PERMANENT_CLASSES: frozenset[LoadErrorClass] = frozenset(
    {
        LoadErrorClass.GATED,
        LoadErrorClass.NOT_FOUND,
        LoadErrorClass.DEPENDENCY,
        LoadErrorClass.CONFIG,
    }
)

_BASE_COOLDOWN_S: dict[LoadErrorClass, float] = {
    LoadErrorClass.OOM: 60.0,
    LoadErrorClass.NETWORK: 30.0,
    LoadErrorClass.TIMEOUT: 30.0,
    LoadErrorClass.PLACEMENT: 30.0,
    LoadErrorClass.STORAGE: 60.0,
    LoadErrorClass.ENGINE: 30.0,
    LoadErrorClass.UNKNOWN: 600.0,
}

MAX_COOLDOWN_S = 1800.0
MAX_TRANSIENT_ATTEMPTS = 8

_TRANSIENT_HTTP_STATUSES = frozenset({408, 429})

# Throttling and transient error codes that botocore's standard retry mode
# retries; S3 returns several of them with a 4xx status.
_TRANSIENT_AWS_ERROR_CODES = frozenset(
    {
        "BandwidthLimitExceeded",
        "EC2ThrottledException",
        "LimitExceededException",
        "PriorRequestNotComplete",
        "ProvisionedThroughputExceededException",
        "RequestLimitExceeded",
        "RequestThrottled",
        "RequestThrottledException",
        "RequestTimeout",
        "RequestTimeoutException",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "ThrottledException",
        "TooManyRequestsException",
        "TransactionInProgressException",
    }
)

_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    httpx.TransportError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    botocore.exceptions.ConnectionError,
    botocore.exceptions.HTTPClientError,
    botocore.exceptions.IncompleteReadError,
)

_NOT_FOUND_EXCEPTIONS: tuple[type[BaseException], ...] = (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    EntryNotFoundError,
    DisabledRepoError,
)

_STORAGE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_STORAGE_MESSAGES = ("no space left on device", "disk quota exceeded")

_MAX_CHAIN_DEPTH = 32


class DevicePlacementError(RuntimeError):
    """No device, or no block of devices, can take this load right now.

    Another model loading or unloading changes the answer, so the failure is
    retried after a cooldown and cleared on unload rather than staying sticky
    the way an unclassified error does.
    """


class ModelConfigurationError(RuntimeError):
    """The model's configuration can never load on this worker as written.

    Classified as the permanent ``CONFIG`` class: only a config change or a
    restart can clear it.
    """


class EngineStartupError(RuntimeError):
    """An engine child process could not start: it crashed during startup or its port was taken."""


class EngineExitedError(RuntimeError):
    """The engine child process serving a loaded model has exited."""


class ModelLoadTimeoutError(TimeoutError):
    """Raised when a post-download load stage exceeds ``SIE_MODEL_LOAD_TIMEOUT_S``.

    Subclasses ``TimeoutError`` so callers that only know about the built-in
    type still match, but the dedicated class lets ``classify_load_error``
    bucket these into :class:`LoadErrorClass.TIMEOUT` rather than the
    generic ``NETWORK`` bucket. Carries structured fields for ops triage.
    """

    def __init__(self, *, model: str, stage: str, elapsed_s: float, timeout_s: float) -> None:
        self.model = model
        self.stage = stage
        self.elapsed_s = elapsed_s
        self.timeout_s = timeout_s
        super().__init__(
            f"Model '{model}' {stage} exceeded timeout: elapsed={elapsed_s:.1f}s, configured={timeout_s:.0f}s"
        )


@dataclass(frozen=True)
class LoadFailureClassification:
    """Result of classifying a load-time exception."""

    error_class: LoadErrorClass
    cooldown_s: float | None
    """Seconds to suppress retries; ``None`` for permanent failures."""

    @property
    def is_permanent(self) -> bool:
        """True if this failure should never auto-retry."""
        return self.cooldown_s is None


def cooldown_for(error_class: LoadErrorClass, attempts: int) -> float | None:
    """Return the retry cooldown after the ``attempts``-th consecutive failure.

    Permanent classes return ``None``. Transient classes double their base
    cooldown with each attempt, capped at :data:`MAX_COOLDOWN_S`, and return
    ``None`` once ``attempts`` exceeds :data:`MAX_TRANSIENT_ATTEMPTS`.
    """
    base = _BASE_COOLDOWN_S.get(error_class)
    if base is None or error_class in _PERMANENT_CLASSES or attempts > MAX_TRANSIENT_ATTEMPTS:
        return None
    return min(base * 2 ** (max(attempts, 1) - 1), MAX_COOLDOWN_S)


def classify_load_error(exc: BaseException, *, attempts: int = 1) -> LoadFailureClassification:
    """Classify a model-load exception by its cause.

    Walks the exception chain (``__cause__``, ``__context__`` and the
    ``last_exception`` that boto3 and s3transfer retry wrappers carry) from
    the outermost exception inwards. The first link that matches a specific
    rule decides the class, so a typed wrapper such as ``GatedModelError``
    or ``ModelLoadTimeoutError`` wins over what it wraps, while an untyped
    wrapper such as huggingface_hub's ``LocalEntryNotFoundError`` is
    classified by the error it wraps. When no link matches, a ``ValueError``
    anywhere in the chain is a permanent ``CONFIG`` error and anything else
    is ``UNKNOWN``.

    Args:
        exc: The exception captured by ``_load_model_background``.
        attempts: Consecutive failed attempts including this one.

    Returns:
        Classification with the canonical class and cooldown.
    """
    chain = list(_exception_chain(exc))
    error_class = next((c for c in map(_classify_link, chain) if c is not None), None)
    if error_class is None:
        error_class = (
            LoadErrorClass.CONFIG if any(isinstance(link, ValueError) for link in chain) else LoadErrorClass.UNKNOWN
        )
    return LoadFailureClassification(error_class=error_class, cooldown_s=cooldown_for(error_class, attempts))


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending and len(seen) < _MAX_CHAIN_DEPTH:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        last_exception = getattr(current, "last_exception", None)
        if isinstance(last_exception, BaseException):
            pending.append(last_exception)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)


def _classify_link(exc: BaseException) -> LoadErrorClass | None:
    try:
        return _classify_single(exc)
    except Exception:  # noqa: BLE001 - a third-party attribute accessor must not break classification
        return None


def _classify_single(exc: BaseException) -> LoadErrorClass | None:
    if isinstance(exc, ModelLoadTimeoutError):
        return LoadErrorClass.TIMEOUT
    if isinstance(exc, DevicePlacementError):
        return LoadErrorClass.PLACEMENT
    if isinstance(exc, GatedModelError | GatedRepoError):
        return LoadErrorClass.GATED
    if isinstance(exc, ModelConfigurationError | OfflineModeIsEnabled):
        return LoadErrorClass.CONFIG
    if isinstance(exc, _NOT_FOUND_EXCEPTIONS) and not isinstance(exc, LocalEntryNotFoundError):
        return LoadErrorClass.NOT_FOUND
    if isinstance(exc, RuntimeError) and is_oom_error(exc):
        return LoadErrorClass.OOM
    if isinstance(exc, ImportError):
        return LoadErrorClass.DEPENDENCY
    if _is_storage_exhausted(exc):
        return LoadErrorClass.STORAGE
    if isinstance(exc, EngineStartupError | EngineExitedError) or (
        isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE
    ):
        return LoadErrorClass.ENGINE
    status = _http_status(exc)
    if status is not None:
        if status in _TRANSIENT_HTTP_STATUSES or 500 <= status <= 599:
            return LoadErrorClass.NETWORK
        if status in (401, 403) and isinstance(exc, HfHubHTTPError):
            return LoadErrorClass.GATED
    if _aws_error_code(exc) in _TRANSIENT_AWS_ERROR_CODES:
        return LoadErrorClass.NETWORK
    if isinstance(exc, _NETWORK_EXCEPTIONS):
        return LoadErrorClass.NETWORK
    return None


def _is_storage_exhausted(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno in _STORAGE_ERRNOS:
        return True
    message = str(exc).lower()
    return any(indicator in message for indicator in _STORAGE_MESSAGES)


def _http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        candidates = [metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None]
    else:
        candidates = [getattr(exc, "status_code", None), getattr(response, "status_code", None)]
    for candidate in candidates:
        if isinstance(candidate, int) and not isinstance(candidate, bool) and 100 <= candidate <= 599:
            return candidate
    return None


def _aws_error_code(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


@dataclass(frozen=True)
class LoadFailure:
    """Recorded load failure for a model.

    Stored in ``ModelRegistry._failed`` to drive the failed branch of the
    state machine and the ``MODEL_LOAD_FAILED`` API response.

    Attributes:
        error_class: The classified error category.
        message: Human-readable summary suitable for API responses.
        attempts: How many consecutive load attempts have failed so far.
        last_attempt_ts: ``time.monotonic()`` value at last attempt.
        cooldown_s: Seconds to suppress further retries; ``None`` means
            the failure is permanent until explicitly cleared.
    """

    error_class: LoadErrorClass
    message: str
    attempts: int
    last_attempt_ts: float
    cooldown_s: float | None

    @property
    def is_permanent(self) -> bool:
        """True when retries are not auto-scheduled."""
        return self.cooldown_s is None

    def in_cooldown(self, now: float) -> bool:
        """Whether the failure is still within its cooldown window.

        Permanent failures are always in cooldown.
        """
        if self.cooldown_s is None:
            return True
        return (now - self.last_attempt_ts) < self.cooldown_s
