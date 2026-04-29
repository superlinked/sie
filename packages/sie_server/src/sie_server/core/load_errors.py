from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from sie_sdk.exceptions import GatedModelError

from sie_server.core.oom import is_oom_error


class LoadErrorClass(StrEnum):
    """Classification of model-load failures.

    Drives the registry's terminal-failed state machine and the API's
    retry-vs-no-retry decision in ``ensure_loaded``.

    Permanent classes (``GATED``, ``NOT_FOUND``, ``DEPENDENCY``) should not
    be retried automatically because re-attempting them produces the same
    error every time and burns request budget. ``OOM`` and ``NETWORK`` are
    transient — the registry holds a cooldown to avoid hot retry loops
    but eventually allows another attempt. ``UNKNOWN`` is treated as
    permanent for safety; operators should inspect the underlying error
    and either fix it or restart the server to clear the failure.
    """

    GATED = "GATED"
    NOT_FOUND = "NOT_FOUND"
    OOM = "OOM"
    NETWORK = "NETWORK"
    DEPENDENCY = "DEPENDENCY"
    UNKNOWN = "UNKNOWN"


# Cooldown windows (seconds) per error class. ``None`` means permanent —
# the failure stays sticky until ``clear_failure`` is invoked (e.g. by a
# config hot-reload, an explicit admin call, or a server restart).
_PERMANENT: float | None = None

_COOLDOWN_BY_CLASS: dict[LoadErrorClass, float | None] = {
    LoadErrorClass.GATED: _PERMANENT,
    LoadErrorClass.NOT_FOUND: _PERMANENT,
    LoadErrorClass.DEPENDENCY: _PERMANENT,
    LoadErrorClass.UNKNOWN: _PERMANENT,
    LoadErrorClass.OOM: 60.0,
    LoadErrorClass.NETWORK: 30.0,
}


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


def classify_load_error(exc: BaseException) -> LoadFailureClassification:
    """Classify a model-load exception into a ``LoadErrorClass``.

    Detection order matters: ``GatedModelError`` is caught before generic
    ``OSError``/``ConnectionError`` because the SDK's gated wrapper subclasses
    ``Exception`` directly, but huggingface's underlying ``GatedRepoError``
    inherits from ``HfHubHTTPError`` which is also an ``OSError``. Likewise
    OOM is detected before generic ``RuntimeError`` because torch raises
    ``RuntimeError("CUDA out of memory")`` and we want it bucketed as
    ``OOM`` not ``UNKNOWN``.

    Args:
        exc: The exception captured by ``_load_model_background``.

    Returns:
        Classification with the canonical class and cooldown.
    """
    # Gated repo (HF auth) — permanent until operator fixes HF_TOKEN.
    if isinstance(exc, GatedModelError):
        return LoadFailureClassification(
            error_class=LoadErrorClass.GATED,
            cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.GATED],
        )

    # Repo missing on the Hub — permanent.
    not_found_cls: type[BaseException] | None
    try:
        from huggingface_hub.utils import RepositoryNotFoundError as _RepositoryNotFoundError

        not_found_cls = _RepositoryNotFoundError
    except ImportError:  # pragma: no cover
        not_found_cls = None

    if not_found_cls is not None and isinstance(exc, not_found_cls):
        return LoadFailureClassification(
            error_class=LoadErrorClass.NOT_FOUND,
            cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.NOT_FOUND],
        )

    # OOM (torch / driver) — transient with cooldown so the LRU evictor or
    # operator can free memory before we try again.
    if isinstance(exc, RuntimeError) and is_oom_error(exc):
        return LoadFailureClassification(
            error_class=LoadErrorClass.OOM,
            cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.OOM],
        )

    # Missing transformer architecture or python dep — permanent until the
    # environment is upgraded. Embeddinggemma in particular needs
    # transformers >= 4.56 for ``Gemma3TextModel``.
    if isinstance(exc, ImportError | ModuleNotFoundError):
        return LoadFailureClassification(
            error_class=LoadErrorClass.DEPENDENCY,
            cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.DEPENDENCY],
        )

    # Network / IO — transient, short cooldown.
    # Includes built-in ConnectionError/TimeoutError plus library-specific
    # network exceptions from httpx and requests, which do NOT inherit from
    # the built-ins. Without these, transient download failures would fall
    # through to UNKNOWN (permanent) and require server restart to recover.
    network_excs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import httpx as _httpx

        network_excs.extend(
            [
                _httpx.TransportError,  # base for ConnectError, ReadTimeout, etc.
                _httpx.HTTPError,  # base for protocol-level errors
            ]
        )
    except ImportError:  # pragma: no cover
        pass
    try:
        import requests.exceptions as _req_exc

        network_excs.extend(
            [
                _req_exc.ConnectionError,
                _req_exc.Timeout,
                _req_exc.ChunkedEncodingError,
            ]
        )
    except ImportError:  # pragma: no cover
        pass

    if isinstance(exc, tuple(network_excs)):
        return LoadFailureClassification(
            error_class=LoadErrorClass.NETWORK,
            cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.NETWORK],
        )

    return LoadFailureClassification(
        error_class=LoadErrorClass.UNKNOWN,
        cooldown_s=_COOLDOWN_BY_CLASS[LoadErrorClass.UNKNOWN],
    )


@dataclass(frozen=True)
class LoadFailure:
    """Recorded load failure for a model.

    Stored in ``ModelRegistry._failed`` to drive the terminal-failed
    branch of the state machine and the ``MODEL_LOAD_FAILED`` API
    response.

    Attributes:
        error_class: The classified error category.
        message: Human-readable summary suitable for API responses.
        attempts: How many load attempts have been recorded so far.
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
