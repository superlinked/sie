from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Substrings that identify an OOM error across CUDA / MPS / generic backends.
# Lower-cased once at import time. The phrases below are specific enough to
# avoid false positives on unrelated error messages — they all appear in real
# torch / CUDA / MPS / kernel OOM strings and don't collide with normal
# inference error wording.
#
# We intentionally do NOT match the bare token "oom" (e.g., via a word-
# boundary regex). Python's ``\b`` treats ``-`` as a word boundary, which
# would mis-classify any error referencing a model whose name contains
# ``oom`` (e.g., ``"failed to load oom-classifier"``). The phrases below
# already cover every real-world OOM message we've seen from PyTorch /
# CUDA / kernel-OOM-killer ("Out of memory") / glibc ("Cannot allocate
# memory") / driver-level ("Failed to allocate").
_OOM_INDICATORS: tuple[str, ...] = (
    "out of memory",  # generic + CUDA + MPS all use this phrase
    "cannot allocate memory",  # OS allocation failures
    "failed to allocate",  # CUDA driver allocation failures
)


def is_oom_error(error: BaseException) -> bool:
    """Return True if ``error`` looks like an out-of-memory error.

    Two-tier detection:

    1. **Type check**: ``ResourceExhaustedError`` is the typed terminal
       error raised by the worker's ``BatchExecutor`` after recovery is
       exhausted. Recognising it by type makes the typed signal
       authoritative and decouples detection from the exact wording of
       the wrapped string.
    2. **Substring match** on ``str(error)`` for the un-wrapped escapes:
       - ``torch.cuda.OutOfMemoryError`` (subclass of ``RuntimeError``)
       - Plain ``RuntimeError("CUDA out of memory ...")``
       - ``RuntimeError("MPS backend out of memory ...")``
       - Generic OS-level allocation failures

    The same matcher is used at load time, at dispatch time, and on the
    HTTP / queue error paths so behaviour is consistent.
    """
    if isinstance(error, ResourceExhaustedError):
        return True
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in _OOM_INDICATORS)


class OomRecoveryAction(StrEnum):
    """Strategies the worker can apply to recover from a dispatch-time OOM.

    Applied in order; each is attempted exactly once before falling through
    to the next. ``SPLIT_BATCH`` is recursive and is the terminal step.
    """

    CACHE_CLEAR = "cache_clear"
    EVICT_LRU = "evict_lru"
    SPLIT_BATCH = "split_batch"


# Default ordering: cheapest mitigation first, most disruptive last.
_DEFAULT_STRATEGY: tuple[OomRecoveryAction, ...] = (
    OomRecoveryAction.CACHE_CLEAR,
    OomRecoveryAction.EVICT_LRU,
    OomRecoveryAction.SPLIT_BATCH,
)


@dataclass(frozen=True)
class OomRecoveryConfig:
    """Worker-side OOM recovery configuration.

    All fields are validated by the pydantic model in ``config/engine.py``;
    this dataclass is the runtime-facing form passed into ``BatchExecutor``.

    Attributes:
        enabled: Master switch. When False, OOM exceptions propagate
            unchanged (legacy behaviour).
        strategy: Ordered tuple of recovery actions. Earlier actions are
            attempted first.
        max_split_depth: Maximum recursion depth for ``SPLIT_BATCH``. Each
            split halves the batch, so ``max_split_depth=4`` permits up to
            ``2**4 = 16`` sub-batches before declaring terminal failure.
        eviction_lock_timeout_s: Soft timeout when waiting for the registry's
            load-lock during ``EVICT_LRU``. Avoids deadlock when many workers
            try to evict simultaneously; on timeout the action is skipped
            and the next strategy is tried.
        retry_after_s: Value placed in the ``Retry-After`` header on the 503
            response when recovery is exhausted. Independent of any SDK-side
            backoff; exists so callers can pace retries.
    """

    enabled: bool = True
    strategy: tuple[OomRecoveryAction, ...] = _DEFAULT_STRATEGY
    max_split_depth: int = 4
    eviction_lock_timeout_s: float = 5.0
    retry_after_s: int = 5


@dataclass
class OomRecoveryStats:
    """Counters for OOM recovery activity on a single worker.

    Exposed via ``WorkerStats.oom_recoveries`` and surfaced in Prometheus
    so a sustained recovery rate can be alerted on (typically a sign of a
    real memory leak rather than transient pressure).

    Note on partial-success semantics under ``split_batch``: when half the
    items succeed via halving and the other half terminally fail, **both**
    ``recoveries_succeeded`` and ``terminal_failures`` increment by 1 — the
    counters track distinct outcome events, not mutually-exclusive request
    fates. Operators dashboarding "% of OOMs that recovered" should use
    ``recoveries_succeeded / recoveries_attempted``; "% lost to clients"
    is ``terminal_failures / recoveries_attempted``. The two ratios sum to
    >=1 in the partial case, which is the correct signal.
    """

    # ``batch_splits`` counts top-level engagement of the split strategy
    # only (not recursive halves) — this matches the metric semantics
    # operators see in ``sie_oom_batch_splits_total``.
    cache_clears: int = 0
    evictions_triggered: int = 0
    batch_splits: int = 0
    terminal_failures: int = 0
    # How many times the recovery loop entered (== distinct OOMs caught at
    # the dispatch boundary, regardless of whether recovery ultimately
    # succeeded).
    recoveries_attempted: int = 0
    # How many recoveries had at least one metadata succeed (under split,
    # this includes partial-success cases where some items still failed).
    recoveries_succeeded: int = 0


@dataclass(frozen=True)
class ResourceExhausted:
    """Marker payload attached to the exception set on terminal failure.

    Carrying a structured marker (rather than relying on the Python type)
    lets ``InferenceErrorHandler`` distinguish "we tried to recover and
    failed" from "OOM raised but recovery is disabled". Both currently map
    to the same HTTP response, but the distinction matters for metrics.
    """

    operation: str
    attempts: int
    original_message: str = field(default="")


class ResourceExhaustedError(RuntimeError):
    """Raised inside the worker when all OOM recovery strategies are exhausted.

    Subclass of ``RuntimeError`` so existing ``except RuntimeError`` blocks
    in adapter / handler code continue to behave correctly. The marker is
    available via ``.marker`` for the API layer.
    """

    def __init__(self, message: str, marker: ResourceExhausted) -> None:
        super().__init__(message)
        self.marker = marker
