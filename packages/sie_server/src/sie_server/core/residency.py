"""Model-residency policy types.

This module is the first seam of the planned ResidencyPolicy (issue #1569),
which will eventually own the eviction decision, the LRU order, and the
OOM-recovery cooldown that are today spread across ``core/registry.py``,
``core/memory.py`` and ``core/worker/oom_recovery.py``.

It deliberately holds only leaf types so both the registry and the worker's
OOM-recovery executor can import it without a cycle (the worker must not
import the registry).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Iterable


class EvictionResult(Enum):
    """Typed outcome of a single LRU eviction attempt.

    Replaces the bool that ``evict_lru_excluding`` used to return, which
    collapsed three distinct non-eviction outcomes into ``False`` — so the
    caller (the worker's OOM recovery) could not tell "nothing eligible to
    evict" from "the registry was too busy to acquire the lock in time". The
    eviction *decision* now reports why, which the recovery path logs and a
    later ResidencyPolicy slice can act on.
    """

    EVICTED = "evicted"
    """A sibling model was unloaded and its GPU memory freed."""

    NO_CANDIDATE = "no_candidate"
    """No eligible sibling to evict (all pinned, already unloading, not
    loaded, or the only resident model is the caller itself)."""

    LOCK_TIMEOUT = "lock_timeout"
    """The registry load-lock could not be acquired within the soft timeout;
    another residency operation (load/unload/drain) held it. Transient —
    the caller may retry."""

    UNLOAD_FAILED = "unload_failed"
    """A candidate was selected but its unload raised."""


def select_eviction_candidate(
    lru_order: Iterable[str],
    *,
    exclude_name: str,
    is_pinned: Callable[[str], bool],
    loaded: Container[str],
    unloading: Container[str],
    releasing: Container[str] = (),
) -> str | None:
    """Pick the first LRU-ordered model eligible for eviction, or ``None``.

    Walks ``lru_order`` (least-recently-used first) and returns the first entry
    that is not the caller's own ``exclude_name``, not pinned, still ``loaded``,
    not already ``unloading``, and not in ``releasing``. A name in ``releasing``
    still has an optional-memory release on its inference thread. Unload would
    wait for that thread with no deadline. Pure decision — no I/O, no unload —
    so the OOM-recovery eviction *policy* is testable in isolation and lives in
    one place instead of inline in ``ModelRegistry.evict_lru_excluding``.

    The registry state is injected (``is_pinned`` predicate, ``loaded`` /
    ``unloading`` containers) so this stays free of any registry import — the
    worker's OOM-recovery path must not import the registry. This is the next
    slice of the ResidencyPolicy seam described in this module's docstring.
    """
    for candidate in lru_order:
        if candidate == exclude_name:
            continue
        if is_pinned(candidate):
            continue
        if candidate not in loaded:
            continue
        if candidate in unloading:
            continue
        if candidate.lower() in releasing:
            continue
        return candidate
    return None
