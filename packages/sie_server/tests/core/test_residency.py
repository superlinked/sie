# Tests for the pure eviction-decision helper in core/residency.py.
#
# ``select_eviction_candidate`` is the *policy* half of
# ``ModelRegistry.evict_lru_excluding`` (the registry keeps the I/O half).
# Exercising it directly locks the LRU-walk skip rules without standing up a
# registry — the worker's OOM-recovery path must not import the registry.

from __future__ import annotations

from sie_server.core.residency import select_eviction_candidate


def _never_pinned(_name: str) -> bool:
    return False


def test_selects_first_eligible_in_lru_order() -> None:
    got = select_eviction_candidate(
        ["a", "b", "c"],
        exclude_name="z",
        is_pinned=_never_pinned,
        loaded={"a", "b", "c"},
        unloading=set(),
    )
    assert got == "a"


def test_skips_exclude_pinned_ghost_and_unloading() -> None:
    # a == exclude, b pinned, c is a ghost (in LRU order but not loaded),
    # d is already unloading, e is the first eligible sibling.
    got = select_eviction_candidate(
        ["a", "b", "c", "d", "e"],
        exclude_name="a",
        is_pinned=lambda n: n == "b",
        loaded={"a", "b", "d", "e"},  # c absent -> ghost, skipped
        unloading={"d"},
    )
    assert got == "e"


def test_skips_a_model_whose_release_is_still_running() -> None:
    got = select_eviction_candidate(
        ["releasing", "free"],
        exclude_name="caller",
        is_pinned=_never_pinned,
        loaded={"releasing", "free"},
        unloading=set(),
        releasing=frozenset({"releasing"}),
    )
    assert got == "free"


def test_returns_none_when_nothing_eligible() -> None:
    got = select_eviction_candidate(
        ["only-self"],
        exclude_name="only-self",
        is_pinned=_never_pinned,
        loaded={"only-self"},
        unloading=set(),
    )
    assert got is None
