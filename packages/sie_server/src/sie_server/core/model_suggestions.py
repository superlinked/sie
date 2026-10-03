"""Near-match suggestions for unknown model ids.

The catalog carries ~160 models keyed by their full canonical id
(``BAAI/bge-m3``, ``Alibaba-NLP/gte-Qwen2-1.5B-instruct``). There is no
short-name aliasing, so a caller who drops the org prefix, mistypes a case,
or misremembers a suffix gets an exact-match miss. "Model 'X' not found" then
leaves them to guess against a list they cannot see.

Two lookup mistakes dominate, and plain edit distance only catches one:

- **a missing or wrong org prefix** (``bge-m3`` for ``BAAI/bge-m3``) — handled
  by matching the part after the last ``/`` as well as the whole id, so a
  short name still surfaces its owner;
- **a typo or case slip** (``BAAI/bge-M3``, ``qwen/Qwen3-0.6B``) — handled by
  case-insensitive fuzzy matching.

Suggestions are advisory: they only ever add a sentence to an error that is
raised regardless.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

# Below this similarity the "suggestion" is noise; the caller is better served
# by the bare "not found" than by a confidently wrong guess.
_SIMILARITY_CUTOFF = 0.6

MAX_SUGGESTIONS = 3


def _basename(model_id: str) -> str:
    """The id without its org prefix (``BAAI/bge-m3`` -> ``bge-m3``)."""
    return model_id.rpartition("/")[2]


def suggest_models(unknown: str, known: Iterable[str], *, limit: int = MAX_SUGGESTIONS) -> list[str]:
    """Return catalog ids closest to ``unknown``, best first.

    Args:
        unknown: The id the caller asked for.
        known: Every id the catalog serves.
        limit: Maximum suggestions to return.

    Returns:
        Up to ``limit`` ids from ``known``, ordered best-match first. Empty
        when nothing is close enough to be worth printing.
    """
    query = unknown.strip().lower()
    if not query or limit <= 0:
        return []

    candidates = [model_id for model_id in known if model_id]
    if not candidates:
        return []

    # Score each id on its best reading: the whole id, or just its basename.
    # A bare "bge-m3" scores 1.0 against BAAI/bge-m3's basename, so the exact
    # model the caller meant sorts above coincidentally similar full ids.
    scored: list[tuple[float, str]] = []
    query_base = _basename(query)
    for model_id in candidates:
        lowered = model_id.lower()
        score = max(
            difflib.SequenceMatcher(None, query, lowered).ratio(),
            difflib.SequenceMatcher(None, query_base, _basename(lowered)).ratio(),
        )
        if score >= _SIMILARITY_CUTOFF:
            scored.append((score, model_id))

    # Sort by descending score, then by id so equal scores stay deterministic.
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [model_id for _score, model_id in scored[:limit]]


def suggestion_suffix(unknown: str, known: Iterable[str], *, limit: int = MAX_SUGGESTIONS) -> str:
    """A ``. Did you mean ...?`` fragment to append to a not-found message.

    Returns an empty string when there is no confident suggestion, so callers
    can append it unconditionally.
    """
    matches = suggest_models(unknown, known, limit=limit)
    if not matches:
        return ""
    quoted = ", ".join(f"'{model_id}'" for model_id in matches)
    return f" Did you mean {quoted}?"
