"""Per-pair batching cost for the score path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sie_server.core.prepared import ScorePreparedItem
from sie_server.core.timing import RequestTiming

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

SCORE_MEDIA_COST: Final[int] = 1024
MAX_SCORE_ITEMS: Final[int] = 1000


def score_item_cost(item: Item) -> int:
    """Return the approximate batching cost for one side of a score pair."""
    text_cost = len(item.text) if item.text else 0
    media_count = len(item.images or [])
    if item.audio is not None:
        media_count += 1
    if item.video is not None:
        media_count += 1
    if item.document is not None:
        media_count += 1
    return text_cost + (media_count * SCORE_MEDIA_COST)


def score_pair_cost(query: Item, doc: Item) -> int:
    """Return the approximate batching cost for a rerank query/document pair."""
    return score_item_cost(query) + score_item_cost(doc)


def build_score_prepared_items(query: Item, items: list[Item]) -> list[ScorePreparedItem]:
    """Build PreparedItems for a score request.

    ``score_pair_cost`` here is the char-count BATCHING proxy only.
    Authoritative billable counts (§7.3) are the reranker's real per-pair
    tokenizer lengths, surfaced on the resulting ScoreOutput and summed into
    ``ItemOutcome.units`` in ``_score_success_outcome`` — never this proxy.
    """
    if not items:
        raise ValueError("Score requests require at least one candidate item")
    if len(items) > MAX_SCORE_ITEMS:
        raise ValueError(f"Score requests accept at most {MAX_SCORE_ITEMS} candidate items")
    return [ScorePreparedItem(cost=score_pair_cost(query, item), original_index=i) for i, item in enumerate(items)]


def build_score_prepared_items_timed(query: Item, items: list[Item]) -> tuple[list[ScorePreparedItem], RequestTiming]:
    """Build score PreparedItems inside a tokenization-phase timing bracket.

    Single source of truth for the score prepare + timing step shared by the
    HTTP handler (``api.score``) and the sidecar IPC executor
    (``queue_executor.process_score_batch``), so the two ingress paths cannot
    drift on tokenization-phase timing accounting.
    """
    timing = RequestTiming()
    timing.start_tokenization()
    prepared_items = build_score_prepared_items(query, items)
    timing.end_tokenization()
    return prepared_items, timing
