"""Bounded span decoding for GLiFormer's BIO extraction heads.

GLiFormer finds entity, relation-endpoint, and structuring spans from
per-word start, end, and inside scores. Its span decoder
(``SpanDecoder._calculate_span_score``) pairs every start above the threshold
with every end above it in Python, testing each same-label pair with tensor
operations, and removes overlaps (``greedy_search``) by testing each span
against every kept one. The structuring head proposes spans with GLiNER's
``extract_spans_from_tokens``, which builds a starts x ends matrix and keeps
every pair whose range stays inside, then processes each proposal in Python.
All of this grows with the product of start and end counts, which a low
threshold on a long, entity-dense document turns into seconds or minutes.

:func:`pair_spans`, :func:`select_spans`, and :func:`propose_spans` return
the same spans, in the same order and with the same scores, from vectorized
array operations whose cost grows with the number of spans they return, and
they bound that number: a document row keeps at most ``max_candidates``
candidate spans or proposals. When more pass the threshold, only the
highest-scoring ones are kept, cut at a score so that tied candidates are
kept or dropped together. Overlap removal takes spans best score first, so a
cut changes the entities only by leaving out spans that rank below it.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

# Candidate spans (or structuring proposals) one document row keeps. Measured
# on 2048-word documents at the 0.1 threshold floor with 64 labels: up to 903
# for prose and 1768 for a list of names; repetitive text can produce hundreds
# of thousands.
MAX_SPAN_CANDIDATES = 4096

_FLOAT32_BITS_ONE = int(np.float32(1.0).view(np.int32))


def pair_spans(
    start_idx: tuple[list[int], list[int]] | list[list[int]],
    end_idx: tuple[list[int], list[int]] | list[list[int]],
    scores_inside: torch.Tensor,
    scores_start: torch.Tensor,
    scores_end: torch.Tensor,
    id_to_classes: Mapping[int, Any],
    threshold: float,
    *,
    make_span: Callable[..., Any],
    max_width: int | None = None,
    max_candidates: int = MAX_SPAN_CANDIDATES,
) -> list[Any]:
    """Pair start and end positions into scored spans, like ``_calculate_span_score``.

    A span's score is the minimum of its start, end, and inside probabilities
    and of one minus the inside probability just outside each end.

    Args:
        start_idx: ``[positions, labels]`` whose start probability passed the
            threshold, in row-major (position, label) order.
        end_idx: The same for end probabilities.
        scores_inside: ``(L, C)`` inside probabilities.
        scores_start: ``(L, C)`` start probabilities.
        scores_end: ``(L, C)`` end probabilities.
        id_to_classes: Label id to label name; when non-empty, starts of other
            labels are skipped.
        threshold: Every inside probability of a span must reach it.
        make_span: Builds one span from ``start``, ``end``, ``entity_type``
            and ``score`` keyword arguments.
        max_width: Longest span in positions, if bounded.
        max_candidates: Most spans returned; see the module docstring.

    Returns:
        Spans ordered by start position, then label, then end position.
    """
    start_positions, start_labels = (list(part) for part in start_idx)
    end_positions, end_labels = (list(part) for part in end_idx)
    if not start_positions or not end_positions:
        return []
    s_pos = np.asarray(start_positions, dtype=np.int64)
    s_lab = np.asarray(start_labels, dtype=np.int64)
    if id_to_classes:
        known = np.fromiter((label in id_to_classes for label in start_labels), dtype=bool, count=len(start_labels))
        s_pos, s_lab = s_pos[known], s_lab[known]
    # The same element-wise operations as the package's decoder, on the same
    # tensors, so comparisons and complements round exactly as they do there.
    span_start, span_end, label, score = _bounded_pairs(
        s_pos,
        s_lab,
        np.asarray(end_positions, dtype=np.int64),
        np.asarray(end_labels, dtype=np.int64),
        bad=(scores_inside < threshold).detach().cpu().numpy(),
        inside=_as_float32(scores_inside),
        start_scores=_as_float32(scores_start),
        end_scores=_as_float32(scores_end),
        outside=_as_float32(1.0 - scores_inside),
        max_width=max_width,
        max_candidates=max_candidates,
    )
    names: dict[int, Any] = {}
    spans = []
    for st, ed, lab, value in zip(span_start.tolist(), span_end.tolist(), label.tolist(), score.tolist(), strict=True):
        name = names.get(lab)
        if name is None:
            name = names[lab] = id_to_classes.get(lab, str(lab))
        spans.append(make_span(start=st, end=ed, entity_type=name, score=value))
    return spans


def propose_spans(
    scores: torch.Tensor,
    labels: torch.Tensor | None = None,
    threshold: float = 0.5,
    *,
    original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    max_candidates: int = MAX_SPAN_CANDIDATES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Span proposals from BIO logits, like GLiNER's ``extract_spans_from_tokens``.

    GLiFormer's structuring head proposes every span whose start, end, and
    every inside probability exceed the threshold, for any label, and scores
    each proposal against every record slot. A proposal's rank here is the
    minimum of those probabilities, the same value the head uses to accept a
    field for it.

    Args:
        scores: ``(B, L, C, 3)`` start/end/inside logits.
        labels: Gold BIO labels (training); proposals then come from
            ``original`` unchanged.
        threshold: Probability every start, end, and inside score must exceed.
        original: GLiNER's function, for the labelled case.
        max_candidates: Most proposals per row; see the module docstring.

    Returns:
        ``(span_idx, span_mask)``: ``(B, N, 2)`` start/end positions and the
        ``(B, N)`` mask of real proposals, N at least 1.
    """
    if labels is not None:
        return original(scores, labels, threshold)
    batch = scores.shape[0]
    # The same element-wise operations as GLiNER, on the same tensors.
    probs = torch.sigmoid(scores)
    start_mask = probs[..., 0] > threshold
    end_mask = probs[..., 1] > threshold
    inside_mask = probs[..., 2] > threshold
    rows = []
    for row in range(batch):
        starts = start_mask[row].nonzero(as_tuple=False).cpu().numpy()
        ends = end_mask[row].nonzero(as_tuple=False).cpu().numpy()
        if starts.size == 0 or ends.size == 0:
            rows.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
            continue
        row_probs = _as_float32(probs[row])
        span_start, span_end, _, _ = _bounded_pairs(
            starts[:, 0],
            starts[:, 1],
            ends[:, 0],
            ends[:, 1],
            bad=(~inside_mask[row]).cpu().numpy(),
            inside=np.ascontiguousarray(row_probs[..., 2]),
            start_scores=np.ascontiguousarray(row_probs[..., 0]),
            end_scores=np.ascontiguousarray(row_probs[..., 1]),
            outside=None,
            max_width=None,
            max_candidates=max_candidates,
        )
        rows.append((span_start, span_end))
    width = max(1, max((len(span_start) for span_start, _ in rows), default=0))
    span_idx = torch.zeros(batch, width, 2, dtype=torch.long, device=scores.device)
    span_mask = torch.zeros(batch, width, dtype=torch.bool, device=scores.device)
    for row, (span_start, span_end) in enumerate(rows):
        if len(span_start):
            pairs = torch.from_numpy(np.stack([span_start, span_end], axis=1))
            span_idx[row, : len(span_start)] = pairs.to(scores.device)
            span_mask[row, : len(span_start)] = True
    return span_idx, span_mask


def _bounded_pairs(
    s_pos: np.ndarray,
    s_lab: np.ndarray,
    e_pos: np.ndarray,
    e_lab: np.ndarray,
    *,
    bad: np.ndarray,
    inside: np.ndarray,
    start_scores: np.ndarray,
    end_scores: np.ndarray,
    outside: np.ndarray | None,
    max_width: int | None,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Same-label (start, end) pairs whose range has no ``bad`` position, best first cut.

    Pairs come in start order, then by end position. A pair's score is the
    minimum of its start, end, and inside scores and, with ``outside``, of the
    outside scores next to it. When more than ``max_candidates`` pairs
    qualify, only those at or above the lowest score cut that leaves at most
    that many are returned.

    Returns:
        ``(span_start, span_end, label, score)`` arrays.
    """
    empty = np.empty(0, dtype=np.int64)
    if s_pos.size == 0 or e_pos.size == 0:
        return empty, empty, empty, np.empty(0, dtype=np.float32)
    length, labels = inside.shape
    positions = np.arange(length, dtype=np.int64)[:, None]
    infinite = np.float32(np.inf)
    start_val = start_scores[s_pos, s_lab]
    end_val = end_scores[e_pos, e_lab]
    if outside is not None:
        start_val = np.minimum(start_val, np.where(s_pos > 0, outside[np.maximum(s_pos - 1, 0), s_lab], infinite))
        end_val = np.minimum(
            end_val, np.where(e_pos + 1 < length, outside[np.minimum(e_pos + 1, length - 1), e_lab], infinite)
        )

    uncut: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def reach(cut: np.float32 | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kept starts and ends at score ``cut``, and each start's exclusive end limit."""
        if cut is None and uncut:
            return uncut[0]
        if cut is None:
            s_keep = np.ones(s_pos.size, dtype=bool)
            e_keep = np.ones(e_pos.size, dtype=bool)
            limit_bad = _next_true(bad, positions, length)
        else:
            s_keep = start_val >= cut
            e_keep = end_val >= cut
            limit_bad = _next_true(bad | (inside < cut), positions, length)
        limit = limit_bad[s_pos, s_lab]
        if max_width is not None:
            limit = np.minimum(limit, s_pos + max_width)
        result = (s_keep, e_keep, np.maximum(limit, s_pos))
        if cut is None:
            uncut.append(result)
        return result

    def count(cut: np.float32 | None) -> int:
        """Candidates at score ``cut``: kept ends in each kept start's range, from prefix sums."""
        s_keep, e_keep, limit = reach(cut)
        ends = np.zeros((length + 1, labels), dtype=np.int64)
        ends[e_pos[e_keep] + 1, e_lab[e_keep]] = 1  # (position, label) pairs are distinct
        np.cumsum(ends, axis=0, out=ends)
        return int((ends[limit, s_lab] - ends[s_pos, s_lab])[s_keep].sum())

    cut = _score_cut(count, max_candidates) if count(None) > max_candidates else None
    s_keep, e_keep, limit = reach(cut)
    stride = length + 1
    keys = np.sort(e_lab[e_keep] * stride + e_pos[e_keep])
    starts = np.flatnonzero(s_keep)
    lab = s_lab[starts]
    low = np.searchsorted(keys, lab * stride + s_pos[starts], side="left")
    high = np.searchsorted(keys, lab * stride + limit[starts], side="left")
    counts = high - low
    total = int(counts.sum())
    if total == 0:
        return empty, empty, empty, np.empty(0, dtype=np.float32)

    owner = np.repeat(starts, counts)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    span_start, label = s_pos[owner], s_lab[owner]
    span_end = keys[np.repeat(low, counts) + offsets] - label * stride

    # Minimum inside score over each span, from one reduction over the
    # label-major scores (padded so every range end is a valid index).
    flat = np.append(inside.T.ravel(), np.float32(0.0))
    bounds = np.empty(2 * total, dtype=np.int64)
    bounds[0::2] = label * length + span_start
    bounds[1::2] = label * length + span_end + 1
    score = np.minimum.reduceat(flat, bounds)[0::2]
    score = np.minimum(score, start_val[owner])
    score = np.minimum(score, end_scores[span_end, label])
    if outside is not None:
        right = np.where(span_end + 1 < length, outside[np.minimum(span_end + 1, length - 1), label], infinite)
        score = np.minimum(score, right)
    return span_start, span_end, label, score


def _as_float32(tensor: torch.Tensor) -> np.ndarray:
    """Contiguous host float32 copy; exact for float16, bfloat16, and float32 values."""
    return np.ascontiguousarray(tensor.detach().to(device="cpu", dtype=torch.float32).numpy())


def _next_true(mask: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
    """Per column, the first row at or after each row where ``mask`` holds, else ``length``."""
    marked = np.where(mask, positions, length)
    return np.minimum.accumulate(marked[::-1], axis=0)[::-1]


def _score_cut(count: Callable[[np.float32], int], max_candidates: int) -> np.float32:
    """Lowest score cut that leaves at most ``max_candidates`` candidates.

    A candidate's score is a minimum of per-position scores, so it reaches a
    cut exactly when each of them does, and ``count`` needs no enumeration.
    The count only falls as the cut rises, so the lowest passing cut is found
    by bisecting the float32 bit patterns of [0, 1] (about 30 counts).
    """
    low_bits, high_bits = 0, _FLOAT32_BITS_ONE + 1  # nothing scores above 1
    while low_bits < high_bits:
        middle = (low_bits + high_bits) // 2
        if count(np.int32(middle).view(np.float32)) <= max_candidates:
            high_bits = middle
        else:
            low_bits = middle + 1
    return np.int32(low_bits).view(np.float32)


def select_spans(spans: list[Any], flat_ner: bool = True, multi_label: bool = False) -> list[Any]:
    """Remove overlapping spans best first, exactly like the package's ``greedy_search``.

    The package tests each span against every span kept so far. Kept flat
    spans never overlap one another (except for exact repeats of the same
    boundaries, with ``multi_label``), so a flat test only needs the kept span
    starting last at or before the candidate's end: a binary search. Nested
    mode tests every kept span at once with array operations.

    Args:
        spans: Candidate spans with ``start``, ``end``, and ``score``.
        flat_ner: Reject any overlap; otherwise only partial overlaps (nested
            spans are kept).
        multi_label: Keep spans with the same boundaries as a kept span.

    Returns:
        The kept spans, ordered by start position.
    """
    if not spans:
        return []
    ranked = sorted(spans, key=lambda item: -item.score)
    kept = _select_flat(ranked, multi_label) if flat_ner else _select_nested(ranked, multi_label)
    kept.sort(key=lambda item: item.start)
    return kept


def _select_flat(ranked: list[Any], multi_label: bool) -> list[Any]:
    # Distinct kept boundaries, sorted; they are pairwise disjoint.
    starts: list[int] = []
    ends: list[int] = []
    kept: list[Any] = []
    for span in ranked:
        start, end = span.start, span.end
        index = bisect.bisect_right(starts, end) - 1
        if index >= 0 and ends[index] >= start:
            # Overlaps the kept span at ``index`` (the only one it can reach).
            if not (multi_label and starts[index] == start and ends[index] == end):
                continue
            kept.append(span)
            continue
        starts.insert(index + 1, start)
        ends.insert(index + 1, end)
        kept.append(span)
    return kept


def _select_nested(ranked: list[Any], multi_label: bool) -> list[Any]:
    kept_starts = np.empty(len(ranked), dtype=np.int64)
    kept_ends = np.empty(len(ranked), dtype=np.int64)
    kept: list[Any] = []
    for span in ranked:
        start, end = span.start, span.end
        count = len(kept)
        if count:
            starts, ends = kept_starts[:count], kept_ends[:count]
            crossing = (start <= ends) & (starts <= end)
            crossing &= ~(((start <= starts) & (end >= ends)) | ((starts <= start) & (ends >= end)))
            same = (starts == start) & (ends == end)
            if bool(np.where(same, not multi_label, crossing).any()):
                continue
        kept_starts[count], kept_ends[count] = start, end
        kept.append(span)
    return kept
