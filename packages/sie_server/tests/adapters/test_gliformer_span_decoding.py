"""Bounded GLiFormer span decoding: same spans as the package, at bounded cost (no model weights)."""

from __future__ import annotations

import gc
import importlib
import random
import sys
import time
from dataclasses import astuple
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gliner.modeling.utils import extract_spans_from_tokens
from sie_server.adapters.gliformer import adapter as adapter_module
from sie_server.adapters.gliformer import relation_decoding, span_decoding, structuring_decoding
from sie_server.adapters.gliformer.span_decoding import MAX_SPAN_CANDIDATES, pair_spans, propose_spans, select_spans

# Import the package the way the adapter does (without its AutoModel
# overrides); that also installs the bounded decoding on its span decoder.
adapter_module._import_gliformer()
upstream = importlib.import_module("gliformer.tasks.span_decoder")
assert upstream.SpanDecoder._calculate_span_score._sie_bounded is True
assert upstream.SpanDecoder.greedy_search._sie_bounded is True
structuring_model = importlib.import_module("gliformer.tasks.structuring.model")
assert structuring_model.extract_spans_from_tokens._sie_original is extract_spans_from_tokens
# The package's own methods, kept on the replacements.
_UPSTREAM_PAIRING = upstream.SpanDecoder._calculate_span_score._sie_original
_UPSTREAM_GREEDY = upstream.SpanDecoder.greedy_search._sie_original
_DECODER = upstream.SpanDecoder(SimpleNamespace())


def _bio_scores(seed: int, length: int, labels: int, dtype: torch.dtype) -> torch.Tensor:
    """(L, C, 3) start/end/inside logits with entity-like runs and noise."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(length, labels, 3, generator=generator) * 2.5
    for label in range(labels):
        for _ in range(max(1, length // 6)):
            start = int(torch.randint(0, length, (1,), generator=generator))
            end = min(length - 1, start + int(torch.randint(0, 6, (1,), generator=generator)))
            logits[start, label, 0] += 4
            logits[end, label, 1] += 4
            logits[start : end + 1, label, 2] += 4
    return logits.to(dtype)


def _inputs(logits: torch.Tensor, threshold: float) -> tuple[Any, ...]:
    """Arguments of ``_calculate_span_score`` exactly as ``decode_bio_spans`` builds them."""
    scores_start, scores_end, scores_inside = logits.permute(2, 0, 1)
    return (
        _DECODER._get_indices_above_threshold(scores_start, threshold),
        _DECODER._get_indices_above_threshold(scores_end, threshold),
        torch.sigmoid(scores_inside),
        torch.sigmoid(scores_start),
        torch.sigmoid(scores_end),
    )


def _as_tuples(spans: list[Any]) -> list[tuple[Any, ...]]:
    return [astuple(span) for span in spans]


def _upstream(logits: torch.Tensor, id_to_classes: dict[int, Any], threshold: float, **kwargs: Any) -> list[Any]:
    return _UPSTREAM_PAIRING(_DECODER, *_inputs(logits, threshold), id_to_classes, threshold, **kwargs)


def _bounded(logits: torch.Tensor, id_to_classes: dict[int, Any], threshold: float, **kwargs: Any) -> list[Any]:
    return pair_spans(*_inputs(logits, threshold), id_to_classes, threshold, make_span=upstream.Span, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7])
@pytest.mark.parametrize("seed", range(6))
def test_pairing_matches_the_package_span_for_span(seed: int, threshold: float, dtype: torch.dtype) -> None:
    logits = _bio_scores(seed, length=40 + 7 * seed, labels=1 + seed % 4, dtype=dtype)
    labels = logits.shape[1]
    for id_to_classes in ({i: f"label {i}" for i in range(labels)}, {0: "only the first"}, {}):
        expected = _upstream(logits, id_to_classes, threshold)
        actual = _bounded(logits, id_to_classes, threshold)
        # Same spans, same order, bit-identical scores and types.
        assert _as_tuples(actual) == _as_tuples(expected)
        assert all(type(a.score) is type(e.score) and type(a.start) is int for a, e in zip(actual, expected))


@pytest.mark.parametrize("max_width", [0, 1, 3, 12])
def test_pairing_matches_the_package_with_a_width_limit(max_width: int) -> None:
    probabilities = torch.sigmoid(_bio_scores(11, length=60, labels=1, dtype=torch.float32)[:, 0, :])
    # The single-label path: probabilities in, indices recomputed from logits.
    start = _DECODER._get_indices_above_threshold(probabilities[:, 0:1].logit(), 0.2)
    end = _DECODER._get_indices_above_threshold(probabilities[:, 1:2].logit(), 0.2)
    args = (start, end, probabilities[:, 2:3], probabilities[:, 0:1], probabilities[:, 1:2], {0: ""}, 0.2)
    expected = _UPSTREAM_PAIRING(_DECODER, *args, max_width=max_width)
    assert _as_tuples(pair_spans(*args, make_span=upstream.Span, max_width=max_width)) == _as_tuples(expected)


def test_scores_at_the_threshold_compare_as_the_package_does() -> None:
    # Inside exactly at the threshold passes (the test is "< threshold");
    # start and end exactly at it do not ("> threshold").
    threshold = 0.3
    at = torch.tensor(threshold, dtype=torch.float32).logit()
    logits = torch.full((6, 1, 3), -8.0)
    logits[1, 0, 0] = logits[4, 0, 1] = 5.0
    logits[1:5, 0, 2] = at
    logits[2, 0, 0] = at
    expected = _upstream(logits, {0: "x"}, threshold)
    assert expected
    assert _as_tuples(_bounded(logits, {0: "x"}, threshold)) == _as_tuples(expected)


def _head(items: list[Any], limit: int) -> list[Any]:
    """The first ``limit`` items overlap removal would take (stable, best score first), in their order."""
    taken = set(sorted(range(len(items)), key=lambda index: -items[index].score)[:limit])
    return [item for index, item in enumerate(items) if index in taken]


@pytest.mark.parametrize("limit", [1, 7, 50, 200])
@pytest.mark.parametrize("seed", range(4))
def test_capped_pairing_keeps_the_best_candidates_in_order(seed: int, limit: int) -> None:
    logits = _bio_scores(seed, length=120, labels=3, dtype=torch.float32)
    names = {0: "a", 1: "b", 2: "c"}
    full = _upstream(logits, names, 0.1)
    assert len(full) > limit
    capped = _bounded(logits, names, 0.1, max_candidates=limit)
    assert _as_tuples(capped) == _as_tuples(_head(full, limit))
    # Overlap removal takes spans best first, so the capped result is the
    # head of the full one: the spans it would have kept first.
    for flat_ner in (True, False):
        for multi_label in (True, False):
            kept = select_spans(capped, flat_ner, multi_label)
            everything = _UPSTREAM_GREEDY(_DECODER, full, flat_ner, multi_label)
            head = {id(span) for span in _head(full, limit)}
            assert _as_tuples(kept) == _as_tuples([span for span in everything if id(span) in head])


def test_capped_pairing_keeps_tied_candidates_in_the_package_order() -> None:
    # Three identical one-word entities tie: a cap of 2 keeps the first two.
    logits = torch.full((5, 1, 3), -8.0)
    for position in (0, 2, 4):
        logits[position, 0] = 6.0
    names = {0: "x"}
    assert [span.start for span in _bounded(logits, names, 0.5)] == [0, 2, 4]
    assert [span.start for span in _bounded(logits, names, 0.5, max_candidates=2)] == [0, 2]
    assert [span.start for span in _bounded(logits, names, 0.5, max_candidates=1)] == [0]


def test_capped_pairing_fills_the_cap_from_a_large_tie() -> None:
    # Saturated scores: every start, end, and inside probability is 1, so all
    # spans away from the edges tie at 0 (their outside neighbour is 1).
    logits = torch.full((300, 2, 3), 30.0)
    names = {0: "a", 1: "b"}
    full = _upstream(logits[:60], names, 0.5)
    assert len({span.score for span in full}) == 2
    for limit in (1, 2, 3, 100, 1000):
        assert _as_tuples(_bounded(logits[:60], names, 0.5, max_candidates=limit)) == _as_tuples(_head(full, limit))
    assert len(_bounded(logits, names, 0.5)) == MAX_SPAN_CANDIDATES


def _random_spans(seed: int, count: int, length: int) -> list[Any]:
    """Spans with short and long widths, repeated boundaries, and tied scores."""
    rng = random.Random(seed)  # noqa: S311 -- reproducible test data
    spans = []
    for _ in range(count):
        start = rng.randrange(length)
        end = min(length - 1, start + rng.choice((0, 0, 1, 2, 5, 20)))
        score = rng.choice((0.5, 0.75, 0.9)) if rng.random() < 0.3 else rng.random()
        spans.append(upstream.Span(start=start, end=end, entity_type=rng.choice("abc"), score=score))
    return spans


@pytest.mark.parametrize("multi_label", [False, True])
@pytest.mark.parametrize("flat_ner", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_overlap_removal_matches_the_package(seed: int, flat_ner: bool, multi_label: bool) -> None:
    spans = _random_spans(seed, count=10 + 40 * seed, length=5 + 30 * seed)
    expected = _UPSTREAM_GREEDY(_DECODER, list(spans), flat_ner, multi_label)
    assert select_spans(list(spans), flat_ner, multi_label) == expected
    assert select_spans([], flat_ner, multi_label) == []


@pytest.mark.parametrize("multi_label", [False, True])
def test_nested_overlap_removal_matches_the_package_on_chains_and_crossings(multi_label: bool) -> None:
    rng = random.Random(7)  # noqa: S311 -- reproducible test data
    # Fully nested chains, their crossings, and long spans, with ties.
    spans = [upstream.Span(start=i, end=400 - i, entity_type="a", score=rng.choice((0.5, 0.9))) for i in range(200)]
    spans += [upstream.Span(start=i + 1, end=401 - i, entity_type="b", score=rng.random()) for i in range(200)]
    spans += [upstream.Span(start=i, end=i, entity_type="c", score=rng.random()) for i in range(0, 400, 7)]
    spans += [upstream.Span(start=s, end=s, entity_type="d", score=0.5) for s in (3, 3, 5)]
    rng.shuffle(spans)
    assert select_spans(list(spans), False, multi_label) == _UPSTREAM_GREEDY(_DECODER, list(spans), False, multi_label)


def test_fully_nested_spans_are_removed_quickly() -> None:
    spans = [upstream.Span(start=i, end=8190 - i, entity_type="a", score=1.0 - i / 8192) for i in range(4095)]
    kept = select_spans(spans, False, False)
    assert len(kept) == 4095


def test_overlap_removal_through_the_decoder_class_uses_the_bounded_code() -> None:
    spans = _random_spans(1, count=60, length=40)
    assert _DECODER.greedy_search(spans, False, True) == _UPSTREAM_GREEDY(_DECODER, spans, False, True)


def test_worst_case_document_is_bounded() -> None:
    # Every position of every label passes: ~134M same-label pairs uncapped.
    logits = 1.0 + 3.0 * torch.rand(2048, 64, 3, generator=torch.Generator().manual_seed(0))
    spans = _bounded(logits, {i: str(i) for i in range(64)}, 0.1)
    assert MAX_SPAN_CANDIDATES // 2 < len(spans) <= MAX_SPAN_CANDIDATES
    assert select_spans(spans, True, True)


def test_pairing_without_starts_or_ends_returns_nothing() -> None:
    logits = torch.full((4, 2, 3), -8.0)
    assert _bounded(logits, {0: "a", 1: "b"}, 0.5) == []


# -- Per-document allowances --------------------------------------------------------


def _row_inputs(logits: torch.Tensor, threshold: float) -> tuple[Any, ...]:
    return _inputs(logits, threshold)


def test_allowances_belong_to_rows() -> None:
    assert span_decoding.row_allowance() is None  # outside a forward pass
    with span_decoding.document_allowances([5, 7]) as allowances:
        with span_decoding.decoding_row(1):
            assert span_decoding.row_allowance() is allowances[1]
        assert span_decoding.row_allowance(0) is allowances[0]
        with pytest.raises(RuntimeError, match="not one of the pass's documents"):
            span_decoding.row_allowance()
        with span_decoding.decoding_row(2), pytest.raises(RuntimeError, match="not one of the pass's documents"):
            span_decoding.row_allowance()
    allowance = span_decoding.Allowance(10)
    assert allowance.affordable(3) == 3
    assert allowance.affordable(3, reserve=5) == 1
    allowance.spend(25)
    assert allowance.remaining == 0


def _cells(logits: torch.Tensor, threshold: float) -> int:
    """Score cells a row is charged for: its labels, up to one past its last start or end."""
    probs = torch.sigmoid(logits.float())
    passing = ((probs[..., 0] > threshold) | (probs[..., 1] > threshold)).any(dim=1)
    return (int(passing.nonzero().max()) + 2) * logits.shape[1]


def test_pairing_within_an_allowance_keeps_the_best_first_prefix() -> None:
    logits = _bio_scores(2, length=80, labels=3, dtype=torch.float32)
    names = {0: "a", 1: "b", 2: "c"}
    full = _upstream(logits, names, 0.1)
    cells = _cells(logits, 0.1)
    count_cost = -(-cells // 32)
    with span_decoding.document_allowances([10**9, count_cost + len(full), count_cost + cells + 10, 50]) as rows:
        results = []
        for row in range(4):
            with span_decoding.decoding_row(row):
                results.append(_bounded(logits, names, 0.1))
    # Plenty, or exactly enough: everything, at one unit per span plus the count.
    assert _as_tuples(results[0]) == _as_tuples(full)
    assert 10**9 - rows[0].remaining == count_cost + len(full)
    assert _as_tuples(results[1]) == _as_tuples(full)
    assert rows[1].remaining == 0
    # Enough for the cut and 10 spans: the head of the full result.
    assert _as_tuples(results[2]) == _as_tuples(_head(full, 10))
    assert rows[2].remaining == 0
    # Not enough to find a cut: nothing, and only the count is spent.
    assert results[3] == []
    assert rows[3].remaining == 50 - count_cost


def test_candidate_units_raise_the_cost_of_each_span() -> None:
    logits = _bio_scores(4, length=40, labels=2, dtype=torch.float32)
    names = {0: "a", 1: "b"}
    full = _upstream(logits, names, 0.2)
    with span_decoding.document_allowances([10**9]) as rows, span_decoding.decoding_row(0):
        with span_decoding.candidate_units(64):
            _bounded(logits, names, 0.2)
    assert 10**9 - rows[0].remaining == -(-_cells(logits, 0.2) // 32) + 64 * len(full)


def test_proposals_within_an_allowance_keep_the_best_first_prefix() -> None:
    scores = _bio_scores(8, length=30, labels=3, dtype=torch.float32).unsqueeze(0)
    ranked = _ranked_proposals(scores[0], 0.1)
    cells = _cells(scores[0], 0.1)
    units = -(-cells // 32) + cells + span_decoding.PROPOSAL_UNITS * 5
    with span_decoding.document_allowances([units]) as rows:
        kept = _proposals(*_propose(scores, 0.1))[0]
    assert len(ranked) > 5
    assert kept == [(item.start, item.end) for item in _head(ranked, 5)]
    assert rows[0].remaining == 0


def _padded(logits: torch.Tensor, positions: int) -> torch.Tensor:
    """``logits`` followed by padded positions, as the padding mask leaves them."""
    padding = torch.full((positions, *logits.shape[1:]), float("-inf"), dtype=logits.dtype)
    return torch.cat([logits, padding])


@pytest.mark.parametrize("units", [10**9, 300])
def test_pairing_costs_the_same_however_far_the_batch_pads_the_document(units: int) -> None:
    logits = _bio_scores(5, length=40, labels=3, dtype=torch.float32)
    names = {0: "a", 1: "b", 2: "c"}
    kept, spent = [], []
    for row_logits in (logits, _padded(logits, 400)):
        with span_decoding.document_allowances([units]) as rows, span_decoding.decoding_row(0):
            kept.append(_as_tuples(_bounded(row_logits, names, 0.1)))
        spent.append(units - rows[0].remaining)
    assert kept[0]
    assert kept[0] == kept[1]
    assert spent[0] == spent[1]
    assert units > 10**6 or len(kept[0]) < len(_upstream(logits, names, 0.1))


def test_proposals_cost_the_same_however_far_the_batch_pads_the_document() -> None:
    logits = _bio_scores(6, length=30, labels=3, dtype=torch.float32)
    kept, spent = [], []
    for row_logits in (logits, _padded(logits, 300)):
        with span_decoding.document_allowances([10**9]) as rows:
            kept.append(_proposals(*_propose(row_logits.unsqueeze(0), 0.1))[0])
        spent.append(10**9 - rows[0].remaining)
    assert kept[0]
    assert kept[0] == kept[1]
    assert spent[0] == spent[1]


# -- Structuring span proposals ---------------------------------------------------


def _propose(scores: torch.Tensor, threshold: float, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return propose_spans(scores, None, threshold, original=extract_spans_from_tokens, **kwargs)


def _proposals(span_idx: torch.Tensor, span_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    return [[tuple(pair) for pair in idx[mask].tolist()] for idx, mask in zip(span_idx, span_mask, strict=True)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("seed", range(4))
def test_proposals_match_gliner(seed: int, threshold: float, dtype: torch.dtype) -> None:
    scores = torch.stack([_bio_scores(seed * 3 + row, length=50, labels=3, dtype=torch.float32) for row in range(3)])
    scores[1, 30:] = float("-inf")  # a padded row
    scores = scores.to(dtype)
    expected_idx, expected_mask = extract_spans_from_tokens(scores, None, threshold)
    actual_idx, actual_mask = _propose(scores, threshold)
    assert torch.equal(actual_mask, expected_mask)
    assert torch.equal(actual_idx, expected_idx)
    assert actual_idx.dtype == expected_idx.dtype
    assert actual_mask.dtype == expected_mask.dtype


def test_proposals_without_candidates_match_gliner() -> None:
    scores = torch.full((2, 6, 2, 3), -8.0)
    for actual, expected in zip(_propose(scores, 0.5), extract_spans_from_tokens(scores, None, 0.5), strict=True):
        assert torch.equal(actual, expected)


def test_labelled_proposals_use_gliner_unchanged() -> None:
    scores = _bio_scores(1, length=20, labels=2, dtype=torch.float32).unsqueeze(0)
    labels = (torch.rand(scores.shape, generator=torch.Generator().manual_seed(1)) > 0.5).float()
    for actual, expected in zip(
        propose_spans(scores, labels, 0.5, original=extract_spans_from_tokens),
        extract_spans_from_tokens(scores, labels, 0.5),
        strict=True,
    ):
        assert torch.equal(actual, expected)


def _ranked_proposals(scores: torch.Tensor, threshold: float) -> list[Any]:
    """GLiNER's proposals for one row, in its order, each with the head's field score."""
    probs = torch.sigmoid(scores)
    start, end, inside = probs[..., 0] > threshold, probs[..., 1] > threshold, probs[..., 2] > threshold
    ends = end.nonzero().tolist()
    ranked = []
    for s_pos, label in start.nonzero().tolist():
        for e_pos, e_label in ends:
            if e_label == label and e_pos >= s_pos and bool(inside[s_pos : e_pos + 1, label].all()):
                value = min(
                    float(probs[s_pos, label, 0]),
                    float(probs[e_pos, label, 1]),
                    float(probs[s_pos : e_pos + 1, label, 2].min()),
                )
                ranked.append(SimpleNamespace(start=s_pos, end=e_pos, score=value))
    return ranked


@pytest.mark.parametrize("limit", [1, 10, 60])
def test_capped_proposals_keep_the_best_in_order(limit: int) -> None:
    scores = _bio_scores(8, length=90, labels=3, dtype=torch.float32).unsqueeze(0)
    ranked = _ranked_proposals(scores[0], 0.1)
    assert [(item.start, item.end) for item in ranked] == _proposals(*extract_spans_from_tokens(scores, None, 0.1))[0]
    assert len(ranked) > limit
    kept = _proposals(*_propose(scores, 0.1, max_candidates=limit))[0]
    assert kept == [(item.start, item.end) for item in _head(ranked, limit)]


def test_worst_case_proposals_are_bounded() -> None:
    scores = (1.0 + 3.0 * torch.rand(1, 2048, 16, 3, generator=torch.Generator().manual_seed(0))).float()
    _, span_mask = _propose(scores, 0.1)
    assert 0 < int(span_mask.sum()) <= MAX_SPAN_CANDIDATES


# -- Structuring records ----------------------------------------------------------

structuring_decoder = importlib.import_module("gliformer.tasks.structuring.decoder")
_UPSTREAM_STRUCTURING_DECODE = structuring_decoder.StructuringDecoder.decode._sie_original
_STRUCTURING = structuring_decoder.StructuringDecoder(SimpleNamespace())


def _structuring_output(seed: int, *, groups: int = 2, slots: int = 6, entities: int = 24, fields: int = 4) -> Any:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, 30, (groups, entities, 1), generator=generator)
    widths = torch.randint(0, 4, (groups, entities, 1), generator=generator)
    return SimpleNamespace(
        structuring_logits=torch.randn(groups, slots, entities, generator=generator) * 2,
        structuring_field_logits=torch.randn(groups, entities, fields, generator=generator) * 2,
        structuring_span_idx=torch.cat([starts, starts + widths], dim=-1),
        structuring_span_mask=torch.rand(groups, entities, generator=generator) > 0.2,
        structuring_anchor_mask=torch.rand(groups, slots, generator=generator) > 0.3,
        structuring_batch_origin=torch.arange(groups),
        batch_size=groups,
    )


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
@pytest.mark.parametrize(("flat_ner", "multi_label"), [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("seed", range(4))
def test_structuring_decode_matches_the_package(seed: int, threshold: float, flat_ner: bool, multi_label: bool) -> None:
    output = _structuring_output(seed)
    texts = [[f"w{i}" for i in range(40)] for _ in range(2)]
    kwargs = {"threshold": threshold, "flat_ner": flat_ner, "multi_label": multi_label, "texts": texts}
    expected = _UPSTREAM_STRUCTURING_DECODE(_STRUCTURING, output, **kwargs)
    assert expected
    assert any(expected[0])
    assert _STRUCTURING.decode(output, **kwargs) == expected


def test_structuring_decode_without_outputs_matches_the_package() -> None:
    empty = SimpleNamespace(structuring_logits=None, structuring_span_idx=None, structuring_span_mask=None)
    assert _STRUCTURING.decode(empty) == _UPSTREAM_STRUCTURING_DECODE(_STRUCTURING, empty) == []


def _slot_args(seed: int, *, slots: int = 4, entities: int = 40, fields: int = 5, saturate: bool = False) -> tuple:
    output = _structuring_output(seed, groups=1, slots=slots, entities=entities, fields=fields)
    membership = torch.sigmoid(output.structuring_logits[0])
    field_probs = torch.sigmoid(output.structuring_field_logits[0])
    if saturate:
        membership = torch.ones_like(membership)
        field_probs = torch.ones_like(field_probs)
    return (
        membership,
        field_probs,
        output.structuring_span_idx[0],
        torch.where(output.structuring_span_mask[0])[0],
        None,
        0.1,
        {i: f"f{i}" for i in range(fields)},
    )


@pytest.mark.parametrize("saturate", [False, True])
@pytest.mark.parametrize("limit", [1, 5, 30, 100])
def test_capped_documents_keep_each_slots_best_spans(limit: int, saturate: bool) -> None:
    args = _slot_args(3, saturate=saturate)
    full = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_spans=10**9)
    capped = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_spans=limit)
    flat = [(slot, span) for slot, spans in full for span in spans]
    assert len(flat) > limit
    head = {id(span) for span in _head([span for _, span in flat], limit)}
    assert [slot for slot, _ in capped] == [slot for slot, _ in full]
    for (_, everything), (_, kept) in zip(full, capped, strict=True):
        # Each slot keeps the head of its own best-first order...
        assert _as_tuples(kept) == _as_tuples([span for span in everything if id(span) in head])
        # ...so its overlap removal keeps the head of the full result.
        assert _as_tuples(select_spans(kept, False, False)) == _as_tuples(
            [span for span in _UPSTREAM_GREEDY(_DECODER, everything, False, False) if id(span) in head]
        )
    assert sum(len(spans) for _, spans in capped) == limit


def test_record_spans_are_bounded_per_document_not_per_slot() -> None:
    args = _slot_args(5, slots=100, entities=64, fields=8, saturate=True)
    capped = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_spans=1000)
    assert sum(len(spans) for _, spans in capped) == 1000
    assert len(capped) == 100


def test_record_text_is_bounded_per_document() -> None:
    span = upstream.Span
    slots = [
        (0, [span(start=0, end=99, entity_type="a", score=0.9), span(start=0, end=9, entity_type="a", score=0.5)]),
        (1, [span(start=0, end=49, entity_type="b", score=0.7), span(start=60, end=60, entity_type="b", score=0.6)]),
    ]
    assert structuring_decoding._within_word_limit(slots, 161) is slots
    # 100 + 50 words fit in 155; the next best (1 word, score 0.6) fits too; the last (10 words) does not.
    kept = structuring_decoding._within_word_limit(slots, 155)
    assert [(slot, [(s.start, s.end) for s in spans]) for slot, spans in kept] == [
        (0, [(0, 99)]),
        (1, [(0, 49), (60, 60)]),
    ]
    assert structuring_decoding._within_word_limit(slots, 50) == [(0, []), (1, [])]


def test_fully_nested_records_stay_within_the_text_bound() -> None:
    spans = [(i, 8190 - i) for i in range(4095)]
    count = len(spans)
    output = SimpleNamespace(
        structuring_logits=torch.full((1, 3, count), 4.0),
        structuring_field_logits=torch.full((1, count, 1), 4.0),
        structuring_span_idx=torch.tensor([spans]),
        structuring_span_mask=torch.ones(1, count, dtype=torch.bool),
        structuring_batch_origin=torch.arange(1),
        batch_size=1,
    )
    records = _STRUCTURING.decode(output, threshold=0.1, flat_ner=False, texts=[[f"w{i}" for i in range(8191)]])
    fields = [field for group in records[0] for record in group for field in record]
    words = sum(field["end"] - field["start"] + 1 for field in fields)
    assert 0 < words <= structuring_decoding.MAX_RECORD_WORDS


def test_record_spans_within_an_allowance_keep_each_slots_best_spans() -> None:
    args = _slot_args(3)
    full = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_spans=10**9)
    flat = [span for _, spans in full for span in spans]
    proposals = len(args[3])
    cut_cost = (4 * proposals + proposals * 5) // 4  # open slots x proposals + proposals x fields, over 4
    allowance = span_decoding.Allowance(cut_cost + 30)
    capped = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_spans=10**9, allowance=allowance)
    head = {id(span) for span in _head(flat, 30)}
    for (_, everything), (_, kept) in zip(full, capped, strict=True):
        assert _as_tuples(kept) == _as_tuples([span for span in everything if id(span) in head])
    assert allowance.remaining == 0


# -- Replacing the package's functions ---------------------------------------------


def test_package_functions_are_replaced_once_with_their_originals_kept() -> None:
    owner = SimpleNamespace()

    def original(scores, labels=None, threshold=0.5):
        return "original"

    def replacement(scores, labels=None, threshold=0.5):
        return "replacement"

    owner.extract = original
    adapter_module._replace_package_function(owner, "extract", replacement)
    assert owner.extract is replacement
    assert replacement._sie_original is original
    adapter_module._replace_package_function(owner, "extract", lambda scores, labels=None, threshold=0.5: None)
    assert owner.extract is replacement


def test_package_function_replacement_fails_closed() -> None:
    def original(scores, labels=None, threshold=0.5):
        return None

    def other_parameters(scores, threshold=0.5):
        return None

    with pytest.raises(RuntimeError, match="is missing"):
        adapter_module._replace_package_function(SimpleNamespace(), "extract", original)
    with pytest.raises(RuntimeError, match="changed its parameters"):
        adapter_module._replace_package_function(SimpleNamespace(extract=original), "extract", other_parameters)
    with pytest.raises(RuntimeError, match="must be verified again"):
        adapter_module._replace_package_function(
            SimpleNamespace(extract=original),
            "extract",
            lambda scores, labels=None, threshold=0.5: None,
            source_sha256="0" * 64,
        )


def test_the_package_functions_are_bounded_after_import() -> None:
    assert upstream.SpanDecoder._calculate_span_score._sie_bounded is True
    assert upstream.SpanDecoder.greedy_search._sie_bounded is True
    assert structuring_decoder.StructuringDecoder.decode._sie_bounded is True
    assert structuring_model.extract_spans_from_tokens._sie_bounded is True
    # Importing again leaves the replacements in place.
    replaced = upstream.SpanDecoder._calculate_span_score
    adapter_module._import_gliformer()
    assert upstream.SpanDecoder._calculate_span_score is replaced


def test_every_gliformer_binding_of_the_proposal_function_is_bounded() -> None:
    for name in adapter_module._PROPOSAL_MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "extract_spans_from_tokens"):
            assert module.extract_spans_from_tokens._sie_bounded is True
    adapter_module._assert_bounded_decoding()


def test_bounded_decoding_check_rejects_an_override() -> None:
    class _Overriding(upstream.SpanDecoder):
        def greedy_search(self, spans, flat_ner=True, multi_label=False):
            return spans

    try:
        with pytest.raises(RuntimeError, match="overrides"):
            adapter_module._assert_bounded_decoding()
    finally:
        del _Overriding
        gc.collect()
    adapter_module._assert_bounded_decoding()


def test_bounded_decoding_check_rejects_a_leftover_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    leftover = ModuleType("gliformer.tasks.leftover")
    leftover.extract_spans_from_tokens = extract_spans_from_tokens
    monkeypatch.setitem(sys.modules, "gliformer.tasks.leftover", leftover)
    with pytest.raises(RuntimeError, match="still binds"):
        adapter_module._assert_bounded_decoding()


def test_replacements_are_verified_against_the_package_at_load() -> None:
    adapter_module._verify_bounded_decoding(None)


@pytest.mark.parametrize("target", ["propose_spans", "pair_spans", "select_spans", "make_structuring_decode"])
def test_verification_fails_closed_when_a_replacement_differs(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    # Swap the real implementation behind one installed replacement for one
    # that drops its last result, as a changed package would look.
    if target == "make_structuring_decode":
        decode = structuring_decoder.StructuringDecoder.decode

        def changed(self: Any, *args: Any, **kwargs: Any) -> Any:
            return [[group[:-1] for group in item] for item in decode(self, *args, **kwargs)]

        changed._sie_original = decode._sie_original
        changed._sie_bounded = True
        monkeypatch.setattr(structuring_decoder.StructuringDecoder, "decode", changed)
    elif target == "propose_spans":
        real = adapter_module.propose_spans

        def changed(*args: Any, **kwargs: Any) -> Any:
            span_idx, span_mask = real(*args, **kwargs)
            return span_idx, span_mask & False

        monkeypatch.setattr(adapter_module, target, changed)
    else:
        real = getattr(adapter_module, target)
        monkeypatch.setattr(adapter_module, target, lambda *args, **kwargs: real(*args, **kwargs)[:-1])
    with pytest.raises(RuntimeError, match="no longer matches its bounded replacement"):
        adapter_module._verify_bounded_decoding(None)


# -- Padding --------------------------------------------------------------------


def _head_output(logits: torch.Tensor, mask: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(logits=logits, extra={} if mask is None else {"mask": mask})


def test_padded_positions_are_masked_and_decode_as_the_document_alone() -> None:
    alone = _bio_scores(5, length=9, labels=2, dtype=torch.float32)
    # Batched with a longer document: the head scores padding with logit 0.
    padded = torch.zeros(2, 20, 2, 3)
    padded[0, :9] = alone
    padded[1] = _bio_scores(6, length=20, labels=2, dtype=torch.float32)
    mask = torch.zeros(2, 20, dtype=torch.long)
    mask[0, :9] = 1
    mask[1] = 1
    names = {0: "a", 1: "b"}
    for threshold in (0.1, 0.3, 0.5):
        unmasked = _upstream(padded[0], names, threshold)
        assert any(span.end >= 9 or span.score == 0.5 for span in unmasked) or threshold == 0.5
        output = adapter_module._mask_padded_ner_logits(None, (), _head_output(padded.clone(), mask))
        assert torch.equal(output.logits[1], padded[1])
        assert bool(torch.isneginf(output.logits[0, 9:]).all())
        assert _as_tuples(_upstream(output.logits[0], names, threshold)) == _as_tuples(
            _upstream(alone, names, threshold)
        )


def test_padding_mask_leaves_unpadded_batches_alone() -> None:
    logits = torch.randn(2, 5, 3, 3)
    output = _head_output(logits, torch.ones(2, 5, dtype=torch.bool))
    assert adapter_module._mask_padded_ner_logits(None, (), output).logits is logits
    empty = SimpleNamespace(logits=None, extra={})
    assert adapter_module._mask_padded_ner_logits(None, (), empty) is empty


@pytest.mark.parametrize("mask", [None, torch.ones(2, 4, dtype=torch.bool), torch.ones(5, dtype=torch.bool)])
def test_padding_mask_fails_closed_without_a_matching_word_mask(mask: torch.Tensor | None) -> None:
    with pytest.raises(RuntimeError, match="word mask"):
        adapter_module._mask_padded_ner_logits(None, (), _head_output(torch.zeros(2, 5, 3, 3), mask))


def test_padding_mask_is_installed_on_the_ner_head() -> None:
    ner_head = torch.nn.Identity()
    reusing = SimpleNamespace(_owns_ner_head=False, _reused_ner_head=ner_head)
    model = SimpleNamespace(heads={"ner": ner_head, "joint_relex": reusing, "structuring": reusing})
    adapter_module._mask_padded_words(model)
    assert list(ner_head._forward_hooks.values()) == [adapter_module._mask_padded_ner_logits]
    assert "_forward_entity_ner" not in vars(reusing)
    with pytest.raises(RuntimeError, match="no NER head"):
        adapter_module._mask_padded_words(SimpleNamespace(heads={}))


def test_padding_mask_covers_a_head_with_its_own_ner_pass() -> None:
    padded = torch.zeros(1, 6, 2, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
    calls = []

    class _StructuringHead:
        _owns_ner_head = True

        def _forward_entity_ner(self, shared: Any, flat_inputs: Any, **kwargs: Any) -> Any:
            calls.append((shared, flat_inputs, kwargs))
            return _head_output(padded.clone(), mask)

    head = _StructuringHead()
    adapter_module._mask_padded_words(SimpleNamespace(heads={"ner": torch.nn.Identity(), "structuring": head}))
    output = head._forward_entity_ner("shared", "inputs", threshold=0.3)
    assert calls == [("shared", "inputs", {"threshold": 0.3})]
    assert bool(torch.isneginf(output.logits[0, 3:]).all())
    assert torch.equal(output.logits[0, :3], padded[0, :3])


@pytest.mark.parametrize(
    "head",
    [
        SimpleNamespace(_owns_ner_head=False, _reused_ner_head=torch.nn.Identity()),
        SimpleNamespace(_owns_ner_head=True),
        SimpleNamespace(),
    ],
)
def test_padding_mask_fails_closed_on_an_ner_pass_it_cannot_reach(head: Any) -> None:
    with pytest.raises(RuntimeError, match="cannot mask"):
        adapter_module._mask_padded_words(SimpleNamespace(heads={"ner": torch.nn.Identity(), "joint_relex": head}))


# -- Relations --------------------------------------------------------------------

joint_relex_decoder = importlib.import_module("gliformer.tasks.joint_relex.decoder")
_UPSTREAM_RELATION_DECODE = joint_relex_decoder.JointRelexDecoder.decode._sie_original
_RELATIONS = joint_relex_decoder.JointRelexDecoder(SimpleNamespace())


def _relation_output(
    seed: int,
    *,
    rows: int = 2,
    length: int = 30,
    labels: int = 3,
    pairs: int = 40,
    types: int = 4,
    entity_spans: bool = False,
) -> SimpleNamespace:
    generator = torch.Generator().manual_seed(seed)
    ner = torch.stack(
        [_bio_scores(seed * 5 + row, length=length, labels=labels, dtype=torch.float32) for row in range(rows)]
    )
    idx = torch.randint(0, 12, (rows, pairs, 2), generator=generator)
    output = SimpleNamespace(
        ner_logits=ner,
        ner_batch_origin=torch.arange(rows),
        span_logits=None,
        span_idx=None,
        span_mask=None,
        joint_rel_logits=torch.randn(rows, pairs, types, generator=generator) * 2,
        joint_rel_idx=idx,
        joint_rel_mask=torch.rand(rows, pairs, generator=generator) > 0.2,
        joint_rel_batch_origin=torch.arange(rows),
        batch_size=rows,
    )
    if entity_spans:
        starts = torch.randint(0, length - 3, (rows, 12, 1), generator=generator)
        output.joint_rel_entity_spans = torch.cat(
            [starts, starts + torch.randint(0, 3, (rows, 12, 1), generator=generator)], -1
        )
    return output


@pytest.mark.parametrize("entity_spans", [False, True])
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("seed", range(5))
def test_relation_decoding_matches_the_package(seed: int, threshold: float, entity_spans: bool) -> None:
    output = _relation_output(seed, entity_spans=entity_spans)
    texts = [[f"w{i}" for i in range(30)] for _ in range(2)]
    for flat_ner in (True, False):
        kwargs = {"threshold": threshold, "flat_ner": flat_ner, "texts": texts}
        expected = _UPSTREAM_RELATION_DECODE(_RELATIONS, output, **kwargs)
        assert _RELATIONS.decode(output, **kwargs) == expected
    assert entity_spans or any(expected)


def test_relation_decoding_without_relation_outputs_matches_the_package() -> None:
    empty = SimpleNamespace(joint_rel_logits=None, joint_rel_idx=None)
    assert _RELATIONS.decode(empty) == _UPSTREAM_RELATION_DECODE(_RELATIONS, empty) == []


def _flat(relations: list) -> list[dict]:
    return [triple for item in relations for group in item for triple in group]


@pytest.mark.parametrize("limit", [1, 5, 40])
def test_capped_relations_keep_the_best_first_prefix(limit: int) -> None:
    output = _relation_output(1, rows=1, pairs=60, types=6)
    kwargs = {"threshold": 0.1, "texts": [[f"w{i}" for i in range(30)]]}
    full = _flat(_UPSTREAM_RELATION_DECODE(_RELATIONS, output, **kwargs))
    assert len(full) > limit
    capped = _flat(
        relation_decoding.make_relation_decode(
            importlib.import_module("gliformer.tasks.ner.decoder").NERDecoder.decode,
            joint_relex_decoder.unflatten_by_batch_origin,
            max_relations=limit,
        )(_RELATIONS, output, **kwargs)
    )
    expected = [triple for triple in full if id(triple) in {id(t.triple) for t in _head(_scored(full), limit)}]
    assert capped == expected


def _scored(triples: list[dict]) -> list[Any]:
    return [SimpleNamespace(score=triple["score"], triple=triple) for triple in triples]


def test_relation_text_is_bounded_per_document() -> None:
    output = _relation_output(2, rows=1, pairs=60, types=6)
    kwargs = {"threshold": 0.1, "texts": [[f"w{i}" for i in range(30)]]}
    full = _flat(_UPSTREAM_RELATION_DECODE(_RELATIONS, output, **kwargs))
    words = [t["head"]["end"] - t["head"]["start"] + 1 + t["tail"]["end"] - t["tail"]["start"] + 1 for t in full]
    budget = sum(words) // 3
    capped = _flat(
        relation_decoding.make_relation_decode(
            importlib.import_module("gliformer.tasks.ner.decoder").NERDecoder.decode,
            joint_relex_decoder.unflatten_by_batch_origin,
            max_words=budget,
        )(_RELATIONS, output, **kwargs)
    )
    ranked = sorted(range(len(full)), key=lambda index: -full[index]["score"])
    taken, used = set(), 0
    for index in ranked:
        if used + words[index] > budget:
            break
        used += words[index]
        taken.add(index)
    assert capped == [triple for index, triple in enumerate(full) if index in taken]
    assert 0 < len(capped) < len(full)


def test_relations_within_an_allowance_keep_the_best_first_prefix() -> None:
    output = _relation_output(3, rows=1, pairs=60, types=6)
    kwargs = {"threshold": 0.1, "texts": [[f"w{i}" for i in range(30)]]}
    with span_decoding.document_allowances([10**9]) as rows:
        full = _flat(_RELATIONS.decode(output, **kwargs))
        spent = 10**9 - rows[0].remaining
    kept_units = relation_decoding.RELATION_UNITS * len(full)
    # Just enough for everything but the last few relations.
    with span_decoding.document_allowances([spent - kept_units + relation_decoding.RELATION_UNITS * 4]) as rows:
        capped = _flat(_RELATIONS.decode(output, **kwargs))
    head = {id(item.triple) for item in _head(_scored(full), 4)}
    assert capped == [triple for triple in full if id(triple) in head]
    assert rows[0].remaining == 0


def test_relations_cost_the_same_however_far_the_batch_pads_the_pairs() -> None:
    output = _relation_output(7, rows=1, pairs=60, types=6)
    padded = _relation_output(7, rows=1, pairs=60, types=6)
    padded.joint_rel_logits = torch.cat([output.joint_rel_logits, torch.full((1, 500, 6), 3.0)], 1)
    padded.joint_rel_idx = torch.cat([output.joint_rel_idx, torch.zeros(1, 500, 2, dtype=torch.long)], 1)
    padded.joint_rel_mask = torch.cat([output.joint_rel_mask, torch.zeros(1, 500, dtype=torch.bool)], 1)
    kwargs = {"threshold": 0.1, "texts": [[f"w{i}" for i in range(30)]]}
    kept, spent = [], []
    for relation_output in (output, padded):
        with span_decoding.document_allowances([10**9]) as rows:
            kept.append(_RELATIONS.decode(relation_output, **kwargs))
        spent.append(10**9 - rows[0].remaining)
    assert _flat(kept[0])
    assert kept[0] == kept[1]
    assert spent[0] == spent[1]


def test_relation_cells_that_all_fail_decode_quickly() -> None:
    output = _relation_output(4, rows=1, pairs=9900, types=20)
    output.joint_rel_logits = torch.full((1, 9900, 20), -9.0)
    start = time.perf_counter()
    assert _flat(_RELATIONS.decode(output, threshold=0.5, texts=[[f"w{i}" for i in range(30)]])) == []
    assert time.perf_counter() - start < 5
