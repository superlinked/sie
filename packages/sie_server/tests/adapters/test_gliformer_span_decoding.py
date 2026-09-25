"""Bounded GLiFormer span decoding: same spans as the package, at bounded cost (no model weights)."""

from __future__ import annotations

import importlib
import random
from dataclasses import astuple
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gliner.modeling.utils import extract_spans_from_tokens
from sie_server.adapters.gliformer import adapter as adapter_module
from sie_server.adapters.gliformer import structuring_decoding
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


def _cut(spans: list[Any], limit: int) -> float | None:
    """Score of the lowest kept candidate: at most ``limit`` kept, ties kept together."""
    scores = sorted((span.score for span in spans), reverse=True)
    kept = min(limit, len(scores))
    while kept and kept < len(scores) and scores[kept - 1] == scores[kept]:
        kept -= 1
    return scores[kept - 1] if kept else None


@pytest.mark.parametrize("limit", [1, 7, 50, 200])
@pytest.mark.parametrize("seed", range(4))
def test_capped_pairing_keeps_the_best_candidates_in_order(seed: int, limit: int) -> None:
    logits = _bio_scores(seed, length=120, labels=3, dtype=torch.float32)
    names = {0: "a", 1: "b", 2: "c"}
    full = _upstream(logits, names, 0.1)
    assert len(full) > limit
    capped = _bounded(logits, names, 0.1, max_candidates=limit)
    cut = _cut(full, limit)
    expected = [span for span in full if cut is not None and span.score >= cut]
    assert _as_tuples(capped) == _as_tuples(expected)
    assert len(capped) <= limit
    # Overlap removal takes spans best first, so the cut only drops spans
    # that rank below it.
    for flat_ner in (True, False):
        for multi_label in (True, False):
            kept = select_spans(capped, flat_ner, multi_label)
            everything = _UPSTREAM_GREEDY(_DECODER, full, flat_ner, multi_label)
            assert _as_tuples(kept) == _as_tuples([s for s in everything if cut is not None and s.score >= cut])


def test_capped_pairing_drops_candidates_tied_at_the_cut_together() -> None:
    # Three identical one-word entities: all tie, so a cap of 2 keeps none.
    logits = torch.full((5, 1, 3), -8.0)
    for position in (0, 2, 4):
        logits[position, 0] = 6.0
    names = {0: "x"}
    assert len(_bounded(logits, names, 0.5)) == 3
    assert _bounded(logits, names, 0.5, max_candidates=2) == []
    assert len(_bounded(logits, names, 0.5, max_candidates=3)) == 3


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
    cut = _cut(ranked, limit)
    kept = _proposals(*_propose(scores, 0.1, max_candidates=limit))[0]
    assert len(kept) <= limit
    assert kept == [(item.start, item.end) for item in ranked if cut is not None and item.score >= cut]


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


@pytest.mark.parametrize("limit", [1, 5, 30])
def test_capped_slots_keep_their_best_spans(limit: int) -> None:
    output = _structuring_output(3, groups=1, slots=4, entities=40, fields=5)
    args = (
        torch.sigmoid(output.structuring_logits[0]),
        torch.sigmoid(output.structuring_field_logits[0]),
        output.structuring_span_idx[0],
        torch.where(output.structuring_span_mask[0])[0],
        None,
        0.1,
        {i: f"f{i}" for i in range(5)},
    )
    full = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_candidates=10**9)
    capped = structuring_decoding._slot_spans(*args, make_span=upstream.Span, max_candidates=limit)
    for (slot, everything), (capped_slot, kept) in zip(full, capped, strict=True):
        assert slot == capped_slot
        assert len(everything) > limit
        cut = _cut(everything, limit)
        assert _as_tuples(kept) == _as_tuples([s for s in everything if cut is not None and s.score >= cut])


@pytest.mark.parametrize(
    ("scores", "limit", "kept"),
    [
        ([0.9, 0.8, 0.8, 0.8, 0.5], 1, 1),
        ([0.9, 0.8, 0.8, 0.8, 0.5], 2, 1),
        ([0.9, 0.8, 0.8, 0.8, 0.5], 4, 4),
        ([0.9, 0.8, 0.8, 0.8, 0.5], 5, 5),
        ([0.5, 0.5, 0.5], 2, 0),
    ],
)
def test_slot_cut_keeps_ties_together(scores: list[float], limit: int, kept: int) -> None:
    values = np.asarray(scores, dtype=np.float32)
    cut = structuring_decoding._tie_safe_cut(values, limit) if values.size > limit else values.min()
    assert int((values >= cut).sum()) == kept


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
    model = SimpleNamespace(heads={"ner": ner_head})
    adapter_module._mask_padded_words(model)
    assert list(ner_head._forward_hooks.values()) == [adapter_module._mask_padded_ner_logits]
    with pytest.raises(RuntimeError, match="no NER head"):
        adapter_module._mask_padded_words(SimpleNamespace(heads={}))
