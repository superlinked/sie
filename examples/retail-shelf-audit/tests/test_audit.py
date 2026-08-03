from __future__ import annotations

import pytest

from retail_shelf_audit.audit import (
    build_evidence,
    candidate_crop_box,
    nearby_price_candidates,
    ocr_fragments,
    select_gap,
    select_vertical_pair,
)
from retail_shelf_audit.verify import EXPECTED_OCR_FRAGMENTS


def _objects() -> list[dict]:
    return [
        {"label": "empty shelf space", "score": 0.252384, "bbox": [8.0, 2552.5, 4019.2, 210.1]},
        {"label": "empty shelf space", "score": 0.274157, "bbox": [2043.8, 2137.0, 623.6, 402.4]},
        {"label": "price tag", "score": 0.259638, "bbox": [2235.5, 2559.4, 399.2, 186.3]},
        {"label": "price tag", "score": 0.23793, "bbox": [2237.3, 2561.7, 334.9, 181.2]},
        {"label": "price tag", "score": 0.231093, "bbox": [2300.1, 2318.7, 335.8, 244.5]},
        {"label": "price tag", "score": 0.9, "bbox": [20, 20, 100, 100]},
    ]


def _gap() -> dict:
    return _objects()[1]


def _price() -> dict:
    return _objects()[2]


def test_select_gap_rejects_full_width_strip() -> None:
    selected = select_gap(_objects(), (4032, 3024))
    assert selected["score"] == 0.274157


def test_select_gap_fails_when_only_full_width_strips_exist() -> None:
    with pytest.raises(ValueError, match="No non-strip empty shelf detection"):
        select_gap([_objects()[0]], (4032, 3024))


def test_nearby_candidates_are_geometry_selected_and_deduplicated() -> None:
    selected = nearby_price_candidates(_objects(), _gap())
    assert [item["score"] for item in selected] == [0.259638, 0.231093]


def test_nearby_candidates_require_two_distinct_boxes() -> None:
    with pytest.raises(ValueError, match="Fewer than two distinct nearby price-tag candidates"):
        nearby_price_candidates([_price()], _gap())


def test_nearby_candidates_ignore_zero_width_boxes() -> None:
    objects = [_price(), {"label": "price tag", "score": 0.8, "bbox": [2300, 2300, 0, 200]}]
    with pytest.raises(ValueError, match="Fewer than two distinct nearby price-tag candidates"):
        nearby_price_candidates(objects, _gap())


def test_candidate_crop_is_derived_from_detection() -> None:
    assert candidate_crop_box(_price()["bbox"], (4032, 3024)) == [2156, 2513, 2715, 2792]


def test_vertical_roles_are_assigned_by_geometry() -> None:
    upper, lower = select_vertical_pair(nearby_price_candidates(_objects(), _gap()))
    assert upper["score"] == 0.231093
    assert lower["score"] == 0.259638


def test_vertical_pair_requires_aligned_candidates() -> None:
    candidates = [
        {"label": "price tag", "score": 0.4, "bbox": [100, 100, 100, 100]},
        {"label": "price tag", "score": 0.3, "bbox": [500, 300, 100, 100]},
    ]
    with pytest.raises(ValueError, match="No vertically aligned DINO candidate pair"):
        select_vertical_pair(candidates)


def test_ocr_fragments_use_position_and_line_order() -> None:
    upper = "I am temporarily\nout-of-stock\nfrom our supplier\nPlease ask our friendly staff"
    lower = "Please ask our friendly staff\nPanadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹"
    assert ocr_fragments(upper, lower) == EXPECTED_OCR_FRAGMENTS


def test_ocr_fragments_fail_on_incomplete_layout() -> None:
    with pytest.raises(ValueError, match="three upper lines and four distinct lower lines"):
        ocr_fragments("one\ntwo", "three\nfour")


def test_build_evidence_preserves_model_outputs() -> None:
    evidence = build_evidence(
        _gap(),
        _price(),
        "I am temporarily\nout-of-stock\nfrom our supplier",
        "Panadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹",
    )
    assert evidence == {
        "gap_detection": {
            "label": "empty shelf space",
            "score": 0.274157,
            "bbox_xywh": [2043.8, 2137.0, 623.6, 402.4],
        },
        "price_detection": {
            "label": "price tag",
            "score": 0.259638,
            "bbox_xywh": [2235.5, 2559.4, 399.2, 186.3],
        },
        "ocr_fragments": EXPECTED_OCR_FRAGMENTS,
    }
