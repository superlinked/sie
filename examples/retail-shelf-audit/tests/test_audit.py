from __future__ import annotations

from pathlib import Path

import pytest

import retail_shelf_audit.audit as audit_module
from retail_shelf_audit.audit import (
    build_evidence,
    candidate_crop_box,
    evaluation_checks,
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


def test_evaluation_checks_are_derived_from_selected_evidence() -> None:
    upper, lower = select_vertical_pair(nearby_price_candidates(_objects(), _gap()))
    checks = evaluation_checks(
        _objects(),
        _gap(),
        upper,
        lower,
        "I am temporarily\nout-of-stock\nfrom our supplier",
        "Panadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹",
        (4032, 3024),
    )
    assert checks == {
        "non_strip_gap_selected": True,
        "nearby_vertical_price_pair_selected": True,
        "minimum_ocr_fragments_recovered": True,
    }


def test_evaluation_checks_fail_on_incomplete_ocr() -> None:
    upper, lower = select_vertical_pair(nearby_price_candidates(_objects(), _gap()))
    checks = evaluation_checks(
        _objects(),
        _gap(),
        upper,
        lower,
        "one\ntwo",
        "three\nfour",
        (4032, 3024),
    )
    assert checks == {
        "non_strip_gap_selected": True,
        "nearby_vertical_price_pair_selected": True,
        "minimum_ocr_fragments_recovered": False,
    }


def test_evaluation_checks_reject_a_full_width_strip_gap() -> None:
    upper, lower = select_vertical_pair(nearby_price_candidates(_objects(), _gap()))
    checks = evaluation_checks(
        _objects(),
        _objects()[0],
        upper,
        lower,
        "I am temporarily\nout-of-stock\nfrom our supplier",
        "Panadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹",
        (4032, 3024),
    )

    assert checks["non_strip_gap_selected"] is False


def test_evaluation_checks_reject_a_non_nearby_price_pair() -> None:
    checks = evaluation_checks(
        _objects(),
        _gap(),
        _objects()[5],
        _price(),
        "I am temporarily\nout-of-stock\nfrom our supplier",
        "Panadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹",
        (4032, 3024),
    )

    assert checks["nearby_vertical_price_pair_selected"] is False


def test_evaluation_checks_reject_a_valid_but_lower_ranked_pair() -> None:
    gap = {
        "label": "empty shelf space",
        "score": 0.9,
        "bbox": [0, 0, 1000, 1000],
    }
    best_upper = {"label": "price tag", "score": 0.9, "bbox": [100, 100, 100, 100]}
    best_lower = {"label": "price tag", "score": 0.8, "bbox": [100, 210, 100, 100]}
    other_upper = {"label": "price tag", "score": 0.4, "bbox": [400, 100, 100, 100]}
    other_lower = {"label": "price tag", "score": 0.3, "bbox": [400, 210, 100, 100]}
    objects = [gap, best_upper, best_lower, other_upper, other_lower]

    checks = evaluation_checks(
        objects,
        gap,
        other_upper,
        other_lower,
        "I am temporarily\nout-of-stock\nfrom our supplier",
        "Panadol Child\n5-12Yrs Elixir 100ml\n101760\n10⁹⁹",
        (2000, 2000),
    )

    assert checks["nearby_vertical_price_pair_selected"] is False


def test_run_audit_cleans_failed_staging_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit_module, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(audit_module, "load_config", lambda: object())

    def fail(run_dir: Path, _run_id: str, _config: object) -> None:
        (run_dir / "partial.json").write_text("partial", encoding="utf-8")
        raise RuntimeError("evidence failed")

    monkeypatch.setattr(audit_module, "_write_audit", fail)
    with pytest.raises(RuntimeError, match="evidence failed"):
        audit_module.run_audit("retryable")

    assert list(tmp_path.iterdir()) == []

    def succeed(run_dir: Path, _run_id: str, _config: object) -> None:
        (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(audit_module, "_write_audit", succeed)
    final_run_dir = audit_module.run_audit("retryable")

    assert final_run_dir == tmp_path / "retryable"
    assert (final_run_dir / "manifest.json").is_file()
