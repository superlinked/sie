from retail_shelf_audit.verify import recorded_evidence


def test_recorded_evidence_matches_verified_case() -> None:
    evidence = recorded_evidence()
    assert evidence["gap_detection"] == {
        "label": "empty shelf space",
        "score": 0.274157,
        "bbox_xywh": [2043.8, 2137.0, 623.6, 402.4],
    }
    assert evidence["price_detection"] == {
        "label": "price tag",
        "score": 0.259638,
        "bbox_xywh": [2235.5, 2559.4, 399.2, 186.3],
    }
