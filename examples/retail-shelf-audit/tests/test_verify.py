from retail_shelf_audit.verify import EXPECTED_DETECTIONS, EXPECTED_OCR_FRAGMENTS, recorded_evidence


def test_recorded_evidence_matches_verified_case() -> None:
    evidence = recorded_evidence()
    for key, expected in EXPECTED_DETECTIONS.items():
        assert evidence[key] == expected
    assert evidence["ocr_fragments"] == EXPECTED_OCR_FRAGMENTS
