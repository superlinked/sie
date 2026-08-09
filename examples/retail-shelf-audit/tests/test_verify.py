import pytest

import retail_shelf_audit.verify as verify_module
from retail_shelf_audit.verify import (
    EXPECTED_DETECTIONS,
    EXPECTED_OCR_FRAGMENTS,
    recorded_evidence,
    verify_cloud_run,
)


def test_recorded_evidence_matches_verified_case() -> None:
    evidence = recorded_evidence()
    for key, expected in EXPECTED_DETECTIONS.items():
        assert evidence[key] == expected
    assert evidence["ocr_fragments"] == EXPECTED_OCR_FRAGMENTS


def test_verified_cloud_run_rejects_a_different_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = verify_module._load

    def changed_load(path):
        value = load(path)
        if path.name == "manifest.json":
            value["source_input"]["path"] = "assets/different.jpg"
        return value

    monkeypatch.setattr(verify_module, "_load", changed_load)

    with pytest.raises(ValueError, match="source image path differs"):
        verify_cloud_run()


def test_verified_cloud_run_rejects_reused_request_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = verify_module._load

    def changed_load(path):
        value = load(path)
        if path.name == "lighton-ocr-candidate-2.json":
            value["request"]["id"] = "019fe6f0-2381-7ec0-bcf9-87324524058a"
        return value

    monkeypatch.setattr(verify_module, "_load", changed_load)

    with pytest.raises(ValueError, match="reuse a request ID"):
        verify_cloud_run()


def test_verified_cloud_run_rejects_a_non_string_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = verify_module._load

    def changed_load(path):
        value = load(path)
        if path.name == "grounding-dino.json":
            value["request"]["id"] = 42
        return value

    monkeypatch.setattr(verify_module, "_load", changed_load)

    with pytest.raises(ValueError, match="has no request ID"):
        verify_cloud_run()
