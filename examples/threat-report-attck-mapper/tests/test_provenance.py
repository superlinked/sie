from __future__ import annotations

import json

import pytest

from threat_mapper import runner
from threat_mapper.runner import _rate_book_provenance


def test_rate_book_provenance_requires_one_version_and_execution_identity() -> None:
    calls = [
        {
            "request_id": "request-1",
            "credits_debited": 10,
            "rate_book_version": "rates-v1",
            "execution_identity_sha256": "identity-a",
        },
        {
            "request_id": "request-2",
            "credits_debited": 3,
            "rate_book_version": "rates-v1",
            "execution_identity_sha256": "identity-b",
        },
    ]

    result = _rate_book_provenance(calls)

    assert result["version"] == "rates-v1"
    assert result["request_ids"] == ["request-1", "request-2"]
    assert result["execution_identity_sha256"] == ["identity-a", "identity-b"]


def test_rate_book_provenance_rejects_charged_request_without_identity() -> None:
    with pytest.raises(RuntimeError, match="execution identity"):
        _rate_book_provenance(
            [
                {
                    "request_id": "request-1",
                    "credits_debited": 1,
                    "rate_book_version": "rates-v1",
                }
            ]
        )


def test_begin_run_reserves_an_id_against_concurrent_writers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)

    final_dir, staging, reservation = runner._begin_run("one")

    assert final_dir == tmp_path / "one"
    assert staging.is_dir()
    assert reservation.is_dir()
    with pytest.raises(FileExistsError, match="reserved"):
        runner._begin_run("one")


def test_publish_failed_run_keeps_persisted_artifacts(tmp_path) -> None:
    staging = tmp_path / ".run-staging"
    final_dir = tmp_path / "run"
    staging.mkdir()
    (staging / "predictions.jsonl").write_text('{"case_id":"one"}\n', encoding="utf-8")
    (staging / "api-calls.json").write_text("[]\n", encoding="utf-8")

    runner._publish_failed_run(final_dir, staging, RuntimeError("provenance failed"))

    assert not staging.exists()
    assert (final_dir / "predictions.jsonl").is_file()
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "post_processing_failed"
    assert manifest["error"] == {"type": "RuntimeError", "message": "provenance failed"}
    assert [row["path"] for row in manifest["artifacts"]] == ["api-calls.json", "predictions.jsonl"]
