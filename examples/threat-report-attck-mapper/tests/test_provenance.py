from __future__ import annotations

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
