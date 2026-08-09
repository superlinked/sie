from __future__ import annotations

import json
from pathlib import Path

import pytest

import contract_review_agent.evidence as evidence_module
from contract_review_agent.app import ContractReview, RiskFlag
from contract_review_agent.evidence import write_run_record
from contract_review_agent.runtime import Ledger

ROOT = Path(__file__).resolve().parents[1]


def _review() -> ContractReview:
    return ContractReview(
        document_type="agreement",
        parties=["Buyer", "Seller"],
        effective_date="unknown",
        renewal_terms="annual",
        governing_law="unknown",
        executed=False,
        key_obligations=["Give notice"],
        risk_flags=[
            RiskFlag(
                clause="Renewal",
                issue="No early exit",
                severity="high",
                suggested_redline="Add a termination right.",
            )
        ],
        recommendation="Add a termination right.",
    )


def _write_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    run_id: str,
    api_calls: list[dict[str, object]],
) -> Path:
    monkeypatch.setattr(evidence_module, "PROJECT_ROOT", tmp_path)
    scan = tmp_path / "scan.png"
    database = tmp_path / "obligations.db"
    scan.write_bytes(b"scan")
    database.write_bytes(b"database")
    return write_run_record(
        run_id=run_id,
        endpoint="https://api.superlinked.com",
        cfg={"models": {"review": "model-a"}},
        label="example",
        contract_text="contract",
        scan_path=str(scan),
        db_path=str(database),
        findings="findings",
        review=_review(),
        ledger=Ledger(),
        api_calls=api_calls,
        wall_s=1,
    )


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "nested\\run", ".."])
def test_write_run_record_rejects_unsafe_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="one safe directory name"):
        _write_record(tmp_path, monkeypatch, run_id=run_id, api_calls=[])

    assert not (tmp_path / "runs").exists()


def test_write_run_record_rejects_missing_request_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = [
        {
            "function": function,
            "requested_model": "model-a",
            "request_id": "request-1",
            "rate_book_version": "rate-book-1",
            "execution_identity_sha256": None if function == "score" else "a" * 64,
        }
        for function in ("encode", "extract", "generate", "score")
    ]

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    evaluation = json.loads(
        (tmp_path / "runs" / "safe-run" / "evaluation.json").read_text()
    )
    assert evaluation["checks"]["api_calls_have_request_provenance"] is False
    assert evaluation["passed"] is False


def test_verified_run_manifest_pins_complete_passing_evidence() -> None:
    run_dir = ROOT / "verified-run"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["artifacts"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)) == 5
    artifacts = {entry["path"]: entry["sha256"] for entry in entries}

    assert set(artifacts) == {
        "api-calls.json",
        "evaluation.json",
        "investigator-findings.txt",
        "ledger.json",
        "review.json",
    }
    for relative_path, expected_hash in artifacts.items():
        assert evidence_module._sha256(run_dir / relative_path) == expected_hash

    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["passed"] is True
    assert evaluation["checks"]["api_calls_have_request_provenance"] is True
