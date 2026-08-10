from __future__ import annotations

import json
from pathlib import Path

import pytest

import contract_review_agent.evidence as evidence_module
from contract_review_agent.app import ContractReview, RiskFlag
from contract_review_agent.evidence import write_run_record
from contract_review_agent.runtime import Ledger

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "triage": "model-triage",
    "orchestrator": "model-orchestrator",
    "vision": "model-vision",
    "reasoning": "model-reasoning",
    "sql": "model-sql",
    "guard": "model-guard",
    "ocr": "model-ocr",
    "embed": "model-embed",
    "rerank": "model-rerank",
    "entities": "model-entities",
}
CFG = {"models": MODELS}


def _api_calls() -> list[dict[str, object]]:
    return [
        {
            **expected,
            "runtime_model": expected["requested_model"],
            "request_id": f"request-{index}",
            "rate_book_version": "rate-book-1",
            "credits_debited": 1,
            "execution_identity_sha256": "a" * 64,
        }
        for index, expected in enumerate(
            evidence_module._expected_call_sequence(CFG), start=1
        )
    ]


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
                clause="Section 1.1 (Renewal)",
                issue="No early exit",
                severity="high",
                suggested_redline="Add a termination right.",
            )
        ],
        recommendation="Add a termination right.",
    )


def _ledger(verdict: str = "no") -> Ledger:
    ledger = Ledger()
    ledger.record(
        "Safety guardrail (granite-guardian)",
        MODELS["guard"],
        "generate",
        got=verdict,
    )
    return ledger


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
        cfg=CFG,
        label="example",
        contract_text="1.1 Renewal. Annual renewal terms.",
        scan_path=str(scan),
        db_path=str(database),
        findings="findings",
        review=_review(),
        ledger=_ledger(),
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


def test_run_destination_preflight_rejects_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_module, "PROJECT_ROOT", tmp_path)
    destination = tmp_path / "runs" / "existing"
    destination.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="Run evidence already exists"):
        evidence_module.ensure_run_destination_available("existing")


def test_write_run_record_rejects_missing_request_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["execution_identity_sha256"] = None

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    runs_dir = tmp_path / "runs"
    assert not (runs_dir / "safe-run").exists()
    assert list(runs_dir.iterdir()) == []

    run_dir = _write_record(
        tmp_path,
        monkeypatch,
        run_id="safe-run",
        api_calls=_api_calls(),
    )
    assert run_dir == runs_dir / "safe-run"
    assert (run_dir / "manifest.json").is_file()


def test_write_run_record_rejects_missing_runtime_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["runtime_model"] = None

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    assert list((tmp_path / "runs").iterdir()) == []


def test_write_run_record_allows_unreported_encode_and_extract_runtime_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    for call in api_calls:
        if call["function"] in evidence_module.RUNTIME_MODEL_OPTIONAL_FUNCTIONS:
            call["runtime_model"] = None

    run_dir = _write_record(
        tmp_path,
        monkeypatch,
        run_id="safe-run",
        api_calls=api_calls,
    )

    assert (run_dir / "manifest.json").is_file()


def test_verified_api_calls_have_supported_runtime_model_provenance() -> None:
    api_calls = json.loads(
        (ROOT / "verified-run" / "api-calls.json").read_text(encoding="utf-8")
    )

    assert any(call["runtime_model"] is None for call in api_calls)
    assert all(evidence_module._runtime_model_is_valid(call) for call in api_calls)


def test_write_run_record_rejects_duplicate_request_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["request_id"] = api_calls[-2]["request_id"]

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)


def test_write_run_record_rejects_an_unsupported_risk_clause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    monkeypatch.setattr(evidence_module, "PROJECT_ROOT", tmp_path)
    scan = tmp_path / "scan.png"
    database = tmp_path / "obligations.db"
    scan.write_bytes(b"scan")
    database.write_bytes(b"database")
    review = _review()
    review.risk_flags[0].clause = "Section 2.1 (Missing)"

    with pytest.raises(RuntimeError, match="matched 0 source sections"):
        write_run_record(
            run_id="safe-run",
            endpoint="https://api.superlinked.com",
            cfg=CFG,
            label="example",
            contract_text="1.1 Renewal. Annual renewal terms.",
            scan_path=str(scan),
            db_path=str(database),
            findings="findings",
            review=review,
            ledger=_ledger(),
            api_calls=api_calls,
            wall_s=1,
        )


def test_write_run_record_rejects_section_reference_only_in_risk_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    review.risk_flags[0].clause = "Renewal clause"
    review.risk_flags[0].issue = "Section 1.1 has no early exit"
    monkeypatch.setattr(evidence_module, "PROJECT_ROOT", tmp_path)
    scan = tmp_path / "scan.png"
    database = tmp_path / "obligations.db"
    scan.write_bytes(b"scan")
    database.write_bytes(b"database")

    with pytest.raises(RuntimeError, match="no source section reference"):
        write_run_record(
            run_id="safe-run",
            endpoint="https://api.superlinked.com",
            cfg=CFG,
            label="example",
            contract_text="1.1 Renewal. Annual renewal terms.",
            scan_path=str(scan),
            db_path=str(database),
            findings="findings",
            review=review,
            ledger=_ledger(),
            api_calls=_api_calls(),
            wall_s=1,
        )


@pytest.mark.parametrize("credits_debited", [None, False, "1", -1])
def test_write_run_record_rejects_invalid_credit_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    credits_debited: object,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["credits_debited"] = credits_debited

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    assert list((tmp_path / "runs").iterdir()) == []


def test_write_run_record_rejects_a_missing_required_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = [
        call for call in _api_calls() if call["stage"] != "search_clauses:index"
    ]

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    runs_dir = tmp_path / "runs"
    assert not (runs_dir / "safe-run").exists()
    assert list(runs_dir.iterdir()) == []


def test_write_run_record_rejects_a_malformed_guardrail_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_module, "PROJECT_ROOT", tmp_path)
    scan = tmp_path / "scan.png"
    database = tmp_path / "obligations.db"
    scan.write_bytes(b"scan")
    database.write_bytes(b"database")

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        write_run_record(
            run_id="safe-run",
            endpoint="https://api.superlinked.com",
            cfg=CFG,
            label="example",
            contract_text="1.1 Renewal. Annual renewal terms.",
            scan_path=str(scan),
            db_path=str(database),
            findings="findings",
            review=_review(),
            ledger=_ledger("No_of_turn>"),
            api_calls=_api_calls(),
            wall_s=1,
        )

    runs_dir = tmp_path / "runs"
    assert not (runs_dir / "safe-run").exists()
    assert list(runs_dir.iterdir()) == []


def test_model_text_normalization_is_formatting_only() -> None:
    assert evidence_module._normalize_model_text("first  \r\nsecond\t\r\n") == (
        "first\nsecond\n"
    )


def test_verified_run_manifest_pins_complete_passing_evidence() -> None:
    run_dir = ROOT / "verified-run"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["artifacts"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths)) == 6
    artifacts = {entry["path"]: entry["sha256"] for entry in entries}

    assert set(artifacts) == {
        "api-calls.json",
        "evaluation.json",
        "investigator-findings.txt",
        "ledger.json",
        "review.json",
        "source-evidence.json",
    }
    for relative_path, expected_hash in artifacts.items():
        assert evidence_module._sha256(run_dir / relative_path) == expected_hash

    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["passed"] is True
    assert evaluation["checks"]["api_calls_have_request_provenance"] is True
    assert evaluation["checks"]["api_call_request_ids_unique"] is True
    assert evaluation["checks"]["risk_clauses_supported_by_source"] is True
    assert evaluation["checks"]["signature_image_scope_not_overclaimed"] is True
    source_evidence = json.loads(
        (run_dir / "source-evidence.json").read_text(encoding="utf-8")
    )
    assert (
        source_evidence["contract_text_sha256"]
        == manifest["source_inputs"][0]["sha256"]
    )
    sections = {row["section"] for row in source_evidence["risk_clauses"]}
    assert {"1.3", "5.3"} <= sections
    assert evaluation["checks"]["required_api_call_sequence"] is True
    assert evaluation["observed_call_sequence"] == evaluation["expected_call_sequence"]
