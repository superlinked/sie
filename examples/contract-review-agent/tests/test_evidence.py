from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import contract_review_agent.evidence as evidence_module
from contract_review_agent.app import (
    ContractReview,
    PublishedReviewRepair,
    RiskFlag,
    _align_published_signature_recommendation,
    _published_findings_narrative_is_bounded,
    _render_grounded_published_findings,
    _unsupported_published_sections,
)
from contract_review_agent.data.fetch_contracts import _signature_page_text
from contract_review_agent.evidence import write_run_record
from contract_review_agent.runtime import Ledger
from contract_review_agent.tools import ClauseRiskAnalysis

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


def test_published_section_allowlist_rejects_unrelated_citations() -> None:
    assert (
        _unsupported_published_sections(
            "Financial terms (Section 1.6); governing law (Section 6.9)."
        )
        == set()
    )
    assert _unsupported_published_sections("Financial terms are in Section 6.7.") == {
        "6.7"
    }
    assert _unsupported_published_sections("Risks are in Sections 6.7 and 1.3.") == {
        "6.7"
    }
    assert _unsupported_published_sections("Risks are in Sections 1.3 and 6.7.") == {
        "6.7"
    }


def test_optional_findings_repairs_precede_synthesis_in_provenance() -> None:
    stages = [
        call["stage"]
        for call in evidence_module._expected_call_sequence(
            CFG, include_citation_repair=True, include_synthesis_repair=True
        )
    ]
    assert stages[-4:] == [
        "investigator_report",
        "investigator_report:citation_repair",
        "synthesize_review",
        "synthesize_review:repair",
    ]


def test_published_findings_narrative_must_be_complete_and_bounded() -> None:
    complete = "A" * 1_799 + "."
    assert _published_findings_narrative_is_bounded(complete)
    assert _published_findings_narrative_is_bounded("A" * 1_797 + ".**")
    assert not _published_findings_narrative_is_bounded("A" * 1_799)
    assert _published_findings_narrative_is_bounded("A" * 2_999 + ".")
    assert not _published_findings_narrative_is_bounded("A" * 3_000 + ".")


def test_grounded_published_findings_are_complete_and_bounded() -> None:
    risks = json.loads(
        (ROOT / "verified-run" / "review.json").read_text(encoding="utf-8")
    )["risk_flags"]
    findings = _render_grounded_published_findings(
        ClauseRiskAnalysis.model_validate({"risks": risks}),
        as_of_date=date(2026, 8, 10),
    )

    assert _published_findings_narrative_is_bounded(findings)
    assert "June 30, 2026 quarterly compliance attestation is overdue" in findings
    assert "July 1, 2026 annual subscription or license fee is overdue" in findings
    assert "September 15, 2026 renewal or non-renewal notice is upcoming" in findings
    assert evidence_module._published_investigator_findings_are_complete(
        evidence_module.PUBLISHED_CONTRACT_LABEL, findings
    )


def test_published_signature_recommendation_replaces_execution_overclaim() -> None:
    recommendation = _align_published_signature_recommendation(
        "The agreement is not fully executed. Negotiate the renewal notice window."
    )

    assert "not fully executed" not in recommendation
    assert "not established from the visible signature page" in recommendation
    assert "Negotiate the renewal notice window." in recommendation


def test_cuad_scan_renders_signature_block_or_document_tail() -> None:
    text = "title\nbody\nIn witness whereof\nBy: /s/ Example"
    assert _signature_page_text(text) == "In witness whereof\nBy: /s/ Example"
    assert _signature_page_text("0123456789") == "0123456789"
    tail = _signature_page_text(
        "\n".join([*(f"contract line {index}" for index in range(60)), "By: /s/ Tail"])
    )
    assert len(tail.splitlines()) == 46
    assert tail.endswith("By: /s/ Tail")
    unicode_prefix = "Straße before the marker\nSIGNATURE PAGE\nBy: /s/ Example"
    assert _signature_page_text(unicode_prefix) == ("SIGNATURE PAGE\nBy: /s/ Example")


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


def _published_repair() -> PublishedReviewRepair:
    return PublishedReviewRepair(
        document_type="agreement",
        parties=["Buyer", "Seller"],
        effective_date="unknown",
        governing_law="Illinois",
        executed=False,
        illinois_exclusive_distributorship=True,
        initial_term_years=10,
        initial_term_starts_on_last_sample_delivery=True,
        renewal_period_years=1,
        renewal_max_additional_years=10,
        renewal_requires_distributor_compliance=True,
        letter_of_credit_amount_usd=500_000,
        letter_of_credit_is_irrevocable=True,
        monthly_purchase_order_amount_usd=250_000,
        first_product_year_unit_minimum=375,
        quarterly_reports_during_first_year=True,
    )


def test_published_repair_assembles_every_named_obligation() -> None:
    repair = _published_repair()
    risk_flags = [
        RiskFlag(
            clause="Section 1.3",
            issue="Missing notice mechanics",
            severity="high",
            suggested_redline="Add explicit notice mechanics.",
        )
    ]

    review = repair.to_contract_review(
        risk_flags=risk_flags,
        recommendation="review",
    )
    assert review.document_type == "Distributor Agreement"
    assert review.parties == [
        "Electric City Corp. (Company)",
        "Electric City of Illinois L.L.C. (Distributor)",
    ]
    assert review.renewal_terms == (
        "Conditional annual renewal for 1-year terms up to 10 additional years if "
        "Distributor complies with all terms of the Agreement (Section 1.3)"
    )
    assert review.key_obligations == [
        "Exclusive distributorship within the Illinois Market (Section 1.1)",
        "Ten-year initial term beginning on delivery of the last Sample (Section 1.3)",
        review.renewal_terms,
        (
            "Distributor must issue an irrevocable $500,000 letter of credit to "
            "Company (Section 1.6)"
        ),
        (
            "Company must receive a $250,000 purchase order from Distributor by the "
            "first day of each month (Section 1.6)"
        ),
        (
            "Distributor must purchase at least 375 units during the first Product "
            "Year (Section 1.6)"
        ),
        (
            "Distributor must submit written reports each quarter during the first "
            "year of the Term (Section 4.1)"
        ),
    ]
    assert review.risk_flags == risk_flags
    assert "risk_flags" not in PublishedReviewRepair.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("renewal_period_years", 2, "renewal period"),
        ("renewal_max_additional_years", 9, "maximum renewal duration"),
    ],
)
def test_published_repair_rejects_changed_renewal_durations(
    field_name: str,
    value: int,
    match: str,
) -> None:
    repair = _published_repair().model_copy(update={field_name: value})

    with pytest.raises(ValueError, match=match):
        repair.to_contract_review(risk_flags=[], recommendation="review")


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
    label: str = "example",
    findings: str = "Execution is not established from the visible signature page.",
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
        label=label,
        contract_text="1.1 Renewal. Annual renewal terms.",
        scan_path=str(scan),
        db_path=str(database),
        findings=findings,
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


def test_write_run_record_rejects_incomplete_published_findings_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        RuntimeError, match="Published investigator findings are incomplete"
    ):
        _write_record(
            tmp_path,
            monkeypatch,
            run_id="safe-run",
            api_calls=_api_calls(),
            label=evidence_module.PUBLISHED_CONTRACT_LABEL,
            findings="Incomplete findings.",
        )

    assert not (tmp_path / "runs").exists()


def test_write_run_record_rejects_missing_runtime_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["runtime_model"] = None

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)

    assert list((tmp_path / "runs").iterdir()) == []


def test_write_run_record_rejects_substituted_runtime_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["runtime_model"] = "model-other"

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

    assert all(evidence_module._runtime_model_is_valid(call) for call in api_calls)


def test_write_run_record_rejects_duplicate_request_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_calls = _api_calls()
    api_calls[-1]["request_id"] = api_calls[-2]["request_id"]

    with pytest.raises(RuntimeError, match="Production evidence checks failed"):
        _write_record(tmp_path, monkeypatch, run_id="safe-run", api_calls=api_calls)


def test_published_risk_target_requires_three_high_and_one_medium() -> None:
    review = _review()
    review.risk_flags = [
        RiskFlag(
            clause=f"Section {section}",
            issue=f"Grounded risk in {section}",
            severity=severity,
            suggested_redline=f"Clarify Section {section}.",
        )
        for section, severity in evidence_module.PUBLISHED_RISK_TARGETS.items()
    ]

    assert evidence_module._published_risk_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )

    review.risk_flags[0].severity = "medium"
    assert not evidence_module._published_risk_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )


def test_published_fact_target_preserves_source_grounded_commercial_details() -> None:
    review = _review()
    review.document_type = "Distributor Agreement"
    review.parties = ["Electric City Corp.", "Electric City of Illinois LLC"]
    review.renewal_terms = (
        "If Distributor complies with the Agreement, annual renewal is available for "
        "up to ten additional years (Section 1.3)."
    )
    review.governing_law = "Illinois"
    review.key_obligations = [
        "Distributor must issue a $500,000 letter of credit (Section 1.6).",
        "Company must receive a $250,000 purchase order monthly (Section 1.6).",
        "Distributor has a 375-unit first-year minimum (Section 1.6).",
        "Distributor has exclusive rights in Illinois (Section 1.1).",
        "The ten-year term starts upon delivery of the last Sample (Section 1.3).",
        "Distributor reports quarterly during the first year (Section 4.1).",
    ]

    assert evidence_module._published_fact_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )

    review.renewal_terms = review.renewal_terms.replace("annual", "one (1) year")
    review.renewal_terms = review.renewal_terms.replace("ten", "10")
    assert evidence_module._published_fact_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )

    review.key_obligations[0] = "Distributor must issue a 500k LC (Section 1.6)."
    assert evidence_module._published_fact_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )

    review.key_obligations[0] = review.key_obligations[0].replace("500k", "50k")
    assert not evidence_module._published_fact_coverage_is_preserved(
        evidence_module.PUBLISHED_CONTRACT_LABEL, review
    )


def test_risk_claim_gate_rejects_unsupported_automatic_renewal_variants() -> None:
    review = _review()
    review.risk_flags[0].issue = "The clause implies automatic annual renewals."
    source_evidence = {
        "risk_clauses": [
            {
                "section": "1.1",
                "excerpt": "1.1 Renewal is available if Distributor complies.",
            }
        ]
    }

    assert not evidence_module._risk_claims_are_source_supported(
        review, source_evidence, "example"
    )

    review.risk_flags[
        0
    ].issue = (
        "It is unclear whether the term automatically renews or requires an election."
    )
    assert evidence_module._risk_claims_are_source_supported(
        review, source_evidence, "example"
    )


def test_risk_claim_gate_accepts_distinct_notice_and_cure_periods() -> None:
    review = _review()
    review.risk_flags[0].clause = "Section 4.2"
    review.risk_flags[0].issue = (
        "The termination notice is 30 days, while curable defaults have a separate "
        "commercially reasonable cure period."
    )
    source_evidence = {
        "risk_clauses": [
            {
                "section": "4.2",
                "excerpt": (
                    "4.2 Termination requires 30 days' notice; material defaults may "
                    "be cured within a commercially reasonable time."
                ),
            }
        ]
    }

    assert evidence_module._risk_claims_are_source_supported(
        review, source_evidence, evidence_module.PUBLISHED_CONTRACT_LABEL
    )


def test_risk_claim_gate_requires_correct_repurchase_option_and_exception() -> None:
    review = _review()
    review.risk_flags[0].clause = "Section 4.4"
    source_evidence = {
        "risk_clauses": [
            {
                "section": "4.4",
                "excerpt": (
                    "4.4 Company has the option to repurchase unopened Products; "
                    "provided that if Company terminates without cause, Company shall "
                    "repurchase them."
                ),
            }
        ]
    }
    review.risk_flags[0].issue = (
        "Company has a general repurchase option, with a mandatory repurchase "
        "exception when Company terminates without cause, leaving inventory exposure "
        "after other expirations or terminations."
    )
    assert evidence_module._risk_claims_are_source_supported(
        review, source_evidence, evidence_module.PUBLISHED_CONTRACT_LABEL
    )

    review.risk_flags[0].issue = (
        "Company has the repurchase option only if it terminates without cause, and "
        "the mandatory exception is unclear."
    )
    assert not evidence_module._risk_claims_are_source_supported(
        review, source_evidence, evidence_module.PUBLISHED_CONTRACT_LABEL
    )


def test_risk_claim_gate_does_not_apply_published_terms_to_other_contracts() -> None:
    review = _review()
    review.risk_flags[0].clause = "Section 4.4"
    review.risk_flags[0].issue = "The source-specific repurchase wording is unclear."
    source_evidence = {
        "risk_clauses": [
            {
                "section": "4.4",
                "excerpt": "4.4 A contract-specific repurchase term.",
            }
        ]
    }

    assert evidence_module._risk_claims_are_source_supported(
        review, source_evidence, "other-contract"
    )


def test_signature_scope_rejects_a_review_level_execution_overclaim() -> None:
    review = _review()
    review.recommendation = "The agreement is not executed."

    assert not evidence_module._signature_scope_is_supported(
        "Execution is not established from the visible signature page.", review
    )


def test_published_investigator_findings_require_complete_marketing_evidence() -> None:
    findings = (
        "Distributor Agreement; Section 1.3, Section 1.6, Section 4.2, "
        "Section 4.4, Section 5.3, and "
        "Section 6.9; not at fault; 2026-06-30; 2026-07-01; 2026-09-15; "
        "$500,000 letter of credit; $250,000 monthly purchase order; 375 units; "
        "quarterly compliance attestation; annual subscription fee; renewal notice; "
        "Deadline status as of August 10, 2026: June 30, 2026 quarterly compliance "
        "attestation is overdue; July 1, 2026 annual subscription or license fee is "
        "overdue; September 15, 2026 renewal or non-renewal notice is upcoming. "
        "The visible signature page shows "
        "By: /s/ Joseph Marino and a Jim Stump signatory block, but no execution "
        "dates. Execution is not established from the visible signature page."
    )
    assert evidence_module._published_investigator_findings_are_complete(
        evidence_module.PUBLISHED_CONTRACT_LABEL, findings
    )

    assert not evidence_module._published_investigator_findings_are_complete(
        evidence_module.PUBLISHED_CONTRACT_LABEL,
        findings.replace("Section 5.3", "Indemnification"),
    )
    assert not evidence_module._published_investigator_findings_are_complete(
        evidence_module.PUBLISHED_CONTRACT_LABEL,
        findings.replace("By: /s/ Joseph Marino", "No actual signatures are present"),
    )
    assert not evidence_module._published_investigator_findings_are_complete(
        evidence_module.PUBLISHED_CONTRACT_LABEL,
        findings.replace("attestation is overdue", "attestation is upcoming"),
    )


def test_published_investigator_primary_narrative_must_be_bounded() -> None:
    narrative = "A" * 1_799 + "."
    findings = (
        narrative
        + "\n\nObligations and deadlines (validated exact-contract rows):\n"
        + "2026-06-30 | quarterly compliance"
    )

    assert evidence_module._published_investigator_narrative_is_bounded(
        evidence_module.PUBLISHED_CONTRACT_LABEL, findings
    )
    assert not evidence_module._published_investigator_narrative_is_bounded(
        evidence_module.PUBLISHED_CONTRACT_LABEL,
        "Too short.\n\nObligations and deadlines "
        "(validated exact-contract rows):\n2026-06-30",
    )


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


def test_write_run_record_rejects_a_delayed_section_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    review.risk_flags[0].clause = "Section summary 1.1"
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
            findings="Execution is not established from the visible signature page.",
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
    recorded_files = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file() and path != run_dir / "manifest.json"
    }
    assert set(artifacts) == recorded_files

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
    recorded_review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
    assert manifest["source_inputs"][0]["parties"] == recorded_review["parties"]
    sections = {row["section"] for row in source_evidence["risk_clauses"]}
    assert {"1.3", "5.3"} <= sections
    assert evaluation["checks"]["required_api_call_sequence"] is True
    assert evaluation["observed_call_sequence"] == evaluation["expected_call_sequence"]
