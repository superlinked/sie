from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .app import (
    _INVESTIGATOR_TOOL_SEQUENCE,
    ContractReview,
    _published_review_missing_labels,
    _source_section_excerpt,
)
from .config import PROJECT_ROOT
from .runtime import Ledger

RUNTIME_MODEL_OPTIONAL_FUNCTIONS = frozenset({"encode", "extract"})
PUBLISHED_CONTRACT_LABEL = "CUAD · limeenergyco-09-09-1999-ex-10-distributor-agreement"
PUBLISHED_RISK_TARGETS = {
    "1.3": "high",
    "4.2": "high",
    "4.4": "medium",
    "5.3": "high",
}
PUBLISHED_FACT_SECTIONS = ("1.1", "1.3", "1.6", "4.1", "6.9")


def _runtime_model_is_valid(call: dict[str, Any]) -> bool:
    runtime_model = call.get("runtime_model")
    if runtime_model is None:
        return call.get("function") in RUNTIME_MODEL_OPTIONAL_FUNCTIONS
    return (
        isinstance(runtime_model, str)
        and bool(runtime_model)
        and runtime_model == call.get("requested_model")
    )


def _expected_call_sequence(
    cfg: dict[str, Any],
    *,
    include_citation_repair: bool = False,
    include_synthesis_repair: bool = False,
) -> list[dict[str, str]]:
    models = cfg["models"]
    sequence = [
        ("safety_guardrail", "generate", "guard"),
        ("classify_document", "generate", "triage"),
        ("ocr_signature_page", "extract", "ocr"),
        ("extract_entities", "extract", "entities"),
        ("read_signature_page", "generate", "vision"),
        ("search_clauses:index", "encode", "embed"),
    ]
    search_rounds = sum(
        1 for name, _ in _INVESTIGATOR_TOOL_SEQUENCE if name == "search_clauses"
    )
    for _ in range(search_rounds):
        sequence.extend(
            [
                ("search_clauses:encode", "encode", "embed"),
                ("search_clauses:score", "score", "rerank"),
            ]
        )
    sequence.extend(
        [
            ("analyze_clause_risks", "generate", "reasoning"),
            ("query_obligations_db", "generate", "sql"),
            ("investigator_report", "generate", "orchestrator"),
        ]
    )
    if include_citation_repair:
        sequence.append(
            ("investigator_report:citation_repair", "generate", "orchestrator")
        )
    sequence.append(("synthesize_review", "generate", "orchestrator"))
    if include_synthesis_repair:
        sequence.append(("synthesize_review:repair", "generate", "orchestrator"))
    return [
        {"stage": stage, "function": function, "requested_model": models[role]}
        for stage, function, role in sequence
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _normalize_model_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.rstrip().splitlines()) + "\n"


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or run_id in {
        ".",
        "..",
    }:
        raise ValueError(
            "run_id must be one safe directory name containing only letters, "
            "digits, '.', '_', or '-'"
        )
    return run_id


def ensure_run_destination_available(run_id: str) -> Path:
    validate_run_id(run_id)
    destination = PROJECT_ROOT / "runs" / run_id
    if destination.exists():
        raise FileExistsError(f"Run evidence already exists at {destination}")
    return destination


def _risk_clause_source_evidence(
    contract_text: str, review: ContractReview, label: str
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    for risk in review.risk_flags:
        if not re.match(r"^Sections?\s+\d+(?:\.\d+)+\b", risk.clause):
            raise RuntimeError(
                f"Risk clause has no source section reference: {risk.clause}"
            )
        sections = list(dict.fromkeys(re.findall(r"\b\d+(?:\.\d+)+\b", risk.clause)))
        if not sections:
            raise RuntimeError(
                f"Risk clause has no source section reference: {risk.clause}"
            )
        for section in sections:
            excerpt = _source_section_excerpt(
                contract_text,
                section,
                error_prefix=f"Risk clause {risk.clause!r}",
            )
            if not excerpt:
                raise RuntimeError(f"Risk clause has no source excerpt: {risk.clause}")
            rows.append(
                {
                    "clause": risk.clause,
                    "citation": f"Section {section}",
                    "section": section,
                    "excerpt": excerpt,
                    "excerpt_sha256": _sha256_bytes(excerpt.encode()),
                }
            )
    commercial_fact_clauses: list[dict[str, str]] = []
    if label == PUBLISHED_CONTRACT_LABEL:
        for section in PUBLISHED_FACT_SECTIONS:
            excerpt = _source_section_excerpt(
                contract_text,
                section,
                error_prefix="Published fact",
            )
            assert excerpt is not None
            commercial_fact_clauses.append(
                {
                    "citation": f"Section {section}",
                    "section": section,
                    "excerpt": excerpt,
                    "excerpt_sha256": _sha256_bytes(excerpt.encode()),
                }
            )
    return {
        "contract_text_sha256": _sha256_bytes(contract_text.encode()),
        "risk_clauses": rows,
        "commercial_fact_clauses": commercial_fact_clauses,
    }


def _risk_claims_are_source_supported(
    review: ContractReview, source_evidence: dict[str, Any], label: str
) -> bool:
    excerpts = {
        str(row["section"]): str(row["excerpt"]).casefold()
        for row in source_evidence["risk_clauses"]
    }
    all_source = "\n".join(excerpts.values())
    all_claims = "\n".join(
        [
            review.recommendation,
            *(risk.issue for risk in review.risk_flags),
        ]
    ).casefold()
    automatic_renewal = re.compile(
        r"\bautomatic(?:ally)?(?:\s+\w+){0,2}\s+renew(?:al|als|s|ed|ing)\b"
    )
    if automatic_renewal.search(all_claims) and not automatic_renewal.search(
        all_source
    ):
        for match in automatic_renewal.finditer(all_claims):
            qualifier = all_claims[max(0, match.start() - 80) : match.start()]
            if not re.search(
                r"\b(?:ambiguity|uncertain|uncertainty|unclear|whether)\b", qualifier
            ):
                return False
    if (
        "right of first refusal" in all_claims
        and "right of first refusal" not in all_source
    ):
        return False
    for risk in review.risk_flags:
        sections = re.findall(r"\b\d+(?:\.\d+)+\b", risk.clause)
        source = "\n".join(excerpts.get(section, "") for section in sections)
        claim = risk.issue.casefold()
        if (
            "cure" in claim
            and "30 day" in claim
            and "commercially reasonable" in source
            and "commercially reasonable" not in claim
        ):
            return False
        if (
            label == PUBLISHED_CONTRACT_LABEL
            and sections == ["4.2"]
            and (
                "commercially reasonable" not in claim
                or "without a cure" in claim
                or "no cure" in claim
            )
        ):
            return False
        if (
            label == PUBLISHED_CONTRACT_LABEL
            and sections == ["4.4"]
            and (
                "without cause" not in claim
                or ("mandatory" not in claim and "shall" not in claim)
                or "option" not in claim
                or "inventory" not in claim
                or re.search(
                    r"\boption\b.{0,60}\bonly\b.{0,60}\bwithout cause\b", claim
                )
                or re.search(r"\bmandatory exception\b.{0,80}\boptional\b", claim)
                or "drafted as optional" in claim
            )
        ):
            return False
        if (
            label == PUBLISHED_CONTRACT_LABEL
            and sections == ["5.3"]
            and (
                "not at fault" not in claim
                or "indemnif" in claim
                and "for company's negligence" in claim
            )
        ):
            return False
    return True


def _signature_scope_is_supported(findings: str, review: ContractReview) -> bool:
    normalized = " ".join(findings.casefold().split())
    review_text = " ".join(review.recommendation.casefold().split())
    if review.executed:
        return "not executed" not in normalized and "not executed" not in review_text
    return (
        "not established from the visible signature page" in normalized
        and "not executed" not in normalized
        and "not executed" not in review_text
        and "unexecuted" not in review_text
        and "draft or unsigned copy" not in normalized
        and "document ends at" not in normalized
    )


def _published_investigator_findings_missing(label: str, findings: str) -> list[str]:
    if label != PUBLISHED_CONTRACT_LABEL:
        return []
    normalized = " ".join(findings.casefold().split())
    requirements = {
        "all cited risk and fact sections": all(
            section in normalized
            for section in (
                "section 1.3",
                "section 1.6",
                "section 4.2",
                "section 4.4",
                "section 5.3",
                "section 6.9",
            )
        ),
        "commercial amounts and unit minimum": all(
            fact in normalized for fact in ("$500,000", "$250,000", "375")
        ),
        "quarterly compliance obligation": (
            "quarter" in normalized and "compliance" in normalized
        ),
        "annual subscription obligation": (
            "annual" in normalized and "subscription" in normalized
        ),
        "renewal notice obligation": (
            "renewal" in normalized and "notice" in normalized
        ),
        "upcoming obligations section": "upcoming obligation" in normalized,
        "indemnity fault condition": "not at fault" in normalized,
        "June 30, 2026 deadline": any(
            value in normalized
            for value in ("2026-06-30", "june 30, 2026", "june 30 2026")
        ),
        "July 1, 2026 deadline": any(
            value in normalized
            for value in ("2026-07-01", "july 1, 2026", "july 1 2026")
        ),
        "September 15, 2026 deadline": any(
            value in normalized
            for value in ("2026-09-15", "september 15, 2026", "september 15 2026")
        ),
        "no unsupported Section 6.7 citation": "section 6.7" not in normalized,
        "no unrelated contract rows": "centrack" not in normalized,
        "exact document classification": (
            "distributor agreement" in normalized
            and "master services agreement" not in normalized
            and re.search(r"\bmsa\b", normalized) is None
        ),
        "partial visible signature evidence": (
            "/s/" in normalized
            and "joseph marino" in normalized
            and "jim stump" in normalized
            and "no actual signature" not in normalized
            and "no signatures are present" not in normalized
        ),
        "visible-page execution qualification": (
            "not established from the visible signature page" in normalized
            and any(
                wording in normalized
                for wording in ("no execution dates", "no dates", "not dated")
            )
        ),
    }
    return [name for name, present in requirements.items() if not present]


def _published_investigator_findings_are_complete(label: str, findings: str) -> bool:
    return not _published_investigator_findings_missing(label, findings)


def _published_risk_coverage_is_preserved(label: str, review: ContractReview) -> bool:
    if label != PUBLISHED_CONTRACT_LABEL:
        return True
    actual: dict[str, str] = {}
    for risk in review.risk_flags:
        sections = re.findall(r"\b\d+(?:\.\d+)+\b", risk.clause)
        if len(sections) != 1 or sections[0] in actual:
            return False
        actual[sections[0]] = risk.severity.casefold()
    return actual == PUBLISHED_RISK_TARGETS


def _published_fact_coverage_is_preserved(label: str, review: ContractReview) -> bool:
    if label != PUBLISHED_CONTRACT_LABEL:
        return True
    return not _published_review_missing_labels(review)


def _published_fact_source_coverage_is_preserved(
    label: str, source_evidence: dict[str, Any]
) -> bool:
    if label != PUBLISHED_CONTRACT_LABEL:
        return True
    sections = [
        str(row.get("section"))
        for row in source_evidence.get("commercial_fact_clauses", [])
    ]
    return sections == list(PUBLISHED_FACT_SECTIONS)


def _guardrail_was_accepted(ledger: Ledger) -> bool:
    entries = [
        entry
        for entry in ledger.entries
        if entry.step == "Safety guardrail (granite-guardian)"
    ]
    return len(entries) == 1 and entries[0].got.strip().casefold() == "no"


def write_run_record(
    *,
    run_id: str,
    endpoint: str,
    cfg: dict[str, Any],
    label: str,
    contract_text: str,
    scan_path: str,
    db_path: str,
    findings: str,
    review: ContractReview,
    ledger: Ledger,
    api_calls: list[dict[str, Any]],
    wall_s: float,
) -> Path:
    missing_findings = _published_investigator_findings_missing(label, findings)
    if missing_findings:
        raise RuntimeError(
            "Published investigator findings are incomplete: "
            + ", ".join(missing_findings)
        )
    final_run_dir = ensure_run_destination_available(run_id)
    runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    reservation_dir = runs_dir / f".{run_id}.lock"
    try:
        reservation_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"Run ID is already reserved: {run_id}") from exc
    try:
        if final_run_dir.exists():
            raise FileExistsError(f"Run evidence already exists at {final_run_dir}")
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs_dir))
        try:
            _write_run_record(
                run_dir=staging_dir,
                run_id=run_id,
                endpoint=endpoint,
                cfg=cfg,
                label=label,
                contract_text=contract_text,
                scan_path=scan_path,
                db_path=db_path,
                findings=findings,
                review=review,
                ledger=ledger,
                api_calls=api_calls,
                wall_s=wall_s,
            )
            staging_dir.rename(final_run_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    finally:
        reservation_dir.rmdir()
    return final_run_dir


def _write_run_record(
    *,
    run_dir: Path,
    run_id: str,
    endpoint: str,
    cfg: dict[str, Any],
    label: str,
    contract_text: str,
    scan_path: str,
    db_path: str,
    findings: str,
    review: ContractReview,
    ledger: Ledger,
    api_calls: list[dict[str, Any]],
    wall_s: float,
) -> None:

    review_path = run_dir / "review.json"
    findings_path = run_dir / "investigator-findings.txt"
    ledger_path = run_dir / "ledger.json"
    calls_path = run_dir / "api-calls.json"
    evaluation_path = run_dir / "evaluation.json"
    source_evidence_path = run_dir / "source-evidence.json"
    _write_json(review_path, review.model_dump(mode="json"))
    findings_path.write_text(_normalize_model_text(findings), encoding="utf-8")
    _write_json(ledger_path, [asdict(entry) for entry in ledger.entries])
    _write_json(calls_path, api_calls)
    source_evidence = _risk_clause_source_evidence(contract_text, review, label)
    _write_json(source_evidence_path, source_evidence)

    requested_models = sorted({call["requested_model"] for call in api_calls})
    expected_models = sorted(set(cfg["models"].values()))
    observed_functions = sorted({call["function"] for call in api_calls})
    observed_call_sequence = [
        {field: call.get(field) for field in ("stage", "function", "requested_model")}
        for call in api_calls
        if call.get("stage") != "warmup"
    ]
    repair_calls = [
        call
        for call in observed_call_sequence
        if call.get("stage") == "synthesize_review:repair"
    ]
    citation_repair_calls = [
        call
        for call in observed_call_sequence
        if call.get("stage") == "investigator_report:citation_repair"
    ]
    expected_call_sequence = _expected_call_sequence(
        cfg,
        include_citation_repair=len(citation_repair_calls) == 1,
        include_synthesis_repair=len(repair_calls) == 1,
    )
    request_ids = [call.get("request_id") for call in api_calls]
    checks = {
        "structured_review": True,
        "guardrail_was_accepted": _guardrail_was_accepted(ledger),
        "parties_identified": bool(review.parties),
        "risk_flags_identified": bool(review.risk_flags),
        "api_calls_have_request_provenance": bool(api_calls)
        and all(
            isinstance(call.get("credits_debited"), int | float)
            and not isinstance(call["credits_debited"], bool)
            and call["credits_debited"] >= 0
            and _runtime_model_is_valid(call)
            and all(
                isinstance(call.get(field), str) and bool(call[field])
                for field in (
                    "stage",
                    "request_id",
                    "rate_book_version",
                    "execution_identity_sha256",
                )
            )
            for call in api_calls
        ),
        "api_call_request_ids_unique": len(request_ids) == len(set(request_ids)),
        "risk_clauses_supported_by_source": bool(source_evidence["risk_clauses"]),
        "risk_claims_supported_by_source": _risk_claims_are_source_supported(
            review, source_evidence, label
        ),
        "published_risk_coverage_non_regression": (
            _published_risk_coverage_is_preserved(label, review)
        ),
        "published_fact_coverage_non_regression": (
            _published_fact_coverage_is_preserved(label, review)
        ),
        "published_fact_source_coverage_non_regression": (
            _published_fact_source_coverage_is_preserved(label, source_evidence)
        ),
        "signature_image_scope_not_overclaimed": _signature_scope_is_supported(
            findings, review
        ),
        "published_investigator_findings_complete": (
            _published_investigator_findings_are_complete(label, findings)
        ),
        "all_configured_models_called": requested_models == expected_models,
        "native_primitives_called": set(observed_functions)
        >= {"encode", "extract", "generate", "score"},
        "required_api_call_sequence": observed_call_sequence == expected_call_sequence,
        "bounded_citation_repair": len(citation_repair_calls) <= 1,
        "bounded_synthesis_repair": len(repair_calls) <= 1,
    }
    _write_json(
        evaluation_path,
        {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_models": expected_models,
            "requested_models": requested_models,
            "observed_functions": observed_functions,
            "expected_call_sequence": expected_call_sequence,
            "observed_call_sequence": observed_call_sequence,
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"Production evidence checks failed: {checks}")

    artifacts = [
        calls_path,
        evaluation_path,
        findings_path,
        ledger_path,
        review_path,
        source_evidence_path,
    ]
    scan = Path(scan_path)
    database = Path(db_path)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "endpoint": endpoint,
            "execution": "SIE API",
            "models": cfg["models"],
            "source_inputs": [
                {
                    "label": label,
                    "kind": "contract_text",
                    "sha256": _sha256_bytes(contract_text.encode()),
                },
                {"path": scan.name, "kind": "signature_scan", "sha256": _sha256(scan)},
                {
                    "path": database.name,
                    "kind": "obligations_db",
                    "sha256": _sha256(database),
                },
            ],
            "timing_ms": {"end_to_end": round(wall_s * 1000, 1)},
            "timing_note": "Diagnostic run timing, not a benchmark.",
            "artifacts": [
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in artifacts
            ],
        },
    )
