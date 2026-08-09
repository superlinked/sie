from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .app import ContractReview
from .config import PROJECT_ROOT
from .runtime import Ledger


def _expected_call_sequence(cfg: dict[str, Any]) -> list[dict[str, str]]:
    models = cfg["models"]
    sequence = [
        ("safety_guardrail", "generate", "guard"),
        ("classify_document", "generate", "triage"),
        ("ocr_signature_page", "extract", "ocr"),
        ("extract_entities", "extract", "entities"),
        ("read_signature_page", "generate", "vision"),
        ("search_clauses:index", "encode", "embed"),
    ]
    for query in (
        "automatic renewal",
        "limitation of liability",
        "indemnification",
        "termination",
    ):
        sequence.extend(
            [
                (f"search_clauses:{query}:encode", "encode", "embed"),
                (f"search_clauses:{query}:score", "score", "rerank"),
            ]
        )
    sequence.extend(
        [
            ("analyze_clause_risks", "generate", "reasoning"),
            ("query_obligations_db", "generate", "sql"),
            ("investigator_report", "generate", "orchestrator"),
            ("synthesize_review", "generate", "orchestrator"),
        ]
    )
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
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or run_id in {
        ".",
        "..",
    }:
        raise ValueError(
            "run_id must be one safe directory name containing only letters, "
            "digits, '.', '_', or '-'"
        )
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    review_path = run_dir / "review.json"
    findings_path = run_dir / "investigator-findings.txt"
    ledger_path = run_dir / "ledger.json"
    calls_path = run_dir / "api-calls.json"
    evaluation_path = run_dir / "evaluation.json"
    _write_json(review_path, review.model_dump(mode="json"))
    findings_path.write_text(findings.rstrip() + "\n", encoding="utf-8")
    _write_json(ledger_path, [asdict(entry) for entry in ledger.entries])
    _write_json(calls_path, api_calls)

    requested_models = sorted({call["requested_model"] for call in api_calls})
    expected_models = sorted(set(cfg["models"].values()))
    observed_functions = sorted({call["function"] for call in api_calls})
    expected_call_sequence = _expected_call_sequence(cfg)
    observed_call_sequence = [
        {field: call.get(field) for field in ("stage", "function", "requested_model")}
        for call in api_calls
        if call.get("stage") != "warmup"
    ]
    checks = {
        "structured_review": True,
        "parties_identified": bool(review.parties),
        "risk_flags_identified": bool(review.risk_flags),
        "api_calls_have_request_provenance": bool(api_calls)
        and all(
            all(
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
        "all_configured_models_called": requested_models == expected_models,
        "native_primitives_called": set(observed_functions)
        >= {"encode", "extract", "generate", "score"},
        "required_api_call_sequence": observed_call_sequence == expected_call_sequence,
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
    return run_dir
