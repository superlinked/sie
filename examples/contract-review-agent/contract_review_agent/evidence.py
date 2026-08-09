from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .app import ContractReview
from .config import PROJECT_ROOT
from .runtime import Ledger


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
    checks = {
        "structured_review": True,
        "parties_identified": bool(review.parties),
        "risk_flags_identified": bool(review.risk_flags),
        "all_configured_models_called": requested_models == expected_models,
        "native_primitives_called": set(observed_functions)
        >= {"encode", "extract", "generate", "score"},
    }
    _write_json(
        evaluation_path,
        {
            "passed": all(checks.values()),
            "checks": checks,
            "expected_models": expected_models,
            "requested_models": requested_models,
            "observed_functions": observed_functions,
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
            "sie_server_commit": os.environ.get("SIE_SERVER_COMMIT"),
            "run_command": os.environ.get("SIE_RUN_COMMAND"),
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
