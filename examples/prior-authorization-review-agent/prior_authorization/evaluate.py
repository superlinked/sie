from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def evaluate_review(review: dict[str, Any]) -> list[Check]:
    evidence = review.get("ranked_source_evidence", [])
    evidence_text = "\n".join(str(row.get("text", "")) for row in evidence).casefold()
    return [
        Check(
            "published-cms-example-scope",
            review.get("published_example") is True and str(review.get("scope", "")).startswith("Reproduction of CMS"),
            str(review.get("scope")),
        ),
        Check("hcpcs-code", review.get("hcpcs_code") == "L1851", str(review.get("hcpcs_code"))),
        Check(
            "insufficient-documentation-route",
            review.get("route") == "insufficient_documentation",
            str(review.get("route")),
        ),
        Check(
            "six-month-requirement",
            review.get("required_face_to_face_within_months") == 6,
            str(review.get("required_face_to_face_within_months")),
        ),
        Check(
            "seven-month-observation",
            review.get("documented_face_to_face_age_months") == 7,
            str(review.get("documented_face_to_face_age_months")),
        ),
        Check("one-month-gap", review.get("overdue_by_months") == 1, str(review.get("overdue_by_months"))),
        Check(
            "missing-timely-face-to-face",
            review.get("missing_documentation") == ["face-to-face encounter within 6 months of proof of delivery"],
            str(review.get("missing_documentation")),
        ),
        Check(
            "published-payment-action",
            review.get("review_conclusion") == "insufficient documentation error"
            and review.get("payment_action") == "MAC recoups payment",
            f"{review.get('review_conclusion')}; {review.get('payment_action')}",
        ),
        Check(
            "ranked-source-evidence",
            bool(evidence)
            and all(
                token in evidence_text
                for token in (
                    "l1851",
                    "within the 6 months",
                    "face-to-face encounter 7 months ago",
                    "insufficient documentation error",
                    "mac recoups payment",
                )
            ),
            str([row.get("chunk_id") for row in evidence]),
        ),
        Check(
            "no-new-coverage-or-medical-decision",
            review.get("coverage_decision") is None and review.get("medical_decision") is None,
            f"coverage={review.get('coverage_decision')}; medical={review.get('medical_decision')}",
        ),
    ]


def _record_evaluation_artifact(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    evaluation_path = run_dir / "evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = [
        entry
        for entry in manifest.get("artifacts", [])
        if isinstance(entry, dict) and entry.get("path") != "evaluation.json"
    ]
    artifacts.append(
        {
            "path": "evaluation.json",
            "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        }
    )
    manifest["artifacts"] = sorted(artifacts, key=lambda entry: str(entry["path"]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def evaluate_run(run_dir: Path) -> bool:
    review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
    checks = evaluate_review(review)
    passed = all(check.passed for check in checks)
    (run_dir / "evaluation.json").write_text(
        json.dumps({"passed": passed, "checks": [asdict(check) for check in checks]}, indent=2) + "\n",
        encoding="utf-8",
    )
    _record_evaluation_artifact(run_dir)
    table = Table("Check", "Result", "Detail")
    for check in checks:
        table.add_row(check.name, "[green]pass[/]" if check.passed else "[red]fail[/]", check.detail)
    console.print(table)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved CMS L1851 example reproduction")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not evaluate_run(args.run_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
