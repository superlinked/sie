from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()
ARTIFACT_EXCLUDED_PATHS = {Path("README.md"), Path("manifest.json")}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _close(value: object, expected: float) -> bool:
    try:
        return abs(float(value) - expected) < 0.01
    except (TypeError, ValueError):
        return False


def evaluate_review(review: dict[str, Any]) -> list[Check]:
    summary = review.get("appeal_summary", {})
    decision = review.get("decision", {})
    findings = review.get("findings", [])
    categories = {finding.get("category") for finding in findings}
    covered = str(decision.get("covered_scope", "")).casefold()
    excluded = str(decision.get("excluded_scope", "")).casefold()
    evidence = str(decision.get("evidence_needed", "")).casefold()
    overlap = str(decision.get("prior_claim_check", "")).casefold()
    overlap_finding = next(
        (
            str(finding.get("evidence", "")).casefold()
            for finding in findings
            if finding.get("category") == "prior_claim_overlap"
        ),
        "",
    )
    overlap_action = next(
        (
            str(action).casefold()
            for action in review.get("next_actions", [])
            if "prior claim" in str(action).casefold()
            or "previous claim" in str(action).casefold()
            or "payment overlap" in str(action).casefold()
        ),
        "",
    )

    def preserves_prior_claim_timing(value: str) -> bool:
        return (
            ("same area" in value or "underneath the building" in value)
            and "before" in value
            and "july 2019" in value
            and "repair" in value
            and "pric" in value
        )

    return [
        Check(
            "scope-review-route",
            review.get("route") == "scope_review_required",
            str(review.get("route")),
        ),
        Check(
            "proof-of-loss-amount",
            _close(summary.get("proof_of_loss_amount"), 182552),
            str(summary.get("proof_of_loss_amount")),
        ),
        Check(
            "removal-estimate",
            _close(summary.get("removal_estimate"), 49500),
            str(summary.get("removal_estimate")),
        ),
        Check(
            "barge-estimate",
            _close(summary.get("barge_estimate"), 181832.94),
            str(summary.get("barge_estimate")),
        ),
        Check(
            "debris-volume",
            summary.get("debris_cubic_yards_min") == 12 and summary.get("debris_cubic_yards_max") == 15,
            f"{summary.get('debris_cubic_yards_min')} to {summary.get('debris_cubic_yards_max')}",
        ),
        Check(
            "covered-scope",
            "underneath" in covered and "perimeter" in covered,
            str(decision.get("covered_scope")),
        ),
        Check(
            "excluded-scope",
            "barge" in excluded and "disposal" in excluded,
            str(decision.get("excluded_scope")),
        ),
        Check(
            "price-support",
            "estimate" in evidence or "contractor" in evidence,
            str(decision.get("evidence_needed")),
        ),
        Check(
            "prior-claim-overlap",
            all(preserves_prior_claim_timing(value) for value in (overlap, overlap_finding, overlap_action)),
            f"{overlap} | {overlap_finding} | {overlap_action}",
        ),
        Check(
            "finding-categories",
            {
                "covered_removal",
                "excluded_transport",
                "price_support",
                "prior_claim_overlap",
            }
            <= categories,
            ", ".join(sorted(str(category) for category in categories)),
        ),
    ]


def evaluate_run(run_dir: Path) -> bool:
    review_path = run_dir / "review.json"
    if not review_path.exists():
        raise FileNotFoundError(review_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    checks = evaluate_review(review)
    passed = all(check.passed for check in checks)
    (run_dir / "evaluation.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": [asdict(check) for check in checks],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = [
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.relative_to(run_dir) not in ARTIFACT_EXCLUDED_PATHS
        ]
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    table = Table("Check", "Result", "Detail")
    for check in checks:
        table.add_row(
            check.name,
            "[green]pass[/]" if check.passed else "[red]fail[/]",
            check.detail,
        )
    console.print(table)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the saved FEMA appeal review")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not evaluate_run(args.run_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
