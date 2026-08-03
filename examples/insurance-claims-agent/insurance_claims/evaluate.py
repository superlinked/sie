from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def evaluate_review(review: dict[str, Any]) -> list[Check]:
    summary = review.get("claim_summary", {})
    categories = {finding.get("category") for finding in review.get("findings", [])}
    blocking_signature = any(
        finding.get("category") == "missing_signature" and finding.get("severity") == "blocking"
        for finding in review.get("findings", [])
    )
    return [
        Check("manual-review-route", review.get("route") == "manual_review", str(review.get("route"))),
        Check(
            "claimed-total",
            abs(float(summary.get("claimed_total", 0)) - 81060) < 0.01,
            str(summary.get("claimed_total")),
        ),
        Check(
            "attachment-total",
            abs(float(summary.get("attachment_total", 0)) - 80660) < 0.01,
            str(summary.get("attachment_total")),
        ),
        Check(
            "difference",
            abs(float(summary.get("difference", 0)) - 400) < 0.01,
            str(summary.get("difference")),
        ),
        Check("missing-signature", blocking_signature, "blocking finding required"),
        Check("amount-mismatch", "amount_mismatch" in categories, "amount_mismatch finding required"),
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
    parser = argparse.ArgumentParser(description="Check the saved claim-review result")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not evaluate_run(args.run_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
