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


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def evaluate_review(review: dict[str, Any]) -> list[Check]:
    change = review.get("change", {})
    excluded = set(review.get("claims_excluded", []))
    evidence = review.get("ranked_evidence", [])
    evidence_text = "\n".join(str(row.get("text", "")) for row in evidence).casefold()
    return [
        Check("route", review.get("route") == "superseded_figure", str(review.get("route"))),
        Check("net-income-delta", abs(float(change.get("value_millions", 0)) + 9.016) < 0.0001, str(change)),
        Check("eps-delta", abs(float(change.get("diluted_eps", 0)) + 0.34) < 0.0001, str(change)),
        Check("percent-delta", abs(float(change.get("percent", 0)) + 20.0) < 0.05, str(change)),
        Check(
            "source-lineage", len(review.get("controlling_sources", [])) == 3, str(review.get("controlling_sources"))
        ),
        Check(
            "caveat-preserved",
            "over the life of the portfolio" in review.get("company_caveat", ""),
            review.get("company_caveat", ""),
        ),
        Check(
            "ranked-source-evidence",
            bool(evidence)
            and all(
                token in evidence_text
                for token in (
                    "pathward financial",
                    "$45,096",
                    "$36,080",
                    "$1.68",
                    "$1.34",
                    "no longer",
                    "over the life of the portfolio",
                )
            ),
            str([row.get("chunk_id") for row in evidence]),
        ),
        Check(
            "unsafe-claims-excluded",
            {"fraud", "misconduct", "investment recommendation"} <= excluded,
            ", ".join(sorted(excluded)),
        ),
    ]


def evaluate_run(run_dir: Path) -> bool:
    review = json.loads((run_dir / "review.json").read_text(encoding="utf-8"))
    checks = evaluate_review(review)
    passed = all(check.passed for check in checks)
    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(
        json.dumps({"passed": passed, "checks": [asdict(check) for check in checks]}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = [
            artifact for artifact in manifest.get("artifacts", []) if artifact.get("path") != "evaluation.json"
        ]
        artifacts.append(
            {"path": "evaluation.json", "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()}
        )
        manifest["artifacts"] = artifacts
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    table = Table("Check", "Result", "Detail")
    for check in checks:
        table.add_row(check.name, "[green]pass[/]" if check.passed else "[red]fail[/]", check.detail)
    console.print(table)
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved filing review")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if not evaluate_run(args.run_dir):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
