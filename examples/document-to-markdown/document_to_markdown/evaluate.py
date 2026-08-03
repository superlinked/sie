from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from document_to_markdown.config import load_config, select_documents

console = Console()


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _table_count(markdown: str) -> int:
    lines = markdown.splitlines()
    return sum(
        1
        for index, line in enumerate(lines[:-1])
        if line.lstrip().startswith("|")
        and line.count("|") >= 2
        and "-" in lines[index + 1]
        and set(lines[index + 1].replace("|", "").replace(":", "").strip()) <= {"-", " "}
    )


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip().casefold()


def _evaluate_markdown(
    markdown: str, required: tuple[str, ...], ordered: tuple[str, ...], tables: int
) -> list[CheckResult]:
    folded = _normalized(markdown)
    checks = [
        CheckResult(
            name=f"contains:{text}",
            passed=_normalized(text) in folded,
            detail=text,
        )
        for text in required
    ]
    if ordered:
        positions = [folded.find(_normalized(text)) for text in ordered]
        ordered_passed = all(position >= 0 for position in positions) and positions == sorted(positions)
        checks.append(
            CheckResult(
                name="reading-order",
                passed=ordered_passed,
                detail=" -> ".join(ordered),
            )
        )
    actual_tables = _table_count(markdown)
    checks.append(
        CheckResult(
            name="markdown-tables",
            passed=actual_tables >= tables,
            detail=f"{actual_tables} found, {tables} required",
        )
    )
    return checks


def evaluate_run(run_dir: Path, slugs: list[str]) -> bool:
    config = load_config()
    documents = select_documents(config, slugs)
    rows = []
    passed_all = True
    for document in documents:
        markdown_path = run_dir / "markdown" / f"{document.slug}.md"
        if not markdown_path.exists():
            raise FileNotFoundError(markdown_path)
        markdown = markdown_path.read_text(encoding="utf-8")
        checks = _evaluate_markdown(
            markdown,
            document.checks.required_text,
            document.checks.ordered_text,
            document.checks.minimum_markdown_tables,
        )
        passed = sum(check.passed for check in checks)
        passed_all = passed_all and passed == len(checks)
        rows.append(
            {
                "slug": document.slug,
                "passed": passed,
                "total": len(checks),
                "checks": [asdict(check) for check in checks],
            }
        )

    result = {
        "run_dir": str(run_dir),
        "passed": passed_all,
        "documents": rows,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    table = Table("Document", "Checks", "Result")
    for row in rows:
        table.add_row(
            row["slug"],
            f"{row['passed']}/{row['total']}",
            "[green]pass[/]" if row["passed"] == row["total"] else "[red]fail[/]",
        )
    console.print(table)
    return passed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic checks against a saved conversion")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("slugs", nargs="*", default=["all"])
    args = parser.parse_args()
    if not evaluate_run(args.run_dir, args.slugs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
