from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .catalog import load_annoctr_catalog, load_catalog
from .config import load_config
from .data import ensure_sources, find_annoctr_catalog, load_linking_cases
from .evaluation import evaluate_predictions, read_predictions
from .runner import benchmark, full_report_benchmark, map_annoctr_demo, map_report, write_json

console = Console()


def _fetch(args: argparse.Namespace) -> None:
    config = load_config()
    paths = ensure_sources(config, force=args.force)
    techniques = load_catalog(paths["attack"])
    benchmark_techniques = load_annoctr_catalog(find_annoctr_catalog(paths["annoctr"]))
    benchmark_ids = {technique.technique_id for technique in benchmark_techniques}
    table = Table("Source", "Path", "Cases", "Scored", "Missing IDs")
    table.add_row("MITRE ATT&CK Enterprise", str(paths["attack"]), str(len(techniques)), "", "")
    table.add_row("AnnoCTR ATT&CK snapshot", str(paths["annoctr"]), str(len(benchmark_techniques)), "", "")
    for split in ("train", "dev", "test"):
        cases = load_linking_cases(paths["annoctr"], split)
        scored = sum(bool(set(case.gold_ids).intersection(benchmark_ids)) for case in cases)
        historical = sum(len(set(case.gold_ids).difference(benchmark_ids)) for case in cases)
        table.add_row(f"AnnoCTR {split}", str(paths["annoctr"]), str(len(cases)), str(scored), str(historical))
    console.print(table)


def _benchmark(args: argparse.Namespace) -> None:
    config = load_config()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = benchmark(config, split=args.split, limit=args.limit, stage=args.stage, run_id=run_id)
    console.print(f"[green]Wrote[/] {result}")


def _report(args: argparse.Namespace) -> None:
    config = load_config()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = map_report(config, report_path=args.report, run_id=run_id)
    console.print(f"[green]Wrote[/] {result}")


def _full_benchmark(args: argparse.Namespace) -> None:
    config = load_config()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = full_report_benchmark(config, split=args.split, limit=args.limit, run_id=run_id)
    console.print(f"[green]Wrote[/] {result}")


def _demo(args: argparse.Namespace) -> None:
    config = load_config()
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = map_annoctr_demo(
        config,
        document=args.document,
        split=args.split,
        run_id=run_id,
    )
    console.print(f"[green]Wrote[/] {result}")


def _evaluate(args: argparse.Namespace) -> None:
    predictions_path = args.run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions not found: {predictions_path}")
    evaluation = evaluate_predictions(read_predictions(predictions_path))
    output = args.run_dir / "evaluation.json"
    write_json(output, evaluation)
    console.print_json(data=evaluation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Map cited threat-report behavior to MITRE ATT&CK with SIE")
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="Download and verify the pinned ATT&CK and AnnoCTR sources")
    fetch.add_argument("--force", action="store_true")
    fetch.set_defaults(func=_fetch)

    run_benchmark = commands.add_parser("benchmark", help="Run the mention-linking benchmark")
    run_benchmark.add_argument("--split", choices=("dev", "test"), default="test")
    run_benchmark.add_argument("--limit", type=int)
    run_benchmark.add_argument("--stage", choices=("retrieve", "rerank", "verify"), default="retrieve")
    run_benchmark.add_argument("--run-id")
    run_benchmark.set_defaults(func=_benchmark)

    full_benchmark = commands.add_parser(
        "full-benchmark",
        help="Run behavior detection and ATT&CK mapping from complete AnnoCTR reports",
    )
    full_benchmark.add_argument("--split", choices=("dev", "test"), default="dev")
    full_benchmark.add_argument("--limit", type=int)
    full_benchmark.add_argument("--run-id")
    full_benchmark.set_defaults(func=_full_benchmark)

    report = commands.add_parser("report", help="Map behaviors in one text, Markdown, HTML, or PDF report")
    report.add_argument("report", type=Path)
    report.add_argument("--run-id")
    report.set_defaults(func=_report)

    demo = commands.add_parser("demo", help="Map a real report from the pinned AnnoCTR corpus")
    demo.add_argument(
        "--document",
        default="proofpoint_2022-02-03_mfa-psa-oh-my",
        help="AnnoCTR document stem",
    )
    demo.add_argument("--split", choices=("train", "dev", "test"), default="test")
    demo.add_argument("--run-id")
    demo.set_defaults(func=_demo)

    evaluate = commands.add_parser("evaluate", help="Recompute metrics from a saved prediction ledger")
    evaluate.add_argument("run_dir", type=Path)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
