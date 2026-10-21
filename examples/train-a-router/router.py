"""Prepare, fit and evaluate the fixed paired-correctness routing study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import evidence
import frontier
import frontier_analysis
import paired_native
import paired_report
import paired_topic
import paired_training as core
import reproduction


def same_report(expected, actual):
    """Allow numeric roundoff across CPU libraries, never changed counts or metadata."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(same_report(expected[key], actual[key]) for key in expected)
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(same_report(a, b) for a, b in zip(expected, actual, strict=True))
    if isinstance(expected, float):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=1e-10, abs_tol=1e-10
        )
    return expected == actual


def evaluate_recorded(frozen_routing, final_grades, output, *, expected_report=None):
    """Replay the recorded final acquisition; never rebind it to new route masks."""
    output = Path(output)
    core.require(not output.exists(), "Preserve previous analysis")
    bundle, grades = core.read(frozen_routing), core.read(final_grades)
    core.require(bundle["frozen_before_final_model_outcomes"] is True, "Missing original frozen boundary")
    paired_report.check_lineage(bundle, grades, core.sha(frozen_routing))
    result = paired_report.evaluate(grades["rows"], bundle)
    result["provenance"] = {
        "reporter_sha256": core.sha(paired_report.__file__),
        "frozen_routing_sha256": core.sha(frozen_routing),
        "final_grades_sha256": core.sha(final_grades),
        "cost_scope": grades["cost_scope"],
    }
    core.write(output, result)
    if expected_report is not None:
        core.require(same_report(core.read(expected_report), result), "Recorded replay differs; retain output for inspection")
    return {"output": str(output), "sha256": core.sha(output), "questions": result["questions"]}


def show_report(path, target=None):
    """Present the saved result without fitting, API calls or changing a policy."""
    report = core.read(path)
    core.require(
        report["format"] in ("routerbench-paired-final-report-v1", "train-a-router-reproduction-report-v1"),
        "Use a complete paired evaluation report",
    )
    targets = (0, 1, 2, 5) if target is None else (target,)
    methods = {"base": "Laya", "lora": "Laya + LoRA", "embedding": "Embedding + classifier",
               "topic": "Text/topic control", "random": "Random control"}
    rows = [("always_strong", "Always Sol", "—"), ("always_cheap", "Always Luna", "—")]
    rows.extend((f"{method}@{value}pp", label, f"{value} pp")
                for method, label in methods.items() for value in targets)
    lines = [
        f"{report['questions']:,} final questions; GPT-6.1 Sol and GPT-6 Luna answer the requests.",
        "These are sampled cost–quality operating points, not a required quality-loss ceiling.",
        "The recorded setting selected a policy on calibration data. Intervals are 95% family-bootstrap intervals.",
        "",
        "| Method | Recorded setting | Answer quality | Quality loss [95% interval], pp | Answer $/10k | Total $/10k | Net savings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label, setting in rows:
        row = report["policies"][name]
        metrics = row["metrics"]
        interval = row["confidence95"]["quality_gap_from_always_strong_pp"]
        gap = f"{metrics['quality_gap_from_always_strong_pp']:.2f} [{interval['lower']:.2f}, {interval['upper']:.2f}]"
        total = metrics["total_usd_per_10000"]
        savings = metrics["net_savings_percent"]
        total_text = "hosting unpriced" if total is None else f"${total:.2f}"
        savings_text = "hosting unpriced" if savings is None else f"{savings:.1f}%"
        lines.append(
            f"| {label} | {setting} | {metrics['answer_accuracy_percent']:.2f}% | {gap} | "
            f"${metrics['answer_model_usd_per_10000']:.2f} | {total_text} | {savings_text} |"
        )
    lines.extend([
        "",
        "Costs use the recorded answer-token prices; embedding totals include the projected per-request Cloud encoder fee.",
        "Unpriced router hosting is never counted as free. See the separate serving measurements for deployment estimates.",
    ])
    if report.get("new_blind_evaluation") is False:
        lines.append("This is a refit evaluated on already published answers, not a new blind test.")
    return "\n".join(lines)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("fetch", help="Download an explicitly pinned artifact manifest")
    c.add_argument("--manifest-url", help="Override the published release; requires --manifest-sha256")
    c.add_argument("--manifest-sha256", help="Expected hash of an overridden manifest")
    c.add_argument("--release", type=Path, default=Path(__file__).with_name("release.json"), help="Pinned artifact release file")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--group", choices=("core", "weights", "recordings"), action="append")

    c = commands.add_parser("show", help="Print the saved quality/cost comparison and uncertainty")
    c.add_argument("--report", type=Path, required=True)
    c.add_argument("--target-gap-pp", type=int, choices=(0, 1, 2, 5), help="Show one recorded setting; defaults to all")

    c = commands.add_parser("frontier", help="Explore the cost–quality frontier and choose your own operating point")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, help="Defaults to the published data/reference run")
    c.add_argument("--include-test", action="store_true", help="Also inspect retrospective test outcomes; never use these to select a policy")
    c.add_argument("--output", type=Path, required=True, help="Standalone interactive HTML; also writes its source JSON")

    c = commands.add_parser("prepare", help="Create a new run from fixed question-only IDs, families and folds")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--acquisition-protocol-sha256", help="Defaults to the recorded reference plan")

    c = commands.add_parser("encode", help="Regenerate question embeddings through SIE Cloud or your own SIE server")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True, help="A new directory, separate from the downloaded features")
    c.add_argument("--batch-size", type=int, choices=range(1, 17), default=16)
    c.add_argument("--sie-base-url", help="SIE endpoint; defaults to SIE_BASE_URL or SIE Cloud")
    c.add_argument("--self-hosted", action="store_true", help="Leave Cloud billing fields unset for your own server")
    c.add_argument("--retry-unknown", action="store_true", help="Explicitly retry interrupted calls after reviewing their receipts")
    c.add_argument("--max-batches", type=int, help="Stop after this many new batches; resume by rerunning without this limit")

    c = commands.add_parser("fit-embedding", help="Fit two correctness heads using training-family CV")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--features", type=Path, help="Regenerated feature directory; defaults to data/features")

    c = commands.add_parser("fit-native", help="Select and refit attention LoRA on CUDA; training grades only")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--initial-checkpoint", type=Path, help="Defaults to reference/native/base")

    c = commands.add_parser("fit-topic", help="Fit the training-only text-to-topic control and score held inputs")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)

    c = commands.add_parser("reuse-controls", help="Reuse published comparison models for an embedding-only refit")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)

    c = commands.add_parser("score", help="Score calibration/final/development inputs without reading their grades")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--method", choices=("embedding", "native"), required=True)
    c.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    c.add_argument("--features", type=Path, help="Regenerated features for the embedding method; defaults to data/features")

    c = commands.add_parser("calibrate", help="Build the full calibration frontier and retain the four recorded comparison settings")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)

    for name, help_text in (
        ("evaluate", "Evaluate the original frozen routes against their bound final acquisition"),
        ("replay", "Reproduce and compare the recorded report; no models, keys or training"),
    ):
        c = commands.add_parser(name, help=help_text)
        c.add_argument("--data", type=Path, required=True)
        c.add_argument("--output", type=Path, required=True)
        c.add_argument("--frozen-routing", type=Path, help="Defaults to reference/frozen/frozen-final-routes.json")
        if name == "replay":
            c.add_argument("--expected-report", type=Path, help="Defaults to reference/report.json")

    c = commands.add_parser("evaluate-refit", help="Evaluate your refit on published test outcomes; no new blind test")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)

    c = commands.add_parser("answer", help="Route a new question and call the chosen answer model")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--method", choices=("base", "lora", "embedding"), required=True)
    choice = c.add_mutually_exclusive_group(required=True)
    choice.add_argument("--candidate-id", type=int, help="An operating point from this method's calibration frontier")
    choice.add_argument("--target-gap-pp", type=int, choices=(0, 1, 2, 5), help="Explicitly choose one of the four recorded settings")
    c.add_argument("--question", required=True)
    c.add_argument("--answer-type", choices=("exact_string", "integer", "choice"), default="exact_string")
    c.add_argument("--choices", nargs="+", help="Allowed option labels when answer-type is choice")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    c.add_argument("--sie-base-url", help="SIE endpoint; defaults to SIE_BASE_URL or SIE Cloud")
    c.add_argument(
        "--self-hosted",
        action="store_true",
        help="Accept SIE responses without Cloud billing; Sol/Luna still use OpenAI",
    )
    return p


def execute(args):
    if args.command == "fetch":
        core.require(bool(args.manifest_url) == bool(args.manifest_sha256), "Supply both manifest URL and SHA, or use the pinned release")
        if args.manifest_url:
            url, digest = args.manifest_url, args.manifest_sha256
        else:
            release = core.read(args.release)
            core.require(release["format"] == "train-a-router-release-v1", "Unknown artifact release format")
            url, digest = release["manifest_url"], release["manifest_sha256"]
        return evidence.fetch(url, digest, args.output, groups=args.group or ("core",))
    if args.command == "show":
        return show_report(args.report, args.target_gap_pp)
    if args.command == "frontier":
        core.require(args.output.suffix == ".html", "Use an .html output path")
        json_path = args.output.with_suffix(".json")
        core.require(not args.output.exists() and not json_path.exists(), "Preserve previous frontier outputs")
        run = args.run or args.data / "reference"
        report = frontier_analysis.build(args.data, run, include_test=args.include_test)
        result = frontier.write_view(report, args.output, run)
        core.write(json_path, report)
        return {"html": str(args.output), "data": str(json_path), "view": result}
    if args.command == "encode":
        import encode  # Optional service dependencies are unnecessary for recorded CPU replay.

        return encode.collect(
            args.data,
            args.output,
            batch_size=args.batch_size,
            sie_base_url=args.sie_base_url,
            self_hosted=args.self_hosted,
            retry_unknown=args.retry_unknown,
            max_batches=args.max_batches,
        )
    if args.command == "answer":
        contract = {"type": args.answer_type}
        if args.answer_type == "choice":
            core.require(bool(args.choices), "Supply --choices A B C ... for multiple-choice questions")
            contract["allowed"] = args.choices
        import live  # Optional service dependencies are unnecessary for recorded CPU replay.

        return live.answer(
            args.run,
            args.question,
            args.output,
            method=args.method,
            target=args.target_gap_pp,
            candidate_id=args.candidate_id,
            contract=contract,
            sie_base_url=args.sie_base_url,
            self_hosted=args.self_hosted,
            device=args.device,
        )
    data = args.data
    if args.command == "prepare":
        acquisition = args.acquisition_protocol_sha256
        if acquisition is None:
            reference, _ = core.checked_plan(data / "reference")
            acquisition = reference["acquisition_protocol_sha256"]
        plan = core.prepare(data / "source", data / "pair.json", args.run, acquisition)
        return {"plan_sha256": core.sha(args.run / "plan.json"), "training_rows": len(plan["populations"]["train"])}
    if args.command == "fit-embedding":
        result = core.train_embedding(args.run, data / "recordings/grades-train.json", args.features or data / "features")
        return {"selected_penalty": result["fit"]["penalty"], "model_sha256": core.sha(args.run / "embedding.json")}
    if args.command == "fit-native":
        checkpoint = args.initial_checkpoint or data / "reference/native/base"
        result = paired_native.train(args.run, data / "recordings/grades-train.json", checkpoint)
        return {
            "selected_epoch": result["selected_epoch"],
            "trained_sha256": core.sha(args.run / "native/trained.json"),
        }
    if args.command == "fit-topic":
        output = args.run / "topic-scores.json"
        paired_topic.fit_and_score(args.run, data / "recordings/grades-train.json", output)
        return {"scores_sha256": core.sha(output)}
    if args.command == "reuse-controls":
        return reproduction.reuse_controls(data, args.run)
    if args.command == "score":
        output = args.run / f"{args.method}-scores.json"
        if args.method == "embedding":
            core.score_embedding(args.run, args.features or data / "features", output)
        else:
            core.require(args.features is None, "--features applies only to --method embedding")
            paired_native.score_native(args.run, output, device=args.device)
        return {"scores_sha256": core.sha(output), "outcomes_accessed": False}
    if args.command == "calibrate":
        output = args.run / "frozen"
        result = reproduction.calibrate(
            args.run,
            data,
            output,
        )
        return {
            "frozen_routing_sha256": core.sha(output / "frozen-final-routes.json"),
            "evaluation_mode": result["evaluation_mode"],
        }
    if args.command == "evaluate-refit":
        return reproduction.evaluate(data, args.run, args.output)
    routing = args.frozen_routing or data / "reference/frozen/frozen-final-routes.json"
    expected = None
    if args.command == "replay":
        expected = args.expected_report or data / "reference/report.json"
    return evaluate_recorded(routing, data / "recordings/grades-final.json", args.output, expected_report=expected)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        result = execute(args)
    except (FileNotFoundError, ValueError) as error:
        p.exit(2, f"{args.command}: {error}\n")
    print(result if isinstance(result, str) else json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
