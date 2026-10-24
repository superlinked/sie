"""Train routers and compare money saved with answer quality."""

import argparse
import json
from pathlib import Path

import calibration_view
import workflow
from embedding import recipes as core
from reader_runtime import ensure_utf8


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("fetch", help="Download and verify the pinned public data; add weights for Laya")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--group", choices=("core", "weights", "recordings"), action="append")
    for name, help_text in (
        ("prepare", "Create a fresh reproduction run from the supplied dataset"),
        ("base", "Apply unchanged Laya before training; CPU or CUDA"),
        ("fit-native", "Select and refit LoRA on an NVIDIA GPU"),
        ("fit-embedding", "Fit neighbors, small linear heads and simple controls on CPU"),
        ("score", "Predict correctness on calibration/test inputs without their answers"),
        ("fit-cost", "Fit the small request-cost predictor and add its routing variants"),
        ("reuse-native", "Reuse supplied Laya comparisons during a CPU-only embedding exercise"),
        ("calibrate", "Save the full calibration frontier and freeze input-only test decisions"),
        ("evaluate-refit", "Check your frozen policies against published test answers"),
    ):
        c = commands.add_parser(name, help=help_text)
        c.add_argument("--data", type=Path, required=True)
        c.add_argument("--run", type=Path, required=True)
        if name in {"base", "score"}:
            c.add_argument("--device", choices=("cpu", "cuda"), default="cpu" if name == "base" else "cuda")
        if name in {"fit-embedding", "score"}:
            c.add_argument("--features", type=Path, help="Fresh features; defaults to the verified supplied vectors")
        if name == "score":
            c.add_argument("--method", choices=("embedding", "native"), required=True)
        if name == "evaluate-refit":
            c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("replay", help="Recompute and verify the published final results without model calls")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("frontier", help="Explore calibration data before looking at test results")
    c.add_argument("--run", type=Path, required=True, help="Your run, or data/reference for supplied routers")
    c.add_argument("--output", type=Path, required=True)
    c = commands.add_parser("answer", help="Route one new question using a saved calibration candidate")
    c.add_argument("--run", type=Path, required=True, help="Your run, or data/reference for supplied routers")
    c.add_argument("--method", required=True)
    c.add_argument("--candidate-id", type=int, required=True)
    c.add_argument("--question", required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    c.add_argument("--sie-base-url")
    c.add_argument("--self-hosted", action="store_true")
    c = commands.add_parser("encode", help="Regenerate question embeddings through Cloud or your own SIE server")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--budget-usd", default="2.00", help="Ceiling for Cloud calls, including unresolved attempts")
    c.add_argument("--sie-base-url")
    c.add_argument("--self-hosted", action="store_true")
    c = commands.add_parser("collect", help="Optional new Sol/Luna paired answer collection")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--phase", choices=("learn", "calibration", "final"), required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--budget-usd", required=True)
    c.add_argument("--frozen", type=Path)
    c.add_argument("--execute", action="store_true", help="Make paid calls; otherwise show the plan")
    return p


def execute(a):
    if a.command == "fetch":
        return workflow.fetch(a.output, a.group or ["core"])
    if a.command == "prepare":
        return workflow.prepare(a.data, a.run)
    if a.command == "base":
        return workflow.base(a.data, a.run, a.device)
    if a.command == "fit-native":
        return workflow.fit_native(a.data, a.run)
    if a.command == "fit-embedding":
        return workflow.fit_embedding(a.data, a.run, a.features)
    if a.command == "score":
        return workflow.score(a.data, a.run, a.method, a.device, a.features)
    if a.command == "fit-cost":
        return workflow.fit_cost(a.data, a.run)
    if a.command == "reuse-native":
        return workflow.reuse_native(a.data, a.run)
    if a.command == "calibrate":
        return workflow.calibrate(a.data, a.run)
    if a.command == "evaluate-refit":
        return workflow.evaluate_refit(a.data, a.run, a.output)
    if a.command == "replay":
        return workflow.replay(a.data, a.output)
    if a.command == "frontier":
        path = a.run / "internal/calibration-frontiers.json"
        if not path.exists():
            path = a.run / "calibration-frontiers.json"
        return calibration_view.render(path, a.output, reproduction=(a.run / "reproduction.json").exists())
    if a.command == "answer":
        import live  # Optional provider dependencies.

        return live.answer(a.run, a.question, a.output, method=a.method, candidate_id=a.candidate_id,
            device=a.device, base_url=a.sie_base_url, self_hosted=a.self_hosted)
    if a.command == "encode":
        data = workflow.checked_data(a.data)
        args = ["--source", data / "source", "--output", a.output, "--execute"]
        if a.self_hosted:
            core.require(a.sie_base_url, "Supply --sie-base-url for your server")
            args.extend(["--self-hosted", "--sie-base-url", a.sie_base_url])
        else:
            core.require(a.sie_base_url is None, "Use --self-hosted with a custom endpoint")
            args.extend(["--budget-usd", a.budget_usd])
        workflow.invoke("embedding/encode.py", *args)
        return {"output": str(a.output), "features": str(a.output / "features")}
    if a.command == "collect":
        data = workflow.checked_data(a.data)
        args = ["--source", data / "source", "--pair", workflow.ROOT / "pair.json", "--phase", a.phase,
                "--output", a.output, "--budget-usd", a.budget_usd]
        if a.frozen:
            args.extend(["--frozen", a.frozen])
        if a.execute:
            args.append("--execute")
        workflow.invoke("collect_answers.py", *args)
        return {"output": str(a.output), "paid_calls_enabled": a.execute}
    raise ValueError("Unknown command")


if __name__ == "__main__":
    ensure_utf8()
    print(json.dumps(execute(parser().parse_args()), indent=2, allow_nan=False))
