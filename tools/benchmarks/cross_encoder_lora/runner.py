"""Run the frozen six-block experiment in eighteen owned fresh processes."""

# Only the current Python executable and this checkout's benchmark module run.
# ruff: noqa: S603

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest, fingerprint, vector_delta
from tools.benchmarks.cross_encoder_lora.protocol import (
    BASE_REVISION,
    MAX_EXPERIMENT_SECONDS,
    MAX_TRIAL_SECONDS,
    ORDERS,
    RESTORE_ATOL,
    compare,
    require,
)

ROOT = Path(__file__).resolve().parents[3]
WAIT_GRACE_SECONDS = 45
TERMINATE_GRACE_SECONDS = 35
KILL_GRACE_SECONDS = 10


def harness_manifest() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(directory)): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".md"}
        and not any(part in {"_inputs", "_evidence", "__pycache__"} for part in path.relative_to(directory).parts)
    }
    return {"files": files, "file_manifest_sha256": fingerprint(files)}


def source_environment(source: Path) -> dict[str, str]:
    # The harness stays fixed; both server and SDK imports come from the
    # selected immutable archive. Do not run uv sync per arm.
    paths = [source / f"packages/{package}/src" for package in ("sie_server", "sie_sdk", "sie_config", "sie_bundles")]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(path) for path in [*paths, ROOT]),
        "TOKENIZERS_PARALLELISM": "false",
        "SIE_BUNDLE": "default",
    }


def extract(inputs: Path, output: Path) -> dict[str, Any]:
    manifest = json.loads((inputs / "manifest.json").read_text())
    require(manifest["schema"] == 1 and manifest["dependency_lock_identical"], "unexpected source manifest")
    require(manifest["sources"]["B"]["commit"] == BASE_REVISION, "baseline differs from protocol")
    for name, record in manifest["sources"].items():
        require(name in {"B", "candidate"}, "unknown source arm")
        path = inputs / record["archive"]
        require(path.resolve().is_relative_to(inputs.resolve()), "archive path escapes inputs")
        require(digest(path) == record["sha256"], "source archive hash differs")
        target = output / name
        target.mkdir(parents=True)
        with tarfile.open(path) as archive:
            archive.extractall(target, filter="data")
    require(
        (output / "B/uv.lock").read_bytes() == (output / "candidate/uv.lock").read_bytes(), "extracted locks differ"
    )
    return manifest


def cross_checks(trials: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for pair in range(len(ORDERS)):
        rows = {trial["arm"]: trial for trial in trials if trial["pair"] == pair}
        baseline = rows["B"]
        for arm in ("N", "R"):
            candidate = rows[arm]
            require(
                candidate["environment"]["versions"] == baseline["environment"]["versions"],
                "libraries changed between arms",
            )
            require(
                candidate["environment"]["installed_distributions_sha256"]
                == baseline["environment"]["installed_distributions_sha256"],
                "installed distributions changed between arms",
            )
            require(candidate["environment"]["device"] == baseline["environment"]["device"], "GPU changed between arms")
            require(
                vector_delta(candidate["base_scores"], baseline["base_scores"]) <= RESTORE_ATOL,
                "cross-process C1 base scores differ",
            )
            require(
                candidate["initial_memory"]["allocated"] == baseline["initial_memory"]["allocated"],
                "unadapted allocated bytes changed",
            )
            for key, oracle in candidate["shape_oracles"].items():
                require(
                    vector_delta(oracle["scores"], baseline["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                    "cross-process shape oracle differs",
                )
            # Actual shape attribution is available only in observed untimed
            # replays. Timed C4/C16 diagnostics are not shape-qualified evidence.
            for phase in candidate["untimed_replays"].values():
                for batch in phase["batches"]:
                    key = fingerprint({"queries": batch["queries"], "documents": batch["documents"]})
                    require(
                        vector_delta(batch["scores"], baseline["shape_oracles"][key]["scores"]) <= RESTORE_ATOL,
                        "candidate untimed SDK replay differs from the matched B shape",
                    )
            delta = candidate["resident_memory"]["allocated"] - baseline["resident_memory"]["allocated"]
            allowance = candidate.get("adapter_storage", {}).get("rounded_storage_bytes", 0)
            require(0 <= delta <= allowance, "resident allocated bytes exceed measured adapter storage allowance")
            results[f"{pair}-{arm}"] = {"allocated_delta": delta, "allowance": allowance, "passed": True}
    return results


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=KILL_GRACE_SECONDS)


def trial_summary(trial: dict[str, Any]) -> dict[str, Any]:
    """Keep the cumulative index independent of the number of raw requests."""
    fields = (
        "pair",
        "arm",
        "passed",
        "shutdown_complete",
        "smoke",
        "pid",
        "process_returncode",
        "source_commit",
        "terminal_record",
        "terminal_record_sha256",
        "error",
        "runner_error",
    )
    result = {field: trial[field] for field in fields if field in trial}
    if "performance" in trial:
        metrics = ("concurrency", "count", "elapsed_s", "mean_ms", "p95_ms", "rps", "actual_timed_batch_shape")
        result["performance"] = {
            concurrency: {field: phase[field] for field in metrics if field in phase}
            for concurrency, phase in trial["performance"].items()
        }
    return result


def load_trial(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Reopen the complete evidence only after checking its recorded identity."""
    relative = f"trial-{record['pair']}-{record['arm']}/trial.json"
    require(record["terminal_record"] == relative, "unexpected terminal record path")
    path = output / relative
    require(path.resolve().is_relative_to(output.resolve()), "terminal record path escapes output")
    require(digest(path) == record["terminal_record_sha256"], "terminal record hash differs")
    trial = json.loads(path.read_text())
    require((trial["pair"], trial["arm"]) == (record["pair"], record["arm"]), "terminal record identity differs")
    for field, value in trial_summary(trial).items():
        if field == "passed" and "runner_error" in record:
            # A process timeout can invalidate an otherwise successful worker
            # record. The runner's failed verdict must survive that override.
            require(record[field] is False, "runner error lost its failed verdict")
        else:
            require(record.get(field) == value, "terminal record summary differs")
    if "runner_error" in record:
        trial["passed"] = False
        trial["runner_error"] = record["runner_error"]
    if "process_returncode" in record:
        trial["process_returncode"] = record["process_returncode"]
    return trial


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = extract(args.inputs.resolve(), args.output / "sources")
    summary: dict[str, Any] = {
        "schema": 2,
        "passed": False,
        "source_manifest": manifest,
        "harness_manifest": harness_manifest(),
        "smoke": args.smoke,
        "orders": ORDERS,
        "trials": [],
        "admissions": [],
    }
    summary_path = args.output / "summary.json"
    started = time.monotonic()
    absolute_deadline = started + args.deadline_s
    summary["runner_clock"] = {
        "monotonic_start_s": started,
        "wall_start_ns": time.time_ns(),
        "deadline_monotonic_s": absolute_deadline,
        "budget_s": args.deadline_s,
    }
    if args.environment_receipt is not None:
        summary["environment_preparation"] = {
            "sha256": digest(args.environment_receipt),
            "receipt": json.loads(args.environment_receipt.read_text()),
        }
    try:
        schedule = [(0, "R")] if args.smoke else [(pair, arm) for pair, order in enumerate(ORDERS) for arm in order]
        for pair, arm in schedule:
            admitted_at = time.monotonic()
            cleanup_deadline = (
                admitted_at + args.trial_deadline_s + WAIT_GRACE_SECONDS + TERMINATE_GRACE_SECONDS + KILL_GRACE_SECONDS
            )
            admitted = cleanup_deadline < absolute_deadline
            summary["admissions"].append(
                {
                    "pair": pair,
                    "arm": arm,
                    "monotonic_s": admitted_at,
                    "deadline_monotonic_s": absolute_deadline,
                    "trial_limit_s": args.trial_deadline_s,
                    "reserve_s": (
                        args.trial_deadline_s + WAIT_GRACE_SECONDS + TERMINATE_GRACE_SECONDS + KILL_GRACE_SECONDS
                    ),
                    "margin_s": absolute_deadline - cleanup_deadline,
                    "admitted": admitted,
                }
            )
            require(admitted, "insufficient time for a complete bounded next trial and cleanup")
            directory = args.output / f"trial-{pair}-{arm}"
            source = (args.output / "sources" / ("B" if arm == "B" else "candidate")).resolve()
            command = [
                sys.executable,
                "-m",
                "tools.benchmarks.cross_encoder_lora.worker",
                "--source",
                str(source),
                "--fixtures",
                str(args.fixtures.resolve()),
                "--output",
                str(directory.resolve()),
                "--arm",
                arm,
                "--pair",
                str(pair),
                "--deadline-s",
                str(args.trial_deadline_s),
            ]
            if args.snapshot:
                command.extend(["--snapshot", str(args.snapshot.resolve())])
            if args.smoke:
                command.append("--smoke")
            print(f"Starting block {pair + 1}, arm {arm}", flush=True)
            process = None
            pending_error: BaseException | None = None
            returncode = None
            with (args.output / f"trial-{pair}-{arm}.log").open("w") as log:
                try:
                    process = subprocess.Popen(
                        command, env=source_environment(source), cwd=source, stdout=log, stderr=subprocess.STDOUT
                    )
                    returncode = process.wait(timeout=args.trial_deadline_s + WAIT_GRACE_SECONDS)
                except BaseException as error:  # noqa: BLE001 -- persist the trial, then re-raise after owned cleanup
                    pending_error = error
                finally:
                    if process is not None:
                        stop(process)
                        returncode = process.returncode
            trial: dict[str, Any]
            if (directory / "trial.json").is_file():
                trial = json.loads((directory / "trial.json").read_text())
            else:
                trial = {
                    "pair": pair,
                    "arm": arm,
                    "passed": False,
                    "shutdown_complete": False,
                    "error": "worker produced no terminal record",
                }
            trial["process_returncode"] = returncode
            trial["source_commit"] = manifest["sources"]["B" if arm == "B" else "candidate"]["commit"]
            trial["terminal_record_sha256"] = (
                digest(directory / "trial.json") if (directory / "trial.json").exists() else None
            )
            trial["terminal_record"] = (
                str((directory / "trial.json").relative_to(args.output))
                if trial["terminal_record_sha256"] is not None
                else None
            )
            if pending_error is not None:
                trial["passed"] = False
                trial["runner_error"] = {"type": type(pending_error).__name__, "message": str(pending_error)}
            summary["trials"].append(trial_summary(trial))
            atomic_json(summary_path, summary)
            if pending_error is not None:
                raise pending_error
            require(
                returncode == 0 and trial["passed"] and trial["shutdown_complete"],
                "trial failed; raw evidence retained",
            )
            print(f"Finished block {pair + 1}, arm {arm}", flush=True)
        if args.smoke:
            summary["passed"] = True
            summary["qualification"] = False
        else:
            complete_trials = [load_trial(args.output, row) for row in summary["trials"]]
            summary["correctness_memory"] = cross_checks(complete_trials)
            summary["statistics"] = compare(complete_trials)
            summary["passed"] = summary["statistics"]["passed"]
            require(summary["passed"], "fixed non-inferiority gates did not pass")
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--environment-receipt",
        type=Path,
        help="Optional operator-owned JSON receipt of environment preparation (no secrets)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=MAX_EXPERIMENT_SECONDS)
    parser.add_argument("--trial-deadline-s", type=float, default=MAX_TRIAL_SECONDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    require(
        0 < args.deadline_s <= MAX_EXPERIMENT_SECONDS and 0 < args.trial_deadline_s <= MAX_TRIAL_SECONDS,
        "deadline outside bounded protocol",
    )
    signal.signal(signal.SIGTERM, terminate)
    run(args)


def terminate(signum: int, _frame: Any) -> None:
    # Enter the ordinary owned-child cleanup and atomic evidence finally paths.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    main()
