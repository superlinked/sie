"""Short reader commands around the pinned scientific recipe implementations."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cost_reader
import evidence
from embedding import recipes as core
from embedding.cost_iteration import refit as cost_refit
from evaluation import frontier

ROOT = Path(__file__).resolve().parent
ROLES = ("calibration", "final")


def invoke(script, *arguments):
    """Run an optional component with UTF-8 defaults and its original sibling import layout."""
    script = ROOT / script
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(ROOT), str(script.parent)))}
    subprocess.run([sys.executable, "-X", "utf8", str(script), *map(str, arguments)], env=environment, check=True)


def fetch(output, groups, release_path=None):
    release = core.read(release_path or ROOT / "release.json")
    core.require(release["format"] == "train-a-router-release-v1", "Unknown artifact release")
    return evidence.fetch(release["manifest_url"], release["manifest_sha256"], output, groups=groups)


def checked_data(data):
    data = Path(data).resolve()
    release = core.read(ROOT / "release.json")
    evidence.verify(data, manifest_sha256=release["manifest_sha256"], groups=("core",))
    return data


def prepare(data, run):
    data, run = checked_data(data), Path(run).resolve()
    core.require(not run.exists(), "Use a new run directory")
    run.mkdir(parents=True)
    core.prepare(data / "source", ROOT / "pair.json", run / "embedding")
    original = core.read(data / "reference/native/plan.json")
    invoke("native/recipe.py", "prepare", "--source", data / "source", "--pair", ROOT / "pair.json",
           "--lengths", data / "reference/native/lengths.json", "--output", run / "native",
           "--acquisition-protocol-sha256", original["acquisition_protocol_sha256"])
    (run / "scores").mkdir()
    core.write(run / "reproduction.json", {
        "format": "bbeh-public-reproduction-v1",
        "evaluation_mode": "post_publication_reproduction",
        "test_answers_already_published": True,
        "new_blind_evaluation": False,
        "manifest_sha256": core.sha(data / "manifest.json"),
        "artifact_directory": str(data),
        "reference_frozen_sha256": core.sha(data / "reference/frozen-frontiers.json"),
        "source_plan_sha256": original["source_plan_sha256"],
        "purpose": "Repeat the recipe using the published training and held-out answers.",
    })
    return {"run": str(run), "prepared": True, "model_calls": False}


def checked_run(data, run):
    data, run = checked_data(data), Path(run).resolve()
    record = core.read(run / "reproduction.json")
    core.require(record["format"] == "bbeh-public-reproduction-v1"
                 and record["manifest_sha256"] == core.sha(data / "manifest.json")
                 and record["reference_frozen_sha256"] == core.sha(data / "reference/frozen-frontiers.json")
                 and record["new_blind_evaluation"] is False, "Run belongs to another published dataset")
    core.checked_plan(run / "embedding")
    return data, run


def fit_embedding(data, run, features=None):
    data, run = checked_run(data, run)
    model = core.fit(run / "embedding", data / "grades/grades-train.json",
                     data / "grades/grades-development.json", features or data / "features")
    return {"selected": model["selected"], "timing": model["timing"], "run": str(run)}


def fit_native(data, run, *, recipe="reference"):
    core.require(recipe in {"reference", "fast"}, "Choose the reference or fast training recipe")
    data, run = checked_run(data, run)
    script = "native/fast_train.py" if recipe == "fast" else "native/bbeh_native.py"
    arguments = [] if recipe == "fast" else ["--phase", "train"]
    invoke(script, *arguments, "--run", run / "native",
           "--training-grades", data / "grades/grades-train.json",
           "--development-grades", data / "grades/grades-development.json",
           "--initial-checkpoint", data / "reference/native/native/base")
    trained = core.read(run / "native/native/trained.json")
    return {"recipe": recipe, "selected_epoch": trained["selected_epoch"],
            "elapsed_seconds": trained["elapsed_seconds"]}


def base(data, run, device):
    data, run = checked_run(data, run)
    output = run / "unchanged-base"
    if not output.exists():
        invoke("native/base.py", "prepare", "--run", run / "native",
               "--training-grades", data / "grades/grades-train.json",
               "--development-grades", data / "grades/grades-development.json",
               "--initial-checkpoint", data / "reference/native/native/base", "--output", output)
    for role in ROLES:
        invoke("native/base.py", "score", "--run", run / "native", "--base", output,
               "--split", role, "--device", device, "--output", run / f"scores/base-{role}.json")
    return {"run": str(run), "method": "base", "device": device, "held_out_outcomes_accessed": False}


def score(data, run, method, device="cuda", features=None):
    data, run = checked_run(data, run)
    if method == "embedding":
        for role in ROLES:
            target = run / f"scores/embedding-{role}.json"
            core.require(not target.exists(), "Preserve previous scores")
            core.score(run / "embedding", features or data / "features", role,
                       run / f"embedding/scores-{role}.json")
            shutil.copyfile(run / f"embedding/scores-{role}.json", target)
    else:
        invoke("native/reader.py", "--run", run / "native", "--device", device,
               "--output", run / "native/native/input-scores.json")
        for role in ROLES:
            target = run / f"scores/native-{role}.json"
            core.require(not target.exists(), "Preserve previous scores")
            shutil.copyfile(run / f"native/native/scores-{role}.json", target)
    return {"run": str(run), "method": method, "held_out_outcomes_accessed": False}


def reuse_native(data, run):
    data, run = checked_run(data, run)
    reference = data / "reference"
    core.require(core.sha(run / "native/plan.json") == core.sha(reference / "native/plan.json"),
                 "Reference native scores require the identical recipe plan")
    for role in ROLES:
        target = run / f"scores/native-{role}.json"
        core.require(not target.exists(), "Existing native scores; preserve them")
        shutil.copyfile(reference / f"scores/native-{role}.json", target)
    core.write(run / "native-comparison.json", {
        "fitting_performed": False,
        "scores_sha256": {role: core.sha(run / f"scores/native-{role}.json") for role in ROLES},
        "purpose": "Reuse supplied Laya comparisons during an embedding-only refit.",
    })
    return {"run": str(run), "native_predictions_reused": True}


def fit_cost(data, run):
    data, run = checked_run(data, run)
    receipt = cost_reader.fit_score(run / "embedding", data / "grades/grades-train.json",
        data / "grades/grades-development.json", run / "cost")
    for role in ROLES:
        cost_refit.augment_files(run / f"scores/native-{role}.json", run / f"cost/costs-{role}.json",
            run / f"cost/native-scores-{role}.json", run / "cost/native-combined-model.json")
    return {"run": str(run), "cost_model": str(run / "cost/cost-model.json"), "alpha": receipt["alpha"]}


def calibrate(data, run):
    """Keep post-publication artifacts explicitly marked as a reproduction."""
    data, run = checked_run(data, run)
    internal = run / "internal"
    core.require(not internal.exists(), "Preserve the existing calibration; use a fresh run")
    internal.mkdir()
    frontier.calibrate(run / "embedding", data / "grades/grades-calibration.json",
        [run / "cost/scores-calibration.json", run / "cost/native-scores-calibration.json"],
        data / "fees.json", internal / "calibration-frontiers.json")
    frontier.freeze(run / "embedding", internal / "calibration-frontiers.json",
        [run / "cost/scores-final.json", run / "cost/native-scores-final.json"],
        data / "fees.json", internal / "frozen-frontiers.json")
    core.write(run / "calibration.json", {
        "format": "bbeh-public-calibration-v1", "evaluation_mode": "post_publication_reproduction",
        "new_blind_evaluation": False, "final_outcomes_read_by_calibrator": False,
        "calibration_sha256": core.sha(internal / "calibration-frontiers.json"),
        "frozen_sha256": core.sha(internal / "frozen-frontiers.json"),
        "reproduction_sha256": core.sha(run / "reproduction.json"),
        "router_fee_scenario": "Published SIE Cloud singleton fees; self-hosting is unpriced.",
    })
    return {"run": str(run), "calibration": str(run / "calibration.json"), "new_model_calls": 0}


def evaluate_refit(data, run, output):
    data, run = checked_run(data, run)
    output, internal = Path(output).resolve(), run / "internal"
    core.require(not output.exists(), "Use a new report directory")
    envelope = core.read(run / "calibration.json")
    core.require(envelope["new_blind_evaluation"] is False
        and envelope["reproduction_sha256"] == core.sha(run / "reproduction.json")
        and envelope["frozen_sha256"] == core.sha(internal / "frozen-frontiers.json")
        and envelope["calibration_sha256"] == core.sha(internal / "calibration-frontiers.json"),
        "Public reproduction lineage changed")
    output.mkdir(parents=True)
    actual = frontier.evaluate(run / "embedding", internal / "frozen-frontiers.json",
                               data / "grades/grades-final.json", output / "final-report.json")
    invoke("evaluation/source_sensitivity.py", "--run", run / "embedding",
           "--frozen", internal / "frozen-frontiers.json", "--grades", data / "grades/grades-final.json",
           "--primary", output / "final-report.json", "--specification", ROOT / "source-format-sensitivity.json",
           "--output", output / "source-format-sensitivity.json")
    core.write(output / "reproduction.json", {
        **envelope, "final_report_sha256": core.sha(output / "final-report.json"),
        "interpretation": "A refit evaluated on already published test answers, not a new blind test.",
    })
    invoke("evaluation/report.py", "--calibration", internal / "calibration-frontiers.json",
           "--frozen", internal / "frozen-frontiers.json", "--final-report", output / "final-report.json",
           "--output", output / "explorer", "--reproduction")
    return {"report": str(output / "final-report.json"), "open": str(output / "explorer/index.html"),
            "questions": actual["n"], "new_blind_evaluation": False, "new_model_calls": 0}


def same_values(expected, actual):
    """Allow floating-point roundoff across CPUs without ignoring altered results."""
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(same_values(expected[k], actual[k]) for k in expected)
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(same_values(a, b) for a, b in zip(expected, actual, strict=True))
    if isinstance(expected, float):
        return math.isfinite(expected) and math.isfinite(actual) and math.isclose(
            expected, actual, rel_tol=1e-10, abs_tol=1e-10)
    return expected == actual


def replay(data, output):
    data, output = checked_data(data), Path(output).resolve()
    core.require(not output.exists(), "Use a new report directory")
    output.mkdir(parents=True)
    reference = data / "reference"
    actual = frontier.evaluate(reference / "embedding", reference / "frozen-frontiers.json",
                               data / "grades/grades-final.json", output / "final-report.json")
    core.require(same_values(core.read(reference / "final-report.json"), actual),
                 "Replay differs from the published report; keep the output for inspection")
    invoke("evaluation/source_sensitivity.py", "--run", reference / "embedding",
           "--frozen", reference / "frozen-frontiers.json", "--grades", data / "grades/grades-final.json",
           "--primary", output / "final-report.json", "--specification", ROOT / "source-format-sensitivity.json",
           "--output", output / "source-format-sensitivity.json")
    invoke("evaluation/report.py", "--calibration", reference / "calibration-frontiers.json",
           "--frozen", reference / "frozen-frontiers.json", "--final-report", output / "final-report.json",
           "--output", output / "explorer")
    return {"report": str(output / "final-report.json"), "open": str(output / "explorer/index.html"),
            "questions": actual["n"], "matches_published_result": True, "new_model_calls": 0}
