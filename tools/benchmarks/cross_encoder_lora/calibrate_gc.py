"""Bounded model-free calibration of the fixed GC observer allowance, not serving performance."""

from __future__ import annotations

import argparse
import gc
import math
import platform
import statistics
import sys
import time
from array import array
from pathlib import Path
from typing import Any

from tools.benchmarks.cross_encoder_lora.evidence import atomic_json, digest
from tools.benchmarks.cross_encoder_lora.process_metrics import CALLBACK_TAIL_ALLOWANCE_NS, GCRecorder
from tools.benchmarks.cross_encoder_lora.protocol import percentile, require

BLOCKS = 6
COLLECTIONS = 10_000
DIRECT_PAIRS = 500
T95_DF5 = 2.0150483733330233


def bound(values: list[float]) -> dict[str, Any]:
    require(
        len(values) == BLOCKS and all(math.isfinite(value) for value in values), "six finite calibration units required"
    )
    mean = statistics.mean(values)
    upper = mean + T95_DF5 * statistics.stdev(values) / math.sqrt(BLOCKS)
    return {
        "block_mean_residual_ns": values,
        "mean_ns": mean,
        "one_sided_95_upper_ns": upper,
        "allowance_ns": CALLBACK_TAIL_ALLOWANCE_NS,
        "passed": max(values) <= CALLBACK_TAIL_ALLOWANCE_NS and upper <= CALLBACK_TAIL_ALLOWANCE_NS,
    }


def collect_arm(observed: bool, count: int, record: dict[str, Any] | None = None) -> dict[str, Any]:
    if record is None:
        record = {}
    record.update(observed=observed, passed=False, completed=0)
    durations = array("Q", [0]) * count
    collected = array("Q", [0]) * count
    recorder = GCRecorder(capacity=2 * count + 64)
    thresholds, prior = gc.get_threshold(), tuple(gc.callbacks)
    require(gc.isenabled(), "normal GC must remain enabled")
    if observed:
        recorder.start()
    try:
        for index in range(count):
            started = time.perf_counter_ns()
            collected[index] = gc.collect(0)
            durations[index] = time.perf_counter_ns() - started
            record["completed"] = index + 1
    finally:
        try:
            if observed:
                recorder.close()
        finally:
            rows = [list(recorder.buffer[i * 6 : (i + 1) * 6]) for i in range(recorder.count)]
            record.update(
                duration_ns=list(durations),
                collected=list(collected),
                records=rows,
                recorder_errors=recorder.lifecycle_errors,
                invalid=recorder.invalid,
                overflow=recorder.overflow,
            )
    require(gc.isenabled() and gc.get_threshold() == thresholds, "GC configuration changed")
    require(
        len(gc.callbacks) == len(prior) and all(a is b for a, b in zip(gc.callbacks, prior, strict=True)),
        "GC callback identities changed",
    )
    if observed:
        require(
            not recorder.lifecycle_errors and not recorder.invalid and not recorder.overflow,
            "invalid calibration observer",
        )
        require(len(rows) == 2 * count, "unexpected extra/missing calibration GC callbacks")
        require(
            all(row[1] == index % 2 and row[2] == 0 for index, row in enumerate(rows)), "calibration GC pairing differs"
        )
    record["passed"] = True
    return record


def direct(generation: int, pairs: int) -> dict[str, Any]:
    # This is a diagnostic direct call, not an installed GC callback. It covers
    # all field paths while the real-dispatch pairs above exercise CPython GC.
    recorder = GCRecorder(capacity=2 * pairs)
    durations = array("Q", [0]) * (2 * pairs)
    info = {"generation": generation, "collected": 101, "uncollectable": 3}
    callback = recorder._record
    for index in range(2 * pairs):
        phase = "start" if index % 2 == 0 else "stop"
        started = time.perf_counter_ns()
        callback(phase, info)
        durations[index] = time.perf_counter_ns() - started
    self_ns = [recorder.buffer[index * 6 + 5] for index in range(2 * pairs)]
    residual: list[float] = [elapsed - cost for elapsed, cost in zip(durations, self_ns, strict=True)]
    return {
        "generation": generation,
        "info": info,
        "external_ns": list(durations),
        "self_ns": self_ns,
        "residual_ns": residual,
        "mean_residual_ns": statistics.mean(residual),
        "p99_residual_ns": percentile(residual, 0.99),
        "maximum_residual_ns": max(residual),
    }


def run(result: dict[str, Any]) -> None:
    prior, thresholds = tuple(gc.callbacks), gc.get_threshold()
    require(gc.isenabled(), "normal GC must remain enabled")
    blocks: list[dict[str, Any]] = []
    result["blocks"] = blocks
    for index in range(BLOCKS):
        order = (False, True) if index % 2 == 0 else (True, False)
        block: dict[str, Any] = {"block": index, "order": list(order), "arms": {}, "direct": []}
        blocks.append(block)
        arms = block["arms"]
        for observed in order:
            arms[str(observed)] = {}
            collect_arm(observed, COLLECTIONS, arms[str(observed)])
        baseline, observed = arms["False"], arms["True"]
        self_mean = statistics.mean(row[5] for row in observed["records"])
        # Pair means are the independent units. The baseline cost, including
        # the identical loop/timer/storage, is subtracted only in this diagnostic.
        # Signed residuals are retained; none are clipped or selected away.
        residual = (statistics.mean(observed["duration_ns"]) - statistics.mean(baseline["duration_ns"])) / 2 - self_mean
        unsubtracted = [
            (elapsed - observed["records"][2 * j][5] - observed["records"][2 * j + 1][5]) / 2
            for j, elapsed in enumerate(observed["duration_ns"])
        ]
        block.update(
            {
                "residual_ns_per_callback": residual,
                "unsubtracted_residual_p99_ns": percentile(unsubtracted, 0.99),
                "unsubtracted_residual_maximum_ns": max(unsubtracted),
                "direct": [direct(generation, DIRECT_PAIRS) for generation in range(3)],
            }
        )
    gates = {"actual_gc_dispatch": bound([row["residual_ns_per_callback"] for row in blocks])}
    for generation in range(3):
        gates[f"direct_gen{generation}"] = bound([row["direct"][generation]["mean_residual_ns"] for row in blocks])
    require(gc.isenabled() and gc.get_threshold() == thresholds, "GC configuration changed")
    require(
        len(gc.callbacks) == len(prior) and all(a is b for a, b in zip(gc.callbacks, prior, strict=True)),
        "GC callback identities changed",
    )
    result.update(
        {
            "qualification": False,
            "passed": all(value["passed"] for value in gates.values()),
            "python": sys.version,
            "platform": platform.platform(),
            "gc_thresholds": list(thresholds),
            "source_sha256": digest(Path(__file__)),
            "observer_sha256": digest(Path(__file__).with_name("process_metrics.py")),
            "blocks": blocks,
            "gates": gates,
            "interpretation": "Local mean-overhead estimator calibration only. The fixed10us allowance is not a worst-case event bound or a guarantee for another host. All samples/outliers retained; no benchmark subtraction or threshold adjustment.",
            "runtime_budget": "Combined server+client estimated observer cost<=0.001 in every real measured phase, unchanged.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "calibration output must be new")
    result: dict[str, Any] = {
        "qualification": False,
        "passed": False,
        "source_sha256": digest(Path(__file__)),
        "observer_sha256": digest(Path(__file__).with_name("process_metrics.py")),
    }
    try:
        run(result)
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        atomic_json(args.output, result)
    print({"passed": result["passed"], "qualification": False, "output_sha256": digest(args.output)}, flush=True)
    require(result["passed"], "fixed GC allowance calibration failed; retain evidence and investigate")


if __name__ == "__main__":
    main()
