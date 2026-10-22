"""Measure resident two-question inference on a source-only development sample."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import math
import os
import statistics
import time
from pathlib import Path

import recipe as core


def select(rows, count=90):
    candidates = [row for row in rows if row["role"] == "development"]
    ordered = sorted(candidates, key=lambda row: core.text_sha("bbeh-native-timing89:" + row["id"]))
    core.require(len(ordered) >= 2, "Need a separate warmup and timed request")
    count = min(count, len(ordered) - 1)
    return ordered[:count], ordered[count]


def summarize(records):
    values = sorted(row["elapsed_seconds"] for row in records)
    core.require(values and all(math.isfinite(v) and v > 0 for v in values), "Invalid timed durations")

    def quantile(q):
        index = q * (len(values) - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)

    return {
        "requests": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": quantile(0.90),
        "p95_seconds": quantile(0.95),
        "reciprocal_mean_service_seconds": len(values) / sum(values),
        "by_task": {
            task: {
                "n": sum(r["task"] == task for r in records),
                "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in records if r["task"] == task),
            }
            for task in sorted({r["task"] for r in records})
        },
    }


def measure(study, output, *, variant, device, resource_rate_usd_per_second=None, resource_rate_source=None):
    study, output = Path(study), Path(output)
    core.require(variant in ("base", "lora") and device in ("cuda", "cpu"), "Unsupported timing variant/device")
    if resource_rate_usd_per_second is not None:
        core.require(
            math.isfinite(resource_rate_usd_per_second) and resource_rate_usd_per_second > 0 and resource_rate_source,
            "Need an explicit priced resource and source",
        )
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(TORCH_COMPILE_DISABLE="1", TORCHDYNAMO_DISABLE="1", TOKENIZERS_PARALLELISM="false")
    import laya.train as lt
    import numpy as np
    import torch

    import bbeh_native as native

    torch.set_num_threads(8)
    plan, rows = core.checked_plan(study)
    trained = core.read(study / "native/trained.json")
    core.require(trained["plan_sha256"] == core.sha(study / "plan.json"), "Training plan changed")
    sample, warmup = select(rows)
    protocol = {
        "rule": "First90 development IDs by SHA256('bbeh-native-timing89:'+id); nextID is separate warmup; all if fewer than91",
        "ids": [row["id"] for row in sample],
        "warmup_id": warmup["id"],
        "request_sha256": {row["id"]: core.text_sha(row["request"]) for row in sample + [warmup]},
        "source_plan_sha256": plan["source_plan_sha256"],
        "outcomes_read": False,
        "policy_computation": "Signed quality advantage and expected USD; lambda0 is timing-only, not a selected policy",
    }
    core.write(output / "sample-protocol.json", protocol)
    checkpoint = study / "native" / variant
    verification_start = time.monotonic()
    core.require(
        native.checkpoint_files(checkpoint) == trained["checkpoints"][variant]["files_sha256"], "Checkpoint changed"
    )
    verification_seconds = time.monotonic() - verification_start
    load_start = time.monotonic()
    model, tokenizer, saved = native.load_checkpoint_preserving(lt, checkpoint)
    settings = plan["native"]
    maximum, head = settings["max_len"], settings["head_max_len"]
    core.require(saved["max_len"] == maximum and saved["head_max_len"] == head, "Export context differs")
    model.encoder.config.reference_compile = False
    execution_device = torch.device(device)
    model.to(execution_device).eval()
    core.require({str(p.dtype) for p in model.parameters()} == {"torch.float32"}, "Expected FP32 resident inference")
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.monotonic() - load_start

    def timed(row):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        items = native.encode_rows(lt, tokenizer, [row], plan["pair"], settings)
        values = lt.calibration_records(model, tokenizer, items, execution_device, maximum, head, batch_size=2)
        margins = native.margins_from_records(values, np.zeros((1, 2)))
        probabilities = np.exp(-np.logaddexp(0, -margins))
        expected = core.expected_usd(probabilities, trained["conditional_cost_usd"])
        decision = bool(probabilities[0, 1] - probabilities[0, 0] >= 0)
        if device == "cuda":
            torch.cuda.synchronize()
        return {
            "id": row["id"],
            "task": row["task"],
            "elapsed_seconds": time.perf_counter() - started,
            "probabilities": probabilities[0].tolist(),
            "expected_usd": expected[0].tolist(),
            "timing_only_strong": decision,
            "state_tokens": len(items[0]["state_ids"]),
        }

    first = timed(warmup)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    records = [timed(row) for row in sample]
    summary = summarize(records)
    report = {
        "format": "bbeh-resident-native-timing-v1",
        "variant": variant,
        "device": device,
        "plan_sha256": core.sha(study / "plan.json"),
        "trained_sha256": core.sha(study / "native/trained.json"),
        "benchmark_sha256": core.sha(__file__),
        "sample_protocol_sha256": core.sha(output / "sample-protocol.json"),
        "checkpoint_verification_seconds": verification_seconds,
        "model_load_seconds": load_seconds,
        "warmup": first,
        "records": records,
        "summary": summary,
        "native_context": settings,
        "runtime_versions": {name: importlib.metadata.version(name) for name in settings["versions"]},
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "timing_includes": "Tokenization/full-context validation, both resident model questions, sigmoid probabilities and expected-cost/decision arithmetic",
        "timing_excludes": "Checkpoint verification/loading, startup, warmup, artifact I/O, HTTP/network, queueing and idle capacity",
        "limitations": "Sequential resident service measurements; reciprocal mean is not measured concurrent throughput",
        "resource_rate_usd_per_second": resource_rate_usd_per_second,
        "resource_rate_source": resource_rate_source,
        "busy_resource_usd_per_10000": None
        if resource_rate_usd_per_second is None
        else summary["mean_seconds"] * 10000 * resource_rate_usd_per_second,
        "cost_scope": "Timing-sample-based busy-resource projection only when a priced rate is supplied; not observed billing or full deployment cost",
    }
    if device == "cuda":
        report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()
        )
    core.write(output / "report.json", report)
    model, tokenizer = None, None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("base", "lora"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--resource-rate-usd-per-second", type=float)
    parser.add_argument("--resource-rate-source")
    args = parser.parse_args()
    report = measure(
        args.study,
        args.output,
        variant=args.variant,
        device=args.device,
        resource_rate_usd_per_second=args.resource_rate_usd_per_second,
        resource_rate_source=args.resource_rate_source,
    )
    print(report["summary"])


if __name__ == "__main__":
    main()
