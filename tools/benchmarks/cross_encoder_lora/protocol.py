"""Pinned workload and paired gates; README records the longer-window amendment."""

from __future__ import annotations

import math
import statistics
from typing import Any

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
WEIGHTS_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
BASE_REVISION = "d5e024ad3ba6127b1ee176e2c85c794b76077888"
QUERY = {"text": "How do I reset my password?"}
DOCUMENTS = [
    "Use the password reset link on the sign-in page to receive a recovery email.",
    "The capital of France is Paris.",
    "Contact account support if you cannot access your recovery email.",
    "To change a billing address, open the payment settings page.",
]
ITEMS = [{"id": str(i), "text": text} for i, text in enumerate(DOCUMENTS)]
CPU_ORACLE = [3.89416241645813, -11.230953216552734, -10.699657440185547, -9.410902976989746]
ORACLE_ATOL = 0.1  # Existing finite fp32 checkpoint oracle versus the default fp16 CUDA path.
RESTORE_ATOL = 1e-6
EFFECT_FLOOR = 1e-5
EFFECT_DRIFT_MULTIPLIER = 5
ORDERS = ("BNR", "NRB", "RBN", "BRN", "RNB", "NBR")
WARMUP = 30
SEQUENTIAL = 2000
STEADY_SECONDS = 60
CONCURRENCIES = (4, 16)
THREADS = 4
MAX_BATCH_ITEMS = 64
MAX_TRIAL_SECONDS = 270
MAX_EXPERIMENT_SECONDS = 4200
# Measured concurrent phase + request timeout + scheduling margin.
PERFORMANCE_FUTURE_TIMEOUT_S = 120
EXPECTED_VERSIONS = {
    "torch": "2.9.1+cu129",
    "transformers": "4.57.6",
    "sentence-transformers": "5.4.1",
    "peft": "0.19.1",
    "safetensors": "0.7.0",
    "huggingface-hub": "0.36.2",
}
T95_DF5 = 2.0150483733330233
MAX_POINT_RATIO = 1.01
MAX_UPPER_RATIO = 1.03
MAX_PAIR_RATIO = 1.10


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def percentile(values: list[float], fraction: float) -> float:
    require(bool(values) and all(math.isfinite(x) and x > 0 for x in values), "invalid latency samples")
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def paired_gate(ratios: list[float]) -> dict[str, Any]:
    require(len(ratios) == len(ORDERS), "six fresh-process pairs are required")
    require(all(math.isfinite(x) and x > 0 for x in ratios), "invalid paired ratios")
    logs = [math.log(value) for value in ratios]
    mean = statistics.mean(logs)
    point = math.exp(mean)
    upper = math.exp(mean + T95_DF5 * statistics.stdev(logs) / math.sqrt(len(logs)))
    return {
        "paired_ratios": ratios,
        "geometric_ratio": point,
        "one_sided_95_upper": upper,
        "passed": point <= MAX_POINT_RATIO and upper <= MAX_UPPER_RATIO and max(ratios) <= MAX_PAIR_RATIO,
    }


def compare(trials: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(trials) == len(ORDERS) * 3, "all eighteen fresh-process trials are required")
    require(
        [(t["pair"], t["arm"]) for t in trials] == [(i, arm) for i, order in enumerate(ORDERS) for arm in order],
        "trial identities/order differ from the six preregistered triplets",
    )
    require(all(trial["passed"] and trial["shutdown_complete"] for trial in trials), "a trial failed")
    result = {}
    for arm in ("N", "R"):
        metrics = {}
        for concurrency, field in (
            (1, "mean_ms"),
            (1, "p95_ms"),
            (4, "rps"),
            (4, "p95_ms"),
            (16, "rps"),
            (16, "p95_ms"),
        ):
            ratios = []
            for pair in range(len(ORDERS)):
                baseline = next(t for t in trials if t["pair"] == pair and t["arm"] == "B")
                candidate = next(t for t in trials if t["pair"] == pair and t["arm"] == arm)
                b = baseline["performance"][str(concurrency)][field]
                c = candidate["performance"][str(concurrency)][field]
                require(all(math.isfinite(value) and value > 0 for value in (b, c)), "invalid raw performance metric")
                ratios.append(b / c if field == "rps" else c / b)
            metrics[f"C{concurrency}_{field}"] = paired_gate(ratios)
        result[arm] = {"metrics": metrics, "passed": all(value["passed"] for value in metrics.values())}
    return {"arms": result, "passed": all(value["passed"] for value in result.values())}
