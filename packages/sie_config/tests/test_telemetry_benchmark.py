"""Opt-in microbenchmark for the config telemetry request facade.

Run explicitly with::

    SIE_RUN_TELEMETRY_BENCHMARK=1 mise exec -- uv run pytest \
      packages/sie_config/tests/test_telemetry_benchmark.py -s -q

It reports three independently warmed disabled/enabled timing samples and
per-invocation temporary-allocation peaks, then enforces the checked-in
generous regression tripwire in ``telemetry/performance-budgets.json``.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sie_config import managed_metrics as mm
from sie_config import metrics as config_metrics

_EVENTS_PER_ITERATION = 50_000
_WARMUP_EVENTS = 2_000
_ALLOCATION_EVENTS_PER_SAMPLE = 256
_ITERATIONS = 3
_PERFORMANCE_BUDGETS = json.loads(
    (Path(__file__).resolve().parents[3] / "telemetry" / "performance-budgets.json").read_text()
)["budgets"]


@contextmanager
def _facade(facade: mm.ConfigMetrics) -> Iterator[None]:
    previous = mm._MANAGED
    mm._MANAGED = facade
    try:
        yield
    finally:
        mm._MANAGED = previous


def _emit_one() -> None:
    config_metrics.record_http_request(
        method="GET",
        path="/v1/configs/epoch",
        status=200,
        duration_s=0.002,
    )


def _emit(count: int) -> None:
    for _ in range(count):
        _emit_one()


def _samples(facade: mm.ConfigMetrics) -> list[float]:
    samples: list[float] = []
    with _facade(facade):
        for _ in range(_ITERATIONS):
            _emit(_WARMUP_EVENTS)
            started = time.perf_counter_ns()
            _emit(_EVENTS_PER_ITERATION)
            elapsed = time.perf_counter_ns() - started
            samples.append(elapsed / _EVENTS_PER_ITERATION)
    return samples


def _temporary_peak_bytes_per_invocation_samples(facade: mm.ConfigMetrics) -> list[float]:
    samples: list[float] = []
    with _facade(facade):
        for _ in range(_ITERATIONS):
            _emit(_WARMUP_EVENTS)
            invocation_peaks: list[float] = []
            tracemalloc.start()
            try:
                for _ in range(_ALLOCATION_EVENTS_PER_SAMPLE):
                    tracemalloc.reset_peak()
                    current, _ = tracemalloc.get_traced_memory()
                    _emit_one()
                    _, peak = tracemalloc.get_traced_memory()
                    invocation_peaks.append(float(max(peak - current, 0)))
            finally:
                tracemalloc.stop()
            samples.append(statistics.median(invocation_peaks))
    return samples


@pytest.mark.skipif(
    os.environ.get("SIE_RUN_TELEMETRY_BENCHMARK") != "1",
    reason="opt in with SIE_RUN_TELEMETRY_BENCHMARK=1",
)
def test_config_telemetry_hot_path_benchmark() -> None:
    disabled = mm._DISABLED
    reader = InMemoryMetricReader(preferred_temporality=mm._otlp_metric_temporality())
    provider = MeterProvider(metric_readers=[reader], views=mm._metric_views(), shutdown_on_exit=False)
    enabled = mm.ManagedConfigMetrics(provider.get_meter("config-telemetry-benchmark", "1"))
    try:
        disabled_samples = _samples(disabled)
        enabled_samples = _samples(enabled)
        disabled_peak_samples = _temporary_peak_bytes_per_invocation_samples(disabled)
        enabled_peak_samples = _temporary_peak_bytes_per_invocation_samples(enabled)
    finally:
        provider.shutdown()

    disabled_median = statistics.median(disabled_samples)
    enabled_median = statistics.median(enabled_samples)
    result = {
        "events_per_iteration": _EVENTS_PER_ITERATION,
        "iterations": _ITERATIONS,
        "disabled_ns_per_event": disabled_samples,
        "disabled_median_ns_per_event": disabled_median,
        "allocation_events_per_sample": _ALLOCATION_EVENTS_PER_SAMPLE,
        "disabled_temporary_peak_bytes_per_invocation": disabled_peak_samples,
        "disabled_temporary_peak_median_bytes_per_invocation": statistics.median(disabled_peak_samples),
        "enabled_ns_per_event": enabled_samples,
        "enabled_median_ns_per_event": enabled_median,
        "incremental_median_ns_per_event": max(enabled_median - disabled_median, 0.0),
        "enabled_temporary_peak_bytes_per_invocation": enabled_peak_samples,
        "enabled_temporary_peak_median_bytes_per_invocation": statistics.median(enabled_peak_samples),
    }
    assert result["disabled_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["config_disabled_ns_per_event"]
    assert result["enabled_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["config_enabled_ns_per_event"]
    assert result["incremental_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["config_incremental_ns_per_event"]
    assert (
        result["disabled_temporary_peak_median_bytes_per_invocation"]
        <= _PERFORMANCE_BUDGETS["config_disabled_temporary_peak_bytes_per_invocation"]
    )
    assert (
        result["enabled_temporary_peak_median_bytes_per_invocation"]
        <= _PERFORMANCE_BUDGETS["config_enabled_temporary_peak_bytes_per_invocation"]
    )
    print("CONFIG_TELEMETRY_BENCHMARK " + json.dumps(result, sort_keys=True))
