"""Opt-in microbenchmark for the worker telemetry semantic hot path.

Run explicitly with::

    SIE_RUN_TELEMETRY_BENCHMARK=1 mise exec -- uv run pytest \
      packages/sie_server/tests/observability/test_worker_telemetry_benchmark.py -s -q

The benchmark reports three independently warmed raw nanoseconds/completion
samples and their median for the disabled facade and an enabled in-memory OTel
provider. It covers both a single-item completion, a representative
``item_count=32`` batch, and the
checked-in `EngineConfig.max_batch_requests` default so the cost of exact
per-item histogram weighting is measured through the actual configured batch
ceiling. It enforces the checked-in generous regression tripwires in
``telemetry/performance-budgets.json`` and stays skipped in ordinary test runs.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sie_server.config.engine import EngineConfig
from sie_server.observability import generation_diagnostics as gd
from sie_server.observability import worker_telemetry as wt

_EVENTS_PER_ITERATION = 50_000
_WARMUP_EVENTS = 2_000
_BATCH_EVENTS_PER_ITERATION = 10_000
_BATCH_WARMUP_EVENTS = 500
_BATCH_ITEM_COUNT = 32
_MAX_BATCH_EVENTS_PER_ITERATION = 5_000
_MAX_BATCH_WARMUP_EVENTS = 250
_CONFIGURED_MAX_BATCH_ITEM_COUNT = int(EngineConfig.model_fields["max_batch_requests"].default)
_ITERATIONS = 3
_PERFORMANCE_BUDGETS = json.loads(
    (Path(__file__).resolve().parents[4] / "telemetry" / "performance-budgets.json").read_text()
)["budgets"]


class _Config:
    profiles: ClassVar[dict[str, object]] = {"default": object()}

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name == "default"
        return SimpleNamespace(adapter_path="sie_server.adapters.fake.adapter:FakeAdapter")


def _emit_item(count: int) -> None:
    for _ in range(count):
        wt.worker_telemetry().item_completed(
            operation="encode",
            outcome="success",
            model="catalog/model",
            profile="default",
            duration_s=0.012,
            tokenization_s=0.001,
            inference_s=0.01,
            postprocessing_s=0.001,
            units={"input_tokens": 24},
        )


def _emit_item_batch_32(count: int) -> None:
    _emit_item_batch(count, _BATCH_ITEM_COUNT)


def _emit_item_batch_configured_max(count: int) -> None:
    _emit_item_batch(count, _CONFIGURED_MAX_BATCH_ITEM_COUNT)


def _emit_item_batch(count: int, item_count: int) -> None:
    for _ in range(count):
        wt.worker_telemetry().item_completed(
            operation="encode",
            outcome="success",
            model="catalog/model",
            profile="default",
            duration_s=0.012,
            item_count=item_count,
            tokenization_s=0.001,
            inference_s=0.01,
            postprocessing_s=0.001,
            units={"input_tokens": 24 * item_count},
        )


def _emit_generation_stream(count: int) -> None:
    for _ in range(count):
        timer = gd.GenerationStreamTimer("catalog/model")
        timer.mark_yield(has_text=True)
        timer.mark_yield(has_text=True)
        timer.finalize(prompt_tokens=24, completion_tokens=2)


def _samples(
    emit: Callable[[int], None],
    *,
    meter: object | None,
    events_per_iteration: int = _EVENTS_PER_ITERATION,
    warmup_events: int = _WARMUP_EVENTS,
) -> list[float]:
    samples: list[float] = []
    for _ in range(_ITERATIONS):
        wt.configure_worker_telemetry(meter)  # type: ignore[arg-type]
        emit(warmup_events)
        started = time.perf_counter_ns()
        emit(events_per_iteration)
        elapsed = time.perf_counter_ns() - started
        samples.append(elapsed / events_per_iteration)
    return samples


@pytest.mark.skipif(
    os.environ.get("SIE_RUN_TELEMETRY_BENCHMARK") != "1",
    reason="opt in with SIE_RUN_TELEMETRY_BENCHMARK=1",
)
def test_worker_telemetry_hot_path_benchmark() -> None:
    wt.configure_worker_metric_context(
        lane="benchmark|cpu|default",
        configs={"catalog/model": _Config()},  # type: ignore[dict-item]
    )
    reader = InMemoryMetricReader(preferred_temporality=wt.otlp_metric_temporality())
    provider = MeterProvider(metric_readers=[reader], views=wt.metric_views(), shutdown_on_exit=False)
    meter = provider.get_meter("worker-telemetry-benchmark", "1")
    try:
        disabled = _samples(_emit_item, meter=None)
        enabled = _samples(_emit_item, meter=meter)
        batch_32_disabled = _samples(
            _emit_item_batch_32,
            meter=None,
            events_per_iteration=_BATCH_EVENTS_PER_ITERATION,
            warmup_events=_BATCH_WARMUP_EVENTS,
        )
        batch_32_enabled = _samples(
            _emit_item_batch_32,
            meter=meter,
            events_per_iteration=_BATCH_EVENTS_PER_ITERATION,
            warmup_events=_BATCH_WARMUP_EVENTS,
        )
        configured_max_disabled = _samples(
            _emit_item_batch_configured_max,
            meter=None,
            events_per_iteration=_MAX_BATCH_EVENTS_PER_ITERATION,
            warmup_events=_MAX_BATCH_WARMUP_EVENTS,
        )
        configured_max_enabled = _samples(
            _emit_item_batch_configured_max,
            meter=meter,
            events_per_iteration=_MAX_BATCH_EVENTS_PER_ITERATION,
            warmup_events=_MAX_BATCH_WARMUP_EVENTS,
        )
        generation_disabled = _samples(_emit_generation_stream, meter=None)
        generation_enabled = _samples(_emit_generation_stream, meter=meter)
    finally:
        wt.configure_worker_telemetry(None)
        provider.shutdown()

    disabled_median = statistics.median(disabled)
    enabled_median = statistics.median(enabled)
    batch_32_disabled_per_item = statistics.median(batch_32_disabled) / _BATCH_ITEM_COUNT
    batch_32_enabled_per_item = statistics.median(batch_32_enabled) / _BATCH_ITEM_COUNT
    configured_max_disabled_per_item = statistics.median(configured_max_disabled) / _CONFIGURED_MAX_BATCH_ITEM_COUNT
    configured_max_enabled_per_item = statistics.median(configured_max_enabled) / _CONFIGURED_MAX_BATCH_ITEM_COUNT
    generation_disabled_median = statistics.median(generation_disabled)
    generation_enabled_median = statistics.median(generation_enabled)
    result = {
        "events_per_iteration": _EVENTS_PER_ITERATION,
        "iterations": _ITERATIONS,
        "disabled_ns_per_event": disabled,
        "disabled_median_ns_per_event": disabled_median,
        "enabled_ns_per_event": enabled,
        "enabled_median_ns_per_event": enabled_median,
        "incremental_median_ns_per_event": max(enabled_median - disabled_median, 0.0),
        "batch_32_events_per_iteration": _BATCH_EVENTS_PER_ITERATION,
        "batch_32_item_count": _BATCH_ITEM_COUNT,
        "batch_32_disabled_ns_per_completion_event": batch_32_disabled,
        "batch_32_disabled_median_ns_per_completion_event": statistics.median(batch_32_disabled),
        "batch_32_disabled_median_ns_per_item_derived": batch_32_disabled_per_item,
        "batch_32_enabled_ns_per_completion_event": batch_32_enabled,
        "batch_32_enabled_median_ns_per_completion_event": statistics.median(batch_32_enabled),
        "batch_32_enabled_median_ns_per_item_derived": batch_32_enabled_per_item,
        "batch_32_incremental_median_ns_per_item_derived": max(
            batch_32_enabled_per_item - batch_32_disabled_per_item, 0.0
        ),
        "configured_max_batch_events_per_iteration": _MAX_BATCH_EVENTS_PER_ITERATION,
        "configured_max_batch_item_count": _CONFIGURED_MAX_BATCH_ITEM_COUNT,
        "configured_max_batch_disabled_ns_per_completion_event": configured_max_disabled,
        "configured_max_batch_disabled_median_ns_per_completion_event": statistics.median(configured_max_disabled),
        "configured_max_batch_disabled_median_ns_per_item_derived": configured_max_disabled_per_item,
        "configured_max_batch_enabled_ns_per_completion_event": configured_max_enabled,
        "configured_max_batch_enabled_median_ns_per_completion_event": statistics.median(configured_max_enabled),
        "configured_max_batch_enabled_median_ns_per_item_derived": configured_max_enabled_per_item,
        "configured_max_batch_incremental_median_ns_per_item_derived": max(
            configured_max_enabled_per_item - configured_max_disabled_per_item, 0.0
        ),
        "generation_stream_disabled_ns_per_event": generation_disabled,
        "generation_stream_disabled_median_ns_per_event": generation_disabled_median,
        "generation_stream_enabled_ns_per_event": generation_enabled,
        "generation_stream_enabled_median_ns_per_event": generation_enabled_median,
        "generation_stream_incremental_median_ns_per_event": max(
            generation_enabled_median - generation_disabled_median, 0.0
        ),
    }
    assert result["disabled_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["python_worker_disabled_ns_per_item"]
    assert result["enabled_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["python_worker_enabled_ns_per_item"]
    assert result["incremental_median_ns_per_event"] <= _PERFORMANCE_BUDGETS["python_worker_incremental_ns_per_item"]
    assert (
        result["batch_32_disabled_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_disabled_ns_per_item"]
    )
    assert (
        result["batch_32_enabled_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_enabled_ns_per_item"]
    )
    assert (
        result["configured_max_batch_enabled_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_enabled_ns_per_item"]
    )
    assert (
        result["configured_max_batch_disabled_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_disabled_ns_per_item"]
    )
    assert (
        result["batch_32_incremental_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_incremental_ns_per_item"]
    )
    assert (
        result["configured_max_batch_incremental_median_ns_per_item_derived"]
        <= _PERFORMANCE_BUDGETS["python_worker_batch_incremental_ns_per_item"]
    )
    assert (
        result["generation_stream_disabled_median_ns_per_event"]
        <= _PERFORMANCE_BUDGETS["python_worker_generation_disabled_ns_per_event"]
    )
    assert (
        result["generation_stream_enabled_median_ns_per_event"]
        <= _PERFORMANCE_BUDGETS["python_worker_generation_enabled_ns_per_event"]
    )
    assert (
        result["generation_stream_incremental_median_ns_per_event"]
        <= _PERFORMANCE_BUDGETS["python_worker_generation_incremental_ns_per_event"]
    )
    print("WORKER_TELEMETRY_BENCHMARK " + json.dumps(result, sort_keys=True))
