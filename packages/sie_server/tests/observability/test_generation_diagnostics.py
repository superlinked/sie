from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from sie_server.observability import generation_diagnostics as gd
from sie_server.observability import worker_telemetry as wt


class _Config:
    profiles: ClassVar[dict[str, object]] = {"default": object()}

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name == "default"
        return SimpleNamespace(adapter_path="sie_server.adapters.sglang.generation:SGLangGenerationAdapter")


def _metric_map(data: Any) -> dict[str, Any]:
    return {
        metric.name: metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _points(metric: Any) -> list[Any]:
    return list(metric.data.data_points)


@pytest.fixture
def active_generation() -> tuple[InMemoryMetricReader, MeterProvider]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], views=wt.metric_views(), shutdown_on_exit=False)
    wt.configure_worker_metric_context(lane="generation|l4|sglang", configs={"catalog/gen": _Config()})  # type: ignore[dict-item]
    wt.configure_worker_telemetry(provider.get_meter("generation-contract-test", "1"))
    yield reader, provider
    wt.configure_worker_telemetry(None)
    provider.shutdown()


def test_typed_generation_events_emit_bounded_contract_points(
    active_generation: tuple[InMemoryMetricReader, MeterProvider],
) -> None:
    reader, _ = active_generation
    telemetry = wt.worker_telemetry()
    telemetry.first_token_observed(model="catalog/gen", grammar="json_schema", duration_s=0.25)
    telemetry.stream_finished(
        model="catalog/gen",
        grammar="json_schema",
        tpot_s=0.05,
        prompt_tokens=12,
        completion_tokens=4,
    )
    telemetry.state_changed(model="catalog/gen", reserved_tokens=128, inflight=1, budget_tokens=1024)
    telemetry.admission_decided(model="catalog/gen", outcome="rejected", reason="kv_budget")
    telemetry.duplicate_prevented(model="catalog/gen", path="first_chunk_fallback")
    telemetry.grammar_requested(model="catalog/gen", backend="outlines", grammar="json_schema")
    telemetry.grammar_cache_lookup(
        model="catalog/gen",
        backend="outlines",
        grammar="json_schema",
        phase="request",
        result="miss",
    )
    telemetry.grammar_compile_completed(
        model="catalog/gen",
        backend="outlines",
        grammar="json_schema",
        phase="request",
        outcome="success",
        duration_s=0.125,
    )

    by_name = _metric_map(reader.get_metrics_data())
    assert set(by_name) == wt.generation_metric_names()
    for name in (wt.GENERATION_TTFT_METRIC_NAME, wt.GENERATION_TPOT_METRIC_NAME):
        point = _points(by_name[name])[0]
        assert tuple(point.explicit_bounds) == wt.TTFT_TPOT_BUCKETS_S
        assert dict(point.attributes) == {
            "backend": "sglang",
            "lane": "generation|l4|sglang",
            "model": "catalog/gen",
            "profile": "default",
            "grammar": "json_schema",
        }
    token_points = _points(by_name[wt.GENERATION_TOKENS_METRIC_NAME])
    assert {point.attributes["token.type"]: point.value for point in token_points} == {
        "prompt": 12,
        "completion": 4,
    }
    assert _points(by_name[wt.GENERATION_INFLIGHT_METRIC_NAME])[0].value == 1
    assert _points(by_name[wt.GENERATION_KV_RESERVED_METRIC_NAME])[0].value == 128
    assert _points(by_name[wt.GENERATION_KV_BUDGET_METRIC_NAME])[0].value == 1024
    assert dict(_points(by_name[wt.GENERATION_ADMISSION_METRIC_NAME])[0].attributes) | {} == {
        "backend": "sglang",
        "lane": "generation|l4|sglang",
        "model": "catalog/gen",
        "profile": "default",
        "outcome": "rejected",
        "reason": "kv_budget",
    }
    grammar_attributes = {
        "backend": "sglang",
        "lane": "generation|l4|sglang",
        "model": "catalog/gen",
        "profile": "default",
        "grammar.backend": "outlines",
        "grammar": "json_schema",
    }
    assert dict(_points(by_name[wt.GRAMMAR_REQUESTS_METRIC_NAME])[0].attributes) == grammar_attributes
    assert dict(_points(by_name[wt.GRAMMAR_CACHE_LOOKUPS_METRIC_NAME])[0].attributes) == {
        **grammar_attributes,
        "phase": "request",
        "result": "miss",
    }
    compile_point = _points(by_name[wt.GRAMMAR_COMPILE_DURATION_METRIC_NAME])[0]
    assert tuple(compile_point.explicit_bounds) == wt.TTFT_TPOT_BUCKETS_S
    assert dict(compile_point.attributes) == {
        **grammar_attributes,
        "phase": "request",
        "outcome": "success",
    }


def test_stream_timer_records_ttft_tpot_and_tokens_once(
    active_generation: tuple[InMemoryMetricReader, MeterProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _ = active_generation
    clock = iter([10.0, 10.1, 10.5])
    monkeypatch.setattr(gd.time, "perf_counter", lambda: next(clock))
    timer = gd.GenerationStreamTimer("catalog/gen", grammar="regex")
    timer.mark_yield(has_text=True)
    timer.mark_yield(has_text=True)
    timer.finalize(prompt_tokens=8, completion_tokens=2)
    timer.finalize(prompt_tokens=999, completion_tokens=999)

    by_name = _metric_map(reader.get_metrics_data())
    assert _points(by_name[wt.GENERATION_TTFT_METRIC_NAME])[0].sum == pytest.approx(0.1)
    assert _points(by_name[wt.GENERATION_TPOT_METRIC_NAME])[0].sum == pytest.approx(0.2)
    assert sum(point.value for point in _points(by_name[wt.GENERATION_TOKENS_METRIC_NAME])) == 10


def test_disabled_stream_timer_is_a_clock_free_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    wt.configure_worker_telemetry(None)

    def unexpected_clock_read() -> float:
        pytest.fail("disabled generation telemetry must not read the hot-path clock")

    monkeypatch.setattr(gd.time, "perf_counter", unexpected_clock_read)
    timer = gd.GenerationStreamTimer("catalog/gen", grammar="regex")
    timer.mark_yield(has_text=True)
    timer.finalize(prompt_tokens=3, completion_tokens=2)


def test_single_output_event_has_no_measurable_tpot(
    active_generation: tuple[InMemoryMetricReader, MeterProvider],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, _ = active_generation
    clock = iter([10.0, 10.1])
    monkeypatch.setattr(gd.time, "perf_counter", lambda: next(clock))
    timer = gd.GenerationStreamTimer("catalog/gen")
    timer.mark_yield(has_text=True)
    timer.finalize(prompt_tokens=8, completion_tokens=1)

    by_name = _metric_map(reader.get_metrics_data())
    assert wt.GENERATION_TPOT_METRIC_NAME not in by_name


def test_unknown_generation_dimensions_and_enums_collapse_without_leaking(
    active_generation: tuple[InMemoryMetricReader, MeterProvider],
) -> None:
    reader, _ = active_generation
    sentinel = "tenant-secret/raw-request-id"
    wt.worker_telemetry().admission_decided(model=sentinel, outcome=sentinel, reason=sentinel)
    serialized = str(reader.get_metrics_data())
    assert sentinel not in serialized
    point = _points(_metric_map(reader.get_metrics_data())[wt.GENERATION_ADMISSION_METRIC_NAME])[0]
    assert point.attributes["model"] == "other"
    assert point.attributes["outcome"] == "other"
    assert point.attributes["reason"] == "other"
