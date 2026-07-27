from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    MeterProvider,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import AggregationTemporality, InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from sie_server.observability import worker_telemetry as wt


class _Config:
    profiles: ClassVar[dict[str, object]] = {"default": object()}

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name == "default"
        return SimpleNamespace(adapter_path="sie_server.adapters.fake.adapter:FakeAdapter")


class _TwoProfileConfig:
    profiles: ClassVar[dict[str, object]] = {"default": object(), "overflow": object()}

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name in self.profiles
        return SimpleNamespace(adapter_path="sie_server.adapters.fake.adapter:FakeAdapter")


class _ExpandedProfileConfig:
    profiles: ClassVar[dict[str, object]] = {"default": object()}

    def __init__(self, base_model: str, profile: str) -> None:
        self.synthetic_profile_variant_source = (base_model, profile)

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name == "default"
        return SimpleNamespace(adapter_path="sie_server.adapters.fake.adapter:FakeAdapter")


class _ProfilesConfig:
    def __init__(self, profiles: tuple[str, ...]) -> None:
        self.profiles = dict.fromkeys(profiles, object())

    def resolve_profile(self, name: str) -> SimpleNamespace:
        assert name in self.profiles
        return SimpleNamespace(adapter_path="sie_server.adapters.fake.adapter:FakeAdapter")


def _metric_map(data: Any) -> dict[str, Any]:
    return {
        metric.name: metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _points(metric: Any) -> list[Any]:
    return list(metric.data.data_points)


def test_endpoint_log_origin_redacts_credentials_path_and_query() -> None:
    raw = "https://user:secret@collector.example:4318/v1/metrics?token=private#fragment"
    assert wt._endpoint_origin_for_log(raw) == "https://collector.example:4318"
    assert wt._endpoint_origin_for_log("not a URL with secret") == "<redacted>"


@pytest.fixture
def active_telemetry() -> tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        resource=Resource.create(
            {
                "service.name": "sie-worker",
                "service.instance.id": "test-instance",
                "deployment.environment": "test",
                "cloud.region": "us-east-1",
            }
        ),
        views=wt.metric_views(),
        shutdown_on_exit=False,
    )
    telemetry = wt.WorkerTelemetry(provider.get_meter("worker-contract-test", "1"))
    wt.configure_worker_metric_context(
        lane="realtime|l4|default",
        configs={"catalog/model": _Config()},  # type: ignore[dict-item]
    )
    return telemetry, reader, provider


def test_item_completed_is_one_call_with_exact_contract_points(
    active_telemetry: tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider],
) -> None:
    telemetry, reader, provider = active_telemetry

    telemetry.item_completed(
        operation="encode",
        outcome="success",
        model="catalog/model",
        profile="default",
        duration_s=0.125,
        item_count=3,
        tokenization_s=0.01,
        inference_s=0.1,
        postprocessing_s=0.015,
        units={"input_tokens": 42, "raw-user-unit": 999},
    )

    by_name = _metric_map(reader.get_metrics_data())
    assert set(by_name) == {
        wt.REQUESTS_METRIC_NAME,
        wt.REQUEST_DURATION_METRIC_NAME,
        wt.INFERENCE_DURATION_METRIC_NAME,
        wt.UNITS_METRIC_NAME,
    }
    request_point = _points(by_name[wt.REQUESTS_METRIC_NAME])[0]
    assert request_point.value == 3
    assert dict(request_point.attributes) == {
        "operation": "encode",
        "outcome": "success",
        "backend": "python",
        "lane": "realtime|l4|default",
        "model": "catalog/model",
        "profile": "default",
    }
    duration_point = _points(by_name[wt.REQUEST_DURATION_METRIC_NAME])[0]
    assert duration_point.count == 3
    assert duration_point.sum == pytest.approx(0.375)
    assert tuple(duration_point.explicit_bounds) == wt.REQUEST_DURATION_BUCKETS_S

    phase_points = _points(by_name[wt.INFERENCE_DURATION_METRIC_NAME])
    assert {point.attributes["phase"] for point in phase_points} == {
        "tokenization",
        "inference",
        "postprocessing",
    }
    assert all(point.count == 3 for point in phase_points)
    assert {point.attributes["phase"]: point.sum for point in phase_points} == {
        "tokenization": pytest.approx(0.03),
        "inference": pytest.approx(0.3),
        "postprocessing": pytest.approx(0.045),
    }
    assert all(tuple(point.explicit_bounds) == wt.INFERENCE_DURATION_BUCKETS_S for point in phase_points)

    unit_point = _points(by_name[wt.UNITS_METRIC_NAME])[0]
    assert unit_point.value == 42
    assert dict(unit_point.attributes) == {
        "operation": "encode",
        "backend": "python",
        "lane": "realtime|l4|default",
        "model": "catalog/model",
        "profile": "default",
        "unit.type": "input_tokens",
    }
    provider.shutdown()


def test_lifecycle_native_topology_and_oom_attributes_match_contract(
    active_telemetry: tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider],
) -> None:
    telemetry, reader, provider = active_telemetry
    telemetry.queue_released(
        operation="encode",
        model="catalog/model",
        profile="default",
        duration_s=0.02,
    )
    telemetry.queue_depth_changed(
        operation="encode",
        model="catalog/model",
        profile="default",
        depth=0,
    )
    telemetry.batch_formed(
        operation="encode",
        model="catalog/model",
        profile="default",
        size=4,
        cost=128,
        capacity=256,
        flush_reason="coalesce",
    )
    telemetry.queue_pending_observed(
        operation="encode",
        model="catalog/model",
        profile="default",
        pending=3,
    )
    telemetry.runtime_batch_dispatched(
        operation="encode",
        model="catalog/model",
        profile="default",
        total_items=4,
        subgroup_sizes=[3, 1],
    )
    telemetry.model_residency_changed(
        model="catalog/model",
        loaded=True,
        memory_bytes=4096,
    )
    telemetry.model_load_completed(
        model="catalog/model",
        duration_s=1.25,
        outcome="success",
        stage="total",
    )
    telemetry.oom_recovery_completed(
        model="catalog/model",
        strategy="cache_clear",
        outcome="success",
    )
    telemetry.model_evicted(model="catalog/model", reason="idle")
    telemetry.adaptive_snapshot(
        model="catalog/model",
        profile="default",
        wait_ms=12.5,
        cost=8192,
        observed_p50_ms=40.0,
        target_p50_ms=50.0,
        starvation_resets_delta=1,
    )

    by_name = _metric_map(reader.get_metrics_data())
    assert set(by_name) == {
        wt.QUEUE_DURATION_METRIC_NAME,
        wt.QUEUE_DEPTH_METRIC_NAME,
        wt.BATCH_SIZE_METRIC_NAME,
        wt.BATCH_COST_METRIC_NAME,
        wt.BATCH_FILL_RATIO_METRIC_NAME,
        wt.QUEUE_PENDING_AT_DISPATCH_METRIC_NAME,
        wt.RUNTIME_BATCH_SIZE_METRIC_NAME,
        wt.RUNTIME_BATCH_SUBGROUPS_METRIC_NAME,
        wt.RUNTIME_SUBGROUP_SIZE_METRIC_NAME,
        wt.MODEL_LOADED_METRIC_NAME,
        wt.MODEL_MEMORY_METRIC_NAME,
        wt.MODEL_LOAD_DURATION_METRIC_NAME,
        wt.OOM_RECOVERIES_METRIC_NAME,
        wt.MODEL_EVICTIONS_METRIC_NAME,
        wt.ADAPTIVE_WAIT_METRIC_NAME,
        wt.ADAPTIVE_COST_METRIC_NAME,
        wt.ADAPTIVE_P50_METRIC_NAME,
        wt.STARVATION_RESETS_METRIC_NAME,
    }
    assert tuple(_points(by_name[wt.QUEUE_DURATION_METRIC_NAME])[0].explicit_bounds) == wt.QUEUE_DURATION_BUCKETS_S
    assert _points(by_name[wt.QUEUE_DEPTH_METRIC_NAME])[0].value == 0
    queue_attributes = {
        "operation": "encode",
        "lane": "realtime|l4|default",
        "model": "catalog/model",
        "profile": "default",
    }
    assert dict(_points(by_name[wt.QUEUE_DEPTH_METRIC_NAME])[0].attributes) == queue_attributes
    batch_point = _points(by_name[wt.BATCH_SIZE_METRIC_NAME])[0]
    assert batch_point.attributes == queue_attributes
    assert tuple(batch_point.explicit_bounds) == wt.BATCH_SIZE_BUCKETS
    batch_cost_point = _points(by_name[wt.BATCH_COST_METRIC_NAME])[0]
    assert batch_cost_point.sum == 128
    assert batch_cost_point.attributes == queue_attributes
    batch_fill_point = _points(by_name[wt.BATCH_FILL_RATIO_METRIC_NAME])[0]
    assert batch_fill_point.value == pytest.approx(0.5)
    assert batch_fill_point.attributes == {**queue_attributes, "flush.reason": "coalesce"}
    assert _points(by_name[wt.QUEUE_PENDING_AT_DISPATCH_METRIC_NAME])[0].sum == 3
    runtime_batch_point = _points(by_name[wt.RUNTIME_BATCH_SIZE_METRIC_NAME])[0]
    assert runtime_batch_point.sum == 4
    assert dict(runtime_batch_point.attributes) == {
        "operation": "encode",
        "backend": "python",
        "lane": "realtime|l4|default",
        "model": "catalog/model",
        "profile": "default",
    }
    assert _points(by_name[wt.RUNTIME_BATCH_SUBGROUPS_METRIC_NAME])[0].sum == 2
    assert sum(point.sum for point in _points(by_name[wt.RUNTIME_SUBGROUP_SIZE_METRIC_NAME])) == 4

    model_attributes = {
        "backend": "python",
        "lane": "realtime|l4|default",
        "model": "catalog/model",
        "profile": "default",
    }
    assert dict(_points(by_name[wt.MODEL_LOADED_METRIC_NAME])[0].attributes) == model_attributes
    assert dict(_points(by_name[wt.MODEL_MEMORY_METRIC_NAME])[0].attributes) == model_attributes
    assert dict(_points(by_name[wt.MODEL_LOAD_DURATION_METRIC_NAME])[0].attributes) == {
        "outcome": "success",
        "stage": "total",
        **model_attributes,
    }
    assert dict(_points(by_name[wt.OOM_RECOVERIES_METRIC_NAME])[0].attributes) == {
        "strategy": "cache_clear",
        "outcome": "success",
        **model_attributes,
    }
    assert dict(_points(by_name[wt.MODEL_EVICTIONS_METRIC_NAME])[0].attributes) == {
        "reason": "idle",
        **model_attributes,
    }
    assert _points(by_name[wt.ADAPTIVE_WAIT_METRIC_NAME])[0].value == pytest.approx(0.0125)
    assert _points(by_name[wt.ADAPTIVE_COST_METRIC_NAME])[0].value == 8192
    p50_points = _points(by_name[wt.ADAPTIVE_P50_METRIC_NAME])
    assert {point.attributes["kind"]: point.value for point in p50_points} == {
        "observed": pytest.approx(0.04),
        "target": pytest.approx(0.05),
    }
    assert _points(by_name[wt.STARVATION_RESETS_METRIC_NAME])[0].value == 1
    provider.shutdown()


def test_unknown_values_collapse_and_never_become_attributes(
    active_telemetry: tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider],
) -> None:
    telemetry, reader, provider = active_telemetry
    telemetry.item_completed(
        operation="private-operation",
        outcome="raw exception text",
        model="tenant-secret/model-id",
        profile="customer-123",
        duration_s=0.01,
        units={"input_tokens": 9},
    )

    serialized = str(reader.get_metrics_data())
    for forbidden in (
        "private-operation",
        "raw exception text",
        "tenant-secret/model-id",
        "customer-123",
    ):
        assert forbidden not in serialized
    provider.shutdown()


def test_generate_operation_is_retained_by_the_contract_domain(
    active_telemetry: tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider],
) -> None:
    telemetry, reader, provider = active_telemetry
    telemetry.item_completed(
        operation="generate",
        outcome="success",
        model="catalog/model",
        profile="default",
        duration_s=0.01,
    )

    point = _points(_metric_map(reader.get_metrics_data())[wt.REQUESTS_METRIC_NAME])[0]
    assert point.attributes["operation"] == "generate"
    provider.shutdown()


def test_cancelled_worker_outcome_is_retained_by_the_contract_domain(
    active_telemetry: tuple[wt.WorkerTelemetry, InMemoryMetricReader, MeterProvider],
) -> None:
    telemetry, reader, provider = active_telemetry
    telemetry.item_completed(
        operation="generate",
        outcome="cancelled",
        model="catalog/model",
        profile="default",
        duration_s=0.01,
    )

    point = _points(_metric_map(reader.get_metrics_data())[wt.REQUESTS_METRIC_NAME])[0]
    assert point.attributes["outcome"] == "cancelled"
    provider.shutdown()


def test_expanded_profile_aliases_share_canonical_pairs_and_invalid_profiles_collapse_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wt, "_admitted_catalog_pairs", set())
    monkeypatch.setattr(wt, "_catalog_collapse_warning_emitted", False)
    monkeypatch.setattr(
        wt,
        "_context",
        wt._WorkerMetricContext(lane="other", profiles={}, backends={}, aliases={}, overflow_pairs=frozenset()),
    )

    model = "BAAI/bge-m3"
    profiles = ("default", "dense", "sparse", "multivector", "hybrid", "fast", "precise", "candle")
    configs: dict[str, Any] = {model: _ProfilesConfig(profiles)}
    configs.update(
        {f"{model}:{profile}": _ExpandedProfileConfig(model, profile) for profile in profiles if profile != "default"}
    )

    wt.configure_worker_metric_context(lane="realtime|l4|default", configs=configs)

    expected_pairs = {(model, profile) for profile in profiles}
    assert wt._admitted_catalog_pairs == expected_pairs
    assert len(wt._admitted_catalog_pairs) == 8
    assert wt._dimensions(f"{model}:sparse", "default") == (
        "realtime|l4|default",
        model,
        "sparse",
        "python",
    )
    assert wt._dimensions(f"{model}:sparse", "sparse") == (
        "realtime|l4|default",
        model,
        "sparse",
        "python",
    )
    assert wt._dimensions(model, "caller-defined-profile") == (
        "realtime|l4|default",
        "other",
        "other",
        "other",
    )


def test_catalog_churn_collapses_after_the_process_lifetime_budget(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(wt, "_admitted_catalog_pairs", set())
    monkeypatch.setattr(wt, "_catalog_collapse_warning_emitted", False)
    monkeypatch.setattr(
        wt,
        "_context",
        wt._WorkerMetricContext(lane="other", profiles={}, backends={}, aliases={}, overflow_pairs=frozenset()),
    )

    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        views=wt.metric_views(),
        shutdown_on_exit=False,
    )
    telemetry = wt.WorkerTelemetry(provider.get_meter("worker-cardinality-test", "1"))
    lane = "realtime|l4|default"
    overflow_count = 8

    try:
        with caplog.at_level("WARNING", logger=wt.__name__):
            for index in range(wt._MAX_CATALOG_MODEL_PROFILE_PAIRS + overflow_count):
                model = f"catalog/model-{index:03d}"
                configs = {model: _Config()}  # type: ignore[dict-item]
                if index == 0:
                    wt.configure_worker_metric_context(lane=lane, configs=configs)
                else:
                    wt.refresh_worker_metric_context(configs=configs)
                telemetry.item_completed(
                    operation="encode",
                    outcome="success",
                    model=model,
                    profile="default",
                    duration_s=0.001,
                )

        first_model = "catalog/model-000"
        wt.refresh_worker_metric_context(configs={first_model: _TwoProfileConfig()})  # type: ignore[dict-item]
        assert wt._dimensions(first_model, "default") == (
            lane,
            first_model,
            "default",
            "python",
        )
        assert wt._dimensions(first_model, "overflow") == (lane, "other", "other", "other")
        assert len(wt._admitted_catalog_pairs) == wt._MAX_CATALOG_MODEL_PROFILE_PAIRS

        points = _points(_metric_map(reader.get_metrics_data())[wt.REQUESTS_METRIC_NAME])
        assert len(points) == wt._MAX_CATALOG_MODEL_PROFILE_PAIRS + 1
        overflow = [point for point in points if point.attributes["model"] == "other"]
        assert len(overflow) == 1
        assert overflow[0].attributes["profile"] == "other"
        assert overflow[0].attributes["backend"] == "other"
        assert overflow[0].value == overflow_count

        warnings = [
            record
            for record in caplog.records
            if record.name == wt.__name__ and "lifetime budget" in record.getMessage()
        ]
        assert len(warnings) == 1
    finally:
        provider.shutdown()


def test_disabled_facade_is_noop() -> None:
    wt.configure_worker_telemetry(None)
    assert wt.worker_telemetry_enabled() is False
    wt.worker_telemetry().item_completed(
        operation="encode",
        outcome="success",
        model="catalog/model",
        profile="default",
        duration_s=0.1,
        units={"input_tokens": 10},
    )
    assert wt.worker_telemetry_enabled() is False


def test_resource_identity_appends_stable_process_uuid_to_substrate_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIE_TELEMETRY_INSTANCE_ID", " pod-uid/worker ")
    monkeypatch.setenv("SIE_OTEL_DEPLOYMENT_ENVIRONMENT", "staging")
    monkeypatch.setenv("SIE_OTEL_CLOUD_REGION", "us-east-1")
    resource = wt.worker_resource_attributes()
    assert resource["service.name"] == "sie-worker"
    assert resource["deployment.environment"] == "staging"
    assert resource["cloud.region"] == "us-east-1"
    assert resource["service.instance.id"].startswith("pod-uid/worker/")
    suffix = resource["service.instance.id"].removeprefix("pod-uid/worker/")
    assert str(uuid.UUID(suffix)) == suffix
    assert wt.service_instance_id() == resource["service.instance.id"]

    monkeypatch.delenv("SIE_TELEMETRY_INSTANCE_ID")
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    first = wt.service_instance_id()
    second = wt.service_instance_id()
    assert first == second
    assert uuid.UUID(first)
    assert first == suffix
    monkeypatch.setenv("MODAL_TASK_ID", "modal-task-123/worker")
    assert wt.service_instance_id() == f"modal-task-123/worker/{first}"


def test_resource_identity_changes_when_a_process_restarts() -> None:
    first_process = str(uuid.uuid4())
    restarted_process = str(uuid.uuid4())
    assert wt._compose_service_instance_id("pod-uid/worker", first_process) != wt._compose_service_instance_id(
        "pod-uid/worker", restarted_process
    )


def test_metric_inventory_is_exact() -> None:
    assert wt.metric_names() == {
        "sie.worker.queue.duration",
        "sie.worker.queue.depth",
        "sie.worker.queue.pending_at_dispatch",
        "sie.worker.batch.size",
        "sie.worker.batch.cost",
        "sie.worker.batch.fill_ratio",
        "sie.worker.runtime.batch.size",
        "sie.worker.runtime.batch.subgroups",
        "sie.worker.runtime.subgroup.size",
        "sie.worker.requests",
        "sie.worker.request.duration",
        "sie.worker.inference.duration",
        "sie.worker.units",
        "sie.worker.model.loaded",
        "sie.worker.model.load.duration",
        "sie.worker.model.memory",
        "sie.worker.model.evictions",
        "sie.worker.oom.recoveries",
        "sie.worker.scheduler.adaptive.wait",
        "sie.worker.scheduler.adaptive.cost",
        "sie.worker.scheduler.adaptive.p50",
        "sie.worker.scheduler.starvation.resets",
        "sie.worker.generation.ttft",
        "sie.worker.generation.tpot",
        "sie.worker.generation.tokens",
        "sie.worker.generation.inflight",
        "sie.worker.generation.kv.reserved",
        "sie.worker.generation.kv.budget",
        "sie.worker.generation.admission.decisions",
        "sie.worker.generation.duplicate_prevented",
        "sie.worker.generation.grammar.compile.duration",
        "sie.worker.generation.grammar.cache.lookups",
        "sie.worker.generation.grammar.requests",
    }


def test_metrics_transport_is_signal_specific_then_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", " HTTP/PROTOBUF ")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "grpc")
    assert wt._metrics_protocol() == "http"

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    assert wt._metrics_protocol() == "grpc"


@pytest.mark.parametrize(
    ("protocol", "endpoint"),
    [("http", "http://collector:4318/v1/metrics"), ("grpc", "http://collector:4317")],
)
def test_metric_exporter_pins_additive_delta_and_stateful_cumulative_temporality(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    endpoint: str,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "CUMULATIVE")
    expected_delta = {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }
    assert wt.otlp_metric_temporality() == expected_delta

    exporter = wt._build_metric_exporter(endpoint, protocol)
    for instrument in expected_delta:
        assert exporter._preferred_temporality[instrument] == AggregationTemporality.DELTA
    for instrument in (UpDownCounter, ObservableUpDownCounter, ObservableGauge):
        assert exporter._preferred_temporality.get(instrument, AggregationTemporality.CUMULATIVE) == (
            AggregationTemporality.CUMULATIVE
        )
    exporter.shutdown()


@pytest.mark.parametrize("unsupported", ["http", "http/json", "json", "thrift"])
def test_metrics_transport_rejects_unsupported_protocols(
    monkeypatch: pytest.MonkeyPatch,
    unsupported: str,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", unsupported)
    with pytest.raises(ValueError, match="unsupported OTLP metrics protocol"):
        wt._metrics_protocol()


def test_invalid_metrics_transport_disables_export_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(wt, "_METER_PROVIDER", None)
    monkeypatch.setenv("SIE_METRICS_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/json")

    assert wt.setup_worker_telemetry() is None
    assert not wt.worker_telemetry_enabled()
    assert "continuing without export" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "http/json" not in caplog.text
