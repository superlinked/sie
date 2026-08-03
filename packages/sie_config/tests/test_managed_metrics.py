from __future__ import annotations

import gzip
import threading
import uuid
from concurrent import futures
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

import grpc
import pytest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2_grpc import (
    MetricsServiceServicer,
    add_MetricsServiceServicer_to_server,
)
from opentelemetry.proto.metrics.v1 import metrics_pb2
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from sie_config import managed_metrics as mm
from sie_config import metrics as config_metrics


def _metric_map(data: Any) -> dict[str, Any]:
    return {
        metric.name: metric
        for resource_metrics in data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _point(metric: Any) -> Any:
    points = list(metric.data.data_points)
    assert points
    return points[-1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", 5_000), ("2000", 2_000), ("30000", 5_000), ("invalid", 5_000), ("500", 5_000)],
)
def test_metric_export_interval_never_exceeds_keda_contract(raw: str, expected: int) -> None:
    assert mm._metrics_export_interval_ms(raw) == expected


def test_endpoint_log_origin_redacts_credentials_path_and_query() -> None:
    raw = "https://user:secret@collector.example:4318/v1/metrics?token=private#fragment"
    assert mm._endpoint_origin_for_log(raw) == "https://collector.example:4318"
    assert mm._endpoint_origin_for_log("not a URL with secret") == "<redacted>"


def test_managed_config_contract_is_dotted_bounded_and_complete() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(
        metric_readers=[reader],
        views=mm._metric_views(),
        resource=Resource.create(
            {
                "service.name": "sie-config",
                "deployment.environment": "test",
                "cloud.region": "us-east",
            }
        ),
    )
    contract = mm.ManagedConfigMetrics(provider.get_meter("test"))

    contract.record_request(
        method="USER-SUPPLIED",
        route="/secret/customer/route",
        status_code=999,
        duration_s=0.125,
    )
    contract.set_epoch(7)
    contract.set_models(source="api", count=3)
    contract.record_publish(operation="user-op", outcome="raw-error")
    contract.record_store_write(operation="write_model", outcome="success")
    contract.set_messaging_ready(True)

    data = reader.get_metrics_data()
    by_name = _metric_map(data)
    assert set(by_name) == set(mm.metric_names().values())
    assert all("_" not in name for name in by_name)

    request = _point(by_name[mm.REQUESTS_METRIC_NAME])
    assert request.value == 1
    assert dict(request.attributes) == {
        "http.method": "other",
        "http.route": "other",
        "http.status_code": 0,
    }
    duration_point = _point(by_name[mm.REQUEST_DURATION_METRIC_NAME])
    assert duration_point.sum == pytest.approx(0.125)
    assert duration_point.explicit_bounds == mm.REQUEST_DURATION_BUCKETS_S
    assert _point(by_name[mm.EPOCH_METRIC_NAME]).value == 7
    assert by_name[mm.EPOCH_METRIC_NAME].unit == "{epoch}"
    assert dict(_point(by_name[mm.MODELS_METRIC_NAME]).attributes) == {"source": "api"}
    assert by_name[mm.MODELS_METRIC_NAME].unit == "{model}"
    assert dict(_point(by_name[mm.PUBLISH_METRIC_NAME]).attributes) == {
        "operation": "other",
        "outcome": "other",
    }
    assert dict(_point(by_name[mm.STORE_WRITES_METRIC_NAME]).attributes) == {
        "operation": "write_model",
        "outcome": "success",
    }
    messaging = _point(by_name[mm.MESSAGING_READY_METRIC_NAME])
    assert messaging.value == 1
    assert dict(messaging.attributes) == {"transport": "nats"}

    serialized = str(data)
    for forbidden in ("USER-SUPPLIED", "/secret/customer/route", "raw-error", "user-op"):
        assert forbidden not in serialized
    provider.shutdown()


def test_disabled_semantic_facade_constructs_no_sdk_instruments_or_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIE_METRICS_ENABLED", raising=False)
    monkeypatch.setattr(mm, "_METER_PROVIDER", None)
    monkeypatch.setattr(mm, "_MANAGED", mm._DISABLED)

    def unexpected(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("disabled telemetry must not construct an OTel facade or labels")

    monkeypatch.setattr(mm, "ManagedConfigMetrics", unexpected)
    monkeypatch.setattr(mm, "_bounded", unexpected)

    assert mm.setup_managed_metrics() is None
    assert not config_metrics.telemetry_enabled()
    config_metrics.record_http_request(
        method="GET",
        path="/v1/configs/epoch",
        status=200,
        duration_s=0.001,
    )
    config_metrics.set_epoch(1)
    config_metrics.update_models_gauge(api_count=1, filesystem_count=2)
    config_metrics.record_snapshot_publish(success=True)
    config_metrics.record_nats_publish(config_metrics.NATS_PUBLISH_SUCCESS)
    config_metrics.record_store_write(
        config_metrics.STORE_OP_WRITE_MODEL,
        config_metrics.STORE_RESULT_SUCCESS,
    )
    config_metrics.set_nats_connected(True)


def test_public_semantic_facade_emits_once_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], views=mm._metric_views(), shutdown_on_exit=False)
    facade = mm.ManagedConfigMetrics(provider.get_meter("public-config-facade-test"))
    monkeypatch.setattr(mm, "_MANAGED", facade)

    config_metrics.record_http_request(
        method="GET",
        path="/v1/configs/epoch",
        status=200,
        duration_s=0.002,
    )

    by_name = _metric_map(reader.get_metrics_data())
    assert _point(by_name[mm.REQUESTS_METRIC_NAME]).value == 1
    assert _point(by_name[mm.REQUEST_DURATION_METRIC_NAME]).count == 1
    assert config_metrics.telemetry_enabled()
    provider.shutdown()


def test_resource_has_stable_unique_process_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIE_TELEMETRY_INSTANCE_ID", raising=False)
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)
    monkeypatch.delenv("SIE_OTEL_DEPLOYMENT_ENVIRONMENT", raising=False)
    monkeypatch.delenv("SIE_DEPLOYMENT_ENV", raising=False)
    monkeypatch.delenv("SIE_OTEL_CLOUD_REGION", raising=False)
    monkeypatch.delenv("SIE_CLOUD_REGION", raising=False)
    first_resource = mm._resource_attributes()
    second_resource = mm._resource_attributes()
    first = first_resource["service.instance.id"]
    second = second_resource["service.instance.id"]
    assert first == second
    assert str(uuid.UUID(first)) == first
    assert first_resource["deployment.environment"] == "unknown"
    assert first_resource["cloud.region"] == "unknown"


def test_resource_appends_process_uuid_to_prefix_and_changes_on_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_TELEMETRY_INSTANCE_ID", " pod-uid/config/ ")
    resource = mm._resource_attributes()
    assert resource["service.instance.id"].startswith("pod-uid/config/")
    suffix = resource["service.instance.id"].removeprefix("pod-uid/config/")
    assert str(uuid.UUID(suffix)) == suffix
    assert mm._resource_attributes()["service.instance.id"] == resource["service.instance.id"]

    monkeypatch.delenv("SIE_TELEMETRY_INSTANCE_ID")
    monkeypatch.setenv("MODAL_TASK_ID", "modal-task-123/config")
    assert mm._resource_attributes()["service.instance.id"] == f"modal-task-123/config/{suffix}"

    restarted_process = str(uuid.uuid4())
    assert mm._compose_service_instance_id("pod-uid/config", suffix) != mm._compose_service_instance_id(
        "pod-uid/config", restarted_process
    )


def test_modal_proxy_headers_require_exact_origin_path_and_complete_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MODAL_PROXY_AUTH", "1")
    monkeypatch.setenv("SIE_OTEL_PROXY_AUTH_ORIGIN", "https://workspace--collector.modal.run")
    monkeypatch.setenv("SIE_MODAL_PROXY_TOKEN_ID", "id")
    monkeypatch.setenv("SIE_MODAL_PROXY_TOKEN_SECRET", "secret")

    assert mm._modal_proxy_headers("https://workspace--collector.modal.run/v1/metrics") == {
        "Modal-Key": "id",
        "Modal-Secret": "secret",
    }
    for endpoint in (
        "https://attacker.example/v1/metrics",
        "https://workspace--collector.modal.run/v1/traces",
        "https://workspace--collector.modal.run.evil.example/v1/metrics",
        "http://workspace--collector.modal.run/v1/metrics",
        "https://workspace--collector.modal.run:not-a-port/v1/metrics",
    ):
        with pytest.raises(ValueError, match="untrusted"):
            mm._modal_proxy_headers(endpoint)

    monkeypatch.delenv("SIE_MODAL_PROXY_TOKEN_SECRET")
    with pytest.raises(ValueError, match="complete credential pair"):
        mm._modal_proxy_headers("https://workspace--collector.modal.run/v1/metrics")


@pytest.mark.parametrize(
    ("protocol", "generic_endpoint", "expected"),
    [
        ("grpc", "http://collector:4327", "http://collector:4327"),
        ("http", "http://collector:4318", "http://collector:4318/v1/metrics"),
        ("http", "http://collector:4318/", "http://collector:4318/v1/metrics"),
    ],
)
def test_generic_metrics_endpoint_follows_transport(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    generic_endpoint: str,
    expected: str,
) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", generic_endpoint)
    assert mm._metrics_endpoint(protocol) == expected


def test_signal_specific_metrics_endpoint_is_never_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "https://collector.example/custom")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://ignored.example")
    assert mm._metrics_endpoint("grpc") == "https://collector.example/custom"
    assert mm._metrics_endpoint("http") == "https://collector.example/custom"


def test_metrics_protocol_precedence_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", " HTTP/PROTOBUF ")
    assert mm._metrics_protocol() == "http"

    for unsupported in ("http", "http/json", "json", "thrift"):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", unsupported)
        with pytest.raises(ValueError, match="unsupported OTLP metrics protocol"):
            mm._metrics_protocol()


def test_metrics_protocol_never_inherits_trace_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    assert mm._metrics_protocol() == "grpc"


def test_invalid_metrics_protocol_is_operator_visible_and_fail_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(mm, "_METER_PROVIDER", None)
    monkeypatch.setattr(mm, "_MANAGED", mm._DISABLED)
    monkeypatch.setenv("SIE_METRICS_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://collector:4318/v1/metrics")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "http/json")

    assert mm.setup_managed_metrics() is None
    assert mm.managed_metrics() is mm._DISABLED
    assert "continuing without export" in caplog.text
    assert "error_type=ValueError" in caplog.text
    assert "http/json" not in caplog.text


def test_config_exporter_rejects_modal_credentials_over_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIE_MODAL_PROXY_AUTH", "1")

    with pytest.raises(ValueError, match="requires HTTP"):
        mm._build_exporter("https://workspace--collector.modal.run", "grpc")


class _CaptureHandler(BaseHTTPRequestHandler):
    request_path = ""
    request_headers: ClassVar[dict[str, str]] = {}
    request_body = b""
    requests: ClassVar[list[tuple[dict[str, str], bytes]]] = []

    def do_POST(self) -> None:
        type(self).request_path = self.path
        type(self).request_headers = {key.lower(): value for key, value in self.headers.items()}
        type(self).request_body = self.rfile.read(int(self.headers["content-length"]))
        type(self).requests.append((type(self).request_headers, type(self).request_body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _assert_config_wire_request(request: ExportMetricsServiceRequest, *, epoch: int) -> None:
    metrics = {
        metric.name: metric
        for resource_metrics in request.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name.startswith("sie.")
    }
    assert set(metrics) == {
        mm.REQUESTS_METRIC_NAME,
        mm.REQUEST_DURATION_METRIC_NAME,
        mm.EPOCH_METRIC_NAME,
    }
    assert metrics[mm.REQUESTS_METRIC_NAME].sum.aggregation_temporality == (metrics_pb2.AGGREGATION_TEMPORALITY_DELTA)
    assert metrics[mm.REQUESTS_METRIC_NAME].sum.data_points[0].as_int == 1
    assert metrics[mm.REQUEST_DURATION_METRIC_NAME].histogram.aggregation_temporality == (
        metrics_pb2.AGGREGATION_TEMPORALITY_DELTA
    )
    assert metrics[mm.REQUEST_DURATION_METRIC_NAME].histogram.data_points[0].count == 1
    assert metrics[mm.EPOCH_METRIC_NAME].WhichOneof("data") == "gauge"
    assert metrics[mm.EPOCH_METRIC_NAME].gauge.data_points[0].as_int == epoch


def test_real_otlp_http_export_contains_only_dotted_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _CaptureHandler.requests = []
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE", "CUMULATIVE")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/metrics"
        exporter = mm._build_exporter(endpoint, "http")
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
        provider = MeterProvider(
            metric_readers=[reader],
            views=mm._metric_views(),
            resource=Resource.create({"service.name": "sie-config", "deployment.environment": "test"}),
        )
        contract = mm.ManagedConfigMetrics(provider.get_meter("transport-test"))
        contract.record_request(method="GET", route="/v1/configs/epoch", status_code=200, duration_s=0.01)
        contract.set_epoch(11)
        assert provider.force_flush(timeout_millis=5_000)
        contract.record_request(method="GET", route="/v1/configs/epoch", status_code=200, duration_s=0.02)
        contract.set_epoch(11)
        assert provider.force_flush(timeout_millis=5_000)

        def decode(headers: dict[str, str], body: bytes) -> ExportMetricsServiceRequest:
            if headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            return ExportMetricsServiceRequest.FromString(body)

        assert len(_CaptureHandler.requests) == 2
        requests = [decode(headers, body) for headers, body in _CaptureHandler.requests]
        for request in requests:
            _assert_config_wire_request(request, epoch=11)
        request = requests[0]
        names = {
            metric.name
            for resource_metrics in request.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }
        assert names == {
            mm.REQUESTS_METRIC_NAME,
            mm.REQUEST_DURATION_METRIC_NAME,
            mm.EPOCH_METRIC_NAME,
        }
        assert _CaptureHandler.request_path == "/v1/metrics"
        assert all("_" not in name for name in names)
        duration = next(
            metric
            for resource_metrics in request.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
            if metric.name == mm.REQUEST_DURATION_METRIC_NAME
        )
        assert list(duration.histogram.data_points[0].explicit_bounds) == list(mm.REQUEST_DURATION_BUCKETS_S)
        provider.shutdown()
    finally:
        server.shutdown()
        thread.join(timeout=5)


class _CaptureMetricsService(MetricsServiceServicer):
    def __init__(self) -> None:
        self.request: ExportMetricsServiceRequest | None = None
        self.requests: list[ExportMetricsServiceRequest] = []
        self.received = threading.Event()

    def Export(  # noqa: N802 - generated gRPC service method
        self,
        request: ExportMetricsServiceRequest,
        context: grpc.ServicerContext,
    ) -> ExportMetricsServiceResponse:
        del context
        self.request = request
        self.requests.append(request)
        self.received.set()
        return ExportMetricsServiceResponse()


def test_real_otlp_grpc_export_matches_helm_application_receiver() -> None:
    service = _CaptureMetricsService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_MetricsServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        exporter = mm._build_exporter(f"http://127.0.0.1:{port}", "grpc")
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
        provider = MeterProvider(
            metric_readers=[reader],
            views=mm._metric_views(),
            resource=Resource.create({"service.name": "sie-config", "deployment.environment": "test"}),
        )
        contract = mm.ManagedConfigMetrics(provider.get_meter("grpc-transport-test"))
        contract.record_request(method="GET", route="/v1/configs/epoch", status_code=200, duration_s=0.01)
        contract.set_epoch(12)
        assert provider.force_flush(timeout_millis=5_000)
        contract.record_request(method="GET", route="/v1/configs/epoch", status_code=200, duration_s=0.02)
        contract.set_epoch(12)
        assert provider.force_flush(timeout_millis=5_000)
        assert service.received.wait(timeout=5)
        assert len(service.requests) == 2
        for request in service.requests:
            _assert_config_wire_request(request, epoch=12)
        assert service.request is not None
        names = {
            metric.name
            for resource_metrics in service.request.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
            if metric.name.startswith("sie.")
        }
        assert names == {
            mm.REQUESTS_METRIC_NAME,
            mm.REQUEST_DURATION_METRIC_NAME,
            mm.EPOCH_METRIC_NAME,
        }
        provider.shutdown()
    finally:
        server.stop(grace=0).wait(timeout=5)
