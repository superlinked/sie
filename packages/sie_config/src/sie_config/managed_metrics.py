"""Bounded OpenTelemetry instruments for the config-service facade.

Every topology uses this one producer path. Export is opt-in and fail-open: an
unavailable collector must never make config reads or writes fail.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from functools import cache
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

import requests
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GrpcMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HttpMetricExporter,
)
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import (
    AlwaysOffExemplarFilter,
    Counter,
    Histogram,
    MeterProvider,
    ObservableCounter,
)
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from sie_sdk.redaction import endpoint_origin_for_log as _endpoint_origin_for_log

logger = logging.getLogger(__name__)

REQUESTS_METRIC_NAME: Final = "sie.config.requests"
REQUEST_DURATION_METRIC_NAME: Final = "sie.config.request.duration"
EPOCH_METRIC_NAME: Final = "sie.config.epoch"
MODELS_METRIC_NAME: Final = "sie.config.models"
PUBLISH_METRIC_NAME: Final = "sie.config.publish"
STORE_WRITES_METRIC_NAME: Final = "sie.config.store.writes"
MESSAGING_READY_METRIC_NAME: Final = "sie.config.messaging.ready"


REQUEST_DURATION_BUCKETS_S: Final = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_PUBLISH_OPERATIONS: Final = frozenset({"delta", "snapshot"})
_OUTCOMES: Final = frozenset({"success", "partial", "failure"})
_STORE_OPERATIONS: Final = frozenset({"write_model", "increment_epoch"})
_MODEL_SOURCES: Final = frozenset({"api", "filesystem"})
_HTTP_METHODS: Final = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
_HTTP_ROUTES: Final = frozenset(
    {
        "/healthz",
        "/readyz",
        "/v1/configs/models",
        "/v1/configs/models/{model_id:path}",
        "/v1/configs/resolve",
        "/v1/configs/bundles",
        "/v1/configs/bundles/{bundle_id}",
        "/v1/configs/epoch",
        "/v1/configs/export",
    }
)
_METER_PROVIDER: MeterProvider | None = None
_UNKNOWN_RESOURCE_VALUE: Final = "unknown"


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _bounded(value: str, allowed: frozenset[str]) -> str:
    return value if value in allowed else "other"


def _resource_attributes() -> dict[str, str]:
    environment = (
        os.environ.get("SIE_OTEL_DEPLOYMENT_ENVIRONMENT", "").strip()
        or os.environ.get("SIE_DEPLOYMENT_ENV", "").strip()
        or _UNKNOWN_RESOURCE_VALUE
    )
    region = (
        os.environ.get("SIE_OTEL_CLOUD_REGION", "").strip()
        or os.environ.get("SIE_CLOUD_REGION", "").strip()
        or _UNKNOWN_RESOURCE_VALUE
    )
    instance_prefix = (
        os.environ.get("SIE_TELEMETRY_INSTANCE_ID", "").strip() or os.environ.get("MODAL_TASK_ID", "").strip()
    )
    instance_id = _compose_service_instance_id(instance_prefix, _process_start_uuid())
    return {
        SERVICE_NAME: "sie-config",
        "service.instance.id": instance_id,
        "deployment.environment": environment,
        "cloud.region": region,
    }


def _compose_service_instance_id(configured_prefix: str | None, process_start_uuid: str) -> str:
    prefix = (configured_prefix or "").strip().rstrip("/")
    return f"{prefix}/{process_start_uuid}" if prefix else process_start_uuid


@cache
def _process_start_uuid() -> str:
    return str(uuid.uuid4())


def _metrics_endpoint(protocol: str) -> str | None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "").strip()
    if endpoint:
        return endpoint
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip().rstrip("/")
    if not base:
        return None
    return base if protocol == "grpc" else f"{base}/v1/metrics"


def _metrics_protocol() -> str:
    raw = (
        os.environ.get("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip()
        or "grpc"
    ).lower()
    if raw == "grpc":
        return "grpc"
    if raw == "http/protobuf":
        return "http"
    raise ValueError(f"unsupported OTLP metrics protocol: {raw!r}")


def _trusted_modal_origin(raw: str, *, origin_only: bool, expected_path: str | None = None) -> str | None:
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host.endswith(".modal.run")
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or (origin_only and parsed.path not in ("", "/"))
        or (expected_path is not None and parsed.path != expected_path)
    ):
        return None
    return f"https://{host}"


def _modal_proxy_headers(endpoint: str) -> dict[str, str]:
    if not _truthy("SIE_MODAL_PROXY_AUTH"):
        return {}
    allowed = _trusted_modal_origin(
        os.environ.get("SIE_OTEL_PROXY_AUTH_ORIGIN", "").strip(),
        origin_only=True,
    )
    actual = _trusted_modal_origin(endpoint, origin_only=False, expected_path="/v1/metrics")
    if allowed is None or actual != allowed:
        raise ValueError("refusing Modal proxy credentials for an untrusted OTLP metrics endpoint")
    token_id = os.environ.get("SIE_MODAL_PROXY_TOKEN_ID", "").strip()
    token_secret = os.environ.get("SIE_MODAL_PROXY_TOKEN_SECRET", "").strip()
    if not token_id or not token_secret:
        raise ValueError("Modal OTLP proxy authentication requires a complete credential pair")
    return {"Modal-Key": token_id, "Modal-Secret": token_secret}


def _build_exporter(endpoint: str, protocol: str) -> Any:
    preferred_temporality = _otlp_metric_temporality()
    if protocol == "grpc":
        if _truthy("SIE_MODAL_PROXY_AUTH"):
            raise ValueError("Modal OTLP proxy authentication requires HTTP transport")
        return GrpcMetricExporter(endpoint=endpoint, preferred_temporality=preferred_temporality)
    if protocol != "http":
        raise ValueError(f"unsupported OTLP metrics protocol: {protocol!r}")
    headers = _modal_proxy_headers(endpoint)
    kwargs: dict[str, Any] = {
        "endpoint": endpoint,
        "preferred_temporality": preferred_temporality,
    }
    if _truthy("SIE_MODAL_PROXY_AUTH"):
        # Stop at the trusted origin's first response; custom credentials can
        # never ride a redirect to another origin.
        session = requests.Session()
        session.max_redirects = 0
        kwargs.update(headers=headers, session=session)
    return HttpMetricExporter(**kwargs)


def _otlp_metric_temporality() -> dict[type, AggregationTemporality]:
    """Pin additive application instruments to OTLP DELTA on the wire."""
    return {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }


def _metric_views() -> list[View]:
    return [
        View(
            instrument_name=REQUEST_DURATION_METRIC_NAME,
            aggregation=ExplicitBucketHistogramAggregation(REQUEST_DURATION_BUCKETS_S),
        )
    ]


class ManagedConfigMetrics:
    """The dotted managed metric contract, backed by an injected OTel meter."""

    def __init__(self, meter: Meter) -> None:
        self.requests = meter.create_counter(
            REQUESTS_METRIC_NAME,
            unit="{request}",
            description="Config HTTP requests",
        )
        self.request_duration = meter.create_histogram(
            REQUEST_DURATION_METRIC_NAME,
            unit="s",
            description="Config HTTP request duration",
        )
        self.epoch = meter.create_gauge(
            EPOCH_METRIC_NAME,
            unit="{epoch}",
            description="Current persisted config epoch",
        )
        self.models = meter.create_gauge(
            MODELS_METRIC_NAME,
            unit="{model}",
            description="Models known to the config registry",
        )
        self.publish = meter.create_counter(
            PUBLISH_METRIC_NAME,
            unit="{operation}",
            description="Config publication attempts",
        )
        self.store_writes = meter.create_counter(
            STORE_WRITES_METRIC_NAME,
            unit="{operation}",
            description="Config-store write attempts",
        )
        self.messaging_ready = meter.create_gauge(
            MESSAGING_READY_METRIC_NAME,
            unit="1",
            description="Whether config distribution messaging is ready",
        )

    def record_request(self, *, method: str, route: str, status_code: int, duration_s: float) -> None:
        attributes = {
            "http.method": _bounded(method, _HTTP_METHODS),
            "http.route": _bounded(route, _HTTP_ROUTES),
            "http.status_code": status_code if 100 <= status_code <= 599 else 0,
        }
        self.requests.add(1, attributes)
        self.request_duration.record(max(duration_s, 0.0), attributes)

    def set_epoch(self, epoch: int) -> None:
        self.epoch.set(max(epoch, 0))

    def set_models(self, *, source: str, count: int) -> None:
        self.models.set(max(count, 0), {"source": _bounded(source, _MODEL_SOURCES)})

    def record_publish(self, *, operation: str, outcome: str) -> None:
        self.publish.add(
            1,
            {
                "operation": _bounded(operation, _PUBLISH_OPERATIONS),
                "outcome": _bounded(outcome, _OUTCOMES),
            },
        )

    def record_store_write(self, *, operation: str, outcome: str) -> None:
        self.store_writes.add(
            1,
            {
                "operation": _bounded(operation, _STORE_OPERATIONS),
                "outcome": _bounded(outcome, _OUTCOMES),
            },
        )

    def set_messaging_ready(self, ready: bool) -> None:
        self.messaging_ready.set(1 if ready else 0, {"transport": "nats"})


class ConfigMetrics(Protocol):
    """Runtime-local implementation behind the semantic config facade."""

    def record_request(self, *, method: str, route: str, status_code: int, duration_s: float) -> None: ...

    def set_epoch(self, epoch: int) -> None: ...

    def set_models(self, *, source: str, count: int) -> None: ...

    def record_publish(self, *, operation: str, outcome: str) -> None: ...

    def record_store_write(self, *, operation: str, outcome: str) -> None: ...

    def set_messaging_ready(self, ready: bool) -> None: ...


class _DisabledConfigMetrics:
    """Label-allocation-free sink used when config telemetry is disabled.

    In particular, this is not an OTel ``NoOpMeter``. Calling an instrument
    created by a no-op meter would still construct the attribute mappings on
    every business event. Keeping the disabled implementation at the semantic
    boundary means lifecycle call sites can remain unconditional without
    constructing SDK instruments or labels.
    """

    __slots__ = ()

    def record_request(self, *, method: str, route: str, status_code: int, duration_s: float) -> None:
        del method, route, status_code, duration_s

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def set_models(self, *, source: str, count: int) -> None:
        del source, count

    def record_publish(self, *, operation: str, outcome: str) -> None:
        del operation, outcome

    def record_store_write(self, *, operation: str, outcome: str) -> None:
        del operation, outcome

    def set_messaging_ready(self, ready: bool) -> None:
        del ready


_DISABLED: Final = _DisabledConfigMetrics()
_MANAGED: ConfigMetrics = _DISABLED


def managed_metrics() -> ConfigMetrics:
    return _MANAGED


def managed_metrics_enabled() -> bool:
    """Return whether the facade is backed by real OTel instruments."""
    return _MANAGED is not _DISABLED


def _metrics_export_interval_ms(raw: str) -> int:
    try:
        interval_ms = int(raw) if raw else 5_000
    except ValueError:
        interval_ms = 5_000
    if interval_ms < 1_000:
        interval_ms = 5_000
    return min(interval_ms, 5_000)


def setup_managed_metrics() -> MeterProvider | None:
    """Install the managed OTLP provider once; return ``None`` when disabled."""
    global _MANAGED, _METER_PROVIDER
    if _METER_PROVIDER is not None:
        return _METER_PROVIDER
    if not _truthy("SIE_METRICS_ENABLED"):
        return None
    try:
        protocol = _metrics_protocol()
        endpoint = _metrics_endpoint(protocol)
        if endpoint is None:
            logger.warning("SIE_METRICS_ENABLED set but no OTLP metrics endpoint; config metrics disabled")
            return None
        interval_ms = _metrics_export_interval_ms(os.environ.get("OTEL_METRIC_EXPORT_INTERVAL", "").strip())
        reader = PeriodicExportingMetricReader(
            _build_exporter(endpoint, protocol),
            export_interval_millis=interval_ms,
        )
        provider = MeterProvider(
            resource=Resource(_resource_attributes()),
            metric_readers=[reader],
            exemplar_filter=AlwaysOffExemplarFilter(),
            shutdown_on_exit=True,
            views=_metric_views(),
        )
        metrics.set_meter_provider(provider)
        _MANAGED = ManagedConfigMetrics(provider.get_meter("sie-config", "1"))
        _METER_PROVIDER = provider
        logger.info(
            "config OTLP metrics initialized (endpoint=%s, protocol=%s, proxy_auth=%s)",
            _endpoint_origin_for_log(endpoint),
            protocol,
            bool(_modal_proxy_headers(endpoint)),
        )
        return provider
    except Exception as error:  # noqa: BLE001 - telemetry setup must never fail config service startup
        logger.warning(
            "config OTLP metrics setup failed; continuing without export (error_type=%s)",
            type(error).__name__,
        )
        return None


def metric_names() -> Mapping[str, str]:
    """Stable inventory seam used by contract tests and deployment audits."""
    return {
        "requests": REQUESTS_METRIC_NAME,
        "request_duration": REQUEST_DURATION_METRIC_NAME,
        "epoch": EPOCH_METRIC_NAME,
        "models": MODELS_METRIC_NAME,
        "publish": PUBLISH_METRIC_NAME,
        "store_writes": STORE_WRITES_METRIC_NAME,
        "messaging_ready": MESSAGING_READY_METRIC_NAME,
    }
