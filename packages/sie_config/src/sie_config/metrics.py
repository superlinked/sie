"""Canonical config-service telemetry facade.

Business code records one semantic event here. The facade owns the dotted
OpenTelemetry instruments; OTLP is the only producer path and is a no-op when
no provider is installed. Prometheus compatibility is provided by the
collector exporter, never by a second in-process registry.
"""

from __future__ import annotations

from typing import Final

from sie_config.managed_metrics import managed_metrics, managed_metrics_enabled

NATS_PUBLISH_SUCCESS: Final = "success"
NATS_PUBLISH_PARTIAL: Final = "partial"
NATS_PUBLISH_FAILURE: Final = "failure"

STORE_OP_WRITE_MODEL: Final = "write_model"
STORE_OP_DELETE_MODEL: Final = "delete_model"
STORE_OP_INCREMENT_EPOCH: Final = "increment_epoch"
STORE_RESULT_SUCCESS: Final = "success"
STORE_RESULT_FAILURE: Final = "failure"


def telemetry_enabled() -> bool:
    """Return whether semantic config events have an active OTel sink."""
    return managed_metrics_enabled()


def set_epoch(epoch: int) -> None:
    """Record the authoritative config epoch."""
    managed_metrics().set_epoch(epoch)


def set_nats_connected(connected: bool) -> None:
    """Record whether the config distribution transport is ready."""
    managed_metrics().set_messaging_ready(connected)


def record_http_request(*, method: str, path: str, status: int, duration_s: float) -> None:
    """Record one completed config HTTP request."""
    managed_metrics().record_request(
        method=method,
        route=path,
        status_code=status,
        duration_s=duration_s,
    )


def record_snapshot_publish(*, success: bool) -> None:
    """Record one authoritative snapshot publication."""
    managed_metrics().record_publish(
        operation="snapshot",
        outcome="success" if success else "failure",
    )


def record_nats_publish(result: str) -> None:
    """Record one delta publication outcome."""
    managed_metrics().record_publish(operation="delta", outcome=result)


def record_store_write(op: str, result: str) -> None:
    """Record one config-store write outcome."""
    managed_metrics().record_store_write(operation=op, outcome=result)


def update_models_gauge(api_count: int, filesystem_count: int) -> None:
    """Publish the complete model-source snapshot."""
    managed_metrics().set_models(source="api", count=api_count)
    managed_metrics().set_models(source="filesystem", count=filesystem_count)
