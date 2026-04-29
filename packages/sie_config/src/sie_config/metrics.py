# Prometheus metrics for SIE Config Service.
#
# Why these metrics exist (and no more):
#
# The config service is the control plane's only source of truth for the
# epoch counter and the model catalog. The metrics below are the minimum
# set needed to:
#
# 1. Detect cross-service config drift -- `sie_config_epoch` is the
#    other half of the gateway's `sie_gateway_config_epoch`. The
#    operational alert
#    `(sie_config_epoch - on() sie_gateway_config_epoch) > 0` fires
#    when a gateway lags the control plane for any reason (NATS delta
#    lost, poller stuck, etc.).
# 2. Attribute outages -- `sie_config_http_requests_total` and
#    `sie_config_http_request_duration_seconds` tell us whether a
#    gateway's bootstrap 5xx is on the control-plane side or the
#    gateway's own fetch code. `sie_config_nats_connected` and
#    `sie_config_nats_publishes_total` distinguish "config write
#    succeeded but NATS delta didn't leave the box" from "NATS is down
#    everywhere".
# 3. Catch silent corruption early -- `sie_config_store_writes_total`
#    with `result="failure"` picks up PVC / disk-backend failures that
#    would otherwise only surface on restart.
#
# Cardinality is bounded:
#
# - HTTP `path` label is the FastAPI route template (e.g.
#   `/v1/configs/models/{model_id}`), not the raw URL, so the series
#   count is the number of endpoints, not the number of models.
# - `source` on `sie_config_models_total` is a fixed two-element enum
#   (`api`, `filesystem`).
# - `result` / `op` labels are fixed enums, enumerated by the
#   constants below.
#
# All metrics register into the default `prometheus_client` registry,
# which is what the `/metrics` endpoint exposes.

from __future__ import annotations

import logging
from typing import Final

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# Seconds. Same shape as sie_server's DURATION_BUCKETS so dashboards can
# be reused across services; config API latencies are typically sub-100ms
# (single disk read + registry lookup) with bootstrap export being the
# long tail when the catalog is large.
_DURATION_BUCKETS: Final = (
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


# ---------------------------------------------------------------------------
# HTTP request metrics
#
# Populated by the ASGI middleware in `app_factory.py`. The `path` label
# is the FastAPI route template so per-model reads don't each get their
# own time series. The `status` label is the HTTP status code as a
# string ("200", "404", "503", ...) so PromQL aggregations like
# `sum by (path)(rate(...{status=~"5.."}[5m]))` work directly.
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "sie_config_http_requests_total",
    "Total HTTP requests handled by sie-config",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "sie_config_http_request_duration_seconds",
    "HTTP request duration (wall-clock, end-to-end inside the ASGI app)",
    ["method", "path"],
    buckets=_DURATION_BUCKETS,
)


# ---------------------------------------------------------------------------
# Config plane state
#
# `sie_config_epoch` is the most important metric in this file. It is
# the authoritative epoch (persisted in `ConfigStore`) and is what
# gateway pollers compare their local `sie_gateway_config_epoch`
# against. Populated on startup (from `ConfigStore.read_epoch`) and on
# every successful `increment_epoch`. When no `ConfigStore` is
# configured (standalone / test), it stays at 0, which is correct: a
# gateway with `CONFIG_EPOCH == 0` against `sie_config_epoch == 0` is
# in sync.
# ---------------------------------------------------------------------------

EPOCH = Gauge(
    "sie_config_epoch",
    "Current persisted config epoch (0 if no ConfigStore is configured)",
)

# `source` enum: "api" = added via POST /v1/configs/models (persisted
# in ConfigStore), "filesystem" = seeded from the on-disk bundles/
# models/ directories at startup. The split matters for operations:
# API-added models vanish on PVC loss, filesystem-seeded ones do not.
MODELS_TOTAL = Gauge(
    "sie_config_models_total",
    "Number of models known to the registry, split by origin",
    ["source"],
)


# ---------------------------------------------------------------------------
# NATS publisher
#
# `sie_config_nats_connected` is a simple 0/1 gauge flipped by the
# NATS client callbacks. Matches `sie_gateway_nats_connected` on the
# gateway side — the same NATS core connection feeds both, so a
# disconnect shows up symmetrically on both services.
#
# `sie_config_nats_publishes_total` records one sample per
# `publish_config_notification` call. `result` is enumerated by
# `NATS_PUBLISH_*` constants below. `partial` is emitted when some
# bundle subjects succeeded and others failed — operators need to
# distinguish that from a full failure because the gateway poller
# will close a partial gap within ~30s, but a full failure requires
# NATS-side intervention.
# ---------------------------------------------------------------------------

NATS_CONNECTED = Gauge(
    "sie_config_nats_connected",
    "1 iff sie-config is currently connected to NATS for delta publishing",
)

NATS_PUBLISHES_TOTAL = Counter(
    "sie_config_nats_publishes_total",
    "Config-delta publish attempts to NATS",
    ["result"],
)

NATS_PUBLISH_SUCCESS: Final = "success"
NATS_PUBLISH_PARTIAL: Final = "partial"
NATS_PUBLISH_FAILURE: Final = "failure"


# ---------------------------------------------------------------------------
# ConfigStore persistence
#
# Covers both disk writes (model YAML) and epoch increments. `result`
# is "success" or "failure"; `op` is the enum below. A sustained
# `failure` rate here is a PVC / disk-backend issue and should page.
# ---------------------------------------------------------------------------

STORE_WRITES_TOTAL = Counter(
    "sie_config_store_writes_total",
    "ConfigStore write operations, by op and result",
    ["op", "result"],
)

STORE_OP_WRITE_MODEL: Final = "write_model"
STORE_OP_INCREMENT_EPOCH: Final = "increment_epoch"
STORE_RESULT_SUCCESS: Final = "success"
STORE_RESULT_FAILURE: Final = "failure"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_epoch(epoch: int) -> None:
    """Mirror the authoritative epoch into the gauge."""
    EPOCH.set(epoch)


def set_nats_connected(connected: bool) -> None:
    """Flip the NATS-connected gauge. Centralized so every call site
    uses the same 0/1 convention and we don't have to remember it.
    """
    NATS_CONNECTED.set(1 if connected else 0)


def record_nats_publish(result: str) -> None:
    """Increment `sie_config_nats_publishes_total{result=...}`. Use
    the module-level `NATS_PUBLISH_*` constants for `result` to keep
    the label space enumerable.
    """
    NATS_PUBLISHES_TOTAL.labels(result=result).inc()


def record_store_write(op: str, result: str) -> None:
    """Increment `sie_config_store_writes_total{op=..., result=...}`.
    Use the `STORE_OP_*` and `STORE_RESULT_*` constants to keep the
    label space enumerable.
    """
    STORE_WRITES_TOTAL.labels(op=op, result=result).inc()


def update_models_gauge(api_count: int, filesystem_count: int) -> None:
    """Set both `source` series of `sie_config_models_total`. Called
    from startup and after every successful mutation; cheap because
    `ModelRegistry.list_models()` is an in-memory dict scan.
    """
    MODELS_TOTAL.labels(source="api").set(api_count)
    MODELS_TOTAL.labels(source="filesystem").set(filesystem_count)
