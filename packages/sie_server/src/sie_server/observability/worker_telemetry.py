"""Single-emission OpenTelemetry facade for the Python inference engine.

Business and lifecycle code reports one semantic event here.  This module is
the only Python-engine code allowed to expand that event into the canonical
instruments declared by ``telemetry/contract.yaml``.  Applications export the
result once over OTLP; Prometheus compatibility belongs to the collector.

The facade is a real no-op until a meter is explicitly configured.  That keeps
telemetry-disabled inference free of SDK aggregation work and lets the normal
server and Modal-native runtime supply different OTLP transports without
changing any producer call site.
"""

from __future__ import annotations

import logging
import math
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from numbers import Integral, Real
from threading import Lock
from typing import TYPE_CHECKING, Any, Final, Protocol
from urllib.parse import urlsplit

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

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig

logger = logging.getLogger(__name__)


def _endpoint_origin_for_log(endpoint: str) -> str:
    """Return a credential- and query-free endpoint origin for diagnostics."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return "<redacted>"
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return "<redacted>"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}{f':{port}' if port is not None else ''}"


SERVICE_NAME_VALUE: Final = "sie-worker"
INSTRUMENTATION_SCOPE: Final = "sie_server.worker"
INSTRUMENTATION_VERSION: Final = "1"
UNKNOWN_RESOURCE_VALUE: Final = "unknown"
EXPORT_INTERVAL_MS: Final = 5_000
EXPORT_TIMEOUT_S: Final = 3

QUEUE_DURATION_METRIC_NAME: Final = "sie.worker.queue.duration"
QUEUE_DEPTH_METRIC_NAME: Final = "sie.worker.queue.depth"
BATCH_SIZE_METRIC_NAME: Final = "sie.worker.batch.size"
BATCH_COST_METRIC_NAME: Final = "sie.worker.batch.cost"
BATCH_FILL_RATIO_METRIC_NAME: Final = "sie.worker.batch.fill_ratio"
QUEUE_PENDING_AT_DISPATCH_METRIC_NAME: Final = "sie.worker.queue.pending_at_dispatch"
RUNTIME_BATCH_SIZE_METRIC_NAME: Final = "sie.worker.runtime.batch.size"
RUNTIME_BATCH_SUBGROUPS_METRIC_NAME: Final = "sie.worker.runtime.batch.subgroups"
RUNTIME_SUBGROUP_SIZE_METRIC_NAME: Final = "sie.worker.runtime.subgroup.size"
REQUESTS_METRIC_NAME: Final = "sie.worker.requests"
REQUEST_DURATION_METRIC_NAME: Final = "sie.worker.request.duration"
INFERENCE_DURATION_METRIC_NAME: Final = "sie.worker.inference.duration"
UNITS_METRIC_NAME: Final = "sie.worker.units"
MODEL_LOADED_METRIC_NAME: Final = "sie.worker.model.loaded"
MODEL_LOAD_DURATION_METRIC_NAME: Final = "sie.worker.model.load.duration"
MODEL_MEMORY_METRIC_NAME: Final = "sie.worker.model.memory"
OOM_RECOVERIES_METRIC_NAME: Final = "sie.worker.oom.recoveries"
MODEL_EVICTIONS_METRIC_NAME: Final = "sie.worker.model.evictions"
ADAPTIVE_WAIT_METRIC_NAME: Final = "sie.worker.scheduler.adaptive.wait"
ADAPTIVE_COST_METRIC_NAME: Final = "sie.worker.scheduler.adaptive.cost"
ADAPTIVE_P50_METRIC_NAME: Final = "sie.worker.scheduler.adaptive.p50"
STARVATION_RESETS_METRIC_NAME: Final = "sie.worker.scheduler.starvation.resets"
GENERATION_TTFT_METRIC_NAME: Final = "sie.worker.generation.ttft"
GENERATION_TPOT_METRIC_NAME: Final = "sie.worker.generation.tpot"
GENERATION_TOKENS_METRIC_NAME: Final = "sie.worker.generation.tokens"
GENERATION_INFLIGHT_METRIC_NAME: Final = "sie.worker.generation.inflight"
GENERATION_KV_RESERVED_METRIC_NAME: Final = "sie.worker.generation.kv.reserved"
GENERATION_KV_BUDGET_METRIC_NAME: Final = "sie.worker.generation.kv.budget"
GENERATION_ADMISSION_METRIC_NAME: Final = "sie.worker.generation.admission.decisions"
GENERATION_DUPLICATE_PREVENTED_METRIC_NAME: Final = "sie.worker.generation.duplicate_prevented"
GRAMMAR_COMPILE_DURATION_METRIC_NAME: Final = "sie.worker.generation.grammar.compile.duration"
GRAMMAR_CACHE_LOOKUPS_METRIC_NAME: Final = "sie.worker.generation.grammar.cache.lookups"
GRAMMAR_REQUESTS_METRIC_NAME: Final = "sie.worker.generation.grammar.requests"

QUEUE_DURATION_BUCKETS_S: Final = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
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
BATCH_SIZE_BUCKETS: Final = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
BATCH_COST_BUCKETS: Final = (
    128.0,
    512.0,
    1024.0,
    2048.0,
    4096.0,
    8192.0,
    16384.0,
    32768.0,
    65536.0,
    131072.0,
    262144.0,
)
PENDING_ITEMS_BUCKETS: Final = (0.0, *BATCH_SIZE_BUCKETS, 1024.0)
REQUEST_DURATION_BUCKETS_S: Final = (*QUEUE_DURATION_BUCKETS_S, 60.0, 120.0, 300.0)
INFERENCE_DURATION_BUCKETS_S: Final = REQUEST_DURATION_BUCKETS_S
MODEL_LOAD_DURATION_BUCKETS_S: Final = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    900.0,
)
TTFT_TPOT_BUCKETS_S: Final = (
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
    60.0,
    120.0,
    300.0,
)

_OTHER: Final = "other"
_MAX_DIMENSION_LEN: Final = 256
_MAX_CATALOG_MODEL_PROFILE_PAIRS: Final = 256
_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})
_OPERATIONS: Final = frozenset({"encode", "score", "extract", "embeddings", "moderations", "generate", "other"})
_WORKER_OUTCOMES: Final = frozenset({"success", "error", "retry", "cancelled", "other"})
_INFERENCE_PHASES: Final = frozenset({"tokenization", "inference", "postprocessing"})
_UNIT_TYPES: Final = ("input_tokens", "pages", "images")
_LOAD_OUTCOMES: Final = frozenset({"success", "error", "timeout", "other"})
_LOAD_STAGES: Final = frozenset({"total", "instantiate", "load", "resident", "other"})
_OOM_STRATEGIES: Final = frozenset({"cache_clear", "evict_lru", "split_batch", "other"})
_OOM_OUTCOMES: Final = frozenset({"success", "failed", "terminal", "other"})
_EVICTION_REASONS: Final = frozenset(
    {
        "preload_pressure",
        "load_oom",
        "manual",
        "shutdown",
        "config_change",
        "oom_recovery",
        "idle",
        "memory_pressure",
        "other",
    }
)
_P50_KINDS: Final = frozenset({"observed", "target"})
_GRAMMAR_MODES: Final = frozenset({"none", "json_schema", "regex", "ebnf", _OTHER})
_GRAMMAR_BACKENDS: Final = frozenset({"outlines", "xgrammar", "llguidance", _OTHER})
_GRAMMAR_PHASES: Final = frozenset({"request", "prewarm", _OTHER})
_GRAMMAR_OUTCOMES: Final = frozenset({"success", "error", "timeout", _OTHER})
_CACHE_RESULTS: Final = frozenset({"hit", "miss", _OTHER})
_ADMISSION_OUTCOMES: Final = frozenset({"admitted", "rejected", "error", _OTHER})
_ADMISSION_REASONS: Final = frozenset({"none", "kv_budget", "resolver_error", "missing_budget", _OTHER})
_DUPLICATE_PATHS: Final = frozenset({"first_chunk_fallback", _OTHER})
_FLUSH_REASONS: Final = frozenset(
    {"cost_cap", "count_cap", "timeout", "coalesce", "single_oversize", "idle_bypass", "drain", "other"}
)

_METER_PROVIDER: MeterProvider | None = None


@dataclass(frozen=True)
class _WorkerMetricContext:
    lane: str
    profiles: Mapping[str, frozenset[str]]
    backends: Mapping[tuple[str, str], str]
    aliases: Mapping[str, tuple[str, str]]
    overflow_pairs: frozenset[tuple[str, str]]


_context = _WorkerMetricContext(lane=_OTHER, profiles={}, backends={}, aliases={}, overflow_pairs=frozenset())
_catalog_admission_lock = Lock()
# Python's synchronous OTel aggregators retain every attribute set they have
# observed. Keep admission across catalog replacements so a long-lived worker
# cannot create unbounded model/profile streams through configuration churn.
_admitted_catalog_pairs: set[tuple[str, str]] = set()
_catalog_collapse_warning_emitted = False


class WorkerTelemetryFacade(Protocol):
    def item_completed(
        self,
        *,
        operation: object,
        outcome: object,
        model: object,
        profile: object,
        duration_s: object,
        item_count: object = 1,
        tokenization_s: object | None = None,
        inference_s: object | None = None,
        postprocessing_s: object | None = None,
        units: Mapping[str, int] | None = None,
    ) -> None: ...

    def queue_released(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        duration_s: object,
    ) -> None: ...

    def queue_depth_changed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        depth: object,
    ) -> None: ...

    def batch_formed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        size: object,
        cost: object | None = None,
        capacity: object | None = None,
        flush_reason: object = "other",
    ) -> None: ...

    def queue_pending_observed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        pending: object,
    ) -> None: ...

    def runtime_batch_dispatched(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        total_items: object,
        subgroup_sizes: Iterable[int],
    ) -> None: ...

    def model_residency_changed(
        self,
        *,
        model: object,
        loaded: bool,
        memory_bytes: object | None,
        profile: object = "default",
    ) -> None: ...

    def model_load_completed(
        self,
        *,
        model: object,
        duration_s: object,
        outcome: object,
        stage: object = "total",
        profile: object = "default",
    ) -> None: ...

    def oom_recovery_completed(
        self,
        *,
        model: object,
        strategy: object,
        outcome: object,
        profile: object = "default",
    ) -> None: ...

    def model_evicted(
        self,
        *,
        model: object,
        reason: object,
        profile: object = "default",
    ) -> None: ...

    def adaptive_snapshot(
        self,
        *,
        model: object,
        profile: object,
        wait_ms: object,
        cost: object,
        observed_p50_ms: object | None,
        target_p50_ms: object | None,
        starvation_resets_delta: object,
    ) -> None: ...

    def first_token_observed(self, *, model: object, grammar: object, duration_s: object) -> None: ...

    def stream_finished(
        self,
        *,
        model: object,
        grammar: object,
        tpot_s: object | None,
        prompt_tokens: object | None,
        completion_tokens: object | None,
    ) -> None: ...

    def state_changed(
        self,
        *,
        model: object,
        reserved_tokens: object,
        inflight: object,
        budget_tokens: object | None,
    ) -> None: ...

    def admission_decided(self, *, model: object, outcome: object, reason: object) -> None: ...

    def duplicate_prevented(self, *, model: object, path: object) -> None: ...

    def grammar_requested(self, *, model: object, backend: object, grammar: object) -> None: ...

    def grammar_cache_lookup(
        self,
        *,
        model: object,
        backend: object,
        grammar: object,
        phase: object,
        result: object,
    ) -> None: ...

    def grammar_compile_completed(
        self,
        *,
        model: object,
        backend: object,
        grammar: object,
        phase: object,
        outcome: object,
        duration_s: object,
    ) -> None: ...


class _NoopWorkerTelemetry:
    """Zero-work provider used until OTLP metrics are explicitly enabled."""

    def item_completed(self, **_: Any) -> None:
        return

    def queue_released(self, **_: Any) -> None:
        return

    def queue_depth_changed(self, **_: Any) -> None:
        return

    def batch_formed(self, **_: Any) -> None:
        return

    def queue_pending_observed(self, **_: Any) -> None:
        return

    def runtime_batch_dispatched(self, **_: Any) -> None:
        return

    def model_residency_changed(self, **_: Any) -> None:
        return

    def model_load_completed(self, **_: Any) -> None:
        return

    def oom_recovery_completed(self, **_: Any) -> None:
        return

    def model_evicted(self, **_: Any) -> None:
        return

    def adaptive_snapshot(self, **_: Any) -> None:
        return

    def first_token_observed(self, **_: Any) -> None:
        return

    def stream_finished(self, **_: Any) -> None:
        return

    def state_changed(self, **_: Any) -> None:
        return

    def admission_decided(self, **_: Any) -> None:
        return

    def duplicate_prevented(self, **_: Any) -> None:
        return

    def grammar_requested(self, **_: Any) -> None:
        return

    def grammar_cache_lookup(self, **_: Any) -> None:
        return

    def grammar_compile_completed(self, **_: Any) -> None:
        return


class WorkerTelemetry:
    """Canonical engine instruments backed by one injected OTel meter."""

    def __init__(self, meter: Meter) -> None:
        self._queue_duration = meter.create_histogram(
            QUEUE_DURATION_METRIC_NAME,
            unit="s",
            description="Time a Modal-native inference item waited before engine execution",
        )
        self._queue_depth = meter.create_gauge(
            QUEUE_DEPTH_METRIC_NAME,
            unit="{item}",
            description="Authoritative queued-but-not-running items at a worker queue owner",
        )
        self._batch_size = meter.create_histogram(
            BATCH_SIZE_METRIC_NAME,
            unit="{item}",
            description="Items in a Modal-native engine-owned batch",
        )
        self._batch_cost = meter.create_histogram(
            BATCH_COST_METRIC_NAME,
            unit="{cost}",
            description="Authoritative scheduling cost in a formed engine batch",
        )
        self._batch_fill_ratio = meter.create_gauge(
            BATCH_FILL_RATIO_METRIC_NAME,
            unit="1",
            description="Authoritative batch cost divided by configured capacity",
        )
        self._queue_pending_at_dispatch = meter.create_histogram(
            QUEUE_PENDING_AT_DISPATCH_METRIC_NAME,
            unit="{item}",
            description="Items still pending after an engine batch is selected",
        )
        self._runtime_batch_size = meter.create_histogram(
            RUNTIME_BATCH_SIZE_METRIC_NAME,
            unit="{item}",
            description="Items in one engine execution batch",
        )
        self._runtime_batch_subgroups = meter.create_histogram(
            RUNTIME_BATCH_SUBGROUPS_METRIC_NAME,
            unit="{subgroup}",
            description="GPU forward subgroups produced from one execution batch",
        )
        self._runtime_subgroup_size = meter.create_histogram(
            RUNTIME_SUBGROUP_SIZE_METRIC_NAME,
            unit="{item}",
            description="Items in each GPU forward subgroup produced from an execution batch",
        )
        self._requests = meter.create_counter(
            REQUESTS_METRIC_NAME,
            unit="{item}",
            description="Inference items completed by the engine",
        )
        self._request_duration = meter.create_histogram(
            REQUEST_DURATION_METRIC_NAME,
            unit="s",
            description="End-to-end engine processing time per inference item",
        )
        self._inference_duration = meter.create_histogram(
            INFERENCE_DURATION_METRIC_NAME,
            unit="s",
            description="Per-item engine processing time split into bounded phases",
        )
        self._units = meter.create_counter(
            UNITS_METRIC_NAME,
            unit="{unit}",
            description="Authoritative non-generation units completed by the engine",
        )
        self._model_loaded = meter.create_gauge(
            MODEL_LOADED_METRIC_NAME,
            unit="{model}",
            description="Whether a catalog model is resident in the engine",
        )
        self._model_load_duration = meter.create_histogram(
            MODEL_LOAD_DURATION_METRIC_NAME,
            unit="s",
            description="Model load duration",
        )
        self._model_memory = meter.create_gauge(
            MODEL_MEMORY_METRIC_NAME,
            unit="By",
            description="Authoritative adapter-reported resident model memory",
        )
        self._oom_recoveries = meter.create_counter(
            OOM_RECOVERIES_METRIC_NAME,
            unit="{recovery}",
            description="Completed OOM recovery strategy attempts",
        )
        self._model_evictions = meter.create_counter(
            MODEL_EVICTIONS_METRIC_NAME,
            unit="{model}",
            description="Resident models evicted from the engine by bounded cause",
        )
        self._adaptive_wait = meter.create_gauge(
            ADAPTIVE_WAIT_METRIC_NAME,
            unit="s",
            description="Current engine-owned adaptive scheduler wait ceiling",
        )
        self._adaptive_cost = meter.create_gauge(
            ADAPTIVE_COST_METRIC_NAME,
            unit="{cost}",
            description="Current engine-owned adaptive scheduler batch cost ceiling",
        )
        self._adaptive_p50 = meter.create_gauge(
            ADAPTIVE_P50_METRIC_NAME,
            unit="s",
            description="Observed and target p50 latency used by the adaptive scheduler",
        )
        self._starvation_resets = meter.create_counter(
            STARVATION_RESETS_METRIC_NAME,
            unit="{reset}",
            description="Adaptive scheduler starvation recoveries",
        )
        self._generation_ttft = meter.create_histogram(
            GENERATION_TTFT_METRIC_NAME,
            unit="s",
            description="Adapter-observed generation time to first non-empty token",
        )
        self._generation_tpot = meter.create_histogram(
            GENERATION_TPOT_METRIC_NAME,
            unit="s",
            description="Adapter-observed mean time per output token",
        )
        self._generation_tokens = meter.create_counter(
            GENERATION_TOKENS_METRIC_NAME,
            unit="{token}",
            description="Generation tokens reported by the authoritative engine",
        )
        self._generation_inflight = meter.create_gauge(
            GENERATION_INFLIGHT_METRIC_NAME,
            unit="{request}",
            description="Generation requests currently holding a KV reservation",
        )
        self._generation_kv_reserved = meter.create_gauge(
            GENERATION_KV_RESERVED_METRIC_NAME,
            unit="{token}",
            description="KV-cache tokens reserved by in-flight generation requests",
        )
        self._generation_kv_budget = meter.create_gauge(
            GENERATION_KV_BUDGET_METRIC_NAME,
            unit="{token}",
            description="Effective KV-cache admission budget",
        )
        self._generation_admission = meter.create_counter(
            GENERATION_ADMISSION_METRIC_NAME,
            unit="{decision}",
            description="Generation admission decisions and resolver failures",
        )
        self._generation_duplicate_prevented = meter.create_counter(
            GENERATION_DUPLICATE_PREVENTED_METRIC_NAME,
            unit="{request}",
            description="Generation duplicate executions prevented before decode",
        )
        self._grammar_compile_duration = meter.create_histogram(
            GRAMMAR_COMPILE_DURATION_METRIC_NAME,
            unit="s",
            description="Grammar preparation duration by bounded phase and outcome",
        )
        self._grammar_cache_lookups = meter.create_counter(
            GRAMMAR_CACHE_LOOKUPS_METRIC_NAME,
            unit="{lookup}",
            description="Grammar cache lookup results",
        )
        self._grammar_requests = meter.create_counter(
            GRAMMAR_REQUESTS_METRIC_NAME,
            unit="{request}",
            description="Structured-output generation requests by grammar backend and kind",
        )

    def item_completed(
        self,
        *,
        operation: object,
        outcome: object,
        model: object,
        profile: object,
        duration_s: object,
        item_count: object = 1,
        tokenization_s: object | None = None,
        inference_s: object | None = None,
        postprocessing_s: object | None = None,
        units: Mapping[str, int] | None = None,
    ) -> None:
        """Expand one completion event into its request, phase, and unit instruments."""
        count = _positive_int(item_count)
        if count is None:
            return
        attributes = _request_attributes(
            operation=operation,
            outcome=outcome,
            model=model,
            profile=profile,
        )
        self._requests.add(count, attributes)
        if (duration := _nonnegative_float(duration_s)) is not None:
            for _ in range(count):
                self._request_duration.record(duration, attributes)
        for phase, value in (
            ("tokenization", tokenization_s),
            ("inference", inference_s),
            ("postprocessing", postprocessing_s),
        ):
            if (phase_duration := _nonnegative_float(value)) is not None:
                phase_attributes = {**attributes, "phase": _enum(phase, _INFERENCE_PHASES)}
                for _ in range(count):
                    self._inference_duration.record(phase_duration, phase_attributes)

        if attributes["outcome"] != "success" or units is None:
            return
        unit_attributes = {key: value for key, value in attributes.items() if key != "outcome"}
        for unit_type in _UNIT_TYPES:
            value = _positive_int(units.get(unit_type))
            if value is not None:
                self._units.add(value, {**unit_attributes, "unit.type": unit_type})

    def queue_released(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        duration_s: object,
    ) -> None:
        """Record an authoritative Modal-native queue release."""
        duration = _nonnegative_float(duration_s)
        if duration is None:
            return
        lane, bounded_model, bounded_profile, _ = _dimensions(model, profile)
        self._queue_duration.record(
            duration,
            {
                "operation": _enum(operation, _OPERATIONS),
                "lane": lane,
                "model": bounded_model,
                "profile": bounded_profile,
            },
        )

    def batch_formed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        size: object,
        cost: object | None = None,
        capacity: object | None = None,
        flush_reason: object = "other",
    ) -> None:
        """Record an authoritative Modal-native batch formation."""
        bounded_size = _positive_int(size)
        if bounded_size is None:
            return
        lane, bounded_model, bounded_profile, _ = _dimensions(model, profile)
        core_attributes = {
            "operation": _enum(operation, _OPERATIONS),
            "lane": lane,
            "model": bounded_model,
            "profile": bounded_profile,
        }
        self._batch_size.record(bounded_size, core_attributes)
        bounded_cost = _nonnegative_int(cost)
        if bounded_cost is not None:
            self._batch_cost.record(bounded_cost, core_attributes)
            bounded_capacity = _positive_int(capacity)
            if bounded_capacity is not None:
                self._batch_fill_ratio.set(
                    bounded_cost / bounded_capacity,
                    {
                        **core_attributes,
                        "flush.reason": _enum(flush_reason, _FLUSH_REASONS),
                    },
                )

    def queue_pending_observed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        pending: object,
    ) -> None:
        bounded_pending = _nonnegative_int(pending)
        if bounded_pending is None:
            return
        lane, bounded_model, bounded_profile, _ = _dimensions(model, profile)
        self._queue_pending_at_dispatch.record(
            bounded_pending,
            {
                "operation": _enum(operation, _OPERATIONS),
                "lane": lane,
                "model": bounded_model,
                "profile": bounded_profile,
            },
        )

    def runtime_batch_dispatched(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        total_items: object,
        subgroup_sizes: Iterable[int],
    ) -> None:
        bounded_total = _positive_int(total_items)
        if bounded_total is None:
            return
        attributes = _runtime_attributes(operation=operation, model=model, profile=profile)
        bounded_subgroups = [size for raw in subgroup_sizes if (size := _positive_int(raw)) is not None]
        self._runtime_batch_size.record(bounded_total, attributes)
        self._runtime_batch_subgroups.record(len(bounded_subgroups), attributes)
        for size in bounded_subgroups:
            self._runtime_subgroup_size.record(size, attributes)

    def queue_depth_changed(
        self,
        *,
        operation: object,
        model: object,
        profile: object,
        depth: object,
    ) -> None:
        """Record an authoritative depth snapshot, including explicit zero.

        The Modal engine currently has no call site for this event: Modal's
        queued-but-unassigned backlog is available only through its external
        function-stats API, not inside ``run_batch_bytes``. Keeping the typed
        event here completes the conditional-owner contract without guessing
        depth from a batch that is already executing.
        """
        bounded_depth = _nonnegative_int(depth)
        if bounded_depth is None:
            return
        lane, bounded_model, bounded_profile, _ = _dimensions(model, profile)
        self._queue_depth.set(
            bounded_depth,
            {
                "operation": _enum(operation, _OPERATIONS),
                "lane": lane,
                "model": bounded_model,
                "profile": bounded_profile,
            },
        )

    def model_residency_changed(
        self,
        *,
        model: object,
        loaded: bool,
        memory_bytes: object | None,
        profile: object = "default",
    ) -> None:
        """Record one authoritative residency transition and optional memory value."""
        attributes = _model_attributes(model, profile)
        self._model_loaded.set(1 if loaded else 0, attributes)
        if (memory := _nonnegative_int(memory_bytes)) is not None:
            self._model_memory.set(memory, attributes)

    def model_load_completed(
        self,
        *,
        model: object,
        duration_s: object,
        outcome: object,
        stage: object = "total",
        profile: object = "default",
    ) -> None:
        duration = _nonnegative_float(duration_s)
        if duration is None:
            return
        self._model_load_duration.record(
            duration,
            {
                "outcome": _enum(outcome, _LOAD_OUTCOMES),
                "stage": _enum(stage, _LOAD_STAGES),
                **_model_attributes(model, profile),
            },
        )

    def oom_recovery_completed(
        self,
        *,
        model: object,
        strategy: object,
        outcome: object,
        profile: object = "default",
    ) -> None:
        self._oom_recoveries.add(
            1,
            {
                "strategy": _enum(strategy, _OOM_STRATEGIES),
                "outcome": _enum(outcome, _OOM_OUTCOMES),
                **_model_attributes(model, profile),
            },
        )

    def model_evicted(
        self,
        *,
        model: object,
        reason: object,
        profile: object = "default",
    ) -> None:
        self._model_evictions.add(
            1,
            {"reason": _enum(reason, _EVICTION_REASONS), **_model_attributes(model, profile)},
        )

    def adaptive_snapshot(
        self,
        *,
        model: object,
        profile: object,
        wait_ms: object,
        cost: object,
        observed_p50_ms: object | None,
        target_p50_ms: object | None,
        starvation_resets_delta: object,
    ) -> None:
        attributes = _model_attributes(model, profile)
        if (wait := _nonnegative_float(wait_ms)) is not None:
            self._adaptive_wait.set(wait / 1_000.0, attributes)
        if (bounded_cost := _nonnegative_int(cost)) is not None:
            self._adaptive_cost.set(bounded_cost, attributes)
        for kind, value in (("observed", observed_p50_ms), ("target", target_p50_ms)):
            if (p50 := _nonnegative_float(value)) is not None:
                self._adaptive_p50.set(
                    p50 / 1_000.0,
                    {**attributes, "kind": _enum(kind, _P50_KINDS)},
                )
        if (resets := _positive_int(starvation_resets_delta)) is not None:
            self._starvation_resets.add(resets, attributes)

    def first_token_observed(self, *, model: object, grammar: object, duration_s: object) -> None:
        if (duration := _nonnegative_float(duration_s)) is not None:
            self._generation_ttft.record(duration, self._generation_stream_attributes(model, grammar))

    def stream_finished(
        self,
        *,
        model: object,
        grammar: object,
        tpot_s: object | None,
        prompt_tokens: object | None,
        completion_tokens: object | None,
    ) -> None:
        attributes = self._generation_stream_attributes(model, grammar)
        if (duration := _nonnegative_float(tpot_s)) is not None:
            self._generation_tpot.record(duration, attributes)
        for token_type, raw_value in (("prompt", prompt_tokens), ("completion", completion_tokens)):
            if (value := _positive_int(raw_value)) is not None:
                self._generation_tokens.add(value, {**attributes, "token.type": token_type})

    def state_changed(
        self,
        *,
        model: object,
        reserved_tokens: object,
        inflight: object,
        budget_tokens: object | None,
    ) -> None:
        attributes = _model_attributes(model, "default")
        if (reserved := _nonnegative_int(reserved_tokens)) is not None:
            self._generation_kv_reserved.set(reserved, attributes)
        if (active := _nonnegative_int(inflight)) is not None:
            self._generation_inflight.set(active, attributes)
        if (budget := _positive_int(budget_tokens)) is not None:
            self._generation_kv_budget.set(budget, attributes)

    def admission_decided(self, *, model: object, outcome: object, reason: object) -> None:
        self._generation_admission.add(
            1,
            {
                **_model_attributes(model, "default"),
                "outcome": _enum(outcome, _ADMISSION_OUTCOMES),
                "reason": _enum(reason, _ADMISSION_REASONS),
            },
        )

    def duplicate_prevented(self, *, model: object, path: object) -> None:
        self._generation_duplicate_prevented.add(
            1,
            {
                **_model_attributes(model, "default"),
                "dispatch.path": _enum(path, _DUPLICATE_PATHS),
            },
        )

    def grammar_requested(self, *, model: object, backend: object, grammar: object) -> None:
        self._grammar_requests.add(1, self._grammar_attributes(model, backend, grammar))

    def grammar_cache_lookup(
        self,
        *,
        model: object,
        backend: object,
        grammar: object,
        phase: object,
        result: object,
    ) -> None:
        self._grammar_cache_lookups.add(
            1,
            {
                **self._grammar_attributes(model, backend, grammar),
                "phase": _enum(phase, _GRAMMAR_PHASES),
                "result": _enum(result, _CACHE_RESULTS),
            },
        )

    def grammar_compile_completed(
        self,
        *,
        model: object,
        backend: object,
        grammar: object,
        phase: object,
        outcome: object,
        duration_s: object,
    ) -> None:
        if (duration := _nonnegative_float(duration_s)) is None:
            return
        self._grammar_compile_duration.record(
            duration,
            {
                **self._grammar_attributes(model, backend, grammar),
                "phase": _enum(phase, _GRAMMAR_PHASES),
                "outcome": _enum(outcome, _GRAMMAR_OUTCOMES),
            },
        )

    @staticmethod
    def _generation_stream_attributes(model: object, grammar: object) -> dict[str, str]:
        return {**_model_attributes(model, "default"), "grammar": _enum(grammar, _GRAMMAR_MODES)}

    @staticmethod
    def _grammar_attributes(model: object, backend: object, grammar: object) -> dict[str, str]:
        return {
            **_model_attributes(model, "default"),
            "grammar.backend": _enum(backend, _GRAMMAR_BACKENDS),
            "grammar": _enum(grammar, _GRAMMAR_MODES),
        }


_NOOP = _NoopWorkerTelemetry()
_TELEMETRY: WorkerTelemetryFacade = _NOOP


def worker_telemetry() -> WorkerTelemetryFacade:
    """Return the active semantic facade (a no-op when metrics are disabled)."""
    return _TELEMETRY


def configure_worker_telemetry(meter: Meter | None) -> None:
    """Bind the facade to ``meter`` or atomically return it to no-op mode."""
    global _TELEMETRY
    _TELEMETRY = WorkerTelemetry(meter) if meter is not None else _NOOP


def worker_telemetry_enabled() -> bool:
    return _TELEMETRY is not _NOOP


def setup_worker_telemetry() -> MeterProvider | None:
    """Configure the ordinary server's OTLP provider once, failing open."""
    global _METER_PROVIDER
    if _METER_PROVIDER is not None:
        return _METER_PROVIDER
    if not _truthy("SIE_METRICS_ENABLED"):
        configure_worker_telemetry(None)
        return None

    try:
        protocol = _metrics_protocol()
        endpoint = _metrics_endpoint(protocol)
        if endpoint is None:
            logger.warning("SIE_METRICS_ENABLED set but no OTLP metrics endpoint; worker metrics disabled")
            configure_worker_telemetry(None)
            return None
        exporter = _build_metric_exporter(endpoint, protocol)
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=EXPORT_INTERVAL_MS,
            export_timeout_millis=EXPORT_TIMEOUT_S * 1_000,
        )
        provider = MeterProvider(
            resource=Resource(worker_resource_attributes()),
            metric_readers=[reader],
            exemplar_filter=AlwaysOffExemplarFilter(),
            shutdown_on_exit=False,
            views=metric_views(),
        )
        configure_worker_telemetry(provider.get_meter(INSTRUMENTATION_SCOPE, INSTRUMENTATION_VERSION))
        _METER_PROVIDER = provider
        logger.info(
            "worker OTLP metrics initialized (endpoint=%s, protocol=%s, interval_ms=%d)",
            _endpoint_origin_for_log(endpoint),
            protocol,
            EXPORT_INTERVAL_MS,
        )
        return provider
    except Exception as error:  # noqa: BLE001 - telemetry setup must never fail inference startup
        logger.warning(
            "worker OTLP metrics setup failed; continuing without export (error_type=%s)",
            type(error).__name__,
        )
        configure_worker_telemetry(None)
        return None


def shutdown_worker_telemetry() -> None:
    """Stop accepting observations before shutting down the owned provider."""
    global _METER_PROVIDER
    provider = _METER_PROVIDER
    configure_worker_telemetry(None)
    _METER_PROVIDER = None
    if provider is None:
        return
    try:
        provider.shutdown(timeout_millis=EXPORT_TIMEOUT_S * 1_000)
    except Exception as error:  # noqa: BLE001 - telemetry teardown is best-effort
        logger.warning("worker OTLP metrics shutdown failed (error_type=%s)", type(error).__name__)


def worker_resource_attributes() -> dict[str, str]:
    """Resource identity shared by worker traces and metrics."""
    deployment_environment = (
        _clean_env("SIE_OTEL_DEPLOYMENT_ENVIRONMENT") or _clean_env("SIE_DEPLOYMENT_ENV") or UNKNOWN_RESOURCE_VALUE
    )
    cloud_region = (
        _clean_env("SIE_OTEL_CLOUD_REGION")
        or _clean_env("SIE_CLOUD_REGION")
        or _clean_env("AWS_REGION")
        or _clean_env("AWS_DEFAULT_REGION")
        or UNKNOWN_RESOURCE_VALUE
    )
    return {
        SERVICE_NAME: SERVICE_NAME_VALUE,
        "service.instance.id": service_instance_id(),
        "deployment.environment": deployment_environment,
        "cloud.region": cloud_region,
    }


def service_instance_id() -> str:
    """Stable substrate prefix plus a UUID unique to this process start."""
    configured_prefix = _clean_env("SIE_TELEMETRY_INSTANCE_ID") or _clean_env("MODAL_TASK_ID")
    return _compose_service_instance_id(configured_prefix, _process_start_uuid())


def _compose_service_instance_id(configured_prefix: str | None, process_start_uuid: str) -> str:
    prefix = (configured_prefix or "").strip().rstrip("/")
    return f"{prefix}/{process_start_uuid}" if prefix else process_start_uuid


@cache
def _process_start_uuid() -> str:
    """Generate lazily so Modal snapshot restore does not clone an eager UUID."""
    return str(uuid.uuid4())


def metric_views() -> list[View]:
    schemas = (
        (QUEUE_DURATION_METRIC_NAME, QUEUE_DURATION_BUCKETS_S),
        (BATCH_SIZE_METRIC_NAME, BATCH_SIZE_BUCKETS),
        (BATCH_COST_METRIC_NAME, BATCH_COST_BUCKETS),
        (QUEUE_PENDING_AT_DISPATCH_METRIC_NAME, PENDING_ITEMS_BUCKETS),
        (RUNTIME_BATCH_SIZE_METRIC_NAME, BATCH_SIZE_BUCKETS),
        (RUNTIME_BATCH_SUBGROUPS_METRIC_NAME, BATCH_SIZE_BUCKETS),
        (RUNTIME_SUBGROUP_SIZE_METRIC_NAME, BATCH_SIZE_BUCKETS),
        (REQUEST_DURATION_METRIC_NAME, REQUEST_DURATION_BUCKETS_S),
        (INFERENCE_DURATION_METRIC_NAME, INFERENCE_DURATION_BUCKETS_S),
        (MODEL_LOAD_DURATION_METRIC_NAME, MODEL_LOAD_DURATION_BUCKETS_S),
        (GENERATION_TTFT_METRIC_NAME, TTFT_TPOT_BUCKETS_S),
        (GENERATION_TPOT_METRIC_NAME, TTFT_TPOT_BUCKETS_S),
        (GRAMMAR_COMPILE_DURATION_METRIC_NAME, TTFT_TPOT_BUCKETS_S),
    )
    return [
        View(
            instrument_name=name,
            aggregation=ExplicitBucketHistogramAggregation(boundaries),
        )
        for name, boundaries in schemas
    ]


def metric_names() -> frozenset[str]:
    """Every contract instrument this runtime can authoritatively emit."""
    return frozenset(
        {
            QUEUE_DURATION_METRIC_NAME,
            QUEUE_DEPTH_METRIC_NAME,
            BATCH_SIZE_METRIC_NAME,
            BATCH_COST_METRIC_NAME,
            BATCH_FILL_RATIO_METRIC_NAME,
            QUEUE_PENDING_AT_DISPATCH_METRIC_NAME,
            RUNTIME_BATCH_SIZE_METRIC_NAME,
            RUNTIME_BATCH_SUBGROUPS_METRIC_NAME,
            RUNTIME_SUBGROUP_SIZE_METRIC_NAME,
            REQUESTS_METRIC_NAME,
            REQUEST_DURATION_METRIC_NAME,
            INFERENCE_DURATION_METRIC_NAME,
            UNITS_METRIC_NAME,
            MODEL_LOADED_METRIC_NAME,
            MODEL_LOAD_DURATION_METRIC_NAME,
            MODEL_MEMORY_METRIC_NAME,
            OOM_RECOVERIES_METRIC_NAME,
            MODEL_EVICTIONS_METRIC_NAME,
            ADAPTIVE_WAIT_METRIC_NAME,
            ADAPTIVE_COST_METRIC_NAME,
            ADAPTIVE_P50_METRIC_NAME,
            STARVATION_RESETS_METRIC_NAME,
            GENERATION_TTFT_METRIC_NAME,
            GENERATION_TPOT_METRIC_NAME,
            GENERATION_TOKENS_METRIC_NAME,
            GENERATION_INFLIGHT_METRIC_NAME,
            GENERATION_KV_RESERVED_METRIC_NAME,
            GENERATION_KV_BUDGET_METRIC_NAME,
            GENERATION_ADMISSION_METRIC_NAME,
            GENERATION_DUPLICATE_PREVENTED_METRIC_NAME,
            GRAMMAR_COMPILE_DURATION_METRIC_NAME,
            GRAMMAR_CACHE_LOOKUPS_METRIC_NAME,
            GRAMMAR_REQUESTS_METRIC_NAME,
        }
    )


def generation_metric_names() -> frozenset[str]:
    """Generation-family instruments owned by the worker facade."""
    return frozenset(
        {
            GENERATION_TTFT_METRIC_NAME,
            GENERATION_TPOT_METRIC_NAME,
            GENERATION_TOKENS_METRIC_NAME,
            GENERATION_INFLIGHT_METRIC_NAME,
            GENERATION_KV_RESERVED_METRIC_NAME,
            GENERATION_KV_BUDGET_METRIC_NAME,
            GENERATION_ADMISSION_METRIC_NAME,
            GENERATION_DUPLICATE_PREVENTED_METRIC_NAME,
            GRAMMAR_COMPILE_DURATION_METRIC_NAME,
            GRAMMAR_CACHE_LOOKUPS_METRIC_NAME,
            GRAMMAR_REQUESTS_METRIC_NAME,
        }
    )


def configure_worker_metric_context(*, lane: str, configs: Mapping[str, ModelConfig]) -> None:
    """Install the release catalog within the process-lifetime series budget."""
    candidates: dict[tuple[str, str], str] = {}
    aliases: dict[str, tuple[str, str]] = {}
    for raw_name, config in configs.items():
        variant_source = getattr(config, "synthetic_profile_variant_source", None)
        if variant_source is not None:
            raw_model, raw_profile = variant_source
            model = _bounded_release_value(raw_model)
            profile = _bounded_release_value(raw_profile)
            alias = _bounded_release_value(raw_name)
            if _OTHER in (model, profile, alias):
                continue
            try:
                adapter_path = config.resolve_profile("default").adapter_path
            except (KeyError, ValueError):
                backend = _OTHER
            else:
                backend = _adapter_backend(adapter_path)
            pair = (model, profile)
            candidates[pair] = backend
            aliases[alias] = pair
            continue

        model = _bounded_release_value(raw_name)
        if model == _OTHER:
            continue
        for raw_profile in config.profiles:
            profile = _bounded_release_value(raw_profile)
            if profile == _OTHER:
                continue
            try:
                adapter_path = config.resolve_profile(raw_profile).adapter_path
            except (KeyError, ValueError):
                backend = _OTHER
            else:
                backend = _adapter_backend(adapter_path)
            candidates[(model, profile)] = backend

    profiles: dict[str, frozenset[str]] = {}
    backends: dict[tuple[str, str], str] = {}
    mutable_profiles: dict[str, set[str]] = {}
    overflow_pairs: set[tuple[str, str]] = set()
    warn_once = False

    global _catalog_collapse_warning_emitted, _context
    with _catalog_admission_lock:
        for pair in sorted(candidates):
            admitted = pair in _admitted_catalog_pairs
            if not admitted and len(_admitted_catalog_pairs) < _MAX_CATALOG_MODEL_PROFILE_PAIRS:
                _admitted_catalog_pairs.add(pair)
                admitted = True
            if not admitted:
                overflow_pairs.add(pair)
                continue

            model, profile = pair
            mutable_profiles.setdefault(model, set()).add(profile)
            backends[pair] = candidates[pair]

        profiles = {model: frozenset(model_profiles) for model, model_profiles in mutable_profiles.items()}
        warn_once = bool(overflow_pairs) and not _catalog_collapse_warning_emitted
        _catalog_collapse_warning_emitted |= bool(overflow_pairs)
        _context = _WorkerMetricContext(
            lane=_bounded_release_value(lane),
            profiles=profiles,
            backends=backends,
            aliases=aliases,
            overflow_pairs=frozenset(overflow_pairs),
        )

    if warn_once:
        logger.warning(
            "worker telemetry catalog exceeded its %d-pair lifetime budget; affected observations collapse to other",
            _MAX_CATALOG_MODEL_PROFILE_PAIRS,
        )


def refresh_worker_metric_context(*, configs: Mapping[str, ModelConfig]) -> None:
    """Replace catalog-derived dimensions while preserving the deployed lane."""
    configure_worker_metric_context(lane=_context.lane, configs=configs)


def lane_from_environment() -> str:
    pool = _bounded_release_value(os.environ.get("SIE_POOL", "default"))
    machine = _bounded_release_value(os.environ.get("SIE_MACHINE_PROFILE", "default"))
    bundle = _bounded_release_value(os.environ.get("SIE_BUNDLE", "default"))
    if _OTHER in (pool, machine, bundle):
        return _OTHER
    return _bounded_release_value(f"{pool}|{machine}|{bundle}")


def _request_attributes(
    *,
    operation: object,
    outcome: object,
    model: object,
    profile: object,
) -> dict[str, str]:
    lane, bounded_model, bounded_profile, backend = _dimensions(model, profile)
    return {
        "operation": _enum(operation, _OPERATIONS),
        "outcome": _enum(outcome, _WORKER_OUTCOMES),
        "backend": backend,
        "lane": lane,
        "model": bounded_model,
        "profile": bounded_profile,
    }


def _model_attributes(model: object, profile: object) -> dict[str, str]:
    lane, bounded_model, bounded_profile, backend = _dimensions(model, profile)
    return {
        "backend": backend,
        "lane": lane,
        "model": bounded_model,
        "profile": bounded_profile,
    }


def _runtime_attributes(*, operation: object, model: object, profile: object) -> dict[str, str]:
    lane, bounded_model, bounded_profile, backend = _dimensions(model, profile)
    return {
        "operation": _enum(operation, _OPERATIONS),
        "backend": backend,
        "lane": lane,
        "model": bounded_model,
        "profile": bounded_profile,
    }


def _dimensions(model: object, profile: object) -> tuple[str, str, str, str]:
    raw_model = _bounded_release_value(model)
    raw_profile = _bounded_release_value(profile or "default")
    context = _context
    pair = (raw_model, raw_profile)
    if pair not in context.backends and pair not in context.overflow_pairs:
        alias_pair = context.aliases.get(raw_model)
        if alias_pair is not None and raw_profile in ("default", alias_pair[1]):
            pair = alias_pair
    if pair in context.overflow_pairs:
        return context.lane, _OTHER, _OTHER, _OTHER
    bounded_model, bounded_profile = pair
    if bounded_profile not in context.profiles.get(bounded_model, frozenset()):
        return context.lane, _OTHER, _OTHER, _OTHER
    backend = context.backends.get(pair, _OTHER)
    return context.lane, bounded_model, bounded_profile, backend


def _adapter_backend(adapter_path: str) -> str:
    if adapter_path.startswith("sie_server_rust."):
        return "candle"
    if ".sglang." in adapter_path:
        return "sglang"
    if adapter_path.startswith("sie_server.adapters."):
        return "python"
    return _OTHER


def _bounded_release_value(value: object) -> str:
    text = str(value or "").strip()
    if not text or len(text) > _MAX_DIMENSION_LEN or not text.isascii():
        return _OTHER
    return text


def _enum(value: object, allowed: frozenset[str]) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in allowed else _OTHER


def _nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return max(0.0, result)


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return max(0, int(value))


def _positive_int(value: object) -> int | None:
    result = _nonnegative_int(value)
    return result if result is not None and result > 0 else None


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _metrics_protocol() -> str:
    raw = (
        _clean_env("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL") or _clean_env("OTEL_EXPORTER_OTLP_PROTOCOL") or "grpc"
    ).lower()
    if raw == "grpc":
        return "grpc"
    if raw == "http/protobuf":
        return "http"
    msg = f"unsupported OTLP metrics protocol: {raw!r}"
    raise ValueError(msg)


def _metrics_endpoint(protocol: str) -> str | None:
    explicit = _clean_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if explicit:
        return explicit
    base = _clean_env("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base:
        return None
    if protocol == "grpc":
        return base
    normalized = base.rstrip("/")
    return normalized if normalized.endswith("/v1/metrics") else f"{normalized}/v1/metrics"


def _build_metric_exporter(endpoint: str, protocol: str) -> Any:
    preferred_temporality = otlp_metric_temporality()
    if protocol == "grpc":
        return GrpcMetricExporter(
            endpoint=endpoint,
            timeout=EXPORT_TIMEOUT_S,
            preferred_temporality=preferred_temporality,
        )
    return HttpMetricExporter(
        endpoint=endpoint,
        timeout=EXPORT_TIMEOUT_S,
        preferred_temporality=preferred_temporality,
    )


def otlp_metric_temporality() -> dict[type, AggregationTemporality]:
    """Return the canonical application-wire temporality policy.

    Short-lived managed producers cannot establish a cumulative baseline at a
    remote backend. Emit additive instruments as DELTA while gauges and
    up/down counters retain the SDK's cumulative/current-value semantics. The
    explicit map also prevents ambient OTel environment settings from changing
    this checked-in wire contract.
    """
    return {
        Counter: AggregationTemporality.DELTA,
        Histogram: AggregationTemporality.DELTA,
        ObservableCounter: AggregationTemporality.DELTA,
    }
