//! Canonical single-emission telemetry owned by the realtime sidecar.
//!
//! Queue, batch, and sidecar↔engine IPC call sites report one semantic event
//! through [`SidecarTelemetry`]. This facade is the only code that creates or
//! records the corresponding OpenTelemetry instruments. OTLP is the only
//! application export path; Prometheus compatibility is produced downstream by
//! the collector. Attributes are closed and contain no IDs, LoRA names, or
//! other caller-controlled dimensions.

use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use opentelemetry::metrics::{Gauge, Histogram};
use opentelemetry::{global, KeyValue};

pub const QUEUE_DURATION_METRIC_NAME: &str = "sie.worker.queue.duration";
pub const QUEUE_DEPTH_METRIC_NAME: &str = "sie.worker.queue.depth";
pub const SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME: &str =
    "sie.worker.scheduler.request_batch.dispatch_wait";
pub const SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME: &str =
    "sie.worker.scheduler.request_batch.total";
pub const BATCH_SIZE_METRIC_NAME: &str = "sie.worker.batch.size";
pub const BATCH_COST_METRIC_NAME: &str = "sie.worker.batch.cost";
pub const BATCH_FILL_RATIO_METRIC_NAME: &str = "sie.worker.batch.fill_ratio";
pub const IPC_REQUESTS_METRIC_NAME: &str = "sie.worker.ipc.requests";
pub const IPC_REQUEST_DURATION_METRIC_NAME: &str = "sie.worker.ipc.request.duration";
pub const IPC_RESPONSE_CHUNKS_METRIC_NAME: &str = "sie.worker.ipc.response.chunks";
pub const IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME: &str =
    "sie.worker.ipc.response.reconstructed.size";
pub const IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME: &str = "sie.worker.ipc.response.chunk.count";
pub const IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME: &str = "sie.worker.ipc.response.chunk.reserved";
pub const CONFIG_APPLIES_METRIC_NAME: &str = "sie.worker.config.applies";
pub const CONFIG_EPOCH_METRIC_NAME: &str = "sie.worker.config.epoch";
pub const CONFIG_DEGRADED_METRIC_NAME: &str = "sie.worker.config.degraded";
pub const NATS_OPERATIONS_METRIC_NAME: &str = "sie.worker.nats.operations";
pub const NATS_DELIVERY_ATTEMPTS_METRIC_NAME: &str = "sie.worker.nats.delivery.attempts";
pub const RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME: &str = "sie.worker.result.transport.attempts";
pub const RESULT_CHUNKS_PUBLISHED_METRIC_NAME: &str = "sie.worker.result.chunks.published";
pub const RESULT_CHUNK_SIZE_METRIC_NAME: &str = "sie.worker.result.chunk.size";
pub const PAYLOAD_FETCHES_METRIC_NAME: &str = "sie.worker.payload.fetches";
pub const PAYLOAD_FETCH_DURATION_METRIC_NAME: &str = "sie.worker.payload.fetch.duration";
pub const PAYLOAD_SIZE_METRIC_NAME: &str = "sie.worker.payload.size";
pub const GPU_SLOTS_METRIC_NAME: &str = "sie.worker.gpu.slots";
pub const PENDING_ITEMS_METRIC_NAME: &str = "sie.worker.pending.items";
pub const PENDING_COST_METRIC_NAME: &str = "sie.worker.pending.cost";
pub const INFLIGHT_BATCHES_METRIC_NAME: &str = "sie.worker.inflight.batches";
pub const SATURATED_METRIC_NAME: &str = "sie.worker.saturated";
pub const IPC_CAPACITY_METRIC_NAME: &str = "sie.worker.ipc.capacity";
pub const IPC_INFLIGHT_METRIC_NAME: &str = "sie.worker.ipc.inflight";
pub const IPC_ACQUIRE_DURATION_METRIC_NAME: &str = "sie.worker.ipc.acquire.duration";
pub const ADAPTIVE_WAIT_METRIC_NAME: &str = "sie.worker.scheduler.adaptive.wait";
pub const ADAPTIVE_COST_METRIC_NAME: &str = "sie.worker.scheduler.adaptive.cost";
pub const ADAPTIVE_P50_METRIC_NAME: &str = "sie.worker.scheduler.adaptive.p50";
pub const STARVATION_RESETS_METRIC_NAME: &str = "sie.worker.scheduler.starvation.resets";
pub const GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME: &str =
    "sie.worker.generation.model_loading.responses";
pub const SHUTDOWN_DRAIN_DURATION_METRIC_NAME: &str = "sie.worker.shutdown.drain.duration";

const OTHER: &str = "other";
const DEFAULT_PROFILE: &str = "default";
const MAX_CATALOG_MODEL_LABEL_BYTES: usize = 512;
const MAX_CATALOG_PROFILE_LABEL_BYTES: usize = 128;
/// The Rust OTel SDK defaults to 2,000 streams per instrument. Six declared
/// queue operations times 256 catalog pairs plus one collapsed overflow pair
/// per operation is 1,542 series. Even including the defensive `other`
/// operation leaves 7 * 257 = 1,799 series, below that SDK limit.
const MAX_CATALOG_MODEL_PROFILE_PAIRS: usize = 256;

// Checked-in finite domains used by the sidecar MeterProvider's explicit
// per-instrument views. `lane` is fixed for one process and therefore has a
// cardinality factor of one. The catalog factor includes the single
// `(other, other)` overflow pair in addition to the 256 exact admitted pairs.
pub(crate) const SIDECAR_CATALOG_PAIR_SERIES: usize = MAX_CATALOG_MODEL_PROFILE_PAIRS + 1;
const OPERATION_SERIES: usize = 7;
const FLUSH_REASON_SERIES: usize = 8;
const IPC_METHOD_SERIES: usize = 14;
const IPC_OUTCOME_SERIES: usize = 7;
const IPC_RESPONSE_CHUNK_OUTCOME_SERIES: usize = 3;
const CONFIG_SOURCE_SERIES: usize = 3;
const CONFIG_OPERATION_SERIES: usize = 6;
const CONFIG_OUTCOME_SERIES: usize = 18;
const NATS_OPERATION_SERIES: usize = 7;
const BINARY_OUTCOME_SERIES: usize = 3;
const NATS_REASON_SERIES: usize = 10;
const DELIVERY_REDELIVERED_SERIES: usize = 2;
const RESULT_TRANSPORT_MODE_SERIES: usize = 4;
const RESULT_TRANSPORT_OUTCOME_SERIES: usize = 4;
const PAYLOAD_REASON_SERIES: usize = 8;
const SLOT_STATE_SERIES: usize = 2;
const IPC_TRANSPORT_SERIES: usize = 3;
const SCHEDULER_P50_KIND_SERIES: usize = 2;
const GENERATION_LOADING_STATE_SERIES: usize = 4;
const GENERATION_RESPONSE_OUTCOME_SERIES: usize = 4;
const SHUTDOWN_DRAIN_OUTCOME_SERIES: usize = 3;

pub(crate) const SIDECAR_QUEUE_CARDINALITY_LIMIT: usize =
    OPERATION_SERIES * SIDECAR_CATALOG_PAIR_SERIES;
pub(crate) const SIDECAR_BATCH_FILL_CARDINALITY_LIMIT: usize =
    SIDECAR_QUEUE_CARDINALITY_LIMIT * FLUSH_REASON_SERIES;
pub(crate) const SIDECAR_IPC_CARDINALITY_LIMIT: usize = IPC_METHOD_SERIES * IPC_OUTCOME_SERIES;
pub(crate) const SIDECAR_IPC_RESPONSE_CHUNK_CARDINALITY_LIMIT: usize =
    IPC_RESPONSE_CHUNK_OUTCOME_SERIES;
pub(crate) const SIDECAR_CONFIG_APPLY_CARDINALITY_LIMIT: usize =
    CONFIG_SOURCE_SERIES * CONFIG_OPERATION_SERIES * CONFIG_OUTCOME_SERIES;
pub(crate) const SIDECAR_NATS_CARDINALITY_LIMIT: usize =
    NATS_OPERATION_SERIES * BINARY_OUTCOME_SERIES * NATS_REASON_SERIES;
pub(crate) const SIDECAR_RESULT_TRANSPORT_CARDINALITY_LIMIT: usize =
    RESULT_TRANSPORT_MODE_SERIES * RESULT_TRANSPORT_OUTCOME_SERIES;
pub(crate) const SIDECAR_PAYLOAD_CARDINALITY_LIMIT: usize =
    BINARY_OUTCOME_SERIES * PAYLOAD_REASON_SERIES;
pub(crate) const SIDECAR_IPC_ACQUIRE_CARDINALITY_LIMIT: usize =
    IPC_TRANSPORT_SERIES * BINARY_OUTCOME_SERIES;
pub(crate) const SIDECAR_ADAPTIVE_P50_CARDINALITY_LIMIT: usize =
    SIDECAR_CATALOG_PAIR_SERIES * SCHEDULER_P50_KIND_SERIES;
pub(crate) const SIDECAR_GENERATION_LOADING_CARDINALITY_LIMIT: usize = SIDECAR_CATALOG_PAIR_SERIES
    * GENERATION_LOADING_STATE_SERIES
    * GENERATION_RESPONSE_OUTCOME_SERIES;

/// Return the exact finite attribute-domain ceiling for every sidecar-owned
/// metric stream. The provider uses this for explicit SDK views so valid
/// labels never enter `otel.metric.overflow`; unknown catalog pairs already
/// share one bounded `(other, other)` series before reaching the SDK.
pub(crate) fn sidecar_metric_cardinality_limit(name: &str) -> Option<usize> {
    match name {
        QUEUE_DURATION_METRIC_NAME
        | QUEUE_DEPTH_METRIC_NAME
        | SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME
        | SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME
        | BATCH_SIZE_METRIC_NAME
        | BATCH_COST_METRIC_NAME => Some(SIDECAR_QUEUE_CARDINALITY_LIMIT),
        BATCH_FILL_RATIO_METRIC_NAME => Some(SIDECAR_BATCH_FILL_CARDINALITY_LIMIT),
        IPC_REQUESTS_METRIC_NAME | IPC_REQUEST_DURATION_METRIC_NAME => {
            Some(SIDECAR_IPC_CARDINALITY_LIMIT)
        }
        IPC_RESPONSE_CHUNKS_METRIC_NAME
        | IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME
        | IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME
        | IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME => {
            Some(SIDECAR_IPC_RESPONSE_CHUNK_CARDINALITY_LIMIT)
        }
        CONFIG_APPLIES_METRIC_NAME => Some(SIDECAR_CONFIG_APPLY_CARDINALITY_LIMIT),
        CONFIG_EPOCH_METRIC_NAME | CONFIG_DEGRADED_METRIC_NAME => Some(CONFIG_SOURCE_SERIES),
        NATS_OPERATIONS_METRIC_NAME => Some(SIDECAR_NATS_CARDINALITY_LIMIT),
        NATS_DELIVERY_ATTEMPTS_METRIC_NAME => Some(DELIVERY_REDELIVERED_SERIES),
        RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME
        | RESULT_CHUNKS_PUBLISHED_METRIC_NAME
        | RESULT_CHUNK_SIZE_METRIC_NAME => Some(SIDECAR_RESULT_TRANSPORT_CARDINALITY_LIMIT),
        PAYLOAD_FETCHES_METRIC_NAME
        | PAYLOAD_FETCH_DURATION_METRIC_NAME
        | PAYLOAD_SIZE_METRIC_NAME => Some(SIDECAR_PAYLOAD_CARDINALITY_LIMIT),
        GPU_SLOTS_METRIC_NAME => Some(SLOT_STATE_SERIES),
        PENDING_ITEMS_METRIC_NAME
        | PENDING_COST_METRIC_NAME
        | INFLIGHT_BATCHES_METRIC_NAME
        | SATURATED_METRIC_NAME => Some(1),
        IPC_CAPACITY_METRIC_NAME | IPC_INFLIGHT_METRIC_NAME => Some(IPC_TRANSPORT_SERIES),
        IPC_ACQUIRE_DURATION_METRIC_NAME => Some(SIDECAR_IPC_ACQUIRE_CARDINALITY_LIMIT),
        ADAPTIVE_WAIT_METRIC_NAME | ADAPTIVE_COST_METRIC_NAME | STARVATION_RESETS_METRIC_NAME => {
            Some(SIDECAR_CATALOG_PAIR_SERIES)
        }
        ADAPTIVE_P50_METRIC_NAME => Some(SIDECAR_ADAPTIVE_P50_CARDINALITY_LIMIT),
        GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME => {
            Some(SIDECAR_GENERATION_LOADING_CARDINALITY_LIMIT)
        }
        SHUTDOWN_DRAIN_DURATION_METRIC_NAME => Some(SHUTDOWN_DRAIN_OUTCOME_SERIES),
        _ => None,
    }
}
const QUEUE_DURATION_BUCKETS: &[f64] = &[
    0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0,
];
const BATCH_SIZE_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];
const BATCH_COST_BUCKETS: &[f64] = &[
    128.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16_384.0, 32_768.0, 65_536.0, 131_072.0,
    262_144.0,
];
const IPC_REQUEST_DURATION_BUCKETS: &[f64] = &[
    0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0, 60.0, 300.0, 900.0,
];
const PAYLOAD_FETCH_DURATION_BUCKETS: &[f64] = &[
    0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0,
];
const PAYLOAD_SIZE_BUCKETS: &[f64] = &[
    256.0,
    1024.0,
    4096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];
const IPC_RESPONSE_CHUNK_COUNT_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const IPC_ACQUIRE_DURATION_BUCKETS: &[f64] = &[
    0.000_001, 0.000_01, 0.000_1, 0.000_25, 0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1,
    0.25, 0.5, 1.0, 2.5, 5.0,
];
const DELIVERY_ATTEMPT_BUCKETS: &[f64] = &[1.0, 2.0, 3.0, 5.0, 10.0];
const SHUTDOWN_DRAIN_DURATION_BUCKETS: &[f64] = &[
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0,
];

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct QueueKey {
    operation: &'static str,
    model: String,
    profile: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CatalogPair {
    model: String,
    profile: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CatalogAlias {
    raw_model: Option<String>,
    pair: CatalogPair,
}

#[derive(Default)]
struct CatalogDimensions {
    /// Active semantic pairs, indexed without allocating on the request path.
    active: HashMap<String, HashSet<String>>,
    /// Raw routable model IDs (including `model:profile` variants) mapped to
    /// their semantic metric pair.
    aliases: HashMap<String, CatalogPair>,
    ambiguous_aliases: HashSet<String>,
    /// Lifetime admission is intentional: synchronous OTel aggregators retain
    /// a series after the catalog stops advertising it.
    admitted: HashSet<CatalogPair>,
    warning_emitted: bool,
}

impl CatalogDimensions {
    fn pair_is_active(&self, model: &str, profile: &str) -> bool {
        self.active
            .get(model)
            .is_some_and(|profiles| profiles.contains(profile))
    }

    #[cfg(test)]
    fn active_pair_count(&self) -> usize {
        self.active.values().map(HashSet::len).sum()
    }

    fn admit_pair(&mut self, pair: &CatalogPair) -> bool {
        if self.admitted.contains(pair) {
            return true;
        }
        if self.admitted.len() >= MAX_CATALOG_MODEL_PROFILE_PAIRS {
            return false;
        }
        self.admitted.insert(pair.clone());
        true
    }

    fn activate_pair(&mut self, pair: &CatalogPair) {
        self.active
            .entry(pair.model.clone())
            .or_default()
            .insert(pair.profile.clone());
    }

    fn extend(&mut self, candidates: Vec<CatalogAlias>) -> bool {
        let mut exact = true;
        for candidate in candidates {
            if !self.admit_pair(&candidate.pair) {
                exact = false;
                continue;
            }
            self.activate_pair(&candidate.pair);
            let Some(raw_model) = candidate.raw_model else {
                exact = false;
                continue;
            };
            if self.ambiguous_aliases.contains(&raw_model) {
                exact = false;
                continue;
            }
            match self.aliases.get(&raw_model) {
                Some(existing) if existing != &candidate.pair => {
                    self.aliases.remove(&raw_model);
                    self.ambiguous_aliases.insert(raw_model);
                    exact = false;
                }
                Some(_) => {}
                None => {
                    self.aliases.insert(raw_model, candidate.pair);
                }
            }
        }
        exact
    }

    fn replace(&mut self, candidates: Vec<CatalogAlias>) -> bool {
        let mut next_active: HashMap<String, HashSet<String>> = HashMap::new();
        let mut next_aliases: HashMap<String, CatalogPair> = HashMap::new();
        let mut next_ambiguous = HashSet::new();
        let mut exact = true;

        for candidate in candidates {
            if !self.admit_pair(&candidate.pair) {
                exact = false;
                continue;
            }
            next_active
                .entry(candidate.pair.model.clone())
                .or_default()
                .insert(candidate.pair.profile.clone());
            let Some(raw_model) = candidate.raw_model else {
                exact = false;
                continue;
            };
            if next_ambiguous.contains(&raw_model) {
                exact = false;
                continue;
            }
            match next_aliases.get(&raw_model) {
                Some(existing) if existing != &candidate.pair => {
                    next_aliases.remove(&raw_model);
                    next_ambiguous.insert(raw_model);
                    exact = false;
                }
                Some(_) => {}
                None => {
                    next_aliases.insert(raw_model, candidate.pair);
                }
            }
        }

        self.active = next_active;
        self.aliases = next_aliases;
        self.ambiguous_aliases = next_ambiguous;
        exact
    }

    fn resolve<'a>(
        &self,
        raw_model: &'a str,
        raw_profile: Option<&'a str>,
    ) -> Option<(&'a str, &'a str)> {
        let model = bounded_catalog_model(raw_model)?;
        let profile = match raw_profile {
            Some(raw) if !raw.trim().is_empty() => bounded_catalog_profile(raw)?,
            _ => DEFAULT_PROFILE,
        };

        // Direct `(base model, selected profile)` is the unambiguous form.
        if self.pair_is_active(model, profile) {
            return Some((model, profile));
        }

        // Queue work commonly carries a materialized `model:profile` ID and a
        // wire-compatible `profile_id=default`. Resolve that alias only after
        // the trusted catalog established its exact semantic pair.
        let alias = self.aliases.get(model)?;
        if profile != DEFAULT_PROFILE && profile != alias.profile {
            return None;
        }
        if alias.model == model && alias.profile == DEFAULT_PROFILE {
            return Some((model, DEFAULT_PROFILE));
        }
        let (base, suffix) = model.rsplit_once(':')?;
        if base == alias.model && suffix == alias.profile {
            Some((base, suffix))
        } else {
            None
        }
    }
}

#[derive(Clone)]
struct TelemetryContext {
    lane: Arc<str>,
    catalog: Arc<RwLock<CatalogDimensions>>,
}

impl TelemetryContext {
    fn from_env() -> Self {
        let lane = cleaned_env("SIE_OTEL_LANE").unwrap_or_else(|| OTHER.to_string());
        let model_allowlist = cleaned_env("SIE_METRIC_MODEL_ALLOWLIST")
            .map(|raw| parse_allowlist(&raw))
            .unwrap_or_default();
        Self::new(&lane, model_allowlist)
    }

    fn new(lane: &str, model_allowlist: HashSet<String>) -> Self {
        let (candidates, issues) = startup_catalog_candidates(model_allowlist);
        let mut catalog = CatalogDimensions::default();
        let exact = catalog.replace(candidates) && issues == 0;
        if !exact {
            catalog.warning_emitted = true;
            tracing::warn!(
                max_pairs = MAX_CATALOG_MODEL_PROFILE_PAIRS,
                "sidecar telemetry catalog exceeded or violated its bounded model/profile contract; affected observations collapse to other"
            );
        }
        Self {
            lane: Arc::from(nonempty_or_other(lane)),
            catalog: Arc::new(RwLock::new(catalog)),
        }
    }

    /// Resolve both dimensions against one coherent catalog-pair snapshot.
    /// Unknown or mismatched combinations collapse together so independent
    /// allowlists can never create a Cartesian product of series.
    fn model_profile<'a>(
        &self,
        raw_model: &'a str,
        raw_profile: Option<&'a str>,
    ) -> (&'a str, &'a str) {
        let Ok(catalog) = self.catalog.read() else {
            return (OTHER, OTHER);
        };
        catalog
            .resolve(raw_model, raw_profile)
            .unwrap_or((OTHER, OTHER))
    }

    fn queue_attributes(
        &self,
        operation: &'static str,
        model: &str,
        profile: &str,
    ) -> [KeyValue; 4] {
        [
            KeyValue::new("operation", operation),
            KeyValue::new("lane", self.lane.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("profile", profile.to_string()),
        ]
    }

    fn batch_attributes(
        &self,
        operation: &'static str,
        model: &str,
        profile: &str,
        flush_reason: &'static str,
    ) -> [KeyValue; 5] {
        [
            KeyValue::new("operation", operation),
            KeyValue::new("lane", self.lane.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("profile", profile.to_string()),
            KeyValue::new("flush.reason", flush_reason),
        ]
    }

    fn batch_core_attributes(
        &self,
        operation: &'static str,
        model: &str,
        profile: &str,
    ) -> [KeyValue; 4] {
        self.queue_attributes(operation, model, profile)
    }

    fn ipc_attributes(&self, method: &'static str, outcome: &'static str) -> [KeyValue; 3] {
        [
            KeyValue::new("method", method),
            KeyValue::new("outcome", outcome),
            KeyValue::new("lane", self.lane.to_string()),
        ]
    }

    fn model_profile_attributes(&self, model: &str, profile: &str) -> [KeyValue; 3] {
        [
            KeyValue::new("lane", self.lane.to_string()),
            KeyValue::new("model", model.to_string()),
            KeyValue::new("profile", profile.to_string()),
        ]
    }

    fn lane_attribute(&self) -> KeyValue {
        KeyValue::new("lane", self.lane.to_string())
    }
}

#[derive(Default)]
struct IpcInflightState {
    pool: AtomicU64,
    mux: AtomicU64,
}

#[derive(Default)]
struct IpcCapacityState {
    pool: AtomicU64,
    mux: AtomicU64,
}

/// One successful request-ID occurrence completed by a backend batch.
///
/// Both monotonic intervals start when the request enters the sidecar
/// scheduler. The dispatch interval ends immediately before `RunBatch`; the
/// total interval ends when that RPC replies. Callers deduplicate request IDs
/// within the backend batch before emitting this one semantic event.
#[derive(Clone, Copy, Debug)]
pub struct SchedulerRequestBatchObservation<'a> {
    pub operation: &'a str,
    pub model: &'a str,
    pub profile: Option<&'a str>,
    pub dispatch_wait: Duration,
    pub total: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IpcResponseChunkOutcome {
    Completed,
    ProtocolError,
    BudgetRejected,
}

impl IpcResponseChunkOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::ProtocolError => "protocol_error",
            Self::BudgetRejected => "budget_rejected",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResultTransportMode {
    Single,
    Chunked,
    CompactError,
    Rejected,
}

impl ResultTransportMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Chunked => "chunked",
            Self::CompactError => "compact_error",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResultTransportOutcome {
    Published,
    PublishError,
    FlushError,
    PlanningError,
}

impl ResultTransportOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::PublishError => "publish_error",
            Self::FlushError => "flush_error",
            Self::PlanningError => "planning_error",
        }
    }
}

/// Canonical queue/batch/IPC instruments plus the per-series queue-depth state
/// needed by the synchronous OTel gauge.
///
/// The state map is bounded by the operation enum times the process-local model
/// allowlist.  Unknown model IDs collapse before they reach the map.  A poisoned
/// lock merely drops the gauge update; telemetry must never stop inference.
#[doc(hidden)]
pub struct EnabledSidecarTelemetry {
    queue_duration: Histogram<f64>,
    queue_depth: Gauge<u64>,
    scheduler_request_batch_dispatch_wait: Histogram<f64>,
    scheduler_request_batch_total: Histogram<f64>,
    batch_size: Histogram<u64>,
    batch_cost: Histogram<u64>,
    batch_fill_ratio: Gauge<f64>,
    ipc_requests: opentelemetry::metrics::Counter<u64>,
    ipc_request_duration: Histogram<f64>,
    ipc_response_chunks: opentelemetry::metrics::Counter<u64>,
    ipc_response_reconstructed_size: Histogram<u64>,
    ipc_response_chunk_count: Histogram<u64>,
    ipc_response_chunk_reserved: Gauge<u64>,
    config_applies: opentelemetry::metrics::Counter<u64>,
    config_epoch: Gauge<u64>,
    config_degraded: Gauge<u64>,
    nats_operations: opentelemetry::metrics::Counter<u64>,
    nats_delivery_attempts: Histogram<u64>,
    result_transport_attempts: opentelemetry::metrics::Counter<u64>,
    result_chunks_published: opentelemetry::metrics::Counter<u64>,
    result_chunk_size: Histogram<u64>,
    payload_fetches: opentelemetry::metrics::Counter<u64>,
    payload_fetch_duration: Histogram<f64>,
    payload_size: Histogram<u64>,
    gpu_slots: Gauge<u64>,
    pending_items: Gauge<u64>,
    pending_cost: Gauge<u64>,
    inflight_batches: Gauge<u64>,
    saturated: Gauge<u64>,
    ipc_capacity: Gauge<u64>,
    ipc_inflight: Gauge<u64>,
    ipc_acquire_duration: Histogram<f64>,
    adaptive_wait: Gauge<f64>,
    adaptive_cost: Gauge<u64>,
    adaptive_p50: Gauge<f64>,
    starvation_resets: opentelemetry::metrics::Counter<u64>,
    generation_model_loading_responses: opentelemetry::metrics::Counter<u64>,
    shutdown_drain_duration: Histogram<f64>,
    context: TelemetryContext,
    queue_depths: Arc<Mutex<HashMap<QueueKey, u64>>>,
    ipc_inflight_state: Arc<IpcInflightState>,
    ipc_capacity_state: Arc<IpcCapacityState>,
}

/// Cheap runtime telemetry handle with an explicit disabled state.
///
/// The disabled variant owns neither SDK instruments nor the bounded state
/// maps used by the enabled facade. This keeps the disabled request path to a
/// single branch and prevents merely constructing [`RuntimeState`] from
/// registering instruments with the global meter provider.
#[derive(Clone, Default)]
pub struct SidecarTelemetry {
    inner: Option<Arc<EnabledSidecarTelemetry>>,
}

impl std::fmt::Debug for SidecarTelemetry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SidecarTelemetry")
            .field("enabled", &self.is_enabled())
            .finish()
    }
}

impl Deref for SidecarTelemetry {
    type Target = EnabledSidecarTelemetry;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_deref()
            .expect("disabled sidecar telemetry must be checked before instrument access")
    }
}

impl SidecarTelemetry {
    /// Whether semantic observations will build attributes and reach SDK
    /// instruments. Call sites use this only when they must retain an owned
    /// telemetry-only value across a business-state move.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Bind instruments to the global provider installed during sidecar startup.
    /// If canonical telemetry is disabled, every semantic method is a no-op.
    pub fn from_global() -> Self {
        Self::from_enabled_builder(super::tracing::metrics_provider_enabled(), || {
            Self::build_enabled(
                &global::meter("sie-worker-sidecar"),
                TelemetryContext::from_env(),
            )
        })
    }

    #[cfg(test)]
    fn new(
        meter: &opentelemetry::metrics::Meter,
        context: TelemetryContext,
        enabled: bool,
    ) -> Self {
        Self::from_enabled_builder(enabled, || Self::build_enabled(meter, context))
    }

    fn from_enabled_builder<F>(enabled: bool, build: F) -> Self
    where
        F: FnOnce() -> EnabledSidecarTelemetry,
    {
        Self {
            inner: enabled.then(|| Arc::new(build())),
        }
    }

    fn build_enabled(
        meter: &opentelemetry::metrics::Meter,
        context: TelemetryContext,
    ) -> EnabledSidecarTelemetry {
        let queue_duration = meter
            .f64_histogram(QUEUE_DURATION_METRIC_NAME)
            .with_description("Worker-local scheduler wait before batch execution.")
            .with_unit("s")
            .with_boundaries(QUEUE_DURATION_BUCKETS.to_vec())
            .build();
        let scheduler_request_batch_dispatch_wait = meter
            .f64_histogram(SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME)
            .with_description(
                "Scheduler enqueue to backend batch dispatch for one successful request ID.",
            )
            .with_unit("s")
            .with_boundaries(QUEUE_DURATION_BUCKETS.to_vec())
            .build();
        let scheduler_request_batch_total = meter
            .f64_histogram(SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME)
            .with_description(
                "Scheduler enqueue to backend batch reply for one successful request ID.",
            )
            .with_unit("s")
            .with_boundaries(QUEUE_DURATION_BUCKETS.to_vec())
            .build();
        let queue_depth = meter
            .u64_gauge(QUEUE_DEPTH_METRIC_NAME)
            .with_description("Current worker-local scheduler depth.")
            .with_unit("{item}")
            .build();
        let batch_size = meter
            .u64_histogram(BATCH_SIZE_METRIC_NAME)
            .with_description("Items in a scheduler-formed batch.")
            .with_unit("{item}")
            .with_boundaries(BATCH_SIZE_BUCKETS.to_vec())
            .build();
        let batch_cost = meter
            .u64_histogram(BATCH_COST_METRIC_NAME)
            .with_description("Scheduler cost of a formed batch.")
            .with_unit("{cost}")
            .with_boundaries(BATCH_COST_BUCKETS.to_vec())
            .build();
        let batch_fill_ratio = meter
            .f64_gauge(BATCH_FILL_RATIO_METRIC_NAME)
            .with_description("Rolling mean scheduler batch fill ratio.")
            .with_unit("1")
            .build();
        let ipc_requests = meter
            .u64_counter(IPC_REQUESTS_METRIC_NAME)
            .with_description("Logical sidecar-to-engine IPC RPCs by bounded method and outcome.")
            .with_unit("{request}")
            .build();
        let ipc_request_duration = meter
            .f64_histogram(IPC_REQUEST_DURATION_METRIC_NAME)
            .with_description("End-to-end logical sidecar-to-engine IPC RPC duration.")
            .with_unit("s")
            .with_boundaries(IPC_REQUEST_DURATION_BUCKETS.to_vec())
            .build();
        let ipc_response_chunks = meter
            .u64_counter(IPC_RESPONSE_CHUNKS_METRIC_NAME)
            .with_description("Negotiated IPC response-chunk transfer outcomes.")
            .with_unit("{transfer}")
            .build();
        let ipc_response_reconstructed_size = meter
            .u64_histogram(IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME)
            .with_description("Reconstructed response size for completed IPC chunk transfers.")
            .with_unit("By")
            .with_boundaries(PAYLOAD_SIZE_BUCKETS.to_vec())
            .build();
        let ipc_response_chunk_count = meter
            .u64_histogram(IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME)
            .with_description("Physical chunk count for completed IPC response transfers.")
            .with_unit("{chunk}")
            .with_boundaries(IPC_RESPONSE_CHUNK_COUNT_BUCKETS.to_vec())
            .build();
        let ipc_response_chunk_reserved = meter
            .u64_gauge(IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME)
            .with_description("Process-local bytes reserved for IPC response reassembly.")
            .with_unit("By")
            .build();
        let config_applies = meter
            .u64_counter(CONFIG_APPLIES_METRIC_NAME)
            .with_description("Trusted sidecar config apply outcomes.")
            .with_unit("{apply}")
            .build();
        let config_epoch = meter
            .u64_gauge(CONFIG_EPOCH_METRIC_NAME)
            .with_description("Latest successfully applied config epoch by config source.")
            .with_unit("{epoch}")
            .build();
        let config_degraded = meter
            .u64_gauge(CONFIG_DEGRADED_METRIC_NAME)
            .with_description("Whether config convergence is degraded for a config source.")
            .with_unit("1")
            .build();
        let nats_operations = meter
            .u64_counter(NATS_OPERATIONS_METRIC_NAME)
            .with_description("NATS receive, settlement, fetch, and stream outcomes.")
            .with_unit("{operation}")
            .build();
        let nats_delivery_attempts = meter
            .u64_histogram(NATS_DELIVERY_ATTEMPTS_METRIC_NAME)
            .with_description("JetStream delivery attempt number observed on receive.")
            .with_unit("{attempt}")
            .with_boundaries(DELIVERY_ATTEMPT_BUCKETS.to_vec())
            .build();
        let result_transport_attempts = meter
            .u64_counter(RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME)
            .with_description("Terminal NATS result transport outcomes by bounded mode.")
            .with_unit("{attempt}")
            .build();
        let result_chunks_published = meter
            .u64_counter(RESULT_CHUNKS_PUBLISHED_METRIC_NAME)
            .with_description("Successfully queued NATS result-chunk envelopes.")
            .with_unit("{chunk}")
            .build();
        let result_chunk_size = meter
            .u64_histogram(RESULT_CHUNK_SIZE_METRIC_NAME)
            .with_description("Encoded NATS result-chunk envelope size.")
            .with_unit("By")
            .with_boundaries(PAYLOAD_SIZE_BUCKETS.to_vec())
            .build();
        let payload_fetches = meter
            .u64_counter(PAYLOAD_FETCHES_METRIC_NAME)
            .with_description("Offloaded payload fetch outcomes.")
            .with_unit("{fetch}")
            .build();
        let payload_fetch_duration = meter
            .f64_histogram(PAYLOAD_FETCH_DURATION_METRIC_NAME)
            .with_description("Offloaded payload fetch duration.")
            .with_unit("s")
            .with_boundaries(PAYLOAD_FETCH_DURATION_BUCKETS.to_vec())
            .build();
        let payload_size = meter
            .u64_histogram(PAYLOAD_SIZE_METRIC_NAME)
            .with_description("Successfully fetched offloaded payload size.")
            .with_unit("By")
            .with_boundaries(PAYLOAD_SIZE_BUCKETS.to_vec())
            .build();
        let gpu_slots = meter
            .u64_gauge(GPU_SLOTS_METRIC_NAME)
            .with_description("Worker adapter slots by total or ready state.")
            .with_unit("{slot}")
            .build();
        let pending_items = meter
            .u64_gauge(PENDING_ITEMS_METRIC_NAME)
            .with_description("Current sidecar scheduler pending item count.")
            .with_unit("{item}")
            .build();
        let pending_cost = meter
            .u64_gauge(PENDING_COST_METRIC_NAME)
            .with_description("Current sidecar scheduler pending cost.")
            .with_unit("{cost}")
            .build();
        let inflight_batches = meter
            .u64_gauge(INFLIGHT_BATCHES_METRIC_NAME)
            .with_description("Current sidecar backend batches in flight.")
            .with_unit("{batch}")
            .build();
        let saturated = meter
            .u64_gauge(SATURATED_METRIC_NAME)
            .with_description("Worker admission saturation state.")
            .with_unit("1")
            .build();
        let ipc_capacity = meter
            .u64_gauge(IPC_CAPACITY_METRIC_NAME)
            .with_description("Configured sidecar IPC admission capacity by transport.")
            .with_unit("{request}")
            .build();
        let ipc_inflight = meter
            .u64_gauge(IPC_INFLIGHT_METRIC_NAME)
            .with_description("Current sidecar IPC requests in flight by transport.")
            .with_unit("{request}")
            .build();
        let ipc_acquire_duration = meter
            .f64_histogram(IPC_ACQUIRE_DURATION_METRIC_NAME)
            .with_description("Wait for an IPC request slot by transport and outcome.")
            .with_unit("s")
            .with_boundaries(IPC_ACQUIRE_DURATION_BUCKETS.to_vec())
            .build();
        let adaptive_wait = meter
            .f64_gauge(ADAPTIVE_WAIT_METRIC_NAME)
            .with_description("Adaptive scheduler maximum batch wait.")
            .with_unit("s")
            .build();
        let adaptive_cost = meter
            .u64_gauge(ADAPTIVE_COST_METRIC_NAME)
            .with_description("Adaptive scheduler maximum batch cost.")
            .with_unit("{cost}")
            .build();
        let adaptive_p50 = meter
            .f64_gauge(ADAPTIVE_P50_METRIC_NAME)
            .with_description("Adaptive scheduler observed or target p50 latency.")
            .with_unit("s")
            .build();
        let starvation_resets = meter
            .u64_counter(STARVATION_RESETS_METRIC_NAME)
            .with_description("Adaptive scheduler starvation recovery resets.")
            .with_unit("{reset}")
            .build();
        let generation_model_loading_responses = meter
            .u64_counter(GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME)
            .with_description(
                "Terminal generation responses emitted while model readiness is not ready.",
            )
            .with_unit("{response}")
            .build();
        let shutdown_drain_duration = meter
            .f64_histogram(SHUTDOWN_DRAIN_DURATION_METRIC_NAME)
            .with_description("Backend drain duration during graceful sidecar shutdown.")
            .with_unit("s")
            .with_boundaries(SHUTDOWN_DRAIN_DURATION_BUCKETS.to_vec())
            .build();
        EnabledSidecarTelemetry {
            queue_duration,
            queue_depth,
            scheduler_request_batch_dispatch_wait,
            scheduler_request_batch_total,
            batch_size,
            batch_cost,
            batch_fill_ratio,
            ipc_requests,
            ipc_request_duration,
            ipc_response_chunks,
            ipc_response_reconstructed_size,
            ipc_response_chunk_count,
            ipc_response_chunk_reserved,
            config_applies,
            config_epoch,
            config_degraded,
            nats_operations,
            nats_delivery_attempts,
            result_transport_attempts,
            result_chunks_published,
            result_chunk_size,
            payload_fetches,
            payload_fetch_duration,
            payload_size,
            gpu_slots,
            pending_items,
            pending_cost,
            inflight_batches,
            saturated,
            ipc_capacity,
            ipc_inflight,
            ipc_acquire_duration,
            adaptive_wait,
            adaptive_cost,
            adaptive_p50,
            starvation_resets,
            generation_model_loading_responses,
            shutdown_drain_duration,
            context,
            queue_depths: Arc::new(Mutex::new(HashMap::new())),
            ipc_inflight_state: Arc::new(IpcInflightState::default()),
            ipc_capacity_state: Arc::new(IpcCapacityState::default()),
        }
    }

    /// Extend the live catalog after one trusted config notification applies.
    /// Only exact semantic `(model, profile)` pairs are admitted. The lifetime
    /// cap bounds SDK series even when many catalog generations pass through a
    /// long-lived process; overflow affects telemetry only and collapses to
    /// `(other, other)`.
    pub(crate) fn extend_catalog<I, S>(&self, model: &str, profiles: I) -> bool
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let (candidates, issues) = delta_catalog_candidates(model, profiles);
        if self.inner.is_none() {
            return issues == 0 && candidates.len() <= MAX_CATALOG_MODEL_PROFILE_PAIRS;
        }
        let (exact, warn_once) = {
            let Ok(mut catalog) = self.context.catalog.write() else {
                return false;
            };
            let exact = catalog.extend(candidates) && issues == 0;
            let warn_once = !exact && !catalog.warning_emitted;
            catalog.warning_emitted |= !exact;
            (exact, warn_once)
        };
        if warn_once {
            warn_catalog_collapse();
        }
        exact
    }

    /// Replace the catalog from the backend's successful full snapshot.
    /// Removed queue-depth series are explicitly zeroed and outstanding depth
    /// migrates to the single `(other, other)` pair, keeping queue accounting
    /// balanced across a concurrent catalog replacement.
    pub(crate) fn replace_catalog<I, S, P, Q>(&self, models: I, profiles: P) -> bool
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
        P: IntoIterator<Item = Q>,
        Q: AsRef<str>,
    {
        let (candidates, issues) = snapshot_catalog_candidates(models, profiles);
        if self.inner.is_none() {
            return issues == 0 && candidates.len() <= MAX_CATALOG_MODEL_PROFILE_PAIRS;
        }
        let (updates, exact, warn_once) = {
            let Ok(mut catalog) = self.context.catalog.write() else {
                return false;
            };
            let Ok(mut depths) = self.queue_depths.lock() else {
                return false;
            };

            let exact = catalog.replace(candidates) && issues == 0;
            let removed_keys: Vec<_> = depths
                .keys()
                .filter(|key| {
                    (key.model != OTHER || key.profile != OTHER)
                        && !catalog.pair_is_active(&key.model, &key.profile)
                })
                .cloned()
                .collect();
            let mut updates = Vec::with_capacity(removed_keys.len() * 2);
            for key in removed_keys {
                let value = depths.remove(&key).unwrap_or_default();
                updates.push((key.clone(), 0));
                if value > 0 {
                    let other_key = QueueKey {
                        operation: key.operation,
                        model: OTHER.to_string(),
                        profile: OTHER.to_string(),
                    };
                    let other = depths.entry(other_key.clone()).or_default();
                    *other = other.saturating_add(value);
                    updates.push((other_key, *other));
                }
            }

            let warn_once = !exact && !catalog.warning_emitted;
            catalog.warning_emitted |= !exact;
            (updates, exact, warn_once)
        };

        for (key, value) in updates {
            self.queue_depth.record(
                value,
                &self.context.queue_attributes(
                    key.operation,
                    key.model.as_str(),
                    key.profile.as_str(),
                ),
            );
        }
        if warn_once {
            warn_catalog_collapse();
        }
        exact
    }

    /// Record one item entering the Rust scheduler.
    pub fn queue_enqueued(&self, operation: &str, model: &str, profile: Option<&str>) {
        if self.inner.is_none() {
            return;
        }
        self.update_queue_depth(operation, model, profile, QueueChange::Increment);
    }

    /// Record one item leaving the scheduler for dispatch or cancellation.
    pub fn queue_released(
        &self,
        operation: &str,
        model: &str,
        profile: Option<&str>,
        queued_for: Duration,
    ) {
        if self.inner.is_none() {
            return;
        }
        let operation = bounded_operation(operation);
        let Some((model, profile, depth)) =
            self.update_queue_depth_bounded(operation, model, profile, QueueChange::Decrement)
        else {
            return;
        };
        self.queue_duration.record(
            queued_for.as_secs_f64(),
            &self.context.queue_attributes(operation, model, profile),
        );
        self.queue_depth.record(
            depth,
            &self.context.queue_attributes(operation, model, profile),
        );
    }

    /// Record the two scheduler-local intervals for one successful request ID
    /// in one completed backend batch. The call site owns success filtering
    /// and batch-local request-ID deduplication; this facade owns bounded
    /// dimensions and the two canonical OTel instruments.
    pub fn scheduler_request_batch_completed(
        &self,
        observation: SchedulerRequestBatchObservation<'_>,
    ) {
        if self.inner.is_none() || observation.total.is_zero() {
            return;
        }
        let operation = bounded_operation(observation.operation);
        let (model, profile) = self
            .context
            .model_profile(observation.model, observation.profile);
        let attributes = self.context.queue_attributes(operation, model, profile);
        self.scheduler_request_batch_dispatch_wait
            .record(observation.dispatch_wait.as_secs_f64(), &attributes);
        self.scheduler_request_batch_total
            .record(observation.total.as_secs_f64(), &attributes);
    }

    /// Record one formed batch exactly once, before its backend RPC.
    pub fn batch_formed(
        &self,
        operation: &str,
        model: &str,
        profile: Option<&str>,
        _flush_reason: &str,
        size: usize,
        cost: u64,
    ) {
        if self.inner.is_none() {
            return;
        }
        let operation = bounded_operation(operation);
        let (model, profile) = self.context.model_profile(model, profile);
        let attributes = self
            .context
            .batch_core_attributes(operation, model, profile);
        self.batch_size
            .record(u64::try_from(size).unwrap_or(u64::MAX), &attributes);
        self.batch_cost.record(cost, &attributes);
        // `flush.reason` remains on the fill-ratio diagnostic below. Size and
        // cost answer batch shape and do not need that extra series product.
    }

    /// Publish the adaptive controller's current fill ratio after a primary
    /// batch.  Non-finite values are ignored instead of poisoning the series.
    pub fn batch_fill_observed(
        &self,
        operation: &str,
        model: &str,
        profile: Option<&str>,
        flush_reason: &str,
        fill_ratio: f64,
    ) {
        if self.inner.is_none() || !fill_ratio.is_finite() {
            return;
        }
        let operation = bounded_operation(operation);
        let (model, profile) = self.context.model_profile(model, profile);
        self.batch_fill_ratio.record(
            fill_ratio.clamp(0.0, 1.0),
            &self.context.batch_attributes(
                operation,
                model,
                profile,
                bounded_flush_reason(flush_reason),
            ),
        );
    }

    /// Record one completed logical IPC RPC. `method` and `result` originate
    /// from the typed IPC client, but are still closed here so a future caller
    /// cannot create raw method/error series.
    pub fn ipc_completed(&self, method: &str, result: &str, duration: Duration) {
        if self.inner.is_none() {
            return;
        }
        let attributes = self
            .context
            .ipc_attributes(bounded_ipc_method(method), bounded_ipc_outcome(result));
        self.ipc_requests.add(1, &attributes);
        self.ipc_request_duration
            .record(duration.as_secs_f64(), &attributes);
    }

    /// Record one terminal negotiated IPC response-chunk transfer. A completed
    /// event owns the reconstructed-size and physical-chunk observations too,
    /// so the call site emits one semantic event rather than updating three
    /// instruments independently.
    pub(crate) fn ipc_response_chunk_transfer_completed(
        &self,
        outcome: IpcResponseChunkOutcome,
        reconstructed_bytes: Option<usize>,
        chunk_count: Option<u32>,
    ) {
        if self.inner.is_none() {
            return;
        }
        self.ipc_response_chunks.add(
            1,
            &[
                KeyValue::new("outcome", outcome.as_str()),
                self.context.lane_attribute(),
            ],
        );
        if outcome != IpcResponseChunkOutcome::Completed {
            return;
        }
        let lane = [self.context.lane_attribute()];
        if let Some(reconstructed_bytes) = reconstructed_bytes {
            self.ipc_response_reconstructed_size.record(
                u64::try_from(reconstructed_bytes).unwrap_or(u64::MAX),
                &lane,
            );
        }
        if let Some(chunk_count) = chunk_count {
            self.ipc_response_chunk_count
                .record(u64::from(chunk_count), &lane);
        }
    }

    /// Publish the current process-local IPC response reassembly reservation.
    pub(crate) fn ipc_response_chunk_reserved_changed(&self, bytes: usize) {
        if self.inner.is_none() {
            return;
        }
        self.ipc_response_chunk_reserved.record(
            u64::try_from(bytes).unwrap_or(u64::MAX),
            &[self.context.lane_attribute()],
        );
    }

    /// Record one config-path outcome. An epoch is supplied only after the
    /// corresponding trusted state transition commits. Expected filtering
    /// outcomes leave the degraded gauge unchanged; convergence failures set
    /// it and a later successful apply clears it.
    pub fn config_apply(
        &self,
        source: &str,
        operation: &str,
        outcome: &str,
        applied_epoch: Option<u64>,
    ) {
        if self.inner.is_none() {
            return;
        }
        let source = bounded_config_source(source);
        let operation = bounded_config_operation(operation);
        let outcome = bounded_config_outcome(outcome);
        let attributes = [
            KeyValue::new("source", source),
            KeyValue::new("operation", operation),
            KeyValue::new("outcome", outcome),
            self.context.lane_attribute(),
        ];
        self.config_applies.add(1, &attributes);
        let source_attributes = [
            KeyValue::new("source", source),
            self.context.lane_attribute(),
        ];
        if let Some(epoch) = applied_epoch {
            self.config_epoch.record(epoch, &source_attributes);
        }
        if let Some(degraded) = config_degraded_value(outcome) {
            self.config_degraded
                .record(u64::from(degraded), &source_attributes);
        }
    }

    /// Record a NATS operation with a closed action/outcome/reason vocabulary.
    pub fn nats_operation(&self, operation: &str, outcome: &str, reason: &str, count: u64) {
        if self.inner.is_none() || count == 0 {
            return;
        }
        self.nats_operations.add(
            count,
            &[
                KeyValue::new("operation", bounded_nats_operation(operation)),
                KeyValue::new("outcome", bounded_binary_outcome(outcome)),
                KeyValue::new("reason", bounded_nats_reason(reason)),
                self.context.lane_attribute(),
            ],
        );
    }

    /// Record the one terminal result-transport outcome owned by a published
    /// work result. Mode and outcome are enums so callers cannot create new
    /// series with raw broker errors or request data.
    pub(crate) fn result_transport_completed(
        &self,
        mode: ResultTransportMode,
        outcome: ResultTransportOutcome,
    ) {
        if self.inner.is_none() {
            return;
        }
        self.result_transport_attempts.add(
            1,
            &[
                KeyValue::new("mode", mode.as_str()),
                KeyValue::new("outcome", outcome.as_str()),
                self.context.lane_attribute(),
            ],
        );
    }

    /// Record one successfully queued result-chunk envelope. The counter and
    /// envelope-size histogram are projections of this single semantic event.
    pub(crate) fn result_chunk_published(&self, envelope_bytes: usize) {
        if self.inner.is_none() {
            return;
        }
        let lane = [self.context.lane_attribute()];
        self.result_chunks_published.add(1, &lane);
        self.result_chunk_size
            .record(u64::try_from(envelope_bytes).unwrap_or(u64::MAX), &lane);
    }

    /// Record one JetStream receive and its delivery-attempt ordinal. A
    /// missing metadata envelope remains visible but cannot be mislabeled as
    /// either a first delivery or redelivery.
    pub fn nats_received(&self, delivery_attempt: Option<u64>) {
        if self.inner.is_none() {
            return;
        }
        let reason = match delivery_attempt {
            Some(attempt) if attempt > 1 => "redelivery",
            Some(_) => "first_delivery",
            None => "metadata_unavailable",
        };
        self.nats_operation("receive", "success", reason, 1);
        if let Some(attempt) = delivery_attempt {
            self.nats_delivery_attempts.record(
                attempt.max(1),
                &[
                    KeyValue::new("redelivered", if attempt > 1 { "true" } else { "false" }),
                    self.context.lane_attribute(),
                ],
            );
        }
    }

    /// Record one offloaded payload fetch at the store boundary.
    pub fn payload_fetch_completed(
        &self,
        outcome: &str,
        reason: &str,
        duration: Duration,
        size_bytes: Option<usize>,
    ) {
        if self.inner.is_none() {
            return;
        }
        let outcome = bounded_binary_outcome(outcome);
        let reason = bounded_payload_reason(reason);
        let attributes = [
            KeyValue::new("outcome", outcome),
            KeyValue::new("reason", reason),
            self.context.lane_attribute(),
        ];
        self.payload_fetches.add(1, &attributes);
        self.payload_fetch_duration
            .record(duration.as_secs_f64(), &attributes);
        if let Some(size_bytes) = size_bytes {
            self.payload_size
                .record(u64::try_from(size_bytes).unwrap_or(u64::MAX), &attributes);
        }
    }

    /// Publish the complete process-local pressure snapshot from the shared
    /// runtime heartbeat. This is the single emission point for these current
    /// state families in NATS and local-ingest modes.
    pub fn runtime_snapshot(
        &self,
        total_slots: u64,
        ready_slots: u64,
        pending_items: u64,
        pending_cost: u64,
        inflight_batches: u64,
        saturated: bool,
    ) {
        if self.inner.is_none() {
            return;
        }
        self.gpu_slots.record(
            total_slots,
            &[
                KeyValue::new("state", "total"),
                self.context.lane_attribute(),
            ],
        );
        self.gpu_slots.record(
            ready_slots.min(total_slots),
            &[
                KeyValue::new("state", "ready"),
                self.context.lane_attribute(),
            ],
        );
        let lane = [self.context.lane_attribute()];
        self.pending_items.record(pending_items, &lane);
        self.pending_cost.record(pending_cost, &lane);
        self.inflight_batches.record(inflight_batches, &lane);
        self.saturated.record(u64::from(saturated), &lane);
    }

    /// Register one active IPC transport instance. Admission capacity is
    /// aggregated across adapter children (pool slots or mux command queues),
    /// while inactive pool/mux implementations emit nothing.
    pub fn ipc_transport_registered(&self, transport: &str, capacity: usize) {
        if self.inner.is_none() {
            return;
        }
        let transport = bounded_ipc_transport(transport);
        let capacity = u64::try_from(capacity).unwrap_or(u64::MAX);
        let aggregate = self
            .ipc_capacity_counter(transport)
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(capacity))
            })
            .unwrap_or_default()
            .saturating_add(capacity);
        self.ipc_capacity.record(
            aggregate,
            &[
                KeyValue::new("transport", transport),
                self.context.lane_attribute(),
            ],
        );
        self.ipc_inflight.record(
            self.ipc_inflight_counter(transport).load(Ordering::Acquire),
            &[
                KeyValue::new("transport", transport),
                self.context.lane_attribute(),
            ],
        );
    }

    /// Replace one registered transport instance's contribution after a
    /// builder-level resize without double-counting the discarded pool.
    pub fn ipc_transport_resized(
        &self,
        transport: &str,
        previous_capacity: usize,
        capacity: usize,
    ) {
        if self.inner.is_none() {
            return;
        }
        let transport = bounded_ipc_transport(transport);
        let previous_capacity = u64::try_from(previous_capacity).unwrap_or(u64::MAX);
        let capacity = u64::try_from(capacity).unwrap_or(u64::MAX);
        let aggregate = self
            .ipc_capacity_counter(transport)
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(
                    current
                        .saturating_sub(previous_capacity)
                        .saturating_add(capacity),
                )
            })
            .unwrap_or_default()
            .saturating_sub(previous_capacity)
            .saturating_add(capacity);
        self.ipc_capacity.record(
            aggregate,
            &[
                KeyValue::new("transport", transport),
                self.context.lane_attribute(),
            ],
        );
    }

    /// Record completion of a request-slot acquire. Successful acquires also
    /// advance the shared inflight gauge; callers pair them with
    /// [`Self::ipc_released`].
    pub fn ipc_acquired(&self, transport: &str, outcome: &str, duration: Duration) {
        if self.inner.is_none() {
            return;
        }
        let transport = bounded_ipc_transport(transport);
        let outcome = bounded_binary_outcome(outcome);
        self.ipc_acquire_duration.record(
            duration.as_secs_f64(),
            &[
                KeyValue::new("transport", transport),
                KeyValue::new("outcome", outcome),
                self.context.lane_attribute(),
            ],
        );
        if outcome == "success" {
            let inflight = self.ipc_inflight_counter(transport);
            let value = inflight.fetch_add(1, Ordering::AcqRel).saturating_add(1);
            self.ipc_inflight.record(
                value,
                &[
                    KeyValue::new("transport", transport),
                    self.context.lane_attribute(),
                ],
            );
        }
    }

    /// Release one request slot without allowing a cancelled path to drive the
    /// gauge negative.
    pub fn ipc_released(&self, transport: &str) {
        if self.inner.is_none() {
            return;
        }
        let transport = bounded_ipc_transport(transport);
        let inflight = self.ipc_inflight_counter(transport);
        let value = inflight
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_sub(1))
            })
            .unwrap_or_default()
            .saturating_sub(1);
        self.ipc_inflight.record(
            value,
            &[
                KeyValue::new("transport", transport),
                self.context.lane_attribute(),
            ],
        );
    }

    /// Record the adaptive controller snapshot produced by one primary
    /// scheduler completion. Values are normalized to seconds at the facade.
    #[allow(clippy::too_many_arguments)]
    pub fn adaptive_snapshot(
        &self,
        model: &str,
        profile: Option<&str>,
        wait_ms: f64,
        cost: u64,
        observed_p50_ms: Option<f64>,
        target_p50_ms: Option<f64>,
        starvation_resets_delta: u32,
    ) {
        if self.inner.is_none() {
            return;
        }
        let (model, profile) = self.context.model_profile(model, profile);
        let attributes = self.context.model_profile_attributes(model, profile);
        if wait_ms.is_finite() && wait_ms >= 0.0 {
            self.adaptive_wait.record(wait_ms / 1000.0, &attributes);
        }
        self.adaptive_cost.record(cost, &attributes);
        for (kind, value) in [("observed", observed_p50_ms), ("target", target_p50_ms)] {
            if let Some(value) = value.filter(|value| value.is_finite() && *value >= 0.0) {
                let mut p50_attributes = attributes.to_vec();
                p50_attributes.push(KeyValue::new("kind", kind));
                self.adaptive_p50.record(value / 1000.0, &p50_attributes);
            }
        }
        if starvation_resets_delta > 0 {
            self.starvation_resets
                .add(u64::from(starvation_resets_delta), &attributes);
        }
    }

    /// Record the compound publish+settlement outcome for the typed terminal
    /// generation response emitted while a model is still loading or has
    /// failed to load. Generic NATS settlement remains a separate event.
    pub fn generation_model_loading_response(
        &self,
        model: &str,
        profile: Option<&str>,
        state: &str,
        outcome: &str,
    ) {
        if self.inner.is_none() {
            return;
        }
        let (model, profile) = self.context.model_profile(model, profile);
        let mut attributes = self
            .context
            .model_profile_attributes(model, profile)
            .to_vec();
        attributes.push(KeyValue::new(
            "state",
            bounded_generation_loading_state(state),
        ));
        attributes.push(KeyValue::new(
            "outcome",
            bounded_generation_response_outcome(outcome),
        ));
        self.generation_model_loading_responses.add(1, &attributes);
    }

    /// Record the one backend drain performed during process shutdown. A
    /// deadline-exceeded counter is derived downstream from this histogram's
    /// `outcome` rather than emitted as a second application observation.
    pub fn shutdown_drain_completed(&self, outcome: &str, duration: Duration) {
        if self.inner.is_none() {
            return;
        }
        self.shutdown_drain_duration.record(
            duration.as_secs_f64(),
            &[
                KeyValue::new("outcome", bounded_shutdown_drain_outcome(outcome)),
                self.context.lane_attribute(),
            ],
        );
    }

    fn ipc_inflight_counter(&self, transport: &'static str) -> &AtomicU64 {
        match transport {
            "mux" => &self.ipc_inflight_state.mux,
            _ => &self.ipc_inflight_state.pool,
        }
    }

    fn ipc_capacity_counter(&self, transport: &'static str) -> &AtomicU64 {
        match transport {
            "mux" => &self.ipc_capacity_state.mux,
            _ => &self.ipc_capacity_state.pool,
        }
    }

    fn update_queue_depth(
        &self,
        operation: &str,
        model: &str,
        profile: Option<&str>,
        change: QueueChange,
    ) {
        let operation = bounded_operation(operation);
        let Some((model, profile, value)) =
            self.update_queue_depth_bounded(operation, model, profile, change)
        else {
            return;
        };
        self.queue_depth.record(
            value,
            &self.context.queue_attributes(operation, model, profile),
        );
    }

    fn update_queue_depth_bounded<'a>(
        &'a self,
        operation: &'static str,
        raw_model: &'a str,
        raw_profile: Option<&'a str>,
        change: QueueChange,
    ) -> Option<(&'a str, &'a str, u64)> {
        let catalog = self.context.catalog.read().ok()?;
        let (mut model, mut profile) = catalog
            .resolve(raw_model, raw_profile)
            .unwrap_or((OTHER, OTHER));
        let mut depths = self.queue_depths.lock().ok()?;
        let mut key = QueueKey {
            operation,
            model: model.to_string(),
            profile: profile.to_string(),
        };
        if matches!(change, QueueChange::Decrement)
            && depths.get(&key).copied().unwrap_or_default() == 0
        {
            let other = QueueKey {
                operation,
                model: OTHER.to_string(),
                profile: OTHER.to_string(),
            };
            if depths.get(&other).copied().unwrap_or_default() > 0 {
                key = other;
                model = OTHER;
                profile = OTHER;
            }
        }
        let value = match change {
            QueueChange::Increment => {
                let depth = depths.entry(key).or_default();
                *depth = depth.saturating_add(1);
                *depth
            }
            QueueChange::Decrement => match depths.get(&key).copied().unwrap_or_default() {
                0 => 0,
                1 => {
                    depths.remove(&key);
                    0
                }
                current => {
                    let next = current - 1;
                    depths.insert(key, next);
                    next
                }
            },
        };
        Some((model, profile, value))
    }

    #[cfg(test)]
    pub(crate) fn queue_depth_for_tests(&self, operation: &str, model: &str) -> u64 {
        self.queue_depth_for_profile_tests(operation, model, None)
    }

    #[cfg(test)]
    pub(crate) fn queue_depth_for_profile_tests(
        &self,
        operation: &str,
        model: &str,
        profile: Option<&str>,
    ) -> u64 {
        if self.inner.is_none() {
            return 0;
        }
        let (model, profile) = self.context.model_profile(model, profile);
        let key = QueueKey {
            operation: bounded_operation(operation),
            model: model.to_string(),
            profile: profile.to_string(),
        };
        self.queue_depths
            .lock()
            .ok()
            .and_then(|depths| depths.get(&key).copied())
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn model_allowed_for_tests(&self, model: &str) -> bool {
        if self.inner.is_none() {
            return false;
        }
        self.context.catalog.read().ok().is_some_and(|catalog| {
            catalog.active.contains_key(model) || catalog.aliases.contains_key(model)
        })
    }

    #[cfg(test)]
    pub(crate) fn allowed_model_count_for_tests(&self) -> usize {
        if self.inner.is_none() {
            return 0;
        }
        self.context
            .catalog
            .read()
            .map(|catalog| catalog.active.len())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn allowed_pair_count_for_tests(&self) -> usize {
        if self.inner.is_none() {
            return 0;
        }
        self.context
            .catalog
            .read()
            .map(|catalog| catalog.active_pair_count())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn admitted_pair_count_for_tests(&self) -> usize {
        if self.inner.is_none() {
            return 0;
        }
        self.context
            .catalog
            .read()
            .map(|catalog| catalog.admitted.len())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn pair_allowed_for_tests(&self, model: &str, profile: &str) -> bool {
        if self.inner.is_none() {
            return false;
        }
        self.context
            .catalog
            .read()
            .is_ok_and(|catalog| catalog.pair_is_active(model, profile))
    }

    #[cfg(test)]
    pub(crate) fn profile_allowed_for_tests(&self, profile: &str) -> bool {
        if self.inner.is_none() {
            return false;
        }
        self.context.catalog.read().ok().is_some_and(|catalog| {
            catalog
                .active
                .values()
                .any(|profiles| profiles.contains(profile))
        })
    }

    #[cfg(test)]
    pub(crate) fn queue_series_count_for_tests(&self) -> usize {
        if self.inner.is_none() {
            return 0;
        }
        self.queue_depths
            .lock()
            .map(|depths| depths.len())
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn for_tests(models: &[&str]) -> Self {
        Self::new(
            &global::meter("sie-worker-sidecar-test-context"),
            TelemetryContext::new(
                "test|test|test",
                models.iter().map(|model| (*model).to_string()).collect(),
            ),
            true,
        )
    }

    #[cfg(test)]
    pub(crate) fn enabled_for_tests(&self) -> bool {
        self.inner.is_some()
    }
}

#[derive(Clone, Copy)]
enum QueueChange {
    Increment,
    Decrement,
}

fn cleaned_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_allowlist(raw: &str) -> HashSet<String> {
    bounded_model_set(raw.split(',')).0
}

fn bounded_catalog_model(raw: &str) -> Option<&str> {
    let model = raw.trim();
    if model == OTHER || model.len() > MAX_CATALOG_MODEL_LABEL_BYTES || model.contains("..") {
        return None;
    }

    // Colons are the profile-variant separator, but base SIE IDs are not
    // otherwise forbidden from containing them. Full snapshots disambiguate a
    // trailing profile using the declared profile set and base-model presence.
    if !model
        .split(':')
        .all(|segment| valid_catalog_segment(segment, true))
    {
        return None;
    }

    Some(model)
}

fn bounded_catalog_profile(raw: &str) -> Option<&str> {
    let profile = raw.trim();
    if profile == OTHER
        || profile.len() > MAX_CATALOG_PROFILE_LABEL_BYTES
        || !valid_catalog_segment(profile, false)
    {
        return None;
    }
    Some(profile)
}

fn valid_catalog_segment(segment: &str, allow_slash: bool) -> bool {
    let mut bytes = segment.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric())
        && bytes.all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'.' | b'_' | b'-')
                || (allow_slash && byte == b'/')
        })
}

fn bounded_model_set<I, S>(models: I) -> (HashSet<String>, usize)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut allowed = HashSet::new();
    let mut dropped = 0_usize;
    for raw in models {
        let Some(model) = bounded_catalog_model(raw.as_ref()) else {
            dropped = dropped.saturating_add(1);
            continue;
        };
        if allowed.contains(model) {
            continue;
        }
        if allowed.len() >= MAX_CATALOG_MODEL_PROFILE_PAIRS {
            dropped = dropped.saturating_add(1);
            continue;
        }
        allowed.insert(model.to_string());
    }
    (allowed, dropped)
}

fn bounded_profile_set<I, S>(profiles: I) -> (HashSet<String>, usize)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut allowed = HashSet::new();
    let mut dropped = 0_usize;
    for raw in profiles {
        let Some(profile) = bounded_catalog_profile(raw.as_ref()) else {
            dropped = dropped.saturating_add(1);
            continue;
        };
        if allowed.contains(profile) {
            continue;
        }
        if allowed.len() >= MAX_CATALOG_MODEL_PROFILE_PAIRS {
            dropped = dropped.saturating_add(1);
            continue;
        }
        allowed.insert(profile.to_string());
    }
    (allowed, dropped)
}

fn catalog_alias(model: &str, profile: &str) -> Option<String> {
    if profile == DEFAULT_PROFILE {
        return Some(model.to_string());
    }
    let alias = format!("{model}:{profile}");
    bounded_catalog_model(&alias).map(str::to_string)
}

fn startup_catalog_candidates<I, S>(models: I) -> (Vec<CatalogAlias>, usize)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let (models, issues) = bounded_model_set(models);
    let mut models: Vec<_> = models.into_iter().collect();
    models.sort();
    (
        models
            .into_iter()
            .map(|model| CatalogAlias {
                raw_model: Some(model.clone()),
                pair: CatalogPair {
                    model,
                    profile: DEFAULT_PROFILE.to_string(),
                },
            })
            .collect(),
        issues,
    )
}

fn delta_catalog_candidates<I, S>(model: &str, profiles: I) -> (Vec<CatalogAlias>, usize)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let Some(model) = bounded_catalog_model(model) else {
        return (Vec::new(), 1);
    };
    let (mut profiles, mut issues) = bounded_profile_set(profiles);
    if profiles.is_empty() && issues == 0 {
        profiles.insert(DEFAULT_PROFILE.to_string());
    }
    let mut profiles: Vec<_> = profiles.into_iter().collect();
    profiles.sort();
    let candidates = profiles
        .into_iter()
        .map(|profile| {
            let raw_model = catalog_alias(model, &profile);
            issues += usize::from(raw_model.is_none());
            CatalogAlias {
                raw_model,
                pair: CatalogPair {
                    model: model.to_string(),
                    profile,
                },
            }
        })
        .collect();
    (candidates, issues)
}

fn snapshot_catalog_candidates<I, S, P, Q>(models: I, profiles: P) -> (Vec<CatalogAlias>, usize)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
    P: IntoIterator<Item = Q>,
    Q: AsRef<str>,
{
    let (models, dropped_models) = bounded_model_set(models);
    let (profiles, dropped_profiles) = bounded_profile_set(profiles);
    let mut issues = dropped_models.saturating_add(dropped_profiles);
    if models.is_empty() && profiles.is_empty() {
        return (Vec::new(), issues);
    }
    if models.is_empty() || profiles.is_empty() {
        return (Vec::new(), issues.saturating_add(1));
    }

    let has_default = profiles.contains(DEFAULT_PROFILE);
    let mut represented_profiles = HashSet::new();
    let mut sorted_models: Vec<_> = models.iter().cloned().collect();
    sorted_models.sort();
    let mut candidates = Vec::with_capacity(sorted_models.len());

    for raw_model in sorted_models {
        let variant = raw_model.rsplit_once(':').filter(|(base, profile)| {
            *profile != DEFAULT_PROFILE
                && profiles.contains(*profile)
                && bounded_catalog_model(base).is_some()
                && (models.contains(*base) || !has_default)
        });
        let pair = if let Some((base, profile)) = variant {
            CatalogPair {
                model: base.to_string(),
                profile: profile.to_string(),
            }
        } else if has_default {
            CatalogPair {
                model: raw_model.clone(),
                profile: DEFAULT_PROFILE.to_string(),
            }
        } else {
            issues = issues.saturating_add(1);
            continue;
        };
        represented_profiles.insert(pair.profile.clone());
        candidates.push(CatalogAlias {
            raw_model: Some(raw_model),
            pair,
        });
    }

    issues = issues.saturating_add(profiles.symmetric_difference(&represented_profiles).count());
    (candidates, issues)
}

fn warn_catalog_collapse() {
    tracing::warn!(
        max_pairs = MAX_CATALOG_MODEL_PROFILE_PAIRS,
        "sidecar telemetry catalog exceeded or violated its bounded model/profile contract; affected observations collapse to other"
    );
}

fn nonempty_or_other(value: &str) -> &str {
    let value = value.trim();
    if value.is_empty() {
        OTHER
    } else {
        value
    }
}

fn bounded_operation(operation: &str) -> &'static str {
    match operation {
        "encode" => "encode",
        "score" => "score",
        "extract" => "extract",
        "embeddings" => "embeddings",
        "moderations" => "moderations",
        "generate" => "generate",
        _ => OTHER,
    }
}

fn bounded_flush_reason(reason: &str) -> &'static str {
    match reason {
        "cost_cap" => "cost_cap",
        "count_cap" => "count_cap",
        "timeout" => "timeout",
        "coalesce" => "coalesce",
        "single_oversize" => "single_oversize",
        "idle_bypass" => "idle_bypass",
        "drain" => "drain",
        _ => OTHER,
    }
}

fn bounded_ipc_method(method: &str) -> &'static str {
    match method {
        "ApplyModelConfig" => "apply_model_config",
        "Drain" => "drain",
        "EnsureModelReady" => "ensure_model_ready",
        "Ping" => "ping",
        "ProcessEncodeBatch" => "process_encode_batch",
        "ProcessExtractBatch" => "process_extract_batch",
        "ProcessGenerate" => "process_generate",
        "ProcessScoreBatch" => "process_score_batch",
        "ReplaceModelConfigs" => "replace_model_configs",
        "RunBatch" => "run_batch",
        "SetPinnedModels" => "set_pinned_models",
        "SignalGenerateCancel" => "signal_generate_cancel",
        "WorkerCapabilities" => "worker_capabilities",
        _ => OTHER,
    }
}

fn bounded_ipc_outcome(result: &str) -> &'static str {
    match result {
        "ok" => "success",
        "ok_after_retry" => "success_after_retry",
        "io" => "transport_error",
        "encode" | "decode" | "frame_too_large" | "response_chunk" | "version_mismatch" => {
            "protocol_error"
        }
        "timeout" => "timeout",
        "server" => "server_error",
        _ => OTHER,
    }
}

fn bounded_binary_outcome(outcome: &str) -> &'static str {
    match outcome {
        "success" | "ok" => "success",
        "error" | "failed" => "error",
        _ => OTHER,
    }
}

fn bounded_config_source(source: &str) -> &'static str {
    match source {
        "notification" => "notification",
        "export" => "export",
        _ => OTHER,
    }
}

fn bounded_config_operation(operation: &str) -> &'static str {
    match operation {
        "model_config" => "model_config",
        "epoch_bump" => "epoch_bump",
        "full_export" | "export" => "full_export",
        "poll" => "poll",
        "subscribe" => "subscribe",
        _ => OTHER,
    }
}

fn bounded_config_outcome(outcome: &str) -> &'static str {
    match outcome {
        "applied" => "applied",
        "no_change" => "no_change",
        "no_relevant_models" => "no_relevant_models",
        "stale" | "stale_export" => "stale",
        "skipped_pool" => "skipped_pool",
        "rejected_untrusted" => "rejected_untrusted",
        "rejected_bundle" => "rejected_bundle",
        "rejected_oversized" => "rejected_oversized",
        "parse_error" => "parse_error",
        "client_error" => "client_error",
        "fetch_error" => "fetch_error",
        "epoch_error" => "epoch_error",
        "apply_error" => "apply_error",
        "apply_rejected" => "apply_rejected",
        "hash_mismatch" => "hash_mismatch",
        "partial" => "partial",
        "shutdown" => "shutdown",
        _ => OTHER,
    }
}

fn config_degraded_value(outcome: &str) -> Option<bool> {
    match outcome {
        "applied" | "no_change" | "no_relevant_models" => Some(false),
        "client_error" | "fetch_error" | "epoch_error" | "apply_error" | "apply_rejected"
        | "hash_mismatch" | "partial" => Some(true),
        _ => None,
    }
}

fn bounded_nats_operation(operation: &str) -> &'static str {
    match operation {
        "receive" => "receive",
        "ack" => "ack",
        "nak" => "nak",
        "progress" => "progress",
        "fetch" => "fetch",
        "stream" => "stream",
        _ => OTHER,
    }
}

fn bounded_nats_reason(reason: &str) -> &'static str {
    match reason {
        "none" => "none",
        "completed" => "completed",
        "retry" => "retry",
        "pool_not_assigned" => "pool_not_assigned",
        "first_delivery" => "first_delivery",
        "redelivery" => "redelivery",
        "metadata_unavailable" => "metadata_unavailable",
        "transport" => "transport",
        "stream_ended" => "stream_ended",
        _ => OTHER,
    }
}

fn bounded_payload_reason(reason: &str) -> &'static str {
    match reason {
        "none" => "none",
        "not_found" => "not_found",
        "permission_denied" => "permission_denied",
        "io" => "io",
        "invalid_ref" => "invalid_ref",
        "unsupported" => "unsupported",
        "object_store" => "object_store",
        _ => OTHER,
    }
}

fn bounded_ipc_transport(transport: &str) -> &'static str {
    match transport {
        "pool" => "pool",
        "mux" => "mux",
        _ => OTHER,
    }
}

fn bounded_generation_loading_state(state: &str) -> &'static str {
    match state {
        "loading_started" => "loading_started",
        "loading_in_progress" => "loading_in_progress",
        "failed" => "failed",
        _ => OTHER,
    }
}

fn bounded_generation_response_outcome(outcome: &str) -> &'static str {
    match outcome {
        "success" | "published_acked" => "success",
        "ack_error" | "published_ack_failed" => "ack_error",
        "publish_error" | "publish_failed" => "publish_error",
        _ => OTHER,
    }
}

fn bounded_shutdown_drain_outcome(outcome: &str) -> &'static str {
    match outcome {
        "success" | "ok" => "success",
        "deadline_exceeded" => "deadline_exceeded",
        _ => OTHER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{
        InMemoryMetricExporter, InMemoryMetricExporterBuilder, PeriodicReader, SdkMeterProvider,
        Temporality,
    };

    fn test_metrics(meter: &opentelemetry::metrics::Meter, models: &[&str]) -> SidecarTelemetry {
        SidecarTelemetry::new(
            meter,
            TelemetryContext::new(
                "default|l4|default",
                models.iter().map(|model| (*model).to_string()).collect(),
            ),
            true,
        )
    }

    #[test]
    fn queue_depth_balances_and_unknown_values_collapse() {
        let meter = opentelemetry::global::meter("sidecar-telemetry-test");
        let metrics = test_metrics(&meter, &["model-a"]);
        assert_eq!(
            metrics
                .context
                .model_profile("model-a", Some("caller-selected-runtime"))
                .1,
            OTHER,
            "a raw WorkItem profile_id must collapse until catalog allowlisting is wired"
        );
        assert!(metrics.extend_catalog("model-a", ["catalog-profile"]));
        assert_eq!(
            metrics
                .context
                .model_profile("model-a", Some("catalog-profile"))
                .1,
            "catalog-profile"
        );

        metrics.queue_enqueued("encode", "model-a", Some("catalog-profile"));
        metrics.queue_enqueued("encode", "model-a", Some("catalog-profile"));
        assert_eq!(
            metrics.queue_depth_for_profile_tests("encode", "model-a", Some("catalog-profile")),
            2
        );
        metrics.queue_released(
            "encode",
            "model-a",
            Some("catalog-profile"),
            Duration::from_millis(3),
        );
        metrics.queue_released(
            "encode",
            "model-a",
            Some("catalog-profile"),
            Duration::from_millis(4),
        );
        assert_eq!(
            metrics.queue_depth_for_profile_tests("encode", "model-a", Some("catalog-profile")),
            0
        );
        assert_eq!(metrics.queue_series_count_for_tests(), 0);

        metrics.queue_enqueued("generate", "caller/model", None);
        assert_eq!(metrics.queue_depth_for_tests("generate", "other"), 1);
        metrics.queue_released("generate", "caller/model", None, Duration::ZERO);
        assert_eq!(metrics.queue_depth_for_tests("generate", "other"), 0);
        assert_eq!(metrics.queue_series_count_for_tests(), 0);

        for operation in [
            "encode",
            "score",
            "extract",
            "embeddings",
            "moderations",
            "generate",
        ] {
            assert_eq!(bounded_operation(operation), operation);
        }
        assert_eq!(bounded_operation("caller-defined-operation"), OTHER);
    }

    #[test]
    fn successful_catalog_updates_promote_and_replace_models_exactly() {
        let telemetry = SidecarTelemetry::for_tests(&["startup/model"]);

        assert!(!telemetry.model_allowed_for_tests("live/model"));
        assert!(telemetry.extend_catalog("live/model", [] as [&str; 0]));
        assert!(telemetry.model_allowed_for_tests("live/model"));
        assert!(telemetry.replace_catalog(["export/model-a", "export/model-b"], [DEFAULT_PROFILE]));

        assert!(!telemetry.model_allowed_for_tests("startup/model"));
        assert!(!telemetry.model_allowed_for_tests("live/model"));
        assert!(telemetry.model_allowed_for_tests("export/model-a"));
        assert!(telemetry.model_allowed_for_tests("export/model-b"));
        assert_eq!(telemetry.allowed_model_count_for_tests(), 2);
    }

    #[test]
    fn catalog_allowlist_accepts_one_profile_qualifier() {
        let telemetry = SidecarTelemetry::for_tests(&[]);

        assert!(telemetry.replace_catalog(
            ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-27B:rtx-pro-6000"],
            [DEFAULT_PROFILE, "rtx-pro-6000"],
        ));
        assert!(telemetry.model_allowed_for_tests("Qwen/Qwen3.6-27B"));
        assert!(telemetry.model_allowed_for_tests("Qwen/Qwen3.6-27B:rtx-pro-6000"));
        assert!(telemetry.profile_allowed_for_tests("rtx-pro-6000"));
        assert!(telemetry.pair_allowed_for_tests("Qwen/Qwen3.6-27B", DEFAULT_PROFILE));
        assert!(telemetry.pair_allowed_for_tests("Qwen/Qwen3.6-27B", "rtx-pro-6000"));
        assert_eq!(telemetry.allowed_model_count_for_tests(), 1);
        assert_eq!(telemetry.allowed_pair_count_for_tests(), 2);
    }

    #[test]
    fn catalog_pair_parser_handles_colons_and_rejects_malformed_ids() {
        let telemetry = SidecarTelemetry::for_tests(&[]);

        for model in [":profile", "model:", "model::profile", "model:profile?"] {
            assert!(
                !telemetry.extend_catalog(model, [] as [&str; 0]),
                "accepted {model}"
            );
        }
        assert!(telemetry.extend_catalog("tenant:namespace/model", [] as [&str; 0]));
        assert!(telemetry.replace_catalog(
            [
                "tenant:namespace/model",
                "tenant:namespace/model:throughput"
            ],
            [DEFAULT_PROFILE, "throughput"]
        ));
        assert!(telemetry.pair_allowed_for_tests("tenant:namespace/model", "throughput"));
        assert_eq!(
            telemetry
                .context
                .model_profile("tenant:namespace/model:throughput", Some(DEFAULT_PROFILE)),
            ("tenant:namespace/model", "throughput")
        );
    }

    #[test]
    fn catalog_allowlist_enforces_model_label_byte_bound() {
        let telemetry = SidecarTelemetry::for_tests(&[]);
        let at_bound = format!("{}:p", "x".repeat(MAX_CATALOG_MODEL_LABEL_BYTES - 2));
        let over_bound = format!("{}:p", "x".repeat(MAX_CATALOG_MODEL_LABEL_BYTES - 1));

        assert_eq!(at_bound.len(), MAX_CATALOG_MODEL_LABEL_BYTES);
        assert!(telemetry.extend_catalog(&at_bound, [] as [&str; 0]));
        assert_eq!(over_bound.len(), MAX_CATALOG_MODEL_LABEL_BYTES + 1);
        assert!(!telemetry.extend_catalog(&over_bound, [] as [&str; 0]));
        assert_eq!(telemetry.allowed_model_count_for_tests(), 1);
    }

    #[test]
    fn catalog_allowlist_rejects_invalid_values_and_stays_bounded_under_concurrency() {
        use std::sync::atomic::AtomicUsize;

        let telemetry = Arc::new(SidecarTelemetry::for_tests(&[]));
        assert!(!telemetry.extend_catalog("", [] as [&str; 0]));
        assert!(!telemetry.extend_catalog("../caller-controlled", [] as [&str; 0]));
        assert!(!telemetry.extend_catalog(
            &"x".repeat(MAX_CATALOG_MODEL_LABEL_BYTES + 1),
            [] as [&str; 0]
        ));

        let admitted = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for shard in 0..8 {
            let telemetry = Arc::clone(&telemetry);
            let admitted = Arc::clone(&admitted);
            tasks.push(std::thread::spawn(move || {
                for model in 0..128 {
                    if telemetry
                        .extend_catalog(&format!("catalog/{shard}-{model}"), [] as [&str; 0])
                    {
                        admitted.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        for task in tasks {
            task.join().expect("catalog update task");
        }
        assert_eq!(
            admitted.load(Ordering::Relaxed),
            MAX_CATALOG_MODEL_PROFILE_PAIRS
        );
        assert_eq!(
            telemetry.allowed_pair_count_for_tests(),
            MAX_CATALOG_MODEL_PROFILE_PAIRS
        );
        assert_eq!(
            telemetry.admitted_pair_count_for_tests(),
            MAX_CATALOG_MODEL_PROFILE_PAIRS
        );

        let bounded = SidecarTelemetry::for_tests(&[]);
        let oversized: Vec<_> = (0..=MAX_CATALOG_MODEL_PROFILE_PAIRS)
            .map(|index| format!("catalog/bounded-{index}"))
            .collect();
        assert!(!bounded.replace_catalog(&oversized, [DEFAULT_PROFILE]));
        assert_eq!(
            bounded.allowed_pair_count_for_tests(),
            MAX_CATALOG_MODEL_PROFILE_PAIRS
        );

        bounded.queue_enqueued("encode", "catalog/bounded-256", None);
        assert_eq!(bounded.queue_depth_for_tests("encode", OTHER), 1);
        assert!(!bounded.model_allowed_for_tests("catalog/bounded-256"));
    }

    #[test]
    fn catalog_snapshot_deduplicates_pairs_and_rejects_non_cartesian_mismatch() {
        let telemetry = SidecarTelemetry::for_tests(&[]);

        assert!(telemetry.replace_catalog(
            ["model-a", "model-a", "model-a:fast", "model-b"],
            [DEFAULT_PROFILE, "fast", "fast"]
        ));
        assert_eq!(telemetry.allowed_pair_count_for_tests(), 3);
        assert!(telemetry.pair_allowed_for_tests("model-a", "fast"));
        assert!(!telemetry.pair_allowed_for_tests("model-b", "fast"));
        assert_eq!(
            telemetry.context.model_profile("model-b", Some("fast")),
            (OTHER, OTHER)
        );

        assert!(!telemetry.replace_catalog(
            ["model-a", "model-a:fast", "model-b"],
            [DEFAULT_PROFILE, "fast", "ghost"]
        ));
        assert_eq!(telemetry.allowed_pair_count_for_tests(), 3);
        assert!(!telemetry.profile_allowed_for_tests("ghost"));
    }

    #[test]
    fn catalog_replace_migrates_outstanding_queue_depth_to_other() {
        let meter = opentelemetry::global::meter("sidecar-catalog-replace-test");
        let telemetry = test_metrics(&meter, &["old/model"]);

        telemetry.queue_enqueued("encode", "old/model", None);
        assert!(telemetry.replace_catalog(["new/model"], [DEFAULT_PROFILE]));
        assert_eq!(telemetry.queue_depth_for_tests("encode", OTHER), 1);
        assert!(!telemetry.model_allowed_for_tests("old/model"));
        assert!(telemetry.model_allowed_for_tests("new/model"));

        telemetry.queue_released("encode", "old/model", None, Duration::ZERO);
        assert_eq!(telemetry.queue_depth_for_tests("encode", OTHER), 0);
    }

    #[test]
    fn catalog_replace_migrates_removed_pair_depth_to_other() {
        let meter = opentelemetry::global::meter("sidecar-profile-replace-test");
        let telemetry = test_metrics(&meter, &["model-a", "model-a:fast"]);
        assert!(telemetry.replace_catalog(["model-a", "model-a:fast"], [DEFAULT_PROFILE, "fast"]));
        assert!(telemetry.profile_allowed_for_tests("fast"));

        telemetry.queue_enqueued("encode", "model-a", Some("fast"));
        assert!(telemetry.replace_catalog(["model-a"], [DEFAULT_PROFILE]));
        assert_eq!(telemetry.queue_depth_for_tests("encode", OTHER), 1);

        telemetry.queue_released("encode", "model-a", Some("fast"), Duration::ZERO);
        assert_eq!(telemetry.queue_depth_for_tests("encode", OTHER), 0);
        assert_eq!(telemetry.queue_series_count_for_tests(), 0);
    }

    #[test]
    fn sidecar_cardinality_view_covers_every_owned_instrument() {
        let names = [
            QUEUE_DURATION_METRIC_NAME,
            QUEUE_DEPTH_METRIC_NAME,
            SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME,
            SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME,
            BATCH_SIZE_METRIC_NAME,
            BATCH_COST_METRIC_NAME,
            BATCH_FILL_RATIO_METRIC_NAME,
            IPC_REQUESTS_METRIC_NAME,
            IPC_REQUEST_DURATION_METRIC_NAME,
            IPC_RESPONSE_CHUNKS_METRIC_NAME,
            IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME,
            IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME,
            IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME,
            CONFIG_APPLIES_METRIC_NAME,
            CONFIG_EPOCH_METRIC_NAME,
            CONFIG_DEGRADED_METRIC_NAME,
            NATS_OPERATIONS_METRIC_NAME,
            NATS_DELIVERY_ATTEMPTS_METRIC_NAME,
            RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME,
            RESULT_CHUNKS_PUBLISHED_METRIC_NAME,
            RESULT_CHUNK_SIZE_METRIC_NAME,
            PAYLOAD_FETCHES_METRIC_NAME,
            PAYLOAD_FETCH_DURATION_METRIC_NAME,
            PAYLOAD_SIZE_METRIC_NAME,
            GPU_SLOTS_METRIC_NAME,
            PENDING_ITEMS_METRIC_NAME,
            PENDING_COST_METRIC_NAME,
            INFLIGHT_BATCHES_METRIC_NAME,
            SATURATED_METRIC_NAME,
            IPC_CAPACITY_METRIC_NAME,
            IPC_INFLIGHT_METRIC_NAME,
            IPC_ACQUIRE_DURATION_METRIC_NAME,
            ADAPTIVE_WAIT_METRIC_NAME,
            ADAPTIVE_COST_METRIC_NAME,
            ADAPTIVE_P50_METRIC_NAME,
            STARVATION_RESETS_METRIC_NAME,
            GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME,
            SHUTDOWN_DRAIN_DURATION_METRIC_NAME,
        ];
        assert!(names
            .iter()
            .all(|name| sidecar_metric_cardinality_limit(name).is_some()));
        assert!(sidecar_metric_cardinality_limit("not-a-contract-metric").is_none());
        assert_eq!(SIDECAR_QUEUE_CARDINALITY_LIMIT, 7 * 257);
        assert_eq!(SIDECAR_BATCH_FILL_CARDINALITY_LIMIT, 7 * 257 * 8);
        assert_eq!(SIDECAR_IPC_RESPONSE_CHUNK_CARDINALITY_LIMIT, 3);
        assert_eq!(SIDECAR_RESULT_TRANSPORT_CARDINALITY_LIMIT, 4 * 4);
        assert_eq!(SIDECAR_GENERATION_LOADING_CARDINALITY_LIMIT, 257 * 4 * 4);
    }

    #[test]
    fn high_product_valid_domains_never_use_sdk_overflow_series() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_view(crate::observability::tracing::sidecar_metric_cardinality_view)
            .build();
        let meter = provider.meter("sidecar-cardinality-view-test");
        let telemetry = SidecarTelemetry::new(
            &meter,
            TelemetryContext::new("default|l4|default", HashSet::new()),
            true,
        );
        let models: Vec<_> = (0..MAX_CATALOG_MODEL_PROFILE_PAIRS)
            .map(|index| format!("catalog/model-{index}"))
            .collect();
        assert!(telemetry.replace_catalog(&models, [DEFAULT_PROFILE]));

        let operations = [
            "encode",
            "score",
            "extract",
            "embeddings",
            "moderations",
            "generate",
            "unknown-operation",
        ];
        let flush_reasons = [
            "cost_cap",
            "count_cap",
            "timeout",
            "coalesce",
            "single_oversize",
            "idle_bypass",
            "drain",
            "unknown-reason",
        ];
        let loading_states = [
            "loading_started",
            "loading_in_progress",
            "failed",
            "unknown-state",
        ];
        let loading_outcomes = ["success", "ack_error", "publish_error", "unknown-outcome"];
        for model in models.iter().map(String::as_str).chain(["unknown/model"]) {
            for operation in operations {
                for flush_reason in flush_reasons {
                    telemetry.batch_fill_observed(operation, model, None, flush_reason, 0.5);
                }
            }
            for state in loading_states {
                for outcome in loading_outcomes {
                    telemetry.generation_model_loading_response(model, None, state, outcome);
                }
            }
        }

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let exported: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();

        let fill = exported
            .iter()
            .find(|metric| metric.name() == BATCH_FILL_RATIO_METRIC_NAME)
            .expect("batch fill ratio");
        let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = fill.data() else {
            panic!("batch fill ratio must be an f64 gauge")
        };
        assert_eq!(
            gauge.data_points().count(),
            SIDECAR_BATCH_FILL_CARDINALITY_LIMIT
        );
        assert!(gauge.data_points().all(|point| point
            .attributes()
            .all(|attribute| attribute.key.as_str() != "otel.metric.overflow")));

        let loading = exported
            .iter()
            .find(|metric| metric.name() == GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME)
            .expect("generation loading responses");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = loading.data() else {
            panic!("generation loading responses must be a u64 sum")
        };
        assert_eq!(
            sum.data_points().count(),
            SIDECAR_GENERATION_LOADING_CARDINALITY_LIMIT
        );
        assert!(sum.data_points().all(|point| point
            .attributes()
            .all(|attribute| attribute.key.as_str() != "otel.metric.overflow")));
    }

    #[test]
    fn disabled_state_does_not_build_enabled_sdk_state() {
        let build_called = std::cell::Cell::new(false);
        let telemetry = SidecarTelemetry::from_enabled_builder(false, || {
            build_called.set(true);
            panic!("disabled telemetry must not construct SDK instruments or state maps")
        });

        assert!(!build_called.get());
        assert!(!telemetry.is_enabled());
        assert!(!telemetry.enabled_for_tests());
        telemetry.queue_enqueued("encode", "model-a", None);
        assert_eq!(telemetry.queue_depth_for_tests("encode", "model-a"), 0);
    }

    #[test]
    fn disabled_facade_records_no_observations() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("disabled-sidecar-telemetry-test");
        let telemetry = SidecarTelemetry::new(
            &meter,
            TelemetryContext::new("default|l4|default", HashSet::from(["model-a".to_string()])),
            false,
        );
        assert!(!telemetry.enabled_for_tests());

        telemetry.queue_enqueued("encode", "model-a", None);
        telemetry.queue_released("encode", "model-a", None, Duration::from_millis(1));
        telemetry.scheduler_request_batch_completed(SchedulerRequestBatchObservation {
            operation: "encode",
            model: "model-a",
            profile: None,
            dispatch_wait: Duration::from_millis(1),
            total: Duration::from_millis(2),
        });
        telemetry.batch_formed("encode", "model-a", None, "cost_cap", 1, 128);
        telemetry.batch_fill_observed("encode", "model-a", None, "cost_cap", 0.5);
        telemetry.ipc_completed("RunBatch", "ok", Duration::from_millis(1));
        telemetry.ipc_response_chunk_transfer_completed(
            IpcResponseChunkOutcome::Completed,
            Some(4096),
            Some(2),
        );
        telemetry.ipc_response_chunk_reserved_changed(8192);
        telemetry.config_apply("notification", "model_config", "applied", Some(7));
        telemetry.nats_received(Some(2));
        telemetry.nats_operation("ack", "success", "completed", 1);
        telemetry.result_transport_completed(
            ResultTransportMode::Chunked,
            ResultTransportOutcome::Published,
        );
        telemetry.result_chunk_published(1024);
        telemetry.payload_fetch_completed("success", "none", Duration::from_millis(1), Some(1024));
        telemetry.runtime_snapshot(2, 1, 3, 128, 1, true);
        telemetry.ipc_transport_registered("pool", 4);
        telemetry.ipc_acquired("pool", "success", Duration::from_micros(5));
        telemetry.ipc_released("pool");
        telemetry.adaptive_snapshot("model-a", None, 5.0, 4096, Some(12.0), Some(20.0), 1);
        telemetry.generation_model_loading_response("model-a", None, "loading_started", "success");
        telemetry.shutdown_drain_completed("success", Duration::from_millis(1));

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        assert!(
            resource_metrics
                .iter()
                .flat_map(|resource| resource.scope_metrics())
                .flat_map(|scope| scope.metrics())
                .next()
                .is_none(),
            "disabled telemetry must be a complete no-op"
        );
        assert_eq!(telemetry.queue_depth_for_tests("encode", "model-a"), 0);
    }

    #[test]
    fn exports_exact_dotted_contract_with_bounded_attributes() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("sie-worker-sidecar");
        let metrics = test_metrics(&meter, &["model-a"]);
        assert!(metrics.extend_catalog("model-a", ["fast"]));

        metrics.queue_enqueued("encode", "model-a", Some("fast"));
        metrics.queue_released("encode", "model-a", Some("fast"), Duration::from_millis(5));
        metrics.scheduler_request_batch_completed(SchedulerRequestBatchObservation {
            operation: "encode",
            model: "model-a",
            profile: Some("fast"),
            dispatch_wait: Duration::from_millis(7),
            total: Duration::from_millis(19),
        });
        metrics.batch_formed("encode", "model-a", Some("fast"), "cost_cap", 8, 4096);
        metrics.batch_fill_observed("encode", "model-a", Some("fast"), "cost_cap", 1.75);
        metrics.batch_formed(
            "caller-defined-operation",
            "caller/model",
            Some("caller-profile"),
            "caller-value",
            1,
            1,
        );
        metrics.ipc_completed("RunBatch", "ok_after_retry", Duration::from_millis(7));
        metrics.ipc_completed(
            "caller-defined-method",
            "customer error text",
            Duration::from_millis(9),
        );
        metrics.ipc_response_chunk_transfer_completed(
            IpcResponseChunkOutcome::Completed,
            Some(16_384),
            Some(4),
        );
        metrics.ipc_response_chunk_transfer_completed(
            IpcResponseChunkOutcome::ProtocolError,
            None,
            None,
        );
        metrics.ipc_response_chunk_reserved_changed(32_768);
        metrics.config_apply("notification", "model_config", "apply_error", None);
        metrics.config_apply("notification", "model_config", "applied", Some(17));
        metrics.nats_received(Some(2));
        metrics.nats_operation("ack", "error", "completed", 1);
        metrics.result_transport_completed(
            ResultTransportMode::Chunked,
            ResultTransportOutcome::Published,
        );
        metrics.result_chunk_published(4096);
        metrics.payload_fetch_completed("success", "none", Duration::from_millis(3), Some(16_384));
        metrics.payload_fetch_completed("error", "not_found", Duration::from_millis(1), None);
        metrics.runtime_snapshot(4, 3, 7, 4096, 2, true);
        metrics.ipc_transport_registered("pool", 4);
        metrics.ipc_transport_registered("pool", 2);
        metrics.ipc_transport_resized("pool", 2, 3);
        metrics.ipc_acquired("pool", "success", Duration::from_micros(250));
        metrics.ipc_released("pool");
        metrics.adaptive_snapshot(
            "model-a",
            Some("fast"),
            7.5,
            8192,
            Some(20.0),
            Some(25.0),
            2,
        );
        metrics.generation_model_loading_response(
            "model-a",
            Some("fast"),
            "loading_in_progress",
            "published_ack_failed",
        );
        metrics.generation_model_loading_response(
            "caller/model",
            Some("caller-profile"),
            "caller-state",
            "customer error text",
        );
        metrics.shutdown_drain_completed("deadline_exceeded", Duration::from_secs(12));

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let exported: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        let names: HashSet<_> = exported
            .iter()
            .map(|metric| metric.name().to_string())
            .collect();
        assert_eq!(
            names,
            HashSet::from([
                QUEUE_DURATION_METRIC_NAME.to_string(),
                QUEUE_DEPTH_METRIC_NAME.to_string(),
                SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME.to_string(),
                SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME.to_string(),
                BATCH_SIZE_METRIC_NAME.to_string(),
                BATCH_COST_METRIC_NAME.to_string(),
                BATCH_FILL_RATIO_METRIC_NAME.to_string(),
                IPC_REQUESTS_METRIC_NAME.to_string(),
                IPC_REQUEST_DURATION_METRIC_NAME.to_string(),
                IPC_RESPONSE_CHUNKS_METRIC_NAME.to_string(),
                IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME.to_string(),
                IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME.to_string(),
                IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME.to_string(),
                CONFIG_APPLIES_METRIC_NAME.to_string(),
                CONFIG_EPOCH_METRIC_NAME.to_string(),
                CONFIG_DEGRADED_METRIC_NAME.to_string(),
                NATS_OPERATIONS_METRIC_NAME.to_string(),
                NATS_DELIVERY_ATTEMPTS_METRIC_NAME.to_string(),
                RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME.to_string(),
                RESULT_CHUNKS_PUBLISHED_METRIC_NAME.to_string(),
                RESULT_CHUNK_SIZE_METRIC_NAME.to_string(),
                PAYLOAD_FETCHES_METRIC_NAME.to_string(),
                PAYLOAD_FETCH_DURATION_METRIC_NAME.to_string(),
                PAYLOAD_SIZE_METRIC_NAME.to_string(),
                GPU_SLOTS_METRIC_NAME.to_string(),
                PENDING_ITEMS_METRIC_NAME.to_string(),
                PENDING_COST_METRIC_NAME.to_string(),
                INFLIGHT_BATCHES_METRIC_NAME.to_string(),
                SATURATED_METRIC_NAME.to_string(),
                IPC_CAPACITY_METRIC_NAME.to_string(),
                IPC_INFLIGHT_METRIC_NAME.to_string(),
                IPC_ACQUIRE_DURATION_METRIC_NAME.to_string(),
                ADAPTIVE_WAIT_METRIC_NAME.to_string(),
                ADAPTIVE_COST_METRIC_NAME.to_string(),
                ADAPTIVE_P50_METRIC_NAME.to_string(),
                STARVATION_RESETS_METRIC_NAME.to_string(),
                GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME.to_string(),
                SHUTDOWN_DRAIN_DURATION_METRIC_NAME.to_string(),
            ])
        );
        assert!(names.iter().all(|name| !name.starts_with("sie_worker_")));

        macro_rules! assert_lane_on_points {
            ($points:expr, $name:expr) => {
                for point in $points {
                    assert!(
                        point.attributes().any(|attribute| {
                            attribute.key.as_str() == "lane"
                                && attribute.value.as_str().as_ref() == "default|l4|default"
                        }),
                        "{} point must carry the bounded release lane",
                        $name
                    );
                }
            };
        }
        for metric in &exported {
            match metric.data() {
                AggregatedMetrics::U64(MetricData::Gauge(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                AggregatedMetrics::U64(MetricData::Sum(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                AggregatedMetrics::U64(MetricData::Histogram(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                AggregatedMetrics::F64(MetricData::Gauge(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                AggregatedMetrics::F64(MetricData::Sum(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                AggregatedMetrics::F64(MetricData::Histogram(data)) => {
                    assert_lane_on_points!(data.data_points(), metric.name())
                }
                other => panic!(
                    "unexpected metric aggregation for {}: {other:?}",
                    metric.name()
                ),
            }
        }

        let queue_duration = exported
            .iter()
            .find(|metric| metric.name() == QUEUE_DURATION_METRIC_NAME)
            .expect("queue duration");
        assert_eq!(queue_duration.unit(), "s");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = queue_duration.data() else {
            panic!("queue duration must be an f64 histogram")
        };
        let point = histogram.data_points().next().expect("queue point");
        assert_eq!(point.count(), 1);
        assert_eq!(point.bounds().collect::<Vec<_>>(), QUEUE_DURATION_BUCKETS);
        let queue_attributes: HashMap<_, _> = point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            queue_attributes.keys().copied().collect::<HashSet<_>>(),
            HashSet::from(["operation", "lane", "model", "profile"]),
        );
        assert_eq!(
            queue_attributes.get("profile").map(String::as_str),
            Some("fast"),
            "trusted catalog profiles should survive while the lane machine profile stays separate"
        );

        for (name, expected_sum) in [
            (SCHEDULER_REQUEST_BATCH_DISPATCH_WAIT_METRIC_NAME, 0.007),
            (SCHEDULER_REQUEST_BATCH_TOTAL_METRIC_NAME, 0.019),
        ] {
            let metric = exported
                .iter()
                .find(|metric| metric.name() == name)
                .unwrap_or_else(|| panic!("missing scheduler request-batch metric {name}"));
            assert_eq!(metric.unit(), "s");
            let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = metric.data() else {
                panic!("{name} must be an f64 histogram")
            };
            let point = histogram
                .data_points()
                .next()
                .expect("scheduler batch point");
            assert_eq!(point.count(), 1);
            assert!((point.sum() - expected_sum).abs() < 1e-12);
            assert_eq!(point.bounds().collect::<Vec<_>>(), QUEUE_DURATION_BUCKETS);
            assert_eq!(
                point
                    .attributes()
                    .map(|attribute| attribute.key.as_str())
                    .collect::<HashSet<_>>(),
                HashSet::from(["operation", "lane", "model", "profile"]),
            );
        }

        let batch_size = exported
            .iter()
            .find(|metric| metric.name() == BATCH_SIZE_METRIC_NAME)
            .expect("batch size");
        assert_eq!(batch_size.unit(), "{item}");
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = batch_size.data() else {
            panic!("batch size must be a u64 histogram")
        };
        assert_eq!(
            histogram
                .data_points()
                .map(|point| point.count())
                .sum::<u64>(),
            2
        );
        for point in histogram.data_points() {
            let attributes: HashMap<_, _> = point
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            assert_eq!(attributes.len(), 4);
            assert!(attributes
                .keys()
                .all(|key| { matches!(*key, "operation" | "lane" | "model" | "profile") }));
            assert!(!attributes.contains_key("flush.reason"));
            assert!(!attributes.contains_key("lora"));
            assert!(!attributes.keys().any(|key| key.contains("id")));
            assert!(matches!(
                attributes.get("profile").map(String::as_str),
                Some("fast" | OTHER)
            ));
        }
        assert!(histogram.data_points().any(|point| {
            let attributes: HashMap<_, _> = point
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            attributes.get("operation").map(String::as_str) == Some(OTHER)
                && attributes.get("model").map(String::as_str) == Some(OTHER)
                && attributes.get("profile").map(String::as_str) == Some(OTHER)
        }));

        let batch_cost = exported
            .iter()
            .find(|metric| metric.name() == BATCH_COST_METRIC_NAME)
            .expect("batch cost");
        assert_eq!(batch_cost.unit(), "{cost}");
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = batch_cost.data() else {
            panic!("batch cost must be a u64 histogram")
        };
        assert_eq!(
            histogram
                .data_points()
                .map(|point| point.count())
                .sum::<u64>(),
            2,
            "one batch event must produce one cost observation"
        );

        let queue_depth = exported
            .iter()
            .find(|metric| metric.name() == QUEUE_DEPTH_METRIC_NAME)
            .expect("queue depth");
        assert_eq!(queue_depth.unit(), "{item}");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = queue_depth.data() else {
            panic!("queue depth must be a u64 gauge")
        };
        assert_eq!(
            gauge
                .data_points()
                .next()
                .expect("queue depth point")
                .value(),
            0,
            "queue release must emit an explicit zero"
        );

        let fill_ratio = exported
            .iter()
            .find(|metric| metric.name() == BATCH_FILL_RATIO_METRIC_NAME)
            .expect("fill ratio");
        assert_eq!(fill_ratio.unit(), "1");
        assert!(matches!(
            fill_ratio.data(),
            AggregatedMetrics::F64(MetricData::Gauge(_))
        ));
        let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = fill_ratio.data() else {
            panic!("fill ratio must be an f64 gauge")
        };
        assert_eq!(
            gauge
                .data_points()
                .next()
                .expect("fill ratio point")
                .value(),
            1.0,
            "fill ratio must be clamped to the documented unit interval"
        );
        assert!(gauge
            .data_points()
            .any(|point| point.attributes().any(|attribute| {
                attribute.key.as_str() == "flush.reason"
                    && attribute.value.as_str().as_ref() == "cost_cap"
            })));

        let ipc_requests = exported
            .iter()
            .find(|metric| metric.name() == IPC_REQUESTS_METRIC_NAME)
            .expect("IPC requests");
        assert_eq!(ipc_requests.unit(), "{request}");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = ipc_requests.data() else {
            panic!("IPC requests must be a u64 sum")
        };
        assert_eq!(
            sum.data_points().map(|point| point.value()).sum::<u64>(),
            2,
            "one IPC completion must increment the counter exactly once"
        );

        let ipc_duration = exported
            .iter()
            .find(|metric| metric.name() == IPC_REQUEST_DURATION_METRIC_NAME)
            .expect("IPC duration");
        assert_eq!(ipc_duration.unit(), "s");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = ipc_duration.data() else {
            panic!("IPC duration must be an f64 histogram")
        };
        assert_eq!(
            histogram
                .data_points()
                .map(|point| point.count())
                .sum::<u64>(),
            2
        );
        assert_eq!(
            histogram
                .data_points()
                .next()
                .expect("IPC duration point")
                .bounds()
                .collect::<Vec<_>>(),
            IPC_REQUEST_DURATION_BUCKETS
        );
        let ipc_attributes: Vec<HashMap<_, _>> = histogram
            .data_points()
            .map(|point| {
                point
                    .attributes()
                    .map(|attribute| {
                        (
                            attribute.key.as_str(),
                            attribute.value.as_str().into_owned(),
                        )
                    })
                    .collect()
            })
            .collect();
        assert!(ipc_attributes.iter().all(|attributes| {
            attributes.keys().copied().collect::<HashSet<_>>()
                == HashSet::from(["method", "outcome", "lane"])
        }));
        assert!(ipc_attributes.iter().any(|attributes| {
            attributes.get("method").map(String::as_str) == Some("run_batch")
                && attributes.get("outcome").map(String::as_str) == Some("success_after_retry")
        }));
        assert!(ipc_attributes.iter().any(|attributes| {
            attributes.get("method").map(String::as_str) == Some(OTHER)
                && attributes.get("outcome").map(String::as_str) == Some(OTHER)
        }));
        assert!(ipc_attributes.iter().all(|attributes| {
            !attributes
                .values()
                .any(|value| value == "caller-defined-method" || value == "customer error text")
        }));

        let ipc_response_chunks = exported
            .iter()
            .find(|metric| metric.name() == IPC_RESPONSE_CHUNKS_METRIC_NAME)
            .expect("IPC response chunks");
        assert_eq!(ipc_response_chunks.unit(), "{transfer}");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = ipc_response_chunks.data() else {
            panic!("IPC response chunks must be a u64 sum")
        };
        assert_eq!(sum.data_points().map(|point| point.value()).sum::<u64>(), 2);
        assert!(sum
            .data_points()
            .any(|point| point.attributes().any(|attribute| {
                attribute.key.as_str() == "outcome"
                    && attribute.value.as_str().as_ref() == "protocol_error"
            })));

        let reconstructed = exported
            .iter()
            .find(|metric| metric.name() == IPC_RESPONSE_RECONSTRUCTED_SIZE_METRIC_NAME)
            .expect("IPC reconstructed size");
        assert_eq!(reconstructed.unit(), "By");
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = reconstructed.data() else {
            panic!("IPC reconstructed size must be a u64 histogram")
        };
        let point = histogram.data_points().next().expect("reconstructed point");
        assert_eq!(point.count(), 1);
        assert_eq!(point.sum(), 16_384);
        assert_eq!(point.bounds().collect::<Vec<_>>(), PAYLOAD_SIZE_BUCKETS);

        let chunk_count = exported
            .iter()
            .find(|metric| metric.name() == IPC_RESPONSE_CHUNK_COUNT_METRIC_NAME)
            .expect("IPC response chunk count");
        assert_eq!(chunk_count.unit(), "{chunk}");
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = chunk_count.data() else {
            panic!("IPC response chunk count must be a u64 histogram")
        };
        let point = histogram.data_points().next().expect("chunk count point");
        assert_eq!(point.count(), 1);
        assert_eq!(point.sum(), 4);
        assert_eq!(
            point.bounds().collect::<Vec<_>>(),
            IPC_RESPONSE_CHUNK_COUNT_BUCKETS
        );

        let reserved = exported
            .iter()
            .find(|metric| metric.name() == IPC_RESPONSE_CHUNK_RESERVED_METRIC_NAME)
            .expect("IPC response reserved bytes");
        assert_eq!(reserved.unit(), "By");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = reserved.data() else {
            panic!("IPC response reserved bytes must be a u64 gauge")
        };
        assert_eq!(
            gauge.data_points().next().expect("reserved point").value(),
            32_768
        );

        let config_epoch = exported
            .iter()
            .find(|metric| metric.name() == CONFIG_EPOCH_METRIC_NAME)
            .expect("config epoch");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = config_epoch.data() else {
            panic!("config epoch must be a u64 gauge")
        };
        assert_eq!(gauge.data_points().next().expect("epoch point").value(), 17);
        let config_degraded = exported
            .iter()
            .find(|metric| metric.name() == CONFIG_DEGRADED_METRIC_NAME)
            .expect("config degraded");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = config_degraded.data() else {
            panic!("config degraded must be a u64 gauge")
        };
        assert_eq!(
            gauge.data_points().next().expect("degraded point").value(),
            0,
            "a successful convergence must clear a prior degraded state"
        );

        let nats_operations = exported
            .iter()
            .find(|metric| metric.name() == NATS_OPERATIONS_METRIC_NAME)
            .expect("NATS operations");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = nats_operations.data() else {
            panic!("NATS operations must be a u64 sum")
        };
        let nats_attributes: Vec<HashMap<_, _>> = sum
            .data_points()
            .map(|point| {
                point
                    .attributes()
                    .map(|attribute| {
                        (
                            attribute.key.as_str(),
                            attribute.value.as_str().into_owned(),
                        )
                    })
                    .collect()
            })
            .collect();
        assert!(nats_attributes.iter().any(|attributes| {
            attributes.get("operation").map(String::as_str) == Some("receive")
                && attributes.get("reason").map(String::as_str) == Some("redelivery")
        }));
        assert!(nats_attributes.iter().any(|attributes| {
            attributes.get("operation").map(String::as_str) == Some("ack")
                && attributes.get("outcome").map(String::as_str) == Some("error")
        }));

        let result_transport = exported
            .iter()
            .find(|metric| metric.name() == RESULT_TRANSPORT_ATTEMPTS_METRIC_NAME)
            .expect("result transport attempts");
        assert_eq!(result_transport.unit(), "{attempt}");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = result_transport.data() else {
            panic!("result transport attempts must be a u64 sum")
        };
        let point = sum.data_points().next().expect("result transport point");
        assert_eq!(point.value(), 1);
        let attributes: HashMap<_, _> = point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(attributes.get("mode").map(String::as_str), Some("chunked"));
        assert_eq!(
            attributes.get("outcome").map(String::as_str),
            Some("published")
        );

        let result_chunks = exported
            .iter()
            .find(|metric| metric.name() == RESULT_CHUNKS_PUBLISHED_METRIC_NAME)
            .expect("result chunks published");
        assert_eq!(result_chunks.unit(), "{chunk}");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = result_chunks.data() else {
            panic!("result chunks published must be a u64 sum")
        };
        assert_eq!(
            sum.data_points()
                .next()
                .expect("result chunk point")
                .value(),
            1
        );

        let result_chunk_size = exported
            .iter()
            .find(|metric| metric.name() == RESULT_CHUNK_SIZE_METRIC_NAME)
            .expect("result chunk size");
        assert_eq!(result_chunk_size.unit(), "By");
        let AggregatedMetrics::U64(MetricData::Histogram(histogram)) = result_chunk_size.data()
        else {
            panic!("result chunk size must be a u64 histogram")
        };
        let point = histogram
            .data_points()
            .next()
            .expect("result chunk size point");
        assert_eq!(point.count(), 1);
        assert_eq!(point.sum(), 4096);
        assert_eq!(point.bounds().collect::<Vec<_>>(), PAYLOAD_SIZE_BUCKETS);

        let payload_duration = exported
            .iter()
            .find(|metric| metric.name() == PAYLOAD_FETCH_DURATION_METRIC_NAME)
            .expect("payload fetch duration");
        assert_eq!(payload_duration.unit(), "s");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = payload_duration.data()
        else {
            panic!("payload duration must be an f64 histogram")
        };
        assert_eq!(
            histogram
                .data_points()
                .map(|point| point.count())
                .sum::<u64>(),
            2
        );

        let slots = exported
            .iter()
            .find(|metric| metric.name() == GPU_SLOTS_METRIC_NAME)
            .expect("GPU slots");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = slots.data() else {
            panic!("GPU slots must be a u64 gauge")
        };
        let slot_values: HashMap<_, _> = gauge
            .data_points()
            .map(|point| {
                let state = point
                    .attributes()
                    .find(|attribute| attribute.key.as_str() == "state")
                    .expect("slot state")
                    .value
                    .as_str()
                    .into_owned();
                (state, point.value())
            })
            .collect();
        assert_eq!(slot_values.get("total"), Some(&4));
        assert_eq!(slot_values.get("ready"), Some(&3));

        let ipc_capacity = exported
            .iter()
            .find(|metric| metric.name() == IPC_CAPACITY_METRIC_NAME)
            .expect("IPC capacity");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = ipc_capacity.data() else {
            panic!("IPC capacity must be a u64 gauge")
        };
        assert_eq!(
            gauge.data_points().next().expect("IPC capacity point").value(),
            7,
            "active child capacities must aggregate and a resize must replace only its prior contribution"
        );

        let ipc_inflight = exported
            .iter()
            .find(|metric| metric.name() == IPC_INFLIGHT_METRIC_NAME)
            .expect("IPC inflight");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = ipc_inflight.data() else {
            panic!("IPC inflight must be a u64 gauge")
        };
        assert_eq!(
            gauge
                .data_points()
                .next()
                .expect("IPC inflight point")
                .value(),
            0,
            "acquire/release must balance"
        );

        let adaptive_p50 = exported
            .iter()
            .find(|metric| metric.name() == ADAPTIVE_P50_METRIC_NAME)
            .expect("adaptive p50");
        assert_eq!(adaptive_p50.unit(), "s");
        let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = adaptive_p50.data() else {
            panic!("adaptive p50 must be an f64 gauge")
        };
        assert_eq!(gauge.data_points().count(), 2);

        let starvation_resets = exported
            .iter()
            .find(|metric| metric.name() == STARVATION_RESETS_METRIC_NAME)
            .expect("starvation resets");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = starvation_resets.data() else {
            panic!("starvation resets must be a u64 sum")
        };
        assert_eq!(sum.data_points().map(|point| point.value()).sum::<u64>(), 2);

        let generation_loading = exported
            .iter()
            .find(|metric| metric.name() == GENERATION_MODEL_LOADING_RESPONSES_METRIC_NAME)
            .expect("generation model-loading responses");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = generation_loading.data() else {
            panic!("generation model-loading responses must be a u64 sum")
        };
        assert_eq!(sum.data_points().map(|point| point.value()).sum::<u64>(), 2);
        assert!(sum.data_points().any(|point| {
            let attributes: HashMap<_, _> = point
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            attributes.get("state").map(String::as_str) == Some("loading_in_progress")
                && attributes.get("outcome").map(String::as_str) == Some("ack_error")
                && attributes.get("model").map(String::as_str) == Some("model-a")
                && attributes.get("profile").map(String::as_str) == Some("fast")
        }));
        assert!(sum.data_points().any(|point| {
            let attributes: HashMap<_, _> = point
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            attributes.get("state").map(String::as_str) == Some(OTHER)
                && attributes.get("outcome").map(String::as_str) == Some(OTHER)
                && attributes.get("model").map(String::as_str) == Some(OTHER)
                && attributes.get("profile").map(String::as_str) == Some(OTHER)
        }));

        let shutdown_drain = exported
            .iter()
            .find(|metric| metric.name() == SHUTDOWN_DRAIN_DURATION_METRIC_NAME)
            .expect("shutdown drain duration");
        assert_eq!(shutdown_drain.unit(), "s");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = shutdown_drain.data() else {
            panic!("shutdown drain duration must be an f64 histogram")
        };
        let point = histogram
            .data_points()
            .next()
            .expect("shutdown drain point");
        assert_eq!(point.count(), 1);
        assert_eq!(
            point.bounds().collect::<Vec<_>>(),
            SHUTDOWN_DRAIN_DURATION_BUCKETS
        );
        assert!(point.attributes().any(|attribute| {
            attribute.key.as_str() == "outcome"
                && attribute.value.as_str().as_ref() == "deadline_exceeded"
        }));
    }

    /// Opt-in release benchmark for the per-result-chunk facade event at the
    /// protocol maximum of 64 chunks per transfer. It reports three warmed
    /// samples and normalizes each result to one chunk event:
    ///
    /// `SIE_RUN_RESULT_CHUNK_TELEMETRY_BENCHMARK=1 cargo test --manifest-path packages/sie_server_sidecar/Cargo.toml --release --lib result_chunk_telemetry_benchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "opt-in release result-chunk telemetry benchmark"]
    fn result_chunk_telemetry_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        const CHUNKS_PER_TRANSFER: usize = 64;
        const TRANSFERS_PER_SAMPLE: usize = 10_000;
        const WARMUP_TRANSFERS: usize = 500;
        const SAMPLES: usize = 3;

        assert_eq!(
            std::env::var("SIE_RUN_RESULT_CHUNK_TELEMETRY_BENCHMARK").as_deref(),
            Ok("1"),
            "opt in with SIE_RUN_RESULT_CHUNK_TELEMETRY_BENCHMARK=1"
        );

        let exporter = InMemoryMetricExporterBuilder::new()
            .with_temporality(Temporality::LowMemory)
            .build();
        let reader = PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("sie-worker-sidecar-result-chunk-benchmark");
        let context = TelemetryContext::new("benchmark|l4|default", HashSet::new());
        let disabled = SidecarTelemetry::new(&meter, context.clone(), false);
        let enabled = SidecarTelemetry::new(&meter, context, true);

        let exercise = |telemetry: &SidecarTelemetry, transfers: usize| {
            for _ in 0..transfers {
                for _ in 0..CHUNKS_PER_TRANSFER {
                    black_box(telemetry).result_chunk_published(black_box(1_048_576));
                }
            }
        };
        let sample = |telemetry: &SidecarTelemetry| {
            let started = Instant::now();
            exercise(telemetry, TRANSFERS_PER_SAMPLE);
            started.elapsed().as_nanos() as f64
                / (TRANSFERS_PER_SAMPLE * CHUNKS_PER_TRANSFER) as f64
        };
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for index in 0..SAMPLES {
            exercise(&disabled, WARMUP_TRANSFERS);
            disabled_samples[index] = sample(&disabled);
            exercise(&enabled, WARMUP_TRANSFERS);
            enabled_samples[index] = sample(&enabled);
        }

        let median = |mut values: [f64; SAMPLES]| {
            values.sort_by(f64::total_cmp);
            values[SAMPLES / 2]
        };
        let disabled_median = median(disabled_samples);
        let enabled_median = median(enabled_samples);
        let incremental_median = (enabled_median - disabled_median).max(0.0);
        println!(
            "SIDECAR_RESULT_CHUNK_TELEMETRY_BENCHMARK chunks_per_transfer={CHUNKS_PER_TRANSFER} transfers_per_sample={TRANSFERS_PER_SAMPLE} disabled_ns_per_event={disabled_samples:?} disabled_median_ns_per_event={:.2} enabled_ns_per_event={enabled_samples:?} enabled_median_ns_per_event={:.2} incremental_median_ns_per_event={:.2}",
            disabled_median,
            enabled_median,
            incremental_median,
        );

        let budgets: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../telemetry/performance-budgets.json"
        ))
        .expect("valid checked-in telemetry performance budgets");
        let budget = |name: &str| {
            budgets["budgets"][name].as_f64().unwrap_or_else(|| {
                panic!("sidecar result-chunk telemetry performance budget {name}")
            })
        };
        let disabled_budget = budget("sidecar_result_chunk_disabled_ns_per_event");
        assert!(
            disabled_median <= disabled_budget,
            "sidecar result-chunk telemetry-disabled median {disabled_median:.2} ns/event exceeds {disabled_budget:.2} ns/event budget"
        );
        let enabled_budget = budget("sidecar_result_chunk_enabled_ns_per_event");
        assert!(
            enabled_median <= enabled_budget,
            "sidecar result-chunk telemetry-enabled median {enabled_median:.2} ns/event exceeds {enabled_budget:.2} ns/event budget"
        );
        let incremental_budget = budget("sidecar_result_chunk_incremental_ns_per_event");
        assert!(
            incremental_median <= incremental_budget,
            "sidecar result-chunk telemetry incremental median {incremental_median:.2} ns/event exceeds {incremental_budget:.2} ns/event budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }

    /// Opt-in release benchmark for the scheduler queue lifecycle plus one
    /// successful request-batch completion. It reports three independently
    /// warmed samples and their median for both disabled and enabled telemetry:
    ///
    /// `SIE_RUN_TELEMETRY_BENCHMARK=1 cargo test --manifest-path packages/sie_server_sidecar/Cargo.toml --release --lib telemetry_hot_path_benchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "opt-in release telemetry benchmark"]
    fn telemetry_hot_path_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;

        const ITERATIONS: usize = 50_000;
        const WARMUP_ITERATIONS: usize = 2_000;
        const SAMPLES: usize = 3;

        assert_eq!(
            std::env::var("SIE_RUN_TELEMETRY_BENCHMARK").as_deref(),
            Ok("1"),
            "opt in with SIE_RUN_TELEMETRY_BENCHMARK=1"
        );

        let exporter = InMemoryMetricExporterBuilder::new()
            .with_temporality(Temporality::LowMemory)
            .build();
        let reader = PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("sie-worker-sidecar-benchmark");
        let context = TelemetryContext::new(
            "benchmark|l4|default",
            HashSet::from(["catalog/model".to_string()]),
        );
        let disabled = SidecarTelemetry::new(&meter, context.clone(), false);
        let enabled = SidecarTelemetry::new(&meter, context, true);

        let exercise = |telemetry: &SidecarTelemetry, iterations: usize| {
            for _ in 0..iterations {
                telemetry.queue_enqueued(black_box("encode"), black_box("catalog/model"), None);
                telemetry.queue_released(
                    black_box("encode"),
                    black_box("catalog/model"),
                    None,
                    black_box(Duration::from_millis(2)),
                );
                telemetry.scheduler_request_batch_completed(black_box(
                    SchedulerRequestBatchObservation {
                        operation: "encode",
                        model: "catalog/model",
                        profile: None,
                        dispatch_wait: Duration::from_millis(2),
                        total: Duration::from_millis(5),
                    },
                ));
            }
        };
        let sample = |telemetry: &SidecarTelemetry| {
            let started = Instant::now();
            exercise(telemetry, ITERATIONS);
            started.elapsed().as_nanos() as f64 / ITERATIONS as f64
        };
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for index in 0..SAMPLES {
            exercise(&disabled, WARMUP_ITERATIONS);
            disabled_samples[index] = sample(&disabled);
            exercise(&enabled, WARMUP_ITERATIONS);
            enabled_samples[index] = sample(&enabled);
        }

        let median = |mut values: [f64; SAMPLES]| {
            values.sort_by(f64::total_cmp);
            values[SAMPLES / 2]
        };
        let disabled_median = median(disabled_samples);
        let enabled_median = median(enabled_samples);
        let incremental_median = (enabled_median - disabled_median).max(0.0);
        println!(
            "SIDECAR_TELEMETRY_BENCHMARK iterations_per_sample={ITERATIONS} disabled_ns_per_item={disabled_samples:?} disabled_median_ns_per_item={:.2} enabled_ns_per_item={enabled_samples:?} enabled_median_ns_per_item={:.2} incremental_median_ns_per_item={:.2}",
            disabled_median,
            enabled_median,
            incremental_median,
        );
        let budgets: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../telemetry/performance-budgets.json"
        ))
        .expect("valid checked-in telemetry performance budgets");
        let budget = |name: &str| {
            budgets["budgets"][name]
                .as_f64()
                .unwrap_or_else(|| panic!("sidecar telemetry performance budget {name}"))
        };
        let disabled_budget = budget("sidecar_disabled_ns_per_item");
        assert!(
            disabled_median <= disabled_budget,
            "sidecar telemetry-disabled median {disabled_median:.2} ns/item exceeds {disabled_budget:.2} ns/item budget"
        );
        let enabled_budget = budget("sidecar_enabled_ns_per_item");
        assert!(
            enabled_median <= enabled_budget,
            "sidecar enabled telemetry median {enabled_median:.2} ns/item exceeds {enabled_budget:.2} ns/item budget"
        );
        let incremental_budget = budget("sidecar_incremental_ns_per_item");
        assert!(
            incremental_median <= incremental_budget,
            "sidecar incremental telemetry median {incremental_median:.2} ns/item exceeds {incremental_budget:.2} ns/item budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }
}
