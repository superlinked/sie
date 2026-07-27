//! Canonical gateway telemetry facade.
//!
//! Business and middleware call sites emit one semantic observation here.
//! This module is the only application code that turns those observations
//! into OpenTelemetry instruments. Applications export OTLP only; Prometheus
//! naming and exposition are collector concerns.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, Histogram, ObservableGauge};
use opentelemetry::trace::SpanContext;
use opentelemetry::{global, Context, KeyValue};

use crate::state::demand_tracker::{DemandTracker, PhysicalLane};

pub const REQUESTS_METRIC_NAME: &str = "sie.gateway.requests";
pub const REQUEST_DURATION_METRIC_NAME: &str = "sie.gateway.request.duration";
pub const ADMISSION_DECISIONS_METRIC_NAME: &str = "sie.gateway.admission.decisions";
pub const DISPATCHES_METRIC_NAME: &str = "sie.gateway.dispatches";
pub const DISPATCH_DURATION_METRIC_NAME: &str = "sie.gateway.dispatch.duration";
pub const PENDING_DEMAND_METRIC_NAME: &str = "sie.gateway.pending_demand";
pub const LANE_QUEUE_DEPTH_METRIC_NAME: &str = "sie.gateway.lane.queue.depth";
pub const LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME: &str =
    "sie.gateway.lane.queue.snapshot.timestamp";
pub const ACTIVE_LEASE_GPUS_METRIC_NAME: &str = "sie.gateway.active_lease.gpus";
pub const POOL_WARM_FLOOR_METRIC_NAME: &str = "sie.gateway.pool.warm_floor";
pub const POOL_PINNED_MODEL_LOADED_METRIC_NAME: &str = "sie.gateway.pool.pinned_model.loaded";
pub const REJECTED_REQUESTS_METRIC_NAME: &str = "sie.gateway.rejected.requests";
pub const CAPACITY_SNAPSHOT_TIMESTAMP_METRIC_NAME: &str = "sie.gateway.capacity.snapshot.timestamp";
pub const CONFIG_APPLIED_EPOCH_METRIC_NAME: &str = "sie.gateway.config.applied_epoch";
pub const CONFIG_OPERATIONS_METRIC_NAME: &str = "sie.gateway.config.operations";
pub const CONFIG_BOOTSTRAP_DEGRADED_METRIC_NAME: &str = "sie.gateway.config.bootstrap.degraded";
pub const MESSAGING_CLIENT_READY_METRIC_NAME: &str = "sie.gateway.messaging.client.ready";
pub const QUEUE_PUBLISHES_METRIC_NAME: &str = "sie.gateway.queue.publishes";
pub const QUEUE_PUBLISH_DURATION_METRIC_NAME: &str = "sie.gateway.queue.publish.duration";
pub const QUEUE_PUBLISH_ITEMS_METRIC_NAME: &str = "sie.gateway.queue.publish.items";
pub const QUEUE_RESULT_WAITS_METRIC_NAME: &str = "sie.gateway.queue.result_waits";
pub const QUEUE_RESULT_WAIT_DURATION_METRIC_NAME: &str = "sie.gateway.queue.result_wait.duration";
pub const QUEUE_RESULT_CHUNKS_RECEIVED_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunks.received";
pub const QUEUE_RESULT_CHUNK_BYTES_RECEIVED_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.bytes_received";
pub const QUEUE_RESULT_CHUNK_REJECTIONS_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.rejections";
pub const QUEUE_RESULT_CHUNK_TRANSFERS_COMPLETED_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.transfers_completed";
pub const QUEUE_RESULT_CHUNK_DUPLICATES_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.duplicates";
pub const QUEUE_RESULT_CHUNK_RETRY_REPLACEMENTS_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.retry_replacements";
pub const QUEUE_RESULT_CHUNK_STALE_RETRIES_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.stale_retries";
pub const QUEUE_RESULT_CHUNK_RESERVED_BYTES_METRIC_NAME: &str =
    "sie.gateway.queue.result_chunk.reserved_bytes";
pub const QUEUE_EVENTS_METRIC_NAME: &str = "sie.gateway.queue.events";
pub const PROVISIONING_RESPONSES_METRIC_NAME: &str = "sie.gateway.provisioning.responses";
pub const GENERATION_EVENTS_METRIC_NAME: &str = "sie.gateway.generation.events";
pub const GENERATION_TTFT_METRIC_NAME: &str = "sie.gateway.generation.ttft";
pub const GENERATION_TPOT_METRIC_NAME: &str = "sie.gateway.generation.tpot";
pub const GENERATION_TOKENS_METRIC_NAME: &str = "sie.gateway.generation.tokens";

pub const REQUEST_LATENCY_BUCKETS: [f64; 21] = [
    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75,
    1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
];

pub const QUEUE_PUBLISH_DURATION_BUCKETS: [f64; 9] =
    [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0];
pub const QUEUE_PUBLISH_ITEMS_BUCKETS: [f64; 16] = [
    1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 49.0, 53.0, 57.0, 61.0,
];
pub const QUEUE_RESULT_WAIT_DURATION_BUCKETS: [f64; 13] = [
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0,
];
pub const GENERATION_DURATION_BUCKETS: [f64; 15] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Runtime catalog allowlist for the bounded managed `lane` dimension.
#[allow(dead_code)] // Managed composition API; the standalone binary duplicates this module tree.
const LANE_APP_MAP_ENV: &str = "SIE_LANE_APP_MAP";

type LaneKey = (String, String, String);
type PinnedModelKey = (String, String);

/// Only rejection reasons that are actionable by adding capacity enter the
/// KEDA control counter. Combined with the physical-lane catalog cap, this is
/// the exact maximum number of non-overflow series for the instrument.
pub const KEDA_SCALE_UP_REJECTION_REASON_CARDINALITY: usize = 4;

#[derive(Default)]
struct PinnedModelObservableState {
    current: HashMap<PinnedModelKey, bool>,
    removed: HashSet<PinnedModelKey>,
}

#[derive(Default)]
struct ResultChunkReservationObservableState {
    initialized: AtomicBool,
    reserved_bytes: AtomicU64,
}

/// Canonical routing labels resolved by the handler and consumed by the outer
/// request-completion middleware.
#[derive(Clone, Debug, Default)]
pub struct MetricLabels {
    pub machine_profile: String,
}

/// First-write-wins request-extension carrier for [`MetricLabels`].
#[derive(Clone, Default)]
pub struct MetricLabelsSlot(Arc<OnceLock<MetricLabels>>);

impl MetricLabelsSlot {
    pub fn set(&self, labels: MetricLabels) {
        let _ = self.0.set(labels);
    }

    pub fn get(&self) -> Option<&MetricLabels> {
        self.0.get()
    }
}

/// Bounded admission result carried from policy middleware to the one outer
/// request-completion event.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdmissionOutcome {
    Admitted,
    Unauthenticated,
    Forbidden,
    AuthMisconfigured,
    RegionMismatch,
    LicenseExcluded,
    PayloadTooLarge,
    InsufficientCredits,
    KeySpendLimitExceeded,
    RateLimited,
}

impl AdmissionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Unauthenticated => "unauthenticated",
            Self::Forbidden => "forbidden",
            Self::AuthMisconfigured => "auth_misconfigured",
            Self::RegionMismatch => "region_mismatch",
            Self::LicenseExcluded => "license_excluded",
            Self::PayloadTooLarge => "payload_too_large",
            Self::InsufficientCredits => "insufficient_credits",
            Self::KeySpendLimitExceeded => "key_spend_limit_exceeded",
            Self::RateLimited => "rate_limited",
        }
    }
}

/// First-write-wins request-extension carrier for [`AdmissionOutcome`].
#[derive(Clone, Default)]
pub struct AdmissionOutcomeSlot(Arc<OnceLock<AdmissionOutcome>>);

impl AdmissionOutcomeSlot {
    #[allow(dead_code)]
    pub fn set(&self, outcome: AdmissionOutcome) {
        let _ = self.0.set(outcome);
    }

    pub fn get(&self) -> Option<AdmissionOutcome> {
        self.0.get().copied()
    }
}

/// Request-extension carrier for the enclosing `gateway.request` OTel context.
#[derive(Clone)]
pub struct RequestTraceContext(Context);

impl RequestTraceContext {
    pub fn new(context: Context) -> Self {
        Self(context)
    }

    pub fn get(&self) -> &Context {
        &self.0
    }
}

/// Final transport selected for one managed gateway dispatch.
#[allow(dead_code)] // Managed composition API; unused by the standalone gateway binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchPath {
    I6pn,
    Modal,
}

#[allow(dead_code)]
impl DispatchPath {
    fn as_str(self) -> &'static str {
        match self {
            Self::I6pn => "i6pn",
            Self::Modal => "modal",
        }
    }
}

/// Final bounded outcome of one managed gateway dispatch.
#[allow(dead_code)] // Managed composition API; unused by the standalone gateway binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DispatchOutcome {
    Success,
    Error,
    Timeout,
    Cancelled,
}

#[allow(dead_code)]
impl DispatchOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Why an i6pn attempt was not the final path.
///
/// Raw transport errors deliberately have no representation here. Every
/// variant maps to a finite contract value before it reaches an OTel label.
#[allow(dead_code)] // Managed composition API; unused by the standalone gateway binary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackReason {
    None,
    Disabled,
    NotApplicable,
    UnknownLane,
    NoLiveChannel,
    NoCredit,
    LaneError,
    Timeout,
    TransportFailure,
}

#[allow(dead_code)]
impl FallbackReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Disabled => "disabled",
            Self::NotApplicable => "not_applicable",
            Self::UnknownLane => "unknown_lane",
            Self::NoLiveChannel => "no_live_channel",
            Self::NoCredit => "no_credit",
            Self::LaneError => "lane_error",
            Self::Timeout => "timeout",
            Self::TransportFailure => "transport_failure",
        }
    }
}

/// One final-path dispatch event. Only bounded semantic routing values are
/// accepted; request, tenant, key, job, worker, and container identifiers have
/// no field and therefore cannot accidentally become dimensions.
#[allow(dead_code)] // Managed composition API; unused by the standalone gateway binary.
#[derive(Clone, Copy, Debug)]
pub struct DispatchObservation<'a> {
    pub operation: &'a str,
    pub path: DispatchPath,
    pub outcome: DispatchOutcome,
    pub fallback_reason: FallbackReason,
    pub lane: &'a str,
    pub duration: Duration,
}

/// Bounded control-plane operation. A delta's shape is retained without
/// attaching model, bundle, producer, URL, or raw error dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigOperation {
    Bootstrap,
    Reconcile,
    DeltaModel,
    DeltaEpoch,
}

impl ConfigOperation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Reconcile => "reconcile",
            Self::DeltaModel => "delta_model",
            Self::DeltaEpoch => "delta_epoch",
        }
    }
}

/// Finite outcome vocabulary shared by bootstrap and live delta application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigOutcome {
    Success,
    ClientError,
    FetchError,
    PartialApply,
    ParseError,
    ApplyError,
    RejectedUntrusted,
}

impl ConfigOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::ClientError => "client_error",
            Self::FetchError => "fetch_error",
            Self::PartialApply => "partial_apply",
            Self::ParseError => "parse_error",
            Self::ApplyError => "apply_error",
            Self::RejectedUntrusted => "rejected_untrusted",
        }
    }
}

/// The gateway intentionally uses one `async-nats` client for NATS Core and
/// JetStream. Expose that fact as one finite transport value rather than two
/// gauges that could falsely disagree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MessagingTransport {
    NatsJetstream,
}

impl MessagingTransport {
    fn as_str(self) -> &'static str {
        match self {
            Self::NatsJetstream => "nats_jetstream",
        }
    }
}

/// Submission outcome of one logical queue publish call. JetStream durability
/// completes asynchronously and is reported separately as `PublishAck`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuePublishOutcome {
    Submitted,
    Backpressure,
    NoConsumers,
    Error,
}

impl QueuePublishOutcome {
    pub fn from_error(error: &str) -> Self {
        let normalized = error.to_ascii_lowercase();
        if normalized.contains("backpressure") {
            Self::Backpressure
        } else if normalized.contains("no consumers") {
            Self::NoConsumers
        } else {
            Self::Error
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Backpressure => "backpressure",
            Self::NoConsumers => "no_consumers",
            Self::Error => "error",
        }
    }
}

/// One logical publish-submission observation. A single facade call owns the
/// submission counter, duration, and batch-size instruments.
#[derive(Clone, Copy, Debug)]
pub struct QueuePublishObservation<'a> {
    pub operation: &'a str,
    pub outcome: QueuePublishOutcome,
    pub duration: Duration,
    pub items: u32,
}

/// Final outcome of waiting for a queue result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueResultOutcome {
    Success,
    Timeout,
    ChannelClosed,
    DurabilityError,
    WorkerError,
    Cancelled,
}

impl QueueResultOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Timeout => "timeout",
            Self::ChannelClosed => "channel_closed",
            Self::DurabilityError => "durability_error",
            Self::WorkerError => "worker_error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Bounded validation failure for one chunked queue-result transfer.
///
/// The gateway's result-chunk decoder is the sole source of these values. A
/// typed enum keeps malformed wire data out of metric attributes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueResultChunkRejectionReason {
    Kind,
    Identity,
    Digest,
    ChunkCount,
    ChunkIndex,
    ItemSize,
    PayloadSize,
    AggregateSize,
    GlobalBudget,
    MetadataConflict,
    DuplicateConflict,
    TotalMismatch,
    DigestMismatch,
    Decode,
}

impl QueueResultChunkRejectionReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Kind => "kind",
            Self::Identity => "identity",
            Self::Digest => "digest",
            Self::ChunkCount => "chunk_count",
            Self::ChunkIndex => "chunk_index",
            Self::ItemSize => "item_size",
            Self::PayloadSize => "payload_size",
            Self::AggregateSize => "aggregate_size",
            Self::GlobalBudget => "global_budget",
            Self::MetadataConflict => "metadata_conflict",
            Self::DuplicateConflict => "duplicate_conflict",
            Self::TotalMismatch => "total_mismatch",
            Self::DigestMismatch => "digest_mismatch",
            Self::Decode => "decode",
        }
    }
}

/// Bounded non-rejection outcome of processing a chunked queue result.
///
/// Separate contract instruments retain the existing operational counters,
/// while one enum prevents call sites from selecting an instrumentation
/// backend or minting arbitrary event dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueResultChunkEvent {
    TransferCompleted,
    Duplicate,
    RetryReplacement,
    StaleRetry,
}

/// One successful change to the process-wide result-chunk memory budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueResultChunkReservationChange {
    Reserved(usize),
    Released(usize),
}

/// Durability stage grouped onto one bounded counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueEvent {
    PublishAck,
    DlqForward,
    PayloadOffload,
}

impl QueueEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::PublishAck => "publish_ack",
            Self::DlqForward => "dlq_forward",
            Self::PayloadOffload => "payload_offload",
        }
    }
}

/// Finite durability-event outcome. `Deduplicated` is specific to the DLQ
/// forwarding path but remains valid on the shared family by contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueEventOutcome {
    Success,
    AckError,
    Error,
    Deduplicated,
}

/// HTTP surface on which a semantic provisioning response was returned.
/// This intentionally has no machine-profile, bundle, model, tenant, or
/// request-id field, so those values cannot become metric dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProvisioningSurface {
    Native,
    OpenAi,
}

impl ProvisioningSurface {
    fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::OpenAi => "openai",
        }
    }
}

/// Generation transport stage represented by the consolidated integrity
/// counter. Values are deliberately finite and do not carry request data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationEvent {
    Cancellation,
    Chunk,
    Nak,
}

impl GenerationEvent {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cancellation => "cancellation",
            Self::Chunk => "chunk",
            Self::Nak => "nak",
        }
    }
}

/// Why a generation transport event occurred. This replaces several legacy
/// one-off metric families while preserving their operational distinction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationEventReason {
    BeforeFirstChunk,
    MidStream,
    InvalidKind,
    NonFiniteTtft,
    InvalidFinishReason,
    SequenceGap,
    StaleAttempt,
    Duplicate,
}

impl GenerationEventReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::BeforeFirstChunk => "before_first_chunk",
            Self::MidStream => "mid_stream",
            Self::InvalidKind => "invalid_kind",
            Self::NonFiniteTtft => "non_finite_ttft",
            Self::InvalidFinishReason => "invalid_finish_reason",
            Self::SequenceGap => "sequence_gap",
            Self::StaleAttempt => "stale_attempt",
            Self::Duplicate => "duplicate",
        }
    }
}

/// Terminal classification for one generation transport event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationEventOutcome {
    Cancelled,
    Dropped,
    Rejected,
}

impl GenerationEventOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
            Self::Rejected => "rejected",
        }
    }
}

/// One successful, authoritative terminal generation observation. Timing is
/// accepted in milliseconds to match the worker wire envelope; conversion and
/// finite-value validation remain inside the facade.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerationCompletionObservation {
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

impl QueueEventOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::AckError => "ack_error",
            Self::Error => "error",
            Self::Deduplicated => "deduplicated",
        }
    }
}

/// One current value for a physical KEDA lane.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LaneSnapshot {
    pub pool: String,
    pub machine_profile: String,
    pub bundle: String,
    pub value: f64,
}

/// Gateway-owned KEDA state captured by one business reconciliation.
///
/// The three registry families are complete snapshots. `lane_queue_depth` is
/// intentionally a set of successful per-lane broker updates: omitted lanes
/// retain their last value but do not receive a new lane freshness timestamp.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct KedaCapacitySnapshot {
    pub pending_demand: Vec<LaneSnapshot>,
    pub lane_queue_depth: Vec<LaneSnapshot>,
    pub active_lease_gpus: Vec<LaneSnapshot>,
    pub pool_warm_floor: Vec<LaneSnapshot>,
}

/// One current pinned-model readiness value for a logical pool.
///
/// Both dimensions come from gateway-owned configuration and registry state;
/// caller-supplied request data has no path into this observation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedModelSnapshot {
    pub pool: String,
    pub model: String,
    pub loaded: bool,
}

/// Bound an internal/catalog label before aggregation. This remains a neutral
/// safety helper for KEDA and request context; it is not backend-specific.
pub fn sanitize_label(value: &str) -> String {
    const MAX_LABEL_LEN: usize = 128;
    if value.len() > MAX_LABEL_LEN
        || value.is_empty()
        || !value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'))
    {
        return "other".to_string();
    }
    value.to_string()
}

struct GatewayTelemetry {
    requests: Counter<u64>,
    request_duration: Histogram<f64>,
    admission_decisions: Counter<u64>,
    #[allow(dead_code)] // Managed composition API.
    dispatches: Counter<u64>,
    #[allow(dead_code)] // Managed composition API.
    dispatch_duration: Histogram<f64>,
    pending_demand: Gauge<f64>,
    lane_queue_depth: Gauge<f64>,
    lane_queue_snapshot_timestamp: Gauge<f64>,
    active_lease_gpus: Gauge<f64>,
    pool_warm_floor: Gauge<f64>,
    #[allow(dead_code)] // Retains the observable callback registration.
    pool_pinned_model_loaded: ObservableGauge<u64>,
    rejected_requests: Counter<u64>,
    capacity_snapshot_timestamp: Gauge<f64>,
    config_applied_epoch: Gauge<u64>,
    config_operations: Counter<u64>,
    config_bootstrap_degraded: Gauge<u64>,
    messaging_client_ready: Gauge<u64>,
    queue_publishes: Counter<u64>,
    queue_publish_duration: Histogram<f64>,
    queue_publish_items: Histogram<f64>,
    queue_result_waits: Counter<u64>,
    queue_result_wait_duration: Histogram<f64>,
    queue_result_chunks_received: Counter<u64>,
    queue_result_chunk_bytes_received: Counter<u64>,
    queue_result_chunk_rejections: Counter<u64>,
    queue_result_chunk_transfers_completed: Counter<u64>,
    queue_result_chunk_duplicates: Counter<u64>,
    queue_result_chunk_retry_replacements: Counter<u64>,
    queue_result_chunk_stale_retries: Counter<u64>,
    #[allow(dead_code)] // Retains the observable callback registration.
    queue_result_chunk_reserved_bytes: ObservableGauge<u64>,
    queue_result_chunk_reserved_bytes_state: Arc<ResultChunkReservationObservableState>,
    queue_events: Counter<u64>,
    provisioning_responses: Counter<u64>,
    generation_events: Counter<u64>,
    generation_ttft: Histogram<f64>,
    generation_tpot: Histogram<f64>,
    generation_tokens: Counter<u64>,
    capacity_snapshot_lock: Mutex<()>,
    previous_pending_lanes: Mutex<HashSet<LaneKey>>,
    previous_lease_lanes: Mutex<HashSet<LaneKey>>,
    previous_warm_floor_lanes: Mutex<HashSet<LaneKey>>,
    pinned_model_state: Arc<Mutex<PinnedModelObservableState>>,
}

impl GatewayTelemetry {
    fn new(meter: &opentelemetry::metrics::Meter) -> Self {
        let pinned_model_state = Arc::new(Mutex::new(PinnedModelObservableState::default()));
        let observable_pinned_model_state = Arc::clone(&pinned_model_state);
        let queue_result_chunk_reserved_bytes_state =
            Arc::new(ResultChunkReservationObservableState::default());
        let observable_queue_result_chunk_reserved_bytes_state =
            Arc::clone(&queue_result_chunk_reserved_bytes_state);
        Self {
            requests: meter
                .u64_counter(REQUESTS_METRIC_NAME)
                .with_description(
                    "Count of gateway inference HTTP responses by bounded outcome and profile.",
                )
                .with_unit("{request}")
                .build(),
            request_duration: meter
                .f64_histogram(REQUEST_DURATION_METRIC_NAME)
                .with_description("Gateway-observed inference response duration.")
                .with_unit("s")
                .with_boundaries(REQUEST_LATENCY_BUCKETS.to_vec())
                .build(),
            admission_decisions: meter
                .u64_counter(ADMISSION_DECISIONS_METRIC_NAME)
                .with_description("Count of bounded managed gateway admission decisions.")
                .with_unit("{request}")
                .build(),
            dispatches: meter
                .u64_counter(DISPATCHES_METRIC_NAME)
                .with_description(
                    "Count of final gateway dispatch outcomes by operation, path, fallback, and lane.",
                )
                .with_unit("{request}")
                .build(),
            dispatch_duration: meter
                .f64_histogram(DISPATCH_DURATION_METRIC_NAME)
                .with_description(
                    "Gateway dispatch duration through the final i6pn or Modal path.",
                )
                .with_unit("s")
                .with_boundaries(REQUEST_LATENCY_BUCKETS.to_vec())
                .build(),
            pending_demand: meter
                .f64_gauge(PENDING_DEMAND_METRIC_NAME)
                .with_description("Whether a physical worker lane has refreshable unmet demand.")
                .with_unit("{request}")
                .build(),
            lane_queue_depth: meter
                .f64_gauge(LANE_QUEUE_DEPTH_METRIC_NAME)
                .with_description(
                    "Exact JetStream durable-consumer pending plus unacknowledged work in one physical lane.",
                )
                .with_unit("{item}")
                .build(),
            active_lease_gpus: meter
                .f64_gauge(ACTIVE_LEASE_GPUS_METRIC_NAME)
                .with_description("Distinct active leased workers in one physical lane.")
                .with_unit("{gpu}")
                .build(),
            pool_warm_floor: meter
                .f64_gauge(POOL_WARM_FLOOR_METRIC_NAME)
                .with_description("Configured minimum warm workers for one physical lane.")
                .with_unit("{worker}")
                .build(),
            pool_pinned_model_loaded: meter
                .u64_observable_gauge(POOL_PINNED_MODEL_LOADED_METRIC_NAME)
                .with_description(
                    "Whether a pinned model is loaded on a healthy worker assigned to a logical pool.",
                )
                .with_unit("1")
                .with_callback(move |observer| {
                    let mut state = observable_pinned_model_state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    for ((pool, model), loaded) in &state.current {
                        observer.observe(
                            u64::from(*loaded),
                            &pinned_model_attributes(pool, model),
                        );
                    }
                    for (pool, model) in &state.removed {
                        observer.observe(0, &pinned_model_attributes(pool, model));
                    }
                    state.removed.clear();
                })
                .build(),
            rejected_requests: meter
                .u64_counter(REJECTED_REQUESTS_METRIC_NAME)
                .with_description("Gateway rejections classified by autoscaling action.")
                .with_unit("{request}")
                .build(),
            lane_queue_snapshot_timestamp: meter
                .f64_gauge(LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME)
                .with_description(
                    "Unix start timestamp of the latest successful JetStream backlog read for one physical lane.",
                )
                .with_unit("s")
                .build(),
            capacity_snapshot_timestamp: meter
                .f64_gauge(CAPACITY_SNAPSHOT_TIMESTAMP_METRIC_NAME)
                .with_description(
                    "Unix start timestamp of the latest published gateway KEDA reconciliation.",
                )
                .with_unit("s")
                .build(),
            config_applied_epoch: meter
                .u64_gauge(CONFIG_APPLIED_EPOCH_METRIC_NAME)
                .with_description("Latest control-plane epoch applied by this gateway instance.")
                .with_unit("{epoch}")
                .build(),
            config_operations: meter
                .u64_counter(CONFIG_OPERATIONS_METRIC_NAME)
                .with_description(
                    "Count of bounded config bootstrap and live-delta application outcomes.",
                )
                .with_unit("{operation}")
                .build(),
            config_bootstrap_degraded: meter
                .u64_gauge(CONFIG_BOOTSTRAP_DEGRADED_METRIC_NAME)
                .with_description(
                    "Whether initial control-plane bootstrap is in sustained failure.",
                )
                .with_unit("1")
                .build(),
            messaging_client_ready: meter
                .u64_gauge(MESSAGING_CLIENT_READY_METRIC_NAME)
                .with_description(
                    "Whether the shared NATS Core and JetStream client is connected.",
                )
                .with_unit("1")
                .build(),
            queue_publishes: meter
                .u64_counter(QUEUE_PUBLISHES_METRIC_NAME)
                .with_description("Count of logical queue publish submission outcomes.")
                .with_unit("{publish}")
                .build(),
            queue_publish_duration: meter
                .f64_histogram(QUEUE_PUBLISH_DURATION_METRIC_NAME)
                .with_description("Duration of one logical queue publish submission.")
                .with_unit("s")
                .with_boundaries(QUEUE_PUBLISH_DURATION_BUCKETS.to_vec())
                .build(),
            queue_publish_items: meter
                .f64_histogram(QUEUE_PUBLISH_ITEMS_METRIC_NAME)
                .with_description("Number of work items in one logical queue publish call.")
                .with_unit("{item}")
                .with_boundaries(QUEUE_PUBLISH_ITEMS_BUCKETS.to_vec())
                .build(),
            queue_result_waits: meter
                .u64_counter(QUEUE_RESULT_WAITS_METRIC_NAME)
                .with_description("Count of terminal queue result-wait outcomes.")
                .with_unit("{wait}")
                .build(),
            queue_result_wait_duration: meter
                .f64_histogram(QUEUE_RESULT_WAIT_DURATION_METRIC_NAME)
                .with_description("Duration from successful queue publish to terminal result.")
                .with_unit("s")
                .with_boundaries(QUEUE_RESULT_WAIT_DURATION_BUCKETS.to_vec())
                .build(),
            queue_result_chunks_received: meter
                .u64_counter(QUEUE_RESULT_CHUNKS_RECEIVED_METRIC_NAME)
                .with_description("Count of bounded WorkResult chunks received on the gateway inbox.")
                .with_unit("{chunk}")
                .build(),
            queue_result_chunk_bytes_received: meter
                .u64_counter(QUEUE_RESULT_CHUNK_BYTES_RECEIVED_METRIC_NAME)
                .with_description("WorkResult chunk payload bytes received before reassembly.")
                .with_unit("By")
                .build(),
            queue_result_chunk_rejections: meter
                .u64_counter(QUEUE_RESULT_CHUNK_REJECTIONS_METRIC_NAME)
                .with_description(
                    "Rejected WorkResult chunk transfers by bounded validation reason.",
                )
                .with_unit("{rejection}")
                .build(),
            queue_result_chunk_transfers_completed: meter
                .u64_counter(QUEUE_RESULT_CHUNK_TRANSFERS_COMPLETED_METRIC_NAME)
                .with_description(
                    "WorkResult chunk transfers reassembled, verified, and decoded.",
                )
                .with_unit("{transfer}")
                .build(),
            queue_result_chunk_duplicates: meter
                .u64_counter(QUEUE_RESULT_CHUNK_DUPLICATES_METRIC_NAME)
                .with_description(
                    "Byte-identical duplicate WorkResult chunks accepted idempotently.",
                )
                .with_unit("{chunk}")
                .build(),
            queue_result_chunk_retry_replacements: meter
                .u64_counter(QUEUE_RESULT_CHUNK_RETRY_REPLACEMENTS_METRIC_NAME)
                .with_description(
                    "Partial WorkResult chunk transfers replaced by a new retry layout.",
                )
                .with_unit("{transfer}")
                .build(),
            queue_result_chunk_stale_retries: meter
                .u64_counter(QUEUE_RESULT_CHUNK_STALE_RETRIES_METRIC_NAME)
                .with_description("Delayed or unknown retry fragments ignored safely.")
                .with_unit("{chunk}")
                .build(),
            queue_result_chunk_reserved_bytes: meter
                .u64_observable_gauge(QUEUE_RESULT_CHUNK_RESERVED_BYTES_METRIC_NAME)
                .with_description(
                    "Gateway-wide bytes conservatively reserved for pending WorkResult chunk transfers.",
                )
                .with_unit("By")
                .with_callback(move |observer| {
                    if observable_queue_result_chunk_reserved_bytes_state
                        .initialized
                        .load(Ordering::Acquire)
                    {
                        observer.observe(
                            observable_queue_result_chunk_reserved_bytes_state
                                .reserved_bytes
                                .load(Ordering::Relaxed),
                            &[],
                        );
                    }
                })
                .build(),
            queue_result_chunk_reserved_bytes_state,
            queue_events: meter
                .u64_counter(QUEUE_EVENTS_METRIC_NAME)
                .with_description(
                    "Count of bounded queue ACK, DLQ-forward, and payload-offload outcomes.",
                )
                .with_unit("{event}")
                .build(),
            provisioning_responses: meter
                .u64_counter(PROVISIONING_RESPONSES_METRIC_NAME)
                .with_description(
                    "Count of semantic provisioning responses returned before execution.",
                )
                .with_unit("{response}")
                .build(),
            generation_events: meter
                .u64_counter(GENERATION_EVENTS_METRIC_NAME)
                .with_description(
                    "Count of bounded generation cancellation and transport-integrity events.",
                )
                .with_unit("{event}")
                .build(),
            generation_ttft: meter
                .f64_histogram(GENERATION_TTFT_METRIC_NAME)
                .with_description("Gateway-observed publish-to-first-token duration.")
                .with_unit("s")
                .with_boundaries(GENERATION_DURATION_BUCKETS.to_vec())
                .build(),
            generation_tpot: meter
                .f64_histogram(GENERATION_TPOT_METRIC_NAME)
                .with_description("Gateway-observed mean time per output token.")
                .with_unit("s")
                .with_boundaries(GENERATION_DURATION_BUCKETS.to_vec())
                .build(),
            generation_tokens: meter
                .u64_counter(GENERATION_TOKENS_METRIC_NAME)
                .with_description("Generation tokens accounted from terminal chunk usage.")
                .with_unit("{token}")
                .build(),
            capacity_snapshot_lock: Mutex::new(()),
            previous_pending_lanes: Mutex::new(HashSet::new()),
            previous_lease_lanes: Mutex::new(HashSet::new()),
            previous_warm_floor_lanes: Mutex::new(HashSet::new()),
            pinned_model_state,
        }
    }

    fn request_completed(
        &self,
        operation: &str,
        status: u16,
        machine_profile: &str,
        duration_s: f64,
        admission_outcome: AdmissionOutcome,
    ) {
        let request_attributes = request_attributes(operation, status, machine_profile);
        self.requests.add(1, &request_attributes);
        self.request_duration
            .record(duration_s.max(0.0), &request_attributes);
        self.admission_decisions.add(
            1,
            &admission_attributes(operation, admission_outcome.as_str()),
        );
    }

    #[allow(dead_code)] // Managed composition API.
    fn dispatch_completed(&self, observation: &DispatchObservation<'_>) {
        self.dispatch_completed_with_lanes(observation, manifest_lanes());
    }

    #[allow(dead_code)] // Also provides deterministic contract-test injection.
    fn dispatch_completed_with_lanes(
        &self,
        observation: &DispatchObservation<'_>,
        allowed_lanes: &BTreeSet<String>,
    ) {
        let attributes = dispatch_attributes_with_lanes(observation, allowed_lanes);
        self.dispatches.add(1, &attributes);
        self.dispatch_duration
            .record(observation.duration.as_secs_f64(), &attributes);
    }

    fn set_config_applied_epoch(&self, epoch: u64) {
        self.config_applied_epoch.record(epoch, &[]);
    }

    fn record_config_operation(&self, operation: ConfigOperation, outcome: ConfigOutcome) {
        self.config_operations
            .add(1, &config_attributes(operation, outcome));
    }

    fn set_config_bootstrap_degraded(&self, degraded: bool) {
        self.config_bootstrap_degraded
            .record(u64::from(degraded), &[]);
    }

    fn set_messaging_client_ready(&self, ready: bool) {
        self.messaging_client_ready.record(
            u64::from(ready),
            &[KeyValue::new(
                "transport",
                MessagingTransport::NatsJetstream.as_str(),
            )],
        );
    }

    fn record_queue_publish(&self, observation: &QueuePublishObservation<'_>) {
        let attributes =
            queue_operation_attributes(observation.operation, observation.outcome.as_str());
        self.queue_publishes.add(1, &attributes);
        self.queue_publish_duration
            .record(observation.duration.as_secs_f64(), &attributes);
        self.queue_publish_items
            .record(f64::from(observation.items), &attributes);
    }

    fn record_queue_result_wait(
        &self,
        operation: &str,
        outcome: QueueResultOutcome,
        duration: Duration,
    ) {
        let attributes = queue_operation_attributes(operation, outcome.as_str());
        self.queue_result_waits.add(1, &attributes);
        self.queue_result_wait_duration
            .record(duration.as_secs_f64(), &attributes);
    }

    fn record_queue_result_chunk_received(&self, payload_bytes: Option<u64>) {
        self.queue_result_chunks_received.add(1, &[]);
        if let Some(payload_bytes) = payload_bytes {
            self.queue_result_chunk_bytes_received
                .add(payload_bytes, &[]);
        }
    }

    fn record_queue_result_chunk_rejection(&self, reason: QueueResultChunkRejectionReason) {
        self.queue_result_chunk_rejections
            .add(1, &[KeyValue::new("reason", reason.as_str())]);
    }

    fn record_queue_result_chunk_event(&self, event: QueueResultChunkEvent) {
        match event {
            QueueResultChunkEvent::TransferCompleted => {
                self.queue_result_chunk_transfers_completed.add(1, &[]);
            }
            QueueResultChunkEvent::Duplicate => {
                self.queue_result_chunk_duplicates.add(1, &[]);
            }
            QueueResultChunkEvent::RetryReplacement => {
                self.queue_result_chunk_retry_replacements.add(1, &[]);
            }
            QueueResultChunkEvent::StaleRetry => {
                self.queue_result_chunk_stale_retries.add(1, &[]);
            }
        }
    }

    fn record_queue_result_chunk_reservation_change(
        &self,
        change: QueueResultChunkReservationChange,
    ) {
        match change {
            QueueResultChunkReservationChange::Reserved(bytes) => {
                if let Ok(bytes) = u64::try_from(bytes) {
                    self.queue_result_chunk_reserved_bytes_state
                        .reserved_bytes
                        .fetch_add(bytes, Ordering::Relaxed);
                }
            }
            QueueResultChunkReservationChange::Released(bytes) => {
                if let Ok(bytes) = u64::try_from(bytes) {
                    let _ = self
                        .queue_result_chunk_reserved_bytes_state
                        .reserved_bytes
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                            Some(current.saturating_sub(bytes))
                        });
                }
            }
        }
        self.queue_result_chunk_reserved_bytes_state
            .initialized
            .store(true, Ordering::Release);
    }

    fn record_queue_event(&self, event: QueueEvent, outcome: QueueEventOutcome) {
        self.queue_events.add(
            1,
            &[
                KeyValue::new("event", event.as_str()),
                KeyValue::new("outcome", outcome.as_str()),
            ],
        );
    }

    fn record_provisioning_response(&self, surface: ProvisioningSurface, status: u16) {
        let status = if (100..=599).contains(&status) {
            status
        } else {
            0
        };
        self.provisioning_responses.add(
            1,
            &[
                KeyValue::new("surface", surface.as_str()),
                KeyValue::new("http.status_code", i64::from(status)),
            ],
        );
    }

    fn record_generation_event(
        &self,
        event: GenerationEvent,
        reason: GenerationEventReason,
        outcome: GenerationEventOutcome,
    ) {
        self.generation_events.add(
            1,
            &[
                KeyValue::new("event", event.as_str()),
                KeyValue::new("reason", reason.as_str()),
                KeyValue::new("outcome", outcome.as_str()),
            ],
        );
    }

    fn record_generation_completion(&self, observation: GenerationCompletionObservation) {
        let attributes = [KeyValue::new("operation", "generate")];
        if let Some(ttft_ms) = observation
            .ttft_ms
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.generation_ttft.record(ttft_ms / 1000.0, &attributes);
        }
        if let Some(tpot_ms) = observation
            .tpot_ms
            .filter(|value| value.is_finite() && *value >= 0.0)
        {
            self.generation_tpot.record(tpot_ms / 1000.0, &attributes);
        }
        if let Some(prompt_tokens) = observation.prompt_tokens {
            self.generation_tokens.add(
                prompt_tokens,
                &[
                    KeyValue::new("operation", "generate"),
                    KeyValue::new("token.kind", "prompt"),
                ],
            );
        }
        if let Some(completion_tokens) = observation.completion_tokens {
            self.generation_tokens.add(
                completion_tokens,
                &[
                    KeyValue::new("operation", "generate"),
                    KeyValue::new("token.kind", "completion"),
                ],
            );
        }
    }

    fn set_pending_demand_snapshot(&self, values: &[LaneSnapshot]) {
        set_lane_snapshot(&self.pending_demand, &self.previous_pending_lanes, values);
    }

    fn set_lane_queue_snapshot(&self, values: &[LaneSnapshot]) {
        // The physical catalog is frozen for this process. Omission therefore
        // means this lane's broker read failed, not that the lane was removed.
        // Update successful lanes only: their paired freshness timestamps are
        // refreshed below, while an omitted lane retains its last value with
        // an aging timestamp so KEDA fails closed for exactly that lane.
        record_lane_updates(&self.lane_queue_depth, values);
    }

    fn set_pool_warm_floor_snapshot(&self, values: &[LaneSnapshot]) {
        set_lane_snapshot(
            &self.pool_warm_floor,
            &self.previous_warm_floor_lanes,
            values,
        );
    }

    fn set_pool_pinned_model_loaded_snapshot(&self, values: &[PinnedModelSnapshot]) {
        let current_values: HashMap<PinnedModelKey, bool> = values
            .iter()
            .map(|value| (pinned_model_key(&value.pool, &value.model), value.loaded))
            .collect();
        let mut state = self
            .pinned_model_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed: Vec<_> = state
            .current
            .keys()
            .filter(|key| !current_values.contains_key(*key))
            .cloned()
            .collect();
        state.removed.extend(removed);
        for key in current_values.keys() {
            state.removed.remove(key);
        }
        state.current = current_values;
    }

    fn set_active_lease_snapshot(&self, values: &[LaneSnapshot]) {
        set_lane_snapshot(&self.active_lease_gpus, &self.previous_lease_lanes, values);
    }

    fn record_rejected_request(&self, lane: &PhysicalLane, reason: &str) -> bool {
        let Some(reason) = scale_up_rejection_reason(reason) else {
            return false;
        };
        let attributes = [
            KeyValue::new("pool", lane.pool().to_string()),
            KeyValue::new("machine_profile", lane.machine_profile().to_string()),
            KeyValue::new("bundle", lane.bundle().to_string()),
            KeyValue::new("reason", reason),
            KeyValue::new("scaling_action", "scale_up"),
        ];
        self.rejected_requests.add(1, &attributes);
        true
    }

    fn set_capacity_snapshot_timestamp(&self, unix_time_s: f64) {
        self.capacity_snapshot_timestamp
            .record(unix_time_s.max(0.0), &[]);
    }

    fn set_lane_queue_snapshot_timestamp(&self, values: &[LaneSnapshot], unix_time_s: f64) {
        for value in values {
            self.lane_queue_snapshot_timestamp.record(
                unix_time_s.max(0.0),
                &lane_attributes(&value.pool, &value.machine_profile, &value.bundle),
            );
        }
    }
}

trait KedaSnapshotSink {
    fn record_pending_demand(&self, values: &[LaneSnapshot]);
    fn record_lane_queue_depth(&self, values: &[LaneSnapshot]);
    fn record_lane_queue_timestamp(&self, values: &[LaneSnapshot], unix_time_s: f64);
    fn record_active_lease_gpus(&self, values: &[LaneSnapshot]);
    fn record_pool_warm_floor(&self, values: &[LaneSnapshot]);
    fn record_timestamp(&self, unix_time_s: f64);
}

impl KedaSnapshotSink for GatewayTelemetry {
    fn record_pending_demand(&self, values: &[LaneSnapshot]) {
        self.set_pending_demand_snapshot(values);
    }

    fn record_lane_queue_depth(&self, values: &[LaneSnapshot]) {
        self.set_lane_queue_snapshot(values);
    }

    fn record_lane_queue_timestamp(&self, values: &[LaneSnapshot], unix_time_s: f64) {
        self.set_lane_queue_snapshot_timestamp(values, unix_time_s);
    }

    fn record_active_lease_gpus(&self, values: &[LaneSnapshot]) {
        self.set_active_lease_snapshot(values);
    }

    fn record_pool_warm_floor(&self, values: &[LaneSnapshot]) {
        self.set_pool_warm_floor_snapshot(values);
    }

    fn record_timestamp(&self, unix_time_s: f64) {
        self.set_capacity_snapshot_timestamp(unix_time_s);
    }
}

fn telemetry() -> Option<&'static GatewayTelemetry> {
    if !super::tracing::metrics_exporter_enabled() {
        return None;
    }
    static TELEMETRY: OnceLock<GatewayTelemetry> = OnceLock::new();
    Some(TELEMETRY.get_or_init(|| GatewayTelemetry::new(&global::meter("sie-gateway"))))
}

fn bounded_operation(operation: &str) -> &'static str {
    match operation {
        "encode" => "encode",
        "score" => "score",
        "extract" => "extract",
        "embeddings" => "embeddings",
        "moderations" => "moderations",
        "generate" => "generate",
        _ => "other",
    }
}

#[allow(dead_code)] // Managed composition API.
fn parse_manifest_lanes(raw: Option<&str>) -> BTreeSet<String> {
    raw.and_then(|value| {
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(value).ok()
    })
    .map(|mapping| mapping.into_iter().map(|(lane, _app)| lane).collect())
    .unwrap_or_default()
}

#[allow(dead_code)] // Managed composition API.
fn manifest_lanes() -> &'static BTreeSet<String> {
    static LANES: OnceLock<BTreeSet<String>> = OnceLock::new();
    LANES.get_or_init(|| parse_manifest_lanes(std::env::var(LANE_APP_MAP_ENV).ok().as_deref()))
}

/// Keep `lane` on the release-manifest catalog. An unpinned model initially
/// names a per-model logical lane; when that exact lane is not deployed, both
/// i6pn and Modal remap it to the catalog's shared `default` lane. All other
/// values collapse instead of exposing caller-derived strings.
#[allow(dead_code)] // Managed composition API.
fn bounded_lane(lane: &str, allowed: &BTreeSet<String>) -> String {
    if allowed.contains(lane) {
        return lane.to_string();
    }

    let parts: Vec<&str> = lane.split('|').collect();
    if let [pool, profile, _model] = parts.as_slice() {
        let lazy_lane = format!("{pool}|{profile}|default");
        if allowed.contains(&lazy_lane) {
            return lazy_lane;
        }
    }
    "other".to_string()
}

#[allow(dead_code)] // Managed composition API.
fn dispatch_attributes_with_lanes(
    observation: &DispatchObservation<'_>,
    allowed_lanes: &BTreeSet<String>,
) -> [KeyValue; 5] {
    [
        KeyValue::new("operation", bounded_operation(observation.operation)),
        KeyValue::new("dispatch.path", observation.path.as_str()),
        KeyValue::new("outcome", observation.outcome.as_str()),
        KeyValue::new("fallback.reason", observation.fallback_reason.as_str()),
        KeyValue::new("lane", bounded_lane(observation.lane, allowed_lanes)),
    ]
}

pub(crate) fn request_outcome(status: u16) -> &'static str {
    match status {
        200..=299 => "success",
        300..=399 => "redirect",
        400..=499 => "client_error",
        500..=599 => "server_error",
        _ => "other",
    }
}

pub(crate) fn bounded_http_status(status: u16) -> u16 {
    if (100..=599).contains(&status) {
        status
    } else {
        0
    }
}

fn request_attributes(operation: &str, status: u16, machine_profile: &str) -> [KeyValue; 4] {
    let status = bounded_http_status(status);
    [
        KeyValue::new("operation", bounded_operation(operation)),
        KeyValue::new("outcome", request_outcome(status)),
        KeyValue::new("http.status_code", i64::from(status)),
        KeyValue::new("machine_profile", sanitize_label(machine_profile)),
    ]
}

fn admission_attributes(operation: &str, outcome: &str) -> [KeyValue; 2] {
    [
        KeyValue::new("operation", bounded_operation(operation)),
        KeyValue::new("outcome", outcome.to_string()),
    ]
}

fn config_attributes(operation: ConfigOperation, outcome: ConfigOutcome) -> [KeyValue; 2] {
    [
        KeyValue::new("operation", operation.as_str()),
        KeyValue::new("outcome", outcome.as_str()),
    ]
}

fn queue_operation_attributes(operation: &str, outcome: &'static str) -> [KeyValue; 2] {
    [
        KeyValue::new("operation", bounded_operation(operation)),
        KeyValue::new("outcome", outcome),
    ]
}

fn lane_key(pool: &str, machine_profile: &str, bundle: &str) -> LaneKey {
    let pool = normalized_lane_label(pool, "default");
    let machine_profile = normalized_lane_label(machine_profile, "other");
    let bundle = normalized_lane_label(bundle, "default");
    (
        sanitize_label(&pool),
        sanitize_label(&machine_profile),
        sanitize_label(&bundle),
    )
}

fn pinned_model_key(pool: &str, model: &str) -> PinnedModelKey {
    (sanitize_label(pool), sanitize_model_label(model))
}

fn sanitize_model_label(value: &str) -> String {
    const MAX_MODEL_LABEL_LEN: usize = 256;
    if value.is_empty()
        || value.len() > MAX_MODEL_LABEL_LEN
        || !value.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '.' | '_' | ':' | '/' | '@' | '+' | '-')
        })
    {
        return "other".to_string();
    }
    value.to_string()
}

fn normalized_lane_label(value: &str, empty_fallback: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        empty_fallback.to_string()
    } else {
        normalized
    }
}

fn lane_attributes(pool: &str, machine_profile: &str, bundle: &str) -> [KeyValue; 3] {
    let (pool, machine_profile, bundle) = lane_key(pool, machine_profile, bundle);
    [
        KeyValue::new("pool", pool),
        KeyValue::new("machine_profile", machine_profile),
        KeyValue::new("bundle", bundle),
    ]
}

fn pinned_model_attributes(pool: &str, model: &str) -> [KeyValue; 2] {
    [
        KeyValue::new("pool", pool.to_string()),
        KeyValue::new("model", model.to_string()),
    ]
}

fn set_lane_snapshot(
    instrument: &Gauge<f64>,
    previous: &Mutex<HashSet<LaneKey>>,
    values: &[LaneSnapshot],
) {
    let mut current_values: HashMap<LaneKey, f64> = HashMap::new();
    for value in values {
        let key = lane_key(&value.pool, &value.machine_profile, &value.bundle);
        current_values
            .entry(key)
            .and_modify(|current| *current = current.max(value.value.max(0.0)))
            .or_insert_with(|| value.value.max(0.0));
    }
    let current: HashSet<LaneKey> = current_values.keys().cloned().collect();
    let mut previous = previous
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (pool, machine_profile, bundle) in previous.difference(&current) {
        instrument.record(0.0, &lane_attributes(pool, machine_profile, bundle));
    }
    *previous = current;
    drop(previous);

    let mut current_values: Vec<_> = current_values.into_iter().collect();
    current_values.sort_by(|(left, _), (right, _)| left.cmp(right));
    for ((pool, machine_profile, bundle), value) in current_values {
        instrument.record(value, &lane_attributes(&pool, &machine_profile, &bundle));
    }
}

fn record_lane_updates(instrument: &Gauge<f64>, values: &[LaneSnapshot]) {
    let mut current_values: HashMap<LaneKey, f64> = HashMap::new();
    for value in values {
        let key = lane_key(&value.pool, &value.machine_profile, &value.bundle);
        current_values
            .entry(key)
            .and_modify(|current| *current = current.max(value.value.max(0.0)))
            .or_insert_with(|| value.value.max(0.0));
    }
    let mut current_values: Vec<_> = current_values.into_iter().collect();
    current_values.sort_by(|(left, _), (right, _)| left.cmp(right));
    for ((pool, machine_profile, bundle), value) in current_values {
        instrument.record(value, &lane_attributes(&pool, &machine_profile, &bundle));
    }
}

fn record_request_completed_to(
    telemetry: Option<&GatewayTelemetry>,
    span_context: Option<&SpanContext>,
    operation: &str,
    status: u16,
    machine_profile: &str,
    duration_s: f64,
    admission_outcome: AdmissionOutcome,
) {
    if let Some(telemetry) = telemetry {
        telemetry.request_completed(
            operation,
            status,
            machine_profile,
            duration_s,
            admission_outcome,
        );
    }
    super::tracing::record_inference_completion_log(
        span_context,
        bounded_operation(operation),
        status,
    );
}

/// Record one completed inference request, its admission decision, and its one
/// privacy-safe completion log from a single semantic call site.
pub fn record_request_completed(
    span_context: Option<&SpanContext>,
    operation: &str,
    status: u16,
    machine_profile: &str,
    duration_s: f64,
    admission_outcome: AdmissionOutcome,
) {
    record_request_completed_to(
        telemetry(),
        span_context,
        operation,
        status,
        machine_profile,
        duration_s,
        admission_outcome,
    );
}

#[allow(dead_code)] // Managed composition API plus disabled-path benchmark seam.
fn record_dispatch_to(
    target: Option<&GatewayTelemetry>,
    observation: &DispatchObservation<'_>,
) -> bool {
    let Some(telemetry) = target else {
        return false;
    };
    telemetry.dispatch_completed(observation);
    true
}

/// Record one final-path dispatch event on the canonical counter and duration
/// histogram. The exporter-disabled path returns before allocating attributes.
#[allow(dead_code)] // Managed composition API; unused by the standalone gateway binary.
pub fn record_dispatch(observation: DispatchObservation<'_>) {
    let _ = record_dispatch_to(telemetry(), &observation);
}

fn set_config_applied_epoch_to(target: Option<&GatewayTelemetry>, epoch: u64) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.set_config_applied_epoch(epoch);
    true
}

/// Publish the current gateway-applied control-plane epoch.
pub fn set_config_applied_epoch(epoch: u64) {
    let _ = set_config_applied_epoch_to(telemetry(), epoch);
}

/// Count one terminal bootstrap or live config-delta application outcome.
pub fn record_config_operation(operation: ConfigOperation, outcome: ConfigOutcome) {
    if let Some(telemetry) = telemetry() {
        telemetry.record_config_operation(operation, outcome);
    }
}

/// Set the sustained initial-bootstrap failure state.
pub fn set_config_bootstrap_degraded(degraded: bool) {
    if let Some(telemetry) = telemetry() {
        telemetry.set_config_bootstrap_degraded(degraded);
    }
}

/// Set readiness for the gateway's shared NATS Core and JetStream client.
pub fn set_messaging_client_ready(ready: bool) {
    if let Some(telemetry) = telemetry() {
        telemetry.set_messaging_client_ready(ready);
    }
}

fn record_queue_publish_to(
    target: Option<&GatewayTelemetry>,
    observation: &QueuePublishObservation<'_>,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_queue_publish(observation);
    true
}

/// Record one logical queue submission exactly once. The facade fans that
/// event into the publish counter, duration histogram, and item-count
/// histogram. The detached durability monitor owns the separate ACK event.
pub fn record_queue_publish(observation: QueuePublishObservation<'_>) {
    let _ = record_queue_publish_to(telemetry(), &observation);
}

/// Record one terminal result-wait event exactly once.
pub fn record_queue_result_wait(operation: &str, outcome: QueueResultOutcome, duration: Duration) {
    if let Some(telemetry) = telemetry() {
        telemetry.record_queue_result_wait(operation, outcome, duration);
    }
}

fn record_queue_result_chunk_received_to(
    target: Option<&GatewayTelemetry>,
    payload_bytes: Option<u64>,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_queue_result_chunk_received(payload_bytes);
    true
}

/// Count one received result-chunk envelope. Successfully decoded envelopes
/// also account their bounded payload bytes through this same semantic call.
pub fn record_queue_result_chunk_received(payload_bytes: Option<usize>) {
    let payload_bytes = payload_bytes.and_then(|value| u64::try_from(value).ok());
    let _ = record_queue_result_chunk_received_to(telemetry(), payload_bytes);
}

fn record_queue_result_chunk_rejection_to(
    target: Option<&GatewayTelemetry>,
    reason: QueueResultChunkRejectionReason,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_queue_result_chunk_rejection(reason);
    true
}

/// Count one rejected result-chunk transfer with a finite typed reason.
pub fn record_queue_result_chunk_rejection(reason: QueueResultChunkRejectionReason) {
    let _ = record_queue_result_chunk_rejection_to(telemetry(), reason);
}

fn record_queue_result_chunk_event_to(
    target: Option<&GatewayTelemetry>,
    event: QueueResultChunkEvent,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_queue_result_chunk_event(event);
    true
}

/// Count one bounded result-chunk processing event.
pub fn record_queue_result_chunk_event(event: QueueResultChunkEvent) {
    let _ = record_queue_result_chunk_event_to(telemetry(), event);
}

fn record_queue_result_chunk_reservation_change_to(
    target: Option<&GatewayTelemetry>,
    change: QueueResultChunkReservationChange,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_queue_result_chunk_reservation_change(change);
    true
}

/// Apply one successful process-wide reservation change. The facade-owned
/// observable state uses commutative deltas so concurrent publisher CAS
/// completions cannot leave an out-of-order absolute gauge value behind.
pub fn record_queue_result_chunk_reservation_change(change: QueueResultChunkReservationChange) {
    let _ = record_queue_result_chunk_reservation_change_to(telemetry(), change);
}

/// Count one bounded queue durability event.
pub fn record_queue_event(event: QueueEvent, outcome: QueueEventOutcome) {
    if let Some(telemetry) = telemetry() {
        telemetry.record_queue_event(event, outcome);
    }
}

fn record_provisioning_response_to(
    target: Option<&GatewayTelemetry>,
    surface: ProvisioningSurface,
    status: u16,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_provisioning_response(surface, status);
    true
}

/// Count one semantic pre-execution provisioning response. The typed surface
/// and bounded numeric status are the complete attribute set.
pub fn record_provisioning_response(surface: ProvisioningSurface, status: u16) {
    let _ = record_provisioning_response_to(telemetry(), surface, status);
}

fn record_generation_event_to(
    target: Option<&GatewayTelemetry>,
    event: GenerationEvent,
    reason: GenerationEventReason,
    outcome: GenerationEventOutcome,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_generation_event(event, reason, outcome);
    true
}

/// Count one bounded generation cancellation or transport-integrity event.
pub fn record_generation_event(
    event: GenerationEvent,
    reason: GenerationEventReason,
    outcome: GenerationEventOutcome,
) {
    let _ = record_generation_event_to(telemetry(), event, reason, outcome);
}

fn record_generation_completion_to(
    target: Option<&GatewayTelemetry>,
    observation: GenerationCompletionObservation,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_generation_completion(observation);
    true
}

/// Record gateway-observed timing and token accounting from the single
/// authoritative successful terminal-chunk funnel.
pub fn record_generation_completion(observation: GenerationCompletionObservation) {
    let _ = record_generation_completion_to(telemetry(), observation);
}

fn record_rejected_request_to(
    target: Option<&GatewayTelemetry>,
    demand_tracker: &DemandTracker,
    lane: &PhysicalLane,
    reason: &str,
) -> bool {
    if !demand_tracker.catalog().contains(lane) {
        return false;
    }
    target.is_some_and(|target| target.record_rejected_request(lane, reason))
}

/// Count one scale-worthy rejection for a deployment-configured physical lane.
///
/// This KEDA control stream is deliberately narrower than request diagnostics:
/// non-scale-worthy outcomes are already represented by the canonical request
/// completion counter/log, while caller-manufactured lane tuples are rejected
/// against the gateway's deployment-owned catalog before touching OTel.
pub(crate) fn record_rejected_request(
    demand_tracker: &DemandTracker,
    lane: &PhysicalLane,
    reason: &str,
) {
    let _ = record_rejected_request_to(telemetry(), demand_tracker, lane, reason);
}

fn scale_up_rejection_reason(reason: &str) -> Option<&'static str> {
    match reason {
        "backpressure" => Some("backpressure"),
        "no_consumers" => Some("no_consumers"),
        "publish_ack_failed" => Some("publish_ack_failed"),
        "upstream_result_timeout" => Some("upstream_result_timeout"),
        _ => None,
    }
}

fn record_keda_capacity_snapshot_to(
    target: Option<&dyn KedaSnapshotSink>,
    snapshot: &KedaCapacitySnapshot,
    unix_time_s: f64,
) -> bool {
    let Some(target) = target else {
        return false;
    };
    target.record_pending_demand(&snapshot.pending_demand);
    target.record_lane_queue_depth(&snapshot.lane_queue_depth);
    target.record_lane_queue_timestamp(&snapshot.lane_queue_depth, unix_time_s);
    target.record_active_lease_gpus(&snapshot.active_lease_gpus);
    target.record_pool_warm_floor(&snapshot.pool_warm_floor);
    target.record_timestamp(unix_time_s);
    true
}

/// Emit one gateway-owned KEDA capacity reconciliation. The caller captures
/// `snapshot_started_unix_time_s` before async state collection; this facade
/// records it after all state values so a slow build ages out rather than
/// refreshing stale capacity. Synchronous OTel gauge collection is not claimed
/// to be export-atomic across the five instruments.
pub(crate) fn record_keda_capacity_snapshot(
    snapshot: &KedaCapacitySnapshot,
    snapshot_started_unix_time_s: f64,
) {
    let Some(telemetry) = telemetry() else {
        return;
    };
    let _snapshot_guard = telemetry
        .capacity_snapshot_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _ =
        record_keda_capacity_snapshot_to(Some(telemetry), snapshot, snapshot_started_unix_time_s);
}

/// Emit the complete current pinned-model readiness snapshot.
///
/// Lanes omitted from a later snapshot are explicitly recorded as zero so a
/// deleted pool or removed pin cannot leave a stale healthy series behind.
pub fn record_pool_pinned_model_loaded_snapshot(snapshot: &[PinnedModelSnapshot]) {
    let Some(telemetry) = telemetry() else {
        return;
    };
    telemetry.set_pool_pinned_model_loaded_snapshot(snapshot);
}

#[cfg(test)]
pub(crate) fn benchmark_keda_capacity_emit_export(
    snapshot: &KedaCapacitySnapshot,
) -> (Duration, usize) {
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::{
        InMemoryMetricExporterBuilder, PeriodicReader, SdkMeterProvider, Temporality,
    };

    let exporter = InMemoryMetricExporterBuilder::new()
        .with_temporality(Temporality::LowMemory)
        .build();
    let reader = PeriodicReader::builder(exporter.clone()).build();
    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_view(crate::observability::tracing::keda_metric_cardinality_view)
        .build();
    let telemetry = GatewayTelemetry::new(&provider.meter("sie-gateway-capacity-benchmark"));
    let started = std::time::Instant::now();
    let _ = record_keda_capacity_snapshot_to(Some(&telemetry), snapshot, 1_721_177_600.0);
    provider
        .force_flush()
        .expect("force_flush benchmark metrics");
    let elapsed = started.elapsed();
    let exported_points = exporter
        .get_finished_metrics()
        .expect("finished benchmark metrics")
        .iter()
        .flat_map(|resource| resource.scope_metrics())
        .flat_map(|scope| scope.metrics())
        .filter(|metric| {
            matches!(
                metric.name(),
                PENDING_DEMAND_METRIC_NAME
                    | LANE_QUEUE_DEPTH_METRIC_NAME
                    | LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME
                    | ACTIVE_LEASE_GPUS_METRIC_NAME
                    | POOL_WARM_FLOOR_METRIC_NAME
            )
        })
        .map(|metric| match metric.data() {
            opentelemetry_sdk::metrics::data::AggregatedMetrics::F64(
                opentelemetry_sdk::metrics::data::MetricData::Gauge(gauge),
            ) => gauge.data_points().count(),
            _ => 0,
        })
        .sum();
    provider.shutdown().expect("shutdown benchmark provider");
    (elapsed, exported_points)
}

#[cfg(test)]
pub(crate) fn telemetry_performance_budget(name: &str) -> f64 {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../telemetry/performance-budgets.json");
    let document: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read telemetry budget {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("parse telemetry budget {}: {error}", path.display()));
    document["budgets"][name]
        .as_f64()
        .unwrap_or_else(|| panic!("missing numeric telemetry performance budget {name}"))
}

#[cfg(test)]
pub(crate) fn telemetry_benchmark_median(mut samples: [f64; 3]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[1]
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::hint::black_box;
    use std::sync::Mutex;
    use std::time::Instant;

    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    use super::*;

    fn metric_points() -> (GatewayTelemetry, InMemoryMetricExporter, SdkMeterProvider) {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_view(crate::observability::tracing::keda_metric_cardinality_view)
            .build();
        let telemetry = GatewayTelemetry::new(&provider.meter("sie-gateway-test"));
        (telemetry, exporter, provider)
    }

    #[test]
    fn dispatch_contract_exports_one_counter_and_histogram_observation() {
        let (telemetry, exporter, provider) = metric_points();
        let allowed = parse_manifest_lanes(Some(r#"{"default|l4|BAAI/bge-m3":"sie-cloud-lanes"}"#));
        telemetry.dispatch_completed_with_lanes(
            &DispatchObservation {
                operation: "score",
                path: DispatchPath::I6pn,
                outcome: DispatchOutcome::Success,
                fallback_reason: FallbackReason::None,
                lane: "default|l4|BAAI/bge-m3",
                duration: Duration::from_millis(25),
            },
            &allowed,
        );

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let metrics: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        assert_eq!(metrics.len(), 2, "one event owns exactly two instruments");

        let dispatches = metrics
            .iter()
            .find(|metric| metric.name() == DISPATCHES_METRIC_NAME)
            .expect("dispatch counter");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = dispatches.data() else {
            panic!("dispatches must export as a u64 sum")
        };
        let dispatch_point = sum.data_points().next().expect("dispatch data point");
        assert_eq!(dispatch_point.value(), 1, "one event must increment once");
        assert_eq!(dispatches.unit(), "{request}");
        let dispatch_attributes: HashMap<_, _> = dispatch_point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            dispatch_attributes,
            HashMap::from([
                ("operation", "score".to_string()),
                ("dispatch.path", "i6pn".to_string()),
                ("outcome", "success".to_string()),
                ("fallback.reason", "none".to_string()),
                ("lane", "default|l4|BAAI/bge-m3".to_string()),
            ])
        );

        let duration = metrics
            .iter()
            .find(|metric| metric.name() == DISPATCH_DURATION_METRIC_NAME)
            .expect("dispatch duration");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = duration.data() else {
            panic!("dispatch duration must export as an f64 histogram")
        };
        let duration_point = histogram.data_points().next().expect("duration point");
        assert_eq!(duration.unit(), "s");
        assert_eq!(duration_point.count(), 1);
        assert_eq!(duration_point.sum(), 0.025);
        assert_eq!(
            duration_point.bounds().collect::<Vec<_>>(),
            REQUEST_LATENCY_BUCKETS
        );
        let duration_attributes: HashMap<_, _> = duration_point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(duration_attributes, dispatch_attributes);
    }

    #[test]
    fn operational_contract_exports_grouped_bounded_observations() {
        let (telemetry, exporter, provider) = metric_points();
        telemetry.set_config_applied_epoch(42);
        telemetry.record_config_operation(ConfigOperation::Bootstrap, ConfigOutcome::Success);
        telemetry.set_config_bootstrap_degraded(true);
        telemetry.set_messaging_client_ready(true);
        telemetry.record_queue_publish(&QueuePublishObservation {
            operation: "caller-controlled",
            outcome: QueuePublishOutcome::NoConsumers,
            duration: Duration::from_millis(25),
            items: 5,
        });
        telemetry.record_queue_result_wait(
            "generate",
            QueueResultOutcome::Timeout,
            Duration::from_millis(300),
        );
        telemetry.record_queue_event(QueueEvent::DlqForward, QueueEventOutcome::Deduplicated);

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let exported: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        let by_name: HashMap<_, _> = exported
            .iter()
            .map(|metric| (metric.name().to_string(), *metric))
            .collect();
        assert_eq!(
            by_name.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from([
                CONFIG_APPLIED_EPOCH_METRIC_NAME.to_string(),
                CONFIG_OPERATIONS_METRIC_NAME.to_string(),
                CONFIG_BOOTSTRAP_DEGRADED_METRIC_NAME.to_string(),
                MESSAGING_CLIENT_READY_METRIC_NAME.to_string(),
                QUEUE_PUBLISHES_METRIC_NAME.to_string(),
                QUEUE_PUBLISH_DURATION_METRIC_NAME.to_string(),
                QUEUE_PUBLISH_ITEMS_METRIC_NAME.to_string(),
                QUEUE_RESULT_WAITS_METRIC_NAME.to_string(),
                QUEUE_RESULT_WAIT_DURATION_METRIC_NAME.to_string(),
                QUEUE_EVENTS_METRIC_NAME.to_string(),
            ])
        );

        let config_epoch = by_name[CONFIG_APPLIED_EPOCH_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Gauge(config_epoch)) = config_epoch.data() else {
            panic!("config epoch must be a u64 gauge")
        };
        let config_epoch_point = config_epoch
            .data_points()
            .next()
            .expect("config epoch point");
        assert_eq!(config_epoch_point.value(), 42);
        assert_eq!(config_epoch_point.attributes().count(), 0);

        let config_operations = by_name[CONFIG_OPERATIONS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(config_operations)) = config_operations.data()
        else {
            panic!("config operations must be a u64 counter")
        };
        let config_attributes: HashMap<_, _> = config_operations
            .data_points()
            .next()
            .expect("config operation point")
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            config_attributes,
            HashMap::from([
                ("operation", "bootstrap".to_string()),
                ("outcome", "success".to_string()),
            ])
        );

        let readiness = by_name[MESSAGING_CLIENT_READY_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Gauge(readiness)) = readiness.data() else {
            panic!("messaging readiness must be a u64 gauge")
        };
        let readiness_point = readiness.data_points().next().expect("readiness point");
        assert_eq!(readiness_point.value(), 1);
        let readiness_attributes: HashMap<_, _> = readiness_point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            readiness_attributes,
            HashMap::from([("transport", "nats_jetstream".to_string())])
        );

        let publishes = by_name[QUEUE_PUBLISHES_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(publishes)) = publishes.data() else {
            panic!("publishes must be a u64 counter")
        };
        let publish_attributes: HashMap<_, _> = publishes
            .data_points()
            .next()
            .expect("publish point")
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            publish_attributes,
            HashMap::from([
                ("operation", "other".to_string()),
                ("outcome", "no_consumers".to_string()),
            ])
        );

        let publish_duration = by_name[QUEUE_PUBLISH_DURATION_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Histogram(publish_duration)) =
            publish_duration.data()
        else {
            panic!("publish duration must be an f64 histogram")
        };
        let publish_duration_point = publish_duration
            .data_points()
            .next()
            .expect("publish duration point");
        assert_eq!(publish_duration_point.count(), 1);
        assert_eq!(publish_duration_point.sum(), 0.025);
        assert_eq!(
            publish_duration_point.bounds().collect::<Vec<_>>(),
            QUEUE_PUBLISH_DURATION_BUCKETS
        );

        let publish_items = by_name[QUEUE_PUBLISH_ITEMS_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Histogram(publish_items)) = publish_items.data()
        else {
            panic!("publish items must be an f64 histogram")
        };
        let publish_items_point = publish_items
            .data_points()
            .next()
            .expect("publish items point");
        assert_eq!(publish_items_point.count(), 1);
        assert_eq!(publish_items_point.sum(), 5.0);
        assert_eq!(
            publish_items_point.bounds().collect::<Vec<_>>(),
            QUEUE_PUBLISH_ITEMS_BUCKETS
        );

        let result_waits = by_name[QUEUE_RESULT_WAITS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(result_waits)) = result_waits.data() else {
            panic!("result waits must be a u64 counter")
        };
        let result_attributes: HashMap<_, _> = result_waits
            .data_points()
            .next()
            .expect("result wait point")
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            result_attributes,
            HashMap::from([
                ("operation", "generate".to_string()),
                ("outcome", "timeout".to_string()),
            ])
        );

        let queue_events = by_name[QUEUE_EVENTS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(queue_events)) = queue_events.data() else {
            panic!("queue events must be a u64 counter")
        };
        let event_attributes: HashMap<_, _> = queue_events
            .data_points()
            .next()
            .expect("queue event point")
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            event_attributes,
            HashMap::from([
                ("event", "dlq_forward".to_string()),
                ("outcome", "deduplicated".to_string()),
            ])
        );
    }

    #[test]
    fn result_chunk_contract_exports_exact_bounded_semantics_once() {
        let (telemetry, exporter, provider) = metric_points();
        telemetry.record_queue_result_chunk_received(Some(128));
        telemetry.record_queue_result_chunk_rejection(QueueResultChunkRejectionReason::Decode);
        for event in [
            QueueResultChunkEvent::TransferCompleted,
            QueueResultChunkEvent::Duplicate,
            QueueResultChunkEvent::RetryReplacement,
            QueueResultChunkEvent::StaleRetry,
        ] {
            telemetry.record_queue_result_chunk_event(event);
        }
        telemetry.record_queue_result_chunk_reservation_change(
            QueueResultChunkReservationChange::Reserved(4_096),
        );
        telemetry.record_queue_result_chunk_reservation_change(
            QueueResultChunkReservationChange::Reserved(2_048),
        );
        telemetry.record_queue_result_chunk_reservation_change(
            QueueResultChunkReservationChange::Released(1_024),
        );

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let exported: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        let by_name: HashMap<_, _> = exported
            .iter()
            .map(|metric| (metric.name().to_string(), *metric))
            .collect();
        assert_eq!(
            by_name.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from([
                QUEUE_RESULT_CHUNKS_RECEIVED_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_BYTES_RECEIVED_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_REJECTIONS_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_TRANSFERS_COMPLETED_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_DUPLICATES_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_RETRY_REPLACEMENTS_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_STALE_RETRIES_METRIC_NAME.to_string(),
                QUEUE_RESULT_CHUNK_RESERVED_BYTES_METRIC_NAME.to_string(),
            ])
        );

        for metric_name in [
            QUEUE_RESULT_CHUNKS_RECEIVED_METRIC_NAME,
            QUEUE_RESULT_CHUNK_TRANSFERS_COMPLETED_METRIC_NAME,
            QUEUE_RESULT_CHUNK_DUPLICATES_METRIC_NAME,
            QUEUE_RESULT_CHUNK_RETRY_REPLACEMENTS_METRIC_NAME,
            QUEUE_RESULT_CHUNK_STALE_RETRIES_METRIC_NAME,
        ] {
            let AggregatedMetrics::U64(MetricData::Sum(sum)) = by_name[metric_name].data() else {
                panic!("{metric_name} must be a u64 counter")
            };
            let point = sum.data_points().next().expect("counter point");
            assert_eq!(point.value(), 1, "{metric_name}");
            assert_eq!(point.attributes().count(), 0, "{metric_name}");
        }

        let AggregatedMetrics::U64(MetricData::Sum(bytes)) =
            by_name[QUEUE_RESULT_CHUNK_BYTES_RECEIVED_METRIC_NAME].data()
        else {
            panic!("result chunk bytes must be a u64 counter")
        };
        assert_eq!(
            bytes.data_points().next().expect("bytes point").value(),
            128
        );

        let AggregatedMetrics::U64(MetricData::Sum(rejections)) =
            by_name[QUEUE_RESULT_CHUNK_REJECTIONS_METRIC_NAME].data()
        else {
            panic!("result chunk rejections must be a u64 counter")
        };
        let rejection = rejections
            .data_points()
            .next()
            .expect("result chunk rejection point");
        assert_eq!(rejection.value(), 1);
        assert_eq!(
            rejection
                .attributes()
                .map(|attribute| (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                ))
                .collect::<HashMap<_, _>>(),
            HashMap::from([("reason", "decode".to_string())])
        );

        let AggregatedMetrics::U64(MetricData::Gauge(reserved)) =
            by_name[QUEUE_RESULT_CHUNK_RESERVED_BYTES_METRIC_NAME].data()
        else {
            panic!("result chunk reserved bytes must be a u64 gauge")
        };
        let reserved = reserved.data_points().next().expect("reserved bytes point");
        assert_eq!(reserved.value(), 5_120);
        assert_eq!(reserved.attributes().count(), 0);
    }

    #[test]
    fn result_chunk_reservation_changes_commute_across_threads() {
        let (telemetry, _exporter, provider) = metric_points();
        std::thread::scope(|scope| {
            let telemetry = &telemetry;
            for _ in 0..8 {
                scope.spawn(move || {
                    for _ in 0..1_000 {
                        telemetry.record_queue_result_chunk_reservation_change(
                            QueueResultChunkReservationChange::Reserved(1),
                        );
                        telemetry.record_queue_result_chunk_reservation_change(
                            QueueResultChunkReservationChange::Released(1),
                        );
                    }
                });
            }
        });
        assert_eq!(
            telemetry
                .queue_result_chunk_reserved_bytes_state
                .reserved_bytes
                .load(Ordering::Relaxed),
            0
        );
        provider.shutdown().expect("shutdown test provider");
    }

    #[test]
    fn generation_contract_exports_exact_bounded_semantics_once() {
        let (telemetry, exporter, provider) = metric_points();
        telemetry.record_provisioning_response(ProvisioningSurface::OpenAi, 503);
        telemetry.record_generation_event(
            GenerationEvent::Chunk,
            GenerationEventReason::SequenceGap,
            GenerationEventOutcome::Rejected,
        );
        telemetry.record_generation_completion(GenerationCompletionObservation {
            ttft_ms: Some(125.0),
            tpot_ms: Some(25.0),
            prompt_tokens: Some(11),
            completion_tokens: Some(7),
        });

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let exported: Vec<_> = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        let by_name: HashMap<_, _> = exported
            .iter()
            .map(|metric| (metric.name().to_string(), *metric))
            .collect();
        assert_eq!(
            by_name.keys().cloned().collect::<HashSet<_>>(),
            HashSet::from([
                PROVISIONING_RESPONSES_METRIC_NAME.to_string(),
                GENERATION_EVENTS_METRIC_NAME.to_string(),
                GENERATION_TTFT_METRIC_NAME.to_string(),
                GENERATION_TPOT_METRIC_NAME.to_string(),
                GENERATION_TOKENS_METRIC_NAME.to_string(),
            ])
        );

        let provisioning = by_name[PROVISIONING_RESPONSES_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(provisioning)) = provisioning.data() else {
            panic!("provisioning responses must be a u64 counter")
        };
        let provisioning_point = provisioning
            .data_points()
            .next()
            .expect("provisioning point");
        assert_eq!(provisioning_point.value(), 1);
        let provisioning_attributes: HashMap<_, _> = provisioning_point
            .attributes()
            .map(|attribute| (attribute.key.as_str(), attribute.value.to_string()))
            .collect();
        assert_eq!(
            provisioning_attributes,
            HashMap::from([
                ("surface", "openai".to_string()),
                ("http.status_code", "503".to_string()),
            ])
        );

        let events = by_name[GENERATION_EVENTS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(events)) = events.data() else {
            panic!("generation events must be a u64 counter")
        };
        let event_point = events.data_points().next().expect("generation event point");
        assert_eq!(event_point.value(), 1);
        let event_attributes: HashMap<_, _> = event_point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            event_attributes,
            HashMap::from([
                ("event", "chunk".to_string()),
                ("reason", "sequence_gap".to_string()),
                ("outcome", "rejected".to_string()),
            ])
        );

        for (name, expected_sum) in [
            (GENERATION_TTFT_METRIC_NAME, 0.125),
            (GENERATION_TPOT_METRIC_NAME, 0.025),
        ] {
            let metric = by_name[name];
            let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = metric.data() else {
                panic!("{name} must be an f64 histogram")
            };
            let point = histogram.data_points().next().expect("timing point");
            assert_eq!(point.count(), 1);
            assert_eq!(point.sum(), expected_sum);
            assert_eq!(
                point.bounds().collect::<Vec<_>>(),
                GENERATION_DURATION_BUCKETS
            );
            assert_eq!(
                point
                    .attributes()
                    .next()
                    .expect("operation attribute")
                    .value
                    .as_str(),
                "generate"
            );
        }

        let tokens = by_name[GENERATION_TOKENS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(tokens)) = tokens.data() else {
            panic!("generation tokens must be a u64 counter")
        };
        let token_values: HashMap<_, _> = tokens
            .data_points()
            .map(|point| {
                let kind = point
                    .attributes()
                    .find(|attribute| attribute.key.as_str() == "token.kind")
                    .expect("token.kind")
                    .value
                    .as_str()
                    .into_owned();
                (kind, point.value())
            })
            .collect();
        assert_eq!(
            token_values,
            HashMap::from([("prompt".to_string(), 11), ("completion".to_string(), 7)])
        );
    }

    #[test]
    fn generation_dimensions_are_finite() {
        assert_eq!(
            [
                GenerationEvent::Cancellation,
                GenerationEvent::Chunk,
                GenerationEvent::Nak,
            ]
            .map(GenerationEvent::as_str),
            ["cancellation", "chunk", "nak"]
        );
        assert_eq!(
            [
                GenerationEventReason::BeforeFirstChunk,
                GenerationEventReason::MidStream,
                GenerationEventReason::InvalidKind,
                GenerationEventReason::NonFiniteTtft,
                GenerationEventReason::InvalidFinishReason,
                GenerationEventReason::SequenceGap,
                GenerationEventReason::StaleAttempt,
                GenerationEventReason::Duplicate,
            ]
            .map(GenerationEventReason::as_str),
            [
                "before_first_chunk",
                "mid_stream",
                "invalid_kind",
                "non_finite_ttft",
                "invalid_finish_reason",
                "sequence_gap",
                "stale_attempt",
                "duplicate",
            ]
        );
        assert_eq!(
            [
                GenerationEventOutcome::Cancelled,
                GenerationEventOutcome::Dropped,
                GenerationEventOutcome::Rejected,
            ]
            .map(GenerationEventOutcome::as_str),
            ["cancelled", "dropped", "rejected"]
        );
    }

    #[test]
    fn operational_dimensions_are_finite() {
        assert_eq!(
            [
                ConfigOperation::Bootstrap,
                ConfigOperation::Reconcile,
                ConfigOperation::DeltaModel,
                ConfigOperation::DeltaEpoch,
            ]
            .map(ConfigOperation::as_str),
            ["bootstrap", "reconcile", "delta_model", "delta_epoch"]
        );
        assert_eq!(
            [
                ConfigOutcome::Success,
                ConfigOutcome::ClientError,
                ConfigOutcome::FetchError,
                ConfigOutcome::PartialApply,
                ConfigOutcome::ParseError,
                ConfigOutcome::ApplyError,
                ConfigOutcome::RejectedUntrusted,
            ]
            .map(ConfigOutcome::as_str),
            [
                "success",
                "client_error",
                "fetch_error",
                "partial_apply",
                "parse_error",
                "apply_error",
                "rejected_untrusted",
            ]
        );
        assert_eq!(
            [
                QueuePublishOutcome::Submitted,
                QueuePublishOutcome::Backpressure,
                QueuePublishOutcome::NoConsumers,
                QueuePublishOutcome::Error,
            ]
            .map(QueuePublishOutcome::as_str),
            ["submitted", "backpressure", "no_consumers", "error"]
        );
        assert_eq!(
            QueuePublishOutcome::from_error("NATS: NO CONSUMERS"),
            QueuePublishOutcome::NoConsumers
        );
        assert_eq!(
            [
                QueueEventOutcome::Success,
                QueueEventOutcome::AckError,
                QueueEventOutcome::Error,
                QueueEventOutcome::Deduplicated,
            ]
            .map(QueueEventOutcome::as_str),
            ["success", "ack_error", "error", "deduplicated"]
        );
        assert_eq!(
            [
                QueueResultChunkRejectionReason::Kind,
                QueueResultChunkRejectionReason::Identity,
                QueueResultChunkRejectionReason::Digest,
                QueueResultChunkRejectionReason::ChunkCount,
                QueueResultChunkRejectionReason::ChunkIndex,
                QueueResultChunkRejectionReason::ItemSize,
                QueueResultChunkRejectionReason::PayloadSize,
                QueueResultChunkRejectionReason::AggregateSize,
                QueueResultChunkRejectionReason::GlobalBudget,
                QueueResultChunkRejectionReason::MetadataConflict,
                QueueResultChunkRejectionReason::DuplicateConflict,
                QueueResultChunkRejectionReason::TotalMismatch,
                QueueResultChunkRejectionReason::DigestMismatch,
                QueueResultChunkRejectionReason::Decode,
            ]
            .map(QueueResultChunkRejectionReason::as_str),
            [
                "kind",
                "identity",
                "digest",
                "chunk_count",
                "chunk_index",
                "item_size",
                "payload_size",
                "aggregate_size",
                "global_budget",
                "metadata_conflict",
                "duplicate_conflict",
                "total_mismatch",
                "digest_mismatch",
                "decode",
            ]
        );
    }

    #[test]
    fn disabled_operational_facades_return_before_instrumentation() {
        assert!(!set_config_applied_epoch_to(None, 42));
        assert!(!record_queue_publish_to(
            None,
            &QueuePublishObservation {
                operation: "encode",
                outcome: QueuePublishOutcome::Submitted,
                duration: Duration::ZERO,
                items: 1,
            }
        ));
        assert!(!record_queue_result_chunk_received_to(None, Some(128)));
        assert!(!record_queue_result_chunk_rejection_to(
            None,
            QueueResultChunkRejectionReason::Decode,
        ));
        assert!(!record_queue_result_chunk_event_to(
            None,
            QueueResultChunkEvent::TransferCompleted,
        ));
        assert!(!record_queue_result_chunk_reservation_change_to(
            None,
            QueueResultChunkReservationChange::Reserved(4_096),
        ));
        assert!(!record_provisioning_response_to(
            None,
            ProvisioningSurface::Native,
            503,
        ));
        assert!(!record_generation_event_to(
            None,
            GenerationEvent::Chunk,
            GenerationEventReason::SequenceGap,
            GenerationEventOutcome::Rejected,
        ));
        assert!(!record_generation_completion_to(
            None,
            GenerationCompletionObservation::default(),
        ));
    }

    #[test]
    fn dispatch_dimensions_are_finite_and_manifest_bounded() {
        assert_eq!(
            [DispatchPath::I6pn, DispatchPath::Modal].map(DispatchPath::as_str),
            ["i6pn", "modal"]
        );
        assert_eq!(
            [
                DispatchOutcome::Success,
                DispatchOutcome::Error,
                DispatchOutcome::Timeout,
                DispatchOutcome::Cancelled,
            ]
            .map(DispatchOutcome::as_str),
            ["success", "error", "timeout", "cancelled"]
        );
        assert_eq!(
            [
                FallbackReason::None,
                FallbackReason::Disabled,
                FallbackReason::NotApplicable,
                FallbackReason::UnknownLane,
                FallbackReason::NoLiveChannel,
                FallbackReason::NoCredit,
                FallbackReason::LaneError,
                FallbackReason::Timeout,
                FallbackReason::TransportFailure,
            ]
            .map(FallbackReason::as_str),
            [
                "none",
                "disabled",
                "not_applicable",
                "unknown_lane",
                "no_live_channel",
                "no_credit",
                "lane_error",
                "timeout",
                "transport_failure",
            ]
        );

        let allowed = parse_manifest_lanes(Some(
            r#"{
                "default|l4|default":"sie-cloud-lanes",
                "default|l4|BAAI/bge-m3":"sie-cloud-lanes"
            }"#,
        ));
        assert_eq!(
            bounded_lane("default|l4|BAAI/bge-m3", &allowed),
            "default|l4|BAAI/bge-m3"
        );
        assert_eq!(
            bounded_lane("default|l4|intfloat/e5-base-v2", &allowed),
            "default|l4|default",
            "unpinned catalog models use the shared lazy lane"
        );
        assert_eq!(
            bounded_lane("customer-123|h100|worker-456", &allowed),
            "other"
        );
        assert!(parse_manifest_lanes(Some("not-json")).is_empty());

        let attributes = dispatch_attributes_with_lanes(
            &DispatchObservation {
                operation: "caller-defined-operation",
                path: DispatchPath::Modal,
                outcome: DispatchOutcome::Error,
                fallback_reason: FallbackReason::NoCredit,
                lane: "customer-123|h100|worker-456",
                duration: Duration::ZERO,
            },
            &allowed,
        );
        assert_eq!(attributes[0].value.as_str(), "other");
        assert_eq!(attributes[4].value.as_str(), "other");
    }

    #[test]
    fn disabled_dispatch_facade_returns_before_instrumentation() {
        assert!(!record_dispatch_to(
            None,
            &DispatchObservation {
                operation: "encode",
                path: DispatchPath::Modal,
                outcome: DispatchOutcome::Success,
                fallback_reason: FallbackReason::NotApplicable,
                lane: "default|l4|default",
                duration: Duration::from_millis(1),
            },
        ));
    }

    #[test]
    fn request_and_keda_contract_exports_exact_names_units_labels_and_freshness() {
        let (telemetry, exporter, provider) = metric_points();
        telemetry.request_completed("encode", 200, "l4-spot", 0.025, AdmissionOutcome::Admitted);
        let lane = |value| LaneSnapshot {
            pool: "DEFAULT".to_string(),
            machine_profile: "L4-SPOT".to_string(),
            bundle: "DEFAULT".to_string(),
            value,
        };
        let initial = KedaCapacitySnapshot {
            pending_demand: vec![lane(1.0)],
            lane_queue_depth: vec![lane(3.0)],
            active_lease_gpus: vec![lane(1.0)],
            pool_warm_floor: vec![lane(2.0)],
        };
        assert!(record_keda_capacity_snapshot_to(
            Some(&telemetry),
            &initial,
            1_721_177_600.0,
        ));
        let rejection_lane = PhysicalLane::try_new("DEFAULT", "L4-SPOT", "DEFAULT").unwrap();
        telemetry.record_rejected_request(&rejection_lane, "backpressure");
        assert!(record_keda_capacity_snapshot_to(
            Some(&telemetry),
            &KedaCapacitySnapshot::default(),
            1_721_177_601.0,
        ));

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
                REQUESTS_METRIC_NAME.to_string(),
                REQUEST_DURATION_METRIC_NAME.to_string(),
                ADMISSION_DECISIONS_METRIC_NAME.to_string(),
                PENDING_DEMAND_METRIC_NAME.to_string(),
                LANE_QUEUE_DEPTH_METRIC_NAME.to_string(),
                LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME.to_string(),
                ACTIVE_LEASE_GPUS_METRIC_NAME.to_string(),
                POOL_WARM_FLOOR_METRIC_NAME.to_string(),
                REJECTED_REQUESTS_METRIC_NAME.to_string(),
                CAPACITY_SNAPSHOT_TIMESTAMP_METRIC_NAME.to_string(),
            ])
        );

        let by_name: HashMap<_, _> = exported
            .iter()
            .map(|metric| (metric.name().to_string(), *metric))
            .collect();
        for (name, unit) in [
            (REQUESTS_METRIC_NAME, "{request}"),
            (REQUEST_DURATION_METRIC_NAME, "s"),
            (ADMISSION_DECISIONS_METRIC_NAME, "{request}"),
            (PENDING_DEMAND_METRIC_NAME, "{request}"),
            (LANE_QUEUE_DEPTH_METRIC_NAME, "{item}"),
            (LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME, "s"),
            (ACTIVE_LEASE_GPUS_METRIC_NAME, "{gpu}"),
            (POOL_WARM_FLOOR_METRIC_NAME, "{worker}"),
            (REJECTED_REQUESTS_METRIC_NAME, "{request}"),
            (CAPACITY_SNAPSHOT_TIMESTAMP_METRIC_NAME, "s"),
        ] {
            assert_eq!(by_name[name].unit(), unit, "unit drift for {name}");
        }

        let queue = by_name[LANE_QUEUE_DEPTH_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Gauge(queue)) = queue.data() else {
            panic!("lane queue depth must be an f64 gauge")
        };
        let queue_point = queue.data_points().next().expect("queue point");
        assert_eq!(
            queue_point.value(),
            3.0,
            "an omitted broker lane must retain its last value until KEDA freshness expires it"
        );
        let queue_attributes: HashMap<_, _> = queue_point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            queue_attributes,
            HashMap::from([
                ("pool", "default".to_string()),
                ("machine_profile", "l4-spot".to_string()),
                ("bundle", "default".to_string()),
            ])
        );

        let queue_timestamp = by_name[LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Gauge(queue_timestamp)) = queue_timestamp.data()
        else {
            panic!("lane queue snapshot timestamp must be an f64 gauge")
        };
        let queue_timestamp_point = queue_timestamp
            .data_points()
            .next()
            .expect("lane queue snapshot timestamp point");
        assert_eq!(queue_timestamp_point.value(), 1_721_177_600.0);
        assert_eq!(
            queue_timestamp_point
                .attributes()
                .map(|attribute| (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                ))
                .collect::<HashMap<_, _>>(),
            HashMap::from([
                ("pool", "default".to_string()),
                ("machine_profile", "l4-spot".to_string()),
                ("bundle", "default".to_string()),
            ])
        );

        for name in [
            PENDING_DEMAND_METRIC_NAME,
            ACTIVE_LEASE_GPUS_METRIC_NAME,
            POOL_WARM_FLOOR_METRIC_NAME,
        ] {
            let metric = by_name[name];
            let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = metric.data() else {
                panic!("{name} must be an f64 gauge")
            };
            let point = gauge.data_points().next().expect("KEDA lane point");
            assert_eq!(point.value(), 0.0, "{name} must retain explicit zero");
            let attributes: HashMap<_, _> = point
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            assert_eq!(
                attributes,
                HashMap::from([
                    ("pool", "default".to_string()),
                    ("machine_profile", "l4-spot".to_string()),
                    ("bundle", "default".to_string()),
                ]),
                "KEDA label drift for {name}"
            );
        }

        let rejected = by_name[REJECTED_REQUESTS_METRIC_NAME];
        let AggregatedMetrics::U64(MetricData::Sum(rejected)) = rejected.data() else {
            panic!("rejections must be a u64 counter")
        };
        let rejected_attributes: HashMap<_, _> = rejected
            .data_points()
            .next()
            .expect("rejection point")
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            rejected_attributes.get("reason").map(String::as_str),
            Some("backpressure")
        );
        assert_eq!(
            rejected_attributes
                .get("scaling_action")
                .map(String::as_str),
            Some("scale_up")
        );
        assert_eq!(
            rejected_attributes.get("pool").map(String::as_str),
            Some("default")
        );
        assert_eq!(
            rejected_attributes
                .get("machine_profile")
                .map(String::as_str),
            Some("l4-spot")
        );
        assert_eq!(
            rejected_attributes.get("bundle").map(String::as_str),
            Some("default")
        );
        assert_eq!(rejected_attributes.len(), 5);

        let snapshot_timestamp = by_name[CAPACITY_SNAPSHOT_TIMESTAMP_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Gauge(snapshot_timestamp)) =
            snapshot_timestamp.data()
        else {
            panic!("capacity snapshot timestamp must be an f64 gauge")
        };
        let snapshot_point = snapshot_timestamp
            .data_points()
            .next()
            .expect("capacity snapshot point");
        assert_eq!(snapshot_point.value(), 1_721_177_601.0);
        assert_eq!(snapshot_point.attributes().count(), 0);

        let duration = by_name[REQUEST_DURATION_METRIC_NAME];
        let AggregatedMetrics::F64(MetricData::Histogram(duration)) = duration.data() else {
            panic!("request duration must be a histogram")
        };
        let duration_point = duration.data_points().next().expect("duration point");
        assert_eq!(duration_point.count(), 1);
        assert_eq!(duration_point.sum(), 0.025);
        assert_eq!(
            duration_point.bounds().collect::<Vec<_>>(),
            REQUEST_LATENCY_BUCKETS
        );
    }

    #[test]
    fn full_keda_rejection_domain_never_uses_the_otel_overflow_series() {
        let (telemetry, exporter, provider) = metric_points();
        let scale_up_reasons = [
            "backpressure",
            "no_consumers",
            "publish_ack_failed",
            "upstream_result_timeout",
        ];

        for index in 0..crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES {
            let lane = PhysicalLane::try_new(&format!("pool-{index}"), "l4", "default").unwrap();
            for reason in scale_up_reasons {
                telemetry.record_rejected_request(&lane, reason);
            }
            // Non-scaling and unknown reasons are deliberately absent from
            // this control stream; request counters/logs retain the outcome.
            telemetry.record_rejected_request(&lane, "resource_exhausted");
            telemetry.record_rejected_request(&lane, "future_reason");
        }

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let rejected = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .find(|metric| metric.name() == REJECTED_REQUESTS_METRIC_NAME)
            .expect("scale-worthy rejection counter");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = rejected.data() else {
            panic!("rejections must export as a u64 sum")
        };
        let points: Vec<_> = sum.data_points().collect();
        assert_eq!(
            points.len(),
            crate::observability::tracing::KEDA_REJECTED_REQUESTS_CARDINALITY_LIMIT,
        );
        assert!(
            points.iter().all(|point| point
                .attributes()
                .all(|attribute| attribute.key.as_str() != "otel.metric.overflow")),
            "the full valid catalog × reason domain must remain exact-label addressable"
        );
        assert!(points
            .iter()
            .all(|point| point.attributes().any(|attribute| {
                attribute.key.as_str() == "scaling_action"
                    && attribute.value.as_str().as_ref() == "scale_up"
            })));
    }

    #[test]
    fn rejection_facade_requires_configured_lane_and_scale_worthy_reason() {
        let (telemetry, exporter, provider) = metric_points();
        let configured_lane = PhysicalLane::try_new("default", "l4", "default").unwrap();
        let configured_catalog =
            crate::state::demand_tracker::PhysicalLaneCatalog::try_new([configured_lane.clone()])
                .unwrap();
        let demand_tracker = DemandTracker::new(configured_catalog);
        let foreign_lane = PhysicalLane::try_new("attacker", "l4", "default").unwrap();

        assert!(record_rejected_request_to(
            Some(&telemetry),
            &demand_tracker,
            &configured_lane,
            "backpressure",
        ));
        assert!(!record_rejected_request_to(
            Some(&telemetry),
            &demand_tracker,
            &foreign_lane,
            "backpressure",
        ));
        assert!(!record_rejected_request_to(
            Some(&telemetry),
            &demand_tracker,
            &configured_lane,
            "resource_exhausted",
        ));

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let rejected = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .find(|metric| metric.name() == REJECTED_REQUESTS_METRIC_NAME)
            .expect("configured scale-worthy rejection");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = rejected.data() else {
            panic!("rejections must export as a u64 sum")
        };
        assert_eq!(sum.data_points().count(), 1);
    }

    #[test]
    fn full_keda_lane_gauge_domain_never_uses_the_otel_overflow_series() {
        let (telemetry, exporter, provider) = metric_points();
        let lanes: Vec<_> = (0..crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES)
            .map(|index| LaneSnapshot {
                pool: format!("pool-{index}"),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: (index + 1) as f64,
            })
            .collect();
        let snapshot = KedaCapacitySnapshot {
            pending_demand: lanes.clone(),
            lane_queue_depth: lanes.clone(),
            active_lease_gpus: lanes.clone(),
            pool_warm_floor: lanes,
        };

        assert!(record_keda_capacity_snapshot_to(
            Some(&telemetry),
            &snapshot,
            1_721_177_600.0,
        ));
        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");

        for name in [
            PENDING_DEMAND_METRIC_NAME,
            LANE_QUEUE_DEPTH_METRIC_NAME,
            LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME,
            ACTIVE_LEASE_GPUS_METRIC_NAME,
            POOL_WARM_FLOOR_METRIC_NAME,
        ] {
            let metric = resource_metrics
                .iter()
                .flat_map(|resource| resource.scope_metrics())
                .flat_map(|scope| scope.metrics())
                .find(|metric| metric.name() == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let AggregatedMetrics::F64(MetricData::Gauge(gauge)) = metric.data() else {
                panic!("{name} must export as an f64 gauge")
            };
            let points: Vec<_> = gauge.data_points().collect();
            assert_eq!(
                points.len(),
                crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES,
                "{name} must retain every configured lane",
            );
            assert!(
                points.iter().all(|point| {
                    point
                        .attributes()
                        .all(|attribute| attribute.key.as_str() != "otel.metric.overflow")
                }),
                "{name} must keep every lane exact-label addressable",
            );
            let high_lane = points
                .iter()
                .find(|point| {
                    point.attributes().any(|attribute| {
                        attribute.key.as_str() == "pool"
                            && attribute.value.as_str().as_ref() == "pool-1023"
                    })
                })
                .unwrap_or_else(|| panic!("{name} is missing the highest configured lane"));
            let expected_value = if name == LANE_QUEUE_SNAPSHOT_TIMESTAMP_METRIC_NAME {
                1_721_177_600.0
            } else {
                1024.0
            };
            assert_eq!(
                high_lane.value(),
                expected_value,
                "{name} high-lane value drift"
            );
            let high_attributes: HashMap<_, _> = high_lane
                .attributes()
                .map(|attribute| {
                    (
                        attribute.key.as_str(),
                        attribute.value.as_str().into_owned(),
                    )
                })
                .collect();
            assert_eq!(
                high_attributes,
                HashMap::from([
                    ("pool", "pool-1023".to_string()),
                    ("machine_profile", "l4".to_string()),
                    ("bundle", "default".to_string()),
                ]),
                "{name} must preserve the exact high-lane tuple",
            );
        }
    }

    #[test]
    fn pinned_model_snapshot_exports_exact_labels_and_clears_removed_lanes() {
        let (telemetry, exporter, provider) = metric_points();
        telemetry.set_pool_pinned_model_loaded_snapshot(&[PinnedModelSnapshot {
            pool: "tenant-a".to_string(),
            model: "BAAI/bge-m3".to_string(),
            loaded: true,
        }]);
        telemetry.set_pool_pinned_model_loaded_snapshot(&[]);

        provider.force_flush().expect("force_flush");
        let resource_metrics = exporter.get_finished_metrics().expect("finished metrics");
        let metric = resource_metrics
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .find(|metric| metric.name() == POOL_PINNED_MODEL_LOADED_METRIC_NAME)
            .expect("pinned-model metric");
        assert_eq!(metric.unit(), "1");
        let AggregatedMetrics::U64(MetricData::Gauge(gauge)) = metric.data() else {
            panic!("pinned-model readiness must be a u64 gauge")
        };
        let point = gauge.data_points().next().expect("pinned-model point");
        assert_eq!(point.value(), 0, "removed pins must be explicitly zeroed");
        let attributes: HashMap<_, _> = point
            .attributes()
            .map(|attribute| {
                (
                    attribute.key.as_str(),
                    attribute.value.as_str().into_owned(),
                )
            })
            .collect();
        assert_eq!(
            attributes,
            HashMap::from([
                ("pool", "tenant-a".to_string()),
                ("model", "BAAI/bge-m3".to_string()),
            ])
        );
        let state = telemetry
            .pinned_model_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(state.current.is_empty());
        assert!(
            state.removed.is_empty(),
            "one collection must release removed logical-pool dimensions"
        );
    }

    #[derive(Default)]
    struct RecordingKedaSink {
        calls: Mutex<Vec<&'static str>>,
    }

    impl RecordingKedaSink {
        fn push(&self, call: &'static str) {
            self.calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(call);
        }
    }

    impl KedaSnapshotSink for RecordingKedaSink {
        fn record_pending_demand(&self, _values: &[LaneSnapshot]) {
            self.push("pending_demand");
        }

        fn record_lane_queue_depth(&self, _values: &[LaneSnapshot]) {
            self.push("lane_queue_depth");
        }

        fn record_lane_queue_timestamp(&self, _values: &[LaneSnapshot], _unix_time_s: f64) {
            self.push("lane_queue_timestamp");
        }

        fn record_active_lease_gpus(&self, _values: &[LaneSnapshot]) {
            self.push("active_lease_gpus");
        }

        fn record_pool_warm_floor(&self, _values: &[LaneSnapshot]) {
            self.push("pool_warm_floor");
        }

        fn record_timestamp(&self, _unix_time_s: f64) {
            self.push("timestamp");
        }
    }

    #[test]
    fn capacity_timestamp_is_recorded_after_all_four_state_families() {
        let sink = RecordingKedaSink::default();
        assert!(record_keda_capacity_snapshot_to(
            Some(&sink),
            &KedaCapacitySnapshot::default(),
            1.0,
        ));
        assert_eq!(
            *sink
                .calls
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            [
                "pending_demand",
                "lane_queue_depth",
                "lane_queue_timestamp",
                "active_lease_gpus",
                "pool_warm_floor",
                "timestamp",
            ]
        );
    }

    #[test]
    fn disabled_capacity_snapshot_does_not_touch_instruments() {
        let snapshot = KedaCapacitySnapshot {
            pending_demand: vec![LaneSnapshot {
                pool: "default".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: 1.0,
            }],
            ..Default::default()
        };
        assert!(!record_keda_capacity_snapshot_to(None, &snapshot, 1.0));
    }

    #[test]
    fn failed_queue_lane_retains_value_without_a_producer_refresh() {
        let (telemetry, exporter, provider) = metric_points();
        let lane = LaneSnapshot {
            pool: "default".to_string(),
            machine_profile: "l4".to_string(),
            bundle: "default".to_string(),
            value: 7.0,
        };
        let queue_value = |exporter: &InMemoryMetricExporter| {
            exporter
                .get_finished_metrics()
                .expect("finished metrics")
                .iter()
                .flat_map(|resource| resource.scope_metrics())
                .flat_map(|scope| scope.metrics())
                .find(|metric| metric.name() == LANE_QUEUE_DEPTH_METRIC_NAME)
                .and_then(|metric| match metric.data() {
                    AggregatedMetrics::F64(MetricData::Gauge(gauge)) => {
                        gauge.data_points().next().map(|point| point.value())
                    }
                    _ => None,
                })
                .expect("lane queue gauge point")
        };

        telemetry.set_lane_queue_snapshot(&[lane]);
        provider.force_flush().expect("positive flush");
        assert_eq!(queue_value(&exporter), 7.0);
        exporter.reset();

        telemetry.set_lane_queue_snapshot(&[]);
        provider.force_flush().expect("partial snapshot flush");
        assert_eq!(queue_value(&exporter), 7.0);
        exporter.reset();

        // No producer write occurs between these collections: synchronous
        // last-value aggregation retains the last queue value. Its independent
        // lane freshness timestamp is deliberately not refreshed, so KEDA
        // rejects it rather than interpreting a read failure as zero.
        provider
            .force_flush()
            .expect("subsequent retained-value flush");
        assert_eq!(queue_value(&exporter), 7.0);
    }

    #[test]
    fn labels_and_rejection_domains_are_bounded() {
        assert_eq!(sanitize_label("default"), "default");
        assert_eq!(sanitize_label("customer/value"), "other");
        assert_eq!(sanitize_label(""), "other");
        assert_eq!(sanitize_label(&"p".repeat(128)), "p".repeat(128));
        assert_eq!(bounded_operation("caller-controlled"), "other");
        assert_eq!(
            scale_up_rejection_reason("backpressure"),
            Some("backpressure")
        );
        assert_eq!(scale_up_rejection_reason("resource_exhausted"), None);
        assert_eq!(scale_up_rejection_reason("future_reason"), None);
    }

    /// Reproducible release microbenchmark for emitting a pre-built worst-case
    /// 1,024-lane capacity value through the facade. Snapshot construction and
    /// exporter flush are intentionally excluded and benchmarked separately in
    /// `server`; this reports raw wall-clock time only and does not instrument
    /// allocator traffic.
    ///
    /// One invocation collects three independently warmed samples:
    /// `mise exec -- cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib keda_capacity_emit_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    fn keda_capacity_emit_microbenchmark() {
        const SAMPLES: usize = 3;
        const DISABLED_WARMUP_ITERATIONS: usize = 10_000;
        const ENABLED_WARMUP_ITERATIONS: usize = 2;
        const DISABLED_ITERATIONS: usize = 500_000;
        const ENABLED_ITERATIONS: usize = 25;

        let lanes: Vec<_> = (0..crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES)
            .map(|index| LaneSnapshot {
                pool: format!("pool-{index}"),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: (index + 1) as f64,
            })
            .collect();
        let snapshot = KedaCapacitySnapshot {
            pending_demand: lanes.clone(),
            lane_queue_depth: lanes.clone(),
            active_lease_gpus: lanes.clone(),
            pool_warm_floor: lanes,
        };
        let (telemetry, _exporter, provider) = metric_points();

        assert!(!super::super::tracing::metrics_exporter_enabled());
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            for _ in 0..DISABLED_WARMUP_ITERATIONS {
                record_keda_capacity_snapshot(black_box(&snapshot), black_box(1_721_177_600.0));
            }
            for _ in 0..ENABLED_WARMUP_ITERATIONS {
                let _ = record_keda_capacity_snapshot_to(
                    Some(&telemetry),
                    black_box(&snapshot),
                    black_box(1_721_177_600.0),
                );
            }

            let disabled_started = Instant::now();
            for _ in 0..DISABLED_ITERATIONS {
                record_keda_capacity_snapshot(black_box(&snapshot), black_box(1_721_177_600.0));
            }
            disabled_samples[sample_index] =
                disabled_started.elapsed().as_nanos() as f64 / DISABLED_ITERATIONS as f64;

            let enabled_started = Instant::now();
            for iteration in 0..ENABLED_ITERATIONS {
                let _ = record_keda_capacity_snapshot_to(
                    Some(&telemetry),
                    black_box(&snapshot),
                    black_box(1_721_177_600.0 + iteration as f64),
                );
            }
            enabled_samples[sample_index] =
                enabled_started.elapsed().as_secs_f64() * 1_000.0 / ENABLED_ITERATIONS as f64;
        }

        let disabled_median_ns = telemetry_benchmark_median(disabled_samples);
        let enabled_median_ms = telemetry_benchmark_median(enabled_samples);
        println!(
            "gateway_keda_capacity_emit lanes={} gauge_points_per_emit={} samples={SAMPLES} disabled_public_ns_per_emit={disabled_samples:?} disabled_median_ns_per_emit={disabled_median_ns:.2} enabled_instrumented_ms_per_emit={enabled_samples:?} enabled_median_ms_per_emit={enabled_median_ms:.3} snapshot_build=excluded exporter_flush=excluded allocation_measurement=not_instrumented",
            crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES,
            crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES * 4,
        );
        let disabled_budget = telemetry_performance_budget("gateway_keda_disabled_ns_per_snapshot");
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway telemetry-disabled KEDA snapshot median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let budget = telemetry_performance_budget("gateway_keda_1024_lane_emit_ms_per_snapshot");
        assert!(
            enabled_median_ms <= budget,
            "gateway 1,024-lane KEDA emit median {enabled_median_ms:.3} ms exceeded {budget:.3} ms budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }

    /// Reproducible release microbenchmark for the exact request-completion
    /// facade. One invocation collects three independently warmed samples:
    /// `cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib request_completion_hot_path_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    fn request_completion_hot_path_microbenchmark() {
        const SAMPLES: usize = 3;
        const WARMUP: usize = 20_000;
        const ITERATIONS: usize = 200_000;

        let (telemetry, _exporter, provider) = metric_points();
        let exercise_enabled = |target: &GatewayTelemetry, iterations: usize| {
            for index in 0..iterations {
                record_request_completed_to(
                    Some(target),
                    None,
                    black_box("encode"),
                    black_box(200),
                    black_box("l4-spot"),
                    black_box((index % 100) as f64 / 1_000.0),
                    black_box(AdmissionOutcome::Admitted),
                );
            }
        };
        let exercise_disabled = |iterations: usize| {
            for index in 0..iterations {
                record_request_completed(
                    None,
                    black_box("encode"),
                    black_box(200),
                    black_box("l4-spot"),
                    black_box((index % 100) as f64 / 1_000.0),
                    black_box(AdmissionOutcome::Admitted),
                );
            }
        };

        assert!(!super::super::tracing::metrics_exporter_enabled());
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise_disabled(WARMUP);
            exercise_enabled(&telemetry, WARMUP);

            let disabled_started = Instant::now();
            exercise_disabled(ITERATIONS);
            disabled_samples[sample_index] =
                disabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let enabled_started = Instant::now();
            exercise_enabled(&telemetry, ITERATIONS);
            enabled_samples[sample_index] =
                enabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        }

        let disabled_median_ns = telemetry_benchmark_median(disabled_samples);
        let enabled_median_ns = telemetry_benchmark_median(enabled_samples);
        let incremental_median_ns = (enabled_median_ns - disabled_median_ns).max(0.0);
        println!(
            "gateway_request_completion iterations_per_sample={ITERATIONS} samples={SAMPLES} public_disabled_ns_per_op={disabled_samples:?} public_disabled_median_ns_per_op={disabled_median_ns:.2} enabled_instrumented_ns_per_op={enabled_samples:?} enabled_instrumented_median_ns_per_op={enabled_median_ns:.2} median_incremental_ns_per_op={:.2}",
            incremental_median_ns
        );
        let disabled_budget =
            telemetry_performance_budget("gateway_request_completion_disabled_ns_per_event");
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway telemetry-disabled request completion median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let budget =
            telemetry_performance_budget("gateway_request_completion_enabled_ns_per_event");
        assert!(
            enabled_median_ns <= budget,
            "gateway request completion median {enabled_median_ns:.2} ns exceeded {budget:.2} ns budget"
        );
        let incremental_budget =
            telemetry_performance_budget("gateway_request_completion_incremental_ns_per_event");
        assert!(
            incremental_median_ns <= incremental_budget,
            "gateway incremental request completion median {incremental_median_ns:.2} ns exceeded {incremental_budget:.2} ns budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }

    /// Reproducible release microbenchmark for the steady-state per-frame
    /// result-chunk facade. One item may contain up to 64 frames, so this gate
    /// measures the combined received-count and payload-byte observation:
    /// `cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib result_chunk_hot_path_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    fn result_chunk_hot_path_microbenchmark() {
        const SAMPLES: usize = 3;
        const WARMUP: usize = 20_000;
        const ITERATIONS: usize = 200_000;

        let (telemetry, _exporter, provider) = metric_points();
        let exercise_enabled = |target: &GatewayTelemetry, iterations: usize| {
            for _ in 0..iterations {
                let _ = record_queue_result_chunk_received_to(
                    Some(black_box(target)),
                    Some(black_box(64 * 1_024)),
                );
            }
        };
        let exercise_disabled = |iterations: usize| {
            for _ in 0..iterations {
                record_queue_result_chunk_received(Some(black_box(64 * 1_024)));
            }
        };

        assert!(!super::super::tracing::metrics_exporter_enabled());
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise_disabled(WARMUP);
            exercise_enabled(&telemetry, WARMUP);

            let disabled_started = Instant::now();
            exercise_disabled(ITERATIONS);
            disabled_samples[sample_index] =
                disabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let enabled_started = Instant::now();
            exercise_enabled(&telemetry, ITERATIONS);
            enabled_samples[sample_index] =
                enabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        }

        let disabled_median_ns = telemetry_benchmark_median(disabled_samples);
        let enabled_median_ns = telemetry_benchmark_median(enabled_samples);
        let incremental_median_ns = (enabled_median_ns - disabled_median_ns).max(0.0);
        println!(
            "gateway_result_chunk iterations_per_sample={ITERATIONS} samples={SAMPLES} public_disabled_ns_per_frame={disabled_samples:?} public_disabled_median_ns_per_frame={disabled_median_ns:.2} enabled_instrumented_ns_per_frame={enabled_samples:?} enabled_instrumented_median_ns_per_frame={enabled_median_ns:.2} median_incremental_ns_per_frame={:.2}",
            incremental_median_ns
        );
        let disabled_budget =
            telemetry_performance_budget("gateway_result_chunk_disabled_ns_per_event");
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway telemetry-disabled result chunk median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let budget = telemetry_performance_budget("gateway_result_chunk_enabled_ns_per_event");
        assert!(
            enabled_median_ns <= budget,
            "gateway result chunk median {enabled_median_ns:.2} ns exceeded {budget:.2} ns budget"
        );
        let incremental_budget =
            telemetry_performance_budget("gateway_result_chunk_incremental_ns_per_event");
        assert!(
            incremental_median_ns <= incremental_budget,
            "gateway incremental result chunk median {incremental_median_ns:.2} ns exceeded {incremental_budget:.2} ns budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }

    /// Reproducible release microbenchmark for the shared managed-dispatch
    /// facade. One invocation collects three independently warmed samples:
    /// `SIE_LANE_APP_MAP='{"default|l4|default":"benchmark"}' mise exec -- cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib dispatch_completion_hot_path_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    fn dispatch_completion_hot_path_microbenchmark() {
        const SAMPLES: usize = 3;
        const WARMUP: usize = 20_000;
        const ITERATIONS: usize = 200_000;

        let (telemetry, _exporter, provider) = metric_points();
        let observation = DispatchObservation {
            operation: "encode",
            path: DispatchPath::I6pn,
            outcome: DispatchOutcome::Success,
            fallback_reason: FallbackReason::None,
            lane: "default|l4|default",
            duration: Duration::from_millis(25),
        };
        let exercise_enabled = |target: &GatewayTelemetry, iterations: usize| {
            for _ in 0..iterations {
                let _ = record_dispatch_to(Some(black_box(target)), black_box(&observation));
            }
        };
        let exercise_disabled = |iterations: usize| {
            for _ in 0..iterations {
                record_dispatch(black_box(observation));
            }
        };

        assert!(!super::super::tracing::metrics_exporter_enabled());
        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise_disabled(WARMUP);
            exercise_enabled(&telemetry, WARMUP);

            let disabled_started = Instant::now();
            exercise_disabled(ITERATIONS);
            disabled_samples[sample_index] =
                disabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let enabled_started = Instant::now();
            exercise_enabled(&telemetry, ITERATIONS);
            enabled_samples[sample_index] =
                enabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        }

        let disabled_median_ns = telemetry_benchmark_median(disabled_samples);
        let enabled_median_ns = telemetry_benchmark_median(enabled_samples);
        let incremental_median_ns = (enabled_median_ns - disabled_median_ns).max(0.0);
        println!(
            "gateway_dispatch_completion iterations_per_sample={ITERATIONS} samples={SAMPLES} public_disabled_ns_per_op={disabled_samples:?} public_disabled_median_ns_per_op={disabled_median_ns:.2} enabled_instrumented_ns_per_op={enabled_samples:?} enabled_instrumented_median_ns_per_op={enabled_median_ns:.2} median_incremental_ns_per_op={:.2}",
            incremental_median_ns
        );
        let disabled_budget =
            telemetry_performance_budget("gateway_dispatch_completion_disabled_ns_per_event");
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway telemetry-disabled dispatch completion median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let budget =
            telemetry_performance_budget("gateway_dispatch_completion_enabled_ns_per_event");
        assert!(
            enabled_median_ns <= budget,
            "gateway dispatch completion median {enabled_median_ns:.2} ns exceeded {budget:.2} ns budget"
        );
        let incremental_budget =
            telemetry_performance_budget("gateway_dispatch_completion_incremental_ns_per_event");
        assert!(
            incremental_median_ns <= incremental_budget,
            "gateway incremental dispatch completion median {incremental_median_ns:.2} ns exceeded {incremental_budget:.2} ns budget"
        );
        provider.shutdown().expect("shutdown benchmark provider");
    }
}
