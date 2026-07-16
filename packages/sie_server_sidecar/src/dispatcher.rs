//! Per-batch dispatcher: decode NATS JetStream messages into WorkItems,
//! fan out to the backend over IPC, publish results, and
//! ACK/NAK each message.
//!
//!   fetch -> decode + validate (subject, reply_subject, model_id)
//!         -> group by model_id
//!         -> per model (concurrent, capped by batch_semaphore):
//!              EnsureModelReady
//!              apply per-model batch_budget (overflow -> NAK fast)
//!              fan out encode/score/extract concurrently:
//!                 resolve payload -> IPC ProcessXxxBatch -> publish + ACK
//!                 on IPC failure -> NAK group with transient delay

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use async_nats::jetstream::Message;
use futures_util::future::join_all;
use rmpv::Value as MsgValue;
use serde_json::Value as Json;
use thiserror::Error;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::backend::{AdapterWorkerPool, BackendError, SharedBackend};
use crate::batch_cancel::BatchCancelState;
use crate::config_subscriber::ConfigApplyState;
use crate::delivery::Delivery;
use crate::ipc_client::IpcError;
use crate::ipc_types::{
    BatchOutcome, Disposition, EncodeBatchItem, ExtractBatchItem, GenerateEvent, ItemOutcome,
    PreparedTokens, ProcessEncodeBatchRequest, ProcessExtractBatchRequest, ProcessGenerateRequest,
    ProcessScoreBatchRequest, ReadinessState, RunBatchRequest, ScoreBatchItem,
};
use crate::latency::LatencyTracker;
use crate::log_util::ErrChain;
use crate::metrics::MetricsRegistry;
use crate::payload_store::{PayloadError, PayloadStore};
use crate::pool_admission::PoolAdmissionGate;
use crate::publisher::{shape_and_build_work_result, should_publish, Timings, WorkPublisher};
use crate::scheduler::{
    lora_from_options, HasCost, LoraKey, Op as SchedOp, ProductionScheduler,
    ProductionSchedulerRegistry, SchedulerItem, SchedulerMeta,
};
use crate::shutdown::Shutdown;
use crate::subject::{extract_model_id, is_worker_direct_work_subject};
use crate::tokenize::TokenizerRegistry;
use crate::work_types::WorkItem;

const MODEL_LOADING_ERROR_CODE: &str = "MODEL_LOADING";
/// Terminal, non-retryable model-load failure. Emitted on the `WorkResult`
/// (encode/score/extract) or the generation terminal chunk when the Python
/// executor reports [`ReadinessState::Failed`] (registry holds a PERMANENT
/// `LoadFailure`). The gateway maps this code to a typed HTTP 502 via
/// `build_model_load_failed_response` (unary/batch path) — the fast-path twin
/// of the #1786 `run_batch` mapping. Kept byte-identical to the gateway
/// constant `sie_gateway::handlers::proxy::MODEL_LOAD_FAILED_ERROR_CODE` and
/// the Python `ErrorCode.MODEL_LOAD_FAILED`.
const MODEL_LOAD_FAILED_ERROR_CODE: &str = "MODEL_LOAD_FAILED";

#[derive(serde::Serialize)]
struct GenerateTerminalErrorChunk<'a> {
    kind: &'static str,
    request_id: &'a str,
    attempt_id: String,
    seq: u32,
    text_delta: &'static str,
    done: bool,
    finish_reason: &'static str,
    error: GenerateTerminalError<'a>,
}

#[derive(serde::Serialize)]
struct GenerateTerminalError<'a> {
    code: &'a str,
    message: &'a str,
}

#[derive(Debug, Clone)]
struct DeliveryContext {
    subject: String,
    stream: String,
    consumer: String,
    stream_sequence: u64,
    consumer_sequence: u64,
    delivered: i64,
    pending: u64,
    has_metadata: bool,
}

impl DeliveryContext {
    fn from_message(msg: &Message) -> Self {
        match msg.info() {
            Ok(info) => Self {
                subject: msg.subject.to_string(),
                stream: info.stream.to_string(),
                consumer: info.consumer.to_string(),
                stream_sequence: info.stream_sequence,
                consumer_sequence: info.consumer_sequence,
                delivered: info.delivered,
                pending: info.pending,
                has_metadata: true,
            },
            Err(_) => Self {
                subject: msg.subject.to_string(),
                stream: String::new(),
                consumer: String::new(),
                stream_sequence: 0,
                consumer_sequence: 0,
                delivered: 0,
                pending: 0,
                has_metadata: false,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct GenerateDeliveryLogContext {
    work_item_id: String,
    request_id: String,
    model_id: String,
    delivery: DeliveryContext,
}

/// Base retry delay in milliseconds. Mirrors Python's `_NAK_DELAY_S`
/// (default 5 000 ms, overridable via `SIE_NAK_DELAY_S`). Used for:
///
/// * `retry_later` readiness (unknown error path)
/// * generic transient IPC / executor failures
///
/// Local model-loading waits use JetStream progress ACKs instead of NAKs, with
/// `loading_in_progress` sleeping for `2 × base` between readiness probes.
pub(crate) fn base_nak_delay_ms() -> u64 {
    std::env::var("SIE_NAK_DELAY_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .map(|s| (s * 1000.0) as u64)
        .unwrap_or(5_000)
}

/// NAK delay on fair-dispatch overflow — we want fast redelivery because
/// the item is otherwise ready, we're just flow-controlling. Matches
/// Python's hardcoded 0.1 s overflow NAK.
const NAK_DELAY_OVERFLOW_MS: u64 = 100;

/// NAK delay used when the backend reports `BackendError::Draining`. The
/// local backend is going away — another worker should pick the message
/// up promptly, so we use a short delay (100ms) instead of the generic
/// `base_nak_delay_ms` (~5s) that is intended for transient retryable
/// errors. Holding the message back longer just starves redelivery.
const NAK_DELAY_DRAINING_MS: u64 = 100;
const READINESS_PROGRESS_ACK_WAIT_FRACTION: u64 = 2;

/// Pick the right NAK delay for a given backend error — `Draining` is
/// fast because we want redelivery to another worker, everything else
/// uses the shared base delay.
pub(crate) fn nak_delay_for_backend_error(err: &BackendError) -> u64 {
    match err {
        BackendError::Draining => NAK_DELAY_DRAINING_MS,
        _ => base_nak_delay_ms(),
    }
}

fn readiness_progress_delay_ms(state: &ReadinessState, base_delay_ms: u64) -> Option<u64> {
    let max_delay_ms = crate::nats_consumer::ACK_WAIT_SECS
        .saturating_mul(1000)
        .checked_div(READINESS_PROGRESS_ACK_WAIT_FRACTION)
        .unwrap_or(1)
        .max(1);
    match state {
        ReadinessState::LoadingStarted => Some(base_delay_ms.min(max_delay_ms)),
        ReadinessState::LoadingInProgress => {
            Some(base_delay_ms.saturating_mul(2).min(max_delay_ms))
        }
        // `Failed` is terminal: no progress delay — the caller dead-letters
        // the group instead of re-driving `EnsureModelReady`.
        ReadinessState::Ready | ReadinessState::RetryLater | ReadinessState::Failed => None,
    }
}

/// Reply subjects must live under `_INBOX.` (NATS conventions). Non-empty
/// subjects outside this prefix are rejected as a crude anti-injection
/// check — a malicious producer could otherwise aim results at an
/// arbitrary subject. Empty reply_subjects are allowed (fire-and-forget).
const INBOX_PREFIX: &str = "_INBOX.";

fn msg_value_key_eq(key: &MsgValue, expected: &str) -> bool {
    match key {
        MsgValue::String(s) => s.as_str() == Some(expected),
        MsgValue::Binary(b) => std::str::from_utf8(b).ok() == Some(expected),
        _ => false,
    }
}

fn msg_map_get<'a>(value: &'a MsgValue, key: &str) -> Option<&'a MsgValue> {
    let MsgValue::Map(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find(|(k, _)| msg_value_key_eq(k, key))
        .map(|(_, v)| v)
}

fn msg_as_str(value: &MsgValue) -> Option<&str> {
    match value {
        MsgValue::String(s) => s.as_str(),
        MsgValue::Binary(b) => std::str::from_utf8(b).ok(),
        _ => None,
    }
}

/// True if `reply_subject` is acceptable for use on a `WorkItem`.
/// Empty is allowed (fire-and-forget). Non-empty subjects must start
/// with `_INBOX.` so malicious producers can't redirect results.
pub(crate) fn reply_subject_is_safe(reply_subject: &str) -> bool {
    reply_subject.is_empty() || reply_subject.starts_with(INBOX_PREFIX)
}

/// Bind each outcome (in order) to a `resolved` index by `work_item_id`,
/// consuming duplicate ids in FIFO arrival order. Returns a vector the
/// same length as `outcomes`, where `Some(idx)` means "outcome[i]
/// handles resolved[idx]" and `None` means "ghost outcome — no matching
/// work item remaining to consume".
///
/// Factored out of [`Dispatcher::apply_outcomes`] so the bookkeeping is
/// unit-testable without mocking JetStream `Message`s.
pub(crate) fn resolve_outcome_indices<'a, I>(
    resolved_wiids: &[&'a str],
    outcomes_wiids: I,
) -> Vec<Option<usize>>
where
    I: IntoIterator<Item = &'a str>,
{
    use std::collections::HashMap;

    let mut by_wiid: HashMap<&str, Vec<usize>> = HashMap::with_capacity(resolved_wiids.len());
    for (idx, wiid) in resolved_wiids.iter().enumerate() {
        by_wiid.entry(*wiid).or_default().push(idx);
    }
    // Reverse so `pop()` yields earliest-inserted index first.
    for v in by_wiid.values_mut() {
        v.reverse();
    }
    outcomes_wiids
        .into_iter()
        .map(|wiid| by_wiid.get_mut(wiid).and_then(|v| v.pop()))
        .collect()
}

/// Fallback per-model batch budget when the Python side doesn't report
/// one on EnsureModelReady (for example, while the model is not loaded).
/// Reads `SIE_NATS_FETCH_BUDGET` (default
/// 64), the same env var that the pull loop uses for its per-fetch
/// credit — so operators have one knob that means "how many messages a
/// worker should grab at a time" across both layers (matches Python's
/// historical `_DEFAULT_BATCH_BUDGET`).
fn default_batch_budget() -> u32 {
    std::env::var("SIE_NATS_FETCH_BUDGET")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(64)
}

/// Cap on concurrent model-group processing per worker. Matches Python's
/// `_MAX_CONCURRENT_BATCHES = 4` — enough to keep the IPC pipeline full
/// without risking ACK-timeout storms under backpressure. Override with
/// `SIE_MAX_CONCURRENT_BATCHES`.
///
/// Exposed publicly so `main.rs` can use the same value as the fallback
/// for `SIE_IPC_POOL_SIZE` when it's unset — the IPC pool should never
/// be smaller than the dispatcher's concurrency cap or it becomes the
/// binding constraint.
pub fn default_max_concurrent_batches() -> usize {
    std::env::var("SIE_MAX_CONCURRENT_BATCHES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(4)
}

#[derive(Debug, Error)]
pub enum DispatchError {
    /// Backend-layer error — IPC transport, native inference failure,
    /// or backend-level drain. Wraps [`BackendError`].
    #[error("backend: {0}")]
    Backend(#[from] BackendError),
    #[error("payload: {0}")]
    Payload(#[from] PayloadError),
    #[error("publish: {0}")]
    Publish(#[from] crate::publisher::PublishError),
    #[error("ipc: {0}")]
    Ipc(#[from] IpcError),
    #[error("nats ack: {0}")]
    Ack(String),
}

/// A JetStream message plus an optional pull-loop admission permit — the
/// pull-loop intake type consumed by [`Dispatcher::handle_batch`].
///
/// The permit is intentionally carried with the message until the request
/// settles. This makes the sidecar's queue intake behave like the Python
/// worker queue: a pulled item occupies capacity until the scheduler/backend
/// path has actually finished with it, not merely until it was enqueued.
/// At decode time `handle_batch` moves the permit into the item's
/// [`Delivery`], which owns it for the rest of the pipeline.
pub(crate) struct QueuedMessage {
    msg: Message,
    admission_permit: Option<OwnedSemaphorePermit>,
}

impl QueuedMessage {
    #[must_use]
    pub(crate) fn new(msg: Message, admission_permit: Option<OwnedSemaphorePermit>) -> Self {
        Self {
            msg,
            admission_permit,
        }
    }

    fn into_parts(self) -> (Message, Option<OwnedSemaphorePermit>) {
        (self.msg, self.admission_permit)
    }
}

impl From<Message> for QueuedMessage {
    fn from(msg: Message) -> Self {
        Self::new(msg, None)
    }
}

impl std::ops::Deref for QueuedMessage {
    type Target = Message;

    fn deref(&self) -> &Self::Target {
        &self.msg
    }
}

/// Pre-bundled handles the dispatcher needs — avoids a giant fn signature.
pub struct Dispatcher {
    /// Inference backend — typically a [`crate::backend::BackendRouter`]
    /// composing a native backend and/or [`crate::backend::PythonIpcBackend`].
    /// Held behind a trait object so the dispatcher doesn't care which
    /// backend runs which model.
    pub backend: SharedBackend,
    /// Adapter worker pool used for streaming generation. The batch backend
    /// trait remains outcome-oriented; generation is event-streaming, so
    /// the dispatcher asks the pool to pick the concrete child socket.
    pub worker_pool: Arc<AdapterWorkerPool>,
    pub payload_store: Arc<dyn PayloadStore>,
    /// NATS result publisher. `None` in local-ingest mode (P2.10, §4.6)
    /// where results ride each [`Delivery::Local`]'s event channel instead
    /// — the invariant "a [`Delivery::Nats`] item implies `Some`" holds by
    /// construction (`run()` wires both NATS and the publisher; `run_local()`
    /// wires neither).
    pub publisher: Option<Arc<WorkPublisher>>,
    /// Stable worker id stamped on locally-delivered `WorkResult`s. Same
    /// value the NATS `WorkPublisher` stamps on its results.
    pub worker_id: String,
    pub metrics: Arc<MetricsRegistry>,
    /// Rolling latency tracker shared with the pull loop. On every
    /// successful publish we record `inference_ms + postprocess_ms`
    /// (default) or `queue_ms + inference_ms + postprocess_ms` when
    /// `SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS=1`. See
    /// `crate::pull_quantum_includes_queue_ms` for the rationale.
    pub latency_tracker: Arc<Mutex<LatencyTracker>>,
    /// Caps concurrent model-group processing. Acquired once per group,
    /// held for the lifetime of the encode/score/extract fan-out.
    pub batch_semaphore: Arc<Semaphore>,
    /// Rust-side tokenizer registry. Always present; an empty
    /// registry means "no model has registered a tokeniser yet" and
    /// every `get(model_id)` returns `None`, which collapses the
    /// dispatcher to the Python-tokenise fallback path.
    ///
    /// Tokenisers are ingested lazily on the first
    /// [`crate::backend::InferenceBackend::ensure_model_ready`]
    /// response that carries a populated [`crate::ipc_types::ModelDescriptor`] for the
    /// model — see `Dispatcher::handle_model_group`.
    pub tokenizer_registry: Arc<TokenizerRegistry>,
    /// Optional Rust-side scheduler registry. When `Some`, every model
    /// is routed through the Rust scheduler
    /// (batch formation + adaptive control) with flushed batches
    /// shipped to the backend via `run_batch`. When `None`, the
    /// dispatcher falls back to the op-scoped `process_*_batch`
    /// path for every request — this mode is kept for unit tests
    /// that don't need the scheduler plumbing.
    ///
    /// The [`Shutdown`] signal below is observed by the per-model
    /// drain loops which are spawned lazily on first traffic by
    /// `Dispatcher::resolve_scheduler`.
    pub scheduler_registry: Option<Arc<ProductionSchedulerRegistry>>,
    /// Shutdown signal the per-model scheduler drain loops observe.
    /// `None` when no [`Self::scheduler_registry`] is configured —
    /// keeps the field inert for the Python-only topology.
    pub shutdown: Option<Arc<Shutdown>>,
    /// Current sidecar-applied bundle config state. Shared with the live config
    /// subscriber/reconciler and NATS health publisher.
    pub config_apply_state: Option<Arc<ConfigApplyState>>,
    /// Optional admission gate for both the physical queue and logical
    /// `admission_pool` labels carried on default-backed work items.
    pub pool_admission: Option<Arc<PoolAdmissionGate>>,
    /// Per-model scheduler drain handles, populated lazily when a
    /// new scheduler is materialised on first traffic. The shutdown
    /// path in `lib.rs` drains this map and awaits every handle so
    /// final-drain windows have a chance to complete.
    pub scheduler_drain_handles: Arc<Mutex<HashMap<String, JoinHandle<()>>>>,
    pub generation_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pub batch_cancel_state: BatchCancelState,
}

impl Dispatcher {
    /// Construct with a default-sized concurrency semaphore from env.
    ///
    /// `scheduler_registry` + `shutdown` together gate the scheduler
    /// path: if both are `Some`, every model routes
    /// through `Self::enqueue_encode_into_scheduler` (and the
    /// score / extract twins) with per-model drain loops spawned
    /// lazily on first traffic by `Self::resolve_scheduler`.
    /// Otherwise every model keeps the legacy
    /// `Self::handle_encode` / `Self::handle_score` /
    /// `Self::handle_extract` flow. It's a programming error to
    /// pass exactly one as `Some` — we assert it below rather than
    /// carry a half-wired state through the hot path.
    #[allow(clippy::too_many_arguments)] // each arg is a distinct dependency
    pub fn new(
        backend: SharedBackend,
        worker_pool: Arc<AdapterWorkerPool>,
        payload_store: Arc<dyn PayloadStore>,
        publisher: Option<Arc<WorkPublisher>>,
        worker_id: String,
        metrics: Arc<MetricsRegistry>,
        latency_tracker: Arc<Mutex<LatencyTracker>>,
        tokenizer_registry: Arc<TokenizerRegistry>,
        scheduler_registry: Option<Arc<ProductionSchedulerRegistry>>,
        shutdown: Option<Arc<Shutdown>>,
        config_apply_state: Option<Arc<ConfigApplyState>>,
        pool_admission: Option<Arc<PoolAdmissionGate>>,
        batch_cancel_state: BatchCancelState,
    ) -> Self {
        debug_assert_eq!(
            scheduler_registry.is_some(),
            shutdown.is_some(),
            "scheduler_registry and shutdown must be wired together — one half invalidates the drain-loop lifecycle"
        );
        Self {
            backend,
            worker_pool,
            payload_store,
            publisher,
            worker_id,
            metrics,
            latency_tracker,
            batch_semaphore: Arc::new(Semaphore::new(default_max_concurrent_batches())),
            tokenizer_registry,
            scheduler_registry,
            shutdown,
            config_apply_state,
            pool_admission,
            scheduler_drain_handles: Arc::new(Mutex::new(HashMap::new())),
            generation_handles: Arc::new(Mutex::new(Vec::new())),
            batch_cancel_state,
        }
    }
}

impl Dispatcher {
    fn current_bundle_config_hash(&self) -> Option<String> {
        self.config_apply_state
            .as_ref()
            .map(|state| state.current_bundle_config_hash())
    }

    fn is_cancelled_worker_direct_batch(
        &self,
        request_id: &str,
        operation: &str,
        worker_direct: bool,
    ) -> bool {
        worker_direct && operation != "generate" && self.batch_cancel_state.is_cancelled(request_id)
    }

    async fn ack_cancelled_worker_direct_batch(&self, wi: &WorkItem, delivery: &Delivery) {
        debug!(
            request_id = %wi.request_id,
            work_item_id = %wi.work_item_id,
            operation = %wi.operation,
            origin = %delivery.log_ref(),
            "batch-direct: ACKing cancelled worker-direct item"
        );
        match ack(delivery).await {
            Ok(()) => self.metrics.messages_acked_total.inc(),
            Err(e) => {
                warn!(error = %e, "ack failed on cancelled worker-direct batch item");
                self.metrics.jetstream_ack_failures_total.inc();
            }
        }
    }
}

fn unknown_bundle_config_hash<'a>(
    items: impl IntoIterator<Item = &'a WorkItem>,
    state: Option<&ConfigApplyState>,
) -> Option<(&'a str, usize)> {
    let state = state?;
    let mut first_unknown: Option<&'a str> = None;
    let mut count = 0usize;
    for wi in items {
        if !state.accepts_bundle_config_hash(&wi.bundle_config_hash) {
            count += 1;
            if first_unknown.is_none() {
                first_unknown = Some(wi.bundle_config_hash.as_str());
            }
        }
    }
    first_unknown.map(|hash| (hash, count))
}

impl Dispatcher {
    /// Process a full fetched batch.
    ///
    /// Legacy op handlers return after every message has either been ACKed
    /// (we published a result) or NAKed (we'll see it again). Scheduler-routed
    /// handlers return after enqueue, but any pull-loop admission permit
    /// rides the item's [`Delivery`] (inside [`SchedulerMeta`]) until the
    /// scheduler/backend path settles it.
    ///
    /// Model-id routing: derived from the NATS **subject**, falling back
    /// to `WorkItem.model_id` when the subject is malformed. On
    /// disagreement the subject wins (that's what JetStream used to
    /// dispatch us) and we warn.
    ///
    /// Concurrency:
    /// * group by `model_id` only.
    /// * model groups run concurrently, capped by `batch_semaphore` to
    ///   avoid ACK-timeout storms.
    /// * within a model, encode/score/extract run concurrently via
    ///   `tokio::join!` so slow payload fetches don't block other ops.
    pub(crate) async fn handle_batch(self: &Arc<Self>, messages: Vec<QueuedMessage>) {
        let batch_started = Instant::now();
        let batch_size = messages.len();
        self.metrics
            .messages_received_total
            .inc_by(batch_size as u64);
        info!(batch_size, "handle_batch: start");

        let base_delay_ms = base_nak_delay_ms();
        let mut decoded: Vec<(WorkItem, Delivery)> = Vec::with_capacity(batch_size);
        for queued in messages {
            // The admission permit moves into the [`Delivery`] here: every
            // reject path below settles (ACK/NAK) and drops it immediately,
            // and successful decodes carry it until downstream settlement.
            let (msg, admission_permit) = queued.into_parts();
            // Record JetStream delivery count so operators can spot hot
            // redelivery (which usually means a worker is hitting ack_wait).
            // `info()` returns Ok for pull-based messages; core NATS messages
            // in tests return Err and are quietly skipped.
            if let Ok(info) = msg.info() {
                // Operation is part of the WorkItem, not the subject; use
                // "all" here and rely on `sie_pull_loop_batch_process_seconds`
                // (already labelled by operation) for per-op breakdowns.
                self.metrics
                    .nats_deliver_count
                    .with_label_values(&["all"])
                    .observe(info.delivered as f64);
                if info.delivered > 1 {
                    self.metrics.nats_redelivery_total.inc();
                }
            }
            // Source of truth for routing is the NATS subject (JetStream
            // already used it to dispatch to this consumer). If the subject
            // doesn't yield a model_id, we can't trust the payload either,
            // so NAK for redelivery and let max_deliver → DLQ handle it.
            let subject_model = match extract_model_id(&msg.subject) {
                Some(m) => m,
                None => {
                    warn!(
                        subject = %msg.subject,
                        "could not extract model_id from subject — NAKing for redelivery",
                    );
                    nak_one(
                        &Delivery::Nats(msg, admission_permit),
                        base_delay_ms,
                        &self.metrics,
                    )
                    .await;
                    continue;
                }
            };
            match rmp_serde::from_slice::<WorkItem>(&msg.payload) {
                Ok(mut wi) => {
                    if !reply_subject_is_safe(&wi.reply_subject) {
                        // ACK-to-drop (not NAK): the subject is attacker-
                        // controlled; retrying just amplifies the attempt.
                        warn!(
                            work_item_id = %wi.work_item_id,
                            reply_subject = %truncate(&wi.reply_subject, 60),
                            "rejecting WorkItem with suspicious reply_subject — ACKing to drop",
                        );
                        match ack(&Delivery::Nats(msg, admission_permit)).await {
                            Ok(()) => self.metrics.messages_acked_total.inc(),
                            Err(e) => {
                                warn!(error = %e, "ack failed on drop");
                                self.metrics.jetstream_ack_failures_total.inc();
                            }
                        }
                        continue;
                    }
                    let worker_direct = is_worker_direct_work_subject(&msg.subject);
                    if self.is_cancelled_worker_direct_batch(
                        &wi.request_id,
                        &wi.operation,
                        worker_direct,
                    ) {
                        self.ack_cancelled_worker_direct_batch(
                            &wi,
                            &Delivery::Nats(msg, admission_permit),
                        )
                        .await;
                        continue;
                    }
                    if let Some(gate) = self.pool_admission.as_ref() {
                        let admission_pool = wi.admission_pool.trim();
                        if !gate.admits_work_item_pool(admission_pool) {
                            debug!(
                                work_item_id = %wi.work_item_id,
                                request_id = %wi.request_id,
                                admission_pool = %admission_pool,
                                physical_pool = %wi.pool_name,
                                "WorkItem admission_pool does not assign this worker — NAKing for redelivery"
                            );
                            self.metrics
                                .pool_admission_naks_total
                                .with_label_values(&["logical_pool_not_assigned"])
                                .inc();
                            nak_one(
                                &Delivery::Nats(msg, admission_permit),
                                base_delay_ms,
                                &self.metrics,
                            )
                            .await;
                            continue;
                        }
                    }
                    if wi.model_id != subject_model {
                        warn!(
                            work_item_id = %wi.work_item_id,
                            subject = %msg.subject,
                            wi_model_id = %wi.model_id,
                            subject_model_id = %subject_model,
                            "WorkItem.model_id disagrees with subject — trusting subject",
                        );
                        wi.model_id = subject_model;
                    }
                    decoded.push((wi, Delivery::Nats(msg, admission_permit)));
                }
                Err(e) => {
                    // NAK so JetStream redelivers (possibly to a worker on a
                    // newer wire version) and eventually DLQs after
                    // max_deliver. ACK-dropping would silently discard the
                    // item on a transient msgpack glitch.
                    warn!(error = %e, subject = %msg.subject, "failed to decode WorkItem — NAKing for redelivery");
                    nak_one(
                        &Delivery::Nats(msg, admission_permit),
                        base_delay_ms,
                        &self.metrics,
                    )
                    .await;
                }
            }
        }
        if decoded.is_empty() {
            info!(
                batch_size,
                elapsed_ms = batch_started.elapsed().as_millis() as u64,
                "handle_batch: done (all messages rejected before dispatch)"
            );
            return;
        }
        self.dispatch_decoded(decoded, batch_size, batch_started)
            .await;
    }

    /// Dispatch already-decoded `(WorkItem, Delivery)` pairs — the shared
    /// tail of [`Self::handle_batch`], also fed directly by the local-ingest
    /// server (P2.10, §4.6) whose items arrive pre-decoded over the UDS
    /// rather than as NATS messages. Both ingest paths coalesce in the same
    /// per-model scheduler batch assembly downstream.
    pub async fn dispatch_decoded(
        self: &Arc<Self>,
        decoded: Vec<(WorkItem, Delivery)>,
        batch_size: usize,
        batch_started: Instant,
    ) {
        let mut generate_items = Vec::new();
        let mut regular_items = Vec::with_capacity(decoded.len());
        for (wi, delivery) in decoded {
            if wi.operation == "generate" {
                match delivery {
                    // Generation bypasses the scheduler; re-bundle the permit
                    // with the message so intake capacity stays held until
                    // the generate task settles the delivery.
                    Delivery::Nats(msg, permit) => {
                        generate_items.push((wi, QueuedMessage::new(msg, permit)))
                    }
                    delivery @ Delivery::Local(_) => {
                        // Generation streams chunk envelopes over NATS reply
                        // subjects; the local-ingest publish_work op is
                        // one-shot (PROTOCOL.md v0.1 — streaming ops land
                        // with P2.6). Answer a typed error instead of
                        // silently sinking the item.
                        let _ = self
                            .publish_error(
                                &wi,
                                &delivery,
                                "bad_operation",
                                "generate is not supported on the local-ingest lane (P2.6)",
                            )
                            .await;
                    }
                }
            } else {
                regular_items.push((wi, delivery));
            }
        }
        let generate_count = generate_items.len();
        if generate_count > 0 {
            self.spawn_generate_items(generate_items).await;
        }
        if regular_items.is_empty() {
            info!(
                batch_size,
                generate = generate_count,
                elapsed_ms = batch_started.elapsed().as_millis() as u64,
                "handle_batch: done (generation items handed off)"
            );
            return;
        }

        let model_groups = group_by_model_only(regular_items);
        let group_count = model_groups.len();
        let mut futs = Vec::with_capacity(group_count);
        for (model_id, items) in model_groups {
            let this = Arc::clone(self);
            futs.push(async move {
                let permit = match this.batch_semaphore.clone().acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => {
                        warn!(model = %model_id, "batch semaphore closed — dropping group");
                        return;
                    }
                };
                let count_model_group_inflight = this.scheduler_registry.is_none();
                if count_model_group_inflight {
                    this.metrics.inflight_batches.inc();
                }
                let result = this.handle_model_group(&model_id, items).await;
                if count_model_group_inflight {
                    this.metrics.inflight_batches.dec();
                }
                drop(permit);
                if let Err(e) = result {
                    warn!(model = %model_id, error = %ErrChain(&e), "model group handling failed");
                }
            });
        }
        join_all(futs).await;
        info!(
            batch_size,
            group_count,
            elapsed_ms = batch_started.elapsed().as_millis() as u64,
            "handle_batch: done"
        );
    }

    async fn spawn_generate_items(self: &Arc<Self>, items: Vec<(WorkItem, QueuedMessage)>) {
        let mut handles = self.generation_handles.lock().await;
        handles.retain(|h| !h.is_finished());
        for (wi, msg) in items {
            let this = Arc::clone(self);
            handles.push(tokio::spawn(async move {
                this.handle_generate_item(wi, msg).await;
            }));
        }
    }

    async fn handle_generate_item(self: Arc<Self>, mut wi: WorkItem, msg: QueuedMessage) {
        let model_id = wi.model_id.clone();
        let delivery = DeliveryContext::from_message(&msg);
        let model_lbl = self.metrics.model_label(&model_id).into_owned();
        let _timer = self
            .metrics
            .pull_batch_process_seconds
            .with_label_values(&[&model_lbl, "generate"])
            .start_timer();
        let base_delay_ms = base_nak_delay_ms();
        info!(
            work_item_id = %wi.work_item_id,
            request_id = %wi.request_id,
            model = %model_id,
            subject = %delivery.subject,
            stream = %delivery.stream,
            consumer = %delivery.consumer,
            stream_seq = delivery.stream_sequence,
            consumer_seq = delivery.consumer_sequence,
            delivery_count = delivery.delivered,
            pending = delivery.pending,
            has_metadata = delivery.has_metadata,
            "generate delivery received"
        );

        let readiness_resp = match self.backend.ensure_model_ready(&model_id).await {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    error = %ErrChain(&e),
                    "EnsureModelReady failed for generate — NAKing"
                );
                nak_msg(&msg, nak_delay_for_backend_error(&e), &self.metrics).await;
                return;
            }
        };
        match &readiness_resp.state {
            ReadinessState::Ready => {}
            ReadinessState::LoadingStarted => {
                info!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    readiness = ?readiness_resp.state,
                    delay_ms = base_delay_ms,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    "generate model started loading — publishing MODEL_LOADING chunk + ACK"
                );
                let message = format!("Model '{model_id}' is loading; retry later.");
                match self
                    .publish_generate_terminal_error(&wi, MODEL_LOADING_ERROR_CODE, &message)
                    .await
                {
                    Ok(_) => match ack_msg(&msg).await {
                        Ok(()) => {
                            self.metrics.messages_acked_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[
                                    &model_lbl,
                                    "loading_started",
                                    "published_acked",
                                ])
                                .inc();
                        }
                        Err(e) => {
                            warn!(
                                work_item_id = %wi.work_item_id,
                                request_id = %wi.request_id,
                                model = %model_id,
                                stream_seq = delivery.stream_sequence,
                                delivery_count = delivery.delivered,
                                error = %e,
                                "ack after MODEL_LOADING chunk publish failed"
                            );
                            self.metrics.jetstream_ack_failures_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[
                                    &model_lbl,
                                    "loading_started",
                                    "published_ack_failed",
                                ])
                                .inc();
                        }
                    },
                    Err(_) => {
                        self.metrics
                            .generate_model_loading_responses_total
                            .with_label_values(&[&model_lbl, "loading_started", "publish_failed"])
                            .inc();
                        nak_msg(&msg, base_delay_ms, &self.metrics).await;
                    }
                }
                return;
            }
            ReadinessState::RetryLater => {
                info!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    readiness = ?readiness_resp.state,
                    delay_ms = base_delay_ms,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    "generate model retry requested — NAKing"
                );
                nak_msg(&msg, base_delay_ms, &self.metrics).await;
                return;
            }
            ReadinessState::LoadingInProgress => {
                let delay_ms = readiness_progress_delay_ms(&readiness_resp.state, base_delay_ms)
                    .unwrap_or_else(|| base_delay_ms.saturating_mul(2));
                info!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    readiness = ?readiness_resp.state,
                    delay_ms,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    "generate model still loading — publishing MODEL_LOADING chunk + ACK"
                );
                let message = format!("Model '{model_id}' is still loading; retry later.");
                match self
                    .publish_generate_terminal_error(&wi, MODEL_LOADING_ERROR_CODE, &message)
                    .await
                {
                    Ok(_) => match ack_msg(&msg).await {
                        Ok(()) => {
                            self.metrics.messages_acked_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[
                                    &model_lbl,
                                    "loading_in_progress",
                                    "published_acked",
                                ])
                                .inc();
                        }
                        Err(e) => {
                            warn!(
                                work_item_id = %wi.work_item_id,
                                request_id = %wi.request_id,
                                model = %model_id,
                                stream_seq = delivery.stream_sequence,
                                delivery_count = delivery.delivered,
                                error = %e,
                                "ack after MODEL_LOADING chunk publish failed"
                            );
                            self.metrics.jetstream_ack_failures_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[
                                    &model_lbl,
                                    "loading_in_progress",
                                    "published_ack_failed",
                                ])
                                .inc();
                        }
                    },
                    Err(_) => {
                        self.metrics
                            .generate_model_loading_responses_total
                            .with_label_values(&[
                                &model_lbl,
                                "loading_in_progress",
                                "publish_failed",
                            ])
                            .inc();
                        nak_msg(&msg, delay_ms, &self.metrics).await;
                    }
                }
                return;
            }
            ReadinessState::Failed => {
                // Terminal load failure (permanent cooldown). Re-driving would
                // hang the streaming client forever (#1786 fast-path gap), so
                // publish a terminal MODEL_LOAD_FAILED chunk + ACK — the
                // gateway's streaming collector maps the code to a typed
                // failure exactly like the batch path.
                info!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    readiness = ?readiness_resp.state,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    "generate model load failed terminally — publishing MODEL_LOAD_FAILED chunk + ACK"
                );
                let message = format!("Model '{model_id}' failed to load permanently.");
                match self
                    .publish_generate_terminal_error(&wi, MODEL_LOAD_FAILED_ERROR_CODE, &message)
                    .await
                {
                    Ok(_) => match ack_msg(&msg).await {
                        Ok(()) => {
                            self.metrics.messages_acked_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[&model_lbl, "failed", "published_acked"])
                                .inc();
                        }
                        Err(e) => {
                            warn!(
                                work_item_id = %wi.work_item_id,
                                request_id = %wi.request_id,
                                model = %model_id,
                                stream_seq = delivery.stream_sequence,
                                delivery_count = delivery.delivered,
                                error = %e,
                                "ack after MODEL_LOAD_FAILED chunk publish failed"
                            );
                            self.metrics.jetstream_ack_failures_total.inc();
                            self.metrics
                                .generate_model_loading_responses_total
                                .with_label_values(&[&model_lbl, "failed", "published_ack_failed"])
                                .inc();
                        }
                    },
                    Err(_) => {
                        self.metrics
                            .generate_model_loading_responses_total
                            .with_label_values(&[&model_lbl, "failed", "publish_failed"])
                            .inc();
                        // Terminal failure but the publish itself failed: NAK
                        // so redelivery can retry surfacing the typed error
                        // rather than silently dropping the client's request.
                        nak_msg(&msg, base_delay_ms, &self.metrics).await;
                    }
                }
                return;
            }
        }

        // Resolve an offloaded generate payload. The gateway offloads large
        // vision work items to the object store (``payload_ref`` set,
        // ``generate`` blanked); fetch + inline so the adapter worker — which
        // has no object-store access — receives a self-contained WorkItem. The
        // blob is base64-string image data, so it decodes cleanly as
        // ``serde_json::Value`` (msgpack ``bin`` would not).
        if let Some(payload_ref) = wi.payload_ref.clone() {
            match self.payload_store.get(&payload_ref).await {
                Ok(bytes) => match decode_offloaded_generate(&bytes) {
                    Ok(generate) => {
                        wi.generate = Some(generate);
                        wi.payload_ref = None;
                    }
                    Err(e) => {
                        warn!(
                            work_item_id = %wi.work_item_id,
                            request_id = %wi.request_id,
                            model = %model_id,
                            subject = %delivery.subject,
                            stream = %delivery.stream,
                            consumer = %delivery.consumer,
                            stream_seq = delivery.stream_sequence,
                            consumer_seq = delivery.consumer_sequence,
                            delivery_count = delivery.delivered,
                            pending = delivery.pending,
                            error = %e,
                            "failed to decode offloaded generate payload — publishing error + ACK"
                        );
                        match self
                            .publish_error_generate(
                                &wi,
                                "internal_error",
                                "failed to decode offloaded generate payload",
                                is_worker_direct_work_subject(&msg.subject),
                            )
                            .await
                        {
                            Ok(_) => match ack_msg(&msg).await {
                                Ok(()) => self.metrics.messages_acked_total.inc(),
                                Err(e) => {
                                    warn!(
                                        work_item_id = %wi.work_item_id,
                                        request_id = %wi.request_id,
                                        model = %model_id,
                                        stream_seq = delivery.stream_sequence,
                                        delivery_count = delivery.delivered,
                                        error = %e,
                                        "ack after offload-decode error failed"
                                    );
                                    self.metrics.jetstream_ack_failures_total.inc();
                                }
                            },
                            Err(_) => nak_msg(&msg, base_delay_ms, &self.metrics).await,
                        }
                        return;
                    }
                },
                Err(e) => {
                    warn!(
                        work_item_id = %wi.work_item_id,
                        request_id = %wi.request_id,
                        model = %model_id,
                        subject = %delivery.subject,
                        stream = %delivery.stream,
                        consumer = %delivery.consumer,
                        stream_seq = delivery.stream_sequence,
                        consumer_seq = delivery.consumer_sequence,
                        delivery_count = delivery.delivered,
                        pending = delivery.pending,
                        error = %e,
                        "failed to resolve offloaded generate payload — NAKing"
                    );
                    nak_msg(&msg, base_delay_ms, &self.metrics).await;
                    return;
                }
            }
        }

        let work_item_msgpack = match rmp_serde::to_vec_named(&wi) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    subject = %delivery.subject,
                    stream = %delivery.stream,
                    consumer = %delivery.consumer,
                    stream_seq = delivery.stream_sequence,
                    consumer_seq = delivery.consumer_sequence,
                    delivery_count = delivery.delivered,
                    pending = delivery.pending,
                    error = %e,
                    "failed to re-encode generate WorkItem — publishing error + ACK"
                );
                match self
                    .publish_error_generate(
                        &wi,
                        "internal_error",
                        "failed to encode generate work item",
                        is_worker_direct_work_subject(&msg.subject),
                    )
                    .await
                {
                    Ok(_) => match ack_msg(&msg).await {
                        Ok(()) => self.metrics.messages_acked_total.inc(),
                        Err(e) => {
                            warn!(
                                work_item_id = %wi.work_item_id,
                                request_id = %wi.request_id,
                                model = %model_id,
                                stream_seq = delivery.stream_sequence,
                                delivery_count = delivery.delivered,
                                error = %e,
                                "ack after generate encode error failed"
                            );
                            self.metrics.jetstream_ack_failures_total.inc();
                        }
                    },
                    Err(_) => nak_msg(&msg, base_delay_ms, &self.metrics).await,
                }
                return;
            }
        };

        let Some(publisher) = self.publisher.clone() else {
            // Unreachable by construction: generate items only arrive here
            // as NATS deliveries, and `run()` always wires the publisher.
            warn!(
                work_item_id = %wi.work_item_id,
                "generate item without a NATS publisher — NAKing"
            );
            nak_msg(&msg, base_delay_ms, &self.metrics).await;
            return;
        };
        let settled = Arc::new(AtomicBool::new(false));
        let msg = Arc::new(msg);
        let delivery_log = Arc::new(GenerateDeliveryLogContext {
            work_item_id: wi.work_item_id.clone(),
            request_id: wi.request_id.clone(),
            model_id: model_id.clone(),
            delivery,
        });
        let metrics = Arc::clone(&self.metrics);
        let settled_for_events = Arc::clone(&settled);
        let msg_for_events = Arc::clone(&msg);
        let delivery_log_for_events = Arc::clone(&delivery_log);
        self.metrics.inflight_batches.inc();
        let result = self
            .worker_pool
            .process_generate(
                ProcessGenerateRequest {
                    model_id: model_id.clone(),
                    work_item_msgpack,
                },
                move |event| {
                    let publisher = Arc::clone(&publisher);
                    let metrics = Arc::clone(&metrics);
                    let settled = Arc::clone(&settled_for_events);
                    let msg = Arc::clone(&msg_for_events);
                    let delivery_log = Arc::clone(&delivery_log_for_events);
                    async move {
                        handle_generate_event(event, publisher, metrics, settled, msg, delivery_log)
                            .await
                            .map_err(|e| IpcError::Server(e.to_string()))
                    }
                },
            )
            .await;
        decrement_gauge(&self.metrics.inflight_batches, 1);

        match result {
            Ok(()) => {
                if !settled.load(Ordering::SeqCst) {
                    warn!(
                        work_item_id = %wi.work_item_id,
                        request_id = %wi.request_id,
                        model = %model_id,
                        subject = %delivery_log.delivery.subject,
                        stream = %delivery_log.delivery.stream,
                        consumer = %delivery_log.delivery.consumer,
                        stream_seq = delivery_log.delivery.stream_sequence,
                        consumer_seq = delivery_log.delivery.consumer_sequence,
                        delivery_count = delivery_log.delivery.delivered,
                        pending = delivery_log.delivery.pending,
                        "ProcessGenerate ended without ACK/NAK event — NAKing"
                    );
                    nak_msg(&msg, base_delay_ms, &self.metrics).await;
                }
            }
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    request_id = %wi.request_id,
                    model = %model_id,
                    subject = %delivery_log.delivery.subject,
                    stream = %delivery_log.delivery.stream,
                    consumer = %delivery_log.delivery.consumer,
                    stream_seq = delivery_log.delivery.stream_sequence,
                    consumer_seq = delivery_log.delivery.consumer_sequence,
                    delivery_count = delivery_log.delivery.delivered,
                    pending = delivery_log.delivery.pending,
                    error = %e,
                    "ProcessGenerate failed — NAKing if unsettled"
                );
                if !settled.swap(true, Ordering::SeqCst) {
                    nak_msg(&msg, base_delay_ms, &self.metrics).await;
                }
            }
        }
    }

    async fn handle_model_group(
        self: &Arc<Self>,
        model_id: &str,
        items: Vec<(WorkItem, Delivery)>,
    ) -> Result<(), DispatchError> {
        let group_started = Instant::now();
        let group_size = items.len();
        let base_delay_ms = base_nak_delay_ms();
        if let Some((expected_hash, unknown_hash_count)) = unknown_bundle_config_hash(
            items.iter().map(|(wi, _)| wi),
            self.config_apply_state.as_deref(),
        ) {
            let local_hash = self.current_bundle_config_hash().unwrap_or_default();
            info!(
                model = %model_id,
                unknown_hash_count,
                group_size,
                local_hash = %local_hash,
                expected_hash,
                "request bundle config hash is unknown locally — NAKing group"
            );
            nak_all(&items, base_delay_ms, &self.metrics).await;
            return Ok(());
        }
        let readiness_resp = loop {
            // `ipc_requests_total` / `ipc_failures_total` identify the IPC
            // transport counters even though the backend trait is generic (see
            // `crate::backend::InferenceBackend`).
            self.metrics.ipc_requests_total.inc();
            let readiness_resp = match self.backend.ensure_model_ready(model_id).await {
                Ok(r) => r,
                Err(e) => {
                    self.metrics.ipc_failures_total.inc();
                    warn!(
                        model = %model_id,
                        group_size,
                        error = %ErrChain(&e),
                        "EnsureModelReady failed — NAKing group"
                    );
                    nak_all(&items, base_delay_ms, &self.metrics).await;
                    return Err(e.into());
                }
            };
            if readiness_resp.state == ReadinessState::Ready {
                break readiness_resp;
            }
            if readiness_resp.state == ReadinessState::Failed {
                // Terminal load failure (permanent cooldown on the Python
                // registry). Re-driving `EnsureModelReady` would loop forever
                // and hang the client (#1786 fast-path gap), so dead-letter
                // the whole group as `MODEL_LOAD_FAILED` — the gateway maps
                // that code to a typed 502, exactly like the batch/`run_batch`
                // path. ACK each item after publishing its error so JetStream
                // stops redelivering the doomed work.
                info!(
                    model = %model_id,
                    group_size,
                    "model load failed terminally — dead-lettering group as MODEL_LOAD_FAILED"
                );
                self.dead_letter_all(&items, model_id).await;
                return Ok(());
            }
            if let Some(delay_ms) =
                readiness_progress_delay_ms(&readiness_resp.state, base_delay_ms)
            {
                info!(
                    model = %model_id,
                    group_size,
                    readiness = ?readiness_resp.state,
                    delay_ms,
                    "model loading — progress ACKing group before retry"
                );
                if !progress_all(&items, &self.metrics).await {
                    nak_all(&items, base_delay_ms, &self.metrics).await;
                    return Ok(());
                }
                if self.sleep_or_shutdown(delay_ms).await {
                    warn!(
                        model = %model_id,
                        group_size,
                        "shutdown while waiting for model load — NAKing group"
                    );
                    nak_all(&items, NAK_DELAY_DRAINING_MS, &self.metrics).await;
                    return Ok(());
                }
                continue;
            }
            info!(model = %model_id, "model not available — NAKing group");
            nak_all(&items, base_delay_ms, &self.metrics).await;
            return Ok(());
        };

        // Adapter handshake: fold the adapter's `ModelDescriptor`
        // (if any) into our local registries. Idempotent on
        // re-handshake — the registry hashes the loaded
        // tokenizer.json and short-circuits if the declared
        // `tokenizer_id` already matches what's cached.
        if let Some(descriptor) = readiness_resp.descriptor.as_ref() {
            match self
                .tokenizer_registry
                .register_from_descriptor(model_id, descriptor)
            {
                Ok(true) => {
                    debug!(
                        model = %model_id,
                        "rust-tokenize: registered tokeniser from EnsureModelReady descriptor"
                    );
                }
                Ok(false) => {} // no path, idempotent, or hash mismatch (warning logged inside)
                Err(e) => {
                    // Non-fatal: model just falls back to Python
                    // tokenisation, exactly the same as if no
                    // descriptor had been declared at all.
                    warn!(
                        model = %model_id,
                        error = %e,
                        "rust-tokenize: descriptor load failed — Python will tokenise this model"
                    );
                }
            }
        }

        // Cap the group at the per-model batch budget reported by Python.
        // Overflow gets NAK'd with a short delay so it redelivers to
        // (possibly) another worker — keeps one hot model from starving
        // the others on this worker's GPU.
        let budget = readiness_resp
            .batch_budget
            .filter(|&b| b > 0)
            .unwrap_or_else(default_batch_budget) as usize;
        let (dispatch, overflow) = split_by_budget(items, budget);
        if !overflow.is_empty() {
            debug!(
                model = %model_id,
                budget,
                overflow = overflow.len(),
                "fair dispatch: NAKing overflow"
            );
            nak_all(&overflow, NAK_DELAY_OVERFLOW_MS, &self.metrics).await;
        }
        if dispatch.is_empty() {
            return Ok(());
        }
        let model_lbl = self.metrics.model_label(model_id);
        self.metrics
            .pull_items_fetched
            .with_label_values(&[&model_lbl])
            .observe(dispatch.len() as f64);

        // Fan out by operation; each op runs concurrently below. The IPC
        // client serialises at the socket but payload resolution, NATS
        // ACKs and publishes all overlap across ops.
        let mut encode_items = Vec::new();
        let mut score_items = Vec::new();
        let mut extract_items = Vec::new();
        let mut unknown_items: Vec<(WorkItem, Delivery)> = Vec::new();
        for (wi, delivery) in dispatch {
            match wi.operation.as_str() {
                "encode" => encode_items.push((wi, delivery)),
                "score" => score_items.push((wi, delivery)),
                "extract" => extract_items.push((wi, delivery)),
                _ => unknown_items.push((wi, delivery)),
            }
        }

        for (wi, delivery) in &unknown_items {
            warn!(op = %wi.operation, "unknown operation — publishing error + ACK");
            match self
                .publish_error(wi, delivery, "bad_operation", "unknown operation")
                .await
            {
                Ok(_) => match ack(delivery).await {
                    Ok(()) => self.metrics.messages_acked_total.inc(),
                    Err(e) => {
                        warn!(error = %e, "ack after bad_operation error-publish failed");
                        self.metrics.jetstream_ack_failures_total.inc();
                    }
                },
                Err(_) => {
                    // Error publish failed — NAK so JetStream redelivers
                    // and we get another chance to either succeed or hit
                    // max_deliver → DLQ (preserves the failure rather
                    // than silently dropping it).
                    nak_one(delivery, base_nak_delay_ms(), &self.metrics).await;
                }
            }
        }

        let encode_n = encode_items.len();
        let score_n = score_items.len();
        let extract_n = extract_items.len();
        let unknown_n = unknown_items.len();

        // When `scheduler_registry` is wired, every op routes through
        // the scheduler's submit-then-drain path instead of the per-op
        // `process_*_batch` path. The scheduler owns batch formation +
        // adaptive control and hands flushed batches to the backend via
        // `run_batch`; the per-model drain loop (spawned lazily on first
        // traffic inside [`Self::resolve_scheduler`]) handles inference
        // + publish + ACK/NAK.
        //
        // Registry absent ⇒ legacy path unchanged. Only unit tests
        // exercise that branch today.
        let scheduler_opt = self.resolve_scheduler(model_id).await;
        let encode_fut = async {
            if encode_items.is_empty() {
                return Ok(());
            }
            if let Some(sched) = scheduler_opt.as_ref() {
                self.enqueue_encode_into_scheduler(model_id, sched, encode_items)
                    .await;
                return Ok(());
            }
            self.handle_encode(model_id, encode_items).await
        };
        let score_fut = async {
            if score_items.is_empty() {
                return Ok(());
            }
            if let Some(sched) = scheduler_opt.as_ref() {
                self.enqueue_score_into_scheduler(model_id, sched, score_items)
                    .await;
                return Ok(());
            }
            self.handle_score(model_id, score_items).await
        };
        let extract_fut = async {
            if extract_items.is_empty() {
                return Ok(());
            }
            if let Some(sched) = scheduler_opt.as_ref() {
                self.enqueue_extract_into_scheduler(model_id, sched, extract_items)
                    .await;
                return Ok(());
            }
            self.handle_extract(model_id, extract_items).await
        };
        let (r_enc, r_score, r_ext) = tokio::join!(encode_fut, score_fut, extract_fut);
        if let Err(e) = &r_enc {
            warn!(model = %model_id, error = %ErrChain(e), "encode batch failed");
        }
        if let Err(e) = &r_score {
            warn!(model = %model_id, error = %ErrChain(e), "score batch failed");
        }
        if let Err(e) = &r_ext {
            warn!(model = %model_id, error = %ErrChain(e), "extract batch failed");
        }
        info!(
            model = %model_id,
            group_size,
            encode = encode_n,
            score = score_n,
            extract = extract_n,
            unknown = unknown_n,
            encode_ok = r_enc.is_ok(),
            score_ok = r_score.is_ok(),
            extract_ok = r_ext.is_ok(),
            elapsed_ms = group_started.elapsed().as_millis() as u64,
            "handle_model_group: done"
        );
        Ok(())
    }

    // -- encode -----------------------------------------------------------

    async fn handle_encode(
        &self,
        model_id: &str,
        items: Vec<(WorkItem, Delivery)>,
    ) -> Result<(), DispatchError> {
        let model_lbl = self.metrics.model_label(model_id);
        let _timer = self
            .metrics
            .pull_batch_process_seconds
            .with_label_values(&[&model_lbl, "encode"])
            .start_timer();
        let mut resolved: Vec<(WorkItem, Delivery, MsgValue, f64)> =
            Vec::with_capacity(items.len());
        for (wi, delivery) in items {
            let (item_json, fetch_ms) = match self.resolve_item(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        error = %ErrChain(&e),
                        work_item_id = %wi.work_item_id,
                        request_id = %wi.request_id,
                        model = %wi.model_id,
                        "failed to resolve encode item"
                    );
                    match self
                        .publish_error(&wi, &delivery, "payload_error", "failed to resolve item")
                        .await
                    {
                        Ok(_) => match ack(&delivery).await {
                            Ok(()) => self.metrics.messages_acked_total.inc(),
                            Err(e) => {
                                warn!(error = %e, "ack after error-publish failed");
                                self.metrics.jetstream_ack_failures_total.inc();
                            }
                        },
                        Err(_) => {
                            // NATS publish failed — NAK so redelivery
                            // gives another attempt (or surfaces the
                            // error later). Swallowed publish errors
                            // would otherwise silently drop the item.
                            nak_one(&delivery, base_nak_delay_ms(), &self.metrics).await;
                        }
                    }
                    continue;
                }
            };
            resolved.push((wi, delivery, item_json, fetch_ms));
        }
        if resolved.is_empty() {
            return Ok(());
        }

        let batch_items: Vec<EncodeBatchItem> = resolved
            .iter()
            .map(|(wi, _msg, item, fm)| {
                let prepared_tokens = self.maybe_prepare_encode_tokens(model_id, wi, item);
                EncodeBatchItem {
                    work_item_id: wi.work_item_id.clone(),
                    request_id: wi.request_id.clone(),
                    item_index: wi.item_index,
                    total_items: wi.total_items,
                    timestamp: wi.timestamp,
                    item: item.clone(),
                    output_types: wi.output_types.clone(),
                    instruction: wi.instruction.clone(),
                    is_query: wi.is_query,
                    options: wi.options.clone(),
                    profile_id: opt_non_empty(&wi.profile_id),
                    bundle_config_hash: opt_non_empty(&wi.bundle_config_hash),
                    payload_fetch_ms: *fm,
                    prepared_tokens,
                }
            })
            .collect();

        self.metrics.ipc_requests_total.inc();
        let outcome = match self
            .backend
            .process_encode_batch(ProcessEncodeBatchRequest {
                model_id: model_id.to_string(),
                items: batch_items,
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                self.metrics.ipc_failures_total.inc();
                let delay = nak_delay_for_backend_error(&e);
                warn!(
                    model = %model_id,
                    error = %ErrChain(&e),
                    nak_delay_ms = delay,
                    batch_size = resolved.len(),
                    "ProcessEncodeBatch failed — NAKing group"
                );
                let msgs_only: Vec<(WorkItem, Delivery)> =
                    resolved.into_iter().map(|(wi, m, _, _)| (wi, m)).collect();
                nak_all(&msgs_only, delay, &self.metrics).await;
                return Err(e.into());
            }
        };

        self.apply_outcomes(
            outcome,
            resolved
                .into_iter()
                .map(|(wi, m, _, fm)| (wi, m, fm))
                .collect(),
        )
        .await;
        Ok(())
    }

    // -- score ------------------------------------------------------------

    async fn handle_score(
        &self,
        model_id: &str,
        items: Vec<(WorkItem, Delivery)>,
    ) -> Result<(), DispatchError> {
        let model_lbl = self.metrics.model_label(model_id);
        let _timer = self
            .metrics
            .pull_batch_process_seconds
            .with_label_values(&[&model_lbl, "score"])
            .start_timer();
        let mut prepared: Vec<(WorkItem, Delivery, MsgValue, Vec<MsgValue>, f64)> =
            Vec::with_capacity(items.len());
        for (wi, delivery) in items {
            let (query, score_items, fetch_ms) = match self.resolve_score(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        error = %ErrChain(&e),
                        work_item_id = %wi.work_item_id,
                        request_id = %wi.request_id,
                        model = %wi.model_id,
                        "failed to resolve score payload"
                    );
                    match self
                        .publish_error(
                            &wi,
                            &delivery,
                            "payload_error",
                            "failed to resolve score payload",
                        )
                        .await
                    {
                        Ok(_) => match ack(&delivery).await {
                            Ok(()) => self.metrics.messages_acked_total.inc(),
                            Err(e) => {
                                warn!(error = %e, "ack after error-publish failed");
                                self.metrics.jetstream_ack_failures_total.inc();
                            }
                        },
                        Err(_) => {
                            nak_one(&delivery, base_nak_delay_ms(), &self.metrics).await;
                        }
                    }
                    continue;
                }
            };
            prepared.push((wi, delivery, query, score_items, fetch_ms));
        }
        if prepared.is_empty() {
            return Ok(());
        }

        // Rust-tokenisation wire-noop on score: Python's
        // `_process_single_score` does not consume `prepared_tokens`
        // — the cross-encoder adapter tokenises query+doc pairs
        // internally using model-specific pair-building policy
        // (`[CLS] q [SEP] d [SEP]`, pair padding, etc.) that lives
        // adapter-side. We always set `prepared_tokens = None` here
        // so the Python path stays the source of truth. See the
        // score-path note in `docs/architecture-guide.md`.
        let batch_items: Vec<ScoreBatchItem> = prepared
            .iter()
            .map(|(wi, _, q, it, fm)| ScoreBatchItem {
                work_item_id: wi.work_item_id.clone(),
                request_id: wi.request_id.clone(),
                item_index: wi.item_index,
                total_items: wi.total_items,
                timestamp: wi.timestamp,
                query_item: q.clone(),
                score_items: it.clone(),
                instruction: wi.instruction.clone(),
                options: wi.options.clone(),
                profile_id: opt_non_empty(&wi.profile_id),
                payload_fetch_ms: *fm,
                prepared_tokens: None,
            })
            .collect();

        self.metrics.ipc_requests_total.inc();
        let outcome = match self
            .backend
            .process_score_batch(ProcessScoreBatchRequest {
                model_id: model_id.to_string(),
                items: batch_items,
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                self.metrics.ipc_failures_total.inc();
                let delay = nak_delay_for_backend_error(&e);
                warn!(
                    model = %model_id,
                    error = %ErrChain(&e),
                    nak_delay_ms = delay,
                    batch_size = prepared.len(),
                    "ProcessScoreBatch failed — NAKing group"
                );
                let msgs_only: Vec<(WorkItem, Delivery)> = prepared
                    .into_iter()
                    .map(|(wi, m, _, _, _)| (wi, m))
                    .collect();
                nak_all(&msgs_only, delay, &self.metrics).await;
                return Err(e.into());
            }
        };

        self.apply_outcomes(
            outcome,
            prepared
                .into_iter()
                .map(|(wi, m, _, _, fm)| (wi, m, fm))
                .collect(),
        )
        .await;
        Ok(())
    }

    // -- extract ----------------------------------------------------------

    async fn handle_extract(
        &self,
        model_id: &str,
        items: Vec<(WorkItem, Delivery)>,
    ) -> Result<(), DispatchError> {
        let model_lbl = self.metrics.model_label(model_id);
        let _timer = self
            .metrics
            .pull_batch_process_seconds
            .with_label_values(&[&model_lbl, "extract"])
            .start_timer();
        let mut resolved: Vec<(WorkItem, Delivery, MsgValue, f64)> =
            Vec::with_capacity(items.len());
        for (wi, delivery) in items {
            let (item_json, fetch_ms) = match self.resolve_item(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        error = %ErrChain(&e),
                        work_item_id = %wi.work_item_id,
                        request_id = %wi.request_id,
                        model = %wi.model_id,
                        "failed to resolve extract item"
                    );
                    match self
                        .publish_error(&wi, &delivery, "payload_error", "failed to resolve item")
                        .await
                    {
                        Ok(_) => match ack(&delivery).await {
                            Ok(()) => self.metrics.messages_acked_total.inc(),
                            Err(e) => {
                                warn!(error = %e, "ack after error-publish failed");
                                self.metrics.jetstream_ack_failures_total.inc();
                            }
                        },
                        Err(_) => {
                            nak_one(&delivery, base_nak_delay_ms(), &self.metrics).await;
                        }
                    }
                    continue;
                }
            };
            resolved.push((wi, delivery, item_json, fetch_ms));
        }
        if resolved.is_empty() {
            return Ok(());
        }

        let batch_items: Vec<ExtractBatchItem> = resolved
            .iter()
            .map(|(wi, _, item, fm)| ExtractBatchItem {
                work_item_id: wi.work_item_id.clone(),
                request_id: wi.request_id.clone(),
                item_index: wi.item_index,
                total_items: wi.total_items,
                timestamp: wi.timestamp,
                item: item.clone(),
                labels: wi.labels.clone(),
                output_schema: wi.output_schema.clone(),
                instruction: wi.instruction.clone(),
                options: wi.options.clone(),
                profile_id: opt_non_empty(&wi.profile_id),
                bundle_config_hash: opt_non_empty(&wi.bundle_config_hash),
                payload_fetch_ms: *fm,
            })
            .collect();

        self.metrics.ipc_requests_total.inc();
        let outcome = match self
            .backend
            .process_extract_batch(ProcessExtractBatchRequest {
                model_id: model_id.to_string(),
                items: batch_items,
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                self.metrics.ipc_failures_total.inc();
                let delay = nak_delay_for_backend_error(&e);
                warn!(
                    model = %model_id,
                    error = %ErrChain(&e),
                    nak_delay_ms = delay,
                    batch_size = resolved.len(),
                    "ProcessExtractBatch failed — NAKing group"
                );
                let msgs_only: Vec<(WorkItem, Delivery)> =
                    resolved.into_iter().map(|(wi, m, _, _)| (wi, m)).collect();
                nak_all(&msgs_only, delay, &self.metrics).await;
                return Err(e.into());
            }
        };

        self.apply_outcomes(
            outcome,
            resolved
                .into_iter()
                .map(|(wi, m, _, fm)| (wi, m, fm))
                .collect(),
        )
        .await;
        Ok(())
    }

    async fn sleep_or_shutdown(&self, delay_ms: u64) -> bool {
        let delay = Duration::from_millis(delay_ms);
        let Some(shutdown) = self.shutdown.as_ref() else {
            tokio::time::sleep(delay).await;
            return false;
        };
        tokio::select! {
            _ = tokio::time::sleep(delay) => false,
            _ = shutdown.wait() => true,
        }
    }

    // -- outcome / publish helpers ---------------------------------------

    async fn apply_outcomes(
        &self,
        outcome: BatchOutcome,
        resolved: Vec<(WorkItem, Delivery, f64)>,
    ) {
        // Decide which index (if any) each outcome binds to, using the
        // pure `resolve_outcome_indices` helper. Indices are into
        // `resolved`; `None` means "ghost outcome, no matching item".
        let wiids: Vec<&str> = resolved
            .iter()
            .map(|(wi, _, _)| wi.work_item_id.as_str())
            .collect();
        let bindings = resolve_outcome_indices(
            &wiids,
            outcome.outcomes.iter().map(|o| o.work_item_id.as_str()),
        );

        let mut resolved: Vec<Option<(WorkItem, Delivery, f64)>> =
            resolved.into_iter().map(Some).collect();

        for (outcome_idx, item_outcome) in outcome.outcomes.into_iter().enumerate() {
            let Some(idx) = bindings[outcome_idx] else {
                warn!(
                    work_item_id = %item_outcome.work_item_id,
                    "outcome for unknown or already-consumed work_item_id — ignoring"
                );
                continue;
            };
            let Some((wi, delivery, fetch_ms)) = resolved[idx].take() else {
                continue;
            };
            self.apply_outcome(&wi, &delivery, &item_outcome, fetch_ms)
                .await;
        }

        // Any messages left without a corresponding outcome: the executor
        // dropped them. NAK so they get redelivered.
        for slot in resolved.iter_mut() {
            let Some((_wi, delivery, _fm)) = slot.take() else {
                continue;
            };
            warn!(
                origin = %delivery.log_ref(),
                "no outcome from executor — NAKing"
            );
            nak_one(&delivery, base_nak_delay_ms(), &self.metrics).await;
        }
    }

    async fn apply_outcome(
        &self,
        wi: &WorkItem,
        delivery: &Delivery,
        outcome: &ItemOutcome,
        payload_fetch_ms: f64,
    ) {
        match outcome.disposition {
            Disposition::PublishAndAck | Disposition::PublishErrorAndAck => {
                if should_publish(&outcome.disposition) {
                    let queue_ms = queue_ms_from(wi.timestamp);
                    let timings = Some(Timings {
                        queue_ms,
                        payload_fetch_ms,
                    });
                    match self.deliver_result(wi, delivery, outcome, timings).await {
                        Ok(()) => {
                            // Record latency only on the success path —
                            // sampling error-path latency would bias the
                            // FetchExpiry controller toward shrinking the
                            // pull-loop quantum.
                            //
                            // By default `queue_ms_from(wi.timestamp)`
                            // (gateway-publish → NATS-pull) is **excluded**:
                            // including it would feed upstream queue depth
                            // into the tracker that drives the pull-loop
                            // quantum, collapsing the quantum to its floor
                            // under saturation even though the pull-loop
                            // itself isn't the bottleneck. Mirrors the semantics applied
                            // to the scheduler's adaptive-batch tracker
                            // (see `dispatch_batch_inner` per_item_total_ms).
                            //
                            // Operators can opt in to whole-path latency
                            // feedback (queue + inference + postprocess) via
                            // `SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS=1`. See
                            // [`crate::pull_quantum_includes_queue_ms`] and
                            // `docs/architecture-guide.md`.
                            if matches!(outcome.disposition, Disposition::PublishAndAck) {
                                let inference_ms = outcome.inference_ms.unwrap_or(0.0);
                                let postprocess_ms = outcome.postprocessing_ms.unwrap_or(0.0);
                                let mut total = inference_ms + postprocess_ms;
                                if crate::pull_quantum_includes_queue_ms() {
                                    total += queue_ms;
                                }
                                self.latency_tracker.lock().await.record(total);

                                // Prometheus backend timing histograms. Observations
                                // are per-item (one outcome == one source request), so
                                // each completed backend batch contributes N samples
                                // per timing bucket. Only the success path is sampled
                                // to match the rationale above on latency_tracker.
                                //
                                // A zero field means "backend didn't report this phase"
                                // (e.g. tokenize time for image-only models, or
                                // postprocess when no postprocessor ran) — skip those
                                // so the histograms reflect reality rather than
                                // adding a pile of 0-bucket samples that bias p50.
                                //
                                // Cardinality: `operation` is a free-form String on
                                // WorkItem, but PublishAndAck is only produced by the
                                // encode/score/extract code paths (unknown ops route
                                // to publish_error_and_ack and get filtered by the
                                // outer match). We still check against the known set
                                // defensively — a future refactor shouldn't be able
                                // to sneak a new label value into Prometheus without
                                // touching this list. If it's somehow not one of the
                                // three, skip the observation (don't `return` — that
                                // would bypass the ACK below and cause redelivery).
                                if let op @ ("encode" | "score" | "extract") = wi.operation.as_str()
                                {
                                    let model_label =
                                        self.metrics.model_label(&wi.model_id).into_owned();
                                    let record_phase = |phase: &str, ms: f64| {
                                        if ms > 0.0 {
                                            self.metrics
                                                .backend_phase_seconds
                                                .with_label_values(&[
                                                    op,
                                                    model_label.as_str(),
                                                    phase,
                                                ])
                                                .observe(ms / 1_000.0);
                                        }
                                    };
                                    record_phase(
                                        "tokenize",
                                        outcome.tokenization_ms.unwrap_or(0.0),
                                    );
                                    record_phase("inference", inference_ms);
                                    record_phase("postprocess", postprocess_ms);
                                }
                            }
                        }
                        Err(crate::publisher::PublishError::EmptyReplySubject) => {
                            // Fire-and-forget work item — ACK anyway so
                            // JetStream doesn't redeliver forever.
                            debug!(
                                work_item_id = %wi.work_item_id,
                                "skipping publish — empty reply_subject; will still ACK"
                            );
                        }
                        Err(e) => {
                            // NATS publish failed — skip ACK so JetStream
                            // redelivers (caller may still NAK explicitly).
                            warn!(
                                work_item_id = %wi.work_item_id,
                                error = %e,
                                "failed to publish WorkResult — skipping ACK",
                            );
                            return;
                        }
                    }
                }
                match ack(delivery).await {
                    Ok(()) => self.metrics.messages_acked_total.inc(),
                    Err(e) => {
                        warn!(error = %e, "ack failed");
                        self.metrics.jetstream_ack_failures_total.inc();
                    }
                }
            }
            Disposition::NakRetry => {
                nak_one(
                    delivery,
                    outcome.nak_delay_ms.unwrap_or_else(base_nak_delay_ms),
                    &self.metrics,
                )
                .await;
            }
        }
    }

    /// Route one publishable outcome to its delivery-appropriate result
    /// sink: NATS reply-subject publish for [`Delivery::Nats`] (verbatim
    /// legacy behaviour, including the fire-and-forget empty-reply-subject
    /// contract), or an in-process [`crate::delivery::LocalDeliveryEvent`]
    /// for [`Delivery::Local`] — same `WorkResult` bytes either way via
    /// [`shape_and_build_work_result`].
    async fn deliver_result(
        &self,
        wi: &WorkItem,
        delivery: &Delivery,
        outcome: &ItemOutcome,
        timings: Option<Timings>,
    ) -> Result<(), crate::publisher::PublishError> {
        match delivery {
            Delivery::Nats(..) => {
                let Some(publisher) = self.publisher.as_ref() else {
                    // Unreachable by construction (`run()` wires NATS +
                    // publisher together); keep the item redeliverable
                    // rather than sinking it if the invariant ever breaks.
                    warn!(
                        work_item_id = %wi.work_item_id,
                        "NATS delivery without a publisher — skipping publish"
                    );
                    return Err(crate::publisher::PublishError::NoPublisher);
                };
                publisher
                    .publish_result(
                        &wi.reply_subject,
                        outcome,
                        timings,
                        delivery.worker_direct(),
                    )
                    .await
            }
            Delivery::Local(local) => {
                // reply_subject is NATS-only; local results always ride the
                // ingest socket back to the caller that is awaiting them.
                let result = shape_and_build_work_result(
                    outcome,
                    &self.worker_id,
                    timings,
                    delivery.worker_direct(),
                );
                if !local.send_result(result) {
                    debug!(
                        work_item_id = %wi.work_item_id,
                        origin = %delivery.log_ref(),
                        "local ingest caller gone — dropping WorkResult"
                    );
                }
                Ok(())
            }
        }
    }

    /// Publish a synthetic error `WorkResult` on `wi.reply_subject`
    /// ([`Delivery::Nats`]) or straight to the local ingest caller
    /// ([`Delivery::Local`]).
    ///
    /// Returns:
    /// * `Ok(true)` — published (or fire-and-forget: empty reply_subject).
    ///   The caller may safely ACK the NATS message.
    /// * `Ok(false)` — (reserved) never returned today, kept for future
    ///   cases where the caller should NAK without logging.
    /// * `Err(_)` — NATS publish itself failed. The caller MUST NOT ACK:
    ///   the client never got the error reply, so we rely on redelivery
    ///   to give another worker (or this one, later) a chance to surface
    ///   the failure.
    async fn publish_error(
        &self,
        wi: &WorkItem,
        delivery: &Delivery,
        code: &str,
        message: &str,
    ) -> Result<bool, crate::publisher::PublishError> {
        let outcome = synthetic_error_outcome(wi, code, message);
        match self.deliver_result(wi, delivery, &outcome, None).await {
            Ok(()) => Ok(true),
            Err(crate::publisher::PublishError::EmptyReplySubject) => {
                // Fire-and-forget work item. No one is waiting for the
                // error; ACKing lets JetStream drop it on the floor,
                // which is the right behaviour.
                debug!(work_item_id = %wi.work_item_id, "skipping error publish — empty reply_subject");
                Ok(true)
            }
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    error = %e,
                    "failed to publish error WorkResult"
                );
                Err(e)
            }
        }
    }

    /// Dead-letter a whole (op, model) group on a TERMINAL load failure:
    /// publish a typed `MODEL_LOAD_FAILED` error `WorkResult` for every item
    /// and ACK it. This is the [`ReadinessState::Failed`] fast-path twin of
    /// the batch/`run_batch` `MODEL_LOAD_FAILED` mapping (#1786) — the gateway
    /// turns the code into an HTTP 502 so the client fails fast instead of
    /// blocking while the sidecar re-drives a doomed model forever.
    ///
    /// Per-item settlement mirrors [`Self::apply_outcome`] for a
    /// `PublishErrorAndAck` outcome: ACK once the error is published (or the
    /// item is fire-and-forget); if the NATS publish itself fails, NAK so
    /// JetStream redelivers and another attempt can surface the failure.
    async fn dead_letter_all(&self, items: &[(WorkItem, Delivery)], model_id: &str) {
        let message = format!("Model '{model_id}' failed to load permanently.");
        for (wi, delivery) in items {
            match self
                .publish_error(wi, delivery, MODEL_LOAD_FAILED_ERROR_CODE, &message)
                .await
            {
                Ok(_) => match ack(delivery).await {
                    Ok(()) => self.metrics.messages_acked_total.inc(),
                    Err(e) => {
                        warn!(
                            work_item_id = %wi.work_item_id,
                            error = %e,
                            "ack after MODEL_LOAD_FAILED publish failed"
                        );
                        self.metrics.jetstream_ack_failures_total.inc();
                    }
                },
                Err(_) => {
                    // Publish failed — the client never got the typed error,
                    // so rely on redelivery rather than ACKing it away.
                    nak_one(delivery, base_nak_delay_ms(), &self.metrics).await;
                }
            }
        }
        debug!(
            count = items.len(),
            model = %model_id,
            "dead-lettered group as MODEL_LOAD_FAILED"
        );
    }

    /// Generate-path twin of [`Self::publish_error`]. The generate flow is
    /// NATS-only (local-ingest generate items are rejected before reaching
    /// it, see [`Self::dispatch_decoded`]) and holds its [`Message`] in an
    /// `Arc` for the event stream, so it cannot wrap one into a
    /// [`Delivery`]; publish straight through the NATS publisher.
    async fn publish_error_generate(
        &self,
        wi: &WorkItem,
        code: &str,
        message: &str,
        worker_direct: bool,
    ) -> Result<bool, crate::publisher::PublishError> {
        let Some(publisher) = self.publisher.as_ref() else {
            return Err(crate::publisher::PublishError::NoPublisher);
        };
        let outcome = synthetic_error_outcome(wi, code, message);
        match publisher
            .publish_result(&wi.reply_subject, &outcome, None, worker_direct)
            .await
        {
            Ok(()) => Ok(true),
            Err(crate::publisher::PublishError::EmptyReplySubject) => {
                debug!(work_item_id = %wi.work_item_id, "skipping error publish — empty reply_subject");
                Ok(true)
            }
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    error = %e,
                    "failed to publish error WorkResult"
                );
                Err(e)
            }
        }
    }

    /// Publish a terminal generation chunk error on `wi.reply_subject`.
    ///
    /// Generation requests are tracked by the gateway's streaming collector, so
    /// a one-shot `WorkResult` would not unblock the client. Use the same chunk
    /// envelope Python's `StreamingProcessor` emits for terminal pre-execution
    /// errors.
    async fn publish_generate_terminal_error(
        &self,
        wi: &WorkItem,
        code: &str,
        message: &str,
    ) -> Result<bool, crate::publisher::PublishError> {
        if wi.reply_subject.is_empty() {
            debug!(
                work_item_id = %wi.work_item_id,
                "skipping generate terminal error publish — empty reply_subject"
            );
            return Ok(true);
        }
        let Some(publisher) = self.publisher.as_ref() else {
            return Err(crate::publisher::PublishError::NoPublisher);
        };
        let payload = encode_generate_terminal_error_chunk(wi, code, message)?;
        match publisher.publish_raw(&wi.reply_subject, payload).await {
            Ok(()) => Ok(true),
            Err(crate::publisher::PublishError::EmptyReplySubject) => {
                debug!(
                    work_item_id = %wi.work_item_id,
                    "skipping generate terminal error publish — empty reply_subject"
                );
                Ok(true)
            }
            Err(e) => {
                warn!(
                    work_item_id = %wi.work_item_id,
                    error = %e,
                    "failed to publish generate terminal error chunk"
                );
                Err(e)
            }
        }
    }

    // -- scheduler -------------------------------------------------------

    /// Return the per-model [`ProductionScheduler`] when
    /// [`Self::scheduler_registry`] is present. `None` means
    /// "legacy path: submit straight to `process_*_batch`" — used
    /// only in unit tests that don't wire the scheduler.
    ///
    /// Lazily materialises the scheduler on first touch. When a new
    /// one is created, also spawns that model's drain loop so the
    /// submitted items get consumed — that's the counterpart to the
    /// old eager-at-startup `spawn_scheduler_drains`. Schedulers are
    /// now materialised only for active models that land on a sidecar
    /// worker, so boot does not need a model list to iterate.
    async fn resolve_scheduler(
        self: &Arc<Self>,
        model_id: &str,
    ) -> Option<Arc<ProductionScheduler>> {
        let registry = self.scheduler_registry.as_ref()?;
        let shutdown = self.shutdown.as_ref()?;
        let (sched, created) = registry.get_or_create(model_id).await;
        if created {
            let mut handles = self.scheduler_drain_handles.lock().await;
            // Double-check under the lock: a concurrent `resolve_scheduler`
            // for the same model could have won the `get_or_create` race
            // and already inserted a handle. Without this guard we'd spawn
            // two drain loops racing the same scheduler queue.
            if !handles.contains_key(model_id) {
                let disp = Arc::clone(self);
                let sched_c = Arc::clone(&sched);
                let shutdown_c = Arc::clone(shutdown);
                let model_id_s = model_id.to_owned();
                let model_id_log = model_id.to_owned();
                let handle = tokio::spawn(async move {
                    scheduler_drain_loop(model_id_s, disp, sched_c, shutdown_c).await;
                });
                handles.insert(model_id.to_owned(), handle);
                // Increment the live `models_total` gauge the first time
                // we materialise a scheduler for this model. Dashboards
                // and the shutdown log both read this; we deliberately
                // keep it monotonic-per-process (schedulers never get
                // dropped mid-run) so a decrement path isn't needed.
                self.metrics.scheduler.models_total.inc();
                info!(
                    model = %model_id_log,
                    "rust-scheduler: drain loop spawned on first traffic",
                );
            }
        }
        Some(sched)
    }

    /// Remove and return every scheduler drain handle registered so
    /// far. Called at shutdown from `lib.rs` so the main shutdown
    /// path can `await` each task to completion (bounded inside the
    /// loop by `DEFAULT_SCHEDULER_DRAIN_DEADLINE_MS`, overridable via
    /// `SIE_SCHEDULER_DRAIN_DEADLINE_MS`).
    pub async fn take_scheduler_drain_handles(&self) -> Vec<JoinHandle<()>> {
        let mut guard = self.scheduler_drain_handles.lock().await;
        guard.drain().map(|(_, h)| h).collect()
    }

    /// Remove and return all in-flight generation task handles. Called
    /// during shutdown after the pull loops stop so long-running streams can
    /// settle before the backend drain RPC closes backend-side state.
    pub async fn take_generation_handles(&self) -> Vec<JoinHandle<()>> {
        let mut guard = self.generation_handles.lock().await;
        guard.drain(..).collect()
    }

    /// Resolve every encode item's payload then submit it into the
    /// model scheduler under its `options["lora"]` key. Items whose
    /// payload resolution fails follow the same publish_error + ACK
    /// (or NAK on publish failure) path as [`Self::handle_encode`];
    /// the scheduler never sees them.
    ///
    /// Returns immediately once all items are enqueued — the drain
    /// loop owns the actual backend call + outcome publish.
    async fn enqueue_encode_into_scheduler(
        &self,
        model_id: &str,
        scheduler: &Arc<ProductionScheduler>,
        items: Vec<(WorkItem, Delivery)>,
    ) {
        let mut grouped: HashMap<LoraKey, Vec<(SchedulerItem, SchedulerMeta)>> = HashMap::new();
        for (wi, delivery) in items {
            let (item_json, fetch_ms) = match self.resolve_item(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    self.fail_resolve(&wi, &delivery, &e, "failed to resolve encode item")
                        .await;
                    continue;
                }
            };
            let prepared_tokens = self.maybe_prepare_encode_tokens(model_id, &wi, &item_json);
            let lora = lora_from_options(&wi.options);
            let ebi = EncodeBatchItem {
                work_item_id: wi.work_item_id.clone(),
                request_id: wi.request_id.clone(),
                item_index: wi.item_index,
                total_items: wi.total_items,
                timestamp: wi.timestamp,
                item: item_json,
                output_types: wi.output_types.clone(),
                instruction: wi.instruction.clone(),
                is_query: wi.is_query,
                options: wi.options.clone(),
                profile_id: opt_non_empty(&wi.profile_id),
                bundle_config_hash: opt_non_empty(&wi.bundle_config_hash),
                payload_fetch_ms: fetch_ms,
                prepared_tokens,
            };
            let item = SchedulerItem::Encode(ebi);
            let child_index = self.record_scheduler_enqueue(model_id, &item);
            let worker_direct = delivery.worker_direct();
            let meta = SchedulerMeta::new_with_worker_direct(wi, delivery, fetch_ms, worker_direct)
                .with_worker_child_index(child_index);
            grouped.entry(lora).or_default().push((item, meta));
        }
        let enqueued = grouped.values().map(Vec::len).sum::<usize>();
        if enqueued > 0 {
            self.metrics
                .scheduler
                .enqueued_items_total
                .with_label_values(&[&self.metrics.model_label(model_id), "encode"])
                .inc_by(enqueued as u64);
        }
        for (lora, grouped_items) in grouped {
            scheduler
                .submit_many(SchedOp::Encode, lora, grouped_items)
                .await;
        }
    }

    /// Score twin of [`Self::enqueue_encode_into_scheduler`]. See
    /// [`Self::handle_score`] for the Rust-tokenisation note on
    /// `prepared_tokens` being `None` (cross-encoder tokenisation
    /// stays Python-side for now).
    ///
    /// Routing policy: score always goes to `LoraKey::base`
    /// regardless of what's on `options["lora"]`. That's enforced
    /// inside [`crate::scheduler::Scheduler::submit`] so the call
    /// here passes the parsed key through transparently.
    async fn enqueue_score_into_scheduler(
        &self,
        model_id: &str,
        scheduler: &Arc<ProductionScheduler>,
        items: Vec<(WorkItem, Delivery)>,
    ) {
        let mut grouped: Vec<(SchedulerItem, SchedulerMeta)> = Vec::new();
        for (wi, delivery) in items {
            let (query, score_items, fetch_ms) = match self.resolve_score(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    self.fail_resolve(&wi, &delivery, &e, "failed to resolve score payload")
                        .await;
                    continue;
                }
            };
            let sbi = ScoreBatchItem {
                work_item_id: wi.work_item_id.clone(),
                request_id: wi.request_id.clone(),
                item_index: wi.item_index,
                total_items: wi.total_items,
                timestamp: wi.timestamp,
                query_item: query,
                score_items,
                instruction: wi.instruction.clone(),
                options: wi.options.clone(),
                profile_id: opt_non_empty(&wi.profile_id),
                payload_fetch_ms: fetch_ms,
                prepared_tokens: None,
            };
            let item = SchedulerItem::Score(sbi);
            let child_index = self.record_scheduler_enqueue(model_id, &item);
            let worker_direct = delivery.worker_direct();
            let meta = SchedulerMeta::new_with_worker_direct(wi, delivery, fetch_ms, worker_direct)
                .with_worker_child_index(child_index);
            grouped.push((item, meta));
        }
        if !grouped.is_empty() {
            self.metrics
                .scheduler
                .enqueued_items_total
                .with_label_values(&[&self.metrics.model_label(model_id), "score"])
                .inc_by(grouped.len() as u64);
            scheduler
                .submit_many(SchedOp::Score, LoraKey::base(), grouped)
                .await;
        }
    }

    /// Extract twin of [`Self::enqueue_encode_into_scheduler`].
    /// Extract items don't emit `prepared_tokens` on the Rust side
    /// (Python owns extract tokenisation in v1), so the outgoing
    /// [`ExtractBatchItem`] matches the current
    /// [`Self::handle_extract`] shape.
    async fn enqueue_extract_into_scheduler(
        &self,
        model_id: &str,
        scheduler: &Arc<ProductionScheduler>,
        items: Vec<(WorkItem, Delivery)>,
    ) {
        let mut grouped: HashMap<LoraKey, Vec<(SchedulerItem, SchedulerMeta)>> = HashMap::new();
        for (wi, delivery) in items {
            let (item_json, fetch_ms) = match self.resolve_item(&wi).await {
                Ok(v) => v,
                Err(e) => {
                    self.fail_resolve(&wi, &delivery, &e, "failed to resolve extract item")
                        .await;
                    continue;
                }
            };
            let lora = lora_from_options(&wi.options);
            let xbi = ExtractBatchItem {
                work_item_id: wi.work_item_id.clone(),
                request_id: wi.request_id.clone(),
                item_index: wi.item_index,
                total_items: wi.total_items,
                timestamp: wi.timestamp,
                item: item_json,
                labels: wi.labels.clone(),
                output_schema: wi.output_schema.clone(),
                instruction: wi.instruction.clone(),
                options: wi.options.clone(),
                profile_id: opt_non_empty(&wi.profile_id),
                bundle_config_hash: opt_non_empty(&wi.bundle_config_hash),
                payload_fetch_ms: fetch_ms,
            };
            let item = SchedulerItem::Extract(xbi);
            let child_index = self.record_scheduler_enqueue(model_id, &item);
            let worker_direct = delivery.worker_direct();
            let meta = SchedulerMeta::new_with_worker_direct(wi, delivery, fetch_ms, worker_direct)
                .with_worker_child_index(child_index);
            grouped.entry(lora).or_default().push((item, meta));
        }
        let enqueued = grouped.values().map(Vec::len).sum::<usize>();
        if enqueued > 0 {
            self.metrics
                .scheduler
                .enqueued_items_total
                .with_label_values(&[&self.metrics.model_label(model_id), "extract"])
                .inc_by(enqueued as u64);
        }
        for (lora, grouped_items) in grouped {
            scheduler
                .submit_many(SchedOp::Extract, lora, grouped_items)
                .await;
        }
    }

    fn record_scheduler_enqueue(&self, model_id: &str, item: &SchedulerItem) -> usize {
        let cost = item.cost();
        self.metrics.worker_queue_depth.inc();
        self.metrics.worker_pending_cost.add(clamp_u64_to_i64(cost));
        self.worker_pool
            .record_model_pending_enqueue(model_id, cost)
    }

    /// Shared failure tail for the three scheduler-enqueue paths.
    /// Mirrors the payload-resolve error branch in the legacy
    /// [`Self::handle_encode`] / [`Self::handle_score`] /
    /// [`Self::handle_extract`] flows: publish a synthetic
    /// `payload_error` WorkResult then ACK; if the publish itself
    /// fails, NAK so JetStream redelivers.
    async fn fail_resolve(
        &self,
        wi: &WorkItem,
        delivery: &Delivery,
        err: &PayloadError,
        log_msg: &str,
    ) {
        warn!(
            error = %ErrChain(err),
            work_item_id = %wi.work_item_id,
            request_id = %wi.request_id,
            model = %wi.model_id,
            "{log_msg}"
        );
        match self
            .publish_error(wi, delivery, "payload_error", "failed to resolve item")
            .await
        {
            Ok(_) => match ack(delivery).await {
                Ok(()) => self.metrics.messages_acked_total.inc(),
                Err(e) => {
                    warn!(error = %e, "ack after error-publish failed");
                    self.metrics.jetstream_ack_failures_total.inc();
                }
            },
            Err(_) => {
                nak_one(delivery, base_nak_delay_ms(), &self.metrics).await;
            }
        }
    }

    // -- payload resolution ----------------------------------------------

    async fn resolve_item(&self, wi: &WorkItem) -> Result<(MsgValue, f64), PayloadError> {
        if let Some(item) = &wi.item {
            return Ok((item.clone(), 0.0));
        }
        let Some(payload_ref) = &wi.payload_ref else {
            return Err(PayloadError::InvalidRef(format!(
                "work item {} has neither item nor payload_ref",
                wi.work_item_id
            )));
        };
        let start = std::time::Instant::now();
        let bytes = self.payload_store.get(payload_ref).await?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let item: MsgValue = rmp_serde::from_slice(&bytes)
            .map_err(|e| PayloadError::InvalidRef(format!("decode payload: {e}")))?;
        Ok((item, ms))
    }

    // -- Rust-side tokenisation -------------------------------------------
    //
    // `maybe_prepare_encode_tokens` consults the tokenizer registry
    // and returns `Some(PreparedTokens)` when the v2 safety rules
    // hold:
    //
    //   1. Registry has an entry for `model_id` (the adapter declared
    //      a tokeniser on `EnsureModelReady`).
    //   2. The msgpack-native `item` payload has a populated string `text`
    //      field. Image / audio / multimodal items fall through to
    //      Python as today.
    //
    // The registry-backed path no longer bails out on `is_query=true`
    // or `instruction!=""`: when a model has shipped its template
    // defaults via `ModelDescriptor.default_query_template` /
    // `default_doc_template`, the sidecar applies the template via
    // [`crate::prep::text_prep::TextPrep`] before tokenising — bit-exact
    // with Python's `_utils.extract_texts` for the two known
    // placeholders. Per-request `options.query_template` /
    // `options.doc_template` overrides still win.
    //
    // Any tokenise error at runtime returns `None` and the Python
    // adapter tokenises from `item` exactly like today. Failures are
    // logged at `debug` so they don't drown out real incidents.
    //
    // Score path: there is no Rust-side fast path. The cross-encoder
    // adapter on the Python side owns pair-building + tokenisation
    // (model-specific `[CLS] q [SEP] d [SEP]` policies plus pair
    // padding), so Rust always sets `ScoreBatchItem.prepared_tokens =
    // None`. Re-introducing a `maybe_prepare_score_tokens` helper
    // is straightforward when the Python score path grows a
    // `prepared_tokens` consumer; until then the dead helper has
    // been removed to keep the surface honest.
    fn maybe_prepare_encode_tokens(
        &self,
        model_id: &str,
        wi: &WorkItem,
        item: &MsgValue,
    ) -> Option<PreparedTokens> {
        let entry = self.tokenizer_registry.get(model_id)?;

        // Text-only inputs. Treat absent / non-string / empty text as
        // "not a fast-path request" and defer to Python. An empty
        // string would tokenise to a 2-token `[CLS][SEP]` padding
        // sequence — harmless but pure IPC overhead vs letting Python
        // short-circuit on its own empty-text guard.
        let raw_text = msg_map_get(item, "text")
            .and_then(msg_as_str)
            .filter(|s| !s.is_empty())?;

        // Resolve per-request template overrides; fall back to the
        // adapter's defaults from the handshake. Same precedence as
        // Python's `resolve_embedding_options`.
        let (query_template, doc_template) = crate::prep::text_prep::extract_templates_from_options(
            wi.options.as_ref(),
            entry.default_query_template(),
            entry.default_doc_template(),
        );

        // Apply the template / instruction transform. Borrowing-style
        // `apply` so plain (non-templated, non-instructed) text is a
        // no-op pass-through with no allocation beyond the input.
        let prep = crate::prep::text_prep::TextPrep {
            instruction: wi.instruction.as_deref(),
            is_query: wi.is_query,
            query_template,
            doc_template,
        };
        let prepared_text = prep.apply(raw_text);
        let text: &str = prepared_text.as_str();

        let rag = match entry.tokenize(&[text]) {
            Ok(r) if r.len() == 1 => r,
            Ok(_) => return None, // empty/unexpected — defer
            Err(e) => {
                tracing::debug!(
                    model = %model_id,
                    error = %e,
                    "rust-tokenize: encode tokenise failed; deferring to Python"
                );
                return None;
            }
        };

        Some(rag_to_wire(entry.tokenizer_id(), entry.max_seq_len(), rag))
    }

    async fn resolve_score(
        &self,
        wi: &WorkItem,
    ) -> Result<(MsgValue, Vec<MsgValue>, f64), PayloadError> {
        // Inline path: both query + items provided on the WorkItem.
        if let (Some(q), Some(items)) = (&wi.query_item, &wi.score_items) {
            return Ok((q.clone(), items.clone(), 0.0));
        }

        // Offloaded path: query_payload_ref points at a msgpack-encoded
        // `{"query": ..., "items": [...]}` blob.
        let Some(ref_key) = &wi.query_payload_ref else {
            return Err(PayloadError::InvalidRef(format!(
                "score item {} missing query/items and query_payload_ref",
                wi.work_item_id
            )));
        };
        let start = std::time::Instant::now();
        let bytes = self.payload_store.get(ref_key).await?;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let decoded: MsgValue = rmp_serde::from_slice(&bytes)
            .map_err(|e| PayloadError::InvalidRef(format!("decode score payload: {e}")))?;
        let query = msg_map_get(&decoded, "query")
            .cloned()
            .ok_or_else(|| PayloadError::InvalidRef("score payload missing 'query'".into()))?;
        let items = match msg_map_get(&decoded, "items") {
            Some(MsgValue::Array(items)) => items.clone(),
            _ => {
                return Err(PayloadError::InvalidRef(
                    "score payload missing 'items' array".into(),
                ));
            }
        };
        Ok((query, items, ms))
    }
}

// -----------------------------------------------------------------------------
// Grouping (pure logic, tested independently)
// -----------------------------------------------------------------------------

/// Group by `(model_id, operation)`. Retained for tests and callers
/// that want explicit per-op grouping; the hot path uses
/// [`group_by_model_only`] so encode/score/extract for the same model
/// can run concurrently under a single readiness check.
pub fn group_by_model<T>(
    items: Vec<(WorkItem, T)>,
) -> BTreeMap<(String, String), Vec<(WorkItem, T)>> {
    let mut groups: BTreeMap<(String, String), Vec<(WorkItem, T)>> = BTreeMap::new();
    for (wi, extra) in items {
        let key = (wi.model_id.clone(), wi.operation.clone());
        groups.entry(key).or_default().push((wi, extra));
    }
    groups
}

/// Group decoded messages by `model_id` only — hot-path grouping.
pub fn group_by_model_only<T>(items: Vec<(WorkItem, T)>) -> BTreeMap<String, Vec<(WorkItem, T)>> {
    let mut groups: BTreeMap<String, Vec<(WorkItem, T)>> = BTreeMap::new();
    for (wi, extra) in items {
        let key = wi.model_id.clone();
        groups.entry(key).or_default().push((wi, extra));
    }
    groups
}

/// Split a per-model group at the batch budget. First `budget` items go
/// to dispatch, the rest to overflow (the caller NAKs with a short delay).
pub(crate) type DispatchSplit<T> = (Vec<(WorkItem, T)>, Vec<(WorkItem, T)>);

pub(crate) fn split_by_budget<T>(items: Vec<(WorkItem, T)>, budget: usize) -> DispatchSplit<T> {
    if items.len() <= budget {
        return (items, Vec::new());
    }
    let mut it = items.into_iter();
    let dispatch: Vec<(WorkItem, T)> = it.by_ref().take(budget).collect();
    let overflow: Vec<(WorkItem, T)> = it.collect();
    (dispatch, overflow)
}

/// Compute `queue_ms` for a `WorkItem` given its embedded `timestamp`
/// (unix seconds, set by the gateway publisher). Returns 0 for
/// missing/future timestamps so we never report a negative latency.
/// Compute one `total_ms` sample per *unique* `request_id` in the
/// batch. Mirrors Python's `_complete_requests` dedup pattern (see
/// `model_worker.py:947-976` on main `bbe409c3`):
///
/// ```python
/// completed_metadata: set[int] = set()
/// for metadata in batch.metadata:
///     meta_id = id(metadata)
///     if meta_id in completed_metadata:
///         continue
///     completed_metadata.add(meta_id)
///     ...
///     self._latency_tracker.record(metadata.timing.total_ms)
/// ```
///
/// In the Rust scheduler each NATS message is its own
/// [`SchedulerMeta`] with its own `submitted_at`; when the gateway
/// fans out a multi-item client request it produces N work-items with
/// the same `request_id` and distinct `item_index`es. We pick the
/// **first** occurrence's `submitted_at` (lowest `item_index` is not
/// guaranteed because the BatchFormer sorts by cost, but all items in
/// a request share the same gateway publish time so any of them is a
/// fair stand-in for Python's request-level `_start_time`).
///
/// Items are skipped if:
///   * the IPC reply marked them anything other than `PublishAndAck`
///     (errors / NAK paths shouldn't bias the controller toward fast
///     no-op replies);
///   * `total_ms` came out non-positive — represents IPC error paths
///     where the per-item timing fields were left at zero. Letting
///     these in would bias `observed_p50_ms` toward zero and pin the
///     wait knob at the floor.
///
/// Takes an iterator of `(request_id, submitted_at, outcome)` triples
/// rather than a `&[SchedulerMeta]` so unit tests don't need to
/// fabricate a `jetstream::Message` (which has no public constructor).
fn dedupe_per_request_totals<'a, I>(rows: I, now: Instant) -> Vec<f64>
where
    I: IntoIterator<Item = (&'a str, Instant, &'a ItemOutcome)>,
{
    let iter = rows.into_iter();
    let (lower, _) = iter.size_hint();
    let mut seen: HashSet<&str> = HashSet::with_capacity(lower);
    let mut out: Vec<f64> = Vec::with_capacity(lower);
    for (request_id, submitted_at, o) in iter {
        if !matches!(o.disposition, Disposition::PublishAndAck) {
            continue;
        }
        if !seen.insert(request_id) {
            continue;
        }
        // Saturating subtract: in tests / replays where
        // `submitted_at` could be set in the future relative to
        // `now`, treat it as zero wait rather than panicking.
        let batcher_wait_ms = now.saturating_duration_since(submitted_at).as_secs_f64() * 1000.0;
        let inf = o.inference_ms.unwrap_or(0.0);
        let post = o.postprocessing_ms.unwrap_or(0.0);
        let total = batcher_wait_ms + inf + post;
        if total > 0.0 {
            out.push(total);
        }
    }
    out
}

fn queue_ms_from(timestamp_s: f64) -> f64 {
    if timestamp_s <= 0.0 {
        return 0.0;
    }
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(timestamp_s);
    let delta_ms = (now_s - timestamp_s) * 1000.0;
    if delta_ms < 0.0 {
        0.0
    } else {
        delta_ms
    }
}

/// Convert a [`crate::tokenize::RaggedTokens`] bundle into the wire
/// [`PreparedTokens`] form. Wrapping here (rather than inline at each
/// call site) keeps the "what to elide to save bytes" policy in one
/// place: BERT-style all-zero `token_type_ids` skip the wire, since
/// Python treats an empty outer vec as "all zeros".
fn rag_to_wire(
    tokenizer_id: &str,
    max_seq_len: usize,
    rag: crate::tokenize::RaggedTokens,
) -> PreparedTokens {
    let token_type_ids = if rag.token_type_ids_all_zero() {
        Vec::new()
    } else {
        rag.token_type_ids
    };
    PreparedTokens {
        input_ids: rag.input_ids,
        attention_mask: rag.attention_mask,
        token_type_ids,
        tokenizer_id: tokenizer_id.to_string(),
        max_seq_len: max_seq_len as u32,
    }
}

/// Synthetic error `ItemOutcome` for pre-execution failures (payload
/// resolution, bad operation, offload decode). Shared by the NATS and
/// local delivery error paths so both emit the identical wire shape.
/// No timings: error-only publishes omit queue_ms / processing_ms /
/// payload_fetch_ms.
fn synthetic_error_outcome(wi: &WorkItem, code: &str, message: &str) -> ItemOutcome {
    ItemOutcome {
        work_item_id: wi.work_item_id.clone(),
        request_id: wi.request_id.clone(),
        item_index: wi.item_index,
        disposition: Disposition::PublishErrorAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: Some(message.to_string()),
        error_code: Some(code.to_string()),
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None,
        units: None,
    }
}

fn opt_non_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((idx, _)) => &s[..idx],
        None => s,
    }
}

fn encode_generate_terminal_error_chunk(
    wi: &WorkItem,
    code: &str,
    message: &str,
) -> Result<Vec<u8>, rmp_serde::encode::Error> {
    let chunk = GenerateTerminalErrorChunk {
        kind: "chunk",
        request_id: &wi.request_id,
        attempt_id: format!("{}:model-loading", wi.work_item_id),
        seq: 0,
        text_delta: "",
        done: true,
        finish_reason: "error",
        error: GenerateTerminalError { code, message },
    };
    rmp_serde::to_vec_named(&chunk)
}

async fn handle_generate_event(
    event: GenerateEvent,
    publisher: Arc<WorkPublisher>,
    metrics: Arc<MetricsRegistry>,
    settled: Arc<AtomicBool>,
    msg: Arc<QueuedMessage>,
    delivery_log: Arc<GenerateDeliveryLogContext>,
) -> Result<(), DispatchError> {
    match event.kind.as_str() {
        "publish" => {
            publisher
                .publish_raw(&event.reply_subject, event.payload)
                .await?;
        }
        "ack" => {
            if !settled.swap(true, Ordering::SeqCst) {
                match ack_msg(&msg).await {
                    Ok(()) => {
                        info!(
                            work_item_id = %delivery_log.work_item_id,
                            request_id = %delivery_log.request_id,
                            model = %delivery_log.model_id,
                            subject = %delivery_log.delivery.subject,
                            stream = %delivery_log.delivery.stream,
                            consumer = %delivery_log.delivery.consumer,
                            stream_seq = delivery_log.delivery.stream_sequence,
                            consumer_seq = delivery_log.delivery.consumer_sequence,
                            delivery_count = delivery_log.delivery.delivered,
                            pending = delivery_log.delivery.pending,
                            "generate delivery ACKed"
                        );
                        metrics.messages_acked_total.inc();
                    }
                    Err(e) => {
                        warn!(
                            work_item_id = %delivery_log.work_item_id,
                            request_id = %delivery_log.request_id,
                            model = %delivery_log.model_id,
                            subject = %delivery_log.delivery.subject,
                            stream = %delivery_log.delivery.stream,
                            consumer = %delivery_log.delivery.consumer,
                            stream_seq = delivery_log.delivery.stream_sequence,
                            consumer_seq = delivery_log.delivery.consumer_sequence,
                            delivery_count = delivery_log.delivery.delivered,
                            pending = delivery_log.delivery.pending,
                            error = %e,
                            "generate ACK failed"
                        );
                        metrics.jetstream_ack_failures_total.inc();
                    }
                }
            }
        }
        "nak" => {
            if !settled.swap(true, Ordering::SeqCst) {
                let delay_ms = event.delay_ms.unwrap_or_else(base_nak_delay_ms);
                warn!(
                    work_item_id = %delivery_log.work_item_id,
                    request_id = %delivery_log.request_id,
                    model = %delivery_log.model_id,
                    delay_ms,
                    subject = %delivery_log.delivery.subject,
                    stream = %delivery_log.delivery.stream,
                    consumer = %delivery_log.delivery.consumer,
                    stream_seq = delivery_log.delivery.stream_sequence,
                    consumer_seq = delivery_log.delivery.consumer_sequence,
                    delivery_count = delivery_log.delivery.delivered,
                    pending = delivery_log.delivery.pending,
                    "generate delivery NAKed by Python"
                );
                nak_msg(&msg, delay_ms, &metrics).await;
            }
        }
        "in_progress" => {
            if !settled.load(Ordering::SeqCst) {
                match msg.ack_with(async_nats::jetstream::AckKind::Progress).await {
                    Ok(()) => {
                        debug!(
                            work_item_id = %delivery_log.work_item_id,
                            request_id = %delivery_log.request_id,
                            model = %delivery_log.model_id,
                            subject = %delivery_log.delivery.subject,
                            stream = %delivery_log.delivery.stream,
                            consumer = %delivery_log.delivery.consumer,
                            stream_seq = delivery_log.delivery.stream_sequence,
                            consumer_seq = delivery_log.delivery.consumer_sequence,
                            delivery_count = delivery_log.delivery.delivered,
                            pending = delivery_log.delivery.pending,
                            "generate delivery progress ACKed"
                        );
                    }
                    Err(e) => {
                        debug!(
                            work_item_id = %delivery_log.work_item_id,
                            request_id = %delivery_log.request_id,
                            model = %delivery_log.model_id,
                            subject = %delivery_log.delivery.subject,
                            stream = %delivery_log.delivery.stream,
                            consumer = %delivery_log.delivery.consumer,
                            stream_seq = delivery_log.delivery.stream_sequence,
                            consumer_seq = delivery_log.delivery.consumer_sequence,
                            delivery_count = delivery_log.delivery.delivered,
                            pending = delivery_log.delivery.pending,
                            error = %e,
                            "generate in-progress ACK failed"
                        );
                        metrics.jetstream_ack_failures_total.inc();
                    }
                }
            }
        }
        other => {
            warn!(event = %other, "unknown ProcessGenerate event from Python");
        }
    }
    Ok(())
}

async fn ack(delivery: &Delivery) -> Result<(), DispatchError> {
    delivery.ack().await.map_err(DispatchError::Ack)
}

async fn nak_all(items: &[(WorkItem, Delivery)], delay_ms: u64, metrics: &MetricsRegistry) {
    for (_, d) in items {
        nak_one(d, delay_ms, metrics).await;
    }
    debug!(count = items.len(), delay_ms, "NAKed group");
}

async fn nak_one(delivery: &Delivery, delay_ms: u64, metrics: &MetricsRegistry) {
    match delivery.nak(delay_ms).await {
        Ok(()) => metrics.messages_naked_total.inc(),
        Err(e) => {
            warn!(error = %e, "nak failed");
            metrics.jetstream_nak_failures_total.inc();
        }
    }
}

async fn progress_all(items: &[(WorkItem, Delivery)], metrics: &MetricsRegistry) -> bool {
    let mut all_ok = true;
    for (_, d) in items {
        if !progress_one(d, metrics).await {
            all_ok = false;
        }
    }
    if all_ok {
        debug!(count = items.len(), "progress ACKed group");
    }
    all_ok
}

async fn progress_one(delivery: &Delivery, metrics: &MetricsRegistry) -> bool {
    match delivery.progress().await {
        Ok(()) => true,
        Err(e) => {
            warn!(error = %e, "progress ack failed");
            metrics.jetstream_ack_failures_total.inc();
            false
        }
    }
}

// Generate-path (NATS-only) settlement helpers. The generation flow shares
// its `Message` behind an `Arc` with the streaming-event callback, so it
// cannot move it into a [`Delivery`]; these mirror `ack`/`nak_one` exactly.

async fn ack_msg(msg: &Message) -> Result<(), DispatchError> {
    msg.ack()
        .await
        .map_err(|e| DispatchError::Ack(e.to_string()))
}

async fn nak_msg(msg: &Message, delay_ms: u64, metrics: &MetricsRegistry) {
    let delay = std::time::Duration::from_millis(delay_ms);
    match msg
        .ack_with(async_nats::jetstream::AckKind::Nak(Some(delay)))
        .await
    {
        Ok(()) => metrics.messages_naked_total.inc(),
        Err(e) => {
            warn!(error = %e, "nak failed");
            metrics.jetstream_nak_failures_total.inc();
        }
    }
}

// -----------------------------------------------------------------------------
// Scheduler drain loop
// -----------------------------------------------------------------------------

/// Default deadline (ms) for the shutdown-time drain of a model's
/// scheduler queue. After this the loop exits and any residual items
/// redeliver via JetStream's `ack_wait` — correct but slower. Tuned
/// to stay well under `DRAIN_DEADLINE_MS` on the backend so the
/// overall shutdown budget isn't exceeded. Overridable with
/// `SIE_SCHEDULER_DRAIN_DEADLINE_MS` for ops; see
/// [`scheduler_drain_deadline_ms`].
const DEFAULT_SCHEDULER_DRAIN_DEADLINE_MS: u64 = 10_000;

/// Resolved drain deadline honouring the `SIE_SCHEDULER_DRAIN_DEADLINE_MS`
/// env override. Parsed per call (the drain loop reads it exactly
/// once, at shutdown-time) so tests + ops can nudge it without
/// restarting threads. Invalid / non-positive values fall back to
/// the default rather than silently producing a zero deadline.
fn scheduler_drain_deadline_ms() -> u64 {
    std::env::var("SIE_SCHEDULER_DRAIN_DEADLINE_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_SCHEDULER_DRAIN_DEADLINE_MS)
}

/// Process-wide monotonic batch id. Shared across every per-model
/// drain loop so a batch id collision across models can't happen,
/// which keeps log-correlation unambiguous on the Python side.
static SCHEDULER_BATCH_ID_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Role of a batch within a scheduler wave.
///
/// A wave starts with one **primary** batch and may continue with zero
/// or more **drain** batches from the same `(op, lora)` queue. The
/// adaptive controller must step exactly once per wave, using the
/// primary batch size. Drains still feed inference-time and per-request
/// latency samples, but they do not trigger efficiency records or PI
/// controller steps; otherwise the controller sees a stream of smaller
/// drain batches and drives the wait knob too low under saturation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaveRole {
    /// First batch in a wave (or shutdown final-drain — every
    /// flush there is its own degenerate wave with no following
    /// drains). Drives a full `record_completion` →
    /// efficiency-record + controller step + cap propagation.
    Primary,
    /// Continuation batch produced by `try_drain_same` on the same
    /// `(op, lora)` queue as the wave's primary. Feeds the
    /// inference-calibration tracker (one sample) + the per-request
    /// latency tracker, but does **not** step the controller and
    /// does **not** record an efficiency sample. The caps from the
    /// preceding primary's step stay in effect for the rest of the
    /// wave; adaptive Prometheus gauges retain the primary's value
    /// (`gauge.set` is idempotent), so dashboards show one update
    /// per wave instead of one per IPC roundtrip.
    Drain,
}

fn release_scheduler_pressure(
    metrics: &MetricsRegistry,
    worker_pool: &AdapterWorkerPool,
    model_id: &str,
    items: &[SchedulerItem],
    metadata: &[SchedulerMeta],
) {
    if items.is_empty() {
        return;
    }
    assert_eq!(
        items.len(),
        metadata.len(),
        "FormattedBatch items and metadata must stay aligned",
    );

    let mut total_cost = 0_u64;
    let mut child_pending: BTreeMap<usize, (usize, u64)> = BTreeMap::new();
    let mut unattributed_count = 0_usize;
    let mut unattributed_cost = 0_u64;

    for (item, meta) in items.iter().zip(metadata.iter()) {
        let item_cost = item.cost();
        total_cost = total_cost.saturating_add(item_cost);
        if let Some(child_index) = meta.worker_child_index {
            let entry = child_pending.entry(child_index).or_insert((0, 0));
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(item_cost);
        } else {
            unattributed_count += 1;
            unattributed_cost = unattributed_cost.saturating_add(item_cost);
        }
    }

    decrement_gauge(&metrics.worker_queue_depth, items.len() as i64);
    decrement_gauge(&metrics.worker_pending_cost, clamp_u64_to_i64(total_cost));
    for (child_index, (item_count, cost)) in child_pending {
        worker_pool.record_child_pending_dequeue(child_index, item_count, cost);
    }
    if unattributed_count > 0 {
        worker_pool.record_model_pending_dequeue(model_id, unattributed_count, unattributed_cost);
    }
}

/// Per-batch scheduler-tick: pack the flushed batch into a
/// [`RunBatchRequest`], hand it to the backend, apply outcomes
/// through the existing dispatcher publish/ACK/NAK path, and feed
/// the adaptive controller with one completion sample.
///
/// Factored out of [`scheduler_drain_loop`] so the happy-path + the
/// shutdown final-drain share the same code. Pure async fn with no
/// hidden state: the scheduler is borrowed and the dispatcher is
/// [`Arc`]'d.
///
/// `role` decides whether this batch's completion triggers a
/// controller step (see [`WaveRole`] for the cadence rules).
async fn process_scheduler_batch(
    model_id: &str,
    dispatcher: &Arc<Dispatcher>,
    scheduler: &Arc<ProductionScheduler>,
    op: SchedOp,
    lora: crate::scheduler::LoraKey,
    batch: crate::scheduler::FormattedBatch<SchedulerItem, SchedulerMeta>,
    role: WaveRole,
) {
    if batch.items.is_empty() {
        return;
    }
    assert_eq!(
        batch.items.len(),
        batch.metadata.len(),
        "FormattedBatch items and metadata must stay aligned",
    );
    let crate::scheduler::FormattedBatch {
        items,
        metadata,
        total_cost: _,
        flush_reason,
    } = batch;
    let mut kept_items = Vec::with_capacity(items.len());
    let mut kept_metadata = Vec::with_capacity(metadata.len());
    let mut kept_cost = 0_u64;
    let mut cancelled = 0_usize;
    for (item, meta) in items.into_iter().zip(metadata) {
        if dispatcher.is_cancelled_worker_direct_batch(
            &meta.wi.request_id,
            &meta.wi.operation,
            meta.worker_direct,
        ) {
            release_scheduler_pressure(
                &dispatcher.metrics,
                &dispatcher.worker_pool,
                model_id,
                std::slice::from_ref(&item),
                std::slice::from_ref(&meta),
            );
            dispatcher
                .ack_cancelled_worker_direct_batch(&meta.wi, &meta.delivery)
                .await;
            cancelled += 1;
            continue;
        }
        kept_cost += item.cost();
        kept_items.push(item);
        kept_metadata.push(meta);
    }
    if cancelled > 0 {
        debug!(
            model = %model_id,
            cancelled,
            "batch-direct: dropped cancelled worker-direct scheduler items before IPC"
        );
    }
    let batch = crate::scheduler::FormattedBatch {
        items: kept_items,
        metadata: kept_metadata,
        total_cost: kept_cost,
        flush_reason,
    };
    if batch.items.is_empty() {
        return;
    }
    let batch_size = batch.items.len();
    let total_cost = batch.total_cost;
    let flush_reason = batch.flush_reason.as_label();
    release_scheduler_pressure(
        &dispatcher.metrics,
        &dispatcher.worker_pool,
        model_id,
        &batch.items,
        &batch.metadata,
    );
    let op_label = match op {
        SchedOp::Encode => "encode",
        SchedOp::Score => "score",
        SchedOp::Extract => "extract",
    };
    // `lora.as_str()` yields `None` for the base key; on the wire
    // we send an empty string so Python's `lora_key or None` chain
    // roundtrips to the same value.
    let lora_str = lora.as_str().unwrap_or("").to_string();
    // Worker-local monotonic batch id. `Relaxed` is fine: the value
    // is log-only (Python writes it on METHOD_RUN_BATCH) and no
    // ordering guarantees hang off it across threads.
    let batch_id = SCHEDULER_BATCH_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Zip each scheduler item with its parallel metadata so we can copy
    // the originating `WorkItem`'s W3C trace context onto the wire item.
    // `batch.metadata` is borrowed here and only consumed later in this
    // function (the NAK / `apply_outcomes` paths), so `.iter()` is fine.
    // Guard the parallel-vector invariant: `zip()` would silently truncate
    // and drop work items if `items` and `metadata` ever drifted apart.
    assert_eq!(
        batch.items.len(),
        batch.metadata.len(),
        "FormattedBatch items and metadata must stay aligned",
    );
    // Open the `sidecar.dispatch` span so the queue hop is visible in
    // the trace: `gateway.proxy → sidecar.dispatch → worker.run_batch`.
    // It is created here (before the IPC dispatch) and kept alive — but
    // only *entered* synchronously, never across an `.await` — so its
    // OTel duration covers the dispatch and it is not held on the
    // worker thread while the task is parked. When no OTLP exporter is
    // configured the `tracing_opentelemetry` layer is absent, so this
    // span produces no OTel span and the gateway-context fallback below
    // keeps `gateway → worker` linkage intact (propagator-only parity).
    let dispatch_span = tracing::info_span!(
        "sidecar.dispatch",
        otel.name = "sidecar.dispatch",
        sie.op = op_label,
        sie.model = %model_id,
        sie.batch_id = batch_id,
        sie.batch_size = batch_size,
    );
    {
        // A coalesced batch can mix items from several gateway traces.
        // A span has exactly one parent but may carry many links, so we
        // parent on the first valid gateway context and record the
        // remaining distinct contexts as links (OTel batch convention).
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let mut linked: HashSet<&str> = HashSet::new();
        // Parent on the first item carrying a *valid* inbound context,
        // not merely the first item. A leading untraced item would
        // otherwise root the sidecar span on a fresh trace and — once
        // the outgoing items are overwritten with that context below —
        // strip the gateway lineage from the batch's genuinely-traced
        // items. `set_parent` then always receives a valid context, so
        // its `Result` is an `Ok` we discard.
        if let Some(parent) = batch.metadata.iter().find(|meta| {
            crate::observability::propagation::remote_span_context(
                meta.wi.traceparent.as_deref(),
                meta.wi.tracestate.as_deref(),
            )
            .is_some()
        }) {
            let tp = parent.wi.traceparent.as_deref();
            let ts = parent.wi.tracestate.as_deref();
            let _ = dispatch_span.set_parent(
                crate::observability::propagation::extract_context_from_w3c(tp, ts),
            );
            if let Some(tp) = tp {
                linked.insert(tp);
            }
        }
        for meta in batch.metadata.iter() {
            let Some(tp) = meta.wi.traceparent.as_deref() else {
                continue;
            };
            if !linked.insert(tp) {
                continue;
            }
            if let Some(sc) = crate::observability::propagation::remote_span_context(
                Some(tp),
                meta.wi.tracestate.as_deref(),
            ) {
                dispatch_span.add_link(sc);
            }
        }
    }

    // Serialise the sidecar span back into W3C strings (entered
    // synchronously — no `.await` inside). With an exporter active this
    // yields the sidecar span's context (new span_id under the inbound
    // trace); without one it returns `(None, None)` and the per-item
    // fallback below keeps the gateway context.
    let (sidecar_tp, sidecar_ts) =
        dispatch_span.in_scope(crate::observability::propagation::inject_current_context);

    let rb_items: Vec<crate::ipc_types::RunBatchItem> = batch
        .items
        .into_iter()
        .zip(batch.metadata.iter())
        .map(|(item, meta)| {
            // `into_run_batch_item_with_trace` copies the gateway
            // context as the fallback; override with the sidecar span's
            // context when one exists so `worker.run_batch` nests under
            // `sidecar.dispatch`.
            let mut rbi = item.into_run_batch_item_with_trace(&meta.wi);
            if let Some(tp) = &sidecar_tp {
                rbi.traceparent = Some(tp.clone());
                rbi.tracestate = sidecar_ts.clone();
            }
            rbi
        })
        .collect();
    let req = RunBatchRequest {
        model_id: model_id.to_string(),
        batch_id,
        lora_key: lora_str,
        total_cost,
        items: rb_items,
    };

    let model_lbl = dispatcher.metrics.model_label(model_id);
    // Basic scheduler observability: one sample per flushed batch of
    // (items, total_cost) against `(model, operation, lora)`. Record
    // *before* the RPC so we capture the batch shape even if the
    // backend call ends up failing and the batch gets NAKed — the
    // shape itself is what the scheduler produced.
    //
    // On the wire `lora_str` is `""` (matches Python's `lora_key or
    // None` contract) but for dashboards we prefer the explicit
    // `"base"` sentinel so PromQL doesn't need `lora=""` predicates.
    // `lora.as_str()` yields `None` for the base key too, so this
    // doesn't duplicate logic — we just pick a clearer label.
    let lora_metric_label = lora.as_str().unwrap_or("base");
    dispatcher
        .metrics
        .scheduler
        .batch_items
        .with_label_values(&[&model_lbl, op_label, lora_metric_label])
        .observe(batch_size as f64);
    dispatcher
        .metrics
        .scheduler
        .batch_cost
        .with_label_values(&[&model_lbl, op_label, lora_metric_label])
        .observe(total_cost as f64);
    dispatcher
        .metrics
        .scheduler
        .flush_reason_total
        .with_label_values(&[&model_lbl, op_label, lora_metric_label, flush_reason])
        .inc();
    let _timer = dispatcher
        .metrics
        .pull_batch_process_seconds
        .with_label_values(&[&model_lbl, op_label])
        .start_timer();
    let started = Instant::now();
    dispatcher.metrics.ipc_requests_total.inc();
    dispatcher.metrics.inflight_batches.inc();

    let outcome = match dispatcher.backend.run_batch(req).await {
        Ok(o) => o,
        Err(e) => {
            dispatcher.metrics.inflight_batches.dec();
            dispatcher.metrics.ipc_failures_total.inc();
            let delay = nak_delay_for_backend_error(&e);
            warn!(
                model = %model_id,
                op = op_label,
                error = %ErrChain(&e),
                nak_delay_ms = delay,
                batch_size,
                "scheduler RunBatch failed — NAKing batch",
            );
            let msgs_only: Vec<(WorkItem, Delivery)> = batch
                .metadata
                .into_iter()
                .map(|m| (m.wi, m.delivery))
                .collect();
            nak_all(&msgs_only, delay, &dispatcher.metrics).await;
            return;
        }
    };
    dispatcher.metrics.inflight_batches.dec();

    // Collect per-item timings BEFORE we move the metadata into
    // `apply_outcomes`. Two telemetry flows out of this loop:
    //
    //  1. `inference_ms_sample` — first reported inference_ms (all
    //     items in a GPU batch share the forward pass so any one is
    //     representative). Feeds the auto-calibration tracker so
    //     `target_p50_ms` derives from GPU forward time, not the
    //     batcher+post sum.
    //  2. `per_item_total_ms` — one entry per `PublishAndAck`
    //     outcome, value = `batcher_wait + inference + postprocess`
    //     for *that* item. Fed verbatim into the controller's
    //     latency tracker so `observed_p50_ms` matches Python
    //     `RequestTiming.total_ms` semantics. The signal must include
    //     the Rust BatchFormer wait, must exclude upstream NATS queue
    //     depth, and must not collapse a batch to its max item latency;
    //     each of those alternatives biases the PI loop enough to pin
    //     the wait knob at a floor or ceiling.
    //
    //     The semantically correct mirror of Python's `total_ms` is
    //     `(now - submitted_at) + inference + post`. `submitted_at`
    //     (`SchedulerMeta.submitted_at`) is stamped when the dispatcher
    //     enqueues the item into our scheduler — equivalent to when
    //     Python's `RequestTiming()` is constructed at the top of
    //     `EncodePipeline.run_encode`, which is *after* the NATS pull
    //     but *before* batch formation. The delta (`now -
    //     submitted_at`) covers the time the item spent inside our
    //     own (Rust) `BatchFormer` waiting to flush — the analogue
    //     of Python's `total_ms` via `metadata.timing._start_time`
    //     getting set in `EncodePipeline.run_encode`. Note: the sidecar
    //     calls Python's pre-formed batch IPC entrypoint, so Python does
    //     not run its per-LoRA BatchFormer on this path; the entire
    //     batch-form delta is observed here on the Rust side instead of
    //     being split across Rust and Python batchers.
    let now = Instant::now();
    let inference_ms_sample = outcome
        .outcomes
        .iter()
        .find_map(|o| o.inference_ms)
        .unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0);
    // Per-request totals (deduped by `request_id`). When the gateway
    // splits a multi-item client request into N NATS work-items they
    // share the same `request_id` but get distinct `item_index`es —
    // they may all land in one Rust batch. Python's
    // `_complete_requests` dedupes via a `seen: set[id(metadata)]` so a
    // multi-item request deposits **one** latency sample, not N (see
    // `model_worker.py:947-976` on main `bbe409c3`). The tracker p50 is
    // a per-request signal, not a per-item one — recording N samples
    // here would let a single 10-item request dominate `observed_p50_ms`
    // 10× more than ten 1-item requests at the same throughput, which
    // is exactly the bias the calibration tracker is designed to avoid.
    // Picking the first occurrence's
    // `submitted_at` keeps `batcher_wait_ms` aligned with Python's
    // `RequestTiming._start_time`, which is set once when the request
    // enters the worker process and shared across all its items.
    let per_request_total_ms: Vec<f64> = dedupe_per_request_totals(
        batch
            .metadata
            .iter()
            .zip(outcome.outcomes.iter())
            .map(|(m, o)| (m.wi.request_id.as_str(), m.submitted_at, o)),
        now,
    );

    let resolved: Vec<(WorkItem, Delivery, f64)> = batch
        .metadata
        .into_iter()
        .map(|m| (m.wi, m.delivery, m.fetch_ms))
        .collect();
    dispatcher.apply_outcomes(outcome, resolved).await;

    // Feed the adaptive controller. Only record completion when we
    // actually saw a successful item — a batch of all-errors would
    // bias the controller toward shrinking caps based on fast
    // no-op replies.
    //
    // Calibration sample rate: **one sample per batch** (not per
    // item). Feeding one sample per item can make Rust's calibration
    // latch from a single cold-start batch, pinning the target and
    // wait knob before the GPU reaches steady state. Per-batch
    // sampling keeps calibration aging across multiple batches while
    // the GPU warms.
    //
    // PI signal sample rate: per **unique request_id** (deduped).
    // The tracker p50 is a per-request signal — a single 10-item
    // request shouldn't dominate `observed_p50_ms` 10× more than ten
    // 1-item requests at the same throughput. For typical bench
    // workloads (1 item per request) dedup is a no-op.
    //
    // Cadence (per [`WaveRole`]): Primary triggers
    // `record_completion` (efficiency record + controller step +
    // cap propagation + Prometheus push). Drain feeds inference +
    // latency samples only; the controller is **not** stepped — the
    // wave's caps were already updated by the primary, and Python's
    // `_process_loop` likewise steps once per wave with the primary
    // batch size (`model_worker.py:828, 855-870`).
    //
    // `SIE_RUST_WAVE_CADENCE=off` flips back to per-batch stepping
    // (every Drain also calls `record_completion`). Off-by-default;
    // see [`crate::wave_cadence_enabled`] for the rationale and the
    // p50/p99 trade-off.
    if !per_request_total_ms.is_empty() {
        scheduler.record_inference_sample(inference_ms_sample).await;
        scheduler
            .record_latency_samples(&per_request_total_ms)
            .await;
        if matches!(role, WaveRole::Primary) || !crate::wave_cadence_enabled() {
            let snapshot = scheduler.record_completion(total_cost, batch_size).await;

            // Push the controller snapshot to Prometheus. In passthrough
            // mode the Python adaptive controller never runs, so the Rust
            // scheduler is the sole source of these metrics. `model_lbl`
            // is already in scope from the pre-RPC path; re-using it keeps
            // the label string stable across the
            // batch_items / batch_cost / adaptive_* families so a
            // PromQL `on(model)` join works.
            let m = &dispatcher.metrics.scheduler;
            m.adaptive_wait_ms
                .with_label_values(&[&model_lbl])
                .set(snapshot.new_wait_ms);
            m.adaptive_cost
                .with_label_values(&[&model_lbl])
                .set(snapshot.new_batch_cost as f64);
            if let Some(o) = snapshot.observed_p50_ms {
                m.adaptive_observed_p50_ms
                    .with_label_values(&[&model_lbl])
                    .set(o);
            }
            if let Some(t) = snapshot.target_p50_ms {
                m.adaptive_target_p50_ms
                    .with_label_values(&[&model_lbl])
                    .set(t);
            }
            if let Some(f) = snapshot.fill_ratio {
                m.adaptive_fill_ratio
                    .with_label_values(&[&model_lbl])
                    .set(f);
            }
            if snapshot.starvation_resets_delta > 0 {
                m.starvation_resets_total
                    .with_label_values(&[&model_lbl])
                    .inc_by(u64::from(snapshot.starvation_resets_delta));
            }
        }
    }

    debug!(
        model = %model_id,
        op = op_label,
        batch_id,
        batch_size,
        total_cost,
        flush_reason,
        role = ?role,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "scheduler batch complete",
    );
}

fn clamp_u64_to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn decrement_gauge(gauge: &prometheus::IntGauge, value: i64) {
    if value <= 0 {
        return;
    }
    gauge.sub(value);
}

/// Per-model background task: consume flushed batches from the
/// model's [`ProductionScheduler`], run them, and cycle back. Exits
/// on the shared [`Shutdown`] signal after a bounded final-drain
/// window; any items still in the scheduler when the deadline
/// expires redeliver via JetStream `ack_wait`.
///
/// Maximum number of in-flight `process_scheduler_batch` invocations
/// per model scheduler. This controls the depth of the IPC dispatch
/// pipeline.
///
/// Default is **2**: one batch can be on the GPU while the next has
/// already crossed the IPC boundary and is waiting on Python's
/// passthrough lock. That preserves the dual-buffer behaviour the
/// Python batcher used to provide before batch formation moved into
/// Rust.
///
/// `1` is still useful as a regression-bisect and high-saturation
/// escape hatch: it forces strict serial dispatch and removes queued
/// IPC frames from the request's tail. The default stays at `2` because
/// normal saturated loads benefit from hiding IPC roundtrip/decode time
/// behind the current forward pass.
///
/// Values are clamped to `[1, 8]`. The upper bound is intentionally
/// conservative: Python inference is still serialized by the adapter's
/// single CUDA stream, so large depths mainly park decoded batches on
/// the lock and inflate per-batch latency.
fn pipeline_depth() -> usize {
    std::env::var("SIE_RUST_PIPELINE_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, 8))
        .unwrap_or(2)
}

/// Whether the scheduler may flush the first item after an idle gap
/// immediately instead of waiting for the coalesce window.
///
/// Default true preserves the current low-load latency behavior.
/// Operators can set `SIE_BATCHER_IDLE_BYPASS_ENABLED=false` for
/// passthrough runtimes such as Candle when multi-worker fanout makes
/// each worker see a thinner local stream and early singleton flushes
/// under-fill GPU forwards.
fn scheduler_idle_bypass_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SIE_BATCHER_IDLE_BYPASS_ENABLED")
            .ok()
            .map(|raw| {
                !matches!(
                    raw.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(true)
    })
}

/// Acquire one permit from `pipeline_sem` (blocking the consume loop
/// once the active depth — `ceiling` minus any autotune-parked
/// permits — batches are in flight) then spawn the dispatch as an
/// independent task. The task releases its permit when it returns.
///
/// Backpressure direction: if the pipeline is saturated, the
/// `acquire_owned` here yields the consume loop, which means the
/// underlying `BatchFormer` keeps accumulating items. The next
/// `consume_next` call therefore sees a fuller batch — net positive
/// at saturation.
// Each argument is independently sourced from the consume loop's
// stack frame (model_id from the per-model task, dispatcher +
// scheduler are shared Arcs, pipeline_sem is loop-local, op/lora/
// batch come from the just-flushed scheduler tick, role comes from
// the wave's primary-vs-drain role assignment). Bundling them into a
// struct would introduce a one-shot wrapper type whose only purpose
// is to satisfy this lint — net less readable, so explicit allow.
#[allow(clippy::too_many_arguments)]
async fn spawn_pipelined_batch(
    model_id: &str,
    dispatcher: &Arc<Dispatcher>,
    scheduler: &Arc<ProductionScheduler>,
    pipeline_sem: &Arc<Semaphore>,
    op: SchedOp,
    lora: crate::scheduler::LoraKey,
    batch: crate::scheduler::FormattedBatch<SchedulerItem, SchedulerMeta>,
    role: WaveRole,
) {
    if batch.items.is_empty() {
        return;
    }
    let permit = match pipeline_sem.clone().acquire_owned().await {
        Ok(p) => p,
        Err(_) => {
            // Semaphore closed. Should be impossible while the loop
            // owns it, but log loudly and fall back to a serial
            // dispatch on the consume loop's task so we don't drop
            // the batch on the floor.
            warn!(
                model = %model_id,
                "rust-scheduler: pipeline semaphore closed; running batch inline",
            );
            process_scheduler_batch(model_id, dispatcher, scheduler, op, lora, batch, role).await;
            return;
        }
    };
    let model_id_c = model_id.to_owned();
    let disp_c = Arc::clone(dispatcher);
    let sched_c = Arc::clone(scheduler);
    tokio::spawn(async move {
        process_scheduler_batch(&model_id_c, &disp_c, &sched_c, op, lora, batch, role).await;
        drop(permit);
    });
}

pub(crate) async fn scheduler_drain_loop(
    model_id: String,
    dispatcher: Arc<Dispatcher>,
    scheduler: Arc<ProductionScheduler>,
    shutdown: Arc<Shutdown>,
) {
    let depth = pipeline_depth();
    let idle_bypass_enabled = scheduler_idle_bypass_enabled();
    info!(
        model = %model_id,
        pipeline_depth = depth,
        idle_bypass_enabled,
        "rust-scheduler: drain loop started",
    );

    // Python runs the pre-formed batch IPC entrypoint here: there is no
    // per-LoRA BatchFormer on the queue path and every IPC frame is one
    // caller-formed GPU dispatch, serialized by ModelWorker's adapter
    // dispatch lock. All queue batching now lives in the Rust scheduler
    // (`ProductionScheduler` -> `BatchFormer`).
    //
    // What `depth = 2` still buys us in this regime: the Python IPC
    // server (`ipc_server.py::_handle_request`) spawns each `RUN_BATCH`
    // as an `asyncio.create_task` (non-blocking on the read loop), so
    // shipping batch N+1 while N is on the GPU lets N+1 cross the IPC
    // boundary, msgpack-decode, and park on the contended
    // adapter dispatch lock. When N's forward pass finishes the lock
    // releases, N+1 enters the forward pass without paying the IPC
    // roundtrip + decode on the critical path.
    //
    // We cap depth at 2 by default — Python's adapter is still single-
    // CUDA-stream so true concurrent inference isn't possible; depth
    // > 2 just inflates per-batch latency by parking more frames on
    // the lock without any throughput gain. Operators can override
    // via `SIE_RUST_PIPELINE_DEPTH`: `1` falls back to strict serial
    // dispatch (one outstanding IPC roundtrip). At very high
    // concurrency, operators may prefer `1` to avoid parking a second
    // frame behind a long GPU forward pass.
    let pipeline_sem = Arc::new(Semaphore::new(depth));

    // Idle-bypass + continuous-batching state. On the queue path,
    // Python's `model_worker.py::_process_loop` is bypassed and the Rust
    // scheduler is the sole batcher in the system, so the logic below
    // stands alone:
    //
    //   * `was_idle` starts true and is forwarded to `consume_next`
    //     as the `immediate` flag when idle-bypass is enabled. When
    //     the worker just polled an empty queue, the next item to
    //     arrive can flush at once instead of paying the full
    //     `max_batch_wait_ms` (50 ms in the auto-calibrated regime).
    //     Without this, low-concurrency traffic eats one full wait
    //     window per batch and p50 ends up dominated by the
    //     controller's wait knob. Operators can disable this via
    //     `SIE_BATCHER_IDLE_BYPASS_ENABLED=false` for passthrough GPU
    //     runtimes whose per-worker stream gets too thin under
    //     multi-worker fanout.
    //
    //   * After every consumed batch we loop on `try_drain_same` for
    //     the same `(op, lora)` until it returns `None`. With
    //     pipelining this is still useful: items that landed in the
    //     Rust `BatchFormer` between the timer firing and us reaching
    //     this point get shipped immediately rather than waiting for
    //     the next coalesce window. Each drained sub-batch goes
    //     through the same permit-bounded spawn, so total in-flight
    //     stays capped by `depth`.
    //
    //   * `was_idle` is reset to true only when the just-finished
    //     wave produced no drained tail and the kicking batch was a
    //     singleton — i.e. real evidence the worker outran demand.
    //     Multi-item batches or non-empty drain tails mean traffic
    //     is steady and the next iteration should accumulate (no
    //     immediate flush).
    let mut was_idle = true;
    loop {
        tokio::select! {
            (op, lora, batch) = scheduler.consume_next(was_idle && idle_bypass_enabled) => {
                let initial_batch_size = batch.items.len();
                // First batch in the wave drives the controller step.
                spawn_pipelined_batch(
                    &model_id,
                    &dispatcher,
                    &scheduler,
                    &pipeline_sem,
                    op,
                    lora.clone(),
                    batch,
                    WaveRole::Primary,
                ).await;

                let mut drained_any = false;
                while let Some(drain_batch) =
                    scheduler.try_drain_same(op, lora.clone()).await
                {
                    if drain_batch.items.is_empty() {
                        break;
                    }
                    drained_any = true;
                    // Drains continue the wave: feed inference +
                    // latency samples, but no controller step.
                    spawn_pipelined_batch(
                        &model_id,
                        &dispatcher,
                        &scheduler,
                        &pipeline_sem,
                        op,
                        lora.clone(),
                        drain_batch,
                        WaveRole::Drain,
                    ).await;
                }

                was_idle = !drained_any && initial_batch_size <= 1;
            }
            _ = shutdown.wait() => {
                info!(
                    model = %model_id,
                    "rust-scheduler: shutdown observed — entering final drain",
                );
                break;
            }
        }
    }

    // Quiesce the pipeline: acquire all `depth` permits so every
    // spawned dispatch task has finished its IPC roundtrip and
    // released its permit. Only then do we enter synchronous shutdown
    // drain below. Without this the final-drain loop
    // could race with still-in-flight pipelined batches and double-
    // submit work to Python.
    if let Ok(permits) = pipeline_sem.clone().acquire_many_owned(depth as u32).await {
        // Hold the permits for the rest of the function so no further
        // spawns can sneak in (`spawn_pipelined_batch` is no longer
        // called past this point, but defence-in-depth).
        std::mem::forget(permits);
    }

    // Shutdown drain: flush whatever's still enqueued, up to a
    // deadline. `try_consume_next` is non-blocking; a `None` return
    // with `total_pending_count() > 0` means flush triggers haven't
    // fired yet — we sleep briefly to let coalesce windows expire
    // then try again.
    let started = Instant::now();
    let deadline = std::time::Duration::from_millis(scheduler_drain_deadline_ms());
    loop {
        if started.elapsed() >= deadline {
            let remaining = scheduler.total_pending_count().await;
            if remaining > 0 {
                warn!(
                    model = %model_id,
                    remaining,
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "rust-scheduler: drain deadline exceeded — residual items will redeliver via JetStream ack_wait",
                );
            }
            break;
        }
        match scheduler.try_consume_next().await {
            Some((op, lora, batch)) => {
                // Final-drain flushes are standalone waves (no
                // following `try_drain_same` loop here), so each
                // counts as its own primary.
                process_scheduler_batch(
                    &model_id,
                    &dispatcher,
                    &scheduler,
                    op,
                    lora,
                    batch,
                    WaveRole::Primary,
                )
                .await;
            }
            None => {
                if scheduler.total_pending_count().await == 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
        }
    }
    info!(
        model = %model_id,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "rust-scheduler: drain loop exited",
    );
}

/// Decode an offloaded generate payload blob (msgpack) into the inline
/// `generate` value. Extracted so the transport contract is directly
/// testable: the gateway offloads `generate` carrying **base64-string** image
/// data, which round-trips through `serde_json::Value`; the original, buggy
/// msgpack-`bin` (`serde_bytes`) shape does NOT decode here.
fn decode_offloaded_generate(bytes: &[u8]) -> Result<Json, rmp_serde::decode::Error> {
    rmp_serde::from_slice(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::delivery::LocalDelivery;

    fn text_item(text: &str) -> MsgValue {
        MsgValue::Map(vec![(MsgValue::from("text"), MsgValue::from(text))])
    }

    fn adapter_pool_with_metrics(metrics: Arc<MetricsRegistry>) -> Arc<AdapterWorkerPool> {
        let paths = [PathBuf::from("/tmp/sie-test-ipc-0.sock")];
        AdapterWorkerPool::new(&paths, 1, 60, 900, metrics)
    }

    fn wi(request: &str, idx: u32, model: &str, op: &str) -> WorkItem {
        WorkItem {
            work_item_id: format!("{}.{}", request, idx),
            request_id: request.into(),
            item_index: idx,
            total_items: 1,
            operation: op.into(),
            model_id: model.into(),
            profile_id: String::new(),
            engine: String::new(),
            pool_name: "l4".into(),
            admission_pool: String::new(),
            machine_profile: String::new(),
            item: Some(text_item("x")),
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.a".into(),
            traceparent: None,
            tracestate: None,
            timestamp: 0.0,
        }
    }

    fn encode_scheduler_item(work: &WorkItem) -> SchedulerItem {
        SchedulerItem::Encode(EncodeBatchItem {
            work_item_id: work.work_item_id.clone(),
            request_id: work.request_id.clone(),
            item_index: work.item_index,
            total_items: work.total_items,
            timestamp: work.timestamp,
            item: work.item.clone().expect("test item"),
            output_types: work.output_types.clone(),
            instruction: work.instruction.clone(),
            is_query: work.is_query,
            options: work.options.clone(),
            profile_id: Some(work.profile_id.clone()),
            bundle_config_hash: Some(work.bundle_config_hash.clone()),
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        })
    }

    #[test]
    fn releasing_cancelled_scheduled_item_clears_global_and_child_pressure() {
        let metrics = Arc::new(MetricsRegistry::new().unwrap());
        let worker_pool = adapter_pool_with_metrics(Arc::clone(&metrics));
        let work = wi("req-cancel", 0, "model-a", "encode");
        let item = encode_scheduler_item(&work);
        let child_index = worker_pool.record_model_pending_enqueue(&work.model_id, item.cost());
        metrics.worker_queue_depth.inc();
        metrics
            .worker_pending_cost
            .add(clamp_u64_to_i64(item.cost()));

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let meta = SchedulerMeta::new_with_worker_direct(
            work.clone(),
            Delivery::Local(LocalDelivery::new(0, 0, tx)),
            0.0,
            true,
        )
        .with_worker_child_index(child_index);

        release_scheduler_pressure(
            &metrics,
            &worker_pool,
            &work.model_id,
            std::slice::from_ref(&item),
            std::slice::from_ref(&meta),
        );

        assert_eq!(metrics.worker_queue_depth.get(), 0);
        assert_eq!(metrics.worker_pending_cost.get(), 0);
        let child_label = child_index.to_string();
        assert_eq!(
            metrics
                .worker_child_queue_depth
                .with_label_values(&[child_label.as_str()])
                .get(),
            0
        );
        assert_eq!(
            metrics
                .worker_child_pending_cost
                .with_label_values(&[child_label.as_str()])
                .get(),
            0
        );
    }

    #[test]
    fn generate_terminal_error_chunk_uses_streaming_wire_shape() {
        #[derive(serde::Deserialize)]
        struct DecodedChunk {
            kind: String,
            request_id: String,
            attempt_id: String,
            seq: u32,
            text_delta: String,
            done: bool,
            finish_reason: String,
            error: DecodedError,
        }

        #[derive(serde::Deserialize)]
        struct DecodedError {
            code: String,
            message: String,
        }

        let work = wi("req-load", 0, "Qwen/Qwen3-4B-Instruct-2507", "generate");
        let bytes =
            encode_generate_terminal_error_chunk(&work, MODEL_LOADING_ERROR_CODE, "loading")
                .expect("chunk encodes");
        let decoded: DecodedChunk = rmp_serde::from_slice(&bytes).expect("chunk decodes");

        assert_eq!(decoded.kind, "chunk");
        assert_eq!(decoded.request_id, "req-load");
        assert_eq!(decoded.attempt_id, "req-load.0:model-loading");
        assert_eq!(decoded.seq, 0);
        assert_eq!(decoded.text_delta, "");
        assert!(decoded.done);
        assert_eq!(decoded.finish_reason, "error");
        assert_eq!(decoded.error.code, MODEL_LOADING_ERROR_CODE);
        assert_eq!(decoded.error.message, "loading");
    }

    #[tokio::test]
    async fn offloaded_generate_blob_with_base64_images_resolves_via_object_store() {
        use crate::payload_store::LocalPayloadStore;

        // Gateway offload shape: `generate` params with base64-STRING image data.
        let generate = serde_json::json!({
            "messages": [{
                "role": "user",
                "content": "what is this?",
                "images": [{"data": "aGVsbG8=", "format": "png"}], // base64 of b"hello"
            }],
            "max_new_tokens": 8,
        });
        let blob = rmp_serde::to_vec_named(&generate).unwrap();

        // Real object store (filesystem), the gateway's `{request_id}_0.bin` key.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("req-1_0.bin"), &blob).unwrap();
        let store = LocalPayloadStore::new(Some(dir.path()));

        // The exact sidecar resolution ops: fetch from the object store + decode.
        let bytes = store
            .get("req-1_0.bin")
            .await
            .expect("blob fetched from object store");
        let decoded = decode_offloaded_generate(&bytes)
            .expect("base64-image generate blob must decode as serde_json::Value");
        assert_eq!(decoded["messages"][0]["images"][0]["data"], "aGVsbG8=");

        // Inline into a WorkItem and re-encode → decode (the sidecar → Python
        // worker hop): the base64 image survives the full round trip.
        let mut work = wi("req-1", 0, "Qwen/Qwen3.5-4B", "generate");
        work.item = None;
        work.generate = Some(decoded);
        let reencoded = rmp_serde::to_vec_named(&work).unwrap();
        let back: WorkItem = rmp_serde::from_slice(&reencoded).unwrap();
        let g = back.generate.expect("generate inlined onto the work item");
        assert_eq!(g["messages"][0]["images"][0]["data"], "aGVsbG8=");
    }

    #[test]
    fn bin_generate_blob_fails_to_decode_proving_base64_is_required() {
        // The original (buggy) shape: image bytes as a msgpack `bin` via
        // serde_bytes. `serde_json::Value` cannot hold `bin`, so the sidecar
        // decode fails — this is exactly the transport bug the base64-string
        // representation fixes.
        #[derive(serde::Serialize)]
        struct BinImage {
            #[serde(with = "serde_bytes")]
            data: Vec<u8>,
        }
        #[derive(serde::Serialize)]
        struct BinGenerate {
            images: Vec<BinImage>,
        }
        let blob = rmp_serde::to_vec_named(&BinGenerate {
            images: vec![BinImage {
                data: vec![0xFF, 0xD8, 0xFF, 0xE0],
            }],
        })
        .unwrap();
        assert!(
            decode_offloaded_generate(&blob).is_err(),
            "msgpack bin must NOT decode as serde_json::Value — the exact bug base64 transport fixes"
        );
    }

    #[test]
    fn groups_by_model_and_operation() {
        let items = vec![
            (wi("r1", 0, "A", "encode"), ()),
            (wi("r1", 1, "A", "encode"), ()),
            (wi("r2", 0, "B", "encode"), ()),
            (wi("r3", 0, "A", "score"), ()),
        ];
        let groups = group_by_model(items);
        assert_eq!(groups[&("A".to_string(), "encode".to_string())].len(), 2);
        assert_eq!(groups[&("A".to_string(), "score".to_string())].len(), 1);
        assert_eq!(groups[&("B".to_string(), "encode".to_string())].len(), 1);
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn group_by_model_only_collapses_ops() {
        // For the hot path: all ops for a single model should land together
        // so they can be dispatched concurrently within the group.
        let items = vec![
            (wi("r1", 0, "A", "encode"), ()),
            (wi("r1", 1, "A", "score"), ()),
            (wi("r2", 0, "A", "extract"), ()),
            (wi("r3", 0, "B", "encode"), ()),
        ];
        let groups = group_by_model_only(items);
        assert_eq!(groups["A"].len(), 3);
        assert_eq!(groups["B"].len(), 1);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn opt_non_empty_maps_empty_to_none() {
        assert_eq!(opt_non_empty(""), None);
        assert_eq!(opt_non_empty("x"), Some("x".to_string()));
    }

    #[test]
    fn unknown_bundle_config_hash_accepts_current_recent_and_empty_hashes() {
        let state = ConfigApplyState::new("hash-1".into());
        state.set_bundle_hash("hash-2".into());

        let mut current = wi("r1", 0, "A", "encode");
        current.bundle_config_hash = "hash-2".into();
        let mut recent = wi("r1", 1, "A", "encode");
        recent.bundle_config_hash = "hash-1".into();
        let legacy = wi("r1", 2, "A", "encode");

        let items = [current, recent, legacy];
        let unknown = unknown_bundle_config_hash(items.iter(), Some(&state));
        assert!(unknown.is_none());
    }

    #[test]
    fn unknown_bundle_config_hash_reports_first_unknown_and_count() {
        let state = ConfigApplyState::new("hash-1".into());

        let mut first = wi("r1", 0, "A", "encode");
        first.bundle_config_hash = "missing-a".into();
        let mut accepted = wi("r1", 1, "A", "encode");
        accepted.bundle_config_hash = "hash-1".into();
        let mut second = wi("r1", 2, "A", "encode");
        second.bundle_config_hash = "missing-b".into();

        let items = [first, accepted, second];
        let unknown = unknown_bundle_config_hash(items.iter(), Some(&state));
        assert_eq!(unknown, Some(("missing-a", 2)));
    }

    #[test]
    fn reply_subject_is_safe_rules() {
        assert!(reply_subject_is_safe(""));
        assert!(reply_subject_is_safe("_INBOX.ab"));
        assert!(reply_subject_is_safe("_INBOX.a.b.c"));
        assert!(!reply_subject_is_safe("sie.work.foo.l4"));
        assert!(!reply_subject_is_safe("something.evil"));
        assert!(!reply_subject_is_safe("_INBOX")); // no trailing dot
    }

    #[test]
    fn truncate_respects_char_boundaries() {
        assert_eq!(truncate("hello", 3), "hel");
        assert_eq!(truncate("hi", 10), "hi");
        // 'é' is two bytes — truncate should not split it.
        assert_eq!(truncate("café", 3), "caf");
        assert_eq!(truncate("café", 4), "café");
    }

    #[test]
    fn queue_ms_from_zero_timestamp_is_zero() {
        assert_eq!(queue_ms_from(0.0), 0.0);
        assert_eq!(queue_ms_from(-1.0), 0.0);
    }

    #[test]
    fn queue_ms_from_past_timestamp_is_positive() {
        let one_second_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            - 1.0;
        let ms = queue_ms_from(one_second_ago);
        // Allow ±200ms wobble for test scheduling; we mostly want to verify
        // "not zero, not negative, in the right ballpark".
        assert!((800.0..=1200.0).contains(&ms), "expected ~1000ms, got {ms}");
    }

    #[test]
    fn split_by_budget_under_limit_passes_through() {
        let items = vec![
            (wi("r", 0, "m", "encode"), ()),
            (wi("r", 1, "m", "encode"), ()),
        ];
        let (dispatch, overflow) = split_by_budget(items, 5);
        assert_eq!(dispatch.len(), 2);
        assert!(overflow.is_empty());
    }

    #[test]
    fn split_by_budget_over_limit_splits() {
        let items: Vec<(WorkItem, ())> = (0..10).map(|i| (wi("r", i, "m", "encode"), ())).collect();
        let (dispatch, overflow) = split_by_budget(items, 3);
        assert_eq!(dispatch.len(), 3);
        assert_eq!(overflow.len(), 7);
        // Original order preserved across the split.
        assert_eq!(dispatch[0].0.item_index, 0);
        assert_eq!(dispatch[2].0.item_index, 2);
        assert_eq!(overflow[0].0.item_index, 3);
        assert_eq!(overflow[6].0.item_index, 9);
    }

    #[test]
    fn split_by_budget_zero_sends_all_to_overflow() {
        let items: Vec<(WorkItem, ())> = (0..3).map(|i| (wi("r", i, "m", "encode"), ())).collect();
        let (dispatch, overflow) = split_by_budget(items, 0);
        assert!(dispatch.is_empty());
        assert_eq!(overflow.len(), 3);
    }

    #[test]
    fn default_max_concurrent_batches_env_parsing() {
        // Env-free default should be 4. We only assert the env-free
        // branch here to avoid polluting global state for other tests.
        //
        // SAFETY: this is a #[test] in a cfg(test) module and we don't
        // actually mutate env here — we just check the function is sane
        // when the variable is absent in CI.
        assert!(default_max_concurrent_batches() >= 1);
    }

    #[test]
    fn pipeline_depth_default_is_two() {
        // Env-free assertion only. We can't safely mutate env vars
        // in unit tests without polluting other tests running in
        // the same process.
        if std::env::var("SIE_RUST_PIPELINE_DEPTH").is_err() {
            assert_eq!(pipeline_depth(), 2);
        }
    }

    #[test]
    fn scheduler_idle_bypass_enabled_default_is_true() {
        // Env-free assertion only. The function caches its first
        // result, so tests must not mutate the env var in-process.
        if std::env::var("SIE_BATCHER_IDLE_BYPASS_ENABLED").is_err() {
            assert!(
                scheduler_idle_bypass_enabled(),
                "idle bypass must default on for low-load latency parity"
            );
        }
    }

    #[test]
    fn queue_ms_from_future_timestamp_clamps_to_zero() {
        let in_the_future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            + 60.0;
        assert_eq!(queue_ms_from(in_the_future), 0.0);
    }

    #[test]
    fn base_nak_delay_ms_env_free_default() {
        // Env-free default must match Python's `_NAK_DELAY_S = 5.0` (5000ms)
        // so Rust and Python adapter processes produce the same JetStream delivery
        // pressure under back-off. We don't mutate the environment here to
        // avoid cross-test flakiness; we just assert the lower bound.
        //
        // If `SIE_NAK_DELAY_S` is set in the surrounding environment the
        // value may differ, but the function must always yield a positive
        // delay.
        let v = base_nak_delay_ms();
        assert!(v >= 1, "base_nak_delay_ms must be > 0, got {v}");
    }

    #[test]
    fn nak_delay_for_draining_is_short_not_base() {
        // Draining must NOT use the generic base delay (~5s). Another
        // worker needs the redelivery now; a long NAK would starve
        // throughput while the draining pod walks through shutdown.
        let delay = nak_delay_for_backend_error(&BackendError::Draining);
        assert_eq!(delay, NAK_DELAY_DRAINING_MS);
        assert!(
            delay < base_nak_delay_ms(),
            "draining delay ({delay}ms) must be tighter than base ({}ms)",
            base_nak_delay_ms()
        );
    }

    #[test]
    fn nak_delay_for_transient_and_inference_uses_base() {
        // Everything except Draining shares the base delay so Rust and
        // Python adapter processes throttle retries identically.
        assert_eq!(
            nak_delay_for_backend_error(&BackendError::Transient("x".into())),
            base_nak_delay_ms()
        );
        assert_eq!(
            nak_delay_for_backend_error(&BackendError::Inference("y".into())),
            base_nak_delay_ms()
        );
        assert_eq!(
            nak_delay_for_backend_error(&BackendError::UnsupportedModel("z".into())),
            base_nak_delay_ms()
        );
    }

    #[test]
    fn readiness_progress_delay_only_covers_local_loading_states() {
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::LoadingStarted, 5_000),
            Some(5_000)
        );
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::LoadingInProgress, 5_000),
            Some(10_000)
        );
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::Ready, 5_000),
            None
        );
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::RetryLater, 5_000),
            None
        );
        // Terminal failure is NOT a loading state — no progress-ACK re-drive;
        // the caller dead-letters instead.
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::Failed, 5_000),
            None
        );
    }

    #[test]
    fn readiness_progress_delay_is_clamped_below_pool_ack_wait() {
        let max_progress_delay_ms =
            crate::nats_consumer::ACK_WAIT_SECS * 1_000 / READINESS_PROGRESS_ACK_WAIT_FRACTION;
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::LoadingStarted, 20_000),
            Some(max_progress_delay_ms)
        );
        assert_eq!(
            readiness_progress_delay_ms(&ReadinessState::LoadingInProgress, 20_000),
            Some(max_progress_delay_ms)
        );
    }

    #[test]
    fn resolve_outcome_indices_matches_by_wiid_in_arrival_order() {
        // Happy path: every outcome lines up with exactly one resolved
        // row, same order.
        let resolved = vec!["a", "b", "c"];
        let outcomes = ["a", "b", "c"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(bindings, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn resolve_outcome_indices_handles_out_of_order_outcomes() {
        // The executor may reorder outcomes within a batch (adapter
        // dedup / parallelism). Binding is by wiid, not slot.
        let resolved = vec!["a", "b", "c"];
        let outcomes = ["c", "a", "b"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(bindings, vec![Some(2), Some(0), Some(1)]);
    }

    #[test]
    fn resolve_outcome_indices_handles_duplicate_wiids_fifo() {
        // Two messages with the same wiid (pathological redelivery or
        // upstream bug) must each bind to a *different* resolved slot
        // in arrival order, so neither is silently dropped.
        let resolved = vec!["dup", "dup", "uniq"];
        let outcomes = ["dup", "uniq", "dup"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(
            bindings,
            vec![Some(0), Some(2), Some(1)],
            "duplicate 'dup' outcomes must consume resolved slots 0 then 1"
        );
    }

    #[test]
    fn resolve_outcome_indices_extra_outcome_is_ghost() {
        // Executor emitted more outcomes than items. The surplus must
        // be flagged (`None`) so the caller can log + drop it instead
        // of silently ACKing a phantom item.
        let resolved = vec!["a"];
        let outcomes = ["a", "ghost"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(bindings, vec![Some(0), None]);
    }

    #[test]
    fn resolve_outcome_indices_missing_outcome_leaves_orphan() {
        // Executor dropped an outcome. Returned bindings cover just
        // what was supplied; the caller walks `resolved` to NAK any
        // index not present in any `Some(idx)` binding.
        let resolved = vec!["a", "b", "c"];
        let outcomes = ["a", "c"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(bindings, vec![Some(0), Some(2)]);
        // Verifying the orphan-detection contract at the call site:
        let used: std::collections::HashSet<usize> = bindings.iter().filter_map(|o| *o).collect();
        let orphans: Vec<usize> = (0..resolved.len()).filter(|i| !used.contains(i)).collect();
        assert_eq!(orphans, vec![1], "index 1 ('b') should be the sole orphan");
    }

    #[test]
    fn resolve_outcome_indices_unknown_wiid_is_ghost() {
        // Outcome for a wiid that isn't in the batch at all.
        let resolved = vec!["a"];
        let outcomes = ["unknown"];
        let bindings = resolve_outcome_indices(&resolved, outcomes.iter().copied());
        assert_eq!(bindings, vec![None]);
    }

    // ----- dedupe_per_request_totals ----------------------------------------

    fn outcome(
        request_id: &str,
        item_index: u32,
        disposition: Disposition,
        inference_ms: Option<f64>,
        post_ms: Option<f64>,
    ) -> ItemOutcome {
        ItemOutcome {
            work_item_id: format!("{request_id}.{item_index}"),
            request_id: request_id.into(),
            item_index,
            disposition,
            nak_delay_ms: None,
            result_msgpack: Vec::new(),
            error: None,
            error_code: None,
            inference_ms,
            tokenization_ms: None,
            postprocessing_ms: post_ms,
            raw_output: None,
            units: None,
        }
    }

    #[test]
    fn dedupe_collapses_multiple_items_of_same_request() {
        // Three NATS work-items for `req-A` (multi-item /encode) plus
        // one solo `req-B` land in the same batch. Python's
        // `_complete_requests` records 2 latency samples (one per
        // unique `id(metadata)`); Rust must do the same so a 10-item
        // request doesn't pull `observed_p50_ms` 10× harder than ten
        // 1-item requests at matched throughput.
        let now = Instant::now();
        let t0 = now - std::time::Duration::from_millis(50);
        let oa0 = outcome(
            "req-A",
            0,
            Disposition::PublishAndAck,
            Some(20.0),
            Some(2.0),
        );
        let oa1 = outcome(
            "req-A",
            1,
            Disposition::PublishAndAck,
            Some(20.0),
            Some(2.0),
        );
        let oa2 = outcome(
            "req-A",
            2,
            Disposition::PublishAndAck,
            Some(20.0),
            Some(2.0),
        );
        let ob0 = outcome(
            "req-B",
            0,
            Disposition::PublishAndAck,
            Some(20.0),
            Some(2.0),
        );

        let totals = dedupe_per_request_totals(
            [
                ("req-A", t0, &oa0),
                ("req-A", t0, &oa1),
                ("req-A", t0, &oa2),
                ("req-B", t0, &ob0),
            ],
            now,
        );

        assert_eq!(totals.len(), 2, "one sample per unique request_id");
        // Each sample is wait(50) + inf(20) + post(2) = 72 ms within
        // a tight tolerance for the test scheduler.
        for s in &totals {
            assert!(
                (70.0..=80.0).contains(s),
                "sample {s} should be ~72ms (50 wait + 20 inf + 2 post)"
            );
        }
    }

    #[test]
    fn dedupe_skips_non_publish_dispositions() {
        // Errors / NAK paths must not bias the controller toward fast
        // no-op replies (they often have inference_ms = 0). Mirrors
        // Python's per-item `if not metadata.future.done()` gate.
        let now = Instant::now();
        let t0 = now - std::time::Duration::from_millis(10);
        let nak = outcome("req-X", 0, Disposition::NakRetry, Some(20.0), Some(0.0));
        let err = outcome(
            "req-Y",
            0,
            Disposition::PublishErrorAndAck,
            Some(20.0),
            Some(0.0),
        );
        let ok = outcome(
            "req-Z",
            0,
            Disposition::PublishAndAck,
            Some(20.0),
            Some(0.0),
        );

        let totals = dedupe_per_request_totals(
            [("req-X", t0, &nak), ("req-Y", t0, &err), ("req-Z", t0, &ok)],
            now,
        );

        assert_eq!(totals.len(), 1, "only PublishAndAck contributes");
    }

    #[test]
    fn dedupe_skips_zero_or_negative_totals() {
        // IPC failure with no timing fields populated — both `inf` and
        // `post` default to 0 and `submitted_at == now` makes the wait
        // 0 too. A zero sample would pin `observed_p50_ms` at the
        // floor and starve the wait knob.
        let now = Instant::now();
        let zero = outcome("req-zero", 0, Disposition::PublishAndAck, None, None);
        let totals = dedupe_per_request_totals([("req-zero", now, &zero)], now);
        assert!(totals.is_empty(), "zero-total samples must be dropped");
    }

    #[test]
    fn dedupe_keeps_first_occurrence_submitted_at() {
        // BatchFormer sorts items by cost so the first item we see for
        // a given request_id may not be `item_index == 0`. The dedup
        // should latch onto whichever submitted_at appears first in
        // iteration order (parity with Python, which uses the
        // request-level _start_time set once at request entry).
        let now = Instant::now();
        let early = now - std::time::Duration::from_millis(100);
        let late = now - std::time::Duration::from_millis(10);
        let o_late = outcome("req-A", 5, Disposition::PublishAndAck, Some(0.0), Some(0.0));
        let o_early = outcome("req-A", 0, Disposition::PublishAndAck, Some(0.0), Some(0.0));

        // Late item appears first (BatchFormer sorted by cost).
        let totals =
            dedupe_per_request_totals([("req-A", late, &o_late), ("req-A", early, &o_early)], now);

        assert_eq!(totals.len(), 1);
        let s = totals[0];
        assert!(
            (5.0..=20.0).contains(&s),
            "sample {s} should reflect the first-seen submitted_at (~10ms wait)"
        );
    }

    #[test]
    fn dedupe_empty_input_yields_empty_output() {
        let totals = dedupe_per_request_totals(std::iter::empty(), Instant::now());
        assert!(totals.is_empty());
    }
}
