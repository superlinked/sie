//! Inference backend abstraction.
//!
//! A [`InferenceBackend`] is the thing the dispatcher calls to turn a
//! batch of `WorkItem`s into results. The queue-mode default is
//! [`AdapterWorkerPool`], which owns one IPC client per colocated adapter
//! child. [`PythonIpcBackend`] remains the single-socket IPC backend used by
//! tests and specialised call sites.
//!
//! # Why this exists
//!
//! The worker-sidecar owns NATS consumption, scheduling, and dispatch while
//! adapter processes own model execution over UDS + msgpack RPC. This trait
//! keeps those concerns separated: Python/PyTorch and Rust/Candle children can
//! share the dispatcher, IPC protocol, metrics, and config fanout paths.
//!
//! # Not a runtime fallback
//!
//! A [`BackendRouter`] composes multiple backends and routes each
//! operation to the first one that claims the model. Routing is
//! deterministic and config-driven (order of construction = priority).
//!
//! Critically, routing happens at readiness time — it is **not** a
//! retry-on-error fallback. If a native backend claims a model and
//! inference fails, the backend is responsible for fanning out
//! per-item error outcomes in an `Ok(BatchOutcome)` and the dispatcher
//! ACKs the messages (see the error handling contract below). The
//! router does **not** silently retry via Python. Numeric divergence
//! between two embedders is a correctness bug, not a feature, and we
//! refuse to paper over it.
//!
//! # Error handling contract
//!
//! The dispatcher's only lever is ACK vs NAK vs per-item publish. It
//! uses the backend's return value this way:
//!
//! * `Ok(BatchOutcome)` → dispatcher honours every [`crate::ipc_types::ItemOutcome`]'s
//!   [`crate::ipc_types::Disposition`]. A well-behaved backend that
//!   handles its own inference failures returns `Ok` with some items
//!   set to `PublishErrorAndAck` — the per-item path.
//! * `Err(BackendError::Transient)` → NAK the whole group with the
//!   base delay. Use for **infrastructure** failures (broken IPC pipe,
//!   CUDA unavailable, connection reset) where a different pod or the
//!   next redeliver might succeed.
//! * `Err(BackendError::UnsupportedModel)` → the router falls through
//!   to the next backend; if none claim it, the dispatcher NAKs so a
//!   different pod can pick it up.
//! * `Err(BackendError::Draining)` → NAK fast; we're shutting down.
//! * `Err(BackendError::Inference)` → **currently unused at the batch
//!   level.** Reserved for a future protocol where per-batch inference
//!   failure maps to synthetic per-item errors inside the dispatcher.
//!   Do not emit this from new backends — fan out to `PublishErrorAndAck`
//!   items inside `Ok(BatchOutcome)` instead. Emitting `Inference` today
//!   would cause the same retry-storm behaviour as `Transient`.
//!
//! # Contract
//!
//! * [`InferenceBackend::supports`] is advisory — used by the router
//!   to pick a backend cheaply before any RPC. The authoritative answer
//!   is [`InferenceBackend::ensure_model_ready`].
//! * [`InferenceBackend::ensure_model_ready`] returns the same
//!   [`EnsureModelReadyResponse`] the dispatcher already handles
//!   (Ready / LoadingStarted / LoadingInProgress / RetryLater plus an
//!   optional `batch_budget`). Backends populate it from their own
//!   load state; the dispatcher's NAK/ACK policy is unchanged.
//! * `process_*_batch` runs the batch and returns a [`BatchOutcome`]
//!   whose [`crate::ipc_types::ItemOutcome`]s drive per-message publish/ACK/NAK. The
//!   dispatcher has already capped the input at the budget reported
//!   by `ensure_model_ready`, so backends don't need to re-split.
//! * [`InferenceBackend::drain`] is called once at shutdown. The
//!   deadline is advisory — drains should finish promptly and never
//!   block shutdown indefinitely.

use std::sync::Arc;

use async_trait::async_trait;

use crate::ipc_types::{
    BatchOutcome, EnsureModelReadyResponse, ProcessEncodeBatchRequest, ProcessExtractBatchRequest,
    ProcessScoreBatchRequest, RunBatchRequest,
};

pub mod adapter_pool;
pub mod python_ipc;
pub mod router;

// The sidecar talks to colocated adapter children over IPC. GPU-independent
// pieces (tokenize, text prep, IPC types, output formatting) live in this crate
// under `prep` and `protocol`.

pub use adapter_pool::AdapterWorkerPool;
pub use python_ipc::PythonIpcBackend;
pub use router::BackendRouter;

/// Error taxonomy for backend calls.
///
/// See the **Error handling contract** in the module docstring above
/// for how each variant maps to ACK / NAK / per-item dispatcher
/// behaviour. In particular: **do not emit [`BackendError::Inference`]
/// from new backends**; convert per-batch inference failures into
/// per-item `PublishErrorAndAck` outcomes inside an `Ok(BatchOutcome)`.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Transport / infrastructure glitch. Dispatcher NAKs with the
    /// base delay; JetStream redelivers.
    #[error("transient: {0}")]
    Transient(String),

    /// **Reserved.** Batch-level inference failure. Currently no
    /// dispatcher mapping — emitting this causes NAK-and-redeliver
    /// (same as [`BackendError::Transient`]), which can retry-storm.
    /// New backends should fan out per-item errors inside a successful
    /// [`BatchOutcome`] instead.
    #[error("inference: {0}")]
    Inference(String),

    /// No backend in the router claims this model (or the one that
    /// did later rejected it). Router tries the next; if none
    /// remain, dispatcher NAKs so another pod can pick it up.
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),

    /// Backend is shutting down. NAK fast.
    #[error("draining")]
    Draining,
}

impl BackendError {
    /// True if the dispatcher should NAK the group rather than error
    /// out per-item. Transport / readiness / drain failures fall here.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            BackendError::Transient(_) | BackendError::UnsupportedModel(_) | BackendError::Draining
        )
    }
}

/// Trait object implementing inference for one or more models.
///
/// Backends are long-lived, shared across the dispatcher fan-out via
/// `Arc`, and must be `Send + Sync`. A backend that loads weights
/// eagerly should do so at construction; one that loads lazily should
/// surface the load state through [`InferenceBackend::ensure_model_ready`].
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Short, stable name for routing logs (e.g. `"python-ipc"`,
    /// `"native-bert"`).
    fn name(&self) -> &'static str;

    /// Cheap, synchronous check: does this backend handle `model_id`?
    ///
    /// The router uses this to pick a backend before issuing any RPC.
    /// Returning `true` does **not** mean the model is loaded — only
    /// that this backend is responsible for it.
    ///
    /// A backend that serves every model (like [`PythonIpcBackend`])
    /// returns `true` unconditionally. A specialised backend returns
    /// `true` only for models in its responsibility set.
    fn supports(&self, model_id: &str) -> bool;

    /// Ensure the model is loaded (or kick off a load) and report
    /// readiness + the per-batch budget the dispatcher should honour.
    ///
    /// Semantics match the Python `ensure_model_ready` IPC method
    /// exactly: see [`crate::ipc_types::ReadinessState`] for the
    /// state machine.
    async fn ensure_model_ready(
        &self,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, BackendError>;

    /// Run an encode batch. Inputs are a flat list of prepared items;
    /// the backend is responsible for its own internal batching and
    /// tokenisation.
    async fn process_encode_batch(
        &self,
        req: ProcessEncodeBatchRequest,
    ) -> Result<BatchOutcome, BackendError>;

    /// Run a score batch (one query vs. N documents, typically).
    async fn process_score_batch(
        &self,
        req: ProcessScoreBatchRequest,
    ) -> Result<BatchOutcome, BackendError>;

    /// Run an extract batch.
    async fn process_extract_batch(
        &self,
        req: ProcessExtractBatchRequest,
    ) -> Result<BatchOutcome, BackendError>;

    /// Run a pre-formed batch assembled by the Rust-side scheduler
    /// (see [`crate::scheduler::Scheduler`]). Every item in the
    /// batch shares the same `op` (the scheduler's batchers are
    /// keyed per-`(op, lora)`), so a backend that lacks native
    /// `RunBatch` support can fan out into the matching
    /// `process_*_batch` call internally — see
    /// [`crate::backend::PythonIpcBackend::run_batch`] for the
    /// reference implementation.
    ///
    /// The default implementation returns
    /// [`BackendError::UnsupportedModel`] so a backend that only
    /// implements the per-op `process_*_batch` calls doesn't silently
    /// swallow `RunBatch` requests. The [`crate::backend::BackendRouter`]
    /// falls through to the next registered backend on this error so
    /// the Python IPC catch-all still wins.
    async fn run_batch(&self, req: RunBatchRequest) -> Result<BatchOutcome, BackendError> {
        let _ = req;
        Err(BackendError::UnsupportedModel(format!(
            "run_batch not implemented on {}",
            self.name()
        )))
    }

    /// Graceful shutdown. Called once when the worker has stopped
    /// pulling from NATS. Backends should:
    ///
    /// * Reject new work (return `BackendError::Draining` from
    ///   subsequent `process_*` calls if any race in).
    /// * Wait for in-flight requests to settle, up to `deadline_ms`.
    /// * Release resources (close sockets, free GPU memory).
    ///
    /// The default implementation is a no-op so native backends can
    /// opt out when their state lives in-process.
    async fn drain(&self, _deadline_ms: u64) {}
}

/// Convenience alias used throughout the dispatcher.
pub type SharedBackend = Arc<dyn InferenceBackend>;
