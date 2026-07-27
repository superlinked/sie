//! Rust-side batch formation, adaptive control, and worker scheduling.
//!
//! This module is **live** in the dispatcher hot path. Every model
//! served by a worker-sidecar routes encode / score / extract
//! traffic through [`Scheduler`]: each request lands in a per-
//! `(operation, LoRA)` [`BatchFormer`], an [`AdaptiveBatchController`]
//! governs the flush caps, and a per-model drain loop (spawned
//! lazily on first traffic by
//! `crate::dispatcher::Dispatcher::resolve_scheduler`) packs the
//! flushed batch into a [`crate::ipc_types::RunBatchRequest`] and
//! hands it to the [`crate::backend::InferenceBackend`].
//!
//! Byte/semantics-for-byte ports of the Python originals in:
//!
//! * `sie_server/core/batcher.py` → [`BatchConfig`], [`FormattedBatch`],
//!   [`PendingRequest`], [`HasCost`], [`BatchFormer`].
//! * `sie_server/core/adaptive_batching.py` →
//!   [`BatchEfficiencyTracker`], [`AdaptiveBatchController`],
//!   [`AdaptiveBatchState`]. (The `LatencyTracker` used by the
//!   controller is shared with the existing IPC-fetch controller in
//!   [`crate::latency`].)
//!
//! See `docs/architecture-guide.md` for the current design notes. The
//! old per-model scheduler env list has been retired: the question of
//! "does this active model run under the Rust scheduler" is answered by
//! which worker pool handles the model, not by a per-worker env var.
//!
//! Invariants to preserve (same as the Python code; any deviation is a
//! production bug, not a design call):
//!
//! * **Flush triggers**: cost cap, count cap, `max_batch_wait_ms`
//!   timeout, or the adaptive coalesce window.
//! * **Oversize item**: a single item ≥ `max_batch_cost` flushes alone.
//! * **Cost-sorted sub-batch packing** on every extract.
//! * **Starvation recovery**: consecutive tiny batches at both floors
//!   trip a deadlock reset.
//! * **Auto-calibration**: inference-only p50 × `calibration_multiplier`
//!   sets the controller target when none was configured.

pub mod adaptive;
pub mod batch_config;
pub mod batch_former;
pub mod engine;
pub mod item;
pub mod registry;
pub mod trackers;

pub use adaptive::{AdaptiveBatchController, AdaptiveBatchControllerBuilder, AdaptiveBatchState};
pub use batch_config::BatchConfig;
pub use batch_former::{BatchFormer, FlushReason, FormattedBatch, HasCost, PendingRequest};
pub use engine::{LoraKey, Op, RecordCompletionOutcome, Scheduler, SchedulerBuilder};
pub use item::{lora_from_options, SchedulerItem, SchedulerMeta};
pub use registry::SchedulerRegistry;
pub use trackers::BatchEfficiencyTracker;

/// The concrete per-model scheduler type wired into the dispatcher
/// hot path — a [`Scheduler`] parameterised by the production item /
/// metadata pair from [`item`]. Type aliases keep the big signatures
/// readable in `dispatcher.rs` and `lib.rs` without forcing callers
/// to repeat the generic parameters.
pub type ProductionScheduler = Scheduler<SchedulerItem, SchedulerMeta>;

/// The concrete [`SchedulerRegistry`] type passed into the dispatcher.
pub type ProductionSchedulerRegistry = SchedulerRegistry<SchedulerItem, SchedulerMeta>;
