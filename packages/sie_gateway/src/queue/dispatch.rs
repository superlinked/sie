//! Transport-neutral dispatch seam between the HTTP handlers and the
//! queue publisher (roadmap P2.1, design §4.2/§13).
//!
//! [`WorkDispatcher`] captures the exact handler-facing surface of
//! [`WorkPublisher`] — publish, cancel, republish, and pending-state
//! snapshots — so handlers depend on `Arc<dyn WorkDispatcher>` rather
//! than the concrete NATS-backed publisher. NATS stays the only
//! implementation; lifecycle methods (inbox subscription, backpressure
//! monitor, cleanup, drain) deliberately stay on the concrete type
//! owned by `main.rs`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use thiserror::Error;
use tokio::sync::{broadcast, oneshot};
use tracing::warn;

use super::publisher::WorkPublisher;
// Re-exports for future consumers of the dispatch seam (also serve as
// this module's own imports for the trait signatures below).
pub use super::publisher::{PendingGenerationSnapshot, PublishTarget, WorkParams, WorkResult};
pub use super::streaming::{ChunkEnvelope, StreamOutcome};

/// Transport-neutral completion of the dispatch durability boundary.
///
/// NATS resolves this only after every initial JetStream publish is durably
/// acknowledged. Managed Modal dispatch resolves it at its existing accepted
/// local-dispatch boundary. Handlers use it to hand pending demand over to the
/// authoritative queue signal without waiting on a broker RTT in the request
/// path.
pub struct DispatchDurability {
    completion: Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'static>>,
}

/// Collector shape installed before a dispatch is submitted.
///
/// The durability monitor uses this transport-neutral discriminator to tear
/// down the exact pending collector if durable acceptance later fails.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PendingDispatchKind {
    Result,
    Stream,
}

impl DispatchDurability {
    pub(crate) fn from_future(
        completion: impl Future<Output = Result<(), String>> + Send + 'static,
    ) -> Self {
        Self {
            completion: Box::pin(completion),
        }
    }

    #[allow(dead_code)] // Managed Modal dispatcher boundary.
    pub fn accepted() -> Self {
        Self::from_result(Ok(()))
    }

    pub(crate) fn from_result(result: Result<(), String>) -> Self {
        Self::from_future(async move { result })
    }

    pub async fn wait(self) -> Result<(), String> {
        self.completion.await
    }
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct DispatchPayloadTooLarge {
    message: String,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct DispatchInvalidInput {
    message: String,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct DispatchBackpressure {
    message: String,
}

impl From<String> for DispatchPayloadTooLarge {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for DispatchPayloadTooLarge {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl From<String> for DispatchInvalidInput {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for DispatchInvalidInput {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl From<String> for DispatchBackpressure {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for DispatchBackpressure {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error(transparent)]
    PayloadTooLarge(#[from] DispatchPayloadTooLarge),
    #[error(transparent)]
    InvalidInput(#[from] DispatchInvalidInput),
    #[error(transparent)]
    Backpressure(#[from] DispatchBackpressure),
    #[error("{0}")]
    Other(String),
}

impl DispatchPayloadTooLarge {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl DispatchInvalidInput {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl DispatchBackpressure {
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<String> for DispatchError {
    fn from(message: String) -> Self {
        Self::Other(message)
    }
}

/// Handler→publisher dispatch boundary. Signatures mirror the inherent
/// [`WorkPublisher`] methods exactly; see the inherent methods for the
/// full behavioural documentation.
#[async_trait]
pub trait WorkDispatcher: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn publish_work(
        self: Arc<Self>,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<
        (
            String,
            oneshot::Receiver<Vec<WorkResult>>,
            DispatchDurability,
        ),
        DispatchError,
    >;

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            Arc<tokio::sync::Notify>,
            DispatchDurability,
        ),
        String,
    >;

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
            DispatchDurability,
        ),
        String,
    >;

    async fn publish_cancel(&self, request_id: &str);

    /// Remove request-local collector state after a late durability failure.
    ///
    /// Implementations with external payload storage should also begin
    /// best-effort payload cleanup. The default keeps managed/test dispatchers
    /// source-compatible; their accepted boundary cannot fail after return.
    async fn abort_pending_dispatch(&self, request_id: &str, kind: PendingDispatchKind) {
        if kind == PendingDispatchKind::Stream {
            self.drop_pending_stream(request_id);
        }
    }

    /// Claim synchronous ownership of a non-streaming request's abandonment.
    /// Returns true only for the caller that won the transport's terminal race.
    fn begin_work_abandonment(&self, request_id: &str) -> bool;

    /// Finish transport-specific cancellation and retained-resource cleanup.
    async fn finish_work_abandonment(&self, request_id: &str);

    async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String>;

    async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String>;

    fn drop_pending_stream(&self, request_id: &str);

    fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot;

    fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot;

    fn stream_observed_first_chunk(&self, request_id: &str) -> bool;

    fn stream_chunk_timing(&self, request_id: &str) -> Option<(Option<Instant>, Option<Instant>)>;
}

#[async_trait]
impl WorkDispatcher for WorkPublisher {
    async fn publish_work(
        self: Arc<Self>,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<
        (
            String,
            oneshot::Receiver<Vec<WorkResult>>,
            DispatchDurability,
        ),
        DispatchError,
    > {
        WorkPublisher::publish_work(
            &self,
            target,
            admission_pool,
            endpoint,
            model,
            engine,
            bundle_config_hash,
            items,
            params,
        )
        .await
        .map_err(DispatchError::from)
    }

    async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            Arc<tokio::sync::Notify>,
            DispatchDurability,
        ),
        String,
    > {
        WorkPublisher::publish_generate_streaming(
            self,
            target,
            display_model,
            engine,
            bundle_config_hash,
            params,
            admission_pool,
        )
        .await
    }

    async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
            DispatchDurability,
        ),
        String,
    > {
        WorkPublisher::publish_generate_streaming_sse(
            self,
            target,
            display_model,
            engine,
            bundle_config_hash,
            params,
            admission_pool,
        )
        .await
    }

    async fn publish_cancel(&self, request_id: &str) {
        WorkPublisher::publish_cancel(self, request_id).await
    }

    async fn abort_pending_dispatch(&self, request_id: &str, kind: PendingDispatchKind) {
        WorkPublisher::abort_pending_dispatch(self, request_id, kind).await
    }

    fn begin_work_abandonment(&self, request_id: &str) -> bool {
        WorkPublisher::drop_pending_result(self, request_id)
    }

    async fn finish_work_abandonment(&self, request_id: &str) {
        WorkPublisher::finish_abandoned_work(self, request_id).await
    }

    async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        WorkPublisher::republish_to_pool(self, request_id, reason).await
    }

    async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        WorkPublisher::republish_pending_result_to_pool(self, request_id, reason).await
    }

    fn drop_pending_stream(&self, request_id: &str) {
        WorkPublisher::drop_pending_stream(self, request_id)
    }

    fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
        WorkPublisher::pending_generation_snapshot(self)
    }

    fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot {
        WorkPublisher::pending_generation_for_model(self, model_id)
    }

    fn stream_observed_first_chunk(&self, request_id: &str) -> bool {
        WorkPublisher::stream_observed_first_chunk(self, request_id)
    }

    fn stream_chunk_timing(&self, request_id: &str) -> Option<(Option<Instant>, Option<Instant>)> {
        WorkPublisher::stream_chunk_timing(self, request_id)
    }
}

/// Extension methods that need an owned `Arc<dyn WorkDispatcher>` handle
/// (they spawn detached tasks holding a clone), so they cannot live on
/// the object-safe [`WorkDispatcher`] trait itself.
pub trait WorkDispatcherExt {
    /// Arm a one-shot recovery for non-streaming worker-direct batch work.
    ///
    /// Capped logical pools direct-dispatch encode/score/extract to one
    /// admitted worker so unassigned peers do not burn the JetStream
    /// delivery budget. If that worker or its private consumer disappears
    /// after the gateway publishes, the worker-specific stream cannot fail
    /// over by itself. This timer republishes any still-missing items to the
    /// lane's pool subject once; late duplicate results are harmless because
    /// the result collector removes itself on first complete response.
    fn spawn_batch_direct_fallback(&self, request_id: String, delay: Duration);
}

impl WorkDispatcherExt for Arc<dyn WorkDispatcher> {
    fn spawn_batch_direct_fallback(&self, request_id: String, delay: Duration) {
        let publisher = Arc::clone(self);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            if let Err(e) = publisher
                .republish_pending_result_to_pool(&request_id, "batch_direct_timeout")
                .await
            {
                warn!(
                    request_id = %request_id,
                    error = %e,
                    "batch direct-dispatch fallback failed"
                );
            }
        });
    }
}

#[cfg(test)]
mod durability_tests {
    use super::DispatchDurability;

    #[tokio::test]
    async fn immediate_transport_acceptance_resolves_successfully() {
        DispatchDurability::accepted()
            .wait()
            .await
            .expect("accepted dispatch");
    }

    #[tokio::test]
    async fn failed_completion_fails_closed() {
        let durability =
            DispatchDurability::from_result(Err("durability monitor unavailable".to_string()));
        assert!(durability
            .wait()
            .await
            .unwrap_err()
            .contains("monitor unavailable"));
    }
}

#[cfg(test)]
mod performance_tests {
    use std::hint::black_box;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    use super::*;
    use crate::handlers::proxy::monitor_dispatch_durability;
    use crate::state::demand_tracker::{DemandTracker, PhysicalLane, PhysicalLaneCatalog};

    #[derive(Default)]
    struct AckCompletingDispatcher {
        cancel_calls: AtomicUsize,
        aborted: Mutex<Vec<(String, PendingDispatchKind)>>,
        cleanup_completed: tokio::sync::Notify,
    }

    #[async_trait::async_trait]
    impl WorkDispatcher for AckCompletingDispatcher {
        async fn publish_work(
            self: Arc<Self>,
            _target: PublishTarget,
            _admission_pool: &str,
            _endpoint: &str,
            _model: &str,
            _engine: &str,
            _bundle_config_hash: &str,
            _items: Vec<rmpv::Value>,
            _params: &WorkParams,
        ) -> Result<
            (
                String,
                oneshot::Receiver<Vec<WorkResult>>,
                DispatchDurability,
            ),
            DispatchError,
        > {
            let (_result_sender, result_receiver) = oneshot::channel();
            let (ack_sender, ack_receiver) = oneshot::channel();
            ack_sender
                .send(())
                .map_err(|_| "benchmark ACK receiver closed".to_string())?;
            let durability = DispatchDurability::from_future(async move {
                ack_receiver
                    .await
                    .map_err(|_| "benchmark ACK sender closed".to_string())
            });
            Ok(("benchmark-request".to_string(), result_receiver, durability))
        }

        async fn publish_generate_streaming(
            &self,
            _target: PublishTarget,
            _display_model: &str,
            _engine: &str,
            _bundle_config_hash: &str,
            _params: &WorkParams,
            _admission_pool: &str,
        ) -> Result<
            (
                String,
                oneshot::Receiver<StreamOutcome>,
                Arc<tokio::sync::Notify>,
                DispatchDurability,
            ),
            String,
        > {
            panic!("streaming dispatch is outside this benchmark")
        }

        async fn publish_generate_streaming_sse(
            &self,
            _target: PublishTarget,
            _display_model: &str,
            _engine: &str,
            _bundle_config_hash: &str,
            _params: &WorkParams,
            _admission_pool: &str,
        ) -> Result<
            (
                String,
                oneshot::Receiver<StreamOutcome>,
                broadcast::Receiver<ChunkEnvelope>,
                DispatchDurability,
            ),
            String,
        > {
            panic!("SSE dispatch is outside this benchmark")
        }

        async fn publish_cancel(&self, _request_id: &str) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn abort_pending_dispatch(&self, request_id: &str, kind: PendingDispatchKind) {
            self.aborted
                .lock()
                .expect("benchmark abort lock")
                .push((request_id.to_string(), kind));
            self.cleanup_completed.notify_one();
        }

        fn begin_work_abandonment(&self, _request_id: &str) -> bool {
            false
        }

        async fn finish_work_abandonment(&self, _request_id: &str) {}

        async fn republish_to_pool(
            &self,
            _request_id: &str,
            _reason: &'static str,
        ) -> Result<bool, String> {
            Ok(false)
        }

        async fn republish_pending_result_to_pool(
            &self,
            _request_id: &str,
            _reason: &'static str,
        ) -> Result<bool, String> {
            Ok(false)
        }

        fn drop_pending_stream(&self, _request_id: &str) {}

        fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
            PendingGenerationSnapshot::default()
        }

        fn pending_generation_for_model(&self, _model_id: &str) -> PendingGenerationSnapshot {
            PendingGenerationSnapshot::default()
        }

        fn stream_observed_first_chunk(&self, _request_id: &str) -> bool {
            false
        }

        fn stream_chunk_timing(
            &self,
            _request_id: &str,
        ) -> Option<(Option<Instant>, Option<Instant>)> {
            None
        }
    }

    async fn exercise_dispatch_handoff(
        dispatcher: &Arc<dyn WorkDispatcher>,
        demand_tracker: &Arc<DemandTracker>,
        physical_lane: &PhysicalLane,
        target: &PublishTarget,
        params: &WorkParams,
        iterations: usize,
    ) {
        for _ in 0..iterations {
            let (request_id, result_receiver, durability) = Arc::clone(dispatcher)
                .publish_work(
                    black_box(target.clone()),
                    black_box("default"),
                    black_box("encode"),
                    black_box("catalog/model"),
                    black_box("benchmark"),
                    black_box(""),
                    Vec::new(),
                    black_box(params),
                )
                .await
                .expect("benchmark dispatch");
            drop(result_receiver);
            monitor_dispatch_durability(
                Arc::clone(demand_tracker),
                physical_lane.clone(),
                durability,
                Arc::clone(dispatcher),
                request_id,
                PendingDispatchKind::Result,
            )
            .await
            .expect("durability monitor channel")
            .expect("durable dispatch");
        }
        assert!(
            demand_tracker.active_lanes().is_empty(),
            "durable completion must clear the exact pending lane"
        );
    }

    async fn exercise_dispatch_without_handoff(
        dispatcher: &Arc<dyn WorkDispatcher>,
        target: &PublishTarget,
        params: &WorkParams,
        iterations: usize,
    ) {
        for _ in 0..iterations {
            let (_request_id, result_receiver, durability) = Arc::clone(dispatcher)
                .publish_work(
                    black_box(target.clone()),
                    black_box("default"),
                    black_box("encode"),
                    black_box("catalog/model"),
                    black_box("benchmark"),
                    black_box(""),
                    Vec::new(),
                    black_box(params),
                )
                .await
                .expect("benchmark dispatch");
            drop(result_receiver);
            durability.wait().await.expect("benchmark durability");
        }
    }

    fn demand_tracker_with_lanes(
        lanes: &[PhysicalLane],
    ) -> (Arc<DemandTracker>, Arc<AckCompletingDispatcher>) {
        let catalog = PhysicalLaneCatalog::try_new(lanes.iter().cloned()).unwrap();
        (
            Arc::new(DemandTracker::new(catalog)),
            Arc::new(AckCompletingDispatcher::default()),
        )
    }

    #[tokio::test]
    async fn successful_durability_releases_only_its_request_scoped_lease() {
        let physical_lane = PhysicalLane::try_new("default", "l4", "default").unwrap();
        let unrelated_lane = PhysicalLane::try_new("other", "l4", "default").unwrap();
        let (demand_tracker, dispatcher) =
            demand_tracker_with_lanes(&[physical_lane.clone(), unrelated_lane.clone()]);
        let handoff = demand_tracker
            .begin_dispatch_handoff(&physical_lane)
            .unwrap();
        assert!(demand_tracker.record(&unrelated_lane));

        let dispatcher_dyn: Arc<dyn WorkDispatcher> = dispatcher.clone();
        let completion = monitor_dispatch_durability(
            Arc::clone(&demand_tracker),
            physical_lane.clone(),
            DispatchDurability::from_result(Ok(())),
            dispatcher_dyn,
            "success-request".to_string(),
            PendingDispatchKind::Result,
        );
        completion
            .await
            .expect("durability monitor channel")
            .expect("durable dispatch");

        assert_eq!(
            demand_tracker.active_lanes(),
            vec![physical_lane.clone(), unrelated_lane.clone()],
            "the monitor must release only its own lease"
        );
        demand_tracker.finish_dispatch_handoff(&physical_lane, handoff, true);
        assert_eq!(demand_tracker.active_lanes(), vec![unrelated_lane.clone()]);
        demand_tracker.clear(&unrelated_lane);
        assert!(demand_tracker.active_lanes().is_empty());
        assert_eq!(dispatcher.cancel_calls.load(Ordering::Relaxed), 0);
        assert!(dispatcher.aborted.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn failed_durability_notifies_retains_exact_lane_and_cleans_up() {
        let physical_lane = PhysicalLane::try_new("default", "l4", "default").unwrap();
        let unrelated_lane = PhysicalLane::try_new("other", "l4", "default").unwrap();
        let (demand_tracker, dispatcher) =
            demand_tracker_with_lanes(&[physical_lane.clone(), unrelated_lane.clone()]);
        assert!(demand_tracker.record(&unrelated_lane));

        let dispatcher_dyn: Arc<dyn WorkDispatcher> = dispatcher.clone();
        let completion = monitor_dispatch_durability(
            Arc::clone(&demand_tracker),
            physical_lane.clone(),
            DispatchDurability::from_result(Err("broker rejected publish".to_string())),
            dispatcher_dyn,
            "failed-request".to_string(),
            PendingDispatchKind::Result,
        );
        let error = completion
            .await
            .expect("durability monitor channel")
            .expect_err("failed durability must notify the request driver");
        assert!(error.contains("broker rejected publish"));
        tokio::time::timeout(
            Duration::from_secs(1),
            dispatcher.cleanup_completed.notified(),
        )
        .await
        .expect("late-ACK cleanup must complete");

        assert_eq!(
            demand_tracker.active_lanes(),
            vec![physical_lane, unrelated_lane],
            "failure must retain only the failed exact lane plus unrelated demand"
        );
        assert_eq!(dispatcher.cancel_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            *dispatcher.aborted.lock().unwrap(),
            vec![("failed-request".to_string(), PendingDispatchKind::Result)]
        );
    }

    /// Release microbenchmark for the real handler-facing dispatch seam and
    /// detached durability handoff. The disabled baseline publishes and awaits
    /// the same ACK-shaped durability future directly; the enabled path adds
    /// the production handoff task, completion notification, and exact-lane
    /// demand record/clear. Broker RTT is excluded. One run collects three
    /// independently warmed paired samples and gates disabled, enabled, and
    /// incremental medians.
    /// Run with:
    /// `SIE_RUN_TELEMETRY_BENCHMARK=1 mise exec -- cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib dispatch_durability_lifecycle_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    async fn dispatch_durability_lifecycle_microbenchmark() {
        const SAMPLES: usize = 3;
        const WARMUP_ITERATIONS: usize = 200;
        const ITERATIONS: usize = 2_000;

        assert_eq!(
            std::env::var("SIE_RUN_TELEMETRY_BENCHMARK").as_deref(),
            Ok("1"),
            "opt in with SIE_RUN_TELEMETRY_BENCHMARK=1"
        );

        let physical_lane = PhysicalLane::try_new("default", "l4", "default").unwrap();
        let (demand_tracker, dispatcher) =
            demand_tracker_with_lanes(std::slice::from_ref(&physical_lane));
        let dispatcher: Arc<dyn WorkDispatcher> = dispatcher;
        let target = PublishTarget::Pool {
            pool: "default".to_string(),
            machine_profile: "l4".to_string(),
            bundle: "default".to_string(),
            model: "catalog/model".to_string(),
        };
        let params = WorkParams::default();

        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise_dispatch_without_handoff(&dispatcher, &target, &params, WARMUP_ITERATIONS)
                .await;
            exercise_dispatch_handoff(
                &dispatcher,
                &demand_tracker,
                &physical_lane,
                &target,
                &params,
                WARMUP_ITERATIONS,
            )
            .await;

            let started = std::time::Instant::now();
            exercise_dispatch_without_handoff(&dispatcher, &target, &params, ITERATIONS).await;
            disabled_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let started = std::time::Instant::now();
            exercise_dispatch_handoff(
                &dispatcher,
                &demand_tracker,
                &physical_lane,
                &target,
                &params,
                ITERATIONS,
            )
            .await;
            enabled_samples[sample_index] = started.elapsed().as_nanos() as f64 / ITERATIONS as f64;
        }

        let disabled_median_ns =
            crate::observability::metrics::telemetry_benchmark_median(disabled_samples);
        let enabled_median_ns =
            crate::observability::metrics::telemetry_benchmark_median(enabled_samples);
        let incremental_median_ns = (enabled_median_ns - disabled_median_ns).max(0.0);
        println!(
            "gateway_dispatch_durability_lifecycle disabled_samples={disabled_samples:?} disabled_median_ns_per_dispatch={disabled_median_ns:.2} enabled_samples={enabled_samples:?} enabled_median_ns_per_dispatch={enabled_median_ns:.2} incremental_median_ns_per_dispatch={incremental_median_ns:.2} iterations_per_sample={ITERATIONS} broker_rtt=excluded"
        );
        let disabled_budget = crate::observability::metrics::telemetry_performance_budget(
            "gateway_dispatch_durability_disabled_ns_per_dispatch",
        );
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway durability-disabled median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let enabled_budget = crate::observability::metrics::telemetry_performance_budget(
            "gateway_dispatch_durability_enabled_ns_per_dispatch",
        );
        assert!(
            enabled_median_ns <= enabled_budget,
            "gateway durability-enabled median {enabled_median_ns:.2} ns exceeded {enabled_budget:.2} ns budget"
        );
        let incremental_budget = crate::observability::metrics::telemetry_performance_budget(
            "gateway_dispatch_durability_incremental_ns_per_dispatch",
        );
        assert!(
            incremental_median_ns <= incremental_budget,
            "gateway durability incremental median {incremental_median_ns:.2} ns exceeded {incremental_budget:.2} ns budget"
        );
    }
}
