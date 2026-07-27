//! Library root for the `sie-server-sidecar` binary.
//!
//! Sidecar-only build artifacts use the `sie-server-sidecar` name, while the
//! Kubernetes container remains `worker-sidecar` because it runs in worker Pods.
//! See
//! `packages/sie_server_sidecar/docs/architecture-guide.md` for the current
//! runtime contract.

pub mod audio_prep;
pub mod backend;
pub mod batch_cancel;
pub mod config;
pub mod config_reconciler;
pub mod config_subscriber;
pub mod delivery;
pub mod dispatcher;
pub mod health_publisher;
pub mod ipc_client;
pub mod ipc_mux;
pub mod ipc_types;
pub mod latency;
pub mod local_ingest;
pub mod log_util;
pub mod nats_consumer;
pub mod observability;
pub mod output;
pub mod payload_store;
pub mod pinned_reconciler;
pub mod pool_admission;
pub mod prep;
pub mod protocol;
pub mod publisher;
pub mod readiness;
pub mod runtime_state;
pub mod scheduler;
pub mod shutdown;
pub mod subject;
pub mod tokenize;
pub mod work_types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use futures_util::StreamExt;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio::time::{interval, sleep, timeout};
use tracing::{debug, info, warn};

use crate::backend::{AdapterWorkerPool, BackendRouter, SharedBackend};
use crate::batch_cancel::{
    request_id_from_batch_cancel_subject, request_id_from_work_cancel_subject, BatchCancelState,
    RequestCancelState,
};
use crate::config::WorkerConfig;
use crate::config_subscriber::ConfigApplyState;
use crate::dispatcher::{Dispatcher, QueuedMessage};
use crate::latency::{FetchExpiryController, LatencyTracker};
use crate::log_util::ErrChain;
use crate::nats_consumer::{
    connect, ensure_stream_and_consumer, ensure_worker_stream_and_consumer,
    reconcile_stream_and_consumer, reconcile_worker_stream_and_consumer, work_cancel_tombstone_ttl,
    NatsConsumer,
};
use crate::payload_store::{create_payload_store, with_telemetry as payload_store_with_telemetry};
use crate::pool_admission::PoolAdmissionGate;
use crate::publisher::WorkPublisher;
use crate::readiness::Readiness;
use crate::runtime_state::RuntimeState;
use crate::shutdown::Shutdown;
use crate::tokenize::TokenizerRegistry;

/// How many messages to request per NATS pull-stream refill. Matches
/// Python's historical `_DEFAULT_BATCH_BUDGET` (64) so Rust and Python
/// workers have the same per-fetch capacity.
///
/// Reads `SIE_NATS_FETCH_BUDGET` (the operator-facing env var wired by
/// the helm chart — see
/// `deploy/helm/sie-cluster/templates/worker-statefulset.yaml`), then
/// falls back to 64.
pub(crate) fn fetch_batch_size() -> usize {
    std::env::var("SIE_NATS_FETCH_BUDGET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// Server-side pull-request expiry. When async-nats opens a pull
/// subscription it asks the server to send up to `fetch_batch_size()`
/// messages or wait this long before replying with a `408 TIMEOUT`. The
/// stream auto-issues the next pull; this is not the dispatcher batch
/// size nor the client-side coalesce quantum.
///
/// Must stay strictly less than `ack_wait` (30 s) so a stuck stream
/// surfaces as a heartbeat/timeout error instead of silently blocking
/// redelivery. Override with `SIE_NATS_PULL_EXPIRES_S` (default 5).
pub(crate) fn pull_stream_expires() -> Duration {
    let secs = std::env::var("SIE_NATS_PULL_EXPIRES_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0 && n < 30)
        .unwrap_or(5);
    Duration::from_secs(secs)
}

const NATS_CONSUMER_RECONCILE_INTERVAL_ENV: &str = "SIE_NATS_CONSUMER_RECONCILE_INTERVAL_MS";
const DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS: u64 = 30_000;
const MIN_NATS_CONSUMER_RECONCILE_INTERVAL_MS: u64 = 10_000;

fn parse_nats_consumer_reconcile_interval_ms(raw: Option<&str>) -> Option<u64> {
    match raw.map(str::trim) {
        Some("0") => None,
        Some(value) if !value.is_empty() => value
            .parse::<u64>()
            .ok()
            .filter(|millis| *millis > 0)
            .map(|millis| millis.max(MIN_NATS_CONSUMER_RECONCILE_INTERVAL_MS))
            .or(Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS)),
        _ => Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS),
    }
}

fn nats_consumer_reconcile_interval() -> Option<Duration> {
    parse_nats_consumer_reconcile_interval_ms(
        std::env::var(NATS_CONSUMER_RECONCILE_INTERVAL_ENV)
            .ok()
            .as_deref(),
    )
    .map(Duration::from_millis)
}

/// Drain deadline sent to adapter IPC on shutdown.
const DRAIN_DEADLINE_MS: u64 = 15_000;

/// Whether the pull-loop quantum's [`LatencyTracker`] should observe
/// the upstream `queue_ms` (gateway-publish → NATS-pull) on top of
/// `inference_ms + postprocess_ms`.
///
/// **Default `false`** — excludes `queue_ms` from the controller's
/// input. Rationale (see `dispatcher.rs::publish_outcome_for_item`):
/// under saturation `queue_ms` rises into seconds, which would drive
/// the controller's `headroom_ms = target − observed` deeply negative
/// and collapse the quantum to its floor, even though the pull-loop
/// itself is not the bottleneck.
///
/// **Set to `true`** to recover parity with the former Python queue-loop
/// tracker, where `total_ms = queue_times[idx] + inference_ms +
/// postprocess_ms`. With the tighter local defaults
/// (`MIN_QUANTUM_MS=2`, `MAX_QUANTUM_MS=15`, `TARGET_P50_MS=50`)
/// the asymmetry has narrowed substantially, but the flag remains as
/// an explicit compatibility knob.
/// Tracked as a known parity divergence in `docs/architecture-guide.md`.
///
/// Cached after first call: read once per process, atomic check
/// thereafter — keeps the per-outcome `record` path branch-free
/// in the hot loop without paying for `std::env::var()` each time.
pub(crate) fn pull_quantum_includes_queue_ms() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS")
            .ok()
            .map(|v| {
                let trimmed = v.trim();
                matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

/// Whether the scheduler should step the adaptive controller exactly
/// once per wave (Primary only) or once per batch (Primary + every
/// Drain).
///
/// Defaults to **true** — per-wave stepping. Set
/// `SIE_RUST_WAVE_CADENCE=off` (also accepted: `0`,
/// `false`, `no`, `disabled`) to fall back to per-batch stepping.
///
/// **Why a toggle.** Per-wave stepping matches Python `_process_loop`'s
/// cadence and stops the PI loop from biasing toward the smaller
/// drain-batch sizes. Per-batch stepping is useful only as an explicit
/// latency/throughput tradeoff knob.
///
/// The cost: per-wave stepping can leave the wait knob at a value that
/// produces larger batches than the old Python-only path. Operators who
/// care more about light-load p50 than saturated p99 can flip this off
/// to recover smaller batch density.
///
/// At `SIE_RUST_PIPELINE_DEPTH=1`, backend calls remain strictly serial, but
/// work accumulated during a Primary can still form a bounded Drain after that
/// call releases the sole permit. The toggle therefore still controls adaptive
/// controller cadence at depth one; it does not control dispatch concurrency.
///
/// Cached after first call (`OnceLock`): same hot-path treatment as
/// `pull_quantum_includes_queue_ms` — the per-batch gate must stay
/// branch-free without paying for `std::env::var()` each call.
pub(crate) fn wave_cadence_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("SIE_RUST_WAVE_CADENCE")
            .ok()
            .map(|v| {
                let trimmed = v.trim();
                !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "off" | "false" | "no" | "disabled"
                )
            })
            .unwrap_or(true)
    })
}

/// Whether the NATS health-heartbeat publisher should run.
///
/// Defaults to `true` because the sidecar pod runs without
/// `/ws/status` (that endpoint lives on the Python container), which
/// forces the gateway into `health_mode=nats` — and that mode is a
/// no-op until at least
/// one worker publishes heartbeats. Override with
/// `SIE_HEALTH_PUBLISH_ENABLED=0` (or `false` / `no`) on legacy
/// WS-health deployments where the gateway ignores `sie.health.>`.
pub(crate) fn health_publish_enabled() -> bool {
    let raw = match std::env::var("SIE_HEALTH_PUBLISH_ENABLED") {
        Ok(v) => v,
        Err(_) => return true,
    };
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off" | ""
    )
}

/// Max number of `handle_batch` tasks the pull loop may have in flight at
/// once. Each in-flight task holds a full NATS fetch in memory and a
/// model-group IPC request against Python, so this bounds both memory and
/// the effective Rust→Python pipeline depth. Reads
/// `SIE_PULL_LOOP_INFLIGHT`, falling back to
/// [`dispatcher::default_max_concurrent_batches`] (4) so operators get one
/// knob that matches both the per-model-group semaphore and the IPC pool
/// by default.
///
/// Must be ≥ 1; values < 1 are clamped to 1 to avoid a never-dispatching
/// pull loop.
pub(crate) fn pull_loop_inflight() -> usize {
    std::env::var("SIE_PULL_LOOP_INFLIGHT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(crate::dispatcher::default_max_concurrent_batches)
        .max(1)
}

/// Max number of queue messages this sidecar may have admitted locally but
/// not yet finished through the scheduler/backend path.
///
/// Python workers have an explicit pending-queue capacity check before
/// accepting work. The Rust sidecar owns NATS pulling in queue mode, so this
/// cap is the equivalent local admission guard. Defaulting to one fetch keeps
/// enough work to form full batches while preventing a single worker from
/// hoarding the shared durable consumer when its local scheduler backs up.
pub(crate) fn pull_loop_max_outstanding(fetch_batch: usize, inflight_cap: usize) -> usize {
    let raw = std::env::var("SIE_PULL_LOOP_MAX_OUTSTANDING").ok();
    parse_pull_loop_max_outstanding(raw.as_deref(), fetch_batch, inflight_cap)
}

fn parse_pull_loop_max_outstanding(
    raw: Option<&str>,
    fetch_batch: usize,
    inflight_cap: usize,
) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| fetch_batch.max(inflight_cap).max(1))
}

/// How long we wait at shutdown for already-spawned pull-loop dispatch
/// tasks to finish before logging and moving on to backend drain. Kept
/// generous (one Python GPU forward pass + ack round-trip is a few
/// hundred ms) but still < `DRAIN_DEADLINE_MS` so the drain RPC has its
/// own headroom.
const PULL_LOOP_TASK_JOIN_DEADLINE_MS: u64 = 10_000;

/// Top-level entrypoint: wire everything together and run until shutdown.
#[allow(clippy::vec_init_then_push)]
pub async fn run(config: WorkerConfig) -> anyhow::Result<()> {
    let shutdown = Arc::new(Shutdown::new());
    shutdown::install_signal_handlers(shutdown.clone());

    // Shared sidecar readiness state for `/readyz` and NATS health
    // heartbeats. The first successful IPC Ping flips the sidecar ready;
    // shutdown flips it draining before backend drain starts.
    if config.ready_stale_mult == 0 {
        warn!(
            "SIE_WORKER_READYZ_STALE_MULT=0; falling back to readiness::DEFAULT_STALE_MULT ({})",
            crate::readiness::DEFAULT_STALE_MULT
        );
    }
    let readiness = Arc::new(Readiness::new(
        config.ping_interval_ms,
        config.ready_stale_mult,
    ));
    info!(
        ping_interval_ms = config.ping_interval_ms,
        ready_stale_mult = config.ready_stale_mult,
        freshness_ms = readiness.freshness_ms(),
        "readiness: /readyz freshness window configured"
    );

    // Construct the single telemetry facade and neutral runtime-pressure state
    // before wiring components that report semantic observations through it.
    let runtime_state = Arc::new(RuntimeState::new());
    runtime_state
        .worker_gpu_slots_total
        .set(config.gpu_count.max(1).into());
    runtime_state.worker_gpu_slots_ready.set(0);
    let probe_handle = runtime_state::spawn_probe_server(
        config.probe_port,
        Arc::clone(&readiness),
        shutdown.clone(),
    )
    .context("spawn probe server")?;

    // --- Adapter worker pool (connects lazily per child socket).
    let worker_pool = AdapterWorkerPool::new(
        &config.ipc_socket_paths,
        config.ipc_pool_size,
        config.ipc_request_timeout_s,
        config.model_ready_timeout_s,
        Arc::clone(&runtime_state),
    );
    info!(
        primary_socket = %config.ipc_socket_path.display(),
        socket_count = worker_pool.child_count(),
        ipc_pool_size = config.ipc_pool_size,
        ipc_request_timeout_s = config.ipc_request_timeout_s,
        model_ready_timeout_s = config.model_ready_timeout_s,
        "Adapter worker pool initialized"
    );

    // --- NATS
    let nats_url = config
        .nats_url
        .as_deref()
        .context("SIE_NATS_URL is required for the NATS ingest mode")?;
    let (nats_client, jetstream) = connect(nats_url).await.context("connect NATS")?;
    info!(nats = %nats_url, "NATS connected");
    let consumer = ensure_stream_and_consumer(&jetstream, &config)
        .await
        .context("ensure NATS stream/consumer")?;
    let nats_consumer = NatsConsumer::new(consumer);
    let nats_direct_active = Arc::new(AtomicBool::new(false));
    let nats_consumer_reconciler_handle = spawn_nats_consumer_reconciler(
        jetstream.clone(),
        config.clone(),
        shutdown.clone(),
        Arc::clone(&nats_direct_active),
    );

    // --- Payload store (wrapped so every fetch records metrics).
    let payload_store = create_payload_store(config.payload_store_url.as_deref())
        .await
        .context("create payload store")?;
    let payload_store =
        payload_store_with_telemetry(payload_store, runtime_state.telemetry.clone());

    // --- Publisher
    let publisher = Arc::new(WorkPublisher::new(
        nats_client.clone(),
        &config.worker_id,
        runtime_state.telemetry.clone(),
    ));

    // --- Adaptive fetch / latency tracker (shared across dispatcher + loop).
    // 200-sample window, 10-sample warm-up before p50 is reported.
    let latency_tracker = Arc::new(Mutex::new(LatencyTracker::new(200, 10)));
    let fetch_ctrl = FetchExpiryController::from_env_or_default();

    let pool_admission =
        PoolAdmissionGate::from_worker_config(&config).context("construct pool admission gate")?;
    let pool_admission_handle = pool_admission.as_ref().map(|gate| {
        let gate = Arc::clone(gate);
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            gate.run(shutdown).await;
        })
    });

    // Pinned-model reconciler: polls the pool's PoolSpec.pinned_models and pushes
    // it to the adapter worker over IPC. Decoupled from admission so pinned models
    // work even when the admission gate is disabled or fails open.
    let pinned_reconciler_handle =
        crate::pinned_reconciler::spawn(&config, Arc::clone(&worker_pool), Arc::clone(&shutdown))
            .context("spawn pinned reconciler")?;

    // --- Backend router
    //
    // The sidecar talks to the co-deployed adapter workers
    // through a pool that owns child placement and per-socket health.
    let adapter_backend: SharedBackend = worker_pool.clone();
    let backends: Vec<SharedBackend> = vec![Arc::clone(&adapter_backend)];

    let router = BackendRouter::from_backends(backends);
    info!(
        backends = ?router.names(),
        "backend router initialised"
    );
    let backend: SharedBackend = router;

    // --- Rust-side tokenizer registry. Always created; tokenisers are
    // ingested lazily on the `EnsureModelReady` handshake when the adapter
    // declares a `tokenizer_path` in its `ModelDescriptor`.
    //
    // An empty registry is zero-cost: every `get(model_id)` returns
    // `None` and the dispatcher falls through to the Python-tokenise
    // fallback path, exactly like the old `Option::None` branch did.
    let tokenizer_registry = TokenizerRegistry::empty();
    debug!(
        "rust-tokenize: registry empty at startup — tokenisers will be \
         ingested from EnsureModelReady descriptors as adapters declare them"
    );

    // Rust-side scheduler. Always on when the worker runs:
    // model-to-image routing is decided on the gateway from the
    // config-model API, so every model that lands here goes through
    // the Rust scheduler. Per-model drain loops spawn lazily on first
    // traffic inside `Dispatcher::resolve_scheduler`; the shutdown
    // path below awaits every handle collected along the way so
    // final-drain windows have a chance to complete.
    let scheduler_registry: Arc<crate::scheduler::ProductionSchedulerRegistry> =
        Arc::new(crate::scheduler::ProductionSchedulerRegistry::new(
            // BatchConfig pulls `SIE_BATCHER_*` overrides; the
            // controller (constructed lazily per model in
            // `Scheduler::builder().build()`) pulls
            // `SIE_ADAPTIVE_BATCH_*` overrides — see the doc comments
            // on `BatchConfig::from_env_or_default` and
            // `AdaptiveBatchController::from_env_or_default` for the
            // full list.
            crate::scheduler::BatchConfig::from_env_or_default(),
        ));
    info!("rust-scheduler: enabled for every model (per-model drain loops spawn on first traffic)");

    let config_apply_state = Arc::new(ConfigApplyState::new(config.bundle_config_hash.clone()));
    let loaded_models = config_apply_state.loaded_models();
    let batch_cancel_state = BatchCancelState::default();
    let request_cancel_state = RequestCancelState::new(work_cancel_tombstone_ttl());

    let dispatcher = Arc::new(Dispatcher::new(
        Arc::clone(&backend),
        Arc::clone(&worker_pool),
        payload_store,
        Some(Arc::clone(&publisher)),
        config.worker_id.clone(),
        Arc::clone(&runtime_state),
        Arc::clone(&latency_tracker),
        tokenizer_registry,
        Some(Arc::clone(&scheduler_registry)),
        Some(shutdown.clone()),
        Some(Arc::clone(&config_apply_state)),
        pool_admission.clone(),
        batch_cancel_state.clone(),
        request_cancel_state.clone(),
    ));

    let generation_direct_dispatch = Arc::new(GenerationDirectDispatch::new(
        jetstream.clone(),
        nats_client.clone(),
        config.clone(),
        Arc::clone(&dispatcher),
        shutdown.clone(),
        fetch_ctrl.clone(),
        Arc::clone(&latency_tracker),
        pool_admission.clone(),
        Arc::clone(&worker_pool),
        Arc::clone(&nats_direct_active),
        batch_cancel_state,
        request_cancel_state,
    ));
    generation_direct_dispatch
        .activate("startup")
        .await
        .context("activate worker direct-dispatch at startup")?;

    let config_subscriber_handle = crate::config_subscriber::spawn(
        nats_client.clone(),
        config.bundle.clone(),
        Arc::clone(&worker_pool),
        Arc::clone(&config_apply_state),
        runtime_state.telemetry.clone(),
        shutdown.clone(),
        crate::config_subscriber::ConfigSubscriberOptions {
            trusted_producers: config.nats_config_trusted_producers.clone(),
            pool: config.pool.clone(),
        },
    );
    let config_reconciler_handle = crate::config_reconciler::spawn(
        config.config_service_url.as_ref().map(|base_url| {
            crate::config_reconciler::ReconcilerConfig {
                base_url: base_url.clone(),
                admin_token: config.config_service_token.clone(),
                bundle: config.bundle.clone(),
                pool: config.pool.clone(),
                poll_interval: Duration::from_millis(config.config_poll_interval_ms),
                full_export_interval: if config.config_full_export_interval_ms == 0 {
                    None
                } else {
                    Some(Duration::from_millis(config.config_full_export_interval_ms))
                },
            }
        }),
        Arc::clone(&worker_pool),
        Arc::clone(&config_apply_state),
        runtime_state.telemetry.clone(),
        shutdown.clone(),
    );

    // --- Background: heartbeat pings into IPC so /readyz reflects liveness
    let heartbeat_handle = spawn_heartbeat(
        Arc::clone(&worker_pool),
        Arc::clone(&runtime_state),
        Duration::from_millis(config.ping_interval_ms),
        Arc::clone(&readiness),
        Arc::clone(&config_apply_state),
        Arc::clone(&loaded_models),
        shutdown.clone(),
    );

    // --- Background: NATS health-heartbeat publisher.
    //
    // In `health_mode=nats` the gateway's `WorkerRegistry` is built
    // exclusively from `sie.health.>` messages — no publisher means
    // no registered workers, which makes `resolve_queue_route`
    // return `None` and every inbound request `202 provisioning`.
    //
    // Skipped silently when `SIE_HEALTH_PUBLISH_ENABLED=false` (or
    // any value that parses to `false`) so operators on the legacy
    // WS-health gateway aren't forced to consume heartbeats they
    // ignore. Default is enabled because the sidecar pod has no
    // `/ws/status` endpoint (that endpoint lives on the Python
    // container) and therefore forces the gateway into `health_mode=nats`.
    let health_publisher_config = if health_publish_enabled() {
        Some(crate::health_publisher::HealthPublisherConfig {
            worker_id: config.worker_id.clone(),
            bundle: config.bundle.clone(),
            pool_name: config.pool.clone(),
            machine_profile: config.machine_profile.clone(),
            gpu_count: config.gpu_count,
            bundle_config_hash: config_apply_state.bundle_config_hash(),
            loaded_models: Arc::clone(&loaded_models),
            runtime_state: Arc::clone(&runtime_state),
            interval: Duration::from_millis(config.health_publish_interval_ms),
        })
    } else {
        info!("nats-health: SIE_HEALTH_PUBLISH_ENABLED=false; skipping heartbeat publisher");
        None
    };
    let health_publisher_handle = health_publisher_config.as_ref().map(|pub_cfg| {
        crate::health_publisher::spawn(
            nats_client.clone(),
            pub_cfg.clone(),
            Arc::clone(&readiness),
            shutdown.clone(),
        )
    });

    // --- Main pull loop
    run_pull_loop(
        &nats_consumer,
        dispatcher.clone(),
        shutdown.clone(),
        &fetch_ctrl,
        Arc::clone(&latency_tracker),
        pool_admission.clone(),
    )
    .await;
    generation_direct_dispatch.abort_and_join().await;

    // --- Graceful shutdown: the pull loop has exited and awaited every
    // dispatch task it spawned (bounded by `PULL_LOOP_TASK_JOIN_DEADLINE_MS`).
    // Anything left running after that deadline is best-effort and will
    // be observed by the backend drain path below. Now:
    //   1. Stop the background heartbeat task BEFORE drain so its IPC ping
    //      doesn't race the Python drain RPC.
    //   2. Drain every registered backend (Python gets a Drain RPC;
    //      native backends wait for in-flight work + free resources).
    //   3. Flush any buffered NATS publishes so WorkResults don't get lost.
    //   4. Stop the metrics task.
    info!("pull loop exited; stopping heartbeat before drain");
    readiness.mark_draining();
    heartbeat_handle.abort();
    let _ = heartbeat_handle.await; // best-effort join

    // Let the periodic NATS health publisher observe shutdown and emit
    // its tombstone before we move on. Aborting it immediately here can
    // cut off that publish during Kubernetes termination, leaving the
    // gateway with a stale unhealthy registry row until the fallback TTL.
    if let Some(h) = health_publisher_handle {
        let mut h = h;
        tokio::select! {
            res = &mut h => {
                if let Err(e) = res {
                    warn!(error = %e, "nats-health: heartbeat publisher task failed during shutdown");
                }
            }
            _ = sleep(Duration::from_secs(2)) => {
                warn!("nats-health: heartbeat publisher did not stop before tombstone deadline");
                h.abort();
                let _ = h.await;
            }
        }
    }
    // Idempotent fallback: the publisher emits the first tombstone as
    // soon as shutdown fires; this second publish covers the case where
    // that task had already exited or hit a transient publish error.
    if let Some(pub_cfg) = &health_publisher_config {
        if let Err(e) = crate::health_publisher::publish_tombstone(&nats_client, pub_cfg).await {
            warn!(error = %e, "nats-health: shutdown tombstone publish failed");
        }
    }
    if let Some(h) = nats_consumer_reconciler_handle {
        h.abort();
        let _ = h.await;
    }
    if let Some(h) = pool_admission_handle {
        h.abort();
        let _ = h.await;
    }
    if let Some(h) = pinned_reconciler_handle {
        h.abort();
        let _ = h.await;
    }
    config_subscriber_handle.abort();
    let _ = config_subscriber_handle.await;
    if let Some(h) = config_reconciler_handle {
        h.abort();
        let _ = h.await;
    }

    // Wait for scheduler drain loops to finish their own final drain
    // window (bounded by SIE_SCHEDULER_DRAIN_DEADLINE_MS inside each
    // loop; default 10 s). They exit on `shutdown.wait()` which has already fired
    // by this point (the pull loop returned). Items still pending
    // after the internal deadline redeliver via JetStream ack_wait.
    //
    // Drain handles accumulate lazily as models see their first
    // traffic; on a worker that never received any traffic this
    // list is empty.
    let scheduler_drain_handles = dispatcher.take_scheduler_drain_handles().await;
    if !scheduler_drain_handles.is_empty() {
        let scheduler_drain_start = Instant::now();
        let active_count = scheduler_registry.active_count().await;
        info!(
            drain_loop_count = scheduler_drain_handles.len(),
            active_models = active_count,
            "rust-scheduler: awaiting per-model drain loops"
        );
        for h in scheduler_drain_handles {
            if let Err(e) = h.await {
                warn!(error = %e, "scheduler drain task join error");
            }
        }
        info!(
            elapsed_ms = scheduler_drain_start.elapsed().as_millis() as u64,
            "rust-scheduler: drain loops exited"
        );
    }

    let generation_handles = dispatcher.take_generation_handles().await;
    if !generation_handles.is_empty() {
        let generation_drain_start = Instant::now();
        info!(
            generation_task_count = generation_handles.len(),
            "generation: awaiting in-flight streams before backend drain"
        );
        let joined = timeout(Duration::from_millis(DRAIN_DEADLINE_MS), async {
            for h in generation_handles {
                let _ = h.await;
            }
        })
        .await;
        if joined.is_err() {
            warn!(
                deadline_ms = DRAIN_DEADLINE_MS,
                "generation streams did not settle before drain deadline"
            );
        }
        info!(
            elapsed_ms = generation_drain_start.elapsed().as_millis() as u64,
            "generation: in-flight stream wait complete"
        );
    }

    info!("draining backends");
    let drain_start = std::time::Instant::now();
    backend.drain(DRAIN_DEADLINE_MS).await;
    let drain_elapsed = drain_start.elapsed();
    let drain_result = if drain_elapsed.as_millis() as u64 > DRAIN_DEADLINE_MS {
        "deadline_exceeded"
    } else {
        "success"
    };
    runtime_state
        .telemetry
        .shutdown_drain_completed(drain_result, drain_elapsed);
    info!(
        drain_seconds = drain_elapsed.as_secs_f64(),
        drain_result, "backends drained"
    );

    // Flush NATS publishes. async-nats buffers writes in a background
    // task; without a flush we can lose results in flight when the
    // client drops.
    if let Err(e) = nats_client.flush().await {
        warn!(error = %e, "NATS flush on shutdown failed");
    } else {
        debug!("NATS client flushed");
    }

    probe_handle.abort();
    info!("sie-server-sidecar shutdown complete");
    Ok(())
}

/// Local-ingest entrypoint: the broker-less sidecar lane. Selected with
/// `SIE_SIDECAR_INGEST=local`.
///
/// Wires the exact dispatcher/prep/scheduler/backend pipeline [`run`]
/// wires, but the work feed is the [`crate::local_ingest`] UDS server
/// instead of the NATS pull loop, and every NATS-coupled subsystem is
/// skipped by construction:
///
/// * no NATS connection / streams / consumers / reconciler — the ingest
///   socket is the only work source;
/// * no [`WorkPublisher`] — results ride each local delivery's channel
///   back to the ingest caller (`Dispatcher.publisher = None`);
/// * no NATS health publisher — the platform autoscaler owns worker health;
///   the co-located shim watches this process + `/readyz`;
/// * no NATS config subscriber — config freshness is the HTTP epoch-poll
///   reconciler alone (`SIE_CONFIG_SERVICE_URL`, optional), which is
///   transport-independent;
/// * no gateway pool-admission gate / pinned reconciler — the lane has a
///   fixed pool identity enforced by the ingest-side re-check, and model
///   pinning is done by the lane's engine at snapshot time.
///
/// The IPC heartbeat stays: it drives `/readyz` freshness and the
/// bundle-config hash exactly as on Kubernetes.
pub async fn run_local(config: WorkerConfig) -> anyhow::Result<()> {
    let shutdown = Arc::new(Shutdown::new());
    shutdown::install_signal_handlers(shutdown.clone());

    let local_socket = config
        .local_socket_path
        .clone()
        .context("SIE_SIDECAR_LOCAL_SOCKET is required for the local ingest mode")?;

    if config.ready_stale_mult == 0 {
        warn!(
            "SIE_WORKER_READYZ_STALE_MULT=0; falling back to readiness::DEFAULT_STALE_MULT ({})",
            crate::readiness::DEFAULT_STALE_MULT
        );
    }
    let readiness = Arc::new(Readiness::new(
        config.ping_interval_ms,
        config.ready_stale_mult,
    ));

    let runtime_state = Arc::new(RuntimeState::new());
    runtime_state
        .worker_gpu_slots_total
        .set(config.gpu_count.max(1).into());
    runtime_state.worker_gpu_slots_ready.set(0);
    let probe_handle = runtime_state::spawn_probe_server(
        config.probe_port,
        Arc::clone(&readiness),
        shutdown.clone(),
    )
    .context("spawn probe server")?;

    let worker_pool = AdapterWorkerPool::new(
        &config.ipc_socket_paths,
        config.ipc_pool_size,
        config.ipc_request_timeout_s,
        config.model_ready_timeout_s,
        Arc::clone(&runtime_state),
    );
    info!(
        primary_socket = %config.ipc_socket_path.display(),
        socket_count = worker_pool.child_count(),
        ipc_pool_size = config.ipc_pool_size,
        ipc_request_timeout_s = config.ipc_request_timeout_s,
        model_ready_timeout_s = config.model_ready_timeout_s,
        "local-ingest: adapter worker pool initialized"
    );

    let payload_store = create_payload_store(config.payload_store_url.as_deref())
        .await
        .context("create payload store")?;
    let payload_store =
        payload_store_with_telemetry(payload_store, runtime_state.telemetry.clone());

    let adapter_backend: SharedBackend = worker_pool.clone();
    let backend: SharedBackend = BackendRouter::from_backends(vec![Arc::clone(&adapter_backend)]);

    // Tokenizer registry fills from `EnsureModelReady` descriptors exactly
    // like the NATS path — the engine (which carries the image-baked model
    // configs) is the config source; no sie-config dependency needed for
    // Rust-side tokenisation.
    let tokenizer_registry = TokenizerRegistry::empty();

    let scheduler_registry: Arc<crate::scheduler::ProductionSchedulerRegistry> =
        Arc::new(crate::scheduler::ProductionSchedulerRegistry::new(
            crate::scheduler::BatchConfig::from_env_or_default(),
        ));
    let config_apply_state = Arc::new(ConfigApplyState::new(config.bundle_config_hash.clone()));
    let loaded_models = config_apply_state.loaded_models();
    let latency_tracker = Arc::new(Mutex::new(LatencyTracker::new(200, 10)));

    let dispatcher = Arc::new(Dispatcher::new(
        Arc::clone(&backend),
        Arc::clone(&worker_pool),
        payload_store,
        None, // no NATS publisher: results ride the local delivery channels
        config.worker_id.clone(),
        Arc::clone(&runtime_state),
        Arc::clone(&latency_tracker),
        tokenizer_registry,
        Some(Arc::clone(&scheduler_registry)),
        Some(shutdown.clone()),
        Some(Arc::clone(&config_apply_state)),
        None, // admission is the ingest-side re-check (fixed pool identity)
        // Batch-cancel rides the NATS cancel subject; local ingest has no
        // such feed, so a default (empty) state means "nothing cancelled".
        BatchCancelState::default(),
        // Request-wide cancellation is also a NATS signal; local ingest has
        // no subscription and therefore starts with an empty tombstone set.
        RequestCancelState::new(work_cancel_tombstone_ttl()),
    ));

    let heartbeat_handle = spawn_heartbeat(
        Arc::clone(&worker_pool),
        Arc::clone(&runtime_state),
        Duration::from_millis(config.ping_interval_ms),
        Arc::clone(&readiness),
        Arc::clone(&config_apply_state),
        Arc::clone(&loaded_models),
        shutdown.clone(),
    );

    let config_reconciler_handle = crate::config_reconciler::spawn(
        config.config_service_url.as_ref().map(|base_url| {
            crate::config_reconciler::ReconcilerConfig {
                base_url: base_url.clone(),
                admin_token: config.config_service_token.clone(),
                bundle: config.bundle.clone(),
                pool: config.pool.clone(),
                poll_interval: Duration::from_millis(config.config_poll_interval_ms),
                full_export_interval: if config.config_full_export_interval_ms == 0 {
                    None
                } else {
                    Some(Duration::from_millis(config.config_full_export_interval_ms))
                },
            }
        }),
        Arc::clone(&worker_pool),
        Arc::clone(&config_apply_state),
        runtime_state.telemetry.clone(),
        shutdown.clone(),
    );

    // --- Main ingest server: serves until shutdown. A bind failure is
    // fatal — main exits non-zero and the co-located shim replaces the
    // container (fail-fast supervision).
    let ingest_result = crate::local_ingest::run_local_ingest(
        &local_socket,
        Arc::clone(&dispatcher),
        &config.pool,
        &config.worker_id,
        Arc::clone(&shutdown),
    )
    .await;

    info!("local ingest exited; stopping heartbeat before drain");
    readiness.mark_draining();
    heartbeat_handle.abort();
    let _ = heartbeat_handle.await;
    if let Some(h) = config_reconciler_handle {
        h.abort();
        let _ = h.await;
    }

    // Same scheduler final-drain contract as `run()` — residual items
    // surface to the ingest callers as shutdown "cancelled" errors instead
    // of JetStream redelivery.
    let scheduler_drain_handles = dispatcher.take_scheduler_drain_handles().await;
    if !scheduler_drain_handles.is_empty() {
        let scheduler_drain_start = Instant::now();
        info!(
            drain_loop_count = scheduler_drain_handles.len(),
            "rust-scheduler: awaiting per-model drain loops"
        );
        for h in scheduler_drain_handles {
            if let Err(e) = h.await {
                warn!(error = %e, "scheduler drain task join error");
            }
        }
        info!(
            elapsed_ms = scheduler_drain_start.elapsed().as_millis() as u64,
            "rust-scheduler: drain loops exited"
        );
    }

    info!("draining backends");
    let drain_start = Instant::now();
    backend.drain(DRAIN_DEADLINE_MS).await;
    let drain_elapsed = drain_start.elapsed();
    let drain_result = if drain_elapsed.as_millis() as u64 > DRAIN_DEADLINE_MS {
        "deadline_exceeded"
    } else {
        "success"
    };
    runtime_state
        .telemetry
        .shutdown_drain_completed(drain_result, drain_elapsed);
    info!(
        drain_seconds = drain_elapsed.as_secs_f64(),
        drain_result, "backends drained"
    );
    probe_handle.abort();
    info!("sie-server-sidecar (local ingest) shutdown complete");
    ingest_result
}

#[derive(Default)]
struct GenerationDirectHandles {
    pull: Option<JoinHandle<()>>,
    generation_cancel: Option<JoinHandle<()>>,
    batch_cancel: Option<JoinHandle<()>>,
    work_cancel: Option<JoinHandle<()>>,
}

/// Owns the worker-specific direct-dispatch stream and cancel subscriptions.
///
/// The stream is always active: capped logical batch requests can target an
/// assigned worker directly, and generation uses the same worker subject for
/// cache-aware direct dispatch.
struct GenerationDirectDispatch {
    jetstream: async_nats::jetstream::Context,
    nats_client: async_nats::Client,
    config: WorkerConfig,
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<Shutdown>,
    fetch_ctrl: FetchExpiryController,
    latency_tracker: Arc<Mutex<LatencyTracker>>,
    pool_admission: Option<Arc<PoolAdmissionGate>>,
    worker_pool: Arc<AdapterWorkerPool>,
    active: Arc<AtomicBool>,
    batch_cancel_state: BatchCancelState,
    request_cancel_state: RequestCancelState,
    handles: Mutex<GenerationDirectHandles>,
}

impl GenerationDirectDispatch {
    #[allow(clippy::too_many_arguments)]
    fn new(
        jetstream: async_nats::jetstream::Context,
        nats_client: async_nats::Client,
        config: WorkerConfig,
        dispatcher: Arc<Dispatcher>,
        shutdown: Arc<Shutdown>,
        fetch_ctrl: FetchExpiryController,
        latency_tracker: Arc<Mutex<LatencyTracker>>,
        pool_admission: Option<Arc<PoolAdmissionGate>>,
        worker_pool: Arc<AdapterWorkerPool>,
        active: Arc<AtomicBool>,
        batch_cancel_state: BatchCancelState,
        request_cancel_state: RequestCancelState,
    ) -> Self {
        Self {
            jetstream,
            nats_client,
            config,
            dispatcher,
            shutdown,
            fetch_ctrl,
            latency_tracker,
            pool_admission,
            worker_pool,
            active,
            batch_cancel_state,
            request_cancel_state,
            handles: Mutex::new(GenerationDirectHandles::default()),
        }
    }

    async fn activate(&self, reason: &str) -> anyhow::Result<bool> {
        if self
            .active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(false);
        }

        let worker_consumer = match ensure_worker_stream_and_consumer(&self.jetstream, &self.config)
            .await
            .context("ensure worker direct-dispatch stream/consumer")
        {
            Ok(consumer) => consumer,
            Err(e) => {
                self.active.store(false, Ordering::Release);
                return Err(e);
            }
        };

        let work_cancel = match spawn_work_cancel_subscriber(
            self.nats_client.clone(),
            self.request_cancel_state.clone(),
            Arc::clone(&self.shutdown),
        )
        .await
        .context("start request-wide work cancel subscriber")
        {
            Ok(handle) => handle,
            Err(e) => {
                self.active.store(false, Ordering::Release);
                return Err(e);
            }
        };

        let batch_cancel = match spawn_batch_direct_cancel_subscriber(
            self.nats_client.clone(),
            self.config.worker_id.clone(),
            self.batch_cancel_state.clone(),
            Arc::clone(&self.shutdown),
        )
        .await
        .context("start batch direct-dispatch cancel subscriber")
        {
            Ok(handle) => handle,
            Err(e) => {
                work_cancel.abort();
                self.active.store(false, Ordering::Release);
                return Err(e);
            }
        };

        let nats_worker_consumer = NatsConsumer::new(worker_consumer);
        let direct_fetch_ctrl = self.fetch_ctrl.clone();
        let direct_dispatcher = Arc::clone(&self.dispatcher);
        let direct_shutdown = Arc::clone(&self.shutdown);
        let direct_latency_tracker = Arc::clone(&self.latency_tracker);
        let direct_pool_admission = self.pool_admission.clone();
        let pull = tokio::spawn(async move {
            run_pull_loop(
                &nats_worker_consumer,
                direct_dispatcher,
                direct_shutdown,
                &direct_fetch_ctrl,
                direct_latency_tracker,
                direct_pool_admission,
            )
            .await;
        });
        let generation_cancel = spawn_generation_cancel_subscriber(
            self.nats_client.clone(),
            Arc::clone(&self.worker_pool),
            Arc::clone(&self.shutdown),
        );

        let mut handles = self.handles.lock().await;
        handles.pull = Some(pull);
        handles.generation_cancel = Some(generation_cancel);
        handles.batch_cancel = Some(batch_cancel);
        handles.work_cancel = Some(work_cancel);
        info!(reason, "worker direct-dispatch activated");
        Ok(true)
    }

    async fn abort_and_join(&self) {
        let handles = {
            let mut guard = self.handles.lock().await;
            GenerationDirectHandles {
                pull: guard.pull.take(),
                generation_cancel: guard.generation_cancel.take(),
                batch_cancel: guard.batch_cancel.take(),
                work_cancel: guard.work_cancel.take(),
            }
        };

        if let Some(h) = handles.pull {
            if !h.is_finished() {
                h.abort();
            }
            let _ = h.await;
        }
        if let Some(h) = handles.generation_cancel {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = handles.batch_cancel {
            h.abort();
            let _ = h.await;
        }
        if let Some(h) = handles.work_cancel {
            h.abort();
            let _ = h.await;
        }
    }
}

fn spawn_nats_consumer_reconciler(
    jetstream: async_nats::jetstream::Context,
    config: WorkerConfig,
    shutdown: Arc<Shutdown>,
    worker_direct_active: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let every = nats_consumer_reconcile_interval()?;
    Some(tokio::spawn(async move {
        info!(
            interval_ms = every.as_millis() as u64,
            minimum_interval_ms = MIN_NATS_CONSUMER_RECONCILE_INTERVAL_MS,
            "nats-consumer: stream/durable reconciler enabled"
        );
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                _ = sleep(every) => {}
            }
            match reconcile_stream_and_consumer(&jetstream, &config).await {
                Ok(()) => {}
                Err(e) => warn!(
                    error = %e,
                    "nats-consumer: stream/durable reconcile failed"
                ),
            }
            if worker_direct_active.load(Ordering::Acquire) {
                match reconcile_worker_stream_and_consumer(&jetstream, &config).await {
                    Ok(()) => {}
                    Err(e) => warn!(
                        error = %e,
                        "nats-consumer: worker direct-dispatch reconcile failed"
                    ),
                }
            }
        }
    }))
}

async fn spawn_work_cancel_subscriber(
    nats_client: async_nats::Client,
    request_cancel_state: RequestCancelState,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<JoinHandle<()>> {
    let subject = "work_cancel.>".to_string();
    let mut sub = nats_client
        .subscribe(subject.clone())
        .await
        .context("subscribe to work_cancel.>")?;

    Ok(tokio::spawn(async move {
        info!("work-cancel: request-wide subscription started");
        let retry_backoff = Duration::from_millis(500);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                maybe_msg = sub.next() => {
                    let Some(msg) = maybe_msg else {
                        warn!("work-cancel: subscription ended; retrying");
                        loop {
                            match nats_client.subscribe(subject.clone()).await {
                                Ok(next_sub) => {
                                    sub = next_sub;
                                    info!("work-cancel: subscription restarted");
                                    break;
                                }
                                Err(e) => warn!(
                                    error = %e,
                                    "work-cancel: failed to resubscribe to work_cancel.>"
                                ),
                            }
                            tokio::select! {
                                biased;
                                _ = shutdown.wait() => return,
                                _ = sleep(retry_backoff) => {}
                            }
                        }
                        continue;
                    };
                    let subject = msg.subject.to_string();
                    let Some((router_id, request_id)) = request_id_from_work_cancel_subject(&subject) else {
                        debug!(subject = %subject, "work-cancel: ignoring malformed subject");
                        continue;
                    };
                    request_cancel_state.cancel(router_id.clone(), request_id.clone());
                    debug!(
                        router_id = %router_id,
                        request_id = %request_id,
                        "work-cancel: recorded request-wide cancellation tombstone"
                    );
                }
            }
        }
    }))
}

fn spawn_generation_cancel_subscriber(
    nats_client: async_nats::Client,
    worker_pool: Arc<AdapterWorkerPool>,
    shutdown: Arc<Shutdown>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut sub = match nats_client.subscribe("cancel.>".to_string()).await {
            Ok(sub) => sub,
            Err(e) => {
                warn!(error = %e, "generation: failed to subscribe to cancel.>");
                return;
            }
        };
        info!("generation: cancel subscription started");
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                maybe_msg = sub.next() => {
                    let Some(msg) = maybe_msg else {
                        warn!("generation: cancel subscription ended");
                        return;
                    };
                    let subject = msg.subject.to_string();
                    let Some(request_id) = request_id_from_cancel_subject(&subject) else {
                        debug!(subject = %subject, "generation: ignoring malformed cancel subject");
                        continue;
                    };
                    match worker_pool.signal_generate_cancel(request_id.clone()).await {
                        Ok(resp) => {
                            debug!(
                                request_id = %request_id,
                                matched = resp.matched,
                                "generation: cancel signal forwarded to Python"
                            );
                        }
                        Err(e) => warn!(
                            request_id = %request_id,
                            error = %e,
                            "generation: failed to forward cancel signal to Python"
                        ),
                    }
                }
            }
        }
    })
}

async fn spawn_batch_direct_cancel_subscriber(
    nats_client: async_nats::Client,
    worker_id: String,
    batch_cancel_state: BatchCancelState,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<JoinHandle<()>> {
    let subject = "batch_cancel.>".to_string();
    let mut sub = nats_client
        .subscribe(subject.clone())
        .await
        .context("subscribe to batch_cancel.>")?;

    Ok(tokio::spawn(async move {
        info!("batch-direct: cancel subscription started");
        let retry_backoff = Duration::from_millis(500);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                maybe_msg = sub.next() => {
                    let Some(msg) = maybe_msg else {
                        warn!("batch-direct: cancel subscription ended; retrying");
                        loop {
                            match nats_client.subscribe(subject.clone()).await {
                                Ok(next_sub) => {
                                    sub = next_sub;
                                    info!("batch-direct: cancel subscription restarted");
                                    break;
                                }
                                Err(e) => warn!(
                                    error = %e,
                                    "batch-direct: failed to resubscribe to batch_cancel.>"
                                ),
                            }
                            tokio::select! {
                                biased;
                                _ = shutdown.wait() => return,
                                _ = sleep(retry_backoff) => {}
                            }
                        }
                        continue;
                    };
                    let subject = msg.subject.to_string();
                    let Some(request_id) =
                        request_id_from_batch_cancel_subject(&subject, &worker_id)
                    else {
                        debug!(subject = %subject, "batch-direct: ignoring non-local cancel subject");
                        continue;
                    };
                    batch_cancel_state.cancel(request_id.clone());
                    debug!(
                        request_id = %request_id,
                        worker_id = %worker_id,
                        "batch-direct: cancel recorded for worker-direct queued work"
                    );
                }
            }
        }
    }))
}

fn request_id_from_cancel_subject(subject: &str) -> Option<String> {
    let mut parts = subject.splitn(3, '.');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("cancel"), Some(_router_id), Some(request_id)) if !request_id.is_empty() => {
            Some(request_id.to_string())
        }
        _ => None,
    }
}

/// Ping every adapter IPC server on a ticker. A failure is logged but
/// non-fatal — the consumer loop will surface real problems via
/// EnsureModelReady / Process* errors.
///
/// Each successful ready ping refreshes the [`Readiness`] heartbeat timestamp.
fn spawn_heartbeat(
    worker_pool: Arc<AdapterWorkerPool>,
    runtime_state: Arc<RuntimeState>,
    every: Duration,
    readiness: Arc<Readiness>,
    config_apply_state: Arc<ConfigApplyState>,
    loaded_models: crate::health_publisher::SharedLoadedModels,
    shutdown: Arc<Shutdown>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut tick = interval(every);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Track consecutive failures so we warn on transitions
        // (0 -> 1 "heartbeat broke", N -> 0 "heartbeat recovered")
        // without spamming once per tick while the backend is down.
        let mut consecutive_failures: u64 = 0;
        loop {
            let wait = shutdown.wait();
            tokio::select! {
                biased;
                _ = wait => return,
                _ = tick.tick() => {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64() * 1000.0;
                    let ping_results = worker_pool.ping_all(ts).await;
                    let ready_children = worker_pool.ready_child_count();
                    crate::health_publisher::record_runtime_telemetry(&runtime_state);
                    let successful_ready: Vec<_> = ping_results
                        .iter()
                        .filter_map(|(_index, result)| result.as_ref().ok())
                        .filter(|resp| resp.ready)
                        .collect();
                    if let Some(resp) = (ready_children > 0).then(|| successful_ready.first()).flatten() {
                        if config_apply_state.current_bundle_config_hash().is_empty()
                            && !resp.bundle_config_hash.is_empty()
                        {
                            config_apply_state.set_bundle_hash(resp.bundle_config_hash.clone());
                        }
                        let mut merged_loaded_models = Vec::new();
                        for resp in &successful_ready {
                            merged_loaded_models.extend(resp.loaded_models.iter().cloned());
                        }
                        merged_loaded_models.sort();
                        merged_loaded_models.dedup();
                        match loaded_models.write() {
                            Ok(mut guard) => {
                                *guard = merged_loaded_models;
                            }
                            Err(_) => {
                                warn!("loaded_models lock poisoned; NATS health will keep the previous model list");
                            }
                        }
                        readiness.record_ping_success();
                        if consecutive_failures > 0 {
                            info!(
                                consecutive_failures,
                                ready_children,
                                total_children = worker_pool.child_count(),
                                "IPC heartbeat recovered"
                            );
                        }
                        consecutive_failures = 0;
                    } else {
                        let error = ping_results
                            .iter()
                            .find_map(|(_index, result)| result.as_ref().err())
                            .map(|e| ErrChain(e).to_string())
                            .unwrap_or_else(|| "all adapter children reported not ready".to_string());
                        consecutive_failures = consecutive_failures.saturating_add(1);
                        // First failure and "big milestones" warn;
                        // in between, stay quiet (the
                        // canonical IPC completion telemetry still records
                        // everything for the configured collector).
                        if consecutive_failures == 1
                            || consecutive_failures == 5
                            || consecutive_failures == 30
                            || consecutive_failures.is_multiple_of(120)
                        {
                            warn!(
                                consecutive_failures,
                                ready_children,
                                total_children = worker_pool.child_count(),
                                error = %error,
                                "IPC heartbeat failed"
                            );
                        } else {
                            debug!(
                                consecutive_failures,
                                ready_children,
                                total_children = worker_pool.child_count(),
                                error = %error,
                                "IPC heartbeat still failing"
                            );
                        }
                    }
                }
            }
        }
    })
}

/// Compute the client-side batch-coalesce quantum from the adaptive
/// controller, given current latency observations. Bounded by
/// `fetch_ctrl.{min,max}`. Under cold start (fewer than `min_samples`
/// observations) returns `fetch_ctrl.min`.
fn batch_quantum(ctrl: &FetchExpiryController, tracker: &LatencyTracker) -> Duration {
    // Reuse the existing adaptive controller output so the operator knobs
    // (`SIE_ADAPTIVE_{MIN,MAX}_QUANTUM_MS`, `SIE_ADAPTIVE_TARGET_P50_MS`)
    // still apply.
    ctrl.adjust(ctrl.min, tracker)
}

/// Open a long-lived pull stream, retrying on transient build failures
/// until either we succeed or shutdown fires. Returns `None` on
/// shutdown.
async fn build_pull_stream(
    consumer: &NatsConsumer,
    shutdown: &Shutdown,
    telemetry: &crate::observability::metrics::SidecarTelemetry,
    fetch_batch: usize,
    expires: Duration,
    backoff: Duration,
) -> Option<async_nats::jetstream::consumer::pull::Stream> {
    loop {
        match consumer.messages(fetch_batch, expires).await {
            Ok(s) => {
                telemetry.nats_operation("fetch", "success", "none", 1);
                return Some(s);
            }
            Err(e) => {
                telemetry.nats_operation("fetch", "error", "transport", 1);
                warn!(error = %e, "failed to open pull stream; retrying");
                tokio::select! {
                    biased;
                    _ = shutdown.wait() => {
                        info!("shutdown while (re)building pull stream");
                        return None;
                    }
                    _ = sleep(backoff) => continue,
                }
            }
        }
    }
}

async fn acquire_pull_admission_permit(
    permits: &Arc<Semaphore>,
    shutdown: &Arc<Shutdown>,
) -> Option<OwnedSemaphorePermit> {
    tokio::select! {
        biased;
        _ = shutdown.wait() => None,
        permit = permits.clone().acquire_owned() => match permit {
            Ok(permit) => Some(permit),
            Err(_) => {
                warn!("pull-loop admission semaphore closed — exiting loop");
                None
            }
        },
    }
}

/// Pull messages in a loop and dispatch each batch through the Dispatcher.
///
/// # Why one long-lived [`async_nats::jetstream::consumer::pull::Stream`]
///
/// async-nats' [`pull::Stream`] issues background pull requests to the
/// JetStream server and auto-refills credits in the background. When
/// the stream is dropped, any messages the server has already routed
/// but which the client hasn't consumed become `ack_pending` from the
/// server's point of view — they sit stuck until `ack_wait` (30 s)
/// elapses and the server redelivers them. Under concurrent load the
/// original "create a stream, pull for N ms, drop it" pattern therefore
/// collapsed to ~1 message per 30 s throughput with the rest 504'ing at
/// the gateway.
///
/// So: keep one stream open while the worker is admitted to its pool, poll it
/// continuously, and only rebuild on terminal (`None`) stream death or after a
/// pool-admission pause. The stream expire inside `pull_stream_expires()`
/// governs per-pull-request server-side lifetime (default 5 s -> a no-data
/// `408` every 5 s when idle) and is invisible to callers — `poll_next` handles
/// the refill.
///
/// # Batching
///
/// Each loop iteration:
/// 1. Block on `stream.next()` for the first message of a batch.
/// 2. Compute a short client-side **quantum** from the adaptive
///    controller (`batch_quantum`), bounded to
///    `SIE_ADAPTIVE_{MIN,MAX}_QUANTUM_MS`.
/// 3. Greedily drain any already-arrived messages until `fetch_batch` or
///    the quantum deadline, whichever first.
/// 4. Hand the batch to the dispatcher **as a spawned task**, bounded by
///    [`pull_loop_inflight`] so the next fetch can start immediately
///    while the backend processes the previous batch. Without this, a single
///    pull loop serialises `fetch → IPC → GPU → NATS ACK → next fetch`,
///    leaving the IPC pool and GPU under-fed under distributed load.
///
/// The quantum adds at most ~20 ms of latency to the first message of a
/// burst. A single concurrent request always completes end-to-end in
/// `quantum + IPC` ≈ tens of ms, independent of ack_wait.
async fn run_pull_loop(
    consumer: &NatsConsumer,
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<Shutdown>,
    fetch_ctrl: &FetchExpiryController,
    latency_tracker: Arc<Mutex<LatencyTracker>>,
    pool_admission: Option<Arc<PoolAdmissionGate>>,
) {
    let fetch_batch = fetch_batch_size();
    let pull_expires = pull_stream_expires();
    let rebuild_backoff = Duration::from_millis(500);
    let inflight_cap = pull_loop_inflight();
    let dispatch_permits = Arc::new(Semaphore::new(inflight_cap));
    let outstanding_cap = pull_loop_max_outstanding(fetch_batch, inflight_cap);
    let admission_permits = Arc::new(Semaphore::new(outstanding_cap));
    let mut inflight_tasks: Vec<JoinHandle<()>> = Vec::new();

    info!(
        fetch_batch,
        pull_expires_s = pull_expires.as_secs(),
        quantum_min_ms = fetch_ctrl.min.as_millis() as u64,
        quantum_max_ms = fetch_ctrl.max.as_millis() as u64,
        inflight_cap,
        outstanding_cap,
        "pull loop starting",
    );

    let mut stream: Option<async_nats::jetstream::consumer::pull::Stream> = None;

    'outer: loop {
        if let Some(gate) = pool_admission.as_ref() {
            if !gate.pull_admitted() {
                // Drop the long-lived pull stream while not admitted. Keeping
                // it open would leave server-side pull credits active and let
                // JetStream deliver messages into this client even though this
                // worker is outside the physical queue or every assigned
                // logical pool backed by that queue.
                stream = None;
                tokio::select! {
                    biased;
                    _ = shutdown.wait() => break 'outer,
                    _ = sleep(gate.pause_duration()) => continue 'outer,
                }
            }
        }

        if stream.is_none() {
            stream = match build_pull_stream(
                consumer,
                &shutdown,
                &dispatcher.runtime_state.telemetry,
                fetch_batch,
                pull_expires,
                rebuild_backoff,
            )
            .await
            {
                Some(s) => Some(s),
                None => break 'outer,
            };
        }

        // Step 1: acquire local admission capacity, then block for the first
        // message of the next batch. The permit travels with the message until
        // the scheduler/backend path completes, which prevents one sidecar from
        // draining the shared durable into an unbounded local scheduler backlog.
        let Some(first_permit) = acquire_pull_admission_permit(&admission_permits, &shutdown).await
        else {
            break 'outer;
        };
        let first = tokio::select! {
            biased;
            _ = shutdown.wait() => break 'outer,
            next = stream.as_mut().expect("pull stream is initialized").next() => next,
        };
        let first_msg = match first {
            Some(Ok(msg)) => QueuedMessage::new(msg, Some(first_permit)),
            Some(Err(e)) => {
                // Recoverable per-poll errors (server TIMEOUT 408, missed
                // heartbeat, etc.) surface here. The underlying Stream is
                // still usable — async-nats will have re-armed another pull
                // internally. Log, count, and keep going.
                debug!(error = %e, "pull stream poll error; continuing");
                dispatcher
                    .runtime_state
                    .telemetry
                    .nats_operation("fetch", "error", "transport", 1);
                drop(first_permit);
                continue 'outer;
            }
            None => {
                // Terminal end of stream (internal async-nats task died,
                // connection gone, etc.). Rebuild. Any messages that had
                // been routed to the old stream but not yet consumed are
                // recoverable via ack_wait → redelivery; we can't do
                // better than that without a fresh stream.
                warn!("pull stream returned None (terminal); rebuilding");
                dispatcher.runtime_state.telemetry.nats_operation(
                    "stream",
                    "error",
                    "stream_ended",
                    1,
                );
                stream = None;
                drop(first_permit);
                continue 'outer;
            }
        };

        // Step 2: coalesce. Compute the quantum once, relative to now.
        let started = Instant::now();
        let mut batch = Vec::with_capacity(fetch_batch);
        batch.push(first_msg);

        let quantum = {
            let tracker = latency_tracker.lock().await;
            batch_quantum(fetch_ctrl, &tracker)
        };
        let deadline = started + quantum;

        while batch.len() < fetch_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Ok(next_permit) = admission_permits.clone().try_acquire_owned() else {
                break;
            };
            tokio::select! {
                biased;
                _ = shutdown.wait() => break,
                _ = sleep(remaining) => break,
                next = stream.as_mut().expect("pull stream is initialized").next() => {
                    match next {
                        Some(Ok(msg)) => batch.push(QueuedMessage::new(msg, Some(next_permit))),
                        Some(Err(e)) => {
                            // Mid-batch recoverable error — yield what we
                            // have; the next iteration's block-on-first
                            // will resynchronise with the stream.
                            debug!(error = %e, "pull stream error mid-batch; flushing partial batch");
                            dispatcher
                                .runtime_state
                                .telemetry
                                .nats_operation("fetch", "error", "transport", 1);
                            drop(next_permit);
                            break;
                        }
                        None => {
                            // Terminal mid-batch. Flush what we have, then
                            // the next iteration will rebuild.
                            warn!("pull stream ended mid-batch; flushing partial batch");
                            dispatcher
                                .runtime_state
                                .telemetry
                                .nats_operation("stream", "error", "stream_ended", 1);
                            stream = None;
                            drop(next_permit);
                            break;
                        }
                    }
                }
            }
        }

        // Step 3: dispatch. Acquire a dispatch permit (bounded by
        // `inflight_cap`) and spawn `handle_batch`, so the next fetch can
        // start immediately while Python processes this batch. Acquiring
        // *before* spawning is important: it keeps memory bounded when
        // Python is slower than NATS, and naturally applies backpressure
        // up to the stream (unacked messages will stop being pulled once
        // the server-side window fills).
        let got = batch.len();
        debug!(
            batch_size = got,
            quantum_ms = quantum.as_millis() as u64,
            assembled_ms = started.elapsed().as_millis() as u64,
            "batch assembled"
        );

        // Opportunistically reap finished task handles so the vec doesn't
        // grow unboundedly across a long-running loop. Cheap: only
        // allocates when there's actual movement and doesn't block.
        inflight_tasks.retain(|h| !h.is_finished());

        let permit = tokio::select! {
            biased;
            _ = shutdown.wait() => break 'outer,
            permit = dispatch_permits.clone().acquire_owned() => match permit {
                Ok(p) => p,
                Err(_) => {
                    warn!("pull-loop dispatch semaphore closed — exiting loop");
                    break 'outer;
                }
            },
        };

        let dispatcher_clone = Arc::clone(&dispatcher);
        let handle = tokio::spawn(async move {
            dispatcher_clone.handle_batch(batch).await;
            drop(permit);
        });
        inflight_tasks.push(handle);
    }

    // Wait for any spawned dispatch tasks to finish so callers can rely
    // on the pull-loop exit being a quiescence point (no still-running
    // IPC / NATS ACK work). We keep the deadline generous but bounded so
    // a pathological backend can't indefinitely stall shutdown.
    let pending = inflight_tasks.len();
    if pending > 0 {
        info!(
            pending_tasks = pending,
            deadline_ms = PULL_LOOP_TASK_JOIN_DEADLINE_MS,
            "pull loop exited; waiting for in-flight dispatch tasks"
        );
        let join_deadline = Duration::from_millis(PULL_LOOP_TASK_JOIN_DEADLINE_MS);
        let joined = timeout(join_deadline, async {
            for h in inflight_tasks.drain(..) {
                let _ = h.await;
            }
        })
        .await;
        if joined.is_err() {
            warn!(
                deadline_ms = PULL_LOOP_TASK_JOIN_DEADLINE_MS,
                "in-flight dispatch tasks did not finish before join deadline — proceeding to drain"
            );
        }
    }
    info!("pull loop exiting");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pull_loop_inflight_defaults_are_positive() {
        // Env-free default should be at least 1 and match the dispatcher
        // concurrency default when `SIE_PULL_LOOP_INFLIGHT` is unset. We
        // don't mutate env here to avoid polluting other tests in the
        // same process.
        assert!(pull_loop_inflight() >= 1);
        assert_eq!(
            pull_loop_inflight(),
            crate::dispatcher::default_max_concurrent_batches().max(1)
        );
    }

    #[test]
    fn pull_loop_max_outstanding_defaults_to_one_fetch_floor() {
        assert_eq!(parse_pull_loop_max_outstanding(None, 64, 4), 64);
        assert_eq!(parse_pull_loop_max_outstanding(None, 1, 4), 4);
        assert_eq!(parse_pull_loop_max_outstanding(Some("0"), 64, 4), 64);
        assert_eq!(parse_pull_loop_max_outstanding(Some("17"), 64, 4), 17);
    }

    #[test]
    fn nats_consumer_reconcile_interval_defaults_and_disables() {
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(None),
            Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(Some("")),
            Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
        assert_eq!(parse_nats_consumer_reconcile_interval_ms(Some("0")), None);
    }

    #[test]
    fn nats_consumer_reconcile_interval_clamps_to_slow_floor() {
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(Some("1")),
            Some(MIN_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(Some("10000")),
            Some(MIN_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(Some("30000")),
            Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
        assert_eq!(
            parse_nats_consumer_reconcile_interval_ms(Some("bad")),
            Some(DEFAULT_NATS_CONSUMER_RECONCILE_INTERVAL_MS)
        );
    }

    #[test]
    fn wave_cadence_enabled_default_is_true() {
        // Per-wave stepping is the default tail-latency posture.
        // Operators flip off via `SIE_RUST_WAVE_CADENCE=off` for the
        // inverse trade (smaller batches, lower p50, looser p99). We
        // don't mutate env here to avoid contaminating sibling tests
        // via the function's `OnceLock` cache.
        if std::env::var("SIE_RUST_WAVE_CADENCE").is_err() {
            assert!(
                wave_cadence_enabled(),
                "wave-cadence must default ON; see lib.rs docstring for the p99 rationale"
            );
        }
    }

    #[test]
    fn pull_quantum_includes_queue_ms_default_is_false() {
        // Env-free default must exclude `queue_ms` from the
        // FetchExpiryController input.
        // Operators opt in via `SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS=1`.
        // We don't mutate env here to avoid cross-test flakiness; the
        // function's `OnceLock` cache makes any toggle in a sibling
        // test a permanent contamination across the process.
        if std::env::var("SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS").is_err() {
            assert!(
                !pull_quantum_includes_queue_ms(),
                "queue_ms must be excluded by default — see Dispatcher docstring"
            );
        }
    }

    #[test]
    fn cancel_subject_parser_extracts_request_id() {
        assert_eq!(
            request_id_from_cancel_subject("cancel.gateway-1.018f9b0c-req"),
            Some("018f9b0c-req".to_string())
        );
        assert_eq!(
            request_id_from_cancel_subject("cancel.gateway-1.req.with.dots"),
            Some("req.with.dots".to_string())
        );
        assert_eq!(request_id_from_cancel_subject("cancel.gateway-1"), None);
        assert_eq!(request_id_from_cancel_subject("other.gateway-1.req"), None);
    }
}
