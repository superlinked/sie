use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use tokio::sync::RwLock;

use crate::routing::hrw::{RingEntry, RingSnapshot};
use crate::state::pool_manager::normalize_pool_name;
use crate::types::worker::MAX_UNSUPPORTED_MODELS;
use crate::types::{
    ClusterStatus, ModelInfo, WorkerHealth, WorkerInfo, WorkerState, WorkerStatusMessage,
};

/// Rate-limit gate for the duplicate-worker-name warning, in whole
/// seconds since process start. `ring_snapshot_for` runs on the
/// per-request hot path, so an unconditional `warn!` on a persistent
/// operator misconfiguration would flood the log. Emit at most once per
/// `DUP_NAME_WARN_INTERVAL` regardless of how many requests observe it.
static LAST_DUP_NAME_WARN_SECS: AtomicU64 = AtomicU64::new(0);
const DUP_NAME_WARN_INTERVAL: Duration = Duration::from_secs(60);

/// Warn (rate-limited) that the ring snapshot dropped `dropped` entries
/// because two or more workers advertised the same `name`. This is an
/// operator misconfiguration: worker `name` doubles as the HRW ring key
/// and per-worker subject id, so duplicates collapse onto one subject.
fn warn_duplicate_worker_names(model: &str, pool: &str, dropped: usize) {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    let now_secs = START.get_or_init(Instant::now).elapsed().as_secs();
    let last = LAST_DUP_NAME_WARN_SECS.load(Ordering::Relaxed);
    if now_secs.saturating_sub(last) < DUP_NAME_WARN_INTERVAL.as_secs() && last != 0 {
        return;
    }
    // Best-effort CAS: a lost race just means another thread already
    // warned within the window, which is the behaviour we want.
    if LAST_DUP_NAME_WARN_SECS
        .compare_exchange(last, now_secs.max(1), Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        tracing::warn!(
            model = %model,
            pool = %pool,
            dropped = dropped,
            "duplicate worker names in HRW ring — operator misconfig; \
             dedup kept the lowest-url entry per name (worker `name` must \
             be unique: it is the ring key and per-worker subject id)"
        );
    }
}

pub type OnWorkerHealthy = Arc<dyn Fn(&WorkerState) + Send + Sync>;
/// Fired when a worker transitions out of the
/// "eligible for HRW dispatch" set. Two triggers:
/// 1. `mark_unhealthy` (explicit health failure).
/// 2. Saturation rising edge — `saturated` flipped from `false` to `true`.
///
/// Used by the proxy/publisher to invalidate cached per-`(model, pool)`
/// ring snapshots and (optionally) to log a structured trace.
pub type OnWorkerDegraded = Arc<dyn Fn(&WorkerState) + Send + Sync>;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct HeartbeatSweep {
    pub unhealthy: Vec<String>,
    pub evicted: Vec<String>,
}

/// Concrete queue lane resolved from the worker registry.
///
/// `pool_name` is the logical tenant/capacity pool. `machine_profile`
/// is the hardware lane that must be embedded in the NATS subject so
/// non-interchangeable worker groups sharing the same pool do not
/// overlap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueRoute {
    pub pool_name: String,
    pub machine_profile: String,
}

/// Result of resolving a queue lane for one model.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueueRouteLookup {
    pub route: Option<QueueRoute>,
    /// A lane was passed over because a worker in it that reports the
    /// expected hash cannot serve the model.
    pub excluded_unsupported: bool,
}

/// Pre-computed snapshot of healthy workers, indexed by bundle.
/// Rebuilt on every worker state change and swapped atomically via Arc.
#[derive(Default)]
struct RegistrySnapshot {
    /// All healthy workers.
    all_healthy: Vec<WorkerState>,
    /// Healthy workers indexed by lowercase bundle name.
    by_bundle: HashMap<String, Vec<WorkerState>>,
}

impl RegistrySnapshot {
    fn build(workers: &HashMap<String, WorkerState>) -> Self {
        let mut all_healthy = Vec::new();
        let mut by_bundle: HashMap<String, Vec<WorkerState>> = HashMap::new();

        for w in workers.values() {
            if !w.healthy() {
                continue;
            }
            all_healthy.push(w.clone());
            by_bundle
                .entry(w.bundle.to_lowercase())
                .or_default()
                .push(w.clone());
        }

        Self {
            all_healthy,
            by_bundle,
        }
    }
}

pub struct WorkerRegistry {
    workers: RwLock<HashMap<String, WorkerState>>,
    heartbeat_timeout: Duration,
    stale_evict_after: Duration,
    on_worker_healthy: Option<OnWorkerHealthy>,
    on_worker_degraded: Option<OnWorkerDegraded>,
    /// Pre-computed snapshot for lock-free select_worker lookups.
    snapshot: ArcSwap<RegistrySnapshot>,

    // QPS tracking. `record_request` is called on every queue-mode
    // success path, so we keep it to a single relaxed atomic
    // increment — the prior `DashMap<String, i64>` flavour paid
    // `String` allocation + hash + bucket lookup for every request
    // despite there only ever being one caller (`"queue"`).
    // `get_cluster_status` swaps the counter and divides by the
    // elapsed time since the last read.
    request_count: AtomicU64,
    last_qps_calculation: RwLock<Instant>,
    current_qps: RwLock<f64>,
}

impl WorkerRegistry {
    /// Construct without a degrade callback. The main gateway uses
    /// [`Self::with_callbacks`] directly so the dead-code lint fires
    /// here under default-feature builds; tests and other binaries
    /// still use this wrapper as the ergonomic no-degrade-callback
    /// entry point.
    #[allow(dead_code)]
    pub fn new(heartbeat_timeout: Duration, on_worker_healthy: Option<OnWorkerHealthy>) -> Self {
        Self::with_callbacks(heartbeat_timeout, on_worker_healthy, None)
    }

    /// Same as [`Self::new`] but also accepts an
    /// [`OnWorkerDegraded`] callback fired on health → unhealthy and
    /// saturation false → true transitions. Used by the HRW
    /// ring cache invalidation; pre-direct-dispatch callers can keep using
    /// [`Self::new`].
    pub fn with_callbacks(
        heartbeat_timeout: Duration,
        on_worker_healthy: Option<OnWorkerHealthy>,
        on_worker_degraded: Option<OnWorkerDegraded>,
    ) -> Self {
        let stale_evict_after = heartbeat_timeout.checked_mul(20).unwrap_or(Duration::MAX);
        Self {
            workers: RwLock::new(HashMap::new()),
            heartbeat_timeout,
            stale_evict_after,
            on_worker_healthy,
            on_worker_degraded,
            snapshot: ArcSwap::from_pointee(RegistrySnapshot::default()),
            request_count: AtomicU64::new(0),
            last_qps_calculation: RwLock::new(Instant::now()),
            current_qps: RwLock::new(0.0),
        }
    }

    /// Rebuild the pre-computed snapshot from the current worker state.
    /// Lock-free via ArcSwap — safe to call while holding the workers write lock.
    fn rebuild_snapshot(&self, workers: &HashMap<String, WorkerState>) {
        self.snapshot
            .store(Arc::new(RegistrySnapshot::build(workers)));
    }

    pub async fn workers(&self) -> HashMap<String, WorkerState> {
        self.workers.read().await.clone()
    }

    pub async fn healthy_workers(&self) -> Vec<WorkerState> {
        self.snapshot.load().all_healthy.clone()
    }

    pub async fn update_worker(&self, url: &str, msg: WorkerStatusMessage) -> bool {
        let (became_healthy, became_degraded, worker_copy) = {
            let mut workers = self.workers.write().await;
            let exists = workers.contains_key(url);
            let w = workers
                .entry(url.to_string())
                .or_insert_with(|| WorkerState {
                    url: url.to_string(),
                    name: String::new(),
                    health: WorkerHealth::Unknown,
                    gpu_count: 1,
                    ready_gpu_slots: 0,
                    machine_profile: String::new(),
                    bundle: "default".to_string(),
                    bundle_config_hash: String::new(),
                    supports_execution_authority_v1: false,
                    models: Vec::new(),
                    queue_depth: 0,
                    pending_cost: 0,
                    inflight_batches: 0,
                    memory_used_bytes: 0,
                    memory_total_bytes: 0,
                    last_heartbeat: Instant::now(),
                    pool_name: String::new(),
                    saturated: false,
                    unsupported_models: Arc::from([]),
                    unsupported_overflow: false,
                });

            let was_healthy = w.healthy();
            let was_eligible = w.eligible_for_dispatch();

            w.name = if msg.name.is_empty() {
                url.to_string()
            } else {
                msg.name.clone()
            };
            let reported_gpu_slots = msg.total_gpu_slots.unwrap_or(msg.gpu_count);
            w.gpu_count = reported_gpu_slots.max(1);
            w.bundle = if msg.bundle.is_empty() {
                "default".to_string()
            } else {
                msg.bundle.clone()
            };
            w.bundle_config_hash = msg.bundle_config_hash.clone();
            w.supports_execution_authority_v1 = msg.supports_execution_authority_v1;
            let overflow = msg.unsupported_models.len() > MAX_UNSUPPORTED_MODELS;
            if overflow && !w.unsupported_overflow {
                tracing::warn!(
                    worker = %w.name,
                    reported = msg.unsupported_models.len(),
                    max = MAX_UNSUPPORTED_MODELS,
                    "worker reported more unsupported models than the gateway accepts; \
                     routing no model to it until it reports fewer"
                );
            }
            w.unsupported_overflow = overflow;
            w.unsupported_models = msg
                .unsupported_models
                .iter()
                .take(MAX_UNSUPPORTED_MODELS)
                .cloned()
                .collect();
            w.machine_profile = msg.machine_profile.clone();
            w.pool_name = msg.pool_name.clone();
            w.models = msg.loaded_models.clone();

            // Aggregate queue depth from models (fallback to compact top-level field)
            w.queue_depth = if !msg.models.is_empty() {
                msg.models.iter().map(|m| m.queue_depth).sum()
            } else {
                msg.queue_depth.unwrap_or(0)
            };
            w.pending_cost = msg.pending_cost.unwrap_or(0).max(0);
            w.inflight_batches = msg.inflight_batches.unwrap_or(0).max(0);
            w.ready_gpu_slots = msg
                .ready_gpu_slots
                .unwrap_or(if msg.ready { w.gpu_count } else { 0 })
                .clamp(0, w.gpu_count);

            // Aggregate GPU memory (fallback to compact top-level fields)
            if !msg.gpus.is_empty() {
                w.memory_used_bytes = msg.gpus.iter().map(|g| g.memory_used_bytes).sum();
                w.memory_total_bytes = msg.gpus.iter().map(|g| g.memory_total_bytes).sum();
            } else {
                w.memory_used_bytes = msg.memory_used_bytes.unwrap_or(0);
                w.memory_total_bytes = msg.memory_total_bytes.unwrap_or(0);
            }

            w.last_heartbeat = Instant::now();

            // A worker that heartbeats `ready: false` or has zero ready GPU
            // slots is no longer serviceable and must leave the dispatch set /
            // HRW ring promptly. Waiting for the heartbeat timeout would keep
            // routing direct dispatches to a worker that has declared no usable
            // child capacity. The exact transition depends on its prior state.
            if msg.ready && w.ready_gpu_slots > 0 {
                w.health = WorkerHealth::Healthy;
            } else if w.health == WorkerHealth::Healthy {
                // Was serving and now reports not-ready: the worker has
                // regressed (e.g. a wedged GPU per issue #1025, or draining for
                // shutdown). Heartbeats keep arriving, so the heartbeat-timeout
                // path won't catch it — downgrade explicitly so cluster-status
                // and routing stop treating it as healthy.
                w.health = WorkerHealth::Unhealthy;
            }
            // else: leave the current state unchanged. A worker that never
            // became healthy stays Unknown (still starting up); one already
            // downgraded to Unhealthy stays Unhealthy on every subsequent
            // not-ready heartbeat instead of flipping back to Unknown, so a
            // persistently wedged worker keeps reporting as regressed (#1025).

            // Ingest the saturation flag verbatim. The
            // worker owns hysteresis; the gateway just consumes the
            // boolean and fires the degrade callback on the rising
            // edge so the HRW ring cache can invalidate.
            w.saturated = msg.saturated;

            let became_healthy = w.healthy() && (!exists || !was_healthy);
            // Falling edge into "ineligible for HRW dispatch", reported
            // exactly once so the degrade callback (ring invalidation)
            // fires the same way regardless of the trigger. Two triggers
            // are folded here so we don't double-fire:
            //   * not-ready demotion (healthy → Unhealthy), and
            //   * saturation rising edge (was eligible, now saturated).
            // `mark_unhealthy` reports explicit health failures on its
            // own falling edge; this path covers the heartbeat-driven
            // `ready: false` and saturation transitions.
            let became_degraded = was_eligible && !w.eligible_for_dispatch();
            let worker_copy = w.clone();
            // Rebuild snapshot inside write lock — lock-free ArcSwap, no deadlock
            self.rebuild_snapshot(&workers);
            (became_healthy, became_degraded, worker_copy)
        };

        if became_healthy {
            if let Some(cb) = &self.on_worker_healthy {
                cb(&worker_copy);
            }
        }
        if became_degraded {
            if let Some(cb) = &self.on_worker_degraded {
                cb(&worker_copy);
            }
        }

        became_healthy
    }

    pub async fn remove_worker(&self, url: &str) -> bool {
        let mut workers = self.workers.write().await;
        let removed = workers.remove(url).is_some();
        self.rebuild_snapshot(&workers);
        removed
    }

    pub async fn mark_unhealthy(&self, url: &str) {
        let (degraded_copy,) = {
            let mut workers = self.workers.write().await;
            let copy = if let Some(w) = workers.get_mut(url) {
                let was_eligible = w.eligible_for_dispatch();
                w.health = WorkerHealth::Unhealthy;
                // Only report degradation on the falling edge.
                if was_eligible {
                    Some(w.clone())
                } else {
                    None
                }
            } else {
                None
            };
            self.rebuild_snapshot(&workers);
            (copy,)
        };
        if let Some(w) = degraded_copy {
            if let Some(cb) = &self.on_worker_degraded {
                cb(&w);
            }
        }
    }

    /// Dispatch-eligible workers for a `(model, pool, machine_profile, bundle)` —
    /// healthy, not saturated, with `model` in `loaded_models` and not in
    /// `unsupported_models`, a matching bundle config hash, and the full
    /// queue lane matching. Pool /
    /// machine-profile / bundle matches are case-insensitive to mirror
    /// [`resolve_queue_route_matching`]. Used by the proxy to build
    /// per-request HRW snapshots.
    ///
    /// A `model:profile` request is matched as its own model id. The server
    /// loader expands non-default profiles into concrete `base:profile`
    /// configs with profile-specific SGLang launch args, so collapsing the
    /// suffix here would direct-dispatch to a worker that only has the base
    /// model resident and can silently launch the wrong precision/profile.
    /// If no worker has the variant loaded yet, the request falls through to
    /// the pool subject and lazy-loads that concrete variant.
    #[cfg(test)]
    pub fn dispatch_workers_for(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
    ) -> Vec<WorkerState> {
        self.dispatch_workers_for_matching(
            model,
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            None,
        )
    }

    fn dispatch_workers_for_matching(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> Vec<WorkerState> {
        let snap = self.snapshot.load();
        // Narrow to the lane's bundle via the `by_bundle` index (O(N_bundle))
        // instead of scanning the whole-cluster `all_healthy` list
        // (O(N_cluster)) — mirroring `lazy_lane_workers_for_matching`, which
        // already does this on the same request. Every worker in the bucket
        // already has this bundle, so the `w.bundle` predicate is dropped.
        // See #1607.
        let bundle_lower = bundle.to_lowercase();
        let Some(candidates) = snap.by_bundle.get(&bundle_lower) else {
            return Vec::new();
        };
        candidates
            .iter()
            .filter(|w| {
                !w.saturated
                    && worker_allowed_by_admission(w, admitted_worker_names)
                    && !w.machine_profile.is_empty()
                    && w.pool_name.eq_ignore_ascii_case(pool)
                    && w.machine_profile.eq_ignore_ascii_case(machine_profile)
                    && bundle_config_hash_matches(&w.bundle_config_hash, bundle_config_hash)
                    && w.models.iter().any(|m| m.eq_ignore_ascii_case(model))
                    && w.supports_model(model)
            })
            .cloned()
            .collect()
    }

    fn lazy_lane_workers_for_matching(
        &self,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> Vec<WorkerState> {
        let snap = self.snapshot.load();
        let bundle_lower = bundle.to_lowercase();
        let Some(candidates) = snap.by_bundle.get(&bundle_lower) else {
            return Vec::new();
        };
        candidates
            .iter()
            .filter(|w| {
                !w.saturated
                    && worker_allowed_by_admission(w, admitted_worker_names)
                    && !w.machine_profile.is_empty()
                    && w.pool_name.eq_ignore_ascii_case(pool)
                    && w.machine_profile.eq_ignore_ascii_case(machine_profile)
                    && bundle_config_hash_matches(&w.bundle_config_hash, bundle_config_hash)
            })
            .cloned()
            .collect()
    }

    /// Workers that can receive a pool fallback for this concrete lane.
    ///
    /// Unlike [`Self::dispatch_workers_for`], this intentionally does not
    /// require `model` to already be present in `loaded_models`: the pool
    /// fallback path can land on a healthy same-lane worker that lazy-loads
    /// the model. This count is used only to decide whether first-chunk
    /// failover has an alternate consumer.
    pub fn pool_fallback_lane_worker_count(
        &self,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> usize {
        self.lazy_lane_workers_for_matching(
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            admitted_worker_names,
        )
        .len()
    }

    /// Build a worker ring for capped logical batch dispatch.
    ///
    /// Unlike [`Self::ring_snapshot_for_admitted`], this intentionally does not
    /// require the target model to already be in `loaded_models`: encode/score/
    /// extract pool work is allowed to lazy-load the model. The admission filter
    /// still keeps the ring restricted to workers assigned to the logical pool.
    pub fn lazy_lane_ring_snapshot_for_admitted(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: &HashSet<String>,
    ) -> RingSnapshot {
        let mut workers = self.lazy_lane_workers_for_matching(
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            Some(admitted_worker_names),
        );
        workers.retain(|w| w.supports_model(model));
        workers.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.url.cmp(&b.url)));
        let before = workers.len();
        workers.dedup_by(|a, b| a.name == b.name);
        if workers.len() != before {
            warn_duplicate_worker_names(model, pool, before - workers.len());
        }
        RingSnapshot::from_entries(workers.into_iter().map(|w| {
            RingEntry::with_pressure(
                w.name,
                w.ready_gpu_slots,
                w.queue_depth,
                w.pending_cost,
                w.inflight_batches,
            )
        }))
    }

    /// Build a [`RingSnapshot`] for the given concrete queue lane. Uses
    /// the worker `name` (direct-dispatch worker_id convention) as the ring
    /// key; the same name shows up on the per-worker NATS subject
    /// `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
    ///
    /// The output is sorted by worker name so HRW tie-breaks are
    /// identical across gateway replicas regardless of the upstream
    /// `HashMap` iteration order in [`RegistrySnapshot::build`].
    ///
    /// Worker `name` is operator-set and has no uniqueness guarantee. Two
    /// workers sharing a `name` would hash to the same `worker_id_hash`
    /// and tie-break equally, making the HRW pick ambiguous and pointing
    /// both at the same per-worker subject. We therefore dedup ring
    /// entries by `name`, keeping the one whose `url` sorts first so the
    /// choice is deterministic across replicas, and warn (rate-limited)
    /// that an operator misconfiguration was observed. The wire-level
    /// subject format is unchanged — only which workers enter the ring.
    #[cfg(test)]
    pub fn ring_snapshot_for(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
    ) -> RingSnapshot {
        self.ring_snapshot_for_matching(
            model,
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            None,
        )
    }

    pub fn ring_snapshot_for_admitted(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> RingSnapshot {
        self.ring_snapshot_for_matching(
            model,
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            admitted_worker_names,
        )
    }

    fn ring_snapshot_for_matching(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        bundle_config_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> RingSnapshot {
        let mut workers = self.dispatch_workers_for_matching(
            model,
            pool,
            machine_profile,
            bundle,
            bundle_config_hash,
            admitted_worker_names,
        );
        // Sort by `(name, url)` so duplicate names are adjacent and the
        // smallest-`url` entry comes first within each name group.
        workers.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.url.cmp(&b.url)));
        let before = workers.len();
        // Keep the first entry per `name` (smallest `url`).
        workers.dedup_by(|a, b| a.name == b.name);
        if workers.len() != before {
            warn_duplicate_worker_names(model, pool, before - workers.len());
        }
        RingSnapshot::from_entries(workers.into_iter().map(|w| {
            RingEntry::with_pressure(
                w.name,
                w.ready_gpu_slots,
                w.queue_depth,
                w.pending_cost,
                w.inflight_batches,
            )
        }))
    }

    pub async fn check_heartbeats(&self) -> HeartbeatSweep {
        let mut workers = self.workers.write().await;
        let now = Instant::now();
        let mut sweep = HeartbeatSweep::default();
        let mut evict = Vec::new();
        for (url, w) in workers.iter_mut() {
            let age = now.duration_since(w.last_heartbeat);
            if age > self.stale_evict_after {
                evict.push(url.clone());
            } else if w.healthy() && age > self.heartbeat_timeout {
                w.health = WorkerHealth::Unhealthy;
                sweep.unhealthy.push(url.clone());
            }
        }
        for url in evict {
            if workers.remove(&url).is_some() {
                sweep.evicted.push(url);
            }
        }
        if !sweep.unhealthy.is_empty() || !sweep.evicted.is_empty() {
            self.rebuild_snapshot(&workers);
        }
        sweep
    }

    /// Resolve the NATS pool name for queue mode routing.
    /// Finds a healthy worker matching the bundle (and optionally GPU) that has
    /// a pool_name set and reports the expected bundle config hash.
    #[cfg(test)]
    pub async fn resolve_queue_pool(
        &self,
        bundle: &str,
        gpu: &str,
        bundle_config_hash: &str,
    ) -> Option<String> {
        self.resolve_queue_route_matching(bundle, gpu, None, bundle_config_hash, "")
            .route
            .map(|route| route.pool_name)
    }

    /// Resolve the NATS pool name, constrained to a specific logical pool.
    #[cfg(test)]
    pub async fn resolve_queue_pool_in_pool(
        &self,
        bundle: &str,
        gpu: &str,
        pool_name: &str,
        bundle_config_hash: &str,
    ) -> Option<String> {
        self.resolve_queue_route_in_pool(bundle, gpu, pool_name, bundle_config_hash)
            .await
            .map(|route| route.pool_name)
    }

    /// Resolve a concrete queue lane, constrained to a specific logical pool.
    #[cfg(test)]
    pub async fn resolve_queue_route_in_pool(
        &self,
        bundle: &str,
        gpu: &str,
        pool_name: &str,
        bundle_config_hash: &str,
    ) -> Option<QueueRoute> {
        self.resolve_queue_route_matching(bundle, gpu, Some(pool_name), bundle_config_hash, "")
            .route
    }

    /// Resolve a concrete queue lane for `model`, constrained to a logical pool.
    ///
    /// Every worker of a lane consumes the lane subject, so a lane is eligible
    /// only when no worker in it that reports the expected hash lists `model`
    /// in `unsupported_models`. Otherwise such a worker would receive the
    /// model's items and NAK them, spending JetStream delivery attempts.
    pub fn resolve_queue_route_for_model(
        &self,
        bundle: &str,
        gpu: &str,
        pool_name: &str,
        bundle_config_hash: &str,
        model: &str,
    ) -> QueueRouteLookup {
        self.resolve_queue_route_matching(bundle, gpu, Some(pool_name), bundle_config_hash, model)
    }

    fn resolve_queue_route_matching(
        &self,
        bundle: &str,
        gpu: &str,
        pool_name: Option<&str>,
        bundle_config_hash: &str,
        model: &str,
    ) -> QueueRouteLookup {
        let snap = self.snapshot.load();

        // `by_bundle` is keyed with `w.bundle.to_lowercase()` in
        // `RegistrySnapshot::build`, so the lookup has to use the
        // same Unicode-aware lowercase to keep non-ASCII bundle
        // ids reachable. We only
        // allocate once per request on this branch, so the cost
        // is negligible compared to the per-candidate
        // `to_lowercase()` that `eq_ignore_ascii_case` replaced
        // below — gpu labels are `l4`, `a100`, `l4-spot`, … and
        // ASCII-only in practice.
        let bundle_lower = bundle.to_lowercase();

        // Use pre-computed by_bundle index for efficient lock-free lookup
        let Some(candidates) = snap.by_bundle.get(&bundle_lower) else {
            return QueueRouteLookup::default();
        };

        let mut eligible: Vec<&WorkerState> = Vec::new();
        let mut unsupporting: Vec<&WorkerState> = Vec::new();
        for w in candidates {
            if w.pool_name.is_empty() {
                continue;
            }
            if w.machine_profile.is_empty() {
                continue;
            }
            if let Some(pool_name) = pool_name {
                if !w.pool_name.eq_ignore_ascii_case(pool_name) {
                    continue;
                }
            }
            if !gpu.is_empty() && !w.machine_profile.eq_ignore_ascii_case(gpu) {
                continue;
            }
            if !bundle_config_hash_matches(&w.bundle_config_hash, bundle_config_hash) {
                continue;
            }
            if model.is_empty() || w.supports_model(model) {
                eligible.push(w);
            } else {
                unsupporting.push(w);
            }
        }
        let same_lane = |a: &WorkerState, b: &WorkerState| {
            a.pool_name.eq_ignore_ascii_case(&b.pool_name)
                && a.machine_profile.eq_ignore_ascii_case(&b.machine_profile)
        };
        let route = eligible
            .into_iter()
            .find(|w| !unsupporting.iter().any(|u| same_lane(u, w)))
            .map(|w| QueueRoute {
                pool_name: normalize_pool_name(&w.pool_name),
                machine_profile: w.machine_profile.clone(),
            });
        QueueRouteLookup {
            route,
            excluded_unsupported: !unsupporting.is_empty(),
        }
    }

    /// Select only a fresh, positively capable worker for verified direct dispatch.
    /// The versioned subject keeps the execution fence even after a worker rollback.
    pub fn execution_authority_worker(
        &self,
        model: &str,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
        expected_hash: &str,
        admitted_worker_names: Option<&HashSet<String>>,
    ) -> Option<String> {
        if expected_hash.is_empty() {
            return None;
        }
        let snap = self.snapshot.load();
        let candidates = snap.by_bundle.get(&bundle.to_lowercase())?;
        let mut name_counts = HashMap::<&str, usize>::with_capacity(snap.all_healthy.len());
        for worker in &snap.all_healthy {
            *name_counts.entry(worker.name.as_str()).or_default() += 1;
        }
        let ring = RingSnapshot::from_entries(
            candidates
                .iter()
                .filter(|w| {
                    w.eligible_for_dispatch()
                && w.supports_execution_authority_v1
                && w.last_heartbeat.elapsed() <= self.heartbeat_timeout
                && w.bundle_config_hash == expected_hash
                && w.pool_name.eq_ignore_ascii_case(pool)
                && w.machine_profile.eq_ignore_ascii_case(machine_profile)
                && w.supports_model(model)
                && worker_allowed_by_admission(w, admitted_worker_names)
                // Avoid subject normalization collisions and ambiguous worker identities.
                && !w.name.is_empty()
                && w.name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
                && name_counts.get(w.name.as_str()) == Some(&1)
                })
                .map(|w| {
                    RingEntry::with_pressure(
                        w.name.clone(),
                        w.ready_gpu_slots,
                        w.queue_depth,
                        w.pending_cost,
                        w.inflight_batches,
                    )
                }),
        );
        let seed = uuid::Uuid::now_v7().to_string();
        let key = crate::routing::key::RoutingKeyResolved {
            hash: Some(crate::routing::key::hash_bytes(&seed)),
            source: crate::routing::key::KeySource::RoutingKey,
            #[cfg(feature = "raw-routing-logs")]
            raw_for_debug: None,
        };
        crate::routing::pick_worker(&ring, &key).map(str::to_owned)
    }

    pub async fn get_models(&self) -> HashMap<String, Vec<String>> {
        let snap = self.snapshot.load();
        let mut models: HashMap<String, Vec<String>> = HashMap::new();
        for w in &snap.all_healthy {
            for m in &w.models {
                models.entry(m.clone()).or_default().push(w.url.clone());
            }
        }
        models
    }

    pub async fn get_gpu_types(&self) -> Vec<String> {
        let snap = self.snapshot.load();
        let mut seen = std::collections::HashSet::new();
        for w in &snap.all_healthy {
            if !w.machine_profile.is_empty() {
                seen.insert(w.machine_profile.clone());
            }
        }
        seen.into_iter().collect()
    }

    /// Record a single completed request for the QPS gauge.
    ///
    /// The body is a single `fetch_add(_, Ordering::Relaxed)` on a
    /// process-wide `AtomicU64` — the previous
    /// `DashMap<String, i64>` flavour paid a `String` allocation +
    /// hash + bucket lookup per request, and every call site only
    /// ever passed the constant `"queue"` label. Kept `async` and
    /// the `_label` parameter so the call shape matches `main` and
    /// callers don't have to change if the API ever grows a real
    /// per-label counter back.
    pub async fn record_request(&self, _label: &str) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn get_cluster_status(&self) -> ClusterStatus {
        // Calculate QPS
        let qps = {
            let now = Instant::now();
            let mut last_calc = self.last_qps_calculation.write().await;
            let elapsed = now.duration_since(*last_calc).as_secs_f64();
            if elapsed >= 1.0 {
                // Drain the counter atomically so no request is
                // counted twice across the window boundary.
                let total = self.request_count.swap(0, Ordering::Relaxed);
                let qps = if elapsed > 0.0 {
                    total as f64 / elapsed
                } else {
                    0.0
                };
                *self.current_qps.write().await = qps;
                *last_calc = now;
            }
            *self.current_qps.read().await
        };

        let workers = self.workers.read().await;
        let mut worker_infos = Vec::new();
        let mut total_gpus = 0i32;
        let mut model_workers: HashMap<String, Vec<(String, String, i32)>> = HashMap::new();

        for w in workers.values() {
            worker_infos.push(WorkerInfo {
                name: w.name.clone(),
                url: w.url.clone(),
                gpu: w.machine_profile.clone(),
                gpu_count: w.gpu_count,
                ready_gpu_slots: w.ready_gpu_slots,
                loaded_models: w.models.clone(),
                queue_depth: w.queue_depth,
                pending_cost: w.pending_cost,
                inflight_batches: w.inflight_batches,
                memory_used_bytes: w.memory_used_bytes,
                memory_total_bytes: w.memory_total_bytes,
                healthy: w.healthy(),
                bundle: w.bundle.clone(),
                bundle_config_hash: w.bundle_config_hash.clone(),
                unsupported_models: w.unsupported_models.to_vec(),
                unsupported_models_overflow: w.unsupported_overflow,
            });

            if w.healthy() {
                total_gpus += w.gpu_count;
                for m in &w.models {
                    model_workers.entry(m.clone()).or_default().push((
                        w.name.clone(),
                        w.machine_profile.clone(),
                        w.queue_depth,
                    ));
                }
            }
        }

        let mut models = Vec::new();
        for (name, mw) in &model_workers {
            let mut gpu_set = std::collections::HashSet::new();
            let mut total_qd = 0i32;
            for (_, gpu, qd) in mw {
                gpu_set.insert(gpu.clone());
                total_qd += qd;
            }
            models.push(ModelInfo {
                name: name.clone(),
                state: "loaded".to_string(),
                worker_count: mw.len() as i32,
                gpu_types: gpu_set.into_iter().collect(),
                total_queue_depth: total_qd,
            });
        }

        let healthy_count = workers.values().filter(|w| w.healthy()).count() as i32;

        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as f64
            / 1000.0;

        ClusterStatus {
            timestamp: now_millis,
            worker_count: healthy_count,
            gpu_count: total_gpus,
            models_loaded: model_workers.len() as i32,
            total_qps: qps,
            workers: worker_infos,
            models,
        }
    }
}

fn bundle_config_hash_matches(worker_hash: &str, expected_hash: &str) -> bool {
    expected_hash.is_empty() || worker_hash == expected_hash
}

fn worker_allowed_by_admission(
    worker: &WorkerState,
    admitted_worker_names: Option<&HashSet<String>>,
) -> bool {
    match admitted_worker_names {
        Some(names) => names.contains(&worker.name),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker::{GpuStatus, ModelStatus};
    use crate::types::WorkerStatusMessage;
    fn make_msg(ready: bool) -> WorkerStatusMessage {
        WorkerStatusMessage {
            supports_execution_authority_v1: false,
            name: "worker-1".into(),
            ready,
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: "l4-spot".into(),
            pool_name: String::new(),
            bundle: "default".into(),
            bundle_config_hash: "abc123".into(),
            loaded_models: vec!["BAAI/bge-m3".into()],
            models: vec![ModelStatus { queue_depth: 2 }],
            gpus: vec![GpuStatus {
                memory_used_bytes: 1000,
                memory_total_bytes: 4000,
            }],
            queue_depth: None,
            pending_cost: None,
            inflight_batches: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated: false,
            terminated: false,
            unsupported_models: Vec::new(),
        }
    }

    fn registry() -> WorkerRegistry {
        WorkerRegistry::new(Duration::from_secs(30), None)
    }

    // ── update_worker ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_update_worker_registers_new() {
        let reg = registry();
        let became_healthy = reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert!(became_healthy);

        let workers = reg.workers().await;
        assert_eq!(workers.len(), 1);
        let w = workers.get("http://w1:8080").unwrap();
        assert_eq!(w.name, "worker-1");
        assert_eq!(w.health, WorkerHealth::Healthy);
        assert_eq!(w.machine_profile, "l4-spot");
        assert_eq!(w.bundle, "default");
        assert_eq!(w.models, vec!["BAAI/bge-m3".to_string()]);
        assert_eq!(w.queue_depth, 2);
        assert_eq!(w.memory_used_bytes, 1000);
        assert_eq!(w.memory_total_bytes, 4000);
    }

    #[tokio::test]
    async fn test_update_worker_ready_with_zero_ready_slots_is_not_healthy() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "default".into();
        msg.total_gpu_slots = Some(4);
        msg.ready_gpu_slots = Some(0);

        let became_healthy = reg.update_worker("http://w1:8080", msg).await;

        assert!(!became_healthy);
        let workers = reg.workers().await;
        let w = &workers["http://w1:8080"];
        assert_eq!(w.gpu_count, 4);
        assert_eq!(w.ready_gpu_slots, 0);
        assert_eq!(w.health, WorkerHealth::Unknown);
        assert!(reg
            .resolve_queue_route_in_pool("default", "l4-spot", "default", "abc123")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_update_worker_first_seen_not_ready_stays_unknown() {
        let reg = registry();
        let became = reg.update_worker("http://w1:8080", make_msg(false)).await;
        assert!(!became);

        // A never-Healthy worker reporting `ready: false` is still starting up
        // (issue #1025 readiness): it stays `Unknown` — not dispatchable, but
        // not yet a regression — rather than being demoted to `Unhealthy`.
        let workers = reg.workers().await;
        assert_eq!(workers["http://w1:8080"].health, WorkerHealth::Unknown);
    }

    #[tokio::test]
    async fn test_update_worker_ready_false_demotes_healthy_immediately() {
        // Regression: a worker already `Healthy` that heartbeats
        // `ready: false` must be demoted right away (not left in the
        // dispatch set until heartbeat timeout) and must fire the degrade
        // callback on the falling edge so the HRW ring cache invalidates.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let degrade_calls = Arc::new(AtomicUsize::new(0));
        let degrade_clone = degrade_calls.clone();
        let reg = WorkerRegistry::with_callbacks(
            Duration::from_secs(30),
            None,
            Some(Arc::new(move |_w: &WorkerState| {
                degrade_clone.fetch_add(1, Ordering::SeqCst);
            })),
        );

        // First heartbeat: ready → Healthy, in the dispatch set.
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert_eq!(reg.healthy_workers().await.len(), 1);
        assert_eq!(degrade_calls.load(Ordering::SeqCst), 0);

        // Second heartbeat: ready=false → demoted to Unhealthy, removed
        // from the dispatch set, degrade callback fired exactly once.
        let became = reg.update_worker("http://w1:8080", make_msg(false)).await;
        assert!(!became);
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Unhealthy
        );
        assert!(reg.healthy_workers().await.is_empty());
        assert_eq!(degrade_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_update_worker_saturation_rising_edge_fires_degrade_once() {
        // The saturation falling edge (eligible → saturated) must still
        // fire the degrade callback exactly once, and not double-fire on
        // a repeated saturated heartbeat.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let degrade_calls = Arc::new(AtomicUsize::new(0));
        let degrade_clone = degrade_calls.clone();
        let reg = WorkerRegistry::with_callbacks(
            Duration::from_secs(30),
            None,
            Some(Arc::new(move |_w: &WorkerState| {
                degrade_clone.fetch_add(1, Ordering::SeqCst);
            })),
        );

        reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert_eq!(degrade_calls.load(Ordering::SeqCst), 0);

        let mut sat = make_msg(true);
        sat.saturated = true;
        reg.update_worker("http://w1:8080", sat).await;
        assert_eq!(degrade_calls.load(Ordering::SeqCst), 1);

        // Still saturated on the next heartbeat — no new falling edge.
        let mut sat2 = make_msg(true);
        sat2.saturated = true;
        reg.update_worker("http://w1:8080", sat2).await;
        assert_eq!(degrade_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_update_worker_healthy_then_not_ready_downgrades() {
        // A worker that was serving and then reports ready=false has regressed
        // (e.g. a wedged GPU per issue #1025, or draining for shutdown). The
        // gateway must stop advertising it as healthy even though heartbeats
        // keep arriving, so cluster-status and routing reflect reality.
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Healthy
        );

        let became = reg.update_worker("http://w1:8080", make_msg(false)).await;
        assert!(!became);
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Unhealthy
        );

        // A second (and every subsequent) not-ready heartbeat must keep the
        // worker Unhealthy rather than flipping it to Unknown, so cluster-status
        // keeps showing a persistently wedged worker as regressed (#1025).
        let became = reg.update_worker("http://w1:8080", make_msg(false)).await;
        assert!(!became);
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Unhealthy
        );
    }

    #[tokio::test]
    async fn test_update_worker_already_healthy_not_became() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        let became = reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert!(!became); // was already healthy
    }

    #[tokio::test]
    async fn test_update_worker_defaults_empty_name() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.name = String::new();
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.name, "http://w1:8080");
    }

    #[tokio::test]
    async fn test_update_worker_defaults_gpu_count_zero() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.gpu_count = 0;
        reg.update_worker("http://w1:8080", msg).await;

        assert_eq!(reg.workers().await["http://w1:8080"].gpu_count, 1);
    }

    #[tokio::test]
    async fn test_update_worker_defaults_empty_bundle() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.bundle = String::new();
        reg.update_worker("http://w1:8080", msg).await;

        assert_eq!(reg.workers().await["http://w1:8080"].bundle, "default");
    }

    #[tokio::test]
    async fn test_update_worker_aggregates_queue_depth() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.models = vec![
            ModelStatus { queue_depth: 3 },
            ModelStatus { queue_depth: 5 },
        ];
        reg.update_worker("http://w1:8080", msg).await;

        assert_eq!(reg.workers().await["http://w1:8080"].queue_depth, 8);
    }

    #[tokio::test]
    async fn test_update_worker_aggregates_gpu_memory() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.gpus = vec![
            GpuStatus {
                memory_used_bytes: 100,
                memory_total_bytes: 500,
            },
            GpuStatus {
                memory_used_bytes: 200,
                memory_total_bytes: 500,
            },
        ];
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.memory_used_bytes, 300);
        assert_eq!(w.memory_total_bytes, 1000);
    }

    #[tokio::test]
    async fn test_update_worker_compact_field_fallback() {
        let reg = registry();
        let msg = WorkerStatusMessage {
            supports_execution_authority_v1: false,
            name: "w-compact".into(),
            ready: true,
            gpu_count: 1,
            total_gpu_slots: Some(1),
            ready_gpu_slots: Some(1),
            machine_profile: "l4".into(),
            pool_name: String::new(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            loaded_models: vec![],
            models: vec![], // empty — should use compact fallback
            gpus: vec![],   // empty — should use compact fallback
            queue_depth: Some(7),
            pending_cost: Some(42),
            inflight_batches: Some(3),
            memory_used_bytes: Some(2000),
            memory_total_bytes: Some(8000),
            saturated: false,
            terminated: false,
            unsupported_models: Vec::new(),
        };
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.queue_depth, 7);
        assert_eq!(w.ready_gpu_slots, 1);
        assert_eq!(w.pending_cost, 42);
        assert_eq!(w.inflight_batches, 3);
        assert_eq!(w.memory_used_bytes, 2000);
        assert_eq!(w.memory_total_bytes, 8000);
    }

    #[tokio::test]
    async fn test_update_worker_compact_fields_none_defaults_to_zero() {
        let reg = registry();
        let msg = WorkerStatusMessage {
            supports_execution_authority_v1: false,
            name: "w-none".into(),
            ready: true,
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: "l4".into(),
            pool_name: String::new(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            loaded_models: vec![],
            models: vec![],
            gpus: vec![],
            queue_depth: None,
            pending_cost: None,
            inflight_batches: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated: false,
            terminated: false,
            unsupported_models: Vec::new(),
        };
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.queue_depth, 0);
        assert_eq!(w.ready_gpu_slots, 1);
        assert_eq!(w.pending_cost, 0);
        assert_eq!(w.inflight_batches, 0);
        assert_eq!(w.memory_used_bytes, 0);
        assert_eq!(w.memory_total_bytes, 0);
    }

    #[tokio::test]
    async fn test_update_worker_callback_on_healthy() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();
        let reg = WorkerRegistry::new(
            Duration::from_secs(30),
            Some(Arc::new(move |_w: &WorkerState| {
                called_clone.store(true, Ordering::SeqCst);
            })),
        );

        reg.update_worker("http://w1:8080", make_msg(true)).await;
        assert!(called.load(Ordering::SeqCst));
    }

    // ── healthy_workers ────────────────────────────────────────────

    #[tokio::test]
    async fn test_healthy_workers_filters() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        reg.update_worker("http://w2:8080", make_msg(false)).await;

        let healthy = reg.healthy_workers().await;
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].url, "http://w1:8080");
    }

    // ── remove_worker ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_remove_worker() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        reg.remove_worker("http://w1:8080").await;
        assert!(reg.workers().await.is_empty());
    }

    // ── mark_unhealthy ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_mark_unhealthy() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        reg.mark_unhealthy("http://w1:8080").await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.health, WorkerHealth::Unhealthy);
    }

    #[tokio::test]
    async fn test_mark_unhealthy_nonexistent_is_noop() {
        let reg = registry();
        reg.mark_unhealthy("http://nonexistent:8080").await;
        assert!(reg.workers().await.is_empty());
    }

    #[tokio::test]
    async fn verified_worker_selection_requires_fresh_positive_authority_and_admission() {
        let reg = registry();
        let mut good = make_msg(true);
        good.pool_name = "tenant".into();
        good.supports_execution_authority_v1 = true;
        let pick = |hash: &str, names: Option<&HashSet<String>>| {
            reg.execution_authority_worker(
                "BAAI/bge-m3",
                "tenant",
                "l4-spot",
                "default",
                hash,
                names,
            )
        };
        reg.update_worker("w1", good.clone()).await;
        assert_eq!(pick("abc123", None).as_deref(), Some("worker-1"));
        assert!(pick("", None).is_none());
        assert!(pick("different", None).is_none());
        assert!(pick("abc123", Some(&HashSet::new())).is_none());
        let admitted = HashSet::from(["worker-1".to_string()]);
        assert!(pick("abc123", Some(&admitted)).is_some());
        for case in 0..7 {
            let mut bad = good.clone();
            match case {
                0 => bad.supports_execution_authority_v1 = false,
                1 => bad.ready = false,
                2 => bad.saturated = true,
                3 => bad.ready_gpu_slots = Some(0),
                4 => bad.unsupported_models = vec!["BAAI/bge-m3".into()],
                5 => bad.name = "worker.1".into(),
                _ => bad.bundle_config_hash.clear(),
            }
            reg.update_worker("w1", bad).await;
            assert!(pick("abc123", None).is_none(), "case {case}");
        }
        reg.update_worker("w1", good.clone()).await;
        let mut duplicate = good;
        duplicate.pool_name = "another-tenant".into();
        reg.update_worker("w2", duplicate).await;
        assert!(pick("abc123", None).is_none(), "duplicate worker identity");
    }

    #[tokio::test]
    async fn verified_selection_uses_pressure_and_distributes_equal_pressure() {
        let reg = registry();
        for (name, cost) in [("hot", 100), ("idle-a", 0), ("idle-b", 0)] {
            let mut msg = make_msg(true);
            msg.name = name.into();
            msg.pool_name = "tenant".into();
            msg.supports_execution_authority_v1 = true;
            msg.pending_cost = Some(cost);
            reg.update_worker(name, msg).await;
        }
        let selected: HashSet<_> = (0..128)
            .map(|_| {
                reg.execution_authority_worker(
                    "BAAI/bge-m3",
                    "tenant",
                    "l4-spot",
                    "default",
                    "abc123",
                    None,
                )
                .unwrap()
            })
            .collect();
        assert_eq!(
            selected,
            HashSet::from(["idle-a".to_string(), "idle-b".to_string()])
        );
    }

    #[tokio::test]
    async fn verified_selection_refuses_stale_snapshot_before_heartbeat_sweep() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "tenant".into();
        msg.supports_execution_authority_v1 = true;
        reg.update_worker("w1", msg).await;
        let mut workers = reg.workers.write().await;
        workers.get_mut("w1").unwrap().last_heartbeat =
            Instant::now() - reg.heartbeat_timeout - Duration::from_secs(1);
        reg.rebuild_snapshot(&workers);
        drop(workers);
        assert!(reg
            .execution_authority_worker(
                "BAAI/bge-m3",
                "tenant",
                "l4-spot",
                "default",
                "abc123",
                None
            )
            .is_none());
    }

    // ── check_heartbeats ───────────────────────────────────────────

    #[tokio::test]
    async fn test_check_heartbeats_healthy_within_timeout() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;

        let sweep = reg.check_heartbeats().await;
        assert!(sweep.unhealthy.is_empty());
        assert!(sweep.evicted.is_empty());
    }

    #[tokio::test]
    async fn test_check_heartbeats_expired() {
        let reg = WorkerRegistry::new(Duration::from_millis(1), None);
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        {
            let mut workers = reg.workers.write().await;
            workers.get_mut("http://w1:8080").unwrap().last_heartbeat =
                Instant::now() - Duration::from_millis(10);
        }

        let sweep = reg.check_heartbeats().await;
        assert_eq!(sweep.unhealthy, vec!["http://w1:8080"]);
        assert!(sweep.evicted.is_empty());
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Unhealthy
        );
    }

    #[tokio::test]
    async fn test_check_heartbeats_evicts_stale_worker() {
        let reg = WorkerRegistry::new(Duration::from_millis(1), None);
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        {
            let mut workers = reg.workers.write().await;
            workers.get_mut("http://w1:8080").unwrap().last_heartbeat =
                Instant::now() - Duration::from_millis(25);
        }

        let sweep = reg.check_heartbeats().await;
        assert!(sweep.unhealthy.is_empty());
        assert_eq!(sweep.evicted, vec!["http://w1:8080"]);
        assert!(reg.workers().await.is_empty());
        assert!(reg.healthy_workers().await.is_empty());
    }

    #[tokio::test]
    async fn test_check_heartbeats_keeps_not_ready_worker_with_fresh_heartbeat() {
        let reg = WorkerRegistry::new(Duration::from_millis(1), None);
        reg.update_worker("http://w1:8080", make_msg(true)).await;
        reg.update_worker("http://w1:8080", make_msg(false)).await;

        let sweep = reg.check_heartbeats().await;
        assert!(sweep.unhealthy.is_empty());
        assert!(sweep.evicted.is_empty());
        assert_eq!(
            reg.workers().await["http://w1:8080"].health,
            WorkerHealth::Unhealthy
        );
    }

    // ── select_worker ──────────────────────────────────────────────

    async fn setup_workers(reg: &WorkerRegistry) {
        let mut msg1 = make_msg(true);
        msg1.name = "w1".into();
        msg1.machine_profile = "l4-spot".into();
        msg1.bundle = "default".into();
        msg1.loaded_models = vec!["BAAI/bge-m3".into()];
        msg1.models = vec![ModelStatus { queue_depth: 5 }];
        reg.update_worker("http://w1:8080", msg1).await;

        let mut msg2 = make_msg(true);
        msg2.name = "w2".into();
        msg2.machine_profile = "a100".into();
        msg2.bundle = "premium".into();
        msg2.loaded_models = vec!["BAAI/bge-m3".into(), "openai/clip".into()];
        msg2.models = vec![ModelStatus { queue_depth: 2 }];
        reg.update_worker("http://w2:8080", msg2).await;

        let mut msg3 = make_msg(true);
        msg3.name = "w3".into();
        msg3.machine_profile = "l4-spot".into();
        msg3.bundle = "default".into();
        msg3.loaded_models = vec!["openai/clip".into()];
        msg3.models = vec![ModelStatus { queue_depth: 1 }];
        reg.update_worker("http://w3:8080", msg3).await;
    }

    // ── dispatch_workers_for / profile-variant routing ─────────────

    #[tokio::test]
    async fn test_dispatch_workers_for_profile_variant_requires_variant_worker() {
        let reg = registry();
        // A worker advertising only the *base* model name in a named pool.
        let mut msg = make_msg(true);
        msg.name = "w1".into();
        msg.pool_name = "default".into();
        msg.loaded_models = vec!["BAAI/bge-m3".into()];
        reg.update_worker("http://w1:8080", msg).await;

        // A `:profile` request must not direct-dispatch to the base-model
        // worker. The profile variant carries its own launch args.
        let direct =
            reg.dispatch_workers_for("BAAI/bge-m3:fast", "default", "l4-spot", "default", "");
        assert!(
            direct.is_empty(),
            "profile variant must wait for/load a worker with that exact variant"
        );

        // The bare base id still works (regression guard).
        let base = reg.dispatch_workers_for("BAAI/bge-m3", "default", "l4-spot", "default", "");
        assert_eq!(base.len(), 1);

        // The ring snapshot (built from the same filter) stays empty
        // until the concrete variant is reported loaded.
        let ring = reg.ring_snapshot_for("BAAI/bge-m3:fast", "default", "l4-spot", "default", "");
        assert_eq!(ring.len(), 0);

        let mut msg = make_msg(true);
        msg.name = "w1".into();
        msg.pool_name = "default".into();
        msg.loaded_models = vec!["BAAI/bge-m3:fast".into()];
        reg.update_worker("http://w1:8080", msg).await;

        let direct =
            reg.dispatch_workers_for("BAAI/bge-m3:fast", "default", "l4-spot", "default", "");
        assert_eq!(
            direct.len(),
            1,
            "loaded profile variant should match exactly"
        );
        assert_eq!(direct[0].name, "w1");

        let ring = reg.ring_snapshot_for("BAAI/bge-m3:fast", "default", "l4-spot", "default", "");
        assert_eq!(ring.len(), 1);

        // An unknown base id still finds nothing.
        assert!(reg
            .dispatch_workers_for("unknown/model:fast", "default", "l4-spot", "default", "")
            .is_empty());
    }

    #[tokio::test]
    async fn test_dispatch_workers_for_scopes_by_bundle() {
        // Two workers, same model/pool/profile but DIFFERENT bundles. The
        // by_bundle-indexed lookup (#1607) must stay equivalent to the old
        // `w.bundle` predicate: dispatch for one bundle returns only that
        // bundle's worker, and an empty bundle bucket returns nothing.
        let reg = registry();

        let mut a = make_msg(true);
        a.name = "wa".into();
        a.pool_name = "default".into();
        a.bundle = "bundle-a".into();
        reg.update_worker("http://wa:8080", a).await;

        let mut b = make_msg(true);
        b.name = "wb".into();
        b.pool_name = "default".into();
        b.bundle = "bundle-b".into();
        reg.update_worker("http://wb:8080", b).await;

        let a_only = reg.dispatch_workers_for("BAAI/bge-m3", "default", "l4-spot", "bundle-a", "");
        assert_eq!(a_only.len(), 1);
        assert_eq!(a_only[0].name, "wa");

        let b_only = reg.dispatch_workers_for("BAAI/bge-m3", "default", "l4-spot", "bundle-b", "");
        assert_eq!(b_only.len(), 1);
        assert_eq!(b_only[0].name, "wb");

        // A bundle with no workers hits the by_bundle miss path → empty.
        assert!(reg
            .dispatch_workers_for("BAAI/bge-m3", "default", "l4-spot", "bundle-c", "")
            .is_empty());
    }

    // ── ring_snapshot_for dedup ────────────────────────────────────

    #[tokio::test]
    async fn test_ring_snapshot_dedups_duplicate_worker_names() {
        let reg = registry();
        // Two distinct workers (different urls) advertising the SAME
        // operator-set name in the same (model, pool).
        let mut a = make_msg(true);
        a.name = "dup".into();
        a.pool_name = "default".into();
        a.loaded_models = vec!["BAAI/bge-m3".into()];
        reg.update_worker("http://w-z:8080", a).await;

        let mut b = make_msg(true);
        b.name = "dup".into();
        b.pool_name = "default".into();
        b.loaded_models = vec!["BAAI/bge-m3".into()];
        reg.update_worker("http://w-a:8080", b).await;

        // Both are dispatch-eligible…
        assert_eq!(
            reg.dispatch_workers_for("BAAI/bge-m3", "default", "l4-spot", "default", "")
                .len(),
            2
        );
        // …but the ring collapses them to a single deterministic entry so
        // the HRW pick is unambiguous and only one per-worker subject is
        // targeted.
        let ring = reg.ring_snapshot_for("BAAI/bge-m3", "default", "l4-spot", "default", "");
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.entries[0].worker_id, "dup");
    }

    #[tokio::test]
    async fn test_ring_snapshot_keeps_distinct_names() {
        let reg = registry();
        let mut a = make_msg(true);
        a.name = "w-a".into();
        a.pool_name = "default".into();
        a.loaded_models = vec!["BAAI/bge-m3".into()];
        reg.update_worker("http://w1:8080", a).await;

        let mut b = make_msg(true);
        b.name = "w-b".into();
        b.pool_name = "default".into();
        b.loaded_models = vec!["BAAI/bge-m3".into()];
        reg.update_worker("http://w2:8080", b).await;

        let ring = reg.ring_snapshot_for("BAAI/bge-m3", "default", "l4-spot", "default", "");
        assert_eq!(ring.len(), 2);
    }

    // ── get_models ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_models() {
        let reg = registry();
        setup_workers(&reg).await;

        let models = reg.get_models().await;
        assert!(models.contains_key("BAAI/bge-m3"));
        assert!(models.contains_key("openai/clip"));

        // bge-m3 is on w1 and w2
        assert_eq!(models["BAAI/bge-m3"].len(), 2);
        // clip is on w2 and w3
        assert_eq!(models["openai/clip"].len(), 2);
    }

    #[tokio::test]
    async fn test_get_models_excludes_unhealthy() {
        let reg = registry();
        setup_workers(&reg).await;
        reg.mark_unhealthy("http://w1:8080").await;

        let models = reg.get_models().await;
        // bge-m3 should only be on w2 now
        assert_eq!(models["BAAI/bge-m3"].len(), 1);
    }

    // ── get_gpu_types ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_gpu_types() {
        let reg = registry();
        setup_workers(&reg).await;

        let mut gpu_types = reg.get_gpu_types().await;
        gpu_types.sort();
        assert_eq!(gpu_types, vec!["a100", "l4-spot"]);
    }

    #[tokio::test]
    async fn test_get_gpu_types_excludes_unhealthy() {
        let reg = registry();
        setup_workers(&reg).await;
        reg.mark_unhealthy("http://w2:8080").await;

        let gpu_types = reg.get_gpu_types().await;
        assert_eq!(gpu_types, vec!["l4-spot"]);
    }

    // ── record_request ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_record_request_bumps_counter() {
        let reg = registry();
        reg.record_request("queue").await;
        reg.record_request("queue").await;
        reg.record_request("queue").await;

        assert_eq!(reg.request_count.load(Ordering::Relaxed), 3);
    }

    // ── get_cluster_status ─────────────────────────────────────────

    #[tokio::test]
    async fn test_get_cluster_status_empty() {
        let reg = registry();
        let status = reg.get_cluster_status().await;
        assert_eq!(status.worker_count, 0);
        assert_eq!(status.gpu_count, 0);
        assert_eq!(status.models_loaded, 0);
    }

    #[tokio::test]
    async fn test_get_cluster_status_with_workers() {
        let reg = registry();
        setup_workers(&reg).await;

        let status = reg.get_cluster_status().await;
        assert_eq!(status.worker_count, 3);
        assert_eq!(status.gpu_count, 3); // 1 + 1 + 1
        assert_eq!(status.models_loaded, 2); // bge-m3 and clip
        assert_eq!(status.workers.len(), 3);
        assert!(status.timestamp > 0.0);
    }

    #[tokio::test]
    async fn test_get_cluster_status_counts_only_healthy() {
        let reg = registry();
        setup_workers(&reg).await;
        reg.mark_unhealthy("http://w1:8080").await;

        let status = reg.get_cluster_status().await;
        assert_eq!(status.worker_count, 2);
        // Workers list still contains all workers (for visibility)
        assert_eq!(status.workers.len(), 3);
    }

    // ── resolve_queue_pool_matching ───────────────────────────────

    #[tokio::test]
    async fn test_resolve_queue_pool_found() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        msg.machine_profile = "l4-spot".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("default", "l4-spot", "").await;
        assert_eq!(pool, Some("pool-a".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_canonicalizes_worker_pool_case() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "Customer-Acme".into();
        msg.bundle = "default".into();
        msg.machine_profile = "l4-spot".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg
            .resolve_queue_pool_in_pool("default", "l4-spot", "customer-acme", "")
            .await;
        assert_eq!(pool, Some("customer-acme".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_in_pool_filters_pool_name() {
        let reg = registry();

        let mut default_msg = make_msg(true);
        default_msg.pool_name = "default".into();
        default_msg.bundle = "default".into();
        default_msg.machine_profile = "l4-spot".into();
        reg.update_worker("http://w1:8080", default_msg).await;

        let mut isolated_msg = make_msg(true);
        isolated_msg.name = "worker-2".into();
        isolated_msg.pool_name = "eval-l4".into();
        isolated_msg.bundle = "default".into();
        isolated_msg.machine_profile = "l4-spot".into();
        reg.update_worker("http://w2:8080", isolated_msg).await;

        let pool = reg
            .resolve_queue_pool_in_pool("default", "l4-spot", "eval-l4", "")
            .await;
        assert_eq!(pool, Some("eval-l4".to_string()));

        let missing = reg
            .resolve_queue_pool_in_pool("default", "l4-spot", "missing", "")
            .await;
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_bundle_only() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-b".into();
        msg.bundle = "premium".into();
        reg.update_worker("http://w1:8080", msg).await;

        // No GPU filter
        let pool = reg.resolve_queue_pool("premium", "", "").await;
        assert_eq!(pool, Some("pool-b".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_no_match() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("premium", "", "").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_skips_no_pool_name() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = String::new();
        msg.bundle = "default".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("default", "", "").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_skips_unhealthy() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        reg.update_worker("http://w1:8080", msg).await;
        reg.mark_unhealthy("http://w1:8080").await;

        let pool = reg.resolve_queue_pool("default", "", "").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_requires_matching_config_hash_when_expected() {
        let reg = registry();
        let mut stale = make_msg(true);
        stale.name = "stale".into();
        stale.pool_name = "old-pool".into();
        stale.bundle = "default".into();
        stale.bundle_config_hash = "old".into();
        reg.update_worker("http://w1:8080", stale).await;

        let mut current = make_msg(true);
        current.name = "current".into();
        current.pool_name = "current-pool".into();
        current.bundle = "default".into();
        current.bundle_config_hash = "new".into();
        reg.update_worker("http://w2:8080", current).await;

        let pool = reg.resolve_queue_pool("default", "", "new").await;
        assert_eq!(pool, Some("current-pool".to_string()));

        let missing = reg.resolve_queue_pool("default", "", "missing").await;
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_empty_expected_hash_keeps_legacy_match() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        msg.bundle_config_hash = String::new();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("default", "", "").await;
        assert_eq!(pool, Some("pool-a".to_string()));
    }

    // ── dispatch_workers_for / ring_snapshot_for ───────────────────

    async fn make_dispatch_msg(
        ready: bool,
        saturated: bool,
        pool: &str,
        models: &[&str],
    ) -> WorkerStatusMessage {
        WorkerStatusMessage {
            supports_execution_authority_v1: false,
            name: "w".into(),
            ready,
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: "l4".into(),
            pool_name: pool.into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            loaded_models: models.iter().map(|s| (*s).into()).collect(),
            models: vec![],
            gpus: vec![],
            queue_depth: None,
            pending_cost: None,
            inflight_batches: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated,
            terminated: false,
            unsupported_models: Vec::new(),
        }
    }

    #[tokio::test]
    async fn test_dispatch_workers_for_filters_by_lane_model_and_saturation() {
        let reg = registry();
        let mut a = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        a.name = "worker-a".into();
        reg.update_worker("http://a:8080", a).await;

        let mut b = make_dispatch_msg(true, true, "p1", &["BAAI/bge-m3"]).await;
        b.name = "worker-b".into();
        reg.update_worker("http://b:8080", b).await;

        let mut c = make_dispatch_msg(true, false, "p2", &["BAAI/bge-m3"]).await;
        c.name = "worker-c".into();
        reg.update_worker("http://c:8080", c).await;

        let mut d = make_dispatch_msg(true, false, "p1", &["other/model"]).await;
        d.name = "worker-d".into();
        reg.update_worker("http://d:8080", d).await;

        let mut e = make_dispatch_msg(false, false, "p1", &["BAAI/bge-m3"]).await;
        e.name = "worker-e".into();
        reg.update_worker("http://e:8080", e).await;

        let mut f = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        f.name = "worker-f".into();
        f.machine_profile = "a100".into();
        reg.update_worker("http://f:8080", f).await;

        let mut g = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        g.name = "worker-g".into();
        g.bundle = "sglang".into();
        reg.update_worker("http://g:8080", g).await;

        let elig = reg.dispatch_workers_for("BAAI/bge-m3", "p1", "l4", "default", "");
        let names: Vec<_> = elig.iter().map(|w| w.name.clone()).collect();
        assert_eq!(names, vec!["worker-a".to_string()]);
    }

    #[tokio::test]
    async fn test_dispatch_workers_for_filters_stale_bundle_hash() {
        let reg = registry();
        let mut current = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        current.name = "current".into();
        current.bundle_config_hash = "h1".into();
        reg.update_worker("http://current:8080", current).await;

        let mut stale = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        stale.name = "stale".into();
        stale.bundle_config_hash = "old".into();
        reg.update_worker("http://stale:8080", stale).await;

        let ring = reg.ring_snapshot_for("BAAI/bge-m3", "p1", "l4", "default", "h1");
        let ids: Vec<_> = ring.entries.iter().map(|e| e.worker_id.clone()).collect();
        assert_eq!(ids, vec!["current".to_string()]);
    }

    #[tokio::test]
    async fn test_pool_fallback_lane_worker_count_includes_cold_workers() {
        let reg = registry();
        let mut warm = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        warm.name = "warm".into();
        reg.update_worker("http://warm:8080", warm).await;

        let mut cold = make_dispatch_msg(true, false, "p1", &[]).await;
        cold.name = "cold".into();
        reg.update_worker("http://cold:8080", cold).await;

        assert_eq!(
            reg.dispatch_workers_for("BAAI/bge-m3", "p1", "l4", "default", "")
                .len(),
            1
        );
        assert_eq!(
            reg.pool_fallback_lane_worker_count("p1", "l4", "default", "", None),
            2
        );
    }

    #[tokio::test]
    async fn test_pool_fallback_lane_worker_count_filters_unusable_workers() {
        let reg = registry();
        let mut active = make_dispatch_msg(true, false, "p1", &[]).await;
        active.name = "active".into();
        active.bundle_config_hash = "h1".into();
        reg.update_worker("http://active:8080", active).await;

        let mut saturated = make_dispatch_msg(true, true, "p1", &[]).await;
        saturated.name = "saturated".into();
        saturated.bundle_config_hash = "h1".into();
        reg.update_worker("http://saturated:8080", saturated).await;

        let mut stale_hash = make_dispatch_msg(true, false, "p1", &[]).await;
        stale_hash.name = "stale-hash".into();
        stale_hash.bundle_config_hash = "old".into();
        reg.update_worker("http://stale-hash:8080", stale_hash)
            .await;

        let mut other_bundle = make_dispatch_msg(true, false, "p1", &[]).await;
        other_bundle.name = "other-bundle".into();
        other_bundle.bundle = "sglang".into();
        other_bundle.bundle_config_hash = "h1".into();
        reg.update_worker("http://other-bundle:8080", other_bundle)
            .await;

        assert_eq!(
            reg.pool_fallback_lane_worker_count("p1", "l4", "default", "h1", None),
            1
        );
    }

    #[tokio::test]
    async fn test_pool_fallback_lane_worker_count_respects_admitted_workers() {
        let reg = registry();
        let mut assigned = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        assigned.name = "assigned".into();
        assigned.bundle_config_hash = "h1".into();
        reg.update_worker("http://assigned:8080", assigned).await;

        let mut paused = make_dispatch_msg(true, false, "p1", &["BAAI/bge-m3"]).await;
        paused.name = "paused".into();
        paused.bundle_config_hash = "h1".into();
        reg.update_worker("http://paused:8080", paused).await;

        let admitted = HashSet::from(["assigned".to_string()]);

        assert_eq!(
            reg.pool_fallback_lane_worker_count("p1", "l4", "default", "h1", None),
            2
        );
        assert_eq!(
            reg.pool_fallback_lane_worker_count("p1", "l4", "default", "h1", Some(&admitted)),
            1
        );

        let ring = reg.ring_snapshot_for_admitted(
            "BAAI/bge-m3",
            "p1",
            "l4",
            "default",
            "h1",
            Some(&admitted),
        );
        assert_eq!(ring.len(), 1);
        assert_eq!(ring.entries[0].worker_id, "assigned");
    }

    #[tokio::test]
    async fn test_lazy_lane_ring_snapshot_for_admitted_includes_cold_assigned_worker() {
        let reg = registry();
        let mut assigned = make_dispatch_msg(true, false, "p1", &[]).await;
        assigned.name = "assigned-cold".into();
        assigned.bundle_config_hash = "h1".into();
        reg.update_worker("http://assigned-cold:8080", assigned)
            .await;

        let mut unassigned = make_dispatch_msg(true, false, "p1", &[]).await;
        unassigned.name = "unassigned-cold".into();
        unassigned.bundle_config_hash = "h1".into();
        reg.update_worker("http://unassigned-cold:8080", unassigned)
            .await;

        let admitted = HashSet::from(["assigned-cold".to_string()]);

        let loaded_ring = reg.ring_snapshot_for_admitted(
            "BAAI/bge-m3",
            "p1",
            "l4",
            "default",
            "h1",
            Some(&admitted),
        );
        assert_eq!(loaded_ring.len(), 0);

        let lazy_ring = reg.lazy_lane_ring_snapshot_for_admitted(
            "BAAI/bge-m3",
            "p1",
            "l4",
            "default",
            "h1",
            &admitted,
        );
        assert_eq!(lazy_ring.len(), 1);
        assert_eq!(lazy_ring.entries[0].worker_id, "assigned-cold");
    }

    #[tokio::test]
    async fn test_ring_snapshot_for_contains_eligible_only() {
        let reg = registry();
        for (i, sat) in [false, true, false].iter().enumerate() {
            let mut m = make_dispatch_msg(true, *sat, "p1", &["m"]).await;
            m.name = format!("w-{i}");
            reg.update_worker(&format!("http://h{i}:8080"), m).await;
        }
        let snap = reg.ring_snapshot_for("m", "p1", "l4", "default", "");
        let ids: Vec<_> = snap.entries.iter().map(|e| e.worker_id.clone()).collect();
        assert!(ids.contains(&"w-0".to_string()));
        assert!(!ids.contains(&"w-1".to_string()));
        assert!(ids.contains(&"w-2".to_string()));
    }

    #[tokio::test]
    async fn test_ring_snapshot_carries_worker_pressure() {
        let reg = registry();
        let mut msg = make_dispatch_msg(true, false, "p1", &["m"]).await;
        msg.name = "pressured".into();
        msg.total_gpu_slots = Some(4);
        msg.ready_gpu_slots = Some(3);
        msg.queue_depth = Some(9);
        msg.pending_cost = Some(90);
        msg.inflight_batches = Some(6);
        reg.update_worker("http://pressured:8080", msg).await;

        let snap = reg.ring_snapshot_for("m", "p1", "l4", "default", "");

        assert_eq!(snap.len(), 1);
        assert_eq!(snap.entries[0].worker_id, "pressured");
        assert_eq!(snap.entries[0].ready_gpu_slots, 3);
        assert_eq!(snap.entries[0].queue_depth, 9);
        assert_eq!(snap.entries[0].pending_cost, 90);
        assert_eq!(snap.entries[0].inflight_batches, 6);
    }

    #[tokio::test]
    async fn test_degrade_callback_fires_on_saturation_rising_edge() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let count = Arc::new(AtomicUsize::new(0));
        let count_cb = count.clone();
        let reg = WorkerRegistry::with_callbacks(
            Duration::from_secs(30),
            None,
            Some(Arc::new(move |_w: &WorkerState| {
                count_cb.fetch_add(1, Ordering::SeqCst);
            })),
        );

        let msg = make_dispatch_msg(true, false, "p", &["m"]).await;
        reg.update_worker("http://w:8080", msg).await;
        assert_eq!(count.load(Ordering::SeqCst), 0);

        let mut sat = make_dispatch_msg(true, true, "p", &["m"]).await;
        sat.name = "w".into();
        reg.update_worker("http://w:8080", sat).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        // Stays saturated: no re-fire.
        let mut sat2 = make_dispatch_msg(true, true, "p", &["m"]).await;
        sat2.name = "w".into();
        reg.update_worker("http://w:8080", sat2).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        // Drops back to not-saturated: still no degrade fire (only rising edge).
        let mut clear = make_dispatch_msg(true, false, "p", &["m"]).await;
        clear.name = "w".into();
        reg.update_worker("http://w:8080", clear).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_degrade_callback_fires_on_mark_unhealthy() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let count = Arc::new(AtomicUsize::new(0));
        let count_cb = count.clone();
        let reg = WorkerRegistry::with_callbacks(
            Duration::from_secs(30),
            None,
            Some(Arc::new(move |_w: &WorkerState| {
                count_cb.fetch_add(1, Ordering::SeqCst);
            })),
        );

        reg.update_worker(
            "http://w:8080",
            make_dispatch_msg(true, false, "p", &["m"]).await,
        )
        .await;
        reg.mark_unhealthy("http://w:8080").await;
        assert_eq!(count.load(Ordering::SeqCst), 1);

        // Marking an already-unhealthy worker is a no-op (no double-fire).
        reg.mark_unhealthy("http://w:8080").await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    // ── per-model support within one bundle config hash ────────────

    /// Laya and GLiFormer route ids that an earlier default-bundle image
    /// reports as unsupported once the control plane adds their adapters.
    const ADDED_FAMILY_ROUTES: [&str; 5] = [
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
        "knowledgator/gliformer-base-v1",
        "knowledgator/gliformer-large-v1",
    ];

    async fn lane_worker(
        reg: &WorkerRegistry,
        name: &str,
        machine_profile: &str,
        hash: &str,
        loaded: &[&str],
        unsupported: &[&str],
    ) {
        let mut msg = make_dispatch_msg(true, false, "default", loaded).await;
        msg.name = name.into();
        msg.machine_profile = machine_profile.into();
        msg.bundle_config_hash = hash.into();
        msg.unsupported_models = unsupported.iter().map(|m| (*m).into()).collect();
        reg.update_worker(&format!("http://{name}:8080"), msg).await;
    }

    fn route(reg: &WorkerRegistry, gpu: &str, hash: &str, model: &str) -> QueueRouteLookup {
        reg.resolve_queue_route_for_model("default", gpu, "default", hash, model)
    }

    #[tokio::test]
    async fn test_earlier_image_keeps_other_models_routable_when_the_bundle_gains_adapters() {
        let reg = registry();
        lane_worker(&reg, "old-0", "l4", "h-083", &[], &ADDED_FAMILY_ROUTES).await;

        for model in ["BAAI/bge-m3", "urchade/gliner_multi-v2.1"] {
            let lookup = route(&reg, "", "h-083", model);
            assert_eq!(
                lookup.route.map(|r| r.machine_profile).as_deref(),
                Some("l4"),
                "{model} must stay routable on the earlier image"
            );
            assert!(!lookup.excluded_unsupported);
        }
        for model in ADDED_FAMILY_ROUTES {
            let lookup = route(&reg, "", "h-083", model);
            assert!(
                lookup.route.is_none(),
                "{model} must not reach the earlier image"
            );
            assert!(lookup.excluded_unsupported);
        }
        assert!(route(&reg, "", "h-083", "Convaiinnovations/LAYA")
            .route
            .is_none());

        // The same worker before this change hashed with its image's adapter
        // list, so it advertised a different hash and served nothing.
        let before = registry();
        lane_worker(&before, "old-0", "l4", "h-082-image-scope", &[], &[]).await;
        assert!(route(&before, "", "h-083", "BAAI/bge-m3").route.is_none());
    }

    #[tokio::test]
    async fn test_mixed_lane_is_not_eligible_for_a_model_one_worker_cannot_serve() {
        let reg = registry();
        lane_worker(&reg, "old-0", "l4", "h1", &[], &["org/new-family"]).await;
        lane_worker(&reg, "new-0", "l4", "h1", &[], &[]).await;
        lane_worker(&reg, "new-1", "a100", "h1", &[], &[]).await;

        let lookup = route(&reg, "", "h1", "org/new-family");
        assert_eq!(
            lookup.route.map(|r| r.machine_profile).as_deref(),
            Some("a100")
        );
        assert!(lookup.excluded_unsupported);
        assert!(route(&reg, "l4", "h1", "org/new-family").route.is_none());
        assert!(route(&reg, "l4", "h1", "org/kept").route.is_some());
    }

    #[tokio::test]
    async fn test_workers_without_the_field_or_the_hash_do_not_block_a_model() {
        let reg = registry();
        // A worker that predates the field serves everything its hash covers.
        lane_worker(&reg, "older", "l4", "h1", &[], &[]).await;
        // A worker on another hash does not consume this hash's work.
        lane_worker(&reg, "stale", "l4", "h0", &[], &["org/new-family"]).await;

        let lookup = route(&reg, "l4", "h1", "org/new-family");
        assert!(lookup.route.is_some());
        assert!(!lookup.excluded_unsupported);
    }

    #[tokio::test]
    async fn test_worker_direct_rings_exclude_workers_that_cannot_serve_the_model() {
        let reg = registry();
        lane_worker(
            &reg,
            "old-0",
            "l4",
            "h1",
            &["org/new-family"],
            &["org/new-family"],
        )
        .await;
        lane_worker(&reg, "new-0", "l4", "h1", &["org/new-family"], &[]).await;

        let direct: Vec<String> = reg
            .dispatch_workers_for("org/new-family", "default", "l4", "default", "h1")
            .into_iter()
            .map(|w| w.name)
            .collect();
        assert_eq!(direct, ["new-0"]);

        let admitted: HashSet<String> = ["old-0".to_string(), "new-0".to_string()].into();
        let lazy = reg.lazy_lane_ring_snapshot_for_admitted(
            "org/new-family",
            "default",
            "l4",
            "default",
            "h1",
            &admitted,
        );
        assert_eq!(lazy.len(), 1);
        let kept = reg.lazy_lane_ring_snapshot_for_admitted(
            "org/kept", "default", "l4", "default", "h1", &admitted,
        );
        assert_eq!(kept.len(), 2);
    }

    #[tokio::test]
    async fn test_oversized_unsupported_list_fails_closed_and_is_capped() {
        let reg = registry();
        let oversized: Vec<String> = (0..=MAX_UNSUPPORTED_MODELS)
            .map(|i| format!("org/model-{i}"))
            .collect();
        let mut msg = make_dispatch_msg(true, false, "default", &["org/kept"]).await;
        msg.name = "flooded".into();
        msg.bundle_config_hash = "h1".into();
        msg.unsupported_models = oversized;
        reg.update_worker("http://flooded:8080", msg).await;

        let workers = reg.workers().await;
        let flooded = &workers["http://flooded:8080"];
        assert!(flooded.unsupported_overflow);
        assert_eq!(flooded.unsupported_models.len(), MAX_UNSUPPORTED_MODELS);
        assert!(!flooded.supports_model("org/kept"));
        assert!(route(&reg, "", "h1", "org/kept").route.is_none());
        let status = reg.get_cluster_status().await;
        assert!(status.workers[0].unsupported_models_overflow);

        lane_worker(&reg, "flooded", "l4", "h1", &["org/kept"], &["org/other"]).await;
        let recovered = reg.workers().await;
        let flooded = &recovered["http://flooded:8080"];
        assert!(!flooded.unsupported_overflow);
        assert!(flooded.supports_model("org/kept"));
    }

    #[tokio::test]
    async fn test_cluster_status_reports_unsupported_models() {
        let reg = registry();
        lane_worker(&reg, "old-0", "l4", "h1", &[], &["org/new-family"]).await;
        lane_worker(&reg, "new-0", "l4", "h1", &[], &[]).await;

        let status = reg.get_cluster_status().await;
        let by_name: HashMap<_, _> = status
            .workers
            .iter()
            .map(|w| (w.name.as_str(), w.unsupported_models.clone()))
            .collect();
        assert_eq!(by_name["old-0"], ["org/new-family"]);
        assert!(by_name["new-0"].is_empty());
        let json = serde_json::to_value(&status).unwrap();
        let workers = json["workers"].as_array().unwrap();
        assert_eq!(
            workers
                .iter()
                .filter(|w| w.get("unsupported_models").is_some())
                .count(),
            1,
            "the field is omitted for workers that serve every model"
        );
    }
}
