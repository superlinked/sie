use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use tokio::sync::RwLock;

use crate::types::{
    ClusterStatus, ModelInfo, WorkerHealth, WorkerInfo, WorkerState, WorkerStatusMessage,
};

pub type OnWorkerHealthy = Arc<dyn Fn(&WorkerState) + Send + Sync>;

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
    on_worker_healthy: Option<OnWorkerHealthy>,
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
    pub fn new(heartbeat_timeout: Duration, on_worker_healthy: Option<OnWorkerHealthy>) -> Self {
        Self {
            workers: RwLock::new(HashMap::new()),
            heartbeat_timeout,
            on_worker_healthy,
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
        let (became_healthy, worker_copy) = {
            let mut workers = self.workers.write().await;
            let exists = workers.contains_key(url);
            let w = workers
                .entry(url.to_string())
                .or_insert_with(|| WorkerState {
                    url: url.to_string(),
                    name: String::new(),
                    health: WorkerHealth::Unknown,
                    gpu_count: 1,
                    machine_profile: String::new(),
                    bundle: "default".to_string(),
                    bundle_config_hash: String::new(),
                    models: Vec::new(),
                    queue_depth: 0,
                    memory_used_bytes: 0,
                    memory_total_bytes: 0,
                    last_heartbeat: Instant::now(),
                    pool_name: String::new(),
                });

            let was_healthy = w.healthy();

            w.name = if msg.name.is_empty() {
                url.to_string()
            } else {
                msg.name.clone()
            };
            w.gpu_count = if msg.gpu_count == 0 { 1 } else { msg.gpu_count };
            w.bundle = if msg.bundle.is_empty() {
                "default".to_string()
            } else {
                msg.bundle.clone()
            };
            w.bundle_config_hash = msg.bundle_config_hash.clone();
            w.machine_profile = msg.machine_profile.clone();
            w.pool_name = msg.pool_name.clone();
            w.models = msg.loaded_models.clone();

            // Aggregate queue depth from models (fallback to compact top-level field)
            w.queue_depth = if !msg.models.is_empty() {
                msg.models.iter().map(|m| m.queue_depth).sum()
            } else {
                msg.queue_depth.unwrap_or(0)
            };

            // Aggregate GPU memory (fallback to compact top-level fields)
            if !msg.gpus.is_empty() {
                w.memory_used_bytes = msg.gpus.iter().map(|g| g.memory_used_bytes).sum();
                w.memory_total_bytes = msg.gpus.iter().map(|g| g.memory_total_bytes).sum();
            } else {
                w.memory_used_bytes = msg.memory_used_bytes.unwrap_or(0);
                w.memory_total_bytes = msg.memory_total_bytes.unwrap_or(0);
            }

            w.last_heartbeat = Instant::now();

            if msg.ready {
                w.health = WorkerHealth::Healthy;
            } else if w.health != WorkerHealth::Healthy {
                w.health = WorkerHealth::Unknown;
            }

            let became_healthy = msg.ready && (!exists || !was_healthy);
            let worker_copy = w.clone();
            // Rebuild snapshot inside write lock — lock-free ArcSwap, no deadlock
            self.rebuild_snapshot(&workers);
            (became_healthy, worker_copy)
        };

        if became_healthy {
            if let Some(cb) = &self.on_worker_healthy {
                cb(&worker_copy);
            }
        }

        became_healthy
    }

    pub async fn remove_worker(&self, url: &str) {
        let mut workers = self.workers.write().await;
        workers.remove(url);
        self.rebuild_snapshot(&workers);
    }

    pub async fn mark_unhealthy(&self, url: &str) {
        let mut workers = self.workers.write().await;
        if let Some(w) = workers.get_mut(url) {
            w.health = WorkerHealth::Unhealthy;
        }
        self.rebuild_snapshot(&workers);
    }

    pub async fn check_heartbeats(&self) -> Vec<String> {
        let mut workers = self.workers.write().await;
        let now = Instant::now();
        let mut unhealthy = Vec::new();
        for (url, w) in workers.iter_mut() {
            if w.healthy() && now.duration_since(w.last_heartbeat) > self.heartbeat_timeout {
                w.health = WorkerHealth::Unhealthy;
                unhealthy.push(url.clone());
            }
        }
        if !unhealthy.is_empty() {
            self.rebuild_snapshot(&workers);
        }
        unhealthy
    }

    /// Resolve the NATS pool name for queue mode routing.
    /// Finds a healthy worker matching the bundle (and optionally GPU) that has a pool_name set.
    pub async fn resolve_queue_pool(&self, bundle: &str, gpu: &str) -> Option<String> {
        let snap = self.snapshot.load();

        // `by_bundle` is keyed with `w.bundle.to_lowercase()` in
        // `RegistrySnapshot::build`, so the lookup has to use the
        // same Unicode-aware lowercase to keep non-ASCII bundle
        // ids reachable (review feedback on PR #716). We only
        // allocate once per request on this branch, so the cost
        // is negligible compared to the per-candidate
        // `to_lowercase()` that `eq_ignore_ascii_case` replaced
        // below — gpu labels are `l4`, `a100`, `l4-spot`, … and
        // ASCII-only in practice.
        let bundle_lower = bundle.to_lowercase();

        // Use pre-computed by_bundle index for efficient lock-free lookup
        let candidates = snap.by_bundle.get(&bundle_lower)?;

        for w in candidates {
            if w.pool_name.is_empty() {
                continue;
            }
            if !gpu.is_empty() && !w.machine_profile.eq_ignore_ascii_case(gpu) {
                continue;
            }
            return Some(w.pool_name.clone());
        }
        None
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
                loaded_models: w.models.clone(),
                queue_depth: w.queue_depth,
                memory_used_bytes: w.memory_used_bytes,
                memory_total_bytes: w.memory_total_bytes,
                healthy: w.healthy(),
                bundle: w.bundle.clone(),
                bundle_config_hash: w.bundle_config_hash.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker::{GpuStatus, ModelStatus};
    use crate::types::WorkerStatusMessage;
    fn make_msg(ready: bool) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: "worker-1".into(),
            ready,
            gpu_count: 1,
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
            memory_used_bytes: None,
            memory_total_bytes: None,
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
    async fn test_update_worker_not_ready_stays_unknown() {
        let reg = registry();
        let became = reg.update_worker("http://w1:8080", make_msg(false)).await;
        assert!(!became);

        let workers = reg.workers().await;
        assert_eq!(workers["http://w1:8080"].health, WorkerHealth::Unknown);
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
            name: "w-compact".into(),
            ready: true,
            gpu_count: 1,
            machine_profile: "l4".into(),
            pool_name: String::new(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            loaded_models: vec![],
            models: vec![], // empty — should use compact fallback
            gpus: vec![],   // empty — should use compact fallback
            queue_depth: Some(7),
            memory_used_bytes: Some(2000),
            memory_total_bytes: Some(8000),
        };
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.queue_depth, 7);
        assert_eq!(w.memory_used_bytes, 2000);
        assert_eq!(w.memory_total_bytes, 8000);
    }

    #[tokio::test]
    async fn test_update_worker_compact_fields_none_defaults_to_zero() {
        let reg = registry();
        let msg = WorkerStatusMessage {
            name: "w-none".into(),
            ready: true,
            gpu_count: 1,
            machine_profile: "l4".into(),
            pool_name: String::new(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            loaded_models: vec![],
            models: vec![],
            gpus: vec![],
            queue_depth: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
        };
        reg.update_worker("http://w1:8080", msg).await;

        let w = &reg.workers().await["http://w1:8080"];
        assert_eq!(w.queue_depth, 0);
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

    // ── check_heartbeats ───────────────────────────────────────────

    #[tokio::test]
    async fn test_check_heartbeats_healthy_within_timeout() {
        let reg = registry();
        reg.update_worker("http://w1:8080", make_msg(true)).await;

        let unhealthy = reg.check_heartbeats().await;
        assert!(unhealthy.is_empty());
    }

    #[tokio::test]
    async fn test_check_heartbeats_expired() {
        let reg = WorkerRegistry::new(Duration::from_millis(1), None);
        reg.update_worker("http://w1:8080", make_msg(true)).await;

        tokio::time::sleep(Duration::from_millis(10)).await;

        let unhealthy = reg.check_heartbeats().await;
        assert_eq!(unhealthy, vec!["http://w1:8080"]);
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

    // ── resolve_queue_pool ────────────────────────────────────────

    #[tokio::test]
    async fn test_resolve_queue_pool_found() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        msg.machine_profile = "l4-spot".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("default", "l4-spot").await;
        assert_eq!(pool, Some("pool-a".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_bundle_only() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-b".into();
        msg.bundle = "premium".into();
        reg.update_worker("http://w1:8080", msg).await;

        // No GPU filter
        let pool = reg.resolve_queue_pool("premium", "").await;
        assert_eq!(pool, Some("pool-b".to_string()));
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_no_match() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = "pool-a".into();
        msg.bundle = "default".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("premium", "").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_resolve_queue_pool_skips_no_pool_name() {
        let reg = registry();
        let mut msg = make_msg(true);
        msg.pool_name = String::new();
        msg.bundle = "default".into();
        reg.update_worker("http://w1:8080", msg).await;

        let pool = reg.resolve_queue_pool("default", "").await;
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

        let pool = reg.resolve_queue_pool("default", "").await;
        assert!(pool.is_none());
    }
}
