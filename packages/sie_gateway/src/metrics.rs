use prometheus::{
    linear_buckets, Counter, CounterVec, Gauge, GaugeVec, HistogramOpts, HistogramVec, IntGauge,
    Opts, Registry,
};
use std::sync::{Arc, LazyLock, OnceLock};

// Routing decision that the HTTP metrics middleware reads as label
// values for `sie_gateway_requests_total` and
// `sie_gateway_request_latency_seconds`. The proxy handler writes this
// once after it has split `pool/profile`, resolved `l4 -> l4-spot`,
// and validated against `configured_gpus`, so the middleware records
// the *canonical* `machine_profile` the router actually used instead
// of whatever the client shoved into `x-sie-machine-profile`. Keeping
// both halves of the request flow in sync is what lets operators join
// these histograms against every other `{machine_profile}` metric on
// the dashboard (rejections, pending demand, active leases, …).
#[derive(Clone, Debug, Default)]
pub struct MetricLabels {
    pub machine_profile: String,
}

// Request-extension carrier for `MetricLabels`. Installed empty by the
// metrics middleware, filled by the proxy handler post-normalization,
// and read by the middleware after the inner service responds. The
// `OnceLock` is the cheap cell: one uncontended write + one read per
// inference request, no lock and no async-unsafe primitives.
//
// If a handler returns before calling `set` (e.g. the `model is
// required` early exit, which has no GPU yet), the middleware falls
// back to the `"unknown"` label — a real, operator-friendly value
// rather than an empty string or a noisy panic.
#[derive(Clone, Default)]
pub struct MetricLabelsSlot(Arc<OnceLock<MetricLabels>>);

impl MetricLabelsSlot {
    pub fn set(&self, labels: MetricLabels) {
        // First write wins. A handler that accidentally sets twice
        // (cleanup path + main path, for example) should not panic
        // here — observability code must never take down the request.
        let _ = self.0.set(labels);
    }

    pub fn get(&self) -> Option<&MetricLabels> {
        self.0.get()
    }
}

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let r = Registry::new();
    r.register(Box::new(REQUEST_COUNT.clone())).unwrap();
    r.register(Box::new(REQUEST_LATENCY.clone())).unwrap();
    r.register(Box::new(PROVISIONING_RESPONSES.clone()))
        .unwrap();
    r.register(Box::new(PENDING_DEMAND.clone())).unwrap();
    r.register(Box::new(REJECTED_REQUESTS.clone())).unwrap();
    r.register(Box::new(ACTIVE_LEASE_GPUS.clone())).unwrap();
    r.register(Box::new(WORKER_COUNT.clone())).unwrap();
    r.register(Box::new(WORKER_QUEUE_DEPTH.clone())).unwrap();
    r.register(Box::new(WORKER_MEMORY_USED.clone())).unwrap();
    r.register(Box::new(MODEL_WORKERS.clone())).unwrap();
    r.register(Box::new(QUEUE_PUBLISH_SECONDS.clone())).unwrap();
    r.register(Box::new(QUEUE_ITEMS_PUBLISHED.clone())).unwrap();
    r.register(Box::new(QUEUE_RESULT_WAIT.clone())).unwrap();
    r.register(Box::new(QUEUE_PAYLOAD_OFFLOADS.clone()))
        .unwrap();
    r.register(Box::new(QUEUE_INBOX_SKIPS.clone())).unwrap();
    r.register(Box::new(QUEUE_ACK_FAILURES.clone())).unwrap();
    r.register(Box::new(DLQ_EVENTS.clone())).unwrap();
    r.register(Box::new(CONFIG_BOOTSTRAP_FAILURES.clone()))
        .unwrap();
    r.register(Box::new(CONFIG_BOOTSTRAP_DEGRADED.clone()))
        .unwrap();
    r.register(Box::new(CONFIG_EPOCH.clone())).unwrap();
    r.register(Box::new(CONFIG_DELTAS.clone())).unwrap();
    r.register(Box::new(NATS_CONNECTED.clone())).unwrap();
    r.register(Box::new(POOL_EVENTS.clone())).unwrap();
    r.register(Box::new(DLQ_REPUBLISH_FAILURES.clone()))
        .unwrap();
    r
});

// 1. sie_gateway_requests_total
pub static REQUEST_COUNT: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new("sie_gateway_requests_total", "Total proxied requests"),
        &["endpoint", "status", "machine_profile"],
    )
    .unwrap()
});

// 2. sie_gateway_request_latency_seconds
pub static REQUEST_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new("sie_gateway_request_latency_seconds", "Request latency").buckets(vec![
            // The sub-5 ms range is where the gateway's own
            // overhead for small JSON requests now lives — without
            // these extra low buckets p50 gets squashed into the
            // `0.005` bucket and small wins disappear on dashboards.
            0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0,
            10.0, 30.0, 60.0,
        ]),
        &["endpoint", "machine_profile"],
    )
    .unwrap()
});

// 3. sie_gateway_provisioning_responses_total
pub static PROVISIONING_RESPONSES: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_provisioning_responses_total",
            "202 provisioning responses",
        ),
        &["machine_profile"],
    )
    .unwrap()
});

// 4. sie_gateway_pending_demand
pub static PENDING_DEMAND: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_pending_demand",
            "Pools with unmet demand (KEDA trigger)",
        ),
        &["machine_profile", "bundle"],
    )
    .unwrap()
});

// 5. sie_gateway_active_lease_gpus
pub static ACTIVE_LEASE_GPUS: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_active_lease_gpus",
            "GPUs from active leases (KEDA trigger)",
        ),
        &["machine_profile", "bundle"],
    )
    .unwrap()
});

pub static REJECTED_REQUESTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_rejected_requests_total",
            "Requests rejected before reaching a worker",
        ),
        &["machine_profile", "bundle", "reason"],
    )
    .unwrap()
});

// 6. sie_gateway_workers
pub static WORKER_COUNT: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_workers", "Worker count by health status"),
        &["status"],
    )
    .unwrap()
});

// 7. sie_gateway_worker_queue_depth
pub static WORKER_QUEUE_DEPTH: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_worker_queue_depth", "Per-worker queue depth"),
        &["worker", "machine_profile", "bundle"],
    )
    .unwrap()
});

// 8. sie_gateway_worker_memory_used_bytes
pub static WORKER_MEMORY_USED: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_worker_memory_used_bytes",
            "Per-worker GPU memory used",
        ),
        &["worker", "machine_profile", "bundle"],
    )
    .unwrap()
});

// 9. sie_gateway_model_workers
pub static MODEL_WORKERS: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_model_workers", "Workers per model"),
        &["model"],
    )
    .unwrap()
});

// 10. sie_gateway_queue_publish_seconds
pub static QUEUE_PUBLISH_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_queue_publish_seconds",
            "Work item publish time",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["operation"],
    )
    .unwrap()
});

// 11. sie_gateway_queue_items_published
pub static QUEUE_ITEMS_PUBLISHED: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_queue_items_published",
            "Items per publish batch",
        )
        .buckets(linear_buckets(1.0, 4.0, 16).unwrap()),
        &["operation"],
    )
    .unwrap()
});

// 12. sie_gateway_queue_result_wait_seconds
pub static QUEUE_RESULT_WAIT: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new("sie_gateway_queue_result_wait_seconds", "Result wait time").buckets(
            vec![
                0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0,
            ],
        ),
        &["operation"],
    )
    .unwrap()
});

// 13. sie_gateway_queue_payload_offloads_total
pub static QUEUE_PAYLOAD_OFFLOADS: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_payload_offloads_total",
        "Large payload offloads",
    )
    .unwrap()
});

// 14. sie_gateway_queue_inbox_skips_total
pub static QUEUE_INBOX_SKIPS: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_inbox_skips_total",
        "Inbox messages skipped via fast-path request_id check",
    )
    .unwrap()
});

pub static QUEUE_ACK_FAILURES: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_ack_failures_total",
        "JetStream publish acks that failed (fire-and-forget monitoring)",
    )
    .unwrap()
});

pub static DLQ_EVENTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_dlq_events_total",
            "Messages successfully forwarded to dead letter queue after max delivery attempts",
        ),
        &["stream", "consumer"],
    )
    .unwrap()
});

// sie_gateway_config_bootstrap_failures_total
//
// Incremented every time the background bootstrap retry task
// (`state::config_bootstrap::spawn_bootstrap_retry`) fails to fetch a
// clean export from `sie-config`. A sustained nonzero rate on this
// counter combined with a stale `config_epoch` is the operator signal
// that the gateway has been serving the filesystem seed for a long time
// and is at risk of diverging from the control plane.
pub static CONFIG_BOOTSTRAP_FAILURES: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_config_bootstrap_failures_total",
        "Failed attempts to fetch the initial config snapshot from sie-config",
    )
    .unwrap()
});

// sie_gateway_config_bootstrap_degraded
//
// Binary gauge: 1 while the bootstrap has never succeeded AND at least
// `DEGRADED_THRESHOLD` has elapsed since the gateway started; 0 once
// the bootstrap catches up (or from the start if `sie-config` is not
// configured). Alerts on value=1 are the SRE-visible "gateway stuck on
// the filesystem seed" signal.
pub static CONFIG_BOOTSTRAP_DEGRADED: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "sie_gateway_config_bootstrap_degraded",
        "1 iff the background bootstrap from sie-config has not yet succeeded after the degraded threshold",
    )
    .unwrap()
});

// sie_gateway_config_epoch
//
// The furthest-known control-plane epoch observed by this gateway.
// Updated by: bootstrap fetch, NATS Core delta handler, and the epoch
// poller. The single most useful alert in the whole system joins this
// against `sie_config_epoch` (published by sie-config):
//
//     (sie_config_epoch - on() sie_gateway_config_epoch) > 0
//
// If the gateway lags the control plane by any amount for more than a
// minute, either NATS deltas are not flowing or the poller is stuck.
pub static CONFIG_EPOCH: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "sie_gateway_config_epoch",
        "Control-plane epoch currently applied on this gateway",
    )
    .unwrap()
});

// sie_gateway_config_deltas_total
//
// Counts config-delta events processed from NATS Core, split by the
// kind of delta and whether the apply succeeded. A nonzero
// `result="error"` rate is the signal to investigate payload
// compatibility or the poller; a sustained zero `result="applied"`
// rate during known config writes is the signal that NATS Core
// subscriptions are broken (the fallback epoch poller will catch the
// drift eventually, but slowly).
pub static CONFIG_DELTAS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_config_deltas_total",
            "Config deltas processed from NATS Core, by kind and apply result",
        ),
        &["kind", "result"],
    )
    .unwrap()
});

// sie_gateway_nats_connected
//
// 1 iff the gateway is currently connected to NATS. Unlabeled: the
// gateway uses a single `async-nats` client for JetStream (inference
// work + results) and NATS Core (config deltas), so all three
// logical streams share one connection state. Introducing a `stream`
// label would suggest independent connectivity the code does not
// track and cannot claim. If we ever split clients, reintroduce the
// label then.
pub static NATS_CONNECTED: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::new(
        "sie_gateway_nats_connected",
        "1 iff the gateway is currently connected to NATS",
    )
    .unwrap()
});

// sie_gateway_pool_events_total
//
// Pool lifecycle counter. `ACTIVE_LEASE_GPUS` shows current state but
// not the rate of churn; this counter gives it. Use it with a 5m rate
// window to catch runaway pool creation or unexpected expirations.
pub static POOL_EVENTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new("sie_gateway_pool_events_total", "Pool lifecycle events"),
        &["event"],
    )
    .unwrap()
});

// sie_gateway_dlq_republish_failures_total
//
// Companion to `DLQ_EVENTS`: the existing counter only counts
// *successful* DLQ forwards, so an outage in the DLQ publish path is
// completely invisible on the success counter alone. This one counts
// the other half.
pub static DLQ_REPUBLISH_FAILURES: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_dlq_republish_failures_total",
            "Messages that failed to be forwarded to the dead letter queue",
        ),
        &["stream", "consumer"],
    )
    .unwrap()
});

/// Helper: flip `NATS_CONNECTED` to 1.0 (connected) or 0.0
/// (disconnected). Centralized so call sites don't have to remember
/// the 0/1 convention.
pub fn set_nats_connected(connected: bool) {
    NATS_CONNECTED.set(if connected { 1.0 } else { 0.0 });
}

/// Pre-instantiate request/demand `*Vec` metrics with empty-string
/// label sentinels so the metric family always appears in `/metrics`,
/// even on a freshly booted gateway pod that has served zero proxied
/// traffic yet.
///
/// Without this, Prometheus' text encoder silently elides empty `*Vec`
/// families, so dashboards and alerts on a quiet gateway (right after
/// rolling restart, or in a dev cluster with no callers) see "no data"
/// for `sie_gateway_requests_total`, `sie_gateway_pending_demand`, etc.
/// until the first matching observation lands. The empty-label sample
/// is a flat zero that real queries naturally filter out
/// (`{machine_profile=~".+"}` etc.) and `rate()` over a zero series is
/// also zero, so dashboards that don't filter still show 0 instead of
/// "no data" — strictly an improvement.
pub fn init_metric_families() {
    REQUEST_COUNT.with_label_values(&["", "", ""]).inc_by(0.0);
    PROVISIONING_RESPONSES.with_label_values(&[""]).inc_by(0.0);
    PENDING_DEMAND.with_label_values(&["", ""]).set(0.0);
    ACTIVE_LEASE_GPUS.with_label_values(&["", ""]).set(0.0);
    REJECTED_REQUESTS
        .with_label_values(&["", "", ""])
        .inc_by(0.0);
}

pub struct WorkerSnapshot {
    pub name: String,
    pub machine_profile: String,
    pub bundle: String,
    pub queue_depth: i32,
    pub memory_used_bytes: i64,
    pub healthy: bool,
}

pub fn update_worker_metrics(workers: &[WorkerSnapshot]) {
    // Reset per-worker gauges before repopulating. Without this, a
    // worker that disappears (pod termination, GPU drain, bundle
    // reassignment) leaves a ghost series with its last-known queue
    // depth / memory stuck forever — dashboards would treat a
    // long-gone worker as still active. `ACTIVE_LEASE_GPUS` uses the
    // same reset-then-repopulate pattern in `update_pool_metrics`.
    WORKER_QUEUE_DEPTH.reset();
    WORKER_MEMORY_USED.reset();

    let mut healthy_count = 0;
    let mut unhealthy_count = 0;

    for w in workers {
        if w.healthy {
            healthy_count += 1;
        } else {
            unhealthy_count += 1;
        }
        WORKER_QUEUE_DEPTH
            .with_label_values(&[&w.name, &w.machine_profile, &w.bundle])
            .set(w.queue_depth as f64);
        WORKER_MEMORY_USED
            .with_label_values(&[&w.name, &w.machine_profile, &w.bundle])
            .set(w.memory_used_bytes as f64);
    }

    // `WORKER_COUNT` has only two fixed label values (`healthy` /
    // `unhealthy`), so an always-set strategy is enough — nothing to
    // reset.
    WORKER_COUNT
        .with_label_values(&["healthy"])
        .set(healthy_count as f64);
    WORKER_COUNT
        .with_label_values(&["unhealthy"])
        .set(unhealthy_count as f64);
}

pub fn update_model_metrics(models: &std::collections::HashMap<String, usize>) {
    // Reset before repopulating: otherwise, models removed from the
    // registry (explicit deletion or config reload that drops them)
    // keep exporting their last-known worker count until the gateway
    // restarts. Scrape frequency is low enough that the reset window
    // is invisible to Prometheus rate() queries.
    MODEL_WORKERS.reset();
    for (name, count) in models {
        MODEL_WORKERS.with_label_values(&[name]).set(*count as f64);
    }
}

pub fn record_rejected_request(machine_profile: &str, bundle: &str, reason: &str) {
    let effective_profile = if machine_profile.is_empty() {
        "unknown"
    } else {
        machine_profile
    };
    REJECTED_REQUESTS
        .with_label_values(&[effective_profile, bundle, reason])
        .inc();
}

/// Clear PENDING_DEMAND for GPU types that now have healthy workers.
/// Called when worker metrics are updated to cancel stale demand signals.
/// Uses the DemandTracker to properly cancel expiry timers when clearing.
pub fn clear_fulfilled_demand(
    healthy_gpu_types: &std::collections::HashSet<String>,
    demand_tracker: &crate::state::demand_tracker::DemandTracker,
) {
    // Gather current demand label pairs from the metric
    let metric_families = REGISTRY.gather();
    for mf in &metric_families {
        if mf.get_name() != "sie_gateway_pending_demand" {
            continue;
        }
        for m in mf.get_metric() {
            let labels: std::collections::HashMap<&str, &str> = m
                .get_label()
                .iter()
                .map(|l| (l.get_name(), l.get_value()))
                .collect();
            if let Some(&gpu) = labels.get("machine_profile") {
                if healthy_gpu_types.contains(&gpu.to_lowercase()) {
                    let bundle = labels.get("bundle").copied().unwrap_or("default");
                    let current = PENDING_DEMAND.with_label_values(&[gpu, bundle]).get();
                    if current > 0.0 {
                        demand_tracker.clear(gpu, bundle);
                        tracing::info!(
                            gpu = gpu,
                            bundle = bundle,
                            "cleared pending demand — healthy workers available"
                        );
                    }
                }
            }
        }
    }
}

pub fn update_pool_metrics(pools: &[crate::types::Pool]) {
    use crate::state::pool_manager::DEFAULT_POOL_NAME;
    use crate::types::PoolState;

    // Reset gauge to avoid stale entries from deleted pools, then
    // re-seed the empty-label sentinel so the family stays visible
    // in `/metrics` on an idle gateway whose only pool is the
    // default pool (which is skipped below by design). Without this
    // re-seed the `init_metric_families()` sentinel is wiped on the
    // very first scrape.
    ACTIVE_LEASE_GPUS.reset();
    ACTIVE_LEASE_GPUS.with_label_values(&["", ""]).set(0.0);

    for pool in pools {
        if pool.status.state == PoolState::Active && pool.spec.name != DEFAULT_POOL_NAME {
            let bundle = pool.spec.bundle.as_deref().unwrap_or("default");
            for (gpu_type, count) in &pool.spec.gpus {
                ACTIVE_LEASE_GPUS
                    .with_label_values(&[gpu_type, bundle])
                    .add(*count as f64);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{Pool, PoolSpec, PoolState, PoolStatus};
    use std::collections::HashMap;
    use std::sync::Mutex;

    static METRICS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_test_metrics() {
        ACTIVE_LEASE_GPUS.reset();
        WORKER_COUNT.reset();
        WORKER_QUEUE_DEPTH.reset();
        WORKER_MEMORY_USED.reset();
        MODEL_WORKERS.reset();
        PENDING_DEMAND.reset();
    }

    fn make_pool(
        name: &str,
        state: PoolState,
        gpus: HashMap<String, u32>,
        bundle: Option<String>,
    ) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle,
                gpus,
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state,
                assigned_workers: vec![],
                created_at: 0.0,
                last_renewed: 0.0,
            },
        }
    }

    #[test]
    fn test_update_pool_metrics_active_pool() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        // Force registry init
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        let pools = vec![make_pool(
            "eval-pool",
            PoolState::Active,
            gpus,
            Some("premium".to_string()),
        )];

        update_pool_metrics(&pools);

        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["l4-spot", "premium"])
            .get();
        assert!((val - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_skips_default_pool() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 999);
        let pools = vec![make_pool("default", PoolState::Active, gpus, None)];

        update_pool_metrics(&pools);

        // Default pool should not contribute to active lease GPUs
        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["l4-spot", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_skips_pending() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("a100".to_string(), 4);
        let pools = vec![make_pool(
            "pending-pool",
            PoolState::Pending,
            gpus,
            Some("default".to_string()),
        )];

        update_pool_metrics(&pools);

        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["a100", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_reset_clears_stale() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        ACTIVE_LEASE_GPUS
            .with_label_values(&["stale-gpu", "stale-bundle"])
            .set(10.0);

        update_pool_metrics(&[]);

        // Stale series must be gone, but the empty-label sentinel
        // re-seeded by `update_pool_metrics` itself must remain so the
        // family stays visible in `/metrics`.
        let families = REGISTRY.gather();
        let lease = families
            .iter()
            .find(|mf| mf.get_name() == "sie_gateway_active_lease_gpus")
            .expect("sie_gateway_active_lease_gpus should still be present after reset");
        let stale_present = lease.get_metric().iter().any(|m| {
            m.get_label().iter().any(|l| {
                (l.get_name() == "machine_profile" && l.get_value() == "stale-gpu")
                    || (l.get_name() == "bundle" && l.get_value() == "stale-bundle")
            })
        });
        assert!(!stale_present, "stale label combination survived reset");
        let sentinel_present = lease
            .get_metric()
            .iter()
            .any(|m| m.get_label().iter().all(|l| l.get_value().is_empty()));
        assert!(
            sentinel_present,
            "empty-label sentinel missing after update_pool_metrics(&[])"
        );
    }

    #[test]
    fn test_update_worker_metrics() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let workers = vec![
            WorkerSnapshot {
                name: "w1".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                queue_depth: 5,
                memory_used_bytes: 1000,
                healthy: true,
            },
            WorkerSnapshot {
                name: "w2".to_string(),
                machine_profile: "a100".to_string(),
                bundle: "premium".to_string(),
                queue_depth: 2,
                memory_used_bytes: 2000,
                healthy: false,
            },
        ];

        update_worker_metrics(&workers);

        let healthy = WORKER_COUNT.with_label_values(&["healthy"]).get();
        let unhealthy = WORKER_COUNT.with_label_values(&["unhealthy"]).get();
        assert!((healthy - 1.0).abs() < f64::EPSILON);
        assert!((unhealthy - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_model_metrics() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut models = HashMap::new();
        models.insert("BAAI/bge-m3".to_string(), 3usize);
        models.insert("openai/clip".to_string(), 1usize);

        update_model_metrics(&models);

        let val = MODEL_WORKERS.with_label_values(&["BAAI/bge-m3"]).get();
        assert!((val - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_worker_metrics_evicts_stale_series() {
        // Regression: previously, a worker that disappeared between
        // scrapes (pod termination, GPU drain) left a ghost series
        // with its last-known queue depth / memory. We now reset
        // before repopulating, so only the current snapshot is
        // visible.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let first = vec![WorkerSnapshot {
            name: "ghost-worker".to_string(),
            machine_profile: "l4-spot".to_string(),
            bundle: "default".to_string(),
            queue_depth: 42,
            memory_used_bytes: 123_456,
            healthy: true,
        }];
        update_worker_metrics(&first);
        let d = WORKER_QUEUE_DEPTH
            .with_label_values(&["ghost-worker", "l4-spot", "default"])
            .get();
        assert!((d - 42.0).abs() < f64::EPSILON);

        // Simulate the worker disappearing: next snapshot does not
        // include it. A different worker is reported instead.
        let second = vec![WorkerSnapshot {
            name: "other-worker".to_string(),
            machine_profile: "a100".to_string(),
            bundle: "premium".to_string(),
            queue_depth: 7,
            memory_used_bytes: 999,
            healthy: true,
        }];
        update_worker_metrics(&second);

        // The new worker is visible.
        let d2 = WORKER_QUEUE_DEPTH
            .with_label_values(&["other-worker", "a100", "premium"])
            .get();
        assert!((d2 - 7.0).abs() < f64::EPSILON);

        // The stale series must be gone. `get()` on a
        // non-existent label set would create it at 0.0; instead we
        // inspect the registry directly to confirm the series was
        // dropped.
        let families = REGISTRY.gather();
        let stale_present = families
            .iter()
            .filter(|mf| mf.get_name() == "sie_gateway_worker_queue_depth")
            .flat_map(|mf| mf.get_metric().iter())
            .any(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.get_name() == "worker" && l.get_value() == "ghost-worker")
            });
        assert!(
            !stale_present,
            "ghost-worker series must be evicted after it disappears from the snapshot"
        );
    }

    #[test]
    fn test_update_model_metrics_evicts_stale_series() {
        // Same regression for MODEL_WORKERS: removing a model from
        // the registry must clear its series.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut first = HashMap::new();
        first.insert("org/ghost-model".to_string(), 5usize);
        update_model_metrics(&first);
        let d = MODEL_WORKERS.with_label_values(&["org/ghost-model"]).get();
        assert!((d - 5.0).abs() < f64::EPSILON);

        let mut second = HashMap::new();
        second.insert("org/other-model".to_string(), 2usize);
        update_model_metrics(&second);

        let families = REGISTRY.gather();
        let stale_present = families
            .iter()
            .filter(|mf| mf.get_name() == "sie_gateway_model_workers")
            .flat_map(|mf| mf.get_metric().iter())
            .any(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.get_name() == "model" && l.get_value() == "org/ghost-model")
            });
        assert!(
            !stale_present,
            "ghost-model series must be evicted after it disappears from the registry"
        );
    }

    #[test]
    fn test_clear_fulfilled_demand() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let tracker = crate::state::demand_tracker::DemandTracker::new();

        // Set demand for l4-spot
        PENDING_DEMAND
            .with_label_values(&["l4-spot", "default"])
            .set(1.0);

        // Healthy workers include l4-spot
        let mut gpus = std::collections::HashSet::new();
        gpus.insert("l4-spot".to_string());

        clear_fulfilled_demand(&gpus, &tracker);

        let val = PENDING_DEMAND
            .with_label_values(&["l4-spot", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_fulfilled_demand_preserves_unmatched() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let tracker = crate::state::demand_tracker::DemandTracker::new();

        // Set demand for h100 (no healthy workers)
        PENDING_DEMAND
            .with_label_values(&["h100", "default"])
            .set(1.0);

        // Only l4-spot is healthy
        let mut gpus = std::collections::HashSet::new();
        gpus.insert("l4-spot".to_string());

        clear_fulfilled_demand(&gpus, &tracker);

        // h100 demand should remain
        let val = PENDING_DEMAND.with_label_values(&["h100", "default"]).get();
        assert!((val - 1.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // Direct emission tests for the 5 metrics added in the audit pass.
    //
    // These tests read the global Prometheus registry, so they share
    // `METRICS_TEST_LOCK` with every other test in this module to avoid
    // racing with concurrent increments.
    // ------------------------------------------------------------------

    #[test]
    fn test_config_epoch_mirrors_on_set_max() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let epoch = crate::state::config_epoch::ConfigEpoch::new();
        // Pick a baseline well above any value other tests would set so
        // our assertions aren't clobbered by an unrelated test that ran
        // before we took the lock (Prometheus counters/gauges are global).
        let base = CONFIG_EPOCH.get().max(0) as u64 + 1_000_000;

        assert!(epoch.set_max(base + 1));
        assert_eq!(CONFIG_EPOCH.get(), (base + 1) as i64);

        // Non-advancing set_max is a no-op on the gauge too.
        CONFIG_EPOCH.set((base + 1) as i64);
        assert!(!epoch.set_max(base));
        assert_eq!(CONFIG_EPOCH.get(), (base + 1) as i64);

        // force_set mirrors even on a backward move (recovery path).
        epoch.force_set(base);
        assert_eq!(CONFIG_EPOCH.get(), base as i64);
    }

    #[test]
    fn test_config_deltas_labels_and_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        // All four `(kind, result)` pairs we emit in manager.rs must
        // register on the CounterVec. We just verify that `.inc()` moves
        // the observed value; the label validation is implicit (Prometheus
        // would panic on a cardinality mismatch).
        for labels in &[
            ("model_added", "applied"),
            ("model_added", "parse_error"),
            ("model_added", "apply_error"),
            ("model_added", "rejected_untrusted"),
            ("epoch_bump", "applied"),
            ("epoch_bump", "rejected_untrusted"),
        ] {
            let before = CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).get();
            CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).inc();
            let after = CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).get();
            assert!(
                (after - before - 1.0).abs() < f64::EPSILON,
                "({}, {}) did not increment",
                labels.0,
                labels.1
            );
        }
    }

    #[test]
    fn test_nats_connected_gauge_flips() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        set_nats_connected(true);
        assert_eq!(NATS_CONNECTED.get(), 1.0);

        set_nats_connected(false);
        assert_eq!(NATS_CONNECTED.get(), 0.0);

        // Idempotent: flipping to the same value does not panic or
        // double-count (gauges are set, not incremented).
        set_nats_connected(false);
        assert_eq!(NATS_CONNECTED.get(), 0.0);
    }

    #[test]
    fn test_pool_events_labels_and_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        for event in &["created", "renewed", "deleted", "expired"] {
            let before = POOL_EVENTS.with_label_values(&[event]).get();
            POOL_EVENTS.with_label_values(&[event]).inc();
            let after = POOL_EVENTS.with_label_values(&[event]).get();
            assert!(
                (after - before - 1.0).abs() < f64::EPSILON,
                "pool_events{{event={}}} did not increment",
                event
            );
        }
    }

    #[test]
    fn test_dlq_republish_failures_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let before = DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .get();
        DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .inc();
        let after = DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .get();
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    /// Without `init_metric_families`, `*Vec` metrics with no
    /// observations are silently elided by Prometheus' text encoder,
    /// so a freshly booted gateway shows "no data" for the
    /// request/demand panels. This test pins the contract: after
    /// resetting the relevant vecs and calling the init helper, every
    /// targeted family must show up in `REGISTRY.gather()`.
    #[test]
    fn test_init_metric_families_exposes_empty_vecs() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        REQUEST_COUNT.reset();
        PROVISIONING_RESPONSES.reset();
        PENDING_DEMAND.reset();
        ACTIVE_LEASE_GPUS.reset();
        REJECTED_REQUESTS.reset();

        // Sanity: post-reset, none of the targeted families should
        // appear in `gather()`. If this precondition ever stops
        // holding, the test below stops proving anything useful.
        let pre = REGISTRY.gather();
        for needle in [
            "sie_gateway_requests_total",
            "sie_gateway_pending_demand",
            "sie_gateway_active_lease_gpus",
            "sie_gateway_rejected_requests_total",
            "sie_gateway_provisioning_responses_total",
        ] {
            assert!(
                !pre.iter().any(|mf| mf.get_name() == needle),
                "precondition violated: {needle} still in gather() after reset"
            );
        }

        init_metric_families();
        // Simulate a scrape on an idle gateway: no pools, no workers,
        // no models. `update_pool_metrics(&[])` is the riskiest path
        // because it calls `ACTIVE_LEASE_GPUS.reset()` before
        // repopulating, and on an idle gateway there's nothing to
        // repopulate from. If the family disappears here, the bug
        // ships unnoticed.
        update_pool_metrics(&[]);

        let post = REGISTRY.gather();
        for needle in [
            "sie_gateway_requests_total",
            "sie_gateway_pending_demand",
            "sie_gateway_active_lease_gpus",
            "sie_gateway_rejected_requests_total",
            "sie_gateway_provisioning_responses_total",
        ] {
            let found = post
                .iter()
                .find(|mf| mf.get_name() == needle)
                .unwrap_or_else(|| panic!("{needle} missing after init + scrape path"));
            assert!(
                !found.get_metric().is_empty(),
                "{needle} family present but has zero samples"
            );
        }
    }
}
