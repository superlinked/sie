//! NATS health-heartbeat publisher.
//!
//! In `health_mode=nats` the gateway expects every worker to
//! periodically publish a [`WorkerStatusMessage`]-shaped payload to
//! `sie.health.<worker_id>` so its `WorkerRegistry` knows the worker
//! exists, which `bundle` it serves, and which `pool_name` to route
//! NATS work to. Without this signal the gateway returns
//! `202 provisioning` for every request.
//!
//! See `packages/sie_gateway/src/discovery/nats_health.rs` for the
//! consumer side and `docs/queue-based-routing.md` for the routing
//! contract.
//!
//! # What the gateway actually reads
//!
//! Only a small subset of fields is load-bearing for routing:
//!
//! * `bundle` — must match the bundle resolved by the gateway's
//!   model registry (lowercase compared); routes only consider
//!   workers whose bundle index entry contains them.
//! * `pool_name` — non-empty so [`resolve_queue_route`] can return
//!   the pool the dispatcher publishes work onto. An empty
//!   `pool_name` makes the worker invisible to the queue path.
//! * `machine_profile` — the hardware lane embedded in the queue
//!   subject. `X-SIE-MACHINE-PROFILE` constrains route resolution to
//!   this value; otherwise the gateway infers it from a healthy worker.
//! * `ready` — drives the `WorkerHealth::Healthy` transition. We
//!   gate on the local [`Readiness`] state so the heartbeat agrees
//!   with `/readyz`.
//!
//! `loaded_models` and the GPU memory fields are mostly informational, but
//! the gateway also uses `loaded_models` as the per-model dispatch readiness
//! signal. Backends that need explicit residency, such as Candle, must only
//! publish models here after their readiness handshake has made them serveable.
//!
//! [`WorkerStatusMessage`]: https://github.com/search?q=WorkerStatusMessage+repo%3Asie-internal
//! [`resolve_queue_route`]: https://github.com/search?q=resolve_queue_route+repo%3Asie-internal

use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use async_nats::Client;
use serde::Serialize;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::{debug, info, warn};

use crate::readiness::Readiness;
use crate::runtime_state::RuntimeState;
use crate::shutdown::Shutdown;

pub type SharedBundleConfigHash = Arc<RwLock<String>>;
pub type SharedLoadedModels = Arc<RwLock<Vec<String>>>;

/// Default subject prefix used by the gateway. Workers always
/// publish to `sie.health.<worker_id>` because the gateway's
/// `extract_worker_url_from_status` reads `splitn(3, '.')` on the
/// subject, so any sub-token segments would confuse URL extraction.
pub const HEALTH_SUBJECT_PREFIX: &str = "sie.health";

/// Default interval between heartbeats. Matches the gateway's
/// `NatsHealthManager::start_heartbeat_loop` 5 s tick — anything
/// faster wastes NATS bandwidth and anything slower risks the
/// gateway flipping the worker unhealthy via `check_heartbeats`
/// (default `heartbeat_timeout` is 30 s, so 5 s gives a 6× margin).
pub const DEFAULT_PUBLISH_INTERVAL: Duration = Duration::from_secs(5);
const SATURATION_SET_PERCENT: i64 = 90;
const SATURATION_CLEAR_PERCENT: i64 = 70;

/// Static config for the heartbeat publisher.
///
/// Most fields are captured at startup. The ready flag comes from
/// [`Readiness`], and `bundle_config_hash` is shared with the live config
/// subscriber so gateway readiness can observe successful worker-side apply.
#[derive(Clone)]
pub struct HealthPublisherConfig {
    /// Stable identifier used as the subject leaf and as the
    /// fallback URL the gateway constructs. Should match
    /// `WorkerConfig::worker_id`.
    pub worker_id: String,
    /// The bundle this pod's adapter is serving. The gateway's
    /// per-bundle index keys on this (lowercased), so it must
    /// match `BundleConfig.id`. Sourced from `SIE_BUNDLE`.
    pub bundle: String,
    /// Pool name. The gateway publishes work onto
    /// `sie.work.<pool_name>.<machine_profile>.<bundle>.<model>`; the
    /// worker must echo the same pool here or queue-route resolution
    /// returns `None` and requests 202-provision forever. Sourced from
    /// `SIE_POOL`.
    pub pool_name: String,
    /// Optional machine-profile label (e.g. `l4`, `a100`).
    /// Consulted only when the inbound request carries
    /// `X-SIE-MACHINE-PROFILE`. Empty means "any GPU".
    pub machine_profile: String,
    /// Reported GPU count. The registry coerces `0 -> 1` so a
    /// missing value still routes; we still want to surface the
    /// real count when known so admin views show accurate totals.
    pub gpu_count: i32,
    /// Bundle-config hash so the gateway can correlate against its own
    /// model registry epoch. Updated by the config subscriber after a
    /// successful backend config apply.
    pub bundle_config_hash: SharedBundleConfigHash,
    /// Models the colocated backend reports as loaded. Updated by the IPC
    /// heartbeat, not by config apply.
    pub loaded_models: SharedLoadedModels,
    /// Runtime pressure/capacity gauges mirrored into the heartbeat payload.
    pub runtime_state: Arc<RuntimeState>,
    /// How often to publish. Defaults to [`DEFAULT_PUBLISH_INTERVAL`].
    pub interval: Duration,
}

impl HealthPublisherConfig {
    /// Subject this pod publishes to. Kept as a method so the
    /// gateway-side parser (`splitn(3, '.')`) and this side stay
    /// in lockstep — change the prefix here and the gateway will
    /// see no health messages.
    pub fn subject(&self) -> String {
        format!("{}.{}", HEALTH_SUBJECT_PREFIX, self.worker_id)
    }
}

/// Wire shape published to NATS. Mirrors the gateway's
/// [`crate::types::WorkerStatusMessage`] subset that's actually
/// load-bearing for routing.
///
/// Serialised as JSON because the gateway's
/// `handle_nats_message` tries msgpack first and falls back to
/// JSON; JSON keeps the payload trivially `nats sub`-debuggable
/// from a shell, which matters for the post-deploy smoke test
/// we do every time. Both formats are accepted, so we can switch
/// to msgpack later without a flag day on either side.
#[derive(Debug, Serialize)]
struct WorkerStatusPayload<'a> {
    name: &'a str,
    ready: bool,
    terminated: bool,
    gpu_count: i32,
    total_gpu_slots: i32,
    ready_gpu_slots: i32,
    machine_profile: &'a str,
    pool_name: &'a str,
    bundle: &'a str,
    bundle_config_hash: &'a str,
    loaded_models: &'a [String],
    queue_depth: i32,
    pending_cost: i64,
    inflight_batches: i32,
    saturated: bool,
}

fn encode_payload(
    config: &HealthPublisherConfig,
    ready: bool,
    terminated: bool,
) -> Result<Vec<u8>, serde_json::Error> {
    let hash = config
        .bundle_config_hash
        .read()
        .expect("bundle config hash lock poisoned");
    let loaded_models_guard = match config.loaded_models.read() {
        Ok(guard) => Some(guard),
        Err(_) => {
            warn!("loaded_models lock poisoned; publishing empty loaded model list");
            None
        }
    };
    let loaded_models = match &loaded_models_guard {
        Some(guard) => guard.as_slice(),
        None => &[],
    };
    let configured_slots = config.gpu_count.max(1);
    let total_gpu_slots = clamp_i64_to_i32(
        config
            .runtime_state
            .worker_gpu_slots_total
            .get()
            .max(configured_slots.into()),
    );
    let ready_gpu_slots = if ready {
        clamp_i64_to_i32(
            config
                .runtime_state
                .worker_gpu_slots_ready
                .get()
                .clamp(0, total_gpu_slots.into()),
        )
    } else {
        0
    };
    config
        .runtime_state
        .worker_gpu_slots_total
        .set(total_gpu_slots.into());
    config
        .runtime_state
        .worker_gpu_slots_ready
        .set(ready_gpu_slots.into());
    let queue_depth = clamp_i64_to_i32(config.runtime_state.worker_queue_depth.get().max(0));
    let pending_cost = config.runtime_state.worker_pending_cost.get().max(0);
    let inflight_batches = clamp_i64_to_i32(config.runtime_state.inflight_batches.get().max(0));
    refresh_worker_saturation(&config.runtime_state);
    let saturated = config.runtime_state.worker_saturated.get() > 0;
    let payload = WorkerStatusPayload {
        name: &config.worker_id,
        ready,
        terminated,
        gpu_count: config.gpu_count,
        total_gpu_slots,
        ready_gpu_slots,
        machine_profile: &config.machine_profile,
        pool_name: &config.pool_name,
        bundle: &config.bundle,
        bundle_config_hash: hash.as_str(),
        loaded_models,
        queue_depth,
        pending_cost,
        inflight_batches,
        saturated,
    };
    serde_json::to_vec(&payload)
}

fn clamp_i64_to_i32(value: i64) -> i32 {
    i32::try_from(value).unwrap_or_else(|_| if value.is_negative() { 0 } else { i32::MAX })
}

fn refresh_worker_saturation(runtime_state: &RuntimeState) {
    let ready_slots = runtime_state.worker_gpu_slots_ready.get().max(0);
    if ready_slots == 0 {
        runtime_state.worker_saturated.set(0);
        return;
    }

    let capacity = ready_slots
        .saturating_mul(crate::dispatcher::default_max_concurrent_batches().max(1) as i64)
        .max(1);
    let load = runtime_state
        .worker_queue_depth
        .get()
        .max(0)
        .saturating_add(runtime_state.inflight_batches.get().max(0));
    let threshold = if runtime_state.worker_saturated.get() > 0 {
        SATURATION_CLEAR_PERCENT
    } else {
        SATURATION_SET_PERCENT
    };
    let saturated = load.saturating_mul(100) >= capacity.saturating_mul(threshold);
    runtime_state.worker_saturated.set(i64::from(saturated));
}

/// Emit the complete runtime-pressure snapshot once per shared IPC heartbeat.
/// The heartbeat exists in both NATS and local-ingest modes, so this avoids a
/// topology-specific duplicate observation path.
pub(crate) fn record_runtime_telemetry(runtime_state: &RuntimeState) {
    refresh_worker_saturation(runtime_state);
    let nonnegative = |value: i64| u64::try_from(value.max(0)).unwrap_or(u64::MAX);
    runtime_state.telemetry.runtime_snapshot(
        nonnegative(runtime_state.worker_gpu_slots_total.get()),
        nonnegative(runtime_state.worker_gpu_slots_ready.get()),
        nonnegative(runtime_state.worker_queue_depth.get()),
        nonnegative(runtime_state.worker_pending_cost.get()),
        nonnegative(runtime_state.inflight_batches.get()),
        runtime_state.worker_saturated.get() > 0,
    );
}

pub async fn publish_tombstone(
    nats: &Client,
    config: &HealthPublisherConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let subject = config.subject();
    let bytes = encode_payload(config, false, true)?;
    nats.publish(subject.clone(), bytes.into()).await?;
    nats.flush().await?;
    info!(subject = %subject, "nats-health: shutdown tombstone published");
    Ok(())
}

/// Spawn the heartbeat-publisher loop. The returned [`JoinHandle`]
/// is owned by `run()` so shutdown can `.abort()` it after the
/// pull loop exits — same lifecycle as `crate::spawn_heartbeat`.
///
/// Failure modes (non-fatal):
///
/// * NATS publish errors — logged, counter bumped via WARN every
///   5 / 30 / 120 ticks (same backoff pattern as the IPC heartbeat
///   to avoid spamming during a NATS outage).
/// * Serde encode errors — should be unreachable for a static
///   payload shape; logged and skipped.
pub fn spawn(
    nats: Client,
    config: HealthPublisherConfig,
    readiness: Arc<Readiness>,
    shutdown: Arc<Shutdown>,
) -> JoinHandle<()> {
    let subject = config.subject();
    info!(
        subject = %subject,
        bundle = %config.bundle,
        pool = %config.pool_name,
        machine_profile = %config.machine_profile,
        gpu_count = config.gpu_count,
        interval_ms = config.interval.as_millis() as u64,
        "nats-health: heartbeat publisher starting"
    );

    tokio::spawn(async move {
        let mut tick = interval(config.interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        let mut consecutive_failures: u64 = 0;
        loop {
            let wait = shutdown.wait();
            tokio::select! {
                biased;
                _ = wait => {
                    debug!(subject = %subject, "nats-health: shutdown observed; publishing tombstone");
                    if let Err(e) = publish_tombstone(&nats, &config).await {
                        warn!(
                            subject = %subject,
                            error = %e,
                            "nats-health: shutdown tombstone publish failed"
                        );
                    }
                    return;
                }
                _ = tick.tick() => {
                    let snap = readiness.snapshot();
                    let ready = snap.is_ready();
                    let bytes = match encode_payload(&config, ready, false) {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(
                                error = %e,
                                "nats-health: serde_json encode failed (unexpected for static shape)"
                            );
                            continue;
                        }
                    };
                    match nats.publish(subject.clone(), bytes.into()).await {
                        Ok(_) => {
                            if consecutive_failures > 0 {
                                info!(
                                    consecutive_failures,
                                    subject = %subject,
                                    "nats-health: publish recovered"
                                );
                            }
                            consecutive_failures = 0;
                            debug!(subject = %subject, ready, "nats-health: heartbeat published");
                        }
                        Err(e) => {
                            consecutive_failures = consecutive_failures.saturating_add(1);
                            // Same warn cadence as `spawn_heartbeat`:
                            // first failure, then 5 / 30 / every 120
                            // so a sustained NATS outage doesn't
                            // drown the log file.
                            if consecutive_failures == 1
                                || consecutive_failures == 5
                                || consecutive_failures == 30
                                || consecutive_failures.is_multiple_of(120)
                            {
                                warn!(
                                    consecutive_failures,
                                    subject = %subject,
                                    error = %e,
                                    "nats-health: publish failed"
                                );
                            } else {
                                debug!(
                                    consecutive_failures,
                                    subject = %subject,
                                    error = %e,
                                    "nats-health: publish still failing"
                                );
                            }
                        }
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> HealthPublisherConfig {
        HealthPublisherConfig {
            worker_id: "worker-l4-0".into(),
            bundle: "default".into(),
            pool_name: "l4".into(),
            machine_profile: "l4".into(),
            gpu_count: 1,
            bundle_config_hash: Arc::new(RwLock::new("hash-abc".into())),
            loaded_models: Arc::new(RwLock::new(Vec::new())),
            runtime_state: Arc::new(RuntimeState::new()),
            interval: DEFAULT_PUBLISH_INTERVAL,
        }
    }

    #[test]
    fn subject_is_prefix_dot_worker_id() {
        // Gateway's `extract_worker_url_from_status` does
        // `splitn(3, '.')` so the worker_id must occupy a single
        // subject token. We don't enforce that here (NATS
        // tolerates more dots) but the convention is one token.
        assert_eq!(cfg().subject(), "sie.health.worker-l4-0");
    }

    #[test]
    fn payload_roundtrips_to_gateway_shape() {
        // Serialises to the exact field set
        // `WorkerStatusMessage::deserialize` reads — every
        // load-bearing field present, nothing extra.
        let c = cfg();
        let json = {
            let hash = c.bundle_config_hash.read().unwrap();
            let payload = WorkerStatusPayload {
                name: &c.worker_id,
                ready: true,
                terminated: false,
                gpu_count: c.gpu_count,
                total_gpu_slots: 1,
                ready_gpu_slots: 1,
                machine_profile: &c.machine_profile,
                pool_name: &c.pool_name,
                bundle: &c.bundle,
                bundle_config_hash: hash.as_str(),
                queue_depth: 0,
                pending_cost: 0,
                inflight_batches: 0,
                saturated: false,
                loaded_models: &[],
            };
            serde_json::to_value(&payload).unwrap()
        };
        assert_eq!(json["name"], "worker-l4-0");
        assert_eq!(json["ready"], true);
        assert_eq!(json["terminated"], false);
        assert_eq!(json["gpu_count"], 1);
        assert_eq!(json["total_gpu_slots"], 1);
        assert_eq!(json["ready_gpu_slots"], 1);
        assert_eq!(json["queue_depth"], 0);
        assert_eq!(json["pending_cost"], 0);
        assert_eq!(json["inflight_batches"], 0);
        assert_eq!(json["saturated"], false);
        assert_eq!(json["machine_profile"], "l4");
        assert_eq!(json["pool_name"], "l4");
        assert_eq!(json["bundle"], "default");
        assert_eq!(json["bundle_config_hash"], "hash-abc");
        assert_eq!(json["loaded_models"], serde_json::json!([]));
    }

    #[test]
    fn payload_reads_updated_bundle_config_hash() {
        let c = cfg();
        {
            let mut hash = c.bundle_config_hash.write().unwrap();
            *hash = "hash-next".into();
        }
        let json = {
            let hash = c.bundle_config_hash.read().unwrap();
            let payload = WorkerStatusPayload {
                name: &c.worker_id,
                ready: true,
                terminated: false,
                gpu_count: c.gpu_count,
                total_gpu_slots: 1,
                ready_gpu_slots: 1,
                machine_profile: &c.machine_profile,
                pool_name: &c.pool_name,
                bundle: &c.bundle,
                bundle_config_hash: hash.as_str(),
                queue_depth: 0,
                pending_cost: 0,
                inflight_batches: 0,
                saturated: false,
                loaded_models: &[],
            };
            serde_json::to_value(&payload).unwrap()
        };
        assert_eq!(json["bundle_config_hash"], "hash-next");
    }

    #[test]
    fn payload_reads_updated_loaded_models() {
        let c = cfg();
        {
            let mut models = c.loaded_models.write().unwrap();
            *models = vec!["model/a".into(), "model/b".into()];
        }
        let json: serde_json::Value =
            serde_json::from_slice(&encode_payload(&c, true, false).unwrap()).unwrap();
        assert_eq!(
            json["loaded_models"],
            serde_json::json!(["model/a", "model/b"])
        );
    }

    #[test]
    fn tombstone_payload_marks_not_ready_and_terminated() {
        let c = cfg();
        let json: serde_json::Value =
            serde_json::from_slice(&encode_payload(&c, false, true).unwrap()).unwrap();

        assert_eq!(json["name"], "worker-l4-0");
        assert_eq!(json["ready"], false);
        assert_eq!(json["terminated"], true);
        assert_eq!(json["gpu_count"], 1);
        assert_eq!(json["total_gpu_slots"], 1);
        assert_eq!(json["ready_gpu_slots"], 0);
        assert_eq!(json["machine_profile"], "l4");
        assert_eq!(json["pool_name"], "l4");
        assert_eq!(json["bundle"], "default");
        assert_eq!(json["bundle_config_hash"], "hash-abc");
        assert_eq!(json["loaded_models"], serde_json::json!([]));
    }

    #[test]
    fn payload_includes_runtime_pressure_metrics() {
        let c = cfg();
        c.runtime_state.worker_gpu_slots_ready.set(1);
        c.runtime_state.worker_queue_depth.set(1_000_000);
        c.runtime_state.worker_pending_cost.set(1234);
        c.runtime_state.inflight_batches.set(2);

        let json: serde_json::Value =
            serde_json::from_slice(&encode_payload(&c, true, false).unwrap()).unwrap();

        assert_eq!(json["queue_depth"], 1_000_000);
        assert_eq!(json["pending_cost"], 1234);
        assert_eq!(json["inflight_batches"], 2);
        assert_eq!(json["saturated"], true);
    }

    #[test]
    fn payload_derives_saturation_from_worker_pressure() {
        let c = cfg();
        c.runtime_state.worker_gpu_slots_ready.set(1);
        c.runtime_state.worker_queue_depth.set(1_000_000);

        let saturated: serde_json::Value =
            serde_json::from_slice(&encode_payload(&c, true, false).unwrap()).unwrap();
        assert_eq!(saturated["saturated"], true);

        c.runtime_state.worker_queue_depth.set(0);
        let cleared: serde_json::Value =
            serde_json::from_slice(&encode_payload(&c, true, false).unwrap()).unwrap();
        assert_eq!(cleared["saturated"], false);
    }

    #[test]
    fn payload_uses_local_readiness_for_ready_flag() {
        // Heartbeat must reflect /readyz state — once draining
        // we publish ready=false so the gateway transitions us
        // to Unhealthy ASAP rather than waiting for the 30 s
        // heartbeat_timeout to elapse.
        let r = Readiness::new(2_000, 3);
        r.record_ping_success();
        assert!(r.snapshot().is_ready());

        r.mark_draining();
        assert!(!r.snapshot().is_ready());
    }
}
