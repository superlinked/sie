use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use axum::Router;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::handlers::{audio, config_api, health, models, pools, proxy};
use crate::middleware::audit::AuditLayer;
use crate::middleware::auth::AuthLayer;
use crate::middleware::metrics::apply_request_telemetry;
use crate::observability::metrics as telemetry;
use crate::openapi;
use crate::queue::backlog::{LaneBacklogSnapshot, LaneBacklogSource};
use crate::queue::dispatch::WorkDispatcher;
use crate::state::config_epoch::ConfigEpoch;
use crate::state::demand_tracker::{DemandTracker, PhysicalLane, PhysicalLaneCatalog};
use crate::state::model_registry::ModelRegistry;
use crate::state::pool_manager::{CapacityPoolSnapshot, PoolManager, DEFAULT_POOL_NAME};
use crate::state::worker_registry::WorkerRegistry;
use crate::types::PoolState;
use tracing::warn;

/// A model served from a dedicated sandbox rather than a catalog bundle (#1841).
///
/// The managed service returns one of these from [`ModelAccessPolicy::sealed_route`]
/// for an org-registered custom model: it carries the routing tuple the dispatch
/// path would otherwise derive from a resolved bundle, so the proxy skips bundle /
/// profile resolution (a custom model has no bundle) and dispatches straight to the
/// sealed lane. Every value is supplied by the managed policy — the OSS proxy only
/// learns the generic rule "if a route is returned, use it verbatim." The
/// load-bearing field is `engine`: it is the SOLE discriminator the dispatcher
/// co-process uses to route a request to the per-org sealed sandbox.
pub struct SealedRoute {
    /// The wire engine marker (`"sealed"`) — the dispatcher co-process routes on it.
    pub engine: String,
    /// Synthetic bundle label; matches the registered sealed worker lane.
    pub bundle: String,
    /// Synthetic admission pool; matches the registered sealed worker lane.
    pub pool: String,
    /// Synthetic machine profile; the middle segment of the dispatch lane string.
    pub machine_profile: String,
}

/// A deployment hook to keep certain models visible/servable only to authorized
/// callers, decided on the RESOLVED model id and the request extensions (which
/// carry the caller identity). The OSS self-host build installs no policy (every
/// model is global); the managed service supplies one that scopes org-registered
/// custom models to their owning org (#1841). It is consulted at model
/// resolution (the serve gate) AND at the `/v1/models` listing (the enumeration
/// filter), so a model hidden from a caller is indistinguishable from an absent
/// one — no cross-org existence oracle.
pub trait ModelAccessPolicy: Send + Sync {
    /// Whether `resolved_model` (the canonical, alias/`__`/case-resolved id) is
    /// visible to the caller identified in `ext`. `false` ⇒ treat as not found.
    fn visible(&self, resolved_model: &str, ext: &axum::http::Extensions) -> bool;

    /// If `resolved_model` is served from a dedicated sandbox (a #1841 org custom
    /// model) rather than a catalog bundle, return its [`SealedRoute`]. Consulted
    /// ONLY after [`Self::visible`] has passed, so the caller already owns any
    /// org-scoped model — implementations MUST NOT re-derive ownership here. The
    /// default (catalog-only OSS build) returns `None`, so with no policy installed
    /// there are no sealed models by construction.
    fn sealed_route(
        &self,
        _resolved_model: &str,
        _ext: &axum::http::Extensions,
    ) -> Option<SealedRoute> {
        None
    }
}

pub struct AppState {
    pub registry: Arc<WorkerRegistry>,
    pub config: Arc<Config>,
    pub model_registry: Arc<ModelRegistry>,
    pub pool_manager: Arc<PoolManager>,
    pub work_publisher: Option<Arc<dyn WorkDispatcher>>,
    pub lane_backlog_source: Option<Arc<dyn LaneBacklogSource>>,
    pub demand_tracker: Arc<DemandTracker>,
    /// Optional model-visibility policy (managed service; `None` = OSS, all
    /// models global). See [`ModelAccessPolicy`].
    pub model_access_policy: Option<Arc<dyn ModelAccessPolicy>>,
    /// Monotonic view of the furthest-known control-plane epoch. Written by
    /// bootstrap, the NATS delta handler, and the epoch poller; read by
    /// `GET /v1/configs/models/{id}/status`.
    pub config_epoch: ConfigEpoch,
}

const KEDA_CAPACITY_RECONCILE_INTERVAL: Duration = Duration::from_secs(5);

/// Whether this gateway composition owns an authoritative durable queue
/// backlog. Managed Modal dispatch has no JetStream queue; it still exports
/// demand/lease/floor snapshots but must omit queue depth and freshness without
/// reporting an operational failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaneBacklogMode {
    JetStreamRequired,
    // Constructed by the managed gateway composition in the sibling crate.
    #[allow(dead_code)]
    NotApplicable,
}

impl LaneBacklogMode {
    fn requires_jetstream(self) -> bool {
        matches!(self, Self::JetStreamRequired)
    }
}
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn keda_capacity_reconciler_interval(metrics_exporter_enabled: bool) -> Option<Duration> {
    metrics_exporter_enabled.then_some(KEDA_CAPACITY_RECONCILE_INTERVAL)
}

fn keda_capacity_reconciler_jitter(seed: &str, interval: Duration) -> Duration {
    let interval_nanos = u64::try_from(interval.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1);
    let hash = seed.bytes().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    });
    Duration::from_nanos(hash % interval_nanos)
}

/// Start the one shared KEDA capacity loop used by OSS and managed gateway
/// compositions. With no metrics exporter this returns before spawning, so
/// telemetry-disabled gateways pay no reconciliation or task cost.
pub fn spawn_keda_capacity_reconciler(
    state: Arc<AppState>,
    lane_backlog_mode: LaneBacklogMode,
) -> Option<tokio::task::JoinHandle<()>> {
    let interval_duration = keda_capacity_reconciler_interval(
        crate::observability::tracing::metrics_exporter_enabled(),
    )?;
    let initial_delay = keda_capacity_reconciler_jitter(
        &crate::observability::tracing::process_start_uuid(),
        interval_duration,
    );
    Some(tokio::spawn(async move {
        // Keep each process on a stable offset within the cadence. Replica
        // startups often coincide during a rollout; staggering both the first
        // and subsequent ticks prevents every gateway from issuing its
        // bounded JetStream lane scan in the same burst.
        let first_tick = tokio::time::Instant::now() + initial_delay;
        let mut interval = tokio::time::interval_at(first_tick, interval_duration);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            record_keda_capacity_snapshot(&state, lane_backlog_mode).await;
        }
    }))
}

/// Publish the gateway-owned KEDA state through the canonical OTel facade.
/// Capture reconciliation start time before any async registry reads. The
/// timestamp is recorded only after the state values, so a slow reconciliation
/// publishes an already-aged freshness marker and fails closed.
pub async fn record_keda_capacity_snapshot(state: &AppState, lane_backlog_mode: LaneBacklogMode) {
    let snapshot_started_unix_time_s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    match keda_capacity_snapshot(state, lane_backlog_mode).await {
        Ok(snapshot) => {
            telemetry::record_keda_capacity_snapshot(&snapshot, snapshot_started_unix_time_s)
        }
        Err(error) => warn!(
            error,
            "KEDA capacity reconciliation incomplete; retaining the prior values without refreshing freshness"
        ),
    }
}

/// Build the authoritative KEDA snapshot independently of the telemetry SDK.
/// Keeping this as a plain value makes control-path semantics directly
/// testable before the facade translates the event to OTel instruments.
pub(crate) async fn keda_capacity_snapshot(
    state: &AppState,
    lane_backlog_mode: LaneBacklogMode,
) -> Result<telemetry::KedaCapacitySnapshot, String> {
    let catalog = state.demand_tracker.catalog();
    let lane_backlogs = if catalog.is_empty() {
        LaneBacklogSnapshot::default()
    } else {
        match state.lane_backlog_source.as_ref() {
            Some(source) => match source.snapshot().await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    // Broker availability is lane-local to queue freshness.
                    // Keep the independent demand/lease/floor control state
                    // fresh while every missing queue lane ages out.
                    warn!(
                        error = %error,
                        failed_lanes = catalog.len(),
                        "JetStream lane backlog source failed; queue lanes will age out independently"
                    );
                    LaneBacklogSnapshot::default()
                }
            },
            None => {
                if lane_backlog_mode.requires_jetstream() {
                    warn!(
                        failed_lanes = catalog.len(),
                        "physical KEDA lanes configured without a JetStream backlog source; queue lanes will age out"
                    );
                }
                LaneBacklogSnapshot::default()
            }
        }
    };
    if let Some((lane, detail)) = lane_backlogs.failures().first_key_value() {
        warn!(
            failed_lanes = lane_backlogs.failures().len(),
            first_failed_lane = %lane,
            first_error = %detail,
            "JetStream lane backlog reconciliation was partial; successful lanes remain fresh"
        );
    }
    let pools = state.pool_manager.capacity_pools().await;
    let pending_lanes = state.demand_tracker.active_lanes();
    build_keda_capacity_snapshot(catalog, &lane_backlogs, &pools, pending_lanes)
}

fn build_keda_capacity_snapshot(
    catalog: &PhysicalLaneCatalog,
    lane_backlogs: &LaneBacklogSnapshot,
    pools: &[CapacityPoolSnapshot],
    pending_lanes: Vec<PhysicalLane>,
) -> Result<telemetry::KedaCapacitySnapshot, String> {
    if lane_backlogs
        .values()
        .keys()
        .any(|lane| !catalog.contains(lane))
        || lane_backlogs
            .failures()
            .keys()
            .any(|lane| !catalog.contains(lane))
    {
        return Err(format!(
            "JetStream backlog snapshot contains a lane outside the physical catalog (snapshot={}, catalog={})",
            lane_backlogs.len(),
            catalog.len()
        ));
    }
    let catalog_lanes = catalog.lanes();
    let lane_queue_depth = catalog_lanes
        .iter()
        .filter_map(|lane| {
            let value = lane_backlogs.get(lane).copied()?;
            Some(telemetry::LaneSnapshot {
                pool: lane.pool().to_string(),
                machine_profile: lane.machine_profile().to_string(),
                bundle: lane.bundle().to_string(),
                value: value as f64,
            })
        })
        .collect();
    let pending_lanes: HashSet<_> = pending_lanes.into_iter().collect();
    let pending_demand = complete_lane_snapshot(&catalog_lanes, |lane| {
        if pending_lanes.contains(lane) {
            1.0
        } else {
            0.0
        }
    });
    let pool_warm_floor_by_lane: HashMap<_, _> =
        crate::state::warm_floor::warm_floor_values_from_capacity(pools)
            .into_iter()
            .filter_map(|value| {
                let lane = catalog.resolve(&value.pool, &value.machine_profile, &value.bundle)?;
                Some((lane, f64::from(value.value)))
            })
            .collect();
    let pool_warm_floor = complete_lane_snapshot(&catalog_lanes, |lane| {
        pool_warm_floor_by_lane.get(lane).copied().unwrap_or(0.0)
    });
    Ok(telemetry::KedaCapacitySnapshot {
        pending_demand,
        lane_queue_depth,
        active_lease_gpus: active_lease_values(pools, catalog, &catalog_lanes),
        pool_warm_floor,
    })
}

fn complete_lane_snapshot(
    catalog_lanes: &[PhysicalLane],
    mut value_for: impl FnMut(&PhysicalLane) -> f64,
) -> Vec<telemetry::LaneSnapshot> {
    catalog_lanes
        .iter()
        .map(|lane| telemetry::LaneSnapshot {
            pool: lane.pool().to_string(),
            machine_profile: lane.machine_profile().to_string(),
            bundle: lane.bundle().to_string(),
            value: value_for(lane).max(0.0),
        })
        .collect()
}

fn active_lease_values(
    pools: &[CapacityPoolSnapshot],
    catalog: &PhysicalLaneCatalog,
    catalog_lanes: &[PhysicalLane],
) -> Vec<telemetry::LaneSnapshot> {
    let mut active_workers_by_lane: HashMap<PhysicalLane, HashSet<String>> = HashMap::new();
    for pool in pools {
        if pool.name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) || pool.state != PoolState::Active {
            continue;
        }
        for worker in &pool.assigned_workers {
            let worker_key = if worker.name.trim().is_empty() {
                worker.url.clone()
            } else {
                worker.name.clone()
            };
            let bundle = if worker.bundle.trim().is_empty() {
                "default"
            } else {
                worker.bundle.as_str()
            };
            let Some(lane) = catalog.resolve(&pool.queue_pool, &worker.gpu, bundle) else {
                continue;
            };
            active_workers_by_lane
                .entry(lane)
                .or_default()
                .insert(worker_key);
        }
    }

    complete_lane_snapshot(catalog_lanes, |lane| {
        active_workers_by_lane
            .get(lane)
            .map_or(0.0, |workers| workers.len() as f64)
    })
}

pub fn create_router(state: Arc<AppState>, config: Arc<Config>) -> Router {
    apply_request_telemetry(create_router_core(state, config))
}

/// Build the reusable gateway route stack without the outer request telemetry
/// layer.
///
/// Managed compositions add admission middleware and cloud-owned routes around
/// this core, then call [`apply_request_telemetry`] after the final merge so
/// early 401/402/403 responses are observed whenever a request signal is live.
/// Standalone callers should use [`create_router`], which preserves the OSS
/// composition and conditionally adds that layer.
pub fn create_router_core(state: Arc<AppState>, config: Arc<Config>) -> Router {
    Router::new()
        // Status page
        .route("/", get(health::status_page))
        // Health endpoints
        .route("/healthz", get(health::healthz))
        .route("/readyz", get(health::readyz))
        .route("/health", get(health::health))
        // API description
        .route("/openapi.json", get(openapi::openapi_json))
        // Rendered API reference (Redoc) + its vendored, self-contained bundle.
        .route("/docs", get(openapi::docs_ui))
        .route("/docs/redoc.standalone.js", get(openapi::redoc_asset))
        // Models endpoints. `{*model}` accepts slash-bearing IDs.
        .route("/v1/models", get(models::get_models))
        .route("/v1/models/{*model}", get(models::get_model))
        // Pool endpoints
        .route("/v1/pools", post(pools::create_pool))
        .route("/v1/pools", get(pools::list_pools))
        .route("/v1/pools/{name}", get(pools::get_pool))
        .route("/v1/pools/{name}", delete(pools::delete_pool))
        .route("/v1/pools/{name}/renew", post(pools::renew_pool))
        // Config API endpoints. Gateway is read-only; config writes go to
        // `packages/sie_config`. `POST /v1/configs/models` is not
        // registered, so axum returns `405 Method Not Allowed` for write
        // attempts against the gateway (no shim, no redirect body).
        //
        // The `{*id}` wildcard exists because model IDs commonly contain
        // slashes (e.g. `BAAI/bge-m3`). A trailing `/status` suffix on
        // the same wildcard is disambiguated inside
        // `get_model_config_or_status` — see that handler's doc comment
        // for the routing contract.
        .route("/v1/configs/models", get(config_api::get_model_configs))
        .route(
            "/v1/configs/models/{*id}",
            get(config_api::get_model_config_or_status),
        )
        .route("/v1/configs/bundles", get(config_api::get_bundle_configs))
        .route(
            "/v1/configs/bundles/{id}",
            get(config_api::get_bundle_config),
        )
        .route("/v1/configs/resolve", post(config_api::resolve_config))
        // WebSocket cluster status
        .route("/ws/cluster-status", get(health::ws_cluster_status))
        .route(
            "/v1/audio/transcriptions",
            post(audio::proxy_openai_transcription)
                .layer(DefaultBodyLimit::max(audio::MAX_MULTIPART_BYTES)),
        )
        .route("/v1/embeddings", post(proxy::proxy_openai_embeddings))
        .route("/v1/rerank", post(proxy::proxy_rerank))
        .route("/v2/rerank", post(proxy::proxy_rerank_v2))
        .route("/v1/chat/completions", post(proxy::proxy_chat))
        // OpenAI legacy Completions — raw-prompt continuation (non-streaming).
        .route("/v1/completions", post(proxy::proxy_completions))
        // OpenAI Responses API (MVP: string input, non-streaming).
        .route("/v1/responses", post(proxy::proxy_responses))
        // OpenAI moderations surface — registered but not implemented (501)
        // until a moderation model + governance store land (Tier 0).
        .route("/v1/moderations", post(proxy::proxy_moderations))
        // Proxy endpoints - use wildcard for model path
        .route("/v1/encode/{*model}", post(proxy::proxy_encode))
        .route("/v1/score/{*model}", post(proxy::proxy_score))
        .route("/v1/extract/{*model}", post(proxy::proxy_extract))
        .route("/v1/generate/{*model}", post(proxy::proxy_generate))
        .layer(AuditLayer::new())
        .layer(AuthLayer::new(config))
        .with_state(state)
}

#[cfg(test)]
mod capacity_snapshot_tests {
    use std::hint::black_box;
    use std::time::Instant;

    use super::*;
    use crate::state::demand_tracker::{PhysicalLane, PhysicalLaneCatalog};
    use crate::types::pool::{AssignedWorker, Pool, PoolSpec, PoolStatus};

    #[test]
    fn keda_capacity_reconciler_has_no_disabled_task_plan() {
        assert_eq!(keda_capacity_reconciler_interval(false), None);
        assert_eq!(
            keda_capacity_reconciler_interval(true),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn managed_backlog_mode_declares_jetstream_not_applicable() {
        assert!(LaneBacklogMode::JetStreamRequired.requires_jetstream());
        assert!(!LaneBacklogMode::NotApplicable.requires_jetstream());
    }

    #[test]
    fn keda_capacity_reconciler_staggers_processes_within_one_interval() {
        let interval = Duration::from_secs(5);
        let first = keda_capacity_reconciler_jitter("gateway-process-a", interval);
        let second = keda_capacity_reconciler_jitter("gateway-process-b", interval);

        assert!(first < interval);
        assert!(second < interval);
        assert_ne!(first, second);
        assert_eq!(
            first,
            keda_capacity_reconciler_jitter("gateway-process-a", interval)
        );
    }

    fn lane_catalog(lanes: &[(&str, &str, &str)]) -> PhysicalLaneCatalog {
        PhysicalLaneCatalog::try_new(
            lanes.iter().map(|(pool, profile, bundle)| {
                PhysicalLane::try_new(pool, profile, bundle).unwrap()
            }),
        )
        .unwrap()
    }

    fn pool(
        name: &str,
        state: PoolState,
        queue_pool: &str,
        workers: Vec<AssignedWorker>,
    ) -> CapacityPoolSnapshot {
        CapacityPoolSnapshot::from_pool(&Pool {
            spec: PoolSpec {
                name: name.to_string(),
                queue_pool: queue_pool.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
            },
            status: PoolStatus {
                state,
                assigned_workers: workers,
                ..Default::default()
            },
        })
    }

    #[test]
    fn active_leases_use_physical_lane_and_deduplicate_workers() {
        let catalog = lane_catalog(&[("shared", "l4", "default")]);
        let catalog_lanes = catalog.lanes();
        let worker = AssignedWorker {
            name: "worker-1".to_string(),
            url: "http://worker-1".to_string(),
            gpu: "L4".to_string(),
            bundle: "DEFAULT".to_string(),
        };
        assert_eq!(
            active_lease_values(
                &[
                    pool(
                        "tenant-a",
                        PoolState::Active,
                        "SHARED",
                        vec![worker.clone()],
                    ),
                    pool("tenant-b", PoolState::Active, "shared", vec![worker]),
                    pool("pending", PoolState::Pending, "shared", vec![]),
                    pool("DEFAULT", PoolState::Active, "", vec![]),
                ],
                &catalog,
                &catalog_lanes,
            ),
            vec![telemetry::LaneSnapshot {
                pool: "shared".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: 1.0,
            }]
        );
    }

    #[test]
    fn active_lease_outside_physical_catalog_cannot_create_nonzero_state() {
        let catalog = lane_catalog(&[("default", "l4", "default")]);
        let catalog_lanes = catalog.lanes();
        let attacker_worker = AssignedWorker {
            name: "worker-attacker".to_string(),
            url: "http://worker-attacker".to_string(),
            gpu: "caller-profile-999".to_string(),
            bundle: "default".to_string(),
        };

        assert_eq!(
            active_lease_values(
                &[pool(
                    "tenant",
                    PoolState::Active,
                    "caller-pool-999",
                    vec![attacker_worker],
                )],
                &catalog,
                &catalog_lanes,
            ),
            vec![telemetry::LaneSnapshot {
                pool: "default".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: 0.0,
            }]
        );
    }

    #[test]
    fn cold_start_seeds_one_zero_per_catalog_lane_for_nonqueue_state() {
        let catalog = lane_catalog(&[("default", "l4", "default"), ("shared", "h100", "sglang")]);

        let snapshot = build_keda_capacity_snapshot(
            &catalog,
            &LaneBacklogSnapshot::default(),
            &[],
            Vec::new(),
        )
        .unwrap();
        let expected = vec![
            telemetry::LaneSnapshot {
                pool: "default".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: 0.0,
            },
            telemetry::LaneSnapshot {
                pool: "shared".to_string(),
                machine_profile: "h100".to_string(),
                bundle: "sglang".to_string(),
                value: 0.0,
            },
        ];

        assert_eq!(snapshot.pending_demand, expected);
        assert_eq!(snapshot.active_lease_gpus, expected);
        assert_eq!(snapshot.pool_warm_floor, expected);
        assert!(snapshot.lane_queue_depth.is_empty());

        let (_, exported_points) = telemetry::benchmark_keda_capacity_emit_export(&snapshot);
        assert_eq!(exported_points, catalog.len() * 3);
    }

    #[test]
    fn broker_backlogs_preserve_exact_lane_labels_and_explicit_zero() {
        let catalog = lane_catalog(&[("default", "l4", "default"), ("shared", "h100", "sglang")]);
        let backlogs = LaneBacklogSnapshot::from([
            (
                PhysicalLane::try_new("default", "l4", "default").unwrap(),
                7,
            ),
            (
                PhysicalLane::try_new("shared", "h100", "sglang").unwrap(),
                0,
            ),
        ]);

        let snapshot = build_keda_capacity_snapshot(&catalog, &backlogs, &[], Vec::new()).unwrap();
        assert_eq!(
            snapshot.lane_queue_depth,
            vec![
                telemetry::LaneSnapshot {
                    pool: "default".to_string(),
                    machine_profile: "l4".to_string(),
                    bundle: "default".to_string(),
                    value: 7.0,
                },
                telemetry::LaneSnapshot {
                    pool: "shared".to_string(),
                    machine_profile: "h100".to_string(),
                    bundle: "sglang".to_string(),
                    value: 0.0,
                },
            ]
        );
    }

    #[test]
    fn incomplete_broker_snapshot_ages_out_only_missing_lane() {
        let catalog = lane_catalog(&[("default", "l4", "default")]);
        let snapshot = build_keda_capacity_snapshot(
            &catalog,
            &LaneBacklogSnapshot::default(),
            &[],
            Vec::new(),
        )
        .unwrap();
        assert!(snapshot.lane_queue_depth.is_empty());
    }

    #[test]
    fn no_queue_topology_retains_nonqueue_capacity_state() {
        let catalog = lane_catalog(&[("default", "l4", "default")]);
        let lane = catalog.resolve("default", "l4", "default").unwrap();
        let snapshot = build_keda_capacity_snapshot(
            &catalog,
            &LaneBacklogSnapshot::default(),
            &[],
            vec![lane],
        )
        .unwrap();

        assert!(snapshot.lane_queue_depth.is_empty());
        assert_eq!(snapshot.pending_demand.len(), 1);
    }

    #[test]
    fn foreign_broker_snapshot_fails_closed() {
        let catalog = lane_catalog(&[("default", "l4", "default")]);
        let foreign = LaneBacklogSnapshot::from([(
            PhysicalLane::try_new("attacker", "l4", "default").unwrap(),
            99,
        )]);
        assert!(build_keda_capacity_snapshot(&catalog, &foreign, &[], Vec::new()).is_err());
    }

    #[test]
    fn one_corrupt_broker_lane_does_not_suppress_healthy_or_global_state() {
        let catalog = lane_catalog(&[("healthy", "l4", "default"), ("corrupt", "h100", "sglang")]);
        let healthy = catalog.resolve("healthy", "l4", "default").unwrap();
        let corrupt = catalog.resolve("corrupt", "h100", "sglang").unwrap();
        let partial = LaneBacklogSnapshot::from([(healthy, 4)]);

        let snapshot =
            build_keda_capacity_snapshot(&catalog, &partial, &[], vec![corrupt.clone()]).unwrap();

        assert_eq!(
            snapshot.lane_queue_depth,
            vec![telemetry::LaneSnapshot {
                pool: "healthy".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                value: 4.0,
            }]
        );
        assert_eq!(
            snapshot.pending_demand,
            vec![
                telemetry::LaneSnapshot {
                    pool: corrupt.pool().to_string(),
                    machine_profile: corrupt.machine_profile().to_string(),
                    bundle: corrupt.bundle().to_string(),
                    value: 1.0,
                },
                telemetry::LaneSnapshot {
                    pool: "healthy".to_string(),
                    machine_profile: "l4".to_string(),
                    bundle: "default".to_string(),
                    value: 0.0,
                },
            ],
            "independent demand state must remain publishable"
        );
    }

    /// Release-only raw wall-clock benchmark for realistic compact capacity
    /// state: 1,024 physical lanes and two logical pools per lane. It includes
    /// broker-view cloning, semantic snapshot construction,
    /// OTel recording, and one exporter flush. Lock wait and allocator counts
    /// are not instrumented and must not be inferred from the output.
    ///
    /// One invocation collects three independently warmed samples:
    /// `mise exec -- cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib keda_capacity_build_emit_export_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    fn keda_capacity_build_emit_export_microbenchmark() {
        const SAMPLES: usize = 3;
        const LANES: usize = crate::state::demand_tracker::MAX_CONFIGURED_PHYSICAL_LANES;
        const LOGICAL_POOLS_PER_LANE: usize = 2;
        const BUILD_ITERATIONS: usize = 10;

        let lanes: Vec<_> = (0..LANES)
            .map(|index| PhysicalLane::try_new(&format!("pool-{index}"), "l4", "default").unwrap())
            .collect();
        let catalog = PhysicalLaneCatalog::try_new(lanes.iter().cloned()).unwrap();
        let backlogs: LaneBacklogSnapshot = lanes
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, lane)| (lane, (index % 11) as u64))
            .collect();
        let pools: Vec<_> = (0..LANES)
            .flat_map(|lane_index| {
                (0..LOGICAL_POOLS_PER_LANE).map(move |pool_index| CapacityPoolSnapshot {
                    name: format!("tenant-{lane_index}-{pool_index}"),
                    queue_pool: format!("pool-{lane_index}"),
                    bundle: "default".to_string(),
                    machine_profiles: vec!["l4".to_string()],
                    minimum_worker_count: 1,
                    state: PoolState::Active,
                    assigned_workers: vec![AssignedWorker {
                        name: format!("worker-{lane_index}-{pool_index}"),
                        url: format!("http://worker-{lane_index}-{pool_index}"),
                        gpu: "l4".to_string(),
                        bundle: "default".to_string(),
                    }],
                })
            })
            .collect();

        let mut snapshot = telemetry::KedaCapacitySnapshot::default();
        let mut build_samples = [0.0; SAMPLES];
        for sample in &mut build_samples {
            let backlog_view = black_box(backlogs.clone());
            let pool_view = black_box(pools.clone());
            let _ = build_keda_capacity_snapshot(
                black_box(&catalog),
                black_box(&backlog_view),
                black_box(&pool_view),
                black_box(lanes.clone()),
            )
            .unwrap();

            let build_started = Instant::now();
            for _ in 0..BUILD_ITERATIONS {
                let backlog_view = black_box(backlogs.clone());
                let pool_view = black_box(pools.clone());
                snapshot = build_keda_capacity_snapshot(
                    black_box(&catalog),
                    black_box(&backlog_view),
                    black_box(&pool_view),
                    black_box(lanes.clone()),
                )
                .unwrap();
            }
            *sample = build_started.elapsed().as_secs_f64() * 1_000.0 / BUILD_ITERATIONS as f64;
        }
        assert_eq!(snapshot.pending_demand.len(), LANES);
        assert_eq!(snapshot.lane_queue_depth.len(), LANES);
        assert_eq!(snapshot.active_lease_gpus.len(), LANES);
        assert_eq!(snapshot.pool_warm_floor.len(), LANES);

        let mut emit_export_samples = [0.0; SAMPLES];
        for sample in &mut emit_export_samples {
            let (_, warmup_points) = telemetry::benchmark_keda_capacity_emit_export(&snapshot);
            assert_eq!(warmup_points, LANES * 5);
            let (elapsed, exported_points) =
                telemetry::benchmark_keda_capacity_emit_export(&snapshot);
            assert_eq!(exported_points, LANES * 5);
            *sample = elapsed.as_secs_f64() * 1_000.0;
        }
        let build_median_ms = telemetry::telemetry_benchmark_median(build_samples);
        let emit_export_median_ms = telemetry::telemetry_benchmark_median(emit_export_samples);
        println!(
            "gateway_keda_capacity_build_emit_export lanes={LANES} broker_lane_values={} logical_pools={} samples={SAMPLES} build_iterations_per_sample={BUILD_ITERATIONS} compact_clone_and_build_ms_per_snapshot={build_samples:?} compact_clone_and_build_median_ms_per_snapshot={build_median_ms:.3} emit_and_force_flush_ms={emit_export_samples:?} emit_and_force_flush_median_ms={emit_export_median_ms:.3} exported_points={} lock_wait=excluded allocation_measurement=not_instrumented",
            backlogs.len(),
            pools.len(),
            LANES * 5,
        );
    }
}

#[cfg(test)]
mod flat_404_tests {
    //! Flat-404 wire contract for managed-service routes (#1757).
    //!
    //! The Files, Batches, batch-cancel, and file-upload surfaces
    //! (`/v1/files*`, `/v1/batches*`, `/v1/batches/{id}/cancel`,
    //! `POST /v1/files`) are OpenAI-compatible routes the *managed
    //! service* fronts (see `sie_tools`/`sie_sdk` `.files`/`.batches`);
    //! the inference-edge gateway does NOT back them. Because they are
    //! not registered in [`create_router`] and there is no custom
    //! `.fallback()`, axum's default fallback answers them with a
    //! **flat 404**: status `404 Not Found` and an **empty body**.
    //!
    //! That is the contract these tests pin: an unbacked managed-service
    //! route returns a clean 404 with no body — never a leaky/verbose
    //! error envelope and never a 500. Contrast `config_api`'s
    //! `test_post_model_config_returns_405_method_not_allowed`, which
    //! covers a *registered* path hit with an unbacked method (405); an
    //! entirely unregistered path 404s instead, which is what these
    //! routes must do.

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use tower::ServiceExt;

    use crate::config::Config;
    use crate::server::{create_router, AppState};
    use crate::state::demand_tracker::DemandTracker;
    use crate::state::model_registry::ModelRegistry;
    use crate::state::pool_manager::PoolManager;
    use crate::state::worker_registry::WorkerRegistry;

    fn test_config(bundles_dir: &str, models_dir: &str) -> Config {
        Config {
            host: "127.0.0.1".to_string(),
            port: 0,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: "default".to_string(),
            k8s_service: "sie-worker".to_string(),
            k8s_port: 8080,
            health_mode: "ws".to_string(),
            nats_url: String::new(),
            nats_config_trusted_producers: vec!["sie-config".to_string()],
            // Auth disabled so requests reach the router and exercise the
            // route table itself — the flat 404 must come from the
            // unmatched-route fallback, not from an auth 401/403.
            auth_mode: "none".to_string(),
            auth_tokens: Vec::new(),
            admin_token: String::new(),
            auth_exempt_operational: false,
            log_level: "info".to_string(),
            json_logs: false,
            enable_pools: false,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 30.0,
            max_stream_pending: 50_000,
            stream_max_age_s: 1_800,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            configured_physical_lanes: Default::default(),
            static_queue_pools: Vec::new(),
            model_aliases: HashMap::new(),
            bundles_dir: bundles_dir.to_string(),
            models_dir: models_dir.to_string(),
            payload_store_url: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_modal_proxy_token: None,
            public_base_url: None,
        }
    }

    // Returned tempdirs must outlive the router so they drop after the test.
    fn build_router() -> (Router, tempfile::TempDir, tempfile::TempDir) {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let config = Arc::new(test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
        ));
        let state = Arc::new(AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::clone(&config),
            model_registry: Arc::new(ModelRegistry::new(
                bundles_dir.path(),
                models_dir.path(),
                false,
            )),
            pool_manager: Arc::new(PoolManager::new(Vec::new())),
            work_publisher: None,
            lane_backlog_source: None,
            demand_tracker: Arc::new(DemandTracker::new(Default::default())),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
            model_access_policy: None,
        });
        let router = create_router(Arc::clone(&state), config);
        (router, bundles_dir, models_dir)
    }

    /// Drive `method uri` (with `body`) through the real router and assert
    /// the flat-404 contract: status `404 Not Found` with an empty body.
    /// `app` is cloned per call so one test can probe several routes.
    async fn assert_flat_404(app: &Router, method: &str, uri: &str, body: Body) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(body)
                    .unwrap(),
            )
            .await
            .unwrap();

        let status = response.status();
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "{method} {uri} must return a flat 404 (unbacked managed-service route), got {status}",
        );
        // Explicitly guard against a 5xx masquerading — the route must be
        // rejected by the fallback, never reach a handler that could 500.
        assert!(
            !status.is_server_error(),
            "{method} {uri} returned a server error {status}; must be a flat 404",
        );

        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(
            bytes.is_empty(),
            "{method} {uri} flat 404 must have an empty body (no leaky/verbose error), got {} bytes: {:?}",
            bytes.len(),
            String::from_utf8_lossy(&bytes),
        );
    }

    #[tokio::test]
    async fn test_files_routes_return_flat_404() {
        let (app, _bundles_dir, _models_dir) = build_router();
        // list, retrieve, download-content, delete — none are backed here.
        assert_flat_404(&app, "GET", "/v1/files", Body::empty()).await;
        assert_flat_404(&app, "GET", "/v1/files/file-nonexistent", Body::empty()).await;
        assert_flat_404(
            &app,
            "GET",
            "/v1/files/file-nonexistent/content",
            Body::empty(),
        )
        .await;
        assert_flat_404(&app, "DELETE", "/v1/files/file-nonexistent", Body::empty()).await;
    }

    #[tokio::test]
    async fn test_batches_routes_return_flat_404() {
        let (app, _bundles_dir, _models_dir) = build_router();
        // list + retrieve — the batch store is not fronted by the gateway.
        assert_flat_404(&app, "GET", "/v1/batches", Body::empty()).await;
        assert_flat_404(&app, "GET", "/v1/batches/batch-nonexistent", Body::empty()).await;
    }

    #[tokio::test]
    async fn test_batch_cancel_route_returns_flat_404() {
        let (app, _bundles_dir, _models_dir) = build_router();
        // POST /v1/batches/{id}/cancel — the OpenAI-parity cancel verb.
        assert_flat_404(
            &app,
            "POST",
            "/v1/batches/batch-nonexistent/cancel",
            Body::empty(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_file_upload_route_returns_flat_404() {
        let (app, _bundles_dir, _models_dir) = build_router();
        // POST /v1/files — the upload verb. A well-formed body still gets a
        // flat 404 because the route is unregistered (rejected before any
        // handler / body parsing runs).
        assert_flat_404(
            &app,
            "POST",
            "/v1/files?purpose=batch",
            Body::from("{\"purpose\":\"batch\"}"),
        )
        .await;
    }
}
