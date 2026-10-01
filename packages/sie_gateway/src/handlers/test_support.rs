//! Handler test fixtures: a gateway with a local lane and a remote lane, and a
//! dispatcher that records what it publishes and answers every request.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::sync::{broadcast, oneshot, Notify};

use crate::config::{Config, StreamStorage};
use crate::queue::dispatch::{
    ChunkEnvelope, DispatchDurability, DispatchError, PendingGenerationSnapshot, PublishTarget,
    StreamOutcome, WorkDispatcher, WorkParams, WorkResult,
};
use crate::queue::streaming::{ChunkApplied, StreamCollector};
use crate::server::AppState;
use crate::state::config_epoch::ConfigEpoch;
use crate::state::demand_tracker::{DemandTracker, PhysicalLane, PhysicalLaneCatalog};
use crate::state::model_registry::ModelRegistry;
use crate::state::pool_manager::PoolManager;
use crate::state::worker_registry::WorkerRegistry;
use crate::types::WorkerStatusMessage;

/// The local lane's `(pool, machine_profile, bundle)`.
pub(crate) const LOCAL_LANE: (&str, &str, &str) = ("default", "l4", "default");
/// The remote lane's `(pool, machine_profile, bundle)`.
pub(crate) const REMOTE_LANE: (&str, &str, &str) = ("default", "cpu", "remote");

const DEFAULT_BUNDLE: &str = "name: default\ndefault: true\nadapters:\n  - sie_server.adapters.bert_flash\n  - sie_server.adapters.sglang\n  - sie_server.adapters.remote.sie\n";
const REMOTE_BUNDLE: &str =
    "name: remote\npriority: 1\ndefault: false\nadapters:\n  - sie_server.adapters.remote.sie\n";

/// A local encode model.
pub(crate) const LOCAL_ENCODE_MODEL: &str = "\
sie_id: acme/local
hf_id: acme/local
tasks:
  encode:
    dense:
      dim: 2
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
    max_batch_tokens: 4096
";

/// An encode model with no local weights, served by the `team-sie` upstream.
pub(crate) const REMOTE_ENCODE_MODEL: &str = "\
sie_id: acme/remote
remote_backed: true
tasks:
  encode:
    dense:
      dim: 2
profiles:
  default:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: team-sie
        upstream_model: acme/remote
";

/// A generation model served locally, with a remote profile on `team-sie`.
pub(crate) const HYBRID_GENERATE_MODEL: &str = "\
sie_id: acme/chat
hf_id: acme/chat
tasks:
  generate: {}
profiles:
  default:
    adapter_path: sie_server.adapters.sglang:SGLangAdapter
    max_batch_tokens: 8192
  remote:
    adapter_path: sie_server.adapters.remote.sie:SieUpstreamAdapter
    max_batch_tokens: 8192
    adapter_options:
      loadtime:
        upstream: team-sie
        upstream_model: acme/chat
";

/// One request as the transport received it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Dispatched {
    pub endpoint: String,
    pub pool: String,
    pub machine_profile: String,
    pub bundle: String,
    pub model: String,
}

impl Dispatched {
    fn new(endpoint: &str, target: &PublishTarget) -> Self {
        Self {
            endpoint: endpoint.to_string(),
            pool: target.pool().to_string(),
            machine_profile: target.machine_profile().to_string(),
            bundle: target.bundle().to_string(),
            model: target.model().to_string(),
        }
    }
}

/// A dispatcher that records every publish and answers it at once.
#[derive(Default)]
pub(crate) struct RecordingDispatcher {
    dispatched: Mutex<Vec<Dispatched>>,
}

impl RecordingDispatcher {
    pub(crate) fn dispatched(&self) -> Vec<Dispatched> {
        self.dispatched.lock().unwrap().clone()
    }

    fn record(&self, dispatched: Dispatched) {
        self.dispatched.lock().unwrap().push(dispatched);
    }
}

fn successful_result(request_id: &str, item_index: u32, payload: serde_json::Value) -> WorkResult {
    let mut result: WorkResult = serde_json::from_value(json!({
        "work_item_id": format!("{request_id}.{item_index}"),
        "request_id": request_id,
        "item_index": item_index,
        "success": true,
    }))
    .unwrap();
    result.result_msgpack = rmp_serde::to_vec_named(&payload).unwrap();
    result
}

fn terminal_chunk_collector(
    display_model: &str,
    bundle_config_hash: &str,
) -> (
    oneshot::Receiver<StreamOutcome>,
    broadcast::Receiver<ChunkEnvelope>,
) {
    let (tx, rx) = oneshot::channel();
    let mut collector = StreamCollector::new(tx, display_model.to_string(), "default".to_string());
    let tap = collector.install_chunk_tap();
    let terminal = serde_json::from_value(json!({
        "kind": "chunk", "request_id": "request-1", "attempt_id": "attempt-1",
        "seq": 0, "text_delta": "ok", "done": true, "is_first": true,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "executed_bundle_config_hash": bundle_config_hash,
    }))
    .unwrap();
    assert_eq!(collector.apply(terminal), ChunkApplied::Terminal);
    let outcome = collector.build_outcome().unwrap();
    collector.sender.take().unwrap().send(outcome).unwrap();
    (rx, tap)
}

#[async_trait::async_trait]
impl WorkDispatcher for RecordingDispatcher {
    async fn publish_work(
        self: Arc<Self>,
        target: PublishTarget,
        _admission_pool: &str,
        endpoint: &str,
        _model: &str,
        _engine: &str,
        _bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        _params: &WorkParams,
    ) -> Result<
        (
            String,
            oneshot::Receiver<Vec<WorkResult>>,
            DispatchDurability,
        ),
        DispatchError,
    > {
        self.record(Dispatched::new(endpoint, &target));
        let request_id = "request-1".to_string();
        let results = if endpoint == "score" {
            vec![successful_result(
                &request_id,
                0,
                json!([{"item_id": "0", "score": 0.5, "rank": 0}]),
            )]
        } else {
            (0..items.len() as u32)
                .map(|index| successful_result(&request_id, index, json!({"dense": [0.5, 0.25]})))
                .collect()
        };
        let (tx, rx) = oneshot::channel();
        tx.send(results).unwrap();
        Ok((request_id, rx, DispatchDurability::accepted()))
    }

    async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        _engine: &str,
        bundle_config_hash: &str,
        _params: &WorkParams,
        _admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            Arc<Notify>,
            DispatchDurability,
        ),
        String,
    > {
        self.record(Dispatched::new("generate", &target));
        let (rx, _tap) = terminal_chunk_collector(display_model, bundle_config_hash);
        Ok((
            "request-1".to_string(),
            rx,
            Arc::new(Notify::new()),
            DispatchDurability::accepted(),
        ))
    }

    async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        _engine: &str,
        bundle_config_hash: &str,
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
        self.record(Dispatched::new("generate", &target));
        let (rx, tap) = terminal_chunk_collector(display_model, bundle_config_hash);
        Ok((
            "request-1".to_string(),
            rx,
            tap,
            DispatchDurability::accepted(),
        ))
    }

    async fn publish_cancel(&self, _request_id: &str) {}

    fn begin_work_abandonment(&self, _request_id: &str) -> bool {
        true
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

    fn stream_chunk_timing(&self, _request_id: &str) -> Option<(Option<Instant>, Option<Instant>)> {
        None
    }
}

/// A gateway whose registry holds `models` (model config YAML documents) and
/// whose lane catalog has [`LOCAL_LANE`] and [`REMOTE_LANE`]. No worker is
/// registered until [`TestGateway::add_worker`].
pub(crate) struct TestGateway {
    pub state: Arc<AppState>,
    pub dispatcher: Arc<RecordingDispatcher>,
    _bundles_dir: tempfile::TempDir,
    _models_dir: tempfile::TempDir,
}

impl TestGateway {
    pub(crate) async fn new(models: &[&str]) -> Self {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(bundles_dir.path().join("default.yaml"), DEFAULT_BUNDLE).unwrap();
        std::fs::write(bundles_dir.path().join("remote.yaml"), REMOTE_BUNDLE).unwrap();
        for (index, model) in models.iter().enumerate() {
            std::fs::write(models_dir.path().join(format!("model-{index}.yaml")), model).unwrap();
        }
        let profiles = vec![LOCAL_LANE.1.to_string(), REMOTE_LANE.1.to_string()];
        let lanes =
            PhysicalLaneCatalog::try_new([LOCAL_LANE, REMOTE_LANE].into_iter().map(
                |(pool, profile, bundle)| PhysicalLane::try_new(pool, profile, bundle).unwrap(),
            ))
            .unwrap();
        let pool_manager = Arc::new(PoolManager::new(profiles.clone()));
        pool_manager.create_default_pool().await;
        let dispatcher = Arc::new(RecordingDispatcher::default());
        let state = AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::new(test_config(
                bundles_dir.path().to_str().unwrap(),
                models_dir.path().to_str().unwrap(),
                profiles,
                lanes.clone(),
            )),
            model_registry: Arc::new(ModelRegistry::new(
                bundles_dir.path(),
                models_dir.path(),
                true,
            )),
            pool_manager,
            work_publisher: Some(dispatcher.clone()),
            lane_backlog_source: None,
            demand_tracker: Arc::new(DemandTracker::new(lanes)),
            config_epoch: ConfigEpoch::new(),
            model_access_policy: None,
        };
        Self {
            state: Arc::new(state),
            dispatcher,
            _bundles_dir: bundles_dir,
            _models_dir: models_dir,
        }
    }

    /// Register a healthy worker on `lane` that reports `loaded` as loaded.
    pub(crate) async fn add_worker(&self, name: &str, lane: (&str, &str, &str), loaded: &[&str]) {
        let (pool, machine_profile, bundle) = lane;
        let status = WorkerStatusMessage {
            name: name.to_string(),
            ready: true,
            gpu_count: 1,
            machine_profile: machine_profile.to_string(),
            pool_name: pool.to_string(),
            bundle: bundle.to_string(),
            bundle_config_hash: self
                .state
                .model_registry
                .compute_bundle_config_hash_for_pool(bundle, pool),
            loaded_models: loaded.iter().map(|model| model.to_string()).collect(),
            ..Default::default()
        };
        self.state
            .registry
            .update_worker(&format!("http://{name}:8080"), status)
            .await;
    }
}

fn test_config(
    bundles_dir: &str,
    models_dir: &str,
    profiles: Vec<String>,
    lanes: PhysicalLaneCatalog,
) -> Config {
    Config {
        host: "127.0.0.1".to_string(),
        port: 0,
        worker_urls: Vec::new(),
        use_kubernetes: false,
        k8s_namespace: "default".to_string(),
        k8s_service: String::new(),
        k8s_port: 0,
        health_mode: "nats".to_string(),
        nats_url: String::new(),
        nats_user: String::new(),
        nats_password: String::new(),
        nats_config_trusted_producers: Vec::new(),
        auth_mode: "none".to_string(),
        auth_tokens: Vec::new(),
        admin_token: String::new(),
        auth_exempt_operational: false,
        log_level: "info".to_string(),
        json_logs: false,
        enable_pools: true,
        hot_reload: false,
        watch_polling: false,
        multi_router: false,
        request_timeout: 30.0,
        max_stream_pending: 1024,
        max_lane_in_flight_items: crate::queue::lane_admission::DEFAULT_MAX_LANE_IN_FLIGHT_ITEMS,
        lane_backpressure_enforce: false,
        stream_max_age_s: 300,
        stream_storage: StreamStorage::Memory,
        stream_num_replicas: 1,
        configured_gpus: profiles.clone(),
        gpu_profile_map: profiles
            .into_iter()
            .map(|profile| (profile.clone(), profile))
            .collect::<HashMap<_, _>>(),
        configured_physical_lanes: lanes,
        static_queue_pools: Vec::new(),
        model_aliases: HashMap::new(),
        published_model_aliases: Default::default(),
        bundles_dir: bundles_dir.to_string(),
        models_dir: models_dir.to_string(),
        config_service_url: None,
        config_service_token: None,
        config_modal_proxy_token: None,
        payload_store_url: String::new(),
        public_base_url: None,
    }
}
