use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::Arc;

use crate::server::AppState;
use crate::state::model_registry::ModelRegistry;
use crate::types::model::{ModelEntry, ModelInfoExtras};

/// Stable `created` timestamp stamped on the OpenAI-shaped model
/// objects. SIE has no per-model creation time and OpenAI clients only
/// use the field for display/sorting, so a fixed epoch keeps the
/// response deterministic (matches the value the demo nginx shim used).
const OPENAI_MODEL_CREATED_TS: i64 = 1_700_000_000;
/// `owned_by` value on OpenAI-shaped model objects.
const OPENAI_MODEL_OWNER: &str = "sie";

/// Build one OpenAI-shaped model object (`{id, object, created,
/// owned_by}`). `id` matches the string `/v1/chat/completions` accepts
/// as `model`, so a vanilla OpenAI client can list-then-call.
fn openai_model_object(name: &str) -> Value {
    json!({
        "id": name,
        "object": "model",
        "created": OPENAI_MODEL_CREATED_TS,
        "owned_by": OPENAI_MODEL_OWNER,
    })
}

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "models",
    responses((status = 200, description = "Models visible to this gateway replica", body = crate::openapi::ModelsResponse))
)]
pub async fn get_models(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> impl IntoResponse {
    let ext = req.extensions();
    let model_workers =
        canonical_worker_models(state.registry.get_models().await, &state.model_registry);

    let model_names: BTreeSet<String> = state
        .model_registry
        .list_models()
        .into_iter()
        .chain(model_workers.keys().cloned())
        // #1841 org-scoped visibility: drop models this caller cannot see (other
        // orgs' custom models), so the listing carries no cross-org existence
        // oracle. No-op in OSS self-host (policy is None).
        .filter(|name| {
            state
                .model_access_policy
                .as_ref()
                .is_none_or(|p| p.visible(name, ext))
        })
        .collect();
    let models: Vec<serde_json::Value> = model_names
        .iter()
        .map(|name| {
            let worker_urls = model_workers.get(name).cloned().unwrap_or_default();
            let loaded = !worker_urls.is_empty();
            let mut body = match state.model_registry.get_model_info(name) {
                Some(entry) => entry.to_model_info_value(loaded),
                None => worker_only_model_info(name, loaded),
            };
            attach_model_revision(&mut body, state.as_ref(), name);
            attach_pending_generation(&mut body, state.as_ref(), name);
            body
        })
        .collect();

    // Hybrid response: the native `models` array (consumed by the SIE
    // Python/TS SDKs) plus the OpenAI list shape (`object` + `data`)
    // that vanilla OpenAI clients and Open WebUI expect for model
    // discovery. Emitting both keeps existing SDK consumers working
    // while removing the need for an external OpenAI-shape shim.
    let data: Vec<Value> = model_names
        .iter()
        .map(|name| openai_model_object(name))
        .collect();

    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": data,
            "models": models,
        })),
    )
        .into_response()
}

/// Detail counterpart to `get_models`.
#[utoipa::path(
    get,
    path = "/v1/models/{model}",
    tag = "models",
    params(("model" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients")),
    responses(
        (status = 200, description = "Model detail", body = crate::openapi::ModelInfoWire),
        (status = 404, description = "Model not found", body = crate::openapi::ModelNotFoundResponse)
    )
)]
pub async fn get_model(
    Path(model): Path<String>,
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Response {
    let model_entry = state.model_registry.get_model_info(&model);
    let known_in_registry = model_entry.is_some();
    // Own the canonical name so it stays valid after `model_entry` is
    // moved into the body match below.
    let canonical_model: String = model_entry
        .as_ref()
        .map(|entry| entry.name.clone())
        .unwrap_or_else(|| model.clone());
    // #1841 org-scoped visibility: a model this caller cannot see 404s exactly
    // like an unknown one (no cross-org existence oracle). No-op in OSS.
    if state
        .model_access_policy
        .as_ref()
        .is_some_and(|p| !p.visible(&canonical_model, req.extensions()))
    {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "detail": {
                    "code": "MODEL_NOT_FOUND",
                    "message": format!("Model '{}' not found", model),
                }
            })),
        )
            .into_response();
    }
    let bundles = state.model_registry.get_model_bundles(&model);
    let model_workers =
        canonical_worker_models(state.registry.get_models().await, &state.model_registry);
    let worker_urls = model_workers
        .get(canonical_model.as_str())
        .cloned()
        .unwrap_or_default();

    if !known_in_registry && bundles.is_empty() && worker_urls.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "detail": {
                    "code": "MODEL_NOT_FOUND",
                    "message": format!("Model '{}' not found", model),
                }
            })),
        )
            .into_response();
    }

    let loaded = !worker_urls.is_empty();
    let mut body = match model_entry {
        Some(entry) => entry.to_model_info_value(loaded),
        None => worker_only_model_info(&model, loaded),
    };

    // Additive OpenAI compatibility: merge the OpenAI model-object
    // fields (`id`/`object`/`created`/`owned_by`) into the native
    // detail payload so a vanilla OpenAI client's "retrieve model"
    // call works. `id` is the canonical model name — the same string
    // `/v1/chat/completions` accepts. Native consumers ignore the
    // extra keys.
    if let Some(map) = body.as_object_mut() {
        if let Value::Object(openai_fields) = openai_model_object(&canonical_model) {
            for (k, v) in openai_fields {
                map.entry(k).or_insert(v);
            }
        }
    }
    attach_model_revision(&mut body, state.as_ref(), &canonical_model);
    attach_pending_generation(&mut body, state.as_ref(), &canonical_model);

    (StatusCode::OK, Json(body)).into_response()
}

fn attach_model_revision(body: &mut Value, state: &AppState, model: &str) {
    let Some(revision) = state.model_registry.get_model_revision(model) else {
        return;
    };
    if let Some(map) = body.as_object_mut() {
        map.insert("revision".to_string(), json!(revision));
    }
}

fn attach_pending_generation(body: &mut Value, state: &AppState, model: &str) {
    let pending_generation = state
        .work_publisher
        .as_ref()
        .map(|publisher| publisher.pending_generation_for_model(model))
        .unwrap_or_default();
    if let Some(map) = body.as_object_mut() {
        map.insert("pending_generation".to_string(), json!(pending_generation));
    }
}

fn canonical_worker_models(
    model_workers: HashMap<String, Vec<String>>,
    model_registry: &ModelRegistry,
) -> HashMap<String, Vec<String>> {
    let mut normalized: HashMap<String, Vec<String>> = HashMap::new();
    for (model, urls) in model_workers {
        let canonical = model_registry
            .get_model_info(&model)
            .map(|entry| entry.name)
            .unwrap_or(model);
        normalized.entry(canonical).or_default().extend(urls);
    }
    normalized
}

/// ``ModelInfo``-shaped JSON when workers advertise a model id that is not
/// present in the local registry snapshot (bootstrap / race window).
fn worker_only_model_info(name: &str, loaded: bool) -> Value {
    ModelEntry {
        name: name.to_string(),
        pool: None,
        bundles: Vec::new(),
        adapter_modules: HashSet::new(),
        profile_names: HashSet::new(),
        profile_configs: HashMap::new(),
        info_extras: ModelInfoExtras {
            inputs: Vec::new(),
            outputs: Vec::new(),
            dims: HashMap::new(),
            max_sequence_length: None,
            revision: None,
            max_output_tokens: None,
            grammar_capabilities: None,
            grammar_profile: None,
            profile_parents: HashMap::new(),
            tools_supported: None,
            code: false,
            sql: false,
            guard: false,
            lora_adapters: None,
            profile_lora_adapters: None,
        },
    }
    .to_model_info_value(loaded)
}

#[cfg(test)]
mod route_tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::response::Response;
    use axum::Router;
    use tower::ServiceExt;

    use crate::config::Config;
    use crate::server::{create_router, AppState};
    use crate::state::demand_tracker::DemandTracker;
    use crate::state::model_registry::ModelRegistry;
    use crate::state::pool_manager::PoolManager;
    use crate::state::worker_registry::WorkerRegistry;
    use crate::types::model::{ModelConfig, ProfileConfig};
    use crate::types::worker::WorkerStatusMessage;

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
            public_base_url: None,
            config_service_url: None,
            config_service_token: None,
            config_modal_proxy_token: None,
        }
    }

    // Returned tempdirs must outlive the router so they drop after the test.
    async fn build_router_with_state(
    ) -> (Router, Arc<AppState>, tempfile::TempDir, tempfile::TempDir) {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();
        std::fs::write(
            bundles_dir.path().join("candle.yaml"),
            "name: candle\npriority: 30\nengine: candle\nadapters:\n  - sie_server_rust.adapters.candle\n",
        )
        .unwrap();

        let config = Arc::new(test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
        ));
        let model_registry = Arc::new(ModelRegistry::new(
            bundles_dir.path(),
            models_dir.path(),
            true,
        ));
        let state = Arc::new(AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::clone(&config),
            model_registry,
            pool_manager: Arc::new(PoolManager::new(Vec::new())),
            work_publisher: None,
            lane_backlog_source: None,
            demand_tracker: Arc::new(DemandTracker::new(Default::default())),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
            model_access_policy: None,
        });
        let router = create_router(Arc::clone(&state), config);
        (router, state, bundles_dir, models_dir)
    }

    fn seed_model(state: &AppState, model_id: &str) {
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        state
            .model_registry
            .add_model_config(ModelConfig {
                name: model_id.to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                tasks: None,
                max_sequence_length: None,
            })
            .unwrap();
    }

    fn seed_candle_variant_model(state: &AppState, model_id: &str) {
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        profiles.insert(
            "candle".to_string(),
            ProfileConfig {
                adapter_path: Some(
                    "sie_server_rust.adapters.candle:CandleEmbeddingAdapter".to_string(),
                ),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        state
            .model_registry
            .add_model_config(ModelConfig {
                name: model_id.to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                tasks: None,
                max_sequence_length: None,
            })
            .unwrap();
    }

    fn seed_profile_only_candle_model(state: &AppState, model_id: &str) {
        let mut profiles = HashMap::new();
        profiles.insert(
            "candle".to_string(),
            ProfileConfig {
                adapter_path: Some(
                    "sie_server_rust.adapters.candle:CandleEmbeddingAdapter".to_string(),
                ),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        state
            .model_registry
            .add_model_config(ModelConfig {
                name: model_id.to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                tasks: None,
                max_sequence_length: None,
            })
            .unwrap();
    }

    // End-to-end through the routing entry point (proxy::resolve_model_and_bundle),
    // not just the pure resolver: a `sql` job alias whose target carries a BF16
    // bundle routes model="sql" to that bundle, a `code` alias to the FP8 bundle
    // of the SAME base model, an explicit caller bundle overrides the alias, and
    // an alias pointing at a bundle the model isn't registered under is a hard
    // error (not a silent FP8 fallback). Validates the #1184 precision-routing
    // mechanism against a real AppState + ModelRegistry.
    #[test]
    fn test_resolve_model_and_bundle_alias_carries_precision_bundle() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        // Two bundles sharing the seeded model's adapter → the model lives in both.
        for name in ["fp8-pytorch", "bf16-pytorch"] {
            std::fs::write(
                bundles_dir.path().join(format!("{name}.yaml")),
                format!("name: {name}\nadapters:\n  - module\n"),
            )
            .unwrap();
        }

        let mut config = test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
        );
        config.model_aliases = HashMap::from([
            ("sql".to_string(), "bf16-pytorch:/test/model".to_string()),
            ("code".to_string(), "fp8-pytorch:/test/model".to_string()),
            ("broken".to_string(), "ghost-bundle:/test/model".to_string()),
        ]);
        let config = Arc::new(config);
        let model_registry = Arc::new(ModelRegistry::new(
            bundles_dir.path(),
            models_dir.path(),
            true,
        ));
        let state = AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::clone(&config),
            model_registry,
            pool_manager: Arc::new(PoolManager::new(Vec::new())),
            work_publisher: None,
            lane_backlog_source: None,
            demand_tracker: Arc::new(DemandTracker::new(Default::default())),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
            model_access_policy: None,
        };
        seed_model(&state, "test/model");

        use crate::handlers::proxy::resolve_model_and_bundle;
        // sql -> BF16 bundle; code -> FP8 bundle (same base model, different precision).
        assert_eq!(
            resolve_model_and_bundle(&state, "sql", &axum::http::Extensions::new()).unwrap(),
            ("test/model".to_string(), "bf16-pytorch".to_string()),
        );
        assert_eq!(
            resolve_model_and_bundle(&state, "code", &axum::http::Extensions::new()).unwrap(),
            ("test/model".to_string(), "fp8-pytorch".to_string()),
        );
        // An explicit caller bundle overrides the alias's bundle.
        assert_eq!(
            resolve_model_and_bundle(&state, "fp8-pytorch:/sql", &axum::http::Extensions::new())
                .unwrap(),
            ("test/model".to_string(), "fp8-pytorch".to_string()),
        );
        // Alias -> a bundle the model isn't registered under: hard 409, no fallback.
        assert!(
            resolve_model_and_bundle(&state, "broken", &axum::http::Extensions::new()).is_err()
        );
    }

    /// Seed a multi-profile model whose profiles declare disjoint LoRA
    /// adapter sets. Used to exercise the per-profile lora_adapter
    /// scoping behavior introduced for M10.
    fn seed_multi_profile_lora_model(state: &AppState, model_id: &str) {
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: Some(serde_json::json!({
                    "loadtime": {
                        "lora_paths": {
                            "a1": "acme/a1",
                            "a2": "acme/a2",
                        }
                    }
                })),
                extends: None,
            },
        );
        profiles.insert(
            "a100".to_string(),
            ProfileConfig {
                adapter_path: Some("module:Adapter".to_string()),
                max_batch_tokens: Some(8192),
                compute_precision: None,
                adapter_options: Some(serde_json::json!({
                    "loadtime": {
                        "lora_paths": {
                            "b1": "acme/b1",
                        }
                    }
                })),
                extends: None,
            },
        );
        // Pass the raw profiles via the YAML tasks field so
        // ``ModelInfoExtras::from_yaml_raw`` picks the lora_paths up.
        // ``add_model_config`` re-serializes ModelConfig to YAML
        // internally, so this just sets up the right shape.
        state
            .model_registry
            .add_model_config(ModelConfig {
                name: model_id.to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                tasks: Some(serde_yaml::from_str("generate: {}").unwrap()),
                max_sequence_length: None,
            })
            .unwrap();
    }

    fn worker_msg(name: &str, loaded_models: Vec<String>) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: name.into(),
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: "A100".into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            ready: true,
            loaded_models,
            queue_depth: Some(0),
            pending_cost: None,
            inflight_batches: None,
            models: Vec::new(),
            memory_used_bytes: Some(0),
            memory_total_bytes: Some(0),
            gpus: Vec::new(),
            pool_name: String::new(),
            saturated: false,
            terminated: false,
        }
    }

    async fn body_json(response: Response) -> serde_json::Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_list_models_returns_seeded_model() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "BAAI/bge-m3");
        assert_eq!(models[0]["loaded"], false);
        assert_eq!(models[0]["pending_generation"]["total"], 0);
    }

    #[tokio::test]
    async fn test_list_models_emits_openai_list_shape() {
        // OpenAI-compat: `/v1/models` must carry the OpenAI list shape
        // (`object: "list"` + `data: [{id, object, created, owned_by}]`)
        // alongside the native `models` array so vanilla OpenAI clients
        // (and Open WebUI) can discover models without an external shim.
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["object"], "list");
        let data = body["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        // `id` must equal the string `/v1/chat/completions` accepts.
        assert_eq!(data[0]["id"], "BAAI/bge-m3");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "sie");
        assert!(data[0]["created"].is_i64());
        // Native shape still present (SDK consumers depend on it).
        assert_eq!(body["models"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_list_models_advertises_profile_variants() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_candle_variant_model(&state, "acme/embed");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;

        let native_ids: BTreeSet<&str> = body["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            native_ids,
            BTreeSet::from(["acme/embed", "acme/embed:candle"])
        );

        let openai_ids: BTreeSet<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["id"].as_str().unwrap())
            .collect();
        assert_eq!(
            openai_ids,
            BTreeSet::from(["acme/embed", "acme/embed:candle"])
        );
    }

    #[tokio::test]
    async fn test_list_models_advertises_profile_only_candle_variants() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_profile_only_candle_model(&state, "acme/candle-only");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;

        let native_ids: BTreeSet<&str> = body["models"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["name"].as_str().unwrap())
            .collect();
        assert_eq!(native_ids, BTreeSet::from(["acme/candle-only:candle"]));

        let openai_ids: BTreeSet<&str> = body["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|model| model["id"].as_str().unwrap())
            .collect();
        assert_eq!(openai_ids, BTreeSet::from(["acme/candle-only:candle"]));
    }

    #[tokio::test]
    async fn test_list_models_includes_registry_and_worker_only_models() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["worker/only".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);

        let by_name: HashMap<_, _> = models
            .iter()
            .map(|model| (model["name"].as_str().unwrap(), model))
            .collect();
        assert_eq!(by_name["BAAI/bge-m3"]["loaded"], false);
        assert_eq!(by_name["worker/only"]["loaded"], true);
        assert!(by_name["worker/only"]["inputs"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(by_name["worker/only"]["outputs"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_list_models_canonicalizes_worker_model_names() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["baai/BGE-M3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "BAAI/bge-m3");
        assert_eq!(models[0]["loaded"], true);
        assert_eq!(models[0]["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_known_model_no_workers() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/BAAI/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], false);
        assert_eq!(body["state"], "available");
        assert!(!body["inputs"].as_array().unwrap().is_empty());
        assert!(body["profiles"].is_object());
        // Additive OpenAI retrieve-model fields are merged in.
        assert_eq!(body["id"], "BAAI/bge-m3");
        assert_eq!(body["object"], "model");
        assert_eq!(body["owned_by"], "sie");
        assert!(body["created"].is_i64());
        assert_eq!(body["pending_generation"]["total"], 0);
    }

    #[tokio::test]
    async fn test_get_model_detail_with_live_worker_reports_loaded() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["BAAI/bge-m3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/BAAI/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], true);
        assert_eq!(body["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_canonicalizes_worker_model_names() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");
        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", vec!["baai/BGE-M3".to_string()]),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/baai/bge-m3")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["name"], "BAAI/bge-m3");
        assert_eq!(body["loaded"], true);
        assert_eq!(body["state"], "loaded");
    }

    #[tokio::test]
    async fn test_get_model_detail_unknown_returns_404() {
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/does/not/exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["detail"]["code"], "MODEL_NOT_FOUND");
        assert!(body["detail"]["message"]
            .as_str()
            .unwrap()
            .contains("does/not/exist"));
    }

    #[tokio::test]
    async fn test_models_response_advertises_per_profile_lora_breakdown() {
        // M10: ``/v1/models`` must surface BOTH the model-level
        // ``lora_adapters`` union (back-compat for existing clients
        // listing advertised adapters) AND the per-profile
        // ``profile_lora_adapters`` breakdown so consumers needing
        // precise routing scope don't have to reverse-engineer it. A
        // model with disjoint adapter sets on ``default`` and ``a100``
        // must expose the union of all four under
        // ``capabilities.lora_adapters`` and the precise per-profile
        // mapping under ``capabilities.profile_lora_adapters``.
        let (app, state, _bundles_dir, _models_dir) = build_router_with_state().await;
        seed_multi_profile_lora_model(&state, "acme/multi");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/acme/multi")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let caps = body["capabilities"]
            .as_object()
            .expect("capabilities block");
        // Union (back-compat) carries every adapter advertised by any
        // profile.
        let mut union: Vec<String> = caps["lora_adapters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        union.sort();
        assert_eq!(
            union,
            vec!["a1".to_string(), "a2".to_string(), "b1".to_string()]
        );
        // Per-profile breakdown scopes adapters by profile name.
        let per_profile = caps["profile_lora_adapters"]
            .as_object()
            .expect("profile_lora_adapters present");
        let mut default_adapters: Vec<String> = per_profile["default"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        default_adapters.sort();
        assert_eq!(default_adapters, vec!["a1".to_string(), "a2".to_string()]);
        let a100_adapters: Vec<String> = per_profile["a100"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(a100_adapters, vec!["b1".to_string()]);
    }

    #[tokio::test]
    async fn test_get_model_detail_unknown_when_registry_empty_returns_404() {
        let (app, _state, _bundles_dir, _models_dir) = build_router_with_state().await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models/anything")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = body_json(response).await;
        assert_eq!(body["detail"]["code"], "MODEL_NOT_FOUND");
        assert!(body["detail"]["message"]
            .as_str()
            .unwrap()
            .contains("anything"));
    }
}
