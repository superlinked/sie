//! Read-only config API surface for the gateway.
//!
//! The authoritative config control plane is `packages/sie_config` (Python).
//! The gateway exposes these read-side endpoints backed by its in-memory
//! `ModelRegistry` (filesystem seed + cold-start snapshot from `sie-config`
//! + live NATS deltas):
//!
//! - `GET /v1/configs/models`, `GET /v1/configs/models/{id}`
//! - `GET /v1/configs/models/{id}/status` — per-replica worker-ack readiness
//! - `GET /v1/configs/bundles`, `GET /v1/configs/bundles/{id}`
//! - `POST /v1/configs/resolve`
//!
//! Config writes are not handled here. `POST /v1/configs/models` is not
//! registered; axum returns `405 Method Not Allowed`. Admin tooling must call
//! `sie-config` directly for mutations.
//!
//! All handlers below are pure in-memory reads — the gateway does not own a
//! persistent config store.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::server::AppState;
use crate::state::model_registry::ResolveError;

fn parse_model_spec(spec: &str) -> (String, String) {
    if let Some(idx) = spec.find(":/") {
        (spec[..idx].to_string(), spec[idx + 2..].to_string())
    } else {
        (String::new(), spec.to_string())
    }
}

fn yaml_response<T: Serialize>(status: StatusCode, value: &T) -> Response {
    let body = match serde_yaml::to_string(value) {
        Ok(body) => body,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"message": format!("Failed to serialize YAML: {}", e)})),
            )
                .into_response()
        }
    };

    Response::builder()
        .status(status)
        .header("content-type", "application/x-yaml")
        .body(Body::from(body))
        .unwrap()
}

/// GET /v1/configs/models - List all model configs visible to this gateway.
pub async fn get_model_configs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let models: Vec<serde_json::Value> = state
        .model_registry
        .list_models()
        .into_iter()
        .map(|model_name| {
            let profiles = state.model_registry.get_model_profile_names(&model_name);
            json!({
                "model_id": model_name,
                "profiles": profiles,
                "source": "gateway-registry",
            })
        })
        .collect();

    (StatusCode::OK, Json(json!({ "models": models }))).into_response()
}

/// GET /v1/configs/models/{*id} — dual-purpose dispatcher.
///
/// Handles two endpoints sharing the same wildcard route:
///
/// - `GET /v1/configs/models/{id}` — YAML view of the gateway's model.
/// - `GET /v1/configs/models/{id}/status` — JSON worker-ack readiness for
///   that model on this specific gateway replica.
///
/// The `{*id}` wildcard is necessary because model IDs routinely contain
/// forward slashes (`BAAI/bge-m3`), so we strip an optional trailing
/// `/status` segment here rather than maintain two overlapping axum routes
/// (axum's `matchit` panics on overlapping catch-all routes).
///
/// Disambiguation when a model literally named `something/status` exists:
/// we prefer the status interpretation only when the `/status`-stripped
/// id resolves to a real model. That keeps an accidental model whose ID
/// ends with `/status` addressable under `GET
/// /v1/configs/models/foo/status` as a config read, and matches what
/// admin tooling expects (status requests are only meaningful against
/// models that actually exist). `sie-config` additionally refuses to
/// register model IDs ending in `/status` so the pathological case where
/// **both** `foo` and `foo/status` are models cannot arise, but we
/// still degrade gracefully if it ever does.
pub async fn get_model_config_or_status(
    State(state): State<Arc<AppState>>,
    Path(raw): Path<String>,
) -> Response {
    if let Some(stripped) = raw.strip_suffix("/status") {
        if state.model_registry.get_model_info(stripped).is_some() {
            return model_status_response(state.as_ref(), stripped).await;
        }
        // `stripped` is not a model — fall through to treat the full
        // path (with the `/status` suffix) as a potential model id.
    }
    model_config_response(state.as_ref(), &raw)
}

fn model_config_response(state: &AppState, id: &str) -> Response {
    let Some(model_info) = state.model_registry.get_model_info(id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Model '{}' not found", id)})),
        )
            .into_response();
    };

    let doc = json!({
        "sie_id": model_info.name,
        "source": "gateway-registry",
        "bundles": model_info.bundles,
    });
    yaml_response(StatusCode::OK, &doc)
}

/// Worker-ack readiness snapshot for a single model on this gateway replica.
///
/// Admin tooling uses this after a `sie-config` write to confirm that
/// configured workers have picked up the new `bundle_config_hash` for each
/// affected bundle. This is strictly a **per-replica** view: the gateway
/// answers with its own `WorkerRegistry` state. Fleet-wide readiness is the
/// union of this endpoint across all gateway replicas, which admin tooling
/// aggregates.
///
/// The endpoint is intentionally not on `sie-config`: the config service
/// owns persistence and fan-out, not the worker registry. Keeping this here
/// preserves the split — `sie-config` is "is it persisted?", the gateway is
/// "is it live on this replica?".
async fn model_status_response(state: &AppState, id: &str) -> Response {
    let Some(model_info) = state.model_registry.get_model_info(id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Model '{}' not found", id)})),
        )
            .into_response();
    };

    let status = compute_model_status(state, &model_info).await;
    (StatusCode::OK, Json(status)).into_response()
}

async fn compute_model_status(
    state: &AppState,
    model_info: &crate::types::model::ModelEntry,
) -> serde_json::Value {
    let workers = state.registry.workers().await;

    let mut bundles_out = Vec::with_capacity(model_info.bundles.len());
    // A model with zero bundles is not "acked" — it's inert. Callers
    // polling this endpoint after a write expect `true` to mean "all
    // workers for this model's bundles have picked up the new hash";
    // answering `true` for "there's nothing to ack" is a silent
    // misreport, so `no_bundles` makes the distinction explicit and
    // `all_bundles_acked` stays false.
    let mut all_acked = !model_info.bundles.is_empty();

    for bundle_id in &model_info.bundles {
        let expected = state.model_registry.compute_bundle_config_hash(bundle_id);

        let mut total_eligible = 0usize;
        let mut acked_workers: Vec<String> = Vec::new();
        let mut pending_workers: Vec<String> = Vec::new();

        for worker in workers.values() {
            // Exact bundle-name match. `compute_bundle_config_hash` hashes
            // against the canonical (case-sensitive) bundle id from the
            // registry; a case-mismatched worker's advertised hash can
            // never line up with that, so counting it as eligible would
            // park it in `pending_workers` indefinitely. The fleet is
            // case-consistent in practice.
            if worker.bundle != *bundle_id {
                continue;
            }
            if !worker.healthy() {
                continue;
            }
            total_eligible += 1;
            if !expected.is_empty() && worker.bundle_config_hash == expected {
                acked_workers.push(worker.name.clone());
            } else {
                pending_workers.push(worker.name.clone());
            }
        }

        let bundle_acked = total_eligible > 0 && pending_workers.is_empty();
        if !bundle_acked {
            all_acked = false;
        }

        bundles_out.push(json!({
            "bundle_id": bundle_id,
            "expected_bundle_config_hash": expected,
            "total_eligible_workers": total_eligible,
            "acked_workers": acked_workers,
            "pending_workers": pending_workers,
            "acked": bundle_acked,
        }));
    }

    json!({
        "model_id": model_info.name,
        "config_epoch": state.config_epoch.get(),
        "all_bundles_acked": all_acked,
        "no_bundles": model_info.bundles.is_empty(),
        "bundles": bundles_out,
        "source": "gateway-registry",
    })
}

/// GET /v1/configs/bundles - List all bundle configs.
pub async fn get_bundle_configs(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let workers = state.registry.workers().await;

    let bundles: Vec<serde_json::Value> = state
        .model_registry
        .list_bundles()
        .into_iter()
        .filter_map(|bundle_name| state.model_registry.get_bundle_info(&bundle_name))
        .map(|info| {
            let connected_workers = workers
                .values()
                .filter(|worker| worker.healthy() && worker.bundle == info.name)
                .count();

            json!({
                "bundle_id": info.name,
                "priority": info.priority,
                "adapter_count": info.adapters.len(),
                "source": "gateway-registry",
                "connected_workers": connected_workers,
            })
        })
        .collect();

    (StatusCode::OK, Json(json!({ "bundles": bundles })))
}

/// GET /v1/configs/bundles/{id} - Get specific bundle config.
pub async fn get_bundle_config(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let Some(info) = state.model_registry.get_bundle_info(&id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Bundle '{}' not found", id)})),
        )
            .into_response();
    };

    let doc = json!({
        "name": info.name,
        "priority": info.priority,
        "source": "gateway-registry",
        "adapters": info.adapters,
    });
    yaml_response(StatusCode::OK, &doc)
}

/// POST /v1/configs/resolve - Resolve bundle for a model.
#[derive(Debug, Deserialize)]
pub struct ResolveRequest {
    pub model: String,
    #[serde(default)]
    pub bundle: Option<String>,
}

pub async fn resolve_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ResolveRequest>,
) -> impl IntoResponse {
    let (parsed_bundle_override, model_name) = parse_model_spec(&req.model);
    let bundle_override = if parsed_bundle_override.is_empty() {
        req.bundle.as_deref()
    } else {
        Some(parsed_bundle_override.as_str())
    };

    match state
        .model_registry
        .resolve_bundle(&model_name, bundle_override)
    {
        Ok(resolved_bundle) => {
            let compatible_bundles = state.model_registry.get_model_bundles(&model_name);
            let profiles = state.model_registry.get_model_profile_names(&model_name);

            (
                StatusCode::OK,
                Json(json!({
                    "model": model_name,
                    "resolved_bundle": resolved_bundle,
                    "compatible_bundles": compatible_bundles,
                    "profiles": profiles,
                })),
            )
                .into_response()
        }
        Err(ResolveError::ModelNotFound(e)) => (
            StatusCode::NOT_FOUND,
            Json(json!({"message": e.to_string(), "model": model_name})),
        )
            .into_response(),
        Err(ResolveError::BundleConflict(e)) => (
            StatusCode::CONFLICT,
            Json(json!({
                "message": e.to_string(),
                "model": model_name,
                "bundle": e.bundle,
                "compatible_bundles": e.compatible_bundles,
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use tower::ServiceExt;

    use crate::config::Config;
    use crate::server::{create_router, AppState};
    use crate::state::demand_tracker::DemandTracker;
    use crate::state::model_registry::ModelRegistry;
    use crate::state::pool_manager::PoolManager;
    use crate::state::worker_registry::WorkerRegistry;

    fn test_config(
        bundles_dir: &str,
        models_dir: &str,
        config_service_url: Option<&str>,
    ) -> Config {
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
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            bundles_dir: bundles_dir.to_string(),
            models_dir: models_dir.to_string(),
            payload_store_url: String::new(),
            config_service_url: config_service_url.map(str::to_string),
            config_service_token: None,
        }
    }

    fn build_test_router(
        bundles_dir: &tempfile::TempDir,
        models_dir: &tempfile::TempDir,
        config_service_url: Option<&str>,
    ) -> Router {
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();

        let config = Arc::new(test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
            config_service_url,
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
            demand_tracker: Arc::new(DemandTracker::new()),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
        });

        create_router(state, config)
    }

    /// Variant of `build_test_router` that also returns the shared
    /// `AppState` so a test can mutate `WorkerRegistry` / `ModelRegistry`
    /// before issuing requests.
    async fn build_test_router_with_state(
        bundles_dir: &tempfile::TempDir,
        models_dir: &tempfile::TempDir,
    ) -> (Router, Arc<AppState>) {
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();

        let config = Arc::new(test_config(
            bundles_dir.path().to_str().unwrap(),
            models_dir.path().to_str().unwrap(),
            None,
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
            demand_tracker: Arc::new(DemandTracker::new()),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
        });
        let router = create_router(Arc::clone(&state), config);
        (router, state)
    }

    async fn body_bytes(response: Response) -> Vec<u8> {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    #[tokio::test]
    async fn test_post_model_config_returns_405_method_not_allowed() {
        // The gateway does not serve POST writes at all. Axum's default router
        // returns 405 for a method that has no matching handler on a path that
        // does exist (the GET route is registered). This is the intended
        // "writes live on sie-config" signal.
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/configs/models")
                    .header("content-type", "application/x-yaml")
                    .body(Body::from("sie_id: test/model\nprofiles: {}\n"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn test_get_bundle_configs_reports_filesystem_seed() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/bundles")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = body_bytes(response).await;
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let bundles = parsed["bundles"].as_array().unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0]["bundle_id"], "default");
        assert_eq!(bundles[0]["source"], "gateway-registry");
    }

    #[tokio::test]
    async fn test_get_bundle_config_by_id() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/bundles/default")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(body_bytes(response).await).unwrap();
        assert!(body.contains("name: default"));
        assert!(body.contains("source: gateway-registry"));
    }

    #[tokio::test]
    async fn test_get_bundle_config_unknown_returns_404() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/bundles/does-not-exist")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_model_configs_empty_without_seed() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(parsed["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_get_model_config_unknown_returns_404() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/not/there")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_resolve_config_unknown_model_returns_404() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let app = build_test_router(&bundles_dir, &models_dir, None);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/configs/resolve")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::to_vec(&serde_json::json!({"model": "nope"})).unwrap(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_parse_model_spec_with_bundle_prefix() {
        let (bundle, model) = parse_model_spec("embedding:/BAAI/bge-m3");
        assert_eq!(bundle, "embedding");
        assert_eq!(model, "BAAI/bge-m3");
    }

    #[test]
    fn test_parse_model_spec_without_prefix() {
        let (bundle, model) = parse_model_spec("BAAI/bge-m3");
        assert_eq!(bundle, "");
        assert_eq!(model, "BAAI/bge-m3");
    }

    /// Helper: seed a model into the registry so status tests have a target.
    fn seed_model(state: &AppState, model_id: &str) {
        use crate::types::model::{ModelConfig, ProfileConfig};

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
                adapter_module: None,
                default_bundle: None,
                profiles,
            })
            .unwrap();
    }

    #[tokio::test]
    async fn test_model_status_unknown_model_returns_404() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, _) = build_test_router_with_state(&bundles_dir, &models_dir).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/nope/there/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_model_status_reports_zero_workers_when_none_connected() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "BAAI/bge-m3");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/BAAI/bge-m3/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(parsed["model_id"], "BAAI/bge-m3");
        assert_eq!(parsed["all_bundles_acked"], false);
        let bundles = parsed["bundles"].as_array().unwrap();
        assert!(!bundles.is_empty());
        for bundle in bundles {
            assert_eq!(bundle["total_eligible_workers"], 0);
            assert!(bundle["acked_workers"].as_array().unwrap().is_empty());
            assert!(bundle["pending_workers"].as_array().unwrap().is_empty());
            assert_eq!(bundle["acked"], false);
        }
    }

    fn worker_msg(
        name: &str,
        bundle: &str,
        hash: &str,
    ) -> crate::types::worker::WorkerStatusMessage {
        crate::types::worker::WorkerStatusMessage {
            name: name.into(),
            gpu_count: 1,
            machine_profile: "A100".into(),
            bundle: bundle.into(),
            bundle_config_hash: hash.into(),
            ready: true,
            loaded_models: Vec::new(),
            queue_depth: Some(0),
            models: Vec::new(),
            memory_used_bytes: Some(0),
            memory_total_bytes: Some(0),
            gpus: Vec::new(),
            pool_name: String::new(),
        }
    }

    #[tokio::test]
    async fn test_model_status_reports_acked_when_hash_matches() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "BAAI/bge-m3");
        let expected_hash = state.model_registry.compute_bundle_config_hash("default");
        assert!(!expected_hash.is_empty(), "bundle hash should be populated");

        state
            .registry
            .update_worker(
                "http://worker-1:8080",
                worker_msg("worker-1", "default", &expected_hash),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/BAAI/bge-m3/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(parsed["all_bundles_acked"], true);
        let bundle = &parsed["bundles"][0];
        assert_eq!(bundle["bundle_id"], "default");
        assert_eq!(bundle["total_eligible_workers"], 1);
        assert_eq!(bundle["acked_workers"].as_array().unwrap().len(), 1);
        assert!(bundle["pending_workers"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_model_status_reports_pending_for_stale_hash() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "BAAI/bge-m3");

        state
            .registry
            .update_worker(
                "http://worker-2:8080",
                worker_msg("worker-2", "default", "old-stale-hash"),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/BAAI/bge-m3/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(parsed["all_bundles_acked"], false);
        let bundle = &parsed["bundles"][0];
        assert_eq!(bundle["total_eligible_workers"], 1);
        assert!(bundle["acked_workers"].as_array().unwrap().is_empty());
        assert_eq!(bundle["pending_workers"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_model_status_empty_bundles_is_not_all_acked() {
        // A model with no routable bundles must report
        // `all_bundles_acked: false` and `no_bundles: true`. Reporting
        // `true` for an empty bundle set would let admin tooling treat
        // "nothing deployed" as "fully deployed".
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;

        // Seed a model whose adapter module does NOT match any bundle.
        use crate::types::model::{ModelConfig, ProfileConfig};
        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some("nonexistent.module:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        // add_model_config rejects unroutable adapters, so we bypass it:
        // directly register the model name via a config with a routable
        // adapter and then verify that bundle list is still populated.
        // Better: add with a routable adapter, then the registry *will*
        // populate bundles. For this test, use a raw insertion trick —
        // simplest is to register with routable adapter so bundles are
        // populated, then confirm behavior: switching to a case that
        // actually exercises the `bundles.is_empty()` branch requires
        // manual registry manipulation. We'll emulate by using a model
        // config that validates but whose ModelInfo carries empty
        // bundles — namely, register then clear bundles by hot-reloading
        // against an empty bundles dir.
        let _ = state.model_registry.add_model_config(ModelConfig {
            name: "empty/model".to_string(),
            adapter_module: None,
            default_bundle: None,
            profiles,
        });
        // The add above will error because 'nonexistent.module' isn't in
        // any bundle — which is actually the intended real-world path:
        // a model with no bundles never makes it into the registry. So
        // exercise the branch differently: add a valid model, then
        // shadow its bundles list via a direct registry path. Since
        // that path isn't exposed, fall back to asserting the simpler
        // invariant: when the registry reports empty bundles, the
        // endpoint's response must NOT say `all_bundles_acked: true`.
        //
        // We synthesize a ModelEntry with empty bundles via
        // compute_model_status directly.
        let entry = crate::types::model::ModelEntry {
            name: "empty/model".to_string(),
            bundles: Vec::new(),
            adapter_modules: Default::default(),
            profile_names: Default::default(),
            profile_configs: Default::default(),
        };
        let status = compute_model_status(state.as_ref(), &entry).await;
        assert_eq!(status["all_bundles_acked"], false);
        assert_eq!(status["no_bundles"], true);
        let bundles = status["bundles"].as_array().unwrap();
        assert!(bundles.is_empty());

        // Also sanity-check: a wildcard URL should still work.
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/nonexistent/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_model_status_exact_case_required_for_worker_match() {
        // Worker bundle matching must be case-sensitive, because
        // `compute_bundle_config_hash` is keyed off the canonical bundle
        // id. A case-mismatched worker counted as eligible would park in
        // `pending_workers` forever (its hash would be computed against
        // its case variant and never line up with the canonical one),
        // turning a live model into perpetual "pending" for admin tooling.
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "BAAI/bge-m3");

        // Worker claims to serve "Default" (capitalized) — registry has
        // canonical "default". Worker must NOT be counted as eligible.
        state
            .registry
            .update_worker(
                "http://worker-case:8080",
                worker_msg("worker-case", "Default", "any-hash"),
            )
            .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/BAAI/bge-m3/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        let bundle = &parsed["bundles"][0];
        assert_eq!(bundle["bundle_id"], "default");
        assert_eq!(
            bundle["total_eligible_workers"], 0,
            "case-mismatched worker must not count toward eligibility"
        );
    }

    #[tokio::test]
    async fn test_model_id_with_status_suffix_falls_back_to_config_read() {
        // A model legitimately named `foo/status` must still be
        // addressable via `GET /v1/configs/models/foo/status` as a config
        // read when no model named `foo` exists. The wildcard dispatcher
        // prefers the status interpretation only when the `/status`-
        // stripped id resolves to a real model, so the legitimate
        // config-read path is preserved.
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "foo/status");
        assert!(state.model_registry.get_model_info("foo").is_none());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/foo/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        // Config YAML body mentions the model id.
        let body = String::from_utf8(body_bytes(response).await).unwrap();
        assert!(body.contains("foo/status"), "body was: {}", body);
    }

    #[tokio::test]
    async fn test_model_status_reports_config_epoch() {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let (app, state) = build_test_router_with_state(&bundles_dir, &models_dir).await;
        seed_model(&state, "BAAI/bge-m3");
        state.config_epoch.set_max(99);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/configs/models/BAAI/bge-m3/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let parsed: serde_json::Value =
            serde_json::from_slice(&body_bytes(response).await).unwrap();
        assert_eq!(parsed["config_epoch"], 99);
    }
}
