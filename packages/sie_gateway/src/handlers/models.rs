use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::server::AppState;

pub async fn get_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let model_workers = state.registry.get_models().await;

    let all_model_names = state.model_registry.list_models();
    let models: Vec<serde_json::Value> = if !all_model_names.is_empty() {
        all_model_names
            .iter()
            .map(|name| {
                let worker_urls = model_workers.get(name).cloned().unwrap_or_default();
                let bundles = state.model_registry.get_model_bundles(name);
                build_model_payload(name, &bundles, &worker_urls)
            })
            .collect()
    } else {
        model_workers
            .into_iter()
            .map(|(name, urls)| build_model_payload(&name, &[], &urls))
            .collect()
    };

    (StatusCode::OK, Json(json!({"models": models}))).into_response()
}

/// Detail counterpart to `get_models`.
pub async fn get_model(Path(model): Path<String>, State(state): State<Arc<AppState>>) -> Response {
    let known_in_registry = state.model_registry.get_model_info(&model).is_some();
    let bundles = state.model_registry.get_model_bundles(&model);
    let model_workers = state.registry.get_models().await;
    let worker_urls = model_workers.get(&model).cloned().unwrap_or_default();

    if !known_in_registry && bundles.is_empty() && worker_urls.is_empty() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Model '{}' not found", model)})),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        Json(build_model_payload(&model, &bundles, &worker_urls)),
    )
        .into_response()
}

fn build_model_payload(name: &str, bundles: &[String], worker_urls: &[String]) -> Value {
    json!({
        "name": name,
        "bundles": bundles,
        "worker_count": worker_urls.len(),
        "workers": worker_urls,
        "loaded": !worker_urls.is_empty(),
    })
}

pub fn extract_bearer_token(headers: &HeaderMap) -> Option<String> {
    let header = headers
        .get("authorization")?
        .to_str()
        .ok()?
        .trim()
        .to_string();
    if header.is_empty() {
        return None;
    }
    let token = if header.to_lowercase().starts_with("bearer ") {
        header[7..].trim().to_string()
    } else {
        header
    };
    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

pub fn mask_token(token: &str) -> String {
    if token.len() <= 4 {
        "****".to_string()
    } else {
        format!(
            "{}{}",
            "*".repeat(token.len() - 4),
            &token[token.len() - 4..]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_bearer_token ───────────────────────────────────────

    #[test]
    fn test_extract_bearer_token_with_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "Bearer my-token-123".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token-123".into()));
    }

    #[test]
    fn test_extract_bearer_token_case_insensitive_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "bearer my-token".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token".into()));
    }

    #[test]
    fn test_extract_bearer_token_without_prefix() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "raw-token-value".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("raw-token-value".into()));
    }

    #[test]
    fn test_extract_bearer_token_missing_header() {
        let h = HeaderMap::new();
        assert_eq!(extract_bearer_token(&h), None);
    }

    #[test]
    fn test_extract_bearer_token_empty_value() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), None);
    }

    #[test]
    fn test_extract_bearer_token_bearer_only() {
        // "Bearer " trims to "Bearer", which doesn't start with "bearer " (missing trailing space),
        // so it's treated as a raw token value.
        let mut h = HeaderMap::new();
        h.insert("authorization", "Bearer ".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("Bearer".into()));
    }

    #[test]
    fn test_extract_bearer_token_whitespace_trimmed() {
        let mut h = HeaderMap::new();
        h.insert("authorization", "  Bearer  my-token  ".parse().unwrap());
        assert_eq!(extract_bearer_token(&h), Some("my-token".into()));
    }

    // ── mask_token ─────────────────────────────────────────────────

    #[test]
    fn test_mask_token_long() {
        assert_eq!(mask_token("secret-token-123"), "************-123");
    }

    #[test]
    fn test_mask_token_short() {
        assert_eq!(mask_token("abc"), "****");
        assert_eq!(mask_token(""), "****");
    }

    #[test]
    fn test_mask_token_exactly_4() {
        assert_eq!(mask_token("abcd"), "****");
    }

    #[test]
    fn test_mask_token_5_chars() {
        assert_eq!(mask_token("12345"), "*2345");
    }
}

#[cfg(test)]
mod route_tests {
    use std::collections::HashMap;
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
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            bundles_dir: bundles_dir.to_string(),
            models_dir: models_dir.to_string(),
            payload_store_url: String::new(),
            config_service_url: None,
            config_service_token: None,
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
            demand_tracker: Arc::new(DemandTracker::new()),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
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
                adapter_module: None,
                default_bundle: None,
                profiles,
            })
            .unwrap();
    }

    fn worker_msg(name: &str, loaded_models: Vec<String>) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: name.into(),
            gpu_count: 1,
            machine_profile: "A100".into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            ready: true,
            loaded_models,
            queue_depth: Some(0),
            models: Vec::new(),
            memory_used_bytes: Some(0),
            memory_total_bytes: Some(0),
            gpus: Vec::new(),
            pool_name: String::new(),
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
        assert_eq!(body["worker_count"], 0);
        assert!(body["workers"].as_array().unwrap().is_empty());
        assert!(!body["bundles"].as_array().unwrap().is_empty());
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
        assert_eq!(body["worker_count"], 1);
        assert_eq!(
            body["workers"].as_array().unwrap()[0],
            "http://worker-1:8080"
        );
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
        assert!(body["message"].as_str().unwrap().contains("does/not/exist"));
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
        assert!(body["message"].as_str().unwrap().contains("anything"));
    }
}
