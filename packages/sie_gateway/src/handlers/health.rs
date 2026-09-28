use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::Json;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

use crate::server::AppState;

/// Static HTML status page
#[utoipa::path(
    get,
    path = "/",
    tag = "health",
    responses((status = 200, description = "HTML gateway status page", body = String, content_type = "text/html"))
)]
pub async fn status_page(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cluster = state.registry.get_cluster_status().await;
    let status_str = if cluster.worker_count > 0 {
        "healthy"
    } else {
        "degraded"
    };

    let workers_html: String = cluster
        .workers
        .iter()
        .map(|w| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                w.name,
                w.url,
                w.gpu,
                w.bundle,
                if w.healthy { "healthy" } else { "unhealthy" },
                w.queue_depth,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let html = format!(
        r#"<!DOCTYPE html>
<html><head><title>SIE Gateway</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:left}}th{{background:#f5f5f5}}.healthy{{color:green}}.degraded{{color:orange}}</style>
</head><body>
<h1>SIE Gateway</h1>
<p>Status: <span class="{status_str}">{status_str}</span></p>
<p>Workers: {wc} | GPUs: {gc} | Models loaded: {ml} | QPS: {qps:.1}</p>
<h2>Workers</h2>
<table><tr><th>Name</th><th>URL</th><th>GPU</th><th>Bundle</th><th>Health</th><th>Queue Depth</th></tr>
{workers_html}
</table>
<h2>Models</h2>
<ul>{models_html}</ul>
<p><a href="/ws/cluster-status">WebSocket feed</a> | <a href="/health">Health JSON</a></p>
</body></html>"#,
        status_str = status_str,
        wc = cluster.worker_count,
        gc = cluster.gpu_count,
        ml = cluster.models_loaded,
        qps = cluster.total_qps,
        workers_html = workers_html,
        models_html = cluster
            .models
            .iter()
            .map(|m| format!("<li>{} ({} workers)</li>", m.name, m.worker_count))
            .collect::<Vec<_>>()
            .join(""),
    );

    Html(html)
}

#[utoipa::path(
    get,
    path = "/healthz",
    tag = "health",
    responses((
        status = 200,
        description = "Liveness probe (plain text, matches sie_server)",
        body = String,
        content_type = "text/plain; charset=utf-8"
    ))
)]
pub async fn healthz() -> impl IntoResponse {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "ok",
    )
}

#[utoipa::path(
    get,
    path = "/readyz",
    tag = "health",
    description = "Gateway readiness. When a config service URL is configured, returns 503 until this replica has applied its first complete configuration snapshot, and 200 from then on, including while the config service is later unreachable. Without a config service URL it returns 200 once the gateway is serving requests. Readiness never depends on worker health: worker readiness is reported by GET /health and by inference responses with retryable provisioning signals from a workerless gateway. This contract supports KEDA scale-from-zero.",
    responses(
        (status = 200, description = "Gateway is ready", body = String, content_type = "text/plain; charset=utf-8"),
        (status = 503, description = "The first configuration snapshot has not been applied yet", body = String, content_type = "text/plain; charset=utf-8")
    )
)]
pub async fn readyz(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.config.config_service_url.is_some() && !state.config_epoch.is_bootstrapped() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            "config bootstrap pending",
        );
    }
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        "ok",
    )
}

#[utoipa::path(
    get,
    path = "/health",
    tag = "health",
    responses((status = 200, description = "Gateway cluster health", body = crate::openapi::HealthResponse))
)]
pub async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cluster = state.registry.get_cluster_status().await;
    let pending_generation = state
        .work_publisher
        .as_ref()
        .map(|publisher| publisher.pending_generation_snapshot())
        .unwrap_or_default();
    let status_str = if cluster.worker_count > 0 {
        "healthy"
    } else {
        "degraded"
    };

    let gpu_types = state.registry.get_gpu_types().await;

    (
        StatusCode::OK,
        Json(json!({
            "status": status_str,
            "type": "gateway",
            "configured_gpu_types": state.config.configured_gpus,
            "live_gpu_types": gpu_types,
            "cluster": {
                "worker_count": cluster.worker_count,
                "gpu_count": cluster.gpu_count,
                "models_loaded": cluster.models_loaded,
                "total_qps": cluster.total_qps,
            },
            "workers": cluster.workers,
            "models": cluster.models,
            "pending_generation": pending_generation,
        })),
    )
}

#[utoipa::path(
    get,
    path = "/ws/cluster-status",
    tag = "observability",
    responses((status = 101, description = "WebSocket cluster status stream"))
)]
pub async fn ws_cluster_status(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_cluster_status_ws(socket, state))
}

async fn handle_cluster_status_ws(mut socket: WebSocket, state: Arc<AppState>) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    loop {
        interval.tick().await;
        let status = state.registry.get_cluster_status().await;
        let pending_generation = state
            .work_publisher
            .as_ref()
            .map(|publisher| publisher.pending_generation_snapshot())
            .unwrap_or_default();
        // nested cluster sub-object in WS feed
        let nested = serde_json::json!({
            "timestamp": status.timestamp,
            "cluster": {
                "worker_count": status.worker_count,
                "gpu_count": status.gpu_count,
                "models_loaded": status.models_loaded,
                "total_qps": status.total_qps,
            },
            "workers": status.workers,
            "models": status.models,
            "pending_generation": pending_generation,
        });
        let json = match serde_json::to_string(&nested) {
            Ok(j) => j,
            Err(_) => break,
        };
        if socket.send(Message::Text(json.into())).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use axum::Router;
    use tempfile::TempDir;
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::config::Config;
    use crate::server::{create_router, AppState};
    use crate::state::bundle_config_hashes_hash::BundleConfigHashesHash;
    use crate::state::bundles_hash::BundlesHash;
    use crate::state::config_bootstrap::{bootstrap_once, BootstrapClient};
    use crate::state::config_epoch::ConfigEpoch;
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
            max_lane_in_flight_items:
                crate::queue::lane_admission::DEFAULT_MAX_LANE_IN_FLIGHT_ITEMS,
            lane_backpressure_enforce: false,
            stream_max_age_s: 1_800,
            stream_storage: crate::config::StreamStorage::Memory,
            stream_num_replicas: 1,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            configured_physical_lanes: Default::default(),
            static_queue_pools: Vec::new(),
            model_aliases: HashMap::new(),
            published_model_aliases: Default::default(),
            bundles_dir: bundles_dir.to_string(),
            models_dir: models_dir.to_string(),
            payload_store_url: String::new(),
            public_base_url: None,
            config_service_url: config_service_url.map(str::to_string),
            config_service_token: None,
            config_modal_proxy_token: None,
        }
    }

    fn gateway(config_service_url: Option<&str>) -> (Router, Arc<AppState>, TempDir) {
        let temp = TempDir::new().unwrap();
        let bundles = temp.path().join("bundles");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&bundles).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let config = Arc::new(test_config(
            bundles.to_str().unwrap(),
            models.to_str().unwrap(),
            config_service_url,
        ));
        let state = Arc::new(AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::clone(&config),
            model_registry: Arc::new(ModelRegistry::new(&bundles, &models, true)),
            pool_manager: Arc::new(PoolManager::new(Vec::new())),
            work_publisher: None,
            lane_backlog_source: None,
            demand_tracker: Arc::new(DemandTracker::new(Default::default())),
            config_epoch: ConfigEpoch::new(),
            model_access_policy: None,
        });
        (create_router(Arc::clone(&state), config), state, temp)
    }

    async fn readyz(app: &Router) -> (StatusCode, String) {
        let response = app
            .clone()
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    async fn serve_config_snapshot(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 1,
                "bundles_hash": "bundles",
                "bundle_config_hashes_hash": "bundle-configs",
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{"bundle_id": "default", "priority": 10, "adapter_count": 1}],
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "snapshot_version": 1,
                "epoch": 1,
                "generated_at": "2026-04-17T00:00:00Z",
                "models": [],
            })))
            .mount(server)
            .await;
    }

    async fn run_bootstrap(state: &AppState, server: &MockServer) -> bool {
        let client = BootstrapClient::new(server.uri(), None).unwrap();
        bootstrap_once(
            &client,
            state.model_registry.as_ref(),
            &state.config_epoch,
            &BundlesHash::new(),
            &BundleConfigHashesHash::new(),
        )
        .await
        .is_ok()
    }

    #[tokio::test]
    async fn readyz_is_ready_without_a_config_service() {
        let (app, _state, _temp) = gateway(None);
        assert_eq!(readyz(&app).await, (StatusCode::OK, "ok".to_string()));
    }

    #[tokio::test]
    async fn readyz_waits_for_the_first_config_snapshot_and_then_stays_ready() {
        let refusing = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&refusing)
            .await;
        let healthy = MockServer::start().await;
        serve_config_snapshot(&healthy).await;
        let (app, state, _temp) = gateway(Some(&refusing.uri()));

        assert_eq!(
            readyz(&app).await,
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "config bootstrap pending".to_string()
            )
        );
        assert!(!run_bootstrap(&state, &refusing).await);
        assert_eq!(readyz(&app).await.0, StatusCode::SERVICE_UNAVAILABLE);

        assert!(run_bootstrap(&state, &healthy).await);
        assert_eq!(readyz(&app).await, (StatusCode::OK, "ok".to_string()));

        assert!(!run_bootstrap(&state, &refusing).await);
        assert_eq!(readyz(&app).await, (StatusCode::OK, "ok".to_string()));
    }
}
