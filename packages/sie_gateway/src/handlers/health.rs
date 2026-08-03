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
    description = "Process readiness only. Always returns 200 once the gateway is serving requests; never returns 503. Worker readiness is reported by GET /health and by inference responses with retryable provisioning signals from a workerless gateway. This contract supports KEDA scale-from-zero.",
    responses(
        (status = 200, description = "Gateway process is ready", body = String, content_type = "text/plain; charset=utf-8")
    )
)]
pub async fn readyz() -> impl IntoResponse {
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
