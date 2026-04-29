use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

use crate::server::AppState;
use crate::state::pool_manager::DEFAULT_POOL_NAME;

#[derive(Debug, Deserialize)]
pub struct CreatePoolRequest {
    pub name: String,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    #[serde(default)]
    pub minimum_worker_count: u32,
}

pub async fn create_pool(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreatePoolRequest>,
) -> impl IntoResponse {
    if req.name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": "Pool name is required"})),
        )
            .into_response();
    }

    if req.gpus.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": "GPU requirements are required"})),
        )
            .into_response();
    }

    match state
        .pool_manager
        .create_pool(
            &req.name,
            req.gpus,
            req.bundle,
            req.ttl_seconds,
            req.minimum_worker_count,
        )
        .await
    {
        Ok(pool) => {
            info!(event = "pool.create", pool = %req.name, status = 201u16, "audit");
            (StatusCode::CREATED, Json(json!(pool))).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn list_pools(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let pools = state.pool_manager.list_pools().await;
    (StatusCode::OK, Json(json!({"pools": pools})))
}

pub async fn get_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.pool_manager.get_pool(&name).await {
        Some(pool) => (StatusCode::OK, Json(json!(pool))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Pool '{}' not found", name)})),
        )
            .into_response(),
    }
}

pub async fn delete_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if name == DEFAULT_POOL_NAME {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({"message": "Cannot delete the default pool"})),
        )
            .into_response();
    }

    match state.pool_manager.delete_pool(&name).await {
        Ok(true) => {
            info!(event = "pool.delete", pool = %name, status = 200u16, "audit");
            (StatusCode::OK, Json(json!({"message": "Pool deleted"}))).into_response()
        }
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Pool '{}' not found", name)})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::FORBIDDEN,
            Json(json!({"message": e.to_string()})),
        )
            .into_response(),
    }
}

pub async fn renew_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if state.pool_manager.renew_pool(&name).await {
        info!(event = "pool.renew", pool = %name, status = 200u16, "audit");
        (StatusCode::OK, Json(json!({"message": "Pool renewed"})))
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({"message": format!("Pool '{}' not found", name)})),
        )
    }
}
