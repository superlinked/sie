use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;
use utoipa::ToSchema;

use crate::http_error::{code as err_code, json_detail};
use crate::server::AppState;
use crate::state::model_registry::ModelRegistry;
use crate::state::pool_manager::DEFAULT_POOL_NAME;

#[derive(Debug, Deserialize, ToSchema)]
pub struct CreatePoolRequest {
    pub name: String,
    /// Physical Helm/NATS queue namespace to draw workers from. Omit for normal
    /// dynamic logical pools; they use `default`. Set only to an
    /// operator-provisioned Helm queue pool declared under
    /// `queueRouting.staticQueuePools` for dedicated capacity.
    #[serde(default)]
    pub queue_pool: Option<String>,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default)]
    pub gpu_caps: HashMap<String, u32>,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    /// Per-pool warm floor (minimum machines kept warm via KEDA). Default 0
    /// keeps scale-from-zero. See `PoolSpec::minimum_worker_count`.
    #[serde(default)]
    pub minimum_worker_count: u32,
    /// Per-pool pinned-model set. Each id is validated against the models the
    /// gateway already tracks and stored canonicalized; ids may be
    /// profile-qualified (`model-name:profile_name`). Default empty leaves
    /// lazy-loading unchanged. See `PoolSpec::pinned_models`.
    #[serde(default)]
    pub pinned_models: Vec<String>,
}

/// Resolve a requested pinned-model set to canonical, deduped model ids,
/// validating each against the gateway's tracked models. Supports
/// profile-qualified ids (`model-name:profile_name`): the registry materializes
/// each non-default profile as its own `{base}:{profile}` model entry, so those
/// resolve like any other tracked model; the `default` profile is the base
/// model itself, so `model:default` folds to the canonical base. Returns
/// `Err(message)` (mapped to a `400` by the caller) describing the first id that
/// is not a tracked model or names an unconfigured profile, so the stored set
/// always matches what `GET /v1/configs/models` reports. Case-variant and
/// duplicate ids fold to a single canonical entry.
fn canonicalize_pinned_models(
    registry: &ModelRegistry,
    requested: &[String],
) -> Result<Vec<String>, String> {
    let mut canonical_models = Vec::with_capacity(requested.len());
    let mut seen = std::collections::HashSet::new();
    for raw in requested {
        // A bare model id or an existing `{base}:{profile}` variant resolves
        // directly (profile variants are first-class registry entries).
        if let Some(canonical) = registry.resolve_canonical_model_name(raw) {
            if seen.insert(canonical.clone()) {
                canonical_models.push(canonical);
            }
            continue;
        }
        // Not directly resolvable. If it is `{base}:{profile}` and the base is
        // tracked, distinguish the `default` profile (folds to the base, which
        // already represents it) from an unconfigured profile (a clear 400).
        if let Some((base, profile)) = raw.rsplit_once(':') {
            if let Some(canonical_base) = registry.resolve_canonical_model_name(base) {
                if profile.eq_ignore_ascii_case("default") {
                    if seen.insert(canonical_base.clone()) {
                        canonical_models.push(canonical_base);
                    }
                    continue;
                }
                return Err(format!(
                    "Pinned model '{raw}': profile '{profile}' is not configured for model '{base}'"
                ));
            }
        }
        return Err(format!("Pinned model '{raw}' is not a tracked model"));
    }
    Ok(canonical_models)
}

#[utoipa::path(
    post,
    path = "/v1/pools",
    tag = "pools",
    request_body = CreatePoolRequest,
    responses(
        (status = 201, description = "Pool created, renewed, or updated", body = crate::types::pool::Pool),
        (status = 400, description = "Invalid pool request", body = crate::openapi::StandardApiError)
    )
)]
pub async fn create_pool(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreatePoolRequest>,
) -> impl IntoResponse {
    if req.name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json_detail(
                err_code::INVALID_REQUEST,
                "Pool name is required",
            )),
        )
            .into_response();
    }

    if req.gpus.is_empty() && req.gpu_caps.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json_detail(
                err_code::INVALID_REQUEST,
                "GPU requirements or caps are required",
            )),
        )
            .into_response();
    }

    // Validate and canonicalize each pinned id (incl. `model:profile`) against
    // the gateway's model registry so the stored set matches what
    // `GET /v1/configs/models` reports. Unknown model / profile → 400.
    let pinned_models = match canonicalize_pinned_models(&state.model_registry, &req.pinned_models)
    {
        Ok(models) => models,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json_detail(err_code::INVALID_REQUEST, message)),
            )
                .into_response()
        }
    };

    match state
        .pool_manager
        .create_pool_with_caps_on_queue(
            &req.name,
            req.queue_pool.as_deref().unwrap_or(DEFAULT_POOL_NAME),
            req.gpus,
            req.gpu_caps,
            req.bundle,
            req.ttl_seconds,
            req.minimum_worker_count,
            pinned_models,
        )
        .await
    {
        Ok(pool) => {
            info!(event = "pool.create", pool = %req.name, status = 201u16, "audit");
            (StatusCode::CREATED, Json(json!(pool))).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json_detail(err_code::INVALID_REQUEST, e.to_string())),
        )
            .into_response(),
    }
}

#[utoipa::path(
    get,
    path = "/v1/pools",
    tag = "pools",
    responses((status = 200, description = "Pools visible to this gateway replica", body = crate::openapi::PoolListResponse))
)]
pub async fn list_pools(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let pools = state.pool_manager.list_pools().await;
    (StatusCode::OK, Json(json!({"pools": pools})))
}

#[utoipa::path(
    get,
    path = "/v1/pools/{name}",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool detail", body = crate::types::pool::Pool),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
pub async fn get_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.pool_manager.get_pool(&name).await {
        Some(pool) => (StatusCode::OK, Json(json!(pool))).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
            .into_response(),
    }
}

#[utoipa::path(
    delete,
    path = "/v1/pools/{name}",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool deleted", body = crate::openapi::MessageResponse),
        (status = 403, description = "Pool cannot be deleted", body = crate::openapi::StandardApiError),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
pub async fn delete_pool(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
        return (
            StatusCode::FORBIDDEN,
            Json(json_detail(
                err_code::DEFAULT_POOL_DELETE_FORBIDDEN,
                "Cannot delete the default pool",
            )),
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
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
            .into_response(),
        Err(e) => (
            StatusCode::FORBIDDEN,
            Json(json_detail(
                err_code::POOL_OPERATION_FORBIDDEN,
                e.to_string(),
            )),
        )
            .into_response(),
    }
}

#[utoipa::path(
    post,
    path = "/v1/pools/{name}/renew",
    tag = "pools",
    params(("name" = String, Path, description = "Pool name")),
    responses(
        (status = 200, description = "Pool renewed", body = crate::openapi::MessageResponse),
        (status = 404, description = "Pool not found", body = crate::openapi::StandardApiError)
    )
)]
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
            Json(json_detail(
                err_code::POOL_NOT_FOUND,
                format!("Pool '{}' not found", name),
            )),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::model::{ModelConfig, ProfileConfig};

    /// A registry seeded with a tracked model `test/model` that has a default
    /// profile plus a non-default `fp8` profile, so `resolve_canonical_model_name`
    /// resolves the base, its case variants, and the `test/model:fp8` profile
    /// variant. The returned `TempDir`s must be kept alive for the registry's
    /// lifetime.
    fn seeded_registry() -> (ModelRegistry, tempfile::TempDir, tempfile::TempDir) {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        let registry = ModelRegistry::new(bundles_dir.path(), models_dir.path(), true);

        let mut profiles = HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                adapter_path: Some(
                    "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                        .to_string(),
                ),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        // A non-default profile so the registry materializes a `test/model:fp8`
        // variant entry (the gateway expands non-default profiles into their
        // own `{base}:{profile}` model entries).
        profiles.insert(
            "fp8".to_string(),
            ProfileConfig {
                adapter_path: Some(
                    "sie_server.adapters.sentence_transformer:SentenceTransformerAdapter"
                        .to_string(),
                ),
                max_batch_tokens: Some(4096),
                compute_precision: Some("fp8".to_string()),
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "test/model".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .expect("seed model");

        (registry, bundles_dir, models_dir)
    }

    #[test]
    fn canonicalize_rejects_unknown_model() {
        let (registry, _bundles, _models) = seeded_registry();
        let err = canonicalize_pinned_models(&registry, &["does/not-exist".to_string()])
            .expect_err("an untracked model must be rejected");
        assert!(err.contains("does/not-exist"), "{err}");
        assert!(err.contains("not a tracked model"), "{err}");
    }

    #[test]
    fn canonicalize_accepts_non_default_profile_variant() {
        let (registry, _bundles, _models) = seeded_registry();
        // `model:profile` for a configured non-default profile is a first-class
        // tracked id and is stored as the canonical variant name.
        let out = canonicalize_pinned_models(&registry, &["test/model:fp8".to_string()])
            .expect("a configured profile variant must be accepted");
        assert_eq!(out, vec!["test/model:fp8".to_string()]);
    }

    #[test]
    fn canonicalize_folds_default_profile_to_base() {
        let (registry, _bundles, _models) = seeded_registry();
        // `model:default` is the base model (no `{base}:default` variant is
        // materialized), so it folds to the canonical base and dedupes against a
        // bare base reference.
        let out = canonicalize_pinned_models(
            &registry,
            &["test/model:default".to_string(), "test/model".to_string()],
        )
        .expect("the default profile folds to the base");
        assert_eq!(out, vec!["test/model".to_string()]);
    }

    #[test]
    fn canonicalize_rejects_unconfigured_profile() {
        let (registry, _bundles, _models) = seeded_registry();
        // The base is tracked but the named profile is not configured: a clear
        // 400, distinct from an unknown base model.
        let err = canonicalize_pinned_models(&registry, &["test/model:bogus".to_string()])
            .expect_err("an unconfigured profile must be rejected");
        assert!(err.contains("profile 'bogus'"), "{err}");
        assert!(err.contains("test/model"), "{err}");
    }

    #[test]
    fn canonicalize_folds_case_variants_and_dedupes() {
        let (registry, _bundles, _models) = seeded_registry();
        // Two case-variant references to the same tracked model collapse to a
        // single canonical entry matching the registry's stored name.
        let out = canonicalize_pinned_models(
            &registry,
            &["test/model".to_string(), "TEST/MODEL".to_string()],
        )
        .expect("a tracked model must be accepted");
        assert_eq!(out, vec!["test/model".to_string()]);
    }

    #[test]
    fn canonicalize_empty_is_ok_empty() {
        let (registry, _bundles, _models) = seeded_registry();
        let out = canonicalize_pinned_models(&registry, &[]).expect("empty set is always ok");
        assert!(out.is_empty());
    }
}
