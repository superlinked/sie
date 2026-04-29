use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use rmp_serde;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::metrics;
use crate::queue::publisher;

use crate::server::AppState;
use crate::state::model_registry::ResolveError;
use crate::state::worker_registry::WorkerRegistry;
use crate::types::AuditEntry;

use super::models::{extract_bearer_token, mask_token};

const GATEWAY_VERSION: &str = env!("CARGO_PKG_VERSION");
static GATEWAY_VERSION_MINOR: std::sync::LazyLock<u32> =
    std::sync::LazyLock::new(|| env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0));
const DEFAULT_RETRY_AFTER: &str = "120";
const BACKPRESSURE_RETRY_AFTER: &str = "5";
const MODEL_LOADING_RETRY_AFTER: &str = "5";
const MODEL_LOADING_ERROR_CODE: &str = "MODEL_LOADING";
/// Server-side OOM recovery exhausted. Workers stamp this on
/// ``WorkResult.error_code`` when the per-batch ``cache_clear → evict_lru
/// → split_batch`` strategy still runs out of GPU memory; the gateway
/// translates it into HTTP 503 + ``Retry-After`` so the SDK auto-retries
/// with bounded exponential backoff. The worker is **not** marked
/// unhealthy — it lost an allocation race, it isn't broken.
const RESOURCE_EXHAUSTED_ERROR_CODE: &str = "RESOURCE_EXHAUSTED";
const RESOURCE_EXHAUSTED_RETRY_AFTER: &str = "5";
/// Worker is loading a LoRA adapter on demand. The SDK retries this with
/// the same ``provision_timeout_s`` budget it uses for ``MODEL_LOADING``;
/// see ``sie_sdk.client._shared.LORA_LOADING_*``.
const LORA_LOADING_ERROR_CODE: &str = "LORA_LOADING";
const LORA_LOADING_RETRY_AFTER: &str = "5";
const ESTIMATED_WAIT_S: u64 = 180;

/// Track which SDK minor versions we've already warned about (to warn once per minor).
static SDK_WARNED_MINORS: std::sync::LazyLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

/// Per-unique-SDK-version cache of the parsed minor-version number.
///
/// Every inference request runs through [`check_sdk_version`] which
/// previously allocated a fresh `Vec<&str>` via `split('.')` and ran
/// a `u32::parse` every call. A gateway sees at most a handful of
/// unique SDK version strings over its lifetime (one per client
/// release), so a DashMap keyed by the raw header value folds the
/// parse work down to a single lookup on the hot path. `Option<u32>`
/// caches "unparseable" so malformed headers don't re-parse either.
///
/// **Hard size cap.** `X-SIE-SDK-Version` is caller-supplied, so a
/// buggy or hostile client could otherwise walk the key space with
/// unique strings on every request. Once `SDK_VERSION_CACHE_CAP`
/// entries are populated we stop memoising and fall back to
/// parse-on-every-request (which is what `main` did before this
/// optimisation existed, so the worst case is still bounded).
/// 1024 entries is well above the real client-release count and
/// well below any memory worry — `Arc<str>` + `Option<u32>` costs
/// ~32 B per entry plus the version string itself.
static SDK_VERSION_CACHE: std::sync::LazyLock<DashMap<Arc<str>, Option<u32>>> =
    std::sync::LazyLock::new(DashMap::new);
const SDK_VERSION_CACHE_CAP: usize = 1024;

/// Outcome of resolving the JetStream pool to publish work for a request.
#[derive(Debug, PartialEq, Eq)]
enum PoolResolution {
    /// A healthy worker is registered and this is the pool to publish to.
    Pool(String),
    /// No healthy worker matches `(bundle, gpu)` and the caller did not
    /// pin a specific pool — the caller should emit `202 provisioning`
    /// and record pending demand so KEDA scales up.
    Provisioning,
}

/// Result of `resolve_effective_pool` — bundles the routing decision
/// with a flag telling the caller whether the registry had an exact
/// `(bundle, gpu)` worker match *before* any bundle-only fallback.
///
/// The gateway records pending demand (for KEDA auto-scale) whenever
/// the caller expressed a GPU preference but the exact tuple has no
/// healthy worker. By reporting `exact_gpu_match` here we can fold
/// that probe into the same registry load the routing decision already
/// does, instead of doing a separate `resolve_queue_pool` call on the
/// hot path.
#[derive(Debug, PartialEq, Eq)]
struct PoolLookup {
    resolution: PoolResolution,
    /// `true` iff a healthy worker with a non-empty pool name existed
    /// for the exact `(bundle, gpu)` tuple at lookup time.
    ///
    /// Always `false` when `gpu.is_empty()` (no exact tuple to match)
    /// or when `pool_name` was caller-pinned (we short-circuit the
    /// registry lookup entirely in that case — callers that care
    /// about demand tracking already know to record it before
    /// trusting the pin).
    exact_gpu_match: bool,
}

/// Pure decision logic for the scale-from-zero branch of `proxy_request`.
/// Kept as a free function so it can be unit-tested without standing up
/// an `AppState` / `WorkPublisher`.
///
/// Rules (see `product/design.md` §10 and
/// `packages/sie_gateway/docs/architecture-guide.md` §2):
/// - If the caller pinned a pool via `X-SIE-Pool`, trust them and publish
///   there unconditionally. This preserves the "power user" path where
///   the client knows exactly which pool it wants (including cold ones
///   that are expected to scale up on demand).
/// - Otherwise look up a healthy worker for `(bundle, gpu)`. If GPU was
///   specified and the exact tuple has no worker, fall back to any worker
///   on `bundle` (covers single-GPU clusters where the profile-level
///   distinction is cosmetic).
/// - If nothing resolves, return `Provisioning` so the caller can emit
///   `202 + Retry-After` — regardless of whether the caller sent
///   `X-SIE-MACHINE-PROFILE`. Before the fix this branch only fired when
///   `gpu` was non-empty, which turned a normal cold start into a queue
///   timeout for default-routing clients.
async fn resolve_effective_pool(
    registry: &WorkerRegistry,
    bundle: &str,
    gpu: &str,
    pool_name: &str,
) -> PoolLookup {
    if !pool_name.is_empty() {
        // Caller pinned a pool. We still try one registry probe to
        // report `exact_gpu_match` honestly (so callers can decide
        // whether to record demand) — but only when a GPU was
        // expressed, otherwise `exact_gpu_match` is definitionally
        // false and we skip the lookup entirely.
        let exact_gpu_match = if gpu.is_empty() {
            false
        } else {
            registry.resolve_queue_pool(bundle, gpu).await.is_some()
        };
        return PoolLookup {
            resolution: PoolResolution::Pool(pool_name.to_string()),
            exact_gpu_match,
        };
    }

    // Primary lookup. Folds the "was the exact tuple routable?"
    // question into the same registry load we use to pick a pool.
    let primary = registry.resolve_queue_pool(bundle, gpu).await;
    let exact_gpu_match = !gpu.is_empty() && primary.is_some();

    // Fallback: caller expressed a GPU preference but nothing matches;
    // try any healthy worker on the bundle. This covers single-GPU
    // clusters where the profile-level distinction is cosmetic.
    let resolved = match primary {
        Some(p) => Some(p),
        None if !gpu.is_empty() => registry.resolve_queue_pool(bundle, "").await,
        None => None,
    };

    let resolution = match resolved {
        Some(p) => PoolResolution::Pool(p),
        None => PoolResolution::Provisioning,
    };
    PoolLookup {
        resolution,
        exact_gpu_match,
    }
}

pub async fn proxy_encode(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "encode").await
}

pub async fn proxy_score(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "score").await
}

pub async fn proxy_extract(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "extract").await
}

async fn proxy_request(
    State(state): State<Arc<AppState>>,
    req: Request,
    endpoint: &str,
) -> Response {
    // SDK version skew detection
    check_sdk_version(req.headers());

    // Extract model from path: /v1/{endpoint}/{model...}
    let prefix = format!("/v1/{}/", endpoint);
    let path = req.uri().path().to_string();
    let model = path.strip_prefix(&prefix).unwrap_or("").to_string();

    if model.is_empty() {
        // Early exit, headers haven't been parsed yet — pass through
        // placeholder labels. `record_rejected_request` normalizes
        // empty `machine_profile` to `"unknown"` internally; `bundle`
        // gets the same treatment here for cardinality discipline.
        metrics::record_rejected_request("", "unknown", "model_required");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": "model is required"})),
        )
            .into_response();
    }

    // Parse model spec: [bundle:/]org/model
    let (bundle_override, model_name) = parse_model_spec(&model);
    let bundle_override_ref = if bundle_override.is_empty() {
        None
    } else {
        Some(bundle_override.as_str())
    };

    // Try model registry resolution. Three cases:
    //   1. Model is known        → resolve bundle (404 on BundleConflict, etc.)
    //   2. Model unknown, registry populated → 404 (fail fast; avoids queueing
    //      requests for typo'd model ids).
    //   3. Model unknown, registry empty     → fall back to caller's bundle
    //      override or "default". This is the pre-bootstrap / no-config
    //      deployment path; workers may still match on bundle+gpu alone.
    let bundle = if state.model_registry.model_exists(&model_name) {
        match state
            .model_registry
            .resolve_bundle(&model_name, bundle_override_ref)
        {
            Ok(b) => b,
            Err(ResolveError::ModelNotFound(e)) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"message": e.to_string()})),
                )
                    .into_response();
            }
            Err(ResolveError::BundleConflict(e)) => {
                return (
                    StatusCode::CONFLICT,
                    Json(json!({
                        "message": e.to_string(),
                        "compatible_bundles": e.compatible_bundles,
                    })),
                )
                    .into_response();
            }
        }
    } else if state.model_registry.has_any_models() {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "message": format!("Unknown model '{}'", model_name),
            })),
        )
            .into_response();
    } else if bundle_override.is_empty() {
        "default".to_string()
    } else {
        bundle_override.clone()
    };

    // Parse GPU from X-SIE-MACHINE-PROFILE header
    let mut gpu = req
        .headers()
        .get("x-sie-machine-profile")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let mut pool_name = req
        .headers()
        .get("x-sie-pool")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    // Parse pool from GPU param (e.g., "eval-l4/l4")
    if !gpu.is_empty() && gpu.contains('/') {
        let parts: Vec<&str> = gpu.splitn(2, '/').collect();
        pool_name = parts[0].to_string();
        gpu = parts[1].to_string();
    }

    // Resolve bare GPU to spot variant
    if !gpu.is_empty() && !state.config.configured_gpus.is_empty() {
        gpu = resolve_machine_profile(&gpu, &state.config.gpu_profile_map);
    }

    // Publish the canonical `machine_profile` to the HTTP metrics
    // middleware via a request extension slot. The middleware reads
    // this AFTER the inner service responds, so every downstream
    // return path below — including the `gpu_not_configured` rejection
    // one block down and all the early exits inside `queue_mode_proxy`
    // — automatically gets the normalized label without each site
    // having to remember to tag its response. A fallback of
    // `"unknown"` kicks in at the middleware if we never set the slot
    // (e.g. the `model is required` exit above, which has no GPU).
    if let Some(slot) = req.extensions().get::<metrics::MetricLabelsSlot>() {
        slot.set(metrics::MetricLabels {
            machine_profile: if gpu.is_empty() {
                "unknown".to_string()
            } else {
                gpu.clone()
            },
        });
    }

    // Validate GPU is configured
    if !gpu.is_empty() && !state.config.configured_gpus.is_empty() {
        let found = state
            .config
            .configured_gpus
            .iter()
            .any(|cg| cg.eq_ignore_ascii_case(&gpu));

        if !found {
            metrics::record_rejected_request(&gpu, &bundle, "gpu_not_configured");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "gpu_not_configured",
                    "gpu": gpu,
                    "configured_gpu_types": state.config.configured_gpus,
                    "message": format!("GPU type '{}' is not configured in this cluster.", gpu),
                })),
            )
                .into_response();
        }
    }

    let Some(work_publisher) = state.work_publisher.as_ref() else {
        metrics::record_rejected_request(&gpu, &bundle, "queue_unavailable");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "message": "Rust gateway is queue-only, but NATS JetStream is unavailable"
            })),
        )
            .into_response();
    };

    // Resolve the effective pool in one shot. `resolve_effective_pool`
    // folds the demand-tracking probe ("was there an exact (bundle,
    // gpu) match?") into the same registry load it uses to pick a
    // pool, so we don't make two `resolve_queue_pool` calls on the
    // hot path. The `exact_gpu_match` flag is what drives pending
    // demand for KEDA — we record whenever the caller expressed a GPU
    // preference but no exact-tuple worker was registered, regardless
    // of whether we still serve this request via bundle-fallback or a
    // pinned pool (see `test_scaling.py::test_pending_demand_*`).
    let lookup = resolve_effective_pool(&state.registry, &bundle, &gpu, &pool_name).await;
    if !gpu.is_empty() && !lookup.exact_gpu_match {
        state.demand_tracker.record(&gpu, &bundle);
    }

    let effective_pool = match lookup.resolution {
        PoolResolution::Pool(p) => p,
        PoolResolution::Provisioning => {
            let gpu_label = if gpu.is_empty() { "any" } else { gpu.as_str() };
            info!(
                gpu = %gpu_label,
                bundle = %bundle,
                "no queue worker available, returning 202",
            );
            metrics::PROVISIONING_RESPONSES
                .with_label_values(&[gpu_label])
                .inc();
            // Note: if the caller expressed a GPU preference, demand
            // was already recorded above via `exact_gpu_match = false`.
            // We still record here (idempotent — see
            // `DemandTracker::record`) so the empty-GPU case also
            // gets a pending-demand entry, matching the prior
            // behavior exactly.
            state.demand_tracker.record(&gpu, &bundle);
            let message = if gpu.is_empty() {
                format!(
                    "No worker available for bundle '{}'. Provisioning in progress.",
                    bundle
                )
            } else {
                format!(
                    "No worker available for GPU type '{}'. Provisioning in progress.",
                    gpu
                )
            };
            let mut resp = (
                StatusCode::ACCEPTED,
                Json(json!({
                    "status": "provisioning",
                    "gpu": gpu,
                    "bundle": bundle,
                    "estimated_wait_s": ESTIMATED_WAIT_S,
                    "message": message,
                })),
            )
                .into_response();
            resp.headers_mut().insert(
                HeaderName::from_static("retry-after"),
                HeaderValue::from_static(DEFAULT_RETRY_AFTER),
            );
            return resp;
        }
    };

    let token_id = extract_bearer_token(req.headers())
        .map(|t| mask_token(&t))
        .unwrap_or_default();
    let content_length = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(-1);

    // Extract the only two header-derived bits `queue_mode_proxy`
    // actually needs (request content-type and response negotiation)
    // *before* consuming the request. This lets us skip cloning the
    // entire `HeaderMap` just to read two flags on the hot path.
    let is_msgpack_in = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("msgpack"))
        .unwrap_or(false);
    let use_msgpack_out = publisher::wants_msgpack(req.headers());

    const MAX_PROXY_BODY: usize = 256 * 1024 * 1024;
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_PROXY_BODY).await {
        Ok(b) => b,
        Err(e) => {
            warn!(error = %e, limit = MAX_PROXY_BODY, "request body too large or read error");
            metrics::record_rejected_request(&gpu, &bundle, "body_too_large");
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json!({"message": format!("Request body too large (max {} bytes)", MAX_PROXY_BODY)})),
            )
                .into_response();
        }
    };

    queue_mode_proxy(
        &state,
        work_publisher,
        endpoint,
        &model_name,
        &bundle,
        &gpu,
        &effective_pool,
        &body_bytes,
        is_msgpack_in,
        use_msgpack_out,
        &token_id,
        content_length,
        Instant::now(),
    )
    .await
}

/// Route request through the queue-only JetStream path.
///
/// `pool` is always pre-resolved by the caller via
/// [`resolve_effective_pool`] and is guaranteed non-empty: the
/// `PoolResolution::Provisioning` branch returns `202` before we get
/// here, and every `PoolResolution::Pool(_)` path produces a non-empty
/// string (either the caller-pinned `X-SIE-Pool`, or a `pool_name`
/// harvested from the registry snapshot — `resolve_queue_pool` filters
/// out empty pool names). We therefore don't need to re-query the
/// registry inside this function.
#[allow(clippy::too_many_arguments)]
async fn queue_mode_proxy(
    state: &AppState,
    work_publisher: &publisher::WorkPublisher,
    endpoint: &str,
    model: &str,
    bundle: &str,
    gpu: &str,
    pool: &str,
    body_bytes: &[u8],
    is_msgpack_in: bool,
    use_msgpack_out: bool,
    token_id: &str,
    content_length: i64,
    start: Instant,
) -> Response {
    // Parse body once, extract items + params (avoids double parse)
    let (items, params) = match parse_queue_request(body_bytes, is_msgpack_in, endpoint) {
        Ok(r) => r,
        Err(e) => {
            metrics::record_rejected_request(gpu, bundle, "body_parse_error");
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"message": format!("Failed to parse request body: {}", e)})),
            )
                .into_response();
        }
    };

    if items.is_empty() && endpoint != "score" {
        metrics::record_rejected_request(gpu, bundle, "empty_items");
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": "No items found in request body"})),
        )
            .into_response();
    }

    // Compute bundle config hash for worker config-skew detection
    let bundle_config_hash = state.model_registry.compute_bundle_config_hash(bundle);

    let publish_start = Instant::now();
    let (request_id, rx) = match work_publisher
        .publish_work(
            pool,
            endpoint,
            model,
            bundle,
            gpu,
            &bundle_config_hash,
            items,
            &params,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "failed to publish work");
            let lower = e.to_lowercase();
            if lower.contains("score request missing query item") {
                metrics::record_rejected_request(gpu, bundle, "score_missing_query");
                return (StatusCode::BAD_REQUEST, Json(json!({"message": e}))).into_response();
            }

            let mut response = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"message": format!("Queue publish failed: {}", e)})),
            )
                .into_response();

            if lower.contains("no consumers") {
                metrics::record_rejected_request(gpu, bundle, "no_consumers");
                response.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_static(DEFAULT_RETRY_AFTER),
                );
            } else if lower.contains("backpressure") {
                metrics::record_rejected_request(gpu, bundle, "backpressure");
                response.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_static(BACKPRESSURE_RETRY_AFTER),
                );
            } else {
                metrics::record_rejected_request(gpu, bundle, "queue_publish_failed");
            }

            response.headers_mut().insert(
                HeaderName::from_static("x-sie-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            response.headers_mut().insert(
                HeaderName::from_static("x-sie-server-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            return response;
        }
    };
    let publish_elapsed = publish_start.elapsed();

    // Wait for results (use configured request_timeout instead of hardcoded 300s)
    let timeout_secs = state.config.request_timeout as u64;
    let wait_start = Instant::now();
    let results = match tokio::time::timeout(Duration::from_secs(timeout_secs), rx).await {
        Ok(Ok(results)) => results,
        Ok(Err(_)) => {
            metrics::record_rejected_request(gpu, bundle, "result_channel_closed");
            return (
                StatusCode::GATEWAY_TIMEOUT,
                Json(json!({"message": "Result channel closed"})),
            )
                .into_response();
        }
        Err(_) => {
            // Upstream timeout: the most common cause is a worker cold-loading
            // the target model on demand. The worker NAKs the JetStream message
            // and redelivers after load, but that cycle typically exceeds our
            // per-request timeout on the first call. Surface this as
            // 503 + MODEL_LOADING so SDK clients with wait_for_capacity=True
            // retry using the existing model-loading contract rather than
            // failing the cold-start request. Genuinely hung upstreams are
            // still bounded by the SDK's provision_timeout_s budget.
            //
            // The rejection reason is intentionally only the specific
            // `upstream_timeout_model_loading` — emitting both that
            // and a generic `result_timeout` for the same event would
            // double-count the timeout on the error-rate dashboards
            // and break rate alerts that sum across reasons.
            metrics::record_rejected_request(gpu, bundle, "upstream_timeout_model_loading");
            return build_model_loading_timeout_response(model, timeout_secs);
        }
    };
    let wait_elapsed = wait_start.elapsed();

    let elapsed = start.elapsed();
    let use_msgpack = use_msgpack_out;

    // Assemble response matching Python's envelope: {"model": "...", "items": [...]}
    let successful: Vec<&publisher::WorkResult> = results.iter().filter(|r| r.success).collect();
    let errors: Vec<&publisher::WorkResult> = results.iter().filter(|r| !r.success).collect();

    if successful.is_empty() && !errors.is_empty() {
        // Translate retryable worker error codes into the SDK-expected 503
        // contract. Without this every per-item failure surfaced as 500
        // ``all_items_failed`` and the SDK retry path never engaged. We
        // require a *unanimous* code across all failed items so we don't
        // mis-translate a mixed batch (e.g. one item OOM, another invalid
        // input) — only homogeneous, unambiguous cases get retried.
        if let Some(code) = unanimous_retryable_error_code(&errors) {
            let first_msg = errors
                .first()
                .and_then(|r| r.error.as_deref())
                .unwrap_or("Worker reported a retryable error");
            metrics::record_rejected_request(gpu, bundle, retryable_metric_reason(code));
            return build_retryable_error_response(code, first_msg);
        }

        let error_details: Vec<serde_json::Value> = errors
            .iter()
            .map(|r| {
                let mut entry = json!({"item_index": r.item_index, "error": r.error});
                // Surface the per-item ``error_code`` for observability —
                // useful when a mixed batch lands here (some
                // RESOURCE_EXHAUSTED, some genuine inference errors).
                if let Some(code) = r.error_code.as_deref() {
                    entry["code"] = json!(code);
                }
                entry
            })
            .collect();
        metrics::record_rejected_request(gpu, bundle, "all_items_failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "all_items_failed", "details": error_details})),
        )
            .into_response();
    }

    let content_key = if endpoint == "score" {
        "scores"
    } else {
        "items"
    };
    let status: u16 = 200;

    let resp_body = if use_msgpack {
        // Msgpack: build {"model": ..., "items"|"scores": [raw_blobs...]} at byte level
        let result_blobs: Vec<&[u8]> = successful
            .iter()
            .map(|r| r.result_msgpack.as_slice())
            .collect();
        let mut packer = rmp::encode::buffer::ByteBuf::new();
        let n_keys = if errors.is_empty() { 2u32 } else { 3 };
        rmp::encode::write_map_len(&mut packer, n_keys).unwrap();
        rmp::encode::write_str(&mut packer, "model").unwrap();
        rmp::encode::write_str(&mut packer, model).unwrap();
        rmp::encode::write_str(&mut packer, content_key).unwrap();
        if endpoint == "score" && result_blobs.len() == 1 {
            // Score: single blob is the complete list, embed directly
            let mut parts = packer.into_vec();
            parts.extend_from_slice(result_blobs[0]);
            parts
        } else {
            rmp::encode::write_array_len(&mut packer, result_blobs.len() as u32).unwrap();
            let mut parts = packer.into_vec();
            for blob in result_blobs {
                parts.extend_from_slice(blob);
            }
            parts
        }
    } else {
        // JSON: decode each blob, convert numpy arrays, wrap in envelope.
        //
        // The result blobs typically carry msgpack_numpy-encoded
        // arrays — `{"nd": true, "type": "<f4", "shape": [...],
        // "data": <binary>}`. `rmpv_to_response_json` detects the
        // sentinel while walking the `rmpv::Value` tree and decodes
        // `Binary` / `Ext` blobs directly from their byte slice, so
        // we never materialise the `data` buffer as a
        // `Vec<serde_json::Value::Number>` (one Number per byte)
        // just to re-collect it back into bytes for dtype decoding.
        let mut result_items: Vec<serde_json::Value> = successful
            .iter()
            .filter_map(|r| {
                let rmpv_val: rmpv::Value = rmp_serde::from_slice(&r.result_msgpack).ok()?;
                Some(rmpv_to_response_json(rmpv_val))
            })
            .collect();
        // Score: the single blob *is* the candidate-list already, so
        // we unwrap it and hand the inner array straight to the
        // envelope. Use `into_iter().next()` to move the item out
        // instead of cloning — score payloads can be thousands of
        // entries long and this used to clone the whole vec.
        let items_val = if endpoint == "score" && result_items.len() == 1 {
            match result_items.pop() {
                Some(serde_json::Value::Array(arr)) => serde_json::Value::Array(arr),
                Some(other) => serde_json::Value::Array(vec![other]),
                None => serde_json::Value::Array(Vec::new()),
            }
        } else {
            serde_json::Value::Array(result_items)
        };
        let mut response_data = json!({
            "model": model,
            content_key: items_val,
        });
        if !errors.is_empty() {
            let error_details: Vec<serde_json::Value> = errors
                .iter()
                .map(|r| json!({"item_index": r.item_index, "error": r.error}))
                .collect();
            response_data["errors"] = json!(error_details);
        }
        serde_json::to_vec(&response_data).unwrap_or_default()
    };

    debug!(
        request_id = %request_id,
        endpoint = endpoint,
        model = %model,
        status = status,
        latency_ms = elapsed.as_millis(),
        "queue mode response"
    );

    // `REQUEST_COUNT` and `REQUEST_LATENCY` are now emitted by
    // `middleware::metrics::MetricsLayer` for *every* response on the
    // inference routes, including early returns (404, 413, 503, 504,
    // 202 provisioning, ...). Do not re-emit them here or the success
    // path would be double-counted. The worker-registry per-request
    // bookkeeping stays — it is independent of Prometheus.
    state.registry.record_request("queue").await;

    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: endpoint.to_string(),
        status,
        token_id: token_id.to_string(),
        model: model.to_string(),
        pool: pool.to_string(),
        gpu: gpu.to_string(),
        worker: format!("queue:{}", request_id),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let content_type = if use_msgpack {
        "application/x-msgpack"
    } else {
        "application/json"
    };

    let mut response = Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::OK))
        .body(Body::from(resp_body))
        .unwrap();
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-request-id"),
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );
    insert_duration_header(
        response.headers_mut(),
        "x-queue-publish-time",
        publish_elapsed,
    );
    insert_duration_header(response.headers_mut(), "x-queue-wait-time", wait_elapsed);
    insert_queue_worker_timing_headers(response.headers_mut(), &successful);
    let worker_tag = format!("queue:{}", request_id);
    if let Ok(val) = HeaderValue::from_str(&worker_tag) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-sie-worker"), val);
    }
    response
}

/// Return the retryable error code shared by every failed item, or
/// ``None`` if the batch is mixed / non-retryable.
///
/// Mixed batches go through the legacy ``all_items_failed`` 500 path so
/// callers can inspect per-item codes; unanimous retryable batches get
/// the dedicated 503 + ``Retry-After`` contract that the SDK already
/// understands. The set of codes that count as retryable here mirrors
/// the SDK's auto-retry table — see ``sie_sdk.client._shared``.
fn unanimous_retryable_error_code(errors: &[&publisher::WorkResult]) -> Option<&'static str> {
    let first = errors.first()?.error_code.as_deref()?;
    let canonical = match first {
        RESOURCE_EXHAUSTED_ERROR_CODE => RESOURCE_EXHAUSTED_ERROR_CODE,
        MODEL_LOADING_ERROR_CODE => MODEL_LOADING_ERROR_CODE,
        LORA_LOADING_ERROR_CODE => LORA_LOADING_ERROR_CODE,
        _ => return None,
    };
    if errors
        .iter()
        .all(|r| r.error_code.as_deref() == Some(canonical))
    {
        Some(canonical)
    } else {
        None
    }
}

/// Metric label used when rejecting a request because workers emitted a
/// retryable error code unanimously. Keeps Prometheus reasons stable
/// instead of folding everything under ``all_items_failed``.
fn retryable_metric_reason(code: &str) -> &'static str {
    match code {
        RESOURCE_EXHAUSTED_ERROR_CODE => "resource_exhausted",
        MODEL_LOADING_ERROR_CODE => "upstream_model_loading",
        LORA_LOADING_ERROR_CODE => "upstream_lora_loading",
        _ => "all_items_failed",
    }
}

/// Build a ``503 + <code>`` response that mirrors the worker-side HTTP
/// contract (see ``packages/sie_server/src/sie_server/api/helpers.py``).
///
///   * status:  503 Service Unavailable
///   * body:    ``{"error": {"code": <code>, "message": <upstream message>}}``
///   * headers: ``Retry-After: 5``, ``X-SIE-Error-Code: <code>``, plus the
///     standard ``X-SIE-*`` version pair.
///
/// The worker is **not** marked unhealthy — these codes are transient
/// per-request signals, not worker-health signals.
fn build_retryable_error_response(code: &'static str, message: &str) -> Response {
    let retry_after = match code {
        RESOURCE_EXHAUSTED_ERROR_CODE => RESOURCE_EXHAUSTED_RETRY_AFTER,
        MODEL_LOADING_ERROR_CODE => MODEL_LOADING_RETRY_AFTER,
        LORA_LOADING_ERROR_CODE => LORA_LOADING_RETRY_AFTER,
        // Defensive default. Should be unreachable given the
        // ``unanimous_retryable_error_code`` allow-list (the only caller
        // pathway), but if a future code is added there without here, fall
        // back to the most conservative retry hint we know rather than
        // panicking in production. The ``debug_assert!`` ensures any such
        // mismatch fails loudly in tests / dev builds.
        _ => {
            debug_assert!(
                false,
                "build_retryable_error_response called with unmapped code: {code}"
            );
            MODEL_LOADING_RETRY_AFTER
        }
    };
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "code": code,
                "message": message,
            },
        })),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from_str(retry_after).unwrap_or_else(|_| HeaderValue::from_static("5")),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_str(code).unwrap_or_else(|_| HeaderValue::from_static("ERROR")),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

/// Build a `503 + MODEL_LOADING` response for an upstream timeout.
///
/// Mirrors the contract the SDK already implements for worker-emitted
/// `MODEL_LOADING` errors (see `sie_sdk.client._shared`):
///   * status:  503 Service Unavailable
///   * body:    `{"error": {"code": "MODEL_LOADING", "message": ...}}`
///   * headers: `Retry-After: 5`, plus the standard `X-SIE-*` version pair.
fn build_model_loading_timeout_response(model: &str, timeout_secs: u64) -> Response {
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "code": MODEL_LOADING_ERROR_CODE,
                "message": format!(
                    "Model '{}' did not respond within {}s; worker may be loading the model on demand. Please retry.",
                    model, timeout_secs
                ),
            },
        })),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from_static(MODEL_LOADING_RETRY_AFTER),
    );
    // Advertised in architecture-guide.md, README.md, and
    // docs/queue-based-routing.md. The SDK parses the body (see
    // sie_sdk.client._shared.get_error_code), but external clients and
    // retry/observability middleware key off this header.
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(MODEL_LOADING_ERROR_CODE),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

fn parse_model_spec(spec: &str) -> (String, String) {
    if let Some(idx) = spec.find(":/") {
        (spec[..idx].to_string(), spec[idx + 2..].to_string())
    } else {
        (String::new(), spec.to_string())
    }
}

fn resolve_machine_profile(
    gpu: &str,
    gpu_profile_map: &std::collections::HashMap<String, String>,
) -> String {
    // Lowercase the input once and reuse it for both the exact and the
    // `-spot` variant lookup. Previously we paid two `to_lowercase()`
    // heap allocations per request even in the common "already
    // canonical, not in the map" case.
    let gpu_lower = gpu.to_ascii_lowercase();
    if let Some(val) = gpu_profile_map.get(&gpu_lower) {
        return val.clone();
    }

    let mut spot_key = gpu_lower;
    spot_key.push_str("-spot");
    if let Some(val) = gpu_profile_map.get(&spot_key) {
        info!(from = gpu, to = %val, "resolved machine_profile");
        return val.clone();
    }

    gpu.to_string()
}

fn insert_duration_header(headers: &mut HeaderMap, name: &'static str, duration: Duration) {
    if let Ok(value) = HeaderValue::from_str(&format!("{:.1}", duration.as_secs_f64() * 1000.0)) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn insert_timing_header(headers: &mut HeaderMap, name: &'static str, value_ms: f64) {
    if let Ok(value) = HeaderValue::from_str(&format!("{value_ms:.1}")) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn max_result_timing<F>(results: &[&publisher::WorkResult], field: F) -> Option<f64>
where
    F: Fn(&publisher::WorkResult) -> Option<f64>,
{
    results
        .iter()
        .filter_map(|result| field(result))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn insert_optional_timing_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value_ms: Option<f64>,
) {
    let Some(value_ms) = value_ms else {
        return;
    };
    if value_ms <= 0.0 {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(&format!("{value_ms:.1}")) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn insert_queue_worker_timing_headers(
    headers: &mut HeaderMap,
    successful: &[&publisher::WorkResult],
) {
    if successful.is_empty() {
        return;
    }

    insert_timing_header(
        headers,
        "x-queue-time",
        max_result_timing(successful, |result| result.queue_ms).unwrap_or(0.0),
    );
    insert_timing_header(
        headers,
        "x-inference-time",
        max_result_timing(successful, |result| result.inference_ms).unwrap_or(0.0),
    );
    insert_optional_timing_header(
        headers,
        "x-tokenization-time",
        max_result_timing(successful, |result| result.tokenization_ms),
    );
    insert_optional_timing_header(
        headers,
        "x-postprocessing-time",
        max_result_timing(successful, |result| result.postprocessing_ms),
    );
    insert_optional_timing_header(
        headers,
        "x-payload-fetch-time",
        max_result_timing(successful, |result| result.payload_fetch_ms),
    );
}

/// Check client SDK version skew.
/// Warns once per minor version if client SDK differs by >1 minor version.
fn check_sdk_version(headers: &HeaderMap) {
    let Some(sdk_version) = headers
        .get("x-sie-sdk-version")
        .and_then(|v| v.to_str().ok())
    else {
        return;
    };

    // Fast path: this SDK version was already parsed on a previous
    // request. A successful hit avoids the `split('.')` allocation
    // and `u32::parse` on every subsequent request.
    let cached = SDK_VERSION_CACHE.get(sdk_version).map(|v| *v);
    let sdk_minor = match cached {
        Some(Some(m)) => Some(m),
        Some(None) => return, // header is malformed; stop re-parsing it
        None => {
            // First request for this version string — parse once and
            // memoise. Parse semver-like string (e.g. "0.2.3" → 2).
            let parsed = sdk_version
                .split('.')
                .nth(1)
                .and_then(|p| p.parse::<u32>().ok());
            // Size-capped insert: once the cache is full we stop
            // memoising so a hostile client can't walk unique
            // header values and grow the map forever. `len()` is
            // a snapshot so two racing inserts can push us one or
            // two entries over the cap — that's fine, the point
            // is bounded growth, not a strict bound.
            if SDK_VERSION_CACHE.len() < SDK_VERSION_CACHE_CAP {
                SDK_VERSION_CACHE.insert(Arc::<str>::from(sdk_version), parsed);
            }
            parsed
        }
    };

    let Some(sdk_minor) = sdk_minor else { return };

    if sdk_minor.abs_diff(*GATEWAY_VERSION_MINOR) > 1 {
        let mut warned = SDK_WARNED_MINORS.lock().unwrap();
        if warned.insert(sdk_minor) {
            warn!(
                sdk_version = %sdk_version,
                gateway_version = GATEWAY_VERSION,
                "client SDK version skew detected (>1 minor version difference)"
            );
        }
    }
}

/// Parse request body once, extract both raw items and work params for queue mode.
///
/// `is_msgpack` is computed by the caller from the `content-type`
/// header before the request body is consumed, which lets us avoid
/// holding on to a full `HeaderMap` clone just to read two flags.
///
/// Items are returned as `rmpv::Value`. This lets msgpack request
/// bodies pass straight through to the worker without the old
/// `msgpack → rmpv::Value → serde_json::Value → msgpack` detour
/// (which in particular blew every `bin` field up into a
/// `Vec<serde_json::Value::Number>` — ~16 MiB of allocations per
/// 1 MiB of binary input). JSON bodies are converted to `rmpv` once
/// via [`json_to_rmpv`]; JSON has no binary or ext types so that
/// conversion is cheap and lossless.
fn parse_queue_request(
    body: &[u8],
    is_msgpack: bool,
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), String> {
    if is_msgpack {
        parse_queue_request_msgpack(body, endpoint)
    } else {
        parse_queue_request_json(body, endpoint)
    }
}

fn parse_queue_request_json(
    body: &[u8],
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), String> {
    let mut parsed: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("json decode: {}", e))?;

    if !parsed.is_object() {
        return Err("Request body must be a JSON object".to_string());
    }

    let params = work_params_from_json(&parsed, endpoint);

    let items_json = if let Some(map) = parsed.as_object_mut() {
        if endpoint == "score" {
            match map.remove("items") {
                Some(serde_json::Value::Array(arr)) => arr,
                Some(_) => return Err("'items' must be an array".to_string()),
                None => Vec::new(),
            }
        } else if let Some(value) = map.remove("items") {
            match value {
                serde_json::Value::Array(arr) => arr,
                _ => return Err("'items' must be an array".to_string()),
            }
        } else if let Some(val) = map.remove("input") {
            match val {
                serde_json::Value::Array(arr) => arr,
                other => vec![other],
            }
        } else if let Some(value) = map.remove("inputs") {
            match value {
                serde_json::Value::Array(arr) => arr,
                _ => return Err("'inputs' must be an array".to_string()),
            }
        } else {
            vec![parsed]
        }
    } else {
        vec![parsed]
    };

    let items = items_json.into_iter().map(json_to_rmpv).collect();
    Ok((items, params))
}

fn parse_queue_request_msgpack(
    body: &[u8],
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), String> {
    let parsed: rmpv::Value =
        rmp_serde::from_slice(body).map_err(|e| format!("msgpack decode: {}", e))?;

    // Parity with `parse_queue_request_json`: top-level must be a
    // map. The JSON path rejects scalars/arrays with 400 before we
    // ever reach a fallback, so reject the same shapes on the
    // msgpack path instead of silently turning e.g. a top-level
    // array into `items = vec![<array>]`, which would only fail
    // later in worker-specific ways.
    let mut map = match parsed {
        rmpv::Value::Map(m) => m,
        _ => return Err("Request body must be a msgpack map".to_string()),
    };

    let params = work_params_from_rmpv(&map, endpoint);

    let items: Vec<rmpv::Value> = if endpoint == "score" {
        match rmpv_map_remove(&mut map, "items") {
            Some(rmpv::Value::Array(arr)) => arr,
            Some(_) => return Err("'items' must be an array".to_string()),
            None => Vec::new(),
        }
    } else if let Some(value) = rmpv_map_remove(&mut map, "items") {
        match value {
            rmpv::Value::Array(arr) => arr,
            _ => return Err("'items' must be an array".to_string()),
        }
    } else if let Some(val) = rmpv_map_remove(&mut map, "input") {
        match val {
            rmpv::Value::Array(arr) => arr,
            other => vec![other],
        }
    } else if let Some(value) = rmpv_map_remove(&mut map, "inputs") {
        match value {
            rmpv::Value::Array(arr) => arr,
            _ => return Err("'inputs' must be an array".to_string()),
        }
    } else {
        // No recognised items key — treat the remaining map as a
        // single item. Rebuild the map value (the original was
        // consumed above — `work_params_from_rmpv` only borrowed it
        // but the items-lookup calls used `rmpv_map_remove`, which
        // mutates; fields that the params extractor cares about are
        // still present because the remove helpers only strip the
        // items-related keys).
        vec![rmpv::Value::Map(map)]
    };

    Ok((items, params))
}

/// Remove the first entry whose key (string or binary UTF-8) matches
/// `key`, returning its value. Lookup is O(n) but n is the number of
/// top-level request fields (≤ ~10), so the cost is negligible and we
/// avoid allocating an intermediate map.
fn rmpv_map_remove(map: &mut Vec<(rmpv::Value, rmpv::Value)>, key: &str) -> Option<rmpv::Value> {
    let pos = map.iter().position(|(k, _)| rmpv_key_eq(k, key))?;
    Some(map.swap_remove(pos).1)
}

fn rmpv_map_get<'a>(map: &'a [(rmpv::Value, rmpv::Value)], key: &str) -> Option<&'a rmpv::Value> {
    map.iter()
        .find(|(k, _)| rmpv_key_eq(k, key))
        .map(|(_, v)| v)
}

fn rmpv_key_eq(key: &rmpv::Value, expected: &str) -> bool {
    match key {
        rmpv::Value::String(s) => s.as_str() == Some(expected),
        // Python msgpack without strict_map_key=True emits bin keys.
        rmpv::Value::Binary(b) => std::str::from_utf8(b).ok() == Some(expected),
        _ => false,
    }
}

fn rmpv_as_str(value: &rmpv::Value) -> Option<&str> {
    match value {
        rmpv::Value::String(s) => s.as_str(),
        _ => None,
    }
}

fn rmpv_as_bool(value: &rmpv::Value) -> Option<bool> {
    match value {
        rmpv::Value::Boolean(b) => Some(*b),
        _ => None,
    }
}

fn rmpv_as_array(value: &rmpv::Value) -> Option<&[rmpv::Value]> {
    match value {
        rmpv::Value::Array(a) => Some(a),
        _ => None,
    }
}

fn rmpv_string_array(value: &rmpv::Value) -> Option<Vec<String>> {
    rmpv_as_array(value).map(|arr| {
        arr.iter()
            .filter_map(|v| rmpv_as_str(v).map(String::from))
            .collect()
    })
}

/// Convert a small config-style `rmpv::Value` back to
/// `serde_json::Value` for the `options` / `output_schema` fields.
/// These are always tiny (a handful of flags/strings) and never
/// carry binary, so the conversion cost is negligible — and keeping
/// the `WorkParams` type stable here avoids a cascade of changes
/// into rest of the gateway that only cares about their structural
/// shape.
fn rmpv_to_json_owned(value: &rmpv::Value) -> serde_json::Value {
    rmpv_to_json(value.clone())
}

fn work_params_from_json(parsed: &serde_json::Value, endpoint: &str) -> publisher::WorkParams {
    if endpoint == "score" {
        return publisher::WorkParams {
            output_types: None,
            instruction: parsed
                .get("instruction")
                .and_then(|v| v.as_str())
                .map(String::from),
            is_query: false,
            options: parsed.get("options").cloned(),
            labels: None,
            output_schema: None,
            query_item: Some(
                parsed
                    .get("query")
                    .cloned()
                    .map(json_to_rmpv)
                    .unwrap_or_else(|| rmpv::Value::Map(Vec::new())),
            ),
        };
    }

    let nested_params = parsed.get("params");
    let shared_field = |key: &str| {
        nested_params
            .and_then(|params| params.get(key))
            .or_else(|| parsed.get(key))
    };
    let options = shared_field("options").cloned();

    publisher::WorkParams {
        output_types: shared_field("output_types")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
        instruction: shared_field("instruction")
            .and_then(|v| v.as_str())
            .map(String::from),
        is_query: shared_field("is_query")
            .and_then(|v| v.as_bool())
            .or_else(|| {
                options
                    .as_ref()
                    .and_then(|value| value.get("is_query"))
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false),
        options,
        labels: shared_field("labels")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            }),
        output_schema: shared_field("output_schema").cloned(),
        query_item: None,
    }
}

fn work_params_from_rmpv(
    parsed: &[(rmpv::Value, rmpv::Value)],
    endpoint: &str,
) -> publisher::WorkParams {
    if endpoint == "score" {
        return publisher::WorkParams {
            output_types: None,
            instruction: rmpv_map_get(parsed, "instruction")
                .and_then(rmpv_as_str)
                .map(String::from),
            is_query: false,
            options: rmpv_map_get(parsed, "options").map(rmpv_to_json_owned),
            labels: None,
            output_schema: None,
            query_item: Some(
                rmpv_map_get(parsed, "query")
                    .cloned()
                    .unwrap_or_else(|| rmpv::Value::Map(Vec::new())),
            ),
        };
    }

    // For `encode`/`extract`, callers may nest the param block under
    // `"params"`; fall through to the top-level if the nested copy
    // doesn't carry the field.
    let nested = rmpv_map_get(parsed, "params").and_then(|v| match v {
        rmpv::Value::Map(m) => Some(m.as_slice()),
        _ => None,
    });
    let shared_field = |key: &str| -> Option<&rmpv::Value> {
        nested
            .and_then(|m| rmpv_map_get(m, key))
            .or_else(|| rmpv_map_get(parsed, key))
    };
    let options_rmpv = shared_field("options");
    let options = options_rmpv.map(rmpv_to_json_owned);
    let is_query = shared_field("is_query")
        .and_then(rmpv_as_bool)
        .or_else(|| {
            options_rmpv
                .and_then(|v| match v {
                    rmpv::Value::Map(m) => rmpv_map_get(m, "is_query"),
                    _ => None,
                })
                .and_then(rmpv_as_bool)
        })
        .unwrap_or(false);

    publisher::WorkParams {
        output_types: shared_field("output_types").and_then(rmpv_string_array),
        instruction: shared_field("instruction")
            .and_then(rmpv_as_str)
            .map(String::from),
        is_query,
        options,
        labels: shared_field("labels").and_then(rmpv_string_array),
        output_schema: shared_field("output_schema").map(rmpv_to_json_owned),
        query_item: None,
    }
}

/// One-shot conversion from `serde_json::Value` to `rmpv::Value`.
/// Used for the JSON request-body path: workers all speak msgpack, so
/// we normalize to `rmpv` once at ingress and avoid having two item
/// representations flowing through the rest of the publisher.
/// JSON has no binary or ext types, so this is lossless and cheap
/// (no per-byte blow-up like `rmpv_to_json` suffered in the opposite
/// direction).
fn json_to_rmpv(value: serde_json::Value) -> rmpv::Value {
    match value {
        serde_json::Value::Null => rmpv::Value::Nil,
        serde_json::Value::Bool(b) => rmpv::Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                rmpv::Value::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                rmpv::Value::Integer(u.into())
            } else if let Some(f) = n.as_f64() {
                rmpv::Value::F64(f)
            } else {
                rmpv::Value::Nil
            }
        }
        serde_json::Value::String(s) => rmpv::Value::String(s.into()),
        serde_json::Value::Array(arr) => {
            rmpv::Value::Array(arr.into_iter().map(json_to_rmpv).collect())
        }
        serde_json::Value::Object(map) => rmpv::Value::Map(
            map.into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), json_to_rmpv(v)))
                .collect(),
        ),
    }
}

fn emit_audit_log(entry: AuditEntry) {
    info!(
        event = %entry.event,
        method = %entry.method,
        endpoint = %entry.endpoint,
        status = entry.status,
        token_id = %entry.token_id,
        model = %entry.model,
        pool = %entry.pool,
        gpu = %entry.gpu,
        worker = %entry.worker,
        latency_ms = entry.latency_ms,
        body_bytes = entry.body_bytes,
        "audit"
    );
}

/// Fuses the `msgpack → serde_json` conversion for response bodies
/// with inline `msgpack_numpy` sentinel decoding, so `bin` / `ext`
/// payloads skip the `Vec<serde_json::Value::Number>` (one number per
/// byte) detour that a generic rmpv-to-json conversion would produce.
///
/// Used on the JSON-response hot path. The `decode_dtype_values` /
/// `reshape_array` helpers below do the actual numeric decoding.
fn rmpv_to_response_json(value: rmpv::Value) -> serde_json::Value {
    match value {
        rmpv::Value::Map(entries) => {
            if let Some(decoded) = try_decode_rmpv_numpy(&entries) {
                return decoded;
            }
            let mut map = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                let key = match k {
                    rmpv::Value::String(s) => s.into_str().unwrap_or_default().to_string(),
                    rmpv::Value::Binary(b) => String::from_utf8(b).unwrap_or_default(),
                    other => format!("{}", other),
                };
                map.insert(key, rmpv_to_response_json(v));
            }
            serde_json::Value::Object(map)
        }
        rmpv::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(rmpv_to_response_json).collect())
        }
        other => rmpv_to_json(other),
    }
}

/// Inspect a map that might be a `msgpack_numpy` sentinel and, if so,
/// decode the packed bytes straight into a nested JSON array without
/// ever materializing a `Vec<Number>` byte-by-byte.
///
/// Returns `None` if the map lacks any required sentinel key — the
/// caller then walks the map generically.
fn try_decode_rmpv_numpy(entries: &[(rmpv::Value, rmpv::Value)]) -> Option<serde_json::Value> {
    let mut is_nd = false;
    let mut dtype: Option<&str> = None;
    let mut data: Option<&[u8]> = None;
    let mut shape_src: Option<&[rmpv::Value]> = None;

    for (k, v) in entries {
        let key = match k {
            rmpv::Value::String(s) => s.as_str(),
            rmpv::Value::Binary(b) => std::str::from_utf8(b).ok(),
            _ => None,
        };
        let Some(key) = key else { continue };
        match key {
            "nd" => {
                if let rmpv::Value::Boolean(b) = v {
                    is_nd = *b;
                }
            }
            "type" => {
                if let rmpv::Value::String(s) = v {
                    dtype = s.as_str();
                }
            }
            "data" => {
                data = match v {
                    rmpv::Value::Binary(b) => Some(b.as_slice()),
                    // Some msgpack_numpy variants pack the buffer as
                    // an ext-type (code 0x15/0x17 etc.); the payload
                    // bytes are still the raw dtype-packed buffer.
                    rmpv::Value::Ext(_, b) => Some(b.as_slice()),
                    _ => None,
                };
            }
            "shape" => {
                if let rmpv::Value::Array(a) = v {
                    shape_src = Some(a.as_slice());
                }
            }
            _ => {}
        }
    }

    if !is_nd {
        return None;
    }
    let dtype = dtype?;
    let data = data?;
    let shape: Vec<usize> = shape_src
        .map(|arr| {
            arr.iter()
                .filter_map(|v| match v {
                    rmpv::Value::Integer(i) => i.as_u64().map(|n| n as usize),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let flat_values = decode_dtype_values(dtype, data)?;
    Some(reshape_array(&flat_values, &shape))
}

/// Convert an rmpv::Value to serde_json::Value, handling binary data
/// by converting it to a JSON array of byte values.
fn rmpv_to_json(value: rmpv::Value) -> serde_json::Value {
    match value {
        rmpv::Value::Nil => serde_json::Value::Null,
        rmpv::Value::Boolean(b) => serde_json::Value::Bool(b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                serde_json::Value::Number(n.into())
            } else if let Some(n) = i.as_u64() {
                serde_json::Value::Number(n.into())
            } else {
                serde_json::Value::Null
            }
        }
        rmpv::Value::F32(f) => serde_json::Number::from_f64(f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        rmpv::Value::F64(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        rmpv::Value::String(s) => {
            match s.into_str() {
                Some(s) => serde_json::Value::String(s.to_string()),
                None => serde_json::Value::Null, // Invalid UTF-8
            }
        }
        rmpv::Value::Binary(bytes) => {
            // Binary outside a numpy sentinel: keep the legacy
            // "array of byte values" shape so non-numpy payloads
            // that happen to contain `bin` (rare — workers prefer
            // the numpy sentinel even for 1-D tensors) still
            // serialize to something JSON can represent.
            serde_json::Value::Array(
                bytes
                    .into_iter()
                    .map(|b| serde_json::Value::from(b as u64))
                    .collect(),
            )
        }
        rmpv::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(rmpv_to_json).collect())
        }
        rmpv::Value::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (k, v) in entries {
                // msgpack map keys can be binary strings from Python
                let key = match k {
                    rmpv::Value::String(s) => s.into_str().unwrap_or_default().to_string(),
                    rmpv::Value::Binary(b) => String::from_utf8(b).unwrap_or_default(),
                    other => format!("{}", other),
                };
                map.insert(key, rmpv_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        rmpv::Value::Ext(_, data) => {
            // Extension types: convert data to byte array like Binary
            serde_json::Value::Array(
                data.into_iter()
                    .map(|b| serde_json::Value::from(b as u64))
                    .collect(),
            )
        }
    }
}

/// Decode raw bytes into a flat array of JSON values based on numpy dtype.
fn decode_dtype_values(dtype: &str, data: &[u8]) -> Option<Vec<serde_json::Value>> {
    match dtype {
        "<f4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as f64)
                    })
                    .collect(),
            )
        }
        "<f8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "<f2" => {
            if !data.len().is_multiple_of(2) {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        let val = f16_to_f32(bits);
                        serde_json::Value::from(val as f64)
                    })
                    .collect(),
            )
        }
        "<i4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as i64)
                    })
                    .collect(),
            )
        }
        "<i8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "<i2" => {
            if !data.len().is_multiple_of(2) {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|chunk| {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        serde_json::Value::from(val as i64)
                    })
                    .collect(),
            )
        }
        "<u4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as u64)
                    })
                    .collect(),
            )
        }
        "<u8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "|b1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::Bool(b != 0))
                .collect(),
        ),
        "|u1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::from(b as u64))
                .collect(),
        ),
        "|i1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::from(b as i8 as i64))
                .collect(),
        ),
        _ => {
            warn!(dtype = %dtype, "unsupported numpy dtype in msgpack_numpy conversion");
            None
        }
    }
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((e + 127) as u32) & 0xff;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xff << 23) | f32_mantissa);
    }

    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Reshape a flat array of values into nested JSON arrays according to the given shape.
fn reshape_array(flat: &[serde_json::Value], shape: &[usize]) -> serde_json::Value {
    if shape.is_empty() || shape.len() == 1 {
        return serde_json::Value::Array(flat.to_vec());
    }
    reshape_recursive(flat, shape, 0).0
}

fn reshape_recursive(
    flat: &[serde_json::Value],
    shape: &[usize],
    dim: usize,
) -> (serde_json::Value, usize) {
    if dim == shape.len() - 1 {
        let n = shape[dim].min(flat.len());
        let arr: Vec<serde_json::Value> = flat[..n].to_vec();
        return (serde_json::Value::Array(arr), n);
    }

    let mut result = Vec::with_capacity(shape[dim]);
    let mut offset = 0;
    for _ in 0..shape[dim] {
        if offset >= flat.len() {
            break;
        }
        let (sub_arr, consumed) = reshape_recursive(&flat[offset..], shape, dim + 1);
        result.push(sub_arr);
        offset += consumed;
    }
    (serde_json::Value::Array(result), offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_model_spec ───────────────────────────────────────────

    #[test]
    fn test_parse_model_spec_no_bundle() {
        let (bundle, model) = parse_model_spec("BAAI/bge-m3");
        assert_eq!(bundle, "");
        assert_eq!(model, "BAAI/bge-m3");
    }

    // ── build_model_loading_timeout_response ──────────────────────
    //
    // The queue proxy used to return 504 Gateway Timeout whenever the
    // JetStream result channel did not reply within
    // `request_timeout` seconds. The SDK has no retry branch for 504,
    // which meant a cold-start request that triggered a worker-side
    // model load (worker NAKs + redelivers after load) would bubble
    // up as a hard failure even though `wait_for_capacity=True` was
    // set. We now map that timeout to 503 + MODEL_LOADING, which the
    // SDK already retries under the same `provision_timeout_s`
    // budget it uses for worker-emitted MODEL_LOADING responses.

    #[test]
    fn test_build_model_loading_timeout_response_is_503_with_error_code() {
        let resp = build_model_loading_timeout_response("BAAI/bge-m3", 30);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_build_model_loading_timeout_response_sets_retry_after_and_version_headers() {
        let resp = build_model_loading_timeout_response("BAAI/bge-m3", 30);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            MODEL_LOADING_RETRY_AFTER
        );
        assert!(headers.get("x-sie-version").is_some());
        assert!(headers.get("x-sie-server-version").is_some());
    }

    #[test]
    fn test_build_model_loading_timeout_response_sets_error_code_header() {
        // The documented contract (architecture-guide.md, README.md,
        // docs/queue-based-routing.md) advertises `X-SIE-Error-Code:
        // MODEL_LOADING` alongside the body field. External clients and
        // observability middleware key off the header, so it must be
        // set even though the SDK itself reads the body.
        let resp = build_model_loading_timeout_response("BAAI/bge-m3", 30);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            MODEL_LOADING_ERROR_CODE
        );
    }

    #[tokio::test]
    async fn test_build_model_loading_timeout_response_body_matches_sdk_error_contract() {
        // SDK parses `error.code` for the retry decision; body must
        // include the MODEL_LOADING code and mention the model id.
        let resp = build_model_loading_timeout_response("BAAI/bge-m3", 30);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body is valid JSON");
        assert_eq!(body["error"]["code"], MODEL_LOADING_ERROR_CODE);
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("BAAI/bge-m3"),
            "message references the model id: {msg}"
        );
        assert!(
            msg.contains("30"),
            "message references the timeout value: {msg}"
        );
    }

    // ── retryable error code translation ───────────────────────────
    //
    // When every item in a batch fails with the *same* retryable code
    // (RESOURCE_EXHAUSTED from worker-side OOM recovery exhaustion, or
    // MODEL_LOADING from a worker still warming up), the gateway emits
    // a 503 with the SDK-expected body / headers so auto-retry kicks
    // in. Mixed batches keep going through the legacy 500
    // `all_items_failed` path so callers can inspect per-item codes.

    fn _err_result(code: Option<&str>, msg: &str) -> publisher::WorkResult {
        publisher::WorkResult {
            work_item_id: "req.0".to_string(),
            request_id: "req".to_string(),
            item_index: 0,
            success: false,
            result_msgpack: Vec::new(),
            error: Some(msg.to_string()),
            error_code: code.map(str::to_string),
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        }
    }

    #[test]
    fn test_unanimous_retryable_error_code_resource_exhausted() {
        let r1 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom 1");
        let r2 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom 2");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(RESOURCE_EXHAUSTED_ERROR_CODE)
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_model_loading() {
        let r1 = _err_result(Some("MODEL_LOADING"), "loading");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(MODEL_LOADING_ERROR_CODE)
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_lora_loading() {
        // LoRA-load-on-demand is also SDK-retryable; gateway must
        // translate it the same way as MODEL_LOADING.
        let r1 = _err_result(Some("LORA_LOADING"), "loading lora adapter");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(LORA_LOADING_ERROR_CODE)
        );
    }

    #[test]
    fn test_retryable_metric_reason_lora_loading() {
        assert_eq!(
            retryable_metric_reason(LORA_LOADING_ERROR_CODE),
            "upstream_lora_loading"
        );
    }

    #[test]
    fn test_build_retryable_error_response_lora_loading_status_and_headers() {
        let resp =
            build_retryable_error_response(LORA_LOADING_ERROR_CODE, "Loading lora adapter 'foo'");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            LORA_LOADING_RETRY_AFTER
        );
        assert_eq!(
            headers.get("x-sie-error-code").unwrap(),
            LORA_LOADING_ERROR_CODE
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_mixed_batch_returns_none() {
        // Mixed batches (one OOM, one inference_error) must NOT be
        // collapsed into a 503 — caller needs to see per-item details.
        let r1 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom");
        let r2 = _err_result(Some("inference_error"), "shape mismatch");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_unanimous_retryable_error_code_unknown_code_returns_none() {
        // An unrecognized code — even if unanimous — does NOT trigger
        // 503 retry; we only opt-in known retryable codes.
        let r1 = _err_result(Some("CUSTOM_BACKEND_ERROR"), "x");
        let r2 = _err_result(Some("CUSTOM_BACKEND_ERROR"), "y");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_unanimous_retryable_error_code_missing_code_returns_none() {
        let r1 = _err_result(None, "no code");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_build_retryable_error_response_resource_exhausted_status_and_headers() {
        let resp = build_retryable_error_response(
            RESOURCE_EXHAUSTED_ERROR_CODE,
            "CUDA out of memory after recovery",
        );
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            RESOURCE_EXHAUSTED_RETRY_AFTER
        );
        assert_eq!(
            headers.get("x-sie-error-code").unwrap(),
            RESOURCE_EXHAUSTED_ERROR_CODE
        );
        assert!(headers.get("x-sie-version").is_some());
        assert!(headers.get("x-sie-server-version").is_some());
    }

    #[tokio::test]
    async fn test_build_retryable_error_response_resource_exhausted_body() {
        // SDK reads `error.code` to decide whether to retry; body must
        // carry the structured envelope.
        let resp = build_retryable_error_response(
            RESOURCE_EXHAUSTED_ERROR_CODE,
            "CUDA out of memory after recovery",
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body is valid JSON");
        assert_eq!(body["error"]["code"], RESOURCE_EXHAUSTED_ERROR_CODE);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("out of memory"),
            "message preserves upstream error text: {body}"
        );
    }

    #[test]
    fn test_retryable_metric_reason_known_codes() {
        assert_eq!(
            retryable_metric_reason(RESOURCE_EXHAUSTED_ERROR_CODE),
            "resource_exhausted"
        );
        assert_eq!(
            retryable_metric_reason(MODEL_LOADING_ERROR_CODE),
            "upstream_model_loading"
        );
        // Unknown code falls back to the legacy bucket.
        assert_eq!(retryable_metric_reason("anything-else"), "all_items_failed");
    }

    #[test]
    fn test_parse_model_spec_with_bundle() {
        let (bundle, model) = parse_model_spec("premium:/BAAI/bge-m3");
        assert_eq!(bundle, "premium");
        assert_eq!(model, "BAAI/bge-m3");
    }

    #[test]
    fn test_parse_model_spec_plain_name() {
        let (bundle, model) = parse_model_spec("my-model");
        assert_eq!(bundle, "");
        assert_eq!(model, "my-model");
    }

    #[test]
    fn test_parse_model_spec_empty() {
        let (bundle, model) = parse_model_spec("");
        assert_eq!(bundle, "");
        assert_eq!(model, "");
    }

    // ── check_sdk_version cache ────────────────────────────────────

    #[test]
    fn test_check_sdk_version_caches_parsed_minor() {
        // Use a deliberately unique version string so concurrent
        // tests don't fight over the cache entry.
        let version = "99.42.7-test-parsed";
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sie-sdk-version",
            HeaderValue::from_static("99.42.7-test-parsed"),
        );

        assert!(SDK_VERSION_CACHE.get(version).is_none());
        check_sdk_version(&headers);

        let cached = SDK_VERSION_CACHE
            .get(version)
            .map(|v| *v)
            .expect("cache entry should be populated after first hit");
        assert_eq!(cached, Some(42));
    }

    #[test]
    fn test_check_sdk_version_caches_unparseable_as_none() {
        let version = "garbage-no-dots-test";
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sie-sdk-version",
            HeaderValue::from_static("garbage-no-dots-test"),
        );

        check_sdk_version(&headers);
        // Second call hits the `Some(None)` fast path; no crash,
        // no re-parse, no re-insertion.
        check_sdk_version(&headers);

        let cached = SDK_VERSION_CACHE
            .get(version)
            .map(|v| *v)
            .expect("cache entry should be populated even for garbage");
        assert_eq!(cached, None);
    }

    // ── resolve_machine_profile ────────────────────────────────────

    fn make_gpu_map(gpus: &[&str]) -> std::collections::HashMap<String, String> {
        gpus.iter()
            .map(|g| (g.to_lowercase(), g.to_string()))
            .collect()
    }

    #[test]
    fn test_resolve_machine_profile_exact_match() {
        let m = make_gpu_map(&["l4-spot", "a100-40gb"]);
        assert_eq!(resolve_machine_profile("l4-spot", &m), "l4-spot");
    }

    #[test]
    fn test_resolve_machine_profile_case_insensitive() {
        let m = make_gpu_map(&["L4-Spot"]);
        assert_eq!(resolve_machine_profile("l4-spot", &m), "L4-Spot");
    }

    #[test]
    fn test_resolve_machine_profile_spot_fallback() {
        let m = make_gpu_map(&["l4-spot"]);
        assert_eq!(resolve_machine_profile("l4", &m), "l4-spot");
    }

    #[test]
    fn test_resolve_machine_profile_no_match() {
        let m = make_gpu_map(&["l4-spot"]);
        assert_eq!(resolve_machine_profile("h100", &m), "h100");
    }

    #[test]
    fn test_parse_queue_request_reads_nested_params() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}],
            "params": {
                "output_types": ["dense"],
                "instruction": "search",
                "options": {
                    "truncate": true,
                    "is_query": true
                }
            }
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "encode").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(params.instruction, Some("search".to_string()));
        assert!(params.is_query);
        assert_eq!(params.options.unwrap()["truncate"], true);
    }

    #[test]
    fn test_parse_queue_request_score_keeps_query_and_items() {
        let body = serde_json::to_vec(&json!({
            "query": {"text": "hello"},
            "items": [{"text": "a"}, {"text": "b"}],
            "instruction": "rank"
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            params.query_item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )]))
        );
        assert_eq!(params.instruction, Some("rank".to_string()));
    }

    #[test]
    fn test_parse_queue_request_score_defaults_missing_query_to_empty_object() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}]
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.query_item, Some(rmpv::Value::Map(Vec::new())));
    }

    #[test]
    fn test_parse_queue_request_score_keeps_non_object_query() {
        let body = serde_json::to_vec(&json!({
            "query": "hello",
            "items": [{"text": "a"}]
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.query_item, Some(rmpv::Value::from("hello")));
    }

    /// Msgpack-in encode request: the whole body is a msgpack map,
    /// `items` is an array of maps, and `work_params_from_rmpv`
    /// should read `output_types` / `instruction` / `options` the
    /// same way the JSON path does.
    #[test]
    fn test_parse_queue_request_msgpack_encode_reads_params() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                    rmpv::Value::from("text"),
                    rmpv::Value::from("hello"),
                )])]),
            ),
            (
                rmpv::Value::from("output_types"),
                rmpv::Value::Array(vec![rmpv::Value::from("dense")]),
            ),
            (
                rmpv::Value::from("instruction"),
                rmpv::Value::from("search"),
            ),
            (
                rmpv::Value::from("options"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("is_query"),
                    rmpv::Value::Boolean(true),
                )]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, params) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(items.len(), 1);
        // The per-item value must stay an `rmpv::Value::Map` —
        // the whole point of the passthrough path is that msgpack-in
        // items never round-trip through `serde_json::Value`.
        assert!(matches!(&items[0], rmpv::Value::Map(_)));
        assert_eq!(params.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(params.instruction, Some("search".to_string()));
        assert!(params.is_query);
    }

    /// Regression guard for the rmpv-passthrough correctness fix:
    /// a msgpack-in request carrying a `bin` blob (e.g. a raw
    /// numpy buffer the SDK packs with `msgpack_numpy`) must reach
    /// the publisher as `rmpv::Value::Binary` byte-for-byte. The
    /// old serde_json-intermediate path expanded every byte into a
    /// `Value::Number`, so the wire bytes workers received were
    /// corrupted for any binary-heavy request.
    #[test]
    fn test_parse_queue_request_msgpack_preserves_binary_in_item() {
        let payload: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02];
        let body_value = rmpv::Value::Map(vec![(
            rmpv::Value::from("items"),
            rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                rmpv::Value::from("blob"),
                rmpv::Value::Binary(payload.clone()),
            )])]),
        )]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, _) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(items.len(), 1);
        let rmpv::Value::Map(entries) = &items[0] else {
            panic!("expected Map, got {:?}", items[0]);
        };
        let blob = entries
            .iter()
            .find(|(k, _)| matches!(k, rmpv::Value::String(s) if s.as_str() == Some("blob")))
            .map(|(_, v)| v)
            .expect("blob field missing");
        assert_eq!(blob, &rmpv::Value::Binary(payload));
    }

    /// Top-level msgpack bodies must be maps — parity with the
    /// JSON path which rejects non-object bodies with 400. A
    /// top-level array / scalar is almost always a mis-encoded
    /// client request; silently accepting it as a single item
    /// used to let the request fail later in worker-specific
    /// ways instead of at ingress (see review feedback on
    /// PR #716).
    #[test]
    fn test_parse_queue_request_msgpack_rejects_non_map_top_level() {
        let array_body = rmp_serde::to_vec(&rmpv::Value::Array(vec![
            rmpv::Value::from(1),
            rmpv::Value::from(2),
        ]))
        .unwrap();
        let err = parse_queue_request(&array_body, true, "encode").unwrap_err();
        assert!(err.contains("msgpack map"), "unexpected error: {err}");

        let scalar_body = rmp_serde::to_vec(&rmpv::Value::from(42)).unwrap();
        let err = parse_queue_request(&scalar_body, true, "encode").unwrap_err();
        assert!(err.contains("msgpack map"), "unexpected error: {err}");
    }

    /// Msgpack-in score request: `query` + `items` live at the
    /// top level, and `params.query_item` must carry the rmpv
    /// representation (not a round-tripped serde_json blob).
    #[test]
    fn test_parse_queue_request_msgpack_score_keeps_query_as_rmpv() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("query"),
                rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("q"))]),
            ),
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::from("a"), rmpv::Value::from("b")]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, params) = parse_queue_request(&body, true, "score").unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            params.query_item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("q"),
            )]))
        );
    }

    #[test]
    fn test_parse_queue_request_score_ignores_nested_params() {
        let body = serde_json::to_vec(&json!({
            "items": [],
            "instruction": "top-level",
            "options": {"truncate": false},
            "params": {
                "instruction": "nested",
                "options": {"truncate": true}
            }
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert!(items.is_empty());
        assert_eq!(params.query_item, Some(rmpv::Value::Map(Vec::new())));
        assert_eq!(params.instruction, Some("top-level".to_string()));
        assert_eq!(params.options, Some(json!({"truncate": false})));
    }

    // ── msgpack_numpy conversion tests ──────────────────────────
    //
    // These exercise the fused `rmpv_to_response_json` path. All
    // fixtures use the exact rmpv shape that Python workers emit
    // via `msgpack_numpy` — a `Map` whose `data` key holds a
    // `Binary` blob — so the tests double as a wire-format guard:
    // if Python-side encoding ever changes, these flip first.

    /// Helper: build a `{"nd": true, "type": ..., "shape": [...],
    /// "data": <binary>}` rmpv sentinel for dtype/shape tests.
    fn numpy_sentinel(dtype: &str, shape: &[usize], bytes: Vec<u8>) -> rmpv::Value {
        rmpv::Value::Map(vec![
            (rmpv::Value::from("nd"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("type"), rmpv::Value::from(dtype)),
            (
                rmpv::Value::from("shape"),
                rmpv::Value::Array(
                    shape
                        .iter()
                        .map(|&n| rmpv::Value::Integer((n as u64).into()))
                        .collect(),
                ),
            ),
            (rmpv::Value::from("data"), rmpv::Value::Binary(bytes)),
        ])
    }

    #[test]
    fn test_rmpv_to_response_json_f32_array() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = numpy_sentinel("<f4", &[3], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmpv_to_response_json_f64_array() {
        let bytes: Vec<u8> = [1.5f64, 2.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let value = numpy_sentinel("<f8", &[2], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 2);
        assert!((arr[0].as_f64().unwrap() - 1.5).abs() < 1e-10);
        assert!((arr[1].as_f64().unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_rmpv_to_response_json_i32_array() {
        let bytes: Vec<u8> = [42i32, -7].iter().flat_map(|i| i.to_le_bytes()).collect();
        let value = numpy_sentinel("<i4", &[2], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_i64().unwrap(), 42);
        assert_eq!(arr[1].as_i64().unwrap(), -7);
    }

    #[test]
    fn test_rmpv_to_response_json_bool_array() {
        let value = numpy_sentinel("|b1", &[3], vec![1, 0, 1]);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 3);
        assert!(arr[0].as_bool().unwrap());
        assert!(!arr[1].as_bool().unwrap());
        assert!(arr[2].as_bool().unwrap());
    }

    #[test]
    fn test_rmpv_to_response_json_u8_array() {
        let value = numpy_sentinel("|u1", &[3], vec![0, 128, 255]);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr[0].as_u64().unwrap(), 0);
        assert_eq!(arr[1].as_u64().unwrap(), 128);
        assert_eq!(arr[2].as_u64().unwrap(), 255);
    }

    #[test]
    fn test_rmpv_to_response_json_2d_shape() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = numpy_sentinel("<f4", &[2, 3], bytes);
        let outer = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(outer.len(), 2);
        let row0 = outer[0].as_array().unwrap();
        assert_eq!(row0.len(), 3);
        assert!((row0[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((row0[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        let row1 = outer[1].as_array().unwrap();
        assert!((row1[0].as_f64().unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmpv_to_response_json_empty_data() {
        let value = numpy_sentinel("<f4", &[0], Vec::new());
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 0);
    }

    /// Non-sentinel maps walk through untouched and keep their
    /// original shape — we must not misdecode user maps that happen
    /// to carry a `"data"` key.
    #[test]
    fn test_rmpv_to_response_json_no_sentinel_unchanged() {
        let value = rmpv::Value::Map(vec![
            (rmpv::Value::from("key"), rmpv::Value::from("value")),
            (
                rmpv::Value::from("nested"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("a"),
                    rmpv::Value::Integer(1.into()),
                )]),
            ),
        ]);
        let json_val = rmpv_to_response_json(value);
        assert_eq!(json_val["key"], "value");
        assert_eq!(json_val["nested"]["a"], 1);
    }

    /// Guards that a sentinel-shaped map whose `nd` is `false` is
    /// left alone: the `nd == true` marker is load-bearing, plain
    /// user maps that happen to share the key set must pass through
    /// unchanged.
    #[test]
    fn test_rmpv_to_response_json_ignores_nd_false() {
        let value = rmpv::Value::Map(vec![
            (rmpv::Value::from("nd"), rmpv::Value::Boolean(false)),
            (rmpv::Value::from("type"), rmpv::Value::from("<f4")),
            (
                rmpv::Value::from("shape"),
                rmpv::Value::Array(vec![rmpv::Value::Integer(1.into())]),
            ),
            (rmpv::Value::from("data"), rmpv::Value::Binary(vec![0; 4])),
        ]);
        let json_val = rmpv_to_response_json(value);
        assert_eq!(json_val["nd"], false);
        assert_eq!(json_val["type"], "<f4");
    }

    #[test]
    fn test_f16_to_f32_basic() {
        let result = f16_to_f32(0x3C00); // f16 1.0
        assert!((result - 1.0).abs() < 1e-6);
        let result = f16_to_f32(0x0000); // f16 zero
        assert_eq!(result, 0.0);
        let result = f16_to_f32(0x8000); // f16 negative zero
        assert!(result.is_sign_negative());
        assert_eq!(result, -0.0f32);
    }

    #[test]
    fn test_queue_result_single_item_msgpack_passthrough() {
        let payload = rmp_serde::to_vec(&json!({"embedding": [1.0, 2.0]})).unwrap();
        let result = publisher::WorkResult {
            work_item_id: "r1.0".to_string(),
            request_id: "r1".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: payload.clone(),
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let resp_body = result.result_msgpack.clone();
        assert_eq!(resp_body, payload);
    }

    #[test]
    fn test_queue_result_single_item_json_decode() {
        let payload = rmp_serde::to_vec(&json!({"embedding": [1.0, 2.0]})).unwrap();
        let json_val: serde_json::Value = rmp_serde::from_slice(&payload).unwrap();
        let resp_body = serde_json::to_vec(&json_val).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(parsed["embedding"][0], 1.0);
        assert_eq!(parsed["embedding"][1], 2.0);
    }

    #[test]
    fn test_queue_result_multi_item_msgpack_decode() {
        let payload1 = rmp_serde::to_vec(&json!({"result": "a"})).unwrap();
        let payload2 = rmp_serde::to_vec(&json!({"result": "b"})).unwrap();
        let results = [
            publisher::WorkResult {
                work_item_id: "r1.0".to_string(),
                request_id: "r1".to_string(),
                item_index: 0,
                success: true,
                result_msgpack: payload1,
                error: None,
                error_code: None,
                inference_ms: None,
                queue_ms: None,
                processing_ms: None,
                worker_id: None,
                tokenization_ms: None,
                postprocessing_ms: None,
                payload_fetch_ms: None,
            },
            publisher::WorkResult {
                work_item_id: "r1.1".to_string(),
                request_id: "r1".to_string(),
                item_index: 1,
                success: true,
                result_msgpack: payload2,
                error: None,
                error_code: None,
                inference_ms: None,
                queue_ms: None,
                processing_ms: None,
                worker_id: None,
                tokenization_ms: None,
                postprocessing_ms: None,
                payload_fetch_ms: None,
            },
        ];
        let items: Vec<serde_json::Value> = results
            .iter()
            .filter_map(|r| rmp_serde::from_slice(&r.result_msgpack).ok())
            .collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["result"], "a");
        assert_eq!(items[1]["result"], "b");
        let combined = json!({"items": items});
        let body = serde_json::to_vec(&combined).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["items"].as_array().unwrap().len(), 2);
    }

    /// End-to-end guard: build a msgpack payload the way a real
    /// Python worker would (sentinel map with `data` as `Binary`),
    /// decode via `rmp_serde::from_slice` → `rmpv_to_response_json`,
    /// and check that bin data is decoded inline without byte-array
    /// inflation. Non-numpy fields (`text`) must survive untouched.
    #[test]
    fn test_rmpv_to_response_json_from_real_msgpack_bytes() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let payload = MsgValue::Map(vec![
            (
                MsgValue::String("embedding".into()),
                numpy_sentinel("<f4", &[3], f32_bytes),
            ),
            (
                MsgValue::String("text".into()),
                MsgValue::String("hello".into()),
            ),
        ]);

        let msgpack_bytes = rmp_serde::to_vec(&payload).unwrap();
        let decoded: rmpv::Value = rmp_serde::from_slice(&msgpack_bytes).unwrap();
        let json_val = rmpv_to_response_json(decoded);

        let arr = json_val["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(json_val["text"], "hello");
    }

    /// The hot path fuses msgpack_numpy decode into the rmpv walk.
    /// This test builds the exact shape Python workers produce — a
    /// map whose `data` field is an `rmpv::Value::Binary` blob —
    /// and confirms the fused function decodes the dtype directly
    /// from the byte slice, without ever materialising a
    /// byte-per-`Number` intermediate.
    #[test]
    fn test_rmpv_to_response_json_decodes_numpy_binary_directly() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let sentinel = MsgValue::Map(vec![
            (MsgValue::String("nd".into()), MsgValue::Boolean(true)),
            (
                MsgValue::String("type".into()),
                MsgValue::String("<f4".into()),
            ),
            (
                MsgValue::String("shape".into()),
                MsgValue::Array(vec![MsgValue::Integer(3.into())]),
            ),
            (MsgValue::String("data".into()), MsgValue::Binary(f32_bytes)),
        ]);
        let payload = MsgValue::Map(vec![
            (MsgValue::String("embedding".into()), sentinel),
            (
                MsgValue::String("text".into()),
                MsgValue::String("hello".into()),
            ),
        ]);

        let json_val = rmpv_to_response_json(payload);

        let arr = json_val["embedding"]
            .as_array()
            .expect("embedding should be a flat array");
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(json_val["text"], "hello");
    }

    /// Some msgpack_numpy variants pack the dtype buffer as an ext
    /// type instead of plain binary; the fused decode path treats
    /// both identically.
    #[test]
    fn test_rmpv_to_response_json_decodes_numpy_ext_data() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [0.25f32, 0.5f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let sentinel = MsgValue::Map(vec![
            (MsgValue::String("nd".into()), MsgValue::Boolean(true)),
            (
                MsgValue::String("type".into()),
                MsgValue::String("<f4".into()),
            ),
            (
                MsgValue::String("shape".into()),
                MsgValue::Array(vec![MsgValue::Integer(2.into())]),
            ),
            (
                MsgValue::String("data".into()),
                MsgValue::Ext(0x15, f32_bytes),
            ),
        ]);

        let arr = rmpv_to_response_json(sentinel)
            .as_array()
            .expect("ext data should decode to array")
            .clone();
        assert_eq!(arr.len(), 2);
        assert!((arr[0].as_f64().unwrap() - 0.25).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    /// Non-sentinel maps must pass through unchanged — no accidental
    /// decode of user-defined maps that happen to include `"data"`.
    #[test]
    fn test_rmpv_to_response_json_passes_through_non_sentinel_maps() {
        use rmpv::Value as MsgValue;

        let map = MsgValue::Map(vec![
            (
                MsgValue::String("type".into()),
                MsgValue::String("user-data".into()),
            ),
            (
                MsgValue::String("data".into()),
                MsgValue::String("some value".into()),
            ),
        ]);

        let json_val = rmpv_to_response_json(map);
        assert_eq!(json_val["type"], "user-data");
        assert_eq!(json_val["data"], "some value");
    }

    /// Python msgpack sometimes emits binary keys (e.g. `b"nd"`,
    /// `b"type"`) when the encoder is not in `strict_map_key=True`
    /// mode. `rmpv_to_response_json` must still decode both the
    /// sentinel (at the embedding level) and the surrounding map
    /// keys back into string-keyed JSON.
    #[test]
    fn test_rmpv_to_response_json_handles_binary_map_keys() {
        use rmpv::Value as MsgValue;

        let map = MsgValue::Map(vec![(
            MsgValue::Binary(b"key".to_vec()),
            MsgValue::String("value".into()),
        )]);
        let json_val = rmpv_to_response_json(map);
        assert_eq!(json_val["key"], "value");
    }

    // ── resolve_effective_pool (scale-from-zero decision) ──────────
    //
    // These tests guard the contract that `proxy_request` emits
    // `202 provisioning` whenever no healthy worker is registered for
    // `(bundle, gpu)` — regardless of whether the caller sent an
    // `X-SIE-MACHINE-PROFILE` header. An earlier regression gated the
    // 202 branch on `!gpu.is_empty()`, so default-routing cold starts
    // fell through to a `"default"` pool publish and hung.

    use crate::types::worker::{GpuStatus, ModelStatus, WorkerStatusMessage};
    use std::time::Duration as StdDuration;

    fn pool_registry() -> WorkerRegistry {
        WorkerRegistry::new(StdDuration::from_secs(30), None)
    }

    fn worker_msg(bundle: &str, gpu: &str, pool: &str) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: "worker-1".into(),
            ready: true,
            gpu_count: 1,
            machine_profile: gpu.into(),
            pool_name: pool.into(),
            bundle: bundle.into(),
            bundle_config_hash: "abc".into(),
            loaded_models: vec![],
            models: vec![ModelStatus { queue_depth: 0 }],
            gpus: vec![GpuStatus {
                memory_used_bytes: 0,
                memory_total_bytes: 4000,
            }],
            queue_depth: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
        }
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_provisioning_when_empty_and_no_gpu() {
        // No workers at all. Caller sends no `X-SIE-MACHINE-PROFILE`.
        // Before the fix this returned `Pool("default")` and the gateway
        // published to a nonexistent consumer. After the fix we emit
        // `Provisioning` so the caller returns `202 + Retry-After` and
        // records pending demand for KEDA.
        let reg = pool_registry();
        let out = resolve_effective_pool(&reg, "default", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        // No GPU expressed → exact_gpu_match is definitionally false.
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_provisioning_when_no_gpu_match() {
        // Worker exists but for a different GPU and a different bundle.
        // Nothing matches → provision.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "premium", "l4", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_bundle_only_fallback_when_gpu_mismatch() {
        // Client pinned `l4` but the cluster only has an `a100` worker on
        // the same bundle. We still route (the profile distinction is
        // cosmetic here) rather than forcing a scale-up.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "default", "l4", "").await;
        assert_eq!(out.resolution, PoolResolution::Pool("pool-a".to_string()));
        // Exact tuple (default, l4) had no worker — even though we
        // still routed via the bundle-only fallback, the caller must
        // record demand so KEDA scales up the l4 pool.
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_pool_when_worker_matches_no_gpu() {
        // Common default-routing flow: no GPU header, one worker on the
        // requested bundle → route to its pool directly (no 202).
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "l4-spot", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "default", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Pool("pool-a".to_string()));
        // No GPU preference → no demand tracking applicable.
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_reports_exact_gpu_match_when_tuple_exists() {
        // Exact (bundle, gpu) worker exists → `exact_gpu_match` is
        // `true` and the caller skips the demand-tracking write.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "l4-spot", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "default", "l4-spot", "").await;
        assert_eq!(out.resolution, PoolResolution::Pool("pool-a".to_string()));
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_honours_explicit_pool_name() {
        // Caller pinned a pool. Trust them unconditionally — even an
        // empty registry must route there so a known cold pool can be
        // targeted (worker will scale up independently).
        let reg = pool_registry();
        let out = resolve_effective_pool(&reg, "default", "", "my-bench").await;
        assert_eq!(out.resolution, PoolResolution::Pool("my-bench".to_string()));
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_honours_explicit_pool_even_when_worker_exists() {
        // Explicit pool overrides the registry lookup entirely, but we
        // still probe the registry so `exact_gpu_match` reflects the
        // actual state — the caller uses that to avoid recording
        // spurious demand when a worker is in fact available.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "l4-spot", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "default", "l4-spot", "my-bench").await;
        assert_eq!(out.resolution, PoolResolution::Pool("my-bench".to_string()));
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_pinned_pool_with_missing_gpu_tuple_reports_no_match() {
        // Caller pinned a pool AND expressed a GPU preference, but no
        // worker matches the exact tuple. We still route to the pin,
        // and `exact_gpu_match=false` tells the caller to record
        // demand for KEDA.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, "default", "l4", "my-bench").await;
        assert_eq!(out.resolution, PoolResolution::Pool("my-bench".to_string()));
        assert!(!out.exact_gpu_match);
    }
}
