use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use subtle::ConstantTimeEq;
use tower::{Layer, Service};

use crate::config::Config;

/// Paths that are always exempt from auth. Kubernetes liveness/readiness
/// probes carry no credentials; gating them would take the pod out of
/// rotation during an auth misconfiguration. `/health` (rich status JSON
/// with worker URLs, bundle assignments, queue depth, GPU inventory) is
/// intentionally NOT in this list — see `EXEMPT_OPERATIONAL_PATHS`.
const EXEMPT_PROBE_PATHS: &[&str] = &["/healthz", "/readyz"];

/// Paths that expose operational data (status page, rich `/health`,
/// `/metrics`, `/ws/*`). Exempt from auth only when
/// `SIE_AUTH_EXEMPT_OPERATIONAL=true`; default is fail-closed.
const EXEMPT_OPERATIONAL_PATHS: &[&str] = &["/", "/health", "/metrics"];

#[derive(Clone)]
pub struct AuthLayer {
    config: Arc<Config>,
}

impl AuthLayer {
    pub fn new(config: Arc<Config>) -> Self {
        Self { config }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthMiddleware {
            inner,
            config: Arc::clone(&self.config),
        }
    }
}

#[derive(Clone)]
pub struct AuthMiddleware<S> {
    inner: S,
    config: Arc<Config>,
}

impl<S> Service<Request<Body>> for AuthMiddleware<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let config = Arc::clone(&self.config);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            if !is_auth_enabled(&config.auth_mode) {
                return inner.call(req).await;
            }

            let path = req.uri().path();

            if EXEMPT_PROBE_PATHS.contains(&path) {
                return inner.call(req).await;
            }

            if config.auth_exempt_operational
                && (EXEMPT_OPERATIONAL_PATHS.contains(&path) || path.starts_with("/ws/"))
            {
                return inner.call(req).await;
            }

            if config.auth_tokens.is_empty() {
                return Ok(error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Gateway auth enabled but no tokens configured",
                ));
            }

            let token = match extract_bearer_token(req.headers()) {
                Some(t) => t,
                None => {
                    return Ok(error_response(
                        StatusCode::UNAUTHORIZED,
                        "Missing Authorization header",
                    ));
                }
            };

            let is_admin = is_admin_endpoint(req.method(), path);

            if is_admin {
                if config.admin_token.is_empty() {
                    return Ok(error_response(
                        StatusCode::FORBIDDEN,
                        "Admin token not configured",
                    ));
                }

                if !constant_time_eq_str(&token, &config.admin_token) {
                    return Ok(error_response(
                        StatusCode::FORBIDDEN,
                        "Admin token required",
                    ));
                }

                return inner.call(req).await;
            }

            let valid = config
                .auth_tokens
                .iter()
                .any(|valid_token| constant_time_eq_str(&token, valid_token));

            if !valid {
                return Ok(error_response(StatusCode::UNAUTHORIZED, "Invalid token"));
            }

            inner.call(req).await
        })
    }
}

/// Returns true when `auth_mode` should make the middleware enforce auth.
/// Accepts both `"static"` (the code's legacy name) and `"token"` (the
/// name used in the gateway README and Helm values) as aliases.
pub fn is_auth_enabled(mode: &str) -> bool {
    matches!(mode, "static" | "token")
}

/// Returns true for mutations that require the admin token. Reads on
/// these paths (including `GET`) are checked with the regular token.
///
/// Admin-gated prefixes:
/// - `/v1/config*` — reserved; config writes are currently 405.
/// - `/v1/admin*` — admin surface.
/// - `/v1/pools*` — pool create/delete/renew mutate cluster-wide
///   routing and isolation state.
fn is_admin_endpoint(method: &Method, path: &str) -> bool {
    if !matches!(method, &Method::POST | &Method::PUT | &Method::DELETE) {
        return false;
    }
    path.starts_with("/v1/config") || path.starts_with("/v1/admin") || path.starts_with("/v1/pools")
}

fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
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

fn constant_time_eq_str(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let body = serde_json::json!({"message": message});
    (status, axum::Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bearer_token() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "Bearer test-token-123".parse().unwrap());
        assert_eq!(
            extract_bearer_token(&headers),
            Some("test-token-123".to_string())
        );
    }

    #[test]
    fn test_extract_bearer_token_missing() {
        let headers = axum::http::HeaderMap::new();
        assert_eq!(extract_bearer_token(&headers), None);
    }

    #[test]
    fn test_extract_bearer_token_no_prefix() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "raw-token".parse().unwrap());
        assert_eq!(
            extract_bearer_token(&headers),
            Some("raw-token".to_string())
        );
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq_str("abc", "abc"));
        assert!(!constant_time_eq_str("abc", "def"));
        assert!(!constant_time_eq_str("abc", "abcd"));
    }

    #[test]
    fn test_is_admin_endpoint() {
        assert!(is_admin_endpoint(&Method::POST, "/v1/configs/models"));
        assert!(is_admin_endpoint(&Method::PUT, "/v1/configs/models"));
        assert!(is_admin_endpoint(&Method::DELETE, "/v1/admin/reload"));
        // Pool create/delete/renew require the admin token.
        assert!(is_admin_endpoint(&Method::POST, "/v1/pools"));
        assert!(is_admin_endpoint(&Method::DELETE, "/v1/pools/gpu-l4"));
        assert!(is_admin_endpoint(&Method::POST, "/v1/pools/gpu-l4/renew"));
        // GET to config / pool paths is NOT admin (read-only, uses regular auth)
        assert!(!is_admin_endpoint(&Method::GET, "/v1/configs/models"));
        assert!(!is_admin_endpoint(&Method::GET, "/v1/configs/bundles"));
        assert!(!is_admin_endpoint(&Method::GET, "/v1/pools"));
        assert!(!is_admin_endpoint(&Method::GET, "/v1/pools/gpu-l4"));
        // Inference paths are never admin.
        assert!(!is_admin_endpoint(&Method::POST, "/v1/encode/model"));
    }

    #[test]
    fn test_is_auth_enabled_accepts_static_and_token() {
        assert!(is_auth_enabled("static"));
        assert!(is_auth_enabled("token"));
        assert!(!is_auth_enabled("none"));
        assert!(!is_auth_enabled(""));
        assert!(!is_auth_enabled("disabled"));
        // Typos and unknown modes fail-closed-to-bypass by design: the
        // pair (auth_mode, auth_tokens) is validated at startup in
        // `Config::load`, which logs a warning when tokens are configured
        // but the mode is disabled. This prevents a typo in auth_mode
        // from becoming a silent auth enforcement while still allowing
        // the legacy "turn off auth" workflow.
        assert!(!is_auth_enabled("staitc"));
    }

    #[test]
    fn test_exempt_paths_partitioned() {
        // Probe paths are always exempt; operational paths are not.
        assert!(EXEMPT_PROBE_PATHS.contains(&"/healthz"));
        assert!(EXEMPT_PROBE_PATHS.contains(&"/readyz"));
        assert!(!EXEMPT_PROBE_PATHS.contains(&"/health"));
        assert!(!EXEMPT_PROBE_PATHS.contains(&"/metrics"));
        assert!(EXEMPT_OPERATIONAL_PATHS.contains(&"/"));
        assert!(EXEMPT_OPERATIONAL_PATHS.contains(&"/metrics"));
        assert!(EXEMPT_OPERATIONAL_PATHS.contains(&"/health"));
    }

    // ── End-to-end middleware tests ────────────────────────────────
    //
    // These exercise the full request pipeline (AuthLayer wraps a
    // dummy handler) and cover what the helper-only tests cannot:
    //   - which paths actually reach the handler vs return 401/403
    //   - admin vs user token routing on admin-gated prefixes
    //   - fail-open behavior when auth is disabled / unknown
    //   - the `SIE_AUTH_EXEMPT_OPERATIONAL` toggle

    use axum::routing::{get, post};
    use axum::Router;
    use http::Request as HttpRequest;
    use tower::util::ServiceExt;

    use std::collections::HashMap;

    fn cfg_for_middleware(
        mode: &str,
        tokens: Vec<&str>,
        admin: &str,
        exempt_operational: bool,
    ) -> Arc<Config> {
        Arc::new(Config {
            host: String::new(),
            port: 0,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: String::new(),
            k8s_service: String::new(),
            k8s_port: 0,
            health_mode: String::new(),
            nats_url: String::new(),
            nats_config_trusted_producers: Vec::new(),
            auth_mode: mode.to_string(),
            auth_tokens: tokens.into_iter().map(String::from).collect(),
            admin_token: admin.to_string(),
            auth_exempt_operational: exempt_operational,
            log_level: String::new(),
            json_logs: false,
            enable_pools: false,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 0.0,
            max_stream_pending: 0,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            bundles_dir: String::new(),
            models_dir: String::new(),
            config_service_url: None,
            config_service_token: None,
            payload_store_url: String::new(),
        })
    }

    /// Minimal router with the endpoint shapes the middleware cares
    /// about: probes, operational surfaces, a regular inference path,
    /// and admin-gated prefixes.
    fn test_router(config: Arc<Config>) -> Router {
        Router::new()
            .route("/healthz", get(|| async { "ok" }))
            .route("/readyz", get(|| async { "ready" }))
            .route("/", get(|| async { "index" }))
            .route("/health", get(|| async { "rich-health" }))
            .route("/metrics", get(|| async { "metrics" }))
            .route("/ws/cluster-status", get(|| async { "ws" }))
            .route("/v1/encode/{*model}", post(|| async { "encoded" }))
            .route("/v1/pools", post(|| async { "created" }))
            .route("/v1/pools", get(|| async { "list" }))
            .route("/v1/pools/{pool}", get(|| async { "pool" }))
            .route("/v1/admin/reload", post(|| async { "reloaded" }))
            .layer(AuthLayer::new(config))
    }

    async fn send(router: Router, method: Method, path: &str, bearer: Option<&str>) -> StatusCode {
        let mut req = HttpRequest::builder().method(method).uri(path);
        if let Some(tok) = bearer {
            req = req.header("authorization", format!("Bearer {tok}"));
        }
        let response = router
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap();
        response.status()
    }

    #[tokio::test]
    async fn middleware_disabled_auth_passes_everything_through() {
        let cfg = cfg_for_middleware("none", vec![], "", false);
        let r = test_router(cfg);
        assert_eq!(send(r, Method::GET, "/healthz", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_probes_always_exempt() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(send(r, Method::GET, "/healthz", None).await, StatusCode::OK);
        let r = test_router(cfg);
        assert_eq!(send(r, Method::GET, "/readyz", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_operational_paths_fail_closed_by_default() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send(r, Method::GET, "/health", None).await,
            StatusCode::UNAUTHORIZED,
            "/health must require auth when exempt flag is off"
        );
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send(r, Method::GET, "/metrics", None).await,
            StatusCode::UNAUTHORIZED
        );
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send(r, Method::GET, "/", None).await,
            StatusCode::UNAUTHORIZED
        );
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::GET, "/ws/cluster-status", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn middleware_operational_paths_exempt_when_flag_set() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", true);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(send(r, Method::GET, "/health", None).await, StatusCode::OK);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(send(r, Method::GET, "/metrics", None).await, StatusCode::OK);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(send(r, Method::GET, "/", None).await, StatusCode::OK);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::GET, "/ws/cluster-status", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_accepts_static_alias_for_token_mode() {
        let cfg = cfg_for_middleware("static", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", Some("user")).await,
            StatusCode::OK,
            "`static` must be recognized identically to `token`"
        );
    }

    #[tokio::test]
    async fn middleware_missing_token_is_401() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn middleware_wrong_token_is_401() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", Some("bogus")).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn middleware_pool_mutation_rejects_user_token() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/pools", Some("user")).await,
            StatusCode::FORBIDDEN,
            "pool mutations must require the admin token, not any inference token"
        );
    }

    #[tokio::test]
    async fn middleware_pool_mutation_accepts_admin_token() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/pools", Some("admin")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_pool_read_allows_user_token() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send(r, Method::GET, "/v1/pools", Some("user")).await,
            StatusCode::OK,
            "pool reads are not admin-gated"
        );
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::GET, "/v1/pools/gpu-l4", Some("user")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_admin_mutation_rejects_user_token() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/admin/reload", Some("user")).await,
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn middleware_admin_mutation_refuses_when_admin_token_unset() {
        let cfg = cfg_for_middleware("token", vec!["user"], "", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/admin/reload", Some("user")).await,
            StatusCode::FORBIDDEN,
            "missing admin token must fail-closed, never accept a user token"
        );
    }

    #[tokio::test]
    async fn middleware_enabled_without_tokens_is_500() {
        let cfg = cfg_for_middleware("token", vec![], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", Some("whatever")).await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[tokio::test]
    async fn middleware_unknown_auth_mode_is_fail_open() {
        // Typos in SIE_AUTH_MODE must NOT silently enforce auth (that
        // would lock every pod out of every non-probe request). The
        // `audit_auth()` warning covers operator visibility.
        let cfg = cfg_for_middleware("staitc", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::OK
        );
    }
}
