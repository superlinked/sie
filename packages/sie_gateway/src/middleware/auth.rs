use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::response::{IntoResponse, Response};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use subtle::ConstantTimeEq;
use tower::{Layer, Service};

use crate::http_error::code as err_code;
use crate::observability::metrics::{AdmissionOutcome, AdmissionOutcomeSlot};

use crate::config::Config;

/// Probe paths shared by every gateway surface (auth exemption here,
/// audit classification in `middleware::audit`, OpenAPI security
/// patching in `openapi`, and downstream key-auth route classifiers).
/// Kubernetes liveness/readiness probes carry no
/// credentials; gating them would take the pod out of rotation during
/// an auth misconfiguration. `/health` (rich status JSON with worker
/// URLs, bundle assignments, queue depth, GPU inventory) is
/// intentionally NOT in this list — see `EXEMPT_OPERATIONAL_PATHS`.
/// `/livez` is a server-side probe, not a gateway route, so it
/// is deliberately absent. Consumers with deliberate extras compose
/// them explicitly instead of re-declaring this pair.
pub const PROBE_PATHS: &[&str] = &["/healthz", "/readyz"];

/// Public API description + rendered reference. Keep client-codegen,
/// discovery, and the browsable `/docs` page usable even when request auth
/// is enabled. The Redoc page is read-only (no in-browser request console),
/// so exempting it carries no token-leak risk.
const EXEMPT_DOC_PATHS: &[&str] = &["/openapi.json", "/docs", "/docs/redoc.standalone.js"];

/// Paths that expose operational data (status page, rich `/health`, `/ws/*`).
/// Exempt from auth only when
/// `SIE_AUTH_EXEMPT_OPERATIONAL=true`; default is fail-closed.
const EXEMPT_OPERATIONAL_PATHS: &[&str] = &["/", "/health"];

#[derive(Clone)]
pub struct AuthLayer {
    config: Arc<Config>,
    /// First `AuditLevel::Error` from [`Config::audit_auth`], resolved once at
    /// layer construction because `Config` is immutable after load and the
    /// audit allocates. `Some` means the auth configuration has no valid
    /// interpretation, which this middleware serves as a refusal rather than
    /// as "auth off" — see [`AuthMiddleware::call`].
    misconfigured: Option<Arc<str>>,
    /// Set once the warning about repeated `Authorization` values is logged.
    repeated_authorization_warned: Arc<AtomicBool>,
}

impl AuthLayer {
    pub fn new(config: Arc<Config>) -> Self {
        let misconfigured: Option<Arc<str>> = config.auth_config_error().map(Arc::from);
        if let Some(reason) = misconfigured.as_deref() {
            tracing::error!(
                audit = "auth",
                "{reason} Refusing every non-probe request until it is corrected."
            );
        }
        Self {
            config,
            misconfigured,
            repeated_authorization_warned: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = AuthMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuthMiddleware {
            inner,
            config: Arc::clone(&self.config),
            misconfigured: self.misconfigured.clone(),
            repeated_authorization_warned: Arc::clone(&self.repeated_authorization_warned),
        }
    }
}

#[derive(Clone)]
pub struct AuthMiddleware<S> {
    inner: S,
    config: Arc<Config>,
    misconfigured: Option<Arc<str>>,
    repeated_authorization_warned: Arc<AtomicBool>,
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
        let misconfigured = self.misconfigured.clone();
        let repeated_authorization_warned = Arc::clone(&self.repeated_authorization_warned);
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let path = req.uri().path();

            // Probes and the read-only API description stay reachable in every
            // configuration, including the broken ones below: gating them would
            // take the pod out of rotation and hide the very misconfiguration
            // the operator needs to see.
            if PROBE_PATHS.contains(&path) || EXEMPT_DOC_PATHS.contains(&path) {
                return inner.call(req).await;
            }

            // A configuration `audit_auth` classifies as an error has no valid
            // interpretation, so it is served as a refusal. Treating it as
            // "auth off" is the strictly worse reading: an unrecognised
            // `SIE_AUTH_MODE`, or tokens supplied against a disabled mode, both
            // describe an operator who intended auth and did not get it. The
            // reason was logged once at layer construction and is never
            // echoed, because it quotes operator input. This gate is
            // deliberately above the operational exemption: an auth config
            // that cannot be read cannot be read as consent to publish worker
            // URLs, queue depth and GPU inventory either.
            if misconfigured.is_some() {
                record_admission_rejection(&req, AdmissionOutcome::AuthMisconfigured);
                return Ok(error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    err_code::GATEWAY_AUTH_MISCONFIGURED,
                    "Gateway auth is misconfigured; refusing the request",
                ));
            }

            if !is_auth_enabled(&config.auth_mode) {
                return inner.call(req).await;
            }

            if config.auth_exempt_operational
                && (EXEMPT_OPERATIONAL_PATHS.contains(&path) || path.starts_with("/ws/"))
            {
                return inner.call(req).await;
            }

            let token = match extract_request_credential(req.headers()) {
                RequestCredential::Token(t) => t,
                RequestCredential::Missing => {
                    record_admission_rejection(&req, AdmissionOutcome::Unauthenticated);
                    return Ok(error_response(
                        StatusCode::UNAUTHORIZED,
                        err_code::UNAUTHORIZED,
                        "Missing Authorization or x-api-key header",
                    ));
                }
                RequestCredential::Conflicting => {
                    record_admission_rejection(&req, AdmissionOutcome::Unauthenticated);
                    return Ok(error_response(
                        StatusCode::UNAUTHORIZED,
                        err_code::UNAUTHORIZED,
                        "Repeated x-api-key headers carry different tokens",
                    ));
                }
                RequestCredential::Unreadable => {
                    record_admission_rejection(&req, AdmissionOutcome::Unauthenticated);
                    return Ok(error_response(
                        StatusCode::UNAUTHORIZED,
                        err_code::UNAUTHORIZED,
                        "Invalid token",
                    ));
                }
            };

            let is_admin = is_admin_endpoint(req.method(), path);

            if is_admin {
                if config.admin_token.is_empty() {
                    record_admission_rejection(&req, AdmissionOutcome::Forbidden);
                    return Ok(error_response(
                        StatusCode::FORBIDDEN,
                        err_code::FORBIDDEN,
                        "Admin token not configured",
                    ));
                }

                if !constant_time_eq_str(&token, &config.admin_token) {
                    record_admission_rejection(&req, AdmissionOutcome::Forbidden);
                    return Ok(error_response(
                        StatusCode::FORBIDDEN,
                        err_code::FORBIDDEN,
                        "Admin token required",
                    ));
                }

                warn_on_repeated_authorization(&req, &repeated_authorization_warned);
                return inner.call(req).await;
            }

            // The admin token grants everything a client token does, so one
            // credential can both manage pools and use them.
            let valid = config
                .auth_tokens
                .iter()
                .any(|valid_token| constant_time_eq_str(&token, valid_token))
                || (!config.admin_token.is_empty()
                    && constant_time_eq_str(&token, &config.admin_token));

            if !valid {
                record_admission_rejection(&req, AdmissionOutcome::Unauthenticated);
                return Ok(error_response(
                    StatusCode::UNAUTHORIZED,
                    err_code::UNAUTHORIZED,
                    "Invalid token",
                ));
            }

            warn_on_repeated_authorization(&req, &repeated_authorization_warned);
            inner.call(req).await
        })
    }
}

/// Log, once per gateway, that an authenticated request repeated
/// `Authorization` with values that 0.13.0 refuses. Only requests that pass
/// authentication count, so an unauthenticated client cannot spend the
/// warning.
fn warn_on_repeated_authorization(req: &Request<Body>, warned: &AtomicBool) {
    if authorization_values_disagree(req.headers()) && !warned.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            audit = "auth",
            "A request repeated the Authorization header with different values. The gateway ignored the later Authorization values; from 0.13.0 it refuses such requests with 401. Send one Authorization header."
        );
    }
}

fn record_admission_rejection(req: &Request<Body>, outcome: AdmissionOutcome) {
    if let Some(slot) = req.extensions().get::<AdmissionOutcomeSlot>() {
        slot.set(outcome);
    }
}

/// Returns true when `auth_mode` should make the middleware enforce auth.
/// Accepts both `"static"` (the code's legacy name) and `"token"` (the
/// name used in the gateway README and Helm values) as aliases.
pub fn is_auth_enabled(mode: &str) -> bool {
    matches!(mode, "static" | "token")
}

/// Returns true for mutations that require the admin token. Reads on
/// these paths (including `GET`) are checked with the regular token, which
/// the admin token also satisfies.
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

/// Extract the token from an `Authorization` header, accepting
/// `Bearer <token>` (case-insensitive prefix) or a raw token value.
///
/// Exported through the crate's library target (`src/lib.rs`) for
/// composition crates that read `Authorization` only. This gateway reads
/// request credentials with [`extract_request_credential`].
#[allow(dead_code)]
pub fn extract_bearer_token(headers: &axum::http::HeaderMap) -> Option<String> {
    authorization_token(headers.get("authorization")?.to_str().ok()?)
}

fn authorization_token(header: &str) -> Option<String> {
    let header = header.trim();
    if header.is_empty() {
        return None;
    }
    let token = if header.to_lowercase().starts_with("bearer ") {
        header[7..].trim()
    } else {
        header
    };
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

/// The credential a request presents.
#[derive(Debug, PartialEq, Eq)]
pub enum RequestCredential {
    Missing,
    Token(String),
    /// Repeated `x-api-key` values carry different tokens.
    Conflicting,
    /// A credential header whose value is not visible ASCII.
    Unreadable,
}

/// Read the request's token. The first `Authorization` value (`Bearer
/// <token>` or a raw value, as [`extract_bearer_token`]) decides when it
/// carries a token, and `x-api-key` is then ignored. A first value that
/// cannot be read is refused, and `Bearer` alone is read as the raw token
/// `Bearer`. When the first value is empty or blank, `x-api-key` decides, and
/// its repeated values must carry the same token. Later `Authorization` values
/// are ignored, as they always were; see [`authorization_values_disagree`].
pub fn extract_request_credential(headers: &axum::http::HeaderMap) -> RequestCredential {
    let authorization = match headers.get("authorization").map(|value| value.to_str()) {
        None => RequestCredential::Missing,
        Some(Err(_)) => RequestCredential::Unreadable,
        Some(Ok(raw)) => {
            authorization_token(raw).map_or(RequestCredential::Missing, RequestCredential::Token)
        }
    };
    match authorization {
        RequestCredential::Missing => header_credential(headers.get_all("x-api-key"), |raw| {
            let raw = raw.trim();
            (!raw.is_empty()).then(|| raw.to_string())
        }),
        decided => decided,
    }
}

fn header_credential<'a>(
    values: impl IntoIterator<Item = &'a axum::http::HeaderValue>,
    parse: impl Fn(&str) -> Option<String>,
) -> RequestCredential {
    let mut found: Option<String> = None;
    for value in values {
        let Ok(raw) = value.to_str() else {
            return RequestCredential::Unreadable;
        };
        let Some(token) = parse(raw) else {
            continue;
        };
        match &found {
            None => found = Some(token),
            Some(existing) if *existing == token => {}
            Some(_) => return RequestCredential::Conflicting,
        }
    }
    found.map_or(RequestCredential::Missing, RequestCredential::Token)
}

/// Whether the request repeats `Authorization` with values that carry
/// different tokens, with one that cannot be read, or with a token only after
/// a first value that carries none. Until 0.13.0 the gateway ignores the later
/// values; from 0.13.0 such a request is refused.
pub fn authorization_values_disagree(headers: &axum::http::HeaderMap) -> bool {
    let values = headers.get_all("authorization");
    values.iter().nth(1).is_some()
        && match header_credential(values, authorization_token) {
            RequestCredential::Conflicting | RequestCredential::Unreadable => true,
            RequestCredential::Token(_) => extract_bearer_token(headers).is_none(),
            RequestCredential::Missing => false,
        }
}

/// The masked request token for audit records, empty when the request
/// presents no single readable token.
pub fn request_token_id(headers: &axum::http::HeaderMap) -> String {
    match extract_request_credential(headers) {
        RequestCredential::Token(token) => mask_token(&token),
        _ => String::new(),
    }
}

/// Mask a token for logs/audit records: keep the last 4 characters,
/// star the rest; tokens of 4 chars or fewer are fully starred.
///
/// Canonical shared implementation: also consumed by `handlers::proxy`
/// and `middleware::audit`.
pub fn mask_token(token: &str) -> String {
    // Char-based, not byte-based: byte slicing panics on a UTF-8 boundary
    // for non-ASCII header values, and this runs in the per-request audit
    // path. Identical output for ASCII tokens; mirrors the Python
    // `sie_sdk.redaction.mask_token`.
    let chars: Vec<char> = token.chars().collect();
    if chars.len() <= 4 {
        "****".to_string()
    } else {
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{}{tail}", "*".repeat(chars.len() - 4))
    }
}

fn constant_time_eq_str(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

fn error_response(status: StatusCode, code: &'static str, message: &str) -> Response {
    let body = serde_json::json!({
        "detail": {
            "code": code,
            "message": message,
        }
    });
    let mut response = (status, axum::Json(body)).into_response();
    if status == StatusCode::UNAUTHORIZED {
        response.headers_mut().insert(
            axum::http::header::WWW_AUTHENTICATE,
            axum::http::HeaderValue::from_static("Bearer"),
        );
    }
    response
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
    fn test_extract_bearer_token_case_insensitive_prefix() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "bearer my-token".parse().unwrap());
        assert_eq!(extract_bearer_token(&headers), Some("my-token".into()));
    }

    #[test]
    fn test_extract_bearer_token_empty_value() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "".parse().unwrap());
        assert_eq!(extract_bearer_token(&headers), None);
    }

    #[test]
    fn test_extract_bearer_token_bearer_only() {
        // "Bearer " trims to "Bearer", which doesn't start with "bearer " (missing trailing space),
        // so it's treated as a raw token value.
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "Bearer ".parse().unwrap());
        assert_eq!(extract_bearer_token(&headers), Some("Bearer".into()));
    }

    #[test]
    fn test_extract_bearer_token_whitespace_trimmed() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert("authorization", "  Bearer  my-token  ".parse().unwrap());
        assert_eq!(extract_bearer_token(&headers), Some("my-token".into()));
    }

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

    #[test]
    fn test_mask_token_multibyte_does_not_panic() {
        // A multibyte char straddling the old byte-index cut point must not
        // panic and must mask by characters.
        assert_eq!(mask_token("secrèt-tokén"), "********okén");
        assert_eq!(mask_token("ééé"), "****");
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
        // This predicate answers "should tokens be checked", and an
        // unrecognised mode is not a mode whose tokens can be checked. It is
        // NOT the bypass decision: the middleware refuses an unrecognised mode
        // outright via `Config::auth_config_error`, so reaching this `false`
        // for a typo is unreachable in the request path.
        assert!(!is_auth_enabled("staitc"));
    }

    #[test]
    fn test_exempt_paths_partitioned() {
        // Probe paths are always exempt; operational paths are not.
        assert!(PROBE_PATHS.contains(&"/healthz"));
        assert!(PROBE_PATHS.contains(&"/readyz"));
        assert!(EXEMPT_DOC_PATHS.contains(&"/openapi.json"));
        assert!(EXEMPT_DOC_PATHS.contains(&"/docs"));
        assert!(EXEMPT_DOC_PATHS.contains(&"/docs/redoc.standalone.js"));
        assert!(!PROBE_PATHS.contains(&"/health"));
        assert!(EXEMPT_OPERATIONAL_PATHS.contains(&"/"));
        assert!(EXEMPT_OPERATIONAL_PATHS.contains(&"/health"));
    }

    // ── End-to-end middleware tests ────────────────────────────────
    //
    // These exercise the full request pipeline (AuthLayer wraps a
    // dummy handler) and cover what the helper-only tests cannot:
    //   - which paths actually reach the handler vs return 401/403
    //   - admin vs user token routing on admin-gated prefixes
    //   - pass-through when auth is deliberately disabled, and refusal when
    //     the auth configuration is an error
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
            nats_user: String::new(),
            nats_password: String::new(),
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
            max_item_text_bytes: 2 * 1024 * 1024,
            max_stream_pending: 0,
            max_lane_in_flight_items: 0,
            lane_backpressure_enforce: false,
            stream_max_age_s: 0,
            stream_storage: crate::config::StreamStorage::Memory,
            stream_num_replicas: 1,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            configured_physical_lanes: Default::default(),
            static_queue_pools: Vec::new(),
            model_aliases: HashMap::new(),
            published_model_aliases: Default::default(),
            bundles_dir: String::new(),
            models_dir: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_modal_proxy_token: None,
            payload_store_url: String::new(),
            public_base_url: None,
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
            .route("/openapi.json", get(|| async { "{}" }))
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
    async fn middleware_openapi_is_always_exempt() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::GET, "/openapi.json", None).await,
            StatusCode::OK
        );
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
    async fn middleware_marks_bounded_admission_outcomes_when_outer_slot_is_present() {
        let unauthenticated = AdmissionOutcomeSlot::default();
        let r = test_router(cfg_for_middleware("token", vec!["user"], "admin", false))
            .layer(axum::Extension(unauthenticated.clone()));
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            unauthenticated.get(),
            Some(AdmissionOutcome::Unauthenticated)
        );

        let misconfigured = AdmissionOutcomeSlot::default();
        let r = test_router(cfg_for_middleware("token", vec![], "admin", false))
            .layer(axum::Extension(misconfigured.clone()));
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", Some("whatever")).await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            misconfigured.get(),
            Some(AdmissionOutcome::AuthMisconfigured)
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
    async fn middleware_admin_token_also_serves_client_routes() {
        // One credential can create a pool and then send it traffic.
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        for (method, path) in [
            (Method::POST, "/v1/encode/any"),
            (Method::GET, "/v1/pools"),
            (Method::GET, "/v1/pools/gpu-l4"),
        ] {
            let r = test_router(Arc::clone(&cfg));
            assert_eq!(
                send(r, method.clone(), path, Some("admin")).await,
                StatusCode::OK,
                "{method} {path}"
            );
            let r = test_router(Arc::clone(&cfg));
            assert_eq!(
                send_headers(r, method.clone(), path, &[("x-api-key", "admin")]).await,
                StatusCode::OK,
                "{method} {path} via x-api-key"
            );
        }
    }

    #[tokio::test]
    async fn middleware_unset_admin_token_adds_no_client_credential() {
        let cfg = cfg_for_middleware("token", vec!["user"], "", false);
        for bearer in [None, Some("admin"), Some("bogus")] {
            let r = test_router(Arc::clone(&cfg));
            assert_eq!(
                send(r, Method::POST, "/v1/encode/any", bearer).await,
                StatusCode::UNAUTHORIZED,
                "{bearer:?}"
            );
        }
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
    async fn middleware_unknown_auth_mode_refuses() {
        // A typo in SIE_AUTH_MODE used to disable auth for every route while
        // `audit_auth()` logged an error nobody read. An unrecognised mode has
        // no valid reading, so it is refused rather than served unauthenticated.
        let cfg = cfg_for_middleware("staitc", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[tokio::test]
    async fn middleware_tokens_without_enabled_mode_refuses() {
        // The likelier operator mistake: tokens supplied, mode left disabled.
        // `audit_auth()` already calls this dead configuration; serving it as
        // "auth off" hands an operator who configured a token an open gateway.
        let cfg = cfg_for_middleware("none", vec!["user"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", Some("user")).await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[tokio::test]
    async fn middleware_admin_token_alone_with_disabled_mode_passes_through() {
        // With `SIE_CONFIG_SERVICE_TOKEN` unset, `SIE_ADMIN_TOKEN` is also the
        // sie-config credential, so "auth off + admin token present" is an
        // ordinary deployment shape.
        let cfg = cfg_for_middleware("none", vec![], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_deliberately_disabled_auth_still_passes_through() {
        let cfg = cfg_for_middleware("none", vec![], "", false);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::POST, "/v1/encode/any", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_probes_survive_a_misconfigured_auth_mode() {
        // The refusal must not take the pod out of rotation: a crashlooping or
        // unready gateway hides the misconfiguration instead of surfacing it.
        for path in ["/healthz", "/readyz", "/openapi.json"] {
            let cfg = cfg_for_middleware("staitc", vec!["user"], "admin", false);
            let r = test_router(cfg);
            assert_eq!(
                send(r, Method::GET, path, None).await,
                StatusCode::OK,
                "{path} must stay reachable while auth config is broken"
            );
        }
    }

    #[tokio::test]
    async fn middleware_misconfigured_auth_overrides_the_operational_exemption() {
        let cfg = cfg_for_middleware("staitc", vec!["user"], "admin", true);
        let r = test_router(cfg);
        assert_eq!(
            send(r, Method::GET, "/health", None).await,
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    fn headers(pairs: &[(&str, &str)]) -> axum::http::HeaderMap {
        let mut map = axum::http::HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                axum::http::HeaderValue::from_str(value).unwrap(),
            );
        }
        map
    }

    #[test]
    fn request_credential_reads_either_header() {
        for pairs in [
            vec![("authorization", "Bearer tok")],
            vec![("authorization", "bearer tok")],
            vec![("authorization", "tok")],
            vec![("x-api-key", "tok")],
            vec![("x-api-key", "  tok  ")],
            vec![("authorization", "Bearer tok"), ("x-api-key", "tok")],
            vec![("authorization", "Bearer tok"), ("x-api-key", "other")],
            vec![("authorization", "Bearer tok"), ("authorization", "tok")],
            vec![
                ("authorization", "Bearer tok"),
                ("authorization", "Bearer other"),
            ],
            vec![("authorization", ""), ("x-api-key", "tok")],
        ] {
            assert_eq!(
                extract_request_credential(&headers(&pairs)),
                RequestCredential::Token("tok".into()),
                "{pairs:?}"
            );
        }
    }

    #[test]
    fn request_credential_lets_authorization_decide() {
        let unreadable_key = {
            let mut map = headers(&[("authorization", "Bearer tok")]);
            map.insert(
                "x-api-key",
                axum::http::HeaderValue::from_bytes(b"t\xffk").unwrap(),
            );
            map
        };
        assert_eq!(
            extract_request_credential(&unreadable_key),
            RequestCredential::Token("tok".into())
        );
        let unreadable_authorization = {
            let mut map = headers(&[("x-api-key", "tok")]);
            map.insert(
                "authorization",
                axum::http::HeaderValue::from_bytes(b"t\xffk").unwrap(),
            );
            map
        };
        assert_eq!(
            extract_request_credential(&unreadable_authorization),
            RequestCredential::Unreadable
        );
    }

    fn with_unreadable(
        mut map: axum::http::HeaderMap,
        name: &'static str,
    ) -> axum::http::HeaderMap {
        map.append(
            name,
            axum::http::HeaderValue::from_bytes(b"t\xffk").unwrap(),
        );
        map
    }

    #[test]
    fn request_credential_refuses_repeated_x_api_key_that_disagrees() {
        assert_eq!(
            extract_request_credential(&headers(&[("x-api-key", "tok"), ("x-api-key", "other")])),
            RequestCredential::Conflicting
        );
        assert_eq!(
            extract_request_credential(&with_unreadable(
                headers(&[("x-api-key", "tok")]),
                "x-api-key"
            )),
            RequestCredential::Unreadable
        );
    }

    #[test]
    fn request_credential_uses_the_first_authorization_value() {
        assert_eq!(
            extract_request_credential(&with_unreadable(
                headers(&[("authorization", "Bearer tok")]),
                "authorization"
            )),
            RequestCredential::Token("tok".into())
        );
        assert_eq!(
            extract_request_credential(&headers(&[
                ("authorization", ""),
                ("authorization", "Bearer other")
            ])),
            RequestCredential::Missing
        );
        let mut unreadable_first = with_unreadable(axum::http::HeaderMap::new(), "authorization");
        unreadable_first.append(
            "authorization",
            axum::http::HeaderValue::from_static("Bearer tok"),
        );
        assert_eq!(
            extract_request_credential(&unreadable_first),
            RequestCredential::Unreadable
        );
        unreadable_first.append("x-api-key", axum::http::HeaderValue::from_static("tok"));
        assert_eq!(
            extract_request_credential(&unreadable_first),
            RequestCredential::Unreadable
        );
        assert_eq!(
            extract_request_credential(&headers(&[
                ("authorization", "Bearer "),
                ("x-api-key", "tok")
            ])),
            RequestCredential::Token("Bearer".into())
        );
    }

    #[test]
    fn authorization_values_disagree_only_for_repeats_with_another_token() {
        for pairs in [
            vec![
                ("authorization", "Bearer tok"),
                ("authorization", "Bearer other"),
            ],
            vec![("authorization", "Bearer tok"), ("authorization", "other")],
            vec![("authorization", ""), ("authorization", "Bearer tok")],
            vec![
                ("authorization", " "),
                ("authorization", "tok"),
                ("authorization", "Bearer tok"),
                ("x-api-key", "other"),
            ],
        ] {
            assert!(authorization_values_disagree(&headers(&pairs)), "{pairs:?}");
        }
        assert!(authorization_values_disagree(&with_unreadable(
            headers(&[("authorization", "Bearer tok")]),
            "authorization"
        )));
        for pairs in [
            vec![("authorization", "Bearer tok")],
            vec![("authorization", "Bearer tok"), ("authorization", "tok")],
            vec![("authorization", "Bearer tok"), ("authorization", "")],
            vec![("authorization", ""), ("authorization", " ")],
            vec![("x-api-key", "tok"), ("x-api-key", "other")],
        ] {
            assert!(
                !authorization_values_disagree(&headers(&pairs)),
                "{pairs:?}"
            );
        }
    }

    #[test]
    fn request_credential_missing_and_unreadable() {
        assert_eq!(
            extract_request_credential(&headers(&[])),
            RequestCredential::Missing
        );
        assert_eq!(
            extract_request_credential(&headers(&[("authorization", ""), ("x-api-key", " ")])),
            RequestCredential::Missing
        );
        let mut map = axum::http::HeaderMap::new();
        map.insert(
            "x-api-key",
            axum::http::HeaderValue::from_bytes(b"t\xffk").unwrap(),
        );
        assert_eq!(
            extract_request_credential(&map),
            RequestCredential::Unreadable
        );
    }

    #[test]
    fn request_token_id_masks_either_header() {
        assert_eq!(
            request_token_id(&headers(&[("x-api-key", "secret-token-123")])),
            "************-123"
        );
        assert_eq!(
            request_token_id(&headers(&[("authorization", "Bearer secret-token-123")])),
            "************-123"
        );
        assert_eq!(
            request_token_id(&headers(&[
                ("authorization", "Bearer secret-token-123"),
                ("x-api-key", "other-token-456")
            ])),
            "************-123"
        );
        assert_eq!(
            request_token_id(&headers(&[
                ("authorization", "Bearer a-token-111"),
                ("authorization", "Bearer b-token-222")
            ])),
            mask_token("a-token-111")
        );
        assert_eq!(
            request_token_id(&headers(&[
                ("x-api-key", "a-token-111"),
                ("x-api-key", "b-token-222")
            ])),
            ""
        );
    }

    async fn send_headers(
        router: Router,
        method: Method,
        path: &str,
        pairs: &[(&str, &str)],
    ) -> StatusCode {
        let mut req = HttpRequest::builder().method(method).uri(path);
        for (name, value) in pairs {
            req = req.header(*name, *value);
        }
        router
            .oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn middleware_401_names_the_bearer_scheme() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let response = test_router(cfg)
            .oneshot(
                HttpRequest::builder()
                    .method(Method::POST)
                    .uri("/v1/encode/any")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()["www-authenticate"], "Bearer");
    }

    #[tokio::test]
    async fn middleware_accepts_x_api_key() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send_headers(r, Method::POST, "/v1/encode/any", &[("x-api-key", "user")]).await,
            StatusCode::OK
        );
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send_headers(r, Method::POST, "/v1/encode/any", &[("x-api-key", "bogus")]).await,
            StatusCode::UNAUTHORIZED
        );
        let r = test_router(cfg);
        assert_eq!(
            send_headers(
                r,
                Method::POST,
                "/v1/encode/any",
                &[("authorization", "Bearer user"), ("x-api-key", "user")]
            )
            .await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn middleware_lets_authorization_decide_over_x_api_key() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        for (pairs, path, expected) in [
            (
                [("authorization", "Bearer user"), ("x-api-key", "bogus")],
                "/v1/encode/any",
                StatusCode::OK,
            ),
            (
                [("authorization", "Bearer bogus"), ("x-api-key", "user")],
                "/v1/encode/any",
                StatusCode::UNAUTHORIZED,
            ),
            (
                [("authorization", "Bearer admin"), ("x-api-key", "user")],
                "/v1/pools",
                StatusCode::OK,
            ),
            (
                [("authorization", "Bearer user"), ("x-api-key", "admin")],
                "/v1/pools",
                StatusCode::FORBIDDEN,
            ),
        ] {
            let r = test_router(Arc::clone(&cfg));
            assert_eq!(
                send_headers(r, Method::POST, path, &pairs).await,
                expected,
                "{pairs:?} {path}"
            );
        }
    }

    #[derive(Clone, Default)]
    struct CapturedLog(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLog {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedLog {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl CapturedLog {
        fn count(&self, needle: &str) -> usize {
            String::from_utf8_lossy(&self.0.lock().unwrap())
                .matches(needle)
                .count()
        }
    }

    #[tokio::test]
    async fn middleware_uses_the_first_authorization_value_and_warns_once() {
        use tracing::instrument::WithSubscriber;

        let log = CapturedLog::default();
        let dispatch = tracing::Dispatch::new(
            tracing_subscriber::fmt()
                .with_writer(log.clone())
                .with_ansi(false)
                .finish(),
        );
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let router = test_router(cfg);
        let repeated = || {
            HttpRequest::builder()
                .method(Method::POST)
                .uri("/v1/encode/any")
                .header("authorization", "Bearer user")
                .header("authorization", "Bearer bogus")
        };
        let unreadable_repeat = HttpRequest::builder()
            .method(Method::POST)
            .uri("/v1/encode/any")
            .header("authorization", "Bearer user")
            .header(
                "authorization",
                axum::http::HeaderValue::from_bytes(b"t\xffk").unwrap(),
            );
        let bogus_first = HttpRequest::builder()
            .method(Method::POST)
            .uri("/v1/encode/any")
            .header("authorization", "Bearer bogus")
            .header("authorization", "Bearer user");
        let user_first_admin_key = HttpRequest::builder()
            .method(Method::POST)
            .uri("/v1/pools")
            .header("authorization", "Bearer user")
            .header("authorization", "Bearer admin")
            .header("x-api-key", "admin");

        for (request, expected) in [
            (repeated(), StatusCode::OK),
            (repeated(), StatusCode::OK),
            (unreadable_repeat, StatusCode::OK),
            (bogus_first, StatusCode::UNAUTHORIZED),
            (user_first_admin_key, StatusCode::FORBIDDEN),
        ] {
            let status = router
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .with_subscriber(dispatch.clone())
                .await
                .unwrap()
                .status();
            assert_eq!(status, expected);
        }
        assert_eq!(log.count("repeated the Authorization header"), 1);
        assert_eq!(log.count("from 0.13.0 it refuses such requests"), 1);
    }

    fn repeated_authorization(path: &str, first: &str, second: &str) -> Request<Body> {
        HttpRequest::builder()
            .method(Method::POST)
            .uri(path)
            .header("authorization", format!("Bearer {first}"))
            .header("authorization", format!("Bearer {second}"))
            .body(Body::empty())
            .unwrap()
    }

    fn repeated_authorization_after_an_empty_one(x_api_keys: &[&str]) -> axum::http::HeaderMap {
        let mut pairs = vec![
            ("authorization", ""),
            ("authorization", "Bearer later"),
            ("authorization", "Bearer bogus"),
        ];
        pairs.extend(x_api_keys.iter().map(|key| ("x-api-key", *key)));
        headers(&pairs)
    }

    fn encode_with(headers: axum::http::HeaderMap) -> Request<Body> {
        let mut request = HttpRequest::builder()
            .method(Method::POST)
            .uri("/v1/encode/any")
            .body(Body::empty())
            .unwrap();
        *request.headers_mut() = headers;
        request
    }

    #[tokio::test]
    async fn middleware_warns_only_for_a_repeat_that_passes_authentication() {
        use tracing::instrument::WithSubscriber;

        let empty_first = repeated_authorization_after_an_empty_one;
        for (case, admin, refused, refusal, passing) in [
            (
                "invalid token",
                "admin",
                repeated_authorization("/v1/encode/any", "bogus", "user"),
                (StatusCode::UNAUTHORIZED, "Invalid token"),
                repeated_authorization("/v1/encode/any", "user", "bogus"),
            ),
            (
                "admin token required",
                "admin",
                repeated_authorization("/v1/pools", "user", "admin"),
                (StatusCode::FORBIDDEN, "Admin token required"),
                repeated_authorization("/v1/pools", "admin", "user"),
            ),
            (
                "admin token not configured",
                "",
                repeated_authorization("/v1/pools", "admin", "user"),
                (StatusCode::FORBIDDEN, "Admin token not configured"),
                repeated_authorization("/v1/encode/any", "user", "bogus"),
            ),
            (
                "missing x-api-key",
                "admin",
                encode_with(empty_first(&[])),
                (
                    StatusCode::UNAUTHORIZED,
                    "Missing Authorization or x-api-key header",
                ),
                encode_with(empty_first(&["user"])),
            ),
            (
                "unreadable x-api-key",
                "admin",
                encode_with(with_unreadable(empty_first(&[]), "x-api-key")),
                (StatusCode::UNAUTHORIZED, "Invalid token"),
                encode_with(empty_first(&["user"])),
            ),
            (
                "conflicting x-api-key",
                "admin",
                encode_with(empty_first(&["user", "other"])),
                (
                    StatusCode::UNAUTHORIZED,
                    "Repeated x-api-key headers carry different tokens",
                ),
                encode_with(empty_first(&["user"])),
            ),
        ] {
            let log = CapturedLog::default();
            let dispatch = tracing::Dispatch::new(
                tracing_subscriber::fmt()
                    .with_writer(log.clone())
                    .with_ansi(false)
                    .finish(),
            );
            let router = test_router(cfg_for_middleware("token", vec!["user"], admin, false));

            let response = router
                .clone()
                .oneshot(refused)
                .with_subscriber(dispatch.clone())
                .await
                .unwrap();
            assert_eq!(response.status(), refusal.0, "{case}");
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            assert!(
                String::from_utf8_lossy(&body).contains(refusal.1),
                "{case}: {body:?}"
            );
            assert_eq!(log.count("repeated the Authorization header"), 0, "{case}");

            let status = router
                .oneshot(passing)
                .with_subscriber(dispatch)
                .await
                .unwrap()
                .status();
            assert_eq!(status, StatusCode::OK, "{case}");
            assert_eq!(log.count("repeated the Authorization header"), 1, "{case}");
            assert_eq!(
                log.count("The gateway ignored the later Authorization values"),
                1,
                "{case}"
            );
        }
    }

    #[tokio::test]
    async fn middleware_lets_x_api_key_decide_after_an_empty_authorization_and_warns() {
        use tracing::instrument::WithSubscriber;

        let cfg = cfg_for_middleware(
            "token",
            vec!["user-token-aaaa", "key-token-bbbb"],
            "admin-token-zzzz",
            false,
        );
        for (path, pairs, expected, token, warnings) in [
            (
                "/v1/encode/any",
                vec![
                    ("authorization", ""),
                    ("authorization", "Bearer user-token-aaaa"),
                    ("x-api-key", "key-token-bbbb"),
                ],
                StatusCode::OK,
                Some("key-token-bbbb"),
                1,
            ),
            (
                "/v1/encode/any",
                vec![
                    ("authorization", " "),
                    ("authorization", "Bearer unknown-cccc"),
                    ("x-api-key", "key-token-bbbb"),
                ],
                StatusCode::OK,
                Some("key-token-bbbb"),
                1,
            ),
            (
                "/v1/pools",
                vec![
                    ("authorization", ""),
                    ("authorization", "Bearer user-token-aaaa"),
                    ("x-api-key", "admin-token-zzzz"),
                ],
                StatusCode::OK,
                Some("admin-token-zzzz"),
                1,
            ),
            (
                "/v1/pools",
                vec![
                    ("authorization", ""),
                    ("authorization", "Bearer admin-token-zzzz"),
                    ("x-api-key", "user-token-aaaa"),
                ],
                StatusCode::FORBIDDEN,
                Some("user-token-aaaa"),
                0,
            ),
            (
                "/v1/encode/any",
                vec![
                    ("authorization", ""),
                    ("authorization", "Bearer user-token-aaaa"),
                ],
                StatusCode::UNAUTHORIZED,
                None,
                0,
            ),
        ] {
            let request_headers = headers(&pairs);
            assert_eq!(
                request_token_id(&request_headers),
                token.map(mask_token).unwrap_or_default(),
                "{pairs:?}"
            );
            let log = CapturedLog::default();
            let dispatch = tracing::Dispatch::new(
                tracing_subscriber::fmt()
                    .with_writer(log.clone())
                    .with_ansi(false)
                    .finish(),
            );
            let mut request = HttpRequest::builder()
                .method(Method::POST)
                .uri(path)
                .body(Body::empty())
                .unwrap();
            *request.headers_mut() = request_headers;

            let status = test_router(Arc::clone(&cfg))
                .oneshot(request)
                .with_subscriber(dispatch)
                .await
                .unwrap()
                .status();
            assert_eq!(status, expected, "{pairs:?}");
            assert_eq!(
                log.count("The gateway ignored the later Authorization values"),
                warnings,
                "{pairs:?}"
            );
        }
    }

    #[tokio::test]
    async fn middleware_refuses_repeated_x_api_key_that_disagrees() {
        let cfg = cfg_for_middleware("token", vec!["user", "other"], "admin", false);
        let r = test_router(cfg);
        assert_eq!(
            send_headers(
                r,
                Method::POST,
                "/v1/encode/any",
                &[("x-api-key", "user"), ("x-api-key", "other")]
            )
            .await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn middleware_admin_routes_accept_the_admin_token_in_x_api_key() {
        let cfg = cfg_for_middleware("token", vec!["user"], "admin", false);
        let r = test_router(Arc::clone(&cfg));
        assert_eq!(
            send_headers(r, Method::POST, "/v1/pools", &[("x-api-key", "admin")]).await,
            StatusCode::OK
        );
        let r = test_router(cfg);
        assert_eq!(
            send_headers(r, Method::POST, "/v1/pools", &[("x-api-key", "user")]).await,
            StatusCode::FORBIDDEN
        );
    }
}
