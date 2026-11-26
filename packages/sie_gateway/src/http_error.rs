//! Canonical JSON error bodies (FastAPI-style ``{"detail": {"code", "message", ...}}``).
//! SDK-stable responses (``502 MODEL_LOAD_FAILED``, ``503`` retryable ``{"error":{...}}``)
//! intentionally stay on their existing shapes.
//!
//! The OpenAI-compatible surfaces introduce a parallel OpenAI-shaped error envelope
//! (see :mod:`openai_code` / :func:`json_openai_error`). Generation-compatible
//! endpoints build that envelope directly. ``/v1/embeddings`` calls the SIE-native
//! encode path internally and translates non-200 responses back into the OpenAI
//! envelope. Native encode/score/extract endpoints keep ``json_detail`` to preserve
//! existing SDK error-handling contracts. [`openai_error_envelope`] re-renders the
//! ``detail`` refusals that reach an OpenAI-compatible route from outside its
//! handler, such as the auth middleware's.

use axum::body::{to_bytes, Body, HttpBody};
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use serde_json::{json, Map, Value};

/// Response-extension marker: this error response is a fault the GATEWAY owns,
/// produced AFTER the worker already ran, and the client receives none of the
/// work it paid for.
///
/// A handler that translates a successful worker result into a client-facing
/// shape can fail on its own — a subtitle timestamp the model left open, a
/// response body larger than the read bound, an unexpected item count. Those are
/// SIE's bugs, not the customer's request, and unlike a partial worker failure
/// there is no delivered output to bill for. Attaching this to the response lets
/// a downstream billing composition tell the two apart; without it, "the
/// response is non-2xx" is indistinguishable from "one item of a batch failed",
/// which legitimately still settles its successful siblings.
///
/// The OSS gateway attaches it and does nothing else with it. `stage` names the
/// translation step for the alert; keep it a small closed vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatewayOwnedFault {
    pub stage: &'static str,
}

impl GatewayOwnedFault {
    pub const fn new(stage: &'static str) -> Self {
        Self { stage }
    }

    /// Attach the marker to an already-built error response.
    ///
    /// **This marker must survive every layer between the handler that sets it
    /// and the composition that reads it, and it fails toward CHARGING if it
    /// does not.** A dropped marker is not a visible error — the response is
    /// unchanged and the customer is simply billed for a result they never
    /// received, which is the exact outcome the marker exists to prevent.
    ///
    /// Wrapping a response is safe: `Response::from_parts(parts, new_body)`
    /// carries `parts.extensions` through, which is how the managed edge's
    /// permit-body layer preserves it today. REBUILDING one is not — any layer
    /// that constructs a fresh `Response` from a status and body, rather than
    /// re-using the original's parts, silently discards it. A layer that must
    /// rebuild has to copy the extensions across explicitly.
    pub fn mark(self, mut response: Response) -> Response {
        response.extensions_mut().insert(self);
        response
    }
}

/// Stable ``detail.code`` values for gateway-generated errors.
pub mod code {
    pub const MODEL_NOT_FOUND: &str = "MODEL_NOT_FOUND";
    pub const BUNDLE_NOT_FOUND: &str = "BUNDLE_NOT_FOUND";
    pub const POOL_NOT_FOUND: &str = "POOL_NOT_FOUND";
    pub const INVALID_REQUEST: &str = "INVALID_REQUEST";
    pub const PAYLOAD_TOO_LARGE: &str = "PAYLOAD_TOO_LARGE";
    pub const QUEUE_UNAVAILABLE: &str = "QUEUE_UNAVAILABLE";
    pub const GPU_NOT_CONFIGURED: &str = "GPU_NOT_CONFIGURED";
    pub const BUNDLE_ROUTING_CONFLICT: &str = "BUNDLE_ROUTING_CONFLICT";
    pub const CONFIG_RESOLVE_BUNDLE_CONFLICT: &str = "BUNDLE_ROUTING_CONFLICT";
    pub const INTERNAL_ERROR: &str = "INTERNAL_SERVER_ERROR";
    pub const GATEWAY_TIMEOUT: &str = "GATEWAY_TIMEOUT";
    pub const SERIALIZATION_ERROR: &str = "SERIALIZATION_ERROR";
    pub const UNAUTHORIZED: &str = "UNAUTHORIZED";
    pub const FORBIDDEN: &str = "FORBIDDEN";
    pub const NOT_FOUND: &str = "NOT_FOUND";
    pub const RATE_LIMIT: &str = "RATE_LIMIT";
    pub const GATEWAY_AUTH_MISCONFIGURED: &str = "GATEWAY_AUTH_MISCONFIGURED";
    pub const DEFAULT_POOL_DELETE_FORBIDDEN: &str = "DEFAULT_POOL_DELETE_FORBIDDEN";
    pub const POOL_OPERATION_FORBIDDEN: &str = "POOL_OPERATION_FORBIDDEN";
    pub const POOL_CAPACITY_UNAVAILABLE: &str = "POOL_CAPACITY_UNAVAILABLE";
    pub const PROVISIONING: &str = "PROVISIONING";
}

/// ``{"detail": {"code", "message"}}``
pub fn json_detail(code: &str, message: impl Into<String>) -> Value {
    json!({
        "detail": {
            "code": code,
            "message": message.into(),
        }
    })
}

/// ``{"detail": { "code", "message", ...extra }}`` — merge extra keys into ``detail``.
pub fn json_detail_merge(
    code: &str,
    message: impl Into<String>,
    extra: Map<String, Value>,
) -> Value {
    let mut d = Map::new();
    d.insert("code".to_string(), json!(code));
    d.insert("message".to_string(), json!(message.into()));
    d.extend(extra);
    json!({ "detail": Value::Object(d) })
}

/// Stable ``error.type`` values for the OpenAI-shaped envelope used by
/// ``/v1/generate/{model}`` and ``/v1/chat/completions``.
pub mod openai_type {
    pub const INVALID_REQUEST: &str = "invalid_request_error";
    pub const MODEL_NOT_FOUND: &str = "model_not_found";
    pub const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";
    /// OpenAI's rate-limit type. Surfaced by the chat handler when the
    /// gateway returns 429 — currently emitted when a NAKed request
    /// also fails to republish to the pool (KV budget exhausted across
    /// the whole pool).
    pub const RATE_LIMIT: &str = "rate_limit_error";
    pub const SERVER_ERROR: &str = "server_error";
    pub const AUTHENTICATION: &str = "authentication_error";
    pub const PERMISSION: &str = "permission_error";
}

/// Stable ``error.code`` values paired with the types above. These are
/// the SIE-native discriminators clients should branch on; the
/// human-readable ``message`` is for display only.
pub mod openai_code {
    pub const UNSUPPORTED_FIELD: &str = "unsupported_field";
    pub const PAYLOAD_TOO_LARGE: &str = "payload_too_large";
    pub const MODEL_NOT_FOUND: &str = "model_not_found";
    pub const CONTEXT_EXCEEDED: &str = "context_exceeded";
    pub const TRANSPORT_FAILURE: &str = "transport_failure";
    pub const CANCELLED: &str = "cancelled";
    pub const INVALID_REQUEST: &str = "invalid_request";
    /// The requested ``lora_adapter`` served-name is not among the model's
    /// advertised adapters (see ``/v1/models`` ``capabilities.lora_adapters``).
    pub const UNKNOWN_LORA_ADAPTER: &str = "unknown_lora_adapter";
    /// Surfaced by ``proxy_chat`` when the worker's terminal chunk is
    /// missing the ``usage`` block — the OpenAI envelope requires it,
    /// so we cannot synthesize a valid response.
    #[allow(dead_code)]
    pub const MALFORMED_WORKER_RESPONSE: &str = "malformed_worker_response";
    pub const FIRST_CHUNK_TIMEOUT: &str = "first_chunk_timeout";
    pub const INTER_CHUNK_TIMEOUT: &str = "inter_chunk_timeout";
    pub const OVERALL_TIMEOUT: &str = "overall_timeout";
    /// Surfaced when the gateway returns 429. Today's only emitter is
    /// the NAK-then-pool-republish-fails path; future rate limiters
    /// (per-token-bucket, per-tenant-quota) would land here too.
    pub const RATE_LIMIT_EXCEEDED: &str = "rate_limit_exceeded";
    /// OpenAI-compatible scale-from-zero / worker provisioning signal.
    pub const PROVISIONING: &str = "provisioning";
    /// Worker terminal for a generation that stopped but produced no visible
    /// output text or tool calls (e.g. only private reasoning, #3104/#3136).
    /// A budget that runs out first keeps ``finish_reason: "length"`` instead.
    /// Terminal and non-retryable: tokens were genuinely consumed, so it
    /// settles exactly like ``inference_error``.
    pub const EMPTY_MODEL_OUTPUT: &str = "empty_model_output";
    /// Worker terminal for a guard model whose first output positions carry
    /// no complete Yes/No verdict distribution. Terminal and non-retryable;
    /// settles like ``empty_model_output``.
    pub const INVALID_GUARD_VERDICT: &str = "invalid_guard_verdict";
}

/// OpenAI-shaped error body:
///
/// ```json
/// {"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}
/// ```
///
/// ``param`` carries the offending field name when known (e.g. ``"messages"``,
/// ``"max_completion_tokens"``); pass ``None`` when the error is not field-specific.
pub fn json_openai_error(
    message: impl Into<String>,
    err_type: &'static str,
    param: Option<&str>,
    code: &'static str,
) -> Value {
    json!({
        "error": {
            "message": message.into(),
            "type": err_type,
            "param": param,
            "code": code,
        }
    })
}

/// Translate a SIE-native ``detail.code`` (see :mod:`code`) into the
/// matching OpenAI ``(type, code)`` pair. Used by the OpenAI-shaped
/// surface (``/v1/embeddings``) to re-surface inner SIE-native errors —
/// e.g. an inner ``/v1/encode`` failure — in the OpenAI envelope rather
/// than leaking a ``detail`` body. Unknown codes fall back to a generic
/// ``server_error`` / ``transport_failure`` (never echoing raw internals).
pub fn openai_error_from_detail_code(code: &str) -> (&'static str, &'static str) {
    match code {
        // 400 / 413 — caller-fixable input problems.
        "INVALID_REQUEST" | "INVALID_INPUT" | "INPUT_TOO_LONG" => {
            (openai_type::INVALID_REQUEST, openai_code::INVALID_REQUEST)
        }
        "PAYLOAD_TOO_LARGE" => (openai_type::INVALID_REQUEST, openai_code::PAYLOAD_TOO_LARGE),
        // 404 — model id not routable.
        "MODEL_NOT_FOUND" => (openai_type::INVALID_REQUEST, openai_code::MODEL_NOT_FOUND),
        // 409 / 400 — bundle override conflicts with model routing; pool/bundle
        // not found. All caller-fixable, so stay in invalid_request_error.
        "BUNDLE_ROUTING_CONFLICT" | "POOL_NOT_FOUND" | "BUNDLE_NOT_FOUND" => {
            (openai_type::INVALID_REQUEST, openai_code::INVALID_REQUEST)
        }
        // 503 — pool/queue saturated or unavailable.
        "QUEUE_UNAVAILABLE" | "POOL_CAPACITY_UNAVAILABLE" => {
            (openai_type::SERVER_ERROR, openai_code::TRANSPORT_FAILURE)
        }
        "PROVISIONING" => (openai_type::SERVER_ERROR, openai_code::PROVISIONING),
        // 504 — overall deadline blown.
        "GATEWAY_TIMEOUT" => (openai_type::SERVER_ERROR, openai_code::OVERALL_TIMEOUT),
        // 429 — forward-compatible with Tier 0 rate limiting.
        "RATE_LIMIT" => (openai_type::RATE_LIMIT, openai_code::RATE_LIMIT_EXCEEDED),
        // 500 + anything unmapped.
        _ => (openai_type::SERVER_ERROR, openai_code::TRANSPORT_FAILURE),
    }
}

/// OpenAI-envelope error for the ``/v1/embeddings`` surface, keyed by a
/// SIE-native ``code`` (see :mod:`code`). Thin wrapper over
/// :func:`json_openai_error` + :func:`openai_error_from_detail_code` so
/// every in-handler reject and every re-surfaced inner error share one
/// classifier. ``param`` carries the offending field name when known.
pub fn embeddings_error(sie_code: &str, param: Option<&str>, message: impl Into<String>) -> Value {
    let (err_type, err_code) = openai_error_from_detail_code(sie_code);
    json_openai_error(message, err_type, param, err_code)
}

const OPENAI_COMPATIBLE_PATHS: [&str; 7] = [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
    "/v1/embeddings",
    "/v1/models",
    "/v1/audio/transcriptions",
    "/v1/moderations",
];

const MAX_ERROR_BODY: usize = 1024 * 1024;

/// Whether a path is one of the gateway's OpenAI-compatible routes or below
/// one: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
/// `/v1/embeddings`, `/v1/models`, `/v1/audio/transcriptions` and
/// `/v1/moderations`. `/v1/messages` and the SIE-native routes are not.
pub fn is_openai_path(path: &str) -> bool {
    OPENAI_COMPATIBLE_PATHS.iter().any(|base| {
        path.strip_prefix(base)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
    })
}

/// The ``error.type`` an outer refusal on an OpenAI-compatible route gets.
fn openai_type_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => openai_type::AUTHENTICATION,
        403 => openai_type::PERMISSION,
        429 => openai_type::RATE_LIMIT,
        400..=499 => openai_type::INVALID_REQUEST,
        _ => openai_type::SERVER_ERROR,
    }
}

fn code_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => code::UNAUTHORIZED,
        403 => code::FORBIDDEN,
        404 => code::NOT_FOUND,
        413 => code::PAYLOAD_TOO_LARGE,
        429 => code::RATE_LIMIT,
        400..=499 => code::INVALID_REQUEST,
        _ => code::INTERNAL_ERROR,
    }
}

/// The OpenAI envelope for a ``{"detail": ...}`` error body, or `None` when
/// the body is not that envelope (already OpenAI-shaped, or anything else).
fn openai_error_from_detail_body(status: StatusCode, body: &Value) -> Option<Value> {
    let body = body.as_object()?;
    if body.get("error").is_some_and(Value::is_object) {
        return None;
    }
    let mut error = Map::new();
    match body.get("detail")? {
        Value::String(message) => {
            error.insert("message".to_string(), json!(message));
            error.insert("type".to_string(), json!(openai_type_for_status(status)));
            error.insert("param".to_string(), Value::Null);
            error.insert("code".to_string(), json!(code_for_status(status)));
        }
        Value::Object(detail) => {
            let message = detail
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| status.canonical_reason())
                .unwrap_or_default();
            let code = detail
                .get("code")
                .and_then(Value::as_str)
                .unwrap_or_else(|| code_for_status(status));
            let param = detail
                .get("param")
                .filter(|param| param.is_string())
                .cloned()
                .unwrap_or(Value::Null);
            error.insert("message".to_string(), json!(message));
            error.insert("type".to_string(), json!(openai_type_for_status(status)));
            error.insert("param".to_string(), param);
            error.insert("code".to_string(), json!(code));
            for (key, value) in detail {
                if !matches!(key.as_str(), "message" | "type" | "param" | "code") {
                    error.insert(key.clone(), value.clone());
                }
            }
        }
        _ => return None,
    }
    Some(json!({ "error": error }))
}

fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(|mime| mime.trim().to_ascii_lowercase())
        .is_some_and(|mime| mime == "application/json" || mime.ends_with("+json"))
}

/// One answer in the OpenAI envelope, rewritten as [`openai_error_envelope`]
/// rewrites the answers on its routes. A composition that serves further
/// OpenAI-compatible routes applies it to theirs. Successes pass through
/// untouched.
pub async fn openai_error_response(response: Response) -> Response {
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    let fits = response
        .body()
        .size_hint()
        .upper()
        .is_some_and(|len| len <= MAX_ERROR_BODY as u64);
    if !fits || !is_json(response.headers()) {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let Ok(bytes) = to_bytes(body, MAX_ERROR_BODY).await else {
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::empty());
    };
    let rendered = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|body| openai_error_from_detail_body(parts.status, &body))
        .and_then(|envelope| serde_json::to_vec(&envelope).ok());
    let Some(rendered) = rendered else {
        return Response::from_parts(parts, Body::from(bytes));
    };
    parts
        .headers
        .insert(header::CONTENT_LENGTH, HeaderValue::from(rendered.len()));
    Response::from_parts(parts, Body::from(rendered))
}

/// Middleware that gives every error answer on the OpenAI-compatible routes
/// ([`is_openai_path`]) the OpenAI envelope.
///
/// Compositions that add admission layers around the gateway's routes apply
/// it outermost (`axum::middleware::from_fn(openai_error_envelope)`), so
/// their refusals take the same shape. A 4xx or 5xx JSON answer whose body is
/// ``{"detail": {...}}`` or ``{"detail": "..."}`` becomes
/// ``{"error": {"message", "type", "param", "code", ...}}``:
///
/// - `message` is `detail.message`, or `detail` when it is a string.
/// - `code` is `detail.code` verbatim, so clients keep branching on SIE
///   codes such as `UNAUTHORIZED`. A string `detail` gets the SIE code of its
///   status: `UNAUTHORIZED` (401), `FORBIDDEN` (403), `NOT_FOUND` (404),
///   `PAYLOAD_TOO_LARGE` (413), `RATE_LIMIT` (429), `INVALID_REQUEST` (other
///   4xx), `INTERNAL_SERVER_ERROR` (5xx).
/// - `type` follows the status: `authentication_error` (401),
///   `permission_error` (403), `rate_limit_error` (429),
///   `invalid_request_error` (other 4xx), `server_error` (5xx).
/// - `param` is `detail.param` when it is a string, otherwise `null`.
/// - Every other `detail` field, such as `key_region`, is kept in `error`.
///
/// The status, the response extensions and every header are kept, with
/// `Content-Length` set to the new body. Answers that are already in the
/// OpenAI envelope, are not JSON, or are not errors pass through untouched,
/// and so does every answer on another path: `/v1/messages` belongs to
/// `anthropic_error_envelope`, and the SIE-native routes keep `detail`.
pub async fn openai_error_envelope(request: Request, next: Next) -> Response {
    if !is_openai_path(request.uri().path()) {
        return next.run(request).await;
    }
    openai_error_response(next.run(request).await).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_openai_error_shape_with_param() {
        let body = json_openai_error(
            "tools are not supported",
            openai_type::INVALID_REQUEST,
            Some("tools"),
            openai_code::UNSUPPORTED_FIELD,
        );
        let err = body.get("error").expect("error envelope");
        assert_eq!(err["message"], "tools are not supported");
        assert_eq!(err["type"], "invalid_request_error");
        assert_eq!(err["param"], "tools");
        assert_eq!(err["code"], "unsupported_field");
    }

    #[test]
    fn test_json_openai_error_shape_without_param() {
        let body = json_openai_error(
            "Result channel closed",
            openai_type::SERVER_ERROR,
            None,
            openai_code::TRANSPORT_FAILURE,
        );
        let err = body.get("error").expect("error envelope");
        assert_eq!(err["message"], "Result channel closed");
        assert!(err["param"].is_null(), "param must serialise as JSON null");
        assert_eq!(err["code"], "transport_failure");
    }

    #[test]
    fn test_openai_error_from_detail_code_maps_each_row() {
        // (sie code, expected type, expected code)
        let cases = [
            (
                code::INVALID_REQUEST,
                "invalid_request_error",
                "invalid_request",
            ),
            ("INVALID_INPUT", "invalid_request_error", "invalid_request"),
            ("INPUT_TOO_LONG", "invalid_request_error", "invalid_request"),
            (
                code::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "payload_too_large",
            ),
            (
                code::MODEL_NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
            ),
            (code::QUEUE_UNAVAILABLE, "server_error", "transport_failure"),
            (
                code::POOL_CAPACITY_UNAVAILABLE,
                "server_error",
                "transport_failure",
            ),
            (code::PROVISIONING, "server_error", "provisioning"),
            (code::GATEWAY_TIMEOUT, "server_error", "overall_timeout"),
            ("RATE_LIMIT", "rate_limit_error", "rate_limit_exceeded"),
            (code::INTERNAL_ERROR, "server_error", "transport_failure"),
            // Unknown / unmapped → generic server_error, never leaking the code.
            ("SOMETHING_UNMAPPED", "server_error", "transport_failure"),
        ];
        for (sie_code, want_type, want_code) in cases {
            let (t, c) = openai_error_from_detail_code(sie_code);
            assert_eq!(t, want_type, "type for {sie_code}");
            assert_eq!(c, want_code, "code for {sie_code}");
        }
    }

    #[test]
    fn test_embeddings_error_builds_openai_envelope_with_param() {
        let body = embeddings_error(
            code::INVALID_REQUEST,
            Some("model"),
            "field \"model\" is required",
        );
        let err = body.get("error").expect("error envelope");
        assert_eq!(err["message"], "field \"model\" is required");
        assert_eq!(err["type"], "invalid_request_error");
        assert_eq!(err["code"], "invalid_request");
        assert_eq!(err["param"], "model");
        // No legacy detail shape on the OpenAI surface.
        assert!(body.get("detail").is_none());
    }

    #[test]
    fn test_embeddings_error_inner_failure_translates_to_server_error() {
        let body = embeddings_error(code::QUEUE_UNAVAILABLE, None, "queue unavailable");
        let err = body.get("error").expect("error envelope");
        assert_eq!(err["type"], "server_error");
        assert_eq!(err["code"], "transport_failure");
        assert!(err["param"].is_null());
    }

    mod openai_envelope {
        use super::*;
        use axum::http::HeaderName;
        use axum::response::IntoResponse;
        use axum::Router;
        use tower::ServiceExt;

        const OPENAI_PATHS: [&str; 11] = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/responses/resp_1",
            "/v1/embeddings",
            "/v1/models",
            "/v1/models/BAAI/bge-m3",
            "/v1/audio/transcriptions",
            "/v1/moderations",
            "/v1/chat/completions/chatcmpl_1",
            "/v1/completions/",
        ];

        const OTHER_PATHS: [&str; 13] = [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/encode/BAAI/bge-m3",
            "/v1/score/m",
            "/v1/extract/m",
            "/v1/generate/m",
            "/v1/rerank",
            "/v1/systemone",
            "/v1/decisions",
            "/v1/pools",
            "/v1/modelsx",
            "/v1/completionsx",
            "/health",
        ];

        type MakeResponse = fn() -> Response;

        async fn through(path: &str, make: MakeResponse) -> Response {
            serve(path, make, true).await
        }

        async fn serve(path: &str, make: MakeResponse, envelope: bool) -> Response {
            let router = Router::new().fallback(move || async move { make() });
            let router = if envelope {
                router.layer(axum::middleware::from_fn(openai_error_envelope))
            } else {
                router
            };
            router
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .body(Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response")
        }

        async fn bytes(response: Response) -> Vec<u8> {
            to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("body")
                .to_vec()
        }

        fn missing_token() -> Response {
            let mut response = (
                StatusCode::UNAUTHORIZED,
                axum::Json(json_detail(
                    code::UNAUTHORIZED,
                    "Missing Authorization or x-api-key header",
                )),
            )
                .into_response();
            response
                .headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            response
        }

        fn region_refusal() -> Response {
            let mut extra = Map::new();
            extra.insert("key_region".to_string(), json!("eu"));
            extra.insert(
                "region_endpoint".to_string(),
                json!("https://eu.example.test"),
            );
            extra.insert("epoch".to_string(), json!(7));
            extra.insert("param".to_string(), json!("model"));
            let mut response = (
                StatusCode::FORBIDDEN,
                [
                    (header::RETRY_AFTER, "30"),
                    (HeaderName::from_static("x-sie-error-code"), "KEY_REGION"),
                    (HeaderName::from_static("x-sie-ratelimit-rps"), "5"),
                    (HeaderName::from_static("x-sie-ratelimit-scope"), "key"),
                ],
                axum::Json(json_detail_merge(
                    "KEY_REGION",
                    "this key belongs to another region",
                    extra,
                )),
            )
                .into_response();
            response
                .extensions_mut()
                .insert(GatewayOwnedFault::new("test"));
            response
        }

        fn string_detail() -> Response {
            (
                StatusCode::NOT_FOUND,
                axum::Json(json!({"detail": "Not Found"})),
            )
                .into_response()
        }

        #[test]
        fn openai_paths_cover_the_routes_and_the_paths_below_them() {
            for path in OPENAI_PATHS {
                assert!(is_openai_path(path), "{path}");
            }
            for path in OTHER_PATHS {
                assert!(!is_openai_path(path), "{path}");
            }
        }

        #[tokio::test]
        async fn the_anthropic_and_openai_layers_never_both_apply() {
            for (path, anthropic) in [
                ("/v1/messages", true),
                ("/v1/messages/count_tokens", true),
                ("/v1/chat/completions", false),
                ("/v1/responses", false),
                ("/v1/models/m", false),
            ] {
                let response = Router::new()
                    .fallback(|| async { missing_token() })
                    .layer(axum::middleware::from_fn(
                        crate::handlers::messages::anthropic_error_envelope,
                    ))
                    .layer(axum::middleware::from_fn(openai_error_envelope))
                    .oneshot(
                        Request::builder()
                            .uri(path)
                            .body(Body::empty())
                            .expect("request"),
                    )
                    .await
                    .expect("response");
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
                let body: Value = serde_json::from_slice(&bytes(response).await).expect("json");
                assert_eq!(body["error"]["type"], "authentication_error", "{path}");
                assert_eq!(
                    body.get("type") == Some(&json!("error")),
                    anthropic,
                    "{path}"
                );
                assert_eq!(body["error"].get("code").is_some(), !anthropic, "{path}");
            }
        }

        #[test]
        fn the_type_follows_the_status() {
            for (status, want_type, want_code) in [
                (400, "invalid_request_error", "INVALID_REQUEST"),
                (401, "authentication_error", "UNAUTHORIZED"),
                (402, "invalid_request_error", "INVALID_REQUEST"),
                (403, "permission_error", "FORBIDDEN"),
                (404, "invalid_request_error", "NOT_FOUND"),
                (405, "invalid_request_error", "INVALID_REQUEST"),
                (413, "invalid_request_error", "PAYLOAD_TOO_LARGE"),
                (429, "rate_limit_error", "RATE_LIMIT"),
                (500, "server_error", "INTERNAL_SERVER_ERROR"),
                (502, "server_error", "INTERNAL_SERVER_ERROR"),
                (503, "server_error", "INTERNAL_SERVER_ERROR"),
                (504, "server_error", "INTERNAL_SERVER_ERROR"),
            ] {
                let status = StatusCode::from_u16(status).expect("status");
                let body = openai_error_from_detail_body(status, &json!({"detail": "refused"}))
                    .expect("envelope");
                assert_eq!(body["error"]["type"], want_type, "{status}");
                assert_eq!(body["error"]["code"], want_code, "{status}");
                let body = openai_error_from_detail_body(
                    status,
                    &json!({"detail": {"code": "SOME_SIE_CODE", "message": "refused"}}),
                )
                .expect("envelope");
                assert_eq!(body["error"]["type"], want_type, "{status}");
                assert_eq!(body["error"]["code"], "SOME_SIE_CODE", "{status}");
            }
        }

        #[tokio::test]
        async fn a_detail_refusal_becomes_the_openai_envelope_on_every_openai_path() {
            for path in OPENAI_PATHS {
                let response = through(path, missing_token).await;
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{path}");
                assert_eq!(response.headers()["www-authenticate"], "Bearer", "{path}");
                assert_eq!(
                    response.headers()["content-type"],
                    "application/json",
                    "{path}"
                );
                let length = response.headers()[header::CONTENT_LENGTH].clone();
                let body = bytes(response).await;
                assert_eq!(length, body.len().to_string().as_str(), "{path}");
                let body: Value = serde_json::from_slice(&body).expect("json");
                assert_eq!(
                    body,
                    json!({"error": {
                        "message": "Missing Authorization or x-api-key header",
                        "type": "authentication_error",
                        "param": null,
                        "code": "UNAUTHORIZED",
                    }}),
                    "{path}"
                );
            }
        }

        #[tokio::test]
        async fn one_answer_is_rewritten_as_the_layer_rewrites_it() {
            let response = openai_error_response(missing_token()).await;
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(response.headers()["www-authenticate"], "Bearer");
            let body: Value = serde_json::from_slice(&bytes(response).await).expect("json");
            assert_eq!(body["error"]["type"], "authentication_error");
            assert_eq!(body["error"]["code"], "UNAUTHORIZED");

            let success = (StatusCode::OK, axum::Json(json!({"detail": "kept"}))).into_response();
            let response = openai_error_response(success).await;
            assert_eq!(response.status(), StatusCode::OK);
            let body: Value = serde_json::from_slice(&bytes(response).await).expect("json");
            assert_eq!(body, json!({"detail": "kept"}));
        }

        #[tokio::test]
        async fn a_string_detail_gets_the_code_of_its_status() {
            let response = through("/v1/responses/resp_1", string_detail).await;
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let body: Value = serde_json::from_slice(&bytes(response).await).expect("json");
            assert_eq!(
                body,
                json!({"error": {
                    "message": "Not Found",
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "NOT_FOUND",
                }})
            );
        }

        #[tokio::test]
        async fn detail_fields_headers_and_extensions_survive() {
            let response = through("/v1/chat/completions", region_refusal).await;
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            let headers = response.headers();
            assert_eq!(headers["retry-after"], "30");
            assert_eq!(headers["x-sie-error-code"], "KEY_REGION");
            assert_eq!(headers["x-sie-ratelimit-rps"], "5");
            assert_eq!(headers["x-sie-ratelimit-scope"], "key");
            assert_eq!(
                response.extensions().get::<GatewayOwnedFault>(),
                Some(&GatewayOwnedFault::new("test"))
            );
            let body: Value = serde_json::from_slice(&bytes(response).await).expect("json");
            assert_eq!(
                body,
                json!({"error": {
                    "message": "this key belongs to another region",
                    "type": "permission_error",
                    "param": "model",
                    "code": "KEY_REGION",
                    "key_region": "eu",
                    "region_endpoint": "https://eu.example.test",
                    "epoch": 7,
                }})
            );
        }

        #[tokio::test]
        async fn other_answers_pass_through_untouched() {
            fn openai_body() -> Response {
                (
                    StatusCode::BAD_REQUEST,
                    [(header::CONTENT_TYPE, "application/json")],
                    r#"{"error": {"code": "unsupported_field", "message": "no", "type": "invalid_request_error", "param": "tools"}, "detail": {"code": "X"}}"#,
                )
                    .into_response()
            }
            fn plain_text() -> Response {
                (
                    StatusCode::UNAUTHORIZED,
                    [(header::CONTENT_TYPE, "text/plain")],
                    r#"{"detail": {"code": "UNAUTHORIZED", "message": "no"}}"#,
                )
                    .into_response()
            }
            fn success() -> Response {
                (
                    StatusCode::OK,
                    axum::Json(json!({"detail": {"code": "OK", "message": "fine"}})),
                )
                    .into_response()
            }
            fn validation_list() -> Response {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    axum::Json(json!({"detail": [{"loc": ["body"], "msg": "required"}]})),
                )
                    .into_response()
            }
            fn empty() -> Response {
                StatusCode::METHOD_NOT_ALLOWED.into_response()
            }
            fn not_json() -> Response {
                (
                    StatusCode::BAD_GATEWAY,
                    [(header::CONTENT_TYPE, "application/json")],
                    "{\"detail\": ",
                )
                    .into_response()
            }

            let cases: [(&str, MakeResponse); 6] = [
                ("openai", openai_body),
                ("plain text", plain_text),
                ("success", success),
                ("validation list", validation_list),
                ("empty", empty),
                ("not json", not_json),
            ];
            for (case, make) in cases {
                assert_untouched("/v1/chat/completions", make, case).await;
            }
        }

        async fn assert_untouched(path: &str, make: MakeResponse, case: &str) {
            let expected = serve(path, make, false).await;
            let response = through(path, make).await;
            assert_eq!(response.status(), expected.status(), "{case} {path}");
            assert_eq!(response.headers(), expected.headers(), "{case} {path}");
            assert_eq!(
                bytes(response).await,
                bytes(expected).await,
                "{case} {path}"
            );
        }

        #[tokio::test]
        async fn other_paths_keep_the_detail_envelope() {
            for path in OTHER_PATHS {
                assert_untouched(path, missing_token, "missing token").await;
            }
        }
    }
}
