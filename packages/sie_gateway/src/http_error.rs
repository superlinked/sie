//! Canonical JSON error bodies (FastAPI-style ``{"detail": {"code", "message", ...}}``).
//! SDK-stable responses (``502 MODEL_LOAD_FAILED``, ``503`` retryable ``{"error":{...}}``)
//! intentionally stay on their existing shapes.
//!
//! The OpenAI-compatible surfaces introduce a parallel OpenAI-shaped error envelope
//! (see :mod:`openai_code` / :func:`json_openai_error`). Generation-compatible
//! endpoints build that envelope directly. ``/v1/embeddings`` calls the SIE-native
//! encode path internally and translates non-200 responses back into the OpenAI
//! envelope. Native encode/score/extract endpoints keep ``json_detail`` to preserve
//! existing SDK error-handling contracts.

use serde_json::{json, Map, Value};

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
}
