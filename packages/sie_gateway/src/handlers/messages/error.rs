//! The Anthropic error envelope for `/v1/messages`.
//!
//! Every non-2xx answer on the Messages paths leaves the gateway as
//! `{"type":"error","error":{"type","message"},"request_id"}` with the status
//! the inner route chose. The error type follows the status. `Retry-After`
//! is kept and capped at [`MAX_RETRY_AFTER_S`], the longest wait Anthropic
//! clients keep retrying through.

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use crate::handlers::proxy::is_openai_compat_forwarded_header;

pub(crate) const MAX_RETRY_AFTER_S: u64 = 60;

const MAX_ERROR_BODY: usize = 1024 * 1024;

/// An error decided in Anthropic terms: the status and the client-facing
/// message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MessagesError {
    pub status: StatusCode,
    pub message: String,
}

impl MessagesError {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    pub(crate) fn into_response(self) -> Response {
        let mut response = (
            self.status,
            axum::Json(envelope(
                error_type_for_status(self.status),
                &self.message,
                None,
            )),
        )
            .into_response();
        response.headers_mut().insert(
            HeaderName::from_static("x-sie-error-code"),
            HeaderValue::from_static(sie_code_for_status(self.status)),
        );
        response
    }
}

/// What the Messages handler knows about the request that the chat route's
/// error cannot say in Anthropic terms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ErrorContext {
    /// The Anthropic path of the first image block, named when the model
    /// refuses image input.
    pub first_image: Option<String>,
}

/// The Anthropic `error.type` for an HTTP status.
pub(crate) fn error_type_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "authentication_error",
        402 => "billing_error",
        403 => "permission_error",
        404 => "not_found_error",
        408 | 504 => "timeout_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    }
}

fn sie_code_for_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        413 => "PAYLOAD_TOO_LARGE",
        400..=499 => "INVALID_REQUEST",
        _ => "INTERNAL_SERVER_ERROR",
    }
}

fn default_message(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid request",
        401 => "invalid or missing credential",
        402 => "payment required",
        403 => "permission denied",
        404 => "not found",
        405 => "method not allowed",
        408 | 504 => "request timed out",
        413 => "request too large",
        429 => "rate limited",
        503 | 529 => "overloaded",
        _ => "internal server error",
    }
}

pub(crate) fn envelope(error_type: &str, message: &str, request_id: Option<&str>) -> Value {
    json!({
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": request_id,
    })
}

/// The `(type, message, SIE code)` an inner error answer becomes.
///
/// `body` is the inner answer as JSON: the OpenAI `{error}` envelope, the
/// SIE `{detail}` envelope, or anything else. Messages are rewritten only
/// where Anthropic clients match the wording: the output cap, a prompt that
/// does not fit, and media the model cannot read.
pub(crate) fn translate_error_body(
    status: StatusCode,
    body: &Value,
    context: &ErrorContext,
) -> (&'static str, String, Option<String>) {
    let error_type = error_type_for_status(status);
    let inner = body
        .get("error")
        .filter(|value| value.is_object())
        .or_else(|| body.get("detail").filter(|value| value.is_object()));
    let code = inner
        .and_then(|value| value.get("code"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let inner_type = inner
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str);
    let param = inner
        .and_then(|value| value.get("param"))
        .and_then(Value::as_str);
    let message = inner
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .or_else(|| body.get("detail").and_then(Value::as_str))
        .or_else(|| body.get("message").and_then(Value::as_str))
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| default_message(status))
        .to_string();
    if status != StatusCode::BAD_REQUEST {
        return (error_type, message, code);
    }
    if let Some((requested, cap)) = output_cap(&message) {
        return (
            error_type,
            format!(
                "max_tokens: {requested} > {cap}, which is the maximum allowed number of output tokens for this model"
            ),
            code,
        );
    }
    let too_long = matches!(code.as_deref(), Some("context_exceeded" | "INPUT_TOO_LONG"))
        || inner_type == Some("context_length_exceeded");
    if too_long {
        let message = match prompt_size(&message) {
            Some((tokens, limit)) => {
                format!("prompt is too long: {tokens} tokens > {limit} maximum")
            }
            None => "prompt is too long".to_string(),
        };
        return (error_type, message, code);
    }
    if message.contains("does not support image input") {
        if let Some(path) = context.first_image.as_deref() {
            return (error_type, format!("{path}: {message}"), code);
        }
    }
    if message.contains("does not support tool calling") {
        return (error_type, format!("tools: {message}"), code);
    }
    let message = message.replace("max_completion_tokens", "max_tokens");
    match param.map(anthropic_param) {
        Some(path) if !message.starts_with(&format!("{path}:")) => {
            (error_type, format!("{path}: {message}"), code)
        }
        _ => (error_type, message, code),
    }
}

fn anthropic_param(param: &str) -> String {
    if param == "max_completion_tokens" {
        return "max_tokens".to_string();
    }
    if param == "stop" {
        return "stop_sequences".to_string();
    }
    if let Some(rest) = param.strip_prefix("tools[") {
        if let Some((index, field)) = rest.split_once("].function.") {
            let field = if field == "parameters" {
                "input_schema"
            } else {
                field
            };
            return format!("tools.{index}.{field}");
        }
    }
    param.to_string()
}

fn leading_number(text: &str) -> Option<(u64, &str)> {
    let end = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    if end == 0 {
        return None;
    }
    Some((text[..end].parse().ok()?, &text[end..]))
}

/// `(requested, cap)` from the chat route's or the worker's output-cap refusal.
fn output_cap(message: &str) -> Option<(u64, u64)> {
    for prefix in [
        "max_completion_tokens (",
        "max_tokens (",
        "max_new_tokens (",
    ] {
        let Some(rest) = message.strip_prefix(prefix) else {
            continue;
        };
        let (requested, rest) = leading_number(rest)?;
        let rest = rest
            .strip_prefix(") exceeds model max_output_tokens (")
            .or_else(|| rest.strip_prefix(") exceeds model cap ("))?;
        let (cap, rest) = leading_number(rest)?;
        return (rest == ")").then_some((requested, cap));
    }
    None
}

/// `(tokens, limit)` from a worker's `... = T exceeds context_length (C) ...`
/// or `prompt_tokens (T) exceeds context_length (C) ...`.
fn prompt_size(message: &str) -> Option<(u64, u64)> {
    let (before, after) = message.split_once(" exceeds context_length (")?;
    let (limit, _) = leading_number(after)?;
    let tokens = match before.rsplit_once("= ") {
        Some((_, total)) => leading_number(total)?.0,
        None => {
            let open = before.strip_prefix("prompt_tokens (")?;
            leading_number(open)?.0
        }
    };
    Some((tokens, limit))
}

fn clamp_retry_after(headers: &mut HeaderMap) {
    let Some(value) = headers.get(header::RETRY_AFTER) else {
        return;
    };
    let clamped = value
        .to_str()
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(|seconds| seconds.clamp(1, MAX_RETRY_AFTER_S));
    match clamped {
        Some(seconds) => {
            headers.insert(header::RETRY_AFTER, HeaderValue::from(seconds));
            headers.insert(
                HeaderName::from_static("x-should-retry"),
                HeaderValue::from_static("true"),
            );
        }
        None => {
            headers.remove(header::RETRY_AFTER);
        }
    }
}

fn is_anthropic_envelope(body: &Value) -> bool {
    body.get("type").and_then(Value::as_str) == Some("error")
        && body
            .get("error")
            .and_then(|error| error.get("type"))
            .and_then(Value::as_str)
            .is_some()
}

/// Re-render a non-2xx answer in the Anthropic envelope, keeping the status,
/// the response extensions, `Retry-After` and the SIE headers.
pub(crate) async fn anthropic_error_response(
    response: Response,
    context: &ErrorContext,
) -> Response {
    let (mut parts, body) = response.into_parts();
    let bytes = to_bytes(body, MAX_ERROR_BODY).await.unwrap_or_default();
    let parsed: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let request_id = parts
        .headers
        .get("x-sie-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut headers = HeaderMap::new();
    for (name, value) in parts.headers.iter() {
        let name_str = name.as_str();
        if is_openai_compat_forwarded_header(name_str)
            || name_str.eq_ignore_ascii_case("x-sie-error-code")
            || name_str.eq_ignore_ascii_case("x-sie-error-version")
            || name_str.eq_ignore_ascii_case("retry-after")
            || name_str.eq_ignore_ascii_case("www-authenticate")
        {
            headers.append(name.clone(), value.clone());
        }
    }
    let body = if is_anthropic_envelope(&parsed) {
        bytes.to_vec()
    } else {
        let (error_type, message, code) = translate_error_body(parts.status, &parsed, context);
        if !headers.contains_key("x-sie-error-code") {
            if let Some(value) = code.and_then(|code| HeaderValue::from_str(&code).ok()) {
                headers.insert(HeaderName::from_static("x-sie-error-code"), value);
            }
        }
        serde_json::to_vec(&envelope(error_type, &message, request_id.as_deref()))
            .unwrap_or_default()
    };
    clamp_retry_after(&mut headers);
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if let Some(value) = request_id.and_then(|id| HeaderValue::from_str(&id).ok()) {
        headers.insert(HeaderName::from_static("request-id"), value);
    }
    parts.headers = headers;
    Response::from_parts(parts, Body::from(body))
}

pub(crate) fn is_messages_path(path: &str) -> bool {
    path == "/v1/messages" || path.starts_with("/v1/messages/")
}

/// Middleware that gives every error answer on `/v1/messages` and the paths
/// below it the Anthropic envelope.
///
/// Compositions that add admission layers around the gateway's routes apply
/// it outermost (`axum::middleware::from_fn(anthropic_error_envelope)`), so
/// their refusals take the same shape. Already-Anthropic bodies pass through
/// with `Retry-After` capped.
pub async fn anthropic_error_envelope(request: Request, next: Next) -> Response {
    let applies = is_messages_path(request.uri().path());
    let response = next.run(request).await;
    let status = response.status();
    if !applies || !(status.is_client_error() || status.is_server_error()) {
        return response;
    }
    anthropic_error_response(response, &ErrorContext::default()).await
}
