//! `POST /v1/messages`: the Anthropic Messages API as a translation onto
//! `/v1/chat/completions`.
//!
//! The request is validated and translated into a chat completions body
//! ([`request`]), the gateway's own chat handler serves it in process on the
//! same task, and its answer is translated back: a chat completion into a
//! Message ([`response`]), chat SSE chunks into Messages events ([`stream`]),
//! and every error into the Anthropic envelope ([`error`]). Every chat gate,
//! limit, timeout and metering point therefore applies unchanged.

mod error;
mod request;
mod response;
pub mod schema;
mod stream;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

pub use error::anthropic_error_envelope;

use crate::handlers::proxy::{
    buffer_compat_response_body, chat_completions_dispatch_inputs, is_length_limit_error,
    is_openai_compat_inner_request_header, proxy_chat, CompatResponseBody, MAX_JSON_BODY_BYTES,
};
use crate::http_error::GatewayOwnedFault;
use crate::queue::publisher;
use crate::server::AppState;

use self::error::{anthropic_error_response, ErrorContext, MessagesError};

/// The request body bound, the same as `/v1/chat/completions`.
const MAX_MESSAGES_BODY: usize = MAX_JSON_BODY_BYTES;

fn request_id(response: &Response) -> String {
    response
        .headers()
        .get("x-sie-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

fn stamp_request_id(response: &mut Response, request_id: &str) {
    if let Ok(value) = HeaderValue::from_str(request_id) {
        if !request_id.is_empty() {
            response
                .headers_mut()
                .insert(HeaderName::from_static("request-id"), value);
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/messages",
    tag = "inference",
    description = "Anthropic Messages API. The request is translated into a \
                   /v1/chat/completions request, served by the chat route with all of its \
                   limits, and the answer is translated back. Buffered and streaming answers \
                   are supported. POST /v1/messages/count_tokens is not served.",
    request_body = crate::handlers::messages::schema::MessagesRequest,
    params(
        ("anthropic-version" = Option<String>, Header, description = "Absent or 2023-06-01"),
        ("anthropic-beta" = Option<String>, Header, description = "Accepted and ignored"),
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override")
    ),
    responses(
        (status = 200, description = "Message, or a text/event-stream of Messages events when stream is true", body = crate::handlers::messages::schema::MessagesResponse),
        (status = 400, description = "Invalid or unsupported request", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 401, description = "Missing or invalid bearer token", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 413, description = "Request body is too large", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 500, description = "Worker emitted a malformed response", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 429, description = "Rate limited; retry after Retry-After", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 502, description = "The model failed to load", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 503, description = "Capacity is not ready; retry after Retry-After", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
        (status = 504, description = "Generation timed out", body = crate::handlers::messages::schema::AnthropicErrorEnvelope),
    )
)]
pub async fn proxy_messages(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let headers = request.headers().clone();
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(body, MAX_MESSAGES_BODY).await {
        Ok(bytes) => bytes,
        Err(error) if is_length_limit_error(&error) => {
            return MessagesError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("request body is larger than {MAX_MESSAGES_BODY} bytes"),
            )
            .into_response();
        }
        Err(_) => return MessagesError::invalid("failed to read the request body").into_response(),
    };
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(error) => {
            return MessagesError::invalid(format!("body: invalid JSON: {error}")).into_response()
        }
    };
    let version = match headers.get("anthropic-version") {
        None => None,
        Some(value) => Some(value.to_str().unwrap_or("")),
    };
    let translation = match request::messages_to_chat(&body, version) {
        Ok(translation) => translation,
        Err(error) => return error.into_response(),
    };
    let context = ErrorContext {
        first_image: translation.first_image.clone(),
    };
    let chat_bytes = match serde_json::to_vec(&translation.chat) {
        Ok(bytes) => bytes,
        Err(_) => {
            return MessagesError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build the chat request",
            )
            .into_response()
        }
    };
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .version(parts.version);
    for (name, value) in headers.iter() {
        if is_openai_compat_inner_request_header(name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    builder = builder.header(header::CONTENT_TYPE, "application/json");
    let mut inner = match builder.body(Body::from(chat_bytes)) {
        Ok(inner) => inner,
        Err(_) => {
            return MessagesError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build the chat request",
            )
            .into_response()
        }
    };
    *inner.extensions_mut() = parts.extensions;

    let answer = proxy_chat(State(state), inner).await;
    if !answer.status().is_success() {
        return anthropic_error_response(answer, &context).await;
    }
    let request_id = request_id(&answer);
    let message_id = format!("msg_{request_id}");
    if translation.stream {
        let (mut parts, body) = answer.into_parts();
        parts.headers.remove(header::CONTENT_LENGTH);
        let machine = stream::StreamMachine::new(message_id, translation.model);
        let mut response = Response::from_parts(
            parts,
            stream::stream_body(body, machine, stream::PING_INTERVAL),
        );
        stamp_request_id(&mut response, &request_id);
        return response;
    }
    buffered_message(answer, &translation.model, &request_id).await
}

fn translation_fault(stage: &'static str, message: &str) -> Response {
    GatewayOwnedFault::new(stage)
        .mark(MessagesError::new(StatusCode::INTERNAL_SERVER_ERROR, message).into_response())
}

async fn buffered_message(answer: Response, model: &str, request_id: &str) -> Response {
    let (parts, body) = answer.into_parts();
    let bytes = match buffer_compat_response_body(body).await {
        CompatResponseBody::Buffered(bytes) => bytes,
        CompatResponseBody::TooLarge { .. } | CompatResponseBody::Unreadable => {
            return translation_fault("read_chat_response", "failed to read the chat completion")
        }
    };
    let message = serde_json::from_slice::<Value>(&bytes)
        .map_err(|_| "the chat completion is not valid JSON".to_string())
        .and_then(|chat| response::chat_completion_to_message(&chat, model, request_id));
    let message = match message {
        Ok(message) => message,
        Err(reason) => return translation_fault("translate_chat_response", &reason),
    };
    let mut response = (StatusCode::OK, axum::Json(message)).into_response();
    for (name, value) in parts.headers.iter() {
        if crate::handlers::proxy::is_openai_compat_forwarded_header(name.as_str()) {
            response.headers_mut().insert(name.clone(), value.clone());
        }
    }
    *response.extensions_mut() = parts.extensions;
    stamp_request_id(&mut response, request_id);
    response
}

/// The `(model, WorkParams)` a `/v1/messages` body dispatches with: the
/// translation composed with the chat seam, so an estimate or an admission
/// check prices and admits exactly what the chat route would run.
///
/// A translation refusal is answered in the Anthropic envelope; a chat
/// refusal keeps the chat envelope, which [`anthropic_error_envelope`]
/// re-renders on the way out.
///
/// `dead_code`-allowed because the `sie-gateway` binary compiles this module
/// tree independently of the library, and only a downstream composition
/// calls this seam.
#[allow(clippy::result_large_err, dead_code)]
pub fn messages_dispatch_inputs(body: &Value) -> Result<(String, publisher::WorkParams), Response> {
    let translation =
        request::messages_to_chat(body, None).map_err(MessagesError::into_response)?;
    chat_completions_dispatch_inputs(&translation.chat)
}
