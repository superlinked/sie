//! `POST /v1/responses`: the OpenAI Responses API as a translation onto
//! `/v1/chat/completions`.
//!
//! The request is validated and translated into a chat completions body
//! ([`request`]), the gateway's own chat handler serves it in process on the
//! same task, and its answer is translated back: a chat completion into a
//! Response ([`response`]), chat SSE chunks into Responses events
//! ([`stream`]), and a chat error into the same OpenAI error with the field
//! renamed ([`error`]). Every chat gate, limit, timeout and metering point
//! therefore applies unchanged. Responses are not stored, so the
//! `/v1/responses/{id}` routes are not served.

mod error;
mod request;
mod response;
pub mod schema;
mod stream;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

use crate::handlers::proxy::{
    buffer_compat_response_body, chat_completions_dispatch_inputs, compat_request_body_limit,
    is_openai_compat_forwarded_header, is_openai_compat_inner_request_header, proxy_chat,
    request_body_error, CompatResponseBody,
};
use crate::http_error::GatewayOwnedFault;
use crate::queue::publisher;
use crate::server::AppState;

use self::error::{responses_error_response, ResponsesError};
use self::response::{chat_completion_to_response, ResponseShell};

fn request_id(response: &Response) -> String {
    response
        .headers()
        .get("x-sie-request-id")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "inference",
    description = "OpenAI Responses API. The request is translated into a \
                   /v1/chat/completions request, served by the chat route with all of its \
                   limits, and the answer is translated back. Buffered and streaming answers \
                   are supported. Function and custom tools are supported; hosted tools are \
                   refused. Responses are not stored: previous_response_id, conversation, \
                   store: true and background: true are refused, and the /v1/responses/{id} \
                   routes are not served.",
    request_body = crate::handlers::responses::schema::ResponsesRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override")
    ),
    responses(
        (status = 200, description = "Response, or a text/event-stream of Responses events when stream is true", content(
            (crate::handlers::responses::schema::ResponsesResponse = "application/json"),
            (String = "text/event-stream"),
        )),
        (status = 400, description = "Invalid or unsupported request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body is too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Worker emitted malformed response", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning in progress, queue unavailable, or model loading", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
pub async fn proxy_responses(State(state): State<Arc<AppState>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let limit = compat_request_body_limit("/v1/responses");
    let bytes = match to_bytes(body, limit).await {
        Ok(bytes) => bytes,
        Err(error) => {
            let (status, _, message) = request_body_error(&error, limit);
            return ResponsesError::request(status, message).into_response();
        }
    };
    let body: Value = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(error) => {
            return ResponsesError::request(
                StatusCode::BAD_REQUEST,
                format!("invalid JSON: {error}"),
            )
            .into_response()
        }
    };
    let translation = match request::responses_to_chat(&body) {
        Ok(translation) => translation,
        Err(error) => return error.into_response(),
    };
    let Ok(chat_bytes) = serde_json::to_vec(&translation.chat) else {
        return ResponsesError::internal("failed to build the chat request").into_response();
    };
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .version(parts.version);
    for (name, value) in parts.headers.iter() {
        if is_openai_compat_inner_request_header(name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    builder = builder.header(header::CONTENT_TYPE, "application/json");
    let Ok(mut inner) = builder.body(Body::from(chat_bytes)) else {
        return ResponsesError::internal("failed to build the chat request").into_response();
    };
    *inner.extensions_mut() = parts.extensions;

    let answer = proxy_chat(State(state), inner).await;
    if !answer.status().is_success() {
        return responses_error_response(answer).await;
    }
    let shell = ResponseShell::new(translation.echo, request_id(&answer), now());
    if translation.stream {
        let (mut parts, body) = answer.into_parts();
        parts.headers.remove(header::CONTENT_LENGTH);
        let machine = stream::StreamMachine::new(shell);
        return Response::from_parts(parts, stream::stream_body(body, machine));
    }
    buffered_response(answer, &shell).await
}

async fn buffered_response(answer: Response, shell: &ResponseShell) -> Response {
    let (parts, body) = answer.into_parts();
    let translated = match buffer_compat_response_body(body).await {
        CompatResponseBody::Buffered(bytes) => serde_json::from_slice::<Value>(&bytes)
            .map_err(|_| "the chat completion is not valid JSON".to_string())
            .and_then(|chat| chat_completion_to_response(&chat, shell))
            .map_err(|reason| ("translate_chat_response", reason)),
        CompatResponseBody::TooLarge { .. } | CompatResponseBody::Unreadable => Err((
            "read_chat_response",
            "failed to read the chat completion".to_string(),
        )),
    };
    let (mut response, fault) = match translated {
        Ok(translated) => (
            (StatusCode::OK, axum::Json(translated)).into_response(),
            None,
        ),
        Err((stage, reason)) => (
            ResponsesError::internal(reason).into_response(),
            Some(stage),
        ),
    };
    for (name, value) in parts.headers.iter() {
        if is_openai_compat_forwarded_header(name.as_str()) {
            response.headers_mut().insert(name.clone(), value.clone());
        }
    }
    *response.extensions_mut() = parts.extensions;
    match fault {
        Some(stage) => GatewayOwnedFault::new(stage).mark(response),
        None => response,
    }
}

/// The `(model, WorkParams)` a `/v1/responses` body dispatches with: the
/// translation composed with the chat seam, so an estimate or an admission
/// check prices and admits exactly what the chat route would run.
///
/// A translation refusal is the Responses error answer; a chat refusal is
/// the chat route's own.
///
/// `dead_code`-allowed because the `sie-gateway` binary compiles this module
/// tree independently of the library, and only a downstream composition
/// calls this seam.
#[allow(clippy::result_large_err, dead_code)]
pub fn responses_dispatch_inputs(
    body: &Value,
) -> Result<(String, publisher::WorkParams), Response> {
    let translation = request::responses_to_chat(body).map_err(ResponsesError::into_response)?;
    chat_completions_dispatch_inputs(&translation.chat)
}
