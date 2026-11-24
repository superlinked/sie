//! Errors on `/v1/responses`, in the OpenAI envelope
//! `{"error":{"message","type","param","code"}}` the chat route already uses.
//!
//! A translation refusal names the field as the caller sent it. A chat error
//! keeps its status, headers, extensions and body; only `error.param` is
//! renamed from the chat field to the Responses field it was translated from.

use axum::body::{to_bytes, Body};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;

use crate::http_error::{json_openai_error, openai_code, openai_type};

const MAX_ERROR_BODY: usize = 1024 * 1024;

/// An error decided in Responses terms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResponsesError {
    pub status: StatusCode,
    pub error_type: &'static str,
    pub code: &'static str,
    pub param: Option<String>,
    pub message: String,
}

impl ResponsesError {
    fn refusal(code: &'static str, param: String, reason: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: openai_type::INVALID_REQUEST,
            code,
            message: format!("{param}: {reason}"),
            param: Some(param),
        }
    }

    pub(crate) fn invalid(param: impl Into<String>, reason: &str) -> Self {
        Self::refusal(openai_code::INVALID_REQUEST, param.into(), reason)
    }

    pub(crate) fn unsupported(param: impl Into<String>, reason: &str) -> Self {
        Self::refusal(openai_code::UNSUPPORTED_FIELD, param.into(), reason)
    }

    pub(crate) fn extra(param: impl Into<String>) -> Self {
        Self::unsupported(param, "Extra inputs are not permitted")
    }

    pub(crate) fn required(param: impl Into<String>) -> Self {
        Self::invalid(param, "Field required")
    }

    /// A refusal of the request as a whole, before any field is read.
    pub(crate) fn request(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            error_type: openai_type::INVALID_REQUEST,
            code: openai_code::INVALID_REQUEST,
            param: None,
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: openai_type::SERVER_ERROR,
            code: openai_code::MALFORMED_WORKER_RESPONSE,
            param: None,
            message: message.into(),
        }
    }

    pub(crate) fn into_response(self) -> Response {
        (
            self.status,
            Json(json_openai_error(
                self.message,
                self.error_type,
                self.param.as_deref(),
                self.code,
            )),
        )
            .into_response()
    }
}

/// The Responses name of a field a chat error names.
pub(crate) fn responses_param(param: &str) -> String {
    match param {
        "max_completion_tokens" => return "max_output_tokens".to_string(),
        "reasoning_effort" => return "reasoning.effort".to_string(),
        "response_format" => return "text.format".to_string(),
        _ => {}
    }
    if let Some(field) = param
        .strip_prefix("response_format.json_schema.")
        .filter(|field| !field.is_empty())
    {
        return format!("text.format.{field}");
    }
    if let Some((index, field)) = param
        .strip_prefix("tools[")
        .and_then(|rest| rest.split_once("].function."))
    {
        if !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()) && !field.is_empty()
        {
            return format!("tools[{index}].{field}");
        }
    }
    param.to_string()
}

/// A non-2xx chat answer as `/v1/responses` returns it.
pub(crate) async fn responses_error_response(response: Response) -> Response {
    let (mut parts, body) = response.into_parts();
    let bytes = to_bytes(body, MAX_ERROR_BODY).await.unwrap_or_default();
    let renamed = serde_json::from_slice::<Value>(&bytes)
        .ok()
        .and_then(|mut answer| {
            let param = answer.get_mut("error")?.get_mut("param")?;
            let chat_param = param.as_str()?;
            let renamed = responses_param(chat_param);
            if renamed == chat_param {
                return None;
            }
            *param = Value::String(renamed);
            serde_json::to_vec(&answer).ok()
        });
    parts.headers.remove(header::CONTENT_LENGTH);
    let body = match renamed {
        Some(renamed) => Body::from(renamed),
        None => Body::from(bytes),
    };
    Response::from_parts(parts, body)
}
