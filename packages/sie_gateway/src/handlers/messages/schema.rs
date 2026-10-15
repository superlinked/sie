//! OpenAPI shapes of the Anthropic Messages surface.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::ToSchema;

/// Anthropic Messages request, as accepted by ``POST /v1/messages``.
///
/// The request is translated into a ``/v1/chat/completions`` request and
/// served by the chat route, so every chat limit applies. Fields outside this
/// set fail with ``400 invalid_request_error`` and the message
/// ``<field>: Extra inputs are not permitted``. ``metadata`` and every
/// ``cache_control`` are accepted and have no effect: prefix caching is
/// automatic and ``metadata`` is neither logged nor forwarded.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MessagesRequest {
    /// The model id, used as on ``/v1/chat/completions`` (``model`` or
    /// ``model:profile``, or an operator alias).
    pub model: String,
    /// Output token limit, at least 1 and at most the serving profile's
    /// ``max_output_tokens``.
    pub max_tokens: u32,
    /// ``user`` and ``assistant`` turns. User content blocks: ``text``,
    /// base64 ``image`` and ``tool_result``. Assistant content blocks:
    /// ``text``, ``tool_use``, and ``thinking`` / ``redacted_thinking``,
    /// which are accepted and do not reach the model. ``role: "system"``
    /// turns are refused; system text belongs in ``system``. The last turn
    /// must be a user turn.
    pub messages: Value,
    /// A string or a list of ``text`` blocks, concatenated in order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    /// Custom tools (``name``, ``description``, ``input_schema``), at most
    /// 64, names matching ``^[a-zA-Z0-9_-]{1,64}$``. Anthropic-defined tool
    /// types are refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    /// ``auto``, ``any``, ``tool`` (with ``name``) or ``none``, with
    /// ``disable_parallel_tool_use``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    /// ``enabled`` and ``adaptive`` return the model's reasoning as
    /// ``thinking`` blocks; ``disabled``, ``display: "omitted"`` or an absent
    /// field do not. Whether the model reasons is a property of its serving
    /// profile, and ``budget_tokens`` is validated but does not bound
    /// reasoning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// In ``[0, 1]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// In ``(0, 1]``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// At least 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Server-sent events: ``message_start``, content block events,
    /// ``message_delta`` with the final usage, ``message_stop``, and ``ping``
    /// whenever no other event was sent for 10 seconds. A failure after the
    /// stream opened is an ``error`` event, and the stream then ends without
    /// ``message_stop``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Accepted and ignored (``user_id`` only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Token usage of a Message. ``input_tokens`` plus
/// ``cache_read_input_tokens`` equals the prompt tokens the chat route
/// reports; ``cache_creation_input_tokens`` is always 0.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MessagesUsage {
    pub input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    /// Includes reasoning tokens.
    pub output_tokens: u64,
}

/// Anthropic Message answered by ``POST /v1/messages`` without ``stream``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct MessagesResponse {
    /// ``msg_`` followed by the gateway request id.
    pub id: String,
    /// Always ``message``.
    #[serde(rename = "type")]
    pub kind: String,
    /// Always ``assistant``.
    pub role: String,
    pub model: String,
    /// ``thinking`` (when requested and produced), then ``text``, then one
    /// ``tool_use`` per tool call. ``thinking.signature`` is an opaque
    /// constant that SIE does not verify.
    pub content: Value,
    /// ``end_turn``, ``tool_use``, ``max_tokens`` or ``refusal``.
    pub stop_reason: String,
    /// Always ``null``.
    pub stop_sequence: Option<String>,
    pub usage: MessagesUsage,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnthropicErrorBody {
    /// Follows the status: ``invalid_request_error`` (400),
    /// ``authentication_error`` (401), ``billing_error`` (402),
    /// ``permission_error`` (403), ``not_found_error`` (404),
    /// ``request_too_large`` (413), ``rate_limit_error`` (429),
    /// ``api_error`` (500, 502), ``overloaded_error`` (503),
    /// ``timeout_error`` (504).
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

/// Anthropic error envelope used on ``/v1/messages``. The SIE error code is
/// in the ``X-SIE-Error-Code`` header. ``Retry-After`` is at most 60.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnthropicErrorEnvelope {
    /// Always ``error``.
    #[serde(rename = "type")]
    pub kind: String,
    pub error: AnthropicErrorBody,
    pub request_id: Option<String>,
}
