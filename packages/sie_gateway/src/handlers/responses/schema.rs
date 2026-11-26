//! OpenAPI shapes of the OpenAI Responses surface.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::openapi::schema::{ArrayBuilder, ObjectBuilder, OneOfBuilder, Schema, Type};
use utoipa::openapi::RefOr;
use utoipa::ToSchema;

fn input_schema() -> RefOr<Schema> {
    OneOfBuilder::new()
        .description(Some(
            "A string, sent as one user message, or a non-empty list of items: messages \
             (`role` `user`, `assistant`, `system` or `developer`, with string content or \
             `input_text`, `output_text` and `text` parts, and base64 `data:` `input_image` \
             parts in user messages), `function_call`, `custom_tool_call`, their `*_output` \
             items, and `reasoning` items, which do not reach the model. Item `id` and \
             `status` are ignored. Other item and part types are refused.",
        ))
        .item(ObjectBuilder::new().schema_type(Type::String))
        .item(
            ArrayBuilder::new()
                .items(ObjectBuilder::new().schema_type(Type::Object))
                .min_items(Some(1)),
        )
        .into()
}

fn null_only() -> RefOr<Schema> {
    ObjectBuilder::new()
        .schema_type(Type::Null)
        .description(Some("Accepted only `null`: responses are not stored."))
        .into()
}

/// OpenAI Responses request, as accepted by ``POST /v1/responses``.
///
/// The request is translated into a ``/v1/chat/completions`` request and
/// served by the chat route, so every chat limit and check applies, and a
/// chat refusal names the Responses field (``tools[i].name``,
/// ``max_output_tokens``, ``reasoning.effort``, ``text.format``). A field
/// sent as ``null`` is treated as absent.
///
/// Responses are not stored: ``store: true`` and a non-null
/// ``previous_response_id`` or ``conversation`` fail with ``400
/// unsupported_field`` and the message ``<field>: SIE does not store
/// responses; send the full conversation in input``. ``background: true``,
/// a ``truncation`` other than ``disabled`` and an ``include`` entry other
/// than ``reasoning.encrypted_content`` fail the same way. ``prompt``,
/// ``max_tool_calls``, ``top_logprobs``, ``stream_options`` and every other
/// field outside this set fail with ``<field>: Extra inputs are not
/// permitted``. ``client_metadata``, ``reasoning.summary``,
/// ``reasoning.generate_summary`` and ``text.verbosity`` are accepted and
/// have no effect.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponsesRequest {
    /// The model id, used as on ``/v1/chat/completions`` (``model`` or
    /// ``model:profile``, or an operator alias).
    #[schema(min_length = 1)]
    pub model: String,
    #[schema(schema_with = input_schema)]
    pub input: Value,
    /// Prepended to the leading ``system`` and ``developer`` messages as
    /// one system message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Output token limit, at least 1 and at most the serving profile's
    /// ``max_output_tokens``. Absent: the chat route's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(minimum = 1)]
    pub max_output_tokens: Option<u32>,
    /// Checked as on ``/v1/chat/completions``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Checked as on ``/v1/chat/completions``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Checked as on ``/v1/chat/completions``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Server-sent events: ``response.created``, ``response.in_progress``,
    /// output item, content part, text and call-argument events, then
    /// ``response.completed`` or ``response.incomplete`` with the full
    /// Response. Every event carries ``sequence_number``. A failure after
    /// the stream opened is ``response.failed``, and the stream then ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// ``function`` tools (``name``, ``description``, ``parameters``,
    /// ``strict``, which the chat route enforces) and ``custom`` tools
    /// (``name``, ``description``, ``format``, which is not enforced), at most 64,
    /// names matching ``^[a-zA-Z0-9_-]{1,64}$``. Hosted tool types are
    /// refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    /// ``auto``, ``none``, ``required``, or ``{"type": "function" |
    /// "custom", "name"}`` naming a declared tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// ``effort`` becomes the chat ``reasoning_effort``. Reasoning is not
    /// returned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
    /// ``format`` ``text``, ``json_object`` or ``json_schema`` (``name``,
    /// ``schema``, ``strict``, ``description``) becomes the chat
    /// ``response_format``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<Value>,
    /// Passed to the chat route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    /// Passed to the chat route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Passed to the chat route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Passed to the chat route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Passed to the chat route.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// Accepted only ``false``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Accepted only ``false``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Accepted only ``disabled``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    /// Accepted only with ``reasoning.encrypted_content`` entries, which
    /// have no effect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    /// Accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(schema_with = null_only)]
    pub previous_response_id: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(schema_with = null_only)]
    pub conversation: Option<Value>,
}

/// Token usage of a Response, from the chat route's usage.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    /// ``cached_tokens``: prompt tokens served from the prefix cache.
    pub input_tokens_details: Value,
    /// Includes reasoning tokens.
    pub output_tokens: u64,
    /// ``reasoning_tokens``.
    pub output_tokens_details: Value,
    pub total_tokens: u64,
}

/// Response answered by ``POST /v1/responses`` without ``stream``.
#[derive(Debug, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ResponsesResponse {
    /// ``resp-`` followed by the gateway request id.
    pub id: String,
    /// Always ``response``.
    pub object: String,
    pub created_at: u64,
    /// The model the request named.
    pub model: String,
    /// ``completed`` or ``incomplete``.
    pub status: String,
    /// ``{"reason": "max_output_tokens" | "content_filter"}`` when
    /// ``incomplete``.
    pub incomplete_details: Option<Value>,
    /// Always ``null``.
    pub error: Option<Value>,
    /// Always ``false``.
    pub store: bool,
    /// As the request sent it, ``true`` when absent.
    pub parallel_tool_calls: Value,
    /// As the request sent it, ``auto`` when absent.
    pub tool_choice: Value,
    /// A ``message`` item with one ``output_text`` part when the model
    /// wrote text, then one ``function_call`` or ``custom_tool_call`` item
    /// per tool call.
    pub output: Value,
    pub usage: ResponsesUsage,
}
