//! Buffered response translation: a chat completion into an Anthropic
//! Message.

use serde_json::{json, Map, Value};

/// The signature carried by every `thinking` block SIE answers. SIE does not
/// sign reasoning and ignores the signature of thinking blocks sent back.
pub(crate) const THINKING_SIGNATURE: &str = "sie-unsigned";

/// The Anthropic `stop_reason` for a chat `finish_reason`.
pub(crate) fn stop_reason(finish_reason: Option<&str>, has_tool_use: bool) -> &'static str {
    match finish_reason {
        Some("length") => "max_tokens",
        Some("content_filter") => "refusal",
        Some("tool_calls" | "function_call") => "tool_use",
        _ if has_tool_use => "tool_use",
        _ => "end_turn",
    }
}

/// Anthropic usage from a chat usage block: `input_tokens` excludes the
/// prompt tokens served from the prefix cache, which are reported as
/// `cache_read_input_tokens`, so the two add up to the chat `prompt_tokens`.
pub(crate) fn anthropic_usage(usage: &Value) -> Option<Value> {
    let prompt = usage.get("prompt_tokens")?.as_u64()?;
    let completion = usage.get("completion_tokens")?.as_u64()?;
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(prompt);
    Some(json!({
        "input_tokens": prompt - cached,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "output_tokens": completion,
    }))
}

/// The tool input a chat tool call's `arguments` string carries.
pub(crate) fn tool_input(arguments: &str) -> Result<Value, String> {
    if arguments.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(input @ Value::Object(_)) => Ok(input),
        Ok(_) => Err("tool call arguments are not a JSON object".to_string()),
        Err(_) => Err("tool call arguments are not valid JSON".to_string()),
    }
}

/// Translate a chat completion into a Message. `model` is the model id the
/// caller sent; `request_id` the gateway request id.
pub(crate) fn chat_completion_to_message(
    chat: &Value,
    model: &str,
    request_id: &str,
) -> Result<Value, String> {
    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or("chat completion has no choices")?;
    let message = choice
        .get("message")
        .ok_or("chat completion choice has no message")?;
    let mut content = Vec::new();
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|reasoning| !reasoning.is_empty())
    {
        content.push(json!({
            "type": "thinking",
            "thinking": reasoning,
            "signature": THINKING_SIGNATURE,
        }));
    }
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
    {
        content.push(json!({"type": "text", "text": text}));
    }
    let mut has_tool_use = false;
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .ok_or("tool call has no id")?;
        let function = call.get("function").ok_or("tool call has no function")?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or("tool call has no name")?;
        let arguments = function
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        content.push(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": tool_input(arguments)?,
        }));
        has_tool_use = true;
    }
    let usage = chat
        .get("usage")
        .and_then(anthropic_usage)
        .ok_or("chat completion has no usage")?;
    Ok(json!({
        "id": format!("msg_{request_id}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(choice.get("finish_reason").and_then(Value::as_str), has_tool_use),
        "stop_sequence": Value::Null,
        "usage": usage,
    }))
}
