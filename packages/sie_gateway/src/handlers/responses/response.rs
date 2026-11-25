//! Buffered response translation: a chat completion into a Response.

use serde_json::{json, Map, Value};

/// What a Response repeats from its request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Echo {
    pub model: String,
    pub parallel_tool_calls: Value,
    pub tool_choice: Value,
    /// Names of the declared custom tools, whose calls come back as
    /// `custom_tool_call` items.
    pub custom_tools: Vec<String>,
}

/// The parts of a Response that do not depend on the answer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResponseShell {
    echo: Echo,
    request_id: String,
    created_at: u64,
}

impl ResponseShell {
    pub(crate) fn new(echo: Echo, request_id: String, created_at: u64) -> Self {
        Self {
            echo,
            request_id,
            created_at,
        }
    }

    pub(crate) fn is_custom(&self, name: &str) -> bool {
        self.echo.custom_tools.iter().any(|custom| custom == name)
    }

    pub(crate) fn message_id(&self) -> String {
        format!("msg-{}", self.request_id)
    }

    pub(crate) fn later_message_id(&self, output_index: u64) -> String {
        format!("msg-{}-{output_index}", self.request_id)
    }

    pub(crate) fn call_id(&self, custom: bool, position: u64) -> String {
        let prefix = if custom { "ctc" } else { "fc" };
        format!("{prefix}-{}-{position}", self.request_id)
    }

    /// A `function_call` or `custom_tool_call` item. `value` is the
    /// arguments of a function call or the input of a custom-tool call.
    pub(crate) fn call_item(
        &self,
        position: u64,
        call_id: &str,
        name: &str,
        value: &str,
        status: &str,
    ) -> Value {
        let custom = self.is_custom(name);
        let (kind, field) = if custom {
            ("custom_tool_call", "input")
        } else {
            ("function_call", "arguments")
        };
        let mut item = Map::new();
        item.insert("type".into(), json!(kind));
        item.insert("id".into(), json!(self.call_id(custom, position)));
        item.insert("call_id".into(), json!(call_id));
        item.insert("name".into(), json!(name));
        item.insert(field.into(), json!(value));
        item.insert("status".into(), json!(status));
        Value::Object(item)
    }

    pub(crate) fn object(
        &self,
        status: &str,
        incomplete_details: Value,
        error: Value,
        output: &[Value],
        usage: Value,
    ) -> Value {
        json!({
            "id": format!("resp-{}", self.request_id),
            "object": "response",
            "created_at": self.created_at,
            "model": self.echo.model,
            "status": status,
            "incomplete_details": incomplete_details,
            "error": error,
            "store": false,
            "parallel_tool_calls": self.echo.parallel_tool_calls,
            "tool_choice": self.echo.tool_choice,
            "output": output,
            "usage": usage,
        })
    }
}

/// An `output_text` content part.
pub(crate) fn output_text(text: &str) -> Value {
    json!({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
}

/// A `message` item. An item still in progress has no content yet.
pub(crate) fn message_item(id: &str, text: &str, status: &str) -> Value {
    let content = if status == "in_progress" {
        json!([])
    } else {
        json!([output_text(text)])
    };
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": status,
        "content": content,
    })
}

/// The input a custom-tool call carries: the string `input` of its
/// arguments, or the arguments as sent when they hold none.
pub(crate) fn custom_input(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(fields)) => match fields.get("input") {
            Some(Value::String(input)) => input.clone(),
            _ => arguments.to_string(),
        },
        _ => arguments.to_string(),
    }
}

/// The Response `status` and `incomplete_details` for a chat
/// `finish_reason`.
pub(crate) fn status_for(finish_reason: Option<&str>) -> (&'static str, Value) {
    match finish_reason {
        Some("length") => ("incomplete", json!({"reason": "max_output_tokens"})),
        Some("content_filter") => ("incomplete", json!({"reason": "content_filter"})),
        _ => ("completed", Value::Null),
    }
}

/// Responses usage from a chat usage block, with absent details counted
/// as 0.
pub(crate) fn responses_usage(usage: &Value) -> Option<Value> {
    let prompt = usage.get("prompt_tokens")?.as_u64()?;
    let completion = usage.get("completion_tokens")?.as_u64()?;
    let total = usage.get("total_tokens")?.as_u64()?;
    let detail = |pointer: &str| usage.pointer(pointer).and_then(Value::as_u64).unwrap_or(0);
    Some(json!({
        "input_tokens": prompt,
        "input_tokens_details": {"cached_tokens": detail("/prompt_tokens_details/cached_tokens")},
        "output_tokens": completion,
        "output_tokens_details": {"reasoning_tokens": detail("/completion_tokens_details/reasoning_tokens")},
        "total_tokens": total,
    }))
}

/// Translate a chat completion into a Response.
pub(crate) fn chat_completion_to_response(
    chat: &Value,
    shell: &ResponseShell,
) -> Result<Value, String> {
    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or("chat completion has no choices")?;
    let message = choice
        .get("message")
        .ok_or("chat completion choice has no message")?;
    let mut output = Vec::new();
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        output.push(message_item(&shell.message_id(), text, "completed"));
    }
    for (position, call) in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let call_id = call
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
        let value = if shell.is_custom(name) {
            custom_input(arguments)
        } else {
            arguments.to_string()
        };
        output.push(shell.call_item(position as u64, call_id, name, &value, "completed"));
    }
    let (status, incomplete_details) =
        status_for(choice.get("finish_reason").and_then(Value::as_str));
    if let Some(last) = output.last_mut() {
        last["status"] = json!(status);
    }
    let usage = chat
        .get("usage")
        .and_then(responses_usage)
        .unwrap_or(Value::Null);
    Ok(shell.object(status, incomplete_details, Value::Null, &output, usage))
}
