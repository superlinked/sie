//! Request translation: an Anthropic Messages body into a chat completions
//! body.
//!
//! Validation happens here, in Anthropic terms, so a refusal names the field
//! the caller sent. Unknown fields fail with `<path>: Extra inputs are not
//! permitted`. The accepted-but-inert set is `metadata`, `cache_control` in
//! every position, `thinking.display: "summarized"` and the tool flag
//! `eager_input_streaming`.

use serde_json::{json, Map, Value};

use super::error::MessagesError;
use crate::handlers::grammar::{parse_grammar, GrammarParseResult};

pub(crate) const SUPPORTED_ANTHROPIC_VERSION: &str = "2023-06-01";
pub(crate) const MAX_TOOLS: usize = 64;
const MAX_TOOL_NAME_LEN: usize = 64;
const MAX_USER_ID_CHARS: usize = 512;
const MIN_THINKING_BUDGET: u64 = 1024;
const IMAGE_MEDIA_TYPES: [&str; 4] = ["image/jpeg", "image/png", "image/gif", "image/webp"];

pub(crate) const SYSTEM_TURN_REFUSAL: &str =
    "role \"system\" is not supported in messages; use the top-level system parameter";
pub(crate) const CONTEXT_MANAGEMENT_REFUSAL: &str = "context_management: Extra inputs are not permitted (Claude Code: set CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1)";

/// A translated request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatTranslation {
    pub chat: Value,
    pub model: String,
    pub stream: bool,
    /// Anthropic path of the first image block, named if the model refuses
    /// image input.
    pub first_image: Option<String>,
}

fn invalid(message: impl Into<String>) -> MessagesError {
    MessagesError::invalid(message)
}

fn extra(path: &str) -> MessagesError {
    invalid(format!("{path}: Extra inputs are not permitted"))
}

fn required(path: &str) -> MessagesError {
    invalid(format!("{path}: Field required"))
}

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, MessagesError> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{path}: Input should be a valid dictionary")))
}

fn only_keys(map: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<(), MessagesError> {
    match map.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(extra(&format!("{path}.{key}"))),
        None => Ok(()),
    }
}

fn present<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|value| !value.is_null())
}

fn string<'a>(
    map: &'a Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<&'a str, MessagesError> {
    match map.get(key) {
        None | Some(Value::Null) => Err(required(&format!("{path}.{key}"))),
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(invalid(format!(
            "{path}.{key}: Input should be a valid string"
        ))),
    }
}

fn cache_control(map: &Map<String, Value>, path: &str) -> Result<(), MessagesError> {
    match present(map, "cache_control") {
        None | Some(Value::Object(_)) => Ok(()),
        Some(_) => Err(invalid(format!(
            "{path}.cache_control: Input should be a valid dictionary"
        ))),
    }
}

fn citations(map: &Map<String, Value>, path: &str) -> Result<(), MessagesError> {
    match present(map, "citations") {
        None => Ok(()),
        Some(Value::Array(items)) if items.is_empty() => Ok(()),
        Some(_) => Err(invalid(format!(
            "{path}.citations: citations are not supported"
        ))),
    }
}

/// Text accumulated in one turn, with image parts kept in order.
#[derive(Default)]
struct Parts {
    items: Vec<Part>,
}

enum Part {
    Text(String),
    Image(String),
}

impl Parts {
    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if let Some(Part::Text(last)) = self.items.last_mut() {
            last.push_str(text);
        } else {
            self.items.push(Part::Text(text.to_string()));
        }
    }

    fn image(&mut self, data_uri: String) {
        self.items.push(Part::Image(data_uri));
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn into_content(self) -> Value {
        if self.items.iter().all(|part| matches!(part, Part::Text(_))) {
            let text: String = self
                .items
                .into_iter()
                .map(|part| match part {
                    Part::Text(text) => text,
                    Part::Image(_) => String::new(),
                })
                .collect();
            return Value::String(text);
        }
        Value::Array(
            self.items
                .into_iter()
                .map(|part| match part {
                    Part::Text(text) => json!({"type": "text", "text": text}),
                    Part::Image(url) => json!({"type": "image_url", "image_url": {"url": url}}),
                })
                .collect(),
        )
    }
}

/// Translate an Anthropic Messages body into a chat completions body.
///
/// `anthropic_version` is the `anthropic-version` header, absent or
/// `2023-06-01`.
pub(crate) fn messages_to_chat(
    body: &Value,
    anthropic_version: Option<&str>,
) -> Result<ChatTranslation, MessagesError> {
    if anthropic_version.is_some_and(|version| version != SUPPORTED_ANTHROPIC_VERSION) {
        return Err(invalid(format!(
            "anthropic-version: only {SUPPORTED_ANTHROPIC_VERSION} is supported"
        )));
    }
    let Some(body) = body.as_object() else {
        return Err(invalid("body: Input should be a valid dictionary"));
    };
    for key in body.keys() {
        match key.as_str() {
            "model" | "max_tokens" | "messages" | "system" | "stop_sequences" | "temperature"
            | "top_p" | "top_k" | "stream" | "tools" | "tool_choice" | "thinking" | "metadata"
            | "cache_control" => {}
            "output_config" => output_config(body.get(key).unwrap_or(&Value::Null))?,
            "context_management" => return Err(invalid(CONTEXT_MANAGEMENT_REFUSAL)),
            other => return Err(extra(other)),
        }
    }

    let model = match body.get("model") {
        None | Some(Value::Null) => return Err(required("model")),
        Some(Value::String(model)) if !model.is_empty() => model.clone(),
        Some(Value::String(_)) => {
            return Err(invalid("model: String should have at least 1 character"))
        }
        Some(_) => return Err(invalid("model: Input should be a valid string")),
    };
    let max_tokens = match body.get("max_tokens") {
        None | Some(Value::Null) => return Err(required("max_tokens")),
        Some(value) => match value.as_u64() {
            Some(0) => {
                return Err(invalid(
                    "max_tokens: Input should be greater than or equal to 1",
                ))
            }
            Some(tokens) if tokens <= u64::from(u32::MAX) => tokens,
            _ if value.as_i64().is_some_and(|tokens| tokens < 1) => {
                return Err(invalid(
                    "max_tokens: Input should be greater than or equal to 1",
                ))
            }
            _ => return Err(invalid("max_tokens: Input should be a valid integer")),
        },
    };
    let stream = match body.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => return Err(invalid("stream: Input should be a valid boolean")),
    };
    let temperature = bounded_number(body, "temperature", 0.0, true)?;
    let top_p = bounded_number(body, "top_p", 0.0, false)?;
    let top_k = match present(body, "top_k") {
        None => None,
        Some(value) => match value.as_u64() {
            Some(k) if (1..=u64::from(u32::MAX)).contains(&k) => Some(k),
            Some(_) => return Err(invalid("top_k: Input should be greater than or equal to 1")),
            None if value.as_i64().is_some() => {
                return Err(invalid("top_k: Input should be greater than or equal to 1"))
            }
            None => return Err(invalid("top_k: Input should be a valid integer")),
        },
    };
    let stop = stop_sequences(body)?;
    metadata(body)?;
    if let Some(value) = present(body, "cache_control") {
        if !value.is_object() {
            return Err(invalid("cache_control: Input should be a valid dictionary"));
        }
    }
    let include_reasoning = thinking(body, max_tokens)?;
    let (tools, tool_names) = tools(body)?;
    let (tool_choice, parallel_tool_calls) = tool_choice(body, &tool_names)?;

    let mut first_image = None;
    let mut chat_messages = Vec::new();
    if let Some(system) = system(body)? {
        chat_messages.push(json!({"role": "system", "content": system}));
    }
    let messages = match body.get("messages") {
        None | Some(Value::Null) => return Err(required("messages")),
        Some(Value::Array(messages)) if messages.is_empty() => {
            return Err(invalid("messages: List should have at least 1 item"))
        }
        Some(Value::Array(messages)) => messages,
        Some(_) => return Err(invalid("messages: Input should be a valid list")),
    };
    for (index, message) in messages.iter().enumerate() {
        translate_message(message, index, &mut chat_messages, &mut first_image)?;
    }
    if messages
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("assistant")
    {
        return Err(invalid(
            "messages: assistant message prefill is not supported; the conversation must end with a user message",
        ));
    }

    let mut chat = Map::new();
    chat.insert("model".into(), json!(model));
    chat.insert("messages".into(), Value::Array(chat_messages));
    chat.insert("max_tokens".into(), json!(max_tokens));
    if let Some(stop) = stop {
        chat.insert("stop".into(), json!(stop));
    }
    if let Some(temperature) = temperature {
        chat.insert("temperature".into(), temperature);
    }
    if let Some(top_p) = top_p {
        chat.insert("top_p".into(), top_p);
    }
    if let Some(top_k) = top_k {
        chat.insert("top_k".into(), json!(top_k));
    }
    chat.insert("stream".into(), json!(stream));
    if stream {
        chat.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if let Some(tools) = tools {
        chat.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = tool_choice {
        chat.insert("tool_choice".into(), tool_choice);
    }
    if let Some(parallel) = parallel_tool_calls {
        chat.insert("parallel_tool_calls".into(), json!(parallel));
    }
    chat.insert("include_reasoning".into(), json!(include_reasoning));
    Ok(ChatTranslation {
        chat: Value::Object(chat),
        model,
        stream,
        first_image,
    })
}

fn output_config(value: &Value) -> Result<(), MessagesError> {
    if value.is_null() {
        return Ok(());
    }
    match object(value, "output_config")?.keys().next() {
        Some(key) => Err(extra(&format!("output_config.{key}"))),
        None => Ok(()),
    }
}

fn bounded_number(
    body: &Map<String, Value>,
    key: &str,
    low: f64,
    low_inclusive: bool,
) -> Result<Option<Value>, MessagesError> {
    let Some(value) = present(body, key) else {
        return Ok(None);
    };
    let Some(number) = value.as_f64().filter(|number| number.is_finite()) else {
        return Err(invalid(format!("{key}: Input should be a valid number")));
    };
    if low_inclusive && number < low {
        return Err(invalid(format!(
            "{key}: Input should be greater than or equal to {low}"
        )));
    }
    if !low_inclusive && number <= low {
        return Err(invalid(format!(
            "{key}: Input should be greater than {low}"
        )));
    }
    if number > 1.0 {
        return Err(invalid(format!(
            "{key}: Input should be less than or equal to 1"
        )));
    }
    Ok(Some(value.clone()))
}

fn stop_sequences(body: &Map<String, Value>) -> Result<Option<Vec<String>>, MessagesError> {
    let Some(value) = present(body, "stop_sequences") else {
        return Ok(None);
    };
    let Some(items) = value.as_array() else {
        return Err(invalid("stop_sequences: Input should be a valid list"));
    };
    let mut stop = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item.as_str() {
            Some(sequence) => stop.push(sequence.to_string()),
            None => {
                return Err(invalid(format!(
                    "stop_sequences.{index}: Input should be a valid string"
                )))
            }
        }
    }
    Ok((!stop.is_empty()).then_some(stop))
}

fn metadata(body: &Map<String, Value>) -> Result<(), MessagesError> {
    let Some(value) = present(body, "metadata") else {
        return Ok(());
    };
    let metadata = object(value, "metadata")?;
    only_keys(metadata, &["user_id"], "metadata")?;
    match present(metadata, "user_id") {
        None => Ok(()),
        Some(Value::String(user_id)) if user_id.chars().count() <= MAX_USER_ID_CHARS => Ok(()),
        Some(Value::String(_)) => Err(invalid(format!(
            "metadata.user_id: String should have at most {MAX_USER_ID_CHARS} characters"
        ))),
        Some(_) => Err(invalid("metadata.user_id: Input should be a valid string")),
    }
}

/// Whether the request opts into public reasoning.
fn thinking(body: &Map<String, Value>, max_tokens: u64) -> Result<bool, MessagesError> {
    let Some(value) = present(body, "thinking") else {
        return Ok(false);
    };
    let thinking = object(value, "thinking")?;
    let kind = match thinking.get("type") {
        None | Some(Value::Null) => return Err(required("thinking.type")),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => return Err(invalid("thinking.type: Input should be a valid string")),
    };
    let path = format!("thinking.{kind}");
    match kind {
        "enabled" => {
            only_keys(thinking, &["type", "budget_tokens", "display"], &path)?;
            let budget = match thinking.get("budget_tokens") {
                None | Some(Value::Null) => return Err(required(&format!("{path}.budget_tokens"))),
                Some(value) => value.as_u64().ok_or_else(|| {
                    invalid(format!(
                        "{path}.budget_tokens: Input should be a valid integer"
                    ))
                })?,
            };
            if budget < MIN_THINKING_BUDGET {
                return Err(invalid(format!(
                    "{path}.budget_tokens: Input should be greater than or equal to {MIN_THINKING_BUDGET}"
                )));
            }
            if budget >= max_tokens {
                return Err(invalid(format!(
                    "{path}.budget_tokens: must be less than max_tokens"
                )));
            }
        }
        "adaptive" => only_keys(thinking, &["type", "display"], &path)?,
        "disabled" => {
            only_keys(thinking, &["type"], &path)?;
            return Ok(false);
        }
        "between_tools" => {
            return Err(invalid(
                "\"thinking.type.between_tools\" is not supported for this model.",
            ))
        }
        _ => {
            return Err(invalid(
                "thinking.type: Input should be 'enabled', 'adaptive' or 'disabled'",
            ))
        }
    }
    match present(thinking, "display") {
        None => Ok(true),
        Some(Value::String(display)) if display == "summarized" => Ok(true),
        Some(Value::String(display)) if display == "omitted" => Ok(false),
        Some(_) => Err(invalid(format!(
            "{path}.display: Input should be 'summarized' or 'omitted'"
        ))),
    }
}

fn system(body: &Map<String, Value>) -> Result<Option<String>, MessagesError> {
    let Some(value) = present(body, "system") else {
        return Ok(None);
    };
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Array(blocks) => {
            let mut text = String::new();
            for (index, block) in blocks.iter().enumerate() {
                let path = format!("system.{index}");
                let block = object(block, &path)?;
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {}
                    None => return Err(required(&format!("{path}.type"))),
                    Some(_) => return Err(invalid(format!("{path}.type: Input should be 'text'"))),
                }
                only_keys(
                    block,
                    &["type", "text", "cache_control", "citations"],
                    &path,
                )?;
                cache_control(block, &path)?;
                citations(block, &path)?;
                text.push_str(string(block, "text", &path)?);
            }
            text
        }
        _ => return Err(invalid("system: Input should be a valid string or list")),
    };
    Ok((!text.is_empty()).then_some(text))
}

fn tools(body: &Map<String, Value>) -> Result<(Option<Vec<Value>>, Vec<String>), MessagesError> {
    let Some(value) = present(body, "tools") else {
        return Ok((None, Vec::new()));
    };
    let Some(items) = value.as_array() else {
        return Err(invalid("tools: Input should be a valid list"));
    };
    if items.len() > MAX_TOOLS {
        return Err(invalid(format!(
            "tools: at most {MAX_TOOLS} tools are supported, got {}",
            items.len()
        )));
    }
    let mut tools = Vec::with_capacity(items.len());
    let mut names = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let path = format!("tools.{index}");
        let tool = object(item, &path)?;
        match tool.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "custom" => {}
            Some(Value::String(kind)) => {
                return Err(invalid(format!(
                    "{path}.type: tool type '{kind}' is not supported; only custom tools with an input_schema are"
                )))
            }
            Some(_) => return Err(invalid(format!("{path}.type: Input should be a valid string"))),
        }
        only_keys(
            tool,
            &[
                "type",
                "name",
                "description",
                "input_schema",
                "cache_control",
                "eager_input_streaming",
            ],
            &path,
        )?;
        cache_control(tool, &path)?;
        match present(tool, "eager_input_streaming") {
            None | Some(Value::Bool(_)) => {}
            Some(_) => {
                return Err(invalid(format!(
                    "{path}.eager_input_streaming: Input should be a valid boolean"
                )))
            }
        }
        let name = string(tool, "name", &path)?;
        let valid_chars = !name.is_empty()
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-');
        if valid_chars && name.len() > MAX_TOOL_NAME_LEN {
            return Err(invalid(format!(
                "{path}.name: tool names longer than {MAX_TOOL_NAME_LEN} characters are not supported"
            )));
        }
        if !valid_chars {
            return Err(invalid(format!(
                "{path}.name: String should match pattern '^[a-zA-Z0-9_-]{{1,{MAX_TOOL_NAME_LEN}}}$'"
            )));
        }
        let mut function = Map::new();
        function.insert("name".into(), json!(name));
        match present(tool, "description") {
            None => {}
            Some(Value::String(description)) => {
                function.insert("description".into(), json!(description));
            }
            Some(_) => {
                return Err(invalid(format!(
                    "{path}.description: Input should be a valid string"
                )))
            }
        }
        let schema = match tool.get("input_schema") {
            None | Some(Value::Null) => return Err(required(&format!("{path}.input_schema"))),
            Some(schema @ Value::Object(_)) => schema,
            Some(_) => {
                return Err(invalid(format!(
                    "{path}.input_schema: Input should be a valid dictionary"
                )))
            }
        };
        if let GrammarParseResult::Err(_) = parse_grammar(&json!({"json_schema": schema})) {
            return Err(invalid(format!(
                "{path}.input_schema: the schema uses an unsupported JSON Schema keyword (if, then, else, $dynamicRef, unevaluatedProperties, dependentSchemas) or exceeds the size or depth limits"
            )));
        }
        function.insert("parameters".into(), schema.clone());
        tools.push(json!({"type": "function", "function": Value::Object(function)}));
        names.push(name.to_string());
    }
    if tools.is_empty() {
        return Ok((None, names));
    }
    Ok((Some(tools), names))
}

fn tool_choice(
    body: &Map<String, Value>,
    tool_names: &[String],
) -> Result<(Option<Value>, Option<bool>), MessagesError> {
    let Some(value) = present(body, "tool_choice") else {
        return Ok((None, None));
    };
    let choice = object(value, "tool_choice")?;
    let kind = match choice.get("type") {
        None | Some(Value::Null) => return Err(required("tool_choice.type")),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => return Err(invalid("tool_choice.type: Input should be a valid string")),
    };
    let path = format!("tool_choice.{kind}");
    let translated = match kind {
        "auto" | "any" | "none" => {
            only_keys(choice, &["type", "disable_parallel_tool_use"], &path)?;
            json!(match kind {
                "auto" => "auto",
                "any" => "required",
                _ => "none",
            })
        }
        "tool" => {
            only_keys(
                choice,
                &["type", "name", "disable_parallel_tool_use"],
                &path,
            )?;
            let name = string(choice, "name", &path)?;
            if !tool_names.iter().any(|declared| declared == name) {
                return Err(invalid(format!(
                    "{path}.name: tool '{name}' is not declared in tools"
                )));
            }
            json!({"type": "function", "function": {"name": name}})
        }
        _ => {
            return Err(invalid(
                "tool_choice.type: Input should be 'auto', 'any', 'tool' or 'none'",
            ))
        }
    };
    let parallel = match present(choice, "disable_parallel_tool_use") {
        None => None,
        Some(Value::Bool(disable)) => disable.then_some(false),
        Some(_) => {
            return Err(invalid(format!(
                "{path}.disable_parallel_tool_use: Input should be a valid boolean"
            )))
        }
    };
    if tool_names.is_empty() {
        return match kind {
            "auto" | "none" => Ok((None, None)),
            _ => Err(invalid(format!("{path}: tool_choice requires tools"))),
        };
    }
    Ok((Some(translated), parallel))
}

fn translate_message(
    message: &Value,
    index: usize,
    out: &mut Vec<Value>,
    first_image: &mut Option<String>,
) -> Result<(), MessagesError> {
    let path = format!("messages.{index}");
    let message = object(message, &path)?;
    let role = match message.get("role") {
        None | Some(Value::Null) => return Err(required(&format!("{path}.role"))),
        Some(Value::String(role)) => role.as_str(),
        Some(_) => {
            return Err(invalid(format!(
                "{path}.role: Input should be a valid string"
            )))
        }
    };
    match role {
        "user" | "assistant" => {}
        "system" => return Err(invalid(format!("{path}.role: {SYSTEM_TURN_REFUSAL}"))),
        _ => {
            return Err(invalid(format!(
                "{path}.role: Input should be 'user' or 'assistant'"
            )))
        }
    }
    only_keys(message, &["role", "content"], &path)?;
    let content = match message.get("content") {
        None | Some(Value::Null) => return Err(required(&format!("{path}.content"))),
        Some(Value::String(text)) => {
            out.push(json!({"role": role, "content": text}));
            return Ok(());
        }
        Some(Value::Array(blocks)) if blocks.is_empty() => {
            return Err(invalid(format!(
                "{path}.content: List should have at least 1 item"
            )))
        }
        Some(Value::Array(blocks)) => blocks,
        Some(_) => {
            return Err(invalid(format!(
                "{path}.content: Input should be a valid string or list"
            )))
        }
    };
    if role == "user" {
        user_blocks(content, &path, out, first_image)
    } else {
        assistant_blocks(content, &path, out)
    }
}

fn block_type<'a>(block: &'a Map<String, Value>, path: &str) -> Result<&'a str, MessagesError> {
    match block.get("type") {
        None | Some(Value::Null) => Err(required(&format!("{path}.type"))),
        Some(Value::String(kind)) => Ok(kind),
        Some(_) => Err(invalid(format!(
            "{path}.type: Input should be a valid string"
        ))),
    }
}

fn unsupported_block(path: &str, kind: &str) -> MessagesError {
    match kind {
        "text" | "image" | "tool_result" | "tool_use" | "thinking" | "redacted_thinking"
        | "document" | "search_result" | "container_upload" | "server_tool_use"
        | "web_search_tool_result" | "web_fetch_tool_result" | "code_execution_tool_result"
        | "bash_code_execution_tool_result" | "text_editor_code_execution_tool_result"
        | "tool_search_tool_result" | "mcp_tool_use" | "mcp_tool_result" => invalid(format!(
            "{path}.{kind}: content blocks of type '{kind}' are not supported here"
        )),
        _ => invalid(format!(
            "{path}.type: Input tag '{kind}' found using 'type' does not match any of the expected tags"
        )),
    }
}

fn text_block(block: &Map<String, Value>, path: &str) -> Result<String, MessagesError> {
    let path = format!("{path}.text");
    only_keys(
        block,
        &["type", "text", "cache_control", "citations"],
        &path,
    )?;
    cache_control(block, &path)?;
    citations(block, &path)?;
    Ok(string(block, "text", &path)?.to_string())
}

fn image_block(block: &Map<String, Value>, path: &str) -> Result<String, MessagesError> {
    let path = format!("{path}.image");
    only_keys(block, &["type", "source", "cache_control"], &path)?;
    cache_control(block, &path)?;
    let source = match block.get("source") {
        None | Some(Value::Null) => return Err(required(&format!("{path}.source"))),
        Some(source) => object(source, &format!("{path}.source"))?,
    };
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {}
        Some("url" | "file") => {
            return Err(invalid(format!(
                "{path}.source: only base64 image sources are supported"
            )))
        }
        Some(_) | None => {
            return Err(invalid(format!(
                "{path}.source.type: Input should be 'base64'"
            )))
        }
    }
    let base64_path = format!("{path}.source.base64");
    only_keys(source, &["type", "media_type", "data"], &base64_path)?;
    let media_type = string(source, "media_type", &base64_path)?;
    if !IMAGE_MEDIA_TYPES.contains(&media_type) {
        return Err(invalid(format!(
            "{base64_path}.media_type: Input should be 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'"
        )));
    }
    let data = string(source, "data", &base64_path)?;
    Ok(format!("data:{media_type};base64,{data}"))
}

fn user_blocks(
    blocks: &[Value],
    path: &str,
    out: &mut Vec<Value>,
    first_image: &mut Option<String>,
) -> Result<(), MessagesError> {
    let mut tool_messages = Vec::new();
    let mut parts = Parts::default();
    let mut tool_images = Parts::default();
    for (index, block) in blocks.iter().enumerate() {
        let block_path = format!("{path}.content.{index}");
        let block = object(block, &block_path)?;
        match block_type(block, &block_path)? {
            "text" => parts.text(&text_block(block, &block_path)?),
            "image" => {
                parts.image(image_block(block, &block_path)?);
                first_image.get_or_insert_with(|| format!("{block_path}.image"));
            }
            "tool_result" => {
                let result_path = format!("{block_path}.tool_result");
                only_keys(
                    block,
                    &[
                        "type",
                        "tool_use_id",
                        "content",
                        "is_error",
                        "cache_control",
                    ],
                    &result_path,
                )?;
                cache_control(block, &result_path)?;
                let tool_use_id = string(block, "tool_use_id", &result_path)?;
                if tool_use_id.is_empty() {
                    return Err(invalid(format!(
                        "{result_path}.tool_use_id: String should have at least 1 character"
                    )));
                }
                match present(block, "is_error") {
                    None | Some(Value::Bool(_)) => {}
                    Some(_) => {
                        return Err(invalid(format!(
                            "{result_path}.is_error: Input should be a valid boolean"
                        )))
                    }
                }
                let text = tool_result_content(
                    present(block, "content"),
                    &result_path,
                    &mut tool_images,
                    first_image,
                )?;
                tool_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": text,
                }));
            }
            kind => return Err(unsupported_block(&block_path, kind)),
        }
    }
    out.extend(tool_messages);
    let mut combined = tool_images;
    combined.items.extend(parts.items);
    if !combined.is_empty() || out.last().is_none_or(|last| last["role"] != "tool") {
        out.push(json!({"role": "user", "content": combined.into_content()}));
    }
    Ok(())
}

fn tool_result_content(
    content: Option<&Value>,
    path: &str,
    images: &mut Parts,
    first_image: &mut Option<String>,
) -> Result<String, MessagesError> {
    let blocks = match content {
        None => return Ok(String::new()),
        Some(Value::String(text)) => return Ok(text.clone()),
        Some(Value::Array(blocks)) => blocks,
        Some(_) => {
            return Err(invalid(format!(
                "{path}.content: Input should be a valid string or list"
            )))
        }
    };
    let mut text = String::new();
    for (index, block) in blocks.iter().enumerate() {
        let block_path = format!("{path}.content.{index}");
        let block = object(block, &block_path)?;
        match block_type(block, &block_path)? {
            "text" => text.push_str(&text_block(block, &block_path)?),
            "image" => {
                images.image(image_block(block, &block_path)?);
                first_image.get_or_insert_with(|| format!("{block_path}.image"));
            }
            kind => return Err(unsupported_block(&block_path, kind)),
        }
    }
    Ok(text)
}

fn assistant_blocks(
    blocks: &[Value],
    path: &str,
    out: &mut Vec<Value>,
) -> Result<(), MessagesError> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        let block_path = format!("{path}.content.{index}");
        let block = object(block, &block_path)?;
        match block_type(block, &block_path)? {
            "text" => text.push_str(&text_block(block, &block_path)?),
            "tool_use" => {
                let use_path = format!("{block_path}.tool_use");
                only_keys(
                    block,
                    &["type", "id", "name", "input", "cache_control"],
                    &use_path,
                )?;
                cache_control(block, &use_path)?;
                let id = string(block, "id", &use_path)?;
                let name = string(block, "name", &use_path)?;
                if id.is_empty() || name.is_empty() {
                    let field = if id.is_empty() { "id" } else { "name" };
                    return Err(invalid(format!(
                        "{use_path}.{field}: String should have at least 1 character"
                    )));
                }
                let input = match block.get("input") {
                    None | Some(Value::Null) => return Err(required(&format!("{use_path}.input"))),
                    Some(input @ Value::Object(_)) => input,
                    Some(_) => {
                        return Err(invalid(format!(
                            "{use_path}.input: Input should be a valid dictionary"
                        )))
                    }
                };
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                    },
                }));
            }
            "thinking" => {
                let thinking_path = format!("{block_path}.thinking");
                only_keys(block, &["type", "thinking", "signature"], &thinking_path)?;
            }
            "redacted_thinking" => {
                let redacted_path = format!("{block_path}.redacted_thinking");
                only_keys(block, &["type", "data"], &redacted_path)?;
            }
            kind => return Err(unsupported_block(&block_path, kind)),
        }
    }
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    if tool_calls.is_empty() {
        message.insert("content".into(), json!(text));
    } else {
        message.insert(
            "content".into(),
            if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            },
        );
        message.insert("tool_calls".into(), Value::Array(tool_calls));
    }
    out.push(Value::Object(message));
    Ok(())
}
