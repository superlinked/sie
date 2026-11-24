//! Request translation: an OpenAI Responses body into a chat completions
//! body.
//!
//! Validation happens here only where the chat route cannot name the field
//! the caller sent: the request shape, input items, content parts and tool
//! types. Sampling values, tool names and the tool count are the chat route's
//! to check. The accepted-but-inert set is `store: false`, `background:
//! false`, `truncation: "disabled"`, `include` of
//! `reasoning.encrypted_content`, `client_metadata`, `reasoning.summary`,
//! `reasoning.generate_summary` and `text.verbosity`.

use serde_json::{json, Map, Value};

use super::error::ResponsesError;
use super::response::Echo;

const STATELESS: &str = "SIE does not store responses; send the full conversation in input";
const ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";
const PASSED_THROUGH: [&str; 5] = [
    "metadata",
    "prompt_cache_key",
    "user",
    "safety_identifier",
    "service_tier",
];
const TEXT_PART_KEYS: [&str; 2] = ["type", "text"];
const OUTPUT_TEXT_PART_KEYS: [&str; 4] = ["type", "text", "annotations", "logprobs"];
const IMAGE_PART_KEYS: [&str; 4] = ["type", "image_url", "file_id", "detail"];

/// A translated request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatTranslation {
    pub chat: Value,
    pub stream: bool,
    pub echo: Echo,
}

type Result<T> = std::result::Result<T, ResponsesError>;

fn object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| ResponsesError::invalid(path, "Input should be a valid dictionary"))
}

fn only_keys(map: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<()> {
    match map.keys().find(|key| !allowed.contains(&key.as_str())) {
        Some(key) => Err(ResponsesError::extra(format!("{path}.{key}"))),
        None => Ok(()),
    }
}

fn present<'a>(map: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|value| !value.is_null())
}

fn string<'a>(map: &'a Map<String, Value>, key: &str, path: &str) -> Result<&'a str> {
    match present(map, key) {
        None => Err(ResponsesError::required(format!("{path}.{key}"))),
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(ResponsesError::invalid(
            format!("{path}.{key}"),
            "Input should be a valid string",
        )),
    }
}

/// Message content: adjacent texts joined, empty texts dropped, images kept
/// in order.
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

    fn image(&mut self, url: String) {
        self.items.push(Part::Image(url));
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

/// Translate a Responses body into a chat completions body.
pub(crate) fn responses_to_chat(body: &Value) -> Result<ChatTranslation> {
    let Some(body) = body.as_object() else {
        return Err(ResponsesError::invalid(
            "body",
            "Input should be a valid dictionary",
        ));
    };
    for (key, value) in body {
        match key.as_str() {
            "model"
            | "input"
            | "instructions"
            | "max_output_tokens"
            | "temperature"
            | "top_p"
            | "seed"
            | "stream"
            | "tools"
            | "tool_choice"
            | "parallel_tool_calls"
            | "reasoning"
            | "text"
            | "metadata"
            | "prompt_cache_key"
            | "user"
            | "safety_identifier"
            | "service_tier"
            | "client_metadata" => {}
            "store" => match value {
                Value::Null | Value::Bool(false) => {}
                Value::Bool(true) => return Err(ResponsesError::unsupported("store", STATELESS)),
                _ => {
                    return Err(ResponsesError::invalid(
                        "store",
                        "Input should be a valid boolean",
                    ))
                }
            },
            "previous_response_id" | "conversation" => {
                if !value.is_null() {
                    return Err(ResponsesError::unsupported(key.as_str(), STATELESS));
                }
            }
            "background" => match value {
                Value::Null | Value::Bool(false) => {}
                Value::Bool(true) => {
                    return Err(ResponsesError::unsupported(
                        "background",
                        "background responses are not supported",
                    ))
                }
                _ => {
                    return Err(ResponsesError::invalid(
                        "background",
                        "Input should be a valid boolean",
                    ))
                }
            },
            "truncation" => match value {
                Value::Null => {}
                Value::String(truncation) if truncation == "disabled" => {}
                _ => {
                    return Err(ResponsesError::unsupported(
                        "truncation",
                        "only \"disabled\" is supported",
                    ))
                }
            },
            "include" => include(value)?,
            other => return Err(ResponsesError::extra(other)),
        }
    }

    let model = match present(body, "model") {
        None => return Err(ResponsesError::required("model")),
        Some(Value::String(model)) if !model.is_empty() => model.clone(),
        Some(Value::String(_)) => {
            return Err(ResponsesError::invalid(
                "model",
                "String should have at least 1 character",
            ))
        }
        Some(_) => {
            return Err(ResponsesError::invalid(
                "model",
                "Input should be a valid string",
            ))
        }
    };
    let input = match present(body, "input") {
        None => return Err(ResponsesError::required("input")),
        Some(Value::Array(items)) if items.is_empty() => {
            return Err(ResponsesError::invalid(
                "input",
                "List should have at least 1 item",
            ))
        }
        Some(input @ (Value::String(_) | Value::Array(_))) => input,
        Some(_) => {
            return Err(ResponsesError::invalid(
                "input",
                "Input should be a valid string or list",
            ))
        }
    };
    let instructions = match present(body, "instructions") {
        None => None,
        Some(Value::String(instructions)) => Some(instructions.as_str()),
        Some(_) => {
            return Err(ResponsesError::invalid(
                "instructions",
                "Input should be a valid string",
            ))
        }
    };
    let max_output_tokens = max_output_tokens(present(body, "max_output_tokens"))?;
    let stream = match present(body, "stream") {
        None => false,
        Some(Value::Bool(stream)) => *stream,
        Some(_) => {
            return Err(ResponsesError::invalid(
                "stream",
                "Input should be a valid boolean",
            ))
        }
    };
    let effort = reasoning(present(body, "reasoning"))?;
    let response_format = text(present(body, "text"))?;
    let (tools, names, custom_tools) = tools(present(body, "tools"))?;
    let tool_choice = tool_choice(present(body, "tool_choice"), &names)?;
    let messages = messages(input, instructions)?;

    let mut chat = Map::new();
    chat.insert("model".into(), json!(model));
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(tokens) = max_output_tokens {
        chat.insert("max_completion_tokens".into(), json!(tokens));
    }
    for key in ["temperature", "top_p", "seed"] {
        if let Some(value) = present(body, key) {
            chat.insert(key.into(), value.clone());
        }
    }
    chat.insert("stream".into(), json!(stream));
    if stream {
        chat.insert("stream_options".into(), json!({"include_usage": true}));
    }
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tool_choice) = tool_choice {
        chat.insert("tool_choice".into(), tool_choice);
    }
    if let Some(parallel) = present(body, "parallel_tool_calls") {
        chat.insert("parallel_tool_calls".into(), parallel.clone());
    }
    if let Some(response_format) = response_format {
        chat.insert("response_format".into(), response_format);
    }
    for key in PASSED_THROUGH {
        if let Some(value) = present(body, key) {
            chat.insert(key.into(), value.clone());
        }
    }
    chat.insert("include_reasoning".into(), json!(false));
    if let Some(effort) = effort {
        chat.insert("reasoning_effort".into(), effort.clone());
    }
    let echo = Echo {
        model,
        parallel_tool_calls: present(body, "parallel_tool_calls")
            .cloned()
            .unwrap_or(json!(true)),
        tool_choice: present(body, "tool_choice")
            .cloned()
            .unwrap_or(json!("auto")),
        custom_tools,
    };
    Ok(ChatTranslation {
        chat: Value::Object(chat),
        stream,
        echo,
    })
}

fn include(value: &Value) -> Result<()> {
    let entries = match value {
        Value::Null => return Ok(()),
        Value::Array(entries) => entries,
        _ => {
            return Err(ResponsesError::invalid(
                "include",
                "Input should be a valid list",
            ))
        }
    };
    for (index, entry) in entries.iter().enumerate() {
        if entry.as_str() == Some(ENCRYPTED_REASONING) {
            continue;
        }
        let shown = match entry {
            Value::String(entry) => entry.clone(),
            other => other.to_string(),
        };
        return Err(ResponsesError::unsupported(
            format!("include[{index}]"),
            &format!("'{shown}' is not supported"),
        ));
    }
    Ok(())
}

fn max_output_tokens(value: Option<&Value>) -> Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.as_u64() {
        Some(tokens) if (1..=u64::from(u32::MAX)).contains(&tokens) => Ok(Some(tokens)),
        Some(0) => Err(ResponsesError::invalid(
            "max_output_tokens",
            "Input should be greater than or equal to 1",
        )),
        None if value.as_i64().is_some() => Err(ResponsesError::invalid(
            "max_output_tokens",
            "Input should be greater than or equal to 1",
        )),
        _ => Err(ResponsesError::invalid(
            "max_output_tokens",
            "Input should be a valid integer",
        )),
    }
}

/// The chat `reasoning_effort`, passed as sent for the chat route to check.
fn reasoning(value: Option<&Value>) -> Result<Option<&Value>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let reasoning = object(value, "reasoning")?;
    only_keys(
        reasoning,
        &["effort", "summary", "generate_summary"],
        "reasoning",
    )?;
    Ok(present(reasoning, "effort"))
}

/// The chat `response_format` for `text.format`.
fn text(value: Option<&Value>) -> Result<Option<Value>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let text = object(value, "text")?;
    only_keys(text, &["format", "verbosity"], "text")?;
    let Some(format) = present(text, "format") else {
        return Ok(None);
    };
    let format = object(format, "text.format")?;
    let kind = match present(format, "type") {
        None => return Err(ResponsesError::required("text.format.type")),
        Some(Value::String(kind)) => kind.as_str(),
        Some(_) => {
            return Err(ResponsesError::invalid(
                "text.format.type",
                "Input should be a valid string",
            ))
        }
    };
    let allowed: &[&str] = match kind {
        "text" | "json_object" => &["type"],
        "json_schema" => &["type", "name", "schema", "strict", "description"],
        _ => {
            return Err(ResponsesError::unsupported(
                "text.format.type",
                &format!("'{kind}' is not supported"),
            ))
        }
    };
    only_keys(format, allowed, "text.format")?;
    match kind {
        "text" => Ok(None),
        "json_object" => Ok(Some(json!({"type": "json_object"}))),
        _ => {
            let mut schema = Map::new();
            for key in ["name", "schema", "strict", "description"] {
                if let Some(value) = present(format, key) {
                    schema.insert(key.into(), value.clone());
                }
            }
            Ok(Some(
                json!({"type": "json_schema", "json_schema": Value::Object(schema)}),
            ))
        }
    }
}

/// The chat tools, the declared tool names, and the names of the custom
/// tools.
fn tools(value: Option<&Value>) -> Result<(Vec<Value>, Vec<String>, Vec<String>)> {
    let Some(value) = value else {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    };
    let Some(items) = value.as_array() else {
        return Err(ResponsesError::invalid(
            "tools",
            "Input should be a valid list",
        ));
    };
    let mut tools = Vec::with_capacity(items.len());
    let mut names = Vec::with_capacity(items.len());
    let mut custom_tools = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let path = format!("tools[{index}]");
        let tool = object(item, &path)?;
        let custom = match present(tool, "type") {
            None => return Err(ResponsesError::required(format!("{path}.type"))),
            Some(Value::String(kind)) if kind == "function" => false,
            Some(Value::String(kind)) if kind == "custom" => true,
            Some(Value::String(kind)) => {
                return Err(ResponsesError::unsupported(
                    format!("{path}.type"),
                    &format!("'{kind}' tools are not supported; send function or custom tools"),
                ))
            }
            Some(_) => {
                return Err(ResponsesError::invalid(
                    format!("{path}.type"),
                    "Input should be a valid string",
                ))
            }
        };
        if custom {
            only_keys(tool, &["type", "name", "description", "format"], &path)?;
        } else {
            only_keys(
                tool,
                &["type", "name", "description", "parameters", "strict"],
                &path,
            )?;
            match present(tool, "strict") {
                None | Some(Value::Bool(false)) => {}
                Some(Value::Bool(true)) => {
                    return Err(ResponsesError::unsupported(
                        format!("{path}.strict"),
                        "strict function schemas are not supported",
                    ))
                }
                Some(_) => {
                    return Err(ResponsesError::invalid(
                        format!("{path}.strict"),
                        "Input should be a valid boolean",
                    ))
                }
            }
        }
        let mut function = Map::new();
        for key in ["name", "description"] {
            if let Some(value) = present(tool, key) {
                function.insert(key.into(), value.clone());
            }
        }
        if custom {
            function.insert(
                "parameters".into(),
                json!({
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                    "additionalProperties": false,
                }),
            );
        } else if let Some(parameters) = present(tool, "parameters") {
            function.insert("parameters".into(), parameters.clone());
        }
        tools.push(json!({"type": "function", "function": Value::Object(function)}));
        if let Some(name) = tool.get("name").and_then(Value::as_str) {
            names.push(name.to_string());
            if custom {
                custom_tools.push(name.to_string());
            }
        }
    }
    Ok((tools, names, custom_tools))
}

fn tool_choice(value: Option<&Value>, names: &[String]) -> Result<Option<Value>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let choice = match value {
        Value::String(mode) => {
            return match mode.as_str() {
                "auto" | "none" if names.is_empty() => Ok(None),
                "auto" | "none" | "required" => Ok(Some(value.clone())),
                _ => Err(ResponsesError::invalid(
                    "tool_choice",
                    "Input should be 'auto', 'none' or 'required'",
                )),
            }
        }
        Value::Object(choice) => choice,
        _ => {
            return Err(ResponsesError::invalid(
                "tool_choice",
                "Input should be a valid string or dictionary",
            ))
        }
    };
    match present(choice, "type") {
        None => return Err(ResponsesError::required("tool_choice.type")),
        Some(Value::String(kind)) if kind == "function" || kind == "custom" => {}
        Some(Value::String(kind)) => {
            return Err(ResponsesError::unsupported(
                "tool_choice.type",
                &format!("'{kind}' is not supported"),
            ))
        }
        Some(_) => {
            return Err(ResponsesError::invalid(
                "tool_choice.type",
                "Input should be a valid string",
            ))
        }
    }
    only_keys(choice, &["type", "name"], "tool_choice")?;
    let name = string(choice, "name", "tool_choice")?;
    if !names.iter().any(|declared| declared == name) {
        return Err(ResponsesError::invalid(
            "tool_choice.name",
            &format!("tool '{name}' is not declared in tools"),
        ));
    }
    Ok(Some(
        json!({"type": "function", "function": {"name": name}}),
    ))
}

/// The chat messages, built item by item.
///
/// Consecutive assistant output (assistant messages and calls) becomes one
/// chat assistant message. The images of a run of tool outputs become one
/// user message after the run, joined by the user message that follows it.
#[derive(Default)]
struct Builder {
    out: Vec<Value>,
    system: Vec<String>,
    past_leading_system: bool,
    assistant: Option<(String, Vec<Value>)>,
    tool_images: Option<Vec<String>>,
}

impl Builder {
    fn flush_assistant(&mut self) {
        let Some((text, calls)) = self.assistant.take() else {
            return;
        };
        if calls.is_empty() {
            self.out.push(json!({"role": "assistant", "content": text}));
        } else {
            let content = if text.is_empty() {
                Value::Null
            } else {
                json!(text)
            };
            self.out
                .push(json!({"role": "assistant", "content": content, "tool_calls": calls}));
        }
    }

    fn take_tool_images(&mut self) -> Parts {
        let mut parts = Parts::default();
        for url in self.tool_images.take().into_iter().flatten() {
            parts.image(url);
        }
        parts
    }

    fn flush_tool_images(&mut self) {
        let images = self.take_tool_images();
        if !images.items.is_empty() {
            self.out
                .push(json!({"role": "user", "content": images.into_content()}));
        }
    }

    fn assistant(&mut self) -> &mut (String, Vec<Value>) {
        self.assistant.get_or_insert_with(Default::default)
    }

    fn item(&mut self, item: &Value, path: &str) -> Result<()> {
        let item = object(item, path)?;
        let kind = match present(item, "type") {
            None if !item.contains_key("role") => {
                return Err(ResponsesError::required(format!("{path}.role")))
            }
            None => "message",
            Some(Value::String(kind)) => kind.as_str(),
            Some(_) => {
                return Err(ResponsesError::invalid(
                    format!("{path}.type"),
                    "Input should be a valid string",
                ))
            }
        };
        let allowed: &[&str] = match kind {
            "message" => &["type", "id", "status", "role", "content"],
            "function_call" => &["type", "id", "status", "call_id", "name", "arguments"],
            "custom_tool_call" => &["type", "id", "status", "call_id", "name", "input"],
            "function_call_output" | "custom_tool_call_output" => {
                &["type", "id", "status", "call_id", "output"]
            }
            "reasoning" => &[
                "type",
                "id",
                "status",
                "summary",
                "content",
                "encrypted_content",
            ],
            _ => {
                return Err(ResponsesError::unsupported(
                    format!("{path}.type"),
                    &format!("items of type '{kind}' are not supported"),
                ))
            }
        };
        only_keys(item, allowed, path)?;
        match kind {
            "reasoning" => Ok(()),
            "message" => self.message(item, path),
            "function_call" | "custom_tool_call" => {
                self.past_leading_system = true;
                self.flush_tool_images();
                let call_id = string(item, "call_id", path)?;
                let name = string(item, "name", path)?;
                let arguments = if kind == "function_call" {
                    string(item, "arguments", path)?.to_string()
                } else {
                    json!({"input": string(item, "input", path)?}).to_string()
                };
                self.assistant().1.push(json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
                Ok(())
            }
            _ => {
                self.past_leading_system = true;
                self.flush_assistant();
                let call_id = string(item, "call_id", path)?;
                let (text, images) = tool_output(present(item, "output"), path)?;
                self.out.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": text,
                }));
                self.tool_images.get_or_insert_with(Vec::new).extend(images);
                Ok(())
            }
        }
    }

    fn message(&mut self, item: &Map<String, Value>, path: &str) -> Result<()> {
        let role = match present(item, "role") {
            None => return Err(ResponsesError::required(format!("{path}.role"))),
            Some(Value::String(role))
                if matches!(role.as_str(), "user" | "assistant" | "system" | "developer") =>
            {
                role.as_str()
            }
            Some(_) => {
                return Err(ResponsesError::invalid(
                    format!("{path}.role"),
                    "Input should be 'user', 'assistant', 'system' or 'developer'",
                ))
            }
        };
        let parts = content(item, role, path)?;
        match role {
            "system" | "developer" => {
                let text = parts.into_content();
                if !self.past_leading_system {
                    self.system
                        .push(text.as_str().unwrap_or_default().to_string());
                    return Ok(());
                }
                self.flush_assistant();
                self.flush_tool_images();
                self.out.push(json!({"role": "system", "content": text}));
            }
            "assistant" => {
                self.past_leading_system = true;
                self.flush_tool_images();
                let text = parts.into_content();
                self.assistant()
                    .0
                    .push_str(text.as_str().unwrap_or_default());
            }
            _ => {
                self.past_leading_system = true;
                self.flush_assistant();
                let mut merged = self.take_tool_images();
                merged.items.extend(parts.items);
                self.out
                    .push(json!({"role": "user", "content": merged.into_content()}));
            }
        }
        Ok(())
    }
}

fn messages(input: &Value, instructions: Option<&str>) -> Result<Vec<Value>> {
    let mut builder = Builder::default();
    if let Some(instructions) = instructions.filter(|text| !text.is_empty()) {
        builder.system.push(instructions.to_string());
    }
    match input {
        Value::String(text) => builder.out.push(json!({"role": "user", "content": text})),
        items => {
            for (index, item) in items.as_array().into_iter().flatten().enumerate() {
                builder.item(item, &format!("input[{index}]"))?;
            }
        }
    }
    builder.flush_assistant();
    builder.flush_tool_images();
    if !builder.system.is_empty() {
        builder.out.insert(
            0,
            json!({"role": "system", "content": builder.system.join("\n\n")}),
        );
    }
    Ok(builder.out)
}

fn part_type<'a>(part: &'a Map<String, Value>, path: &str) -> Result<&'a str> {
    match present(part, "type") {
        None => Err(ResponsesError::required(format!("{path}.type"))),
        Some(Value::String(kind)) => Ok(kind),
        Some(_) => Err(ResponsesError::invalid(
            format!("{path}.type"),
            "Input should be a valid string",
        )),
    }
}

/// The text of a text part, or `None` when the part is not a text part.
fn text_part(part: &Map<String, Value>, kind: &str, path: &str) -> Result<Option<String>> {
    let allowed: &[&str] = match kind {
        "input_text" | "text" => &TEXT_PART_KEYS,
        "output_text" => &OUTPUT_TEXT_PART_KEYS,
        _ => return Ok(None),
    };
    only_keys(part, allowed, path)?;
    Ok(Some(string(part, "text", path)?.to_string()))
}

fn image_part(part: &Map<String, Value>, path: &str) -> Result<String> {
    only_keys(part, &IMAGE_PART_KEYS, path)?;
    if present(part, "file_id").is_some() {
        return Err(ResponsesError::unsupported(
            format!("{path}.file_id"),
            "only base64 data: URLs are supported",
        ));
    }
    let url = string(part, "image_url", path)?;
    if !url.starts_with("data:") {
        return Err(ResponsesError::unsupported(
            format!("{path}.image_url"),
            "only base64 data: URLs are supported",
        ));
    }
    Ok(url.to_string())
}

fn unsupported_part(path: &str, kind: &str) -> ResponsesError {
    ResponsesError::unsupported(
        format!("{path}.type"),
        &format!("parts of type '{kind}' are not supported"),
    )
}

fn content(item: &Map<String, Value>, role: &str, path: &str) -> Result<Parts> {
    let mut parts = Parts::default();
    let items = match present(item, "content") {
        None => return Err(ResponsesError::required(format!("{path}.content"))),
        Some(Value::String(text)) => {
            parts.text(text);
            return Ok(parts);
        }
        Some(Value::Array(items)) => items,
        Some(_) => {
            return Err(ResponsesError::invalid(
                format!("{path}.content"),
                "Input should be a valid string or list",
            ))
        }
    };
    for (index, part) in items.iter().enumerate() {
        let part_path = format!("{path}.content[{index}]");
        let part = object(part, &part_path)?;
        let kind = part_type(part, &part_path)?;
        if let Some(text) = text_part(part, kind, &part_path)? {
            parts.text(&text);
        } else if kind == "input_image" {
            if role != "user" {
                return Err(ResponsesError::unsupported(
                    format!("{part_path}.type"),
                    "parts of type 'input_image' are only supported in user messages",
                ));
            }
            parts.image(image_part(part, &part_path)?);
        } else {
            return Err(unsupported_part(&part_path, kind));
        }
    }
    Ok(parts)
}

/// The text and the images of a tool output.
fn tool_output(output: Option<&Value>, path: &str) -> Result<(String, Vec<String>)> {
    let items = match output {
        None => return Err(ResponsesError::required(format!("{path}.output"))),
        Some(Value::String(text)) => return Ok((text.clone(), Vec::new())),
        Some(Value::Array(items)) => items,
        Some(_) => {
            return Err(ResponsesError::invalid(
                format!("{path}.output"),
                "Input should be a valid string or list",
            ))
        }
    };
    let mut text = String::new();
    let mut images = Vec::new();
    for (index, part) in items.iter().enumerate() {
        let part_path = format!("{path}.output[{index}]");
        let part = object(part, &part_path)?;
        let kind = part_type(part, &part_path)?;
        if let Some(part_text) = text_part(part, kind, &part_path)? {
            text.push_str(&part_text);
        } else if kind == "input_image" {
            images.push(image_part(part, &part_path)?);
        } else {
            return Err(unsupported_part(&part_path, kind));
        }
    }
    Ok((text, images))
}
