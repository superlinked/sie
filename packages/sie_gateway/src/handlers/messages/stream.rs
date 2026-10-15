//! Streaming translation: chat completion chunks in, Anthropic Messages
//! events out.
//!
//! [`StreamMachine`] is pure: it turns each chat SSE `data` payload into the
//! events it implies. [`stream_body`] drives it over the chat route's SSE body
//! and adds a `ping` whenever no event was sent for the ping interval.
//! Dropping the returned body drops the chat body with it, so a client that
//! goes away cancels the generation the same way a chat client does. An
//! `error` event ends the body the same way, so a stream the translation
//! cannot follow stops generating.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::time::Duration;

use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};

use super::error::{translate_error_body, ErrorContext};
use super::response::{anthropic_usage, stop_reason, THINKING_SIGNATURE};

pub(crate) const PING_INTERVAL: Duration = Duration::from_secs(10);

/// One Anthropic SSE event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Event {
    pub name: &'static str,
    pub data: Value,
}

impl Event {
    fn new(name: &'static str, data: Value) -> Self {
        Self { name, data }
    }

    pub(crate) fn ping() -> Self {
        Self::new("ping", json!({"type": "ping"}))
    }

    pub(crate) fn encode(&self) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", self.name, self.data))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Block {
    Text,
    Thinking,
    ToolUse { chat_index: u64 },
}

/// The chat-chunk to Anthropic-event state machine. One block is open at a
/// time and block indices are contiguous from 0.
#[derive(Debug)]
pub(crate) struct StreamMachine {
    message_id: String,
    model: String,
    next_index: u64,
    open: Option<(Block, u64)>,
    held_whitespace: String,
    tool_indices: Vec<u64>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    started: bool,
    done: bool,
}

fn zero_usage() -> Value {
    json!({
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    })
}

/// The HTTP status a mid-stream chat error code stands for, which decides
/// the Anthropic error type.
fn status_for_code(code: &str) -> u16 {
    match code {
        "invalid_request" | "unsupported_field" | "context_exceeded" | "INPUT_TOO_LONG"
        | "grammar_invalid" => 400,
        "PAYLOAD_TOO_LARGE" => 413,
        "rate_limit_exceeded" | "COLD_START_RATE_LIMITED" => 429,
        "RESOURCE_EXHAUSTED" | "QUEUE_FULL" | "MODEL_LOADING" | "LORA_LOADING"
        | "transport_failure" | "provisioning" | "QUEUE_UNAVAILABLE" => 503,
        "first_chunk_timeout"
        | "inter_chunk_timeout"
        | "overall_timeout"
        | "GATEWAY_TIMEOUT"
        | "cancelled" => 504,
        _ => 500,
    }
}

impl StreamMachine {
    pub(crate) fn new(message_id: String, model: String) -> Self {
        Self {
            message_id,
            model,
            next_index: 0,
            open: None,
            held_whitespace: String::new(),
            tool_indices: Vec::new(),
            finish_reason: None,
            usage: None,
            started: false,
            done: false,
        }
    }

    /// Whether `message_stop` or an `error` event has been produced.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    /// The `message_start` event, sent as soon as the chat stream opens.
    pub(crate) fn start(&mut self) -> Vec<Event> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![Event::new(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": Value::Null,
                    "stop_sequence": Value::Null,
                    "usage": zero_usage(),
                },
            }),
        )]
    }

    /// Feed one chat SSE `data` payload.
    pub(crate) fn on_data(&mut self, data: &str) -> Vec<Event> {
        let mut events = self.start();
        if self.done {
            return events;
        }
        if data == "[DONE]" {
            self.finish(&mut events);
            return events;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            self.fail(
                &mut events,
                "api_error",
                "the generation stream sent a malformed chunk",
            );
            return events;
        };
        if let Some(error) = chunk.get("error").filter(|error| error.is_object()) {
            let code = error.get("code").and_then(Value::as_str).unwrap_or("");
            let status = axum::http::StatusCode::from_u16(status_for_code(code))
                .unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR);
            let (error_type, message, _) =
                translate_error_body(status, &json!({"error": error}), &ErrorContext::default());
            self.fail(&mut events, error_type, &message);
            return events;
        }
        if let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(reasoning) = delta
                .get("reasoning_content")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                self.thinking(reasoning, &mut events);
            }
            if let Some(text) = delta
                .get("content")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                self.text(text, &mut events);
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Err(message) = self.tool_call(call, &mut events) {
                    self.fail(&mut events, "api_error", message);
                    return events;
                }
            }
            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.close(&mut events);
                self.held_whitespace.clear();
                self.finish_reason = Some(finish_reason.to_string());
            }
        }
        if let Some(usage) = chunk.get("usage").and_then(anthropic_usage) {
            self.usage = Some(usage);
        }
        if self.finish_reason.is_some() && self.usage.is_some() {
            self.complete(&mut events);
        }
        events
    }

    /// The chat stream ended. A stream that reported a finish completes;
    /// any other ending is an `error` event.
    pub(crate) fn on_end(&mut self) -> Vec<Event> {
        let mut events = self.start();
        if !self.done {
            self.finish(&mut events);
        }
        events
    }

    /// The chat stream broke before it ended.
    pub(crate) fn on_transport_error(&mut self) -> Vec<Event> {
        let mut events = self.start();
        if !self.done {
            self.fail(
                &mut events,
                "api_error",
                "the generation stream ended unexpectedly",
            );
        }
        events
    }

    fn finish(&mut self, events: &mut Vec<Event>) {
        if self.finish_reason.is_some() {
            self.complete(events);
        } else {
            self.fail(
                events,
                "api_error",
                "the generation stream ended without a finish reason",
            );
        }
    }

    fn complete(&mut self, events: &mut Vec<Event>) {
        self.close(events);
        let has_tool_use = !self.tool_indices.is_empty();
        events.push(Event::new(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason(self.finish_reason.as_deref(), has_tool_use),
                    "stop_sequence": Value::Null,
                },
                "usage": self.usage.clone().unwrap_or_else(zero_usage),
            }),
        ));
        events.push(Event::new("message_stop", json!({"type": "message_stop"})));
        self.done = true;
    }

    fn fail(&mut self, events: &mut Vec<Event>, error_type: &str, message: &str) {
        events.push(Event::new(
            "error",
            json!({
                "type": "error",
                "error": {"type": error_type, "message": message},
            }),
        ));
        self.done = true;
    }

    fn open_block(&mut self, block: Block, content_block: Value, events: &mut Vec<Event>) {
        let index = self.next_index;
        self.next_index += 1;
        self.open = Some((block, index));
        events.push(Event::new(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            }),
        ));
    }

    fn delta(&self, delta: Value, events: &mut Vec<Event>) {
        if let Some((_, index)) = self.open {
            events.push(Event::new(
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ));
        }
    }

    fn close(&mut self, events: &mut Vec<Event>) {
        let Some((block, index)) = self.open.take() else {
            return;
        };
        if block == Block::Thinking {
            events.push(Event::new(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE},
                }),
            ));
        }
        events.push(Event::new(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    fn thinking(&mut self, text: &str, events: &mut Vec<Event>) {
        if !matches!(self.open, Some((Block::Thinking, _))) {
            self.held_whitespace.clear();
            self.close(events);
            self.open_block(
                Block::Thinking,
                json!({"type": "thinking", "thinking": "", "signature": ""}),
                events,
            );
        }
        self.delta(json!({"type": "thinking_delta", "thinking": text}), events);
    }

    fn text(&mut self, text: &str, events: &mut Vec<Event>) {
        if !matches!(self.open, Some((Block::Text, _))) {
            if text.trim().is_empty() {
                self.held_whitespace.push_str(text);
                return;
            }
            self.close(events);
            self.open_block(Block::Text, json!({"type": "text", "text": ""}), events);
            let mut first = std::mem::take(&mut self.held_whitespace);
            first.push_str(text);
            self.delta(json!({"type": "text_delta", "text": first}), events);
            return;
        }
        self.delta(json!({"type": "text_delta", "text": text}), events);
    }

    fn tool_call(&mut self, call: &Value, events: &mut Vec<Event>) -> Result<(), &'static str> {
        let chat_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function").unwrap_or(&Value::Null);
        let id = call.get("id").and_then(Value::as_str);
        let name = function.get("name").and_then(Value::as_str);
        if !self.tool_indices.contains(&chat_index) {
            let (Some(id), Some(name)) = (id, name) else {
                return Err("a tool call arrived without its id and name");
            };
            self.held_whitespace.clear();
            self.close(events);
            self.open_block(
                Block::ToolUse { chat_index },
                json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                events,
            );
            self.tool_indices.push(chat_index);
        }
        let Some(arguments) = function
            .get("arguments")
            .and_then(Value::as_str)
            .filter(|arguments| !arguments.is_empty())
        else {
            return Ok(());
        };
        if self.open.map(|(block, _)| block) != Some(Block::ToolUse { chat_index }) {
            return Err("tool call arguments arrived after another content block started");
        }
        self.delta(
            json!({"type": "input_json_delta", "partial_json": arguments}),
            events,
        );
        Ok(())
    }
}

/// Splits an SSE byte stream into the `data` payloads of its events.
#[derive(Debug, Default)]
pub(crate) struct FrameParser {
    buffer: Vec<u8>,
}

impl FrameParser {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buffer
            .extend(bytes.iter().copied().filter(|byte| *byte != b'\r'));
        let mut payloads = Vec::new();
        while let Some(end) = self.buffer.windows(2).position(|window| window == b"\n\n") {
            let frame: Vec<u8> = self.buffer.drain(..end + 2).collect();
            let frame = String::from_utf8_lossy(&frame);
            let mut data: Option<String> = None;
            for line in frame.split('\n') {
                let Some(rest) = line.strip_prefix("data:") else {
                    continue;
                };
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                match data.as_mut() {
                    Some(data) => {
                        data.push('\n');
                        data.push_str(rest);
                    }
                    None => data = Some(rest.to_string()),
                }
            }
            payloads.extend(data);
        }
        payloads
    }
}

struct Driver {
    inner: axum::body::BodyDataStream,
    parser: FrameParser,
    machine: StreamMachine,
    pending: VecDeque<Bytes>,
    ping_every: Duration,
    deadline: tokio::time::Instant,
    finished: bool,
}

impl Driver {
    fn queue(&mut self, events: Vec<Event>) {
        self.pending.extend(events.iter().map(Event::encode));
    }
}

/// The Anthropic SSE body for a chat SSE body.
pub(crate) fn stream_body(inner: Body, mut machine: StreamMachine, ping_every: Duration) -> Body {
    let start = machine.start();
    let mut driver = Driver {
        inner: inner.into_data_stream(),
        parser: FrameParser::default(),
        machine,
        pending: VecDeque::new(),
        ping_every,
        deadline: tokio::time::Instant::now() + ping_every,
        finished: false,
    };
    driver.queue(start);
    let stream = futures_util::stream::unfold(driver, |mut driver| async move {
        loop {
            if let Some(bytes) = driver.pending.pop_front() {
                driver.deadline = tokio::time::Instant::now() + driver.ping_every;
                return Some((Ok::<Bytes, Infallible>(bytes), driver));
            }
            if driver.finished {
                return None;
            }
            let pinging = !driver.machine.is_done();
            tokio::select! {
                chunk = driver.inner.next() => match chunk {
                    Some(Ok(bytes)) => {
                        for payload in driver.parser.push(&bytes) {
                            let events = driver.machine.on_data(&payload);
                            let failed = events.iter().any(|event| event.name == "error");
                            driver.queue(events);
                            if failed {
                                driver.finished = true;
                                break;
                            }
                        }
                    }
                    Some(Err(_)) => {
                        let events = driver.machine.on_transport_error();
                        driver.queue(events);
                        driver.finished = true;
                    }
                    None => {
                        let events = driver.machine.on_end();
                        driver.queue(events);
                        driver.finished = true;
                    }
                },
                _ = tokio::time::sleep_until(driver.deadline), if pinging => {
                    driver.pending.push_back(Event::ping().encode());
                }
            }
        }
    });
    Body::from_stream(stream)
}
