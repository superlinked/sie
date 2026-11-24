//! Streaming translation: chat completion chunks in, Responses events out.
//!
//! [`StreamMachine`] is pure: it turns each chat SSE `data` payload into the
//! events it implies. [`stream_body`] drives it over the chat route's SSE body
//! and passes the chat route's comment heartbeats through until the terminal
//! event. Dropping the returned body drops the chat body with it, so a client
//! that goes away cancels the generation the same way a chat client does. A
//! `response.failed` event ends the body the same way, so a stream the
//! translation cannot follow stops generating.

use std::collections::VecDeque;
use std::convert::Infallible;

use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

use super::response::{custom_input, message_item, responses_usage, status_for, ResponseShell};

pub(crate) const FAILED: &str = "response.failed";
const SERVER_ERROR: &str = "server_error";

/// One Responses SSE event.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Event {
    pub name: &'static str,
    pub data: Value,
}

impl Event {
    pub(crate) fn encode(&self) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", self.name, self.data))
    }
}

#[derive(Debug)]
struct OpenText {
    output_index: u64,
    item_id: String,
    text: String,
}

#[derive(Debug)]
struct OpenCall {
    chat_index: u64,
    position: u64,
    output_index: u64,
    call_id: String,
    name: String,
    custom: bool,
    arguments: String,
}

/// The chat-chunk to Responses-event state machine. At most one output item
/// is open at a time, and output indices are contiguous from 0.
#[derive(Debug)]
pub(crate) struct StreamMachine {
    shell: ResponseShell,
    sequence: u64,
    output: Vec<Value>,
    next_index: u64,
    text: Option<OpenText>,
    call: Option<OpenCall>,
    calls: u64,
    closed_calls: Vec<u64>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    started: bool,
    done: bool,
}

impl StreamMachine {
    pub(crate) fn new(shell: ResponseShell) -> Self {
        Self {
            shell,
            sequence: 0,
            output: Vec::new(),
            next_index: 0,
            text: None,
            call: None,
            calls: 0,
            closed_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            started: false,
            done: false,
        }
    }

    /// Whether the terminal event has been produced.
    pub(crate) fn is_done(&self) -> bool {
        self.done
    }

    fn event(&mut self, name: &'static str, payload: Value) -> Event {
        let mut data = Map::new();
        data.insert("type".into(), json!(name));
        data.insert("sequence_number".into(), json!(self.sequence));
        if let Value::Object(fields) = payload {
            data.extend(fields);
        }
        self.sequence += 1;
        Event {
            name,
            data: Value::Object(data),
        }
    }

    /// `response.created` and `response.in_progress`, sent as soon as the
    /// chat stream opens.
    pub(crate) fn start(&mut self) -> Vec<Event> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        let response = self
            .shell
            .object("in_progress", Value::Null, Value::Null, &[], Value::Null);
        vec![
            self.event("response.created", json!({"response": response})),
            self.event("response.in_progress", json!({"response": response})),
        ]
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
                json!(SERVER_ERROR),
                json!("the generation stream sent a malformed chunk"),
            );
            return events;
        };
        if let Some(error) = chunk.get("error").filter(|error| error.is_object()) {
            let code = error.get("code").cloned().unwrap_or(Value::Null);
            let message = error.get("message").cloned().unwrap_or(Value::Null);
            self.fail(&mut events, code, message);
            return events;
        }
        if let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(text) = delta
                .get("content")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                self.text_delta(text, &mut events);
            }
            for call in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Err(message) = self.tool_call(call, &mut events) {
                    self.fail(&mut events, json!(SERVER_ERROR), json!(message));
                    return events;
                }
            }
            if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.close_text(&mut events);
                self.close_call(&mut events);
                self.finish_reason = Some(finish_reason.to_string());
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| usage.is_object()) {
            self.usage = Some(usage.clone());
        }
        events
    }

    /// The chat stream ended. A stream that reported a finish completes;
    /// any other ending fails.
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
                json!(SERVER_ERROR),
                json!("the generation stream ended unexpectedly"),
            );
        }
        events
    }

    fn finish(&mut self, events: &mut Vec<Event>) {
        if self.finish_reason.is_none() {
            self.fail(
                events,
                json!(SERVER_ERROR),
                json!("the generation stream ended without a finish reason"),
            );
            return;
        }
        self.close_text(events);
        self.close_call(events);
        let (status, incomplete_details) = status_for(self.finish_reason.as_deref());
        let name = if status == "completed" {
            "response.completed"
        } else {
            "response.incomplete"
        };
        let usage = self
            .usage
            .as_ref()
            .and_then(responses_usage)
            .unwrap_or(Value::Null);
        let response =
            self.shell
                .object(status, incomplete_details, Value::Null, &self.output, usage);
        let event = self.event(name, json!({"response": response}));
        events.push(event);
        self.done = true;
    }

    fn fail(&mut self, events: &mut Vec<Event>, code: Value, message: Value) {
        let response = self.shell.object(
            "failed",
            Value::Null,
            json!({"code": code, "message": message}),
            &self.output,
            Value::Null,
        );
        let event = self.event(FAILED, json!({"response": response}));
        events.push(event);
        self.done = true;
    }

    fn text_delta(&mut self, text: &str, events: &mut Vec<Event>) {
        self.close_call(events);
        if self.text.is_none() {
            let output_index = self.next_index;
            self.next_index += 1;
            let item_id = if output_index == 0 {
                self.shell.message_id()
            } else {
                self.shell.later_message_id(output_index)
            };
            let added = self.event(
                "response.output_item.added",
                json!({
                    "output_index": output_index,
                    "item": message_item(&item_id, "", "in_progress"),
                }),
            );
            let part = self.event(
                "response.content_part.added",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }),
            );
            events.extend([added, part]);
            self.text = Some(OpenText {
                output_index,
                item_id,
                text: String::new(),
            });
        }
        let Some(open) = self.text.as_mut() else {
            return;
        };
        open.text.push_str(text);
        let payload = json!({
            "item_id": open.item_id,
            "output_index": open.output_index,
            "content_index": 0,
            "delta": text,
        });
        let event = self.event("response.output_text.delta", payload);
        events.push(event);
    }

    fn close_text(&mut self, events: &mut Vec<Event>) {
        let Some(open) = self.text.take() else {
            return;
        };
        let item = message_item(&open.item_id, &open.text, "completed");
        let text_done = self.event(
            "response.output_text.done",
            json!({
                "item_id": open.item_id,
                "output_index": open.output_index,
                "content_index": 0,
                "text": open.text,
            }),
        );
        let part_done = self.event(
            "response.content_part.done",
            json!({
                "item_id": open.item_id,
                "output_index": open.output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": open.text, "annotations": []},
            }),
        );
        let item_done = self.event(
            "response.output_item.done",
            json!({"output_index": open.output_index, "item": item}),
        );
        events.extend([text_done, part_done, item_done]);
        self.output.push(item);
    }

    fn tool_call(&mut self, call: &Value, events: &mut Vec<Event>) -> Result<(), &'static str> {
        let chat_index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let function = call.get("function").unwrap_or(&Value::Null);
        if self.call.as_ref().map(|open| open.chat_index) != Some(chat_index) {
            if self.closed_calls.contains(&chat_index) {
                return Err("tool call arguments arrived after the call closed");
            }
            let (Some(call_id), Some(name)) = (
                call.get("id").and_then(Value::as_str),
                function.get("name").and_then(Value::as_str),
            ) else {
                return Err("a tool call arrived without its id and name");
            };
            self.close_text(events);
            self.close_call(events);
            let open = OpenCall {
                chat_index,
                position: self.calls,
                output_index: self.next_index,
                call_id: call_id.to_string(),
                name: name.to_string(),
                custom: self.shell.is_custom(name),
                arguments: String::new(),
            };
            self.calls += 1;
            self.next_index += 1;
            let item =
                self.shell
                    .call_item(open.position, &open.call_id, &open.name, "", "in_progress");
            let added = self.event(
                "response.output_item.added",
                json!({"output_index": open.output_index, "item": item}),
            );
            events.push(added);
            self.call = Some(open);
        }
        let Some(fragment) = function
            .get("arguments")
            .and_then(Value::as_str)
            .filter(|fragment| !fragment.is_empty())
        else {
            return Ok(());
        };
        let Some(open) = self.call.as_mut() else {
            return Ok(());
        };
        open.arguments.push_str(fragment);
        if open.custom {
            return Ok(());
        }
        let payload = json!({
            "item_id": self.shell.call_id(false, open.position),
            "output_index": open.output_index,
            "delta": fragment,
        });
        let event = self.event("response.function_call_arguments.delta", payload);
        events.push(event);
        Ok(())
    }

    fn close_call(&mut self, events: &mut Vec<Event>) {
        let Some(open) = self.call.take() else {
            return;
        };
        let item_id = self.shell.call_id(open.custom, open.position);
        let value = if open.custom {
            let input = custom_input(&open.arguments);
            let delta = self.event(
                "response.custom_tool_call_input.delta",
                json!({"item_id": item_id, "output_index": open.output_index, "delta": input}),
            );
            let done = self.event(
                "response.custom_tool_call_input.done",
                json!({"item_id": item_id, "output_index": open.output_index, "input": input}),
            );
            events.extend([delta, done]);
            input
        } else {
            let done = self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": open.output_index,
                    "arguments": open.arguments,
                }),
            );
            events.push(done);
            open.arguments
        };
        let item = self.shell.call_item(
            open.position,
            &open.call_id,
            &open.name,
            &value,
            "completed",
        );
        let done = self.event(
            "response.output_item.done",
            json!({"output_index": open.output_index, "item": item}),
        );
        events.push(done);
        self.output.push(item);
        self.closed_calls.push(open.chat_index);
    }
}

/// One SSE frame of the chat stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Frame {
    /// The joined `data` lines of an event.
    Data(String),
    /// A frame with comment lines only, as `:<text>`.
    Comment(String),
}

/// Splits an SSE byte stream into frames.
#[derive(Debug, Default)]
pub(crate) struct FrameParser {
    buffer: Vec<u8>,
}

impl FrameParser {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<Frame> {
        self.buffer
            .extend(bytes.iter().copied().filter(|byte| *byte != b'\r'));
        let mut frames = Vec::new();
        while let Some(end) = self.buffer.windows(2).position(|window| window == b"\n\n") {
            let raw: Vec<u8> = self.buffer.drain(..end + 2).collect();
            let raw = String::from_utf8_lossy(&raw);
            let mut data: Option<String> = None;
            let mut comment: Option<&str> = None;
            for line in raw.split('\n') {
                if line.starts_with(':') {
                    comment.get_or_insert(line);
                    continue;
                }
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
            match (data, comment) {
                (Some(data), _) => frames.push(Frame::Data(data)),
                (None, Some(comment)) => frames.push(Frame::Comment(comment.to_string())),
                (None, None) => {}
            }
        }
        frames
    }
}

struct Driver {
    inner: axum::body::BodyDataStream,
    parser: FrameParser,
    machine: StreamMachine,
    pending: VecDeque<Bytes>,
    finished: bool,
}

impl Driver {
    fn queue(&mut self, events: Vec<Event>) {
        self.pending.extend(events.iter().map(Event::encode));
    }

    fn frames(&mut self, bytes: &[u8]) {
        for frame in self.parser.push(bytes) {
            match frame {
                Frame::Comment(comment) if !self.machine.is_done() => {
                    self.pending
                        .push_back(Bytes::from(format!("{comment}\n\n")));
                }
                Frame::Comment(_) => {}
                Frame::Data(payload) => {
                    let events = self.machine.on_data(&payload);
                    let failed = events.iter().any(|event| event.name == FAILED);
                    self.queue(events);
                    if failed {
                        self.finished = true;
                        return;
                    }
                }
            }
        }
    }
}

/// The Responses SSE body for a chat SSE body.
pub(crate) fn stream_body(inner: Body, mut machine: StreamMachine) -> Body {
    let start = machine.start();
    let mut driver = Driver {
        inner: inner.into_data_stream(),
        parser: FrameParser::default(),
        machine,
        pending: VecDeque::new(),
        finished: false,
    };
    driver.queue(start);
    let stream = futures_util::stream::unfold(driver, |mut driver| async move {
        loop {
            if let Some(bytes) = driver.pending.pop_front() {
                return Some((Ok::<Bytes, Infallible>(bytes), driver));
            }
            if driver.finished {
                return None;
            }
            match driver.inner.next().await {
                Some(Ok(bytes)) => driver.frames(&bytes),
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
            }
        }
    });
    Body::from_stream(stream)
}
