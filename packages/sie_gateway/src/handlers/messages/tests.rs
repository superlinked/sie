use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::error::{anthropic_error_response, ErrorContext};
use super::request::{messages_to_chat, MAX_TOOLS};
use super::response::chat_completion_to_message;
use super::stream::{stream_body, FrameParser, StreamMachine};
use super::{messages_dispatch_inputs, proxy_messages};
use crate::handlers::proxy::chat_completions_dispatch_inputs;
use crate::handlers::test_support::{TestGateway, HYBRID_GENERATE_MODEL, LOCAL_LANE};

const CODER_MODEL: &str = "\
sie_id: acme/coder
hf_id: acme/coder
tasks:
  generate:
    max_output_tokens: 64
    capabilities:
      tools: true
profiles:
  default:
    adapter_path: sie_server.adapters.sglang:SGLangAdapter
    max_batch_tokens: 8192
";

fn vectors(name: &str) -> Value {
    let path = format!(
        "{}/../../conformance/anthropic_messages/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn messages_request(body: Value) -> Request {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/messages?beta=true")
        .header("content-type", "application/json")
        .header("anthropic-version", "2023-06-01")
        .header(
            "anthropic-beta",
            "claude-code-20250219,interleaved-thinking-2025-05-14",
        )
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn json_body(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn parse_events(bytes: &[u8]) -> Vec<Value> {
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(|frame| {
            let mut name = "";
            let mut data = Value::Null;
            for line in frame.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    name = rest;
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data = serde_json::from_str(rest).unwrap();
                }
            }
            assert_eq!(data["type"], name, "event name and data type agree");
            json!({"event": name, "data": data})
        })
        .collect()
}

async fn events_of(response: Response) -> Vec<Value> {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    parse_events(&bytes)
}

fn event_names(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|event| event["event"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn request_vectors_translate_onto_the_chat_body() {
    let vectors = vectors("request_vectors.json");
    for case in vectors["accepted"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let version = case.get("anthropic_version").and_then(Value::as_str);
        let translation = messages_to_chat(&case["body"], version)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(translation.chat, case["chat"], "{name}");
        assert_eq!(
            translation.first_image.as_deref(),
            case.get("first_image").and_then(Value::as_str),
            "{name}"
        );
        assert!(
            chat_completions_dispatch_inputs(&translation.chat).is_ok(),
            "{name}: the chat route accepts the translation"
        );
    }
    for case in vectors["rejected"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let version = case.get("anthropic_version").and_then(Value::as_str);
        let error = messages_to_chat(&case["body"], version).expect_err(name);
        assert_eq!(error.status, StatusCode::BAD_REQUEST, "{name}");
        assert_eq!(error.message, case["message"].as_str().unwrap(), "{name}");
    }
}

#[test]
fn every_captured_claude_code_request_translates_or_fails_with_the_recovery_wording() {
    let fixture = vectors("claude_code_requests.json");
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let version = case["headers"]["anthropic-version"].as_str();
        let expect = &case["expect"];
        match messages_to_chat(&case["body"], version) {
            Ok(translation) => {
                let chat = &expect["chat"];
                assert!(chat.is_object(), "{name}: expected a refusal");
                let roles: Vec<&str> = translation.chat["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|message| message["role"].as_str().unwrap())
                    .collect();
                let expected_roles: Vec<&str> = chat["roles"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|role| role.as_str().unwrap())
                    .collect();
                assert_eq!(roles, expected_roles, "{name}");
                assert_eq!(translation.chat["stream"], chat["stream"], "{name}");
                assert_eq!(translation.chat["max_tokens"], chat["max_tokens"], "{name}");
                assert_eq!(
                    translation.chat["include_reasoning"], chat["include_reasoning"],
                    "{name}"
                );
                assert_eq!(
                    translation.chat["tools"].as_array().map_or(0, Vec::len),
                    chat["tool_count"].as_u64().unwrap() as usize,
                    "{name}"
                );
                assert_eq!(
                    translation.chat.get("stop").cloned().unwrap_or(Value::Null),
                    chat.get("stop").cloned().unwrap_or(Value::Null),
                    "{name}"
                );
                assert!(
                    chat_completions_dispatch_inputs(&translation.chat).is_ok(),
                    "{name}: the chat route accepts the translation"
                );
            }
            Err(error) => {
                assert_eq!(
                    error.message,
                    expect["error"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{name}: unexpected refusal {}", error.message)),
                    "{name}"
                );
            }
        }
    }
}

#[test]
fn tool_count_over_the_limit_is_refused() {
    let tools: Vec<Value> = (0..=MAX_TOOLS)
        .map(|index| json!({"name": format!("tool_{index}"), "input_schema": {"type": "object"}}))
        .collect();
    let error = messages_to_chat(
        &json!({"model": "m", "max_tokens": 1, "tools": tools, "messages": [{"role": "user", "content": "x"}]}),
        None,
    )
    .unwrap_err();
    assert_eq!(
        error.message,
        "tools: at most 64 tools are supported, got 65"
    );
}

#[test]
fn response_vectors_translate_the_chat_completion() {
    let vectors = vectors("response_vectors.json");
    for case in vectors["translated"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let message = chat_completion_to_message(
            &case["chat"],
            case["model"].as_str().unwrap(),
            case["request_id"].as_str().unwrap(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(message, case["message"], "{name}");
    }
    for case in vectors["faults"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        assert!(
            chat_completion_to_message(&case["chat"], "m", "r").is_err(),
            "{name}"
        );
    }
}

fn stream_frames(case: &Value) -> Vec<String> {
    case["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|frame| match frame {
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .collect()
}

#[test]
fn stream_vectors_drive_the_state_machine() {
    let vectors = vectors("stream_vectors.json");
    for case in vectors["streams"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut machine = StreamMachine::new(
            case["message_id"].as_str().unwrap().to_string(),
            case["model"].as_str().unwrap().to_string(),
        );
        let mut events = machine.start();
        for frame in stream_frames(case) {
            events.extend(machine.on_data(&frame));
        }
        events.extend(machine.on_end());
        let got: Vec<Value> = events
            .iter()
            .map(|event| json!({"event": event.name, "data": event.data}))
            .collect();
        assert_eq!(&Value::Array(got), &case["events"], "{name}");
    }
}

#[tokio::test]
async fn stream_vectors_survive_the_sse_body_split_at_any_byte() {
    let vectors = vectors("stream_vectors.json");
    for case in vectors["streams"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut sse = String::new();
        for frame in stream_frames(case) {
            sse.push_str(&format!("data: {frame}\n\n"));
            sse.push_str(":\n\n");
        }
        let bytes = sse.into_bytes();
        let pieces: Vec<Result<Bytes, std::io::Error>> = bytes
            .chunks(7)
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect();
        let inner = Body::from_stream(futures_util::stream::iter(pieces));
        let machine = StreamMachine::new(
            case["message_id"].as_str().unwrap().to_string(),
            case["model"].as_str().unwrap().to_string(),
        );
        let body = stream_body(inner, machine, Duration::from_secs(60));
        let collected = to_bytes(body, usize::MAX).await.unwrap();
        assert_eq!(
            &Value::Array(parse_events(&collected)),
            &case["events"],
            "{name}"
        );
    }
}

#[test]
fn frame_parser_joins_multi_line_data_and_skips_comments() {
    let mut parser = FrameParser::default();
    assert!(parser.push(b": keep-alive\r\n\r\ndata: {\"a\"").is_empty());
    assert_eq!(
        parser.push(b":1}\r\n\r\nevent: x\ndata: one\ndata: two\n\n"),
        vec!["{\"a\":1}".to_string(), "one\ntwo".to_string()]
    );
}

#[tokio::test]
async fn error_vectors_render_the_anthropic_envelope() {
    let vectors = vectors("error_vectors.json");
    for case in vectors["errors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let body = match &case["body"] {
            Value::Null => Body::empty(),
            other => Body::from(other.to_string()),
        };
        let mut response = Response::new(body);
        *response.status_mut() =
            StatusCode::from_u16(case["status"].as_u64().unwrap() as u16).unwrap();
        for (header, value) in case
            .get("headers")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            response.headers_mut().insert(
                HeaderName::from_bytes(header.as_bytes()).unwrap(),
                HeaderValue::from_str(value.as_str().unwrap()).unwrap(),
            );
        }
        let context = ErrorContext {
            first_image: case
                .get("first_image")
                .and_then(Value::as_str)
                .map(str::to_string),
        };
        let translated = anthropic_error_response(response, &context).await;
        let expect = &case["expect"];
        assert_eq!(
            translated.status().as_u16() as u64,
            expect["status"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            translated.headers()["content-type"],
            "application/json",
            "{name}"
        );
        for (header, value) in expect
            .get("headers")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
        {
            assert_eq!(
                translated
                    .headers()
                    .get(header.as_str())
                    .map(|v| v.to_str().unwrap()),
                value.as_str(),
                "{name}: {header}"
            );
        }
        for header in expect
            .get("absent_headers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            assert!(
                translated.headers().get(header.as_str().unwrap()).is_none(),
                "{name}: {header}"
            );
        }
        let body = json_body(translated).await;
        assert_eq!(body["type"], "error", "{name}");
        assert_eq!(body["error"]["type"], expect["type"], "{name}");
        assert_eq!(body["error"]["message"], expect["message"], "{name}");
        assert_eq!(
            body["request_id"],
            expect.get("request_id").cloned().unwrap_or(Value::Null),
            "{name}"
        );
    }
}

#[tokio::test]
async fn an_inner_error_keeps_its_response_extensions() {
    let mut response = Response::new(Body::from(
        json!({"error": {"message": "failed", "type": "server_error", "param": null, "code": "x"}})
            .to_string(),
    ));
    *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    let response = crate::http_error::GatewayOwnedFault::new("stage").mark(response);
    let translated = anthropic_error_response(response, &ErrorContext::default()).await;
    assert_eq!(
        translated
            .extensions()
            .get::<crate::http_error::GatewayOwnedFault>()
            .map(|fault| fault.stage),
        Some("stage")
    );
}

#[derive(Clone, Copy)]
enum Step {
    Role,
    Text(&'static str),
    Reasoning,
    ToolStart(u64),
    ToolArgs(u64),
    Finish(&'static str),
    Usage,
    Error,
    Done,
}

fn step_frame(step: Step) -> String {
    let chunk = |delta: Value, finish: Value| {
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    };
    match step {
        Step::Role => chunk(json!({"role": "assistant"}), Value::Null),
        Step::Text(text) => chunk(json!({"content": text}), Value::Null),
        Step::Reasoning => chunk(json!({"reasoning_content": "r"}), Value::Null),
        Step::ToolStart(index) => chunk(
            json!({"tool_calls": [{"index": index, "id": format!("c{index}"), "type": "function", "function": {"name": "f", "arguments": ""}}]}),
            Value::Null,
        ),
        Step::ToolArgs(index) => chunk(
            json!({"tool_calls": [{"index": index, "function": {"arguments": "{}"}}]}),
            Value::Null,
        ),
        Step::Finish(reason) => chunk(json!({}), json!(reason)),
        Step::Usage => json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).to_string(),
        Step::Error => json!({"choices": [], "error": {"message": "m", "type": "server_error", "param": null, "code": "RESOURCE_EXHAUSTED"}}).to_string(),
        Step::Done => "[DONE]".to_string(),
    }
}

/// Every event sequence starts with `message_start`, opens one block at a
/// time with contiguous indices, sends deltas only to the open block with the
/// block's delta type, and ends with `message_delta` then `message_stop`, or
/// with one `error` event, and nothing after.
fn assert_event_grammar(events: &[Value], context: &str) {
    assert_eq!(events[0]["event"], "message_start", "{context}");
    let mut open: Option<(u64, String)> = None;
    let mut next_index = 0;
    let mut ended = false;
    for (position, event) in events.iter().enumerate().skip(1) {
        assert!(!ended, "{context}: event after the end at {position}");
        let data = &event["data"];
        match event["event"].as_str().unwrap() {
            "content_block_start" => {
                assert!(open.is_none(), "{context}: two open blocks");
                assert_eq!(data["index"], next_index, "{context}");
                next_index += 1;
                open = Some((
                    data["index"].as_u64().unwrap(),
                    data["content_block"]["type"].as_str().unwrap().to_string(),
                ));
            }
            "content_block_delta" => {
                let (index, kind) = open.clone().expect("delta without an open block");
                assert_eq!(data["index"], index, "{context}");
                let delta = data["delta"]["type"].as_str().unwrap();
                let allowed: &[&str] = match kind.as_str() {
                    "text" => &["text_delta"],
                    "thinking" => &["thinking_delta", "signature_delta"],
                    _ => &["input_json_delta"],
                };
                assert!(allowed.contains(&delta), "{context}: {delta} in {kind}");
            }
            "content_block_stop" => {
                let (index, _) = open.take().expect("stop without an open block");
                assert_eq!(data["index"], index, "{context}");
            }
            "message_delta" => {
                assert!(
                    open.is_none(),
                    "{context}: message_delta with an open block"
                );
                assert_eq!(events[position + 1]["event"], "message_stop", "{context}");
            }
            "message_stop" => ended = true,
            "error" => ended = true,
            "ping" => {}
            other => panic!("{context}: unexpected event {other}"),
        }
    }
    assert!(ended, "{context}: no terminal event");
}

#[test]
fn any_chat_chunk_sequence_yields_a_valid_event_grammar() {
    let alphabet = [
        Step::Role,
        Step::Text("a"),
        Step::Text(" "),
        Step::Text("\n"),
        Step::Reasoning,
        Step::ToolStart(0),
        Step::ToolStart(1),
        Step::ToolArgs(0),
        Step::ToolArgs(1),
        Step::Finish("stop"),
        Step::Finish("tool_calls"),
        Step::Finish("length"),
        Step::Usage,
        Step::Error,
        Step::Done,
    ];
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for round in 0..4000 {
        let length = (round % 12) + 1;
        let mut steps = Vec::with_capacity(length);
        for _ in 0..length {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            steps.push(alphabet[(seed % alphabet.len() as u64) as usize]);
        }
        let mut machine = StreamMachine::new("msg_x".to_string(), "m".to_string());
        let mut events = machine.start();
        for step in &steps {
            events.extend(machine.on_data(&step_frame(*step)));
        }
        events.extend(machine.on_end());
        let events: Vec<Value> = events
            .iter()
            .map(|event| json!({"event": event.name, "data": event.data}))
            .collect();
        assert_event_grammar(&events, &format!("round {round}"));
    }
}

#[tokio::test]
async fn a_silent_stream_sends_pings_until_output_arrives() {
    let (sender, receiver) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(4);
    let inner = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(receiver));
    let body = stream_body(
        inner,
        StreamMachine::new("msg_p".to_string(), "m".to_string()),
        Duration::from_millis(20),
    );
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(150)).await;
        for frame in [
            json!({"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]})
                .to_string(),
            json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
                .to_string(),
            "[DONE]".to_string(),
        ] {
            sender
                .send(Ok(Bytes::from(format!("data: {frame}\n\n"))))
                .await
                .unwrap();
        }
    });
    let events = parse_events(&to_bytes(body, usize::MAX).await.unwrap());
    let names = event_names(&events);
    assert_eq!(names[0], "message_start");
    let pings = names.iter().filter(|name| *name == "ping").count();
    assert!(pings >= 2, "{names:?}");
    let first_block = names
        .iter()
        .position(|name| name == "content_block_start")
        .unwrap();
    assert!(names[..first_block]
        .iter()
        .skip(1)
        .all(|name| name == "ping"));
    assert_eq!(names.last().unwrap(), "message_stop");
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn an_error_event_ends_the_body_and_drops_the_chat_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = DropFlag(Arc::clone(&dropped));
    let malformed = futures_util::stream::iter([Ok::<Bytes, std::io::Error>(Bytes::from_static(
        b"data: {not json\n\n",
    ))]);
    let inner_stream = malformed
        .chain(futures_util::stream::pending())
        .map(move |item| {
            let _keep = &flag;
            item
        });
    let body = stream_body(
        Body::from_stream(inner_stream.boxed()),
        StreamMachine::new("msg_e".to_string(), "m".to_string()),
        Duration::from_secs(60),
    );
    let collected = tokio::time::timeout(Duration::from_secs(5), to_bytes(body, usize::MAX))
        .await
        .expect("the body ends after the error event")
        .unwrap();
    assert_eq!(
        event_names(&parse_events(&collected)),
        ["message_start", "error"]
    );
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn dropping_the_messages_body_drops_the_chat_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = DropFlag(Arc::clone(&dropped));
    let inner_stream =
        futures_util::stream::pending::<Result<Bytes, std::io::Error>>().map(move |item| {
            let _keep = &flag;
            item
        });
    let body = stream_body(
        Body::from_stream(inner_stream.boxed()),
        StreamMachine::new("msg_d".to_string(), "m".to_string()),
        Duration::from_secs(60),
    );
    let mut data = body.into_data_stream();
    let first = data.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).starts_with("event: message_start"));
    assert!(!dropped.load(Ordering::SeqCst));
    drop(data);
    assert!(dropped.load(Ordering::SeqCst));
}

async fn chat_gateway(models: &[&str], loaded: &[&str]) -> TestGateway {
    let gateway = TestGateway::new(models).await;
    gateway
        .add_verified_worker("local-1", LOCAL_LANE, loaded)
        .await;
    gateway
}

#[tokio::test]
async fn a_buffered_message_is_served_by_the_chat_route() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        messages_request(json!({
            "model": "acme/chat",
            "max_tokens": 16,
            "metadata": {"user_id": "opaque"},
            "system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["request-id"], "request-1");
    assert_eq!(response.headers()["x-sie-request-id"], "request-1");
    let body = json_body(response).await;
    assert_eq!(
        body,
        json!({
            "id": "msg_request-1",
            "type": "message",
            "role": "assistant",
            "model": "acme/chat",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 1},
        })
    );
    let dispatched = gateway.dispatcher.dispatched();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].endpoint, "generate");
    assert_eq!(dispatched[0].model, "acme/chat");
}

#[tokio::test]
async fn a_streamed_message_is_served_by_the_chat_route() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        messages_request(json!({
            "model": "acme/chat",
            "max_tokens": 16,
            "stream": true,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}],
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["request-id"], "request-1");
    let events = events_of(response).await;
    assert_eq!(
        event_names(&events),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ]
    );
    assert_eq!(events[0]["data"]["message"]["id"], "msg_request-1");
    assert_eq!(events[2]["data"]["delta"]["text"], "ok");
    assert_eq!(
        events[4]["data"]["usage"],
        json!({"input_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 1})
    );
    assert_eq!(events[4]["data"]["delta"]["stop_reason"], "end_turn");
}

#[tokio::test]
async fn a_worker_failure_after_output_is_an_error_event() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    gateway.dispatcher.fail_stream(true);
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        messages_request(json!({
            "model": "acme/chat",
            "max_tokens": 16,
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = events_of(response).await;
    assert_eq!(
        event_names(&events),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error"
        ]
    );
    assert_eq!(events[3]["data"]["error"]["type"], "api_error");
}

#[tokio::test]
async fn chat_refusals_come_back_in_the_anthropic_envelope() {
    let gateway = chat_gateway(
        &[HYBRID_GENERATE_MODEL, CODER_MODEL],
        &["acme/chat", "acme/coder"],
    )
    .await;
    let cases = [
        (
            json!({"model": "acme/missing", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}),
            StatusCode::NOT_FOUND,
            "not_found_error",
            None,
        ),
        (
            json!({"model": "acme/chat", "max_tokens": 16, "tools": [{"name": "t", "input_schema": {"type": "object"}}], "messages": [{"role": "user", "content": "hi"}]}),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("tools: Model 'acme/chat' does not support tool calling"),
        ),
        (
            json!({"model": "acme/coder", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("max_tokens: 100 > 64, which is the maximum allowed number of output tokens for this model"),
        ),
        (
            json!({"model": "acme/chat", "max_tokens": 16, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}]}]}),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            Some("messages.0.content.0.image: Model 'acme/chat' does not support image input"),
        ),
    ];
    for (body, status, error_type, message) in cases {
        let response = proxy_messages(
            State(Arc::clone(&gateway.state)),
            messages_request(body.clone()),
        )
        .await;
        assert_eq!(response.status(), status, "{body}");
        let answer = json_body(response).await;
        assert_eq!(answer["type"], "error", "{body}");
        assert_eq!(answer["error"]["type"], error_type, "{body}");
        if let Some(message) = message {
            assert_eq!(answer["error"]["message"], message, "{body}");
        }
    }
    assert!(gateway.dispatcher.dispatched().is_empty());
}

#[tokio::test]
async fn a_tool_request_on_a_tool_model_is_dispatched() {
    let gateway = chat_gateway(&[CODER_MODEL], &["acme/coder"]).await;
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        messages_request(json!({
            "model": "acme/coder",
            "max_tokens": 64,
            "tools": [{"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "read it"}],
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(gateway.dispatcher.dispatched().len(), 1);
}

#[tokio::test]
async fn a_translation_refusal_publishes_nothing() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        messages_request(json!({
            "model": "acme/chat",
            "max_tokens": 16,
            "output_config": {"effort": "high"},
            "messages": [{"role": "user", "content": "hi"}],
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.headers()["x-sie-error-code"], "INVALID_REQUEST");
    let body = json_body(response).await;
    assert_eq!(
        body,
        json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "output_config.effort: Extra inputs are not permitted"},
            "request_id": null,
        })
    );
    assert!(gateway.dispatcher.dispatched().is_empty());
}

#[tokio::test]
async fn a_publish_failure_is_an_overloaded_error() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    gateway.dispatcher.refuse_generation();
    for stream in [false, true] {
        let response = proxy_messages(
            State(Arc::clone(&gateway.state)),
            messages_request(json!({
                "model": "acme/chat",
                "max_tokens": 16,
                "stream": stream,
                "messages": [{"role": "user", "content": "hi"}],
            })),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "stream={stream}"
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["type"], "overloaded_error", "stream={stream}");
    }
}

#[tokio::test]
async fn the_router_serves_messages_and_renders_its_refusals() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let mut config = (*gateway.state.config).clone();
    config.auth_mode = "token".to_string();
    config.auth_tokens = vec!["secret".to_string()];
    let router = crate::server::create_router(Arc::clone(&gateway.state), Arc::new(config));
    let body = json!({"model": "acme/chat", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]});

    let unauthenticated = router
        .clone()
        .oneshot(messages_request(body.clone()))
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        json_body(unauthenticated).await["error"]["type"],
        "authentication_error"
    );

    let authorized = |method: Method, uri: &str, body: Value| {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("authorization", "Bearer secret")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let served = router
        .clone()
        .oneshot(authorized(
            Method::POST,
            "/v1/messages?beta=true",
            body.clone(),
        ))
        .await
        .unwrap();
    assert_eq!(served.status(), StatusCode::OK);
    assert_eq!(json_body(served).await["content"][0]["text"], "ok");

    let count = router
        .clone()
        .oneshot(authorized(
            Method::POST,
            "/v1/messages/count_tokens",
            body.clone(),
        ))
        .await
        .unwrap();
    assert_eq!(count.status(), StatusCode::NOT_FOUND);
    assert_eq!(json_body(count).await["error"]["type"], "not_found_error");

    let wrong_method = router
        .clone()
        .oneshot(authorized(Method::GET, "/v1/messages", Value::Null))
        .await
        .unwrap();
    assert_eq!(wrong_method.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        json_body(wrong_method).await["error"]["type"],
        "invalid_request_error"
    );

    let other = router
        .oneshot(authorized(Method::POST, "/v1/unknown", body))
        .await
        .unwrap();
    assert_eq!(other.status(), StatusCode::NOT_FOUND);
    let other_body = to_bytes(other.into_body(), usize::MAX).await.unwrap();
    assert!(
        serde_json::from_slice::<Value>(&other_body)
            .map_or(true, |value| value.get("type") != Some(&json!("error"))),
        "only the Messages paths take the Anthropic envelope"
    );
}

#[test]
fn dispatch_inputs_compose_the_translation_with_the_chat_seam() {
    let (model, params) = messages_dispatch_inputs(&json!({
        "model": "acme/chat",
        "max_tokens": 48,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    assert_eq!(model, "acme/chat");
    assert_eq!(params.generate.as_ref().unwrap().max_new_tokens, 48);
    let refused = messages_dispatch_inputs(&json!({"model": "acme/chat"})).unwrap_err();
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn an_oversized_body_is_request_too_large() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let padding = "x".repeat(super::MAX_MESSAGES_BODY + 1);
    let response = proxy_messages(
        State(Arc::clone(&gateway.state)),
        Request::builder()
            .method(Method::POST)
            .uri("/v1/messages")
            .body(Body::from(padding))
            .unwrap(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        json_body(response).await["error"]["type"],
        "request_too_large"
    );
}
