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

use super::error::{responses_error_response, ResponsesError};
use super::proxy_responses;
use super::request::responses_to_chat;
use super::response::{chat_completion_to_response, Echo, ResponseShell};
use super::stream::{stream_body, Frame, FrameParser, StreamMachine};
use crate::handlers::proxy::{
    chat_completions_dispatch_inputs, compat_request_body_limit, responses_dispatch_inputs,
};
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
        "{}/../../conformance/openai_responses/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn shell_for(request: &Value, request_id: &str) -> ResponseShell {
    let echo = responses_to_chat(request).unwrap().echo;
    ResponseShell::new(echo, request_id.to_string(), 0)
}

fn responses_request(body: Value) -> Request {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn json_body(response: Response) -> Value {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// The events of an SSE body, and how many comment frames it carried.
fn parse_events(bytes: &[u8]) -> (Vec<Value>, usize) {
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let mut comments = 0;
    let mut events = Vec::new();
    for frame in text.split("\n\n").filter(|frame| !frame.trim().is_empty()) {
        if frame.lines().all(|line| line.starts_with(':')) {
            comments += 1;
            continue;
        }
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
        events.push(json!({"event": name, "data": data}));
    }
    (events, comments)
}

async fn events_of(response: Response) -> Vec<Value> {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    parse_events(&bytes).0
}

fn event_names(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|event| event["event"].as_str().unwrap().to_string())
        .collect()
}

fn assert_refusal(error: &ResponsesError, expect: &Value, name: &str) {
    assert_eq!(
        u64::from(error.status.as_u16()),
        expect["status"].as_u64().unwrap(),
        "{name}"
    );
    assert_eq!(error.error_type, expect["type"], "{name}");
    assert_eq!(error.code, expect["code"], "{name}");
    assert_eq!(error.param.as_deref(), expect["param"].as_str(), "{name}");
    assert_eq!(error.message, expect["message"], "{name}");
}

#[test]
fn request_vectors_translate_onto_the_chat_body() {
    let vectors = vectors("request_vectors.json");
    for case in vectors["accepted"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let translation =
            responses_to_chat(&case["body"]).unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(translation.chat, case["chat"], "{name}");
        assert!(
            chat_completions_dispatch_inputs(&translation.chat).is_ok(),
            "{name}: the chat route accepts the translation"
        );
    }
    for case in vectors["rejected"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let error = responses_to_chat(&case["body"]).expect_err(name);
        assert_refusal(&error, &case["expect"], name);
    }
}

#[test]
fn every_captured_codex_request_translates_or_is_refused_as_expected() {
    let fixture = vectors("codex_requests.json");
    let requests = fixture["requests"].as_array().unwrap();
    assert!(!requests.is_empty());
    for case in requests {
        let name = case["name"].as_str().unwrap();
        let expect = &case["expect"];
        match responses_to_chat(&case["body"]) {
            Ok(translation) => {
                let summary = &expect["chat"];
                assert!(summary.is_object(), "{name}: expected a refusal");
                let chat = &translation.chat;
                let messages = chat["messages"].as_array().unwrap();
                let roles: Vec<&Value> = messages.iter().map(|message| &message["role"]).collect();
                let expected_roles: Vec<&Value> =
                    summary["roles"].as_array().unwrap().iter().collect();
                assert_eq!(roles, expected_roles, "{name}");
                let image_parts = messages
                    .iter()
                    .filter_map(|message| message["content"].as_array())
                    .flatten()
                    .filter(|part| part["type"] == "image_url")
                    .count();
                assert_eq!(
                    image_parts as u64,
                    summary["image_parts"].as_u64().unwrap(),
                    "{name}"
                );
                assert_eq!(
                    chat["tools"].as_array().map_or(0, Vec::len) as u64,
                    summary["tool_count"].as_u64().unwrap(),
                    "{name}"
                );
                for key in [
                    "stream",
                    "tool_choice",
                    "parallel_tool_calls",
                    "include_reasoning",
                    "reasoning_effort",
                ] {
                    assert_eq!(
                        chat.get(key).unwrap_or(&Value::Null),
                        summary.get(key).unwrap_or(&Value::Null),
                        "{name}: {key}"
                    );
                }
                assert!(
                    chat_completions_dispatch_inputs(chat).is_ok(),
                    "{name}: the chat route accepts the translation"
                );
            }
            Err(error) => {
                assert!(
                    expect["refused"].is_object(),
                    "{name}: unexpected refusal {}",
                    error.message
                );
                assert_refusal(&error, &expect["refused"], name);
            }
        }
    }
}

#[test]
fn response_vectors_translate_the_chat_completion() {
    let vectors = vectors("response_vectors.json");
    for case in vectors["translated"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        assert_eq!(case["request"]["model"], case["model"], "{name}");
        let shell = shell_for(&case["request"], case["request_id"].as_str().unwrap());
        let response = chat_completion_to_response(&case["chat"], &shell)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(response, case["response"], "{name}");
    }
}

#[test]
fn a_chat_completion_the_translation_cannot_read_is_a_fault() {
    let shell = shell_for(&json!({"model": "m", "input": "x"}), "r");
    for chat in [
        json!({}),
        json!({"choices": []}),
        json!({"choices": [{"index": 0}]}),
        json!({"choices": [{"message": {"tool_calls": [{"function": {"name": "f"}}]}}]}),
    ] {
        assert!(
            chat_completion_to_response(&chat, &shell).is_err(),
            "{chat}"
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
        let mut machine = StreamMachine::new(shell_for(
            &case["request"],
            case["request_id"].as_str().unwrap(),
        ));
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
        let mut sse = String::from(":\n\n");
        for frame in stream_frames(case) {
            sse.push_str(&format!("data: {frame}\n\n"));
            sse.push_str(": keep-alive\n\n");
        }
        let bytes = sse.into_bytes();
        let pieces: Vec<Result<Bytes, std::io::Error>> = bytes
            .chunks(7)
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect();
        let inner = Body::from_stream(futures_util::stream::iter(pieces));
        let machine = StreamMachine::new(shell_for(
            &case["request"],
            case["request_id"].as_str().unwrap(),
        ));
        let collected = to_bytes(stream_body(inner, machine), usize::MAX)
            .await
            .unwrap();
        let (events, comments) = parse_events(&collected);
        assert_eq!(&Value::Array(events), &case["events"], "{name}");
        assert!(comments >= 1, "{name}: heartbeats pass through");
        let text = String::from_utf8(collected.to_vec()).unwrap();
        let terminal = text.rfind("event: response.").unwrap();
        assert!(
            !text[terminal..].contains("\n\n:"),
            "{name}: nothing follows the terminal event"
        );
        assert!(!text.contains("[DONE]"), "{name}");
    }
}

#[test]
fn frame_parser_joins_multi_line_data_and_keeps_comments() {
    let mut parser = FrameParser::default();
    assert_eq!(
        parser.push(b": keep-alive\r\n\r\ndata: {\"a\""),
        vec![Frame::Comment(": keep-alive".to_string())]
    );
    assert_eq!(
        parser.push(b":1}\r\n\r\nevent: x\ndata: one\ndata: two\n\n:\n\n"),
        vec![
            Frame::Data("{\"a\":1}".to_string()),
            Frame::Data("one\ntwo".to_string()),
            Frame::Comment(":".to_string()),
        ]
    );
}

#[tokio::test]
async fn error_vectors_keep_the_chat_error_with_the_responses_param() {
    let vectors = vectors("error_vectors.json");
    for case in vectors["errors"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut response = Response::new(Body::from(case["body"].to_string()));
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
        let translated = responses_error_response(response).await;
        let expect = &case["expect"];
        assert_eq!(
            u64::from(translated.status().as_u16()),
            expect["status"].as_u64().unwrap(),
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
                    .map(|value| value.to_str().unwrap()),
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
        assert_eq!(json_body(translated).await, expect["body"], "{name}");
    }
}

#[tokio::test]
async fn an_inner_error_keeps_its_response_extensions() {
    let mut response = Response::new(Body::from(
        json!({"error": {"message": "failed", "type": "server_error", "param": "max_completion_tokens", "code": "x"}})
            .to_string(),
    ));
    *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
    let response = crate::http_error::GatewayOwnedFault::new("stage").mark(response);
    let translated = responses_error_response(response).await;
    assert_eq!(
        translated
            .extensions()
            .get::<crate::http_error::GatewayOwnedFault>()
            .map(|fault| fault.stage),
        Some("stage")
    );
    assert_eq!(
        json_body(translated).await["error"]["param"],
        "max_output_tokens"
    );
}

#[derive(Clone, Copy)]
enum Step {
    Role,
    Text(&'static str),
    ToolStart(u64, &'static str),
    ToolArgs(u64),
    Finish(&'static str),
    Usage,
    Error,
    Malformed,
    Done,
}

fn step_frame(step: Step) -> String {
    let chunk = |delta: Value, finish: Value| {
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    };
    match step {
        Step::Role => chunk(json!({"role": "assistant"}), Value::Null),
        Step::Text(text) => chunk(json!({"content": text}), Value::Null),
        Step::ToolStart(index, name) => chunk(
            json!({"tool_calls": [{"index": index, "id": format!("c{index}"), "type": "function", "function": {"name": name, "arguments": ""}}]}),
            Value::Null,
        ),
        Step::ToolArgs(index) => chunk(
            json!({"tool_calls": [{"index": index, "function": {"arguments": "{\"input\":\"x\"}"}}]}),
            Value::Null,
        ),
        Step::Finish(reason) => chunk(json!({}), json!(reason)),
        Step::Usage => json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).to_string(),
        Step::Error => json!({"choices": [], "error": {"message": "m", "type": "server_error", "param": null, "code": "RESOURCE_EXHAUSTED"}}).to_string(),
        Step::Malformed => "{not json".to_string(),
        Step::Done => "[DONE]".to_string(),
    }
}

/// Every event sequence opens with `response.created` and
/// `response.in_progress`, numbers its events from 0 without gaps, has at
/// most one open output item with contiguous output indices, sends item
/// events only to the open item and with that item's kind, and ends with
/// exactly one terminal event and nothing after it. A completed or
/// incomplete response has closed every item it added, and every terminal
/// response lists exactly the items done so far.
fn assert_event_grammar(events: &[Value], context: &str) {
    let names = event_names(events);
    assert_eq!(
        names[..2],
        ["response.created", "response.in_progress"],
        "{context}"
    );
    let mut open: Option<(Value, Value, String)> = None;
    let mut next_index = 0;
    let mut done = Vec::new();
    let mut terminal = None;
    for (position, event) in events.iter().enumerate() {
        let data = &event["data"];
        assert_eq!(data["sequence_number"], position as u64, "{context}");
        assert!(
            terminal.is_none(),
            "{context}: event after the end at {position}"
        );
        let name = event["event"].as_str().unwrap();
        let item_kind =
            match name {
                "response.created" | "response.in_progress" => {
                    assert!(position < 2, "{context}");
                    continue;
                }
                "response.output_item.added" => {
                    assert!(open.is_none(), "{context}: two open items");
                    assert_eq!(data["output_index"], next_index, "{context}");
                    assert_eq!(data["item"]["status"], "in_progress", "{context}");
                    next_index += 1;
                    open = Some((
                        data["output_index"].clone(),
                        data["item"]["id"].clone(),
                        data["item"]["type"].as_str().unwrap().to_string(),
                    ));
                    continue;
                }
                "response.output_item.done" => {
                    let (index, id, kind) = open.take().expect("done without an open item");
                    assert_eq!(data["output_index"], index, "{context}");
                    assert_eq!(data["item"]["id"], id, "{context}");
                    assert_eq!(data["item"]["type"], kind.as_str(), "{context}");
                    assert_eq!(data["item"]["status"], "completed", "{context}");
                    done.push(data["item"].clone());
                    continue;
                }
                "response.completed" | "response.incomplete" | "response.failed" => {
                    if name != "response.failed" {
                        assert!(open.is_none(), "{context}: {name} with an open item");
                    }
                    assert_eq!(
                        data["response"]["output"],
                        Value::Array(done.clone()),
                        "{context}"
                    );
                    terminal = Some(name);
                    continue;
                }
                "response.content_part.added"
                | "response.output_text.delta"
                | "response.output_text.done"
                | "response.content_part.done" => "message",
                "response.function_call_arguments.delta"
                | "response.function_call_arguments.done" => "function_call",
                "response.custom_tool_call_input.delta"
                | "response.custom_tool_call_input.done" => "custom_tool_call",
                other => panic!("{context}: unexpected event {other}"),
            };
        let (index, id, kind) = open.clone().expect("item event without an open item");
        assert_eq!(kind, item_kind, "{context}: {name}");
        assert_eq!(data["output_index"], index, "{context}");
        assert_eq!(data["item_id"], id, "{context}");
    }
    assert!(terminal.is_some(), "{context}: no terminal event");
}

#[test]
fn any_chat_chunk_sequence_yields_a_well_formed_event_sequence() {
    let alphabet = [
        Step::Role,
        Step::Text("a"),
        Step::Text(" "),
        Step::ToolStart(0, "f"),
        Step::ToolStart(1, "patch"),
        Step::ToolStart(2, "f"),
        Step::ToolArgs(0),
        Step::ToolArgs(1),
        Step::ToolArgs(2),
        Step::Finish("stop"),
        Step::Finish("tool_calls"),
        Step::Finish("length"),
        Step::Usage,
        Step::Error,
        Step::Malformed,
        Step::Done,
    ];
    let echo = Echo {
        model: "m".to_string(),
        parallel_tool_calls: json!(true),
        tool_choice: json!("auto"),
        custom_tools: vec!["patch".to_string()],
    };
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    for round in 0..4000 {
        let length = (round % 14) + 1;
        let mut steps = Vec::with_capacity(length);
        for _ in 0..length {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            steps.push(alphabet[(seed % alphabet.len() as u64) as usize]);
        }
        let mut machine = StreamMachine::new(ResponseShell::new(echo.clone(), "x".to_string(), 0));
        let mut events = machine.start();
        for step in &steps {
            events.extend(machine.on_data(&step_frame(*step)));
        }
        events.extend(if round % 5 == 0 {
            machine.on_transport_error()
        } else {
            machine.on_end()
        });
        let events: Vec<Value> = events
            .iter()
            .map(|event| json!({"event": event.name, "data": event.data}))
            .collect();
        assert_event_grammar(&events, &format!("round {round}"));
    }
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn plain_shell() -> ResponseShell {
    shell_for(&json!({"model": "m", "input": "x", "stream": true}), "r")
}

#[tokio::test]
async fn a_failed_event_ends_the_body_and_drops_the_chat_body() {
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
        StreamMachine::new(plain_shell()),
    );
    let collected = tokio::time::timeout(Duration::from_secs(5), to_bytes(body, usize::MAX))
        .await
        .expect("the body ends after the failed event")
        .unwrap();
    let events = parse_events(&collected).0;
    assert_eq!(
        event_names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.failed"
        ]
    );
    assert_eq!(
        events[2]["data"]["response"]["error"],
        json!({"code": "server_error", "message": "the generation stream sent a malformed chunk"})
    );
    assert!(dropped.load(Ordering::SeqCst));
}

#[test]
fn tool_call_deltas_the_translation_cannot_place_fail_the_response() {
    let cases = [
        (
            vec![
                Step::ToolStart(0, "f"),
                Step::Text("a"),
                Step::ToolStart(0, "f"),
            ],
            None,
        ),
        (
            vec![Step::ToolStart(0, "f"), Step::Text("a"), Step::ToolArgs(0)],
            Some("tool call arguments arrived after another output item started"),
        ),
        (
            vec![
                Step::ToolStart(0, "f"),
                Step::ToolStart(1, "f"),
                Step::ToolArgs(0),
            ],
            Some("tool call arguments arrived after another output item started"),
        ),
        (
            vec![Step::ToolArgs(0)],
            Some("a tool call arrived without its id and name"),
        ),
    ];
    for (case, (steps, failure)) in cases.into_iter().enumerate() {
        let mut machine = StreamMachine::new(plain_shell());
        let mut events = machine.start();
        for step in &steps {
            events.extend(machine.on_data(&step_frame(*step)));
        }
        let failed = events.iter().find(|event| event.name == "response.failed");
        assert_eq!(
            failed.map(|event| event.data["response"]["error"].clone()),
            failure.map(|message| json!({"code": "server_error", "message": message})),
            "case {case}"
        );
    }
}

#[tokio::test]
async fn a_broken_chat_stream_fails_the_response() {
    let broken = futures_util::stream::iter([
        Ok::<Bytes, std::io::Error>(Bytes::from_static(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n",
        )),
        Err(std::io::Error::other("reset")),
    ]);
    let body = stream_body(Body::from_stream(broken), StreamMachine::new(plain_shell()));
    let events = parse_events(&to_bytes(body, usize::MAX).await.unwrap()).0;
    let last = events.last().unwrap();
    assert_eq!(last["event"], "response.failed");
    assert_eq!(
        last["data"]["response"]["error"],
        json!({"code": "server_error", "message": "the generation stream ended unexpectedly"})
    );
}

#[tokio::test]
async fn dropping_the_responses_body_drops_the_chat_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let flag = DropFlag(Arc::clone(&dropped));
    let inner_stream =
        futures_util::stream::pending::<Result<Bytes, std::io::Error>>().map(move |item| {
            let _keep = &flag;
            item
        });
    let body = stream_body(
        Body::from_stream(inner_stream.boxed()),
        StreamMachine::new(plain_shell()),
    );
    let mut data = body.into_data_stream();
    let first = data.next().await.unwrap().unwrap();
    assert!(String::from_utf8_lossy(&first).starts_with("event: response.created"));
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

fn settle_created_at(response: &mut Value) {
    assert!(
        response["created_at"].as_u64().is_some_and(|at| at > 0),
        "{response}"
    );
    response["created_at"] = json!(0);
}

#[tokio::test]
async fn a_buffered_response_is_served_by_the_chat_route() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_responses(
        State(Arc::clone(&gateway.state)),
        responses_request(json!({
            "model": "acme/chat",
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "store": false,
            "include": ["reasoning.encrypted_content"],
            "client_metadata": {"session_id": "s"},
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-sie-request-id"], "request-1");
    assert_eq!(response.headers()["content-type"], "application/json");
    let mut body = json_body(response).await;
    settle_created_at(&mut body);
    assert_eq!(
        body,
        json!({
            "id": "resp-request-1",
            "object": "response",
            "created_at": 0,
            "model": "acme/chat",
            "status": "completed",
            "incomplete_details": null,
            "error": null,
            "store": false,
            "parallel_tool_calls": true,
            "tool_choice": "auto",
            "output": [{
                "type": "message",
                "id": "msg-request-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        })
    );
    let dispatched = gateway.dispatcher.dispatched();
    assert_eq!(dispatched.len(), 1);
    assert_eq!(dispatched[0].endpoint, "generate");
    assert_eq!(dispatched[0].model, "acme/chat");
}

#[tokio::test]
async fn a_streamed_response_is_served_by_the_chat_route() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_responses(
        State(Arc::clone(&gateway.state)),
        responses_request(json!({"model": "acme/chat", "input": "hi", "stream": true})),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    assert_eq!(response.headers()["x-sie-request-id"], "request-1");
    assert!(response.headers().get("content-length").is_none());
    let events = events_of(response).await;
    assert_eq!(
        event_names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert_event_grammar(&events, "served stream");
    let mut completed = events[8]["data"]["response"].clone();
    settle_created_at(&mut completed);
    assert_eq!(completed["id"], "resp-request-1");
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["output"][0]["id"], "msg-request-1");
    assert_eq!(completed["output"][0]["content"][0]["text"], "ok");
    assert_eq!(completed["usage"]["total_tokens"], 2);
}

#[tokio::test]
async fn a_worker_failure_after_output_is_a_failed_event() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    gateway.dispatcher.fail_stream(true);
    let response = proxy_responses(
        State(Arc::clone(&gateway.state)),
        responses_request(json!({"model": "acme/chat", "input": "hi", "stream": true})),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let events = events_of(response).await;
    assert_eq!(
        event_names(&events),
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.failed",
        ]
    );
    let failed = &events[5]["data"]["response"];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["output"], json!([]));
    assert!(failed["error"]["code"].is_string(), "{failed}");
    assert!(failed["error"]["message"].is_string(), "{failed}");
}

#[tokio::test]
async fn chat_refusals_name_the_responses_field() {
    let gateway = chat_gateway(
        &[HYBRID_GENERATE_MODEL, CODER_MODEL],
        &["acme/chat", "acme/coder"],
    )
    .await;
    let function =
        |name: &str| json!({"type": "function", "name": name, "parameters": {"type": "object"}});
    let cases = [
        (
            json!({"model": "acme/missing", "input": "hi"}),
            StatusCode::NOT_FOUND,
            "model",
        ),
        (
            json!({"model": "acme/chat", "input": "hi", "tools": [function("t")]}),
            StatusCode::BAD_REQUEST,
            "tools",
        ),
        (
            json!({"model": "acme/coder", "input": "hi", "max_output_tokens": 100}),
            StatusCode::BAD_REQUEST,
            "max_output_tokens",
        ),
        (
            json!({"model": "acme/coder", "input": "hi", "tools": [function("ok"), function("two words")]}),
            StatusCode::BAD_REQUEST,
            "tools[1].name",
        ),
    ];
    for (body, status, param) in cases {
        let response = proxy_responses(
            State(Arc::clone(&gateway.state)),
            responses_request(body.clone()),
        )
        .await;
        assert_eq!(response.status(), status, "{body}");
        let answer = json_body(response).await;
        assert_eq!(answer["error"]["param"], param, "{body}: {answer}");
        assert!(answer["error"]["code"].is_string(), "{body}");
    }
    assert!(gateway.dispatcher.dispatched().is_empty());
}

#[tokio::test]
async fn a_translation_refusal_publishes_nothing() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let response = proxy_responses(
        State(Arc::clone(&gateway.state)),
        responses_request(json!({
            "model": "acme/chat",
            "input": "hi",
            "previous_response_id": "resp-1",
        })),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        json_body(response).await,
        json!({"error": {
            "message": "previous_response_id: SIE does not store responses; send the full conversation in input",
            "type": "invalid_request_error",
            "param": "previous_response_id",
            "code": "unsupported_field",
        }})
    );
    assert!(gateway.dispatcher.dispatched().is_empty());
}

#[tokio::test]
async fn capacity_is_a_retryable_refusal_before_the_stream_opens() {
    for stream in [false, true] {
        let gateway = TestGateway::new(&[HYBRID_GENERATE_MODEL]).await;
        let response = proxy_responses(
            State(Arc::clone(&gateway.state)),
            responses_request(json!({"model": "acme/chat", "input": "hi", "stream": stream})),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "stream={stream}"
        );
        assert!(
            response.headers().get("retry-after").is_some(),
            "stream={stream}"
        );
        assert!(
            response.headers().get("x-sie-error-code").is_some(),
            "stream={stream}"
        );
        let body = json_body(response).await;
        assert!(body["error"]["code"].is_string(), "stream={stream}: {body}");
    }
}

#[tokio::test]
async fn the_router_serves_responses_and_no_stored_response_routes() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let mut config = (*gateway.state.config).clone();
    config.auth_mode = "token".to_string();
    config.auth_tokens = vec!["secret".to_string()];
    let router = crate::server::create_router(Arc::clone(&gateway.state), Arc::new(config));
    let body = json!({"model": "acme/chat", "input": "hi"});

    let unauthenticated = router
        .clone()
        .oneshot(responses_request(body.clone()))
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

    let authorized = |method: Method, uri: &str, body: &Value| {
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
        .oneshot(authorized(Method::POST, "/v1/responses", &body))
        .await
        .unwrap();
    assert_eq!(served.status(), StatusCode::OK);
    assert_eq!(
        json_body(served).await["output"][0]["content"][0]["text"],
        "ok"
    );
    for (method, uri) in [
        (Method::GET, "/v1/responses/resp-request-1"),
        (Method::POST, "/v1/responses/resp-request-1/cancel"),
        (Method::POST, "/v1/responses/input_tokens"),
        (Method::POST, "/v1/responses/compact"),
    ] {
        let answer = router
            .clone()
            .oneshot(authorized(method, uri, &body))
            .await
            .unwrap();
        assert_eq!(answer.status(), StatusCode::NOT_FOUND, "{uri}");
    }
    let wrong_method = router
        .oneshot(authorized(Method::GET, "/v1/responses", &Value::Null))
        .await
        .unwrap();
    assert_eq!(wrong_method.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[test]
fn dispatch_inputs_compose_the_translation_with_the_chat_seam() {
    let (model, params) = responses_dispatch_inputs(&json!({
        "model": "acme/chat",
        "max_output_tokens": 48,
        "input": "hi",
    }))
    .unwrap();
    assert_eq!(model, "acme/chat");
    assert_eq!(params.generate.as_ref().unwrap().max_new_tokens, 48);
    let refused = responses_dispatch_inputs(&json!({"model": "acme/chat"})).unwrap_err();
    assert_eq!(refused.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unreadable_bodies_are_refused_in_the_openai_envelope() {
    let gateway = chat_gateway(&[HYBRID_GENERATE_MODEL], &["acme/chat"]).await;
    let limit = compat_request_body_limit("/v1/responses");
    for (body, status, message) in [
        (
            Body::from("x".repeat(limit + 1)),
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Request body too large (max {limit} bytes)"),
        ),
        (
            Body::from("{"),
            StatusCode::BAD_REQUEST,
            "invalid JSON: EOF while parsing an object at line 1 column 1".to_string(),
        ),
    ] {
        let response = proxy_responses(
            State(Arc::clone(&gateway.state)),
            Request::builder()
                .method(Method::POST)
                .uri("/v1/responses")
                .body(body)
                .unwrap(),
        )
        .await;
        assert_eq!(response.status(), status);
        let answer = json_body(response).await;
        assert_eq!(answer["error"]["message"], message);
        assert_eq!(answer["error"]["type"], "invalid_request_error");
        assert_eq!(answer["error"]["code"], "invalid_request");
    }
    assert!(gateway.dispatcher.dispatched().is_empty());
}
