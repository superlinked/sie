//! OpenAI-compatible text decisions over the existing System One dispatch path.
//!
//! The public questions and answers are ordered arrays. Synthetic native ids
//! preserve unnamed/repeated question names and distinguish boolean choices
//! from strings, without exposing backend options to the caller. The shared
//! `wire-fixtures/decisions.json` pins parity with the single-server route.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{header, response::Parts, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};

use crate::http_error::GatewayOwnedFault;
use crate::server::AppState;

use super::proxy::{
    is_openai_compat_forwarded_header, is_openai_compat_inner_request_header,
    is_valid_compat_model_id, request_body_error,
};
use super::systemone::{self, MAX_SCORE_LEVELS};

const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 1024 * 1024;
const MAX_QUESTIONS: usize = 200;
const MAX_CHOICES: usize = 255;
const MAX_MESSAGES: usize = 4096;
const MAX_FIELD_CHARS: usize = 1_048_576;
const PROBABILITY_TOLERANCE: f64 = 1e-4;

#[derive(Debug)]
struct Invalid {
    message: String,
    param: Option<String>,
    code: &'static str,
}

fn invalid(message: impl Into<String>, param: &str) -> Invalid {
    Invalid {
        message: message.into(),
        param: (!param.is_empty()).then(|| param.to_string()),
        code: "invalid_request",
    }
}

fn text<'a>(value: Option<&'a Value>, param: &str) -> Result<&'a str, Invalid> {
    let value = value
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("must be a string", param))?;
    if value.chars().count() > MAX_FIELD_CHARS {
        return Err(invalid(
            format!("must contain at most {MAX_FIELD_CHARS} characters"),
            param,
        ));
    }
    Ok(value)
}

fn fields(value: &Map<String, Value>, allowed: &[&str], param: &str) -> Result<(), Invalid> {
    for key in value.keys() {
        if !allowed.contains(&key.as_str()) {
            let param = if param.is_empty() {
                key.clone()
            } else {
                format!("{param}.{key}")
            };
            return Err(Invalid {
                code: "unsupported_field",
                ..invalid("unsupported field", &param)
            });
        }
    }
    Ok(())
}

fn validate_input(value: Option<&Value>) -> Result<(), Invalid> {
    if value.is_some_and(Value::is_string) {
        return Ok(());
    }
    let messages = value
        .and_then(Value::as_array)
        .filter(|messages| (1..=MAX_MESSAGES).contains(&messages.len()))
        .ok_or_else(|| {
            invalid(
                format!("must be text or 1 to {MAX_MESSAGES} text messages"),
                "input",
            )
        })?;
    for (index, message) in messages.iter().enumerate() {
        let param = format!("input[{index}]");
        let message = message
            .as_object()
            .ok_or_else(|| invalid("must be a message object", &param))?;
        fields(message, &["type", "role", "content", "phase"], &param)?;
        if message.get("type").is_some_and(|kind| kind != "message") {
            return Err(invalid(
                "SIE decision models support text messages, not tool calls or media",
                &format!("{param}.type"),
            ));
        }
        let role = message.get("role").and_then(Value::as_str);
        if !matches!(role, Some("user" | "developer" | "system" | "assistant")) {
            return Err(invalid(
                "unsupported message role",
                &format!("{param}.role"),
            ));
        }
        if message.get("phase").is_some_and(|phase| {
            !phase.is_null() && phase != "commentary" && phase != "final_answer"
        }) {
            return Err(invalid(
                "unsupported message phase",
                &format!("{param}.phase"),
            ));
        }
        let content = message.get("content");
        if content.is_some_and(Value::is_string) {
            continue;
        }
        let parts = content
            .and_then(Value::as_array)
            .filter(|parts| !parts.is_empty())
            .ok_or_else(|| {
                invalid(
                    "must be text or a non-empty list of text parts",
                    &format!("{param}.content"),
                )
            })?;
        for (index, part) in parts.iter().enumerate() {
            let param = format!("{param}.content[{index}]");
            let part = part
                .as_object()
                .ok_or_else(|| invalid("must be a text part", &param))?;
            let kind = part.get("type").and_then(Value::as_str);
            let supported = if role == Some("assistant") {
                matches!(kind, Some("output_text" | "refusal"))
            } else {
                kind == Some("input_text")
            };
            if !supported {
                return Err(invalid(
                    "SIE decision models read text only; unsupported content part",
                    &format!("{param}.type"),
                ));
            }
            let field = if kind == Some("refusal") {
                "refusal"
            } else {
                "text"
            };
            fields(part, &["type", field], &param)?;
            if !part.get(field).is_some_and(Value::is_string) {
                return Err(invalid("must be a string", &format!("{param}.{field}")));
            }
        }
    }
    Ok(())
}

/// The native request and the original question array needed to render answers.
fn normalize_request(parsed: &Value) -> Result<(Value, Vec<Value>), Invalid> {
    let body = parsed
        .as_object()
        .ok_or_else(|| invalid("request body must be a JSON object", ""))?;
    fields(
        body,
        &["model", "input", "questions", "safety_identifier"],
        "",
    )?;
    let model = text(body.get("model"), "model")?;
    if !is_valid_compat_model_id(model) {
        return Err(invalid("invalid model id", "model"));
    }
    if let Some(identifier) = body
        .get("safety_identifier")
        .filter(|value| !value.is_null())
    {
        if text(Some(identifier), "safety_identifier")?.chars().count() > 128 {
            return Err(invalid(
                "must contain at most 128 characters",
                "safety_identifier",
            ));
        }
    }
    validate_input(body.get("input"))?;
    let questions = body
        .get("questions")
        .and_then(Value::as_array)
        .filter(|questions| (1..=MAX_QUESTIONS).contains(&questions.len()))
        .ok_or_else(|| {
            invalid(
                format!("must contain 1 to {MAX_QUESTIONS} questions"),
                "questions",
            )
        })?;
    let mut native = Map::new();
    for (index, question) in questions.iter().enumerate() {
        let param = format!("questions[{index}]");
        let question = question
            .as_object()
            .ok_or_else(|| invalid("must be a question object", &param))?;
        let kind = question.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("predicate" | "choice" | "score")) {
            return Err(invalid(
                "must be predicate, choice, or score",
                &format!("{param}.type"),
            ));
        }
        let kind = kind.unwrap();
        let mut allowed = vec!["type", "name", "instructions"];
        if kind == "choice" {
            allowed.push("choices");
        } else if kind == "score" {
            allowed.push("levels");
        }
        fields(question, &allowed, &param)?;
        let instructions = text(
            question.get("instructions"),
            &format!("{param}.instructions"),
        )?;
        if let Some(name) = question.get("name") {
            text(Some(name), &format!("{param}.name"))?;
        }
        let mut normalized = json!({
            "type": if kind == "predicate" { "noul" } else { kind },
            "instructions": instructions,
        });
        if kind != "predicate" {
            let (field, minimum, maximum) = if kind == "choice" {
                ("choices", 2, MAX_CHOICES)
            } else {
                ("levels", 1, MAX_SCORE_LEVELS)
            };
            let options = question
                .get(field)
                .and_then(Value::as_array)
                .filter(|options| (minimum..=maximum).contains(&options.len()))
                .ok_or_else(|| {
                    invalid(
                        format!("must contain {minimum} to {maximum} entries"),
                        &format!("{param}.{field}"),
                    )
                })?;
            let mut descriptions = Vec::new();
            let mut seen = Vec::new();
            for (index, option) in options.iter().enumerate() {
                let param = format!("{param}.{field}[{index}]");
                let option = option
                    .as_object()
                    .ok_or_else(|| invalid("must be an object", &param))?;
                let value_field = if kind == "choice" { "value" } else { "label" };
                fields(option, &[value_field, "description"], &param)?;
                let value = option.get(value_field).unwrap_or(&Value::Null);
                if kind == "choice" && !matches!(value, Value::String(_) | Value::Bool(_)) {
                    return Err(invalid(
                        "must be a string or boolean",
                        &format!("{param}.value"),
                    ));
                }
                if kind == "score" || value.is_string() {
                    text(Some(value), &format!("{param}.{value_field}"))?;
                }
                if kind == "choice" {
                    if seen.contains(&value) {
                        return Err(invalid(
                            "choice values must be unique",
                            &format!("{param}.value"),
                        ));
                    }
                    seen.push(value);
                }
                let description = option
                    .get("description")
                    .map(|value| text(Some(value), &format!("{param}.description")))
                    .transpose()?;
                let label = if kind == "choice" {
                    value.to_string()
                } else {
                    value.as_str().unwrap().to_string()
                };
                descriptions.push(match description.filter(|value| !value.is_empty()) {
                    Some(description) => format!("{label}: {description}"),
                    None => label,
                });
            }
            normalized["criteria"] = if kind == "choice" {
                Value::Object(
                    descriptions
                        .into_iter()
                        .enumerate()
                        .map(|(index, description)| (format!("option_{index}"), json!(description)))
                        .collect(),
                )
            } else {
                json!(descriptions)
            };
        }
        native.insert(format!("question_{index}"), normalized);
    }
    Ok((
        json!({"model": model, "state": body["input"], "questions": native}),
        questions.clone(),
    ))
}

fn probability(value: &Value) -> Result<f64, &'static str> {
    value
        .as_f64()
        .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
        .ok_or("invalid decision probability")
}

fn response_from_systemone(body: &Value, questions: &[Value]) -> Result<Value, &'static str> {
    let mut answers = Vec::with_capacity(questions.len());
    for (index, question) in questions.iter().enumerate() {
        let answer = body["answers"]
            .get(format!("question_{index}"))
            .and_then(Value::as_object)
            .ok_or("missing decision answer")?;
        let kind = question["type"].as_str().ok_or("missing question type")?;
        let mut output = json!({"type": kind, "name": question.get("name")});
        if answer.get("type").and_then(Value::as_str) == Some("refusal") {
            output["type"] = json!("refusal");
        } else if kind == "predicate" {
            if answer.get("type").and_then(Value::as_str) != Some("noul") {
                return Err("unexpected predicate answer type");
            }
            output["probability"] = json!(probability(answer.get("noul").unwrap_or(&Value::Null))?);
        } else {
            if answer.get("type").and_then(Value::as_str) != Some(kind) {
                return Err("unexpected decision answer type");
            }
            let options = question[if kind == "choice" {
                "choices"
            } else {
                "levels"
            }]
            .as_array()
            .ok_or("missing decision options")?;
            let keys: Vec<String> = (0..options.len())
                .map(|index| {
                    if kind == "choice" {
                        format!("option_{index}")
                    } else {
                        index.to_string()
                    }
                })
                .collect();
            let probabilities = answer
                .get("probabilities")
                .and_then(Value::as_object)
                .filter(|values| {
                    values.len() == keys.len() && keys.iter().all(|key| values.contains_key(key))
                })
                .ok_or("decision probability keys do not match the options")?;
            let values: Vec<f64> = keys
                .iter()
                .map(|key| probability(&probabilities[key]))
                .collect::<Result<_, _>>()?;
            if (values.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err("decision probabilities must sum to one");
            }
            if kind == "choice" {
                let choice = answer
                    .get("choice")
                    .and_then(Value::as_str)
                    .and_then(|choice| keys.iter().position(|key| key == choice))
                    .ok_or("decision choice is not an allowed option")?;
                output["choice"] = options[choice]["value"].clone();
                output["probabilities"] = Value::Array(
                    options
                        .iter()
                        .zip(&values)
                        .map(|(option, p)| json!({"value": option["value"], "probability": p}))
                        .collect(),
                );
                output["confidence"] = json!(systemone::choice_confidence(&values));
            } else {
                output["score"] = json!(values
                    .iter()
                    .enumerate()
                    .map(|(index, p)| index as f64 * p)
                    .sum::<f64>());
                output["probabilities"] = Value::Array(options.iter().zip(&values).enumerate().map(|(index, (option, p))| {
                    json!({"value": index, "label": option["label"], "probability": p})
                }).collect());
                output["confidence"] = json!(systemone::score_confidence(&values));
            }
        }
        answers.push(output);
    }
    let tokens = body["usage"]["input_tokens"]
        .as_u64()
        .ok_or("invalid input token count")?;
    let model = body["model"].as_str().ok_or("invalid decision model")?;
    Ok(json!({
        "model": model, "answers": answers,
        "usage": {
            "input_tokens": tokens, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": tokens,
        }
    }))
}

fn api_error(status: StatusCode, message: &str, param: Option<&str>, code: &str) -> Response {
    (status, Json(json!({"error": {
        "message": message, "type": if status.is_server_error() { "server_error" } else { "invalid_request_error" },
        "param": param, "code": code,
    }}))).into_response()
}

fn copy_response_parts(parts: Parts, mut output: Response) -> Response {
    *output.extensions_mut() = parts.extensions;
    for (name, value) in &parts.headers {
        if name == header::RETRY_AFTER {
            if let Some((seconds, millis)) =
                value.to_str().ok().and_then(systemone::capped_retry_after)
            {
                if let (Ok(seconds), Ok(millis)) = (
                    HeaderValue::from_str(&seconds),
                    HeaderValue::from_str(&millis),
                ) {
                    output.headers_mut().insert(header::RETRY_AFTER, seconds);
                    output
                        .headers_mut()
                        .insert(HeaderName::from_static("retry-after-ms"), millis);
                }
            }
        } else if is_openai_compat_forwarded_header(name.as_str()) || name == "x-sie-error-code" {
            output.headers_mut().insert(name.clone(), value.clone());
        }
    }
    output
}

fn translation_fault(parts: Parts, stage: &'static str) -> Response {
    let output = copy_response_parts(
        parts,
        api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "model returned an invalid decision result",
            None,
            "INTERNAL_ERROR",
        ),
    );
    GatewayOwnedFault::new(stage).mark(output)
}

async fn translate_response(response: Response, questions: &[Value]) -> Response {
    let (parts, body) = response.into_parts();
    if parts.status != StatusCode::OK {
        let parsed: Value = match to_bytes(body, MAX_ERROR_BYTES).await {
            Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            Err(_) => Value::Null,
        };
        let (code, message) = systemone::error_parts(&parsed);
        let location = parsed["detail"]
            .as_array()
            .and_then(|errors| errors.first())
            .and_then(|error| error["loc"].as_array())
            .map(|loc| {
                loc.iter()
                    .skip(1)
                    .map(|part| match part.as_str() {
                        Some("state") => "input".to_string(),
                        Some(part) => part.to_string(),
                        None => part.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(".")
            })
            .filter(|param| !param.is_empty());
        let status = if parts.status == StatusCode::UNPROCESSABLE_ENTITY {
            StatusCode::BAD_REQUEST
        } else {
            parts.status
        };
        let output = api_error(
            status,
            message.unwrap_or("request failed"),
            location.as_deref(),
            code.unwrap_or("invalid_request"),
        );
        return copy_response_parts(parts, output);
    }
    let bytes = match to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => return translation_fault(parts, "read_decisions_response"),
    };
    let parsed: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return translation_fault(parts, "decode_decisions_response"),
    };
    match response_from_systemone(&parsed, questions) {
        Ok(body) => copy_response_parts(parts, Json(body).into_response()),
        Err(_) => translation_fault(parts, "format_decisions"),
    }
}

#[utoipa::path(
    post, path = "/v1/decisions", tag = "inference",
    description = "OpenAI-compatible text decisions over SIE decision models. Answers 1 to 200 ordered predicate, choice, or score questions; choices preserve string and boolean values. Supports text or text-only messages. Images, tools, cache options, and backend fields are rejected. Oversized model inputs are refused rather than truncated.",
    request_body = crate::openapi::DecisionRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Answers in question order", body = crate::openapi::DecisionResponse),
        (status = 400, description = "Invalid or unsupported input", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 401, description = "Missing or invalid bearer token"),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Invalid model result", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 502, description = "Model load failed", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Model loading or capacity exhausted; Retry-After at most 60 seconds", body = crate::openapi::OpenAIErrorEnvelope)
    )
)]
pub async fn proxy_decisions(State(state): State<Arc<AppState>>, req: Request) -> Response {
    let (mut parts, body) = req.into_parts();
    let bytes = match to_bytes(body, MAX_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(error) => {
            let (status, code, message) = request_body_error(&error, MAX_BODY_BYTES);
            return api_error(status, &message, None, code);
        }
    };
    let parsed: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "request body must be valid JSON",
                None,
                "invalid_request",
            )
        }
    };
    let (native, questions) = match normalize_request(&parsed) {
        Ok(value) => value,
        Err(error) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                &error.message,
                error.param.as_deref(),
                error.code,
            )
        }
    };
    let bytes = match serde_json::to_vec(&native) {
        Ok(bytes) => bytes,
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build decision request",
                None,
                "INTERNAL_ERROR",
            )
        }
    };
    let mut headers = HeaderMap::new();
    for (name, value) in &parts.headers {
        if is_openai_compat_inner_request_header(name.as_str()) {
            headers.append(name.clone(), value.clone());
        }
    }
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    headers.insert(header::ACCEPT, HeaderValue::from_static("application/json"));
    parts.headers = headers;
    parts.method = Method::POST;
    parts.uri = axum::http::Uri::from_static("/v1/systemone");
    // Keep authenticated identity, model access policy context, and all other
    // extensions on the original request. System One applies the native path's
    // routing/header allowlists and sets CompatibilityRoute itself.
    let inner = Request::from_parts(parts, Body::from(bytes));
    let response = systemone::proxy_systemone(State(state), inner).await;
    translate_response(response, &questions).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::test_support::{
        TestGateway, LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL, LOCAL_LANE,
    };
    use crate::server::{create_router, ModelAccessPolicy};
    use tower::ServiceExt;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../wire-fixtures/decisions.json")).unwrap()
    }

    fn predicate_request() -> Value {
        json!({"model": "acme/decide", "input": "Continue?", "questions": [{"type": "predicate", "instructions": "Continue?"}]})
    }

    fn request(body: Value) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/decisions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
    }

    fn assert_close(actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Object(actual), Value::Object(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (key, value) in expected {
                    assert_close(&actual[key], value);
                }
            }
            (Value::Array(actual), Value::Array(expected)) => {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_close(actual, expected);
                }
            }
            (Value::Number(actual), Value::Number(expected)) => {
                assert!((actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() < 1e-9);
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn shared_fixtures_pin_request_and_response_parity() {
        for case in fixture()["requests"].as_array().unwrap() {
            let (native, questions) = normalize_request(&case["request"]).unwrap();
            assert_eq!(native, case["systemone"], "{}", case["name"]);
            let body = json!({"model": native["model"], "answers": case["native_answers"], "usage": {"input_tokens": 42}});
            assert_close(
                &response_from_systemone(&body, &questions).unwrap(),
                &case["response"],
            );
        }
        for case in fixture()["invalid_requests"].as_array().unwrap() {
            let error = normalize_request(&case["request"]).unwrap_err();
            assert_eq!(
                error.param.as_deref(),
                case["param"].as_str(),
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn rejects_unsupported_fields_and_bounds_before_dispatch() {
        for field in [
            "prompt_cache_key",
            "prompt_cache_options",
            "temperature",
            "lora_path",
        ] {
            let mut body = predicate_request();
            body[field] = Value::Null;
            let error = normalize_request(&body).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(field));
            assert_eq!(error.code, "unsupported_field");
        }
        for (field, value) in [
            (
                "questions",
                json!(vec![
                    json!({"type": "predicate", "instructions": "ok"});
                    MAX_QUESTIONS + 1
                ]),
            ),
            (
                "input",
                json!(vec![
                    json!({"role": "user", "content": "hi"});
                    MAX_MESSAGES + 1
                ]),
            ),
            ("safety_identifier", json!("é".repeat(129))),
            ("model", json!("a".repeat(MAX_FIELD_CHARS + 1))),
        ] {
            let mut body = predicate_request();
            body[field] = value;
            assert_eq!(
                normalize_request(&body).unwrap_err().param.as_deref(),
                Some(field)
            );
        }
        for question in [
            json!({"type": "choice", "instructions": "pick", "choices": vec![json!({"value": "a"}); MAX_CHOICES + 1]}),
            json!({"type": "score", "instructions": "rate", "levels": vec![json!({"label": "a"}); MAX_SCORE_LEVELS + 1]}),
            json!({"type": "predicate", "instructions": "x".repeat(MAX_FIELD_CHARS + 1)}),
            json!({"type": "predicate", "instructions": "hi", "name": Value::Null}),
            json!({"type": "predicate", "instructions": "hi", "criteria": []}),
        ] {
            let mut body = predicate_request();
            body["questions"] = json!([question]);
            assert!(normalize_request(&body).is_err());
        }
        let mut body = predicate_request();
        body["safety_identifier"] = json!("é".repeat(128));
        assert!(normalize_request(&body).is_ok());
    }

    #[test]
    fn text_message_roles_and_parts_are_validated() {
        for input in [
            json!([{ "role": "developer", "content": [{"type": "input_text", "text": "rule"}] }]),
            json!([{ "role": "system", "content": "rule", "phase": null }]),
            json!([{ "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "yes"}, {"type": "refusal", "refusal": "no"}] }]),
        ] {
            let mut body = predicate_request();
            body["input"] = input.clone();
            assert_eq!(normalize_request(&body).unwrap().0["state"], input);
        }
        for input in [
            json!([{ "role": "tool", "content": "untrusted" }]),
            json!([{ "role": "assistant", "content": [{"type": "input_text", "text": "bad"}] }]),
            json!([{ "role": "user", "content": [{"type": "input_text", "text": "bad", "prompt_cache_breakpoint": {"mode": "explicit"}}] }]),
            json!([{ "role": "user", "content": [], "phase": "other" }]),
            json!([{ "role": "user", "content": [{"type": "input_text", "text": true}] }]),
            json!({"role": "user", "content": "hi"}),
        ] {
            let mut body = predicate_request();
            body["input"] = input;
            assert!(normalize_request(&body).is_err());
        }
    }

    #[test]
    fn invalid_native_answers_are_not_delivered() {
        let cases = fixture();
        let case = &cases["requests"][0];
        let (_, questions) = normalize_request(&case["request"]).unwrap();
        let valid =
            json!({"model": "m", "answers": case["native_answers"], "usage": {"input_tokens": 42}});
        for value in [
            json!(-0.1),
            json!(1.1),
            json!(true),
            json!("0.5"),
            Value::Null,
        ] {
            let mut body = valid.clone();
            body["answers"]["question_0"]["noul"] = value;
            assert!(response_from_systemone(&body, &questions).is_err());
        }
        for probabilities in [
            json!({"option_0": 0.9}),
            json!({"option_0": 0.9, "option_1": 0.1, "extra": 0.0}),
            json!({"option_0": 0.9, "option_1": 0.3}),
            json!({"option_0": -0.1, "option_1": 1.1}),
        ] {
            let mut body = valid.clone();
            body["answers"]["question_1"]["probabilities"] = probabilities;
            assert!(response_from_systemone(&body, &questions).is_err());
        }
        for value in [json!(-1), json!(true), json!(1.5), Value::Null] {
            let mut body = valid.clone();
            body["usage"]["input_tokens"] = value;
            assert!(response_from_systemone(&body, &questions).is_err());
        }
        let mut body = valid.clone();
        body["answers"]["question_1"]["choice"] = json!("billing");
        assert!(response_from_systemone(&body, &questions).is_err());
        let mut body = valid.clone();
        body["answers"]["question_0"] = json!({"type": "choice", "noul": 0.5});
        assert!(response_from_systemone(&body, &questions).is_err());
        let mut body = valid;
        body["answers"]
            .as_object_mut()
            .unwrap()
            .remove("question_2");
        assert!(response_from_systemone(&body, &questions).is_err());
    }

    #[tokio::test]
    async fn registered_route_dispatches_native_extract_and_preserves_answer_order() {
        for case in fixture()["requests"].as_array().unwrap() {
            let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
            gateway
                .add_verified_worker("local-1", LOCAL_LANE, &["acme/decide"])
                .await;
            gateway
                .dispatcher
                .return_extract_data(case["native_answers"].clone());
            let mut body = case["request"].clone();
            body["model"] = json!("acme/decide");
            let router = create_router(
                Arc::clone(&gateway.state),
                Arc::clone(&gateway.state.config),
            );
            let response = router.oneshot(request(body)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = body_json(response).await;
            assert_close(&body["answers"], &case["response"]["answers"]);
            assert_eq!(body["model"], "acme/decide");
            assert_eq!(body["usage"]["input_tokens"], 0);
            assert_eq!(body["usage"]["total_tokens"], 0);
            let dispatched = gateway.dispatcher.dispatched();
            assert_eq!(dispatched.len(), 1);
            assert_eq!(dispatched[0].model, "acme/decide");
            assert_eq!(dispatched[0].endpoint, "extract");
        }
    }

    #[derive(Clone)]
    struct Caller;

    struct RequireCaller;
    impl ModelAccessPolicy for RequireCaller {
        fn visible(&self, _: &str, ext: &axum::http::Extensions) -> bool {
            ext.get::<Caller>().is_some()
        }
    }

    #[tokio::test]
    async fn configured_aliases_preserve_caller_spelling_and_dispatch_identity() {
        for model in ["decision", "DECISION", "profiled", "bundled"] {
            let model_yaml = format!(
                "{LOCAL_DECISION_MODEL}  alternate:\n    adapter_path: sie_server.adapters.gliner2.decide:GLiNER2DecideAdapter\n    max_batch_tokens: 8192\n"
            );
            let mut gateway = TestGateway::new(&[&model_yaml]).await;
            gateway.add_model_alias("decision", "acme/decide");
            gateway.add_model_alias("profiled", "acme/decide:alternate");
            gateway.add_model_alias("bundled", "default:/acme/decide");
            gateway
                .add_verified_worker(
                    "local-1",
                    LOCAL_LANE,
                    &["acme/decide", "acme/decide:alternate"],
                )
                .await;
            gateway.dispatcher.return_extract_data(json!({
                "question_0":{"type":"noul", "noul":0.9}
            }));
            let mut body = predicate_request();
            body["model"] = json!(model);
            let response = proxy_decisions(State(Arc::clone(&gateway.state)), request(body)).await;
            let status = response.status();
            let body = body_json(response).await;
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            assert_eq!(body["model"], model);
            assert_eq!(body["answers"][0]["probability"], 0.9);
            let dispatched = gateway.dispatcher.dispatched();
            assert_eq!(dispatched.len(), 1);
            assert_eq!(
                dispatched[0].model,
                if model == "profiled" {
                    "acme/decide:alternate"
                } else {
                    "acme/decide"
                }
            );
            assert_eq!(dispatched[0].endpoint, "extract");
        }
    }

    #[tokio::test]
    async fn hidden_and_absent_aliases_have_identical_not_found_responses() {
        for populated in [false, true] {
            let hidden_models = if populated {
                vec![LOCAL_DECISION_MODEL]
            } else {
                vec![]
            };
            let absent_models = if populated {
                vec![LOCAL_ENCODE_MODEL]
            } else {
                vec![]
            };
            let mut hidden = TestGateway::new(&hidden_models).await;
            let mut absent = TestGateway::new(&absent_models).await;
            hidden.add_model_alias("decision", "acme/decide");
            absent.add_model_alias("decision", "acme/decide");
            hidden.install_policy(Arc::new(RequireCaller));
            for model in ["decision", "DECISION"] {
                let mut payload = predicate_request();
                payload["model"] = json!(model);
                let denied =
                    proxy_decisions(State(Arc::clone(&hidden.state)), request(payload.clone()))
                        .await;
                let missing =
                    proxy_decisions(State(Arc::clone(&absent.state)), request(payload)).await;
                assert_eq!(denied.status(), StatusCode::NOT_FOUND);
                assert_eq!(missing.status(), StatusCode::NOT_FOUND);
                let denied = body_json(denied).await;
                assert_eq!(denied["error"]["code"], "MODEL_NOT_FOUND");
                assert_eq!(denied, body_json(missing).await);
                assert!(!denied.to_string().contains("acme/"));
            }
            assert!(hidden.dispatcher.dispatched().is_empty());
            assert!(absent.dispatcher.dispatched().is_empty());
        }
    }

    #[tokio::test]
    async fn alias_visibility_precedes_decision_capability_checks() {
        let mut gateway = TestGateway::new(&[LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL]).await;
        gateway.add_model_alias("decision", "acme/decide");
        gateway.add_model_alias("encode", "acme/local");
        gateway.install_policy(Arc::new(RequireCaller));
        for model in ["decision", "encode"] {
            let mut body = predicate_request();
            body["model"] = json!(model);
            let response = proxy_decisions(State(Arc::clone(&gateway.state)), request(body)).await;
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            let body = body_json(response).await;
            assert!(body.get("error").is_some());
            assert!(!body.to_string().contains("acme/"));
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn authenticated_extensions_reach_the_native_model_policy() {
        let mut gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
        gateway.install_policy(Arc::new(RequireCaller));
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &["acme/decide"])
            .await;
        gateway
            .dispatcher
            .return_extract_data(json!({"question_0": {"type": "noul", "noul": 0.5}}));
        let mut allowed = request(predicate_request());
        allowed.extensions_mut().insert(Caller);
        // The wrapper must replace incompatible framing/negotiation headers.
        allowed
            .headers_mut()
            .insert(header::CONTENT_LENGTH, HeaderValue::from_static("1"));
        allowed.headers_mut().insert(
            header::ACCEPT,
            HeaderValue::from_static("application/msgpack"),
        );
        let response = proxy_decisions(State(Arc::clone(&gateway.state)), allowed).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        let denied = proxy_decisions(
            State(Arc::clone(&gateway.state)),
            request(predicate_request()),
        )
        .await;
        assert_eq!(denied.status(), StatusCode::NOT_FOUND);
        assert!(body_json(denied).await.get("error").is_some());
        assert_eq!(gateway.dispatcher.dispatched().len(), 1);
    }

    #[tokio::test]
    async fn caller_errors_do_not_dispatch_work() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL]).await;
        for (body, status) in [
            (
                json!({"model": "acme/decide", "input": "hi", "questions": []}),
                StatusCode::BAD_REQUEST,
            ),
            (
                json!({"model": "acme/local", "input": "hi", "questions": [{"type": "predicate", "instructions": "ok"}]}),
                StatusCode::BAD_REQUEST,
            ),
            (
                json!({"model": "acme/missing", "input": "hi", "questions": [{"type": "predicate", "instructions": "ok"}]}),
                StatusCode::NOT_FOUND,
            ),
        ] {
            let response = proxy_decisions(State(Arc::clone(&gateway.state)), request(body)).await;
            assert_eq!(response.status(), status);
            let body = body_json(response).await;
            assert!(body.get("error").is_some());
            assert!(body.get("detail").is_none());
        }
        for (body, status) in [
            (Body::from("{"), StatusCode::BAD_REQUEST),
            (
                Body::from(vec![b'x'; MAX_BODY_BYTES + 1]),
                StatusCode::PAYLOAD_TOO_LARGE,
            ),
        ] {
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/decisions")
                .body(body)
                .unwrap();
            let response = proxy_decisions(State(Arc::clone(&gateway.state)), req).await;
            assert_eq!(response.status(), status);
            assert!(body_json(response).await.get("error").is_some());
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn retry_headers_and_existing_gateway_faults_survive_translation() {
        let mut inner = GatewayOwnedFault::new("upstream").mark(
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"detail": {"code": "MODEL_LOADING", "message": "loading"}})),
            )
                .into_response(),
        );
        inner
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("270"));
        inner.headers_mut().insert(
            HeaderName::from_static("x-sie-request-id"),
            HeaderValue::from_static("r1"),
        );
        inner
            .headers_mut()
            .insert(header::SET_COOKIE, HeaderValue::from_static("private=true"));
        let response = translate_response(inner, &[]).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()[header::RETRY_AFTER], "60");
        assert_eq!(response.headers()["retry-after-ms"], "60000");
        assert_eq!(response.headers()["x-sie-request-id"], "r1");
        assert!(!response.headers().contains_key(header::SET_COOKIE));
        assert_eq!(
            response.extensions().get::<GatewayOwnedFault>(),
            Some(&GatewayOwnedFault::new("upstream"))
        );
        assert_eq!(body_json(response).await["error"]["code"], "MODEL_LOADING");
    }

    #[tokio::test]
    async fn native_validation_errors_use_the_public_input_parameter() {
        let inner = (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"detail": [{
                "loc": ["body", "state"], "msg": "does not fit", "code": "INPUT_TOO_LONG"
            }]})),
        )
            .into_response();
        let response = translate_response(inner, &[]).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response).await;
        assert_eq!(body["error"]["param"], "input");
        assert_eq!(body["error"]["code"], "INPUT_TOO_LONG");
    }

    #[tokio::test]
    async fn invalid_or_oversized_success_bodies_are_gateway_owned_faults() {
        for (body, stage) in [
            (Body::from("not JSON"), "decode_decisions_response"),
            (Body::from("{}"), "format_decisions"),
            (
                Body::from(vec![b'x'; MAX_BODY_BYTES + 1]),
                "read_decisions_response",
            ),
        ] {
            let response = translate_response(Response::new(body), &[]).await;
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(
                response.extensions().get::<GatewayOwnedFault>(),
                Some(&GatewayOwnedFault::new(stage))
            );
            assert_eq!(body_json(response).await["error"]["code"], "INTERNAL_ERROR");
        }
    }

    #[tokio::test]
    async fn both_translation_layers_mark_undelivered_model_results() {
        for (data, stage) in [
            (json!({}), "format_systemone"),
            (
                json!({"question_0": {"type": "noul", "noul": 2.0}}),
                "format_decisions",
            ),
        ] {
            let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
            gateway
                .add_verified_worker("local-1", LOCAL_LANE, &["acme/decide"])
                .await;
            gateway.dispatcher.return_extract_data(data);
            let response = proxy_decisions(
                State(Arc::clone(&gateway.state)),
                request(predicate_request()),
            )
            .await;
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(
                response.extensions().get::<GatewayOwnedFault>(),
                Some(&GatewayOwnedFault::new(stage))
            );
            assert_eq!(body_json(response).await["error"]["code"], "INTERNAL_ERROR");
        }
    }
}
