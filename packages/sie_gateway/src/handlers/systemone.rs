//! System One-compatible typed decisions: `POST /v1/systemone`.
//!
//! A client written for the System One API (the TypeSafe SDKs, or anything that
//! sends the same JSON) points its base URL at SIE and keeps its code. The
//! gateway rewrites the request into one native `extract` item for a decision
//! model (GLiNER2.5-Decide or Laya) and the extract result back into the System
//! One response, as the single server's `sie_server.api.systemone` does. Both
//! check `packages/wire-fixtures/systemone.json`.
//!
//! - Request `{model, state, questions}`. Unknown fields are ignored; a
//!   question may omit `instructions` or send `null`; media fields and a
//!   score question without 1 to 10 levels are refused.
//! - Response `{model, answers, usage: {input_tokens, output_tokens}}`, JSON.
//!   Choice and score confidences follow the System One definitions.
//! - The extract call sets `overflow_policy: "error"`, so a state or question
//!   set that does not fit the model's row is refused with 422 instead of
//!   being truncated.
//! - Input errors are 422 in FastAPI's `{"detail": [{loc, msg, type}]}` shape;
//!   other errors `{"detail": {"error_type", "code", "message"}}`. A
//!   retryable error's `Retry-After` is at most 60 seconds, mirrored in
//!   milliseconds as `retry-after-ms`.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};
use tracing::warn;

use crate::http_error::GatewayOwnedFault;
use crate::server::AppState;

use super::proxy::{
    check_sdk_version, compatibility_model_uri, is_openai_compat_forwarded_header,
    is_openai_compat_inner_request_header, is_valid_compat_model_id, proxy_request,
    request_body_error, resolve_model_and_bundle, CompatibilityRoute,
};

/// Request body cap. A System One request is one text state and its
/// questions; 16 MiB is the text-route cap the chat and embeddings routes use.
const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
/// Native extract reply cap, the same 16 MiB as the other compatibility
/// routes' native replies; one item's answers are far smaller.
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
/// Score levels per question, as the System One API accepts.
pub(crate) const MAX_SCORE_LEVELS: usize = 10;
/// Longest `Retry-After` the route sends. System One clients (and LiteLLM)
/// wait at most 60 seconds before falling back.
pub(crate) const MAX_RETRY_AFTER_S: u64 = 60;
/// Adapter modules that answer typed questions: the modules of the single
/// server's `DECISION_ADAPTERS` (`sie_server.api.systemone`), which names
/// their classes.
const DECISION_ADAPTER_MODULES: [&str; 2] = [
    "sie_server.adapters.gliner2.decide",
    "sie_server.adapters.laya.adapter",
];
/// Request fields for media the decision models cannot read.
const MEDIA_FIELDS: [&str; 3] = ["images", "audio", "videos"];

/// A validated System One request: the model, the extract item, and the
/// questions as the decision adapters read them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SystemOneRequest {
    pub model: String,
    pub item: Value,
    pub questions: Map<String, Value>,
    pub lora_options: Map<String, Value>,
}

/// Where a request failed validation (after `body`) and why.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Invalid {
    pub loc: Vec<String>,
    pub msg: String,
}

fn invalid(loc: &[&str], msg: impl Into<String>) -> Invalid {
    Invalid {
        loc: loc.iter().map(|part| (*part).to_string()).collect(),
        msg: msg.into(),
    }
}

/// Validate a System One request and normalize its questions.
pub(crate) fn normalize_request(parsed: &Value) -> Result<SystemOneRequest, Invalid> {
    let body = parsed
        .as_object()
        .ok_or_else(|| invalid(&[], "the request body must be a JSON object"))?;
    let model = match body.get("model") {
        Some(Value::String(model)) if !model.is_empty() => model.clone(),
        Some(Value::String(_)) => return Err(invalid(&["model"], "'model' must not be empty")),
        Some(_) => return Err(invalid(&["model"], "'model' must be a string")),
        None => return Err(invalid(&["model"], "field required")),
    };
    let item = match body.get("state") {
        Some(Value::String(text)) => json!({ "text": text }),
        Some(state @ (Value::Object(_) | Value::Array(_))) => {
            json!({ "metadata": { "state": state } })
        }
        Some(_) => {
            return Err(invalid(
                &["state"],
                "'state' must be a string, an object, or an array",
            ))
        }
        None => return Err(invalid(&["state"], "field required")),
    };
    let questions = match body.get("questions") {
        Some(Value::Object(questions)) if !questions.is_empty() => questions,
        Some(Value::Object(_)) => {
            return Err(invalid(&["questions"], "'questions' must not be empty"))
        }
        Some(_) => {
            return Err(invalid(
                &["questions"],
                "'questions' must be an object of questions by id",
            ))
        }
        None => return Err(invalid(&["questions"], "field required")),
    };
    for field in MEDIA_FIELDS {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            return Err(invalid(
                &[field],
                format!("'{field}' is not supported: SIE decision models read text only"),
            ));
        }
    }
    let mut normalized = Map::new();
    for (qid, question) in questions {
        let Some(question) = question.as_object() else {
            return Err(invalid(
                &["questions", qid],
                format!("question {qid:?} must be an object"),
            ));
        };
        let mut question = question.clone();
        if question.get("instructions").is_none_or(Value::is_null) {
            question.insert("instructions".to_string(), json!(""));
        }
        if question.get("type").and_then(Value::as_str) == Some("score") {
            if let Some(levels) = question.get("criteria").and_then(Value::as_array) {
                if levels.is_empty() {
                    return Err(invalid(
                        &["questions", qid, "criteria"],
                        format!("score question {qid:?} has no levels"),
                    ));
                }
                if levels.len() > MAX_SCORE_LEVELS {
                    return Err(invalid(
                        &["questions", qid, "criteria"],
                        format!(
                            "score question {qid:?} has {} levels; at most {MAX_SCORE_LEVELS} are allowed",
                            levels.len()
                        ),
                    ));
                }
                if levels.iter().any(Value::is_null) {
                    return Err(invalid(
                        &["questions", qid, "criteria"],
                        format!("score question {qid:?} has a null level"),
                    ));
                }
            }
        }
        normalized.insert(qid.clone(), Value::Object(question));
    }
    let lora_options =
        super::lora::selectors(body).map_err(|error| invalid(&[error.field], error.message))?;
    Ok(SystemOneRequest {
        model,
        item,
        questions: normalized,
        lora_options,
    })
}

/// System One choice confidence: `(p_max - 1/n) / (1 - 1/n)`, within [0, 1].
pub(crate) fn choice_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let floor = 1.0 / n as f64;
    let p_max = probabilities.iter().copied().fold(f64::MIN, f64::max);
    ((p_max - floor) / (1.0 - floor)).clamp(0.0, 1.0)
}

/// System One score confidence: `max(0, 1 - spread / even)`, where `spread`
/// is the expected distance from the most likely level and `even` that of a
/// uniform distribution from the middle level.
pub(crate) fn score_confidence(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    if n <= 1 {
        return 1.0;
    }
    let mut mode = 0;
    for (index, p) in probabilities.iter().enumerate() {
        if *p > probabilities[mode] {
            mode = index;
        }
    }
    let spread: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, p)| p * (index as f64 - mode as f64).abs())
        .sum();
    let middle = (n as f64 - 1.0) / 2.0;
    let even: f64 = (0..n)
        .map(|index| (index as f64 - middle).abs())
        .sum::<f64>()
        / n as f64;
    (1.0 - spread / even).clamp(0.0, 1.0)
}

fn probability(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .ok_or_else(|| format!("probability {value} is not a number"))
}

/// One adapter answer in the System One shape, with System One confidences.
///
/// `Err` for a probability that is not a number or a score level key that is
/// not an integer, as the single server refuses them.
pub(crate) fn to_answer(answer: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut result = answer.clone();
    let kind = answer.get("type").and_then(Value::as_str);
    let Some(probabilities) = answer.get("probabilities").and_then(Value::as_object) else {
        return Ok(result);
    };
    let confidence = match kind {
        Some("choice") => {
            let values = probabilities
                .values()
                .map(probability)
                .collect::<Result<Vec<f64>, String>>()?;
            Some(choice_confidence(&values))
        }
        Some("score") => {
            let mut levels = probabilities
                .iter()
                .map(|(key, value)| {
                    let level: u64 = key
                        .parse()
                        .map_err(|_| format!("score level {key:?} is not an integer"))?;
                    Ok((level, probability(value)?))
                })
                .collect::<Result<Vec<(u64, f64)>, String>>()?;
            levels.sort_by_key(|(level, _)| *level);
            let values: Vec<f64> = levels.into_iter().map(|(_, p)| p).collect();
            Some(score_confidence(&values))
        }
        _ => None,
    };
    if let Some(confidence) = confidence {
        result.insert("confidence".to_string(), json!(confidence));
    }
    Ok(result)
}

/// What an extract reply becomes: the System One body, or the 422 for an item
/// the adapter refused (a state that does not fit, an unreadable state) with
/// the adapter's error code.
#[derive(Debug, PartialEq)]
pub(crate) enum Translated {
    Answers(Value),
    Refused(Invalid, String),
}

/// Translate a native extract reply into the System One response body.
///
/// `Err` is a reply this route cannot render: the worker ran, and the gateway
/// failed to deliver its result.
pub(crate) fn response_from_extract(
    native: &Value,
    model: &str,
    questions: &Map<String, Value>,
) -> Result<Translated, String> {
    let item = native
        .get("items")
        .and_then(Value::as_array)
        .filter(|items| items.len() == 1)
        .map(|items| &items[0])
        .ok_or_else(|| "extract response is missing its one item".to_string())?;
    if let Some(error) = item.get("error").filter(|error| !error.is_null()) {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the state could not be read");
        let code = error
            .get("code")
            .and_then(Value::as_str)
            .unwrap_or("INVALID_INPUT");
        return Ok(Translated::Refused(
            invalid(&["state"], message),
            code.to_string(),
        ));
    }
    let data = item
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| "extract response item has no data".to_string())?;
    let mut answers = Map::new();
    for qid in questions.keys() {
        let answer = data
            .get(qid)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("model returned no answer for question {qid:?}"))?;
        let answer = to_answer(answer).map_err(|error| format!("question {qid:?}: {error}"))?;
        answers.insert(qid.clone(), Value::Object(answer));
    }
    // The queued extract reply carries no `usage`, so this is 0 on a cluster;
    // metered deployments report units in `X-SIE-Units-*` headers instead.
    let input_tokens = native
        .get("usage")
        .and_then(|usage| usage.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(Translated::Answers(json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    })))
}

/// The native extract body for a System One request: its one item, the
/// questions as `output_schema`, and `overflow_policy: "error"` so input that
/// does not fit the model's row is refused rather than cut.
pub(crate) fn extract_body(request: &SystemOneRequest) -> Value {
    let mut options = request.lora_options.clone();
    options.insert("overflow_policy".into(), json!("error"));
    json!({
        "items": [request.item],
        "params": {
            "output_schema": request.questions,
            "options": options,
        },
    })
}

fn error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST
        | StatusCode::PAYLOAD_TOO_LARGE
        | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        _ => "api_error",
    }
}

/// A 422 in FastAPI's validation-error shape, which System One clients parse.
pub(crate) fn validation_error(error: &Invalid, code: Option<&str>) -> Response {
    let mut loc = vec![json!("body")];
    loc.extend(error.loc.iter().map(|part| json!(part)));
    let mut entry = json!({"type": "value_error", "loc": loc, "msg": error.msg});
    if let Some(code) = code {
        entry["code"] = json!(code);
    }
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({ "detail": [entry] })),
    )
        .into_response()
}

/// A non-validation error: `{"detail": {"error_type", "code", "message"}}`.
pub(crate) fn api_error(
    status: StatusCode,
    code: Option<&str>,
    message: impl Into<String>,
) -> Response {
    let mut detail = json!({"error_type": error_type(status), "message": message.into()});
    if let Some(code) = code {
        detail["code"] = json!(code);
    }
    (status, Json(json!({ "detail": detail }))).into_response()
}

/// The `Retry-After` to send for an upstream value: capped at 60 seconds, as
/// whole seconds and as milliseconds. `None` for a value that is not a
/// non-negative number of seconds.
pub(crate) fn capped_retry_after(value: &str) -> Option<(String, String)> {
    let seconds: f64 = value.trim().parse().ok()?;
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    let seconds = seconds.min(MAX_RETRY_AFTER_S as f64);
    Some((
        (seconds.ceil() as u64).max(1).to_string(),
        ((seconds * 1000.0) as u64).to_string(),
    ))
}

fn copy_error_headers(from: &HeaderMap, to: &mut HeaderMap) {
    for (name, value) in from {
        let key = name.as_str();
        if key.eq_ignore_ascii_case("retry-after") {
            if let Some((seconds, millis)) = value.to_str().ok().and_then(capped_retry_after) {
                if let (Ok(seconds), Ok(millis)) = (
                    HeaderValue::from_str(&seconds),
                    HeaderValue::from_str(&millis),
                ) {
                    to.insert(axum::http::header::RETRY_AFTER, seconds);
                    to.insert(HeaderName::from_static("retry-after-ms"), millis);
                }
            }
        } else if is_openai_compat_forwarded_header(key)
            || key.eq_ignore_ascii_case("x-sie-error-code")
        {
            to.insert(name.clone(), value.clone());
        }
    }
}

/// The code and message of a native error body. Reads the native and OpenAI
/// envelopes (`{"detail" | "error": {code, message}}`), the batch failure
/// `{"error": "all_items_failed", "details": [{code, error}]}`, and a FastAPI
/// validation list (`{"detail": [{msg}]}`).
pub(crate) fn error_parts(parsed: &Value) -> (Option<&str>, Option<&str>) {
    let envelope = parsed.get("detail").or_else(|| parsed.get("error"));
    if let Some(envelope @ Value::Object(_)) = envelope {
        return (
            envelope.get("code").and_then(Value::as_str),
            envelope.get("message").and_then(Value::as_str),
        );
    }
    if let Some(first) = parsed
        .get("details")
        .and_then(Value::as_array)
        .and_then(|details| details.first())
    {
        return (
            first.get("code").and_then(Value::as_str),
            first.get("error").and_then(Value::as_str),
        );
    }
    if let Some(first) = envelope
        .and_then(Value::as_array)
        .and_then(|detail| detail.first())
    {
        return (
            first.get("code").and_then(Value::as_str),
            first.get("msg").and_then(Value::as_str),
        );
    }
    (None, None)
}

/// Re-surface a native error in the System One shapes. A 400 (invalid input,
/// input too long) becomes 422; other statuses keep their code. The extensions
/// (a gateway-owned-fault marker) and the retry headers survive.
async fn translate_error(response: Response) -> Response {
    // An error envelope: a code and a sanitized message.
    const MAX: usize = 1024 * 1024;
    let (parts, body) = response.into_parts();
    let parsed: Value = match to_bytes(body, MAX).await {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let (code, message) = error_parts(&parsed);
    let code = code.map(|code| {
        if parts.status == StatusCode::NOT_FOUND && code.eq_ignore_ascii_case("MODEL_NOT_FOUND") {
            "MODEL_NOT_FOUND"
        } else {
            code
        }
    });
    if message.is_none() {
        warn!(
            status = parts.status.as_u16(),
            "System One: native error without a readable message"
        );
    }
    let message = message.unwrap_or("request failed");
    let mut out = if parts.status == StatusCode::BAD_REQUEST {
        validation_error(&invalid(&[], message), code)
    } else {
        api_error(parts.status, code, message)
    };
    *out.extensions_mut() = parts.extensions;
    copy_error_headers(&parts.headers, out.headers_mut());
    out
}

fn translation_fault(model: &str, stage: &'static str, message: impl Into<String>) -> Response {
    let message = message.into();
    warn!(model = %model, stage, error = %message, "System One: undelivered extract result");
    GatewayOwnedFault::new(stage).mark(api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        Some("INTERNAL_ERROR"),
        message,
    ))
}

fn is_decision_model(state: &AppState, model: &str) -> Option<bool> {
    let entry = state.model_registry.get_model_info(model)?;
    Some(
        entry
            .adapter_modules
            .iter()
            .any(|module| DECISION_ADAPTER_MODULES.contains(&module.as_str())),
    )
}

#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "inference",
    description = "System One-compatible typed decisions over SIE's native extract primitive. Answers one state (a string, an object, or a list of conversation turns) against typed noul, choice and score questions with a decision model. Clients written for the System One API work by changing their base URL. A state or question set that does not fit the model's window is refused with 422.",
    request_body = crate::openapi::SystemOneRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Every question answered", body = crate::openapi::SystemOneResponse),
        (status = 401, description = "Missing or invalid bearer token"),
        (status = 404, description = "Model not found"),
        (status = 413, description = "Request body too large"),
        (status = 422, description = "Invalid request, a model that does not answer typed questions, or input that does not fit the model's window"),
        (status = 500, description = "Malformed native extract result"),
        (status = 502, description = "MODEL_LOAD_FAILED"),
        (status = 503, description = "Provisioning, model loading, or capacity exhausted; Retry-After at most 60 seconds")
    )
)]
pub async fn proxy_systemone(State(state): State<Arc<AppState>>, req: Request) -> Response {
    check_sdk_version(req.headers());
    let headers = req.headers().clone();
    let (parts, body) = req.into_parts();
    let body = match to_bytes(body, MAX_REQUEST_BYTES).await {
        Ok(body) => body,
        Err(error) => {
            let (status, code, message) = request_body_error(&error, MAX_REQUEST_BYTES);
            return api_error(status, Some(code), message);
        }
    };
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return validation_error(&invalid(&[], format!("invalid JSON: {error}")), None);
        }
    };
    let request = match normalize_request(&parsed) {
        Ok(request) => request,
        Err(error) => return validation_error(&error, None),
    };
    if !is_valid_compat_model_id(&request.model) {
        return validation_error(&invalid(&["model"], "invalid model id"), None);
    }
    let (resolved_model, _) =
        match resolve_model_and_bundle(&state, &request.model, &parts.extensions) {
            Ok(resolved) => resolved,
            Err(response) => return translate_error(response).await,
        };
    match is_decision_model(&state, &resolved_model) {
        None => {
            return api_error(
                StatusCode::NOT_FOUND,
                Some("MODEL_NOT_FOUND"),
                format!("Model '{}' not found", request.model),
            )
        }
        Some(false) => {
            return validation_error(
                &invalid(
                    &["model"],
                    format!(
                        "Model '{}' does not answer typed questions. Use a decision model such as \
                         fastino/GLiNER2.5-Decide or convaiinnovations/laya-typed-decisions.",
                        request.model
                    ),
                ),
                None,
            )
        }
        Some(true) => {}
    }

    let extract_body = extract_body(&request);
    let uri = match compatibility_model_uri("extract", &request.model) {
        Ok(uri) => uri,
        Err(message) => return validation_error(&invalid(&["model"], message), None),
    };
    let extract_bytes = match serde_json::to_vec(&extract_body) {
        Ok(bytes) => bytes,
        Err(error) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                Some("INTERNAL_ERROR"),
                format!("extract body: {error}"),
            )
        }
    };
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .version(parts.version);
    for (name, value) in headers.iter() {
        if is_openai_compat_inner_request_header(name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    builder = builder
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .header(axum::http::header::ACCEPT, "application/json");
    let mut inner = match builder.body(Body::from(extract_bytes)) {
        Ok(request) => request,
        Err(_) => {
            return api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                Some("INTERNAL_ERROR"),
                "failed to build internal extract request",
            )
        }
    };
    *inner.extensions_mut() = parts.extensions;
    inner.extensions_mut().insert(CompatibilityRoute);
    let response = proxy_request(State(state), inner, "extract").await;
    if response.status() != StatusCode::OK {
        return translate_error(response).await;
    }

    // From here the worker has run and reported its units, so a failure to
    // render its result is the gateway's: the caller receives no answers and
    // must not be charged for them. Each arm is marked [`GatewayOwnedFault`].
    let response_headers = response.headers().clone();
    let native: Value = match to_bytes(response.into_body(), MAX_RESPONSE_BYTES).await {
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                return translation_fault(
                    &request.model,
                    "decode_extract_response",
                    format!("extract response is not valid JSON: {error}"),
                )
            }
        },
        Err(_) => {
            return translation_fault(
                &request.model,
                "read_extract_response",
                "failed to read extract response body",
            )
        }
    };
    let mut output = match response_from_extract(&native, &request.model, &request.questions) {
        Ok(Translated::Answers(body)) => (StatusCode::OK, Json(body)).into_response(),
        Ok(Translated::Refused(error, code)) => validation_error(&error, Some(&code)),
        Err(message) => return translation_fault(&request.model, "format_systemone", message),
    };
    for (name, value) in response_headers.iter() {
        if is_openai_compat_forwarded_header(name.as_str()) {
            output.headers_mut().insert(name.clone(), value.clone());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../../wire-fixtures/systemone.json"))
            .expect("systemone fixture")
    }

    #[test]
    fn lora_selectors_survive_systemone_translation() {
        let base = fixture()["requests"][0]["request"].clone();
        for field in super::super::lora::FIELDS {
            for value in [Value::Null, json!("org/legal")] {
                let mut body = base.clone();
                body[field] = value.clone();
                let request = normalize_request(&body).unwrap();
                assert_eq!(extract_body(&request)["params"]["options"][field], value);
            }
        }
        let mut conflict = base;
        conflict["lora_adapter"] = Value::Null;
        conflict["lora"] = json!("legal");
        assert!(normalize_request(&conflict).is_err());
    }

    #[test]
    fn fixture_requests_become_their_extract_calls() {
        for case in fixture()["requests"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let request = normalize_request(&case["request"]).unwrap_or_else(|error| {
                panic!("{name}: {error:?}");
            });
            let expected = &case["extract"];
            assert_eq!(request.model, expected["model"], "{name}");
            assert_eq!(request.item, expected["item"], "{name}");
            let body = extract_body(&request);
            assert_eq!(body["items"], json!([expected["item"]]), "{name}");
            assert_eq!(
                body["params"]["output_schema"], expected["output_schema"],
                "{name}"
            );
            assert_eq!(body["params"]["options"], expected["options"], "{name}");
        }
    }

    #[test]
    fn fixture_invalid_requests_are_refused() {
        for case in fixture()["invalid_requests"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            assert!(
                normalize_request(&case["request"]).is_err(),
                "{name} was accepted"
            );
        }
    }

    fn assert_close(actual: &Value, expected: &Value, path: &str) {
        match expected {
            Value::Object(expected) => {
                let actual = actual
                    .as_object()
                    .unwrap_or_else(|| panic!("{path}: not an object"));
                let mut actual_keys: Vec<_> = actual.keys().collect();
                let mut expected_keys: Vec<_> = expected.keys().collect();
                actual_keys.sort();
                expected_keys.sort();
                assert_eq!(actual_keys, expected_keys, "{path}");
                for (key, value) in expected {
                    assert_close(&actual[key], value, &format!("{path}.{key}"));
                }
            }
            Value::Number(number) if number.is_f64() => {
                let actual = actual
                    .as_f64()
                    .unwrap_or_else(|| panic!("{path}: not a number"));
                assert!(
                    (actual - number.as_f64().unwrap()).abs() < 1e-9,
                    "{path}: {actual} != {number}"
                );
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    #[test]
    fn fixture_responses_are_rendered() {
        for case in fixture()["responses"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let questions: Map<String, Value> = case["questions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|qid| (qid.as_str().unwrap().to_string(), json!({"type": "noul"})))
                .collect();
            let native = json!({
                "model": case["model"],
                "items": [{"id": "item-0", "data": case["data"]}],
                "usage": {"input_tokens": case["input_tokens"]},
            });
            let model = case["model"].as_str().unwrap();
            let Translated::Answers(body) =
                response_from_extract(&native, model, &questions).unwrap()
            else {
                panic!("{name}: refused");
            };
            assert_close(&body, &case["response"], name);
            assert!(body["usage"]["input_tokens"].is_u64(), "{name}");
            assert!(body["usage"]["output_tokens"].is_u64(), "{name}");
        }
    }

    #[test]
    fn fixture_confidences() {
        let fixture = fixture();
        for (kind, function) in [
            ("choice", choice_confidence as fn(&[f64]) -> f64),
            ("score", score_confidence),
        ] {
            for case in fixture["confidence"][kind].as_array().unwrap() {
                let probabilities: Vec<f64> = case["probabilities"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p.as_f64().unwrap())
                    .collect();
                let expected = case["confidence"].as_f64().unwrap();
                assert!(
                    (function(&probabilities) - expected).abs() < 1e-9,
                    "{kind} {probabilities:?}"
                );
            }
        }
    }

    #[test]
    fn fixture_retry_after() {
        for case in fixture()["retry_after"].as_array().unwrap() {
            let (seconds, millis) =
                capped_retry_after(case["retry_after"].as_str().unwrap()).unwrap();
            assert_eq!(seconds, case["expected"].as_str().unwrap());
            assert_eq!(millis, case["expected_ms"].as_str().unwrap());
        }
    }

    #[test]
    fn refused_item_is_a_validation_error() {
        let native = json!({
            "items": [{"id": "item-0", "data": {}, "error": {"code": "INPUT_TOO_LONG", "message": "does not fit"}}],
        });
        let questions: Map<String, Value> = [("q".to_string(), json!({"type": "noul"}))]
            .into_iter()
            .collect();
        assert_eq!(
            response_from_extract(&native, "m", &questions).unwrap(),
            Translated::Refused(
                invalid(&["state"], "does not fit"),
                "INPUT_TOO_LONG".to_string()
            )
        );
        let unreadable = json!({
            "items": [{"id": "item-0", "data": {}, "error": {"code": "INVALID_INPUT", "message": "not JSON"}}],
        });
        assert_eq!(
            response_from_extract(&unreadable, "m", &questions).unwrap(),
            Translated::Refused(invalid(&["state"], "not JSON"), "INVALID_INPUT".to_string())
        );
    }

    #[test]
    fn unreadable_probabilities_are_a_translation_failure() {
        let questions: Map<String, Value> = [("q".to_string(), json!({"type": "score"}))]
            .into_iter()
            .collect();
        for probabilities in [
            json!({"0": 0.5, "high": 0.5}),
            json!({"0": 0.5, "1": "half"}),
        ] {
            let native = json!({"items": [{"id": "item-0", "data": {
                "q": {"type": "score", "score": 0.5, "probabilities": probabilities}
            }}]});
            assert!(
                response_from_extract(&native, "m", &questions).is_err(),
                "{probabilities}"
            );
        }
    }

    #[test]
    fn error_parts_reads_every_native_error_shape() {
        for (body, code, message) in [
            (
                json!({"detail": {"code": "MODEL_LOADING", "message": "loading"}}),
                Some("MODEL_LOADING"),
                Some("loading"),
            ),
            (
                json!({"error": {"code": "INVALID_INPUT", "message": "bad"}}),
                Some("INVALID_INPUT"),
                Some("bad"),
            ),
            (
                json!({"error": "all_items_failed", "details": [{"item_index": 0, "error": "boom", "code": "INFERENCE_ERROR"}]}),
                Some("INFERENCE_ERROR"),
                Some("boom"),
            ),
            (
                json!({"detail": [{"loc": ["body"], "msg": "field required"}]}),
                None,
                Some("field required"),
            ),
            (Value::Null, None, None),
        ] {
            assert_eq!(error_parts(&body), (code, message), "{body}");
        }
    }

    #[test]
    fn missing_answer_is_a_translation_failure() {
        let native = json!({"items": [{"id": "item-0", "data": {}}]});
        let questions: Map<String, Value> = [("q".to_string(), json!({"type": "noul"}))]
            .into_iter()
            .collect();
        assert!(response_from_extract(&native, "m", &questions).is_err());
    }

    #[tokio::test]
    async fn native_errors_take_the_system_one_shapes() {
        let bad = api_error(
            StatusCode::BAD_REQUEST,
            Some("INVALID_INPUT"),
            "bad question",
        );
        let translated = translate_error(bad).await;
        assert_eq!(translated.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body: Value =
            serde_json::from_slice(&to_bytes(translated.into_body(), 1 << 20).await.unwrap())
                .unwrap();
        assert_eq!(body["detail"][0]["msg"], "bad question");

        let mut loading = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"detail": {"code": "MODEL_LOADING", "message": "loading"}})),
        )
            .into_response();
        loading.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            HeaderValue::from_static("270"),
        );
        let translated = translate_error(loading).await;
        assert_eq!(translated.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(translated.headers()["retry-after"], "60");
        assert_eq!(translated.headers()["retry-after-ms"], "60000");
        let body: Value =
            serde_json::from_slice(&to_bytes(translated.into_body(), 1 << 20).await.unwrap())
                .unwrap();
        assert_eq!(body["detail"]["error_type"], "api_error");
        assert_eq!(body["detail"]["code"], "MODEL_LOADING");
    }

    use crate::handlers::test_support::{
        TestGateway, LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL, LOCAL_LANE,
    };

    fn systemone_request(body: Value) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/systemone")
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap()
    }

    async fn body_json(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn answers_through_the_extract_queue() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &["acme/decide"])
            .await;
        gateway.dispatcher.return_extract_data(json!({
            "spam": {"type": "noul", "noul": 0.9, "answer": true, "confidence": 0.9},
            "intent": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.8, "support": 0.1, "other": 0.1},
                "confidence": 0.1
            }
        }));
        let request = systemone_request(json!({
            "model": "acme/decide",
            "state": "I was charged twice.",
            "questions": {
                "spam": {"type": "noul"},
                "intent": {"type": "choice", "criteria": {"billing": null, "support": null, "other": null}}
            },
            "temperature": 0
        }));
        let response = proxy_systemone(State(Arc::clone(&gateway.state)), request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["model"], "acme/decide");
        assert_eq!(body["answers"]["spam"]["noul"], 0.9);
        assert!((body["answers"]["intent"]["confidence"].as_f64().unwrap() - 0.7).abs() < 1e-9);
        assert_eq!(body["usage"]["output_tokens"], 0);
        assert!(body["usage"]["input_tokens"].is_u64());
        let dispatched = gateway.dispatcher.dispatched();
        assert_eq!(dispatched.len(), 1);
        assert_eq!(dispatched[0].endpoint, "extract");
        assert_eq!(dispatched[0].model, "acme/decide");
    }

    #[tokio::test]
    async fn configured_aliases_reach_the_native_extract_queue() {
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
                "q": {"type":"noul", "noul":0.9, "answer":true}
            }));
            let response = proxy_systemone(
                State(Arc::clone(&gateway.state)),
                systemone_request(json!({
                    "model":model, "state":"Continue?", "questions":{"q":{"type":"noul"}}
                })),
            )
            .await;
            let status = response.status();
            let body = body_json(response).await;
            assert_eq!(status, StatusCode::OK, "{model}: {body}");
            assert_eq!(body["model"], model);
            assert_eq!(body["answers"]["q"]["answer"], true);
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

    struct HiddenModels;

    impl crate::server::ModelAccessPolicy for HiddenModels {
        fn visible(&self, _: &str, _: &axum::http::Extensions) -> bool {
            false
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
            hidden.install_policy(Arc::new(HiddenModels));
            for model in ["decision", "DECISION"] {
                let payload = json!({"model": model, "state": "Continue?", "questions": {"q": {"type": "noul"}}});
                let denied = proxy_systemone(
                    State(Arc::clone(&hidden.state)),
                    systemone_request(payload.clone()),
                )
                .await;
                let missing =
                    proxy_systemone(State(Arc::clone(&absent.state)), systemone_request(payload))
                        .await;
                assert_eq!(denied.status(), StatusCode::NOT_FOUND);
                assert_eq!(missing.status(), StatusCode::NOT_FOUND);
                let denied = body_json(denied).await;
                assert_eq!(denied["detail"]["code"], "MODEL_NOT_FOUND");
                assert_eq!(denied, body_json(missing).await);
                assert!(!denied.to_string().contains("acme/"));
            }
            assert!(hidden.dispatcher.dispatched().is_empty());
            assert!(absent.dispatcher.dispatched().is_empty());
        }
    }

    #[tokio::test]
    async fn configured_aliases_preserve_decision_capability_checks() {
        let mut gateway = TestGateway::new(&[LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL]).await;
        gateway.add_model_alias("encode", "acme/local");
        gateway.add_model_alias("missing", "acme/missing");
        for (model, status) in [
            ("encode", StatusCode::UNPROCESSABLE_ENTITY),
            ("missing", StatusCode::NOT_FOUND),
        ] {
            let response = proxy_systemone(
                State(Arc::clone(&gateway.state)),
                systemone_request(json!({
                    "model":model, "state":"Continue?", "questions":{"q":{"type":"noul"}}
                })),
            )
            .await;
            assert_eq!(response.status(), status, "{model}");
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn an_undelivered_result_is_a_gateway_owned_fault() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &["acme/decide"])
            .await;
        gateway
            .dispatcher
            .return_extract_data(json!({"other": {"type": "noul", "noul": 0.5}}));
        let request = systemone_request(json!({
            "model": "acme/decide", "state": "hi", "questions": {"q": {"type": "noul"}}
        }));
        let response = proxy_systemone(State(Arc::clone(&gateway.state)), request).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.extensions().get::<GatewayOwnedFault>(),
            Some(&GatewayOwnedFault::new("format_systemone"))
        );
        assert_eq!(
            body_json(response).await["detail"]["code"],
            "INTERNAL_ERROR"
        );
    }

    #[tokio::test]
    async fn translated_errors_keep_the_fault_marker() {
        let marked = GatewayOwnedFault::new("upstream").mark(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            Some("INTERNAL_ERROR"),
            "boom",
        ));
        let translated = translate_error(marked).await;
        assert_eq!(
            translated.extensions().get::<GatewayOwnedFault>(),
            Some(&GatewayOwnedFault::new("upstream"))
        );
    }

    #[tokio::test]
    async fn refuses_a_model_that_does_not_answer_typed_questions() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL, LOCAL_ENCODE_MODEL]).await;
        let request = systemone_request(json!({
            "model": "acme/local", "state": "hi", "questions": {"q": {"type": "noul"}}
        }));
        let response = proxy_systemone(State(Arc::clone(&gateway.state)), request).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = body_json(response).await;
        assert_eq!(body["detail"][0]["loc"], json!(["body", "model"]));
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn unknown_model_is_not_found() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
        let request = systemone_request(json!({
            "model": "acme/missing", "state": "hi", "questions": {"q": {"type": "noul"}}
        }));
        let response = proxy_systemone(State(Arc::clone(&gateway.state)), request).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body_json(response).await["detail"]["error_type"],
            "not_found_error"
        );
    }

    #[tokio::test]
    async fn malformed_json_is_a_validation_error() {
        let gateway = TestGateway::new(&[LOCAL_DECISION_MODEL]).await;
        let request = Request::builder()
            .method(Method::POST)
            .uri("/v1/systemone")
            .body(Body::from("{"))
            .unwrap();
        let response = proxy_systemone(State(Arc::clone(&gateway.state)), request).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body_json(response).await["detail"].is_array());
    }
}
