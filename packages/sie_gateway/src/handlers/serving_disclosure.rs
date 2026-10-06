//! `X-SIE-Served-By` and `X-SIE-Upstream`: which side served a request.
//!
//! A route records the model it dispatches to once routing has resolved it,
//! and the route's handler stamps its response from that record. The value
//! comes from the dispatched route's configuration, as on the single server:
//! a route whose default profile uses a remote adapter is served remotely, by
//! the upstream that profile names.

use std::sync::{Arc, Mutex, PoisonError};

use axum::extract::Request;
use axum::http::{Extensions, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::Response;
use futures_util::FutureExt;

use crate::observability::metrics as telemetry;
use crate::server::{AppState, RemoteRouteReason};
use crate::types::model::{FallbackTrigger, ServedBy};

pub(crate) const SERVED_BY_HEADER: HeaderName = HeaderName::from_static("x-sie-served-by");
pub(crate) const UPSTREAM_HEADER: HeaderName = HeaderName::from_static("x-sie-upstream");

pub(crate) const REMOTE_HEADER: &str = "x-sie-remote";

/// Absent means ordinary serving; the sole explicit value is `forbid`.
/// Refuse duplicates and non-UTF8/unknown values without echoing their bytes.
pub(crate) fn remote_forbidden(headers: &HeaderMap) -> Result<bool, &'static str> {
    let mut values = headers.get_all(REMOTE_HEADER).iter();
    let Some(value) = values.next() else {
        return Ok(false);
    };
    if values.next().is_some() || value.as_bytes() != b"forbid" {
        return Err("X-SIE-Remote accepts only one value: 'forbid'");
    }
    Ok(true)
}

/// The side a request was dispatched to, recorded by the route that resolved it.
#[derive(Clone, Default)]
pub(crate) struct ServingDisclosure(Arc<Mutex<Option<ServedBy>>>);

impl ServingDisclosure {
    /// The request's record, installed when the request does not carry one yet.
    pub(crate) fn install(req: &mut Request) -> Self {
        if let Some(existing) = req.extensions().get::<Self>() {
            return existing.clone();
        }
        let disclosure = Self::default();
        req.extensions_mut().insert(disclosure.clone());
        disclosure
    }

    /// Record that the request carrying `extensions` is dispatched to `model`.
    pub(crate) fn record(state: &AppState, extensions: &Extensions, model: &str) {
        Self::record_evidence(extensions, state.model_registry.served_by(model));
    }

    /// Record disclosure paired with the immutable execution evidence.
    pub(crate) fn record_evidence(extensions: &Extensions, served_by: Option<ServedBy>) {
        if let Some(disclosure) = extensions.get::<Self>() {
            *disclosure.0.lock().unwrap_or_else(PoisonError::into_inner) = served_by;
        }
    }

    /// Stamp a served response or a server-side refusal. A client error says
    /// nothing about which side serves the model, so it carries neither header.
    pub(crate) fn stamp(&self, status: StatusCode, headers: &mut HeaderMap) {
        if !(status.is_success() || status.is_server_error()) {
            return;
        }
        let served_by = self
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        match served_by {
            None => {}
            Some(ServedBy::Local) => {
                headers.insert(SERVED_BY_HEADER, HeaderValue::from_static("local"));
                headers.remove(UPSTREAM_HEADER);
            }
            Some(ServedBy::Remote { upstream }) => {
                headers.insert(SERVED_BY_HEADER, HeaderValue::from_static("remote"));
                match upstream.and_then(|name| HeaderValue::from_str(&name).ok()) {
                    Some(value) => {
                        headers.insert(UPSTREAM_HEADER, value);
                    }
                    None => {
                        headers.remove(UPSTREAM_HEADER);
                    }
                }
            }
        }
    }
}

/// One request's original pre-acceptance refusal, retained through its bridge.
/// It is not a retry counter: a request may install exactly one remote attempt.
#[derive(Clone, Default)]
pub(crate) struct FallbackAttempt(Arc<Mutex<FallbackState>>);

#[derive(Default)]
struct FallbackState {
    original: Option<LocalRefusal>,
    observation: Option<ServingObservation>,
    route_decisions: Vec<(String, RemoteRouteReason, bool)>,
}

struct ServingObservation {
    model: String,
    operation: &'static str,
}

/// A compatibility facade owns response validation before a bridge commits.
/// This extension is gateway-created and cannot be supplied by a caller.
#[derive(Clone)]
pub(crate) struct DeferredFallbackFinish;

struct LocalRefusal {
    response: Response,
    trigger: FallbackTrigger,
}

/// Marks a failed response whose gateway deadline passed before the
/// dispatched work answered.
#[derive(Clone)]
pub(crate) struct UnansweredBeforeDeadline;

/// The single server's error codes (`ErrorCode` in `sie_server.types.responses`).
const SIE_ERROR_CODES: [&str; 11] = [
    "INVALID_INPUT",
    "MODEL_NOT_FOUND",
    "MODEL_NOT_LOADED",
    "LORA_LOADING",
    "MODEL_LOADING",
    "MODEL_LOAD_FAILED",
    "INFERENCE_ERROR",
    "QUEUE_FULL",
    "INTERNAL_ERROR",
    "RESOURCE_EXHAUSTED",
    "INPUT_TOO_LONG",
];

/// `X-SIE-Fallback-Error` for a failed remote attempt that answered with
/// `status` and error `code`, as the single server's `_fallback_error`
/// (`sie_server.api.routing`) derives it.
fn fallback_error(status: StatusCode, code: Option<&str>) -> &'static str {
    if let Some(code) = code {
        let upper = code.to_uppercase();
        if let Some(known) = SIE_ERROR_CODES.iter().find(|known| **known == upper) {
            return known;
        }
        match code {
            "server_overloaded" => return "QUEUE_FULL",
            "invalid_request" => return "INVALID_INPUT",
            _ => {}
        }
    }
    if status.as_u16() >= 500 {
        "INFERENCE_ERROR"
    } else {
        "INVALID_INPUT"
    }
}

const ATTEMPT_ERROR_BODY_LIMIT: usize = 64 * 1024;

/// A failed attempt's error code: its `X-SIE-Error-Code`, or else the `code`
/// of its JSON `error` or `detail` object. Only a body that is complete
/// without waiting is read.
fn attempt_error_code(response: &mut Response) -> Option<String> {
    if let Some(code) = response
        .headers()
        .get("x-sie-error-code")
        .and_then(|value| value.to_str().ok())
    {
        return Some(code.to_string());
    }
    let body = std::mem::take(response.body_mut());
    let bytes = axum::body::to_bytes(body, ATTEMPT_ERROR_BODY_LIMIT)
        .now_or_never()?
        .ok()?;
    let body: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    body.get("error")
        .filter(|error| error.is_object())
        .or_else(|| body.get("detail"))?
        .get("code")?
        .as_str()
        .map(str::to_string)
}

impl FallbackAttempt {
    pub(crate) fn install(req: &mut Request) -> Self {
        if let Some(existing) = req.extensions().get::<Self>() {
            return existing.clone();
        }
        let attempt = Self::default();
        req.extensions_mut().insert(attempt.clone());
        attempt
    }

    /// Only a resolved catalog route can name a telemetry model. Preserve the
    /// original route while recursion selects a remote physical profile.
    pub(crate) fn record_model(extensions: &Extensions, model: &str, operation: &str) {
        let Some(attempt) = extensions.get::<Self>() else {
            return;
        };
        let operation = match operation {
            "encode" => "encode",
            "score" => "score",
            "extract" => "extract",
            "generate" => "generate",
            _ => return,
        };
        let mut state = attempt.0.lock().unwrap_or_else(PoisonError::into_inner);
        if state.original.is_none() && state.observation.is_none() {
            state.observation = Some(ServingObservation {
                model: model.split(':').next().unwrap_or(model).to_string(),
                operation,
            });
        }
    }

    /// Transfer local streaming observation to the output driver. HTTP 200
    /// alone does not prove a stream served any valid local output.
    pub(crate) fn defer_local_stream(extensions: &Extensions) -> Option<String> {
        let disclosure = extensions.get::<ServingDisclosure>()?;
        if !matches!(
            *disclosure.0.lock().unwrap_or_else(PoisonError::into_inner),
            Some(ServedBy::Local)
        ) {
            return None;
        }
        let attempt = extensions.get::<Self>()?;
        let mut state = attempt.0.lock().unwrap_or_else(PoisonError::into_inner);
        if state.original.is_some() {
            return None;
        }
        state
            .observation
            .take()
            .map(|observation| observation.model)
    }

    /// Retain the response before any local work has been accepted.
    pub(crate) fn begin(&self, response: Response, trigger: FallbackTrigger) -> bool {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        if state.original.is_some() {
            return false;
        }
        state.original = Some(LocalRefusal { response, trigger });
        true
    }

    pub(crate) fn active(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .original
            .is_some()
    }

    /// A deployment's decision on routing this request to `remote_model` for
    /// `reason`. It is made once per request, remote profile and reason, and
    /// every later question gets the same answer.
    pub(crate) fn remote_route_decision(
        extensions: &Extensions,
        remote_model: &str,
        reason: RemoteRouteReason,
        decide: impl FnOnce() -> bool,
    ) -> bool {
        let Some(attempt) = extensions.get::<Self>() else {
            return decide();
        };
        let known = |state: &FallbackState| {
            state
                .route_decisions
                .iter()
                .find(|(model, decided_reason, _)| {
                    model == remote_model && *decided_reason == reason
                })
                .map(|(_, _, admitted)| *admitted)
        };
        if let Some(admitted) = known(&attempt.0.lock().unwrap_or_else(PoisonError::into_inner)) {
            return admitted;
        }
        let admitted = decide();
        let mut state = attempt.0.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(first) = known(&state) {
            return first;
        }
        state
            .route_decisions
            .push((remote_model.to_string(), reason, admitted));
        admitted
    }

    /// The trigger of the local refusal this request's remote attempt stands
    /// in for, while that refusal is held.
    pub(crate) fn trigger(&self) -> Option<FallbackTrigger> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .original
            .as_ref()
            .map(|original| original.trigger)
    }

    /// Called after normal disclosure stamping and before HTTP success/output.
    /// A failed bridge preserves the original body and Retry-After verbatim.
    pub(crate) fn finish(&self, mut response: Response) -> Response {
        let (original, observation) = {
            let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            (state.original.take(), state.observation.take())
        };
        let Some(mut original) = original else {
            if response.status().is_success()
                && response
                    .headers()
                    .get(SERVED_BY_HEADER)
                    .is_some_and(|value| value == "local")
            {
                if let Some(observation) = observation {
                    telemetry::record_local_serving_success(&observation.model);
                }
            }
            return response;
        };
        if let Some(observation) = observation {
            telemetry::record_remote_fallback(
                &observation.model,
                observation.operation,
                original.trigger,
                response.status().is_success(),
            );
        }
        let reason = HeaderValue::from_static(original.trigger.as_str());
        if response.status().is_success() {
            response
                .headers_mut()
                .insert("x-sie-fallback-reason", reason);
            return response;
        }
        let failure = if response
            .extensions()
            .get::<UnansweredBeforeDeadline>()
            .is_some()
        {
            "QUEUE_FULL"
        } else {
            let status = response.status();
            fallback_error(status, attempt_error_code(&mut response).as_deref())
        };
        let headers = original.response.headers_mut();
        headers.insert(SERVED_BY_HEADER, HeaderValue::from_static("local"));
        headers.remove(UPSTREAM_HEADER);
        headers.insert("x-sie-fallback-reason", reason);
        headers.insert("x-sie-fallback-error", HeaderValue::from_static(failure));
        original.response
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Method, StatusCode};
    use axum::response::{IntoResponse, Response};
    use serde_json::json;

    use super::*;

    #[test]
    fn local_stream_observation_transfers_once_and_excludes_remote_routes_and_bridges() {
        for (served_by, bridge, expected) in [
            (Some(ServedBy::Local), false, Some("acme/chat")),
            (Some(ServedBy::Remote { upstream: None }), false, None),
            (None, false, None),
            (Some(ServedBy::Local), true, None),
        ] {
            let mut req = Request::new(Body::empty());
            ServingDisclosure::install(&mut req);
            let attempt = FallbackAttempt::install(&mut req);
            ServingDisclosure::record_evidence(req.extensions(), served_by);
            FallbackAttempt::record_model(req.extensions(), "acme/chat:default", "generate");
            if bridge {
                assert!(attempt.begin(
                    StatusCode::SERVICE_UNAVAILABLE.into_response(),
                    FallbackTrigger::Provisioning,
                ));
            }
            assert_eq!(
                FallbackAttempt::defer_local_stream(req.extensions()).as_deref(),
                expected,
            );
            if expected.is_some() {
                assert!(FallbackAttempt::defer_local_stream(req.extensions()).is_none());
                assert!(attempt.0.lock().unwrap().observation.is_none());
            } else {
                assert!(attempt.0.lock().unwrap().observation.is_some());
            }
        }
    }

    async fn buffered_surface(
        gateway: &TestGateway,
        surface: &str,
        options: serde_json::Value,
    ) -> Response {
        let state = State(Arc::clone(&gateway.state));
        let model = if options.get("profile").is_some() {
            "default:/acme/chat"
        } else {
            "acme/chat"
        };
        match surface {
            "native" => {
                proxy_request(
                    state,
                    json_request(
                        "/v1/generate/acme/chat",
                        json!({"prompt":"hello", "max_new_tokens":4, "options":options}),
                    ),
                    "generate",
                )
                .await
            }
            "chat" => {
                proxy_chat(
                    state,
                    json_request(
                        "/v1/chat/completions",
                        json!({"model":model, "messages":[{"role":"user","content":"hello"}]}),
                    ),
                )
                .await
            }
            "completions" => {
                proxy_completions(
                    state,
                    json_request(
                        "/v1/completions",
                        json!({"model":model, "prompt":"hello", "max_tokens":4}),
                    ),
                )
                .await
            }
            "responses" => {
                proxy_responses(
                    state,
                    json_request("/v1/responses", json!({"model":model, "input":"hello"})),
                )
                .await
            }
            _ => unreachable!(),
        }
    }

    async fn degraded_gateway(config: &str, degradation: &str, model: &str) -> TestGateway {
        let gateway = TestGateway::new(&[config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        if degradation == "saturated" {
            gateway
                .add_saturated_worker("local-1", LOCAL_LANE, &[model])
                .await;
        } else {
            gateway
                .add_verified_worker("local-1", LOCAL_LANE, &[model])
                .await;
            if degradation == "unhealthy" {
                gateway
                    .state
                    .registry
                    .mark_unhealthy("http://local-1:8080")
                    .await;
            } else {
                gateway.dispatcher.saturate_local_queue();
            }
        }
        gateway
    }

    #[tokio::test]
    async fn spill_fallback_bridges_all_buffered_generation_surfaces_before_local_acceptance() {
        for degradation in ["saturated", "unhealthy", "queue_full"] {
            let trigger = if degradation == "unhealthy" {
                "unhealthy"
            } else {
                "saturated"
            };
            let config = format!("{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n  triggers: [{trigger}]\n");
            for surface in ["native", "chat", "completions", "responses"] {
                for fail in [false, true] {
                    let gateway = degraded_gateway(&config, degradation, "acme/chat").await;
                    if fail {
                        gateway.dispatcher.refuse_generation();
                    }
                    let response = buffered_surface(&gateway, surface, json!({})).await;
                    assert_eq!(
                        response.status(),
                        if fail {
                            StatusCode::SERVICE_UNAVAILABLE
                        } else {
                            StatusCode::OK
                        },
                        "{surface}/{degradation}"
                    );
                    assert_eq!(response.headers()["x-sie-fallback-reason"], trigger);
                    assert_eq!(
                        stamped(&response),
                        if fail {
                            (Some("local"), None)
                        } else {
                            (Some("remote"), Some("team-sie"))
                        }
                    );
                    if fail {
                        assert_eq!(
                            response.headers()["retry-after"],
                            if trigger == "unhealthy" { "60" } else { "5" }
                        );
                        assert_eq!(
                            response.headers()["x-sie-fallback-error"],
                            "INFERENCE_ERROR"
                        );
                    }
                    assert_eq!(
                        gateway.dispatcher.dispatched(),
                        if fail {
                            Vec::new()
                        } else {
                            vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
                        }
                    );
                    assert!(gateway
                        .state
                        .demand_tracker
                        .active_lanes()
                        .iter()
                        .any(|lane| lane.bundle() == LOCAL_LANE.2));
                }
            }
        }
    }

    #[tokio::test]
    async fn spill_fallback_streams_preserve_the_before_and_after_output_boundary() {
        for degradation in ["saturated", "unhealthy", "queue_full"] {
            let trigger = if degradation == "unhealthy" {
                "unhealthy"
            } else {
                "saturated"
            };
            let config = format!("{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n  triggers: [{trigger}]\n");
            for surface in ["native", "chat", "completions"] {
                for failure in [None, Some(false), Some(true)] {
                    let gateway = degraded_gateway(&config, degradation, "acme/chat").await;
                    if let Some(after_output) = failure {
                        gateway.dispatcher.fail_stream(after_output);
                    }
                    let state = State(Arc::clone(&gateway.state));
                    let response = match surface {
                        "native" => proxy_request(state, json_request("/v1/generate/acme/chat", json!({"prompt":"hello","max_new_tokens":4,"stream":true})), "generate").await,
                        "chat" => proxy_chat(state, json_request("/v1/chat/completions", json!({"model":"acme/chat","messages":[{"role":"user","content":"hello"}],"stream":true}))).await,
                        "completions" => proxy_completions(state, json_request("/v1/completions", json!({"model":"acme/chat","prompt":"hello","max_tokens":4,"stream":true}))).await,
                        _ => unreachable!(),
                    };
                    assert_eq!(response.headers()["x-sie-fallback-reason"], trigger);
                    if failure == Some(false) {
                        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                        assert_eq!(
                            response.headers()["retry-after"],
                            if trigger == "unhealthy" { "60" } else { "5" }
                        );
                        assert_eq!(stamped(&response), (Some("local"), None));
                    } else {
                        assert_eq!(response.status(), StatusCode::OK);
                        assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
                    }
                    let body = axum::body::to_bytes(response.into_body(), 8192)
                        .await
                        .unwrap();
                    let body = String::from_utf8_lossy(&body);
                    assert!(!body.contains("private upstream failure"));
                    if failure == Some(true) {
                        assert!(body.contains("ok"));
                        assert!(body.contains("inference_error"));
                        assert!(body.contains("[DONE]"));
                    }
                    assert_eq!(
                        gateway.dispatcher.dispatched(),
                        vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn spill_fallback_requires_opt_in_and_preserves_explicit_default() {
        for degradation in ["saturated", "unhealthy", "queue_full"] {
            for opt_in in [false, true] {
                let triggers = if opt_in {
                    "  triggers: [saturated, unhealthy]\n"
                } else {
                    ""
                };
                let config = format!("{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n{triggers}");
                for surface in ["native", "chat", "completions", "responses"] {
                    let gateway = degraded_gateway(&config, degradation, "acme/chat").await;
                    let response = buffered_surface(
                        &gateway,
                        surface,
                        if opt_in {
                            json!({"profile":"default"})
                        } else {
                            json!({})
                        },
                    )
                    .await;
                    assert!(
                        !response.headers().contains_key("x-sie-fallback-reason"),
                        "{surface}/{degradation}/{opt_in}"
                    );
                    assert!(!gateway
                        .dispatcher
                        .dispatched()
                        .iter()
                        .any(|work| work.bundle == REMOTE_LANE.2));
                    if degradation == "unhealthy" {
                        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                        assert!(gateway.dispatcher.dispatched().is_empty());
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn spill_fallback_covers_extraction_json_and_msgpack() {
        for degradation in ["saturated", "unhealthy", "queue_full"] {
            let trigger = if degradation == "unhealthy" {
                "unhealthy"
            } else {
                "saturated"
            };
            let config = format!("{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n  triggers: [{trigger}]\n");
            for msgpack in [false, true] {
                for fail in [false, true] {
                    let gateway = degraded_gateway(&config, degradation, "acme/extract").await;
                    if fail {
                        gateway.dispatcher.refuse_work();
                    }
                    let response = proxy_request(
                        State(Arc::clone(&gateway.state)),
                        extraction_request(msgpack, json!({})),
                        "extract",
                    )
                    .await;
                    assert_eq!(
                        response.status(),
                        if fail {
                            StatusCode::SERVICE_UNAVAILABLE
                        } else {
                            StatusCode::OK
                        }
                    );
                    assert_eq!(response.headers()["x-sie-fallback-reason"], trigger);
                    assert_eq!(
                        gateway.dispatcher.dispatched(),
                        vec![dispatched("extract", REMOTE_LANE, "acme/extract:remote")]
                    );
                    if fail {
                        assert_eq!(
                            response.headers()["retry-after"],
                            if trigger == "unhealthy" { "60" } else { "5" }
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn all_buffered_cluster_fallback_surfaces_warm_before_bridging_and_leave_loaded_models_local(
    ) {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for loaded in [false, true] {
            let gateway = TestGateway::new(&[&config]).await;
            gateway
                .add_verified_worker(
                    "local-1",
                    LOCAL_LANE,
                    if loaded { &["acme/chat"] } else { &[] },
                )
                .await;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            for surface in ["native", "chat", "completions", "responses"] {
                let response = buffered_surface(&gateway, surface, json!({})).await;
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "{surface}, loaded={loaded}"
                );
                if loaded {
                    assert_eq!(stamped(&response), (Some("local"), None));
                } else {
                    assert_eq!(response.headers()["x-sie-fallback-reason"], "model_loading");
                }
            }
            let expected = if loaded {
                vec![dispatched("generate", LOCAL_LANE, "acme/chat"); 4]
            } else {
                [
                    dispatched("load", LOCAL_LANE, "acme/chat"),
                    dispatched("generate", REMOTE_LANE, "acme/chat:remote"),
                ]
                .into_iter()
                .cycle()
                .take(8)
                .collect()
            };
            assert_eq!(gateway.dispatcher.dispatched(), expected);
        }
    }

    #[tokio::test]
    async fn all_buffered_cluster_fallback_surfaces_preserve_explicit_default_and_failed_load_refusal(
    ) {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        for surface in ["native", "chat", "completions", "responses"] {
            let response = buffered_surface(&gateway, surface, json!({"profile":"default"})).await;
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{surface}"
            );
            assert!(!response.headers().contains_key("x-sie-fallback-reason"));
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &[])
            .await;
        gateway.dispatcher.refuse_model_loads();
        for surface in ["native", "chat", "completions", "responses"] {
            let response = buffered_surface(&gateway, surface, json!({})).await;
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{surface}"
            );
            assert_eq!(response.headers()["retry-after"], "5");
            assert!(!response.headers().contains_key("x-sie-fallback-error"));
        }
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![dispatched("load", LOCAL_LANE, "acme/chat"); 4]
        );
    }

    #[tokio::test]
    async fn all_buffered_cluster_fallback_surfaces_restore_original_cold_refusal_after_remote_failure(
    ) {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        gateway.dispatcher.refuse_generation();
        for surface in ["native", "chat", "completions", "responses"] {
            let response = buffered_surface(&gateway, surface, json!({})).await;
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{surface}"
            );
            assert_eq!(response.headers()["retry-after"], "60");
            assert_eq!(
                response.headers()["x-sie-fallback-error"],
                "INFERENCE_ERROR"
            );
            assert_eq!(stamped(&response), (Some("local"), None));
            let body = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            assert!(!String::from_utf8_lossy(&body).contains("private upstream failure"));
        }
    }

    #[tokio::test]
    async fn buffered_cluster_fallback_rejects_model_output_limits_before_warming_or_dispatch() {
        let config = format!(
            "{}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n",
            HYBRID_GENERATE_MODEL.replace("generate: {}", "generate:\n    max_output_tokens: 1")
        );
        for unloaded in [false, true] {
            let gateway = TestGateway::new(&[&config]).await;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            if unloaded {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &[])
                    .await;
            }
            for surface in ["native", "chat", "completions", "responses"] {
                let response = buffered_surface(&gateway, surface, json!({})).await;
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "{surface}, unloaded={unloaded}"
                );
                assert!(!response.headers().contains_key("x-sie-fallback-reason"));
            }
            assert!(gateway.dispatcher.dispatched().is_empty());
            assert!(gateway.state.demand_tracker.active_lanes().is_empty());
        }
    }

    #[tokio::test]
    async fn streaming_cluster_fallback_commits_only_after_first_valid_event() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for surface in ["native", "chat", "completions"] {
            for unloaded in [false, true] {
                for failure in [None, Some(false), Some(true)] {
                    let gateway = TestGateway::new(&[&config]).await;
                    gateway
                        .add_verified_worker("remote-1", REMOTE_LANE, &[])
                        .await;
                    if unloaded {
                        gateway
                            .add_verified_worker("local-1", LOCAL_LANE, &[])
                            .await;
                    }
                    if let Some(after_output) = failure {
                        gateway.dispatcher.fail_stream(after_output);
                    }
                    let state = State(Arc::clone(&gateway.state));
                    let response = match surface {
                    "native" => proxy_request(state, json_request("/v1/generate/acme/chat", json!({"prompt":"hello","max_new_tokens":4,"stream":true})), "generate").await,
                    "chat" => proxy_chat(state, json_request("/v1/chat/completions", json!({"model":"acme/chat","messages":[{"role":"user","content":"hello"}],"stream":true}))).await,
                    "completions" => proxy_completions(state, json_request("/v1/completions", json!({"model":"acme/chat","prompt":"hello","max_tokens":4,"stream":true}))).await,
                    _ => unreachable!(),
                };
                    if failure == Some(false) {
                        assert_eq!(
                            response.status(),
                            StatusCode::SERVICE_UNAVAILABLE,
                            "{surface}"
                        );
                        assert_eq!(
                            response.headers()["retry-after"],
                            if unloaded { "5" } else { "60" }
                        );
                        assert_eq!(stamped(&response), (Some("local"), None));
                        assert_eq!(
                            response.headers()["x-sie-fallback-error"],
                            "INFERENCE_ERROR"
                        );
                    } else {
                        assert_eq!(response.status(), StatusCode::OK, "{surface}");
                        assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
                        assert_eq!(response.headers()["content-type"], "text/event-stream");
                        assert!(!response.headers().contains_key("x-sie-fallback-error"));
                    }
                    let body = axum::body::to_bytes(response.into_body(), 8192)
                        .await
                        .unwrap();
                    let body = String::from_utf8_lossy(&body);
                    assert!(!body.contains("private upstream failure"));
                    if failure == Some(true) {
                        assert!(body.contains("ok"));
                        assert!(body.contains("inference_error"), "{surface}: {body}");
                        assert!(body.contains("[DONE]"));
                    }
                    assert_eq!(
                        gateway.dispatcher.dispatched(),
                        if unloaded {
                            vec![
                                dispatched("load", LOCAL_LANE, "acme/chat"),
                                dispatched("generate", REMOTE_LANE, "acme/chat:remote"),
                            ]
                        } else {
                            vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
                        }
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn streaming_cluster_fallback_restores_refusal_for_terminal_only_failures() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for surface in ["native", "chat", "completions"] {
            for reason in ["cancelled", "error"] {
                let gateway = TestGateway::new(&[&config]).await;
                gateway
                    .add_verified_worker("remote-1", REMOTE_LANE, &[])
                    .await;
                gateway.dispatcher.fail_stream_terminal(reason);
                let state = State(Arc::clone(&gateway.state));
                let response = match surface {
                    "native" => proxy_request(state, json_request("/v1/generate/acme/chat", json!({"prompt":"hello","max_new_tokens":4,"stream":true})), "generate").await,
                    "chat" => proxy_chat(state, json_request("/v1/chat/completions", json!({"model":"acme/chat","messages":[{"role":"user","content":"hello"}],"stream":true}))).await,
                    "completions" => proxy_completions(state, json_request("/v1/completions", json!({"model":"acme/chat","prompt":"hello","max_tokens":4,"stream":true}))).await,
                    _ => unreachable!(),
                };
                assert_eq!(
                    response.status(),
                    StatusCode::SERVICE_UNAVAILABLE,
                    "{surface}, {reason}"
                );
                assert_eq!(response.headers()["retry-after"], "60");
                assert_eq!(
                    response.headers()["x-sie-fallback-error"],
                    "INFERENCE_ERROR"
                );
                assert_eq!(stamped(&response), (Some("local"), None));
                assert_eq!(
                    gateway.dispatcher.dispatched(),
                    vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
                );
            }
        }
    }

    #[tokio::test]
    async fn native_cluster_fallback_bridges_a_cold_model_and_retains_local_pending_demand() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let response = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/generate/acme/chat",
                json!({"prompt":"hello", "max_new_tokens":4}),
            ),
            "generate",
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
        assert_eq!(response.headers()["x-sie-fallback-reason"], "provisioning");
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
        );
        assert!(gateway
            .state
            .demand_tracker
            .active_lanes()
            .iter()
            .any(|lane| lane.machine_profile() == "l4"));
    }

    #[tokio::test]
    async fn native_cluster_fallback_keeps_explicit_default_local_for_json_and_msgpack() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let body = json!({"prompt":"hello", "max_new_tokens":4, "options":{"profile":"default"}});
        for msgpack in [false, true] {
            let request = Request::builder()
                .method(Method::POST)
                .uri("/v1/generate/acme/chat")
                .header(
                    "content-type",
                    if msgpack {
                        "application/msgpack"
                    } else {
                        "application/json"
                    },
                )
                .body(Body::from(if msgpack {
                    rmp_serde::to_vec_named(&body).unwrap()
                } else {
                    serde_json::to_vec(&body).unwrap()
                }))
                .unwrap();
            let refused =
                proxy_request(State(Arc::clone(&gateway.state)), request, "generate").await;
            assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert!(!refused.headers().contains_key("x-sie-fallback-reason"));
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn compatibility_cluster_fallback_routes_the_remote_profile_without_changing_display_model(
    ) {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let state = || State(Arc::clone(&gateway.state));
        let chat = proxy_chat(
            state(),
            json_request(
                "/v1/chat/completions",
                json!({"model":"acme/chat", "messages":[{"role":"user", "content":"hello"}]}),
            ),
        )
        .await;
        let completions = proxy_completions(
            state(),
            json_request(
                "/v1/completions",
                json!({"model":"acme/chat", "prompt":"hello", "max_tokens":4}),
            ),
        )
        .await;
        let responses = proxy_responses(
            state(),
            json_request(
                "/v1/responses",
                json!({"model":"acme/chat", "input":"hello"}),
            ),
        )
        .await;
        for response in [chat, completions, responses] {
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()["x-sie-fallback-reason"], "provisioning");
            assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
            let body = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["model"], "acme/chat");
        }
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote"); 3]
        );
    }

    #[tokio::test]
    async fn native_cluster_fallback_preserves_an_explicit_bundle_route() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let refused = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/generate/default:/acme/chat",
                json!({"prompt":"hello", "max_new_tokens":4}),
            ),
            "generate",
        )
        .await;
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!refused.headers().contains_key("x-sie-fallback-reason"));
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn native_cluster_fallback_durably_warms_an_unloaded_local_model_before_remote_execution()
    {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &[])
            .await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let response = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/generate/acme/chat",
                json!({"prompt":"hello", "max_new_tokens":4}),
            ),
            "generate",
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sie-fallback-reason"], "model_loading");
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![
                dispatched("load", LOCAL_LANE, "acme/chat"),
                dispatched("generate", REMOTE_LANE, "acme/chat:remote")
            ]
        );
    }

    #[tokio::test]
    async fn native_cluster_fallback_refuses_older_remote_workers_and_honors_caller_forbid() {
        let config = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway.add_worker("remote-old", REMOTE_LANE, &[]).await;
        let request = || {
            json_request(
                "/v1/generate/acme/chat",
                json!({"prompt":"hello", "max_new_tokens":4}),
            )
        };
        let refused = proxy_request(State(Arc::clone(&gateway.state)), request(), "generate").await;
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(refused.headers()["retry-after"], "60");
        assert_eq!(refused.headers()["x-sie-fallback-error"], "INFERENCE_ERROR");
        assert_eq!(stamped(&refused), (Some("local"), None));
        let mut forbidden = request();
        forbidden
            .headers_mut()
            .insert(REMOTE_HEADER, HeaderValue::from_static("forbid"));
        let refused = proxy_request(State(Arc::clone(&gateway.state)), forbidden, "generate").await;
        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!refused.headers().contains_key("x-sie-fallback-reason"));
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    const FORBID_SURFACES: [(&str, bool); 9] = [
        ("encode", false),
        ("embeddings", false),
        ("native", false),
        ("native", true),
        ("chat", false),
        ("chat", true),
        ("completions", false),
        ("completions", true),
        ("responses", false),
    ];

    async fn forbidden_surface(
        gateway: &TestGateway,
        surface: &str,
        model: &str,
        stream: bool,
    ) -> Response {
        let state = State(Arc::clone(&gateway.state));
        let messages = json!([{"role":"user", "content":"hello"}]);
        let mut request = match surface {
            "encode" => json_request(
                &format!("/v1/encode/{model}"),
                json!({"items":[{"text":"hello"}]}),
            ),
            "embeddings" => json_request("/v1/embeddings", json!({"model":model, "input":"hello"})),
            "native" => json_request(
                &format!("/v1/generate/{model}"),
                json!({"prompt":"hello", "max_new_tokens":4, "stream":stream}),
            ),
            "chat" => json_request(
                "/v1/chat/completions",
                json!({"model":model, "messages":messages, "stream":stream}),
            ),
            "completions" => json_request(
                "/v1/completions",
                json!({"model":model, "prompt":"hello", "max_tokens":4, "stream":stream}),
            ),
            "responses" => json_request("/v1/responses", json!({"model":model, "input":"hello"})),
            _ => unreachable!(),
        };
        request
            .headers_mut()
            .insert(REMOTE_HEADER, HeaderValue::from_static("forbid"));
        match surface {
            "encode" => proxy_request(state, request, "encode").await,
            "embeddings" => proxy_openai_embeddings(state, request).await,
            "native" => proxy_request(state, request, "generate").await,
            "chat" => proxy_chat(state, request).await,
            "completions" => proxy_completions(state, request).await,
            _ => proxy_responses(state, request).await,
        }
    }

    #[tokio::test]
    async fn remote_forbid_serves_a_model_without_a_remote_route_through_a_transport_without_the_fence(
    ) {
        let gateway = TestGateway::new(&[
            LOCAL_ENCODE_MODEL,
            REMOTE_ENCODE_MODEL,
            HYBRID_GENERATE_MODEL,
        ])
        .await;
        gateway.add_worker("local-1", LOCAL_LANE, &[]).await;
        gateway.add_worker("remote-1", REMOTE_LANE, &[]).await;
        gateway.dispatcher.withdraw_execution_authority();
        for (surface, stream) in FORBID_SURFACES {
            let model = if matches!(surface, "encode" | "embeddings") {
                "acme/local"
            } else {
                "acme/chat"
            };
            let response = forbidden_surface(&gateway, surface, model, stream).await;
            assert_eq!(response.status(), StatusCode::OK, "{surface}/{stream}");
            assert_eq!(
                stamped(&response),
                (Some("local"), None),
                "{surface}/{stream}"
            );
            let _ = axum::body::to_bytes(response.into_body(), 16384)
                .await
                .unwrap();
        }
        let dispatched = gateway.dispatcher.dispatched();
        assert_eq!(dispatched.len(), FORBID_SURFACES.len());
        assert!(dispatched.iter().all(|work| (
            work.pool.as_str(),
            work.machine_profile.as_str(),
            work.bundle.as_str()
        ) == LOCAL_LANE));
        assert_eq!(
            gateway.dispatcher.execution_authority(),
            vec![false; FORBID_SURFACES.len()]
        );
        for (surface, model) in [("encode", "acme/remote"), ("chat", "acme/chat:remote")] {
            let response = forbidden_surface(&gateway, surface, model, false).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{model}");
        }
        assert_eq!(gateway.dispatcher.dispatched().len(), FORBID_SURFACES.len());
    }

    #[tokio::test]
    async fn remote_forbid_fails_closed_for_a_model_with_a_remote_route_unless_execution_is_verified(
    ) {
        let routing = "\nrouting:\n  policy: fallback\n  fallback_profile: remote\n";
        let generate = format!("{HYBRID_GENERATE_MODEL}{routing}");
        let encode = format!("{HYBRID_ENCODE_MODEL}{routing}");
        for (transport_fence, verified_worker) in [(false, true), (true, false), (true, true)] {
            let gateway = TestGateway::new(&[&generate, &encode]).await;
            let loaded = ["acme/chat", "acme/hybrid-encode"];
            if verified_worker {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &loaded)
                    .await;
            } else {
                gateway.add_worker("local-1", LOCAL_LANE, &loaded).await;
            }
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            if !transport_fence {
                gateway.dispatcher.withdraw_execution_authority();
            }
            let served = transport_fence && verified_worker;
            for (surface, stream) in FORBID_SURFACES {
                let model = if matches!(surface, "encode" | "embeddings") {
                    "acme/hybrid-encode"
                } else {
                    "acme/chat"
                };
                let case = format!("{surface}/{stream}/{transport_fence}/{verified_worker}");
                let response = forbidden_surface(&gateway, surface, model, stream).await;
                assert!(!response.headers().contains_key("x-sie-fallback-reason"));
                if served {
                    assert_eq!(response.status(), StatusCode::OK, "{case}");
                    assert_eq!(stamped(&response), (Some("local"), None), "{case}");
                } else {
                    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{case}");
                    assert_eq!(response.headers()["retry-after"], "5", "{case}");
                }
                let _ = axum::body::to_bytes(response.into_body(), 16384)
                    .await
                    .unwrap();
            }
            let dispatched = gateway.dispatcher.dispatched();
            if served {
                assert_eq!(dispatched.len(), FORBID_SURFACES.len());
                assert!(dispatched.iter().all(|work| work.bundle == LOCAL_LANE.2));
                assert_eq!(
                    gateway.dispatcher.execution_authority(),
                    vec![true; FORBID_SURFACES.len()]
                );
            } else {
                assert!(dispatched.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn failed_bridge_preserves_original_refusal_body_and_retry_without_echoing_remote_errors()
    {
        let attempt = FallbackAttempt::default();
        let mut original =
            (StatusCode::SERVICE_UNAVAILABLE, "original local refusal").into_response();
        original
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("60"));
        assert!(attempt.begin(original, FallbackTrigger::Provisioning));
        assert!(!attempt.begin(
            StatusCode::SERVICE_UNAVAILABLE.into_response(),
            FallbackTrigger::Unhealthy
        ));
        let remote = (StatusCode::BAD_GATEWAY, "private upstream failure").into_response();
        let restored = attempt.finish(remote);
        assert_eq!(restored.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(restored.headers()["retry-after"], "60");
        assert_eq!(restored.headers()["x-sie-served-by"], "local");
        assert_eq!(restored.headers()["x-sie-fallback-reason"], "provisioning");
        assert_eq!(
            restored.headers()["x-sie-fallback-error"],
            "INFERENCE_ERROR"
        );
        assert!(!restored.headers().contains_key("x-sie-upstream"));
        assert_eq!(
            axum::body::to_bytes(restored.into_body(), 1024)
                .await
                .unwrap(),
            "original local refusal"
        );
    }

    #[test]
    fn successful_bridge_keeps_remote_disclosure_and_never_carries_a_failure() {
        let attempt = FallbackAttempt::default();
        assert!(attempt.begin(
            StatusCode::SERVICE_UNAVAILABLE.into_response(),
            FallbackTrigger::ModelLoading
        ));
        let mut remote = StatusCode::OK.into_response();
        remote
            .headers_mut()
            .insert(SERVED_BY_HEADER, HeaderValue::from_static("remote"));
        remote
            .headers_mut()
            .insert(UPSTREAM_HEADER, HeaderValue::from_static("team-sie"));
        let response = attempt.finish(remote);
        assert_eq!(response.headers()[SERVED_BY_HEADER], "remote");
        assert_eq!(response.headers()[UPSTREAM_HEADER], "team-sie");
        assert_eq!(response.headers()["x-sie-fallback-reason"], "model_loading");
        assert!(!response.headers().contains_key("x-sie-fallback-error"));
    }

    #[test]
    fn the_fallback_error_is_the_code_the_single_server_derives_from_the_failed_attempt() {
        for code in SIE_ERROR_CODES {
            for spelling in [code.to_string(), code.to_lowercase()] {
                for status in [StatusCode::BAD_REQUEST, StatusCode::SERVICE_UNAVAILABLE] {
                    assert_eq!(fallback_error(status, Some(&spelling)), code, "{spelling}");
                }
            }
        }
        for (status, code, expected) in [
            (503, Some("server_overloaded"), "QUEUE_FULL"),
            (400, Some("invalid_request"), "INVALID_INPUT"),
            (503, Some("SERVER_OVERLOADED"), "INFERENCE_ERROR"),
            (503, Some("PROVISIONING"), "INFERENCE_ERROR"),
            (504, Some("GATEWAY_TIMEOUT"), "INFERENCE_ERROR"),
            (504, Some("first_chunk_timeout"), "INFERENCE_ERROR"),
            (503, Some("transport_failure"), "INFERENCE_ERROR"),
            (413, Some("PAYLOAD_TOO_LARGE"), "INVALID_INPUT"),
            (429, Some("rate_limit_exceeded"), "INVALID_INPUT"),
            (400, Some("context_exceeded"), "INVALID_INPUT"),
            (502, None, "INFERENCE_ERROR"),
            (500, None, "INFERENCE_ERROR"),
            (499, None, "INVALID_INPUT"),
            (404, None, "INVALID_INPUT"),
        ] {
            assert_eq!(
                fallback_error(StatusCode::from_u16(status).unwrap(), code),
                expected,
                "{status} {code:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_failed_attempt_names_its_code_in_its_header_or_body_and_a_deadline_wins() {
        let json = |status: StatusCode, body: serde_json::Value| {
            (status, axum::Json(body)).into_response()
        };
        let with_header = |mut response: Response, code: &'static str| {
            response
                .headers_mut()
                .insert("x-sie-error-code", HeaderValue::from_static(code));
            response
        };
        let unanswered = |mut response: Response| {
            response.extensions_mut().insert(UnansweredBeforeDeadline);
            response
        };
        let pending = Body::from_stream(futures_util::stream::pending::<
            Result<bytes::Bytes, std::io::Error>,
        >());
        let cases = [
            (
                with_header(
                    json(
                        StatusCode::SERVICE_UNAVAILABLE,
                        json!({"error": {"code": "RESOURCE_EXHAUSTED"}}),
                    ),
                    "RESOURCE_EXHAUSTED",
                ),
                "RESOURCE_EXHAUSTED",
            ),
            (
                with_header(
                    json(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        json!({"error": {"code": "invalid_request"}}),
                    ),
                    "PAYLOAD_TOO_LARGE",
                ),
                "INVALID_INPUT",
            ),
            (
                json(
                    StatusCode::BAD_REQUEST,
                    json!({"error": {"code": "INPUT_TOO_LONG", "type": "context_length_exceeded"}}),
                ),
                "INPUT_TOO_LONG",
            ),
            (
                json(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"detail": {"code": "MODEL_LOADING", "message": "loading"}}),
                ),
                "MODEL_LOADING",
            ),
            (
                json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    json!({"error": "all_items_failed", "details": [{"code": "INTERNAL_ERROR"}]}),
                ),
                "INFERENCE_ERROR",
            ),
            (
                (StatusCode::BAD_REQUEST, "not json").into_response(),
                "INVALID_INPUT",
            ),
            (
                (StatusCode::SERVICE_UNAVAILABLE, pending).into_response(),
                "INFERENCE_ERROR",
            ),
            (
                unanswered(with_header(
                    json(
                        StatusCode::GATEWAY_TIMEOUT,
                        json!({"detail": {"code": "GATEWAY_TIMEOUT"}}),
                    ),
                    "GATEWAY_TIMEOUT",
                )),
                "QUEUE_FULL",
            ),
            (
                unanswered(json(
                    StatusCode::SERVICE_UNAVAILABLE,
                    json!({"error": {"code": "RESOURCE_EXHAUSTED"}}),
                )),
                "QUEUE_FULL",
            ),
        ];
        for (index, (failed, expected)) in cases.into_iter().enumerate() {
            let attempt = FallbackAttempt::default();
            let mut original =
                (StatusCode::SERVICE_UNAVAILABLE, "original local refusal").into_response();
            original
                .headers_mut()
                .insert("retry-after", HeaderValue::from_static("60"));
            assert!(attempt.begin(original, FallbackTrigger::ModelLoading));
            let restored = attempt.finish(failed);
            assert_eq!(restored.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(restored.headers()["retry-after"], "60");
            assert_eq!(restored.headers()[SERVED_BY_HEADER], "local");
            assert_eq!(restored.headers()["x-sie-fallback-reason"], "model_loading");
            assert_eq!(
                restored.headers()["x-sie-fallback-error"],
                expected,
                "case {index}"
            );
            assert_eq!(
                axum::body::to_bytes(restored.into_body(), 1024)
                    .await
                    .unwrap(),
                "original local refusal"
            );
        }
    }

    #[test]
    fn remote_control_rejects_ambiguous_and_unrecognized_header_bytes() {
        let mut headers = HeaderMap::new();
        assert_eq!(remote_forbidden(&headers), Ok(false));
        headers.insert(REMOTE_HEADER, HeaderValue::from_static("forbid"));
        assert_eq!(remote_forbidden(&headers), Ok(true));
        headers.append(REMOTE_HEADER, HeaderValue::from_static("forbid"));
        assert!(remote_forbidden(&headers).is_err());
        for value in [
            b"FORBID".as_slice(),
            b"allow",
            b"forbid, forbid",
            b"",
            b"\xff",
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(REMOTE_HEADER, HeaderValue::from_bytes(value).unwrap());
            assert!(remote_forbidden(&headers).is_err());
        }
    }

    use crate::handlers::proxy::{
        proxy_chat, proxy_completions, proxy_openai_embeddings, proxy_request, proxy_responses,
    };
    use crate::handlers::test_support::{
        Dispatched, TestGateway, HYBRID_ENCODE_MODEL, HYBRID_EXTRACT_MODEL, HYBRID_GENERATE_MODEL,
        LOCAL_ENCODE_MODEL, LOCAL_LANE, REMOTE_ENCODE_MODEL, REMOTE_LANE,
    };

    fn json_request(uri: &str, body: serde_json::Value) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn extraction_request(msgpack: bool, params: serde_json::Value) -> Request {
        let body =
            json!({"items":[{"audio":{"data":"UklGRnRlc3Q=", "format":"wav"}}], "params":params});
        if !msgpack {
            return json_request("/v1/extract/acme/extract", body);
        }
        let mut value: rmpv::Value =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&body).unwrap()).unwrap();
        let rmpv::Value::Map(fields) = &mut value else {
            unreachable!()
        };
        let rmpv::Value::Array(items) = &mut fields
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some("items"))
            .unwrap()
            .1
        else {
            unreachable!()
        };
        let rmpv::Value::Map(fields) = &mut items[0] else {
            unreachable!()
        };
        let rmpv::Value::Map(audio) = &mut fields
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some("audio"))
            .unwrap()
            .1
        else {
            unreachable!()
        };
        audio
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some("data"))
            .unwrap()
            .1 = rmpv::Value::Binary(b"RIFFtest".to_vec());
        Request::builder()
            .method(Method::POST)
            .uri("/v1/extract/acme/extract")
            .header("content-type", "application/msgpack")
            .body(Body::from(rmp_serde::to_vec_named(&value).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn extraction_cluster_fallback_warms_before_remote_and_restores_failed_attempts() {
        let config = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for msgpack in [false, true] {
            for local in ["cold", "loading", "loaded"] {
                for fail in [false, true] {
                    let gateway = TestGateway::new(&[&config]).await;
                    gateway
                        .add_verified_worker("remote-1", REMOTE_LANE, &[])
                        .await;
                    if local != "cold" {
                        gateway
                            .add_verified_worker(
                                "local-1",
                                LOCAL_LANE,
                                if local == "loaded" {
                                    &["acme/extract"]
                                } else {
                                    &[]
                                },
                            )
                            .await;
                    }
                    if fail {
                        gateway.dispatcher.refuse_work();
                    }
                    let response = proxy_request(
                        State(Arc::clone(&gateway.state)),
                        extraction_request(msgpack, json!({})),
                        "extract",
                    )
                    .await;
                    if fail && local != "loaded" {
                        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                        assert_eq!(
                            response.headers()["retry-after"],
                            if local == "cold" { "60" } else { "5" }
                        );
                        assert_eq!(
                            response.headers()["x-sie-fallback-error"],
                            "INFERENCE_ERROR"
                        );
                        assert_eq!(stamped(&response), (Some("local"), None));
                    } else if !fail {
                        assert_eq!(
                            response.status(),
                            StatusCode::OK,
                            "{local}, msgpack={msgpack}"
                        );
                        assert_eq!(
                            stamped(&response),
                            if local == "loaded" {
                                (Some("local"), None)
                            } else {
                                (Some("remote"), Some("team-sie"))
                            }
                        );
                        let body = axum::body::to_bytes(response.into_body(), 8192)
                            .await
                            .unwrap();
                        if !msgpack {
                            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                            assert_eq!(body["model"], "acme/extract");
                            assert_eq!(body["items"][0]["data"]["text"], "hello world");
                        }
                    }
                    let mut expected = Vec::new();
                    if local == "loading" {
                        expected.push(dispatched("load", LOCAL_LANE, "acme/extract"));
                    }
                    expected.push(if local == "loaded" {
                        dispatched("extract", LOCAL_LANE, "acme/extract")
                    } else {
                        dispatched("extract", REMOTE_LANE, "acme/extract:remote")
                    });
                    assert_eq!(gateway.dispatcher.dispatched(), expected);
                }
            }
        }
    }

    async fn streaming_surface(
        gateway: &TestGateway,
        surface: &str,
        options: serde_json::Value,
    ) -> Response {
        let state = State(Arc::clone(&gateway.state));
        match surface {
            "native" => proxy_request(state, json_request("/v1/generate/acme/chat", json!({"prompt":"hello","max_new_tokens":4,"stream":true,"options":options})), "generate").await,
            "chat" => proxy_chat(state, json_request("/v1/chat/completions", json!({"model":"acme/chat","messages":[{"role":"user","content":"hello"}],"stream":true}))).await,
            "completions" => proxy_completions(state, json_request("/v1/completions", json!({"model":"acme/chat","prompt":"hello","max_tokens":4,"stream":true}))).await,
            _ => unreachable!(),
        }
    }

    fn assert_restored_cold_refusal(response: &Response, fallback_error: &str, case: &str) {
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{case}");
        assert_eq!(response.headers()["retry-after"], "60", "{case}");
        assert_eq!(stamped(response), (Some("local"), None), "{case}");
        assert_eq!(
            response.headers()["x-sie-fallback-reason"],
            "provisioning",
            "{case}"
        );
        assert_eq!(
            response.headers()["x-sie-fallback-error"],
            fallback_error,
            "{case}"
        );
    }

    #[tokio::test]
    async fn a_failed_bridge_reports_the_code_its_remote_attempt_answered_with() {
        let generation = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for (code, retry_after_s, expected) in [
            ("RESOURCE_EXHAUSTED", Some(7), "RESOURCE_EXHAUSTED"),
            ("MODEL_LOADING", Some(7), "MODEL_LOADING"),
            ("INPUT_TOO_LONG", None, "INPUT_TOO_LONG"),
            ("inference_error", None, "INFERENCE_ERROR"),
        ] {
            for (surface, stream) in [
                ("native", false),
                ("chat", false),
                ("completions", false),
                ("responses", false),
                ("native", true),
                ("chat", true),
                ("completions", true),
            ] {
                let gateway = TestGateway::new(&[&generation]).await;
                gateway
                    .add_verified_worker("remote-1", REMOTE_LANE, &[])
                    .await;
                gateway.dispatcher.refuse_remote_work(code, retry_after_s);
                let response = if stream {
                    streaming_surface(&gateway, surface, json!({})).await
                } else {
                    buffered_surface(&gateway, surface, json!({})).await
                };
                assert_restored_cold_refusal(
                    &response,
                    expected,
                    &format!("{code} {surface} stream={stream}"),
                );
                assert_eq!(
                    gateway.dispatcher.dispatched(),
                    vec![dispatched("generate", REMOTE_LANE, "acme/chat:remote")]
                );
            }
        }
        let extraction = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for (code, retry_after_s, expected) in [
            ("QUEUE_FULL", Some(9), "QUEUE_FULL"),
            ("MODEL_LOADING", Some(9), "MODEL_LOADING"),
            ("RESOURCE_EXHAUSTED", Some(9), "RESOURCE_EXHAUSTED"),
            ("INPUT_TOO_LONG", None, "INPUT_TOO_LONG"),
            ("INVALID_INPUT", None, "INVALID_INPUT"),
            ("INFERENCE_ERROR", None, "INFERENCE_ERROR"),
        ] {
            for msgpack in [false, true] {
                let gateway = TestGateway::new(&[&extraction]).await;
                gateway
                    .add_verified_worker("remote-1", REMOTE_LANE, &[])
                    .await;
                gateway.dispatcher.refuse_remote_work(code, retry_after_s);
                let response = proxy_request(
                    State(Arc::clone(&gateway.state)),
                    extraction_request(msgpack, json!({})),
                    "extract",
                )
                .await;
                assert_restored_cold_refusal(
                    &response,
                    expected,
                    &format!("{code} extract msgpack={msgpack}"),
                );
            }
        }
    }

    #[tokio::test]
    async fn a_bridge_whose_remote_attempt_never_answers_reports_queue_full() {
        let generation = format!(
            "{HYBRID_GENERATE_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let deadlines = json!({"first_chunk_timeout_s": 0.2, "overall_timeout_s": 0.4});
        for (first_chunk_deadline, output_arrived, stream, expected) in [
            (false, false, false, "QUEUE_FULL"),
            (true, false, false, "QUEUE_FULL"),
            (false, false, true, "QUEUE_FULL"),
            (true, false, true, "QUEUE_FULL"),
            (false, true, false, "INFERENCE_ERROR"),
        ] {
            let gateway = TestGateway::new(&[&generation]).await;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            gateway.dispatcher.withhold_remote_answers(output_arrived);
            if first_chunk_deadline {
                gateway.dispatcher.enable_first_chunk_deadline();
            }
            let request = async {
                if stream {
                    streaming_surface(&gateway, "native", deadlines.clone()).await
                } else {
                    buffered_surface(&gateway, "native", deadlines.clone()).await
                }
            };
            let response = tokio::time::timeout(Duration::from_secs(5), request)
                .await
                .expect("a generation deadline ends the remote attempt");
            assert_restored_cold_refusal(
                &response,
                expected,
                &format!(
                    "first_chunk_deadline={first_chunk_deadline} output_arrived={output_arrived} stream={stream}"
                ),
            );
        }
        let extraction = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for msgpack in [false, true] {
            let mut gateway = TestGateway::new(&[&extraction]).await;
            gateway.set_request_timeout(0.2);
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            gateway.dispatcher.withhold_remote_answers(false);
            let request = proxy_request(
                State(Arc::clone(&gateway.state)),
                extraction_request(msgpack, json!({})),
                "extract",
            );
            let response = tokio::time::timeout(Duration::from_secs(5), request)
                .await
                .expect("the queued-result deadline ends the remote attempt");
            assert_restored_cold_refusal(
                &response,
                "QUEUE_FULL",
                &format!("extract msgpack={msgpack}"),
            );
        }
    }

    #[tokio::test]
    async fn extraction_cluster_fallback_rejects_invalid_requests_and_honors_selectors() {
        let config = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        for msgpack in [false, true] {
            let mut forbid = extraction_request(msgpack, json!({}));
            forbid
                .headers_mut()
                .insert("x-sie-remote", HeaderValue::from_static("forbid"));
            for request in [
                forbid,
                extraction_request(msgpack, json!({"options":{"profile":"default"}})),
            ] {
                let response =
                    proxy_request(State(Arc::clone(&gateway.state)), request, "extract").await;
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert!(!response.headers().contains_key("x-sie-fallback-reason"));
            }
            for params in [
                json!({"labels":[3]}),
                json!({"instruction":4}),
                json!({"options":false}),
                json!({"output_schema":[]}),
            ] {
                let response = proxy_request(
                    State(Arc::clone(&gateway.state)),
                    extraction_request(msgpack, params),
                    "extract",
                )
                .await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
        }
        for body in [
            json!({"items":[]}),
            json!({"items":[{"text":"unsupported"}]}),
            json!({"items":[{"audio":{"data":4}}]}),
            json!({"items":[{"audio":{"data":"UklGRg==", "sample_rate":"invalid"}}]}),
        ] {
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                json_request("/v1/extract/acme/extract", body),
                "extract",
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn extraction_cluster_fallback_bounds_invalid_inputs_before_demand_or_work() {
        let config = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for loading in [false, true] {
            let gateway = TestGateway::new(&[&config]).await;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            if loading {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &[])
                    .await;
            }
            for msgpack in [false, true] {
                let mut large_schema = json!({});
                for _ in 0..130 {
                    large_schema = json!({"nested":large_schema});
                }
                for params in [
                    json!({"output_schema":{"values":vec![0;100_001]}}),
                    json!({"output_schema":large_schema}),
                ] {
                    let response = proxy_request(
                        State(Arc::clone(&gateway.state)),
                        extraction_request(msgpack, params),
                        "extract",
                    )
                    .await;
                    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
                }
            }
            for sample_rate in [0, -1] {
                let body =
                    json!({"items":[{"audio":{"data":"UklGRg==", "sample_rate":sample_rate}}]});
                let response = proxy_request(
                    State(Arc::clone(&gateway.state)),
                    json_request("/v1/extract/acme/extract", body),
                    "extract",
                )
                .await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
            let body = json!({"items":[{"audio":{"data":"UklGRg=="}, "metadata":{"large":"x".repeat(2 * 1024 * 1024)}}]});
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                json_request("/v1/extract/acme/extract", body),
                "extract",
            )
            .await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            assert!(gateway.dispatcher.dispatched().is_empty());
            assert!(gateway.state.demand_tracker.active_lanes().is_empty());
        }
    }

    async fn msgpack_extraction_with_metadata(metadata: rmpv::Value) -> Request {
        let request = extraction_request(true, json!({}));
        let (parts, body) = request.into_parts();
        let bytes = axum::body::to_bytes(body, 8192).await.unwrap();
        let mut root: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let rmpv::Value::Map(fields) = &mut root else {
            unreachable!()
        };
        let rmpv::Value::Array(items) = &mut fields
            .iter_mut()
            .find(|(key, _)| key.as_str() == Some("items"))
            .unwrap()
            .1
        else {
            unreachable!()
        };
        let rmpv::Value::Map(item) = &mut items[0] else {
            unreachable!()
        };
        item.push((rmpv::Value::from("metadata"), metadata));
        Request::from_parts(parts, Body::from(rmp_serde::to_vec_named(&root).unwrap()))
    }

    #[tokio::test]
    async fn extraction_cluster_fallback_uses_worker_decoded_metadata_contract() {
        let config = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for loading in [false, true] {
            let mut gateway = TestGateway::new(&[&config]).await;
            let state = Arc::get_mut(&mut gateway.state).unwrap();
            Arc::get_mut(&mut state.config).unwrap().max_item_text_bytes = 16;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            if loading {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &[])
                    .await;
            }
            for metadata in [
                rmpv::Value::Map(vec![(1.into(), "invalid-key".into())]),
                rmpv::Value::Map(vec![(
                    "x".into(),
                    rmpv::Value::Array(vec![rmpv::Value::F32(1.5); 2]),
                )]),
            ] {
                let response = proxy_request(
                    State(Arc::clone(&gateway.state)),
                    msgpack_extraction_with_metadata(metadata).await,
                    "extract",
                )
                .await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            }
            assert!(gateway.dispatcher.dispatched().is_empty());
            assert!(gateway.state.demand_tracker.active_lanes().is_empty());
            let valid = rmpv::Value::Map(vec![("x".into(), rmpv::Value::F32(1.5))]);
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                msgpack_extraction_with_metadata(valid).await,
                "extract",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
        }
        let gateway = TestGateway::new(&[&config]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let opaque = rmpv::Value::Map(vec![("x".into(), rmpv::Value::Ext(1, vec![1]))]);
        let response = proxy_request(
            State(Arc::clone(&gateway.state)),
            msgpack_extraction_with_metadata(opaque).await,
            "extract",
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!response.headers().contains_key("x-sie-fallback-reason"));
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn every_bridged_work_item_carries_the_trigger_of_the_refusal_it_stands_in_for() {
        for trigger in [
            FallbackTrigger::Provisioning,
            FallbackTrigger::ModelLoading,
            FallbackTrigger::Saturated,
            FallbackTrigger::Unhealthy,
        ] {
            let routing = format!(
                "\nrouting:\n  policy: fallback\n  fallback_profile: remote\n  triggers: [{}]\n",
                trigger.as_str()
            );
            let generate = format!("{HYBRID_GENERATE_MODEL}{routing}");
            let extract = format!("{HYBRID_EXTRACT_MODEL}{routing}");
            let gateway = TestGateway::new(&[&generate, &extract]).await;
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            let loaded = ["acme/chat", "acme/extract"];
            match trigger {
                FallbackTrigger::Provisioning => {}
                FallbackTrigger::ModelLoading => {
                    gateway
                        .add_verified_worker("local-1", LOCAL_LANE, &[])
                        .await;
                }
                FallbackTrigger::Saturated => {
                    gateway
                        .add_saturated_worker("local-1", LOCAL_LANE, &loaded)
                        .await;
                }
                FallbackTrigger::Unhealthy => {
                    gateway
                        .add_verified_worker("local-1", LOCAL_LANE, &loaded)
                        .await;
                    gateway
                        .state
                        .registry
                        .mark_unhealthy("http://local-1:8080")
                        .await;
                }
            }
            let mut responses = Vec::new();
            for surface in ["native", "chat", "completions", "responses"] {
                responses.push(buffered_surface(&gateway, surface, json!({})).await);
            }
            for (uri, body) in [
                (
                    "/v1/generate/acme/chat",
                    json!({"prompt":"hello", "max_new_tokens":4, "stream":true}),
                ),
                (
                    "/v1/chat/completions",
                    json!({"model":"acme/chat", "messages":[{"role":"user", "content":"hello"}], "stream":true}),
                ),
                (
                    "/v1/completions",
                    json!({"model":"acme/chat", "prompt":"hello", "max_tokens":4, "stream":true}),
                ),
            ] {
                let state = State(Arc::clone(&gateway.state));
                let request = json_request(uri, body);
                responses.push(match uri {
                    "/v1/generate/acme/chat" => proxy_request(state, request, "generate").await,
                    "/v1/chat/completions" => proxy_chat(state, request).await,
                    _ => proxy_completions(state, request).await,
                });
            }
            responses.push(
                proxy_request(
                    State(Arc::clone(&gateway.state)),
                    extraction_request(false, json!({})),
                    "extract",
                )
                .await,
            );
            let bridged = responses.len();
            for response in responses {
                assert_eq!(response.status(), StatusCode::OK, "{trigger:?}");
                assert_eq!(
                    response.headers()["x-sie-fallback-reason"],
                    trigger.as_str()
                );
                let _ = axum::body::to_bytes(response.into_body(), 16384)
                    .await
                    .unwrap();
            }
            assert!(gateway
                .dispatcher
                .dispatched()
                .iter()
                .filter(|work| work.endpoint != "load")
                .all(|work| work.bundle == REMOTE_LANE.2));
            assert_eq!(
                gateway.dispatcher.fallback_reasons(),
                vec![Some(trigger); bridged],
                "{trigger:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_remote_attempt_refused_by_its_upstream_restores_the_local_refusal_at_once() {
        let config = format!(
            "{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n"
        );
        for (refusal, remote_code, retry_after_s) in [
            ("capped", "QUEUE_FULL", 1),
            ("breaker_open", "QUEUE_FULL", 60),
            ("not_ready", "QUEUE_FULL", 5),
            ("not_ready", "MODEL_LOADING", 5),
        ] {
            for local in ["cold", "loading"] {
                let gateway = TestGateway::new(&[&config]).await;
                gateway
                    .add_verified_worker("remote-1", REMOTE_LANE, &[])
                    .await;
                if local == "loading" {
                    gateway
                        .add_verified_worker("local-1", LOCAL_LANE, &[])
                        .await;
                }
                gateway
                    .dispatcher
                    .answer_only_bridged_remote_work(remote_code, retry_after_s);
                let response = tokio::time::timeout(
                    Duration::from_secs(5),
                    proxy_request(
                        State(Arc::clone(&gateway.state)),
                        extraction_request(false, json!({})),
                        "extract",
                    ),
                )
                .await
                .unwrap_or_else(|_| {
                    panic!("{refusal}/{remote_code}/{local}: the local refusal waited")
                });
                let (trigger, retry_after, code) = if local == "cold" {
                    ("provisioning", "60", "PROVISIONING")
                } else {
                    ("model_loading", "5", "MODEL_LOADING")
                };
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert_eq!(response.headers()["retry-after"], retry_after);
                assert_eq!(response.headers()["x-sie-fallback-reason"], trigger);
                assert_eq!(response.headers()["x-sie-fallback-error"], remote_code);
                assert_eq!(stamped(&response), (Some("local"), None));
                let body: serde_json::Value = serde_json::from_slice(
                    &axum::body::to_bytes(response.into_body(), 8192)
                        .await
                        .unwrap(),
                )
                .unwrap();
                let detail = body.get("detail").unwrap_or(&body["error"]);
                assert_eq!(detail["code"], code, "{refusal}/{local}");
                assert_eq!(
                    gateway.dispatcher.dispatched().last().unwrap(),
                    &dispatched("extract", REMOTE_LANE, "acme/extract:remote")
                );
            }
        }
    }

    fn stamped(response: &Response) -> (Option<&str>, Option<&str>) {
        let header = |name: &HeaderName| {
            response
                .headers()
                .get(name)
                .map(|value| value.to_str().unwrap())
        };
        (header(&SERVED_BY_HEADER), header(&UPSTREAM_HEADER))
    }

    async fn gateway() -> TestGateway {
        let gateway = TestGateway::new(&[
            LOCAL_ENCODE_MODEL,
            REMOTE_ENCODE_MODEL,
            HYBRID_GENERATE_MODEL,
        ])
        .await;
        gateway.add_worker("local-1", LOCAL_LANE, &[]).await;
        gateway.add_worker("remote-1", REMOTE_LANE, &[]).await;
        gateway
    }

    fn dispatched(endpoint: &str, lane: (&str, &str, &str), model: &str) -> Dispatched {
        Dispatched {
            endpoint: endpoint.to_string(),
            pool: lane.0.to_string(),
            machine_profile: lane.1.to_string(),
            bundle: lane.2.to_string(),
            model: model.to_string(),
        }
    }

    #[tokio::test]
    async fn a_remotely_served_response_names_its_upstream_and_a_local_one_does_not() {
        let gateway = gateway().await;
        let body = json!({"items": [{"text": "hello"}]});

        let remote = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request("/v1/encode/acme/remote", body.clone()),
            "encode",
        )
        .await;
        let local = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request("/v1/encode/acme/local", body),
            "encode",
        )
        .await;

        assert_eq!(remote.status(), StatusCode::OK);
        assert_eq!(stamped(&remote), (Some("remote"), Some("team-sie")));
        assert_eq!(local.status(), StatusCode::OK);
        assert_eq!(stamped(&local), (Some("local"), None));
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![
                dispatched("encode", REMOTE_LANE, "acme/remote"),
                dispatched("encode", LOCAL_LANE, "acme/local"),
            ]
        );
    }

    #[tokio::test]
    async fn a_server_side_refusal_names_the_side_that_refused_and_a_client_error_names_none() {
        let gateway = TestGateway::new(&[LOCAL_ENCODE_MODEL]).await;
        let refused = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/encode/acme/local",
                json!({"items": [{"text": "hello"}]}),
            ),
            "encode",
        )
        .await;
        gateway.add_worker("local-1", LOCAL_LANE, &[]).await;
        let invalid = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request("/v1/encode/acme/local", json!({"items": []})),
            "encode",
        )
        .await;
        let unknown = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/encode/acme/absent",
                json!({"items": [{"text": "hello"}]}),
            ),
            "encode",
        )
        .await;

        assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(stamped(&refused), (Some("local"), None));
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        assert_eq!(stamped(&invalid), (None, None));
        assert_eq!(unknown.status(), StatusCode::NOT_FOUND);
        assert_eq!(stamped(&unknown), (None, None));
    }

    /// JetStream-gated: the gateway knows a remote-only model's upstream by name
    /// only. It publishes the request without that name and opens no upstream
    /// connection; a test consumer stands in for the remote worker.
    #[tokio::test]
    async fn the_gateway_publishes_a_remote_only_request_and_never_calls_its_upstream() {
        use futures_util::StreamExt;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        use crate::queue::dispatch::WorkResult;
        use crate::queue::publisher::{WorkPublisher, WorkStreamConfig};

        let Ok(url) = std::env::var("NATS_URL") else {
            assert_ne!(
                std::env::var("SIE_RUN_NATS_PUBLISHER_TEST").as_deref(),
                Ok("1"),
                "mandatory publisher tests require NATS_URL"
            );
            return;
        };
        let pool = format!("egress{}", uuid::Uuid::now_v7().simple());
        let stream_name = format!("WORK_POOL_{pool}");
        let client = async_nats::connect(url)
            .await
            .expect("test NATS connection");
        let jetstream = async_nats::jetstream::new(client.clone());
        let stream = jetstream
            .create_stream(async_nats::jetstream::stream::Config {
                name: stream_name.clone(),
                subjects: vec![format!("sie.work.{pool}.*.*.*")],
                retention: async_nats::jetstream::stream::RetentionPolicy::WorkQueue,
                storage: async_nats::jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(300),
                discard: async_nats::jetstream::stream::DiscardPolicy::New,
                ..Default::default()
            })
            .await
            .unwrap();
        stream
            .create_consumer(async_nats::jetstream::consumer::pull::Config {
                durable_name: Some("remote-1".into()),
                filter_subject: format!("sie.work.{pool}.cpu.remote.*"),
                ..Default::default()
            })
            .await
            .unwrap();

        let upstream = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let upstream_task = tokio::spawn({
            let connections = Arc::clone(&connections);
            async move {
                while let Ok((mut socket, _)) = upstream.accept().await {
                    connections.fetch_add(1, Ordering::SeqCst);
                    let _ = socket.read(&mut [0u8; 1024]).await;
                    let _ = socket
                        .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")
                        .await;
                }
            }
        });

        let model = format!("{REMOTE_ENCODE_MODEL}pool: {pool}\n");
        let mut gateway = TestGateway::with_remote_queue_pool(&[&model], &pool).await;
        gateway
            .add_worker(
                "remote-1",
                (pool.as_str(), REMOTE_LANE.1, REMOTE_LANE.2),
                &[],
            )
            .await;
        let publisher = Arc::new(WorkPublisher::new(
            jetstream.clone(),
            "egress-gateway".into(),
            Arc::new(crate::queue::payload_store::DisabledPayloadStore),
            Duration::from_secs(10),
            1024,
            WorkStreamConfig {
                max_age: Duration::from_secs(300),
                storage: async_nats::jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));
        publisher.start_inbox_subscription(&client).await.unwrap();
        Arc::get_mut(&mut gateway.state).unwrap().work_publisher = Some(publisher);

        let mut work = client
            .subscribe(format!("sie.work.{pool}.cpu.remote.>"))
            .await
            .unwrap();
        client.flush().await.unwrap();
        let consumer = tokio::spawn({
            let client = client.clone();
            async move {
                let message = work.next().await.expect("a published work item");
                let item: rmpv::Value = rmp_serde::from_slice(&message.payload).unwrap();
                let field = |name: &str| {
                    item.as_map()
                        .and_then(|fields| {
                            fields.iter().find(|(key, _)| key.as_str() == Some(name))
                        })
                        .and_then(|(_, value)| value.as_str())
                        .unwrap()
                        .to_string()
                };
                let mut call = tokio::net::TcpStream::connect(upstream_addr).await.unwrap();
                call.write_all(b"POST /v1/encode/acme/remote HTTP/1.1\r\n\r\n")
                    .await
                    .unwrap();
                let _ = call.read(&mut [0u8; 64]).await;
                let mut result: WorkResult = serde_json::from_value(json!({
                    "work_item_id": field("work_item_id"),
                    "request_id": field("request_id"),
                    "item_index": 0,
                    "success": true,
                }))
                .unwrap();
                result.result_msgpack =
                    rmp_serde::to_vec_named(&json!({"dense": [0.5, 0.25]})).unwrap();
                client
                    .publish(
                        field("reply_subject"),
                        rmp_serde::to_vec_named(&result).unwrap().into(),
                    )
                    .await
                    .unwrap();
                client.flush().await.unwrap();
                (field("model_id"), message.payload)
            }
        });

        let response = tokio::time::timeout(
            Duration::from_secs(10),
            proxy_request(
                State(Arc::clone(&gateway.state)),
                json_request(
                    "/v1/encode/acme/remote",
                    json!({"items": [{"text": "hello"}]}),
                ),
                "encode",
            ),
        )
        .await;
        if !matches!(&response, Ok(served) if served.status().is_success()) {
            consumer.abort();
        }
        let consumer = tokio::time::timeout(Duration::from_secs(10), consumer).await;
        upstream_task.abort();
        let _ = jetstream.delete_stream(&stream_name).await;

        let response = response.expect("served through the queue");
        assert_eq!(response.status(), StatusCode::OK);
        let (model_id, published) = consumer
            .expect("the test consumer finished")
            .expect("the test consumer answered");
        assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
        assert_eq!(model_id, "acme/remote");
        assert_eq!(
            connections.load(Ordering::SeqCst),
            1,
            "the test consumer's call is the only upstream connection"
        );
        assert!(
            !String::from_utf8_lossy(&published).contains("team-sie"),
            "the work item names the upstream"
        );
    }

    #[tokio::test]
    async fn the_embeddings_route_forwards_the_disclosure_of_the_encode_it_wraps() {
        let gateway = gateway().await;

        let response = proxy_openai_embeddings(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/embeddings",
                json!({"model": "acme/remote", "input": "hello"}),
            ),
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
    }

    #[tokio::test]
    async fn a_remote_only_request_its_upstream_cannot_serve_now_answers_503_with_the_wait() {
        for (code, retry_after_s, retry_after) in [
            ("QUEUE_FULL", Some(9), "9"),
            ("MODEL_LOADING", Some(7), "7"),
            ("QUEUE_FULL", None, "5"),
            ("MODEL_LOADING", None, "5"),
        ] {
            let gateway = TestGateway::new(&[REMOTE_ENCODE_MODEL]).await;
            gateway.add_worker("remote-1", REMOTE_LANE, &[]).await;
            gateway.dispatcher.refuse_remote_work(code, retry_after_s);
            let native = proxy_request(
                State(Arc::clone(&gateway.state)),
                json_request(
                    "/v1/encode/acme/remote",
                    json!({"items": [{"text": "hello"}]}),
                ),
                "encode",
            )
            .await;
            let embeddings = proxy_openai_embeddings(
                State(Arc::clone(&gateway.state)),
                json_request(
                    "/v1/embeddings",
                    json!({"model": "acme/remote", "input": "hello"}),
                ),
            )
            .await;
            for response in [&native, &embeddings] {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{code}");
                assert_eq!(response.headers()["retry-after"], retry_after, "{code}");
                assert_eq!(response.headers()["x-sie-error-code"], code);
                assert_eq!(stamped(response), (Some("remote"), Some("team-sie")));
            }
            let body: serde_json::Value = serde_json::from_slice(
                &axum::body::to_bytes(native.into_body(), 8192)
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(body["error"]["code"], code);
            assert_eq!(
                gateway.dispatcher.dispatched(),
                vec![dispatched("encode", REMOTE_LANE, "acme/remote"); 2]
            );
        }
    }

    #[tokio::test]
    async fn generation_routes_name_the_profile_they_dispatch_to() {
        let gateway = gateway().await;
        let messages = json!([{"role": "user", "content": "hello"}]);
        let state = || State(Arc::clone(&gateway.state));

        let chat_remote = proxy_chat(
            state(),
            json_request(
                "/v1/chat/completions",
                json!({"model": "acme/chat:remote", "messages": messages}),
            ),
        )
        .await;
        let chat_remote_stream = proxy_chat(
            state(),
            json_request(
                "/v1/chat/completions",
                json!({"model": "acme/chat:remote", "messages": messages, "stream": true}),
            ),
        )
        .await;
        let chat_local = proxy_chat(
            state(),
            json_request(
                "/v1/chat/completions",
                json!({"model": "acme/chat", "messages": messages}),
            ),
        )
        .await;
        let completion = proxy_completions(
            state(),
            json_request(
                "/v1/completions",
                json!({"model": "acme/chat:remote", "prompt": "hello"}),
            ),
        )
        .await;
        let response = proxy_responses(
            state(),
            json_request(
                "/v1/responses",
                json!({"model": "acme/chat", "input": "hello"}),
            ),
        )
        .await;
        let native = proxy_request(
            state(),
            json_request(
                "/v1/generate/acme__chat:remote",
                json!({"prompt": "hello", "max_new_tokens": 4}),
            ),
            "generate",
        )
        .await;

        for (name, served, expected) in [
            (
                "chat remote",
                &chat_remote,
                (Some("remote"), Some("team-sie")),
            ),
            (
                "chat remote stream",
                &chat_remote_stream,
                (Some("remote"), Some("team-sie")),
            ),
            ("chat local", &chat_local, (Some("local"), None)),
            (
                "completion remote",
                &completion,
                (Some("remote"), Some("team-sie")),
            ),
            ("responses local", &response, (Some("local"), None)),
            (
                "native generate remote",
                &native,
                (Some("remote"), Some("team-sie")),
            ),
        ] {
            assert_eq!(served.status(), StatusCode::OK, "{name}");
            assert_eq!(stamped(served), expected, "{name}");
        }
        let routed: Vec<String> = gateway
            .dispatcher
            .dispatched()
            .into_iter()
            .map(|dispatched| format!("{}:{}", dispatched.bundle, dispatched.model))
            .collect();
        assert_eq!(
            routed,
            vec![
                "remote:acme/chat:remote",
                "remote:acme/chat:remote",
                "default:acme/chat",
                "remote:acme/chat:remote",
                "default:acme/chat",
                "remote:acme/chat:remote",
            ]
        );
    }

    #[test]
    fn only_a_served_response_or_a_server_side_refusal_is_stamped() {
        let disclosure = ServingDisclosure::default();
        *disclosure.0.lock().unwrap() = Some(ServedBy::Remote {
            upstream: Some("team-sie".to_string()),
        });
        for (status, carries_headers) in [
            (StatusCode::OK, true),
            (StatusCode::SERVICE_UNAVAILABLE, true),
            (StatusCode::GATEWAY_TIMEOUT, true),
            (StatusCode::BAD_REQUEST, false),
            (StatusCode::NOT_FOUND, false),
        ] {
            let mut response = StatusCode::OK.into_response();
            disclosure.stamp(status, response.headers_mut());
            let expected = if carries_headers {
                (Some("remote"), Some("team-sie"))
            } else {
                (None, None)
            };
            assert_eq!(stamped(&response), expected, "{status}");
        }
    }

    #[test]
    fn a_local_stamp_replaces_a_remote_one_and_drops_its_upstream() {
        let disclosure = ServingDisclosure::default();
        let mut response = StatusCode::OK.into_response();
        *disclosure.0.lock().unwrap() = Some(ServedBy::Remote {
            upstream: Some("team-sie".to_string()),
        });
        disclosure.stamp(StatusCode::OK, response.headers_mut());
        *disclosure.0.lock().unwrap() = Some(ServedBy::Local);
        disclosure.stamp(StatusCode::OK, response.headers_mut());

        assert_eq!(stamped(&response), (Some("local"), None));
    }

    #[test]
    fn the_headers_are_the_ones_the_shared_wire_fixture_declares() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../wire-fixtures/serving_disclosure.json"
        ))
        .unwrap();
        let served_by = &fixture["response_headers"]["served_by"];
        let upstream = &fixture["response_headers"]["upstream"];

        assert!(served_by["name"]
            .as_str()
            .unwrap()
            .eq_ignore_ascii_case(SERVED_BY_HEADER.as_str()));
        assert!(upstream["name"]
            .as_str()
            .unwrap()
            .eq_ignore_ascii_case(UPSTREAM_HEADER.as_str()));
        assert_eq!(served_by["values"], json!(["local", "remote"]));
    }

    fn remote_routing_vectors() -> serde_json::Value {
        serde_json::from_str(include_str!("../../../wire-fixtures/remote_routing.json")).unwrap()
    }

    /// The vectors of one section of `wire-fixtures/remote_routing.json` that
    /// apply to a gateway.
    fn gateway_vectors(section: &str) -> Vec<serde_json::Value> {
        remote_routing_vectors()[section]
            .as_array()
            .unwrap()
            .iter()
            .filter(|vector| {
                vector.get("topologies").is_none_or(|topologies| {
                    topologies.as_array().unwrap().contains(&json!("gateway"))
                })
            })
            .cloned()
            .collect()
    }

    fn vector_status(value: &serde_json::Value) -> StatusCode {
        StatusCode::from_u16(u16::try_from(value.as_u64().unwrap()).unwrap()).unwrap()
    }

    fn header_values(values: &serde_json::Value) -> Vec<&str> {
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap())
            .collect()
    }

    /// A gateway whose `acme/chat` and `acme/hybrid-encode` local profiles are
    /// in the abstract local `state` of the shared vectors, with a verified
    /// remote worker and `triggers` declared on both models.
    async fn gateway_in_local_state(state: &str, triggers: &serde_json::Value) -> TestGateway {
        let mut routing =
            "\nrouting:\n  policy: fallback\n  fallback_profile: remote\n".to_string();
        if !triggers.is_null() {
            routing.push_str(&format!(
                "  triggers: [{}]\n",
                header_values(triggers).join(", ")
            ));
        }
        let generate = format!("{HYBRID_GENERATE_MODEL}{routing}");
        let encode = format!("{HYBRID_ENCODE_MODEL}{routing}");
        let gateway = TestGateway::new(&[&generate, &encode, REMOTE_ENCODE_MODEL]).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let loaded = ["acme/chat", "acme/hybrid-encode"];
        match state {
            "ready" | "unhealthy" => {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &loaded)
                    .await;
            }
            "loading" => {
                gateway
                    .add_verified_worker("local-1", LOCAL_LANE, &[])
                    .await;
            }
            "saturated" => {
                gateway
                    .add_saturated_worker("local-1", LOCAL_LANE, &loaded)
                    .await;
            }
            "no_worker" => {}
            _ => unreachable!("{state}"),
        }
        if state == "unhealthy" {
            gateway
                .state
                .registry
                .mark_unhealthy("http://local-1:8080")
                .await;
        }
        gateway
    }

    fn assert_kept_off_remote(gateway: &TestGateway, response: &Response, case: &str) {
        assert!(
            !response.headers().contains_key("x-sie-fallback-reason"),
            "{case}"
        );
        assert!(
            !response.headers().contains_key("x-sie-fallback-error"),
            "{case}"
        );
        assert_ne!(stamped(response).0, Some("remote"), "{case}");
        assert!(
            !gateway
                .dispatcher
                .dispatched()
                .iter()
                .any(|work| work.bundle == REMOTE_LANE.2),
            "{case}"
        );
    }

    async fn detail_code(response: Response) -> serde_json::Value {
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 16384)
                .await
                .unwrap(),
        )
        .unwrap();
        body["detail"]["code"].clone()
    }

    fn encode_request(model: &str, remote: &[&str]) -> Request {
        let mut request = json_request(
            &format!("/v1/encode/{model}"),
            json!({"items": [{"text": "hello"}]}),
        );
        for value in remote {
            request
                .headers_mut()
                .append(REMOTE_HEADER, HeaderValue::from_str(value).unwrap());
        }
        request
    }

    #[tokio::test]
    async fn the_local_state_decides_the_bridge_as_the_shared_vectors_say() {
        for vector in gateway_vectors("bridge") {
            let case = vector.to_string();
            let gateway =
                gateway_in_local_state(vector["local"].as_str().unwrap(), &vector["triggers"])
                    .await;
            let response = buffered_surface(&gateway, "native", json!({})).await;
            if vector["remote"].as_bool().unwrap() {
                assert_eq!(response.status(), StatusCode::OK, "{case}");
                assert_eq!(
                    stamped(&response),
                    (Some("remote"), Some("team-sie")),
                    "{case}"
                );
                assert_eq!(
                    response.headers()["x-sie-fallback-reason"],
                    vector["fallback_reason"].as_str().unwrap(),
                    "{case}"
                );
                assert!(
                    gateway.dispatcher.dispatched().contains(&dispatched(
                        "generate",
                        REMOTE_LANE,
                        "acme/chat:remote"
                    )),
                    "{case}"
                );
            } else {
                assert_kept_off_remote(&gateway, &response, &case);
            }
        }
    }

    #[tokio::test]
    async fn a_remote_header_other_than_one_exact_forbid_is_refused_as_the_shared_vectors_say() {
        let header = remote_routing_vectors()["remote_header"].clone();
        let gateway = gateway_in_local_state("ready", &serde_json::Value::Null).await;
        for remote in header["refused"].as_array().unwrap() {
            let remote = header_values(remote);
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                encode_request("acme/hybrid-encode", &remote),
                "encode",
            )
            .await;
            assert_eq!(
                response.status(),
                vector_status(&header["refusal"]["status"]),
                "{remote:?}"
            );
            assert_eq!(
                detail_code(response).await,
                header["refusal"]["code"],
                "{remote:?}"
            );
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
        for remote in header["accepted"].as_array().unwrap() {
            let remote = header_values(remote);
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                encode_request("acme/hybrid-encode", &remote),
                "encode",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK, "{remote:?}");
            assert_eq!(stamped(&response), (Some("local"), None), "{remote:?}");
        }
    }

    #[tokio::test]
    async fn forbid_follows_the_shared_vectors() {
        for vector in gateway_vectors("forbid") {
            let case = vector.to_string();
            let state = vector["local"].as_str().unwrap_or("ready");
            let gateway = gateway_in_local_state(state, &serde_json::Value::Null).await;
            let model = match vector["model"].as_str().unwrap() {
                "bare" => "acme/hybrid-encode",
                "remote_profile" => "acme/hybrid-encode:remote",
                "remote_only" => "acme/remote",
                other => unreachable!("{other}"),
            };
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                encode_request(model, &["forbid"]),
                "encode",
            )
            .await;
            assert_kept_off_remote(&gateway, &response, &case);
            if let Some(refusal) = vector.get("refusal") {
                assert_eq!(
                    response.status(),
                    vector_status(&refusal["status"]),
                    "{case}"
                );
                assert_eq!(detail_code(response).await, refusal["code"], "{case}");
                assert!(gateway.dispatcher.dispatched().is_empty(), "{case}");
            } else {
                assert!(!vector["remote"].as_bool().unwrap(), "{case}");
            }
        }
    }

    #[tokio::test]
    async fn a_named_profile_is_served_as_written_as_the_shared_vectors_say() {
        for vector in gateway_vectors("named_profile") {
            let case = vector.to_string();
            let gateway =
                gateway_in_local_state(vector["local"].as_str().unwrap(), &serde_json::Value::Null)
                    .await;
            let response = match vector["model"].as_str().unwrap() {
                "local_profile" => {
                    buffered_surface(&gateway, "native", json!({"profile": "default"})).await
                }
                "remote_profile" => {
                    proxy_chat(
                        State(Arc::clone(&gateway.state)),
                        json_request(
                            "/v1/chat/completions",
                            json!({"model": "acme/chat:remote", "messages": [{"role": "user", "content": "hello"}]}),
                        ),
                    )
                    .await
                }
                other => unreachable!("{other}"),
            };
            if vector["remote"].as_bool().unwrap() {
                assert_eq!(response.status(), StatusCode::OK, "{case}");
                assert_eq!(
                    stamped(&response),
                    (Some("remote"), Some("team-sie")),
                    "{case}"
                );
                assert!(
                    !response.headers().contains_key("x-sie-fallback-reason"),
                    "{case}"
                );
            } else {
                assert_kept_off_remote(&gateway, &response, &case);
            }
        }
    }

    #[test]
    fn the_fallback_error_is_derived_as_the_shared_vectors_say() {
        for vector in gateway_vectors("fallback_error") {
            let status = vector_status(&vector["status"]);
            let code = vector["code"].as_str();
            let expected = vector["fallback_error"].as_str().unwrap();
            assert_eq!(fallback_error(status, code), expected, "{vector}");
            for in_header in [false, true] {
                let attempt = FallbackAttempt::default();
                assert!(attempt.begin(
                    StatusCode::SERVICE_UNAVAILABLE.into_response(),
                    FallbackTrigger::ModelLoading
                ));
                let failed = match code {
                    Some(code) if in_header => {
                        let mut failed = status.into_response();
                        failed
                            .headers_mut()
                            .insert("x-sie-error-code", HeaderValue::from_str(code).unwrap());
                        failed
                    }
                    _ => (status, axum::Json(json!({"error": {"code": code}}))).into_response(),
                };
                let restored = attempt.finish(failed);
                assert_eq!(
                    restored.headers()["x-sie-fallback-error"],
                    expected,
                    "{vector} in_header={in_header}"
                );
            }
        }
    }

    #[tokio::test]
    async fn the_restored_refusal_keeps_the_local_answer_as_the_shared_vectors_say() {
        for vector in gateway_vectors("restored_refusal") {
            let local = &vector["local"];
            let mut original = (
                vector_status(&local["status"]),
                axum::Json(json!({"detail": {"code": local["code"], "message": "local refusal"}})),
            )
                .into_response();
            if let Some(retry_after) = local["retry_after"].as_str() {
                original
                    .headers_mut()
                    .insert("retry-after", HeaderValue::from_str(retry_after).unwrap());
            }
            let trigger: FallbackTrigger =
                serde_json::from_value(vector["fallback_reason"].clone()).unwrap();
            let attempt = FallbackAttempt::default();
            assert!(attempt.begin(original, trigger));
            let failed = match &vector["attempt"] {
                serde_json::Value::String(kind) => {
                    assert_eq!(kind, "unanswered_before_deadline");
                    let mut failed = StatusCode::GATEWAY_TIMEOUT.into_response();
                    failed.extensions_mut().insert(UnansweredBeforeDeadline);
                    failed
                }
                attempt => {
                    let mut failed = vector_status(&attempt["status"]).into_response();
                    if let Some(code) = attempt["code"].as_str() {
                        failed
                            .headers_mut()
                            .insert("x-sie-error-code", HeaderValue::from_str(code).unwrap());
                    }
                    failed
                }
            };
            let restored = attempt.finish(failed);
            assert_eq!(
                restored.status(),
                vector_status(&local["status"]),
                "{vector}"
            );
            assert_eq!(
                restored
                    .headers()
                    .get("retry-after")
                    .map(|value| value.to_str().unwrap()),
                local["retry_after"].as_str(),
                "{vector}"
            );
            assert_eq!(stamped(&restored), (Some("local"), None), "{vector}");
            assert_eq!(
                restored.headers()["x-sie-fallback-reason"],
                vector["fallback_reason"].as_str().unwrap(),
                "{vector}"
            );
            assert_eq!(
                restored.headers()["x-sie-fallback-error"],
                vector["fallback_error"].as_str().unwrap(),
                "{vector}"
            );
            assert_eq!(detail_code(restored).await, local["code"], "{vector}");
        }
    }

    #[tokio::test]
    async fn threshold_shared_decision_routes_remote_without_local_demand_then_wakes_and_bridges() {
        use crate::handlers::test_support::ThresholdBroker;
        use crate::state::threshold_coordinator::{ThresholdDecision, ThresholdSampler};
        let Some(broker) = ThresholdBroker::start().await else {
            return;
        };
        let policy = "\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n";
        let generate = format!("{HYBRID_GENERATE_MODEL}{policy}");
        let extract = format!("{HYBRID_EXTRACT_MODEL}{policy}");
        let gateway = TestGateway::with_threshold_routing(&[&generate, &extract], true).await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let binding = broker.bind(&gateway).await;
        let mut sampler = ThresholdSampler::default();
        binding.coordinator.sample(&mut sampler).await.unwrap();
        for _ in 0..2 {
            tokio::time::sleep(Duration::from_millis(1050)).await;
            binding.coordinator.sample(&mut sampler).await.unwrap();
        }
        assert_eq!(
            binding.coordinator.decision("acme/chat").unwrap(),
            ThresholdDecision::Remote
        );
        for surface in ["native", "chat", "completions", "responses"] {
            let response = buffered_surface(&gateway, surface, json!({})).await;
            assert_eq!(response.status(), StatusCode::OK, "{surface}");
            assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
            assert!(!response.headers().contains_key("x-sie-fallback-reason"));
        }
        for msgpack in [false, true] {
            let response = proxy_request(
                State(Arc::clone(&gateway.state)),
                extraction_request(msgpack, json!({})),
                "extract",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
        }
        assert!(gateway.state.demand_tracker.active_lanes().is_empty());
        assert!(gateway
            .dispatcher
            .dispatched()
            .iter()
            .all(|work| work.bundle == "remote"
                && work.model.ends_with(":remote")
                && work.endpoint != "load"));
        assert_eq!(gateway.dispatcher.execution_authority(), vec![true; 6]);
        assert_eq!(gateway.dispatcher.fallback_reasons(), vec![None; 6]);

        for surface in ["native", "chat", "completions"] {
            let state = State(Arc::clone(&gateway.state));
            let response = match surface {
                "native" => proxy_request(state, json_request("/v1/generate/acme/chat", json!({"prompt":"hello", "max_new_tokens":4, "stream":true})), "generate").await,
                "chat" => proxy_chat(state, json_request("/v1/chat/completions", json!({"model":"acme/chat", "messages":[{"role":"user", "content":"hello"}], "stream":true}))).await,
                "completions" => proxy_completions(state, json_request("/v1/completions", json!({"model":"acme/chat", "prompt":"hello", "max_tokens":4, "stream":true}))).await,
                _ => unreachable!(),
            };
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
            let body = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            assert!(String::from_utf8_lossy(&body).contains("[DONE]"));
        }
        assert!(gateway.state.demand_tracker.active_lanes().is_empty());
        assert_eq!(gateway.dispatcher.execution_authority(), vec![true; 9]);
        assert_eq!(gateway.dispatcher.fallback_reasons(), vec![None; 9]);

        // Demand from two gateways reaches one decision, rather than dividing
        // the configured rate independently at each gateway.
        let replica = TestGateway::with_threshold_routing(&[&generate, &extract], true).await;
        replica
            .add_verified_worker("remote-2", REMOTE_LANE, &[])
            .await;
        let second = broker.bind(&replica).await;
        let mut standby = ThresholdSampler::default();
        let _ = second.coordinator.sample(&mut standby).await;
        tokio::time::sleep(Duration::from_millis(1050)).await;
        binding.coordinator.sample(&mut sampler).await.unwrap();
        for _ in 0..3 {
            second.coordinator.record_request("acme/chat").unwrap();
        }
        let _ = second.coordinator.sample(&mut standby).await;
        tokio::time::sleep(Duration::from_millis(1050)).await;
        binding.coordinator.sample(&mut sampler).await.unwrap();
        assert_eq!(
            binding.coordinator.decision("acme/chat").unwrap(),
            ThresholdDecision::WakeLocal
        );
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &[])
            .await;
        let before = gateway.dispatcher.dispatched().len();
        let response = buffered_surface(&gateway, "native", json!({})).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sie-fallback-reason"], "model_loading");
        assert_eq!(
            &gateway.dispatcher.dispatched()[before..],
            &[
                dispatched("load", LOCAL_LANE, "acme/chat"),
                dispatched("generate", REMOTE_LANE, "acme/chat:remote"),
            ]
        );
        assert_eq!(
            gateway.dispatcher.fallback_reasons().last(),
            Some(&Some(FallbackTrigger::ModelLoading))
        );
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &["acme/chat"])
            .await;
        let response = buffered_surface(&gateway, "chat", json!({})).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(stamped(&response), (Some("local"), None));
        assert_eq!(
            gateway.dispatcher.dispatched().last().unwrap().model,
            "acme/chat"
        );
        assert_eq!(gateway.dispatcher.fallback_reasons().last(), Some(&None));
    }
    #[tokio::test]
    async fn threshold_authority_preserves_selectors_validation_and_configuration_fences() {
        use crate::handlers::test_support::ThresholdBroker;
        use crate::state::threshold_coordinator::ThresholdSampler;
        use sha2::{Digest, Sha256};
        let Some(broker) = ThresholdBroker::start().await else {
            return;
        };
        let policy = "\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n";
        let config = format!("{HYBRID_GENERATE_MODEL}{policy}");
        let gateway = TestGateway::with_threshold_routing(&[&config], true).await;
        // A legacy worker may not become the authority of a threshold rewrite.
        gateway.add_worker("remote-1", REMOTE_LANE, &[]).await;
        let binding = broker.bind(&gateway).await;
        let mut sampler = ThresholdSampler::default();
        binding.coordinator.sample(&mut sampler).await.unwrap();
        for _ in 0..2 {
            tokio::time::sleep(Duration::from_millis(1050)).await;
            binding.coordinator.sample(&mut sampler).await.unwrap();
        }
        let response = proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/generate/acme/chat",
                json!({"prompt":"", "max_new_tokens":0}),
            ),
            "generate",
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let response = buffered_surface(&gateway, "native", json!({"profile":"default"})).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        binding.coordinator.sample(&mut sampler).await.unwrap();
        let key = Sha256::digest(b"acme/chat")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let counts = broker
            .context
            .get_key_value("SIE_THRESHOLD_COUNTS")
            .await
            .unwrap();
        let message = counts
            .stream
            .get_last_raw_message_by_subject(&format!("$KV.SIE_THRESHOLD_COUNTS.{key}"))
            .await
            .unwrap();
        let counter: serde_json::Value = serde_json::from_slice(&message.payload).unwrap();
        assert_eq!(counter["total"], 0);
        assert!(gateway.dispatcher.dispatched().is_empty());

        for lane in gateway.state.demand_tracker.active_lanes() {
            gateway.state.demand_tracker.clear(&lane);
        }
        let before = gateway.dispatcher.dispatched().len();
        let response = buffered_surface(&gateway, "native", json!({})).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(gateway.dispatcher.dispatched().len(), before);
        assert!(gateway
            .state
            .demand_tracker
            .active_lanes()
            .iter()
            .all(|lane| lane.bundle() != "default"));
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        let mut request = json_request(
            "/v1/generate/acme/chat",
            json!({"prompt":"hello", "max_new_tokens":4}),
        );
        request
            .headers_mut()
            .insert("x-sie-remote", HeaderValue::from_static("forbid"));
        let response = proxy_request(State(Arc::clone(&gateway.state)), request, "generate").await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(gateway.dispatcher.dispatched().is_empty());

        // Forbid constrains execution, while valid bare-model demand still
        // contributes to the shared decision to wake local capacity.
        binding.coordinator.sample(&mut sampler).await.unwrap();
        let message = counts
            .stream
            .get_last_raw_message_by_subject(&format!("$KV.SIE_THRESHOLD_COUNTS.{key}"))
            .await
            .unwrap();
        let counter: serde_json::Value = serde_json::from_slice(&message.payload).unwrap();
        assert_eq!(counter["total"], 2);

        // Replace the registry while retaining the same epoch, and even the
        // same execution hashes: the old decision must still lose authority.
        gateway.state.model_registry.reload();
        assert!(gateway
            .state
            .model_registry
            .threshold_remote_route("acme/chat", 1)
            .is_none());
        assert!(gateway
            .state
            .model_registry
            .threshold_remote_route("acme/chat", 2)
            .is_none());
        gateway.state.model_registry.clear_threshold_binding();
        let response = buffered_surface(&gateway, "chat", json!({})).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sie-fallback-reason"], "provisioning");
    }

    const ADMITTED_IDENTITY: &str =
        "v2:sha256:1111111111111111111111111111111111111111111111111111111111111111";
    const UNADMITTED_IDENTITY: &str =
        "v2:sha256:2222222222222222222222222222222222222222222222222222222222222222";
    const MODEL_CONTRACT: &str = "3333333333333333333333333333333333333333333333333333333333333333";

    fn unix_ms_from_now(offset_ms: i64) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        (now + offset_ms) as u64
    }

    fn inventory(profile: serde_json::Value) -> serde_json::Value {
        json!({
            "observed_at_unix_ms": 1,
            "children": [{
                "child_index": 0,
                "status": "observed",
                "snapshot": {
                    "runtime_instance_id": "9".repeat(64),
                    "complete": true,
                    "profiles": [profile],
                },
            }],
        })
    }

    fn remote_admission(expires_in_ms: i64) -> serde_json::Value {
        inventory(json!({
            "model_id": "acme/hybrid-encode",
            "model_contract_sha256": MODEL_CONTRACT,
            "local_identity": null,
            "admission": {
                "sha256": "a".repeat(64),
                "kind": "sie",
                "local_identities": [ADMITTED_IDENTITY],
                "model_contract_sha256": MODEL_CONTRACT,
                "outputs": ["dense"],
                "expires_at_unix_ms": unix_ms_from_now(expires_in_ms),
            },
        }))
    }

    fn local_identity(identity: &str) -> serde_json::Value {
        inventory(json!({
            "model_id": "acme/hybrid-encode",
            "model_contract_sha256": MODEL_CONTRACT,
            "local_identity": identity,
        }))
    }

    async fn numerical_gateway(policy: &str, threshold: bool) -> TestGateway {
        let encode = format!("{HYBRID_ENCODE_MODEL}{policy}");
        let gateway = TestGateway::with_threshold_routing(&[&encode], threshold).await;
        gateway
            .add_numerical_worker("remote-1", REMOTE_LANE, &[], true, remote_admission(60_000))
            .await;
        gateway
    }

    const NUMERICAL_FALLBACK: &str = "\nrouting:\n  policy: fallback\n  fallback_profile: remote\n";

    async fn encode(gateway: &TestGateway, body: serde_json::Value) -> Response {
        proxy_request(
            State(Arc::clone(&gateway.state)),
            json_request("/v1/encode/acme/hybrid-encode", body),
            "encode",
        )
        .await
    }

    fn assert_numerical_work_stayed_local(gateway: &TestGateway, response: &Response, case: &str) {
        assert!(
            !response.headers().contains_key("x-sie-fallback-reason"),
            "{case}"
        );
        assert_ne!(stamped(response).0, Some("remote"), "{case}");
        assert!(
            gateway
                .dispatcher
                .dispatched()
                .iter()
                .all(|work| work.bundle != REMOTE_LANE.2),
            "{case}"
        );
        assert!(
            gateway
                .dispatcher
                .numerical_admissions()
                .iter()
                .all(Option::is_none),
            "{case}"
        );
    }

    #[tokio::test]
    async fn an_admitted_numerical_bridge_names_the_admission_its_worker_advertised() {
        for local in ["cold", "loading"] {
            let gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
            if local == "loading" {
                gateway
                    .add_numerical_worker(
                        "local-1",
                        LOCAL_LANE,
                        &[],
                        true,
                        local_identity(ADMITTED_IDENTITY),
                    )
                    .await;
            }
            let response = encode(&gateway, json!({"items":[{"text":"hello"}]})).await;
            assert_eq!(response.status(), StatusCode::OK, "{local}");
            assert_eq!(
                stamped(&response),
                (Some("remote"), Some("team-sie")),
                "{local}"
            );
            let remote: Vec<_> = gateway
                .dispatcher
                .dispatched()
                .into_iter()
                .filter(|work| work.endpoint != "load")
                .zip(gateway.dispatcher.numerical_admissions())
                .zip(gateway.dispatcher.fallback_reasons())
                .filter(|((work, _), _)| work.bundle == REMOTE_LANE.2)
                .collect();
            assert_eq!(remote.len(), 1, "{local}");
            let ((work, admission), reason) = &remote[0];
            assert_eq!(work.model, "acme/hybrid-encode:remote");
            assert_eq!(
                admission.as_deref(),
                Some("a".repeat(64).as_str()),
                "{local}"
            );
            assert!(reason.is_some(), "{local}");
        }
    }

    #[tokio::test]
    async fn the_embeddings_route_bridges_numerical_work_under_the_same_admission() {
        let gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
        let response = proxy_openai_embeddings(
            State(Arc::clone(&gateway.state)),
            json_request(
                "/v1/embeddings",
                json!({"model":"acme/hybrid-encode", "input":"hello"}),
            ),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gateway.dispatcher.numerical_admissions(),
            vec![Some("a".repeat(64))]
        );
    }

    #[tokio::test]
    async fn numerical_work_stays_local_unless_a_current_admission_covers_every_local_process() {
        for case in [
            "uncovered local identity",
            "local worker without inventory",
            "admission about to expire",
            "remote worker without the fence",
            "profile selected in the body",
            "health heard for less than a heartbeat timeout",
            "quantized output",
            "a runtime option the admission did not measure",
            "an output the admission did not measure",
        ] {
            let mut gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
            gateway.set_request_timeout(1.0);
            let mut body = json!({"items":[{"text":"hello"}]});
            match case {
                "uncovered local identity" => {
                    gateway
                        .add_numerical_worker(
                            "local-1",
                            LOCAL_LANE,
                            &[],
                            true,
                            local_identity(UNADMITTED_IDENTITY),
                        )
                        .await;
                }
                "local worker without inventory" => {
                    gateway
                        .add_verified_worker("local-1", LOCAL_LANE, &[])
                        .await;
                }
                "admission about to expire" => {
                    gateway
                        .add_numerical_worker(
                            "remote-1",
                            REMOTE_LANE,
                            &[],
                            true,
                            remote_admission(4_000),
                        )
                        .await;
                }
                "remote worker without the fence" => {
                    gateway
                        .add_numerical_worker(
                            "remote-1",
                            REMOTE_LANE,
                            &[],
                            false,
                            remote_admission(60_000),
                        )
                        .await;
                }
                "profile selected in the body" => {
                    body = json!({"items":[{"text":"hello"}], "params":{"options":{"profile":"default"}}});
                }
                "health heard for less than a heartbeat timeout" => {
                    gateway.state.registry.health_subscription_started();
                }
                "quantized output" => {
                    body = json!({"items":[{"text":"hello"}], "params":{"output_dtype":"int8"}});
                }
                "a runtime option the admission did not measure" => {
                    body = json!({"items":[{"text":"hello"}], "params":{"options":{"normalize":false}}});
                }
                "an output the admission did not measure" => {
                    body =
                        json!({"items":[{"text":"hello"}], "params":{"output_types":["sparse"]}});
                }
                _ => unreachable!(),
            }
            let response = encode(&gateway, body).await;
            assert!(!response.status().is_success(), "{case}");
            assert_numerical_work_stayed_local(&gateway, &response, case);
        }
    }

    #[tokio::test]
    async fn threshold_routes_numerical_work_remote_only_under_a_covering_admission() {
        use crate::handlers::test_support::ThresholdBroker;
        use crate::state::threshold_coordinator::{ThresholdDecision, ThresholdSampler};
        let policy = "\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n";
        for covered in [true, false] {
            let Some(broker) = ThresholdBroker::start().await else {
                return;
            };
            let mut gateway = numerical_gateway(policy, true).await;
            gateway.set_request_timeout(1.0);
            gateway
                .add_numerical_worker(
                    "local-1",
                    LOCAL_LANE,
                    &["acme/hybrid-encode"],
                    true,
                    local_identity(if covered {
                        ADMITTED_IDENTITY
                    } else {
                        UNADMITTED_IDENTITY
                    }),
                )
                .await;
            let binding = broker.bind(&gateway).await;
            let mut sampler = ThresholdSampler::default();
            binding.coordinator.sample(&mut sampler).await.unwrap();
            for _ in 0..2 {
                tokio::time::sleep(Duration::from_millis(1050)).await;
                binding.coordinator.sample(&mut sampler).await.unwrap();
            }
            assert_eq!(
                binding.coordinator.decision("acme/hybrid-encode").unwrap(),
                ThresholdDecision::Remote
            );
            let response = encode(&gateway, json!({"items":[{"text":"hello"}]})).await;
            assert_eq!(response.status(), StatusCode::OK, "covered={covered}");
            if covered {
                assert_eq!(stamped(&response), (Some("remote"), Some("team-sie")));
                assert_eq!(
                    gateway.dispatcher.numerical_admissions(),
                    vec![Some("a".repeat(64))]
                );
                assert_eq!(gateway.dispatcher.fallback_reasons(), vec![None]);
            } else {
                assert_numerical_work_stayed_local(
                    &gateway,
                    &response,
                    "uncovered threshold route",
                );
            }
        }
    }

    #[tokio::test]
    async fn a_refused_admitted_threshold_attempt_is_answered_at_once() {
        use crate::handlers::test_support::ThresholdBroker;
        use crate::state::threshold_coordinator::{ThresholdDecision, ThresholdSampler};
        let Some(broker) = ThresholdBroker::start().await else {
            return;
        };
        let policy = "\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n";
        let mut gateway = numerical_gateway(policy, true).await;
        gateway.set_request_timeout(30.0);
        let binding = broker.bind(&gateway).await;
        let mut sampler = ThresholdSampler::default();
        binding.coordinator.sample(&mut sampler).await.unwrap();
        for _ in 0..2 {
            tokio::time::sleep(Duration::from_millis(1050)).await;
            binding.coordinator.sample(&mut sampler).await.unwrap();
        }
        assert_eq!(
            binding.coordinator.decision("acme/hybrid-encode").unwrap(),
            ThresholdDecision::Remote
        );
        gateway
            .dispatcher
            .answer_only_bridged_remote_work("INFERENCE_ERROR", 1);
        let started = std::time::Instant::now();
        let response = encode(&gateway, json!({"items":[{"text":"hello"}]})).await;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "1");
        assert_eq!(response.headers()["x-sie-error-code"], "INFERENCE_ERROR");
        assert_eq!(
            gateway.dispatcher.numerical_admissions(),
            vec![Some("a".repeat(64))]
        );
    }

    #[tokio::test]
    async fn a_worker_of_another_pool_does_not_close_the_bridge() {
        let gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
        gateway
            .add_numerical_worker(
                "batch-1",
                ("batch", LOCAL_LANE.1, LOCAL_LANE.2),
                &[],
                true,
                inventory(json!({
                    "model_id": "acme/batch-only",
                    "model_contract_sha256": MODEL_CONTRACT,
                    "local_identity": UNADMITTED_IDENTITY,
                })),
            )
            .await;
        let response = encode(&gateway, json!({"items":[{"text":"hello"}]})).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gateway.dispatcher.numerical_admissions(),
            vec![Some("a".repeat(64))]
        );
    }

    #[tokio::test]
    async fn a_request_that_can_never_bridge_keeps_its_ordinary_local_path() {
        let gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
        gateway
            .add_numerical_worker(
                "local-1",
                LOCAL_LANE,
                &[],
                true,
                local_identity(ADMITTED_IDENTITY),
            )
            .await;
        let response = encode(
            &gateway,
            json!({"items":[{"text":"hello"}], "params":{"output_dtype":"int8"}}),
        )
        .await;
        assert!(
            !response.headers().contains_key("x-sie-fallback-reason"),
            "a request outside the admission is not a bridge candidate"
        );
        assert!(gateway
            .dispatcher
            .dispatched()
            .iter()
            .all(|work| work.endpoint != "load" && work.bundle != REMOTE_LANE.2));
        assert!(gateway
            .dispatcher
            .dispatched()
            .iter()
            .any(|work| work.bundle == LOCAL_LANE.2));
        assert!(gateway
            .dispatcher
            .numerical_admissions()
            .iter()
            .all(Option::is_none));
    }

    #[tokio::test]
    async fn an_admitted_numerical_bridge_accepts_a_query_flag() {
        let gateway = numerical_gateway(NUMERICAL_FALLBACK, false).await;
        let response = encode(
            &gateway,
            json!({"items":[{"text":"hello"}], "params":{"is_query":true, "options":{"is_query":true}}}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gateway.dispatcher.numerical_admissions(),
            vec![Some("a".repeat(64))]
        );
    }

    #[tokio::test]
    async fn an_invalid_numerical_request_is_refused_before_either_side_counts_it() {
        for policy in [
            NUMERICAL_FALLBACK,
            "\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n",
        ] {
            let gateway = numerical_gateway(policy, policy != NUMERICAL_FALLBACK).await;
            gateway
                .add_verified_worker("local-1", LOCAL_LANE, &[])
                .await;
            for body in [
                json!({"items":"not-an-array"}),
                json!({"items":[]}),
            ] {
                let response = encode(&gateway, body.clone()).await;
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
                assert!(
                    !response.headers().contains_key("x-sie-fallback-reason"),
                    "{body}"
                );
            }
            assert!(gateway.dispatcher.dispatched().is_empty());
            assert!(gateway.dispatcher.numerical_admissions().is_empty());
        }
    }

    use crate::observability::metrics::{AdmissionOutcome, AdmissionOutcomeSlot};
    use crate::server::{
        GenerationRequestIntent, GenerationRoutePolicy, GovernedGenerationRoute, ModelAccessPolicy,
    };

    /// A deployment policy that records every remote route the gateway asks
    /// about and answers `admit`.
    #[derive(Default)]
    struct RoutePolicy {
        admit: bool,
        admit_first_only: bool,
        hide_remote: bool,
        refuse_remote_serving: bool,
        govern_generation: bool,
        asked: std::sync::Mutex<Vec<(String, String, RemoteRouteReason)>>,
    }

    impl RoutePolicy {
        fn asked(&self) -> Vec<(String, String, RemoteRouteReason)> {
            self.asked.lock().unwrap().clone()
        }
    }

    impl ModelAccessPolicy for RoutePolicy {
        fn visible(&self, resolved_model: &str, _ext: &axum::http::Extensions) -> bool {
            !(self.hide_remote && resolved_model.ends_with(":remote"))
        }

        fn serving_refusal(
            &self,
            resolved_model: &str,
            ext: &axum::http::Extensions,
        ) -> Option<Response> {
            (self.refuse_remote_serving && resolved_model.ends_with(":remote")).then(|| {
                if let Some(slot) = ext.get::<AdmissionOutcomeSlot>() {
                    slot.set(AdmissionOutcome::Forbidden);
                }
                StatusCode::FORBIDDEN.into_response()
            })
        }

        fn generation_route_policy(&self) -> Option<&dyn GenerationRoutePolicy> {
            self.govern_generation
                .then_some(self as &dyn GenerationRoutePolicy)
        }

        fn remote_route_admitted(
            &self,
            model: &str,
            remote_model: &str,
            reason: RemoteRouteReason,
            _ext: &axum::http::Extensions,
        ) -> bool {
            let mut asked = self.asked.lock().unwrap();
            asked.push((model.to_string(), remote_model.to_string(), reason));
            if self.admit_first_only {
                return asked.len() == 1;
            }
            self.admit
        }
    }

    impl GenerationRoutePolicy for RoutePolicy {
        fn resolve(
            &self,
            customer_model: &str,
            intent: GenerationRequestIntent,
        ) -> Option<GovernedGenerationRoute> {
            (intent == GenerationRequestIntent::Default).then(|| GovernedGenerationRoute {
                model: customer_model.to_string(),
                bundle: LOCAL_LANE.2.to_string(),
                pool: LOCAL_LANE.0.to_string(),
                machine_profile: LOCAL_LANE.1.to_string(),
            })
        }
    }

    /// A policy that only decides visibility, so remote routes take the
    /// trait's default answer.
    struct VisibilityOnlyPolicy;

    impl ModelAccessPolicy for VisibilityOnlyPolicy {
        fn visible(&self, _resolved_model: &str, _ext: &axum::http::Extensions) -> bool {
            true
        }
    }

    fn fallback_config(model: &str) -> String {
        format!("{model}\nrouting:\n  policy: fallback\n  fallback_profile: remote\n")
    }

    /// A gateway with no local worker and one verified remote worker.
    async fn cold_gateway(models: &[&str], policy: Arc<dyn ModelAccessPolicy>) -> TestGateway {
        let mut gateway = TestGateway::new(models).await;
        gateway.install_policy(policy);
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;
        gateway
    }

    async fn extract(gateway: &TestGateway, request: Request) -> Response {
        proxy_request(State(Arc::clone(&gateway.state)), request, "extract").await
    }

    fn assert_local_provisioning(response: &Response, gateway: &TestGateway) {
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "60");
        assert!(!response.headers().contains_key("x-sie-fallback-reason"));
        assert!(gateway.dispatcher.dispatched().is_empty());
    }

    #[tokio::test]
    async fn a_policy_that_does_not_admit_remote_routes_keeps_the_local_answer() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        let gateway = cold_gateway(&[&config], Arc::new(VisibilityOnlyPolicy)).await;

        let response = extract(&gateway, extraction_request(false, json!({}))).await;

        assert_local_provisioning(&response, &gateway);
    }

    #[tokio::test]
    async fn an_admitting_policy_is_asked_once_with_canonical_ids_and_the_trigger() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        for msgpack in [false, true] {
            let policy = Arc::new(RoutePolicy {
                admit: true,
                ..Default::default()
            });
            let gateway = cold_gateway(&[&config], policy.clone()).await;

            let response = extract(&gateway, extraction_request(msgpack, json!({}))).await;

            assert_eq!(response.status(), StatusCode::OK, "msgpack={msgpack}");
            assert_eq!(response.headers()["x-sie-fallback-reason"], "provisioning");
            assert_eq!(
                gateway.dispatcher.dispatched(),
                vec![dispatched("extract", REMOTE_LANE, "acme/extract:remote")]
            );
            assert_eq!(
                policy.asked(),
                vec![(
                    "acme/extract".to_string(),
                    "acme/extract:remote".to_string(),
                    RemoteRouteReason::Fallback(FallbackTrigger::Provisioning),
                )]
            );
        }
    }

    #[tokio::test]
    async fn a_remote_profile_hidden_from_the_caller_is_never_asked_about_or_routed_to() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        let policy = Arc::new(RoutePolicy {
            admit: true,
            hide_remote: true,
            ..Default::default()
        });
        let gateway = cold_gateway(&[&config], policy.clone()).await;

        let response = extract(&gateway, extraction_request(false, json!({}))).await;

        assert_local_provisioning(&response, &gateway);
        assert!(policy.asked().is_empty());
    }

    #[tokio::test]
    async fn remote_forbid_never_asks_the_policy() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        let policy = Arc::new(RoutePolicy {
            admit: true,
            ..Default::default()
        });
        let gateway = cold_gateway(&[&config], policy.clone()).await;
        let mut request = extraction_request(false, json!({}));
        request
            .headers_mut()
            .insert(REMOTE_HEADER, HeaderValue::from_static("forbid"));

        let response = extract(&gateway, request).await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(gateway.dispatcher.dispatched().is_empty());
        assert!(policy.asked().is_empty());
    }

    #[tokio::test]
    async fn a_governed_generation_policy_keeps_generation_local_and_still_bridges_extraction() {
        let generate = fallback_config(HYBRID_GENERATE_MODEL);
        let extraction = fallback_config(HYBRID_EXTRACT_MODEL);
        let policy = Arc::new(RoutePolicy {
            admit: true,
            govern_generation: true,
            ..Default::default()
        });
        let gateway = cold_gateway(&[&generate, &extraction], policy.clone()).await;

        for surface in ["native", "chat", "completions", "responses"] {
            let response = buffered_surface(&gateway, surface, json!({})).await;
            assert_eq!(
                response.status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{surface}"
            );
            assert!(
                !response.headers().contains_key("x-sie-fallback-reason"),
                "{surface}"
            );
        }
        assert!(gateway.dispatcher.dispatched().is_empty());
        assert!(policy.asked().is_empty());

        let response = extract(&gateway, extraction_request(false, json!({}))).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![dispatched("extract", REMOTE_LANE, "acme/extract:remote")]
        );
        assert_eq!(policy.asked().len(), 1);
    }

    #[tokio::test]
    async fn a_threshold_route_asks_the_policy_before_routing_remotely() {
        use crate::handlers::test_support::ThresholdBroker;
        use crate::state::threshold_coordinator::ThresholdSampler;
        let config = format!("{HYBRID_EXTRACT_MODEL}\nrouting:\n  policy: threshold\n  fallback_profile: remote\n  wake_above: 1\n  sleep_below: 0.5\n  window_s: 1\n  cooldown_s: 1\n");
        for admit in [true, false] {
            let Some(broker) = ThresholdBroker::start().await else {
                return;
            };
            let policy = Arc::new(RoutePolicy {
                admit,
                ..Default::default()
            });
            let mut gateway = TestGateway::with_threshold_routing(&[&config], true).await;
            gateway.install_policy(policy.clone());
            gateway
                .add_verified_worker("remote-1", REMOTE_LANE, &[])
                .await;
            let binding = broker.bind(&gateway).await;
            let mut sampler = ThresholdSampler::default();
            binding.coordinator.sample(&mut sampler).await.unwrap();
            for _ in 0..2 {
                tokio::time::sleep(Duration::from_millis(1050)).await;
                binding.coordinator.sample(&mut sampler).await.unwrap();
            }

            let response = extract(&gateway, extraction_request(false, json!({}))).await;

            let asked = policy.asked();
            assert_eq!(
                asked.first(),
                Some(&(
                    "acme/extract".to_string(),
                    "acme/extract:remote".to_string(),
                    RemoteRouteReason::Threshold,
                )),
                "admit={admit}"
            );
            if admit {
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(response.headers()["x-sie-served-by"], "remote");
                assert_eq!(
                    gateway.dispatcher.dispatched(),
                    vec![dispatched("extract", REMOTE_LANE, "acme/extract:remote")]
                );
            } else {
                assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
                assert!(gateway.dispatcher.dispatched().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn a_remote_profile_the_deployment_does_not_serve_keeps_the_bare_model_local() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        for (refuse_remote_serving, admit, bridged, asked) in [
            (true, true, false, 0),
            (false, false, false, 1),
            (false, true, true, 1),
        ] {
            let case = format!("refuse_remote_serving={refuse_remote_serving} admit={admit}");
            let policy = Arc::new(RoutePolicy {
                admit,
                refuse_remote_serving,
                ..Default::default()
            });
            let gateway = cold_gateway(&[&config], policy.clone()).await;
            let outcome = AdmissionOutcomeSlot::default();
            let mut request = extraction_request(false, json!({}));
            request.extensions_mut().insert(outcome.clone());

            let response = extract(&gateway, request).await;

            if bridged {
                assert_eq!(response.status(), StatusCode::OK, "{case}");
                assert_eq!(stamped(&response).0, Some("remote"), "{case}");
            } else {
                assert_local_provisioning(&response, &gateway);
            }
            assert_eq!(policy.asked().len(), asked, "{case}");
            assert_eq!(outcome.get(), None, "{case}");
        }
    }

    #[tokio::test]
    async fn a_named_remote_profile_is_never_asked_and_follows_visible_and_serving_refusal() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        for refuse_remote_serving in [false, true] {
            let policy = Arc::new(RoutePolicy {
                refuse_remote_serving,
                ..Default::default()
            });
            let gateway = cold_gateway(&[&config], policy.clone()).await;
            let mut request = extraction_request(false, json!({}));
            *request.uri_mut() = "/v1/extract/acme/extract:remote".parse().unwrap();

            let response = extract(&gateway, request).await;

            if refuse_remote_serving {
                assert_eq!(response.status(), StatusCode::FORBIDDEN);
                assert!(gateway.dispatcher.dispatched().is_empty());
            } else {
                assert_eq!(response.status(), StatusCode::OK);
                assert_eq!(
                    gateway.dispatcher.dispatched(),
                    vec![dispatched("extract", REMOTE_LANE, "acme/extract:remote")]
                );
            }
            assert!(policy.asked().is_empty());
        }
    }

    #[tokio::test]
    async fn a_policy_is_decided_once_per_request_and_reason() {
        let config = fallback_config(HYBRID_EXTRACT_MODEL);
        let policy = Arc::new(RoutePolicy {
            admit_first_only: true,
            ..Default::default()
        });
        let mut gateway = TestGateway::new(&[&config]).await;
        gateway.install_policy(policy.clone());
        gateway
            .add_verified_worker("local-1", LOCAL_LANE, &[])
            .await;
        gateway
            .add_verified_worker("remote-1", REMOTE_LANE, &[])
            .await;

        let response = extract(&gateway, extraction_request(false, json!({}))).await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-sie-fallback-reason"], "model_loading");
        assert_eq!(
            gateway.dispatcher.dispatched(),
            vec![
                dispatched("load", LOCAL_LANE, "acme/extract"),
                dispatched("extract", REMOTE_LANE, "acme/extract:remote"),
            ]
        );
        assert_eq!(
            policy.asked(),
            vec![(
                "acme/extract".to_string(),
                "acme/extract:remote".to_string(),
                RemoteRouteReason::Fallback(FallbackTrigger::ModelLoading),
            )]
        );
    }
}
