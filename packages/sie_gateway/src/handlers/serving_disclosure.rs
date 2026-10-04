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

use crate::observability::metrics as telemetry;
use crate::server::AppState;
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
        let failure = if response.status().is_client_error() {
            "INVALID_INPUT"
        } else {
            "INFERENCE_ERROR"
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
        Dispatched, TestGateway, HYBRID_EXTRACT_MODEL, HYBRID_GENERATE_MODEL, LOCAL_ENCODE_MODEL,
        LOCAL_LANE, REMOTE_ENCODE_MODEL, REMOTE_LANE,
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
}
