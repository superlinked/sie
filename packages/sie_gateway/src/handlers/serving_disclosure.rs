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

use crate::server::AppState;
use crate::types::model::ServedBy;

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
        if let Some(disclosure) = extensions.get::<Self>() {
            *disclosure.0.lock().unwrap_or_else(PoisonError::into_inner) =
                state.model_registry.served_by(model);
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::body::Body;
    use axum::extract::State;
    use axum::http::{Method, StatusCode};
    use axum::response::{IntoResponse, Response};
    use serde_json::json;

    use super::*;

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
        Dispatched, TestGateway, HYBRID_GENERATE_MODEL, LOCAL_ENCODE_MODEL, LOCAL_LANE,
        REMOTE_ENCODE_MODEL, REMOTE_LANE,
    };

    fn json_request(uri: &str, body: serde_json::Value) -> Request {
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
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
}
