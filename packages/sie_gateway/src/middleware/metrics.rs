//! Canonical request-completion telemetry middleware.
//!
//! Wraps the whole router so every response is observed — including
//! 4xx/5xx early returns, timeouts, axum-generated 500s for panics, and
//! any future exit path that forgets to instrument itself by hand.
//! The terminal increment that used to live inside
//! `handlers::proxy::queue_mode_proxy` has been removed in favour of
//! this layer.
//!
//! Only inference endpoints (`/v1/encode/*`, `/v1/score/*`,
//! `/v1/extract/*`, `/v1/generate/*`, `/v1/embeddings`, `/v1/moderations`, and
//! the OpenAI-compatible generation/audio routes) populate `sie.gateway.requests` +
//! `sie.gateway.request.duration`. Infrastructure paths
//! (`/health*`, `/ws/*`, `/v1/configs/*`, `/v1/pools`,
//! `/v1/models`) are intentionally skipped: they are not traffic, and
//! counting them would drown the inference error-rate dashboards.
//!
//! The `machine_profile` label is *not* read from the raw
//! `x-sie-machine-profile` header. That header can carry a pool
//! prefix (`pool/l4`), a GPU alias (`l4` that resolves to `l4-spot`),
//! or be absent entirely for default-routed requests — using it as a
//! label directly would (a) break joins with every other
//! `{machine_profile}` series produced elsewhere in the gateway and
//! (b) let unbounded client-controlled values create new time series.
//!
//! Instead the middleware installs a [`MetricLabelsSlot`] into the
//! request's extensions before forwarding. `handlers::proxy` fills it
//! once after it has resolved the canonical GPU label, and this layer
//! reads it back after the inner service has produced a response.
//! Anything that returns before the handler has normalized (`model is
//! required`, `/ws/*` misroutes that somehow reach here, …) falls
//! back to the contract value `"other"`.
//!
//! The KEDA rejection control counter (`sie.gateway.rejected.requests`)
//! stays in the handler because its four scale-up reasons (`backpressure`,
//! `no_consumers`, `publish_ack_failed`, `upstream_result_timeout`) are only
//! knowable at the point of rejection. Every other reason is deliberately omitted from that metric;
//! this middleware still records its bounded HTTP request outcome.
//!
//! ## Hot-path cost
//!
//! Non-inference routes exit after a borrowed-path classification. Inference
//! routes create the request span/context and synchronously record one bounded
//! telemetry event before returning the response. Export remains asynchronous,
//! but local allocation and aggregation are part of request latency; measured
//! telemetry-off/on benchmarks, not a source comment, own the cost claim.

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use axum::Router;
use opentelemetry::trace::{FutureExt, TraceContextExt};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tower::{Layer, Service};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::observability::metrics as telemetry;

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestObservation {
    operation: &'static str,
    status: u16,
    machine_profile: String,
}

#[derive(Clone, Default)]
pub struct MetricsLayer {
    #[cfg(test)]
    observations: Option<std::sync::Arc<std::sync::Mutex<Vec<RequestObservation>>>>,
}

impl MetricsLayer {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn observing(observations: std::sync::Arc<std::sync::Mutex<Vec<RequestObservation>>>) -> Self {
        Self {
            observations: Some(observations),
        }
    }
}

/// Install the outer request telemetry layer only when at least one request
/// signal has a live provider. With all signals absent the returned router is
/// the original Tower stack, so disabled telemetry creates no middleware,
/// spans, extension slots, clocks, instruments, or point attributes.
pub fn apply_request_telemetry(router: Router) -> Router {
    apply_request_telemetry_if(
        router,
        crate::observability::tracing::request_telemetry_enabled(),
    )
}

fn apply_request_telemetry_if(router: Router, enabled: bool) -> Router {
    if enabled {
        router.layer(MetricsLayer::new())
    } else {
        router
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsMiddleware {
            inner,
            #[cfg(test)]
            observations: self.observations.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MetricsMiddleware<S> {
    inner: S,
    #[cfg(test)]
    observations: Option<std::sync::Arc<std::sync::Mutex<Vec<RequestObservation>>>>,
}

impl<S> Service<Request<Body>> for MetricsMiddleware<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<Body>) -> Self::Future {
        // Fast path: classify the URI on a borrowed `&str` — no heap
        // allocation — and exit early for non-inference paths. This
        // matters for `/healthz` / `/readyz` probes that kubelet fires
        // every few seconds; we don't want this layer showing up in
        // their handling cost.
        let endpoint = classify_endpoint(req.uri().path());

        let mut inner = self.inner.clone();
        let Some(endpoint) = endpoint else {
            return Box::pin(async move { inner.call(req).await });
        };

        // Install an empty label slot that the handler will fill in
        // once routing normalization (pool split, alias resolution,
        // `configured_gpus` validation) has picked a canonical
        // `machine_profile`. Cloning an `Arc` is cheap and lets us
        // read the slot back after `inner.call(req)` has consumed
        // `req`. See module docs for why the raw header is not used.
        let slot = telemetry::MetricLabelsSlot::default();
        let admission_slot = telemetry::AdmissionOutcomeSlot::default();
        let parent_cx =
            crate::observability::propagation::extract_context_from_headers(req.headers());
        let request_span = tracing::info_span!(
            "gateway.request",
            otel.name = "gateway.request",
            sie.operation = endpoint,
            http.response.status_code = tracing::field::Empty,
            sie.machine_profile = tracing::field::Empty,
            sie.duration_s = tracing::field::Empty,
        );
        let _ = request_span.set_parent(parent_cx);
        let request_cx = request_span.context();
        let request_cx_for_log = request_cx.clone();
        #[cfg(test)]
        let observations = self.observations.clone();
        req.extensions_mut().insert(slot.clone());
        req.extensions_mut().insert(admission_slot.clone());
        req.extensions_mut()
            .insert(telemetry::RequestTraceContext::new(request_cx.clone()));

        let start = Instant::now();
        Box::pin(
            async move {
                let response = inner.call(req).await?;

                let status = response.status().as_u16();
                let elapsed = start.elapsed().as_secs_f64();
                // Canonical profile from the handler, or `"other"` when
                // the request exited before normalization (e.g. `model is
                // required`). Empty strings also collapse to `"other"`
                // so dashboards never render a blank label row.
                let profile_label = slot
                    .get()
                    .map(|l| l.machine_profile.as_str())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("other");

                request_span.record("http.response.status_code", status);
                request_span.record("sie.machine_profile", profile_label);
                request_span.record("sie.duration_s", elapsed);

                let admission_outcome = admission_slot
                    .get()
                    .unwrap_or(telemetry::AdmissionOutcome::Admitted);
                telemetry::record_request_completed(
                    Some(request_cx_for_log.span().span_context()),
                    endpoint,
                    status,
                    profile_label,
                    elapsed,
                    admission_outcome,
                );
                #[cfg(test)]
                if let Some(observations) = observations {
                    observations.lock().unwrap().push(RequestObservation {
                        operation: endpoint,
                        status,
                        machine_profile: profile_label.to_string(),
                    });
                }

                // The span remains open through the log timestamp, but is never
                // entered as the active tracing span. Entering it here would turn
                // existing audit/proxy `tracing` events (some with raw request or
                // error fields) into exported OTel span events and broaden the
                // privacy contract beyond this bounded request spine.
                drop(request_span);
                Ok(response)
            }
            // Attach only the OTel context on every poll, including OpenAI
            // completions/responses handlers that do not create a local span.
            // Do not instrument/enter `request_span`; child spans parent via the
            // request extension and publisher propagation uses Context::current().
            .with_context(request_cx),
        )
    }
}

/// Return the endpoint label (`encode`, `score`, `extract`, `generate`,
/// `embeddings`, `moderations`) when the path matches an inference route,
/// otherwise `None`. Every generation surface
/// intentionally collapses to `generate`: the route spelling is bounded HTTP
/// metadata, while the operational work and service-level objective are shared.
/// The OpenAI audio transcription surface similarly collapses to its native
/// `extract` primitive.
/// Non-inference
/// paths are intentionally excluded — see module-level docs.
///
/// Works on a borrowed `&str` so the middleware fast path is
/// allocation-free for infrastructure traffic (`/healthz`,
/// `/ws/*`, `/v1/configs/*`, ...).
fn classify_endpoint(path: &str) -> Option<&'static str> {
    if path.starts_with("/v1/encode/") {
        Some("encode")
    } else if path.starts_with("/v1/score/") {
        Some("score")
    } else if path.starts_with("/v1/extract/") {
        Some("extract")
    } else if path.starts_with("/v1/generate/")
        || matches!(
            path,
            "/v1/chat/completions" | "/v1/completions" | "/v1/responses"
        )
    {
        Some("generate")
    } else if path == "/v1/embeddings" {
        Some("embeddings")
    } else if path == "/v1/moderations" {
        Some("moderations")
    } else if path == "/v1/audio/transcriptions" {
        Some("extract")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use std::hint::black_box;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    use tower::ServiceExt;

    // A test handler that mimics the real proxy handler: write the
    // canonical machine_profile into the slot that `MetricsLayer`
    // installed. Real callers always normalize first, so the tests
    // exercise that exact path.
    async fn set_profile(req: Request<Body>, profile: &'static str) -> axum::response::Response {
        if let Some(slot) = req.extensions().get::<telemetry::MetricLabelsSlot>() {
            slot.set(telemetry::MetricLabels {
                machine_profile: profile.to_string(),
            });
        }
        axum::response::IntoResponse::into_response((StatusCode::OK, "ok"))
    }

    fn test_router() -> (Router, Arc<Mutex<Vec<RequestObservation>>>) {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let router = Router::new()
            .route(
                "/v1/encode/{*model}",
                post(|req: Request<Body>| async move { set_profile(req, "l4-spot").await }),
            )
            .route(
                "/v1/score/{*model}",
                post(|req: Request<Body>| async move {
                    // Simulate an early exit before normalization: do
                    // not write the slot. Middleware must fall back
                    // to `"other"`.
                    let _ = req;
                    axum::response::IntoResponse::into_response((
                        StatusCode::SERVICE_UNAVAILABLE,
                        "nope",
                    ))
                }),
            )
            .route(
                "/v1/extract/{*model}",
                post(|req: Request<Body>| async move {
                    if let Some(slot) = req.extensions().get::<telemetry::MetricLabelsSlot>() {
                        slot.set(telemetry::MetricLabels {
                            machine_profile: "a100".to_string(),
                        });
                    }
                    axum::response::IntoResponse::into_response((
                        StatusCode::GATEWAY_TIMEOUT,
                        "timeout",
                    ))
                }),
            )
            .route(
                "/v1/embeddings",
                post(|req: Request<Body>| async move { set_profile(req, "l4-spot").await }),
            )
            .route(
                "/v1/chat/completions",
                post(|req: Request<Body>| async move { set_profile(req, "a100").await }),
            )
            .route(
                "/v1/completions",
                post(|req: Request<Body>| async move { set_profile(req, "a100").await }),
            )
            .route(
                "/v1/responses",
                post(|req: Request<Body>| async move { set_profile(req, "a100").await }),
            )
            .route(
                "/v1/moderations",
                post(|| async {
                    axum::response::IntoResponse::into_response((
                        StatusCode::NOT_IMPLEMENTED,
                        "moderations unavailable",
                    ))
                }),
            )
            .route(
                "/v1/audio/transcriptions",
                post(|req: Request<Body>| async move { set_profile(req, "l4-spot").await }),
            )
            .route("/health", axum::routing::get(|| async { "health" }))
            .layer(MetricsLayer::observing(Arc::clone(&observations)));
        (router, observations)
    }

    async fn fire(router: Router, method: Method, uri: &str, profile: Option<&str>) -> StatusCode {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(p) = profile {
            builder = builder.header("x-sie-machine-profile", p);
        }
        let req = builder.body(Body::empty()).unwrap();
        router.oneshot(req).await.unwrap().status()
    }

    fn extension_probe_router(observed: Arc<Mutex<Option<(bool, bool, bool)>>>) -> Router {
        Router::new().route(
            "/v1/encode/{*model}",
            post(move |req: Request<Body>| {
                let observed = Arc::clone(&observed);
                async move {
                    *observed.lock().unwrap() = Some((
                        req.extensions()
                            .get::<telemetry::MetricLabelsSlot>()
                            .is_some(),
                        req.extensions()
                            .get::<telemetry::AdmissionOutcomeSlot>()
                            .is_some(),
                        req.extensions()
                            .get::<telemetry::RequestTraceContext>()
                            .is_some(),
                    ));
                    StatusCode::OK
                }
            }),
        )
    }

    #[tokio::test]
    async fn disabled_router_omits_request_telemetry_middleware() {
        let observed = Arc::new(Mutex::new(None));
        let router =
            apply_request_telemetry_if(extension_probe_router(Arc::clone(&observed)), false);

        assert_eq!(
            fire(router, Method::POST, "/v1/encode/org/model", None).await,
            StatusCode::OK
        );
        assert_eq!(*observed.lock().unwrap(), Some((false, false, false)));
    }

    #[tokio::test]
    async fn enabled_router_installs_complete_request_telemetry_context() {
        let observed = Arc::new(Mutex::new(None));
        let router =
            apply_request_telemetry_if(extension_probe_router(Arc::clone(&observed)), true);

        assert_eq!(
            fire(router, Method::POST, "/v1/encode/org/model", None).await,
            StatusCode::OK
        );
        assert_eq!(*observed.lock().unwrap(), Some((true, true, true)));
    }

    /// Reproducible full-Tower request-path benchmark. Unlike the facade-only
    /// microbenchmark, this includes route matching, middleware extension
    /// slots, context extraction, span/context creation, the clock and terminal
    /// semantic event. One invocation collects three independently warmed
    /// samples and asserts their median:
    /// `cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib full_tower_request_telemetry_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[tokio::test(flavor = "current_thread")]
    #[ignore = "release microbenchmark; run explicitly with --release --ignored --nocapture"]
    async fn full_tower_request_telemetry_microbenchmark() {
        const SAMPLES: usize = 3;
        const WARMUP: usize = 1_000;
        const ITERATIONS: usize = 8_000;

        let base = Router::new().route(
            "/v1/encode/{*model}",
            post(|req: Request<Body>| async move {
                if let Some(slot) = req.extensions().get::<telemetry::MetricLabelsSlot>() {
                    slot.set(telemetry::MetricLabels {
                        machine_profile: "l4-spot".to_string(),
                    });
                }
                StatusCode::OK
            }),
        );
        let disabled = apply_request_telemetry_if(base.clone(), false);
        let enabled = apply_request_telemetry_if(base, true);

        async fn exercise(router: &Router, iterations: usize) {
            for _ in 0..iterations {
                let response = router
                    .clone()
                    .oneshot(
                        Request::builder()
                            .method(Method::POST)
                            .uri(black_box("/v1/encode/org/model"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
            }
        }

        let mut disabled_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        let mut overhead_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise(&disabled, WARMUP).await;
            exercise(&enabled, WARMUP).await;

            let disabled_started = Instant::now();
            exercise(&disabled, ITERATIONS).await;
            disabled_samples[sample_index] =
                disabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;

            let enabled_started = Instant::now();
            exercise(&enabled, ITERATIONS).await;
            enabled_samples[sample_index] =
                enabled_started.elapsed().as_nanos() as f64 / ITERATIONS as f64;
            overhead_samples[sample_index] =
                (enabled_samples[sample_index] - disabled_samples[sample_index]).max(0.0);
        }

        let disabled_median_ns = telemetry::telemetry_benchmark_median(disabled_samples);
        let enabled_median_ns = telemetry::telemetry_benchmark_median(enabled_samples);
        let overhead_median_ns = telemetry::telemetry_benchmark_median(overhead_samples);
        println!(
            "gateway_full_tower_request iterations_per_sample={ITERATIONS} samples={SAMPLES} disabled_ns_per_request={disabled_samples:?} disabled_median_ns_per_request={disabled_median_ns:.2} enabled_ns_per_request={enabled_samples:?} enabled_median_ns_per_request={enabled_median_ns:.2} overhead_ns_per_request={overhead_samples:?} overhead_median_ns_per_request={overhead_median_ns:.2}"
        );
        let disabled_budget =
            telemetry::telemetry_performance_budget("gateway_full_tower_disabled_ns_per_request");
        assert!(
            disabled_median_ns <= disabled_budget,
            "gateway telemetry-disabled Tower median {disabled_median_ns:.2} ns exceeded {disabled_budget:.2} ns budget"
        );
        let budget =
            telemetry::telemetry_performance_budget("gateway_full_tower_overhead_ns_per_request");
        assert!(
            overhead_median_ns <= budget,
            "gateway full-Tower telemetry overhead median {overhead_median_ns:.2} ns exceeded {budget:.2} ns budget"
        );
    }

    #[tokio::test]
    async fn records_200_on_encode_from_handler_slot() {
        let (router, observations) = test_router();

        // Header carries a noisy pool-prefixed value; the handler sets
        // the slot to the normalized form. The middleware must pick
        // the slot value, not the header.
        let status = fire(
            router,
            Method::POST,
            "/v1/encode/org/model",
            Some("eval-l4/l4"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "encode",
                status: 200,
                machine_profile: "l4-spot".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn falls_back_to_other_when_handler_does_not_set_slot() {
        let (router, observations) = test_router();

        // Early-exit path: handler returns before writing the slot.
        // Even with a non-empty header we must not leak the raw
        // client-controlled value into the label.
        let status = fire(
            router,
            Method::POST,
            "/v1/score/x/y",
            Some("definitely-not-a-gpu"),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "score",
                status: 503,
                machine_profile: "other".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn records_504_on_extract_from_handler_slot() {
        let (router, observations) = test_router();

        let status = fire(router, Method::POST, "/v1/extract/x/y", None).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "extract",
                status: 504,
                machine_profile: "a100".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn records_200_on_openai_embeddings_from_handler_slot() {
        let (router, observations) = test_router();

        let status = fire(router, Method::POST, "/v1/embeddings", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "embeddings",
                status: 200,
                machine_profile: "l4-spot".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn records_all_openai_generation_routes_from_handler_slot() {
        let (router, observations) = test_router();

        for uri in ["/v1/chat/completions", "/v1/completions", "/v1/responses"] {
            assert_eq!(
                fire(router.clone(), Method::POST, uri, None).await,
                StatusCode::OK
            );
        }

        assert_eq!(observations.lock().unwrap().len(), 3);
        assert!(observations.lock().unwrap().iter().all(|observation| {
            observation
                == &RequestObservation {
                    operation: "generate",
                    status: 200,
                    machine_profile: "a100".to_string(),
                }
        }));
    }

    #[tokio::test]
    async fn records_openai_audio_from_handler_slot() {
        let (router, observations) = test_router();

        let status = fire(router, Method::POST, "/v1/audio/transcriptions", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "extract",
                status: 200,
                machine_profile: "l4-spot".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn records_registered_moderations_route() {
        let (router, observations) = test_router();

        let status = fire(router, Method::POST, "/v1/moderations", None).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            *observations.lock().unwrap(),
            vec![RequestObservation {
                operation: "moderations",
                status: 501,
                machine_profile: "other".to_string(),
            }]
        );
    }

    #[tokio::test]
    async fn skips_infrastructure_paths() {
        let (router, observations) = test_router();

        let h = fire(router.clone(), Method::GET, "/health", None).await;
        assert_eq!(h, StatusCode::OK);
        let unknown = fire(router, Method::GET, "/unknown", None).await;
        assert_eq!(unknown, StatusCode::NOT_FOUND);

        assert!(observations.lock().unwrap().is_empty());
    }

    #[test]
    fn classify_endpoint_is_exhaustive() {
        assert_eq!(classify_endpoint("/v1/encode/org/model"), Some("encode"));
        assert_eq!(classify_endpoint("/v1/score/org/model"), Some("score"));
        assert_eq!(classify_endpoint("/v1/extract/org/model"), Some("extract"));
        assert_eq!(
            classify_endpoint("/v1/generate/org/model"),
            Some("generate")
        );
        assert_eq!(classify_endpoint("/v1/chat/completions"), Some("generate"));
        assert_eq!(classify_endpoint("/v1/completions"), Some("generate"));
        assert_eq!(classify_endpoint("/v1/responses"), Some("generate"));
        assert_eq!(classify_endpoint("/v1/chat/completions/extra"), None);
        assert_eq!(classify_endpoint("/v1/embeddings"), Some("embeddings"));
        assert_eq!(classify_endpoint("/v1/moderations"), Some("moderations"));
        assert_eq!(classify_endpoint("/v1/moderations/extra"), None);
        assert_eq!(
            classify_endpoint("/v1/audio/transcriptions"),
            Some("extract")
        );
        assert_eq!(classify_endpoint("/health"), None);
        assert_eq!(classify_endpoint("/healthz"), None);
        assert_eq!(classify_endpoint("/readyz"), None);
        assert_eq!(classify_endpoint("/v1/configs/models"), None);
        assert_eq!(classify_endpoint("/v1/pools"), None);
        assert_eq!(classify_endpoint("/v1/models"), None);
        assert_eq!(classify_endpoint("/v1/models/BAAI/bge-m3"), None);
        assert_eq!(classify_endpoint("/ws/cluster-status"), None);
        assert_eq!(classify_endpoint("/"), None);
    }
}
