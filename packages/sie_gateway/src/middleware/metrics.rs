//! Prometheus HTTP request/latency middleware.
//!
//! Wraps the whole router so every response is observed — including
//! 4xx/5xx early returns, timeouts, axum-generated 500s for panics, and
//! any future exit path that forgets to instrument itself by hand.
//! The terminal increment that used to live inside
//! `handlers::proxy::queue_mode_proxy` has been removed in favour of
//! this layer.
//!
//! Only the three inference endpoints (`/v1/encode/*`, `/v1/score/*`,
//! `/v1/extract/*`) populate `sie_gateway_requests_total` +
//! `sie_gateway_request_latency_seconds`. Infrastructure paths
//! (`/health*`, `/metrics`, `/ws/*`, `/v1/configs/*`, `/v1/pools`,
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
//! back to `"unknown"`.
//!
//! The `reason`-labelled rejection counter
//! (`sie_gateway_rejected_requests_total`) stays in the handler
//! because its labels (`gpu_not_configured`, `no_consumers`,
//! `backpressure`, ...) are only knowable at the point of rejection;
//! the HTTP status code this layer sees is coarser.
//!
//! ## Hot-path cost
//!
//! The middleware is designed so non-inference requests (probes,
//! `/metrics` scrapes, config plane calls) pay **zero heap allocations
//! and no clock reads** from this layer — the classifier runs on a
//! borrowed `&str` and exits early when the path isn't one of the
//! three inference prefixes.
//!
//! For inference requests the added overhead is bounded:
//! - One `Instant::now()` / `elapsed()` pair (vDSO, ~20 ns).
//! - One `Arc` clone + one `req.extensions_mut().insert()` to install
//!   the label slot (~40 ns; no heap alloc per request — the `Arc`
//!   allocation is one inline pointer bump on [`MetricLabelsSlot`]'s
//!   `Default::default()`).
//! - One `Box::pin` future allocation (unavoidable Tower pattern;
//!   matches what `AuditLayer` and `AuthLayer` already do).
//! - Two Prometheus `with_label_values` lookups (~50 ns each: FNV
//!   hash + uncontended parking_lot read lock on the `HashMap` of
//!   children).
//! - No `String` allocation for the HTTP status code — it is mapped
//!   to a bounded set of `&'static str`s via `status_label`.
//!
//! Total: ~100–200 ns per inference request, against a 10–100 ms GPU
//! round-trip. The middleware is not on the critical path for
//! latency.

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tower::{Layer, Service};

use crate::metrics;

#[derive(Clone, Default)]
pub struct MetricsLayer;

impl MetricsLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for MetricsLayer {
    type Service = MetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct MetricsMiddleware<S> {
    inner: S,
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
        // every few seconds and for scrape traffic on `/metrics`; we
        // don't want this layer showing up in their handling cost.
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
        let slot = metrics::MetricLabelsSlot::default();
        req.extensions_mut().insert(slot.clone());

        let start = Instant::now();
        Box::pin(async move {
            let response = inner.call(req).await?;

            let status = response.status().as_u16();
            let elapsed = start.elapsed().as_secs_f64();
            // Map the status code to a `&'static str` so we don't
            // allocate a fresh `String` per request just to use it as
            // a Prometheus label. Covers every status code axum /
            // our handlers actually produce; anything outside the set
            // rolls up into a coarse `"xxx"` bucket.
            let status_label = status_label(status);
            // Canonical profile from the handler, or `"unknown"` when
            // the request exited before normalization (e.g. `model is
            // required`). Empty strings also collapse to `"unknown"`
            // so dashboards never render a blank label row.
            let profile_label = slot
                .get()
                .map(|l| l.machine_profile.as_str())
                .filter(|s| !s.is_empty())
                .unwrap_or("unknown");

            metrics::REQUEST_COUNT
                .with_label_values(&[endpoint, status_label, profile_label])
                .inc();
            metrics::REQUEST_LATENCY
                .with_label_values(&[endpoint, profile_label])
                .observe(elapsed);

            Ok(response)
        })
    }
}

/// Return the endpoint label (`encode`, `score`, `extract`) when the
/// path matches an inference route, otherwise `None`. Non-inference
/// paths are intentionally excluded — see module-level docs.
///
/// Works on a borrowed `&str` so the middleware fast path is
/// allocation-free for infrastructure traffic (`/healthz`, `/metrics`,
/// `/ws/*`, `/v1/configs/*`, ...).
fn classify_endpoint(path: &str) -> Option<&'static str> {
    if path.starts_with("/v1/encode/") {
        Some("encode")
    } else if path.starts_with("/v1/score/") {
        Some("score")
    } else if path.starts_with("/v1/extract/") {
        Some("extract")
    } else {
        None
    }
}

/// Map an HTTP status code to a Prometheus-friendly static label.
/// Avoids the per-request `u16::to_string()` allocation that would
/// otherwise happen on every inference response. Covers every
/// status code the gateway's handlers can return plus the standard
/// axum-generated ones (404, 405, 500); anything outside that set
/// rolls up into `"xxx"`, which is visible on dashboards as an
/// "unexpected status" bucket and still keeps cardinality bounded.
fn status_label(status: u16) -> &'static str {
    match status {
        200 => "200",
        201 => "201",
        202 => "202",
        204 => "204",
        400 => "400",
        401 => "401",
        403 => "403",
        404 => "404",
        405 => "405",
        408 => "408",
        409 => "409",
        413 => "413",
        422 => "422",
        429 => "429",
        500 => "500",
        501 => "501",
        502 => "502",
        503 => "503",
        504 => "504",
        _ => "xxx",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Method, StatusCode};
    use axum::routing::post;
    use axum::Router;
    use std::sync::Mutex;
    use tower::ServiceExt;

    // Serialize tests that observe global counter state. All tests in
    // this module bump `sie_gateway_requests_total`, and Cargo runs
    // unit tests in parallel by default — without this lock the
    // "infrastructure paths don't bump counters" assertion races with
    // the other tests and flakes.
    //
    // Held across `.await` deliberately: we need the lock to remain
    // held while the request future resolves so a concurrent test
    // cannot bump the counter between our baseline snapshot and the
    // post-call assertion. There is no cross-task contention — the
    // lock only serializes tests in this module — so the normal
    // deadlock risk that `clippy::await_holding_lock` guards against
    // does not apply. `#[allow]` is per-test.
    static COUNTER_TEST_LOCK: Mutex<()> = Mutex::new(());

    // A test handler that mimics the real proxy handler: write the
    // canonical machine_profile into the slot that `MetricsLayer`
    // installed. Real callers always normalize first, so the tests
    // exercise that exact path.
    async fn set_profile(req: Request<Body>, profile: &'static str) -> axum::response::Response {
        if let Some(slot) = req.extensions().get::<metrics::MetricLabelsSlot>() {
            slot.set(metrics::MetricLabels {
                machine_profile: profile.to_string(),
            });
        }
        axum::response::IntoResponse::into_response((StatusCode::OK, "ok"))
    }

    fn test_router() -> Router {
        Router::new()
            .route(
                "/v1/encode/{*model}",
                post(|req: Request<Body>| async move { set_profile(req, "l4-spot").await }),
            )
            .route(
                "/v1/score/{*model}",
                post(|req: Request<Body>| async move {
                    // Simulate an early exit before normalization: do
                    // not write the slot. Middleware must fall back
                    // to `"unknown"`.
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
                    if let Some(slot) = req.extensions().get::<metrics::MetricLabelsSlot>() {
                        slot.set(metrics::MetricLabels {
                            machine_profile: "a100".to_string(),
                        });
                    }
                    axum::response::IntoResponse::into_response((
                        StatusCode::GATEWAY_TIMEOUT,
                        "timeout",
                    ))
                }),
            )
            .route("/health", axum::routing::get(|| async { "health" }))
            .route("/metrics", axum::routing::get(|| async { "metrics" }))
            .layer(MetricsLayer::new())
    }

    async fn fire(router: Router, method: Method, uri: &str, profile: Option<&str>) -> StatusCode {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(p) = profile {
            builder = builder.header("x-sie-machine-profile", p);
        }
        let req = builder.body(Body::empty()).unwrap();
        router.oneshot(req).await.unwrap().status()
    }

    fn counter_value(endpoint: &str, status: &str, machine_profile: &str) -> f64 {
        metrics::REQUEST_COUNT
            .with_label_values(&[endpoint, status, machine_profile])
            .get()
    }

    fn latency_count(endpoint: &str, machine_profile: &str) -> u64 {
        // HistogramVec has no public count accessor; gather the metric
        // family and look up the matching series.
        let families = metrics::REGISTRY.gather();
        for mf in &families {
            if mf.get_name() != "sie_gateway_request_latency_seconds" {
                continue;
            }
            for m in mf.get_metric() {
                let labels: std::collections::HashMap<&str, &str> = m
                    .get_label()
                    .iter()
                    .map(|l| (l.get_name(), l.get_value()))
                    .collect();
                if labels.get("endpoint").copied() == Some(endpoint)
                    && labels.get("machine_profile").copied() == Some(machine_profile)
                {
                    return m.get_histogram().get_sample_count();
                }
            }
        }
        0
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // see COUNTER_TEST_LOCK doc
    async fn records_200_on_encode_from_handler_slot() {
        let _guard = COUNTER_TEST_LOCK.lock().unwrap();
        let _ = &*metrics::REGISTRY;
        let before = counter_value("encode", "200", "l4-spot");
        let before_lat = latency_count("encode", "l4-spot");

        // Header carries a noisy pool-prefixed value; the handler sets
        // the slot to the normalized form. The middleware must pick
        // the slot value, not the header.
        let status = fire(
            test_router(),
            Method::POST,
            "/v1/encode/org/model",
            Some("eval-l4/l4"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        assert_eq!(counter_value("encode", "200", "l4-spot") - before, 1.0);
        assert_eq!(latency_count("encode", "l4-spot") - before_lat, 1);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // see COUNTER_TEST_LOCK doc
    async fn falls_back_to_unknown_when_handler_does_not_set_slot() {
        let _guard = COUNTER_TEST_LOCK.lock().unwrap();
        let _ = &*metrics::REGISTRY;
        let before = counter_value("score", "503", "unknown");

        // Early-exit path: handler returns before writing the slot.
        // Even with a non-empty header we must not leak the raw
        // client-controlled value into the label.
        let status = fire(
            test_router(),
            Method::POST,
            "/v1/score/x/y",
            Some("definitely-not-a-gpu"),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);

        assert_eq!(counter_value("score", "503", "unknown") - before, 1.0);
        // And the raw header value must not have created a time series.
        assert_eq!(counter_value("score", "503", "definitely-not-a-gpu"), 0.0);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // see COUNTER_TEST_LOCK doc
    async fn records_504_on_extract_from_handler_slot() {
        let _guard = COUNTER_TEST_LOCK.lock().unwrap();
        let _ = &*metrics::REGISTRY;
        let before = counter_value("extract", "504", "a100");

        let status = fire(test_router(), Method::POST, "/v1/extract/x/y", None).await;
        assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);

        assert_eq!(counter_value("extract", "504", "a100") - before, 1.0);
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // see COUNTER_TEST_LOCK doc
    async fn skips_infrastructure_paths() {
        let _guard = COUNTER_TEST_LOCK.lock().unwrap();
        let _ = &*metrics::REGISTRY;
        let router = test_router();

        // Baseline: sum every series on both metrics.
        let baseline_count: f64 = {
            let families = metrics::REGISTRY.gather();
            families
                .iter()
                .filter(|mf| mf.get_name() == "sie_gateway_requests_total")
                .flat_map(|mf| mf.get_metric().iter())
                .map(|m| m.get_counter().get_value())
                .sum()
        };

        let h = fire(router.clone(), Method::GET, "/health", None).await;
        assert_eq!(h, StatusCode::OK);
        let m = fire(router, Method::GET, "/metrics", None).await;
        assert_eq!(m, StatusCode::OK);

        let after_count: f64 = {
            let families = metrics::REGISTRY.gather();
            families
                .iter()
                .filter(|mf| mf.get_name() == "sie_gateway_requests_total")
                .flat_map(|mf| mf.get_metric().iter())
                .map(|m| m.get_counter().get_value())
                .sum()
        };

        assert!(
            (after_count - baseline_count).abs() < f64::EPSILON,
            "infrastructure paths (/health, /metrics) must not bump request counters"
        );
    }

    #[test]
    fn classify_endpoint_is_exhaustive() {
        assert_eq!(classify_endpoint("/v1/encode/org/model"), Some("encode"));
        assert_eq!(classify_endpoint("/v1/score/org/model"), Some("score"));
        assert_eq!(classify_endpoint("/v1/extract/org/model"), Some("extract"));
        assert_eq!(classify_endpoint("/health"), None);
        assert_eq!(classify_endpoint("/healthz"), None);
        assert_eq!(classify_endpoint("/readyz"), None);
        assert_eq!(classify_endpoint("/metrics"), None);
        assert_eq!(classify_endpoint("/v1/configs/models"), None);
        assert_eq!(classify_endpoint("/v1/pools"), None);
        assert_eq!(classify_endpoint("/v1/models"), None);
        assert_eq!(classify_endpoint("/v1/models/BAAI/bge-m3"), None);
        assert_eq!(classify_endpoint("/ws/cluster-status"), None);
        assert_eq!(classify_endpoint("/"), None);
    }

    #[test]
    fn status_label_covers_expected_codes() {
        // Every status code the gateway or axum actually produces
        // must map to its own static string (not "xxx") so dashboards
        // can distinguish them without a per-request allocation.
        for code in [
            200, 201, 202, 204, 400, 401, 403, 404, 405, 408, 409, 413, 422, 429, 500, 501, 502,
            503, 504,
        ] {
            assert_eq!(status_label(code), code.to_string().as_str());
        }
        // Anything outside the set rolls up into "xxx" to keep
        // cardinality bounded.
        assert_eq!(status_label(418), "xxx");
        assert_eq!(status_label(599), "xxx");
        assert_eq!(status_label(0), "xxx");
    }
}
