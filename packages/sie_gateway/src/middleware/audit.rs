//! Content-free per-request audit record (`event="api_request"`).
//!
//! # Placement contract
//!
//! The emitted `status` is presented as the status the customer received, so
//! this layer MUST be installed outside every layer that can replace the
//! response it observes. In the OSS composition nothing rewrites a handler
//! response, so [`crate::server::AuditPlacement::Core`] keeps the layer inside
//! the route stack. The managed composition wraps the same routes in billing
//! admission middleware that replaces an inner dispatch-reservation 503 with
//! the customer-facing 402 (`slab_ledger::reservation_failure_response`), so it
//! selects [`crate::server::AuditPlacement::Composition`] and installs this
//! layer outside those gates instead — otherwise the audit record reports the
//! provisional inner status and disagrees with what the client received.

use axum::body::Body;
use axum::http::Request;
use axum::response::Response;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Instant;
use tower::{Layer, Service};
use tracing::info;

use super::auth::mask_token;
use crate::observability::metrics::AdmissionOutcomeSlot;

#[derive(Clone)]
pub struct AuditLayer;

impl AuditLayer {
    // `new_without_default` only fires on exported items, so this lint
    // surfaced when the lib target (src/lib.rs) made the type public API.
    // Kept as an `allow` rather than a `Default` impl to leave the OSS
    // surface unchanged.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for AuditLayer {
    type Service = AuditMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AuditMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct AuditMiddleware<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for AuditMiddleware<S>
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

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let start = Instant::now();
        let method = req.method().to_string();
        let path = req.uri().path().to_string();

        let token_id = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(|h| {
                let token = if h.to_lowercase().starts_with("bearer ") {
                    h[7..].trim()
                } else {
                    h.trim()
                };
                mask_token(token)
            })
            .unwrap_or_default();

        let content_length = req
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<i64>().ok())
            .unwrap_or(0);

        // Extract routing hints from headers
        let model = req
            .headers()
            .get("x-sie-model")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let pool = req
            .headers()
            .get("x-sie-pool")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let gpu = req
            .headers()
            .get("x-sie-machine-profile")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        // Bounded admission verdict, installed by the outer request-telemetry
        // layer and set by whichever policy gate rejected the request. Cloned
        // out of the extensions here because `inner.call(req)` consumes the
        // request; the slot itself is an `Arc`, so this reads the same cell the
        // gates write. Absent whenever that layer is not installed (telemetry
        // disabled) or the path is not an inference route.
        let admission_slot = req.extensions().get::<AdmissionOutcomeSlot>().cloned();

        let mut inner = self.inner.clone();

        Box::pin(async move {
            let response = inner.call(req).await?;

            let elapsed = start.elapsed();
            let status = response.status().as_u16();
            // Never a raw reason string: a bounded enum label, or empty when no
            // gate recorded a verdict.
            let admission = admission_slot
                .and_then(|slot| slot.get())
                .map(|outcome| outcome.as_str())
                .unwrap_or("");

            let worker = response
                .headers()
                .get("x-sie-worker")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            // Only audit non-health endpoints to reduce noise
            if !is_infrastructure_path(&path) {
                info!(
                    event = "api_request",
                    method = %method,
                    endpoint = %path,
                    status = status,
                    admission = %admission,
                    token_id = %token_id,
                    model = %model,
                    pool = %pool,
                    gpu = %gpu,
                    worker = %worker,
                    latency_ms = elapsed.as_millis() as u64,
                    body_bytes = content_length,
                    "audit"
                );
            }

            Ok(response)
        })
    }
}

fn is_infrastructure_path(path: &str) -> bool {
    // Audit-specific extra: `/health` (the rich legacy status page) is
    // infrastructure noise for audit purposes even though it is NOT an
    // auth-exempt probe (see `auth::EXEMPT_OPERATIONAL_PATHS`).
    path == "/health" || super::auth::PROBE_PATHS.contains(&path)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::Router;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    use crate::observability::metrics::AdmissionOutcome;

    #[test]
    fn test_infrastructure_paths() {
        assert!(is_infrastructure_path("/health"));
        assert!(is_infrastructure_path("/healthz"));
        assert!(is_infrastructure_path("/readyz"));
        assert!(!is_infrastructure_path("/v1/encode/model"));
    }

    /// JSON-line sink for the emitted `tracing` events, so the assertions read
    /// the record an operator actually receives rather than a stand-in.
    #[derive(Clone, Default)]
    struct CapturedEvents(Arc<Mutex<Vec<u8>>>);

    impl CapturedEvents {
        /// The single `event="api_request"` record's fields, or `None` when the
        /// layer emitted nothing.
        fn api_request(&self) -> Option<serde_json::Value> {
            let buffer = self.0.lock().expect("captured events");
            let text = String::from_utf8(buffer.clone()).expect("utf8 log lines");
            let mut records: Vec<serde_json::Value> = text
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .map(|record| record["fields"].clone())
                .filter(|fields| fields["event"] == "api_request")
                .collect();
            assert!(records.len() <= 1, "one audit record per request");
            records.pop()
        }
    }

    impl Write for CapturedEvents {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("captured events")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturedEvents {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Drive one request through `router` with every event captured. The
    /// subscriber is attached to the future (not `with_default`) so events
    /// emitted after an await point are still recorded.
    async fn audit_one_request(
        router: Router,
        slot: Option<AdmissionOutcomeSlot>,
    ) -> (StatusCode, Option<serde_json::Value>) {
        let events = CapturedEvents::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(events.clone())
            .finish();

        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .body(Body::empty())
            .expect("request");
        if let Some(slot) = slot {
            request.extensions_mut().insert(slot);
        }

        let response = router
            .oneshot(request)
            .with_subscriber(subscriber)
            .await
            .expect("response");
        (response.status(), events.api_request())
    }

    /// The placement contract in miniature: an admission layer that replaces
    /// the handler response with the customer-facing one is exactly what the
    /// managed slab wrapper does (#3442).
    async fn rewrite_503_to_402(
        State(slot): State<AdmissionOutcomeSlot>,
        req: Request<Body>,
        next: axum::middleware::Next,
    ) -> Response {
        let inner = next.run(req).await;
        if inner.status() == StatusCode::SERVICE_UNAVAILABLE {
            slot.set(AdmissionOutcome::KeySpendLimitExceeded);
            return StatusCode::PAYMENT_REQUIRED.into_response();
        }
        inner
    }

    fn provisional_503_router() -> Router {
        Router::new().route(
            "/v1/embeddings",
            post(|| async { StatusCode::SERVICE_UNAVAILABLE }),
        )
    }

    #[tokio::test]
    async fn audit_inside_a_rewriting_layer_records_the_provisional_status() {
        let slot = AdmissionOutcomeSlot::default();
        let router = provisional_503_router().layer(AuditLayer::new()).layer(
            axum::middleware::from_fn_with_state(slot.clone(), rewrite_503_to_402),
        );

        let (status, record) = audit_one_request(router, Some(slot)).await;
        let record = record.expect("audit record");

        // Documents the defect this placement causes, so a future move back
        // inside a rewriting layer fails loudly instead of silently lying.
        assert_eq!(status, StatusCode::PAYMENT_REQUIRED);
        assert_eq!(record["status"], 503);
    }

    #[tokio::test]
    async fn audit_outside_a_rewriting_layer_records_the_final_status() {
        let slot = AdmissionOutcomeSlot::default();
        let router = provisional_503_router()
            .layer(axum::middleware::from_fn_with_state(
                slot.clone(),
                rewrite_503_to_402,
            ))
            .layer(AuditLayer::new());

        let (status, record) = audit_one_request(router, Some(slot)).await;
        let record = record.expect("audit record");

        assert_eq!(status, StatusCode::PAYMENT_REQUIRED);
        assert_eq!(record["status"], 402);
        // Bounded label, never a free-form reason string.
        assert_eq!(record["admission"], "key_spend_limit_exceeded");
    }

    #[tokio::test]
    async fn audit_admission_is_empty_without_a_recorded_verdict() {
        let router = Router::new()
            .route("/v1/embeddings", post(|| async { StatusCode::OK }))
            .layer(AuditLayer::new());

        let (status, record) = audit_one_request(router, None).await;
        let record = record.expect("audit record");

        assert_eq!(status, StatusCode::OK);
        assert_eq!(record["status"], 200);
        assert_eq!(record["admission"], "");
    }
}
