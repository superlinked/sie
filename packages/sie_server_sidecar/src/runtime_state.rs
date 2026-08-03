//! Process-local runtime pressure state and HTTP probes.
//!
//! Application observations leave the process only through
//! [`SidecarTelemetry`]. The integer gauges below are deliberately not metric
//! instruments: the scheduler, adapter pool, and health publisher share them
//! to make admission and heartbeat decisions without creating a second export
//! path.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::observability::metrics::SidecarTelemetry;
use crate::readiness::Readiness;
use crate::shutdown::Shutdown;

/// A non-negative process-local value used by scheduling and health logic.
///
/// This type intentionally exposes only the small API the runtime needs. It is
/// not registered with OpenTelemetry or any other exporter.
#[derive(Default)]
pub struct RuntimeGauge(AtomicI64);

impl RuntimeGauge {
    pub fn get(&self) -> i64 {
        self.0.load(Ordering::Acquire)
    }

    pub fn set(&self, value: i64) {
        self.0.store(value.max(0), Ordering::Release);
    }

    pub fn inc(&self) {
        self.add(1);
    }

    pub fn dec(&self) {
        self.sub(1);
    }

    pub fn add(&self, value: i64) {
        if value <= 0 {
            return;
        }
        let _ = self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_add(value))
            });
    }

    pub fn sub(&self, value: i64) {
        if value <= 0 {
            return;
        }
        let _ = self
            .0
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                Some(current.saturating_sub(value).max(0))
            });
    }
}

/// State shared by sidecar runtime components.
///
/// Only the fields consumed by scheduling/admission/health are retained. All
/// exported observations are owned by `telemetry` and emitted once.
pub struct RuntimeState {
    pub telemetry: SidecarTelemetry,
    pub inflight_batches: RuntimeGauge,
    pub worker_gpu_slots_total: RuntimeGauge,
    pub worker_gpu_slots_ready: RuntimeGauge,
    pub worker_queue_depth: RuntimeGauge,
    pub worker_pending_cost: RuntimeGauge,
    pub worker_saturated: RuntimeGauge,
}

impl RuntimeState {
    pub fn new() -> Self {
        Self::with_telemetry(SidecarTelemetry::from_global())
    }

    fn with_telemetry(telemetry: SidecarTelemetry) -> Self {
        Self {
            telemetry,
            inflight_batches: RuntimeGauge::default(),
            worker_gpu_slots_total: RuntimeGauge::default(),
            worker_gpu_slots_ready: RuntimeGauge::default(),
            worker_queue_depth: RuntimeGauge::default(),
            worker_pending_cost: RuntimeGauge::default(),
            worker_saturated: RuntimeGauge::default(),
        }
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Serve sidecar liveness and readiness probes.
///
/// Telemetry export is push-only OTLP and therefore has no HTTP scrape route.
pub fn spawn_probe_server(
    port: u16,
    readiness: Arc<Readiness>,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<JoinHandle<()>> {
    let handle = tokio::spawn(async move {
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();
        let listener = match TcpListener::bind(addr).await {
            Ok(listener) => listener,
            Err(error) => {
                error!(error = %error, addr = %addr, "probe server: bind failed");
                return;
            }
        };
        info!(addr = %addr, "probe server listening");

        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => {
                    info!("probe server stopping (shutdown)");
                    return;
                }
                accepted = listener.accept() => {
                    match accepted {
                        Ok((stream, _peer)) => {
                            let readiness = Arc::clone(&readiness);
                            tokio::spawn(async move {
                                let service = service_fn(move |request| {
                                    handle_request(request, Arc::clone(&readiness))
                                });
                                if let Err(error) = hyper::server::conn::http1::Builder::new()
                                    .serve_connection(TokioIo::new(stream), service)
                                    .await
                                {
                                    warn!(error = %error, "probe connection failed");
                                }
                            });
                        }
                        Err(error) => warn!(error = %error, "probe accept failed"),
                    }
                }
            }
        }
    });

    Ok(handle)
}

async fn handle_request(
    request: Request<hyper::body::Incoming>,
    readiness: Arc<Readiness>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    match (request.method(), request.uri().path()) {
        (&Method::GET, "/healthz") => Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from_static(b"ok")))
            .expect("static health response")),
        (&Method::GET, "/readyz") => Ok(readyz_response(&readiness)),
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from_static(b"not found")))
            .expect("static not-found response")),
    }
}

fn readyz_status_and_body(readiness: &Readiness) -> (StatusCode, String) {
    let snapshot = readiness.snapshot();
    if snapshot.is_ready() {
        (StatusCode::OK, "ok".to_owned())
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("not ready: {}", snapshot.reason()),
        )
    }
}

fn readyz_response(readiness: &Readiness) -> Response<Full<Bytes>> {
    let (status, body) = readyz_status_and_body(readiness);
    Response::builder()
        .status(status)
        .header("Content-Type", "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from(body)))
        .expect("static readiness response")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_gauge_is_non_negative_and_saturating() {
        let value = RuntimeGauge::default();
        value.sub(1);
        assert_eq!(value.get(), 0);
        value.add(i64::MAX);
        value.inc();
        assert_eq!(value.get(), i64::MAX);
        value.set(-1);
        assert_eq!(value.get(), 0);
    }

    #[test]
    fn runtime_state_preserves_the_zero_state_disabled_facade() {
        let state = RuntimeState::with_telemetry(SidecarTelemetry::default());

        assert!(!state.telemetry.enabled_for_tests());
        state
            .telemetry
            .queue_enqueued("encode", "catalog/model", None);
        assert_eq!(
            state
                .telemetry
                .queue_depth_for_tests("encode", "catalog/model"),
            0
        );
    }
}
