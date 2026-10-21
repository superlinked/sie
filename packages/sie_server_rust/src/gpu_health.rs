//! Detects a wedged CUDA context and recovers by exiting the process.
//!
//! A sticky CUDA error (an illegal memory access, a device-side assert, a
//! launch failure) poisons the process's CUDA context: every later CUDA call
//! fails the same way, and the CUDA documentation offers one way back: "To
//! continue using CUDA, the process must be terminated and relaunched."
//!
//! [`GpuHealth`] runs a tiny kernel and a device-to-host copy on its own stream
//! of the worker's CUDA device. The first probe that fails latches the wedge:
//! the IPC server then refuses model work (the sidecar NAKs it for redelivery to
//! a healthy worker), `Ping`, `/readyz` and `/livez` report unhealthy, and the
//! process exits with [`WEDGED_EXIT_CODE`] after a short grace so its container
//! restarts with a fresh context. This mirrors `sie_server.core.gpu_health` in
//! the Python worker.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tracing::{error, warn};

/// Exit status of a worker that stopped because its CUDA context is wedged
/// (`EX_TEMPFAIL`: a fresh process can succeed).
pub const WEDGED_EXIT_CODE: i32 = 75;

/// A wedged context is sticky, so a short-lived cached answer is harmless.
/// Callers that just saw a failure ask for a fresh probe.
const PROBE_TTL: Duration = Duration::from_secs(2);

/// Time between latching a wedge and exiting, so responses being written (the
/// failed batch's answer, refusals of later work) reach the sidecar first.
const EXIT_GRACE: Duration = Duration::from_secs(1);

const OOM_INDICATORS: &[&str] = &["out of memory", "out_of_memory", "failed to allocate"];

/// Result of one probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeOutcome {
    Healthy,
    /// The device is full, which is not a wedge.
    OutOfMemory,
    /// The device cannot run a trivial kernel; carries the error.
    Wedged(String),
}

type ProbeFn = dyn Fn() -> ProbeOutcome + Send + Sync;

pub struct GpuHealth {
    /// `None` on a worker without a CUDA device: there is nothing to wedge.
    probe: Option<Arc<ProbeFn>>,
    exit_on_wedge: bool,
    wedged: AtomicBool,
    cached: Mutex<Option<(Instant, bool)>>,
    /// One probe at a time. A hung device holds this, so callers wait, which a
    /// liveness probe's timeout and the sidecar's IPC timeout turn into failures.
    probing: tokio::sync::Mutex<()>,
}

impl GpuHealth {
    /// Health of the CUDA device this worker runs models on, if any. A wedge
    /// exits the process: this is the serving configuration.
    pub fn for_configured_device() -> Self {
        let probe = crate::candle_embedding::candle_cuda_ordinal().map(cuda_probe);
        Self::new(probe, true)
    }

    /// A worker with nothing to probe.
    pub fn disabled() -> Self {
        Self::new(None, false)
    }

    /// Health driven by `probe`; for tests.
    pub fn with_probe(
        probe: impl Fn() -> ProbeOutcome + Send + Sync + 'static,
        exit_on_wedge: bool,
    ) -> Self {
        Self::new(Some(Arc::new(probe)), exit_on_wedge)
    }

    fn new(probe: Option<Arc<ProbeFn>>, exit_on_wedge: bool) -> Self {
        Self {
            probe,
            exit_on_wedge,
            wedged: AtomicBool::new(false),
            cached: Mutex::new(None),
            probing: tokio::sync::Mutex::new(()),
        }
    }

    /// Whether a probe has found the context wedged. Never touches the device.
    pub fn is_wedged(&self) -> bool {
        self.wedged.load(Ordering::Acquire)
    }

    /// Whether the device can run a kernel. `fresh` skips the cached answer, for
    /// a caller that has just seen a failure.
    pub async fn healthy(&self, fresh: bool) -> bool {
        if self.is_wedged() {
            return false;
        }
        let Some(probe) = self.probe.clone() else {
            return true;
        };
        let _probing = self.probing.lock().await;
        if self.is_wedged() {
            return false;
        }
        if !fresh {
            if let Some((at, healthy)) = *self.cached.lock().expect("GPU health cache poisoned") {
                if at.elapsed() < PROBE_TTL {
                    return healthy;
                }
            }
        }
        let outcome = tokio::task::spawn_blocking(move || probe())
            .await
            .unwrap_or_else(|join| ProbeOutcome::Wedged(format!("GPU probe panicked: {join}")));
        let healthy = match outcome {
            ProbeOutcome::Healthy => true,
            ProbeOutcome::OutOfMemory => {
                warn!("GPU health probe ran out of memory; treating as healthy (memory pressure, not a wedged context)");
                true
            }
            ProbeOutcome::Wedged(reason) => {
                self.latch(&reason);
                false
            }
        };
        *self.cached.lock().expect("GPU health cache poisoned") = Some((Instant::now(), healthy));
        healthy
    }

    fn latch(&self, reason: &str) {
        if self.wedged.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.exit_on_wedge {
            error!(
                error = %reason,
                exit_code = WEDGED_EXIT_CODE,
                grace_ms = EXIT_GRACE.as_millis() as u64,
                "CUDA context is wedged; refusing new work and exiting so the worker restarts with a fresh context"
            );
            std::thread::spawn(|| {
                std::thread::sleep(EXIT_GRACE);
                // No destructors run, so nothing touches the wedged context.
                std::process::exit(WEDGED_EXIT_CODE);
            });
        } else {
            error!(error = %reason, "CUDA context is wedged; refusing new work");
        }
    }
}

fn is_out_of_memory(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    OOM_INDICATORS
        .iter()
        .any(|indicator| message.contains(indicator))
}

fn classify(error: &candle::Error) -> ProbeOutcome {
    let message = error.to_string();
    if is_out_of_memory(&message) {
        ProbeOutcome::OutOfMemory
    } else {
        ProbeOutcome::Wedged(message)
    }
}

/// Probe CUDA device `ordinal` through a device handle with its own stream,
/// created on first use and reused, so a probe never waits behind inference
/// kernels queued on the model streams.
fn cuda_probe(ordinal: usize) -> Arc<ProbeFn> {
    let device: Mutex<Option<candle::Device>> = Mutex::new(None);
    Arc::new(move || {
        let mut device = device.lock().unwrap_or_else(|e| e.into_inner());
        if device.is_none() {
            match candle::Device::new_cuda_with_stream(ordinal) {
                Ok(created) => *device = Some(created),
                Err(error) => return classify(&error),
            }
        }
        let device = device.as_ref().expect("probe device was just created");
        match candle::Tensor::ones(1, candle::DType::F32, device)
            .and_then(|ones| ones.to_vec1::<f32>())
        {
            Ok(_) => ProbeOutcome::Healthy,
            Err(error) => classify(&error),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn counting(outcome: ProbeOutcome) -> (GpuHealth, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let probe_calls = Arc::clone(&calls);
        let health = GpuHealth::with_probe(
            move || {
                probe_calls.fetch_add(1, Ordering::SeqCst);
                outcome.clone()
            },
            false,
        );
        (health, calls)
    }

    #[tokio::test]
    async fn a_worker_without_a_cuda_device_is_healthy() {
        assert!(GpuHealth::disabled().healthy(true).await);
    }

    #[tokio::test]
    async fn a_healthy_answer_is_cached_until_a_fresh_one_is_asked_for() {
        let (health, calls) = counting(ProbeOutcome::Healthy);
        assert!(health.healthy(false).await);
        assert!(health.healthy(false).await);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(health.healthy(true).await);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_wedge_latches_and_is_never_probed_again() {
        let (health, calls) = counting(ProbeOutcome::Wedged("CUDA_ERROR_ILLEGAL_ADDRESS".into()));
        assert!(!health.is_wedged());
        assert!(!health.healthy(true).await);
        assert!(health.is_wedged());
        assert!(!health.healthy(true).await);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn out_of_memory_is_not_a_wedge() {
        let (health, _calls) = counting(ProbeOutcome::OutOfMemory);
        assert!(health.healthy(true).await);
        assert!(!health.is_wedged());
    }

    #[test]
    fn driver_out_of_memory_errors_are_recognized() {
        assert!(is_out_of_memory(
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"
        ));
        assert!(!is_out_of_memory(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
        ));
    }
}
