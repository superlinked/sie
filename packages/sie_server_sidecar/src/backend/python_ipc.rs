//! Default [`InferenceBackend`] — delegates to the Python `sie-server`
//! over the UDS+msgpack IPC that already exists.
//!
//! This is a thin adapter. The on-wire protocol and semantics are
//! defined in [`crate::ipc_client`] and `sie_server.ipc_server`; this
//! module just maps the IPC error types to [`BackendError`].

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::backend::{BackendError, InferenceBackend};
use crate::ipc_client::{IpcClient, IpcError};
use crate::ipc_types::{
    BatchOutcome, EnsureModelReadyResponse, ProcessEncodeBatchRequest, ProcessExtractBatchRequest,
    ProcessScoreBatchRequest, RunBatchRequest,
};

/// Routes every call through [`IpcClient`] to the Python sie-server process.
///
/// Python is the authoritative adapter registry, so this backend claims
/// every model (`supports() == true`). When composing multiple
/// backends via [`crate::backend::BackendRouter`], register this one
/// last so specialised backends get first refusal.
pub struct PythonIpcBackend {
    ipc: Arc<IpcClient>,
}

impl PythonIpcBackend {
    pub fn new(ipc: Arc<IpcClient>) -> Self {
        Self { ipc }
    }
}

/// Map an [`IpcError`] to a [`BackendError`].
///
/// * Transport-level errors (I/O, timeout, frame-too-large, decode) →
///   `Transient`: the dispatcher NAKs and JetStream redelivers. The
///   IPC client already retries once on transport errors internally,
///   so when we get here the glitch is persistent.
/// * Protocol-level errors (`Server`, `VersionMismatch`, `Encode`) →
///   `Transient` as well. These usually indicate a mismatched Python
///   build or a logical executor error; NAK-and-redeliver lets another
///   pod take the work while we recover.
///
/// Note: the IPC contract does not surface `UnsupportedModel` through
/// this channel — the Python server always responds with a readiness
/// state (`RetryLater`) when a model is unknown, which the dispatcher
/// handles separately via [`EnsureModelReadyResponse`].
pub(crate) fn map_ipc_error(e: IpcError) -> BackendError {
    // Log at debug — the dispatcher already logs warn on NAK decisions.
    debug!(error = %e, "python-ipc backend: IPC call failed");
    BackendError::Transient(e.to_string())
}

#[async_trait]
impl InferenceBackend for PythonIpcBackend {
    fn name(&self) -> &'static str {
        "python-ipc"
    }

    fn supports(&self, _model_id: &str) -> bool {
        true
    }

    async fn ensure_model_ready(
        &self,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, BackendError> {
        self.ipc
            .ensure_model_ready(model_id)
            .await
            .map_err(map_ipc_error)
    }

    async fn process_encode_batch(
        &self,
        req: ProcessEncodeBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .process_encode_batch(req)
            .await
            .map_err(map_ipc_error)
    }

    async fn process_score_batch(
        &self,
        req: ProcessScoreBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .process_score_batch(req)
            .await
            .map_err(map_ipc_error)
    }

    async fn process_extract_batch(
        &self,
        req: ProcessExtractBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .process_extract_batch(req)
            .await
            .map_err(map_ipc_error)
    }

    async fn run_batch(&self, req: RunBatchRequest) -> Result<BatchOutcome, BackendError> {
        // Matches the `METHOD_RUN_BATCH` handler Python registers in
        // `sie_server/ipc_server.py`. Transport/protocol failures
        // still map to `Transient` — the dispatcher NAKs and
        // JetStream redelivers into either the scheduler (if the
        // model still routes to the worker-sidecar) or the per-op Python path.
        self.ipc.run_batch(req).await.map_err(map_ipc_error)
    }

    async fn run_batch_with_budget(
        &self,
        req: RunBatchRequest,
        budget: Option<Duration>,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .run_batch_with_budget(req, budget)
            .await
            .map_err(map_ipc_error)
    }

    async fn run_batch_with_execution_authority_v1(
        &self,
        req: RunBatchRequest,
        budget: Option<Duration>,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .run_batch_with_execution_authority_v1(req, budget)
            .await
            .map_err(map_ipc_error)
    }

    async fn run_batch_with_numerical_admission_v1(
        &self,
        req: RunBatchRequest,
        budget: Option<Duration>,
    ) -> Result<BatchOutcome, BackendError> {
        self.ipc
            .run_batch_with_numerical_admission_v1(req, budget)
            .await
            .map_err(map_ipc_error)
    }

    async fn drain(&self, deadline_ms: u64) {
        match self.ipc.drain(deadline_ms).await {
            Ok(resp) => debug!(
                acknowledged = resp.acknowledged,
                "python-ipc drain acknowledged"
            ),
            Err(e) => warn!(error = %e, "python-ipc drain RPC failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn map_ipc_error_produces_transient_for_io() {
        let e = IpcError::Io(io::Error::new(io::ErrorKind::BrokenPipe, "pipe"));
        let be = map_ipc_error(e);
        assert!(matches!(be, BackendError::Transient(_)));
        assert!(be.is_retryable());
    }

    #[test]
    fn map_ipc_error_produces_transient_for_server() {
        let e = IpcError::Server("model load failed".into());
        let be = map_ipc_error(e);
        assert!(matches!(be, BackendError::Transient(_)));
    }

    #[test]
    fn map_ipc_error_produces_transient_for_timeout() {
        let be = map_ipc_error(IpcError::Timeout);
        assert!(matches!(be, BackendError::Transient(_)));
    }
}
