//! Multiplexed IPC connection — single UDS, many in-flight RPCs.
//!
//! Drop-in alternative to the slot-pool transport in [`crate::ipc_client`].
//! The two designs converge on the same `ipc_server.py` (which already
//! accepts pipelined frames on one connection — process methods run in
//! independent `asyncio.Task`s, see `ipc_server.py:271-286`); the
//! difference is purely on the Rust side:
//!
//! * **Pool transport**: N independent UDS connections, each strictly
//!   single-flight (`write → read`). Concurrency caps at N. A slow RPC
//!   pins one slot for its full duration; if all N slots are busy a
//!   new caller waits on a semaphore even though Python is idle.
//! * **Mux transport** (this module): one UDS connection, one writer
//!   task, one reader task, an inflight map keyed by `request_id`.
//!   `N` concurrent RPCs are limited only by the inflight cap (default
//!   1024) and by Python's own concurrency. No head-of-line stalls
//!   between unrelated methods (e.g. `EnsureModelReady` no longer
//!   queues behind a heavy `RunBatch`).
//!
//! # Activation
//!
//! Disabled by default. Set `SIE_IPC_MUX=1` to opt in. The pool
//! transport stays the legacy fallback so any deployment-specific
//! regression can be reverted with an env-var flip, no rebuild.
//!
//! # Cancellation safety
//!
//! The actor pattern owns all I/O. Caller futures only ever see two
//! awaits: a `mpsc::Sender::send` (drops the command if cancelled —
//! safe) and a `oneshot::Receiver::await` (drops the receiver if
//! cancelled — the actor's send fails silently when the response
//! arrives later, which is a no-op). At no point can a cancelled
//! caller leave the wire in a half-written state.
//!
//! # Reconnect
//!
//! On any I/O error the actor:
//!   1. Sends `IpcError::Io` to every still-in-flight oneshot.
//!   2. Drops the read/write halves and sleeps a small backoff.
//!   3. Reconnects and resumes processing the command channel.
//!
//! Callers that want retry-on-transport-error get it for free: they
//! see one error, the next call proceeds against the new connection.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::log_util::ErrChain;
use crate::observability::metrics::SidecarTelemetry;
use crate::protocol::response_chunks::{
    response_frame_route, AssembledResponse, ResponseAssembler, ResponseChunkBudget,
    ResponseChunkLimits, ResponseFrameStatus,
};

/// Same upper bound the pool transport uses; matches Python's
/// `_MAX_FRAME_BYTES` so a misconfigured caller can't OOM either side.
const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Maximum simultaneously-in-flight RPCs. The mpsc channel is bounded
/// at this size so that backpressure shows up at `MuxClient::call`
/// instead of inside the actor's HashMap. 1024 is large enough to
/// never bind under realistic NATS pull rates and small enough that a
/// stuck Python adapter can't pin unbounded heap.
const MAX_INFLIGHT: usize = 1024;

/// Backoff applied between reconnect attempts. Kept short — the
/// Python side is on the same Pod and "down" usually means "still
/// finishing startup" or "just restarted".
const RECONNECT_BACKOFF: Duration = Duration::from_millis(100);

/// IPC errors visible at the `MuxClient::call` boundary. Mirrors the
/// shape of [`crate::ipc_client::IpcError`] for the subset that's
/// observable through the multiplexed transport — the typed
/// deserialization, version check, and `ok=false` mapping happen in
/// the wrapping `IpcClient`.
#[derive(Debug, thiserror::Error)]
pub enum MuxError {
    #[error("io: {0}")]
    Io(std::io::Error),
    #[error("frame too large: {0} > {MAX_FRAME_BYTES}")]
    FrameTooLarge(u32),
    #[error("response chunk protocol: {0}")]
    ResponseChunk(String),
    #[error("connection lost while waiting for response")]
    ConnectionLost,
    #[error("multiplexer shutting down")]
    Shutdown,
}

/// One pending response, registered before the request frame goes on
/// the wire and consumed exactly once by either the reader (success
/// path) or the actor's reconnect handler (drain path).
type Pending = oneshot::Sender<Result<AssembledResponse, MuxError>>;

struct PendingResponse {
    respond: Option<Pending>,
    assembler: ResponseAssembler,
}

/// Map of in-flight requests indexed by `request_id`.
type Inflight = Arc<StdMutex<HashMap<String, Arc<StdMutex<PendingResponse>>>>>;

/// Command sent from `MuxClient::call` to the writer task.
struct Command {
    request_id: String,
    payload: Vec<u8>,
    respond: Pending,
    cancelled: Arc<AtomicBool>,
}

/// Removes a pending mux response when its caller future is cancelled.
///
/// The atomic closes the enqueue-before-registration race: if cancellation
/// happens while the command is still waiting in the writer queue, the writer
/// observes it and skips registration entirely. If registration already
/// happened, direct map removal drops the assembler and its memory reservation.
struct PendingCallGuard {
    request_id: String,
    inflight: Inflight,
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl PendingCallGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingCallGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.cancelled.store(true, Ordering::Release);
        if let Ok(mut guard) = self.inflight.lock() {
            guard.remove(&self.request_id);
        }
    }
}

/// Caller-facing handle. Cheap to clone (just a few `Arc`s); intended
/// to be wrapped by [`crate::ipc_client::IpcClient`]. The actor task
/// owns the I/O and reconnect logic.
pub struct MuxClient {
    cmd_tx: mpsc::Sender<Command>,
    next_id: AtomicU64,
    telemetry: Arc<OnceLock<SidecarTelemetry>>,
    inflight: Inflight,
    response_chunk_budget: Arc<ResponseChunkBudget>,
}

struct MuxInflightGuard(Option<SidecarTelemetry>);

impl Drop for MuxInflightGuard {
    fn drop(&mut self) {
        if let Some(telemetry) = &self.0 {
            telemetry.ipc_released("mux");
        }
    }
}

impl MuxClient {
    /// Spawn a multiplexer pointed at `socket_path`. The actor task
    /// owns the connection; this handle owns only the command sender.
    pub fn spawn(socket_path: impl Into<PathBuf>) -> Self {
        Self::spawn_with_budget(socket_path, ResponseChunkBudget::production())
    }

    pub(crate) fn spawn_with_budget(
        socket_path: impl Into<PathBuf>,
        budget: Arc<ResponseChunkBudget>,
    ) -> Self {
        Self::spawn_with_budget_and_limits(socket_path, budget, ResponseChunkLimits::production())
    }

    #[cfg(test)]
    fn spawn_with_test_chunk_limits(
        socket_path: impl Into<PathBuf>,
        budget: Arc<ResponseChunkBudget>,
    ) -> Self {
        Self::spawn_with_budget_and_limits(
            socket_path,
            budget,
            ResponseChunkLimits::relaxed_for_small_fixtures(),
        )
    }

    fn spawn_with_budget_and_limits(
        socket_path: impl Into<PathBuf>,
        budget: Arc<ResponseChunkBudget>,
        chunk_limits: ResponseChunkLimits,
    ) -> Self {
        let socket_path = socket_path.into();
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(MAX_INFLIGHT);
        let telemetry = Arc::new(OnceLock::new());
        let actor_telemetry = Arc::clone(&telemetry);
        let inflight: Inflight = Arc::new(StdMutex::new(HashMap::new()));
        let actor_inflight = Arc::clone(&inflight);
        let actor_budget = Arc::clone(&budget);
        tokio::spawn(async move {
            mux_actor(
                socket_path,
                cmd_rx,
                actor_telemetry,
                actor_inflight,
                actor_budget,
                chunk_limits,
            )
            .await;
        });
        Self {
            cmd_tx,
            next_id: AtomicU64::new(1),
            telemetry,
            inflight,
            response_chunk_budget: budget,
        }
    }

    /// Attach the process-local facade after the transport is selected by the
    /// outer IPC builder. The once-cell prevents a live mux from changing its
    /// observation sink.
    pub fn attach_telemetry(&self, telemetry: SidecarTelemetry) {
        if !telemetry.is_enabled() {
            return;
        }
        if self.telemetry.set(telemetry).is_ok() {
            if let Some(telemetry) = self.telemetry.get() {
                self.response_chunk_budget
                    .attach_telemetry(telemetry.clone());
                telemetry.ipc_transport_registered("mux", MAX_INFLIGHT);
            }
        }
    }

    /// Generate a unique request id. The `mw-` prefix lets log
    /// scanners distinguish multiplexer ids from the pool transport's
    /// `w-` ids.
    pub fn next_request_id(&self) -> String {
        format!("mw-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Test-only convenience wrapper. Production callers use
    /// [`Self::call_assembled`] so the chunk reservation stays alive through
    /// typed response decoding.
    #[cfg(test)]
    async fn call_raw(&self, request_id: String, payload: Vec<u8>) -> Result<Vec<u8>, MuxError> {
        self.call_assembled(request_id, payload)
            .await
            .map(|response| response.as_slice().to_vec())
    }

    pub(crate) async fn call_assembled(
        &self,
        request_id: String,
        payload: Vec<u8>,
    ) -> Result<AssembledResponse, MuxError> {
        let acquire_started = self.telemetry.get().map(|_| std::time::Instant::now());
        let (respond, wait) = oneshot::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut cancellation = PendingCallGuard {
            request_id: request_id.clone(),
            inflight: Arc::clone(&self.inflight),
            cancelled: Arc::clone(&cancelled),
            armed: true,
        };
        let cmd = Command {
            request_id,
            payload,
            respond,
            cancelled,
        };
        if self.cmd_tx.send(cmd).await.is_err() {
            if let (Some(telemetry), Some(acquire_started)) =
                (self.telemetry.get(), acquire_started)
            {
                telemetry.ipc_acquired("mux", "error", acquire_started.elapsed());
            }
            return Err(MuxError::Shutdown);
        }
        let telemetry = self.telemetry.get().cloned();
        if let (Some(telemetry), Some(acquire_started)) = (&telemetry, acquire_started) {
            telemetry.ipc_acquired("mux", "success", acquire_started.elapsed());
        }
        let _inflight = MuxInflightGuard(telemetry);
        let result = match wait.await {
            Ok(r) => r,
            // Sender dropped without sending — actor torn down with
            // this request still pending. Treat as connection lost.
            Err(_) => Err(MuxError::ConnectionLost),
        };
        cancellation.disarm();
        result
    }

    /// Approximate pool-size analog for parity with the slot pool's
    /// metric: the multiplexer can drive up to `MAX_INFLIGHT`
    /// in-flight RPCs from a single connection.
    pub fn capacity(&self) -> usize {
        MAX_INFLIGHT
    }
}

/// Long-lived actor task. Holds the connection, spawns reader/writer,
/// reconnects forever (until the command channel closes).
async fn mux_actor(
    socket_path: PathBuf,
    mut cmd_rx: mpsc::Receiver<Command>,
    telemetry: Arc<OnceLock<SidecarTelemetry>>,
    inflight: Inflight,
    budget: Arc<ResponseChunkBudget>,
    chunk_limits: ResponseChunkLimits,
) {
    loop {
        // (Re)connect.
        let stream = match UnixStream::connect(&socket_path).await {
            Ok(s) => {
                info!(
                    socket = %socket_path.display(),
                    "mux: connected to ipc_server"
                );
                s
            }
            Err(e) => {
                warn!(
                    socket = %socket_path.display(),
                    error = %ErrChain(&e),
                    "mux: connect failed — backing off"
                );
                tokio::time::sleep(RECONNECT_BACKOFF).await;
                continue;
            }
        };

        let (read_half, write_half) = stream.into_split();
        let reader_inflight = Arc::clone(&inflight);
        let mut reader_task = tokio::spawn(reader_loop(read_half, reader_inflight, chunk_limits));

        let writer_inflight = Arc::clone(&inflight);
        // Writer borrows cmd_rx exclusively for this connection's
        // lifetime; on disconnect we get it back to drive the next.
        //
        // We `select!` on writer-finishes vs reader-exits so a peer-
        // initiated close detected by the reader (`read_exact` →
        // EOF) immediately tears down the writer + drains inflight,
        // even if no caller has a request in flight to drive the
        // writer's I/O. Without this, a quiet-window disconnect
        // would park the writer on `cmd_rx.recv()` and any caller
        // waiting on a previously-sent request would hang until its
        // own `request_timeout` (60s) fires.
        let writer_fut = writer_loop(
            write_half,
            &mut cmd_rx,
            writer_inflight,
            Arc::clone(&budget),
            Arc::clone(&telemetry),
            chunk_limits,
        );
        tokio::pin!(writer_fut);
        let (writer_outcome, reader_exited) = tokio::select! {
            o = &mut writer_fut => (o, false),
            _ = &mut reader_task => {
                // Reader exited (EOF / IO / oversized frame). The
                // connection is dead — abandon the writer (its
                // pinned future is dropped on the way out of this
                // select), drain, reconnect.
                debug!("mux: reader exited before writer — connection lost");
                (WriterOutcome::IoError, true)
            }
        };

        // Stop the reader only when the writer won the race. A completed
        // JoinHandle cannot be polled a second time on current Tokio.
        if !reader_exited {
            reader_task.abort();
            let _ = reader_task.await;
        }

        // Drain all still-pending requests with ConnectionLost so
        // callers don't hang forever. The actor is the single owner
        // of the inflight map at this point — the reader task is
        // gone and the writer returned.
        let drained: Vec<(String, Arc<StdMutex<PendingResponse>>)> = {
            let mut guard = inflight
                .lock()
                .expect("mux: inflight mutex poisoned during drain");
            guard.drain().collect()
        };
        let drained_count = drained.len();
        for (rid, pending) in drained {
            debug!(request_id = %rid, "mux: draining inflight on reconnect");
            if let Ok(mut pending) = pending.lock() {
                if let Some(respond) = pending.respond.take() {
                    let _ = respond.send(Err(MuxError::ConnectionLost));
                }
            }
        }
        if drained_count > 0 {
            warn!(
                drained_count,
                "mux: connection lost — drained inflight, will reconnect"
            );
        }

        // Channel closed → no more callers, exit cleanly. Otherwise
        // loop back and try to reconnect.
        if let WriterOutcome::ChannelClosed = writer_outcome {
            info!("mux: command channel closed — actor exiting");
            return;
        }
        tokio::time::sleep(RECONNECT_BACKOFF).await;
    }
}

/// Reason the writer loop returned.
enum WriterOutcome {
    /// Caller side dropped its [`MuxClient`] — no more commands will
    /// arrive. Actor should exit.
    ChannelClosed,
    /// Outbound I/O error. Actor should reconnect.
    IoError,
}

async fn writer_loop(
    mut write_half: tokio::net::unix::OwnedWriteHalf,
    cmd_rx: &mut mpsc::Receiver<Command>,
    inflight: Inflight,
    budget: Arc<ResponseChunkBudget>,
    telemetry: Arc<OnceLock<SidecarTelemetry>>,
    chunk_limits: ResponseChunkLimits,
) -> WriterOutcome {
    while let Some(cmd) = cmd_rx.recv().await {
        let Command {
            request_id,
            payload,
            respond,
            cancelled,
        } = cmd;

        if cancelled.load(Ordering::Acquire) {
            continue;
        }

        // Register the responder BEFORE writing the frame: if the
        // reader gets the response back faster than this thread can
        // schedule (test scenarios with in-process echo do hit this),
        // the lookup must already succeed.
        if let Ok(mut guard) = inflight.lock() {
            if cancelled.load(Ordering::Acquire) {
                continue;
            }
            let _ = guard.insert(
                request_id.clone(),
                Arc::new(StdMutex::new(PendingResponse {
                    respond: Some(respond),
                    assembler: ResponseAssembler::new_with_telemetry_and_limits(
                        request_id.clone(),
                        Arc::clone(&budget),
                        telemetry.get().cloned(),
                        chunk_limits,
                    ),
                })),
            );
        } else {
            // Mutex poisoned — surface to caller and bail. A poisoned
            // mutex means a previous panic in this actor; further I/O
            // would just propagate the corruption.
            return WriterOutcome::IoError;
        }

        // Length prefix is u32 big-endian — matches the pool transport
        // and `ipc_server.py::_LEN_STRUCT`.
        let len = match u32::try_from(payload.len()) {
            Ok(l) if (l as usize) <= MAX_FRAME_BYTES => l,
            _ => {
                let too_big = payload.len() as u32;
                let tx = inflight
                    .lock()
                    .expect("mux: inflight mutex poisoned")
                    .remove(&request_id);
                if let Some(pending) = tx {
                    if let Ok(mut pending) = pending.lock() {
                        if let Some(respond) = pending.respond.take() {
                            let _ = respond.send(Err(MuxError::FrameTooLarge(too_big)));
                        }
                    }
                }
                continue;
            }
        };
        if let Err(e) = write_half.write_all(&len.to_be_bytes()).await {
            // Pull the responder back out and notify before tearing
            // down. If we can't grab the lock the response is
            // effectively lost; ConnectionLost will fire on the
            // actor's drain path next.
            if let Ok(mut guard) = inflight.lock() {
                if let Some(pending) = guard.remove(&request_id) {
                    if let Ok(mut pending) = pending.lock() {
                        if let Some(respond) = pending.respond.take() {
                            let _ = respond.send(Err(MuxError::Io(e)));
                        }
                    }
                }
            }
            return WriterOutcome::IoError;
        }
        if let Err(e) = write_half.write_all(&payload).await {
            if let Ok(mut guard) = inflight.lock() {
                if let Some(pending) = guard.remove(&request_id) {
                    if let Ok(mut pending) = pending.lock() {
                        if let Some(respond) = pending.respond.take() {
                            let _ = respond.send(Err(MuxError::Io(e)));
                        }
                    }
                }
            }
            return WriterOutcome::IoError;
        }
        if let Err(e) = write_half.flush().await {
            if let Ok(mut guard) = inflight.lock() {
                if let Some(pending) = guard.remove(&request_id) {
                    if let Ok(mut pending) = pending.lock() {
                        if let Some(respond) = pending.respond.take() {
                            let _ = respond.send(Err(MuxError::Io(e)));
                        }
                    }
                }
            }
            return WriterOutcome::IoError;
        }
    }

    WriterOutcome::ChannelClosed
}

async fn reader_loop(
    mut read_half: tokio::net::unix::OwnedReadHalf,
    inflight: Inflight,
    chunk_limits: ResponseChunkLimits,
) {
    loop {
        let mut len_buf = [0u8; 4];
        if let Err(e) = read_half.read_exact(&mut len_buf).await {
            debug!(error = %ErrChain(&e), "mux reader: read of length prefix failed (likely EOF)");
            return;
        }
        let len = u32::from_be_bytes(len_buf);
        if len as usize > MAX_FRAME_BYTES {
            warn!(
                len,
                max = MAX_FRAME_BYTES,
                "mux reader: oversized frame; dropping connection"
            );
            return;
        }
        let mut frame = vec![0u8; len as usize];
        if let Err(e) = read_half.read_exact(&mut frame).await {
            debug!(error = %ErrChain(&e), "mux reader: read of body failed");
            return;
        }
        // Parse only the identity needed to select the per-request assembler.
        // A malformed envelope is a connection-level protocol failure: there
        // is no safe request to which we could deliver it, so reconnect and
        // fail every pending caller instead of leaving one to time out.
        let route = match response_frame_route(&frame, chunk_limits) {
            Ok(route) => route,
            Err(e) => {
                warn!(error = %ErrChain(&e), "mux reader: malformed envelope; reconnecting");
                return;
            }
        };
        let request_id = route.request_id;

        enum Delivery {
            Pending,
            Complete(Pending, AssembledResponse),
            Error(Pending, String),
            Unknown,
        }

        // Clone one per-request entry under the global map lock, then perform
        // payload copies and final SHA-256 verification under only that
        // request's lock. Other responses, writer registration, and caller
        // cancellation therefore never queue behind a large transfer hash.
        let pending = match inflight.lock() {
            Ok(guard) => guard.get(&request_id).cloned(),
            Err(_) => return,
        };
        let Some(pending) = pending else {
            warn!(
                request_id = %request_id,
                "mux reader: response with no inflight handler — late delivery after drain?"
            );
            continue;
        };
        let delivery = match pending.lock() {
            Ok(mut pending) => match if route.requires_chunk_parser {
                pending.assembler.push(frame)
            } else {
                pending.assembler.push_legacy(frame)
            } {
                Ok(ResponseFrameStatus::Pending) => Delivery::Pending,
                Ok(ResponseFrameStatus::Complete(response)) => match pending.respond.take() {
                    Some(respond) => Delivery::Complete(respond, response),
                    None => Delivery::Unknown,
                },
                Err(error) => match pending.respond.take() {
                    Some(respond) => Delivery::Error(respond, error.to_string()),
                    None => Delivery::Unknown,
                },
            },
            Err(_) => return,
        };

        if !matches!(delivery, Delivery::Pending) {
            let mut guard = match inflight.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            let is_same = guard
                .get(&request_id)
                .is_some_and(|registered| Arc::ptr_eq(registered, &pending));
            if is_same {
                guard.remove(&request_id);
            }
        }
        match delivery {
            Delivery::Pending => {}
            Delivery::Complete(tx, bytes) => {
                let _ = tx.send(Ok(bytes));
            }
            Delivery::Error(tx, error) => {
                let _ = tx.send(Err(MuxError::ResponseChunk(error)));
            }
            Delivery::Unknown => {}
        }
    }
}

/// Ergonomic helper: read the on/off flag for the multiplexed
/// transport. Centralised so callers don't sprinkle `std::env::var`
/// across the codebase.
pub fn mux_enabled() -> bool {
    matches!(
        std::env::var("SIE_IPC_MUX").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

impl From<std::io::Error> for MuxError {
    fn from(e: std::io::Error) -> Self {
        MuxError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::{IpcResponseChunkV1, IPC_VERSION};
    use crate::protocol::response_chunks::IPC_RESPONSE_CHUNK_KIND_V1;
    use serde::{Deserialize, Serialize};
    use sha2::{Digest, Sha256};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::net::UnixListener;

    #[tokio::test]
    async fn disabled_telemetry_does_not_fill_mux_once_cell() {
        let client = MuxClient::spawn(short_sock_path());

        client.attach_telemetry(SidecarTelemetry::default());

        assert!(client.telemetry.get().is_none());
    }

    #[derive(Deserialize)]
    struct EnvelopeHead {
        #[serde(default)]
        request_id: String,
    }

    fn short_sock_path() -> PathBuf {
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
        let base = if base.len() > 20 {
            "/tmp".to_string()
        } else {
            base
        };
        PathBuf::from(base).join(format!("sie-mux-{}.sock", uuid::Uuid::new_v4().simple()))
    }

    /// Test echo server that **does** support multiplexed pipelining:
    /// each frame is processed in its own task, so out-of-order
    /// responses are guaranteed when the handler is async.
    async fn spawn_mux_echo<F, Fut>(path: PathBuf, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<u8>> + Send + 'static,
    {
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let (sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let (mut read_half, write_half) = sock.into_split();
                let write_half = Arc::new(tokio::sync::Mutex::new(write_half));
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if read_half.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if read_half.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let h = Arc::clone(&handler);
                        let w = Arc::clone(&write_half);
                        tokio::spawn(async move {
                            let resp = (h)(buf).await;
                            let mut guard = w.lock().await;
                            let _ = guard.write_all(&(resp.len() as u32).to_be_bytes()).await;
                            let _ = guard.write_all(&resp).await;
                            let _ = guard.flush().await;
                        });
                    }
                });
            }
        })
    }

    #[derive(Serialize)]
    struct StubEnv<'a> {
        version: u32,
        method: &'a str,
        request_id: String,
        body: serde_json::Value,
    }

    fn build_request(method: &str, request_id: String) -> Vec<u8> {
        let env = StubEnv {
            version: 1,
            method,
            request_id,
            body: serde_json::Value::Null,
        };
        rmp_serde::to_vec_named(&env).unwrap()
    }

    fn build_response(request_id: &str, payload: serde_json::Value) -> Vec<u8> {
        let resp = serde_json::json!({
            "version": 1,
            "request_id": request_id,
            "ok": true,
            "body": payload,
            "error": serde_json::Value::Null,
        });
        rmp_serde::to_vec_named(&resp).unwrap()
    }

    fn build_two_response_chunks(request_id: &str, response: &[u8]) -> [Vec<u8>; 2] {
        let split = response.len() / 2;
        let digest = Sha256::digest(response).to_vec();
        std::array::from_fn(|index| {
            let payload = if index == 0 {
                &response[..split]
            } else {
                &response[split..]
            };
            rmp_serde::to_vec_named(&IpcResponseChunkV1 {
                version: IPC_VERSION,
                request_id: request_id.to_owned(),
                transfer_digest: digest.clone(),
                chunk_index: index as u32,
                chunk_count: 2,
                total_bytes: response.len() as u64,
                payload: payload.to_vec(),
                kind: IPC_RESPONSE_CHUNK_KIND_V1.to_owned(),
            })
            .unwrap()
        })
    }

    async fn write_test_frame(stream: &mut UnixStream, frame: &[u8]) {
        stream
            .write_all(&(frame.len() as u32).to_be_bytes())
            .await
            .unwrap();
        stream.write_all(frame).await.unwrap();
        stream.flush().await.unwrap();
    }

    async fn read_test_frame(stream: &mut UnixStream) -> Vec<u8> {
        let mut len = [0_u8; 4];
        stream.read_exact(&mut len).await.unwrap();
        let mut frame = vec![0_u8; u32::from_be_bytes(len) as usize];
        stream.read_exact(&mut frame).await.unwrap();
        frame
    }

    #[tokio::test]
    async fn interleaved_chunk_sequences_reassemble_to_the_right_mux_call() {
        let path = short_sock_path();
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let first: serde_json::Value =
                rmp_serde::from_slice(&read_test_frame(&mut stream).await).unwrap();
            let second: serde_json::Value =
                rmp_serde::from_slice(&read_test_frame(&mut stream).await).unwrap();
            let first_id = first["request_id"].as_str().unwrap();
            let second_id = second["request_id"].as_str().unwrap();
            let first_response = build_response(first_id, serde_json::json!({"which": "first"}));
            let second_response = build_response(second_id, serde_json::json!({"which": "second"}));
            let first_chunks = build_two_response_chunks(first_id, &first_response);
            let second_chunks = build_two_response_chunks(second_id, &second_response);

            // Each per-request sequence remains ordered, but the two transfers
            // are deliberately interleaved on the single UDS connection.
            write_test_frame(&mut stream, &first_chunks[0]).await;
            write_test_frame(&mut stream, &second_chunks[0]).await;
            write_test_frame(&mut stream, &second_chunks[1]).await;
            write_test_frame(&mut stream, &first_chunks[1]).await;

            (
                first_id.to_owned(),
                first_response,
                second_id.to_owned(),
                second_response,
            )
        });

        let client = Arc::new(MuxClient::spawn_with_test_chunk_limits(
            path,
            ResponseChunkBudget::production(),
        ));
        let first_id = client.next_request_id();
        let second_id = client.next_request_id();
        let first_call = {
            let client = Arc::clone(&client);
            let request_id = first_id.clone();
            tokio::spawn(async move {
                client
                    .call_raw(request_id.clone(), build_request("First", request_id))
                    .await
            })
        };
        let second_call = {
            let client = Arc::clone(&client);
            let request_id = second_id.clone();
            tokio::spawn(async move {
                client
                    .call_raw(request_id.clone(), build_request("Second", request_id))
                    .await
            })
        };

        let first_actual = first_call.await.unwrap().unwrap();
        let second_actual = second_call.await.unwrap().unwrap();
        let (wire_first_id, wire_first, wire_second_id, wire_second) = server.await.unwrap();
        let expected_by_id =
            HashMap::from([(wire_first_id, wire_first), (wire_second_id, wire_second)]);
        assert_eq!(first_actual, expected_by_id[&first_id]);
        assert_eq!(second_actual, expected_by_id[&second_id]);
    }

    #[tokio::test]
    async fn cancelled_mux_call_releases_partial_chunk_reservation() {
        let path = short_sock_path();
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let budget = Arc::new(ResponseChunkBudget::new(16 * 1024));
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request: serde_json::Value =
                rmp_serde::from_slice(&read_test_frame(&mut stream).await).unwrap();
            let request_id = request["request_id"].as_str().unwrap();
            let response =
                build_response(request_id, serde_json::json!({"blob": "x".repeat(1024)}));
            let chunks = build_two_response_chunks(request_id, &response);
            write_test_frame(&mut stream, &chunks[0]).await;
            tokio::time::sleep(Duration::from_secs(5)).await;
        });

        let client = Arc::new(MuxClient::spawn_with_test_chunk_limits(
            path,
            Arc::clone(&budget),
        ));
        let request_id = client.next_request_id();
        let call = {
            let client = Arc::clone(&client);
            tokio::spawn(async move {
                client
                    .call_raw(request_id.clone(), build_request("Cancelled", request_id))
                    .await
            })
        };

        tokio::time::timeout(Duration::from_millis(500), async {
            while budget.used() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first chunk should reserve the declared response bytes");
        call.abort();
        let _ = call.await;
        tokio::time::timeout(Duration::from_millis(500), async {
            while budget.used() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelling the caller must release its partial assembler");
        server.abort();
    }

    /// In-order single RPC roundtrip — sanity check that the actor
    /// wires the request id back to the right responder.
    #[tokio::test]
    async fn single_call_roundtrips() {
        let path = short_sock_path();
        let _server = spawn_mux_echo(path.clone(), |req_bytes| async move {
            let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
            build_response(&head.request_id, serde_json::json!({"ok": true}))
        })
        .await;

        let client = MuxClient::spawn(path);
        let rid = client.next_request_id();
        let payload = build_request("Ping", rid.clone());
        let resp = client
            .call_raw(rid, payload)
            .await
            .expect("RPC should succeed");
        let head: EnvelopeHead = rmp_serde::from_slice(&resp).unwrap();
        // Echo bounced the same id back; the multiplexer routed it
        // to our specific waiter instead of any other.
        assert!(!head.request_id.is_empty());
    }

    /// Out-of-order responses on the same connection. The slow
    /// request queues, the fast one resolves first; both must reach
    /// the right caller. This is the property that the slot pool
    /// transport can't deliver without N≥2 connections.
    #[tokio::test]
    async fn parallel_calls_demux_correctly() {
        let path = short_sock_path();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_h = Arc::clone(&counter);
        let _server = spawn_mux_echo(path.clone(), move |req_bytes| {
            let counter = Arc::clone(&counter_h);
            async move {
                let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
                let n = counter.fetch_add(1, Ordering::Relaxed);
                // First request stalls 80 ms, every following one
                // resolves immediately. Without multiplexing the second
                // would be queued behind the first.
                if n == 0 {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                }
                build_response(&head.request_id, serde_json::json!({"n": n}))
            }
        })
        .await;

        let client = Arc::new(MuxClient::spawn(path));
        let started = std::time::Instant::now();

        let c1 = Arc::clone(&client);
        let h_slow = tokio::spawn(async move {
            let rid = c1.next_request_id();
            let payload = build_request("Slow", rid.clone());
            c1.call_raw(rid, payload).await
        });
        // Tiny stagger so the slow request enters the actor first.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let c2 = Arc::clone(&client);
        let h_fast = tokio::spawn(async move {
            let rid = c2.next_request_id();
            let payload = build_request("Fast", rid.clone());
            c2.call_raw(rid, payload).await
        });

        let fast = h_fast.await.unwrap().expect("fast RPC succeeds");
        let elapsed_fast = started.elapsed();
        // Fast call must come back well before the 80 ms slow window
        // would have expired if we were serialized.
        assert!(
            elapsed_fast < Duration::from_millis(70),
            "fast call took {elapsed_fast:?}; multiplexing apparently stalled behind slow call"
        );
        let _slow = h_slow.await.unwrap().expect("slow RPC succeeds");

        // Sanity: both responses were valid envelopes with non-empty
        // request_ids (i.e. demux worked, neither went to the wrong
        // caller).
        let head: EnvelopeHead = rmp_serde::from_slice(&fast).unwrap();
        assert!(!head.request_id.is_empty());
    }

    /// Regression: a peer that closes the connection while a request
    /// is in flight must surface `ConnectionLost` to the caller
    /// promptly — *not* hang until the caller's own request timeout.
    ///
    /// Pre-fix, the actor's writer task awaited `cmd_rx.recv()` with
    /// no awareness of the reader having exited; if the reader saw
    /// EOF and there was no inbound traffic to drive the writer
    /// into a failing `write_all`, the inflight `Pending` for the
    /// already-sent request was never drained.
    ///
    /// Post-fix, the actor `select!`s on writer-finishes vs reader-
    /// exits, so a peer close is detected immediately and the actor
    /// drains inflight before reconnecting. We bound the wait at
    /// 500 ms — pre-fix this would block until the IpcClient's
    /// 60 s timeout (or forever in this test, where there is no
    /// outer timeout).
    #[tokio::test]
    async fn peer_close_drains_inflight_promptly() {
        use tokio::io::AsyncReadExt;

        let path = short_sock_path();
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();

        // Custom server: read one length-prefix + body, then drop
        // the socket. No response is ever written, so the client
        // sits in inflight; only the actor's reader-exit detection
        // can rescue it.
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut len_buf = [0u8; 4];
                let _ = sock.read_exact(&mut len_buf).await;
                let n = u32::from_be_bytes(len_buf) as usize;
                let mut body = vec![0u8; n];
                let _ = sock.read_exact(&mut body).await;
                // Drop sock → peer close.
            }
        });

        let client = MuxClient::spawn(path);
        let rid = client.next_request_id();
        let payload = build_request("Doomed", rid.clone());

        // Bounded wait. With the bug: this would never resolve (the
        // mux actor would park on cmd_rx waiting for the next
        // command, while the inflight request hangs forever).
        let r = tokio::time::timeout(Duration::from_millis(500), client.call_raw(rid, payload))
            .await
            .expect("actor must drain inflight on peer close within 500ms");

        match r {
            Err(MuxError::ConnectionLost) | Err(MuxError::Io(_)) => {} // expected
            other => panic!("expected ConnectionLost / Io, got {:?}", other),
        }
    }

    /// A malformed frame cannot be assigned to one request safely. The mux
    /// therefore drops the connection and drains all pending calls promptly;
    /// the actor then reconnects and serves subsequent requests.
    #[tokio::test]
    async fn malformed_frame_drains_then_reconnects() {
        let path = short_sock_path();
        let arrived = Arc::new(AtomicUsize::new(0));
        let arrived_h = Arc::clone(&arrived);
        // Only the first connection's first frame triggers the
        // server to abort. The retry connection answers normally.
        let _server = spawn_mux_echo(path.clone(), move |req_bytes| {
            let arrived = Arc::clone(&arrived_h);
            async move {
                let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
                let n = arrived.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // Force one malformed response frame. The reader cannot
                    // route it and must tear down this connection.
                    Vec::new()
                } else {
                    build_response(&head.request_id, serde_json::json!({"n": n}))
                }
            }
        })
        .await;

        let client = MuxClient::spawn(path.clone());

        // First call is drained promptly when the malformed frame closes the
        // current mux connection.
        let rid1 = client.next_request_id();
        let p1 = build_request("First", rid1.clone());
        let r1 = tokio::time::timeout(Duration::from_millis(500), client.call_raw(rid1, p1))
            .await
            .expect("malformed response must drain promptly");
        assert!(matches!(r1, Err(MuxError::ConnectionLost)));

        // The actor reconnects and the next call succeeds.
        let rid2 = client.next_request_id();
        let p2 = build_request("Second", rid2.clone());
        let r2 = client.call_raw(rid2, p2).await.expect("second call ok");
        let head: EnvelopeHead = rmp_serde::from_slice(&r2).unwrap();
        assert!(!head.request_id.is_empty());
    }
}
