//! Async IPC client over a Unix domain socket, speaking the length-prefixed
//! msgpack protocol shared by SIE backend IPC servers.
//!
//! # Concurrency model: connection pool
//!
//! Each IPC RPC is a strictly synchronous `write_frame → read_frame`
//! round-trip on one connection. A single connection therefore admits at
//! most one in-flight RPC; any second caller blocks behind the first.
//!
//! To let `SIE_MAX_CONCURRENT_BATCHES` actually translate into parallel
//! IPC calls — and thus let the backend-side adaptive batcher / GPU
//! pipeline overlap CPU preprocessing on one batch with GPU inference on
//! another — we run a small **pool** of N independent connections to the
//! same backend IPC server.
//!
//! * Backend IPC servers accept concurrent connections in deployed runtimes; each
//!   Rust slot just looks like another sidecar.
//! * Callers `acquire()` a `SlotGuard` — a fair FIFO checkout — do
//!   their round-trip, and drop it to return the slot. If all N slots
//!   are checked out the caller waits on a [`tokio::sync::Semaphore`],
//!   which avoids the head-of-line blocking a round-robin scheme would
//!   suffer when one slot has a slow in-flight RPC.
//! * A transport error on slot K only resets slot K's socket, so one
//!   broken connection doesn't wipe the other N-1 sessions.
//!
//! Sizing: 1 is the legacy behaviour (restored by `IpcClient::new`).
//! Real deployments set `SIE_IPC_POOL_SIZE` (see `main.rs`) to match
//! `SIE_MAX_CONCURRENT_BATCHES` so the dispatcher's concurrency limit
//! is the binding one, not this socket mutex.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::ipc_mux::{mux_enabled, MuxClient, MuxError};
use crate::ipc_types::{
    ApplyModelConfigRequest, ApplyModelConfigResponse, BatchOutcome, DrainRequest, DrainResponse,
    EnsureModelReadyRequest, EnsureModelReadyResponse, GenerateEvent, PingRequest, PingResponse,
    ProcessEncodeBatchRequest, ProcessExtractBatchRequest, ProcessGenerateRequest,
    ProcessScoreBatchRequest, ReplaceModelConfigsRequest, ReplaceModelConfigsResponse,
    RequestEnvelope, ResponseEnvelope, RunBatchRequest, SetPinnedModelsRequest,
    SetPinnedModelsResponse, SignalGenerateCancelRequest, SignalGenerateCancelResponse,
    WorkerCapabilitiesRequest, WorkerCapabilitiesResponse, IPC_VERSION, METHOD_APPLY_MODEL_CONFIG,
    METHOD_DRAIN, METHOD_ENSURE_MODEL_READY, METHOD_PING, METHOD_PROCESS_ENCODE_BATCH,
    METHOD_PROCESS_EXTRACT_BATCH, METHOD_PROCESS_GENERATE, METHOD_PROCESS_SCORE_BATCH,
    METHOD_REPLACE_MODEL_CONFIGS, METHOD_RUN_BATCH, METHOD_SET_PINNED_MODELS,
    METHOD_SIGNAL_GENERATE_CANCEL, METHOD_WORKER_CAPABILITIES,
};
use crate::log_util::ErrChain;
use crate::observability::metrics::SidecarTelemetry;
use crate::protocol::response_chunks::{
    response_frame_is_exact_chunk, AssembledResponse, ResponseAssembler, ResponseChunkBudget,
    ResponseChunkError, ResponseChunkLimits, ResponseFrameStatus, IPC_RESPONSE_CHUNK_KIND_V1,
};

const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Default pool size when the caller doesn't specify one. Matches the
/// legacy single-connection behaviour — production deployments override
/// via [`IpcClient::with_pool_size`] / `SIE_IPC_POOL_SIZE`.
const DEFAULT_POOL_SIZE: usize = 1;

/// WARN threshold for "slow" RPCs — emits one structured line per slow
/// call so an operator scanning `kubectl logs` sees backend-side
/// stalls without having to open Grafana. The canonical
/// `sie.worker.ipc.request.duration` histogram is the primary source of truth;
/// this is the ergonomic
/// shortcut for live debugging. Keep the budget generous: model
/// loading (ensure_model_ready) can legitimately take 10+ seconds, so
/// we deliberately do NOT pick a threshold low enough to flag that
/// path; it'd be pure noise. We do flag ping / per-batch inference.
const SLOW_RPC_WARN_MS: u128 = 2_000;

/// A bounded result used by the canonical IPC completion facade.
/// Keeps cardinality low (no raw messages).
fn error_label(e: &IpcError) -> &'static str {
    match e {
        IpcError::Io(_) => "io",
        IpcError::Encode(_) => "encode",
        IpcError::Decode(_) => "decode",
        IpcError::Timeout => "timeout",
        IpcError::FrameTooLarge(_) => "frame_too_large",
        IpcError::ResponseChunk(_) => "response_chunk",
        IpcError::Server(_) => "server",
        IpcError::VersionMismatch { .. } => "version_mismatch",
    }
}

/// Classify an `IpcError` as a transport glitch (`true`) or a logical /
/// protocol error (`false`). Only transport glitches are worth retrying.
fn is_transport_error(e: &IpcError) -> bool {
    match e {
        IpcError::Io(io_err) => matches!(
            io_err.kind(),
            std::io::ErrorKind::BrokenPipe
                | std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::UnexpectedEof
                | std::io::ErrorKind::NotFound
                | std::io::ErrorKind::NotConnected
                | std::io::ErrorKind::WouldBlock
        ),
        _ => false,
    }
}

fn generate_frame_timeout() -> Duration {
    let secs = std::env::var("SIE_GENERATE_IPC_FRAME_TIMEOUT_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(300);
    Duration::from_secs(secs)
}

async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, IpcError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let resp_len = u32::from_be_bytes(len_buf);
    if resp_len as usize > MAX_FRAME_BYTES {
        return Err(IpcError::FrameTooLarge(resp_len));
    }
    let mut resp_buf = vec![0u8; resp_len as usize];
    stream.read_exact(&mut resp_buf).await?;
    Ok(resp_buf)
}

fn request_frame_len(payload_len: usize) -> Result<u32, IpcError> {
    if payload_len > MAX_FRAME_BYTES {
        return Err(IpcError::FrameTooLarge(
            u32::try_from(payload_len).unwrap_or(u32::MAX),
        ));
    }
    u32::try_from(payload_len).map_err(|_| IpcError::FrameTooLarge(u32::MAX))
}

#[derive(Debug, Error)]
pub enum IpcError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("decode: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("timeout waiting for IPC response")]
    Timeout,
    #[error("frame too large: {0} > {MAX_FRAME_BYTES}")]
    FrameTooLarge(u32),
    #[error("response chunk protocol: {0}")]
    ResponseChunk(String),
    #[error("server returned error: {0}")]
    Server(String),
    #[error("envelope version mismatch: got {got}, expected {IPC_VERSION}")]
    VersionMismatch { got: u32 },
}

#[derive(Deserialize)]
struct TypedResponseEnvelope<B> {
    version: u32,
    request_id: String,
    ok: bool,
    #[serde(default = "none_typed_response_body")]
    body: Option<B>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    kind: Option<String>,
}

fn none_typed_response_body<B>() -> Option<B> {
    None
}

enum DecodedResponseFrame<B> {
    Legacy(B),
    Chunk,
}

fn decode_response_frame<B: DeserializeOwned>(
    frame: &[u8],
    expected_request_id: &str,
    chunk_limits: ResponseChunkLimits,
) -> Result<DecodedResponseFrame<B>, IpcError> {
    let envelope: TypedResponseEnvelope<B> = match rmp_serde::from_slice(frame) {
        Ok(envelope) => envelope,
        Err(decode_error) => {
            return match response_frame_is_exact_chunk(frame, chunk_limits) {
                Ok(true) => Ok(DecodedResponseFrame::Chunk),
                Ok(false) => Err(IpcError::Decode(decode_error)),
                Err(error) => Err(IpcError::ResponseChunk(error.to_string())),
            };
        }
    };
    if envelope.kind.as_deref() == Some(IPC_RESPONSE_CHUNK_KIND_V1) {
        return Ok(DecodedResponseFrame::Chunk);
    }
    if envelope.version != IPC_VERSION {
        return Err(IpcError::VersionMismatch {
            got: envelope.version,
        });
    }
    if envelope.request_id != expected_request_id {
        return Err(IpcError::ResponseChunk(
            ResponseChunkError::RequestIdMismatch.to_string(),
        ));
    }
    if !envelope.ok {
        return Err(IpcError::Server(envelope.error.unwrap_or_default()));
    }
    envelope
        .body
        .map(DecodedResponseFrame::Legacy)
        .ok_or_else(|| IpcError::Server("response ok=true but body missing".to_string()))
}

/// One physical UDS connection, identified by its slot index. Only ever
/// accessed by whichever task currently holds it (enforced by the pool's
/// checkout semantics), so no internal synchronization is needed.
struct Slot {
    id: usize,
    stream: Option<UnixStream>,
}

impl Slot {
    fn new(id: usize) -> Self {
        Self { id, stream: None }
    }
}

/// Fair-FIFO pool of [`Slot`]s backed by a [`Semaphore`].
///
/// Invariants:
/// * The semaphore's permit count equals `capacity - live_checkouts`.
/// * Permit release is paired with the slot push: we always push the
///   slot back onto the queue **before** the permit is released. A
///   waiter that subsequently wakes therefore always finds a slot
///   ready to pop, so [`Self::acquire`]'s `expect` can never fire.
///
/// We use a `std::sync::Mutex` (not `tokio::sync::Mutex`) for the slot
/// queue because:
/// 1. The critical section is a single `VecDeque::{push_back, pop_front}`
///    — sub-microsecond, no `.await` points, so Tokio's async mutex
///    brings no benefit and its async-only API prevents using it from
///    inside `Drop` (which is where the push-back happens).
/// 2. Holding a blocking lock for that span does not starve the
///    executor — it's shorter than a scheduling quantum.
///
/// Why not round-robin by index? A slow RPC on slot K would block every
/// Kth caller that wrapped back to that slot, even if N-1 others were
/// idle. Checkout-and-return gives strict fairness and head-of-line
/// isolation.
struct Pool {
    capacity: usize,
    slots: StdMutex<VecDeque<Slot>>,
    permits: Arc<Semaphore>,
    telemetry: Option<SidecarTelemetry>,
}

impl Pool {
    fn new(capacity: usize, telemetry: Option<SidecarTelemetry>) -> Self {
        assert!(capacity >= 1, "IPC pool capacity must be ≥ 1");
        let mut slots = VecDeque::with_capacity(capacity);
        for id in 0..capacity {
            slots.push_back(Slot::new(id));
        }
        Self {
            capacity,
            slots: StdMutex::new(slots),
            permits: Arc::new(Semaphore::new(capacity)),
            telemetry,
        }
    }

    /// Wait for a free slot. Returns an RAII guard that returns the slot
    /// to the pool on drop. If the caller wishes to discard a slot (e.g.
    /// after an unrecoverable error) they can still drop the guard —
    /// the slot's socket is either still usable or already `None`.
    async fn acquire(self: &Arc<Self>) -> SlotGuard {
        let started = self.telemetry.as_ref().map(|_| std::time::Instant::now());
        let permit = Arc::clone(&self.permits)
            .acquire_owned()
            .await
            .expect("IPC pool semaphore closed");
        let slot = {
            let mut q = self.slots.lock().expect("pool slot mutex poisoned");
            q.pop_front().expect(
                "pool invariant violated: permit acquired but no slot available \
                 (indicates the SlotGuard Drop order is wrong — the permit is \
                 being released before the slot is pushed back)",
            )
        };
        if let (Some(telemetry), Some(started)) = (&self.telemetry, started) {
            telemetry.ipc_acquired("pool", "success", started.elapsed());
        }
        SlotGuard {
            slot: Some(slot),
            pool: Arc::clone(self),
            dirty: false,
            _permit: permit,
        }
    }

    /// Drop every idle slot's socket. Slots currently checked out will
    /// reset themselves when the caller's `call_inner` observes the
    /// transport error; this only clobbers the ones sitting in the
    /// queue. Used by [`IpcClient::reset`] for shutdown / pathological
    /// recovery.
    fn reset_idle(&self) {
        let mut q = self.slots.lock().expect("pool slot mutex poisoned");
        for s in q.iter_mut() {
            s.stream = None;
        }
    }
}

/// RAII handle to a single [`Slot`] checked out of a [`Pool`]. Dropping
/// the guard returns the slot to the back of the pool and releases the
/// semaphore permit.
///
/// ## Cancellation safety
///
/// The IPC protocol is a strict `write_frame → read_frame` round-trip.
/// If a caller's `call(…)` Future is dropped mid-I/O (e.g. because a
/// gateway-imposed request deadline fires, or the caller wrapped us in
/// `tokio::time::timeout` and hit it), the underlying `UnixStream` may
/// have:
///   * a half-written length prefix but no body, or
///   * a fully written frame but no response read yet.
///
/// Either way the socket is **protocol-desynchronized**: the next
/// request's bytes would be misinterpreted by the backend server as a
/// continuation of the aborted one, producing either a silent hang or
/// a corrupt decode.
///
/// To prevent that we arm a "dirty" flag on the guard before any I/O
/// `.await` and only disarm it on successful protocol completion. If
/// `Drop` runs while the flag is armed (cancellation, panic, early
/// return), the slot's stream is cleared so the next `call_inner` on
/// it reconnects from scratch.
///
/// Field order matters: Rust drops struct fields in declaration order,
/// so `slot` is taken first in our [`Drop`] impl, and the permit in
/// `_permit` is released afterwards. This ordering is what upholds the
/// pool invariant that a waiter waking on the semaphore always finds a
/// slot to pop.
struct SlotGuard {
    slot: Option<Slot>,
    pool: Arc<Pool>,
    /// Set to `true` while I/O is in flight. If the guard is dropped
    /// while armed, the slot's socket is forcibly reset on the way
    /// back into the pool. Reset on success via [`Self::disarm`].
    dirty: bool,
    _permit: OwnedSemaphorePermit,
}

impl SlotGuard {
    fn slot_mut(&mut self) -> &mut Slot {
        self.slot.as_mut().expect("slot taken before guard dropped")
    }

    /// Mark the slot's stream as mid-I/O — any non-clean exit will
    /// force a reconnect on next use. Paired with [`Self::disarm`] on
    /// successful protocol completion, or an explicit
    /// `slot.stream = None` for errors the caller classifies as
    /// transport glitches.
    fn arm(&mut self) {
        self.dirty = true;
    }

    /// Mark the current call as cleanly completed — no stream reset
    /// needed on drop.
    fn disarm(&mut self) {
        self.dirty = false;
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        // Push the slot back *synchronously* before our `_permit` field
        // is dropped below. A std::sync::Mutex over a VecDeque makes
        // this a handful of instructions — we can't use a Tokio async
        // mutex here because Drop is sync. After this scope, the
        // compiler drops `_permit`, which bumps the semaphore and
        // wakes the next waiter; they then pop the slot we just
        // pushed. Reordering would reintroduce the "permit but no
        // slot" race the old async-spawn version suffered.
        if let Some(mut slot) = self.slot.take() {
            if self.dirty {
                // Cancellation / panic / early-return path: the
                // stream may be protocol-desynchronized. Clobber it
                // so the next user reconnects cleanly.
                slot.stream = None;
            }
            let mut q = self.pool.slots.lock().expect("pool slot mutex poisoned");
            q.push_back(slot);
            drop(q);
            if let Some(telemetry) = &self.pool.telemetry {
                telemetry.ipc_released("pool");
            }
        }
    }
}

/// Shared IPC client — a pool of UDS connections to `ipc_server.py`.
///
/// Created with either [`IpcClient::new`] (single connection, legacy)
/// or [`IpcClient::new_pool`] / [`IpcClient::with_pool_size`] for
/// multi-connection deployments. Connections are established lazily on
/// first use per slot.
///
/// The transport can be switched to a multiplexed single-connection
/// design by setting `SIE_IPC_MUX=1` (see [`crate::ipc_mux`]). The
/// public API is identical; only the wire-level concurrency model
/// differs (slot pool vs. id-multiplexing).
pub struct IpcClient {
    socket_path: PathBuf,
    pool: Arc<Pool>,
    next_request_id: AtomicU64,
    request_timeout: Duration,
    model_ready_timeout: Duration,
    telemetry: Option<SidecarTelemetry>,
    /// When `Some`, every `call(…)` is delegated to the multiplexed
    /// transport instead of acquiring a slot from the pool. Lazily
    /// initialised so unit tests that never opt in pay no cost.
    mux: Option<Arc<MuxClient>>,
    /// Shared by pool and mux transports so concurrent oversized responses
    /// cannot reserve unbounded sidecar heap.
    response_chunk_budget: Arc<ResponseChunkBudget>,
    response_chunk_limits: ResponseChunkLimits,
}

impl IpcClient {
    /// Create a single-connection client. Equivalent to the pre-pool
    /// behaviour — kept for tests and simple callers; production
    /// configuration should prefer [`IpcClient::new_pool`].
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self::new_pool(socket_path, DEFAULT_POOL_SIZE)
    }

    /// Create an N-connection client. `pool_size` is clamped to `>= 1`.
    pub fn new_pool(socket_path: impl Into<PathBuf>, pool_size: usize) -> Self {
        let capacity = pool_size.max(1);
        let socket_path = socket_path.into();
        let response_chunk_budget = ResponseChunkBudget::production();
        let mux = if mux_enabled() {
            info!(
                socket = %socket_path.display(),
                "ipc_client: SIE_IPC_MUX=1 — using multiplexed transport"
            );
            Some(Arc::new(MuxClient::spawn_with_budget(
                socket_path.clone(),
                Arc::clone(&response_chunk_budget),
            )))
        } else {
            None
        };
        Self {
            socket_path,
            pool: Arc::new(Pool::new(capacity, None)),
            next_request_id: AtomicU64::new(1),
            request_timeout: Duration::from_secs(60),
            model_ready_timeout: Duration::from_secs(60),
            telemetry: None,
            mux,
            response_chunk_budget,
            response_chunk_limits: ResponseChunkLimits::production(),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn with_model_ready_timeout(mut self, timeout: Duration) -> Self {
        self.model_ready_timeout = timeout;
        self
    }

    /// Attach the canonical telemetry facade used for one logical completion
    /// observation per RPC and negotiated response-chunk transfer.
    pub fn with_telemetry(mut self, telemetry: SidecarTelemetry) -> Self {
        if !telemetry.is_enabled() {
            self.pool = Arc::new(Pool::new(self.pool.capacity, None));
            self.telemetry = None;
            return self;
        }
        if let Some(mux) = &self.mux {
            // The slot pool is inactive in mux mode. Keep it telemetry-free so
            // dashboards never show capacity for a transport that cannot run.
            self.pool = Arc::new(Pool::new(self.pool.capacity, None));
            mux.attach_telemetry(telemetry.clone());
        } else {
            self.response_chunk_budget
                .attach_telemetry(telemetry.clone());
            self.pool = Arc::new(Pool::new(self.pool.capacity, Some(telemetry.clone())));
            telemetry.ipc_transport_registered("pool", self.pool.capacity);
        }
        self.telemetry = Some(telemetry);
        self
    }

    /// Override pool size post-construction. Useful when the size is
    /// parsed from env in `main.rs` after `IpcClient::new` was chosen
    /// for a reasonable default. Clamped to `>= 1`.
    pub fn with_pool_size(mut self, pool_size: usize) -> Self {
        let previous_capacity = self.pool.capacity;
        let capacity = pool_size.max(1);
        let pool_telemetry = if self.mux.is_none() {
            self.telemetry.clone()
        } else {
            None
        };
        self.pool = Arc::new(Pool::new(capacity, pool_telemetry));
        if self.mux.is_none() {
            if let Some(telemetry) = &self.telemetry {
                telemetry.ipc_transport_resized("pool", previous_capacity, capacity);
            }
        }
        self
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Configured pool capacity (≥ 1).
    pub fn pool_size(&self) -> usize {
        self.pool.capacity
    }

    /// Drop every idle pooled socket. Used on shutdown or after a hard
    /// protocol error. Individual transport glitches during a live RPC
    /// are handled in [`IpcClient::call`] and only affect the slot they
    /// happened on. Slots currently checked out are left alone; they
    /// will be reconnected lazily by their caller's `call_inner` on
    /// next use after it has observed the transport error.
    pub async fn reset(&self) {
        self.pool.reset_idle();
    }

    fn next_id(&self) -> String {
        let n = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        format!("w-{n}")
    }

    /// Low-level request/response — send one frame, read one frame.
    ///
    /// Resilience: on an I/O-layer error (broken pipe / reset / UDS EOF /
    /// msgpack decode on a half-read buffer) we reset **the offending
    /// slot only** and transparently retry exactly once, preferring a
    /// fresh slot from the pool so the retry isn't gated on the still-
    /// breaking connection. This avoids NAKing an entire batch just
    /// because the backend side restarted between two fetches, while
    /// also not taking out the other N-1 slots.
    ///
    /// We do *not* retry:
    /// * `Timeout` — if the backend is slow once, it'll likely be slow again,
    ///   and a retry would double our tail latency.
    /// * `Server` / `VersionMismatch` — these are logical errors that
    ///   won't go away on retry.
    /// * `FrameTooLarge` — signals a bug, not a transport glitch.
    /// * `Encode` — serialization on our side; retry won't help.
    ///
    /// We serialize the request envelope exactly once at the top of `call`
    /// and reuse those bytes for the retry. The previous implementation
    /// took `body: Req` by value and `body.clone()`d before the first send
    /// "just in case" a retry was needed — an O(batch) allocation on every
    /// successful RPC. Trait bound `Req: Clone` was removed with this
    /// refactor; only `Serialize` remains.
    pub async fn call<Req, Resp>(&self, method: &str, body: Req) -> Result<Resp, IpcError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        self.call_with_timeout(method, body, self.request_timeout)
            .await
    }

    async fn call_with_timeout<Req, Resp>(
        &self,
        method: &str,
        body: Req,
        request_timeout: Duration,
    ) -> Result<Resp, IpcError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let request_id = self.next_id();
        let envelope = RequestEnvelope {
            version: IPC_VERSION,
            method,
            request_id: request_id.clone(),
            accepts_ipc_response_chunks_v1: true,
            body,
        };
        let payload = rmp_serde::to_vec_named(&envelope)?;
        request_frame_len(payload.len())?;
        debug!(method, request_id = %request_id, bytes = payload.len(), "ipc request");

        let start = std::time::Instant::now();

        // Multiplexed transport (SIE_IPC_MUX=1): single shared
        // connection with id-based response demuxing. The mux actor
        // owns reconnect; we still wrap the call in our own timeout
        // and apply a single transport-class retry to mirror the
        // pool path's "fresh slot" recovery semantics. The retry
        // reuses the same payload bytes (and request_id): the actor
        // drained its inflight map on disconnect, so the original
        // id is no longer registered and the new attempt won't be
        // confused with a stale response.
        if let Some(mux) = &self.mux {
            let first = self
                .mux_call::<Resp>(mux, method, request_id.clone(), &payload, request_timeout)
                .await;
            return match first {
                Ok(v) => {
                    self.record_rpc(method, start, "ok");
                    Ok(v)
                }
                Err(e) if is_transport_error(&e) => {
                    info!(
                        method,
                        error = %ErrChain(&e),
                        "mux IPC transport error — retrying once"
                    );
                    match self
                        .mux_call::<Resp>(mux, method, request_id, &payload, request_timeout)
                        .await
                    {
                        Ok(v) => {
                            self.record_rpc(method, start, "ok_after_retry");
                            Ok(v)
                        }
                        Err(e) => {
                            self.record_rpc(method, start, error_label(&e));
                            Err(e)
                        }
                    }
                }
                Err(e) => {
                    self.record_rpc(method, start, error_label(&e));
                    Err(e)
                }
            };
        }

        // First attempt on some slot. We arm() before the I/O starts
        // so a cancellation/panic between here and disarm() forces a
        // reconnect on that slot (protocol-resync safety).
        let mut slot_guard = self.pool.acquire().await;
        slot_guard.arm();
        let first = timeout(
            request_timeout,
            self.call_inner::<Resp>(method, &request_id, &payload, slot_guard.slot_mut()),
        )
        .await;
        let first = match first {
            Ok(r) => r,
            Err(_) => {
                // Timeout → the slot's UDS is in an indeterminate state
                // (we may be mid-write, or Python may have stalled mid-
                // read). The guard is still armed so Drop will clear
                // the stream; drop() now to release the slot.
                drop(slot_guard);
                self.record_rpc(method, start, "timeout");
                return Err(IpcError::Timeout);
            }
        };
        match first {
            Ok(v) => {
                // Clean protocol completion — don't reset the socket.
                slot_guard.disarm();
                drop(slot_guard);
                self.record_rpc(method, start, "ok");
                Ok(v)
            }
            Err(e) if is_transport_error(&e) => {
                info!(
                    method,
                    slot = slot_guard.slot_mut().id,
                    error = %ErrChain(&e),
                    "IPC transport error — resetting slot and retrying once"
                );
                // Leave armed so Drop clears the stream; checkout
                // again for the retry so we preferentially land on a
                // different slot. With pool_size=1 this is the same
                // slot (will reconnect on the retry's ensure path).
                drop(slot_guard);
                let mut retry_guard = self.pool.acquire().await;
                retry_guard.arm();
                let second = timeout(
                    request_timeout,
                    self.call_inner::<Resp>(method, &request_id, &payload, retry_guard.slot_mut()),
                )
                .await;
                match second {
                    Ok(Ok(v)) => {
                        retry_guard.disarm();
                        drop(retry_guard);
                        self.record_rpc(method, start, "ok_after_retry");
                        Ok(v)
                    }
                    Ok(Err(e)) => {
                        // Armed → Drop clears the stream.
                        drop(retry_guard);
                        self.record_rpc(method, start, error_label(&e));
                        Err(e)
                    }
                    Err(_) => {
                        // Armed → Drop clears the stream.
                        drop(retry_guard);
                        self.record_rpc(method, start, "timeout");
                        Err(IpcError::Timeout)
                    }
                }
            }
            Err(e) => {
                // Non-transport error from call_inner (Server /
                // VersionMismatch / Decode / FrameTooLarge / Encode).
                // For Decode / FrameTooLarge the stream may be
                // desynchronized, so leave the guard armed → Drop
                // clears it. For Server / VersionMismatch / Encode
                // the protocol completed cleanly, so disarm.
                if !matches!(
                    e,
                    IpcError::Decode(_) | IpcError::FrameTooLarge(_) | IpcError::ResponseChunk(_)
                ) {
                    slot_guard.disarm();
                }
                drop(slot_guard);
                self.record_rpc(method, start, error_label(&e));
                Err(e)
            }
        }
    }

    /// Send one request through the multiplexed transport and decode
    /// the typed response. Mirrors `call_inner` for the pool path:
    /// returns the same [`IpcError`] taxonomy so the caller's error
    /// handling is identical regardless of transport.
    async fn mux_call<Resp>(
        &self,
        mux: &MuxClient,
        _method: &str,
        request_id: String,
        payload: &[u8],
        request_timeout: Duration,
    ) -> Result<Resp, IpcError>
    where
        Resp: DeserializeOwned,
    {
        let raw = match timeout(
            request_timeout,
            mux.call_assembled(request_id.clone(), payload.to_vec()),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => return Err(IpcError::Timeout),
        };
        let frame = match raw {
            Ok(f) => f,
            Err(MuxError::Io(e)) => return Err(IpcError::Io(e)),
            Err(MuxError::FrameTooLarge(n)) => return Err(IpcError::FrameTooLarge(n)),
            Err(MuxError::ResponseChunk(e)) => return Err(IpcError::ResponseChunk(e)),
            Err(MuxError::ConnectionLost) => {
                return Err(IpcError::Io(std::io::Error::new(
                    std::io::ErrorKind::ConnectionReset,
                    "mux: connection lost while waiting for response",
                )));
            }
            Err(MuxError::Shutdown) => {
                return Err(IpcError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotConnected,
                    "mux: actor exited",
                )));
            }
        };
        match decode_response_frame::<Resp>(
            frame.as_slice(),
            &request_id,
            self.response_chunk_limits,
        )? {
            DecodedResponseFrame::Legacy(response) => Ok(response),
            DecodedResponseFrame::Chunk => Err(IpcError::ResponseChunk(
                "exact v1 chunk escaped mux response assembly".to_string(),
            )),
        }
    }

    fn record_rpc(&self, method: &str, start: std::time::Instant, result: &str) {
        let elapsed = start.elapsed();
        if let Some(telemetry) = &self.telemetry {
            telemetry.ipc_completed(method, result, elapsed);
        }
        let elapsed_ms = elapsed.as_millis();
        if elapsed_ms >= SLOW_RPC_WARN_MS && (result == "ok" || result == "ok_after_retry") {
            warn!(
                method,
                result,
                elapsed_ms = elapsed_ms as u64,
                threshold_ms = SLOW_RPC_WARN_MS as u64,
                "slow IPC RPC — backend side took longer than expected"
            );
        }
    }

    async fn call_inner<Resp>(
        &self,
        _method: &str,
        request_id: &str,
        payload: &[u8],
        slot: &mut Slot,
    ) -> Result<Resp, IpcError>
    where
        Resp: DeserializeOwned,
    {
        // The slot is checked out exclusively to this call, so no lock
        // is needed around the I/O. Lazy-connect on first use (or after
        // a prior transport error cleared the stream).
        if slot.stream.is_none() {
            match UnixStream::connect(&self.socket_path).await {
                Ok(s) => {
                    info!(
                        socket = %self.socket_path.display(),
                        slot = slot.id,
                        "connected IPC socket"
                    );
                    slot.stream = Some(s);
                }
                Err(e) => {
                    warn!(
                        socket = %self.socket_path.display(),
                        slot = slot.id,
                        error = %ErrChain(&e),
                        "IPC connect failed — will surface to caller for retry"
                    );
                    return Err(IpcError::Io(e));
                }
            }
        }
        let stream = slot
            .stream
            .as_mut()
            .expect("stream was just populated under exclusive slot ownership");

        let len = request_frame_len(payload.len())?;
        stream.write_all(&len.to_be_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        let first_frame = read_frame(stream).await?;
        match decode_response_frame::<Resp>(&first_frame, request_id, self.response_chunk_limits)? {
            DecodedResponseFrame::Legacy(response) => return Ok(response),
            DecodedResponseFrame::Chunk => {}
        }

        let mut assembler = ResponseAssembler::new_with_telemetry_and_limits(
            request_id,
            Arc::clone(&self.response_chunk_budget),
            self.telemetry.clone(),
            self.response_chunk_limits,
        );
        let mut frame = first_frame;
        let resp_buf: AssembledResponse = loop {
            match assembler
                .push(frame)
                .map_err(|error| IpcError::ResponseChunk(error.to_string()))?
            {
                ResponseFrameStatus::Pending => frame = read_frame(stream).await?,
                ResponseFrameStatus::Complete(bytes) => break bytes,
            }
        };

        match decode_response_frame::<Resp>(
            resp_buf.as_slice(),
            request_id,
            self.response_chunk_limits,
        )? {
            DecodedResponseFrame::Legacy(response) => Ok(response),
            DecodedResponseFrame::Chunk => Err(IpcError::ResponseChunk(
                "exact v1 chunk escaped pool response assembly".to_string(),
            )),
        }
    }

    /// Process one streaming generation request. Unlike normal IPC RPCs,
    /// this method receives a sequence of response envelopes:
    /// `publish` / `in_progress` / `ack` / `nak` events followed by
    /// `done`. The Rust sidecar owns the real NATS client and JetStream
    /// message settlement; the backend owns generation adapter execution and
    /// chunk encoding.
    pub async fn process_generate<F, Fut>(
        &self,
        req: ProcessGenerateRequest,
        mut on_event: F,
    ) -> Result<(), IpcError>
    where
        F: FnMut(GenerateEvent) -> Fut,
        Fut: std::future::Future<Output = Result<(), IpcError>>,
    {
        let request_id = self.next_id();
        let envelope = RequestEnvelope {
            version: IPC_VERSION,
            method: METHOD_PROCESS_GENERATE,
            request_id: request_id.clone(),
            accepts_ipc_response_chunks_v1: false,
            body: req,
        };
        let payload = rmp_serde::to_vec_named(&envelope)?;
        request_frame_len(payload.len())?;
        let start = std::time::Instant::now();
        let result = self
            .process_generate_inner(&request_id, &payload, &mut on_event)
            .await;
        let label = match &result {
            Ok(()) => "ok",
            Err(e) => error_label(e),
        };
        self.record_rpc(METHOD_PROCESS_GENERATE, start, label);
        result
    }

    async fn process_generate_inner<F, Fut>(
        &self,
        request_id: &str,
        payload: &[u8],
        on_event: &mut F,
    ) -> Result<(), IpcError>
    where
        F: FnMut(GenerateEvent) -> Fut,
        Fut: std::future::Future<Output = Result<(), IpcError>>,
    {
        let mut stream = UnixStream::connect(&self.socket_path).await?;
        let len = request_frame_len(payload.len())?;
        stream.write_all(&len.to_be_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        loop {
            let frame = timeout(generate_frame_timeout(), read_frame(&mut stream)).await;
            let resp_buf = match frame {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(IpcError::Timeout),
            };
            let envelope: ResponseEnvelope<GenerateEvent> = rmp_serde::from_slice(&resp_buf)?;
            if envelope.version != IPC_VERSION {
                return Err(IpcError::VersionMismatch {
                    got: envelope.version,
                });
            }
            if envelope.request_id != request_id {
                return Err(IpcError::Server(format!(
                    "generate stream response request_id mismatch: got {}, expected {}",
                    envelope.request_id, request_id
                )));
            }
            if !envelope.ok {
                return Err(IpcError::Server(envelope.error.unwrap_or_default()));
            }
            let event = envelope.body.ok_or_else(|| {
                IpcError::Server("generate stream ok=true but body missing".to_string())
            })?;
            if event.kind == "done" {
                return Ok(());
            }
            on_event(event).await?;
        }
    }

    // -- Typed method wrappers -------------------------------------------

    pub async fn ping(&self, timestamp_ms: f64) -> Result<PingResponse, IpcError> {
        self.call(METHOD_PING, PingRequest { timestamp_ms }).await
    }

    pub async fn ensure_model_ready(
        &self,
        model_id: impl Into<String>,
    ) -> Result<EnsureModelReadyResponse, IpcError> {
        self.call_with_timeout(
            METHOD_ENSURE_MODEL_READY,
            EnsureModelReadyRequest {
                model_id: model_id.into(),
            },
            self.model_ready_timeout,
        )
        .await
    }

    pub async fn worker_capabilities(&self) -> Result<WorkerCapabilitiesResponse, IpcError> {
        self.call(METHOD_WORKER_CAPABILITIES, WorkerCapabilitiesRequest {})
            .await
    }

    pub async fn signal_generate_cancel(
        &self,
        request_id: impl Into<String>,
    ) -> Result<SignalGenerateCancelResponse, IpcError> {
        self.call(
            METHOD_SIGNAL_GENERATE_CANCEL,
            SignalGenerateCancelRequest {
                request_id: request_id.into(),
            },
        )
        .await
    }

    pub async fn process_encode_batch(
        &self,
        req: ProcessEncodeBatchRequest,
    ) -> Result<BatchOutcome, IpcError> {
        self.call(METHOD_PROCESS_ENCODE_BATCH, req).await
    }

    pub async fn process_score_batch(
        &self,
        req: ProcessScoreBatchRequest,
    ) -> Result<BatchOutcome, IpcError> {
        self.call(METHOD_PROCESS_SCORE_BATCH, req).await
    }

    pub async fn process_extract_batch(
        &self,
        req: ProcessExtractBatchRequest,
    ) -> Result<BatchOutcome, IpcError> {
        self.call(METHOD_PROCESS_EXTRACT_BATCH, req).await
    }

    /// Dispatch a pre-formed batch through the `RunBatch` RPC. Every
    /// batch that comes out of the Rust-side
    /// scheduler is shipped through this path; legacy per-op paths
    /// (`process_*_batch`) stay available for backends that don't
    /// implement `run_batch` and for unit tests that don't wire a
    /// scheduler.
    pub async fn run_batch(&self, req: RunBatchRequest) -> Result<BatchOutcome, IpcError> {
        self.call(METHOD_RUN_BATCH, req).await
    }

    pub async fn apply_model_config(
        &self,
        req: ApplyModelConfigRequest,
    ) -> Result<ApplyModelConfigResponse, IpcError> {
        self.call(METHOD_APPLY_MODEL_CONFIG, req).await
    }

    pub async fn replace_model_configs(
        &self,
        req: ReplaceModelConfigsRequest,
    ) -> Result<ReplaceModelConfigsResponse, IpcError> {
        self.call(METHOD_REPLACE_MODEL_CONFIGS, req).await
    }

    pub async fn set_pinned_models(
        &self,
        models: Vec<String>,
    ) -> Result<SetPinnedModelsResponse, IpcError> {
        self.call(METHOD_SET_PINNED_MODELS, SetPinnedModelsRequest { models })
            .await
    }

    pub async fn drain(&self, deadline_ms: u64) -> Result<DrainResponse, IpcError> {
        self.call(METHOD_DRAIN, DrainRequest { deadline_ms }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::IpcResponseChunkV1;
    use crate::protocol::response_chunks::IPC_RESPONSE_CHUNK_KIND_V1;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;
    use tokio::time::Instant;

    #[test]
    fn disabled_telemetry_is_not_retained_by_ipc_transports() {
        let client = IpcClient::new(short_sock_path()).with_telemetry(SidecarTelemetry::default());

        assert!(client.telemetry.is_none());
        assert!(client.pool.telemetry.is_none());
    }

    #[test]
    fn request_frame_len_enforces_exact_serialized_limit() {
        assert_eq!(
            request_frame_len(MAX_FRAME_BYTES).unwrap(),
            MAX_FRAME_BYTES as u32
        );
        let error = request_frame_len(MAX_FRAME_BYTES + 1).unwrap_err();
        assert!(
            matches!(error, IpcError::FrameTooLarge(size) if size == (MAX_FRAME_BYTES + 1) as u32)
        );
    }

    #[test]
    fn maximum_audio_run_batch_fits_ipc_frame() {
        const MAX_DURATION_SECONDS: u64 = 12 * 60;
        let sample_count = u64::from(sie_audio_prep::TARGET_SAMPLE_RATE) * MAX_DURATION_SECONDS;
        let audio = crate::ipc_types::PreparedAudioPcm16 {
            pcm_s16le: vec![0; sample_count as usize * 2],
            sample_rate: sie_audio_prep::TARGET_SAMPLE_RATE,
            sample_count,
            duration_ms: MAX_DURATION_SECONDS * 1_000,
            source_sample_rate: 48_000,
            source_sample_count: 48_000 * MAX_DURATION_SECONDS,
            source_channels: 2,
            container: "wav".to_owned(),
        };
        let item = crate::ipc_types::ExtractBatchItem {
            work_item_id: "request.0".to_owned(),
            request_id: "request".to_owned(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: rmpv::Value::Map(Vec::new()),
            labels: None,
            output_schema: None,
            instruction: None,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
            prepared_audio: Some(audio),
        };
        let batch = RunBatchRequest {
            model_id: "openai/whisper-large-v3-turbo".to_owned(),
            batch_id: 1,
            lora_key: String::new(),
            total_cost: MAX_DURATION_SECONDS,
            items: vec![crate::ipc_types::RunBatchItem::extract(item)],
            accepts_batched_f16_multivectors: false,
        };
        let envelope = RequestEnvelope {
            version: IPC_VERSION,
            method: METHOD_RUN_BATCH,
            request_id: "ipc-request".to_owned(),
            accepts_ipc_response_chunks_v1: false,
            body: batch,
        };

        let payload = rmp_serde::to_vec_named(&envelope).unwrap();
        request_frame_len(payload.len()).expect("maximum legal audio must fit IPC");
        assert!(payload.len() < MAX_FRAME_BYTES);
    }

    fn short_sock_path() -> PathBuf {
        // Match the Python test helper — AF_UNIX paths on macOS are <=104 chars.
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
        let base = if base.len() > 20 {
            "/tmp".to_string()
        } else {
            base
        };
        let name = format!("sie-w-{}.sock", uuid::Uuid::new_v4().simple());
        PathBuf::from(base).join(name)
    }

    /// Tiny in-process echo server: reads a request frame and replies with
    /// a fixed response envelope. Enough to exercise framing + typed call.
    async fn spawn_echo<F>(path: PathBuf, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(&[u8]) -> Vec<u8> + Send + Sync + 'static,
    {
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let resp = (handler)(&buf);
                        let len = (resp.len() as u32).to_be_bytes();
                        if sock.write_all(&len).await.is_err() {
                            return;
                        }
                        if sock.write_all(&resp).await.is_err() {
                            return;
                        }
                        if sock.flush().await.is_err() {
                            return;
                        }
                    }
                });
            }
        })
    }

    /// Echo server variant whose per-frame handler can await — lets tests
    /// inject artificial per-request latency to exercise concurrency.
    async fn spawn_echo_async<F, Fut>(path: PathBuf, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<u8>> + Send + 'static,
    {
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let resp = (handler)(buf).await;
                        let len = (resp.len() as u32).to_be_bytes();
                        if sock.write_all(&len).await.is_err() {
                            return;
                        }
                        if sock.write_all(&resp).await.is_err() {
                            return;
                        }
                        if sock.flush().await.is_err() {
                            return;
                        }
                    }
                });
            }
        })
    }

    fn ping_reply_bytes(req_bytes: &[u8], worker_id: &str) -> Vec<u8> {
        let req: serde_json::Value = rmp_serde::from_slice(req_bytes).unwrap();
        let request_id = req["request_id"].as_str().unwrap().to_string();
        let envelope = serde_json::json!({
            "version": IPC_VERSION,
            "request_id": request_id,
            "ok": true,
            "body": serde_json::json!({"timestamp_ms": 0.0, "worker_id": worker_id}),
            "error": serde_json::Value::Null,
        });
        rmp_serde::to_vec_named(&envelope).unwrap()
    }

    #[test]
    fn legacy_response_decodes_directly_and_preserves_wire_bytes() {
        let frame = rmp_serde::to_vec_named(&serde_json::json!({
            "version": IPC_VERSION,
            "request_id": "r-legacy",
            "ok": true,
            "body": {"value": [1, 2, 3]},
            "error": null,
        }))
        .unwrap();
        let original = frame.clone();

        let decoded = decode_response_frame::<serde_json::Value>(
            &frame,
            "r-legacy",
            ResponseChunkLimits::production(),
        )
        .unwrap();

        let DecodedResponseFrame::Legacy(body) = decoded else {
            panic!("ordinary response must stay on the one-decode legacy path");
        };
        assert_eq!(body, serde_json::json!({"value": [1, 2, 3]}));
        assert_eq!(
            frame, original,
            "legacy decoding must not rewrite the frame"
        );
    }

    #[test]
    fn exact_chunk_kind_cannot_hide_inside_a_typed_legacy_envelope() {
        let frame = rmp_serde::to_vec_named(&serde_json::json!({
            "version": IPC_VERSION,
            "request_id": "r-chunk",
            "ok": true,
            "body": {"value": 1},
            "error": null,
            "kind": IPC_RESPONSE_CHUNK_KIND_V1,
        }))
        .unwrap();

        let decoded = decode_response_frame::<serde_json::Value>(
            &frame,
            "r-chunk",
            ResponseChunkLimits::production(),
        )
        .unwrap();

        assert!(matches!(decoded, DecodedResponseFrame::Chunk));
    }

    #[tokio::test]
    async fn ping_roundtrips_typed() {
        let path = short_sock_path();
        let path_clone = path.clone();
        let _server = spawn_echo(path_clone, |req_bytes| {
            let req: serde_json::Value = rmp_serde::from_slice(req_bytes).unwrap();
            let request_id = req["request_id"].as_str().unwrap().to_string();
            let body = rmp_serde::to_vec_named(&serde_json::json!({
                "timestamp_ms": 999.0,
                "worker_id": "echo",
            }))
            .unwrap();
            let envelope = serde_json::json!({
                "version": IPC_VERSION,
                "request_id": request_id,
                "ok": true,
                "body": rmp_serde::from_slice::<serde_json::Value>(&body).unwrap(),
                "error": serde_json::Value::Null,
            });
            rmp_serde::to_vec_named(&envelope).unwrap()
        })
        .await;

        tokio::task::yield_now().await;
        let client = IpcClient::new(path);
        let resp = client.ping(42.0).await.unwrap();
        assert_eq!(resp.worker_id, "echo");
        assert_eq!(resp.timestamp_ms, 999.0);
    }

    #[tokio::test]
    async fn pool_call_negotiates_and_reassembles_chunked_response() {
        let path = short_sock_path();
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut len = [0_u8; 4];
            socket.read_exact(&mut len).await.unwrap();
            let mut request = vec![0_u8; u32::from_be_bytes(len) as usize];
            socket.read_exact(&mut request).await.unwrap();
            let request: serde_json::Value = rmp_serde::from_slice(&request).unwrap();
            assert_eq!(request["accepts_ipc_response_chunks_v1"], true);
            let request_id = request["request_id"].as_str().unwrap();
            let response =
                ping_reply_bytes(&rmp_serde::to_vec_named(&request).unwrap(), "chunked-pool");
            let digest = Sha256::digest(&response).to_vec();
            let split = response.len() / 2;
            for (index, payload) in [&response[..split], &response[split..]]
                .into_iter()
                .enumerate()
            {
                let frame = rmp_serde::to_vec_named(&IpcResponseChunkV1 {
                    version: IPC_VERSION,
                    request_id: request_id.to_owned(),
                    transfer_digest: digest.clone(),
                    chunk_index: index as u32,
                    chunk_count: 2,
                    total_bytes: response.len() as u64,
                    payload: payload.to_vec(),
                    kind: IPC_RESPONSE_CHUNK_KIND_V1.to_owned(),
                })
                .unwrap();
                socket
                    .write_all(&(frame.len() as u32).to_be_bytes())
                    .await
                    .unwrap();
                socket.write_all(&frame).await.unwrap();
                socket.flush().await.unwrap();
            }
        });

        let mut client = IpcClient::new(path);
        client.response_chunk_limits = ResponseChunkLimits::relaxed_for_small_fixtures();
        let response = client.ping(42.0).await.unwrap();
        assert_eq!(response.worker_id, "chunked-pool");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn server_error_surfaces_as_ipc_server() {
        let path = short_sock_path();
        let path_clone = path.clone();
        let _server = spawn_echo(path_clone, |req_bytes| {
            let req: serde_json::Value = rmp_serde::from_slice(req_bytes).unwrap();
            let request_id = req["request_id"].as_str().unwrap().to_string();
            let envelope = serde_json::json!({
                "version": IPC_VERSION,
                "request_id": request_id,
                "ok": false,
                "body": serde_json::Value::Null,
                "error": "boom",
            });
            rmp_serde::to_vec_named(&envelope).unwrap()
        })
        .await;

        tokio::task::yield_now().await;
        let client = IpcClient::new(path);
        let err = client.ping(1.0).await.unwrap_err();
        match err {
            IpcError::Server(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Server error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reconnect_retry_after_broken_pipe() {
        // Server drops the connection on the very first accepted frame, then
        // replies normally on subsequent connections. We expect the client
        // to transparently retry once and succeed.
        use std::sync::atomic::{AtomicU32, Ordering};

        let path = short_sock_path();
        let path_clone = path.clone();
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let connections = Arc::new(AtomicU32::new(0));
        let connections_clone = Arc::clone(&connections);

        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let n = connections_clone.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut len_buf = [0u8; 4];
                    if sock.read_exact(&mut len_buf).await.is_err() {
                        return;
                    }
                    let frame_len = u32::from_be_bytes(len_buf) as usize;
                    let mut buf = vec![0u8; frame_len];
                    if sock.read_exact(&mut buf).await.is_err() {
                        return;
                    }
                    if n == 0 {
                        drop(sock);
                        return;
                    }
                    let req: serde_json::Value = rmp_serde::from_slice(&buf).unwrap();
                    let request_id = req["request_id"].as_str().unwrap().to_string();
                    let envelope = serde_json::json!({
                        "version": IPC_VERSION,
                        "request_id": request_id,
                        "ok": true,
                        "body": serde_json::json!({"timestamp_ms": 1.0, "worker_id": "after-retry"}),
                        "error": serde_json::Value::Null,
                    });
                    let resp = rmp_serde::to_vec_named(&envelope).unwrap();
                    let len = (resp.len() as u32).to_be_bytes();
                    let _ = sock.write_all(&len).await;
                    let _ = sock.write_all(&resp).await;
                    let _ = sock.flush().await;
                });
            }
        });

        tokio::task::yield_now().await;
        let client = IpcClient::new(path);
        let resp = client.ping(1.0).await.expect("retry should succeed");
        assert_eq!(resp.worker_id, "after-retry");
        assert_eq!(connections.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn server_error_is_not_retried() {
        // A protocol-level `ok=false` response is a logical error — retrying
        // would just repeat the same error, so we should surface it on the
        // first attempt and leave the connection alive.
        use std::sync::atomic::{AtomicU32, Ordering};

        let path = short_sock_path();
        let path_clone = path.clone();
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let connections = Arc::new(AtomicU32::new(0));
        let connections_clone = Arc::clone(&connections);

        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                connections_clone.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let frame_len = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; frame_len];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let req: serde_json::Value = rmp_serde::from_slice(&buf).unwrap();
                        let request_id = req["request_id"].as_str().unwrap().to_string();
                        let envelope = serde_json::json!({
                            "version": IPC_VERSION,
                            "request_id": request_id,
                            "ok": false,
                            "body": serde_json::Value::Null,
                            "error": "logic error",
                        });
                        let resp = rmp_serde::to_vec_named(&envelope).unwrap();
                        let len = (resp.len() as u32).to_be_bytes();
                        if sock.write_all(&len).await.is_err() {
                            return;
                        }
                        if sock.write_all(&resp).await.is_err() {
                            return;
                        }
                        let _ = sock.flush().await;
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client = IpcClient::new(path);
        let err = client.ping(1.0).await.unwrap_err();
        assert!(matches!(err, IpcError::Server(_)));
        assert_eq!(connections.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn is_transport_error_classifies_correctly() {
        let io_broken = IpcError::Io(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
        assert!(is_transport_error(&io_broken));
        let io_reset = IpcError::Io(std::io::Error::from(std::io::ErrorKind::ConnectionReset));
        assert!(is_transport_error(&io_reset));
        let io_eof = IpcError::Io(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        assert!(is_transport_error(&io_eof));
        let server = IpcError::Server("x".into());
        assert!(!is_transport_error(&server));
        let ver = IpcError::VersionMismatch { got: 99 };
        assert!(!is_transport_error(&ver));
        let timeout = IpcError::Timeout;
        assert!(!is_transport_error(&timeout));
    }

    #[tokio::test]
    async fn concurrent_reset_does_not_panic_on_connect() {
        // Regression for a TOCTOU that existed in the single-socket
        // version. With the pool, reset() nukes idle slots' streams and
        // any live call just lazily reconnects on its own slot — never
        // panics, never deadlocks. This test hammers reset() against 32
        // concurrent pings and asserts neither.
        use std::sync::atomic::{AtomicBool, Ordering};

        let path = short_sock_path();
        let path_clone = path.clone();
        let _server = spawn_echo(path_clone, |req_bytes| ping_reply_bytes(req_bytes, "x")).await;

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new(path));
        let stop = Arc::new(AtomicBool::new(false));

        let resetter = {
            let client = Arc::clone(&client);
            let stop = Arc::clone(&stop);
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    client.reset().await;
                    tokio::task::yield_now().await;
                }
            })
        };

        let mut pings = Vec::with_capacity(32);
        for _ in 0..32 {
            let client = Arc::clone(&client);
            pings.push(tokio::spawn(async move { client.ping(1.0).await }));
        }
        for p in pings {
            let _ = p.await;
        }

        stop.store(true, Ordering::Relaxed);
        let _ = resetter.await;
    }

    #[tokio::test]
    async fn version_mismatch_is_detected() {
        let path = short_sock_path();
        let path_clone = path.clone();
        let _server = spawn_echo(path_clone, |req_bytes| {
            let req: serde_json::Value = rmp_serde::from_slice(req_bytes).unwrap();
            let request_id = req["request_id"].as_str().unwrap().to_string();
            let envelope = serde_json::json!({
                "version": 999,
                "request_id": request_id,
                "ok": true,
                "body": serde_json::json!({"timestamp_ms": 0.0, "worker_id": "x"}),
                "error": serde_json::Value::Null,
            });
            rmp_serde::to_vec_named(&envelope).unwrap()
        })
        .await;

        tokio::task::yield_now().await;
        let client = IpcClient::new(path);
        let err = client.ping(1.0).await.unwrap_err();
        assert!(matches!(err, IpcError::VersionMismatch { got: 999 }));
    }

    // -----------------------------------------------------------------
    //  Pool-specific tests (new).
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn pool_opens_independent_connections_per_slot() {
        // With pool_size=N and N concurrent blocking RPCs, the server must
        // accept exactly N connections — proving the client did not
        // serialize them on a single socket.
        use std::sync::atomic::{AtomicU32, Ordering};

        const N: usize = 4;
        const HOLD: Duration = Duration::from_millis(200);

        let path = short_sock_path();
        let path_clone = path.clone();
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let accepted = Arc::new(AtomicU32::new(0));
        let accepted_clone = Arc::clone(&accepted);

        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                accepted_clone.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        // Artificial latency so overlap is observable.
                        tokio::time::sleep(HOLD).await;
                        let resp = ping_reply_bytes(&buf, "pooled");
                        let len = (resp.len() as u32).to_be_bytes();
                        let _ = sock.write_all(&len).await;
                        let _ = sock.write_all(&resp).await;
                        let _ = sock.flush().await;
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new_pool(path, N));
        assert_eq!(client.pool_size(), N);

        let start = Instant::now();
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let client = Arc::clone(&client);
            handles.push(tokio::spawn(async move { client.ping(1.0).await }));
        }
        for h in handles {
            h.await.unwrap().expect("ping");
        }
        let elapsed = start.elapsed();

        // Upper bound: if requests were serialized on one socket it
        // would be ≥ N * HOLD = 800ms. With the pool they run in
        // parallel and should finish in ~HOLD + scheduling jitter.
        assert!(
            elapsed < HOLD * 2,
            "pooled pings took {:?}, expected ~{:?} (serialized would be ~{:?})",
            elapsed,
            HOLD,
            HOLD * N as u32,
        );

        // The server must have accepted one connection per slot.
        // `>= N` covers the benign case where a slot's reconnect
        // handler is still pending on shutdown, but on the steady
        // state path we expect exactly N.
        let n = accepted.load(Ordering::SeqCst);
        assert_eq!(n, N as u32, "expected exactly one accept per slot, got {n}",);
    }

    #[tokio::test]
    async fn pool_isolates_slow_rpcs_from_fast_ones() {
        // One slot is artificially slow (5s). Fire N-1 other calls; they
        // must complete on the other slots and must NOT wait for the
        // slow one. This is the head-of-line isolation guarantee that
        // rules out a naive round-robin scheme.
        use std::sync::atomic::{AtomicBool, Ordering};

        const N: usize = 4;
        const FAST: usize = N - 1;
        const SLOW_HOLD: Duration = Duration::from_secs(3);
        const FAST_BUDGET: Duration = Duration::from_millis(500);

        let path = short_sock_path();
        let path_clone = path.clone();
        let first_slot = Arc::new(AtomicBool::new(true));
        let first_slot_clone = Arc::clone(&first_slot);

        let _server = spawn_echo_async(path_clone, move |buf| {
            let is_first = first_slot_clone.swap(false, Ordering::SeqCst);
            async move {
                if is_first {
                    tokio::time::sleep(SLOW_HOLD).await;
                }
                ping_reply_bytes(&buf, if is_first { "slow" } else { "fast" })
            }
        })
        .await;

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new_pool(path, N));

        // Kick off the slow one first so it grabs slot 0.
        let slow_client = Arc::clone(&client);
        let slow = tokio::spawn(async move { slow_client.ping(1.0).await });

        // Give the slow ping a moment to be in-flight on its slot.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Now fire FAST concurrent pings; they must all return inside
        // FAST_BUDGET even though the slow slot is still holding.
        let start = Instant::now();
        let mut fasts = Vec::with_capacity(FAST);
        for _ in 0..FAST {
            let client = Arc::clone(&client);
            fasts.push(tokio::spawn(async move { client.ping(2.0).await }));
        }
        for h in fasts {
            h.await.unwrap().expect("fast ping");
        }
        let fast_elapsed = start.elapsed();
        assert!(
            fast_elapsed < FAST_BUDGET,
            "fast pings took {:?}, expected < {:?} — slow RPC was blocking the pool",
            fast_elapsed,
            FAST_BUDGET,
        );

        // Let the slow one finish so we don't leak its task.
        let _ = slow.await;
    }

    #[tokio::test]
    async fn pool_saturation_blocks_n_plus_one_until_a_slot_frees() {
        // Pool size = 2 with two in-flight slow RPCs. A third caller
        // must wait until one of the first two returns. This exercises
        // the Semaphore-based fair-FIFO acquire.
        const N: usize = 2;
        const HOLD: Duration = Duration::from_millis(300);

        let path = short_sock_path();
        let path_clone = path.clone();
        let _server = spawn_echo_async(path_clone, |buf| async move {
            tokio::time::sleep(HOLD).await;
            ping_reply_bytes(&buf, "held")
        })
        .await;

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new_pool(path, N));

        let c1 = Arc::clone(&client);
        let c2 = Arc::clone(&client);
        let h1 = tokio::spawn(async move { c1.ping(1.0).await });
        let h2 = tokio::spawn(async move { c2.ping(2.0).await });

        // Let h1 / h2 acquire the two slots.
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Third caller — must be forced to wait at acquire() for >= (HOLD - 30ms).
        let c3 = Arc::clone(&client);
        let third_start = Instant::now();
        let h3 = tokio::spawn(async move { c3.ping(3.0).await });

        h1.await.unwrap().expect("h1");
        h2.await.unwrap().expect("h2");
        h3.await.unwrap().expect("h3");
        let third_elapsed = third_start.elapsed();
        assert!(
            third_elapsed >= HOLD.saturating_sub(Duration::from_millis(80)),
            "third caller returned in {:?}; expected to wait at least ~{:?} \
             for a pool slot to free",
            third_elapsed,
            HOLD,
        );
    }

    #[tokio::test]
    async fn pool_reset_on_transport_error_is_scoped_to_one_slot() {
        // Force a transport error on exactly one connection: the per-
        // connection handler drops the socket on its first frame. With
        // pool_size=4, reset semantics must keep the other 3 slots'
        // sockets alive (no `reset_all`), and the client's one-shot
        // retry must succeed on a fresh slot.
        use std::sync::atomic::{AtomicUsize, Ordering};

        const N: usize = 4;

        let path = short_sock_path();
        let path_clone = path.clone();
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let connections_clone = Arc::clone(&connections);

        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let n = connections_clone.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    if n == 0 {
                        // Read one frame then drop — simulate a mid-
                        // session EOF on slot 0.
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let frame_len = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; frame_len];
                        let _ = sock.read_exact(&mut buf).await;
                        drop(sock);
                        return;
                    }
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let frame_len = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; frame_len];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let resp = ping_reply_bytes(&buf, "ok");
                        let len = (resp.len() as u32).to_be_bytes();
                        let _ = sock.write_all(&len).await;
                        let _ = sock.write_all(&resp).await;
                        let _ = sock.flush().await;
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new_pool(path, N));

        // First ping: hits slot 0, triggers retry after the drop, retry
        // lands on a different slot and succeeds.
        let resp = client.ping(1.0).await.expect("retry should succeed");
        assert_eq!(resp.worker_id, "ok");

        // Fire three more pings back-to-back — each should succeed on
        // first try using the healthy slots (1..N).
        for _ in 0..3 {
            let r = client.ping(2.0).await.expect("healthy slots keep working");
            assert_eq!(r.worker_id, "ok");
        }

        // Connection budget: 1 failed + at least 1 healthy for the
        // retry + up to 3 more for subsequent pings landing on already-
        // connected slots. The exact accept count is scheduling-order
        // dependent so we only assert the key bound: we did NOT take
        // out every slot (which would look like N+1 accepts at minimum,
        // one per slot's reconnect after a reset_all).
        let accepted = connections.load(Ordering::SeqCst);
        assert!(
            accepted >= 2,
            "expected at least 2 connections (1 failing + 1 retry), got {accepted}"
        );
        assert!(
            accepted < N * 2,
            "accept count {accepted} suggests reset_all fired; should be scoped to one slot"
        );
    }

    #[tokio::test]
    async fn pool_size_one_preserves_legacy_semantics() {
        // Belt-and-suspenders: `new()` must behave exactly like the
        // pre-pool version — single connection, strict serialization
        // of concurrent callers. This test locks the contract in.
        const HOLD: Duration = Duration::from_millis(150);

        let path = short_sock_path();
        let path_clone = path.clone();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_clone = Arc::clone(&accepted);
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();

        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                accepted_clone.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        tokio::time::sleep(HOLD).await;
                        let resp = ping_reply_bytes(&buf, "single");
                        let len = (resp.len() as u32).to_be_bytes();
                        let _ = sock.write_all(&len).await;
                        let _ = sock.write_all(&resp).await;
                        let _ = sock.flush().await;
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new(path));
        assert_eq!(client.pool_size(), 1);

        let start = Instant::now();
        let h1 = {
            let c = Arc::clone(&client);
            tokio::spawn(async move { c.ping(1.0).await })
        };
        let h2 = {
            let c = Arc::clone(&client);
            tokio::spawn(async move { c.ping(2.0).await })
        };
        h1.await.unwrap().expect("h1");
        h2.await.unwrap().expect("h2");
        let elapsed = start.elapsed();

        // Serialized: should take ~2 * HOLD.
        assert!(
            elapsed >= HOLD * 2 - Duration::from_millis(40),
            "legacy single-connection client should serialize; took only {:?}",
            elapsed,
        );
        // Exactly one connection opened (serialized sharing).
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    // =====================================================================
    // Stress, cancellation, and invariant tests
    // =====================================================================
    //
    // The tests above lock in the happy-path pool semantics. The ones
    // below stress it the way production will: hundreds of concurrent
    // callers, futures being cancelled mid-RPC by caller-side
    // deadlines, oversubscribed pools, and post-burst metric
    // consistency. If any of these regress we will get correctness
    // bugs in prod — particularly the cancellation tests.
    // =====================================================================

    /// Stress: N=8 pool, 256 concurrent callers, every one must
    /// complete successfully. Exercises semaphore contention, FIFO
    /// slot queueing, and metric correctness under heavy load.
    #[tokio::test]
    async fn stress_many_concurrent_callers_all_complete() {
        const POOL: usize = 8;
        const CONCURRENCY: usize = 256;
        const SERVER_DELAY: Duration = Duration::from_millis(5);

        let path = short_sock_path();
        let _server = spawn_echo_async(path.clone(), |req| async move {
            tokio::time::sleep(SERVER_DELAY).await;
            ping_reply_bytes(&req, "stress")
        })
        .await;
        tokio::task::yield_now().await;

        let client = Arc::new(IpcClient::new_pool(path, POOL));
        assert_eq!(client.pool_size(), POOL);

        let mut handles = Vec::with_capacity(CONCURRENCY);
        for i in 0..CONCURRENCY {
            let c = Arc::clone(&client);
            handles.push(tokio::spawn(async move { c.ping(i as f64).await }));
        }
        for (i, h) in handles.into_iter().enumerate() {
            h.await
                .unwrap_or_else(|e| panic!("task {i} panicked: {e:?}"))
                .unwrap_or_else(|e| panic!("task {i} ping failed: {e:?}"));
        }

        // Same client still works after the storm — slots weren't
        // corrupted.
        client.ping(9999.0).await.expect("post-stress ping");
    }

    /// Cancellation safety: if the caller drops the `call` Future
    /// mid-RPC (what happens on a Tokio timeout wrapper, or gateway
    /// deadline), the slot MUST be returned to the pool and the
    /// next caller on that slot must not get a corrupted socket.
    ///
    /// This is the single most important concurrency invariant. A
    /// regression here silently poisons the pool in production and
    /// every subsequent request on the recycled slot sees either a
    /// decode error or a stall.
    #[tokio::test]
    async fn dropping_future_mid_rpc_does_not_poison_slot() {
        const POOL: usize = 2;
        // Server holds responses just long enough that our 20ms
        // caller deadline fires first. 150ms picked to dwarf any
        // scheduler jitter.
        const SERVER_HOLD: Duration = Duration::from_millis(150);

        let path = short_sock_path();
        let _server = spawn_echo_async(path.clone(), |req| async move {
            tokio::time::sleep(SERVER_HOLD).await;
            ping_reply_bytes(&req, "cancel-harness")
        })
        .await;
        tokio::task::yield_now().await;

        let client = Arc::new(IpcClient::new_pool(path, POOL));

        // Burn both pool slots with calls that will be cancelled.
        for i in 0..POOL {
            let c = Arc::clone(&client);
            let res = tokio::time::timeout(Duration::from_millis(20), c.ping(i as f64)).await;
            assert!(
                res.is_err(),
                "iter {i}: expected caller-side timeout, got {res:?}"
            );
        }
        // Let Drop run and any server-side IO settle.
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Fire POOL+2 sequential calls on the "poisoned" slots — each
        // must get a clean response, not a stale half-frame.
        for i in 0..(POOL + 2) {
            client
                .ping((1000 + i) as f64)
                .await
                .unwrap_or_else(|e| panic!("post-cancel ping {i} got stale/corrupt slot: {e:?}"));
        }
    }

    /// Timeout path: when `request_timeout` elapses the call returns
    /// `IpcError::Timeout`, but the slot must be returned and
    /// reusable. The socket is cleared so the next call reconnects.
    #[tokio::test]
    async fn timeout_returns_slot_and_allows_reuse() {
        const POOL: usize = 1;
        // Server: first request hangs forever, later requests answer
        // immediately. Count accepted connections so we can assert
        // that the post-timeout call reconnected on a fresh socket.
        let path = short_sock_path();
        let path_clone = path.clone();
        let accept_count = Arc::new(AtomicUsize::new(0));
        let accept_count_clone = Arc::clone(&accept_count);
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let conn_idx = Arc::new(AtomicUsize::new(0));
        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                accept_count_clone.fetch_add(1, Ordering::SeqCst);
                let idx = conn_idx.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        if idx == 0 {
                            // Hang: never reply. The client-side
                            // request_timeout will fire.
                            std::future::pending::<()>().await;
                            return;
                        }
                        let resp = ping_reply_bytes(&buf, "timeout-reuse");
                        let len = (resp.len() as u32).to_be_bytes();
                        let _ = sock.write_all(&len).await;
                        let _ = sock.write_all(&resp).await;
                        let _ = sock.flush().await;
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client =
            Arc::new(IpcClient::new_pool(path, POOL).with_timeout(Duration::from_millis(50)));

        // First call: times out inside IpcClient::call.
        let err = client.ping(1.0).await.unwrap_err();
        assert!(matches!(err, IpcError::Timeout), "got {err:?}");
        // Second call: must get a brand new connection (the hung
        // socket was clobbered by arm+Drop), not hang on the
        // still-alive but desynchronized old stream.
        let ok = client.ping(2.0).await.expect("post-timeout ping");
        // Worker id baked into the reply proves we reached the
        // "later connection" branch of the server.
        assert_eq!(ok.worker_id, "timeout-reuse");
        // We should have seen at least two connections: the
        // original (hung) and the reconnect.
        assert!(
            accept_count.load(Ordering::SeqCst) >= 2,
            "expected reconnection after timeout, but server only saw {} connections",
            accept_count.load(Ordering::SeqCst)
        );
    }

    /// Fairness / no-starvation: with K callers >> N slots, every
    /// caller must eventually complete. Catches scheduler bugs where
    /// a single slot monopolizes the semaphore or the wake-chain
    /// gets broken.
    #[tokio::test]
    async fn oversubscribed_pool_drains_without_starvation() {
        const POOL: usize = 4;
        const CALLERS: usize = 64;
        const SERVER_DELAY: Duration = Duration::from_millis(20);

        let path = short_sock_path();
        let _server = spawn_echo_async(path.clone(), |req| async move {
            tokio::time::sleep(SERVER_DELAY).await;
            ping_reply_bytes(&req, "fair")
        })
        .await;
        tokio::task::yield_now().await;

        let client = Arc::new(IpcClient::new_pool(path, POOL));

        // Fan out all callers simultaneously.
        let mut handles = Vec::with_capacity(CALLERS);
        let start = Instant::now();
        for i in 0..CALLERS {
            let c = Arc::clone(&client);
            handles.push(tokio::spawn(async move { c.ping(i as f64).await }));
        }

        // Serial lower bound: CALLERS / POOL * SERVER_DELAY = 16 * 20ms = 320ms
        // Generous upper bound: 3x the theoretical lower bound.
        let budget = (SERVER_DELAY * CALLERS as u32 / POOL as u32) * 3;

        for (i, h) in handles.into_iter().enumerate() {
            let res = tokio::time::timeout(budget, h).await.unwrap_or_else(|_| {
                panic!(
                    "caller {i} did not complete within {:?} — starvation or deadlock",
                    budget
                )
            });
            res.unwrap()
                .unwrap_or_else(|e| panic!("caller {i} failed: {e:?}"));
        }
        let elapsed = start.elapsed();
        // Should NOT be fully serial — with POOL=4, ~4x speedup.
        let fully_serial = SERVER_DELAY * CALLERS as u32;
        assert!(
            elapsed < fully_serial,
            "oversubscribed pool looks serialized: elapsed {:?} >= fully_serial {:?}",
            elapsed,
            fully_serial
        );
    }

    /// Mixed success and failure paths must release every pool permit.
    #[tokio::test]
    async fn mixed_burst_releases_pool_permits() {
        const POOL: usize = 4;
        let path = short_sock_path();

        // Echo server that randomly returns valid / invalid frames,
        // triggering the full spectrum of result labels (ok, decode
        // error, transport close).
        let path_clone = path.clone();
        let _ = tokio::fs::remove_file(&path_clone).await;
        let listener = UnixListener::bind(&path_clone).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);
        let _server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let counter = Arc::clone(&counter_clone);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let idx = counter.fetch_add(1, Ordering::SeqCst);
                        let bucket = idx % 3;
                        if bucket == 0 {
                            // Valid reply.
                            let resp = ping_reply_bytes(&buf, "mixed");
                            let len = (resp.len() as u32).to_be_bytes();
                            let _ = sock.write_all(&len).await;
                            let _ = sock.write_all(&resp).await;
                            let _ = sock.flush().await;
                        } else if bucket == 1 {
                            // Close the socket mid-protocol → transport error on client.
                            return;
                        } else {
                            // Junk body → client Decode error.
                            let junk = b"\x00\x00\x00\x04junk";
                            let _ = sock.write_all(junk).await;
                            let _ = sock.flush().await;
                            return;
                        }
                    }
                });
            }
        });

        tokio::task::yield_now().await;
        let client = Arc::new(IpcClient::new_pool(path, POOL));

        // Fire 32 concurrent calls. Individual failures are expected from the
        // fault-injecting server; all tasks must nevertheless settle.
        let mut handles = Vec::new();
        for i in 0..32 {
            let c = Arc::clone(&client);
            handles.push(tokio::spawn(async move {
                let _ = c.ping(i as f64).await;
            }));
        }
        for h in handles {
            let _ = h.await;
        }
        assert!(
            tokio::time::timeout(Duration::from_secs(1), client.ping(999.0))
                .await
                .is_ok(),
            "pool permit leaked after mixed burst"
        );
    }
}
