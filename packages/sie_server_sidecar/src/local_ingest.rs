//! Local-ingest server: a UDS listener speaking the sidecar dispatch
//! protocol, feeding the same dispatcher/prep/backend pipeline the NATS
//! pull loop feeds.
//!
//! Added for the broker-less sidecar lane. In this mode there is no NATS:
//! a thin handler shim forwards each `run_batch` call over this socket as
//! one `publish_work` op, and the sidecar answers with the msgpack
//! `WorkResult` array. Concurrent `publish_work` calls coalesce in the
//! sidecar's per-model scheduler exactly like concurrent NATS fetches do.
//!
//! ## Wire contract (v0.2)
//!
//! * Frame: `u32` little-endian length + msgpack map. 64 MiB cap.
//! * Request: `{"id": u64, "op": str, "body": map}`; response
//!   `{"id": u64, "ok": bool, "error": str|nil, "body": map}`.
//! * Envelope IDs must be unique while an operation is active; a duplicate
//!   closes the connection because its response cannot be correlated safely.
//!   Retained data request frames share one 64 MiB byte budget per connection;
//!   the reader keeps a separate small byte reserve so a valid `cancel` frame
//!   can still be decoded when data bytes are saturated.
//! * Ops: `ping` → `{}`; `publish_work` → `{results: bin}`;
//!   `publish_generate_stream` → an ordered sequence of `{chunk: bin,
//!   seq: n}` responses followed by exactly one `{final: true, outcome:
//!   {...}}` response or an `ok=false` response with an empty body; `cancel`
//!   forwards `body.request_id` only when that generation is active on the same
//!   connection and answers `{}`; unknown or terminal targets are no-ops.
//! * Generate streams are correlated by envelope `id`. One connection may
//!   carry several streams and control calls. Active generation `request_id`s
//!   are claimed process-wide because backend cancellation is keyed by that ID
//!   alone. A byte- and count-bounded writer serializes frames; every backend
//!   event awaits its finite socket write, so the local hop introduces no
//!   unbounded chunk queue. Count and retained-byte admission bounds decoded
//!   and active work. Data-operation saturation is rejected after one bounded
//!   decode slot so it cannot prevent the reader from accepting a reserved
//!   `cancel` control call. A terminal tombstones its stream. Client EOF signals
//!   cancellation for every active stream on that connection.
//! * `publish_work.body` carries opaque `params`, `items`, and
//!   `dispatch_context` bytes plus a domain-separated SHA-256
//!   `payload_digest`. This unkeyed checksum detects field-assembly drift
//!   across trusted local peers by covering the caller context, route, request,
//!   timeout, params, and items. It does not authenticate a caller: socket
//!   permissions and the co-resident process boundary provide access control.
//!   The context is discarded after validation; it is not worker input.
//! * Errors are `"ExceptionType: message"` strings; a decodable request
//!   with an unknown op or bad body answers `ok=false` and keeps the
//!   connection open; an undecodable frame closes the connection.
//!
//! ## Divergences from the NATS ingest, by design
//!
//! * **No `reply_subject` enforcement** — results never touch a NATS
//!   subject; every result rides this socket back to the caller, so the
//!   `_INBOX.` anti-injection check is meaningless here.
//! * **No broker redelivery** — a dispatcher NAK surfaces as a
//!   [`LocalDeliveryEvent::Retry`] and is re-dispatched in-process with a
//!   bounded attempt budget, after which it becomes a typed error result.
//!   Same proportional stand-in the reference Python lane uses
//!   (`_NAK_MAX_ATTEMPTS`).
//! * **Worker-side admission re-check** happens here (the lane has
//!   one pool identity), mirroring the reference lane engine's
//!   `_admission_error`: rejections are typed errors, not NAKs — there is
//!   no differently-assigned worker to redeliver to.

use std::collections::HashSet;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use anyhow::Context;
use opentelemetry::trace::TraceContextExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, oneshot, Mutex, Notify, OwnedSemaphorePermit, Semaphore};
use tracing::{debug, info, warn, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::delivery::{Delivery, LocalDelivery, LocalDeliveryEvent};
use crate::dispatcher::Dispatcher;
use crate::shutdown::Shutdown;
use crate::work_types::{WorkItem, WorkResult};

/// Defensive ceiling on a single frame — matches the reference dispatcher
/// server's `DEFAULT_MAX_FRAME_BYTES` (over-length = protocol error →
/// connection closed).
pub const MAX_LOCAL_INGEST_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Per-object decoded binary limits shared with the public API contract.
/// The outer frame remains the aggregate request bound.
const MAX_IMAGE_OR_DOCUMENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_AUDIO_BYTES: usize = 24 * 1024 * 1024;
const MAX_WORK_ITEMS_PER_CALL: usize = 4_096;
const MAX_GENERATE_STREAMS_PER_CONNECTION: usize = 64;
const MAX_PENDING_WRITES_PER_CONNECTION: usize = MAX_GENERATE_STREAMS_PER_CONNECTION + 16;
const MAX_INFLIGHT_DATA_OPERATIONS_PER_CONNECTION: usize = MAX_PENDING_WRITES_PER_CONNECTION;
const MAX_INFLIGHT_CONTROL_OPERATIONS_PER_CONNECTION: usize = 1;
const MAX_DECODING_FRAMES_PER_CONNECTION: usize = 1;
const MAX_BUFFERED_WRITE_BYTES: usize = MAX_LOCAL_INGEST_FRAME_BYTES + 4;
const MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION: usize = MAX_LOCAL_INGEST_FRAME_BYTES;
const MAX_RESERVED_CONTROL_REQUEST_BYTES_PER_CONNECTION: usize = 1_024;
const MAX_BUFFERED_REQUEST_BYTES_PER_CONNECTION: usize =
    MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION + MAX_RESERVED_CONTROL_REQUEST_BYTES_PER_CONNECTION;
const LOCAL_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const LOCAL_GENERATE_CANCEL_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Domain-separate the local-ingest consistency checksum from every other
/// SHA-256 use in the sidecar. The context is caller-defined and discarded
/// after checksum/route validation. This detects accidental substitution
/// between trusted local layers; because it is unkeyed, it provides no
/// authenticity against a process that can reach the socket.
const PAYLOAD_DIGEST_DOMAIN: &[u8] = b"sie-local-ingest-v1\0";
const PAYLOAD_DIGEST_BYTES: usize = 32;
const MAX_REQUEST_ID_BYTES: usize = 128;

/// Default liveness deadline for a `publish_work` op when the request omits
/// `timeout_ms` (or sends `0`). A missing/zero value must NOT disable the
/// deadline: an unbounded wait pins the connection and leaks the detached
/// dispatch forever if a slot never settles. It therefore falls back to this
/// bounded ceiling; callers may set a shorter explicit positive `timeout_ms`.
const DEFAULT_PUBLISH_WORK_TIMEOUT_MS: u64 = 300_000;
const MAX_PUBLISH_WORK_TIMEOUT_MS: i64 = 300_000;

/// Redelivery budget per item: how many NAK-triggered re-dispatches one
/// item gets before its NAK becomes a typed error result. Matches the
/// Python lane's `_NAK_MAX_ATTEMPTS`.
const LOCAL_REDELIVERY_MAX_ATTEMPTS: u32 = 3;

/// Ceiling on one local retry backoff. NATS NAK delays reach 5s
/// (`SIE_NAK_DELAY_S` default); with a caller synchronously awaiting the
/// batch there is no reason to wait longer per hop. Matches the Python
/// lane's `_NAK_MAX_DELAY_S`.
const LOCAL_RETRY_MAX_DELAY_MS: u64 = 5_000;

/// Error code for the worker-side admission re-check — same string
/// the reference Python lane emits.
const POOL_ADMISSION_ERROR_CODE: &str = "pool_admission_rejected";

const OP_PING: &str = "ping";
const OP_PUBLISH_WORK: &str = "publish_work";
const OP_PUBLISH_GENERATE_STREAM: &str = "publish_generate_stream";
const OP_CANCEL: &str = "cancel";

// ---------------------------------------------------------------------------
// Wire envelopes
// ---------------------------------------------------------------------------

/// One-shot body decode covering every v0.1 op: unknown fields are
/// ignored, absent fields default, so `ping`'s `{}` and `publish_work`'s
/// full body both land here without a second msgpack pass over the
/// (potentially large) `items` bytes.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct RequestBody {
    lane: String,
    endpoint: String,
    model: String,
    engine: String,
    admission_pool: String,
    bundle_config_hash: String,
    request_id: String,
    /// Opaque `WorkParams` bytes. Every executor-relevant field is also
    /// serialized per-item on the WorkItem maps (gateway `WorkItemRef`
    /// contract), so the lane never decodes it — same as the Python lane.
    params: serde_bytes::ByteBuf,
    items: serde_bytes::ByteBuf,
    /// Caller-defined context included in the checksum and discarded after
    /// trust-boundary validation. It lets trusted transport layers detect
    /// route/context assembly drift without making substrate identity part of
    /// the public WorkItem or backend contract.
    dispatch_context: serde_bytes::ByteBuf,
    payload_digest: serde_bytes::ByteBuf,
    timeout_ms: i64,
    #[serde(deserialize_with = "optional_trace_string")]
    traceparent: Option<String>,
    #[serde(deserialize_with = "optional_trace_string")]
    tracestate: Option<String>,
}

// These optional fields used to be ignored as unknown keys. Malformed
// telemetry must continue to leave the underlying request admissible.
fn optional_trace_string<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    struct TraceString;
    impl<'de> serde::de::Visitor<'de> for TraceString {
        type Value = Option<String>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("optional bounded trace string")
        }
        fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
            Ok((value.len() <= 512).then(|| value.to_owned()))
        }
        fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_bytes<E: serde::de::Error>(self, _: &[u8]) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_newtype_struct<D: serde::Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            serde::de::IgnoredAny::deserialize(deserializer)?;
            Ok(None)
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            while sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(None)
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            while map
                .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                .is_some()
            {}
            Ok(None)
        }
    }
    // Skip wrong-typed containers without materializing a potentially large
    // Value tree. String input is borrowed and copied only within the bound.
    deserializer.deserialize_any(TraceString)
}

#[derive(Debug, Deserialize, Serialize)]
struct RequestEnvelope {
    id: u64,
    op: String,
    #[serde(default)]
    body: RequestBody,
}

fn encode_response_payload(id: u64, ok: bool, error: Option<&str>, body: rmpv::Value) -> Vec<u8> {
    let map = rmpv::Value::Map(vec![
        (rmpv::Value::from("id"), rmpv::Value::from(id)),
        (rmpv::Value::from("ok"), rmpv::Value::from(ok)),
        (
            rmpv::Value::from("error"),
            match error {
                Some(e) => rmpv::Value::from(e),
                None => rmpv::Value::Nil,
            },
        ),
        (rmpv::Value::from("body"), body),
    ]);
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &map).expect("msgpack encode to Vec cannot fail");
    payload
}

fn frame_payload(payload: Vec<u8>) -> Vec<u8> {
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn encode_response_with_limit(
    id: u64,
    ok: bool,
    error: Option<&str>,
    body: rmpv::Value,
    max_frame_bytes: usize,
) -> Vec<u8> {
    let payload = encode_response_payload(id, ok, error, body);
    if payload.len() <= max_frame_bytes {
        return frame_payload(payload);
    }

    let error = format!("ResultTooLarge: response frame exceeds {max_frame_bytes} bytes");
    let fallback = encode_response_payload(id, false, Some(&error), empty_body());
    debug_assert!(fallback.len() <= MAX_LOCAL_INGEST_FRAME_BYTES);
    frame_payload(fallback)
}

fn encode_response(id: u64, ok: bool, error: Option<&str>, body: rmpv::Value) -> Vec<u8> {
    encode_response_with_limit(id, ok, error, body, MAX_LOCAL_INGEST_FRAME_BYTES)
}

fn empty_body() -> rmpv::Value {
    rmpv::Value::Map(Vec::new())
}

fn results_body(results_bytes: Vec<u8>) -> rmpv::Value {
    rmpv::Value::Map(vec![(
        rmpv::Value::from("results"),
        rmpv::Value::Binary(results_bytes),
    )])
}

/// Read one length-prefixed frame; `Ok(None)` on clean EOF between frames.
#[cfg(test)]
async fn read_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> std::io::Result<Option<Vec<u8>>> {
    let Some(len) = read_frame_length(reader).await? else {
        return Ok(None);
    };
    read_frame_payload(reader, len).await.map(Some)
}

async fn read_frame_with_budget<R: AsyncReadExt + Unpin>(
    reader: &mut R,
    byte_budget: &Arc<Semaphore>,
) -> std::io::Result<Option<(Vec<u8>, OwnedSemaphorePermit)>> {
    let Some(len) = read_frame_length(reader).await? else {
        return Ok(None);
    };
    let permit = Arc::clone(byte_budget)
        .acquire_many_owned(len)
        .await
        .map_err(|_| std::io::Error::other("local-ingest request byte budget closed"))?;
    let payload = read_frame_payload(reader, len).await?;
    Ok(Some((payload, permit)))
}

async fn read_frame_length<R: AsyncReadExt + Unpin>(
    reader: &mut R,
) -> std::io::Result<Option<u32>> {
    let mut header = [0u8; 4];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(header);
    if len as usize > MAX_LOCAL_INGEST_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {MAX_LOCAL_INGEST_FRAME_BYTES}"),
        ));
    }
    Ok(Some(len))
}

async fn read_frame_payload<R: AsyncReadExt + Unpin>(
    reader: &mut R,
    len: u32,
) -> std::io::Result<Vec<u8>> {
    // Grow the buffer as bytes actually arrive rather than pre-allocating the
    // full declared length: a client can cheaply declare up to the 64 MiB cap,
    // so an eager `vec![0u8; len]` would let a 4-byte header pin 64 MiB before
    // any body arrives (local memory amplification). `take` bounds the read to
    // the declared length; a short/truncated frame is a protocol error.
    let mut payload = Vec::new();
    let read = reader
        .take(u64::from(len))
        .read_to_end(&mut payload)
        .await?;
    if read != len as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("frame truncated: read {read} of {len} declared bytes"),
        ));
    }
    Ok(payload)
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Lane identity + shared handles one connection task needs.
struct IngestShared {
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<Shutdown>,
    active_generate_request_ids: Arc<StdMutex<HashSet<String>>>,
    /// Physical pool this lane serves (`SIE_POOL`), pre-normalized.
    lane_pool: String,
    worker_id: String,
}

/// Bind the local-ingest UDS and serve until shutdown. Consumes the
/// listener task; callers spawn it.
pub async fn run_local_ingest(
    socket_path: &Path,
    dispatcher: Arc<Dispatcher>,
    pool: &str,
    worker_id: &str,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<()> {
    // Unlink a stale socket file (crash leftovers) before binding — same
    // bind-time cleanup as the Python IPC/dispatcher servers.
    if socket_path.exists() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("unlink stale local socket {}", socket_path.display()))?;
    }
    if let Some(parent) = socket_path.parent() {
        let parent_existed = parent.exists();
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create socket dir {}", parent.display()))?;
        // Only tighten a directory we created — never clobber the mode of a
        // pre-existing (possibly shared, e.g. `/tmp`) parent. A dedicated
        // socket dir at 0700 denies traversal to non-owners regardless of the
        // socket node's own mode.
        if !parent_existed {
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("restrict socket dir {}", parent.display()))?;
        }
    }
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("bind local ingest socket {}", socket_path.display()))?;
    // Restrict the socket node to the owner. It is bound with the process
    // umask default (`0o777 & !umask`), so under a permissive container umask
    // (0000/0002) it would be group/world-connectable and any local process
    // could inject WorkItems onto this worker's GPU (compute theft / DoS).
    // connect(2) checks write permission on the node, so 0600 gates it.
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restrict local ingest socket {}", socket_path.display()))?;
    info!(socket = %socket_path.display(), pool, "local-ingest: listening");

    let shared = Arc::new(IngestShared {
        dispatcher,
        shutdown: Arc::clone(&shutdown),
        active_generate_request_ids: Arc::new(StdMutex::new(HashSet::new())),
        lane_pool: normalize_pool(pool),
        worker_id: worker_id.to_string(),
    });

    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait() => break,
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let shared = Arc::clone(&shared);
                        tokio::spawn(async move {
                            handle_connection(stream, shared).await;
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "local-ingest: accept failed");
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_file(socket_path);
    info!("local-ingest: listener stopped");
    Ok(())
}

#[derive(Default)]
struct ConnectionLifecycle {
    closed: AtomicBool,
    closed_notify: Notify,
}

impl ConnectionLifecycle {
    fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.closed_notify.notify_waiters();
        }
    }

    fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    async fn wait_closed(&self) {
        loop {
            let notified = self.closed_notify.notified();
            if self.is_closed() {
                return;
            }
            notified.await;
        }
    }
}

struct QueuedWrite {
    frame: Vec<u8>,
    deadline: tokio::time::Instant,
    accepting: Option<Arc<AtomicBool>>,
    cancelled: Arc<AtomicBool>,
    done: oneshot::Sender<Result<(), String>>,
    _byte_budget: OwnedSemaphorePermit,
}

struct QueuedWriteDropGuard {
    cancelled: Arc<AtomicBool>,
    armed: bool,
}

impl Drop for QueuedWriteDropGuard {
    fn drop(&mut self) {
        if self.armed {
            self.cancelled.store(true, Ordering::Release);
        }
    }
}

#[derive(Clone)]
struct ConnectionWriter {
    tx: mpsc::Sender<QueuedWrite>,
    byte_budget: Arc<Semaphore>,
    lifecycle: Arc<ConnectionLifecycle>,
}

impl ConnectionWriter {
    fn spawn(
        write_half: tokio::net::unix::OwnedWriteHalf,
        lifecycle: Arc<ConnectionLifecycle>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(MAX_PENDING_WRITES_PER_CONNECTION);
        let byte_budget = Arc::new(Semaphore::new(MAX_BUFFERED_WRITE_BYTES));
        tokio::spawn(run_connection_writer(
            write_half,
            rx,
            Arc::clone(&lifecycle),
        ));
        Self {
            tx,
            byte_budget,
            lifecycle,
        }
    }

    async fn send(&self, frame: Vec<u8>) -> Result<(), String> {
        self.send_if(frame, None).await
    }

    async fn send_if(
        &self,
        frame: Vec<u8>,
        accepting: Option<Arc<AtomicBool>>,
    ) -> Result<(), String> {
        let result = self.send_inner(frame, accepting).await;
        if result.is_err() {
            self.lifecycle.close();
        }
        result
    }

    async fn send_inner(
        &self,
        frame: Vec<u8>,
        accepting: Option<Arc<AtomicBool>>,
    ) -> Result<(), String> {
        if self.lifecycle.is_closed() {
            return Err("local-ingest connection is closed".to_string());
        }
        let permits = u32::try_from(frame.len())
            .map_err(|_| "local-ingest response frame is too large".to_string())?;
        let deadline = tokio::time::Instant::now() + LOCAL_WRITE_TIMEOUT;
        let byte_budget = tokio::time::timeout_at(
            deadline,
            Arc::clone(&self.byte_budget).acquire_many_owned(permits),
        )
        .await
        .map_err(|_| "local-ingest response write queue timed out".to_string())?
        .map_err(|_| "local-ingest response writer stopped".to_string())?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let (done, completed) = oneshot::channel();
        let command = QueuedWrite {
            frame,
            deadline,
            accepting,
            cancelled: Arc::clone(&cancelled),
            done,
            _byte_budget: byte_budget,
        };
        let mut drop_guard = QueuedWriteDropGuard {
            cancelled,
            armed: true,
        };
        tokio::time::timeout_at(deadline, self.tx.send(command))
            .await
            .map_err(|_| "local-ingest response write queue timed out".to_string())?
            .map_err(|_| "local-ingest response writer stopped".to_string())?;
        let result = tokio::time::timeout_at(deadline, completed)
            .await
            .map_err(|_| "local-ingest response write timed out".to_string())?
            .map_err(|_| "local-ingest response writer stopped".to_string())?;
        drop_guard.armed = false;
        result
    }
}

struct ActiveOperationIdGuard {
    operation_id: u64,
    active: Arc<StdMutex<HashSet<u64>>>,
}

impl ActiveOperationIdGuard {
    fn claim(operation_id: u64, active: Arc<StdMutex<HashSet<u64>>>) -> Option<Self> {
        let claimed = active
            .lock()
            .expect("local-ingest active operation ids mutex poisoned")
            .insert(operation_id);
        claimed.then_some(Self {
            operation_id,
            active,
        })
    }
}

impl Drop for ActiveOperationIdGuard {
    fn drop(&mut self) {
        self.active
            .lock()
            .expect("local-ingest active operation ids mutex poisoned")
            .remove(&self.operation_id);
    }
}

struct ActiveGenerateClaim {
    request_id: String,
    active: Arc<StdMutex<HashSet<String>>>,
}

impl ActiveGenerateClaim {
    fn claim(request_id: String, active: Arc<StdMutex<HashSet<String>>>) -> Option<Self> {
        let claimed = active
            .lock()
            .expect("local-ingest active generation request ids mutex poisoned")
            .insert(request_id.clone());
        claimed.then_some(Self { request_id, active })
    }
}

impl Drop for ActiveGenerateClaim {
    fn drop(&mut self) {
        self.active
            .lock()
            .expect("local-ingest active generation request ids mutex poisoned")
            .remove(&self.request_id);
    }
}

async fn acquire_connection_operation(
    operation_budget: &Arc<Semaphore>,
    lifecycle: &ConnectionLifecycle,
    shutdown: &Shutdown,
) -> Option<OwnedSemaphorePermit> {
    tokio::select! {
        biased;
        _ = shutdown.wait() => None,
        _ = lifecycle.wait_closed() => None,
        permit = Arc::clone(operation_budget).acquire_owned() => permit.ok(),
    }
}

enum DecodedOperationAdmission {
    Admitted(OwnedSemaphorePermit),
    DataCapacityExhausted,
    Closed,
}

async fn admit_decoded_operation(
    op: &str,
    data_budget: &Arc<Semaphore>,
    control_budget: &Arc<Semaphore>,
    lifecycle: &ConnectionLifecycle,
    shutdown: &Shutdown,
) -> DecodedOperationAdmission {
    if op == OP_CANCEL {
        return match acquire_connection_operation(control_budget, lifecycle, shutdown).await {
            Some(permit) => DecodedOperationAdmission::Admitted(permit),
            None => DecodedOperationAdmission::Closed,
        };
    }
    match Arc::clone(data_budget).try_acquire_owned() {
        Ok(permit) => DecodedOperationAdmission::Admitted(permit),
        Err(tokio::sync::TryAcquireError::NoPermits) => {
            DecodedOperationAdmission::DataCapacityExhausted
        }
        Err(tokio::sync::TryAcquireError::Closed) => DecodedOperationAdmission::Closed,
    }
}

async fn run_connection_writer(
    mut write_half: tokio::net::unix::OwnedWriteHalf,
    mut rx: mpsc::Receiver<QueuedWrite>,
    lifecycle: Arc<ConnectionLifecycle>,
) {
    loop {
        let command = tokio::select! {
            biased;
            _ = lifecycle.wait_closed() => break,
            command = rx.recv() => match command {
                Some(command) => command,
                None => break,
            },
        };
        if command.cancelled.load(Ordering::Acquire)
            || command
                .accepting
                .as_ref()
                .is_some_and(|accepting| !accepting.load(Ordering::Acquire))
        {
            let _ = command.done.send(Ok(()));
            continue;
        }
        let result =
            tokio::time::timeout_at(command.deadline, write_half.write_all(&command.frame))
                .await
                .map_err(|_| "local-ingest response write timed out".to_string())
                .and_then(|result| result.map_err(|error| error.to_string()));
        match result {
            Ok(()) => {
                let _ = command.done.send(Ok(()));
            }
            Err(error) => {
                let _ = command.done.send(Err(error.clone()));
                lifecycle.close();
                break;
            }
        }
    }
    lifecycle.close();
}

#[derive(Clone)]
struct GenerateStreamWriter {
    operation_id: u64,
    writer: ConnectionWriter,
    accepting_chunks: Arc<AtomicBool>,
    terminal: Arc<AtomicBool>,
    chunks_sent: Arc<AtomicU64>,
}

impl GenerateStreamWriter {
    fn new(operation_id: u64, writer: ConnectionWriter) -> Self {
        Self {
            operation_id,
            writer,
            accepting_chunks: Arc::new(AtomicBool::new(true)),
            terminal: Arc::new(AtomicBool::new(false)),
            chunks_sent: Arc::new(AtomicU64::new(0)),
        }
    }

    fn stop_chunks(&self) {
        self.accepting_chunks.store(false, Ordering::Release);
    }

    async fn send_chunk(&self, payload: Vec<u8>) -> Result<(), String> {
        if self.terminal.load(Ordering::Acquire) {
            return Err("generation emitted a chunk after its transport terminal".to_string());
        }
        if !self.accepting_chunks.load(Ordering::Acquire) {
            return Ok(());
        }
        let seq = self.chunks_sent.load(Ordering::Acquire);
        let frame = encode_generate_chunk(self.operation_id, seq, payload)?;
        self.writer
            .send_if(frame, Some(Arc::clone(&self.accepting_chunks)))
            .await?;
        if self.accepting_chunks.load(Ordering::Acquire) {
            self.chunks_sent.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    async fn finish_success(&self) -> Result<(), String> {
        self.finish(encode_generate_final(
            self.operation_id,
            self.chunks_sent.load(Ordering::Acquire),
        ))
        .await
    }

    async fn finish_error(
        &self,
        error: &crate::dispatcher::GenerateDispatchError,
    ) -> Result<(), String> {
        self.finish(encode_generate_error(self.operation_id, error))
            .await
    }

    async fn finish(&self, frame: Vec<u8>) -> Result<(), String> {
        if self.terminal.swap(true, Ordering::AcqRel) {
            return Err("generation transport terminal already emitted".to_string());
        }
        self.stop_chunks();
        self.writer.send(frame).await
    }
}

async fn handle_connection(stream: UnixStream, shared: Arc<IngestShared>) {
    let (mut reader, write_half) = stream.into_split();
    let lifecycle = Arc::new(ConnectionLifecycle::default());
    let writer = ConnectionWriter::spawn(write_half, Arc::clone(&lifecycle));
    let active_generate = Arc::new(Mutex::new(HashSet::<String>::new()));
    let decode_budget = Arc::new(Semaphore::new(MAX_DECODING_FRAMES_PER_CONNECTION));
    let data_operation_budget =
        Arc::new(Semaphore::new(MAX_INFLIGHT_DATA_OPERATIONS_PER_CONNECTION));
    let control_operation_budget = Arc::new(Semaphore::new(
        MAX_INFLIGHT_CONTROL_OPERATIONS_PER_CONNECTION,
    ));
    let request_byte_budget = Arc::new(Semaphore::new(MAX_BUFFERED_REQUEST_BYTES_PER_CONNECTION));
    let data_request_byte_budget =
        Arc::new(Semaphore::new(MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION));
    let active_operation_ids = Arc::new(StdMutex::new(HashSet::<u64>::new()));
    loop {
        let Some(decode_permit) =
            acquire_connection_operation(&decode_budget, &lifecycle, &shared.shutdown).await
        else {
            break;
        };

        let (payload, inbound_permit) = tokio::select! {
            biased;
            _ = shared.shutdown.wait() => break,
            _ = lifecycle.wait_closed() => break,
            frame = read_frame_with_budget(&mut reader, &request_byte_budget) => match frame {
                Ok(Some(p)) => p,
                Ok(None) => break, // clean EOF
                Err(e) => {
                    warn!(error = %e, "local-ingest: closing connection on malformed frame");
                    break;
                }
            },
        };
        let request: RequestEnvelope = match rmp_serde::from_slice(&payload) {
            Ok(r) => r,
            Err(e) => {
                // No usable `id` to correlate an error response — close,
                // per the protocol contract.
                warn!(error = %e, "local-ingest: closing connection on malformed request envelope");
                break;
            }
        };
        let request_bytes = u32::try_from(payload.len()).expect("local-ingest frame length is u32");
        drop(payload);
        let Some(operation_id_guard) =
            ActiveOperationIdGuard::claim(request.id, Arc::clone(&active_operation_ids))
        else {
            warn!(
                id = request.id,
                "local-ingest: duplicate active envelope id; closing connection"
            );
            break;
        };
        let operation_permit = match admit_decoded_operation(
            &request.op,
            &data_operation_budget,
            &control_operation_budget,
            &lifecycle,
            &shared.shutdown,
        )
        .await
        {
            DecodedOperationAdmission::Admitted(permit) => permit,
            DecodedOperationAdmission::DataCapacityExhausted => {
                drop(decode_permit);
                let message = format!(
                    "connection already has {MAX_INFLIGHT_DATA_OPERATIONS_PER_CONNECTION} active data operations"
                );
                let frame = if request.op == OP_PUBLISH_GENERATE_STREAM {
                    encode_generate_error(
                        request.id,
                        &crate::dispatcher::GenerateDispatchError {
                            code: "CAPACITY_EXHAUSTED".to_string(),
                            message,
                        },
                    )
                } else {
                    encode_response(
                        request.id,
                        false,
                        Some(&format!("CapacityExhausted: {message}")),
                        empty_body(),
                    )
                };
                write_response(&writer, frame).await;
                continue;
            }
            DecodedOperationAdmission::Closed => break,
        };
        drop(decode_permit);
        let retained_data_permit = if request.op == OP_CANCEL {
            None
        } else {
            match Arc::clone(&data_request_byte_budget).try_acquire_many_owned(request_bytes) {
                Ok(permit) => Some(permit),
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    let message = format!(
                        "connection already retains {MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION} request bytes"
                    );
                    let frame = if request.op == OP_PUBLISH_GENERATE_STREAM {
                        encode_generate_error(
                            request.id,
                            &crate::dispatcher::GenerateDispatchError {
                                code: "CAPACITY_EXHAUSTED".to_string(),
                                message,
                            },
                        )
                    } else {
                        encode_response(
                            request.id,
                            false,
                            Some(&format!("CapacityExhausted: {message}")),
                            empty_body(),
                        )
                    };
                    write_response(&writer, frame).await;
                    continue;
                }
                Err(tokio::sync::TryAcquireError::Closed) => break,
            }
        };
        if request.op == OP_PUBLISH_GENERATE_STREAM {
            let request_id = request.body.request_id.clone();
            let mut active = active_generate.lock().await;
            let claim = if !valid_request_id(&request_id) {
                Err((
                    "INVALID_TRANSPORT_BINDING",
                    "invalid generation request_id".to_string(),
                ))
            } else if active.contains(&request_id) {
                Err((
                    "DUPLICATE_REQUEST",
                    format!("generation request {request_id:?} is already active"),
                ))
            } else if active.len() >= MAX_GENERATE_STREAMS_PER_CONNECTION {
                Err((
                    "CAPACITY_EXHAUSTED",
                    format!(
                        "connection already has {MAX_GENERATE_STREAMS_PER_CONNECTION} active generation streams"
                    ),
                ))
            } else {
                match ActiveGenerateClaim::claim(
                    request_id.clone(),
                    Arc::clone(&shared.active_generate_request_ids),
                ) {
                    Some(claim) => {
                        active.insert(request_id.clone());
                        Ok(claim)
                    }
                    None => Err((
                        "DUPLICATE_REQUEST",
                        format!("generation request {request_id:?} is already active"),
                    )),
                }
            };
            drop(active);
            let generate_claim = match claim {
                Ok(claim) => claim,
                Err((code, message)) => {
                    let error = crate::dispatcher::GenerateDispatchError {
                        code: code.to_string(),
                        message,
                    };
                    let frame = encode_generate_error(request.id, &error);
                    write_response(&writer, frame).await;
                    continue;
                }
            };

            let shared_op = Arc::clone(&shared);
            let writer_op = writer.clone();
            let lifecycle_op = Arc::clone(&lifecycle);
            let active_op = Arc::clone(&active_generate);
            tokio::spawn(async move {
                let _operation_permit = operation_permit;
                let _retained_data_permit = retained_data_permit;
                let _inbound_permit = inbound_permit;
                let _operation_id_guard = operation_id_guard;
                let _generate_claim = generate_claim;
                let operation_id = request.id;
                let stream_writer = GenerateStreamWriter::new(operation_id, writer_op);
                let mut span = tracing::Span::none();
                let result = publish_generate_stream(
                    request.body,
                    &shared_op,
                    stream_writer.clone(),
                    Arc::clone(&lifecycle_op),
                    &mut span,
                )
                .await;
                async {
                    active_op.lock().await.remove(&request_id);
                    if !lifecycle_op.is_closed() {
                        let write_result = match result {
                            Ok(()) => stream_writer.finish_success().await,
                            Err(error) => stream_writer.finish_error(&error).await,
                        };
                        if let Err(error) = write_result {
                            debug!(
                                request_id,
                                error, "local-ingest: generation terminal write failed"
                            );
                        }
                    }
                }
                .instrument(span)
                .await;
            });
            continue;
        }
        if request.op == OP_CANCEL {
            let shared_op = Arc::clone(&shared);
            let writer_op = writer.clone();
            let active_op = Arc::clone(&active_generate);
            tokio::spawn(async move {
                let _operation_permit = operation_permit;
                let _retained_data_permit = retained_data_permit;
                let _inbound_permit = inbound_permit;
                let _operation_id_guard = operation_id_guard;
                let request_id = request.body.request_id;
                let response = if !valid_request_id(&request_id) {
                    encode_response(
                        request.id,
                        false,
                        Some("InvalidTransportBinding: invalid request_id"),
                        empty_body(),
                    )
                } else {
                    let owns_request = active_op.lock().await.contains(&request_id);
                    if !owns_request {
                        encode_response(request.id, true, None, empty_body())
                    } else {
                        match tokio::time::timeout(
                            LOCAL_GENERATE_CANCEL_DRAIN_TIMEOUT,
                            shared_op
                                .dispatcher
                                .signal_local_generate_cancel(&request_id),
                        )
                        .await
                        {
                            Ok(Ok(_)) => encode_response(request.id, true, None, empty_body()),
                            Ok(Err(error)) => encode_response(
                                request.id,
                                false,
                                Some(&error.to_string()),
                                empty_body(),
                            ),
                            Err(_) => encode_response(
                                request.id,
                                false,
                                Some("TimeoutError: cancellation RPC timed out"),
                                empty_body(),
                            ),
                        }
                    }
                };
                write_response(&writer_op, response).await;
            });
            continue;
        }
        // Each op runs as its own task; responses may interleave and are
        // correlated by `id` (same shape as the Python dispatcher server).
        let shared_op = Arc::clone(&shared);
        let writer_op = writer.clone();
        tokio::spawn(async move {
            let _operation_permit = operation_permit;
            let _retained_data_permit = retained_data_permit;
            let _inbound_permit = inbound_permit;
            let _operation_id_guard = operation_id_guard;
            let mut span = tracing::Span::none();
            let frame = run_op(request, &shared_op, &mut span).await;
            write_response(&writer_op, frame).instrument(span).await;
        });
    }

    lifecycle.close();
    let request_ids: Vec<String> = active_generate.lock().await.drain().collect();
    for request_id in request_ids {
        match tokio::time::timeout(
            LOCAL_GENERATE_CANCEL_DRAIN_TIMEOUT,
            shared.dispatcher.signal_local_generate_cancel(&request_id),
        )
        .await
        {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => {
                debug!(
                    request_id,
                    error = %error,
                    "local-ingest: disconnect cancellation failed"
                );
            }
            Err(_) => {
                debug!(
                    request_id,
                    "local-ingest: disconnect cancellation timed out"
                );
            }
        }
    }
}

async fn run_op(
    request: RequestEnvelope,
    shared: &IngestShared,
    span: &mut tracing::Span,
) -> Vec<u8> {
    match request.op.as_str() {
        OP_PING => encode_response(request.id, true, None, empty_body()),
        OP_PUBLISH_WORK => match publish_work(request.body, shared, span).await {
            Ok(results_bytes) => {
                encode_response(request.id, true, None, results_body(results_bytes))
            }
            Err(e) => encode_response(request.id, false, Some(&e), empty_body()),
        },
        other => encode_response(
            request.id,
            false,
            Some(&format!("ValueError: unknown op: {other:?}")),
            empty_body(),
        ),
    }
}

fn valid_request_id(request_id: &str) -> bool {
    !request_id.is_empty()
        && request_id.len() <= MAX_REQUEST_ID_BYTES
        && !request_id.chars().any(char::is_whitespace)
}

async fn write_response(writer: &ConnectionWriter, frame: Vec<u8>) {
    if let Err(error) = writer.send(frame).await {
        debug!(error = %error, "local-ingest: response write failed (caller gone)");
    }
}

fn generate_chunk_body(seq: u64, payload: Vec<u8>) -> rmpv::Value {
    rmpv::Value::Map(vec![
        (rmpv::Value::from("chunk"), rmpv::Value::Binary(payload)),
        (rmpv::Value::from("seq"), rmpv::Value::from(seq)),
    ])
}

fn encode_generate_chunk_with_limit(
    id: u64,
    seq: u64,
    payload: Vec<u8>,
    max_frame_bytes: usize,
) -> Result<Vec<u8>, String> {
    let payload = encode_response_payload(id, true, None, generate_chunk_body(seq, payload));
    if payload.len() > max_frame_bytes {
        return Err(format!(
            "ResultTooLarge: generation chunk response frame is {} bytes; maximum is {max_frame_bytes}",
            payload.len()
        ));
    }
    Ok(frame_payload(payload))
}

fn encode_generate_chunk(id: u64, seq: u64, payload: Vec<u8>) -> Result<Vec<u8>, String> {
    encode_generate_chunk_with_limit(id, seq, payload, MAX_LOCAL_INGEST_FRAME_BYTES)
}

fn encode_generate_final(id: u64, chunks: u64) -> Vec<u8> {
    let outcome = rmpv::Value::Map(vec![
        (rmpv::Value::from("status"), rmpv::Value::from("complete")),
        (rmpv::Value::from("chunks"), rmpv::Value::from(chunks)),
    ]);
    let body = rmpv::Value::Map(vec![
        (rmpv::Value::from("final"), rmpv::Value::from(true)),
        (rmpv::Value::from("outcome"), outcome),
    ]);
    encode_response(id, true, None, body)
}

fn encode_generate_error(id: u64, error: &crate::dispatcher::GenerateDispatchError) -> Vec<u8> {
    let rendered = format!("{}: {}", error.code, error.message);
    encode_response(id, false, Some(&rendered), empty_body())
}

fn local_ingest_span(body: &RequestBody, items: &mut [WorkItem]) -> tracing::Span {
    let parent = crate::observability::propagation::extract_context_from_w3c(
        body.traceparent
            .as_deref()
            .filter(|value| value.len() <= 256),
        body.tracestate
            .as_deref()
            .filter(|value| value.len() <= 512),
    );
    if !parent.span().span_context().is_valid() {
        return tracing::Span::none();
    }
    let span = tracing::info_span!("sidecar.local_ingest", otel.name = "sidecar.local_ingest");
    let _ = span.set_parent(parent.clone());
    let context = span.context();
    let context = if context.span().span_context().is_valid() {
        context
    } else {
        parent
    };
    let (traceparent, tracestate) = crate::observability::propagation::inject_context(&context);
    for item in items {
        item.traceparent.clone_from(&traceparent);
        item.tracestate.clone_from(&tracestate);
    }
    span
}

async fn publish_generate_stream(
    body: RequestBody,
    shared: &IngestShared,
    stream_writer: GenerateStreamWriter,
    lifecycle: Arc<ConnectionLifecycle>,
    span: &mut tracing::Span,
) -> Result<(), crate::dispatcher::GenerateDispatchError> {
    let semantic_deadline = generate_timeout_deadline(body.timeout_ms).map_err(|message| {
        crate::dispatcher::GenerateDispatchError {
            code: "INVALID_TRANSPORT_BINDING".to_string(),
            message,
        }
    })?;
    let timeout_ms = (body.timeout_ms > 0).then_some(body.timeout_ms as u64);
    validate_payload_digest(&body).map_err(|message| crate::dispatcher::GenerateDispatchError {
        code: "INVALID_TRANSPORT_BINDING".to_string(),
        message,
    })?;
    let mut items: Vec<WorkItem> = rmp_serde::from_slice(&body.items).map_err(|error| {
        crate::dispatcher::GenerateDispatchError {
            code: "DECODE_ERROR".to_string(),
            message: format!("items is not a msgpack WorkItem array: {error}"),
        }
    })?;
    validate_work_items_with_limit(&body, &items, 1).map_err(|message| {
        crate::dispatcher::GenerateDispatchError {
            code: "INVALID_TRANSPORT_BINDING".to_string(),
            message,
        }
    })?;
    if body.endpoint != "generate" {
        return Err(crate::dispatcher::GenerateDispatchError {
            code: "BAD_OPERATION".to_string(),
            message: "publish_generate_stream requires endpoint generate".to_string(),
        });
    }
    *span = local_ingest_span(&body, &mut items);
    publish_validated_generate(
        body,
        items,
        shared,
        stream_writer,
        lifecycle,
        semantic_deadline,
        timeout_ms,
    )
    .instrument(span.clone())
    .await
}

async fn publish_validated_generate(
    body: RequestBody,
    mut items: Vec<WorkItem>,
    shared: &IngestShared,
    stream_writer: GenerateStreamWriter,
    lifecycle: Arc<ConnectionLifecycle>,
    semantic_deadline: Option<tokio::time::Instant>,
    timeout_ms: Option<u64>,
) -> Result<(), crate::dispatcher::GenerateDispatchError> {
    let mut work_item = items.pop().expect("validated exactly one generate item");
    let meta_pool = normalize_pool(&body.admission_pool);
    let item_pool = normalize_pool(&work_item.admission_pool);
    if let Some(message) = admission_error(&shared.lane_pool, &meta_pool, &item_pool) {
        return Err(crate::dispatcher::GenerateDispatchError {
            code: "POOL_ADMISSION_REJECTED".to_string(),
            message,
        });
    }
    if lifecycle.is_closed() {
        return Err(crate::dispatcher::GenerateDispatchError {
            code: "DISCONNECTED".to_string(),
            message: "local-ingest caller disconnected before generation dispatch".to_string(),
        });
    }
    work_item.reply_subject = format!("_LOCAL.{}", work_item.request_id);
    let request_id = body.request_id;
    let callback_writer = stream_writer.clone();
    let generation = shared
        .dispatcher
        .process_local_generate(work_item, move |payload| {
            let writer = callback_writer.clone();
            async move { writer.send_chunk(payload).await }
        });
    tokio::pin!(generation);
    let semantic_timeout = async move {
        match semantic_deadline {
            Some(deadline) => tokio::time::sleep_until(deadline).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(semantic_timeout);

    enum WaitOutcome {
        Complete(Result<(), crate::dispatcher::GenerateDispatchError>),
        Timeout(u64),
        Disconnected,
    }

    let outcome = tokio::select! {
        biased;
        _ = lifecycle.wait_closed() => WaitOutcome::Disconnected,
        result = &mut generation => WaitOutcome::Complete(result),
        _ = &mut semantic_timeout => WaitOutcome::Timeout(
            timeout_ms.expect("semantic timeout future only settles for a positive timeout")
        ),
    };
    let (code, message) = match outcome {
        WaitOutcome::Complete(result) => return result,
        WaitOutcome::Timeout(timeout_ms) => (
            "TIMEOUT",
            format!("generation timed out after {timeout_ms}ms"),
        ),
        WaitOutcome::Disconnected => (
            "DISCONNECTED",
            "local-ingest caller disconnected during generation".to_string(),
        ),
    };

    stream_writer.stop_chunks();
    let cancel_deadline = tokio::time::Instant::now() + LOCAL_GENERATE_CANCEL_DRAIN_TIMEOUT;
    let cancel_error = match tokio::time::timeout_at(
        cancel_deadline,
        shared.dispatcher.signal_local_generate_cancel(&request_id),
    )
    .await
    {
        Ok(Ok(_)) => None,
        Ok(Err(error)) => Some(error.to_string()),
        Err(_) => Some("cancellation RPC timed out".to_string()),
    };
    let drain_deadline = tokio::time::Instant::now() + LOCAL_GENERATE_CANCEL_DRAIN_TIMEOUT;
    let drain_timed_out = tokio::time::timeout_at(drain_deadline, &mut generation)
        .await
        .is_err();
    let mut suffixes = Vec::new();
    if let Some(error) = cancel_error {
        suffixes.push(format!("cancellation failed: {error}"));
    }
    if drain_timed_out {
        suffixes.push("backend cancellation drain timed out".to_string());
    }
    let suffix = if suffixes.is_empty() {
        String::new()
    } else {
        format!("; {}", suffixes.join("; "))
    };
    Err(crate::dispatcher::GenerateDispatchError {
        code: code.to_string(),
        message: format!("{message}{suffix}"),
    })
}

// ---------------------------------------------------------------------------
// publish_work
// ---------------------------------------------------------------------------

fn update_digest_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn compute_payload_digest(body: &RequestBody) -> [u8; PAYLOAD_DIGEST_BYTES] {
    let mut hasher = Sha256::new();
    hasher.update(PAYLOAD_DIGEST_DOMAIN);
    for value in [
        body.dispatch_context.as_ref(),
        body.lane.as_bytes(),
        body.endpoint.as_bytes(),
        body.model.as_bytes(),
        body.engine.as_bytes(),
        body.admission_pool.as_bytes(),
        body.bundle_config_hash.as_bytes(),
        body.request_id.as_bytes(),
        body.params.as_ref(),
        body.items.as_ref(),
    ] {
        update_digest_field(&mut hasher, value);
    }
    hasher.update(body.timeout_ms.to_be_bytes());
    hasher.finalize().into()
}

fn validate_payload_digest(body: &RequestBody) -> Result<(), String> {
    if body.dispatch_context.is_empty() {
        return Err("InvalidTransportBinding: dispatch_context must not be empty".to_string());
    }
    if body.payload_digest.len() != PAYLOAD_DIGEST_BYTES {
        return Err(format!(
            "InvalidTransportBinding: payload_digest must be {PAYLOAD_DIGEST_BYTES} bytes"
        ));
    }
    let expected = compute_payload_digest(body);
    if expected.as_slice() != body.payload_digest.as_ref() {
        return Err("InvalidTransportBinding: payload_digest mismatch".to_string());
    }
    Ok(())
}

fn validate_publish_work_timeout(timeout_ms: i64) -> Result<(), String> {
    if !(0..=MAX_PUBLISH_WORK_TIMEOUT_MS).contains(&timeout_ms) {
        return Err(format!(
            "InvalidTransportBinding: timeout_ms {timeout_ms} must be between 0 and {MAX_PUBLISH_WORK_TIMEOUT_MS}"
        ));
    }
    Ok(())
}

fn generate_timeout_deadline(timeout_ms: i64) -> Result<Option<tokio::time::Instant>, String> {
    if timeout_ms < 0 {
        return Err(format!(
            "InvalidTransportBinding: generation timeout_ms {timeout_ms} must be non-negative"
        ));
    }
    if timeout_ms == 0 {
        return Ok(None);
    }
    tokio::time::Instant::now()
        .checked_add(Duration::from_millis(timeout_ms as u64))
        .map(Some)
        .ok_or_else(|| {
            format!(
                "InvalidTransportBinding: generation timeout_ms {timeout_ms} is not representable"
            )
        })
}

#[derive(Clone, Copy)]
enum MediaKind {
    ImageOrDocument,
    Audio,
}

fn media_kind_for_key(key: &str) -> Option<MediaKind> {
    match key {
        "image" | "images" | "document" => Some(MediaKind::ImageOrDocument),
        "audio" => Some(MediaKind::Audio),
        _ => None,
    }
}

fn validate_media_value_with_limits(
    value: &rmpv::Value,
    media_kind: Option<MediaKind>,
    max_image_or_document_bytes: usize,
    max_audio_bytes: usize,
) -> Result<(), String> {
    match value {
        rmpv::Value::Binary(data) => {
            let Some(kind) = media_kind else {
                return Ok(());
            };
            let (name, limit) = match kind {
                MediaKind::ImageOrDocument => ("image/document", max_image_or_document_bytes),
                MediaKind::Audio => ("audio", max_audio_bytes),
            };
            if data.len() > limit {
                return Err(format!(
                    "MediaTooLarge: {name} binary is {} bytes; maximum is {limit}",
                    data.len()
                ));
            }
            Ok(())
        }
        rmpv::Value::Array(values) => {
            for child in values {
                validate_media_value_with_limits(
                    child,
                    media_kind,
                    max_image_or_document_bytes,
                    max_audio_bytes,
                )?;
            }
            Ok(())
        }
        rmpv::Value::Map(entries) => {
            for (key, child) in entries {
                let child_kind = key.as_str().and_then(media_kind_for_key).or(media_kind);
                validate_media_value_with_limits(
                    child,
                    child_kind,
                    max_image_or_document_bytes,
                    max_audio_bytes,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_media_value(value: &rmpv::Value) -> Result<(), String> {
    validate_media_value_with_limits(value, None, MAX_IMAGE_OR_DOCUMENT_BYTES, MAX_AUDIO_BYTES)
}

fn validate_work_items_with_limit(
    body: &RequestBody,
    items: &[WorkItem],
    max_items: usize,
) -> Result<(), String> {
    if body.request_id.is_empty()
        || body.request_id.len() > MAX_REQUEST_ID_BYTES
        || body.request_id.chars().any(char::is_whitespace)
    {
        return Err("InvalidTransportBinding: invalid request_id".to_string());
    }
    if items.is_empty() {
        return Err("InvalidTransportBinding: items must not be empty".to_string());
    }
    if items.len() > max_items {
        return Err(format!(
            "InvalidTransportBinding: {} work items exceeds maximum {max_items}",
            items.len()
        ));
    }
    let total_items = u32::try_from(items.len())
        .map_err(|_| "InvalidTransportBinding: too many work items".to_string())?;
    let mut indices = HashSet::with_capacity(items.len());
    for wi in items {
        if wi.request_id != body.request_id {
            return Err(format!(
                "InvalidTransportBinding: item request_id {:?} does not match envelope",
                wi.request_id
            ));
        }
        if wi.total_items != total_items {
            return Err(format!(
                "InvalidTransportBinding: item {} total_items {} does not match batch {total_items}",
                wi.work_item_id, wi.total_items
            ));
        }
        if wi.item_index >= total_items || !indices.insert(wi.item_index) {
            return Err(format!(
                "InvalidTransportBinding: duplicate or out-of-range item_index {}",
                wi.item_index
            ));
        }
        let expected_work_item_id = format!("{}.{}", body.request_id, wi.item_index);
        if wi.work_item_id != expected_work_item_id {
            return Err(format!(
                "InvalidTransportBinding: work_item_id {:?} does not match {:?}",
                wi.work_item_id, expected_work_item_id
            ));
        }
        if wi.operation != body.endpoint {
            return Err(format!(
                "InvalidTransportBinding: operation {:?} does not match endpoint {:?}",
                wi.operation, body.endpoint
            ));
        }
        if wi.model_id != body.model {
            return Err(format!(
                "InvalidTransportBinding: model_id {:?} does not match envelope model {:?}",
                wi.model_id, body.model
            ));
        }
        if wi.engine != body.engine {
            return Err(format!(
                "InvalidTransportBinding: engine {:?} does not match envelope engine {:?}",
                wi.engine, body.engine
            ));
        }
        if wi.bundle_config_hash != body.bundle_config_hash {
            return Err(format!(
                "InvalidTransportBinding: bundle_config_hash {:?} does not match envelope",
                wi.bundle_config_hash
            ));
        }
        if wi.payload_ref.is_some() || wi.query_payload_ref.is_some() {
            return Err(format!(
                "InvalidTransportBinding: unresolved payload reference on {}",
                wi.work_item_id
            ));
        }
        if let Some(item) = &wi.item {
            validate_media_value(item)?;
        }
        if let Some(query_item) = &wi.query_item {
            validate_media_value(query_item)?;
        }
        if let Some(score_items) = &wi.score_items {
            for score_item in score_items {
                validate_media_value(score_item)?;
            }
        }
    }
    Ok(())
}

fn validate_work_items(body: &RequestBody, items: &[WorkItem]) -> Result<(), String> {
    validate_work_items_with_limit(body, items, MAX_WORK_ITEMS_PER_CALL)
}

fn normalize_pool(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

/// Worker-side admission re-check — the local-mode mirror of the
/// NATS decode-path `PoolAdmissionGate` check, reduced exactly like the
/// reference lane's `_admission_error`: one pool identity, no
/// assigned-logical-pool list, so "requested pool must be empty or equal
/// the lane pool" with batch meta and per-item fields required to agree.
fn admission_error(lane_pool: &str, meta_pool: &str, item_pool: &str) -> Option<String> {
    if !meta_pool.is_empty() && !item_pool.is_empty() && meta_pool != item_pool {
        return Some(format!(
            "admission_pool mismatch: batch meta={meta_pool:?} item={item_pool:?}"
        ));
    }
    for requested in [meta_pool, item_pool] {
        if !requested.is_empty() && requested != lane_pool {
            return Some(format!(
                "lane pool {lane_pool:?} does not serve admission_pool {requested:?}"
            ));
        }
    }
    None
}

fn error_result(wi: &WorkItem, worker_id: &str, code: &str, message: &str) -> WorkResult {
    WorkResult {
        work_item_id: wi.work_item_id.clone(),
        request_id: wi.request_id.clone(),
        item_index: wi.item_index,
        success: false,
        result_msgpack: Vec::new(),
        error: Some(message.to_string()),
        error_code: Some(code.to_string()),
        inference_ms: None,
        queue_ms: None,
        processing_ms: None,
        worker_id: Some(worker_id.to_string()),
        tokenization_ms: None,
        postprocessing_ms: None,
        payload_fetch_ms: None,
        units: None,
        // Local-ingest deliveries are worker-direct by construction: the
        // caller addressed this worker's socket, not a pool subject.
        worker_direct: true,
        executed_bundle_config_hash: None,
    }
}

async fn publish_work(
    body: RequestBody,
    shared: &IngestShared,
    span: &mut tracing::Span,
) -> Result<Vec<u8>, String> {
    validate_publish_work_timeout(body.timeout_ms)?;
    validate_payload_digest(&body)?;
    let mut items: Vec<WorkItem> = rmp_serde::from_slice(&body.items)
        .map_err(|e| format!("DecodeError: items is not a msgpack WorkItem array: {e}"))?;
    validate_work_items(&body, &items)?;
    *span = local_ingest_span(&body, &mut items);
    publish_validated_work(body, items, shared)
        .instrument(span.clone())
        .await
}

async fn publish_validated_work(
    body: RequestBody,
    items: Vec<WorkItem>,
    shared: &IngestShared,
) -> Result<Vec<u8>, String> {
    let n = items.len();
    let mut results: Vec<Option<WorkResult>> = Vec::with_capacity(n);
    results.resize_with(n, || None);

    let (tx, mut rx) = mpsc::unbounded_channel::<LocalDeliveryEvent>();
    let meta_pool = normalize_pool(&body.admission_pool);
    let mut dispatchable: Vec<(WorkItem, Delivery)> = Vec::with_capacity(n);
    for (slot, wi) in items.iter().enumerate() {
        let item_pool = normalize_pool(&wi.admission_pool);
        if let Some(reason) = admission_error(&shared.lane_pool, &meta_pool, &item_pool) {
            results[slot] = Some(error_result(
                wi,
                &shared.worker_id,
                POOL_ADMISSION_ERROR_CODE,
                &reason,
            ));
            continue;
        }
        dispatchable.push((
            wi.clone(),
            Delivery::Local(LocalDelivery::new(slot, 0, tx.clone())),
        ));
    }
    // `tx` stays alive for the whole collect loop so NAK-triggered
    // re-dispatches can mint fresh senders. Loop termination is driven by
    // the `pending` count (every delivery settles as Result or Retry — the
    // dispatcher's settlement invariant, the same one the NATS path
    // ultimately backstops with ack_wait), plus timeout/shutdown.

    let mut pending = results.iter().filter(|r| r.is_none()).count();
    if pending > 0 {
        let dispatcher = Arc::clone(&shared.dispatcher);
        let batch_size = dispatchable.len();
        tokio::spawn(async move {
            dispatcher
                .dispatch_decoded(dispatchable, batch_size, Instant::now())
                .await;
        });
    }

    // A missing or non-positive `timeout_ms` must NOT disable the deadline:
    // that turns the timeout arm into `pending()` forever, pinning the
    // connection and leaking the detached dispatch if a slot never settles.
    // Fall back to a bounded default so every op is guaranteed to settle.
    let timeout_ms = if body.timeout_ms > 0 {
        body.timeout_ms as u64
    } else {
        DEFAULT_PUBLISH_WORK_TIMEOUT_MS
    };
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    while pending > 0 {
        let event = tokio::select! {
            biased;
            _ = shared.shutdown.wait() => {
                // Wire lifecycle: in-flight ops answer the exact
                // string "cancelled" on shutdown.
                return Err("cancelled".to_string());
            }
            _ = tokio::time::sleep_until(deadline) => {
                return Err(format!(
                    "TimeoutError: publish_work timed out after {timeout_ms}ms (lane '{}')",
                    body.lane
                ));
            }
            event = rx.recv() => match event {
                Some(e) => e,
                // Unreachable while our own `tx` is alive; defensive break.
                None => break,
            },
        };
        match event {
            LocalDeliveryEvent::Result { slot, result } => {
                if slot < results.len() && results[slot].is_none() {
                    results[slot] = Some(*result);
                    pending -= 1;
                } else {
                    debug!(slot, "local-ingest: duplicate/out-of-range result ignored");
                }
            }
            LocalDeliveryEvent::Retry {
                slot,
                attempt,
                delay_ms,
            } => {
                if slot >= results.len() || results[slot].is_some() {
                    debug!(slot, "local-ingest: retry for settled slot ignored");
                    continue;
                }
                if attempt >= LOCAL_REDELIVERY_MAX_ATTEMPTS {
                    let wi = &items[slot];
                    warn!(
                        work_item_id = %wi.work_item_id,
                        attempts = attempt + 1,
                        "local-ingest: redelivery budget exhausted — typed error"
                    );
                    results[slot] = Some(error_result(
                        wi,
                        &shared.worker_id,
                        "inference_error",
                        &format!(
                            "worker requested redelivery (nak) for {:?} but the local-ingest \
                             lane has no broker redelivery; exhausted {LOCAL_REDELIVERY_MAX_ATTEMPTS} \
                             in-lane retries",
                            wi.work_item_id
                        ),
                    ));
                    pending -= 1;
                    continue;
                }
                // Bounded in-process redelivery: re-dispatch this single
                // item after the NAK delay (capped — the caller is
                // synchronously waiting).
                let delay = Duration::from_millis(delay_ms.min(LOCAL_RETRY_MAX_DELAY_MS));
                let wi = items[slot].clone();
                let delivery = Delivery::Local(LocalDelivery::new(slot, attempt + 1, tx.clone()));
                let dispatcher = Arc::clone(&shared.dispatcher);
                debug!(
                    work_item_id = %wi.work_item_id,
                    attempt = attempt + 1,
                    delay_ms = delay.as_millis() as u64,
                    "local-ingest: re-dispatching NAKed item"
                );
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    dispatcher
                        .dispatch_decoded(vec![(wi, delivery)], 1, Instant::now())
                        .await;
                });
            }
        }
    }

    drop(tx);
    let final_results: Vec<WorkResult> = results
        .into_iter()
        .enumerate()
        .map(|(slot, r)| {
            r.unwrap_or_else(|| {
                // A delivery was dropped without settling (should not
                // happen; the dispatcher always answers). Fail typed
                // rather than hanging or omitting the slot.
                error_result(
                    &items[slot],
                    &shared.worker_id,
                    "inference_error",
                    "work item was dropped without an outcome",
                )
            })
        })
        .collect();
    rmp_serde::to_vec_named(&final_results)
        .map_err(|e| format!("EncodeError: results encode failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_work_item() -> WorkItem {
        WorkItem {
            work_item_id: "req-1.0".into(),
            request_id: "req-1".into(),
            item_index: 0,
            total_items: 1,
            accepts_result_chunks: false,
            operation: "encode".into(),
            model_id: "test/model".into(),
            profile_id: "default".into(),
            engine: String::new(),
            pool_name: "default".into(),
            admission_pool: "default".into(),
            machine_profile: "cpu".into(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )])),
            payload_ref: None,
            output_types: Some(vec!["dense".into()]),
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: String::new(),
            traceparent: None,
            tracestate: None,
            timestamp: 0.0,
            deadline: None,
        }
    }

    fn bound_body(items: &[WorkItem]) -> RequestBody {
        let mut body = RequestBody {
            lane: "default|cpu|test/model".into(),
            endpoint: "encode".into(),
            model: "test/model".into(),
            engine: String::new(),
            admission_pool: "default".into(),
            bundle_config_hash: String::new(),
            request_id: "req-1".into(),
            params: serde_bytes::ByteBuf::from(vec![0x80]),
            items: serde_bytes::ByteBuf::from(rmp_serde::to_vec_named(items).unwrap()),
            dispatch_context: serde_bytes::ByteBuf::from(b"opaque-caller-context".to_vec()),
            payload_digest: serde_bytes::ByteBuf::new(),
            timeout_ms: 1_000,
            traceparent: None,
            tracestate: None,
        };
        body.payload_digest = serde_bytes::ByteBuf::from(compute_payload_digest(&body).to_vec());
        body
    }

    #[tokio::test]
    async fn transport_span_parents_decoded_work_and_ends_on_cancel() {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_sdk::trace::{InMemorySpanExporter, SdkTracerProvider};
        use tracing::instrument::WithSubscriber;
        use tracing_subscriber::prelude::*;

        opentelemetry::global::set_text_map_propagator(
            opentelemetry_sdk::propagation::TraceContextPropagator::new(),
        );
        let exporter = InMemorySpanExporter::default();
        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter.clone())
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
        let dispatch = tracing::Dispatch::new(subscriber);
        let mut items = vec![sample_work_item()];
        let mut body = bound_body(&items);
        let original_items = body.items.clone();
        let original_digest = body.payload_digest.clone();
        body.traceparent = Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".into());
        body.tracestate = Some("vendor=value".into());
        let span =
            tracing::dispatcher::with_default(&dispatch, || local_ingest_span(&body, &mut items));
        let child = span.context().span().span_context().clone();
        let extracted = crate::observability::propagation::extract_context_from_w3c(
            items[0].traceparent.as_deref(),
            items[0].tracestate.as_deref(),
        );
        assert_eq!(extracted.span().span_context().span_id(), child.span_id());
        assert_eq!(body.items, original_items);
        assert_eq!(body.payload_digest, original_digest);
        validate_payload_digest(&body).unwrap();
        let (started, ready) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(
            async move {
                let _ = started.send(());
                std::future::pending::<()>().await;
            }
            .instrument(span)
            .with_subscriber(dispatch),
        );
        ready.await.unwrap();
        assert!(exporter.get_finished_spans().unwrap().is_empty());
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        let spans = exporter.get_finished_spans().unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "sidecar.local_ingest");
        assert_eq!(spans[0].parent_span_id.to_string(), "b7ad6b7169203331");
        assert!(spans[0].events.is_empty());
        assert!(spans[0].links.is_empty());
        provider.shutdown().unwrap();
    }

    #[test]
    fn malformed_optional_carriers_do_not_reject_request_decode() {
        for value in [
            rmpv::Value::from(42),
            rmpv::Value::from(true),
            rmpv::Value::from(-1),
            rmpv::Value::from(1.5),
            rmpv::Value::Nil,
            rmpv::Value::Map(vec![(
                "nested".into(),
                rmpv::Value::Array(vec![rmpv::Value::Nil; 100_000]),
            )]),
            rmpv::Value::Binary(vec![1, 2]),
            rmpv::Value::Ext(1, vec![1, 2]),
            rmpv::Value::Array(vec![]),
            rmpv::Value::from("x".repeat(513)),
        ] {
            let frame = rmpv::Value::Map(vec![
                ("traceparent".into(), value.clone()),
                ("tracestate".into(), value),
            ]);
            let body: RequestBody =
                rmp_serde::from_slice(&rmp_serde::to_vec_named(&frame).unwrap()).unwrap();
            assert!(body.traceparent.is_none());
            assert!(body.tracestate.is_none());
        }
    }

    #[test]
    fn missing_or_invalid_transport_parent_keeps_original_items() {
        for parent in [None, Some("private-invalid-value".to_string())] {
            let mut items = vec![sample_work_item()];
            items[0].traceparent = Some("original-parent".into());
            let mut body = bound_body(&items);
            body.traceparent = parent;
            assert!(local_ingest_span(&body, &mut items).is_disabled());
            assert_eq!(items[0].traceparent.as_deref(), Some("original-parent"));
        }
    }

    #[test]
    fn payload_digest_binds_context_route_request_and_bytes() {
        let items = vec![sample_work_item()];
        let mut body = bound_body(&items);
        assert_eq!(validate_payload_digest(&body), Ok(()));
        let mut missing_context = bound_body(&items);
        missing_context.dispatch_context = serde_bytes::ByteBuf::new();
        assert!(validate_payload_digest(&missing_context)
            .unwrap_err()
            .contains("dispatch_context must not be empty"));

        body.dispatch_context[0] ^= 1;
        assert!(validate_payload_digest(&body)
            .unwrap_err()
            .contains("payload_digest mismatch"));
    }

    #[test]
    fn work_item_validation_rejects_cross_request_and_unresolved_ref() {
        let mut item = sample_work_item();
        let body = bound_body(std::slice::from_ref(&item));
        item.request_id = "other".into();
        assert!(validate_work_items(&body, &[item.clone()])
            .unwrap_err()
            .contains("request_id"));

        item.request_id = "req-1".into();
        item.payload_ref = Some("shared/path".into());
        assert!(validate_work_items(&body, &[item])
            .unwrap_err()
            .contains("unresolved payload reference"));
    }

    #[test]
    fn work_item_validation_binds_execution_authority_fields() {
        let item = sample_work_item();
        let body = bound_body(std::slice::from_ref(&item));

        let mut wrong_model = item.clone();
        wrong_model.model_id = "other/model".into();
        assert!(validate_work_items(&body, &[wrong_model])
            .unwrap_err()
            .contains("model_id"));

        let mut wrong_engine = item.clone();
        wrong_engine.engine = "other-engine".into();
        assert!(validate_work_items(&body, &[wrong_engine])
            .unwrap_err()
            .contains("engine"));

        let mut wrong_hash = item;
        wrong_hash.bundle_config_hash = "other-hash".into();
        assert!(validate_work_items(&body, &[wrong_hash])
            .unwrap_err()
            .contains("bundle_config_hash"));
    }

    #[test]
    fn unary_timeout_is_bounded_but_generation_zero_is_unlimited() {
        assert_eq!(validate_publish_work_timeout(0), Ok(()));
        assert_eq!(
            validate_publish_work_timeout(MAX_PUBLISH_WORK_TIMEOUT_MS),
            Ok(())
        );
        assert!(
            validate_publish_work_timeout(MAX_PUBLISH_WORK_TIMEOUT_MS + 1)
                .unwrap_err()
                .contains("must be between")
        );
        assert!(validate_publish_work_timeout(-1).is_err());
        assert!(validate_publish_work_timeout(i64::MIN).is_err());
        assert!(validate_publish_work_timeout(i64::MAX).is_err());

        assert!(generate_timeout_deadline(0).unwrap().is_none());
        assert!(generate_timeout_deadline(600_001).unwrap().is_some());
        assert!(generate_timeout_deadline(i64::MAX).unwrap().is_some());
        assert!(generate_timeout_deadline(-1).is_err());
    }

    #[test]
    fn work_item_count_is_bounded_before_index_set_allocation() {
        let first = sample_work_item();
        let mut second = sample_work_item();
        second.work_item_id = "req-1.1".into();
        second.item_index = 1;
        second.total_items = 2;
        let body = bound_body(&[first.clone(), second.clone()]);
        let error = validate_work_items_with_limit(&body, &[first, second], 1).unwrap_err();
        assert!(error.contains("exceeds maximum 1"));
    }

    #[test]
    fn media_limits_apply_to_nested_binary_without_large_test_allocations() {
        let image = rmpv::Value::Map(vec![(
            rmpv::Value::from("images"),
            rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                rmpv::Value::from("data"),
                rmpv::Value::Binary(vec![0; 9]),
            )])]),
        )]);
        let audio = rmpv::Value::Map(vec![(
            rmpv::Value::from("audio"),
            rmpv::Value::Map(vec![(
                rmpv::Value::from("data"),
                rmpv::Value::Binary(vec![0; 13]),
            )]),
        )]);

        assert!(validate_media_value_with_limits(&image, None, 8, 16)
            .unwrap_err()
            .contains("image/document"));
        assert!(validate_media_value_with_limits(&audio, None, 16, 12)
            .unwrap_err()
            .contains("audio"));
    }

    #[test]
    fn oversized_response_becomes_small_typed_error() {
        let frame =
            encode_response_with_limit(7, true, None, rmpv::Value::Binary(vec![0; 1_024]), 256);
        let payload_len = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
        assert!(payload_len <= 256);
        let response: rmpv::Value = rmp_serde::from_slice(&frame[4..]).unwrap();
        let rmpv::Value::Map(fields) = response else {
            panic!("response map");
        };
        assert_eq!(
            fields
                .iter()
                .find(|(key, _)| key.as_str() == Some("ok"))
                .and_then(|(_, value)| value.as_bool()),
            Some(false)
        );
    }

    fn response_field<'a>(value: &'a rmpv::Value, key: &str) -> &'a rmpv::Value {
        let rmpv::Value::Map(fields) = value else {
            panic!("response map")
        };
        fields
            .iter()
            .find(|(field, _)| field.as_str() == Some(key))
            .map(|(_, value)| value)
            .expect("response field")
    }

    #[test]
    fn generation_frames_match_canonical_v02_contract() {
        let frame = encode_generate_chunk(17, 3, b"chunk".to_vec()).unwrap();
        let response: rmpv::Value = rmp_serde::from_slice(&frame[4..]).unwrap();
        assert_eq!(response_field(&response, "id").as_u64(), Some(17));
        assert_eq!(response_field(&response, "ok").as_bool(), Some(true));
        let body = response_field(&response, "body");
        assert_eq!(response_field(body, "seq").as_u64(), Some(3));
        assert_eq!(
            response_field(body, "chunk"),
            &rmpv::Value::Binary(b"chunk".to_vec())
        );

        let error = crate::dispatcher::GenerateDispatchError {
            code: "INVALID_INPUT".to_string(),
            message: "bad image".to_string(),
        };
        let frame = encode_generate_error(18, &error);
        let response: rmpv::Value = rmp_serde::from_slice(&frame[4..]).unwrap();
        assert_eq!(response_field(&response, "id").as_u64(), Some(18));
        assert_eq!(response_field(&response, "ok").as_bool(), Some(false));
        assert_eq!(
            response_field(&response, "error").as_str(),
            Some("INVALID_INPUT: bad image")
        );
        let body = response_field(&response, "body");
        assert_eq!(body, &empty_body());

        let frame = encode_generate_final(19, 4);
        let response: rmpv::Value = rmp_serde::from_slice(&frame[4..]).unwrap();
        let body = response_field(&response, "body");
        assert_eq!(response_field(body, "final").as_bool(), Some(true));
        let outcome = response_field(body, "outcome");
        assert_eq!(response_field(outcome, "status").as_str(), Some("complete"));
        assert_eq!(response_field(outcome, "chunks").as_u64(), Some(4));
    }

    #[test]
    fn oversized_generation_chunk_is_rejected_before_any_terminal_is_encoded() {
        let error = encode_generate_chunk_with_limit(17, 0, vec![0; 1_024], 256).unwrap_err();
        assert!(error.contains("ResultTooLarge"));
    }

    #[tokio::test]
    async fn generation_terminal_is_last_and_tombstones_late_chunks() {
        let (server, mut client) = UnixStream::pair().unwrap();
        let (_, write_half) = server.into_split();
        let lifecycle = Arc::new(ConnectionLifecycle::default());
        let writer = ConnectionWriter::spawn(write_half, Arc::clone(&lifecycle));
        let stream = GenerateStreamWriter::new(23, writer);

        stream.send_chunk(b"one".to_vec()).await.unwrap();
        let chunk = read_frame(&mut client).await.unwrap().unwrap();
        let chunk: rmpv::Value = rmp_serde::from_slice(&chunk).unwrap();
        let body = response_field(&chunk, "body");
        assert_eq!(response_field(body, "seq").as_u64(), Some(0));

        stream.finish_success().await.unwrap();
        let terminal = read_frame(&mut client).await.unwrap().unwrap();
        let terminal: rmpv::Value = rmp_serde::from_slice(&terminal).unwrap();
        let body = response_field(&terminal, "body");
        assert_eq!(response_field(body, "final").as_bool(), Some(true));
        assert_eq!(
            response_field(response_field(body, "outcome"), "chunks").as_u64(),
            Some(1)
        );

        let error = stream.send_chunk(b"late".to_vec()).await.unwrap_err();
        assert!(error.contains("after its transport terminal"));
        assert!(stream.finish_success().await.is_err());
        lifecycle.close();
    }

    #[tokio::test]
    async fn connection_operation_budget_blocks_until_capacity_returns() {
        let budget = Arc::new(Semaphore::new(1));
        let lifecycle = ConnectionLifecycle::default();
        let shutdown = Shutdown::new();
        let held = acquire_connection_operation(&budget, &lifecycle, &shutdown)
            .await
            .expect("initial operation admitted");

        assert!(
            tokio::time::timeout(
                Duration::from_millis(10),
                acquire_connection_operation(&budget, &lifecycle, &shutdown),
            )
            .await
            .is_err(),
            "a saturated connection must stop admitting work"
        );

        drop(held);
        assert!(
            acquire_connection_operation(&budget, &lifecycle, &shutdown)
                .await
                .is_some(),
            "released capacity must admit the next operation"
        );
    }

    #[tokio::test]
    async fn reserved_cancel_admission_survives_saturated_data_capacity() {
        let data_budget = Arc::new(Semaphore::new(1));
        let control_budget = Arc::new(Semaphore::new(1));
        let lifecycle = ConnectionLifecycle::default();
        let shutdown = Shutdown::new();
        let _held_data = Arc::clone(&data_budget)
            .acquire_owned()
            .await
            .expect("initial data operation admitted");

        assert!(matches!(
            admit_decoded_operation(
                OP_PUBLISH_WORK,
                &data_budget,
                &control_budget,
                &lifecycle,
                &shutdown,
            )
            .await,
            DecodedOperationAdmission::DataCapacityExhausted
        ));
        assert!(matches!(
            admit_decoded_operation(
                OP_CANCEL,
                &data_budget,
                &control_budget,
                &lifecycle,
                &shutdown,
            )
            .await,
            DecodedOperationAdmission::Admitted(_)
        ));
    }

    #[tokio::test]
    async fn reserved_cancel_bytes_survive_saturated_data_bytes() {
        let cancel_payload = rmp_serde::to_vec_named(&RequestEnvelope {
            id: 2,
            op: OP_CANCEL.to_string(),
            body: RequestBody {
                request_id: "x".repeat(MAX_REQUEST_ID_BYTES),
                ..RequestBody::default()
            },
        })
        .unwrap();
        assert!(cancel_payload.len() <= MAX_RESERVED_CONTROL_REQUEST_BYTES_PER_CONNECTION);

        let buffered = Arc::new(Semaphore::new(MAX_BUFFERED_REQUEST_BYTES_PER_CONNECTION));
        let retained_data = Arc::new(Semaphore::new(MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION));
        let _buffered_data = Arc::clone(&buffered)
            .acquire_many_owned(MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION as u32)
            .await
            .unwrap();
        let _retained_data = Arc::clone(&retained_data)
            .acquire_many_owned(MAX_INFLIGHT_REQUEST_BYTES_PER_CONNECTION as u32)
            .await
            .unwrap();

        assert!(retained_data.try_acquire().is_err());
        assert!(
            buffered
                .try_acquire_many(cancel_payload.len() as u32)
                .is_ok(),
            "a valid cancel frame must fit after the data byte budget is saturated"
        );
    }

    #[tokio::test]
    async fn retained_request_byte_budget_blocks_a_second_max_sized_frame() {
        let (mut writer, mut reader) = tokio::io::duplex(32);
        let frame = [4u32.to_le_bytes().as_slice(), b"full"].concat();
        writer.write_all(&frame).await.unwrap();
        writer.write_all(&frame).await.unwrap();

        let budget = Arc::new(Semaphore::new(4));
        let (first, first_permit) = read_frame_with_budget(&mut reader, &budget)
            .await
            .unwrap()
            .expect("first frame");
        assert_eq!(first, b"full");

        let second_budget = Arc::clone(&budget);
        let second =
            tokio::spawn(async move { read_frame_with_budget(&mut reader, &second_budget).await });
        tokio::task::yield_now().await;
        assert!(
            !second.is_finished(),
            "the next full-budget frame must wait without reading its body"
        );

        drop(first_permit);
        let (second, _second_permit) = tokio::time::timeout(Duration::from_millis(100), second)
            .await
            .expect("released byte capacity wakes the next frame")
            .unwrap()
            .unwrap()
            .expect("second frame");
        assert_eq!(second, b"full");
    }

    #[test]
    fn active_envelope_id_is_unique_until_its_terminal_guard_drops() {
        let active = Arc::new(StdMutex::new(HashSet::new()));
        let first = ActiveOperationIdGuard::claim(7, Arc::clone(&active)).expect("first claim");
        assert!(ActiveOperationIdGuard::claim(7, Arc::clone(&active)).is_none());

        drop(first);
        assert!(ActiveOperationIdGuard::claim(7, active).is_some());
    }

    #[test]
    fn generation_request_ids_are_bounded_and_token_safe() {
        assert!(valid_request_id("req-1"));
        assert!(!valid_request_id(""));
        assert!(!valid_request_id("req 1"));
        assert!(!valid_request_id(&"x".repeat(MAX_REQUEST_ID_BYTES + 1)));
    }

    #[tokio::test]
    async fn oversized_declared_frame_is_rejected_before_body_allocation() {
        let (mut writer, mut reader) = tokio::io::duplex(4);
        let declared = (MAX_LOCAL_INGEST_FRAME_BYTES as u32) + 1;
        writer.write_all(&declared.to_le_bytes()).await.unwrap();
        let error = read_frame(&mut reader).await.unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn admission_accepts_empty_and_matching_pools() {
        assert_eq!(admission_error("default", "", ""), None);
        assert_eq!(admission_error("default", "default", ""), None);
        assert_eq!(admission_error("default", "", "default"), None);
        assert_eq!(admission_error("default", "default", "default"), None);
    }

    #[test]
    fn admission_rejects_mismatched_meta_and_item() {
        let err = admission_error("default", "a", "b").expect("mismatch rejected");
        assert!(err.contains("admission_pool mismatch"));
    }

    #[test]
    fn admission_rejects_foreign_pool() {
        let err = admission_error("default", "other", "").expect("foreign pool rejected");
        assert!(err.contains("does not serve"));
    }

    #[test]
    fn maximum_encoded_audio_fits_local_ingest_frame() {
        let audio = vec![0x5a; sie_audio_prep::DEFAULT_MAX_COMPRESSED_BYTES];
        let work_item = WorkItem {
            work_item_id: "request.0".to_owned(),
            request_id: "request".to_owned(),
            item_index: 0,
            total_items: 1,
            accepts_result_chunks: false,
            operation: "extract".to_owned(),
            model_id: "openai/whisper-large-v3-turbo".to_owned(),
            profile_id: "default".to_owned(),
            engine: "pytorch".to_owned(),
            pool_name: "default".to_owned(),
            admission_pool: "default".to_owned(),
            machine_profile: "L4".to_owned(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("audio"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("data"), rmpv::Value::Binary(audio)),
                    (rmpv::Value::from("format"), rmpv::Value::from("wav")),
                ]),
            )])),
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "hash".to_owned(),
            router_id: "router".to_owned(),
            reply_subject: "_INBOX.reply".to_owned(),
            traceparent: None,
            tracestate: None,
            timestamp: 0.0,
            deadline: None,
        };
        let items = rmp_serde::to_vec_named(&vec![work_item]).unwrap();
        let request = RequestEnvelope {
            id: 1,
            op: OP_PUBLISH_WORK.to_owned(),
            body: RequestBody {
                items: serde_bytes::ByteBuf::from(items),
                ..RequestBody::default()
            },
        };

        let payload = rmp_serde::to_vec_named(&request).unwrap();
        assert!(payload.len() <= MAX_LOCAL_INGEST_FRAME_BYTES);
    }
}
