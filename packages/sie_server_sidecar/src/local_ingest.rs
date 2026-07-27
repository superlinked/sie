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
//! * Ops: `ping` → `{}`; `publish_work` → `{results: bin}`; `cancel` →
//!   `{}` (accepted but a no-op here: the lane's one-shot batch calls
//!   are dropped by the caller side on cancel, and the gateway already
//!   discards late results — real in-flight cancellation lands with the
//!   streaming ops later).
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
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

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
async fn read_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> std::io::Result<Option<Vec<u8>>> {
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
    Ok(Some(payload))
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Lane identity + shared handles one connection task needs.
struct IngestShared {
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<Shutdown>,
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

async fn handle_connection(stream: UnixStream, shared: Arc<IngestShared>) {
    let (mut reader, writer) = stream.into_split();
    let writer = Arc::new(Mutex::new(writer));
    loop {
        let payload = tokio::select! {
            biased;
            _ = shared.shutdown.wait() => break,
            frame = read_frame(&mut reader) => match frame {
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
        // Each op runs as its own task; responses may interleave and are
        // correlated by `id` (same shape as the Python dispatcher server).
        let shared_op = Arc::clone(&shared);
        let writer_op = Arc::clone(&writer);
        tokio::spawn(async move {
            let frame = run_op(request, &shared_op).await;
            let mut w = writer_op.lock().await;
            if let Err(e) = w.write_all(&frame).await {
                debug!(error = %e, "local-ingest: response write failed (caller gone)");
            }
        });
    }
}

async fn run_op(request: RequestEnvelope, shared: &IngestShared) -> Vec<u8> {
    match request.op.as_str() {
        OP_PING => encode_response(request.id, true, None, empty_body()),
        OP_CANCEL => encode_response(request.id, true, None, empty_body()),
        OP_PUBLISH_WORK => match publish_work(request.body, shared).await {
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

fn validate_publish_timeout(timeout_ms: i64) -> Result<(), String> {
    if !(0..=MAX_PUBLISH_WORK_TIMEOUT_MS).contains(&timeout_ms) {
        return Err(format!(
            "InvalidTransportBinding: timeout_ms {timeout_ms} must be between 0 and {MAX_PUBLISH_WORK_TIMEOUT_MS}"
        ));
    }
    Ok(())
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

async fn publish_work(body: RequestBody, shared: &IngestShared) -> Result<Vec<u8>, String> {
    validate_publish_timeout(body.timeout_ms)?;
    validate_payload_digest(&body)?;
    let items: Vec<WorkItem> = rmp_serde::from_slice(&body.items)
        .map_err(|e| format!("DecodeError: items is not a msgpack WorkItem array: {e}"))?;
    validate_work_items(&body, &items)?;
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
        };
        body.payload_digest = serde_bytes::ByteBuf::from(compute_payload_digest(&body).to_vec());
        body
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
    fn publish_timeout_is_bounded() {
        assert_eq!(validate_publish_timeout(0), Ok(()));
        assert_eq!(
            validate_publish_timeout(MAX_PUBLISH_WORK_TIMEOUT_MS),
            Ok(())
        );
        assert!(validate_publish_timeout(MAX_PUBLISH_WORK_TIMEOUT_MS + 1)
            .unwrap_err()
            .contains("must be between"));
        assert!(validate_publish_timeout(-1).is_err());
        assert!(validate_publish_timeout(i64::MIN).is_err());
        assert!(validate_publish_timeout(i64::MAX).is_err());
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
