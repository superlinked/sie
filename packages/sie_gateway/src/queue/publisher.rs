use std::sync::Arc;
use std::time::{Duration, Instant};

use async_nats::jetstream;
use dashmap::DashMap;
use futures_util::future::try_join_all;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use rmp::decode::read_str_from_slice;
use rmp::Marker;

use super::payload_store::PayloadStore;
use crate::metrics;

const PAYLOAD_OFFLOAD_THRESHOLD: usize = 1_024 * 1_024; // 1 MB

/// Parameters extracted from the request body for top-level WorkItem fields.
///
/// `query_item` is stored as an `rmpv::Value` rather than
/// `serde_json::Value` so that msgpack-in requests (the hot path for
/// clients that send binary payloads) can pass the decoded value
/// straight through to the worker without an intermediate
/// JSON-shaped round-trip. For JSON-in requests the body is
/// converted once via [`json_to_rmpv`] before being stored here.
///
/// Small configuration fields (`options`, `output_schema`) stay as
/// `serde_json::Value` — they never carry binary data, and keeping
/// them JSON-shaped avoids rewriting the (de)serializer for the
/// small config surface.
#[derive(Debug, Clone, Default)]
pub struct WorkParams {
    pub output_types: Option<Vec<String>>,
    pub instruction: Option<String>,
    pub is_query: bool,
    pub options: Option<serde_json::Value>,
    pub labels: Option<Vec<String>>,
    pub output_schema: Option<serde_json::Value>,
    pub query_item: Option<rmpv::Value>,
}

/// Work item published to JetStream for queue mode.
/// Serialized as a msgpack **map** (named fields) to match the Python WorkItem TypedDict
/// that the worker consumer expects (`wi.get("operation")` etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub operation: String,
    pub model_id: String,
    #[serde(default)]
    pub profile_id: String,
    pub pool_name: String,
    pub machine_profile: String,
    /// Per-item payload. Carried as `rmpv::Value` so msgpack request
    /// bodies (especially those with `bin`/`ext` fields such as
    /// encoded numpy arrays) can round-trip to the worker byte-for-
    /// byte, without the old
    /// `msgpack → rmpv::Value → serde_json::Value → msgpack` detour
    /// that used to expand every `bin` into a `Vec<Number>`.
    #[serde(default)]
    pub item: Option<rmpv::Value>,
    #[serde(default)]
    pub payload_ref: Option<String>,
    #[serde(default)]
    pub output_types: Option<Vec<String>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub is_query: bool,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    /// Score query. Same rationale as `item` above.
    #[serde(default)]
    pub query_item: Option<rmpv::Value>,
    #[serde(default)]
    pub query_payload_ref: Option<String>,
    /// Score items. Same rationale as `item` above.
    #[serde(default)]
    pub score_items: Option<Vec<rmpv::Value>>,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub bundle_config_hash: String,
    #[serde(default)]
    pub router_id: String,
    pub reply_subject: String,
    /// Epoch seconds (f64) when the work item was created (for queue latency tracking).
    #[serde(default)]
    pub timestamp: f64,
}

/// Borrowed, serialize-only view of a [`WorkItem`].
///
/// Used on the publish hot path to avoid cloning per-item the fields
/// that every item in a batch shares (pool/model/gpu/router_id/
/// reply_subject/operation/bundle_config_hash and the whole
/// `WorkParams` block including `options` / `output_schema` which can
/// be non-trivial JSON trees). For an N-item encode/extract request
/// this saves roughly `7N` small-string clones plus `4N` deep
/// `Option<Vec<_>> / Option<serde_json::Value>` clones.
///
/// The field names, order, and serde attributes are kept identical
/// to [`WorkItem`] so that `rmp_serde::to_vec_named(&WorkItemRef)`
/// produces the same msgpack wire bytes as encoding the equivalent
/// owned `WorkItem`. Deserialization is intentionally not supported
/// — results arrive as `WorkResult`, and inbound `WorkItem`s (only
/// in tests) still use the owned form.
#[derive(Debug, Serialize)]
struct WorkItemRef<'a> {
    pub work_item_id: &'a str,
    pub request_id: &'a str,
    pub item_index: u32,
    pub total_items: u32,
    pub operation: &'a str,
    pub model_id: &'a str,
    pub profile_id: &'a str,
    pub pool_name: &'a str,
    pub machine_profile: &'a str,
    pub item: Option<&'a rmpv::Value>,
    pub payload_ref: Option<&'a str>,
    pub output_types: Option<&'a [String]>,
    pub instruction: Option<&'a str>,
    pub is_query: bool,
    pub options: Option<&'a serde_json::Value>,
    pub query_item: Option<&'a rmpv::Value>,
    pub query_payload_ref: Option<&'a str>,
    pub score_items: Option<&'a [rmpv::Value]>,
    pub labels: Option<&'a [String]>,
    pub output_schema: Option<&'a serde_json::Value>,
    pub bundle_config_hash: &'a str,
    pub router_id: &'a str,
    pub reply_subject: &'a str,
    pub timestamp: f64,
}

/// Per-request context that every [`WorkItemRef`] in a batch borrows
/// from. Grouping these lets us build the per-item view with a single
/// struct literal and keeps `publish_single` / `publish_score` from
/// taking a dozen `&str` arguments each.
struct WorkItemShared<'a> {
    request_id: &'a str,
    endpoint: &'a str,
    model: &'a str,
    pool: &'a str,
    gpu: &'a str,
    bundle_config_hash: &'a str,
    router_id: &'a str,
    reply_subject: &'a str,
    params: &'a WorkParams,
    timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    #[serde(default)]
    pub work_item_id: String,
    pub request_id: String,
    #[serde(default)]
    pub item_index: u32,
    #[serde(default)]
    pub success: bool,
    #[serde(default, with = "serde_bytes")]
    pub result_msgpack: Vec<u8>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub inference_ms: Option<f64>,
    #[serde(default)]
    pub queue_ms: Option<f64>,
    #[serde(default)]
    pub processing_ms: Option<f64>,
    #[serde(default)]
    pub worker_id: Option<String>,
    #[serde(default)]
    pub tokenization_ms: Option<f64>,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
    #[serde(default)]
    pub payload_fetch_ms: Option<f64>,
}

struct CachedStreamInfo {
    num_pending: u64,
    num_consumers: usize,
}

pub struct WorkPublisher {
    jetstream: jetstream::Context,
    router_id: String,
    payload_store: Arc<dyn PayloadStore>,
    result_timeout: Duration,
    max_stream_pending: u64,
    /// Pools we've already `get_or_create_stream`'d for, keyed by pool
    /// name so we skip the admin-API round trip on subsequent
    /// requests. The value is the pre-computed JetStream stream name
    /// (`WORK_POOL_{pool}`) so the publish hot path doesn't rebuild
    /// it with `format!` on every call.
    ensured_streams: DashMap<String, Arc<str>>,
    /// Backpressure snapshot, keyed by pool. The background monitor
    /// refreshes this every tick; the first request to a cold pool
    /// primes it synchronously in [`Self::ensure_stream`] so we
    /// don't fail-open during the initial monitor interval.
    stream_info_cache: DashMap<String, CachedStreamInfo>,
    pending_results: DashMap<String, ResultCollector>,
    inbox_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
}

struct ResultCollector {
    _total_items: u32,
    results: Vec<Option<WorkResult>>,
    sender: Option<oneshot::Sender<Vec<WorkResult>>>,
    deadline: Instant,
    operation: String,
    published_at: Instant,
}

fn stream_name(pool: &str) -> String {
    format!("WORK_POOL_{}", pool)
}

/// Normalize a model ID for use as a single NATS subject token.
///
/// The workers' JetStream pull consumer filters on `sie.work.*.{pool}`, which
/// matches **exactly one token** in the model-ID position. NATS subject tokens
/// legally contain `/` but MUST NOT contain `.`, `*`, `>`, or whitespace.
/// Without this normalization, a model with `.` in its id (e.g.
/// `vidore/colqwen2.5-v0.2`) would expand into multiple tokens, the publish
/// would not match the stream's subject filter, and JetStream would reject it
/// (surfacing to the client as a 504 / no-consumer error).
///
/// The mapping must stay in lockstep with the Python SDK
/// (`sie_sdk.queue_types.normalize_model_id`) so that workers and the gateway
/// agree on the wire-level subject:
///
///     `/`     -> `__`
///     `.`     -> `_dot_`
///     `*`     -> `_`
///     `>`     -> `_`
///     ` `     -> `_`
///
/// The encoding is not fully reversible — e.g. `org/a__b` and `org/a/b` both
/// collapse to the same token — but this is safe in practice because
/// HuggingFace model IDs do not contain literal `__`.
fn normalize_model_id(model_id: &str) -> String {
    let mut out = String::with_capacity(model_id.len() + 8);
    for ch in model_id.chars() {
        match ch {
            '/' => out.push_str("__"),
            '.' => out.push_str("_dot_"),
            '*' | '>' | ' ' => out.push('_'),
            c => out.push(c),
        }
    }
    out
}

fn work_subject(model: &str, pool: &str) -> String {
    format!("sie.work.{}.{}", normalize_model_id(model), pool)
}

/// Fast-path extraction of `request_id` from raw msgpack bytes.
/// Returns None on any parse failure (caller falls back to full deserialization).
fn extract_request_id_fast(payload: &[u8]) -> Option<&str> {
    if payload.is_empty() {
        return None;
    }

    let marker = rmp::decode::read_marker(&mut &payload[..]).ok()?;

    match marker {
        // Array format: request_id is the first element
        Marker::FixArray(n) if n >= 1 => {
            // Marker byte consumed 1 byte
            let (request_id, _) = read_str_from_slice(&payload[1..]).ok()?;
            Some(request_id)
        }
        Marker::Array16 => {
            // 1 marker byte + 2 length bytes = 3 bytes header
            if payload.len() < 3 {
                return None;
            }
            let len = u16::from_be_bytes([payload[1], payload[2]]);
            if len < 1 {
                return None;
            }
            let (request_id, _) = read_str_from_slice(&payload[3..]).ok()?;
            Some(request_id)
        }
        Marker::Array32 => {
            // 1 marker byte + 4 length bytes = 5 bytes header
            if payload.len() < 5 {
                return None;
            }
            let len = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
            if len < 1 {
                return None;
            }
            let (request_id, _) = read_str_from_slice(&payload[5..]).ok()?;
            Some(request_id)
        }

        // Map format: scan for "request_id" key
        Marker::FixMap(n) => scan_map_for_request_id(&payload[1..], n as u32),
        Marker::Map16 => {
            if payload.len() < 3 {
                return None;
            }
            let n = u16::from_be_bytes([payload[1], payload[2]]) as u32;
            scan_map_for_request_id(&payload[3..], n)
        }
        Marker::Map32 => {
            if payload.len() < 5 {
                return None;
            }
            let n = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
            scan_map_for_request_id(&payload[5..], n)
        }

        _ => None,
    }
}

/// Scan a msgpack map's key-value pairs for the "request_id" key.
fn scan_map_for_request_id(mut data: &[u8], num_entries: u32) -> Option<&str> {
    for _ in 0..num_entries {
        let (key, rest) = read_str_from_slice(data).ok()?;
        data = rest;

        if key == "request_id" {
            let (value, _) = read_str_from_slice(data).ok()?;
            return Some(value);
        }

        // Skip the value
        data = skip_msgpack_value(data)?;
    }
    None
}

/// Skip one msgpack value in the byte slice, returning the remaining bytes.
fn skip_msgpack_value(data: &[u8]) -> Option<&[u8]> {
    if data.is_empty() {
        return None;
    }

    let marker = rmp::decode::read_marker(&mut &data[..]).ok()?;
    let rest = &data[1..]; // after marker byte

    match marker {
        Marker::Null | Marker::True | Marker::False => Some(rest),
        Marker::FixPos(_) | Marker::FixNeg(_) => Some(rest),

        Marker::U8 | Marker::I8 => rest.get(1..),
        Marker::U16 | Marker::I16 => rest.get(2..),
        Marker::U32 | Marker::I32 | Marker::F32 => rest.get(4..),
        Marker::U64 | Marker::I64 | Marker::F64 => rest.get(8..),

        Marker::FixStr(len) => rest.get(len as usize..),
        Marker::Str8 => {
            let len = *rest.first()? as usize;
            rest.get(1usize.checked_add(len)?..)
        }
        Marker::Str16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Str32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(4usize.checked_add(len)?..)
        }

        Marker::Bin8 => {
            let len = *rest.first()? as usize;
            rest.get(1usize.checked_add(len)?..)
        }
        Marker::Bin16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Bin32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(4usize.checked_add(len)?..)
        }

        Marker::FixArray(n) => {
            let mut d = rest;
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Array16 => {
            if rest.len() < 2 {
                return None;
            }
            let n = u16::from_be_bytes([rest[0], rest[1]]) as u32;
            let mut d = &rest[2..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Array32 => {
            if rest.len() < 4 {
                return None;
            }
            let n = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
            let mut d = &rest[4..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }

        Marker::FixMap(n) => {
            let mut d = rest;
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Map16 => {
            if rest.len() < 2 {
                return None;
            }
            let n = u16::from_be_bytes([rest[0], rest[1]]) as u32;
            let mut d = &rest[2..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }
        Marker::Map32 => {
            if rest.len() < 4 {
                return None;
            }
            let n = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]);
            let mut d = &rest[4..];
            for _ in 0..n {
                d = skip_msgpack_value(d)?;
                d = skip_msgpack_value(d)?;
            }
            Some(d)
        }

        Marker::FixExt1 => rest.get(2..),
        Marker::FixExt2 => rest.get(3..),
        Marker::FixExt4 => rest.get(5..),
        Marker::FixExt8 => rest.get(9..),
        Marker::FixExt16 => rest.get(17..),
        Marker::Ext8 => {
            let len = *rest.first()? as usize;
            rest.get(2usize.checked_add(len)?..)
        }
        Marker::Ext16 => {
            if rest.len() < 2 {
                return None;
            }
            let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
            rest.get(3usize.checked_add(len)?..)
        }
        Marker::Ext32 => {
            if rest.len() < 4 {
                return None;
            }
            let len = u32::from_be_bytes([rest[0], rest[1], rest[2], rest[3]]) as usize;
            rest.get(5usize.checked_add(len)?..)
        }

        Marker::Reserved => None,
    }
}

impl WorkPublisher {
    pub fn new(
        jetstream: jetstream::Context,
        router_id: String,
        payload_store: Arc<dyn PayloadStore>,
        result_timeout: Duration,
        max_stream_pending: u64,
    ) -> Self {
        Self {
            jetstream,
            router_id,
            payload_store,
            result_timeout,
            max_stream_pending,
            ensured_streams: DashMap::new(),
            stream_info_cache: DashMap::new(),
            pending_results: DashMap::new(),
            inbox_handle: tokio::sync::Mutex::new(None),
        }
    }

    #[allow(dead_code)]
    pub fn router_id(&self) -> &str {
        &self.router_id
    }

    /// Ensure the stream exists for the given pool (cached — admin call happens once per pool).
    ///
    /// Returns the cached JetStream stream name as an `Arc<str>` so
    /// callers can avoid rebuilding `format!("WORK_POOL_{pool}")` on
    /// the hot path.
    pub async fn ensure_stream(&self, pool: &str) -> Result<Arc<str>, String> {
        if let Some(existing) = self.ensured_streams.get(pool) {
            return Ok(Arc::clone(&existing));
        }

        let name = stream_name(pool);
        let subjects = vec![format!("sie.work.*.{}", pool)];

        let mut stream = self
            .jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: name.clone(),
                subjects,
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(60),
                max_messages: 100_000,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("create/get stream {}: {}", name, e))?;

        // Prime the backpressure cache so the very first request to
        // this pool sees real consumer/pending numbers instead of
        // fail-open-until-next-monitor-tick (up to ~tick ms window).
        // We swallow errors here: if the info call fails we fall
        // back to the old behaviour of allowing the first request
        // through, matching the pre-change semantics.
        match stream.info().await {
            Ok(info) => {
                self.stream_info_cache.insert(
                    pool.to_string(),
                    CachedStreamInfo {
                        num_pending: info.state.messages,
                        num_consumers: info.state.consumer_count,
                    },
                );
            }
            Err(e) => {
                debug!(
                    stream = %name,
                    error = %e,
                    "priming stream info cache failed; falling back to monitor tick"
                );
            }
        }

        let arc_name: Arc<str> = Arc::from(name.as_str());
        self.ensured_streams
            .insert(pool.to_string(), Arc::clone(&arc_name));
        info!(stream = %name, "ensured JetStream stream");
        Ok(arc_name)
    }

    /// Check backpressure from cached stream info (lock-free DashMap read).
    /// Actual NATS stream.info() calls happen in the background monitor task.
    fn check_backpressure(&self, pool: &str) -> Result<(), String> {
        if let Some(info) = self.stream_info_cache.get(pool) {
            if info.num_consumers == 0 {
                return Err("no consumers available for work stream".to_string());
            }
            if info.num_pending > self.max_stream_pending {
                return Err(format!(
                    "backpressure: {} pending messages exceeds threshold {}",
                    info.num_pending, self.max_stream_pending
                ));
            }
        }
        // No cached info yet (ensure_stream failed to prime) — allow
        // through; the monitor task will fill the cache shortly.
        Ok(())
    }

    /// Clear cached stream state on NATS reconnect.
    /// After a NATS server restart, streams may have been deleted. Without clearing,
    /// the gateway would publish to non-existent streams and requests would timeout.
    pub fn clear_caches(&self) {
        self.ensured_streams.clear();
        self.stream_info_cache.clear();
        info!("cleared ensured_streams and stream_info caches (NATS reconnect)");
    }

    /// Start background task that polls stream info for all known pools.
    ///
    /// The 50 ms tick (down from 200 ms) shortens the window during
    /// which `check_backpressure` sees a stale snapshot after a burst
    /// starts draining. At the QPS this gateway serves the extra
    /// `stream.info()` calls are negligible (one per pool per tick),
    /// and traded directly against tail-latency recovery time when
    /// consumers catch up. The first-hit window itself is now zero:
    /// `ensure_stream` primes the cache synchronously.
    pub fn start_backpressure_monitor(self: &Arc<Self>) {
        let publisher = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(50));
            loop {
                interval.tick().await;

                // Snapshot (pool, stream_name) pairs so we don't have
                // to rebuild the stream name string for each pool on
                // every tick.
                let pools: Vec<(String, Arc<str>)> = publisher
                    .ensured_streams
                    .iter()
                    .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
                    .collect();

                for (pool, name) in pools {
                    let mut stream = match publisher.jetstream.get_stream(name.as_ref()).await {
                        Ok(s) => s,
                        Err(_) => continue,
                    };
                    let info = match stream.info().await {
                        Ok(i) => i,
                        Err(_) => continue,
                    };
                    publisher.stream_info_cache.insert(
                        pool,
                        CachedStreamInfo {
                            num_pending: info.state.messages,
                            num_consumers: info.state.consumer_count,
                        },
                    );
                }
            }
        });
    }

    /// Decompose a request into work items and publish to JetStream.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_work(
        &self,
        pool: &str,
        endpoint: &str,
        model: &str,
        _bundle: &str,
        gpu: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<(String, oneshot::Receiver<Vec<WorkResult>>), String> {
        let start = Instant::now();

        // Ensure stream exists (cached — first call per pool does admin API, subsequent are free)
        self.ensure_stream(pool).await?;

        // Check backpressure (lock-free read from background-updated cache)
        self.check_backpressure(pool)?;

        // UUIDv7 keeps the leading 48 bits as a big-endian Unix
        // millisecond timestamp, so lexicographic / B-tree-indexed
        // storage of request_ids (JetStream subjects, DashMap keys,
        // downstream log aggregators) stays time-sortable without
        // extra fields. v4 gave us uniqueness but nothing else.
        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let total_items = if endpoint == "score" {
            1
        } else {
            items.len() as u32
        };

        let subject = work_subject(model, pool);

        // Set up result collector (DashMap — lock-free per-key insert)
        let (tx, rx) = oneshot::channel();
        self.pending_results.insert(
            request_id.clone(),
            ResultCollector {
                _total_items: total_items,
                results: vec![None; total_items as usize],
                sender: Some(tx),
                deadline: Instant::now() + self.result_timeout,
                operation: endpoint.to_string(),
                published_at: Instant::now(),
            },
        );

        // Build and publish all work items, collecting ack futures for parallel await.
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Every work item in a request shares the same pool / model /
        // params block. `WorkItemRef` borrows these values so we don't
        // pay N × `Option<Vec<String>> / Option<serde_json::Value>`
        // deep clones on the hot path.
        let shared = WorkItemShared {
            request_id: &request_id,
            endpoint,
            model,
            pool,
            gpu,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
        };

        // Running publishes concurrently via `try_join_all` means
        // a late-failing item can race ahead of earlier successes —
        // so on any error we have to unwind the collector entry
        // and whatever payloads the successful siblings already
        // wrote to the offload store, otherwise both leak until
        // the result-timeout sweep kicks in (review feedback on
        // PR #716).
        let publish_outcome = if endpoint == "score" {
            self.publish_score(&shared, items, &subject)
                .await
                .map(|ack| vec![ack])
        } else {
            // JetStream publishes issue the send-and-receive-ack cycle
            // asynchronously per message, so a batch of N items that
            // used to serialize on `.await` now overlap their network
            // round trips. Each future in the set borrows `shared` +
            // owns its per-item `rmpv::Value`.
            let publishes = items.into_iter().enumerate().map(|(index, item_value)| {
                self.publish_single(&shared, total_items, index, item_value, &subject)
            });
            try_join_all(publishes).await
        };
        let ack_futures = match publish_outcome {
            Ok(acks) => acks,
            Err(e) => {
                self.pending_results.remove(&request_id);
                self.cleanup_offloaded_payloads(&request_id, total_items)
                    .await;
                return Err(e);
            }
        };

        // Fire-and-forget: spawn background task to monitor acks.
        // The request handler proceeds immediately to wait for inbox results.
        // JetStream acks confirm durability but our streams are ephemeral
        // (memory, 60s TTL) — clients retry on timeout if messages are lost.
        if !ack_futures.is_empty() {
            tokio::spawn(async move {
                for ack in ack_futures {
                    if let Err(e) = ack.await {
                        warn!(error = %e, "JetStream ack failed (message may be lost)");
                        metrics::QUEUE_ACK_FAILURES.inc();
                    }
                }
            });
        }

        let elapsed = start.elapsed();
        metrics::QUEUE_PUBLISH_SECONDS
            .with_label_values(&[endpoint])
            .observe(elapsed.as_secs_f64());
        metrics::QUEUE_ITEMS_PUBLISHED
            .with_label_values(&[endpoint])
            .observe(total_items as f64);

        debug!(
            request_id = %request_id,
            items = total_items,
            pool = %pool,
            endpoint = %endpoint,
            latency_ms = elapsed.as_millis(),
            "published work items"
        );

        Ok((request_id, rx))
    }

    /// Publish the single work item for a score request.
    ///
    /// The score endpoint collapses the whole request into one work
    /// item that carries the query + all candidate items, so there's
    /// no per-item fan-out here — but we still route it through the
    /// shared borrow helper to keep a single code path for encoding.
    async fn publish_score(
        &self,
        shared: &WorkItemShared<'_>,
        score_items: Vec<rmpv::Value>,
        subject: &str,
    ) -> Result<jetstream::context::PublishAckFuture, String> {
        let query_item = shared
            .params
            .query_item
            .as_ref()
            .ok_or_else(|| "score request missing query item".to_string())?;

        let work_item_id = format!("{}.0", shared.request_id);
        let ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: 0,
            total_items: 1,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            pool_name: shared.pool,
            machine_profile: shared.gpu,
            item: None,
            payload_ref: None,
            output_types: shared.params.output_types.as_deref(),
            instruction: shared.params.instruction.as_deref(),
            is_query: shared.params.is_query,
            options: shared.params.options.as_ref(),
            query_item: Some(query_item),
            query_payload_ref: None,
            score_items: Some(&score_items),
            labels: shared.params.labels.as_deref(),
            output_schema: shared.params.output_schema.as_ref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            // Build the offloaded `{query, items}` envelope by
            // borrowing the already-decoded values (no deep clone of
            // the score_items array). We only pay one extra msgpack
            // encode — the same one we'd pay before — and the resulting
            // `WorkItem` on the wire is far smaller because the items
            // live in object storage.
            let score_payload_value = rmpv::Value::Map(vec![
                (rmpv::Value::from("query"), query_item.clone()),
                (
                    rmpv::Value::from("items"),
                    rmpv::Value::Array(score_items.clone()),
                ),
            ]);
            let score_payload = rmp_serde::to_vec_named(&score_payload_value)
                .map_err(|e| format!("msgpack encode score payload: {}", e))?;
            let ref_key = format!("{}_score.bin", shared.request_id);
            if let Err(e) = self.payload_store.put(&ref_key, &score_payload).await {
                warn!(error = %e, "failed to offload score payload, sending inline");
            } else {
                let offloaded = WorkItemRef {
                    query_item: None,
                    query_payload_ref: Some(&ref_key),
                    score_items: None,
                    ..ref_item
                };
                encoded = rmp_serde::to_vec_named(&offloaded)
                    .map_err(|e| format!("msgpack encode offloaded score: {}", e))?;
                metrics::QUEUE_PAYLOAD_OFFLOADS.inc();
            }
        }

        self.jetstream
            .publish(subject.to_string(), encoded.into())
            .await
            .map_err(|e| format!("publish score work item: {}", e))
    }

    /// Publish one work item of an encode / extract / other fan-out
    /// endpoint. Returns the per-item JetStream `PublishAckFuture` so
    /// callers can await acks in the background.
    async fn publish_single(
        &self,
        shared: &WorkItemShared<'_>,
        total_items: u32,
        index: usize,
        item_value: rmpv::Value,
        subject: &str,
    ) -> Result<jetstream::context::PublishAckFuture, String> {
        let work_item_id = format!("{}.{}", shared.request_id, index);
        let mut ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: index as u32,
            total_items,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            pool_name: shared.pool,
            machine_profile: shared.gpu,
            item: Some(&item_value),
            payload_ref: None,
            output_types: shared.params.output_types.as_deref(),
            instruction: shared.params.instruction.as_deref(),
            is_query: shared.params.is_query,
            options: shared.params.options.as_ref(),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: shared.params.labels.as_deref(),
            output_schema: shared.params.output_schema.as_ref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        let offload_key;
        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            // Fail fast on encode error instead of `unwrap_or_default()`:
            // a silent empty blob in the payload store would only
            // surface as a confusing worker-side decode failure far
            // away from the real cause, and the inline path above
            // already propagates the same error via `map_err`.
            let item_msgpack = rmp_serde::to_vec_named(&item_value)
                .map_err(|e| format!("msgpack encode offloaded item: {}", e))?;
            offload_key = format!("{}_{}.bin", shared.request_id, index);
            if let Err(e) = self.payload_store.put(&offload_key, &item_msgpack).await {
                warn!(error = %e, "failed to offload payload, sending inline");
            } else {
                ref_item.item = None;
                ref_item.payload_ref = Some(&offload_key);
                encoded = rmp_serde::to_vec_named(&ref_item)
                    .map_err(|e| format!("msgpack encode offloaded: {}", e))?;
                metrics::QUEUE_PAYLOAD_OFFLOADS.inc();
            }
        }

        self.jetstream
            .publish(subject.to_string(), encoded.into())
            .await
            .map_err(|e| format!("publish work item {}/{}: {}", index, total_items, e))
    }

    /// Handle an incoming result message (called from inbox subscription).
    pub async fn handle_result(&self, result: WorkResult) {
        let request_id = result.request_id.clone();

        // Insert result into collector (per-key lock via DashMap)
        let all_done = {
            let mut entry = match self.pending_results.get_mut(&request_id) {
                Some(e) => e,
                None => {
                    warn!(request_id = %request_id, "received result for unknown request");
                    return;
                }
            };
            let collector = entry.value_mut();
            let idx = result.item_index as usize;
            if idx < collector.results.len() {
                collector.results[idx] = Some(result);
            }
            collector.results.iter().all(|r| r.is_some())
        };
        // DashMap per-key lock is released here

        if all_done {
            // Remove atomically — only one thread can win this remove
            if let Some((_, mut collector)) = self.pending_results.remove(&request_id) {
                let total = collector.results.len() as u32;
                let operation = collector.operation.clone();
                let wait_secs = collector.published_at.elapsed().as_secs_f64();
                let results: Vec<WorkResult> =
                    collector.results.drain(..).map(|r| r.unwrap()).collect();

                if let Some(sender) = collector.sender.take() {
                    let _ = sender.send(results);
                }

                metrics::QUEUE_RESULT_WAIT
                    .with_label_values(&[&operation])
                    .observe(wait_secs);

                self.cleanup_offloaded_payloads(&request_id, total).await;
            }
        }
    }

    /// Start the inbox subscription for result collection.
    /// Aborts the previous inbox loop if one exists (prevents duplicates on NATS reconnect).
    pub async fn start_inbox_subscription(
        self: &Arc<Self>,
        client: &async_nats::Client,
    ) -> Result<(), String> {
        let inbox_subject = format!("_INBOX.{}.>", self.router_id);

        let subscriber = client
            .subscribe(inbox_subject.clone())
            .await
            .map_err(|e| format!("subscribe inbox: {}", e))?;

        let publisher = Arc::clone(self);
        let new_handle = tokio::spawn(async move {
            publisher.handle_inbox(subscriber).await;
        });

        // Abort previous inbox loop before storing the new handle
        let mut handle_guard = self.inbox_handle.lock().await;
        if let Some(old_handle) = handle_guard.take() {
            old_handle.abort();
            debug!("aborted previous inbox subscription");
        }
        *handle_guard = Some(new_handle);

        info!(subject = %inbox_subject, "inbox subscription started");
        Ok(())
    }

    async fn handle_inbox(&self, mut subscriber: async_nats::Subscriber) {
        while let Some(msg) = subscriber.next().await {
            // Fast-path: extract request_id without full deserialization.
            // DashMap contains_key is lock-free.
            if let Some(request_id) = extract_request_id_fast(&msg.payload) {
                if !self.pending_results.contains_key(request_id) {
                    debug!(
                        request_id = %request_id,
                        "fast-path skip: result for unknown request"
                    );
                    metrics::QUEUE_INBOX_SKIPS.inc();
                    continue;
                }
            }

            let result: WorkResult = match rmp_serde::from_slice(&msg.payload) {
                Ok(r) => r,
                Err(e) => {
                    warn!(error = %e, "failed to decode inbox result");
                    continue;
                }
            };

            self.handle_result(result).await;
        }

        warn!("inbox subscription ended");
    }

    /// Drain pending result collectors on graceful shutdown.
    /// Waits up to `timeout` for in-flight results to arrive, then drops the rest.
    pub async fn drain_pending(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let poll_interval = Duration::from_millis(100);

        loop {
            if self.pending_results.is_empty() {
                info!("all pending queue results drained");
                return;
            }
            let count = self.pending_results.len();
            if Instant::now() >= deadline {
                warn!(
                    remaining = count,
                    "shutdown drain timeout — dropping pending results"
                );
                break;
            }
            debug!(
                remaining = count,
                "waiting for pending queue results to drain"
            );
            tokio::time::sleep(poll_interval).await;
        }

        // Force-complete remaining collectors so senders don't leak
        self.cleanup_expired().await;
    }

    /// Clean up expired result collectors.
    pub async fn cleanup_expired(&self) {
        let now = Instant::now();

        let expired: Vec<(String, u32)> = self
            .pending_results
            .iter()
            .filter(|entry| now > entry.value().deadline)
            .map(|entry| (entry.key().clone(), entry.value().results.len() as u32))
            .collect();

        for (key, total) in &expired {
            if let Some((_, mut collector)) = self.pending_results.remove(key) {
                warn!(request_id = %key, "result collector timed out");
                let results: Vec<WorkResult> = collector.results.drain(..).flatten().collect();
                if let Some(sender) = collector.sender.take() {
                    let _ = sender.send(results);
                }
            }
            self.cleanup_offloaded_payloads(key, *total).await;
        }
    }

    /// Remove offloaded payloads for a completed/expired request.
    async fn cleanup_offloaded_payloads(&self, request_id: &str, total_items: u32) {
        for index in 0..total_items {
            let key = format!("{}_{}.bin", request_id, index);
            if let Err(e) = self.payload_store.delete(&key).await {
                warn!(key = %key, error = %e, "failed to remove offloaded payload");
            }
        }

        let score_key = format!("{}_score.bin", request_id);
        if let Err(e) = self.payload_store.delete(&score_key).await {
            warn!(key = %score_key, error = %e, "failed to remove offloaded score payload");
        }
    }
}

/// Content negotiation: determine if client wants msgpack or JSON.
pub fn wants_msgpack(headers: &axum::http::HeaderMap) -> bool {
    headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|accept| {
            accept.contains("application/msgpack")
                || accept.contains("application/x-msgpack")
                || accept.contains("application/vnd.msgpack")
        })
        .unwrap_or(false)
}

/// Serialize response based on content negotiation.
#[allow(dead_code)]
pub fn encode_response(
    data: &impl Serialize,
    use_msgpack: bool,
) -> Result<(String, Vec<u8>), String> {
    if use_msgpack {
        let bytes = rmp_serde::to_vec(data).map_err(|e| format!("msgpack encode: {}", e))?;
        Ok(("application/msgpack".to_string(), bytes))
    } else {
        let bytes = serde_json::to_vec(data).map_err(|e| format!("json encode: {}", e))?;
        Ok(("application/json".to_string(), bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_name() {
        assert_eq!(stream_name("default"), "WORK_POOL_default");
        assert_eq!(stream_name("eval-l4"), "WORK_POOL_eval-l4");
    }

    #[test]
    fn test_normalize_model_id() {
        assert_eq!(normalize_model_id("BAAI/bge-m3"), "BAAI__bge-m3");
        assert_eq!(normalize_model_id("my-model"), "my-model");
        assert_eq!(
            normalize_model_id("vidore/colqwen2.5-v0.2"),
            "vidore__colqwen2_dot_5-v0_dot_2"
        );
        assert_eq!(
            normalize_model_id("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers__all-MiniLM-L6-v2"
        );
        assert_eq!(normalize_model_id("a*b"), "a_b");
        assert_eq!(normalize_model_id("a>b"), "a_b");
        assert_eq!(normalize_model_id("a b"), "a_b");
    }

    #[test]
    fn test_work_subject() {
        assert_eq!(
            work_subject("BAAI/bge-m3", "default"),
            "sie.work.BAAI__bge-m3.default"
        );
        assert_eq!(
            work_subject("my-model", "eval-l4"),
            "sie.work.my-model.eval-l4"
        );
    }

    /// Regression: model IDs containing `.` must produce exactly 4
    /// subject tokens (`sie`, `work`, `{normalized_model}`, `{pool}`) so
    /// the worker's consumer filter `sie.work.*.{pool}` matches them.
    #[test]
    fn test_work_subject_token_count_with_dotted_model() {
        let subj = work_subject("vidore/colqwen2.5-v0.2", "l4");
        let tokens: Vec<&str> = subj.split('.').collect();
        assert_eq!(
            tokens.len(),
            4,
            "subject {subj} must have 4 tokens to match sie.work.*.{{pool}}"
        );
        assert_eq!(tokens[0], "sie");
        assert_eq!(tokens[1], "work");
        assert_eq!(tokens[3], "l4");
        // Token[2] contains no '.' (it's the normalized model id).
        assert!(!tokens[2].contains('.'));
    }

    #[test]
    fn test_work_item_msgpack_roundtrip() {
        let item = WorkItem {
            work_item_id: "req-1.0".to_string(),
            request_id: "req-1".to_string(),
            item_index: 0,
            total_items: 3,
            operation: "encode".to_string(),
            model_id: "BAAI/bge-m3".to_string(),
            profile_id: String::new(),
            pool_name: "default".to_string(),
            machine_profile: "l4-spot".to_string(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )])),
            payload_ref: None,
            output_types: Some(vec!["dense".to_string()]),
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            bundle_config_hash: "abc123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.r1.req-1".to_string(),
            timestamp: 1700000000.0,
        };

        let encoded = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.work_item_id, "req-1.0");
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.item_index, 0);
        assert_eq!(decoded.total_items, 3);
        assert_eq!(decoded.operation, "encode");
        assert_eq!(decoded.model_id, "BAAI/bge-m3");
        assert_eq!(decoded.pool_name, "default");
        assert_eq!(decoded.machine_profile, "l4-spot");
        assert_eq!(
            decoded.item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )]))
        );
        assert!(decoded.payload_ref.is_none());
        assert_eq!(decoded.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(decoded.bundle_config_hash, "abc123");
        assert_eq!(decoded.router_id, "router-1");
        assert_eq!(decoded.reply_subject, "_INBOX.r1.req-1");
    }

    /// Regression: `WorkItemRef` is the borrowed view we use on the
    /// publish hot path and it **must** serialize to the exact same
    /// msgpack bytes as the owned `WorkItem`. Any drift in field
    /// names/order/serde attrs between the two would silently break
    /// worker deserialization; lock it down here.
    #[test]
    fn test_work_item_ref_matches_owned_msgpack() {
        let owned = WorkItem {
            work_item_id: "req-ref.0".to_string(),
            request_id: "req-ref".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "encode".to_string(),
            model_id: "BAAI/bge-m3".to_string(),
            profile_id: "default".to_string(),
            pool_name: "default".to_string(),
            machine_profile: "l4-spot".to_string(),
            item: Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )])),
            payload_ref: None,
            output_types: Some(vec!["dense".to_string()]),
            instruction: Some("search_document".to_string()),
            is_query: false,
            options: Some(serde_json::json!({"truncate": true})),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: Some(vec!["exp-a".to_string()]),
            output_schema: Some(serde_json::json!({"kind": "dense"})),
            bundle_config_hash: "hash123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.router-1.req-ref".to_string(),
            timestamp: 1_700_000_000.5,
        };

        let item_value = owned.item.clone().unwrap();
        let output_types = owned.output_types.clone().unwrap();
        let labels = owned.labels.clone().unwrap();
        let options = owned.options.clone().unwrap();
        let output_schema = owned.output_schema.clone().unwrap();

        let borrowed = WorkItemRef {
            work_item_id: &owned.work_item_id,
            request_id: &owned.request_id,
            item_index: owned.item_index,
            total_items: owned.total_items,
            operation: &owned.operation,
            model_id: &owned.model_id,
            profile_id: &owned.profile_id,
            pool_name: &owned.pool_name,
            machine_profile: &owned.machine_profile,
            item: Some(&item_value),
            payload_ref: None,
            output_types: Some(&output_types),
            instruction: owned.instruction.as_deref(),
            is_query: owned.is_query,
            options: Some(&options),
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: Some(&labels),
            output_schema: Some(&output_schema),
            bundle_config_hash: &owned.bundle_config_hash,
            router_id: &owned.router_id,
            reply_subject: &owned.reply_subject,
            timestamp: owned.timestamp,
        };

        let owned_bytes = rmp_serde::to_vec_named(&owned).unwrap();
        let ref_bytes = rmp_serde::to_vec_named(&borrowed).unwrap();
        assert_eq!(
            ref_bytes, owned_bytes,
            "WorkItemRef must produce byte-identical msgpack to WorkItem"
        );

        // And the bytes still decode into a WorkItem cleanly.
        let decoded: WorkItem = rmp_serde::from_slice(&ref_bytes).unwrap();
        assert_eq!(decoded.work_item_id, owned.work_item_id);
        assert_eq!(decoded.item, owned.item);
        assert_eq!(decoded.options, owned.options);
    }

    #[test]
    fn test_work_result_msgpack_roundtrip() {
        let result = WorkResult {
            work_item_id: "req-1.2".to_string(),
            request_id: "req-1".to_string(),
            item_index: 2,
            success: true,
            result_msgpack: vec![5, 6, 7],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };

        let encoded = rmp_serde::to_vec(&result).unwrap();
        let decoded: WorkResult = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.item_index, 2);
        assert!(decoded.success);
        assert_eq!(decoded.result_msgpack, vec![5, 6, 7]);
    }

    #[test]
    fn test_work_item_with_payload_ref() {
        let item = WorkItem {
            work_item_id: "req-2.0".to_string(),
            request_id: "req-2".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "encode".to_string(),
            model_id: "model".to_string(),
            profile_id: String::new(),
            pool_name: "default".to_string(),
            machine_profile: String::new(),
            item: None,
            payload_ref: Some("/tmp/payload_req-2_0.bin".to_string()),
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r1.req-2".to_string(),
            timestamp: 0.0,
        };

        let encoded = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();

        assert!(decoded.item.is_none());
        assert_eq!(
            decoded.payload_ref,
            Some("/tmp/payload_req-2_0.bin".to_string())
        );
    }

    #[test]
    fn test_wants_msgpack() {
        let mut headers = axum::http::HeaderMap::new();
        assert!(!wants_msgpack(&headers));

        headers.insert("accept", "application/json".parse().unwrap());
        assert!(!wants_msgpack(&headers));

        headers.insert("accept", "application/msgpack".parse().unwrap());
        assert!(wants_msgpack(&headers));

        headers.insert("accept", "application/x-msgpack".parse().unwrap());
        assert!(wants_msgpack(&headers));
    }

    #[test]
    fn test_encode_response_json() {
        let data = serde_json::json!({"key": "value"});
        let (content_type, bytes) = encode_response(&data, false).unwrap();
        assert_eq!(content_type, "application/json");
        let parsed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_encode_response_msgpack() {
        let data = serde_json::json!({"key": "value"});
        let (content_type, bytes) = encode_response(&data, true).unwrap();
        assert_eq!(content_type, "application/msgpack");
        let parsed: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(parsed["key"], "value");
    }

    #[test]
    fn test_payload_offload_threshold() {
        assert_eq!(PAYLOAD_OFFLOAD_THRESHOLD, 1_048_576);
    }

    // --- Fast-path request_id extraction tests ---

    #[test]
    fn test_extract_request_id_fast_array_format() {
        let result = WorkResult {
            work_item_id: "abc-123-def.0".to_string(),
            request_id: "abc-123-def".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: vec![1, 2, 3],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        assert_eq!(extracted, Some("abc-123-def.0"));
    }

    #[test]
    fn test_extract_request_id_fast_map_format() {
        let result = WorkResult {
            work_item_id: "map-req-456.1".to_string(),
            request_id: "map-req-456".to_string(),
            item_index: 1,
            success: true,
            result_msgpack: vec![10, 20],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec_named(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        // Map format: fast-path scans for "request_id" key and returns its value
        assert_eq!(extracted, Some("map-req-456"));
    }

    #[test]
    fn test_extract_request_id_fast_empty_payload() {
        assert_eq!(extract_request_id_fast(&[]), None);
    }

    #[test]
    fn test_extract_request_id_fast_invalid_payload() {
        // 0xff is a negative fixint (-1), not an array/map — returns None
        assert_eq!(extract_request_id_fast(&[0xff]), None);
        // A single null marker (no string element)
        assert_eq!(extract_request_id_fast(&[0xc0]), None);
        // Truncated data
        assert_eq!(extract_request_id_fast(&[0x91]), None);
    }

    #[test]
    fn test_extract_request_id_fast_uuid() {
        let uuid_str = "550e8400-e29b-41d4-a716-446655440000";
        let result = WorkResult {
            work_item_id: format!("{}.0", uuid_str),
            request_id: uuid_str.to_string(),
            item_index: 0,
            success: true,
            result_msgpack: vec![],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        // First field in array format is work_item_id
        assert!(extracted.is_some());
    }

    #[test]
    fn test_extract_request_id_fast_map_not_first_key() {
        // Build a map where "request_id" is not the first key
        // Use rmp_serde named format on a struct where request_id comes after other fields
        // Since WorkResult has request_id first, we manually build a map
        use rmp::encode::{write_bin, write_map_len, write_str, write_u32};

        let mut buf = Vec::new();
        write_map_len(&mut buf, 4).unwrap();
        // First key: "status"
        write_str(&mut buf, "status").unwrap();
        write_u32(&mut buf, 200).unwrap();
        // Second key: "item_index"
        write_str(&mut buf, "item_index").unwrap();
        write_u32(&mut buf, 0).unwrap();
        // Third key: "request_id"
        write_str(&mut buf, "request_id").unwrap();
        write_str(&mut buf, "found-me").unwrap();
        // Fourth key: "payload"
        write_str(&mut buf, "payload").unwrap();
        write_bin(&mut buf, &[1, 2, 3]).unwrap();

        let extracted = extract_request_id_fast(&buf);
        assert_eq!(extracted, Some("found-me"));
    }

    #[test]
    fn test_skip_msgpack_value_integers() {
        // Positive fixint (0x00..0x7f): single byte
        let data = [0x05, 0xAA]; // fixint 5, then 0xAA
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xAA]);

        // Negative fixint (0xe0..0xff): single byte
        let data = [0xe0, 0xBB]; // fixint -32, then 0xBB
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xBB]);

        // u16: marker + 2 bytes
        let data = [0xcd, 0x01, 0x00, 0xCC]; // u16(256), then 0xCC
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xCC]);
    }

    #[test]
    fn test_skip_msgpack_value_strings() {
        // fixstr "hi" (length 2)
        let data = [0xa2, b'h', b'i', 0xFF];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0xFF]);
    }

    #[test]
    fn test_skip_msgpack_value_nil_and_bools() {
        // nil
        let data = [0xc0, 0x01];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x01]);

        // true
        let data = [0xc3, 0x02];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x02]);

        // false
        let data = [0xc2, 0x03];
        let rest = skip_msgpack_value(&data).unwrap();
        assert_eq!(rest, &[0x03]);
    }
}
