//! Streaming reply collector for the generation endpoint.
//!
//! Workers publish chunk envelopes on ``_INBOX.{router_id}.{request_id}``
//! using a discriminated msgpack-map shape:
//!
//! ```text
//! { kind: "chunk", request_id, attempt_id, seq, text_delta, done,
//!   is_first?, finish_reason?, usage?, ttft_ms?, error? }
//! ```
//!
//! This module owns the gateway-side state machine that turns those
//! per-chunk messages into the aggregated v1 HTTP response. The
//! aggregated body keeps the walking-skeleton's top-level fields (``model``,
//! ``text``, ``finish_reason``, ``usage``) and adds the SIE-native
//! ``attempt_id``, ``ttft_ms``, and ``tpot_ms``.
//!
//! Attempt-ID rule: the first chunk observed for a ``request_id``
//! latches the ``current_attempt_id``. Chunks bearing any other
//! ``attempt_id`` are dropped silently. Pool-republish driven attempt
//! bumping lands with the routing rollout.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, oneshot, Notify};

use crate::observability::metrics::{
    self as telemetry, GenerationEvent, GenerationEventOutcome, GenerationEventReason,
};

/// Broadcast channel capacity for the per-request SSE chunk tap.
///
/// Workers batch outbound chunks every ~50ms / 32 tokens, so a 256-slot
/// ring is roughly 13 seconds of inter-chunk slack — well beyond the
/// inter-chunk timeout (10s) and the HTTP send pace. A slow SSE
/// consumer that lags by more than this will see ``RecvError::Lagged``
/// on the receiver side, which the handler surfaces as an inter-chunk
/// stall (same downstream effect as a real worker stall).
pub const CHUNK_TAP_CAPACITY: usize = 256;

/// One chunk envelope, decoded from the worker's msgpack publish.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChunkEnvelope {
    /// Discriminator; always ``"chunk"`` for the streaming surface. Forward-compat
    /// for the OpenAI chat shape.
    pub kind: String,
    pub request_id: String,
    #[serde(default)]
    pub attempt_id: String,
    #[serde(default)]
    pub seq: u32,
    #[serde(default)]
    pub text_delta: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_first: bool,
    /// Terminal-only fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ChunkError>,
    /// OpenAI ``tool_calls`` delta entries — one or more
    /// ``{index, id?, type, function: {name?, arguments}}`` objects
    /// emitted by the worker's :func:`parse_tool_call_stream`. ``None``
    /// when the chunk carries plain text. Each chunk envelope carries
    /// at most one logical tool-call delta (announcement or arguments
    /// body) so the SSE driver forwards them byte-for-byte.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDeltaWire>>,
    /// OpenAI per-token logprobs for the tokens in this chunk's
    /// ``text_delta`` — a list of ``ChatCompletionTokenLogprob`` objects
    /// (``{token, logprob, bytes, top_logprobs}``) produced verbatim by
    /// the worker adapter (it owns the SGLang→OpenAI translation).
    /// ``None`` when logprobs were not requested. Forwarded byte-for-byte
    /// to the SSE client and concatenated for the non-streaming body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<serde_json::Value>>,
    /// Multi-candidate (`n > 1`) results, carried on the **terminal** chunk
    /// only. For `n == 1` (the default) this is empty and the single-candidate
    /// `text_delta` stream path is used. For `n > 1` the worker runs the
    /// candidates server-side and emits them all here on the terminal chunk;
    /// the gateway turns them into the multi-entry OpenAI `choices` array. A
    /// worker that predates `n > 1` support never sets this, so a new gateway
    /// against an old worker degrades to a single candidate.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<CandidateData>,
    /// Streaming multi-candidate (`n>1 && stream`): the candidate ordinal this
    /// delta belongs to (`[0, n)`). Defaults to 0 — the single-candidate
    /// stream. The SSE driver maps it to `choices[0].index` so clients can
    /// reassemble per-candidate streams.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub choice_index: u32,
    /// Worker-side execution proof added under the stable config barrier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub executed_bundle_config_hash: Option<String>,
    /// Opaque worker-origin execution identity. Managed workers stamp every
    /// chunk; older and self-host workers omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_identity_sha256: Option<String>,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

pub(crate) fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// One candidate of a multi-candidate (`n > 1`) generation, produced
/// server-side by the worker and surfaced as one OpenAI `choices[]` entry.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CandidateData {
    #[serde(default)]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Per-token OpenAI logprobs for this candidate (same shape as the
    /// single-candidate `ChunkEnvelope::logprobs`). `None` unless requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<serde_json::Value>>,
    /// Per-candidate tool calls for the non-streaming `n>1 + tools`
    /// path (H5 non-streaming side). The worker parses each candidate's
    /// text for `<tool_call>` blocks independently and emits them here
    /// in the OpenAI non-streaming shape: `[{id, type, function:
    /// {name, arguments}}, ...]`. `None` when no tool calls were
    /// parsed for this candidate (the common case for plain text).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Wire shape of one OpenAI tool-call delta. Mirrors the worker's
/// ``ToolCallDelta`` dataclass with the function block nested under
/// ``function`` per the OpenAI streaming spec. ``id`` and
/// ``function.name`` appear only on the first delta of a call;
/// ``function.arguments`` accumulates JSON across deltas.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDeltaWire {
    #[serde(default)]
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default = "default_tool_call_type", rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionWire>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunctionWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: String,
}

fn default_tool_call_type() -> String {
    "function".to_string()
}

/// Wire-level allowlist for ``finish_reason`` values produced by the
/// worker. OpenAI canonical values plus the SIE-internal additions for
/// gateway-driven cancellation and error surfacing. Keep in lockstep with
/// ``GenerationChunk.finish_reason`` in the Python adapter.
fn is_known_finish_reason(reason: &str) -> bool {
    matches!(
        reason,
        "stop"
            | "length"
            | "tool_calls"
            | "content_filter"
            | "function_call"
            | "error"
            | "cancelled"
    )
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsageBlock {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChunkError {
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub message: String,
}

/// Outcome delivered to the HTTP handler once a terminal chunk arrives.
#[derive(Debug)]
pub struct StreamOutcome {
    pub text: String,
    pub finish_reason: String,
    pub usage: Option<UsageBlock>,
    pub attempt_id: String,
    pub ttft_ms: Option<f64>,
    pub tpot_ms: Option<f64>,
    pub error: Option<ChunkError>,
    /// Aggregated OpenAI tool calls observed across the request's
    /// chunk stream. ``None`` when no tool calls were emitted. Each
    /// entry is the fully-assembled ``{id, type, function: {name,
    /// arguments}}`` shape — the gateway concatenates incremental
    /// ``arguments`` deltas into one string per ``index`` so the
    /// non-streaming response body matches OpenAI's aggregate shape.
    pub tool_calls: Option<Vec<AggregatedToolCall>>,
    /// Aggregated OpenAI per-token logprobs across the whole stream
    /// (concatenation of every chunk's ``logprobs`` in observed order).
    /// ``None`` when logprobs were not requested. Surfaced on the
    /// non-streaming body as ``choices[0].logprobs.content``.
    pub logprobs: Option<Vec<serde_json::Value>>,
    /// Multi-candidate (`n > 1`) results from the terminal chunk. Empty for
    /// single-candidate requests (the default), in which case the body builder
    /// uses ``text`` / ``logprobs`` for the lone ``choices[0]``. When
    /// non-empty, the body builder emits one ``choices[]`` entry per element.
    pub candidates: Vec<CandidateData>,
    pub executed_bundle_config_hash: Option<String>,
    pub execution_identity_sha256: Option<String>,
}

/// Fully-assembled tool call ready to surface on the non-streaming
/// chat completion response (``choices[0].message.tool_calls[*]``).
///
/// ``index`` is preserved for diagnostics — the BTreeMap key in
/// :class:`StreamCollector` is the canonical ordering, but holding the
/// index on the struct itself lets downstream metrics / logs surface
/// which tool slot each entry belongs to without re-deriving it.
#[derive(Debug, Clone)]
pub struct AggregatedToolCall {
    #[allow(dead_code)] // retained for future per-index metrics
    pub index: u32,
    pub id: String,
    pub kind: String, // "function"
    pub name: String,
    pub arguments: String,
}

/// Per-request streaming aggregator. Owned by ``WorkPublisher.pending_streams``.
pub struct StreamCollector {
    /// Latched on first observed chunk; all later chunks with a
    /// different ``attempt_id`` are silently dropped.
    pub current_attempt_id: Option<String>,
    /// Highest ``seq`` applied so far for the latched attempt, or
    /// ``None`` before the first non-stale chunk lands. JetStream can
    /// redeliver a message within a single attempt (ack lost, consumer
    /// re-pull); a redelivered chunk carries the *same* ``seq`` it had
    /// the first time. ``apply`` drops any chunk whose ``seq`` is not
    /// strictly greater than this watermark so a redelivery cannot
    /// double-count text, fan out a duplicate SSE event, or fire a
    /// duplicate terminal. Reset by ``bump_attempt_generation`` /
    /// ``rewind_attempt_generation`` so the next attempt re-watermarks
    /// from its own first chunk.
    pub last_applied_seq: Option<u32>,
    /// Accumulated text deltas in observed order. ``seq`` is recorded
    /// for dedupe (the gateway is tolerant of re-deliveries within a
    /// single attempt — only one such delivery is expected today).
    pub chunks: Vec<(u32, String)>,
    pub final_meta: Option<TerminalMeta>,
    pub sender: Option<oneshot::Sender<StreamOutcome>>,
    pub published_at: Instant,
    pub first_chunk_at: Option<Instant>,
    pub last_chunk_at: Option<Instant>,
    /// Last output-bearing (text or tool-call) chunk. Unlike
    /// `last_chunk_at`, an empty terminal marker does not advance this clock;
    /// it is the end of the checked-in TPOT measurement window.
    last_output_at: Option<Instant>,
    /// Output-bearing chunks in the current attempt. Used as the TPOT
    /// denominator only when the worker did not provide authoritative token
    /// usage.
    output_event_count: u64,
    /// DISPATCH model id — the routed target (a ``:no-spec`` grammar
    /// variant when grammar routing fired, else the requested base).
    /// Kept for functional dispatch parity with ``PublishTarget.model``
    /// (NATS subject, republish, fallback token bucket). NEVER used to
    /// label metrics — use ``display_model`` for that.
    pub model: String,
    /// DISPLAY model id — the requested base model. Equal to ``model``
    /// unless grammar routing rewrote the dispatch target. All
    /// collector/publisher metric labels read this so no series carries
    /// the ``:no-spec`` variant (#1324). Defaults to ``model`` in
    /// [`Self::new`]; the publish path overrides it with the requested id.
    pub display_model: String,
    pub pool: String,
    /// Pulsed by ``apply()`` on every non-stale chunk. The HTTP handler
    /// uses this to reset the inter-chunk timer (§4.4.3). Wrapped in
    /// ``Arc`` so the handler can clone a reference and ``await`` on it
    /// outside the DashMap lock.
    pub activity: Arc<Notify>,
    /// Monotonically increasing attempt generation. The
    /// publisher bumps this on every NAK-/timeout-driven republish to
    /// the pool. ``apply`` clears the latched ``current_attempt_id``
    /// when it observes a new generation so the next chunk
    /// re-latches on the fresh attempt, preserving the streaming surface's
    /// stale-chunk drop semantics across the direct → pool fallback.
    pub attempt_generation: u64,
    /// Cached encoded ``WorkItem`` for fallback republishes.
    /// Empty on the original streaming path (set by
    /// ``publish_generate_streaming`` when the publish succeeds). Used
    /// by ``WorkPublisher::republish_to_pool`` to re-issue the same
    /// item on the pool subject after a NAK or first-chunk timeout.
    pub encoded_payload: Option<Vec<u8>>,
    /// Republish destination. ``Some`` for items that may
    /// be republished (currently: all generation items). Falls back
    /// to ``Pool { model, pool }`` regardless of the original target.
    pub pool_fallback_subject: Option<String>,
    /// Per-request idempotency guard so that a NAK arriving
    /// roughly concurrently with a first-chunk timeout doesn't
    /// double-publish. Flipped to ``true`` on the first republish; the
    /// publisher checks-and-sets it under the DashMap entry lock.
    pub republished: bool,
    /// Optional per-chunk broadcast tap. ``None`` for non-streaming
    /// (aggregating) requests — the chunk just lands in ``chunks`` and
    /// the HTTP handler awaits the terminal ``StreamOutcome``. ``Some``
    /// for SSE requests: every non-stale chunk applied by ``apply`` is
    /// also fanned out on this sender so the SSE handler can forward
    /// it to the HTTP client as it arrives. The tap is installed by
    /// the publisher (via :meth:`install_chunk_tap`) before the work
    /// item is published, so no chunks can race past the subscriber.
    /// A bounded broadcast channel (``CHUNK_TAP_CAPACITY``) gives a
    /// slow SSE consumer back-pressure without back-propagating into
    /// the publisher hot path — sends are non-blocking and lagged
    /// receivers surface ``RecvError::Lagged`` on their own loop.
    pub chunk_tap: Option<broadcast::Sender<ChunkEnvelope>>,
    /// Accumulated tool-call deltas keyed by ``index``. The first
    /// observed delta for each ``index`` latches ``id`` /
    /// ``function.name``; subsequent deltas append to
    /// ``function.arguments``. Ordered map so the non-streaming
    /// response surfaces calls in index order without an extra sort.
    pub tool_calls_by_index: BTreeMap<u32, AggregatedToolCall>,
    /// Accumulated OpenAI per-token logprobs across all non-stale chunks,
    /// in observed order — concatenated into the non-streaming body's
    /// ``choices[0].logprobs.content``. Empty when logprobs were not
    /// requested. Snapshotted/cleared by the attempt-generation machinery
    /// exactly like ``chunks``.
    logprobs: Vec<serde_json::Value>,
    /// Worker-origin digest latched across every accepted chunk in the current
    /// attempt. One absent, malformed, or mismatching chunk makes the attempt
    /// non-attestable while preserving ordinary streaming compatibility.
    execution_identity_sha256: Option<String>,
    execution_identity_consistent: bool,
    /// Attempt id of the most recently abandoned attempt. Usually set by
    /// [`Self::bump_attempt_generation`] from the old ``current_attempt_id``;
    /// NAK-driven republish may also fill it from the NAK envelope when the
    /// abandoned attempt had not emitted a first chunk yet. While
    /// ``current_attempt_id`` is ``None`` (post-bump, pre-relatch),
    /// :meth:`apply` refuses to re-latch on this id so a stale leftover chunk
    /// from the just-abandoned attempt cannot hijack the latch and starve the
    /// genuinely-new attempt's chunks (which would then be dropped as stale,
    /// hanging the request). Cleared the moment a non-abandoned attempt
    /// latches. uuid4 attempt ids are never reused, so an incoming chunk whose
    /// id equals this is unambiguously from the abandoned attempt.
    pub abandoned_attempt_id: Option<String>,
    /// Snapshot of the per-attempt state cleared by
    /// [`Self::bump_attempt_generation`], consumed only by an
    /// immediately-following [`Self::rewind_attempt_generation`]. When a
    /// bump's JetStream publish fails, ``rewind`` restores the partial
    /// output the bump cleared so the request can fall back to the
    /// abandoned attempt's accumulated chunks/tool-calls/timing rather
    /// than surfacing an empty body. Overwritten on every bump; a
    /// successful new attempt (no rewind) simply never reads it.
    snapshot: Option<AttemptSnapshot>,
}

/// State saved by ``bump_attempt_generation`` so a failed-publish
/// ``rewind_attempt_generation`` can restore it. See
/// [`StreamCollector::snapshot`].
struct AttemptSnapshot {
    chunks: Vec<(u32, String)>,
    tool_calls_by_index: BTreeMap<u32, AggregatedToolCall>,
    logprobs: Vec<serde_json::Value>,
    first_chunk_at: Option<Instant>,
    last_chunk_at: Option<Instant>,
    last_output_at: Option<Instant>,
    output_event_count: u64,
    current_attempt_id: Option<String>,
    last_applied_seq: Option<u32>,
    execution_identity_sha256: Option<String>,
    execution_identity_consistent: bool,
}

#[derive(Debug, Clone)]
pub struct TerminalMeta {
    pub finish_reason: String,
    pub usage: Option<UsageBlock>,
    pub ttft_ms: Option<f64>,
    pub error: Option<ChunkError>,
    pub candidates: Vec<CandidateData>,
    pub executed_bundle_config_hash: Option<String>,
}

impl StreamCollector {
    pub fn new(sender: oneshot::Sender<StreamOutcome>, model: String, pool: String) -> Self {
        Self {
            current_attempt_id: None,
            last_applied_seq: None,
            chunks: Vec::new(),
            final_meta: None,
            sender: Some(sender),
            published_at: Instant::now(),
            first_chunk_at: None,
            last_chunk_at: None,
            last_output_at: None,
            output_event_count: 0,
            display_model: model.clone(),
            model,
            pool,
            activity: Arc::new(Notify::new()),
            attempt_generation: 0,
            encoded_payload: None,
            pool_fallback_subject: None,
            republished: false,
            chunk_tap: None,
            tool_calls_by_index: BTreeMap::new(),
            logprobs: Vec::new(),
            execution_identity_sha256: None,
            execution_identity_consistent: true,
            abandoned_attempt_id: None,
            snapshot: None,
        }
    }

    /// Clone the activity notifier so the HTTP handler can await on
    /// chunk arrivals without holding the DashMap entry lock.
    pub fn activity_handle(&self) -> Arc<Notify> {
        Arc::clone(&self.activity)
    }

    /// Install a broadcast tap on this collector so every non-stale
    /// chunk applied by :meth:`apply` is also fanned out to the
    /// returned receiver. Returns the receiver the SSE handler will
    /// consume.
    ///
    /// Idempotent: calling twice replaces the tap and drops the prior
    /// sender (existing subscribers see ``Closed`` on the next recv).
    /// The publisher installs the tap before publishing the work item
    /// so chunks cannot race past the subscriber.
    pub fn install_chunk_tap(&mut self) -> broadcast::Receiver<ChunkEnvelope> {
        let (tx, rx) = broadcast::channel(CHUNK_TAP_CAPACITY);
        self.chunk_tap = Some(tx);
        rx
    }

    /// Advance the attempt generation and clear all
    /// per-attempt state. The next chunk observed will re-latch
    /// against the new generation's ``attempt_id``. Returns the new
    /// generation number for logging.
    ///
    /// We discard the prior attempt's accumulated text (``chunks``)
    /// because the republish is producing a brand-new generation;
    /// concatenating partial output from the abandoned attempt with
    /// fresh chunks would surface a corrupt body. ``published_at``
    /// is preserved so the queue-latency metric still reflects total
    /// wall-clock from the original gateway publish.
    pub fn bump_attempt_generation(&mut self) -> u64 {
        self.attempt_generation = self.attempt_generation.saturating_add(1);
        // Snapshot the state we are about to clear so an immediately-
        // following ``rewind_attempt_generation`` (publish failed) can
        // restore the abandoned attempt's partial output instead of
        // surfacing an empty body. ``std::mem::take`` leaves the live
        // fields in their cleared state, so the "clear" below is folded
        // into the snapshot capture for the collections. The snapshot is
        // overwritten on every bump and consumed only by a rewind.
        self.snapshot = Some(AttemptSnapshot {
            chunks: std::mem::take(&mut self.chunks),
            tool_calls_by_index: std::mem::take(&mut self.tool_calls_by_index),
            logprobs: std::mem::take(&mut self.logprobs),
            first_chunk_at: self.first_chunk_at,
            last_chunk_at: self.last_chunk_at,
            last_output_at: self.last_output_at,
            output_event_count: self.output_event_count,
            current_attempt_id: self.current_attempt_id.clone(),
            last_applied_seq: self.last_applied_seq,
            execution_identity_sha256: self.execution_identity_sha256.clone(),
            execution_identity_consistent: self.execution_identity_consistent,
        });
        // Record the abandoned attempt id so the ``None`` latch arm of
        // ``apply`` refuses to re-latch on a stale leftover chunk from
        // the just-abandoned attempt — only a genuinely-new attempt may
        // latch. Take it directly so ``current_attempt_id`` is cleared
        // in the same move.
        self.abandoned_attempt_id = self.current_attempt_id.take();
        // Reset the per-attempt seq watermark: the fresh attempt
        // restarts its own ``seq`` sequence, so the next chunk must be
        // accepted regardless of the abandoned attempt's last seq.
        self.last_applied_seq = None;
        // Reset TTFT timing so it measures from the *successful*
        // attempt, not the abandoned one.
        self.first_chunk_at = None;
        self.last_output_at = None;
        self.output_event_count = 0;
        self.execution_identity_sha256 = None;
        self.execution_identity_consistent = true;
        // ``chunks`` / ``tool_calls_by_index`` were already emptied by
        // the ``std::mem::take`` above. Dropping the partial text from
        // the abandoned attempt prevents a mid-stream republish
        // (admission-control's ``worker_shutting_down`` NAK, etc.) from
        // concatenating "old worker's first half" + "new worker's full
        // output" into a malformed response. Today's republish triggers
        // all fire before any chunks arrive, so this is also a no-op on
        // the routing happy path.
        //
        // ``last_chunk_at`` gates the inter-chunk timeout; clearing
        // it forces that timer to wait for the *next* chunk before
        // arming, which is exactly what we want post-republish.
        self.last_chunk_at = None;
        self.attempt_generation
    }

    /// Fill the abandoned-attempt guard from an inbox NAK that caused a
    /// republish before any chunk latched an attempt id.
    pub fn record_abandoned_attempt_id(&mut self, attempt_id: &str) {
        if attempt_id.is_empty()
            || self.current_attempt_id.is_some()
            || self.abandoned_attempt_id.is_some()
        {
            return;
        }
        self.abandoned_attempt_id = Some(attempt_id.to_string());
    }

    /// Undo a [`Self::bump_attempt_generation`] that was followed by
    /// a publish failure. Used by `WorkPublisher::republish_to_pool` to
    /// keep the collector's `attempt_generation` consistent with what
    /// the worker side actually saw — without rewind, a republish whose
    /// JetStream publish failed would leave the counter one ahead, and
    /// the next NAK/timeout would short-circuit at the `republished`
    /// guard and hang the request until overall-timeout.
    pub fn rewind_attempt_generation(&mut self) {
        // `bump_*` uses `saturating_add(1)`; the inverse must be a
        // saturating sub so an unbalanced rewind cannot underflow.
        self.attempt_generation = self.attempt_generation.saturating_sub(1);
        // The bump that we are undoing was followed by a *failed*
        // publish, so the worker never saw the new generation. Restore
        // the per-attempt state the bump cleared so the abandoned
        // attempt's partial output (chunks / tool-calls / timing) is not
        // permanently lost — without this the request falls back to an
        // empty body even though the prior attempt had produced output.
        if let Some(snap) = self.snapshot.take() {
            self.chunks = snap.chunks;
            self.tool_calls_by_index = snap.tool_calls_by_index;
            self.logprobs = snap.logprobs;
            self.first_chunk_at = snap.first_chunk_at;
            self.last_chunk_at = snap.last_chunk_at;
            self.last_output_at = snap.last_output_at;
            self.output_event_count = snap.output_event_count;
            self.current_attempt_id = snap.current_attempt_id;
            self.last_applied_seq = snap.last_applied_seq;
            self.execution_identity_sha256 = snap.execution_identity_sha256;
            self.execution_identity_consistent = snap.execution_identity_consistent;
        } else {
            // Defensive: a rewind with no matching snapshot (should not
            // happen — every rewind follows a bump). Fall back to the
            // pre-fix clearing semantics so the collector stays in a
            // consistent, empty per-attempt state.
            self.current_attempt_id = None;
            self.last_applied_seq = None;
            self.first_chunk_at = None;
            self.last_chunk_at = None;
            self.last_output_at = None;
            self.output_event_count = 0;
            self.execution_identity_sha256 = None;
            self.execution_identity_consistent = true;
        }
        // The abandoned-id guard is meaningless once we have rewound
        // back onto the prior attempt: clear it so the restored
        // attempt's own chunks are not mistaken for stale leftovers.
        self.abandoned_attempt_id = None;
    }

    /// Apply a chunk to this collector. Returns ``true`` iff the chunk
    /// was the terminal one and the caller should complete the request.
    ///
    /// When a broadcast tap is installed (SSE requests) every non-stale
    /// chunk — including the terminal one — is also forwarded to
    /// the tap. The send is best-effort: ``broadcast::Sender::send``
    /// returns ``Err`` only when there are no subscribers, which is a
    /// no-op for the aggregating side of the request.
    pub fn apply(&mut self, chunk: ChunkEnvelope) -> ChunkApplied {
        // Wire-level invariants. ``is_chunk_envelope`` already peeked at
        // ``kind`` before dispatch, but a malicious / buggy worker could
        // still produce a ``kind != "chunk"`` envelope that deserializes
        // successfully — reject those here too instead of letting them
        // corrupt the aggregated outcome.
        if chunk.kind != "chunk" {
            telemetry::record_generation_event(
                GenerationEvent::Chunk,
                GenerationEventReason::InvalidKind,
                GenerationEventOutcome::Rejected,
            );
            return ChunkApplied::Stale;
        }
        if let Some(t) = chunk.ttft_ms {
            if !t.is_finite() {
                telemetry::record_generation_event(
                    GenerationEvent::Chunk,
                    GenerationEventReason::NonFiniteTtft,
                    GenerationEventOutcome::Rejected,
                );
                return ChunkApplied::Stale;
            }
        }
        if let Some(reason) = chunk.finish_reason.as_ref() {
            if !is_known_finish_reason(reason) {
                telemetry::record_generation_event(
                    GenerationEvent::Chunk,
                    GenerationEventReason::InvalidFinishReason,
                    GenerationEventOutcome::Rejected,
                );
                return ChunkApplied::Stale;
            }
        }

        // Latch attempt on first observed chunk; drop mismatches.
        match self.current_attempt_id.as_ref() {
            None => {
                // Post-bump, the latch is open. A leftover chunk from the
                // just-abandoned attempt must NOT re-latch it — if it did,
                // the genuinely-new attempt's chunks (incl. terminal)
                // would then be dropped as stale, hanging the request and
                // losing the new output. Drop it as stale and keep
                // waiting for a genuinely-new attempt id. uuid4 attempt
                // ids are never reused, so this match is unambiguous.
                if self.abandoned_attempt_id.as_deref() == Some(chunk.attempt_id.as_str()) {
                    telemetry::record_generation_event(
                        GenerationEvent::Chunk,
                        GenerationEventReason::StaleAttempt,
                        GenerationEventOutcome::Dropped,
                    );
                    return ChunkApplied::Stale;
                }
                self.current_attempt_id = Some(chunk.attempt_id.clone());
                // A genuinely-new attempt has latched: the abandoned-id
                // guard has done its job and must not reject this
                // attempt's later chunks (they reach the ``Some`` arm
                // from here on, but clear it for hygiene).
                self.abandoned_attempt_id = None;
            }
            Some(current) if current != &chunk.attempt_id => {
                telemetry::record_generation_event(
                    GenerationEvent::Chunk,
                    GenerationEventReason::StaleAttempt,
                    GenerationEventOutcome::Dropped,
                );
                return ChunkApplied::Stale;
            }
            _ => {}
        }

        // Per-attempt seq dedup. JetStream redelivers a message within
        // an attempt (ack lost / consumer re-pull) with the *same*
        // ``seq`` it carried the first time. Drop any chunk that does
        // not advance the watermark so a redelivery cannot double-count
        // text, re-fan-out an SSE event, or fire a duplicate terminal.
        // The check runs *after* the attempt-latch so a fresh attempt
        // (watermark cleared by ``bump_attempt_generation``) always
        // accepts its own first chunk regardless of the abandoned
        // attempt's seq.
        // The terminal chunk (``done == true``) is exempt from the
        // watermark drop: a worker that batches the final text and the
        // terminal marker can legitimately re-use a ``seq`` already seen
        // on a non-terminal chunk, and dropping the terminal as a
        // duplicate would hang the request forever (the caller only
        // completes on a ``Terminal``). Non-terminal chunks still dedupe
        // so a JetStream redelivery cannot double-count text. The
        // watermark is still advanced below so the build path sees the
        // terminal's seq.
        if !chunk.done {
            if let Some(last) = self.last_applied_seq {
                if chunk.seq <= last {
                    telemetry::record_generation_event(
                        GenerationEvent::Chunk,
                        GenerationEventReason::Duplicate,
                        GenerationEventOutcome::Dropped,
                    );
                    return ChunkApplied::Duplicate;
                }
                // H6: a gap in the per-attempt ``seq`` means a required
                // content chunk was dropped between worker and gateway.
                // The worker's own H6 fix guarantees ``seq`` is only
                // advanced after a successful enqueue, so a gap on the
                // wire is a genuine transport failure (not a worker-side
                // skip). The stream is no longer reconstructable —
                // fail the pending request with ``transport_failure``;
                // the caller (handle_chunk) drops the collector and
                // surfaces the error to the client.
                if chunk.seq > last + 1 {
                    telemetry::record_generation_event(
                        GenerationEvent::Chunk,
                        GenerationEventReason::SequenceGap,
                        GenerationEventOutcome::Rejected,
                    );
                    return ChunkApplied::SeqGap;
                }
            }
        }
        self.last_applied_seq = Some(chunk.seq);

        match chunk.execution_identity_sha256.as_ref() {
            Some(value) if is_lower_sha256(value) && self.execution_identity_consistent => {
                if let Some(current) = self.execution_identity_sha256.as_ref() {
                    if current != value {
                        self.execution_identity_consistent = false;
                        self.execution_identity_sha256 = None;
                    }
                } else {
                    self.execution_identity_sha256 = Some(value.clone());
                }
            }
            _ => {
                self.execution_identity_consistent = false;
                self.execution_identity_sha256 = None;
            }
        }

        let now = Instant::now();
        // Arm ``first_chunk_at`` on the first applied chunk that carries
        // ANY payload — text OR tool-call deltas. Tool-call deltas ride
        // in chunks with an empty ``text_delta``; gating only on text
        // left a tool-call-only stream with ``first_chunk_at == None``,
        // so the non-SSE driver kept the 30s first-chunk timeout armed
        // (it gates its inter-chunk timer on ``first_at.is_some()``)
        // throughout an active tool-call stream. Matches the SSE
        // driver's ``first_seen`` semantics.
        let has_output = !chunk.text_delta.is_empty()
            || chunk
                .tool_calls
                .as_ref()
                .is_some_and(|tool_calls| !tool_calls.is_empty());
        if has_output {
            if self.first_chunk_at.is_none() {
                self.first_chunk_at = Some(now);
            }
            self.last_output_at = Some(now);
            self.output_event_count = self.output_event_count.saturating_add(1);
        }
        self.last_chunk_at = Some(now);

        // Fan out to the SSE tap before mutating ``chunks`` so the
        // forwarded envelope reflects the wire-level chunk (including
        // an empty terminal ``text_delta``). The clone is per-chunk
        // and only happens when a tap is installed; non-streaming
        // requests pay nothing extra.
        if let Some(tap) = self.chunk_tap.as_ref() {
            let _ = tap.send(chunk.clone());
        }

        if !chunk.text_delta.is_empty() {
            self.chunks.push((chunk.seq, chunk.text_delta));
        }

        // Absorb any tool-call deltas before mutating the activity
        // notifier so the non-streaming aggregator (which reads
        // ``tool_calls_by_index`` on terminal) always sees the
        // complete picture by the time it builds the outcome.
        if let Some(tcs) = chunk.tool_calls.as_ref() {
            for tc in tcs {
                let entry = self.tool_calls_by_index.entry(tc.index).or_insert_with(|| {
                    AggregatedToolCall {
                        index: tc.index,
                        id: String::new(),
                        kind: tc.kind.clone(),
                        name: String::new(),
                        arguments: String::new(),
                    }
                });
                if entry.id.is_empty() {
                    if let Some(id) = tc.id.as_ref() {
                        entry.id = id.clone();
                    }
                }
                if let Some(func) = tc.function.as_ref() {
                    if entry.name.is_empty() {
                        if let Some(name) = func.name.as_ref() {
                            entry.name = name.clone();
                        }
                    }
                    entry.arguments.push_str(&func.arguments);
                }
            }
        }

        // Accumulate per-token logprobs. The worker already produced the
        // OpenAI ``ChatCompletionTokenLogprob`` shape, so the gateway just
        // concatenates them in observed order. Mirrors ``chunks``: applied
        // for every non-stale chunk and snapshotted/cleared by the
        // attempt-generation machinery. The SSE tap above already
        // forwarded this chunk's slice for the streaming surface.
        if let Some(lps) = chunk.logprobs {
            self.logprobs.extend(lps);
        }

        // Pulse the activity notifier so any pending inter-chunk timer
        // on the HTTP handler wakes up.
        self.activity.notify_one();

        if chunk.done {
            self.final_meta = Some(TerminalMeta {
                finish_reason: chunk.finish_reason.unwrap_or_else(|| "stop".to_string()),
                usage: chunk.usage,
                ttft_ms: chunk.ttft_ms,
                error: chunk.error,
                candidates: chunk.candidates,
                executed_bundle_config_hash: chunk.executed_bundle_config_hash,
            });
            return ChunkApplied::Terminal;
        }
        ChunkApplied::Delta
    }

    /// Build the aggregated outcome from accumulated chunks. Returns
    /// ``None`` if no terminal chunk has been observed (caller should
    /// only call this after ``ChunkApplied::Terminal``).
    pub fn build_outcome(&self) -> Option<StreamOutcome> {
        let meta = self.final_meta.as_ref()?;

        // Concatenate deltas in observed order. The streaming surface trusts the
        // worker's send order on a single core-NATS subject; out-of-
        // order delivery on the same subject is not produced by the
        // worker today, so we don't re-sort by ``seq``.
        let mut text = String::with_capacity(self.chunks.iter().map(|c| c.1.len()).sum());
        for (_, delta) in &self.chunks {
            text.push_str(delta);
        }

        let ttft_ms = meta.ttft_ms.or_else(|| {
            self.first_chunk_at
                .map(|t| (t - self.published_at).as_secs_f64() * 1000.0)
        });

        let tpot_ms = match (self.first_chunk_at, self.last_output_at) {
            (Some(first), Some(last)) if last > first => {
                let completion = meta
                    .usage
                    .as_ref()
                    .and_then(|usage| {
                        (usage.completion_tokens > 0).then_some(u64::from(usage.completion_tokens))
                    })
                    .unwrap_or(self.output_event_count);
                (completion > 0)
                    .then(|| ((last - first).as_secs_f64() * 1000.0) / completion as f64)
            }
            _ => None,
        };

        let tool_calls = if self.tool_calls_by_index.is_empty() {
            None
        } else {
            Some(self.tool_calls_by_index.values().cloned().collect())
        };

        let logprobs = if self.logprobs.is_empty() {
            None
        } else {
            Some(self.logprobs.clone())
        };

        Some(StreamOutcome {
            text,
            finish_reason: meta.finish_reason.clone(),
            usage: meta.usage.clone(),
            attempt_id: self.current_attempt_id.clone().unwrap_or_default(),
            ttft_ms,
            tpot_ms,
            error: meta.error.clone(),
            tool_calls,
            logprobs,
            candidates: meta.candidates.clone(),
            executed_bundle_config_hash: meta.executed_bundle_config_hash.clone(),
            execution_identity_sha256: self
                .execution_identity_consistent
                .then(|| self.execution_identity_sha256.clone())
                .flatten(),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChunkApplied {
    /// Non-terminal delta chunk applied.
    Delta,
    /// Terminal chunk applied; caller should fire the sender.
    Terminal,
    /// Chunk dropped due to stale ``attempt_id``.
    Stale,
    /// Chunk dropped as a within-attempt redelivery (``seq`` did not
    /// advance the per-attempt watermark). Treated identically to
    /// ``Stale`` by callers (non-terminal, no side effects), but kept
    /// as a distinct variant so tests and any future telemetry can
    /// tell a redelivery apart from a cross-attempt stale chunk.
    Duplicate,
    /// Per-attempt ``seq`` skipped one or more values
    /// (``chunk.seq > last_applied_seq + 1``) — a required content
    /// chunk was dropped on the worker → gateway path and the stream
    /// is no longer reconstructable. The caller must fail the pending
    /// request with a ``transport_failure`` error (mirrors the worker's
    /// own no-silent-drop guarantee per H6). Distinct from
    /// ``Duplicate`` so callers can dispatch on it.
    SeqGap,
}

/// Terminal NAK envelope.
///
/// Emitted by the worker on the inbox reply subject when it cannot
/// service a request (model not loaded, KV budget exhausted in slice
/// 06, shutting down). The gateway treats this as a non-stream
/// terminal: it republishes the original work item to the pool
/// subject so another worker can pick it up. Unlike a JetStream
/// `nak()` (which causes redelivery on the *same* subject), the
/// inbox-side `kind:"nak"` envelope says "the gateway should
/// reroute," not "redeliver to me."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NakEnvelope {
    pub kind: String, // "nak"
    pub request_id: String,
    #[serde(default)]
    pub attempt_id: String,
    /// One of: `kv_budget`, `model_not_loaded`, `worker_shutting_down`.
    /// Recorded as a metric label; the routing rollout only emits
    /// `model_not_loaded` from the stub worker path.
    #[serde(default)]
    pub reason: String,
}

// NOTE: the byte-level `is_chunk_envelope` / `is_nak_envelope` peek
// helpers were removed when `WorkPublisher::handle_inbox` was changed to
// decode the msgpack payload exactly once into an `rmpv::Value` and
// dispatch on `publisher::envelope_kind(&value)`. The previous helpers
// re-decoded the raw slice on every call, which is the very
// double/triple-decode the single-decode refactor eliminated.

#[cfg(test)]
mod tests {
    use super::*;

    fn _make_chunk(attempt: &str, seq: u32, text: &str, done: bool) -> ChunkEnvelope {
        ChunkEnvelope {
            kind: "chunk".to_string(),
            request_id: "req-1".to_string(),
            attempt_id: attempt.to_string(),
            seq,
            text_delta: text.to_string(),
            done,
            is_first: false,
            finish_reason: if done { Some("stop".to_string()) } else { None },
            usage: if done {
                Some(UsageBlock {
                    prompt_tokens: 5,
                    completion_tokens: 3,
                    total_tokens: 8,
                })
            } else {
                None
            },
            ttft_ms: None,
            error: None,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            choice_index: 0,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
        }
    }

    #[test]
    fn test_terminal_model_loading_error_chunk_builds_error_outcome() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(
            tx,
            "Qwen/Qwen3-4B-Instruct-2507".to_string(),
            "default".to_string(),
        );
        let mut term = _make_chunk("req-load.0:model-loading", 0, "", true);
        term.finish_reason = Some("error".to_string());
        term.usage = None;
        term.error = Some(ChunkError {
            code: "MODEL_LOADING".to_string(),
            message: "Model 'Qwen/Qwen3-4B-Instruct-2507' is loading; retry later.".to_string(),
        });

        assert_eq!(c.apply(term), ChunkApplied::Terminal);
        let outcome = c.build_outcome().expect("terminal");
        let err = outcome.error.expect("terminal error");
        assert_eq!(outcome.text, "");
        assert_eq!(outcome.finish_reason, "error");
        assert_eq!(outcome.attempt_id, "req-load.0:model-loading");
        assert_eq!(err.code, "MODEL_LOADING");
        assert!(err.message.contains("retry later"));
    }

    /// Multi-candidate (`n>1`): the terminal chunk carries a `candidates`
    /// array which the collector passes through verbatim onto the
    /// `StreamOutcome` for the body builder to expand into `choices[]`.
    #[test]
    fn test_collector_passes_through_terminal_candidates() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut term = _make_chunk("att-A", 0, "", true);
        term.candidates = vec![
            CandidateData {
                text: "one".to_string(),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                tool_calls: None,
            },
            CandidateData {
                text: "two".to_string(),
                finish_reason: Some("length".to_string()),
                logprobs: None,
                tool_calls: None,
            },
        ];
        assert_eq!(c.apply(term), ChunkApplied::Terminal);
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.candidates.len(), 2);
        assert_eq!(outcome.candidates[0].text, "one");
        assert_eq!(
            outcome.candidates[1].finish_reason.as_deref(),
            Some("length")
        );
    }

    /// Per-chunk logprobs accumulate across the stream and surface as the
    /// aggregated ``StreamOutcome.logprobs`` (the non-streaming body's
    /// ``choices[0].logprobs.content``).
    #[test]
    fn test_collector_aggregates_logprobs_across_chunks() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut d0 = _make_chunk("att-A", 0, "Hi", false);
        d0.logprobs = Some(vec![serde_json::json!({
            "token": "Hi", "logprob": -0.1, "bytes": [72, 105], "top_logprobs": []
        })]);
        let mut d1 = _make_chunk("att-A", 1, "!", false);
        d1.logprobs = Some(vec![serde_json::json!({
            "token": "!", "logprob": -0.2, "bytes": [33], "top_logprobs": []
        })]);
        assert_eq!(c.apply(d0), ChunkApplied::Delta);
        assert_eq!(c.apply(d1), ChunkApplied::Delta);
        assert_eq!(
            c.apply(_make_chunk("att-A", 2, "", true)),
            ChunkApplied::Terminal
        );
        let lps = c
            .build_outcome()
            .expect("terminal")
            .logprobs
            .expect("aggregated");
        assert_eq!(lps.len(), 2);
        assert_eq!(lps[0]["token"], "Hi");
        assert_eq!(lps[1]["token"], "!");
    }

    /// No chunk carried logprobs → aggregated outcome is ``None``.
    #[test]
    fn test_collector_logprobs_none_when_absent() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        c.apply(_make_chunk("att-A", 0, "Hi", false));
        c.apply(_make_chunk("att-A", 1, "", true));
        assert!(c.build_outcome().expect("terminal").logprobs.is_none());
    }

    #[test]
    fn test_collector_requires_execution_identity_on_every_chunk() {
        let identity = "a".repeat(64);
        let (tx, _rx) = oneshot::channel();
        let mut consistent = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut delta = _make_chunk("att-A", 0, "Hi", false);
        delta.execution_identity_sha256 = Some(identity.clone());
        let mut terminal = _make_chunk("att-A", 1, "", true);
        terminal.execution_identity_sha256 = Some(identity.clone());
        assert_eq!(consistent.apply(delta), ChunkApplied::Delta);
        assert_eq!(consistent.apply(terminal), ChunkApplied::Terminal);
        assert_eq!(
            consistent
                .build_outcome()
                .expect("terminal")
                .execution_identity_sha256
                .as_deref(),
            Some(identity.as_str())
        );

        let (tx, _rx) = oneshot::channel();
        let mut mismatching = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut delta = _make_chunk("att-B", 0, "Hi", false);
        delta.execution_identity_sha256 = Some(identity);
        let mut terminal = _make_chunk("att-B", 1, "", true);
        terminal.execution_identity_sha256 = Some("b".repeat(64));
        assert_eq!(mismatching.apply(delta), ChunkApplied::Delta);
        assert_eq!(mismatching.apply(terminal), ChunkApplied::Terminal);
        assert!(mismatching
            .build_outcome()
            .expect("terminal")
            .execution_identity_sha256
            .is_none());

        let (tx, _rx) = oneshot::channel();
        let mut missing = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut delta = _make_chunk("att-C", 0, "Hi", false);
        delta.execution_identity_sha256 = Some("c".repeat(64));
        assert_eq!(missing.apply(delta), ChunkApplied::Delta);
        assert_eq!(
            missing.apply(_make_chunk("att-C", 1, "", true)),
            ChunkApplied::Terminal
        );
        assert!(missing
            .build_outcome()
            .expect("terminal")
            .execution_identity_sha256
            .is_none());
    }

    /// ``ChunkEnvelope`` round-trips a ``logprobs`` field over the wire
    /// (msgpack), so the gateway decodes what the worker encodes.
    #[test]
    fn test_chunk_envelope_round_trips_logprobs() {
        let mut d = _make_chunk("att-A", 0, "Hi", false);
        d.logprobs = Some(vec![serde_json::json!({
            "token": "Hi", "logprob": -0.1, "bytes": [72, 105], "top_logprobs": []
        })]);
        let bytes = rmp_serde::to_vec_named(&d).unwrap();
        let back: ChunkEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.logprobs.expect("logprobs")[0]["token"], "Hi");
    }

    #[test]
    fn test_collector_latches_attempt_and_concatenates_text() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());

        assert_eq!(
            collector.apply(_make_chunk("att-A", 0, "Hello", false)),
            ChunkApplied::Delta
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 1, " world", false)),
            ChunkApplied::Delta
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 2, "!", true)),
            ChunkApplied::Terminal
        );

        let outcome = collector.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "Hello world!");
        assert_eq!(outcome.finish_reason, "stop");
        assert_eq!(outcome.attempt_id, "att-A");
        assert_eq!(outcome.usage.as_ref().unwrap().completion_tokens, 3);
    }

    #[test]
    fn test_tpot_uses_authoritative_tokens_then_observed_output_fallback() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        collector.apply(_make_chunk("att-A", 0, "one", false));
        collector.apply(_make_chunk("att-A", 1, "two", false));
        collector.apply(_make_chunk("att-A", 2, "", true));

        let first = Instant::now();
        collector.first_chunk_at = Some(first);
        collector.last_output_at = Some(first + std::time::Duration::from_millis(400));
        collector.output_event_count = 2;
        collector.final_meta.as_mut().expect("terminal").usage = Some(UsageBlock {
            prompt_tokens: 1,
            completion_tokens: 4,
            total_tokens: 5,
        });
        let authoritative = collector.build_outcome().expect("terminal");
        assert!((authoritative.tpot_ms.expect("TPOT") - 100.0).abs() < f64::EPSILON);

        collector.final_meta.as_mut().expect("terminal").usage = None;
        let fallback = collector.build_outcome().expect("terminal");
        assert!((fallback.tpot_ms.expect("TPOT") - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_output_event_has_no_measurable_tpot() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        collector.apply(_make_chunk("att-A", 0, "only", false));
        collector.apply(_make_chunk("att-A", 1, "", true));

        assert_eq!(collector.output_event_count, 1);
        assert_eq!(collector.first_chunk_at, collector.last_output_at);
        assert!(collector
            .build_outcome()
            .expect("terminal")
            .tpot_ms
            .is_none());
    }

    #[test]
    fn test_collector_drops_stale_attempt_chunks() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());

        assert_eq!(
            collector.apply(_make_chunk("att-A", 0, "first", false)),
            ChunkApplied::Delta
        );
        // Different attempt_id — should be dropped silently.
        assert_eq!(
            collector.apply(_make_chunk("att-B", 0, "ignored", false)),
            ChunkApplied::Stale
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 1, "", true)),
            ChunkApplied::Terminal
        );

        let outcome = collector.build_outcome().expect("terminal");
        // The dropped chunk's text is not present.
        assert_eq!(outcome.text, "first");
    }

    // -- NAK envelope --

    #[test]
    fn test_nak_envelope_roundtrip() {
        let nak = NakEnvelope {
            kind: "nak".to_string(),
            request_id: "req-1".to_string(),
            attempt_id: "att-A".to_string(),
            reason: "kv_budget".to_string(),
        };
        let bytes = rmp_serde::to_vec_named(&nak).unwrap();
        let decoded: NakEnvelope = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.reason, "kv_budget");
    }

    /// The optional broadcast tap is non-invasive: when not installed,
    /// `apply()` behaves identically to the pre-SSE path. Regression
    /// guard against a future refactor that always allocates the
    /// channel.
    #[test]
    fn test_collector_without_tap_no_op() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());
        assert!(collector.chunk_tap.is_none());
        // apply works exactly as before — no panic, no fan-out side
        // effect (the broadcast send is gated by the `Some` arm).
        collector.apply(_make_chunk("att-A", 0, "Hello", false));
        assert_eq!(collector.chunks.len(), 1);
        collector.apply(_make_chunk("att-A", 1, "", true));
        let outcome = collector.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "Hello");
    }

    /// With the tap installed, every non-stale chunk reaches the
    /// subscriber. Stale chunks (mismatched attempt_id) do NOT, since
    /// the drop logic runs *before* the fan-out.
    #[tokio::test]
    async fn test_collector_tap_skips_stale_chunks() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());
        let mut tap = collector.install_chunk_tap();

        collector.apply(_make_chunk("att-A", 0, "first", false));
        let got = tap.recv().await.unwrap();
        assert_eq!(got.text_delta, "first");

        // Stale (different attempt_id) — dropped before tap.
        let stale = _make_chunk("att-B", 0, "ignored", false);
        assert_eq!(collector.apply(stale), ChunkApplied::Stale);
        assert!(matches!(
            tap.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty)
        ));

        // Another valid chunk lands as expected.
        collector.apply(_make_chunk("att-A", 1, "next", false));
        let got = tap.recv().await.unwrap();
        assert_eq!(got.text_delta, "next");
    }

    /// JetStream redelivery within an attempt re-sends a chunk with the
    /// *same* ``seq``. The collector must drop it (``Duplicate``) so the
    /// text is not double-counted and no duplicate SSE event fans out.
    #[test]
    fn test_collector_drops_duplicate_seq_redelivery() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());

        assert_eq!(
            collector.apply(_make_chunk("att-A", 0, "Hello", false)),
            ChunkApplied::Delta
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 1, " world", false)),
            ChunkApplied::Delta
        );
        // Redelivery of seq 1 (and seq 0) must be dropped, not appended.
        assert_eq!(
            collector.apply(_make_chunk("att-A", 1, " world", false)),
            ChunkApplied::Duplicate
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 0, "Hello", false)),
            ChunkApplied::Duplicate
        );
        assert_eq!(
            collector.apply(_make_chunk("att-A", 2, "!", true)),
            ChunkApplied::Terminal
        );

        let outcome = collector.build_outcome().expect("terminal");
        // Only one copy of each delta survives.
        assert_eq!(outcome.text, "Hello world!");
    }

    /// H6: a per-attempt seq gap (chunk.seq > last + 1) means a required
    /// content chunk was lost on the worker → gateway transport. The
    /// worker's H6 fix guarantees ``seq`` is only advanced after a
    /// successful enqueue, so any gap on the wire is genuine transport
    /// loss. The collector must reject the gap with ``SeqGap`` so the
    /// publisher can fail the pending stream with ``transport_failure``
    /// rather than silently producing a shortened completion.
    #[test]
    fn test_streaming_gap_rejected_as_stream_error() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());

        assert_eq!(
            collector.apply(_make_chunk("att-A", 1, "first", false)),
            ChunkApplied::Delta
        );
        // seq 2 is skipped; seq 3 must be rejected as a gap, not silently
        // accepted as a shortened stream.
        assert_eq!(
            collector.apply(_make_chunk("att-A", 3, "skipped-ahead", false)),
            ChunkApplied::SeqGap
        );
        // Watermark must not advance on a rejected gap chunk: a
        // subsequent seq 2 (e.g. an out-of-order JetStream redelivery,
        // hypothetical) would still be classed correctly relative to
        // the unchanged watermark. We assert the watermark by sending
        // the in-order seq 2 next and checking it is accepted as Delta.
        assert_eq!(
            collector.apply(_make_chunk("att-A", 2, "in-order", false)),
            ChunkApplied::Delta
        );
    }

    /// H6: the terminal chunk is exempt from the gap check by design —
    /// it follows the same exemption as the stale-seq check, since the
    /// worker is allowed to re-use a ``seq`` already seen on a non-terminal
    /// chunk for the terminal marker. (Dropping the terminal as a gap
    /// would hang the request forever waiting for a Terminal that never
    /// arrives.) This test pins that behaviour so a future refactor
    /// does not accidentally extend the gap check to terminals.
    #[test]
    fn test_streaming_gap_check_does_not_apply_to_terminal() {
        let (tx, _rx) = oneshot::channel();
        let mut collector =
            StreamCollector::new(tx, "test/model".to_string(), "default".to_string());

        assert_eq!(
            collector.apply(_make_chunk("att-A", 0, "hi", false)),
            ChunkApplied::Delta
        );
        // A terminal that "skips ahead" (seq 5 after seq 0) is still
        // accepted as the request-completing chunk.
        assert_eq!(
            collector.apply(_make_chunk("att-A", 5, "", true)),
            ChunkApplied::Terminal
        );
    }

    /// The seq watermark must reset across a pool-republish (attempt
    /// generation bump) so the fresh attempt's seq sequence (which
    /// restarts from 0) is accepted rather than dropped as a duplicate.
    #[test]
    fn test_seq_dedup_resets_on_attempt_bump() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        collector.apply(_make_chunk("att-A", 0, "abandoned", false));
        collector.apply(_make_chunk("att-A", 1, "-text", false));
        assert_eq!(collector.last_applied_seq, Some(1));

        collector.bump_attempt_generation();
        assert_eq!(collector.last_applied_seq, None);

        // Fresh attempt restarts at seq 0 — must be accepted.
        assert_eq!(
            collector.apply(_make_chunk("att-B", 0, "fresh", false)),
            ChunkApplied::Delta
        );
        assert_eq!(
            collector.apply(_make_chunk("att-B", 1, "-out", true)),
            ChunkApplied::Terminal
        );
        let outcome = collector.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "fresh-out");
    }

    #[test]
    fn test_bump_attempt_generation_clears_per_attempt_state() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let identity = "a".repeat(64);
        // First attempt latches and accumulates partial text.
        let mut first = _make_chunk("att-A", 0, "abandoned-", false);
        first.execution_identity_sha256 = Some(identity.clone());
        let mut second = _make_chunk("att-A", 1, "text", false);
        second.execution_identity_sha256 = Some(identity.clone());
        collector.apply(first);
        collector.apply(second);
        assert_eq!(collector.current_attempt_id.as_deref(), Some("att-A"));
        assert_eq!(collector.chunks.len(), 2);
        assert!(collector.first_chunk_at.is_some());
        assert!(collector.last_chunk_at.is_some());
        assert_eq!(
            collector.execution_identity_sha256.as_deref(),
            Some(identity.as_str())
        );

        // Bump generation — routing-review Mi1: chunks/last_chunk_at
        // must clear so a mid-stream republish doesn't surface a
        // corrupt concatenated body.
        let gen = collector.bump_attempt_generation();
        assert_eq!(gen, 1);
        assert!(collector.current_attempt_id.is_none());
        assert!(collector.first_chunk_at.is_none());
        assert!(collector.last_chunk_at.is_none());
        assert!(collector.chunks.is_empty());
        assert!(collector.execution_identity_sha256.is_none());
        assert!(collector.execution_identity_consistent);

        // Fresh attempt re-latches cleanly and a terminal chunk
        // builds an outcome containing only the new text.
        collector.apply(_make_chunk("att-B", 0, "fresh-", false));
        collector.apply(_make_chunk("att-B", 1, "output", true));
        let outcome = collector.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "fresh-output");
        assert_eq!(outcome.attempt_id, "att-B");
    }

    /// Build a tool-call delta chunk: an empty ``text_delta`` plus one
    /// populated ``tool_calls`` entry. Mirrors the worker's announcement
    /// delta (``id`` + ``function.name``).
    fn _make_tool_call_chunk(attempt: &str, seq: u32, done: bool) -> ChunkEnvelope {
        let mut chunk = _make_chunk(attempt, seq, "", done);
        chunk.tool_calls = Some(vec![ToolCallDeltaWire {
            index: 0,
            id: Some("call-1".to_string()),
            kind: "function".to_string(),
            function: Some(ToolCallFunctionWire {
                name: Some("get_weather".to_string()),
                arguments: "{\"city\":".to_string(),
            }),
        }]);
        chunk
    }

    /// BUG A regression: after a bump opens the latch, a stale leftover
    /// chunk from the just-abandoned attempt must NOT re-latch it. If it
    /// did, the genuinely-new attempt's chunks (incl. terminal) would be
    /// dropped as stale and the request would hang with output lost.
    #[test]
    fn test_stale_chunk_does_not_relatch_abandoned_attempt_after_bump() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        c.apply(_make_chunk("att-A", 0, "old-", false));
        c.bump_attempt_generation();
        // Leftover chunk from the abandoned attempt — rejected, latch
        // stays open.
        assert_eq!(
            c.apply(_make_chunk("att-A", 1, "STALE", false)),
            ChunkApplied::Stale
        );
        // The genuinely-new attempt wins and its terminal completes.
        let applied = c.apply(_make_chunk("att-B", 0, "fresh", true));
        assert_eq!(applied, ChunkApplied::Terminal);
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "fresh");
        assert_eq!(outcome.attempt_id, "att-B");
    }

    #[test]
    fn test_nak_recorded_abandoned_attempt_does_not_relatch_before_successor() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        c.bump_attempt_generation();
        c.record_abandoned_attempt_id("att-A");

        assert_eq!(
            c.apply(_make_chunk("att-A", 0, "STALE", false)),
            ChunkApplied::Stale
        );
        assert!(c.current_attempt_id.is_none());

        let applied = c.apply(_make_chunk("att-B", 0, "fresh", true));
        assert_eq!(applied, ChunkApplied::Terminal);
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "fresh");
        assert_eq!(outcome.attempt_id, "att-B");
    }

    /// BUG B regression: a ``bump`` followed by a ``rewind`` (publish
    /// failed) must restore the accumulated chunks the bump cleared,
    /// rather than permanently dropping the abandoned attempt's partial
    /// output.
    #[test]
    fn test_rewind_restores_chunks_dropped_by_bump() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let identity = "a".repeat(64);
        let mut partial = _make_chunk("att-A", 0, "partial", false);
        partial.execution_identity_sha256 = Some(identity.clone());
        c.apply(partial);
        c.bump_attempt_generation();
        c.rewind_attempt_generation();
        assert_eq!(c.chunks.len(), 1); // restored
                                       // Full restore: tool-calls / timing / latch / watermark too.
        assert_eq!(c.current_attempt_id.as_deref(), Some("att-A"));
        assert_eq!(c.last_applied_seq, Some(0));
        assert!(c.first_chunk_at.is_some());
        assert!(c.last_chunk_at.is_some());
        assert_eq!(
            c.execution_identity_sha256.as_deref(),
            Some(identity.as_str())
        );
        assert!(c.execution_identity_consistent);
    }

    #[test]
    fn test_rewind_clears_nak_recorded_abandoned_attempt() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        c.bump_attempt_generation();
        c.record_abandoned_attempt_id("att-A");

        c.rewind_attempt_generation();

        assert!(c.abandoned_attempt_id.is_none());
        assert_eq!(
            c.apply(_make_chunk("att-A", 0, "restored", true)),
            ChunkApplied::Terminal
        );
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "restored");
        assert_eq!(outcome.attempt_id, "att-A");
    }

    /// BUG B regression: a successful new attempt (bump with NO rewind)
    /// must not resurrect the abandoned attempt's chunks — the snapshot
    /// is consumed only by a rewind.
    #[test]
    fn test_bump_without_rewind_does_not_restore_snapshot() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        c.apply(_make_chunk("att-A", 0, "abandoned", false));
        c.bump_attempt_generation();
        // Fresh attempt latches and completes; only its text survives.
        c.apply(_make_chunk("att-B", 0, "fresh", true));
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "fresh");
        assert_eq!(outcome.attempt_id, "att-B");
    }

    /// BUG C regression: a chunk carrying an empty ``text_delta`` but a
    /// populated ``tool_calls`` vec must arm ``first_chunk_at`` (the
    /// non-SSE driver gates its inter-chunk timer on it; without this a
    /// tool-call-only stream keeps the first-chunk timeout armed).
    #[test]
    fn test_tool_call_only_chunk_arms_first_chunk_at() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        assert!(c.first_chunk_at.is_none());
        let applied = c.apply(_make_tool_call_chunk("att-A", 0, false));
        assert_eq!(applied, ChunkApplied::Delta);
        assert!(
            c.first_chunk_at.is_some(),
            "tool-call-only chunk must arm first_chunk_at"
        );
    }

    /// BUG E regression: a terminal chunk re-using a ``seq`` already seen
    /// on a non-terminal chunk must NOT be dropped as a duplicate — the
    /// caller only completes on ``Terminal``, so dropping it hangs the
    /// request.
    #[test]
    fn test_terminal_chunk_reusing_seq_not_dropped_as_duplicate() {
        let (tx, _rx) = oneshot::channel();
        let mut c = StreamCollector::new(tx, "m".to_string(), "p".to_string());
        assert_eq!(
            c.apply(_make_chunk("att-A", 0, "hi", false)),
            ChunkApplied::Delta
        );
        // Terminal re-uses seq 0 — must apply, not drop.
        assert_eq!(
            c.apply(_make_chunk("att-A", 0, "", true)),
            ChunkApplied::Terminal
        );
        let outcome = c.build_outcome().expect("terminal");
        assert_eq!(outcome.text, "hi");
    }
}
