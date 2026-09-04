use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use async_nats::jetstream;
use dashmap::{DashMap, DashSet};
use futures_util::stream::FuturesUnordered;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{broadcast, oneshot};
use tracing::{debug, info, warn};
use utoipa::ToSchema;

use rmp::decode::read_str_from_slice;
use rmp::Marker;

use super::dispatch::{DispatchDurability, PendingDispatchKind};
use super::identity::canonical_work_item_id;
use super::keyed_worker_pool::{KeyedWorkerPool, ShardKeyed};
use super::lane_admission::{LaneAdmissionControl, LaneAdmissionOutcome, LaneKey, LaneReservation};
use super::payload_store::PayloadStore;
use super::stream_durability;
use super::streaming::{
    ChunkApplied, ChunkEnvelope, ChunkError, NakEnvelope, StreamCollector, StreamOutcome,
    StreamOutcomeOrigin,
};
use crate::endpoint::InferenceEndpoint;
use crate::observability::metrics::{
    self as telemetry, QueueEvent, QueueEventOutcome, QueuePublishObservation, QueuePublishOutcome,
};

const PAYLOAD_OFFLOAD_THRESHOLD: usize = 1_024 * 1_024; // 1 MB
/// Public queue request contract. This is deliberately much larger than the
/// worker's internal coalescing window while bounding per-request result/ACK
/// state under the gateway's 16 MiB body limit.
pub(crate) const MAX_QUEUE_REQUEST_ITEMS: usize = 4_096;
/// Bound simultaneous JetStream sends without serializing large valid batches.
const MAX_CONCURRENT_INITIAL_PUBLISHES: usize = 64;
const PUBLISH_ACK_COMPLETION_TIMEOUT: Duration = Duration::from_secs(6);

/// Metric label for the inbox handler pool.
const INBOX_POOL_NAME: &str = "inbox";
/// Ordering domains for inbox work. A shard is a tokio task multiplexed onto
/// the existing runtime workers, not a thread, so this is a concurrency width
/// rather than a core count: wide enough that unrelated requests rarely share
/// a shard, narrow enough to keep worst-case buffered bytes small.
const INBOX_POOL_SHARDS: usize = 8;
/// Queue depth per inbox shard. Deliberately shallow. A queued item is a fully
/// decoded envelope whose size is bounded only by the NATS max payload, so
/// `SHARDS * DEPTH` is what bounds the transient duplicate residency of result
/// bytes ahead of the per-request reservation. With cleanup moved off them,
/// the handlers are DashMap work plus a channel send, so a shallow queue is
/// already enough to decouple decode from a momentarily slow handler.
const INBOX_POOL_DEPTH: usize = 16;
/// Metric label for the object-store cleanup pool.
const CLEANUP_POOL_NAME: &str = "payload_cleanup";
const CLEANUP_POOL_SHARDS: usize = 4;
/// A queued cleanup item is one request id, so this queue can be deep without
/// costing memory. It is still bounded; overflow falls back to the periodic
/// `cleanup_expired` reconcile.
const CLEANUP_POOL_DEPTH: usize = 256;

const RESULT_CHUNK_KIND: &str = "result_chunk_v1";
const MAX_RESULT_CHUNK_ITEM_BYTES: usize = 16 * 1_024 * 1_024;
const MAX_RESULT_CHUNKS_PER_ITEM: u32 = 64;
/// The ceiling on result-transfer bytes ONE request may have reserved.
///
/// `pub(crate)` because it is also the anchor for
/// `handlers::proxy::MAX_COMPAT_RESPONSE_BODY` (#2617): the OpenAI-compatible
/// wrappers buffer an inner native reply whole, and they should bound that
/// buffer at the number this crate already sizes single-request result state
/// against rather than at a second, independently chosen one.
pub(crate) const MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST: usize = 64 * 1_024 * 1_024;
/// One eighth of the gateway's 2 GiB container limit. This leaves 1.75 GiB
/// for normal request/response buffers, JSON expansion, NATS, and allocator
/// headroom even when result chunk traffic is at its reservation ceiling.
const MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL: usize = 256 * 1_024 * 1_024;
/// Reserve for partial fragments, the contiguous reassembly/decode copy, and
/// the decoded `WorkResult` retained until every item in the request arrives.
const RESULT_CHUNK_COPY_RESERVATION_MULTIPLIER: usize = 3;
/// Covers the chunk vector, strings, digest, and allocator metadata, including
/// adversarial tiny transfers whose payload alone would not cover bookkeeping.
const RESULT_CHUNK_TRANSFER_OVERHEAD_BYTES: usize = 4 * 1_024;
const MAX_RETIRED_RESULT_CHUNK_DIGESTS: usize = 8;
const MAX_RETIRED_RESULT_CHUNK_LAYOUTS: usize = 8;

// H9 — first-chunk-fallback rate limit defaults.
//
// The gateway republishes generation work to the pool subject when the
// direct-dispatched worker doesn't send a first chunk within the
// first-chunk window. Without bounds an in-cluster outage (cold workers,
// model-load storm) trips the fallback for every request in flight,
// doubling the JetStream pressure on the pool stream and amplifying the
// load that already caused the timeouts. The token bucket caps the
// fallback rate per (model, pool); requests beyond the burst are refused
// with a 504 so the client can decide whether to retry.
//
// 5 / sec sustained with a burst of 10 covers a healthy cluster's
// occasional cold-start fallbacks without rate-limiting them, while
// preventing a runaway storm from consuming the pool stream's pending
// budget. Values are tunable per deployment via the constructor in a
// future iteration; the constants here are the "safe default" the audit
// recommended.
const FALLBACK_RATE_PER_SEC_DEFAULT: f64 = 5.0;
const FALLBACK_BURST_DEFAULT: f64 = 10.0;

/// Simple monotonic-time token bucket. Not thread-safe on its own; the
/// caller wraps a single instance in the appropriate concurrency
/// primitive (the publisher uses ``DashMap<key, Mutex<TokenBucket>>``).
///
/// ``try_take`` returns true and decrements the available tokens by one
/// when at least one whole token is available; otherwise it returns
/// false without mutating state. Tokens accrue continuously at
/// ``rate_per_sec`` up to ``burst``.
#[derive(Debug)]
struct TokenBucket {
    /// Currently available tokens (fractional). May briefly exceed
    /// ``burst`` if the system clock is set backwards — capped on the
    /// next refill.
    tokens: f64,
    /// Steady-state refill rate.
    rate_per_sec: f64,
    /// Maximum tokens. ``rate_per_sec * window + 1`` is the upper bound
    /// on bursts the bucket will permit before refusing.
    burst: f64,
    /// Last accrual computation timestamp.
    last_refill: Instant,
}

impl TokenBucket {
    /// `now` is injected rather than read from `Instant::now()` inside, so the
    /// refill / burst-cap logic is unit-testable by advancing a synthetic
    /// clock instead of `thread::sleep`. Production callers pass
    /// `Instant::now()`. First seam of the injectable-clock work in #1575.
    fn new(rate_per_sec: f64, burst: f64, now: Instant) -> Self {
        Self {
            tokens: burst,
            rate_per_sec,
            burst,
            last_refill: now,
        }
    }

    fn try_take(&mut self, now: Instant) -> bool {
        let elapsed = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        self.last_refill = now;
        self.tokens = (self.tokens + elapsed * self.rate_per_sec).min(self.burst);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

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
///
/// `Serialize` exists for dispatch implementations behind the
/// [`super::dispatch::WorkDispatcher`] seam that ship the params block
/// whole over a transport (the NATS path instead borrows individual
/// fields into `WorkItemRef`); it is not used on the JetStream wire.
#[derive(Debug, Clone, Default, Serialize)]
pub struct WorkParams {
    pub output_types: Option<Vec<String>>,
    pub instruction: Option<String>,
    pub is_query: bool,
    pub options: Option<serde_json::Value>,
    pub labels: Option<Vec<String>>,
    pub output_schema: Option<serde_json::Value>,
    pub query_item: Option<rmpv::Value>,
    /// Generate-only typed params (walking-skeleton wire shape). Mutually exclusive
    /// with the encode/score/extract fields above; populated only when
    /// ``endpoint == "generate"``.
    pub generate: Option<GenerateParams>,
    /// Routing-affinity hint. Carried onto the work
    /// envelope and made visible to routing logic without
    /// unpacking ``generate``. Currently read-but-ignored by the worker.
    pub routing_key: Option<String>,
    /// Prompt-cache hint. Same semantics as
    /// :attr:`routing_key`.
    pub prompt_cache_key: Option<String>,
}

/// Discriminated input for a generate work item.
///
/// The original wire shape was ``{prompt: String, ...}`` flat under
/// ``GenerateParams``. The chat-completions surface introduces an
/// OpenAI-shaped ``{messages: [...], ...}`` variant; the two are
/// mutually exclusive on a single work item. Both shapes still
/// serialise / deserialise as flat keys under :class:`GenerateParams`
/// (``#[serde(untagged)]`` + ``#[serde(flatten)]`` on the enclosing
/// field) so a prompt-only worker receiving a chat-shaped published
/// prompt item is wire-compatible.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum GenerateInput {
    /// Raw prompt string — prompt wire shape.
    Prompt { prompt: String },
    /// OpenAI-shaped chat messages. The worker renders the
    /// chat template with the model's tokenizer before forwarding to
    /// the generation adapter.
    Messages { messages: Vec<ChatMessage> },
}

impl Default for GenerateInput {
    fn default() -> Self {
        GenerateInput::Prompt {
            prompt: String::new(),
        }
    }
}

/// One image attached to a chat message, extracted from an OpenAI
/// ``image_url`` data URI at the gateway.
///
/// ``data`` remains a base64 payload string at gateway ingress so the public
/// JSON shape and existing queue publishers stay compatible. The Rust
/// sidecar validates and normalizes it to bounded msgpack binary before
/// Python model execution; mixed-version Python workers accept both binary
/// and the legacy base64 representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatImage {
    /// Base64-encoded image bytes (standard alphabet, no ``data:`` prefix).
    pub data: String,
    /// Format hint parsed from the data-URI MIME subtype (e.g. ``"png"``).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// One ordered content part of a multimodal chat message — either a text
/// fragment or an image placeholder (the bytes ride the message's flat
/// ``images`` list, consumed in order as ``Image`` parts are encountered).
/// Serializes internally-tagged so the worker reads ``{"type":"text",…}`` /
/// ``{"type":"image"}`` and round-trips through ``serde_json::Value`` on the
/// sidecar (no msgpack ``bin`` — same constraint as :struct:`ChatImage`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Image,
}

/// One chat message in the OpenAI request shape. Role is validated
/// against the allowed set at the gateway; ``content`` carries the text
/// (multi-part text parts are concatenated). Any ``image_url`` parts are
/// decoded into ``images`` and gated on the model's vision capability
/// after model resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Assistant tool-call requests (OpenAI shape), preserved across a
    /// multi-turn tool exchange so the worker can replay them into the
    /// chat template. ``None`` on plain messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Links a ``role:"tool"`` result message back to the assistant
    /// tool-call it answers. Required by OpenAI on tool messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Vision input: images decoded from the message's ``image_url`` /
    /// ``input_image`` content parts. ``None`` on text-only messages. The
    /// worker renders one placeholder per image, in order, and forwards the
    /// bytes to the engine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChatImage>>,
    /// Ordered content layout for multimodal messages: the original
    /// text↔image interleaving from the OpenAI ``content`` parts array.
    /// ``None`` on text-only / non-interleaved-legacy messages — the worker
    /// then falls back to the images-first ``content`` + ``images`` render.
    /// When present, the worker emits one image placeholder per ``Image`` part
    /// *in place*, pulling bytes from ``images`` in order; only placeholder
    /// positions change (the flat ``images`` list / ``image_data`` to the
    /// engine is unchanged).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_parts: Option<Vec<ContentPart>>,
}

/// Structured-output grammar.
///
/// Wire shape mirrors the Python ``sie_server.types.grammar.GrammarSpec``
/// dataclass. Serde discriminates on ``kind`` (``"json_schema"`` |
/// ``"regex"``) and carries the schema-or-pattern payload under
/// ``value`` plus optional ``label`` / ``strict`` from the OpenAI
/// ``response_format.json_schema`` wrapper.
///
/// The two variants are mutually exclusive on a single request; the
/// gateway's :func:`handlers::grammar::parse_grammar` is the only place
/// that builds these and enforces all safety caps (payload size, schema
/// depth, regex length, JSON-Schema reject-list) before the worker sees
/// the grammar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GrammarSpec {
    JsonSchema {
        value: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    Regex {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// EBNF / context-free grammar. Forwarded to the worker which
    /// dispatches to Outlines or XGrammar (both support EBNF natively).
    /// Subject to :const:`MAX_GRAMMAR_BYTES` at the gateway; no
    /// further structural walk is performed (the gateway does not
    /// parse EBNF — the worker's backend is the authority).
    Ebnf {
        value: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Generation parameters carried verbatim from the HTTP request body
/// through the work envelope to the worker's ``StreamingProcessor``.
///
/// ``input`` flattens into the parent map so the on-the-wire shape stays
/// ``{prompt | messages, max_new_tokens, ...}`` — backwards-compatible
/// with the original streaming work items.
///
/// `routing_key` / `prompt_cache_key` are caller-supplied
/// affinity hints used by the gateway for HRW direct-dispatch (xxh3
/// hash → per-worker subject). The raw strings are forwarded to the
/// worker so it can use them for cache lookups; the gateway hashes them
/// before any logging.
///
/// **Privacy contract:** `safety_identifier` is intentionally absent.
/// The HTTP layer parses it and discards it without logging the value.
/// Adding it here would put potentially-PII strings on the JetStream wire.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerateParams {
    #[serde(flatten)]
    pub input: GenerateInput,
    pub max_new_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// OpenAI ``frequency_penalty``: range validated upstream to
    /// ``[-2.0, 2.0]``. Forwarded verbatim to the worker; absent → the
    /// worker uses the sampler default (typically 0.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    /// OpenAI ``presence_penalty``: same shape and validation as
    /// :attr:`frequency_penalty`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// Non-OpenAI ``top_k`` (Together / Fireworks / vLLM extension):
    /// integer ``>= 1``, gateway-validated. Forwarded to SGLang's
    /// ``sampling_params["top_k"]``. Absent → top-k disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Non-OpenAI ``repetition_penalty``: float in ``(0.0, 2.0]``,
    /// gateway-validated. Forwarded to SGLang's
    /// ``sampling_params["repetition_penalty"]``. Absent → sampler default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    /// SGLang ``sampling_params["min_new_tokens"]``: integer ``>= 0``,
    /// gateway-validated. Minimum tokens the model must emit before
    /// any stop condition can fire. Absent → sampler default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    /// Per-request kwargs forwarded to the tokenizer's
    /// ``apply_chat_template``. Worker merges them under the model YAML
    /// defaults (YAML wins on conflict). The gateway accepts only the
    /// bounded public schema and the worker revalidates it after dequeue.
    /// Absent → only YAML defaults apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// Structured-output spec. Absent when the request omitted the
    /// ``grammar`` field (SIE-native) or ``response_format`` (OpenAI
    /// chat). Populated by :func:`handlers::grammar::parse_grammar`
    /// after all gateway-side safety caps have passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<GrammarSpec>,
    /// Caller-supplied routing affinity hint. Highest-priority input
    /// to the HRW key resolution in `crate::routing::key`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<String>,
    /// OpenAI-compatible cache-key hint. Second-priority input to HRW
    /// key resolution; also passed verbatim so the worker can use it
    /// for adapter-level cache lookups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// OpenAI ``tools``: non-empty array of ``{type: "function",
    /// function: {name, description?, parameters?}}``. Forwarded
    /// verbatim to the worker; the worker uses presence (rather than
    /// the schemas themselves) to enable the
    /// ``parse_tool_call_stream`` pipeline. The gateway has run the
    /// JSON-Schema safety walker on each ``function.parameters``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// OpenAI ``tool_choice``: one of ``"auto"`` / ``"none"`` /
    /// ``"required"`` or ``{type:"function", function:{name}}``.
    /// Informational on the worker today (Qwen3's chat template emits
    /// ``<tool_call>`` blocks based on the model's own decision); we
    /// still plumb it so future sampler-level constraints can read it
    /// without a wire-shape bump.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI ``parallel_tool_calls`` (default ``true``). Currently
    /// informational; surfaced on the envelope so worker-side
    /// observability can label requests by parallelism intent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI ``seed`` — signed 64-bit per-request sampling seed. Forwarded
    /// unchanged; the active generation adapter owns its reproducibility
    /// semantics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// OpenAI ``logit_bias`` — per-token additive bias on the
    /// sampler. Keys are token-id strings (OpenAI's wire shape), values
    /// are floats in ``[-100.0, 100.0]``. Gateway clamps the map size
    /// and per-value range; the worker forwards verbatim to SGLang's
    /// ``logit_bias`` sampling-param. Absent → unbiased.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::BTreeMap<String, f64>>,
    /// OpenAI ``logprobs`` flag — when ``true`` the worker requests
    /// per-token log-probabilities from SGLang and surfaces them on
    /// each generation chunk. Absent or ``false`` → no logprobs in
    /// the response (``choices[i].logprobs: null``).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// OpenAI ``top_logprobs`` — how many alternative tokens to
    /// surface per position. Range ``[0, 20]`` per OpenAI's spec;
    /// gateway clamps. Requires ``logprobs: true``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// OpenAI ``n`` — number of candidate generations. ``1`` is the
    /// gateway default. ``n>1`` is supported only on non-streaming
    /// requests; the chat handler returns 400 for ``n>1 && stream:true``.
    /// Forwarded to the worker which sets SGLang's ``sampling_params.n``
    /// and surfaces ``n`` outputs in a single ``WorkResult``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// OpenAI ``best_of``: generate this many candidates server-side and return
    /// the top ``n`` ranked by cumulative logprob. Must satisfy ``best_of >= n``.
    /// Non-streaming only. ``None`` → behaves as ``best_of == n``.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    /// Whether the client requested SSE streaming. The worker only needs this
    /// for ``n>1``: streaming fans the candidates out as per-``choice_index``
    /// delta chunks, vs. the single terminal ``candidates[]`` array used for the
    /// non-streaming aggregate. ``false``/absent → unchanged single-candidate
    /// behaviour.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// Multi-LoRA: the public served-name of a LoRA adapter to apply (declared
    /// in the model profile's ``lora_paths``). ``None`` → the base model.
    /// Forwarded to the worker, which passes it as SGLang's
    /// ``sampling_params.lora_path`` (in-batch per-request adapter selection).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_adapter: Option<String>,
}

/// Borrowed, serialize-only view of the worker work-item envelope.
///
/// Used on the publish hot path to avoid cloning per-item the fields
/// that every item in a batch shares (pool/model/gpu/router_id/
/// reply_subject/operation/bundle_config_hash and the whole
/// `WorkParams` block including `options` / `output_schema` which can
/// be non-trivial JSON trees). For an N-item encode/extract request
/// this saves roughly `7N` small-string clones plus `4N` deep
/// `Option<Vec<_>> / Option<serde_json::Value>` clones.
///
/// The field names, order, and serde attributes are guarded by tests
/// against an owned envelope shape so `rmp_serde::to_vec_named(&WorkItemRef)`
/// keeps producing the msgpack map the Python worker consumes.
/// Deserialization is intentionally not supported; results arrive as
/// `WorkResult`.
#[derive(Debug, Serialize)]
struct WorkItemRef<'a> {
    pub work_item_id: &'a str,
    pub request_id: &'a str,
    pub item_index: u32,
    pub total_items: u32,
    pub operation: &'a str,
    pub model_id: &'a str,
    pub profile_id: &'a str,
    pub engine: &'a str,
    pub pool_name: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    pub admission_pool: &'a str,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate: Option<&'a GenerateParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing_key: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<&'a str>,
    pub bundle_config_hash: &'a str,
    pub router_id: &'a str,
    pub reply_subject: &'a str,
    pub timestamp: f64,
    /// Rolling-upgrade negotiation for `result_chunk_v1` specifically. Older
    /// workers ignore this unknown map field; workers must keep publishing the
    /// legacy one-shot ``WorkResult`` unless it is true. A future chunk version
    /// requires a new capability field rather than reusing this boolean.
    pub accepts_result_chunks: bool,
    /// W3C Trace Context. Skipped on serialisation when `None`,
    /// preserving the compact pre-observability wire shape when no
    /// trace context is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<&'a str>,
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
    admission_pool: &'a str,
    gpu: &'a str,
    engine: &'a str,
    bundle_config_hash: &'a str,
    router_id: &'a str,
    reply_subject: &'a str,
    params: &'a WorkParams,
    timestamp: f64,
    /// W3C trace context captured once at publish-call time. The
    /// shared block is what every per-item [`WorkItemRef`] borrows
    /// from, so the propagator runs once per request rather than
    /// once per fan-out item.
    traceparent: Option<&'a str>,
    tracestate: Option<&'a str>,
}

/// Deserialize a msgpack field that is EITHER a byte array OR `nil`.
///
/// `WorkResult.result_msgpack` is typed `bytes | None` on the Python worker
/// (`sie_sdk.queue_types.WorkResult`): a per-item FAILURE legitimately
/// carries `result_msgpack: None`, which msgpack encodes as `nil` —
/// `serde_bytes` alone rejects that with "invalid type: unit value,
/// expected byte array", turning a worker's typed per-item error into an
/// opaque transport-level decode failure (the whole result batch is
/// dropped and the request times out). Same fix and visitor shape as the
/// sidecar's `deserialize_optional_bytes`
/// (`sie_server_sidecar/src/protocol/ipc_types.rs`).
fn deserialize_optional_bytes<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{Error, Visitor};
    use std::fmt;

    struct OptBytesVisitor;

    impl<'de> Visitor<'de> for OptBytesVisitor {
        type Value = Vec<u8>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("bytes, a byte sequence, or nil")
        }

        fn visit_unit<E: Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_none<E: Error>(self) -> Result<Self::Value, E> {
            Ok(Vec::new())
        }
        fn visit_some<D: serde::Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Self::Value, D::Error> {
            deserializer.deserialize_any(self)
        }

        fn visit_borrowed_bytes<E: Error>(self, v: &'de [u8]) -> Result<Self::Value, E> {
            Ok(v.to_vec())
        }
        fn visit_bytes<E: Error>(self, v: &[u8]) -> Result<Self::Value, E> {
            Ok(v.to_vec())
        }
        fn visit_byte_buf<E: Error>(self, v: Vec<u8>) -> Result<Self::Value, E> {
            Ok(v)
        }
        fn visit_str<E: Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.as_bytes().to_vec())
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut out = seq.size_hint().map(Vec::with_capacity).unwrap_or_default();
            while let Some(b) = seq.next_element::<u8>()? {
                out.push(b);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_any(OptBytesVisitor)
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
    #[serde(
        default,
        serialize_with = "serde_bytes::serialize",
        deserialize_with = "deserialize_optional_bytes"
    )]
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
    /// Authoritative billable-unit counts emitted by the worker engine
    /// (`sie_server.ipc_types.UnitCounts`): `input_tokens` is the real
    /// tokenizer count taken post-tokenization, never an estimate.
    /// `audio_ms` carries exact accepted-audio duration.
    /// `#[serde(default)]` keeps the wire backward-compatible — workers
    /// that don't emit it (and array-encoded legacy results) decode to
    /// `None`. Consumed by metering edges (e.g. the managed gateway's
    /// settle path); the OSS gateway itself only carries it through.
    #[serde(default)]
    pub units: Option<UnitCounts>,
    #[serde(default)]
    pub worker_direct: bool,
    /// Bundle hash held stable by the worker's shared execution barrier for
    /// this result. Missing on legacy workers and error-only results.
    #[serde(default)]
    pub executed_bundle_config_hash: Option<String>,
    /// Opaque worker-origin digest for one immutable managed deployment and
    /// observed resource shape. Optional for rolling/self-host compatibility.
    #[serde(default)]
    pub execution_identity_sha256: Option<String>,
    /// Runtime-independent release and deployment-route binding. Optional for
    /// rolling/self-host compatibility.
    #[serde(default)]
    pub execution_binding_sha256: Option<String>,
}

/// One bounded fragment of a named-msgpack encoded [`WorkResult`].
///
/// This envelope is deliberately distinguished before trying to deserialize a
/// permissive ``WorkResult``: many ``WorkResult`` fields have serde defaults,
/// so a chunk map could otherwise be mistaken for a terminal failure result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct ResultChunkV1 {
    kind: String,
    work_item_id: String,
    request_id: String,
    item_index: u32,
    #[serde(with = "serde_bytes")]
    transfer_digest: Vec<u8>,
    chunk_index: u32,
    chunk_count: u32,
    total_bytes: u64,
    #[serde(with = "serde_bytes")]
    payload: Vec<u8>,
}

/// Typed unit counts on the result path (see [`WorkResult::units`]).
/// Every field optional: a field is present only when the worker had an
/// authoritative count for it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitCounts {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub pages: Option<u64>,
    #[serde(default)]
    pub images: Option<u64>,
    /// Exact duration of accepted audio input in integer milliseconds.
    #[serde(default)]
    pub audio_ms: Option<u64>,
    /// Exact number of successfully scored query-document pairs. Appended to
    /// preserve the legacy positional MessagePack field order.
    #[serde(default)]
    pub pairs: Option<u64>,
    /// Host-measured GPU-time share for sole-tenant sealed custom serving,
    /// in whole seconds (ceil, request minimum 1). Stamped by the trusted
    /// dispatcher — never by in-jail code — and only on sealed lanes;
    /// catalog workers never emit it. Appended like `pairs`.
    #[serde(default)]
    pub gpu_second: Option<u64>,
    /// Generated tokens on the per-ITEM result path (#2432). The realtime
    /// generate surface carries its counts on the terminal chunk's
    /// ``UsageBlock`` instead; this field is how ONE buffered generation
    /// item's authoritative `completion_tokens` reaches the jobs/batch
    /// settle path, which rates every item from `WorkResult.units`.
    /// Appended last, like `pairs`/`gpu_second`, so the legacy positional
    /// MessagePack field order is preserved.
    #[serde(default)]
    pub output_tokens: Option<u64>,
}

struct CachedStreamInfo {
    num_pending: u64,
    num_consumers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct PendingGenerationGroup {
    pub model: String,
    pub display_model: String,
    pub pool: String,
    pub count: u64,
    pub waiting_first_chunk: u64,
    pub active_streams: u64,
    pub republished: u64,
    pub oldest_request_age_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, ToSchema)]
pub struct PendingGenerationSnapshot {
    pub total: u64,
    pub groups: Vec<PendingGenerationGroup>,
}

impl PendingGenerationSnapshot {
    pub fn for_model(&self, model_id: &str) -> Self {
        let groups: Vec<PendingGenerationGroup> = self
            .groups
            .iter()
            .filter(|group| group.display_model == model_id || group.model == model_id)
            .cloned()
            .collect();
        Self {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        }
    }
}

fn accumulate_pending_generation_group(
    grouped: &mut BTreeMap<(String, String, String), PendingGenerationGroup>,
    collector: &StreamCollector,
    now: Instant,
) {
    let key = (
        collector.model.clone(),
        collector.display_model.clone(),
        collector.pool.clone(),
    );
    let age_ms = now
        .saturating_duration_since(collector.published_at)
        .as_millis()
        .min(u128::from(u64::MAX)) as u64;

    let group = grouped
        .entry(key.clone())
        .or_insert_with(|| PendingGenerationGroup {
            model: key.0,
            display_model: key.1,
            pool: key.2,
            count: 0,
            waiting_first_chunk: 0,
            active_streams: 0,
            republished: 0,
            oldest_request_age_ms: 0,
        });
    group.count += 1;
    if collector.first_chunk_at.is_some() {
        group.active_streams += 1;
    } else {
        group.waiting_first_chunk += 1;
    }
    if collector.republished {
        group.republished += 1;
    }
    group.oldest_request_age_ms = group.oldest_request_age_ms.max(age_ms);
}

/// The shape of the `WORK_POOL_{pool}` streams this gateway creates.
///
/// Grouped rather than passed as loose constructor arguments because the
/// three move together: they are the part of the stream config that the
/// worker sidecars must mirror exactly (`SIE_STREAM_MAX_AGE_S`,
/// `SIE_STREAM_STORAGE`, `SIE_STREAM_REPLICAS`, all rendered onto both by
/// Helm), since whoever creates a stream first wins.
#[derive(Debug, Clone, Copy)]
pub struct WorkStreamConfig {
    pub max_age: Duration,
    /// Storage backing. Defaults to `Memory`, which is the only behaviour
    /// that existed before the knob: queued work lives in the broker's RAM
    /// and a NATS restart erases it. See [`crate::config::StreamStorage`]
    /// for the trade-off and [`WorkPublisher::ensure_stream`] for why an
    /// existing stream's storage type is reported rather than reconciled.
    pub storage: jetstream::stream::StorageType,
    /// JetStream replica count (default 1). Unlike `storage`, this IS
    /// reconcilable in place.
    pub num_replicas: usize,
}

pub struct WorkPublisher {
    jetstream: jetstream::Context,
    router_id: String,
    payload_store: Arc<dyn PayloadStore>,
    result_timeout: Duration,
    max_stream_pending: u64,
    stream: WorkStreamConfig,
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
    /// Per-lane in-flight accounting (B5, first half). Complements
    /// `max_stream_pending`, which is whole-stream and therefore pool-wide.
    /// Its decision is always computed; whether it sheds is its own flag.
    lane_admission: LaneAdmissionControl,
    pending_results: DashMap<String, ResultCollector>,
    /// Process-wide result-chunk reservation pool shared by every pending
    /// request owned by this publisher.
    result_chunk_budget: Arc<ResultChunkBudget>,
    /// Streaming aggregator. Keyed on ``request_id`` like
    /// ``pending_results`` but holds the per-chunk state for one
    /// generation request. Populated by ``publish_generate_streaming``
    /// and drained by ``handle_inbox`` when chunk envelopes arrive.
    pending_streams: DashMap<String, StreamCollector>,
    /// Request ids whose generate work item was offloaded to the object
    /// store (``payload_ref`` set, ``generate`` blanked). The blob lives at
    /// ``{request_id}_0.bin`` and must be deleted once the stream finishes —
    /// promptly at the terminal funnel (``handle_chunk``) and, as a backstop
    /// for failure/timeout terminations, reconciled against ``pending_streams``
    /// by the periodic ``cleanup_expired`` sweep.
    offloaded_streams: DashSet<String>,
    /// Object-store payload keys written or attempted for non-streaming
    /// requests. Keyed by request id so cleanup deletes only concrete keys
    /// instead of deriving every possible ``{id}_{i}.bin`` / ``{id}_score.bin``
    /// name from ``total_items``. Failed deletes stay tracked so
    /// ``cleanup_expired`` can retry them after the request collector is gone.
    offloaded_payload_keys: DashMap<String, BTreeSet<String>>,
    /// Core NATS client (cancel publishes use core NATS, not
    /// JetStream). Populated lazily by
    /// :meth:`start_inbox_subscription`; reads on the cancel path
    /// tolerate ``None`` (cancellation simply becomes a no-op).
    nats_client: tokio::sync::RwLock<Option<async_nats::Client>>,
    inbox_handle: tokio::sync::Mutex<Option<tokio::task::JoinHandle<()>>>,
    /// Runs inbox handlers off the single decode task, keyed by request id so
    /// one request's messages stay strictly ordered while different requests
    /// proceed concurrently. Created on the first
    /// [`Self::start_inbox_subscription`] and reused across NATS reconnects,
    /// so a re-subscribe never abandons already-queued work.
    inbox_pool: tokio::sync::OnceCell<KeyedWorkerPool<InboxWork>>,
    /// Runs object-store payload deletes off both the decode task and the
    /// inbox handlers. See [`Self::defer_cleanup`] for the bound policy.
    cleanup_pool: tokio::sync::OnceCell<KeyedWorkerPool<CleanupWork>>,
    /// H9 — first-chunk-fallback rate-limit buckets, keyed by
    /// ``"{model}|{pool}"``. Each bucket gates calls into
    /// [`Self::republish_to_pool`] tagged with the
    /// ``first_chunk_timeout`` reason. The NAK-driven republish path
    /// bypasses the bucket because that path is already throttled by
    /// the worker's own NAK rate. Wrapped in ``std::sync::Mutex``
    /// because the critical section is purely CPU-bound (refill +
    /// compare) and we don't hold it across awaits.
    fallback_buckets: DashMap<String, std::sync::Mutex<TokenBucket>>,
    fallback_rate_per_sec: f64,
    fallback_burst: f64,
}

/// Owns a pending collector while `publish_work` may still yield during
/// offload and JetStream publication. If that future is dropped before it can
/// hand ownership to the HTTP guard, this guard wins the same atomic removal
/// race and starts cooperative cancellation/resource cleanup.
struct PendingPublishGuard {
    publisher: Arc<WorkPublisher>,
    request_id: String,
    armed: bool,
}

impl PendingPublishGuard {
    fn new(publisher: Arc<WorkPublisher>, request_id: String) -> Self {
        Self {
            publisher,
            request_id,
            armed: true,
        }
    }

    fn defuse(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingPublishGuard {
    fn drop(&mut self) {
        if !self.armed || !self.publisher.drop_pending_result(&self.request_id) {
            return;
        }
        let publisher = Arc::clone(&self.publisher);
        let request_id = self.request_id.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    publisher.finish_abandoned_work(&request_id).await;
                });
            }
            Err(_) => {
                debug!(
                    request_id = %request_id,
                    "runtime unavailable while abandoning interrupted publish"
                );
            }
        }
    }
}

struct ResultCollector {
    _total_items: u32,
    results: Vec<Option<WorkResult>>,
    /// Partial named-msgpack ``WorkResult`` transfers keyed by work item id.
    /// Their lifetime is exactly the pending request lifetime: removing this
    /// collector on completion, timeout, or publish failure drops every buffer.
    result_chunk_transfers: BTreeMap<String, PartialResultTransfer>,
    /// Actual payload bytes still held as partial fragments.
    result_chunk_buffered_bytes: usize,
    /// Conservative reservations stay held after an accepted item completes
    /// because its decoded `WorkResult` remains in `results` until the whole
    /// request completes. A completed result rejected by first-result or
    /// direct-fallback semantics releases its reservation immediately. [`Drop`]
    /// releases the remaining request reservation on every terminal path.
    result_chunk_reserved_bytes: usize,
    result_chunk_reserved_limit: usize,
    result_chunk_budget: Arc<ResultChunkBudget>,
    sender: Option<oneshot::Sender<Vec<WorkResult>>>,
    deadline: Instant,
    operation: String,
    pool_fallback_subject: Option<String>,
    direct_fallback_worker_id: Option<String>,
    direct_fallback_republished_indices: BTreeSet<usize>,
    direct_fallback_payloads: Vec<Option<Vec<u8>>>,
    direct_fallback_republished: bool,
    /// Per-lane in-flight reservation, released when this collector drops.
    /// Every terminal path (completion, timeout, publish failure, the expiry
    /// sweep) drops the collector, so no path needs its own release call.
    _lane_reservation: Option<LaneReservation>,
}

#[derive(Debug)]
struct PartialResultTransfer {
    request_id: String,
    item_index: u32,
    transfer_digest: [u8; 32],
    chunk_count: u32,
    total_bytes: usize,
    chunks: Vec<Option<Vec<u8>>>,
    buffered_bytes: usize,
    reservation_bytes: usize,
    /// Recently superseded retry digests. This bounded tombstone set stops a
    /// delayed old chunk (including chunk zero) from flipping the active
    /// transfer back to an abandoned attempt.
    retired_digests: Vec<[u8; 32]>,
    /// Superseded chunk counts for the active digest. For a given digest,
    /// total length, and chunk count the layout is canonical; retaining old
    /// counts lets delayed fragments from a same-digest chunk-zero restart be
    /// ignored instead of mixed into the replacement.
    retired_chunk_counts: Vec<u32>,
}

#[derive(Debug)]
struct ResultChunkBudget {
    reserved_bytes: AtomicUsize,
    limit: usize,
}

static RESULT_CHUNK_BUDGET: LazyLock<Arc<ResultChunkBudget>> = LazyLock::new(|| {
    Arc::new(ResultChunkBudget::new(
        MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
    ))
});

impl ResultChunkBudget {
    fn new(limit: usize) -> Self {
        Self {
            reserved_bytes: AtomicUsize::new(0),
            limit,
        }
    }

    fn try_reserve(&self, bytes: usize) -> bool {
        let mut current = self.reserved_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return false;
            };
            if next > self.limit {
                return false;
            }
            match self.reserved_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    telemetry::record_queue_result_chunk_reservation_change(
                        telemetry::QueueResultChunkReservationChange::Reserved(bytes),
                    );
                    return true;
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        let mut current = self.reserved_bytes.load(Ordering::Acquire);
        loop {
            debug_assert!(current >= bytes, "result chunk reservation underflow");
            let released = current.min(bytes);
            let next = current - released;
            match self.reserved_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    telemetry::record_queue_result_chunk_reservation_change(
                        telemetry::QueueResultChunkReservationChange::Released(released),
                    );
                    return;
                }
                Err(observed) => current = observed,
            }
        }
    }

    #[cfg(test)]
    fn current(&self) -> usize {
        self.reserved_bytes.load(Ordering::Acquire)
    }
}

fn result_chunk_reservation_bytes(total_bytes: usize) -> Option<usize> {
    total_bytes
        .checked_mul(RESULT_CHUNK_COPY_RESERVATION_MULTIPLIER)
        .and_then(|bytes| bytes.checked_add(RESULT_CHUNK_TRANSFER_OVERHEAD_BYTES))
}

impl ResultCollector {
    fn reserve_result_chunk_transfer(
        &mut self,
        total_bytes: usize,
    ) -> Result<usize, ResultChunkReject> {
        let reservation_bytes =
            result_chunk_reservation_bytes(total_bytes).ok_or(ResultChunkReject::AggregateSize)?;
        let request_total = self
            .result_chunk_reserved_bytes
            .checked_add(reservation_bytes)
            .ok_or(ResultChunkReject::AggregateSize)?;
        if request_total > self.result_chunk_reserved_limit {
            return Err(ResultChunkReject::AggregateSize);
        }
        if !self.result_chunk_budget.try_reserve(reservation_bytes) {
            return Err(ResultChunkReject::GlobalBudget);
        }
        self.result_chunk_reserved_bytes = request_total;
        Ok(reservation_bytes)
    }

    fn release_result_chunk_reservation(&mut self, bytes: usize) {
        self.result_chunk_reserved_bytes = self.result_chunk_reserved_bytes.saturating_sub(bytes);
        self.result_chunk_budget.release(bytes);
    }
}

impl Drop for ResultCollector {
    fn drop(&mut self) {
        self.result_chunk_budget
            .release(self.result_chunk_reserved_bytes);
        self.result_chunk_reserved_bytes = 0;
    }
}

#[derive(Debug)]
struct CompletedResultTransfer {
    work_item_id: String,
    request_id: String,
    item_index: u32,
    encoded_work_result: Vec<u8>,
    reservation_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResultChunkReject {
    Kind,
    Identity,
    Digest,
    ChunkCount,
    ChunkIndex,
    ItemSize,
    PayloadSize,
    AggregateSize,
    GlobalBudget,
    MetadataConflict,
    DuplicateConflict,
    TotalMismatch,
    DigestMismatch,
    Decode,
}

impl ResultChunkReject {
    fn metric_label(self) -> &'static str {
        match self {
            Self::Kind => "kind",
            Self::Identity => "identity",
            Self::Digest => "digest",
            Self::ChunkCount => "chunk_count",
            Self::ChunkIndex => "chunk_index",
            Self::ItemSize => "item_size",
            Self::PayloadSize => "payload_size",
            Self::AggregateSize => "aggregate_size",
            Self::GlobalBudget => "global_budget",
            Self::MetadataConflict => "metadata_conflict",
            Self::DuplicateConflict => "duplicate_conflict",
            Self::TotalMismatch => "total_mismatch",
            Self::DigestMismatch => "digest_mismatch",
            Self::Decode => "decode",
        }
    }

    fn telemetry_reason(self) -> telemetry::QueueResultChunkRejectionReason {
        match self {
            Self::Kind => telemetry::QueueResultChunkRejectionReason::Kind,
            Self::Identity => telemetry::QueueResultChunkRejectionReason::Identity,
            Self::Digest => telemetry::QueueResultChunkRejectionReason::Digest,
            Self::ChunkCount => telemetry::QueueResultChunkRejectionReason::ChunkCount,
            Self::ChunkIndex => telemetry::QueueResultChunkRejectionReason::ChunkIndex,
            Self::ItemSize => telemetry::QueueResultChunkRejectionReason::ItemSize,
            Self::PayloadSize => telemetry::QueueResultChunkRejectionReason::PayloadSize,
            Self::AggregateSize => telemetry::QueueResultChunkRejectionReason::AggregateSize,
            Self::GlobalBudget => telemetry::QueueResultChunkRejectionReason::GlobalBudget,
            Self::MetadataConflict => telemetry::QueueResultChunkRejectionReason::MetadataConflict,
            Self::DuplicateConflict => {
                telemetry::QueueResultChunkRejectionReason::DuplicateConflict
            }
            Self::TotalMismatch => telemetry::QueueResultChunkRejectionReason::TotalMismatch,
            Self::DigestMismatch => telemetry::QueueResultChunkRejectionReason::DigestMismatch,
            Self::Decode => telemetry::QueueResultChunkRejectionReason::Decode,
        }
    }
}

impl std::fmt::Display for ResultChunkReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.metric_label())
    }
}

#[derive(Debug)]
enum ResultChunkStatus {
    Buffered,
    Duplicate,
    StaleRetry,
    Complete(CompletedResultTransfer),
}

#[derive(Debug)]
struct ResultChunkApply {
    status: ResultChunkStatus,
    retry_replaced: bool,
}

fn apply_result_chunk(
    collector: &mut ResultCollector,
    chunk: ResultChunkV1,
) -> Result<ResultChunkApply, ResultChunkReject> {
    if chunk.kind != RESULT_CHUNK_KIND {
        return Err(ResultChunkReject::Kind);
    }
    if chunk.request_id.is_empty() || chunk.work_item_id.is_empty() {
        return Err(ResultChunkReject::Identity);
    }
    let item_index = chunk.item_index as usize;
    if item_index >= collector.results.len()
        || chunk.work_item_id != canonical_work_item_id(&chunk.request_id, chunk.item_index)
    {
        return Err(ResultChunkReject::Identity);
    }
    let transfer_digest: [u8; 32] = chunk
        .transfer_digest
        .as_slice()
        .try_into()
        .map_err(|_| ResultChunkReject::Digest)?;
    if chunk.chunk_count == 0 || chunk.chunk_count > MAX_RESULT_CHUNKS_PER_ITEM {
        return Err(ResultChunkReject::ChunkCount);
    }
    if chunk.chunk_index >= chunk.chunk_count {
        return Err(ResultChunkReject::ChunkIndex);
    }
    if chunk.total_bytes == 0 || chunk.total_bytes > MAX_RESULT_CHUNK_ITEM_BYTES as u64 {
        return Err(ResultChunkReject::ItemSize);
    }
    let total_bytes = chunk.total_bytes as usize;
    if chunk.payload.len() > total_bytes {
        return Err(ResultChunkReject::PayloadSize);
    }

    // A legacy full result may have won while chunk fragments were in flight.
    // Keep first-result semantics and avoid allocating state for late chunks.
    if collector.results[item_index].is_some() {
        return Ok(ResultChunkApply {
            status: ResultChunkStatus::Duplicate,
            retry_replaced: false,
        });
    }

    // A worker retry may have a new digest, or may retry the same encoded
    // bytes with a different canonical chunk count. Fragments from abandoned
    // attempts must never mix with the active transfer.
    let mut replacement_retired_digests = Vec::new();
    let mut replacement_retired_chunk_counts = Vec::new();
    let mut replacement_reservation_bytes = None;
    let mut retry_replaced = false;
    let differing_digest = collector
        .result_chunk_transfers
        .get(&chunk.work_item_id)
        .is_some_and(|partial| partial.transfer_digest != transfer_digest);
    if differing_digest {
        let current = collector
            .result_chunk_transfers
            .get(&chunk.work_item_id)
            .ok_or(ResultChunkReject::MetadataConflict)?;
        // Sidecars publish a retry in order. A non-zero fragment with an
        // unknown digest is therefore a delayed/stale fragment, not authority
        // to discard the active transfer. Likewise, tombstoned digests can
        // never become active again even if their delayed chunk zero arrives.
        if chunk.chunk_index != 0 || current.retired_digests.contains(&transfer_digest) {
            return Ok(ResultChunkApply {
                status: ResultChunkStatus::StaleRetry,
                retry_replaced: false,
            });
        }
        let stale = collector
            .result_chunk_transfers
            .remove(&chunk.work_item_id)
            .ok_or(ResultChunkReject::MetadataConflict)?;
        collector.result_chunk_buffered_bytes = collector
            .result_chunk_buffered_bytes
            .saturating_sub(stale.buffered_bytes);
        collector.release_result_chunk_reservation(stale.reservation_bytes);
        replacement_retired_digests = stale.retired_digests;
        if replacement_retired_digests.len() == MAX_RETIRED_RESULT_CHUNK_DIGESTS {
            replacement_retired_digests.remove(0);
        }
        replacement_retired_digests.push(stale.transfer_digest);
        retry_replaced = true;
    } else if let Some(current) = collector.result_chunk_transfers.get(&chunk.work_item_id) {
        if current.total_bytes != total_bytes {
            // Equal SHA-256 digests identify equal encoded bytes, so their
            // declared length cannot legitimately change.
            return Err(ResultChunkReject::MetadataConflict);
        }
        if current.chunk_count != chunk.chunk_count {
            if current.retired_chunk_counts.contains(&chunk.chunk_count) {
                return Ok(ResultChunkApply {
                    status: ResultChunkStatus::StaleRetry,
                    retry_replaced: false,
                });
            }
            // Chunk zero is the only authoritative restart marker. Unknown
            // non-zero layouts fail closed; delayed known layouts are handled
            // by the tombstone check above.
            if chunk.chunk_index != 0 {
                return Err(ResultChunkReject::MetadataConflict);
            }
            let stale = collector
                .result_chunk_transfers
                .remove(&chunk.work_item_id)
                .ok_or(ResultChunkReject::MetadataConflict)?;
            collector.result_chunk_buffered_bytes = collector
                .result_chunk_buffered_bytes
                .saturating_sub(stale.buffered_bytes);
            replacement_retired_digests = stale.retired_digests;
            replacement_retired_chunk_counts = stale.retired_chunk_counts;
            if replacement_retired_chunk_counts.len() == MAX_RETIRED_RESULT_CHUNK_LAYOUTS {
                replacement_retired_chunk_counts.remove(0);
            }
            replacement_retired_chunk_counts.push(stale.chunk_count);
            // Same bytes have the same conservative reservation. Transfer it
            // to the replacement state without briefly releasing the global
            // budget and racing another request for the capacity.
            replacement_reservation_bytes = Some(stale.reservation_bytes);
            retry_replaced = true;
        }
    }

    let chunk_index = chunk.chunk_index as usize;
    if let Some(partial) = collector
        .result_chunk_transfers
        .get_mut(&chunk.work_item_id)
    {
        if partial.request_id != chunk.request_id
            || partial.item_index != chunk.item_index
            || partial.chunk_count != chunk.chunk_count
            || partial.total_bytes != total_bytes
        {
            return Err(ResultChunkReject::MetadataConflict);
        }
        if let Some(existing) = &partial.chunks[chunk_index] {
            return if existing == &chunk.payload {
                Ok(ResultChunkApply {
                    status: ResultChunkStatus::Duplicate,
                    retry_replaced,
                })
            } else {
                Err(ResultChunkReject::DuplicateConflict)
            };
        }
        let new_transfer_bytes = partial
            .buffered_bytes
            .checked_add(chunk.payload.len())
            .ok_or(ResultChunkReject::PayloadSize)?;
        if new_transfer_bytes > partial.total_bytes {
            return Err(ResultChunkReject::PayloadSize);
        }
        let new_request_bytes = collector
            .result_chunk_buffered_bytes
            .checked_add(chunk.payload.len())
            .ok_or(ResultChunkReject::AggregateSize)?;
        partial.chunks[chunk_index] = Some(chunk.payload);
        partial.buffered_bytes = new_transfer_bytes;
        collector.result_chunk_buffered_bytes = new_request_bytes;
    } else {
        let reservation_bytes = match replacement_reservation_bytes {
            Some(bytes) => bytes,
            None => collector.reserve_result_chunk_transfer(total_bytes)?,
        };
        let new_request_bytes = collector
            .result_chunk_buffered_bytes
            .checked_add(chunk.payload.len())
            .ok_or(ResultChunkReject::AggregateSize)?;
        let mut chunks = vec![None; chunk.chunk_count as usize];
        let payload_len = chunk.payload.len();
        chunks[chunk_index] = Some(chunk.payload);
        collector.result_chunk_transfers.insert(
            chunk.work_item_id.clone(),
            PartialResultTransfer {
                request_id: chunk.request_id,
                item_index: chunk.item_index,
                transfer_digest,
                chunk_count: chunk.chunk_count,
                total_bytes,
                chunks,
                buffered_bytes: payload_len,
                reservation_bytes,
                retired_digests: replacement_retired_digests,
                retired_chunk_counts: replacement_retired_chunk_counts,
            },
        );
        collector.result_chunk_buffered_bytes = new_request_bytes;
    }

    let complete = collector
        .result_chunk_transfers
        .get(&chunk.work_item_id)
        .is_some_and(|partial| partial.chunks.iter().all(Option::is_some));
    if !complete {
        return Ok(ResultChunkApply {
            status: ResultChunkStatus::Buffered,
            retry_replaced,
        });
    }

    let partial = collector
        .result_chunk_transfers
        .remove(&chunk.work_item_id)
        .ok_or(ResultChunkReject::MetadataConflict)?;
    collector.result_chunk_buffered_bytes = collector
        .result_chunk_buffered_bytes
        .saturating_sub(partial.buffered_bytes);
    if partial.buffered_bytes != partial.total_bytes {
        return Err(ResultChunkReject::TotalMismatch);
    }

    let mut encoded_work_result = Vec::with_capacity(partial.total_bytes);
    let mut transfer_hasher = Sha256::new();
    for payload in partial.chunks {
        let payload = payload.ok_or(ResultChunkReject::TotalMismatch)?;
        transfer_hasher.update(&payload);
        encoded_work_result.extend_from_slice(&payload);
    }
    if encoded_work_result.len() != partial.total_bytes {
        return Err(ResultChunkReject::TotalMismatch);
    }
    let actual_digest = transfer_hasher.finalize();
    if actual_digest[..] != partial.transfer_digest {
        return Err(ResultChunkReject::DigestMismatch);
    }

    Ok(ResultChunkApply {
        status: ResultChunkStatus::Complete(CompletedResultTransfer {
            work_item_id: chunk.work_item_id,
            request_id: partial.request_id,
            item_index: partial.item_index,
            encoded_work_result,
            reservation_bytes: partial.reservation_bytes,
        }),
        retry_replaced,
    })
}

fn decode_completed_result(
    completed: CompletedResultTransfer,
) -> Result<WorkResult, ResultChunkReject> {
    let result: WorkResult = rmp_serde::from_slice(&completed.encoded_work_result)
        .map_err(|_| ResultChunkReject::Decode)?;
    if result.work_item_id != completed.work_item_id
        || result.request_id != completed.request_id
        || result.item_index != completed.item_index
    {
        return Err(ResultChunkReject::Identity);
    }
    Ok(result)
}

fn drop_result_chunk_transfer(collector: &mut ResultCollector, work_item_id: &str) {
    if let Some(partial) = collector.result_chunk_transfers.remove(work_item_id) {
        collector.result_chunk_buffered_bytes = collector
            .result_chunk_buffered_bytes
            .saturating_sub(partial.buffered_bytes);
        collector.release_result_chunk_reservation(partial.reservation_bytes);
    }
}

fn drop_pending_result_collector(
    pending: &DashMap<String, ResultCollector>,
    request_id: &str,
) -> bool {
    pending.remove(request_id).is_some()
}

fn drop_disconnected_stream_collector(
    pending: &DashMap<String, StreamCollector>,
    request_id: &str,
) -> bool {
    let Some(collector) = pending.get(request_id) else {
        return false;
    };
    collector.latch_client_disconnected();
    drop(collector);
    pending.remove(request_id).is_some()
}

fn fail_pending_result_chunk_request(
    pending_results: &DashMap<String, ResultCollector>,
    request_id: &str,
) -> Option<()> {
    let (_, mut collector) = pending_results.remove(request_id)?;
    let results = (0..collector.results.len())
        .map(|item_index| WorkResult {
            work_item_id: canonical_work_item_id(request_id, item_index as u32),
            request_id: request_id.to_string(),
            item_index: item_index as u32,
            success: false,
            result_msgpack: Vec::new(),
            error: Some("Worker result transport validation failed".to_string()),
            error_code: Some("transport_failure".to_string()),
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        })
        .collect();
    if let Some(sender) = collector.sender.take() {
        let _ = sender.send(results);
    }
    Some(())
}

struct PublishedWorkItem {
    index: usize,
    ack: Option<jetstream::context::PublishAckFuture>,
    encoded: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StoreResultOutcome {
    Stored,
    Duplicate,
    StaleDirectFallback,
    OutOfRange,
}

type DirectFallbackPayload = (usize, Vec<u8>);
type DirectFallbackPayloads = Vec<DirectFallbackPayload>;

fn take_direct_fallback_payloads(
    collector: &mut ResultCollector,
) -> Option<(String, Option<String>, DirectFallbackPayloads)> {
    if collector.direct_fallback_republished {
        return None;
    }
    let subject = collector.pool_fallback_subject.clone()?;
    let worker_id = collector.direct_fallback_worker_id.clone();
    let mut payloads = Vec::new();
    for (index, payload) in collector.direct_fallback_payloads.iter().enumerate() {
        let still_missing = collector.results.get(index).is_some_and(Option::is_none);
        if still_missing {
            if let Some(bytes) = payload.clone() {
                payloads.push((index, bytes));
            }
        }
    }
    if payloads.is_empty() {
        return None;
    }
    collector.direct_fallback_republished = true;
    Some((subject, worker_id, payloads))
}

fn store_result_if_missing(
    collector: &mut ResultCollector,
    result: WorkResult,
) -> StoreResultOutcome {
    let idx = result.item_index as usize;
    if idx >= collector.results.len() {
        return StoreResultOutcome::OutOfRange;
    }
    if result.worker_direct && collector.direct_fallback_republished_indices.contains(&idx) {
        return StoreResultOutcome::StaleDirectFallback;
    }
    if collector.results[idx].is_some() {
        return StoreResultOutcome::Duplicate;
    }
    collector.results[idx] = Some(result);
    StoreResultOutcome::Stored
}

fn store_result_with_completed_chunk_reservation(
    collector: &mut ResultCollector,
    result: WorkResult,
    reservation_bytes: usize,
) -> StoreResultOutcome {
    let outcome = store_result_if_missing(collector, result);
    if outcome != StoreResultOutcome::Stored {
        collector.release_result_chunk_reservation(reservation_bytes);
    }
    outcome
}

fn mark_direct_fallback_republished_indices(
    collector: &mut ResultCollector,
    indices: impl IntoIterator<Item = usize>,
) {
    collector
        .direct_fallback_republished_indices
        .extend(indices);
}

fn mark_direct_fallback_confirmed(
    collector: &mut ResultCollector,
    indices: impl IntoIterator<Item = usize>,
) {
    collector.direct_fallback_republished = true;
    mark_direct_fallback_republished_indices(collector, indices);
}

fn take_ack_futures(items: &mut [PublishedWorkItem]) -> Vec<jetstream::context::PublishAckFuture> {
    items
        .iter_mut()
        .filter_map(|item| item.ack.take())
        .collect()
}

async fn await_batch_direct_fallback_acks(
    request_id: &str,
    acks: Vec<jetstream::context::PublishAckFuture>,
) -> bool {
    await_publish_acks(request_id, "batch direct fallback", acks)
        .await
        .is_ok()
}

async fn await_publish_acks(
    request_id: &str,
    context: &'static str,
    acks: Vec<jetstream::context::PublishAckFuture>,
) -> Result<(), String> {
    let mut first_error = None;
    let mut acknowledgements: FuturesUnordered<_> = acks
        .into_iter()
        .map(|ack| async move { ack.await })
        .collect();
    let deadline = tokio::time::Instant::now() + PUBLISH_ACK_COMPLETION_TIMEOUT;
    while !acknowledgements.is_empty() {
        match tokio::time::timeout_at(deadline, acknowledgements.next()).await {
            Ok(Some(Ok(_))) => {
                telemetry::record_queue_event(QueueEvent::PublishAck, QueueEventOutcome::Success)
            }
            Ok(Some(Err(error))) => {
                telemetry::record_queue_event(QueueEvent::PublishAck, QueueEventOutcome::AckError);
                warn!(
                    request_id = %request_id,
                    error = %error,
                    context,
                    "JetStream publish acknowledgement failed"
                );
                first_error.get_or_insert_with(|| error.to_string());
            }
            Ok(None) => break,
            Err(_) => {
                let unresolved = acknowledgements.len();
                for _ in 0..unresolved {
                    telemetry::record_queue_event(
                        QueueEvent::PublishAck,
                        QueueEventOutcome::AckError,
                    );
                }
                warn!(
                    request_id = %request_id,
                    context,
                    unresolved,
                    timeout_ms = PUBLISH_ACK_COMPLETION_TIMEOUT.as_millis(),
                    "JetStream publish acknowledgement monitor timed out"
                );
                first_error.get_or_insert_with(|| {
                    format!(
                        "{unresolved} acknowledgements exceeded {} ms",
                        PUBLISH_ACK_COMPLETION_TIMEOUT.as_millis()
                    )
                });
                break;
            }
        }
    }

    match first_error {
        Some(error) => Err(format!("{context} acknowledgement failed: {error}")),
        None => Ok(()),
    }
}

fn monitor_publish_acks(
    request_id: String,
    context: &'static str,
    acks: Vec<jetstream::context::PublishAckFuture>,
) -> DispatchDurability {
    DispatchDurability::from_future(
        async move { await_publish_acks(&request_id, context, acks).await },
    )
}

pub(crate) fn initial_publish_ack_count(endpoint: &str, request_items: usize) -> usize {
    if endpoint == "score" || endpoint == "generate" {
        return 1;
    }
    request_items
}

pub(crate) fn validate_queue_request_item_count(request_items: usize) -> Result<(), String> {
    if request_items <= MAX_QUEUE_REQUEST_ITEMS {
        Ok(())
    } else {
        Err(format!(
            "Queue request contains {request_items} items; the maximum is {MAX_QUEUE_REQUEST_ITEMS}"
        ))
    }
}

fn stream_name(pool: &str) -> String {
    format!("WORK_POOL_{}", pool)
}

fn canonical_stream_subjects(observed: Vec<String>, desired: &str) -> Option<Vec<String>> {
    if observed.len() == 1 && observed.first().is_some_and(|subject| subject == desired) {
        None
    } else {
        Some(vec![desired.to_string()])
    }
}

/// Normalize a model ID or operator-provided lane id for use as a single
/// NATS subject token.
///
/// The workers' JetStream pull consumer filters on
/// `sie.work.{pool}.{machine_profile}.{bundle}.*`, which matches
/// **exactly one token** in the model-ID position. NATS subject tokens
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
/// ```text
/// `/`     -> `__`
/// `.`     -> `_dot_`
/// `*`     -> `_`
/// `>`     -> `_`
/// ` `     -> `_`
/// ```
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
            // Control chars (`\n`, `\r`, `\t`, …) are whitespace/illegal
            // in a NATS subject token but are not caught by the literal
            // arms above. Map them to `_` too so the empty-registry
            // fallback path (which can interpolate an unsanitized model
            // id) can't emit a malformed subject.
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    out
}

fn work_subject(pool: &str, machine_profile: &str, bundle: &str, model: &str) -> String {
    format!(
        "sie.work.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model)
    )
}

/// Per-worker subject `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
///
/// Matches `sie_sdk.queue_types.work_worker_subject` so workers and
/// the gateway agree on the wire-level subject. The pool stream
/// filters on `sie.work.{pool}.*.*.*`, so this worker-addressed subject
/// cannot be captured by it — exactly the double-delivery guarantee
/// the design requires.
fn work_subject_worker(
    pool: &str,
    machine_profile: &str,
    bundle: &str,
    model: &str,
    worker_id: &str,
) -> String {
    // Worker ids are operator-controlled (set via `WorkerStatusMessage.name`,
    // ultimately sourced from `SIE_WORKER_ID` / `HOSTNAME` / `POD_NAME` on the
    // worker side) and would otherwise be interpolated verbatim into a NATS
    // subject — an id containing `.`, `*`, `>`, or whitespace would produce
    // an illegal subject and every HRW pick that landed on it would fail
    // with "no responders". Apply the same scrub as the model id so wonky
    // names (notably Kubernetes pod hostnames like
    // `sie-worker-7d9f-default-0.sie-worker.default.svc`) degrade to
    // deterministic underscore-joined tokens instead of disappearing from
    // the cluster.
    //
    // CROSS-LANGUAGE CONTRACT (workstream G-M5): this normalization MUST
    // produce byte-identical output to `sie_sdk.queue_types.normalize_worker_id`
    // in Python (which delegates to `normalize_model_id` for exactly this
    // reason). If you change the mapping here, mirror the change in the
    // Python helper or direct-dispatch will silently miss every worker whose
    // raw id contains the newly-changed character.
    format!(
        "sie.work.{}.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model),
        normalize_model_id(worker_id)
    )
}

/// Where a `WorkItem` should be published.
///
/// Direct-dispatch: when the HRW pick yields an
/// eligible worker, the gateway publishes to that worker's per-worker
/// subject. When no worker is eligible (empty ring, key resolution
/// missed, post-NAK / post-timeout republish), it falls back to the
/// pool subject — same target the original walking-skeleton code always used.
#[derive(Debug, Clone)]
pub enum PublishTarget {
    /// Direct-dispatch to a specific worker. Subject:
    /// `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`.
    Worker {
        pool: String,
        machine_profile: String,
        bundle: String,
        model: String,
        worker_id: String,
    },
    /// Pool fan-out — any worker subscribed to
    /// `sie.work.{pool}.{machine_profile}.{bundle}.*` can pick it up.
    Pool {
        pool: String,
        machine_profile: String,
        bundle: String,
        model: String,
    },
}

impl PublishTarget {
    /// Resolve the JetStream subject for this target.
    pub fn subject(&self) -> String {
        match self {
            PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                worker_id,
            } => work_subject_worker(pool, machine_profile, bundle, model, worker_id),
            PublishTarget::Pool {
                pool,
                machine_profile,
                bundle,
                model,
            } => work_subject(pool, machine_profile, bundle, model),
        }
    }

    /// Stable metric label describing the target kind. Wired up to
    /// publish-side metrics in a follow-up; carried in the API
    /// surface now so future callers don't have to extend `PublishTarget`.
    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            PublishTarget::Worker { .. } => "worker",
            PublishTarget::Pool { .. } => "pool",
        }
    }

    /// Build the `Pool` fallback that corresponds to a `Worker` target.
    /// Carried in the API for admission-control use cases that may need to
    /// materialise the fallback target outside `republish_to_pool`.
    #[allow(dead_code)]
    pub fn as_pool_fallback(&self) -> PublishTarget {
        match self {
            PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                ..
            }
            | PublishTarget::Pool {
                pool,
                machine_profile,
                bundle,
                model,
            } => PublishTarget::Pool {
                pool: pool.clone(),
                machine_profile: machine_profile.clone(),
                bundle: bundle.clone(),
                model: model.clone(),
            },
        }
    }

    pub fn model(&self) -> &str {
        match self {
            PublishTarget::Worker { model, .. } | PublishTarget::Pool { model, .. } => model,
        }
    }

    pub fn pool(&self) -> &str {
        match self {
            PublishTarget::Worker { pool, .. } | PublishTarget::Pool { pool, .. } => pool,
        }
    }

    pub fn machine_profile(&self) -> &str {
        match self {
            PublishTarget::Worker {
                machine_profile, ..
            }
            | PublishTarget::Pool {
                machine_profile, ..
            } => machine_profile,
        }
    }

    pub fn bundle(&self) -> &str {
        match self {
            PublishTarget::Worker { bundle, .. } | PublishTarget::Pool { bundle, .. } => bundle,
        }
    }

    /// The `(pool, machine_profile, bundle)` lane this target publishes onto.
    ///
    /// Both variants share it: a worker-direct target is one worker inside the
    /// same lane as its pool fallback, so per-lane admission accounting covers
    /// direct dispatch too — the blind spot recorded in
    /// [`WorkPublisher::check_backpressure`]'s doc block, where a saturated
    /// worker inbox reads as a healthy pool stream, does not apply here.
    pub fn lane(&self) -> LaneKey {
        LaneKey::new(self.pool(), self.machine_profile(), self.bundle())
    }
}

/// Peek the top-level `kind` discriminator on an already-decoded
/// `rmpv::Value`. Returns `Some("chunk")` / `Some("nak")` / `Some(...)`
/// for a map carrying a string `kind`, or `None` for a non-map value
/// (e.g. the array-shaped `WorkResult`) or a map without a string
/// `kind`. Lets `handle_inbox` dispatch on the decoded value without a
/// second decode from the raw slice.
fn envelope_kind(value: &rmpv::Value) -> Option<&str> {
    let rmpv::Value::Map(entries) = value else {
        return None;
    };
    for (k, v) in entries {
        if let rmpv::Value::String(s) = k {
            if s.as_str() == Some("kind") {
                return match v {
                    rmpv::Value::String(s) => s.as_str(),
                    _ => None,
                };
            }
        }
    }
    None
}

/// One decoded inbox message together with the request it belongs to.
///
/// Produced only by [`inbox_work_for_payload`], which owns the single msgpack
/// decode. Handing this — rather than the raw payload — to the worker pool is
/// what keeps the "decode exactly once, on the decode task" property while
/// the handlers run elsewhere.
#[derive(Debug)]
enum InboxWork {
    ResultChunk(Box<ResultChunkV1>),
    /// A `result_chunk_v1` envelope that failed to decode. The request is
    /// known (from the fast-path peek) and must be failed, and that failure
    /// has to be ordered against the request's other inbox work.
    ResultChunkDecodeFailure {
        request_id: String,
    },
    Chunk(Box<ChunkEnvelope>),
    Nak(Box<NakEnvelope>),
    Result(Box<WorkResult>),
}

/// Every variant carries exactly one request id, so all of a request's inbox
/// work lands on one shard and is handled in arrival order.
impl ShardKeyed for InboxWork {
    fn shard_key(&self) -> &str {
        match self {
            Self::ResultChunk(chunk) => &chunk.request_id,
            Self::ResultChunkDecodeFailure { request_id } => request_id,
            Self::Chunk(chunk) => &chunk.request_id,
            Self::Nak(nak) => &nak.request_id,
            Self::Result(result) => &result.request_id,
        }
    }
}

/// Turn one raw inbox payload into the work its handler needs, or `None` when
/// the message is to be dropped.
///
/// This is the entire decode side of the inbox loop, extracted from
/// [`WorkPublisher::handle_inbox`] so it is unit-testable without a broker.
/// In order:
///
/// 1. Fast path — peek `request_id` out of the raw bytes and drop the message
///    outright when this gateway tracks no such request, without decoding it.
/// 2. Decode the msgpack payload **exactly once** into an `rmpv::Value`.
/// 3. Peek the `kind` discriminator on that decoded value and convert the
///    *same* value into the typed envelope via `rmpv::ext::from_value` — never
///    a second decode from the raw slice. Older code re-decoded up to three
///    times per chunk, which is the hot path for streaming generation.
///
/// Only the exact v1 discriminator enters the v1 decoder. That fails malformed
/// exact-v1 envelopes closed while preserving forward compatibility for
/// ordinary/future `WorkResult` fields such as `total_bytes`.
fn inbox_work_for_payload(payload: &[u8], is_tracked: impl Fn(&str) -> bool) -> Option<InboxWork> {
    let request_id_hint = extract_request_id_fast(payload);
    if let Some(request_id) = request_id_hint {
        if !is_tracked(request_id) {
            debug!(
                request_id = %request_id,
                "fast-path skip: result for unknown request"
            );
            return None;
        }
    }

    let value: rmpv::Value = match rmp_serde::from_slice(payload) {
        Ok(value) => value,
        Err(e) => {
            warn!(error = %e, "failed to decode inbox payload");
            return None;
        }
    };

    match envelope_kind(&value) {
        Some(RESULT_CHUNK_KIND) => match rmpv::ext::from_value::<ResultChunkV1>(value) {
            Ok(chunk) => Some(InboxWork::ResultChunk(Box::new(chunk))),
            Err(e) => {
                telemetry::record_queue_result_chunk_received(None);
                telemetry::record_queue_result_chunk_rejection(
                    telemetry::QueueResultChunkRejectionReason::Decode,
                );
                warn!(error = %e, "failed to decode result chunk envelope");
                request_id_hint.map(|request_id| InboxWork::ResultChunkDecodeFailure {
                    request_id: request_id.to_string(),
                })
            }
        },
        // Chunk envelopes feed the streaming aggregator.
        Some("chunk") => match rmpv::ext::from_value::<ChunkEnvelope>(value) {
            Ok(chunk) => Some(InboxWork::Chunk(Box::new(chunk))),
            Err(e) => {
                warn!(error = %e, "failed to decode chunk envelope");
                None
            }
        },
        // Worker-emitted NAK. The publisher republishes the cached work item
        // to the pool subject; if the republish fails we surface a
        // transport-style terminal outcome to the HTTP handler.
        Some("nak") => match rmpv::ext::from_value::<NakEnvelope>(value) {
            Ok(nak) => Some(InboxWork::Nak(Box::new(nak))),
            Err(e) => {
                warn!(error = %e, "failed to decode nak envelope");
                None
            }
        },
        // Anything else (encode/score/extract WorkResults — an array-shaped
        // payload, or a map with an unknown/absent `kind`) flows to the
        // ResultCollector path.
        _ => match rmpv::ext::from_value::<WorkResult>(value) {
            Ok(result) => Some(InboxWork::Result(Box::new(result))),
            Err(e) => {
                warn!(error = %e, "failed to decode inbox result");
                None
            }
        },
    }
}

/// Deferred object-store cleanup for one finished request.
///
/// Both variants are best-effort deletes with an independent backstop in
/// [`WorkPublisher::cleanup_expired`], which is what makes it safe to shed
/// them at the queue bound.
#[derive(Debug)]
enum CleanupWork {
    /// Non-streaming request payload blobs (`offloaded_payload_keys`).
    Payloads(String),
    /// Offloaded generate blob for a streaming request (`offloaded_streams`).
    Generate(String),
}

impl ShardKeyed for CleanupWork {
    fn shard_key(&self) -> &str {
        match self {
            Self::Payloads(request_id) | Self::Generate(request_id) => request_id,
        }
    }
}

/// Hand one cleanup to the pool, or give it back for the caller to run inline.
///
/// Extracted from [`WorkPublisher::defer_cleanup`] so the hand-off policy is
/// unit-testable without a live JetStream context (matching the treatment of
/// [`cleanup_offloaded_payloads_inner`] for issue #1471).
///
/// * `pool` present and with room — queued, `None`. **The caller does not
///   await the object-store round trip.**
/// * `pool` present and saturated — shed, `None`. Safe because
///   [`WorkPublisher::cleanup_expired`] reconciles orphaned keys every 10s.
/// * `pool` absent (a publisher with no inbox subscription) — `Some(work)`,
///   so cleanup still happens inline rather than being silently skipped.
fn queue_cleanup(
    pool: Option<&KeyedWorkerPool<CleanupWork>>,
    work: CleanupWork,
) -> Option<CleanupWork> {
    let Some(pool) = pool else {
        return Some(work);
    };
    let request_id = work.shard_key().to_string();
    if !pool.try_dispatch(work) {
        debug!(
            request_id = %request_id,
            "cleanup queue saturated — leaving the blob to the periodic reconcile"
        );
    }
    None
}

/// Fast-path extraction of `request_id` from raw msgpack bytes.
/// Returns None on any parse failure (caller falls back to full deserialization).
fn extract_request_id_fast(payload: &[u8]) -> Option<&str> {
    if payload.is_empty() {
        return None;
    }

    let marker = rmp::decode::read_marker(&mut &payload[..]).ok()?;

    match marker {
        // Array format follows WorkResult field order:
        // work_item_id, request_id, item_index, ...
        Marker::FixArray(n) if n >= 2 => {
            // Marker byte consumed 1 byte
            let data = skip_msgpack_value(&payload[1..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
            Some(request_id)
        }
        Marker::Array16 => {
            // 1 marker byte + 2 length bytes = 3 bytes header
            if payload.len() < 3 {
                return None;
            }
            let len = u16::from_be_bytes([payload[1], payload[2]]);
            if len < 2 {
                return None;
            }
            let data = skip_msgpack_value(&payload[3..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
            Some(request_id)
        }
        Marker::Array32 => {
            // 1 marker byte + 4 length bytes = 5 bytes header
            if payload.len() < 5 {
                return None;
            }
            let len = u32::from_be_bytes([payload[1], payload[2], payload[3], payload[4]]);
            if len < 2 {
                return None;
            }
            let data = skip_msgpack_value(&payload[5..])?;
            let (request_id, _) = read_str_from_slice(data).ok()?;
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

/// Outcome of [`WorkPublisher::republish_to_pool_outcome`].
///
/// The legacy [`WorkPublisher::republish_to_pool`] wrapper collapses the
/// non-`Republished` arms to `false`, but `handle_nak` needs to tell
/// "already republished" apart from "no collector / no payload": a NAK
/// that arrives after the request already fell back to the pool needs a
/// second collector-state check: before a successor emits its first chunk,
/// a late NAK from the abandoned attempt is indistinguishable from a NAK
/// emitted by the successor. Once a current attempt has latched, the NAK can
/// be classified precisely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepublishOutcome {
    /// The item was re-issued to the pool subject.
    Republished,
    /// A prior NAK/timeout already republished this request; nothing was
    /// done now.
    AlreadyRepublished,
    /// Nothing could be republished: no live collector, no cached
    /// payload, or no fallback subject.
    NotPossible,
    /// H9 — the per-(model, pool) first-chunk-fallback token bucket
    /// was empty; the republish was deliberately refused so the
    /// fallback rate stays bounded under a cold-start storm. Callers
    /// should surface a 504 to the client (the request has already
    /// failed-over once; we won't safely re-fallback now).
    RateLimited,
}

/// Whether a NAK must be ignored because it is from an ABANDONED attempt.
///
/// After a first-chunk-timeout (or earlier NAK) republishes a request to the
/// pool, a healthy successor worker relatches a newer `current_attempt_id` and
/// streams into the same collector. A late NAK from the abandoned attempt would
/// otherwise tear that live successor down. The NAK is stale iff it carries a
/// non-empty attempt id and either:
///
/// - the collector has latched a different `current_attempt_id`, or
/// - the collector is between republish and successor first chunk, and the NAK
///   matches an explicitly recorded abandoned attempt id.
///
/// When neither guard can identify the NAK as abandoned, this predicate returns
/// `false`; callers then decide whether the NAK is actionable or ambiguous for
/// their state. See #1601.
fn nak_is_stale(
    current_attempt_id: Option<&str>,
    abandoned_attempt_id: Option<&str>,
    nak_attempt_id: &str,
) -> bool {
    if nak_attempt_id.is_empty() {
        return false;
    }
    match current_attempt_id {
        Some(current) => current != nak_attempt_id,
        None => abandoned_attempt_id.is_some_and(|abandoned| abandoned == nak_attempt_id),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AlreadyRepublishedNakDecision {
    DropStale,
    WaitForSuccessor,
    Fail,
}

fn already_republished_nak_decision(
    current_attempt_id: Option<&str>,
    abandoned_attempt_id: Option<&str>,
    nak_attempt_id: &str,
) -> AlreadyRepublishedNakDecision {
    if nak_is_stale(current_attempt_id, abandoned_attempt_id, nak_attempt_id) {
        return AlreadyRepublishedNakDecision::DropStale;
    }
    if current_attempt_id.is_none() {
        return AlreadyRepublishedNakDecision::WaitForSuccessor;
    }
    AlreadyRepublishedNakDecision::Fail
}

/// Pool-wide admission from one cached `WORK_POOL_{pool}` snapshot.
///
/// `num_pending` is a whole-stream count and a stream is per pool, so this
/// number cannot tell a hot model's backlog from a cold model's. That is the
/// defect [`check_lane_backpressure`] addresses; this stays exactly as it was
/// and remains the sole gate whenever per-lane enforcement is off.
///
/// Free function so the composition of the two checks is testable without a
/// live broker.
fn pool_backpressure_error(info: &CachedStreamInfo, max_stream_pending: u64) -> Option<String> {
    if info.num_consumers == 0 {
        return Some("no consumers available for work stream".to_string());
    }
    if info.num_pending > max_stream_pending {
        return Some(format!(
            "backpressure: {} pending messages exceeds threshold {}",
            info.num_pending, max_stream_pending
        ));
    }
    None
}

/// Evaluate one publish against its lane's own in-flight ceiling.
///
/// The decision is computed on every publish regardless of the enforcement
/// flag, and recorded, so it can be evaluated against production traffic
/// before it is acted on. With enforcement off this always returns `Ok`, which
/// leaves [`pool_backpressure_error`] as the only gate — today's behaviour
/// exactly.
///
/// Returns the reservation the caller must keep alive for the lifetime of the
/// request; dropping it releases the lane.
fn check_lane_backpressure(
    lane_admission: &LaneAdmissionControl,
    lane: &LaneKey,
    items: u64,
) -> Result<Option<LaneReservation>, String> {
    let admission = lane_admission.admit(lane, items);
    // Lane labels are bounded by the deployment's physical lane catalog. A
    // lane outside it is logged but not counted rather than counted under a
    // synthetic label — see `gateway_lane_admission_domain` in
    // `telemetry/contract.yaml`.
    if let Some(resolved) = lane_admission.resolve(lane) {
        telemetry::record_queue_lane_admission_decision(&resolved, admission.outcome);
    }
    match admission.outcome {
        LaneAdmissionOutcome::Admitted => Ok(admission.reservation),
        LaneAdmissionOutcome::ShedShadow => {
            debug!(
                pool = %lane.pool,
                machine_profile = %lane.machine_profile,
                bundle = %lane.bundle,
                in_flight = admission.in_flight,
                threshold = admission.threshold,
                "lane backpressure would shed this publish (shadow mode; not enforced)"
            );
            Ok(admission.reservation)
        }
        LaneAdmissionOutcome::ShedEnforced => {
            warn!(
                pool = %lane.pool,
                machine_profile = %lane.machine_profile,
                bundle = %lane.bundle,
                in_flight = admission.in_flight,
                threshold = admission.threshold,
                "lane backpressure shed this publish"
            );
            // The "backpressure:" prefix is load-bearing: it is what
            // `QueuePublishOutcome::from_error` classifies on and what
            // `handlers::proxy::record_publish_failure` matches to return the
            // same 503 + `Retry-After` as a pool-wide shed and to record
            // pending demand so KEDA scales the saturated lane.
            //
            // The lane's identity stays out of the message. This string
            // reaches the client in the `QUEUE_UNAVAILABLE` body, and a pool
            // name is operator-defined and may encode tenant context. The
            // `warn!` above and the decision metric carry the identity. The
            // wording still distinguishes a lane shed from the pool-wide one.
            Err(format!(
                "backpressure: work lane saturated ({} in-flight items exceeds threshold {})",
                admission.in_flight, admission.threshold
            ))
        }
    }
}

impl WorkPublisher {
    pub fn new(
        jetstream: jetstream::Context,
        router_id: String,
        payload_store: Arc<dyn PayloadStore>,
        result_timeout: Duration,
        max_stream_pending: u64,
        stream: WorkStreamConfig,
    ) -> Self {
        Self {
            jetstream,
            router_id,
            payload_store,
            result_timeout,
            max_stream_pending,
            stream: WorkStreamConfig {
                num_replicas: stream.num_replicas.max(1),
                ..stream
            },
            ensured_streams: DashMap::new(),
            stream_info_cache: DashMap::new(),
            lane_admission: LaneAdmissionControl::default(),
            pending_results: DashMap::new(),
            result_chunk_budget: Arc::clone(&RESULT_CHUNK_BUDGET),
            pending_streams: DashMap::new(),
            offloaded_streams: DashSet::new(),
            offloaded_payload_keys: DashMap::new(),
            nats_client: tokio::sync::RwLock::new(None),
            inbox_handle: tokio::sync::Mutex::new(None),
            inbox_pool: tokio::sync::OnceCell::new(),
            cleanup_pool: tokio::sync::OnceCell::new(),
            fallback_buckets: DashMap::new(),
            fallback_rate_per_sec: FALLBACK_RATE_PER_SEC_DEFAULT,
            fallback_burst: FALLBACK_BURST_DEFAULT,
        }
    }

    /// Install the deployment's per-lane admission policy.
    ///
    /// Constructed separately from [`Self::new`] because it is a policy, not
    /// part of the queue's wiring: [`Self::new`] installs the shadow-mode
    /// default so a publisher built without one still records the decision.
    pub fn with_lane_admission(mut self, lane_admission: LaneAdmissionControl) -> Self {
        self.lane_admission = lane_admission;
        self
    }

    /// Try to consume one first-chunk-fallback token for ``(model, pool)``.
    /// Returns true when a republish is permitted, false when the bucket is
    /// empty (the caller MUST surface a 504 / refused-republish outcome).
    /// Helper exposed so the `first_chunk_timeout`-driven call sites in
    /// `republish_to_pool_outcome` and any future fallback entry-points
    /// share the same bucket state.
    fn try_take_fallback_token(&self, model: &str, pool: &str) -> bool {
        let key = format!("{}|{}", model, pool);
        let bucket = self.fallback_buckets.entry(key).or_insert_with(|| {
            std::sync::Mutex::new(TokenBucket::new(
                self.fallback_rate_per_sec,
                self.fallback_burst,
                Instant::now(),
            ))
        });
        // Lock is a plain (non-async) ``std::sync::Mutex``; the critical
        // section is a few additions + a comparison, so we never hold it
        // across an ``await``. Poisoning means an earlier panic in the
        // critical section — treat as "deny" so a panicked process state
        // doesn't silently allow unbounded fallbacks.
        let mut guard = match bucket.value().lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.try_take(Instant::now())
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
    ///
    /// Reconciliation of an existing stream covers `subjects`, `max_age`,
    /// `discard`, and `num_replicas`. It deliberately does NOT cover
    /// `storage`: JetStream refuses an in-place storage-type change, so a
    /// mismatch is logged as a warning and the stream is left as-is.
    pub async fn ensure_stream(&self, pool: &str) -> Result<Arc<str>, String> {
        if let Some(existing) = self.ensured_streams.get(pool) {
            return Ok(Arc::clone(&existing));
        }

        let name = stream_name(pool);
        let desired_subject = format!("sie.work.{}.*.*.*", pool);
        let subjects = vec![desired_subject.clone()];

        let mut stream = self
            .jetstream
            .get_or_create_stream(jetstream::stream::Config {
                name: name.clone(),
                subjects,
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                // Configurable via `SIE_STREAM_STORAGE` / `SIE_STREAM_REPLICAS`;
                // defaults are `Memory` / 1, i.e. unchanged from before the knob
                // existed. Must match the worker sidecars — whoever creates the
                // stream first wins.
                storage: self.stream.storage,
                num_replicas: self.stream.num_replicas,
                max_age: self.stream.max_age,
                max_messages: 100_000,
                // Must match the sidecar's stream creators: at the
                // `max_messages` cap the broker rejects the new publish
                // (surfacing on the existing 503 queue-unavailable path)
                // instead of silently discarding the oldest queued work.
                discard: jetstream::stream::DiscardPolicy::New,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("create/get stream {}: {}", name, e))?;
        let observed_subjects = stream.cached_info().config.subjects.clone();
        if let Some(updated_subjects) =
            canonical_stream_subjects(observed_subjects.clone(), &desired_subject)
        {
            let mut updated = stream.cached_info().config.clone();
            updated.subjects = updated_subjects;
            self.jetstream
                .update_stream(updated)
                .await
                .map_err(|e| format!("update stream {} subjects: {}", name, e))?;
            stream = self
                .jetstream
                .get_stream(name.clone())
                .await
                .map_err(|e| format!("refresh stream {} after subject update: {}", name, e))?;
            info!(
                stream = %name,
                observed_subjects = ?observed_subjects,
                desired_subject = %desired_subject,
                "reconciled JetStream stream subjects to canonical queue lane routing"
            );
        }
        let observed_max_age = stream.cached_info().config.max_age;
        if observed_max_age != self.stream.max_age {
            let mut updated = stream.cached_info().config.clone();
            updated.max_age = self.stream.max_age;
            self.jetstream
                .update_stream(updated)
                .await
                .map_err(|e| format!("update stream {} max_age: {}", name, e))?;
            stream = self
                .jetstream
                .get_stream(name.clone())
                .await
                .map_err(|e| format!("refresh stream {} after max_age update: {}", name, e))?;
            info!(
                stream = %name,
                observed_max_age_s = observed_max_age.as_secs(),
                desired_max_age_s = self.stream.max_age.as_secs(),
                "reconciled JetStream stream max_age"
            );
        }
        // Repair pre-existing streams created with the async-nats default
        // (`Old`), which silently discards the oldest queued work at the
        // `max_messages` cap instead of rejecting the new publish.
        let observed_discard = stream.cached_info().config.discard;
        if observed_discard != jetstream::stream::DiscardPolicy::New {
            let mut updated = stream.cached_info().config.clone();
            updated.discard = jetstream::stream::DiscardPolicy::New;
            self.jetstream
                .update_stream(updated)
                .await
                .map_err(|e| format!("update stream {} discard: {}", name, e))?;
            stream = self
                .jetstream
                .get_stream(name.clone())
                .await
                .map_err(|e| format!("refresh stream {} after discard update: {}", name, e))?;
            info!(
                stream = %name,
                observed_discard = ?observed_discard,
                "reconciled JetStream stream discard policy to New (reject publishes at cap)"
            );
        }
        // Durability reconcile policy (replicas repaired, storage reported
        // but never repaired) is shared with the DLQ stream — see
        // `queue::stream_durability` for why the two fields differ.
        stream_durability::reconcile_num_replicas(
            &self.jetstream,
            &mut stream,
            &name,
            self.stream.num_replicas,
        )
        .await?;
        stream_durability::warn_on_storage_mismatch(&stream, &name, self.stream.storage);

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
    ///
    /// **Direct-dispatch caveat:** this check measures the pool stream's
    /// pending count. When HRW routes to a worker's private subject
    /// (`sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`)
    /// the pool count can read low while the chosen worker's inbox is
    /// saturated. The first-chunk timeout in `proxy::stream_generate_response`
    /// is the safety net: on timeout, `republish_to_pool` redrives the work
    /// item onto the lane's pool subject so any healthy matching worker can
    /// pick it up. Tighter per-worker admission is M5+ work (tracked
    /// alongside the §6 mixed-pool fairness scheduler).
    fn check_backpressure(&self, pool: &str) -> Result<(), String> {
        // No cached info yet (ensure_stream failed to prime) — allow
        // through; the monitor task will fill the cache shortly.
        let Some(info) = self.stream_info_cache.get(pool) else {
            return Ok(());
        };
        pool_backpressure_error(&info, self.max_stream_pending).map_or(Ok(()), Err)
    }

    /// Evaluate this publish against its lane's own in-flight ceiling, on top
    /// of the pool-wide [`Self::check_backpressure`].
    fn check_lane_backpressure(
        &self,
        lane: &LaneKey,
        items: u64,
    ) -> Result<Option<LaneReservation>, String> {
        check_lane_backpressure(&self.lane_admission, lane, items)
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
        self: &Arc<Self>,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<
        (
            String,
            oneshot::Receiver<Vec<WorkResult>>,
            DispatchDurability,
        ),
        String,
    > {
        let started = Instant::now();
        let item_count = if endpoint == "score" || endpoint == "generate" {
            1
        } else {
            u32::try_from(items.len()).unwrap_or(u32::MAX)
        };
        let result = self
            .publish_work_inner(
                target,
                admission_pool,
                endpoint,
                model,
                engine,
                bundle_config_hash,
                items,
                params,
                started,
            )
            .await;
        let outcome = result
            .as_ref()
            .map(|_| QueuePublishOutcome::Submitted)
            .unwrap_or_else(|error| QueuePublishOutcome::from_error(error));
        telemetry::record_queue_publish(QueuePublishObservation {
            operation: endpoint,
            outcome,
            duration: started.elapsed(),
            items: item_count,
        });
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn publish_work_inner(
        self: &Arc<Self>,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
        started: Instant,
    ) -> Result<
        (
            String,
            oneshot::Receiver<Vec<WorkResult>>,
            DispatchDurability,
        ),
        String,
    > {
        validate_queue_request_item_count(items.len())?;
        let ack_count = initial_publish_ack_count(endpoint, items.len());
        let pool = target.pool().to_string();
        let gpu = target.machine_profile().to_string();

        // Ensure stream exists (cached — first call per pool does admin API, subsequent are free)
        self.ensure_stream(&pool).await?;

        // Check backpressure (lock-free read from background-updated cache)
        self.check_backpressure(&pool)?;
        let total_items = if endpoint == "score" || endpoint == "generate" {
            1
        } else {
            items.len() as u32
        };
        // Then the lane this request actually lands on. The reservation lives
        // in the result collector below; if any of the early returns between
        // here and that insert fires, dropping it releases the lane.
        let lane_reservation =
            self.check_lane_backpressure(&target.lane(), u64::from(total_items))?;

        // UUIDv7 keeps the leading 48 bits as a big-endian Unix
        // millisecond timestamp, so lexicographic / B-tree-indexed
        // storage of request_ids (JetStream subjects, DashMap keys,
        // downstream log aggregators) stays time-sortable without
        // extra fields. v4 gave us uniqueness but nothing else.
        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);

        let subject = target.subject();
        let (pool_fallback_subject, direct_fallback_worker_id) = match (&target, endpoint) {
            (PublishTarget::Worker { .. }, "generate") | (PublishTarget::Pool { .. }, _) => {
                (None, None)
            }
            (PublishTarget::Worker { worker_id, .. }, _) => (
                Some(target.as_pool_fallback().subject()),
                Some(worker_id.clone()),
            ),
        };
        let direct_publish_fallback_subject = pool_fallback_subject.clone();
        let direct_publish_fallback_worker_id = direct_fallback_worker_id.clone();
        let direct_publish_retry_items = if direct_publish_fallback_subject.is_some() {
            Some(items.clone())
        } else {
            None
        };

        // Set up result collector (DashMap — lock-free per-key insert)
        let (tx, rx) = oneshot::channel();
        self.pending_results.insert(
            request_id.clone(),
            ResultCollector {
                _total_items: total_items,
                results: vec![None; total_items as usize],
                result_chunk_transfers: BTreeMap::new(),
                result_chunk_buffered_bytes: 0,
                result_chunk_reserved_bytes: 0,
                result_chunk_reserved_limit: MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
                result_chunk_budget: Arc::clone(&self.result_chunk_budget),
                sender: Some(tx),
                deadline: Instant::now() + self.result_timeout,
                operation: endpoint.to_string(),
                pool_fallback_subject,
                direct_fallback_worker_id,
                direct_fallback_republished_indices: BTreeSet::new(),
                direct_fallback_payloads: vec![None; total_items as usize],
                direct_fallback_republished: false,
                _lane_reservation: lane_reservation,
            },
        );
        let mut publish_guard = PendingPublishGuard::new(Arc::clone(self), request_id.clone());

        // Build and publish all work items, collecting ack futures for parallel await.
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // All recognized endpoints carry W3C trace context on the work
        // envelope so the worker span attaches to the gateway/client
        // trace (issue #1500). The endpoint classifier gates this and
        // fails closed for unknown labels; `inject_current_context()` is
        // a no-op when no context is active.
        let (traceparent, tracestate) = Self::trace_context_for_endpoint(endpoint);

        // Every work item in a request shares the same pool / model /
        // params block. `WorkItemRef` borrows these values so we don't
        // pay N × `Option<Vec<String>> / Option<serde_json::Value>`
        // deep clones on the hot path.
        let shared = Arc::new(WorkItemShared {
            request_id: &request_id,
            endpoint,
            model,
            pool: &pool,
            admission_pool,
            gpu: &gpu,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        });

        // Bounded concurrent publishes mean a late-failing item can race
        // ahead of earlier successes —
        // so on any error we have to unwind the collector entry
        // and whatever payloads the successful siblings already
        // wrote to the offload store, otherwise both leak until
        // the result-timeout sweep kicks in.
        let publish_outcome = self
            .publish_batch_items(endpoint, Arc::clone(&shared), total_items, items, &subject)
            .await;
        let published_items = match publish_outcome {
            Ok(items) => items,
            Err(e) => {
                if let (Some(fallback_subject), Some(retry_items)) =
                    (direct_publish_fallback_subject, direct_publish_retry_items)
                {
                    warn!(
                        request_id = %request_id,
                        subject = %subject,
                        fallback_subject = %fallback_subject,
                        error = %e,
                        "worker-direct publish failed; retrying non-streaming work on pool subject"
                    );
                    match self
                        .publish_batch_items(
                            endpoint,
                            Arc::clone(&shared),
                            total_items,
                            retry_items,
                            &fallback_subject,
                        )
                        .await
                    {
                        Ok(mut items) => {
                            let fallback_indices: Vec<usize> =
                                items.iter().map(|item| item.index).collect();
                            let fallback_acks = take_ack_futures(&mut items);
                            if let Err(ack_error) = await_publish_acks(
                                &request_id,
                                "initial pool fallback publish",
                                fallback_acks,
                            )
                            .await
                            {
                                self.pending_results.remove(&request_id);
                                self.cleanup_offloaded_payloads(&request_id).await;
                                self.cleanup_offloaded_generate(&request_id).await;
                                return Err(format!("{e}; {ack_error}"));
                            }
                            if let Some(mut entry) = self.pending_results.get_mut(&request_id) {
                                mark_direct_fallback_confirmed(entry.value_mut(), fallback_indices);
                            }
                            if let Some(worker_id) = direct_publish_fallback_worker_id.as_deref() {
                                self.publish_batch_direct_cancel(worker_id, &request_id)
                                    .await;
                            }
                            items
                        }
                        Err(fallback_err) => {
                            return Err(format!(
                                "{}; pool fallback publish failed: {}",
                                e, fallback_err
                            ));
                        }
                    }
                } else {
                    return Err(e);
                }
            }
        };
        let mut ack_futures = Vec::with_capacity(ack_count);
        let should_cache_direct_fallback = self
            .pending_results
            .get(&request_id)
            .is_some_and(|entry| entry.pool_fallback_subject.is_some());
        if should_cache_direct_fallback {
            if let Some(mut entry) = self.pending_results.get_mut(&request_id) {
                for item in &published_items {
                    if item.index < entry.direct_fallback_payloads.len() {
                        entry.direct_fallback_payloads[item.index] = Some(item.encoded.clone());
                    }
                }
            }
        }
        for mut item in published_items {
            if let Some(ack) = item.ack.take() {
                ack_futures.push(ack);
            }
        }

        // Do not add a broker round trip to the request path. The handler
        // receives a transport-neutral completion and clears pending demand
        // only after this detached monitor confirms every durable ACK.
        let durability =
            monitor_publish_acks(request_id.clone(), "initial work publish", ack_futures);

        let elapsed = started.elapsed();
        debug!(
            request_id = %request_id,
            items = total_items,
            pool = %pool,
            target = target.label(),
            endpoint = %endpoint,
            latency_ms = elapsed.as_millis(),
            "published work items"
        );

        // The HTTP abandonment guard owns cancellation after this handoff.
        publish_guard.defuse();
        Ok((request_id, rx, durability))
    }

    async fn publish_batch_items(
        &self,
        endpoint: &str,
        shared: Arc<WorkItemShared<'_>>,
        total_items: u32,
        items: Vec<rmpv::Value>,
        subject: &str,
    ) -> Result<Vec<PublishedWorkItem>, String> {
        if endpoint == "score" {
            self.publish_score(shared.as_ref(), items, subject)
                .await
                .map(|(ack, encoded)| {
                    vec![PublishedWorkItem {
                        index: 0,
                        ack: Some(ack),
                        encoded,
                    }]
                })
        } else if endpoint == "generate" {
            self.publish_generate(shared.as_ref(), subject)
                .await
                .map(|(ack, encoded)| {
                    vec![PublishedWorkItem {
                        index: 0,
                        ack: Some(ack),
                        encoded,
                    }]
                })
        } else {
            // JetStream publishes issue the send-and-receive-ack cycle
            // asynchronously per message, so a batch of N items that
            // used to serialize on `.await` now overlap their network
            // round trips. Each future in the set borrows `shared` +
            // owns its per-item `rmpv::Value`.
            let publishes = futures_util::stream::iter(items.into_iter().enumerate().map(
                |(index, item_value)| {
                    let subject = subject.to_string();
                    let shared = Arc::clone(&shared);
                    async move {
                        self.publish_single(
                            shared.as_ref(),
                            total_items,
                            index,
                            item_value,
                            &subject,
                        )
                        .await
                        .map(|(ack, encoded)| PublishedWorkItem {
                            index,
                            ack: Some(ack),
                            encoded,
                        })
                    }
                },
            ));
            let mut publishes = publishes.buffer_unordered(MAX_CONCURRENT_INITIAL_PUBLISHES);
            let mut published = Vec::with_capacity(total_items as usize);
            while let Some(item) = publishes.next().await {
                published.push(item?);
            }
            // Completion order is intentionally unordered; restore request
            // order for deterministic fallback caching and diagnostics.
            published.sort_unstable_by_key(|item| item.index);
            Ok(published)
        }
    }

    // NOTE: main's inherent `spawn_batch_direct_fallback` lives on the
    // composition seam instead (queue/dispatch.rs `WorkDispatcherExt` on
    // `Arc<dyn WorkDispatcher>`) so proprietary dispatchers inherit the
    // same one-shot direct-batch recovery.

    pub async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        let (subject, direct_worker_id, payloads, operation) = {
            let Some(mut entry) = self.pending_results.get_mut(request_id) else {
                return Ok(false);
            };
            let Some((subject, direct_worker_id, payloads)) =
                take_direct_fallback_payloads(entry.value_mut())
            else {
                return Ok(false);
            };
            let operation = entry.operation.clone();
            (subject, direct_worker_id, payloads, operation)
        };

        let fallback_indices: Vec<usize> = payloads.iter().map(|(index, _)| *index).collect();
        let mut acks = Vec::with_capacity(payloads.len());
        for (index, payload) in payloads {
            let ack = self
                .jetstream
                .publish(subject.clone(), payload.into())
                .await
                .map_err(|e| {
                    format!(
                        "publish batch fallback item {} for {}: {}",
                        index, request_id, e
                    )
                })?;
            acks.push(ack);
        }

        let fallback_confirmed = await_batch_direct_fallback_acks(request_id, acks).await;
        if fallback_confirmed {
            if let Some(mut entry) = self.pending_results.get_mut(request_id) {
                mark_direct_fallback_confirmed(entry.value_mut(), fallback_indices);
            }
            if let Some(worker_id) = direct_worker_id.as_deref() {
                self.publish_batch_direct_cancel(worker_id, request_id)
                    .await;
            }
        } else {
            warn!(
                request_id = %request_id,
                "skipping batch direct-dispatch cancel because pool fallback was not durably acked"
            );
        }

        info!(
            request_id = %request_id,
            operation = %operation,
            subject = %subject,
            reason,
            fallback_confirmed,
            "republished non-streaming direct-dispatch work to pool fallback"
        );

        Ok(true)
    }

    fn trace_context_for_endpoint(endpoint: &str) -> (Option<String>, Option<String>) {
        if Self::should_propagate_queue_trace(endpoint) {
            crate::observability::propagation::inject_current_context()
        } else {
            (None, None)
        }
    }

    fn should_propagate_queue_trace(endpoint: &str) -> bool {
        InferenceEndpoint::from_label(endpoint).injects_queue_trace_context()
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
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        let query_item = shared
            .params
            .query_item
            .as_ref()
            .ok_or_else(|| "score request missing query item".to_string())?;

        let work_item_id = canonical_work_item_id(shared.request_id, 0);
        let ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: 0,
            total_items: 1,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
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
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            accepts_result_chunks: true,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
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
            // Track before the store write (issue #1471 review): a future dropped
            // mid-`put` may still have landed the blob, so it must stay eligible
            // for cleanup.
            record_offloaded_payload_key(&self.offloaded_payload_keys, shared.request_id, &ref_key);
            match put_payload_for_plain_queue_key(
                self.payload_store.as_ref(),
                &ref_key,
                &score_payload,
            )
            .await
            {
                Err(e) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Error,
                    );
                    warn!(error = %e, "failed to offload score payload, sending inline");
                }
                Ok(queue_key) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Success,
                    );
                    let offloaded = WorkItemRef {
                        query_item: None,
                        query_payload_ref: Some(&queue_key),
                        score_items: None,
                        ..ref_item
                    };
                    encoded = rmp_serde::to_vec_named(&offloaded)
                        .map_err(|e| format!("msgpack encode offloaded score: {}", e))?;
                }
            }
        }

        let ack = self
            .jetstream
            .publish(subject.to_string(), encoded.clone().into())
            .await
            .map_err(|e| format!("publish score work item: {}", e))?;
        Ok((ack, encoded))
    }

    /// Publish a generation work item and wire up a streaming collector.
    /// Replaces the walking-skeleton path: instead of a one-shot
    /// ``WorkResult`` collector, this installs a [`StreamCollector`]
    /// that accumulates per-chunk envelopes and fires a
    /// [`StreamOutcome`] when the worker emits a terminal chunk.
    ///
    /// Returns the request id plus a receiver for the outcome. The
    /// caller should await with whatever timeout / cancellation logic
    /// is appropriate; the gateway-side timeout taxonomy lives in
    /// ``handlers/proxy.rs`` (Phase F).
    #[allow(clippy::too_many_arguments)]
    /// ``display_model`` is the requested base id surfaced on all
    /// metric labels for this request; ``target.model()`` is the
    /// dispatch id (a ``:no-spec`` grammar variant when routing fired)
    /// used for the NATS subject / work item. Equal unless routing fired.
    pub async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            std::sync::Arc<tokio::sync::Notify>,
            DispatchDurability,
        ),
        String,
    > {
        let started = Instant::now();
        let result = self
            .publish_generate_streaming_inner(
                target,
                display_model,
                engine,
                bundle_config_hash,
                params,
                admission_pool,
            )
            .await;
        let outcome = result
            .as_ref()
            .map(|_| QueuePublishOutcome::Submitted)
            .unwrap_or_else(|error| QueuePublishOutcome::from_error(error));
        telemetry::record_queue_publish(QueuePublishObservation {
            operation: "generate",
            outcome,
            duration: started.elapsed(),
            items: 1,
        });
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming_inner(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            std::sync::Arc<tokio::sync::Notify>,
            DispatchDurability,
        ),
        String,
    > {
        let model = target.model().to_string();
        let pool = target.pool().to_string();
        let machine_profile = target.machine_profile().to_string();
        // Reuse the same JetStream stream + backpressure plumbing as
        // the batch path. The stream collector replaces ResultCollector.
        self.ensure_stream(&pool).await?;
        self.check_backpressure(&pool)?;
        // One streamed generation is one queued work item, so it reserves one
        // in-flight unit on its lane.
        let lane_reservation = self.check_lane_backpressure(&target.lane(), 1)?;

        if params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let subject = target.subject();
        let pool_fallback_subject = target.as_pool_fallback().subject();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Set up the streaming aggregator before publishing the work so
        // that a fast worker can't beat us to the inbox.
        let (tx, rx) = oneshot::channel::<StreamOutcome>();
        let mut collector = StreamCollector::new(tx, model.clone(), pool.clone());
        // Metric labels surface the requested (display) id, never the
        // ``:no-spec`` dispatch variant (#1324).
        collector.display_model = display_model.to_string();
        collector.pool_fallback_subject = Some(pool_fallback_subject.clone());
        collector.lane_reservation = lane_reservation;
        // Capture the activity handle before the collector moves into
        // ``pending_streams`` so the caller never has to re-look it up
        // (eliminates the .expect() race window where an error path
        // could remove the collector before the handler retrieves it).
        let activity = collector.activity_handle();
        self.pending_streams.insert(request_id.clone(), collector);

        // M5: capture the W3C Trace Context once for envelope injection.
        let (traceparent, tracestate) = crate::observability::propagation::inject_current_context();

        let shared = WorkItemShared {
            request_id: &request_id,
            endpoint: "generate",
            model: &model,
            pool: &pool,
            admission_pool,
            gpu: &machine_profile,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        };

        let (ack, encoded) = match self.publish_generate(&shared, &subject).await {
            Ok(pair) => pair,
            Err(e) => {
                self.pending_streams.remove(&request_id);
                return Err(e);
            }
        };
        // Cache the encoded payload on the collector so
        // ``republish_to_pool`` can re-issue the same item without
        // re-running the full serialization pipeline.
        if let Some(mut entry) = self.pending_streams.get_mut(&request_id) {
            entry.encoded_payload = Some(encoded);
        }

        let durability =
            monitor_publish_acks(request_id.clone(), "initial generate publish", vec![ack]);

        Ok((request_id, rx, activity, durability))
    }

    /// Streaming-SSE variant of [`Self::publish_generate_streaming`].
    /// Installs a per-chunk broadcast tap on the collector *before*
    /// publishing the work item, so an SSE handler can subscribe to
    /// every non-stale chunk delivered through ``handle_chunk``
    /// without racing against early arrivals. Returns the request id,
    /// the terminal-outcome receiver (unchanged from the
    /// non-SSE path — the SSE handler uses the broadcast receiver for
    /// per-chunk forwarding, but the oneshot remains the canonical
    /// completion signal for cancel-guard defuse and timing
    /// accounting), and the broadcast receiver to forward chunks.
    #[allow(clippy::too_many_arguments)]
    pub async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
            DispatchDurability,
        ),
        String,
    > {
        let started = Instant::now();
        let result = self
            .publish_generate_streaming_sse_inner(
                target,
                display_model,
                engine,
                bundle_config_hash,
                params,
                admission_pool,
            )
            .await;
        let outcome = result
            .as_ref()
            .map(|_| QueuePublishOutcome::Submitted)
            .unwrap_or_else(|error| QueuePublishOutcome::from_error(error));
        telemetry::record_queue_publish(QueuePublishObservation {
            operation: "generate",
            outcome,
            duration: started.elapsed(),
            items: 1,
        });
        result
    }

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming_sse_inner(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
            DispatchDurability,
        ),
        String,
    > {
        let model = target.model().to_string();
        let pool = target.pool().to_string();
        let machine_profile = target.machine_profile().to_string();
        self.ensure_stream(&pool).await?;
        self.check_backpressure(&pool)?;
        // One streamed generation is one queued work item, so it reserves one
        // in-flight unit on its lane.
        let lane_reservation = self.check_lane_backpressure(&target.lane(), 1)?;

        if params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let request_id = uuid::Uuid::now_v7().to_string();
        let reply_subject = format!("_INBOX.{}.{}", self.router_id, request_id);
        let subject = target.subject();
        let pool_fallback_subject = target.as_pool_fallback().subject();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Install the collector + broadcast tap atomically before
        // publishing so no chunk envelope can race past the
        // subscriber.
        let (tx, rx) = oneshot::channel::<StreamOutcome>();
        let mut collector = StreamCollector::new(tx, model.clone(), pool.clone());
        // Metric labels surface the requested (display) id, never the
        // ``:no-spec`` dispatch variant (#1324).
        collector.display_model = display_model.to_string();
        collector.pool_fallback_subject = Some(pool_fallback_subject.clone());
        collector.lane_reservation = lane_reservation;
        let chunk_rx = collector.install_chunk_tap();
        self.pending_streams.insert(request_id.clone(), collector);

        // M5: capture the W3C Trace Context once for envelope injection.
        let (traceparent, tracestate) = crate::observability::propagation::inject_current_context();

        let shared = WorkItemShared {
            request_id: &request_id,
            endpoint: "generate",
            model: &model,
            pool: &pool,
            admission_pool,
            gpu: &machine_profile,
            engine,
            bundle_config_hash,
            router_id: &self.router_id,
            reply_subject: &reply_subject,
            params,
            timestamp,
            traceparent: traceparent.as_deref(),
            tracestate: tracestate.as_deref(),
        };

        let (ack, encoded) = match self.publish_generate(&shared, &subject).await {
            Ok(pair) => pair,
            Err(e) => {
                self.pending_streams.remove(&request_id);
                return Err(e);
            }
        };
        if let Some(mut entry) = self.pending_streams.get_mut(&request_id) {
            entry.encoded_payload = Some(encoded);
        }
        let durability = monitor_publish_acks(
            request_id.clone(),
            "initial generate-sse publish",
            vec![ack],
        );

        Ok((request_id, rx, chunk_rx, durability))
    }

    /// Remove a non-streaming collector immediately when its HTTP waiter is
    /// gone. The atomic DashMap remove is the terminal ownership boundary:
    /// completion, timeout, and disconnect can race, but only one path can
    /// take the sender and release the result-chunk reservation.
    pub fn drop_pending_result(&self, request_id: &str) -> bool {
        let removed = drop_pending_result_collector(&self.pending_results, request_id);
        if removed {
            debug!(
                request_id = %request_id,
                "dropped abandoned non-streaming result collector"
            );
        }
        removed
    }

    /// Finish request abandonment after the collector has been removed.
    /// Signal workers before deleting offloaded input so a queued delivery can
    /// observe the tombstone instead of repeatedly fetching a missing blob.
    pub async fn finish_abandoned_work(&self, request_id: &str) {
        self.publish_work_cancel(request_id).await;
        self.cleanup_offloaded_payloads(request_id).await;
    }

    /// Forcibly drop a streaming collector — used by the HTTP handler
    /// when the client disconnects or a timeout fires, so the inbox
    /// subscriber stops accumulating chunks that nobody will read.
    pub fn drop_pending_stream(&self, request_id: &str) {
        self.pending_streams.remove(request_id);
    }

    /// Drop a stream after its client-disconnect grace window and latch that
    /// cause for every transport holder before the collector disappears.
    pub fn drop_pending_stream_client_disconnect(&self, request_id: &str) {
        drop_disconnected_stream_collector(&self.pending_streams, request_id);
    }

    /// Tear down collector and payload state after the initial JetStream
    /// publish was submitted but its durable ACK later failed.
    ///
    /// Removal happens before the awaited object-store deletes, so request
    /// state stops accumulating even if cleanup needs the periodic retry
    /// backstop. Both cleanup helpers are idempotent and exact-key scoped.
    pub async fn abort_pending_dispatch(&self, request_id: &str, kind: PendingDispatchKind) {
        match kind {
            PendingDispatchKind::Result => {
                self.pending_results.remove(request_id);
            }
            PendingDispatchKind::Stream => {
                self.pending_streams.remove(request_id);
            }
        }
        self.cleanup_offloaded_payloads(request_id).await;
        self.cleanup_offloaded_generate(request_id).await;
    }

    pub fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
        let now = Instant::now();
        let mut grouped: BTreeMap<(String, String, String), PendingGenerationGroup> =
            BTreeMap::new();

        for entry in self.pending_streams.iter() {
            accumulate_pending_generation_group(&mut grouped, entry.value(), now);
        }

        let groups: Vec<PendingGenerationGroup> = grouped.into_values().collect();
        PendingGenerationSnapshot {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        }
    }

    pub fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot {
        self.pending_generation_snapshot().for_model(model_id)
    }

    /// Terminate an in-flight streaming request with a synthetic
    /// error outcome and fire the result sender so the HTTP handler
    /// returns immediately. Used by :meth:`handle_nak` when both the
    /// direct-dispatched worker NAKed *and* the pool republish failed
    /// — the request is unrecoverable, and the existing first-chunk
    /// timeout would otherwise make the client wait the full window
    /// for a response we already know will fail.
    ///
    /// The synthesised :class:`StreamOutcome` carries ``error =
    /// Some({code, message})``; the HTTP handler maps the code to
    /// the canonical OpenAI error envelope and HTTP status.
    fn fail_pending_stream(&self, request_id: &str, code: &str, message: &str) {
        let Some((_, mut collector)) = self.pending_streams.remove(request_id) else {
            return;
        };
        let outcome = crate::queue::streaming::StreamOutcome {
            text: String::new(),
            finish_reason: "error".to_string(),
            usage: None,
            attempt_id: collector.current_attempt_id.clone().unwrap_or_default(),
            ttft_ms: None,
            tpot_ms: None,
            error: Some(crate::queue::streaming::ChunkError {
                code: code.to_string(),
                message: message.to_string(),
                param: None,
                retry_after_s: None,
            }),
            origin: StreamOutcomeOrigin::GatewaySynthetic,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        if let Some(sender) = collector.sender.take() {
            if sender.send(outcome).is_err() {
                debug!(
                    request_id = %request_id,
                    "terminal-error outcome receiver dropped (client likely disconnected)"
                );
            }
        }
    }

    /// Cancel signal. Publishes an empty message on the core
    /// NATS subject ``cancel.{router_id}.{request_id}`` so any worker
    /// currently processing this request stops driving the adapter and
    /// emits a terminal ``finish_reason: "cancelled"`` chunk. Best-
    /// effort: silently no-ops if the core NATS client is not yet
    /// available (deployments without an active NATS connection).
    pub async fn publish_cancel(&self, request_id: &str) {
        let client_opt = { self.nats_client.read().await.clone() };
        let Some(client) = client_opt else {
            return;
        };
        let subject = format!("cancel.{}.{}", self.router_id, request_id);
        if let Err(e) = client.publish(subject, Vec::new().into()).await {
            warn!(error = %e, request_id = %request_id, "failed to publish cancel signal");
        }
    }

    /// Best-effort request-wide cancellation for abandoned non-streaming
    /// encode/score/extract work. Every sidecar records this Core NATS signal
    /// as a bounded tombstone and ACK-drops matching pool or direct deliveries
    /// before costly preparation. Static inference already inside IPC is not
    /// preempted; its late result cannot re-enter the removed collector.
    async fn publish_work_cancel(&self, request_id: &str) {
        let client_opt = { self.nats_client.read().await.clone() };
        let Some(client) = client_opt else {
            return;
        };
        let subject = format!("work_cancel.{}.{}", self.router_id, request_id);
        if let Err(e) = client.publish(subject, Vec::new().into()).await {
            warn!(
                error = %e,
                request_id = %request_id,
                "failed to publish non-streaming work cancel signal"
            );
        }
    }

    /// Best-effort cancel for abandoned non-streaming worker-direct batch
    /// work. Unlike the generation gateway-request cancel signal, this is
    /// scoped to the original direct worker. Sidecars only drop items pulled
    /// from worker-direct subjects, so the pool fallback with the same request
    /// id remains eligible to run.
    pub async fn publish_batch_direct_cancel(&self, worker_id: &str, request_id: &str) {
        let client_opt = { self.nats_client.read().await.clone() };
        let Some(client) = client_opt else {
            return;
        };
        let subject = format!(
            "batch_cancel.{}.{}.{}",
            self.router_id,
            normalize_model_id(worker_id),
            request_id
        );
        if let Err(e) = client.publish(subject, Vec::new().into()).await {
            warn!(
                error = %e,
                request_id = %request_id,
                worker_id = %worker_id,
                "failed to publish batch direct-dispatch cancel signal"
            );
        }
    }

    /// Returns `true` iff the streaming collector for `request_id` has
    /// observed at least one chunk. Cancellation telemetry uses this to
    /// distinguish abandonment before the first chunk from mid-stream exits.
    pub fn stream_observed_first_chunk(&self, request_id: &str) -> bool {
        self.pending_streams
            .get(request_id)
            .map(|entry| entry.value().first_chunk_at.is_some())
            .unwrap_or(false)
    }

    /// Snapshot of the collector's timing state for inter-chunk timeout
    /// arming. Returns ``(first_chunk_at, last_chunk_at)``.
    pub fn stream_chunk_timing(
        &self,
        request_id: &str,
    ) -> Option<(Option<Instant>, Option<Instant>)> {
        self.pending_streams
            .get(request_id)
            .map(|entry| (entry.value().first_chunk_at, entry.value().last_chunk_at))
    }

    /// Publish a single generation work item (walking-skeleton path).
    ///
    /// Mirrors ``publish_score``: one WorkItem per HTTP request, no fan-out.
    /// All sampling fields live under ``shared.params.generate`` and travel
    /// through the typed ``WorkItemRef.generate`` field; the worker reads
    /// them via ``WorkItem.get("generate")`` in
    /// ``processors/streaming.py:StreamingProcessor.process``.
    async fn publish_generate(
        &self,
        shared: &WorkItemShared<'_>,
        subject: &str,
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        if shared.params.generate.is_none() {
            return Err("generate request missing 'prompt' / 'max_new_tokens'".to_string());
        }

        let work_item_id = canonical_work_item_id(shared.request_id, 0);
        let mut ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index: 0,
            total_items: 1,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
            machine_profile: shared.gpu,
            item: None,
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
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            accepts_result_chunks: true,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
        };

        let mut encoded =
            rmp_serde::to_vec_named(&ref_item).map_err(|e| format!("msgpack encode: {}", e))?;

        // Offload large generate work items (e.g. multi-MB document images) to
        // the object store so they don't exceed NATS's max payload — mirrors
        // the encode/score paths. The blob is the ``generate`` params alone
        // (base64-string image data, so it stays serde_json::Value-decodable on
        // the sidecar). ``payload_ref`` lives at ``{request_id}_0.bin``; the
        // sidecar fetches + inlines it before handing the work item to Python.
        let offload_key;
        if encoded.len() > PAYLOAD_OFFLOAD_THRESHOLD {
            let generate = shared
                .params
                .generate
                .as_ref()
                .ok_or_else(|| "generate request missing params for offload".to_string())?;
            let gen_blob = rmp_serde::to_vec_named(generate)
                .map_err(|e| format!("msgpack encode offloaded generate: {}", e))?;
            offload_key = format!("{}_0.bin", shared.request_id);
            match put_payload_for_plain_queue_key(
                self.payload_store.as_ref(),
                &offload_key,
                &gen_blob,
            )
            .await
            {
                Err(e) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Error,
                    );
                    warn!(error = %e, "failed to offload generate payload, sending inline");
                }
                Ok(queue_key) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Success,
                    );
                    ref_item.generate = None;
                    ref_item.payload_ref = Some(&queue_key);
                    encoded = rmp_serde::to_vec_named(&ref_item)
                        .map_err(|e| format!("msgpack encode offloaded generate item: {}", e))?;
                    self.offloaded_streams.insert(shared.request_id.to_string());
                }
            }
        }

        // Defense-in-depth dedup: stamp `Nats-Msg-Id` = request_id so a
        // gateway-side retry (or a duplicate publish racing on the same
        // request) collapses to a single message inside the stream's
        // dedup window. This is the *initial* attempt, so the bare
        // request_id is the right identity; the pool-republish path uses
        // a generation-suffixed id (see `republish_to_pool`) precisely so
        // it is NOT swallowed as a duplicate of this publish.
        let ack = self
            .jetstream
            .send_publish(
                subject.to_string(),
                jetstream::message::PublishMessage::build()
                    .message_id(shared.request_id)
                    .payload(encoded.clone().into()),
            )
            .await
            .map_err(|e| format!("publish generate work item: {}", e))?;
        Ok((ack, encoded))
    }

    /// Re-issue a previously-published generation item to the
    /// pool subject after a NAK or first-chunk timeout. Idempotent via
    /// the per-collector ``republished`` guard so a concurrent NAK +
    /// timeout race cannot double-publish.
    ///
    /// Returns `Ok(true)` if the item was republished, `Ok(false)` if
    /// nothing was done (no collector, no cached payload, or already
    /// republished), and `Err` when the publish or its bounded JetStream
    /// acknowledgement fails.
    ///
    /// Thin wrapper over [`Self::republish_to_pool_outcome`] that keeps
    /// the historical `bool` contract for callers (the first-chunk-timeout
    /// paths in `proxy.rs` / `sse.rs`) that only need "did we republish".
    ///
    /// H9 — a refused republish (token bucket empty) collapses to
    /// `Ok(false)` for backward compatibility; callers that need to
    /// distinguish "refused / rate-limited" from "nothing to do"
    /// should call [`Self::republish_to_pool_status`] instead. The
    /// canonical request telemetry records the rate-limit outcome independently
    /// of either return contract; the collector derives backend-specific names.
    pub async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        Ok(matches!(
            self.republish_to_pool_outcome(request_id, reason).await?,
            RepublishOutcome::Republished
        ))
    }

    /// Three-state variant of [`Self::republish_to_pool`] for the
    /// first-chunk-timeout call sites that need to distinguish "rate
    /// limited" (surface a 504) from "nothing was republished" (surface
    /// the underlying first_chunk timeout). Returns ``Ok(true)`` on a
    /// successful republish, ``Ok(false)`` on any non-rate-limit
    /// no-op, and ``Err(...)`` for the rate-limit refusal so the call
    /// site doesn't have to introduce a new sentinel type. The error
    /// string is wire-stable: ``"fallback_rate_limited"``.
    #[allow(dead_code)] // reserved for future proxy.rs/sse.rs adoption (H9)
    pub async fn republish_to_pool_or_rate_limited(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        match self.republish_to_pool_outcome(request_id, reason).await? {
            RepublishOutcome::Republished => Ok(true),
            RepublishOutcome::RateLimited => Err("fallback_rate_limited".to_string()),
            RepublishOutcome::AlreadyRepublished | RepublishOutcome::NotPossible => Ok(false),
        }
    }

    /// Republish variant that distinguishes "already republished" from
    /// "nothing to republish" (see [`RepublishOutcome`]). `handle_nak`
    /// uses it to surface a 429 when a NAK lands on a request that has
    /// already fallen back to the pool.
    async fn republish_to_pool_outcome(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<RepublishOutcome, String> {
        self.republish_to_pool_outcome_with_abandoned_attempt(request_id, reason, None)
            .await
    }

    async fn republish_to_pool_outcome_with_abandoned_attempt(
        &self,
        request_id: &str,
        reason: &'static str,
        abandoned_attempt_id: Option<&str>,
    ) -> Result<RepublishOutcome, String> {
        // Take a short lock to: check + flip `republished`, copy out
        // the encoded bytes + the pool subject, bump the attempt
        // generation. We drop the entry lock before awaiting the
        // JetStream publish to avoid holding the DashMap shard.
        let (subject, payload, generation) = {
            let Some(mut entry) = self.pending_streams.get_mut(request_id) else {
                return Ok(RepublishOutcome::NotPossible);
            };
            if entry.republished {
                return Ok(RepublishOutcome::AlreadyRepublished);
            }
            let Some(subject) = entry.pool_fallback_subject.clone() else {
                return Ok(RepublishOutcome::NotPossible);
            };
            let Some(payload) = entry.encoded_payload.clone() else {
                return Ok(RepublishOutcome::NotPossible);
            };
            // H9 — gate the first-chunk-fallback path on a per-(model,
            // pool) token bucket. Skipped for the NAK-driven reasons
            // because that path is already self-throttled by the
            // worker's own NAK rate; rate-limiting it again would
            // surface a noisy 429/504 during a single misbehaving
            // worker draining its queue. The check happens BEFORE we
            // flip ``entry.republished`` so a refused attempt can be
            // retried after the bucket refills (the deadline-armed
            // caller in proxy.rs/sse.rs already breaks out on the
            // refused signal — no spin).
            if reason == "first_chunk_timeout"
                && !self.try_take_fallback_token(entry.model.as_str(), entry.pool.as_str())
            {
                return Ok(RepublishOutcome::RateLimited);
            }
            entry.republished = true;
            let gen = entry.bump_attempt_generation();
            if let Some(attempt_id) = abandoned_attempt_id {
                entry.record_abandoned_attempt_id(attempt_id);
            }
            (subject, payload, gen)
        };

        info!(
            request_id = %request_id,
            reason = reason,
            generation = generation,
            subject = %subject,
            "republishing generate item to pool"
        );

        // Defense-in-depth dedup: stamp a generation-suffixed
        // `Nats-Msg-Id`. It must differ from the initial publish's
        // `request_id` (and from any earlier republish) so JetStream's
        // dedup window does NOT swallow the republish as a duplicate of
        // the original — that would silently break the fallback. The
        // `attempt_generation` monotonically increases per republish, so
        // `{request_id}#{generation}` is unique per attempt while still
        // collapsing an accidental duplicate republish of the *same*
        // generation.
        let msg_id = format!("{}#{}", request_id, generation);
        match self
            .jetstream
            .send_publish(
                subject.clone(),
                jetstream::message::PublishMessage::build()
                    .message_id(&msg_id)
                    .payload(payload.into()),
            )
            .await
        {
            Ok(ack) => {
                // A fallback is not durable merely because Core NATS accepted
                // the send. Recovery paths are infrequent, so await the same
                // bounded JetStream ACK contract used by initial dispatch and
                // report a prompt failure instead of waiting for a later
                // first/inter-chunk timeout.
                match await_publish_acks(request_id, "pool republish", vec![ack]).await {
                    Ok(()) => Ok(RepublishOutcome::Republished),
                    Err(error) => {
                        self.rollback_republish_attempt(request_id, generation);
                        Err(error)
                    }
                }
            }
            Err(e) => {
                // Roll back the `republished`/`attempt_generation`
                // flip so a downstream NAK or first-chunk-timeout can
                // retry. Without this, the request hangs until
                // overall-timeout: subsequent `republish_to_pool` calls
                // hit the `entry.republished` short-circuit at the top
                // and return `AlreadyRepublished`, but the worker never
                // received anything to begin with.
                self.rollback_republish_attempt(request_id, generation);
                Err(format!("republish to pool failed: {}", e))
            }
        }
    }

    fn rollback_republish_attempt(&self, request_id: &str, generation: u64) {
        if let Some(mut entry) = self.pending_streams.get_mut(request_id) {
            // Only undo the attempt we initiated. This protects future code
            // from rewinding a newer generation if another recovery action is
            // ever allowed while an ACK is outstanding.
            if entry.republished && entry.attempt_generation == generation {
                entry.republished = false;
                entry.rewind_attempt_generation();
            }
        }
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
    ) -> Result<(jetstream::context::PublishAckFuture, Vec<u8>), String> {
        let item_index = index as u32;
        let work_item_id = canonical_work_item_id(shared.request_id, item_index);
        let mut ref_item = WorkItemRef {
            work_item_id: &work_item_id,
            request_id: shared.request_id,
            item_index,
            total_items,
            operation: shared.endpoint,
            model_id: shared.model,
            profile_id: "default",
            engine: shared.engine,
            pool_name: shared.pool,
            admission_pool: shared.admission_pool,
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
            generate: shared.params.generate.as_ref(),
            routing_key: shared.params.routing_key.as_deref(),
            prompt_cache_key: shared.params.prompt_cache_key.as_deref(),
            bundle_config_hash: shared.bundle_config_hash,
            router_id: shared.router_id,
            reply_subject: shared.reply_subject,
            timestamp: shared.timestamp,
            accepts_result_chunks: true,
            traceparent: shared.traceparent,
            tracestate: shared.tracestate,
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
            // Track the request as offloaded BEFORE the store write: this future
            // can be dropped mid-`put` when a concurrent sibling publish fails,
            // yet the object-store write may still have landed.
            // Recording the id first keeps the blob eligible for cleanup on the
            // error/timeout unwind (issue #1471 review).
            record_offloaded_payload_key(
                &self.offloaded_payload_keys,
                shared.request_id,
                &offload_key,
            );
            match put_payload_for_plain_queue_key(
                self.payload_store.as_ref(),
                &offload_key,
                &item_msgpack,
            )
            .await
            {
                Err(e) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Error,
                    );
                    warn!(error = %e, "failed to offload payload, sending inline");
                }
                Ok(queue_key) => {
                    telemetry::record_queue_event(
                        QueueEvent::PayloadOffload,
                        QueueEventOutcome::Success,
                    );
                    ref_item.item = None;
                    ref_item.payload_ref = Some(&queue_key);
                    encoded = rmp_serde::to_vec_named(&ref_item)
                        .map_err(|e| format!("msgpack encode offloaded: {}", e))?;
                }
            }
        }

        let ack = self
            .jetstream
            .publish(subject.to_string(), encoded.clone().into())
            .await
            .map_err(|e| format!("publish work item {}/{}: {}", index, total_items, e))?;
        Ok((ack, encoded))
    }

    /// Fail a non-streaming request immediately after a malformed or
    /// over-budget result transfer. The message is intentionally static: wire
    /// bytes and attacker-controlled metadata never cross the public error
    /// boundary. Replacing every item with the same typed failure prevents a
    /// partial-success response from silently hiding a corrupt item.
    async fn fail_result_chunk_request(&self, request_id: &str) {
        let Some(()) = fail_pending_result_chunk_request(&self.pending_results, request_id) else {
            return;
        };
        // Removing the collector drops all partial transfer buffers. Keep the
        // HTTP handler as the one result-wait telemetry authority, while the
        // transport still cancels queued work and releases retained payloads.
        self.finish_abandoned_work(request_id).await;
    }

    async fn handle_result_chunk(&self, chunk: ResultChunkV1) {
        let request_id = chunk.request_id.clone();
        let work_item_id = chunk.work_item_id.clone();
        telemetry::record_queue_result_chunk_received(Some(chunk.payload.len()));

        let applied = {
            let mut entry = match self.pending_results.get_mut(&request_id) {
                Some(entry) => entry,
                None => {
                    debug!(request_id = %request_id, "result chunk for unknown request");
                    return;
                }
            };
            apply_result_chunk(entry.value_mut(), chunk)
        };

        let applied = match applied {
            Ok(applied) => applied,
            Err(reason) => {
                telemetry::record_queue_result_chunk_rejection(reason.telemetry_reason());
                warn!(
                    request_id = %request_id,
                    work_item_id = %work_item_id,
                    reason = reason.metric_label(),
                    "rejecting invalid result chunk transfer"
                );
                self.fail_result_chunk_request(&request_id).await;
                return;
            }
        };

        if applied.retry_replaced {
            telemetry::record_queue_result_chunk_event(
                telemetry::QueueResultChunkEvent::RetryReplacement,
            );
        }
        match applied.status {
            ResultChunkStatus::Buffered => {}
            ResultChunkStatus::Duplicate => {
                telemetry::record_queue_result_chunk_event(
                    telemetry::QueueResultChunkEvent::Duplicate,
                );
                debug!(
                    request_id = %request_id,
                    work_item_id = %work_item_id,
                    "accepted byte-identical duplicate result chunk"
                );
            }
            ResultChunkStatus::StaleRetry => {
                telemetry::record_queue_result_chunk_event(
                    telemetry::QueueResultChunkEvent::StaleRetry,
                );
                debug!(
                    request_id = %request_id,
                    work_item_id = %work_item_id,
                    "ignored stale result chunk retry fragment"
                );
            }
            ResultChunkStatus::Complete(completed) => {
                let reservation_bytes = completed.reservation_bytes;
                match decode_completed_result(completed) {
                    Ok(result) => {
                        telemetry::record_queue_result_chunk_event(
                            telemetry::QueueResultChunkEvent::TransferCompleted,
                        );
                        self.handle_result_with_chunk_reservation(result, reservation_bytes)
                            .await;
                    }
                    Err(reason) => {
                        telemetry::record_queue_result_chunk_rejection(reason.telemetry_reason());
                        warn!(
                            request_id = %request_id,
                            work_item_id = %work_item_id,
                            reason = reason.metric_label(),
                            "rejecting reconstructed result transfer"
                        );
                        self.fail_result_chunk_request(&request_id).await;
                    }
                }
            }
        }
    }

    /// Handle an incoming result message (called from inbox subscription).
    pub async fn handle_result(&self, result: WorkResult) {
        self.handle_result_with_chunk_reservation(result, 0).await;
    }

    async fn handle_result_with_chunk_reservation(
        &self,
        result: WorkResult,
        completed_reservation_bytes: usize,
    ) {
        let request_id = result.request_id.clone();
        let work_item_id = result.work_item_id.clone();
        let item_index = result.item_index;

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
            match store_result_with_completed_chunk_reservation(
                collector,
                result,
                completed_reservation_bytes,
            ) {
                StoreResultOutcome::Stored => {
                    // A legacy one-shot result may win a rolling-upgrade race
                    // after some fragments arrived. Release those bytes now.
                    drop_result_chunk_transfer(collector, &work_item_id);
                }
                StoreResultOutcome::Duplicate => {
                    debug!(
                        request_id = %request_id,
                        item_index,
                        "ignoring duplicate result for already-completed batch item"
                    );
                }
                StoreResultOutcome::StaleDirectFallback => {
                    debug!(
                        request_id = %request_id,
                        item_index,
                        "ignoring stale worker-direct result for republished batch fallback item"
                    );
                }
                StoreResultOutcome::OutOfRange => {
                    warn!(
                        request_id = %request_id,
                        item_index,
                        total_items = collector.results.len(),
                        "received result with out-of-range item_index"
                    );
                }
            }
            collector.results.iter().all(|r| r.is_some())
        };
        // DashMap per-key lock is released here

        if all_done {
            // Remove atomically — only one thread can win this remove
            if let Some((_, mut collector)) = self.pending_results.remove(&request_id) {
                let results: Vec<WorkResult> =
                    collector.results.drain(..).map(|r| r.unwrap()).collect();

                if let Some(sender) = collector.sender.take() {
                    let _ = sender.send(results);
                }

                self.defer_cleanup(CleanupWork::Payloads(request_id.clone()))
                    .await;
            }
        }
    }

    /// The keyed pool that runs inbox handlers off the decode task.
    ///
    /// The handler holds a `Weak` so the pool tasks never keep the publisher
    /// alive on their own, and a dropped publisher makes them no-ops.
    async fn inbox_pool(self: &Arc<Self>) -> &KeyedWorkerPool<InboxWork> {
        self.inbox_pool
            .get_or_init(|| async {
                let publisher = Arc::downgrade(self);
                KeyedWorkerPool::new(
                    INBOX_POOL_NAME,
                    INBOX_POOL_SHARDS,
                    INBOX_POOL_DEPTH,
                    move |work: InboxWork| {
                        let publisher = publisher.clone();
                        async move {
                            let Some(publisher) = publisher.upgrade() else {
                                return;
                            };
                            publisher.apply_inbox_work(work).await;
                        }
                    },
                )
            })
            .await
    }

    /// The keyed pool that runs object-store cleanup off every hot path.
    async fn cleanup_pool(self: &Arc<Self>) -> &KeyedWorkerPool<CleanupWork> {
        self.cleanup_pool
            .get_or_init(|| async {
                let publisher = Arc::downgrade(self);
                KeyedWorkerPool::new(
                    CLEANUP_POOL_NAME,
                    CLEANUP_POOL_SHARDS,
                    CLEANUP_POOL_DEPTH,
                    move |work: CleanupWork| {
                        let publisher = publisher.clone();
                        async move {
                            let Some(publisher) = publisher.upgrade() else {
                                return;
                            };
                            publisher.run_cleanup(work).await;
                        }
                    },
                )
            })
            .await
    }

    /// Run one decoded inbox message's handler. Called only from an inbox
    /// worker task, never from the decode task.
    async fn apply_inbox_work(&self, work: InboxWork) {
        match work {
            InboxWork::ResultChunk(chunk) => self.handle_result_chunk(*chunk).await,
            InboxWork::ResultChunkDecodeFailure { request_id } => {
                self.fail_result_chunk_request(&request_id).await;
            }
            InboxWork::Chunk(chunk) => self.handle_chunk(*chunk).await,
            InboxWork::Nak(nak) => self.handle_nak(*nak).await,
            InboxWork::Result(result) => self.handle_result(*result).await,
        }
    }

    /// Queue object-store cleanup for a finished request instead of awaiting
    /// it here.
    ///
    /// Cleanup is an object-store round trip. Awaiting it on a hot path — for
    /// the completion funnels below, historically the single inbox decode task
    /// — stalls every other in-flight request in the process, which is the
    /// defect this exists to remove.
    ///
    /// **At the bound this sheds rather than blocks.** Both cleanups are
    /// already best-effort with an independent backstop: `cleanup_expired`
    /// reconciles `offloaded_payload_keys` and `offloaded_streams` against the
    /// in-flight maps every 10s and retries whatever is orphaned. So a shed
    /// enqueue costs at most one reconcile interval of object-store residency
    /// and nothing else, whereas blocking would put the stall back on the
    /// caller. This is the opposite policy from the inbox pool, where nothing
    /// may be dropped.
    async fn defer_cleanup(&self, work: CleanupWork) {
        if let Some(work) = queue_cleanup(self.cleanup_pool.get(), work) {
            self.run_cleanup(work).await;
        }
    }

    async fn run_cleanup(&self, work: CleanupWork) {
        match work {
            CleanupWork::Payloads(request_id) => {
                self.cleanup_offloaded_payloads(&request_id).await;
            }
            CleanupWork::Generate(request_id) => {
                self.cleanup_offloaded_generate(&request_id).await;
            }
        }
    }

    /// Start the inbox subscription for result collection.
    /// Aborts the previous inbox loop if one exists (prevents duplicates on NATS reconnect).
    pub async fn start_inbox_subscription(
        self: &Arc<Self>,
        client: &async_nats::Client,
    ) -> Result<(), String> {
        // Stash the client so the cancel path can publish
        // ``cancel.{router_id}.{request_id}`` without re-acquiring a
        // handle from the nats manager.
        {
            let mut slot = self.nats_client.write().await;
            *slot = Some(client.clone());
        }

        // Bring both worker pools up before the first message can arrive.
        // They are `OnceCell`s, so a reconnect re-subscribe reuses the running
        // pools instead of abandoning whatever they still have queued.
        self.inbox_pool().await;
        self.cleanup_pool().await;

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

    /// The single inbox decode task.
    ///
    /// It now only decodes (exactly once per message, in
    /// [`inbox_work_for_payload`]) and hands the typed result to the keyed
    /// worker pool. Handlers no longer run here, so one slow handler cannot
    /// hold up decode for every other in-flight request in the process.
    async fn handle_inbox(self: &Arc<Self>, mut subscriber: async_nats::Subscriber) {
        let pool = self.inbox_pool().await;
        while let Some(msg) = subscriber.next().await {
            // DashMap contains_key is lock-free; this is the fast-path skip
            // that keeps unknown-request traffic from ever being decoded.
            let Some(work) = inbox_work_for_payload(&msg.payload, |request_id| {
                self.pending_results.contains_key(request_id)
                    || self.pending_streams.contains_key(request_id)
            }) else {
                continue;
            };

            // Bounded hand-off keyed by request id: all of one request's
            // messages take the same shard and stay in arrival order, while
            // different requests run concurrently.
            //
            // At the bound this blocks rather than sheds. A dropped message
            // here would be a lost result (request times out) or a lost chunk
            // (`StreamCollector::apply` sees a `seq` gap and fails the stream
            // with `transport_failure`) — a throughput fix must not buy
            // itself a new failure mode. Blocking is also never worse than
            // the behaviour this replaces, where *every* message was handled
            // inline on this task.
            if !pool.dispatch(work).await {
                warn!("inbox worker pool unavailable — ending inbox loop");
                break;
            }
        }

        warn!("inbox subscription ended");
    }

    /// Handle a worker-emitted NAK envelope.
    ///
    /// Maps the envelope's ``reason`` field into one of a small
    /// closed set of metric labels so dashboards can distinguish
    /// the failure mode (KV budget vs. model-not-loaded vs.
    /// worker-shutting-down) without unbounded cardinality.
    /// Unknown reasons fall through to the generic ``nak`` bucket so
    /// we never lose data on a forward-compat addition.
    async fn handle_nak(&self, nak: NakEnvelope) {
        // Attempt-isolation guard (mirrors the stale-chunk filter in
        // StreamCollector::apply, streaming.rs): a NAK from an ABANDONED
        // attempt must not act on the request. After a first-chunk-timeout (or
        // earlier NAK) republishes to the pool, a healthy successor worker
        // relatches a newer `current_attempt_id` and streams. A late NAK from
        // the abandoned attempt would otherwise reach the `AlreadyRepublished`
        // arm and `fail_pending_stream` the live successor with a spurious 429,
        // dropping its already-produced terminal. If the collector has either
        // latched a different current_attempt_id or recorded this NAK's attempt
        // id as abandoned during a prior republish, it is a stale leftover —
        // drop it. See #1601.
        // Capture the latched/abandoned attempt ids plus the bounded metric labels
        // (display_model/pool) in one lookup so the stale-drop path can record
        // a metric without a second DashMap probe. The `Ref` guard is dropped
        // at the end of the closure — nothing is held across the `.inc()`.
        let collector_state = self.pending_streams.get(&nak.request_id).map(|e| {
            let collector = e.value();
            (
                collector.current_attempt_id.clone(),
                collector.abandoned_attempt_id.clone(),
                collector.display_model.clone(),
                collector.pool.clone(),
            )
        });
        let latched_attempt = collector_state
            .as_ref()
            .and_then(|(current, ..)| current.clone());
        let abandoned_attempt = collector_state
            .as_ref()
            .and_then(|(_, abandoned, ..)| abandoned.clone());
        if nak_is_stale(
            latched_attempt.as_deref(),
            abandoned_attempt.as_deref(),
            &nak.attempt_id,
        ) {
            telemetry::record_generation_event(
                telemetry::GenerationEvent::Nak,
                telemetry::GenerationEventReason::StaleAttempt,
                telemetry::GenerationEventOutcome::Dropped,
            );
            debug!(
                request_id = %nak.request_id,
                nak_attempt_id = %nak.attempt_id,
                "dropping stale NAK from an abandoned attempt"
            );
            return;
        }

        let reason: &'static str = match nak.reason.as_str() {
            "kv_budget" => "nak_kv_budget",
            "model_not_loaded" => "nak_model_not_loaded",
            "worker_shutting_down" => "nak_worker_shutting_down",
            other => {
                // Surface unknown reasons via a warn so a future worker
                // adding a new reason without a matching gateway update is
                // visible in logs. Production degrades gracefully via the
                // `"nak"` catch-all bucket (bounded cardinality). We do
                // NOT `debug_assert!` here: a newer worker emitting a
                // not-yet-known reason is a legitimate forward-compat
                // scenario, and the assert would crash debug/test builds
                // for any non-empty unknown reason rather than degrade.
                tracing::warn!(
                    request_id = %nak.request_id,
                    reason = %other,
                    "unknown NAK reason — bucketing as `nak`"
                );
                "nak"
            }
        };
        match self
            .republish_to_pool_outcome_with_abandoned_attempt(
                &nak.request_id,
                reason,
                Some(nak.attempt_id.as_str()),
            )
            .await
        {
            Ok(RepublishOutcome::Republished) => debug!(
                request_id = %nak.request_id,
                reason = %nak.reason,
                "NAK observed, republished to pool"
            ),
            Ok(RepublishOutcome::AlreadyRepublished) => {
                // Re-read under the collector lock before deciding. A
                // first-chunk-timeout fallback can set `republished` before
                // any worker attempt id has latched; in that window, a late
                // NAK from the abandoned direct attempt is indistinguishable
                // from a NAK emitted by the pool successor. Failing the
                // request here would recreate #1601's spurious 429. Once a
                // current attempt has latched, we can classify the NAK
                // precisely and keep the immediate 429 for a live attempt
                // that really NAKs after fallback.
                let Some(decision) =
                    self.pending_streams
                        .get_mut(&nak.request_id)
                        .map(|mut entry| {
                            let decision = already_republished_nak_decision(
                                entry.current_attempt_id.as_deref(),
                                entry.abandoned_attempt_id.as_deref(),
                                &nak.attempt_id,
                            );
                            if matches!(decision, AlreadyRepublishedNakDecision::WaitForSuccessor) {
                                entry.record_abandoned_attempt_id(&nak.attempt_id);
                            }
                            decision
                        })
                else {
                    debug!(
                        request_id = %nak.request_id,
                        reason = %nak.reason,
                        "NAK on already-republished request after collector completed"
                    );
                    return;
                };
                match decision {
                    AlreadyRepublishedNakDecision::DropStale => {
                        telemetry::record_generation_event(
                            telemetry::GenerationEvent::Nak,
                            telemetry::GenerationEventReason::StaleAttempt,
                            telemetry::GenerationEventOutcome::Dropped,
                        );
                        debug!(
                            request_id = %nak.request_id,
                            nak_attempt_id = %nak.attempt_id,
                            "dropping stale NAK from an abandoned attempt"
                        );
                        return;
                    }
                    AlreadyRepublishedNakDecision::WaitForSuccessor => {
                        debug!(
                            request_id = %nak.request_id,
                            nak_attempt_id = %nak.attempt_id,
                            reason = %nak.reason,
                            "ignoring ambiguous NAK on already-republished request before successor attempt latched"
                        );
                        return;
                    }
                    AlreadyRepublishedNakDecision::Fail => {}
                }
                warn!(
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK on already-republished request — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "KV cache saturated and request already retried on the pool",
                );
            }
            Ok(RepublishOutcome::NotPossible) => debug!(
                request_id = %nak.request_id,
                "NAK observed but no republish performed (no collector or payload)"
            ),
            Ok(RepublishOutcome::RateLimited) => {
                // H9 — only the `first_chunk_timeout` reason path
                // exercises the bucket; the NAK path always supplies
                // a NAK-reason string and never trips this arm. Log
                // defensively in case a future refactor routes a NAK
                // through the bucket, so the request isn't silently
                // stranded.
                warn!(
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK republish unexpectedly rate-limited — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "fallback rate limit reached during NAK republish",
                );
            }
            Err(e) => {
                // Pool republish failed for an already-NAKed request:
                // both the direct-dispatched worker and the pool target
                // are unable to service the request. Surface a typed
                // ``rate_limit_exceeded`` outcome so the HTTP handler
                // returns 429 + Retry-After immediately instead of
                // waiting out the first-chunk timeout.
                warn!(
                    error = %e,
                    request_id = %nak.request_id,
                    reason = %nak.reason,
                    "NAK republish to pool failed — surfacing 429 to client"
                );
                self.fail_pending_stream(
                    &nak.request_id,
                    "rate_limit_exceeded",
                    "KV cache saturated and pool republish failed",
                );
            }
        }
    }

    /// Apply a streaming chunk envelope to the per-request collector.
    /// On terminal chunks, fire the outcome sender and remove the entry.
    async fn handle_chunk(&self, chunk: ChunkEnvelope) {
        let request_id = chunk.request_id.clone();
        let applied = {
            let mut entry = match self.pending_streams.get_mut(&request_id) {
                Some(e) => e,
                None => {
                    debug!(request_id = %request_id, "stream chunk for unknown request");
                    return;
                }
            };
            entry.value_mut().apply(chunk)
        };

        match applied {
            ChunkApplied::Terminal => {
                if let Some((_, mut collector)) = self.pending_streams.remove(&request_id) {
                    let outcome = collector.build_outcome();
                    // This remove-on-terminal branch is the single completion
                    // authority shared by native generate, chat, completions,
                    // responses, and SSE. Record performance here—not in the
                    // HTTP surface handlers—so one worker terminal can never
                    // double-emit TTFT, TPOT, or token accounting.
                    if let Some(completed) = outcome.as_ref().filter(|completed| {
                        completed.error.is_none()
                            && !matches!(completed.finish_reason.as_str(), "cancelled" | "error")
                    }) {
                        telemetry::record_generation_completion(
                            telemetry::GenerationCompletionObservation {
                                ttft_ms: completed.ttft_ms,
                                tpot_ms: completed.tpot_ms,
                                prompt_tokens: completed
                                    .usage
                                    .as_ref()
                                    .map(|usage| u64::from(usage.prompt_tokens)),
                                completion_tokens: completed
                                    .usage
                                    .as_ref()
                                    .map(|usage| u64::from(usage.completion_tokens)),
                            },
                        );
                    }
                    if let (Some(sender), Some(outcome)) = (collector.sender.take(), outcome) {
                        if sender.send(outcome).is_err() {
                            debug!(
                                request_id = %request_id,
                                "terminal outcome receiver dropped (client likely disconnected)"
                            );
                        }
                    }
                    // Stream finished — drop any offloaded generate blob now
                    // (the periodic reconcile is only the failure-path
                    // backstop). Queued, not awaited: this funnel runs on an
                    // inbox worker, and an object-store round trip here would
                    // hold up every other request sharing that shard.
                    self.defer_cleanup(CleanupWork::Generate(request_id.clone()))
                        .await;
                }
            }
            ChunkApplied::SeqGap => {
                // H6: a per-attempt seq gap means a required content
                // chunk was lost on the worker → gateway transport.
                // Mirror the worker's no-silent-drop guarantee on the
                // gateway side: fail the pending stream with
                // ``transport_failure`` so the client sees an explicit
                // error rather than a silently shortened completion.
                warn!(
                    request_id = %request_id,
                    "streaming seq gap detected — failing pending stream as transport_failure"
                );
                self.fail_pending_stream(
                    &request_id,
                    "transport_failure",
                    "streaming chunk sequence gap (missing chunk between worker and gateway)",
                );
            }
            ChunkApplied::Delta | ChunkApplied::Stale | ChunkApplied::Duplicate => {}
        }
    }

    /// Drain pending result collectors on graceful shutdown.
    /// Waits up to `timeout` for in-flight results to arrive, then drops the rest.
    pub async fn drain_pending(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        let poll_interval = Duration::from_millis(100);

        loop {
            if self.pending_results.is_empty() && self.pending_streams.is_empty() {
                info!("all pending queue results drained");
                return;
            }
            let count = self.pending_results.len() + self.pending_streams.len();
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
        self.cleanup_pending_streams().await;
    }

    /// Clean up expired result collectors.
    pub async fn cleanup_expired(&self) {
        let now = Instant::now();

        // Reconcile retained payload-delete failures for completed/failed
        // non-streaming requests. Live requests stay protected by their
        // ``pending_results`` entry; once that entry is gone, only the exact
        // recorded object-store keys are retried.
        let orphaned_payloads = collect_orphaned_offloaded_payload_requests(
            &self.offloaded_payload_keys,
            |request_id| self.pending_results.contains_key(request_id),
        );
        for request_id in orphaned_payloads {
            self.cleanup_offloaded_payloads(&request_id).await;
        }

        let expired: Vec<String> = self
            .pending_results
            .iter()
            .filter(|entry| now > entry.value().deadline)
            .map(|entry| entry.key().clone())
            .collect();

        for key in &expired {
            if let Some((_, collector)) = self.pending_results.remove(key) {
                warn!(request_id = %key, "result collector timed out");
                // Expiry is abandonment, never successful partial completion.
                // Dropping the sender makes the handler's receiver resolve as
                // closed; publishing the tombstone here makes this sweep an
                // equal owner in the race with the handler timeout/drop guard.
                drop(collector);
                self.finish_abandoned_work(key).await;
            } else {
                // The handler or inbox completion won the collector-removal
                // race. Exact-key cleanup is idempotent and remains a useful
                // retry for a prior object-store delete failure.
                self.cleanup_offloaded_payloads(key).await;
            }
        }

        // Reconcile offloaded generate blobs: any tracked request whose stream
        // is no longer pending has terminated (success cleans up at the
        // terminal funnel; this is the backstop for failure/timeout paths,
        // including the sync ``fail_pending_stream`` which can't await).
        //
        // Check BOTH in-flight maps: ``publish_generate`` is shared with the
        // batch generate arm (``publish_work``), which tracks requests in
        // ``pending_results`` rather than ``pending_streams``. Excluding only
        // ``pending_streams`` would let this sweep delete a batch request's
        // offloaded blob mid-flight and break redelivery. (That arm is dead
        // today — all generate goes through the streaming path — but this keeps
        // the reconcile correct if it's ever wired up.)
        let orphaned: Vec<String> = self
            .offloaded_streams
            .iter()
            .filter(|id| {
                !self.pending_streams.contains_key(id.key())
                    && !self.pending_results.contains_key(id.key())
            })
            .map(|id| id.key().clone())
            .collect();
        for request_id in orphaned {
            self.cleanup_offloaded_generate(&request_id).await;
        }
    }

    /// Delete an offloaded generate blob (``{request_id}_0.bin``) and drop its
    /// tracking entry. Best-effort + idempotent: a no-op when the request had
    /// no offload, so it is safe to call on every stream termination.
    async fn cleanup_offloaded_generate(&self, request_id: &str) {
        // No-op for non-offloaded requests.
        if !self.offloaded_streams.contains(request_id) {
            return;
        }
        let key = format!("{request_id}_0.bin");
        // Delete the blob FIRST and only drop the tracker on success, so a
        // transient delete failure leaves the entry for the next periodic
        // reconcile to retry instead of leaking the blob.
        match self.payload_store.delete(&key).await {
            Ok(()) => {
                self.offloaded_streams.remove(request_id);
            }
            Err(e) => {
                warn!(
                    key = %key,
                    error = %e,
                    "failed to delete offloaded generate payload; will retry on next reconcile"
                );
            }
        }
    }

    async fn cleanup_pending_streams(&self) {
        let expired: Vec<String> = self
            .pending_streams
            .iter()
            .map(|entry| entry.key().clone())
            .collect();

        for key in expired {
            if let Some((_, mut collector)) = self.pending_streams.remove(&key) {
                warn!(request_id = %key, "stream collector timed out");
                let outcome = collector.build_outcome().unwrap_or_else(|| StreamOutcome {
                    text: String::new(),
                    finish_reason: "error".to_string(),
                    usage: None,
                    attempt_id: collector.current_attempt_id.clone().unwrap_or_default(),
                    ttft_ms: None,
                    tpot_ms: None,
                    error: Some(ChunkError {
                        code: "shutdown".to_string(),
                        message: "gateway shutdown before stream completed".to_string(),
                        param: None,
                        retry_after_s: None,
                    }),
                    origin: StreamOutcomeOrigin::GatewaySynthetic,
                    tool_calls: None,
                    logprobs: None,
                    candidates: Vec::new(),
                    executed_bundle_config_hash: None,
                    execution_identity_sha256: None,
                    execution_binding_sha256: None,
                });
                if let Some(sender) = collector.sender.take() {
                    if sender.send(outcome).is_err() {
                        debug!(
                            request_id = %key,
                            "shutdown-drain receiver dropped (client likely disconnected)"
                        );
                    }
                }
                // Drop any offloaded blob for this shutdown-dropped stream. The
                // periodic reconcile filters against ``pending_streams``, so an
                // entry removed here would otherwise be missed at shutdown.
                self.cleanup_offloaded_generate(&key).await;
            }
        }
    }

    /// Remove offloaded payloads for a completed/expired request. No-op when the
    /// request offloaded nothing, and exact-key cleanup for mixed requests that
    /// offloaded only a subset of possible payload blobs.
    async fn cleanup_offloaded_payloads(&self, request_id: &str) {
        cleanup_offloaded_payloads_inner(
            self.payload_store.as_ref(),
            &self.offloaded_payload_keys,
            request_id,
        )
        .await;
    }
}

fn collect_orphaned_offloaded_payload_requests(
    offloaded: &DashMap<String, BTreeSet<String>>,
    is_pending: impl Fn(&str) -> bool,
) -> Vec<String> {
    offloaded
        .iter()
        .filter(|entry| !is_pending(entry.key()))
        .map(|entry| entry.key().clone())
        .collect()
}

fn record_offloaded_payload_key(
    offloaded: &DashMap<String, BTreeSet<String>>,
    request_id: &str,
    key: &str,
) {
    offloaded
        .entry(request_id.to_string())
        .or_default()
        .insert(key.to_string());
}

/// Persist one payload while preserving the queue's plain-key wire contract.
/// Cloud stores return a canonical full URL for operator visibility, but the
/// publisher must queue the same safe key that cleanup later deletes.
async fn put_payload_for_plain_queue_key(
    payload_store: &dyn PayloadStore,
    key: &str,
    payload: &[u8],
) -> Result<String, String> {
    let _canonical_reference = payload_store.put(key, payload).await?;
    Ok(key.to_owned())
}

/// Delete the object-store blobs an offloaded request wrote or attempted. No-op
/// when the request offloaded nothing. Successful deletes are dropped from
/// tracking; failed deletes are retained so the periodic reconcile can retry.
/// Extracted from [`WorkPublisher::cleanup_offloaded_payloads`] so the guard is
/// unit-testable without a live JetStream context (issue #1471).
async fn cleanup_offloaded_payloads_inner(
    payload_store: &dyn PayloadStore,
    offloaded: &DashMap<String, BTreeSet<String>>,
    request_id: &str,
) {
    // Snapshot the current keys without removing their tracking entry. The
    // payload-store delete is an await point and this future is deliberately
    // bounded by callers; removing everything up front would make a timeout
    // cancellation forget the current and unvisited keys permanently.
    let Some(keys) = offloaded
        .get(request_id)
        .map(|entry| entry.iter().cloned().collect::<Vec<_>>())
    else {
        return;
    };

    for key in keys {
        match payload_store.delete(&key).await {
            Ok(()) => {
                // Retire only the key whose delete completed. Everything else
                // remains discoverable if this future is cancelled at the next
                // await, and a concurrently-recorded sibling is preserved.
                if let Some(mut tracked) = offloaded.get_mut(request_id) {
                    tracked.remove(&key);
                }
                offloaded.remove_if(request_id, |_, tracked| tracked.is_empty());
            }
            Err(e) => {
                warn!(
                    key = %key,
                    error = %e,
                    "failed to remove offloaded payload; will retry on next reconcile"
                );
            }
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

    /// Two lanes on one pool, differing only in machine profile. The pool-wide
    /// check cannot tell them apart — the whole point of B5's first half.
    fn hot_lane() -> LaneKey {
        LaneKey::new("shared", "h100", "default")
    }

    fn cold_lane() -> LaneKey {
        LaneKey::new("shared", "a10g", "default")
    }

    fn lane_control(enforce: bool) -> LaneAdmissionControl {
        LaneAdmissionControl::new(
            10,
            enforce,
            crate::state::demand_tracker::PhysicalLaneCatalog::default(),
        )
    }

    /// Fill `lane` past its ceiling and hold the reservations.
    fn saturate(control: &LaneAdmissionControl, lane: &LaneKey) -> Option<LaneReservation> {
        check_lane_backpressure(control, lane, 11).expect("first publish is admitted")
    }

    /// Enforced: the deep lane sheds and a cold lane on the SAME pool still
    /// admits.
    ///
    /// Reverting to pool-granular backpressure makes this fail: a pool-wide
    /// gate reads one `num_pending` for the whole `WORK_POOL_shared` stream,
    /// so it either admits both lanes or sheds both. There is no pool-wide
    /// arrangement of inputs under which one of these two assertions holds
    /// and the other does not.
    #[test]
    fn a_saturated_lane_sheds_only_that_lane() {
        let control = lane_control(true);
        let _held = saturate(&control, &hot_lane());

        let hot = check_lane_backpressure(&control, &hot_lane(), 1);
        let cold = check_lane_backpressure(&control, &cold_lane(), 1);

        assert!(hot.is_err(), "the saturated lane must shed");
        assert!(
            cold.is_ok(),
            "a cold lane on the same pool must still admit"
        );
    }

    /// Shadow mode records the decision and changes nothing about admission.
    #[test]
    fn shadow_mode_records_the_decision_without_changing_admission() {
        let shadow = lane_control(false);
        let _held = saturate(&shadow, &hot_lane());

        // Identical lane state to `a_saturated_lane_sheds_only_that_lane`.
        assert!(
            check_lane_backpressure(&shadow, &hot_lane(), 1).is_ok(),
            "shadow mode must not shed"
        );
        // The decision itself is still computed, and it is a shed.
        assert_eq!(
            shadow.admit(&hot_lane(), 1).outcome,
            LaneAdmissionOutcome::ShedShadow
        );
    }

    /// Flag off reproduces today's behaviour exactly: the pool-wide check is
    /// the only gate, and it is byte-for-byte the message it always was.
    #[test]
    fn the_flag_off_leaves_pool_wide_backpressure_as_the_only_gate() {
        let shadow = lane_control(false);
        let _held = saturate(&shadow, &hot_lane());
        assert!(check_lane_backpressure(&shadow, &hot_lane(), 4_096).is_ok());

        let healthy = CachedStreamInfo {
            num_pending: 10,
            num_consumers: 1,
        };
        assert_eq!(pool_backpressure_error(&healthy, 50_000), None);

        let saturated_pool = CachedStreamInfo {
            num_pending: 50_001,
            num_consumers: 1,
        };
        assert_eq!(
            pool_backpressure_error(&saturated_pool, 50_000).as_deref(),
            Some("backpressure: 50001 pending messages exceeds threshold 50000")
        );

        let no_consumers = CachedStreamInfo {
            num_pending: 0,
            num_consumers: 0,
        };
        assert_eq!(
            pool_backpressure_error(&no_consumers, 50_000).as_deref(),
            Some("no consumers available for work stream")
        );
    }

    /// The pool-wide gate is unchanged and still applies with enforcement on:
    /// per-lane admission is added in front of it, not substituted for it.
    #[test]
    fn the_pool_wide_gate_survives_enforcement() {
        let control = lane_control(true);
        assert!(check_lane_backpressure(&control, &cold_lane(), 1).is_ok());

        let saturated_pool = CachedStreamInfo {
            num_pending: 50_001,
            num_consumers: 1,
        };
        assert!(pool_backpressure_error(&saturated_pool, 50_000).is_some());
    }

    /// An enforced lane shed must classify exactly like a pool-wide shed, so
    /// it reaches the client as the same 503 + `Retry-After` and records
    /// pending demand for the saturated lane (`record_publish_failure`).
    #[test]
    fn an_enforced_lane_shed_classifies_as_backpressure() {
        let control = lane_control(true);
        let _held = saturate(&control, &hot_lane());
        let error = check_lane_backpressure(&control, &hot_lane(), 1).expect_err("shed");

        assert_eq!(
            QueuePublishOutcome::from_error(&error),
            QueuePublishOutcome::Backpressure
        );
        assert!(error.to_ascii_lowercase().contains("backpressure"));
        assert!(
            error.contains("work lane saturated"),
            "a lane shed must read differently from a pool-wide shed: {error}"
        );
        // The message reaches the client. Pool names are operator-defined and
        // may encode tenant context, so lane identity stays in the log and the
        // decision metric.
        for token in ["shared", "h100", "default"] {
            assert!(
                !error.contains(token),
                "lane identity must not reach the client body: {error}"
            );
        }
    }

    /// Releasing a request's reservation reopens its lane, and only its lane.
    #[test]
    fn releasing_a_request_reopens_its_own_lane() {
        let control = lane_control(true);
        {
            let _held = saturate(&control, &hot_lane());
            assert!(check_lane_backpressure(&control, &hot_lane(), 1).is_err());
        }
        assert_eq!(control.in_flight(&hot_lane()), 0);
        assert!(check_lane_backpressure(&control, &hot_lane(), 1).is_ok());
    }

    /// A worker-direct target and its pool fallback are the same lane, so
    /// direct dispatch is accounted too — the blind spot `check_backpressure`
    /// documents for the pool-wide count.
    #[test]
    fn worker_direct_and_pool_targets_share_one_lane() {
        let direct = PublishTarget::Worker {
            pool: "shared".to_string(),
            machine_profile: "h100".to_string(),
            bundle: "default".to_string(),
            model: "BAAI/bge-m3".to_string(),
            worker_id: "worker-1".to_string(),
        };
        let fanout = PublishTarget::Pool {
            pool: "shared".to_string(),
            machine_profile: "h100".to_string(),
            bundle: "default".to_string(),
            // A different model on the same lane shares the accounting: the
            // lane is the physical worker set, not the model.
            model: "intfloat/e5-base-v2".to_string(),
        };
        assert_eq!(direct.lane(), fanout.lane());
        assert_eq!(direct.lane(), hot_lane());
    }

    #[test]
    fn unit_counts_decodes_legacy_positional_array_without_field_shift() {
        let legacy = rmp_serde::to_vec(&(Some(11_u64), Some(2_u64), Some(3_u64), Some(4_000_u64)))
            .expect("legacy positional units");
        let decoded: UnitCounts = rmp_serde::from_slice(&legacy).expect("additive decode");

        assert_eq!(decoded.input_tokens, Some(11));
        assert_eq!(decoded.pages, Some(2));
        assert_eq!(decoded.images, Some(3));
        assert_eq!(decoded.audio_ms, Some(4_000));
        assert_eq!(decoded.pairs, None);
        assert_eq!(decoded.gpu_second, None);
        assert_eq!(decoded.output_tokens, None);
    }

    /// `output_tokens` (#2432) is additive: a worker that already stamps
    /// `pairs` and `gpu_second` positionally must keep decoding without a
    /// field shift, and the new dimension must read as absent — never as a
    /// fabricated zero the settle path would bill.
    #[test]
    fn unit_counts_output_tokens_is_wire_additive() {
        let legacy = rmp_serde::to_vec(&(
            Some(11_u64),
            Some(2_u64),
            Some(3_u64),
            Some(4_000_u64),
            Some(5_u64),
            Some(6_u64),
        ))
        .expect("legacy positional units through gpu_second");
        let decoded: UnitCounts = rmp_serde::from_slice(&legacy).expect("additive decode");

        assert_eq!(decoded.input_tokens, Some(11));
        assert_eq!(decoded.pages, Some(2));
        assert_eq!(decoded.images, Some(3));
        assert_eq!(decoded.audio_ms, Some(4_000));
        assert_eq!(decoded.pairs, Some(5));
        assert_eq!(decoded.gpu_second, Some(6));
        assert_eq!(decoded.output_tokens, None);
    }

    #[test]
    fn initial_durability_completion_preserves_the_existing_batch_contract() {
        assert_eq!(initial_publish_ack_count("encode", 0), 0);
        assert_eq!(initial_publish_ack_count("encode", 4_096), 4_096);
        assert_eq!(
            initial_publish_ack_count("score", usize::MAX),
            1,
            "score folds candidates into one published work item"
        );
        assert_eq!(
            initial_publish_ack_count("generate", usize::MAX),
            1,
            "generation publishes one work item"
        );
    }

    #[test]
    fn queue_request_item_limit_is_an_explicit_api_contract() {
        assert!(validate_queue_request_item_count(MAX_QUEUE_REQUEST_ITEMS).is_ok());
        assert_eq!(
            validate_queue_request_item_count(MAX_QUEUE_REQUEST_ITEMS + 1),
            Err(format!(
                "Queue request contains 4097 items; the maximum is {MAX_QUEUE_REQUEST_ITEMS}"
            )),
        );
    }

    // --- issue #1471: payload-store cleanup guard ---
    use async_trait::async_trait;

    /// Records every `delete` so tests can assert the cleanup guard issues no
    /// object-store DELETEs for inline (non-offloaded) requests.
    #[derive(Default)]
    struct CountingPayloadStore {
        deleted: std::sync::Mutex<Vec<String>>,
        fail_delete: bool,
    }

    #[async_trait]
    impl PayloadStore for CountingPayloadStore {
        async fn put(&self, _key: &str, _data: &[u8]) -> Result<String, String> {
            Ok(String::new())
        }
        async fn delete(&self, key: &str) -> Result<(), String> {
            self.deleted.lock().unwrap().push(key.to_string());
            if self.fail_delete {
                Err("simulated object-store delete failure".to_string())
            } else {
                Ok(())
            }
        }
    }

    struct FullReferencePayloadStore {
        full_reference: String,
        puts: std::sync::Mutex<Vec<(String, usize)>>,
        deleted: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl PayloadStore for FullReferencePayloadStore {
        async fn put(&self, key: &str, data: &[u8]) -> Result<String, String> {
            self.puts.lock().unwrap().push((key.to_owned(), data.len()));
            Ok(self.full_reference.clone())
        }

        async fn delete(&self, key: &str) -> Result<(), String> {
            self.deleted.lock().unwrap().push(key.to_owned());
            Ok(())
        }
    }

    #[tokio::test]
    async fn large_oss_payload_queues_plain_key_and_cleans_exact_key() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../wire-fixtures/oss_payload_store.json"
        ))
        .unwrap();
        let key = fixture["plain_key"].as_str().unwrap();
        let store = FullReferencePayloadStore {
            full_reference: fixture["full_reference"].as_str().unwrap().to_owned(),
            puts: std::sync::Mutex::new(Vec::new()),
            deleted: std::sync::Mutex::new(Vec::new()),
        };
        let payload = vec![0; PAYLOAD_OFFLOAD_THRESHOLD + 1];
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "request_00000000", key);

        let queued_ref = put_payload_for_plain_queue_key(&store, key, &payload)
            .await
            .unwrap();
        assert_eq!(queued_ref, key, "the queue must retain the safe plain key");
        assert_eq!(
            store.puts.lock().unwrap().as_slice(),
            &[(key.to_owned(), PAYLOAD_OFFLOAD_THRESHOLD + 1)]
        );

        cleanup_offloaded_payloads_inner(&store, &offloaded, "request_00000000").await;
        assert_eq!(store.deleted.lock().unwrap().as_slice(), &[key.to_owned()]);
        assert!(!offloaded.contains_key("request_00000000"));
    }

    // Common case (issue #1471): a small/inline request that never offloaded a
    // payload must issue zero object-store DELETEs. The unconditional
    // per-request DELETE storm is exactly what capped throughput.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_noop_when_nothing_offloaded() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-inline").await;
        let deleted = store.deleted.lock().unwrap();
        assert!(
            deleted.is_empty(),
            "inline request must not trigger payload-store deletes, got {:?}",
            *deleted,
        );
    }

    #[test]
    fn orphaned_offloaded_payload_requests_skip_live_pending_results() {
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-done", "req-done_0.bin");
        record_offloaded_payload_key(&offloaded, "req-live", "req-live_0.bin");
        let got = collect_orphaned_offloaded_payload_requests(&offloaded, |request_id| {
            request_id == "req-live"
        });
        assert_eq!(
            got,
            vec!["req-done".to_string()],
            "reconcile must retry only tracked payloads whose request is no longer pending",
        );
    }

    // A request that actually offloaded deletes only the concrete keys that
    // were recorded. This keeps mixed batches from deleting every possible
    // `{id}_{i}.bin` sibling just because one item crossed the threshold.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_deletes_only_tracked_keys() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-big", "req-big_1.bin");
        record_offloaded_payload_key(&offloaded, "req-big", "req-big_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-big").await;
        let mut got = store.deleted.lock().unwrap().clone();
        got.sort();
        assert_eq!(
            got,
            vec!["req-big_1.bin".to_string(), "req-big_score.bin".to_string(),],
            "cleanup must delete only the keys that were actually recorded",
        );
        assert!(
            !offloaded.contains_key("req-big"),
            "tracking entry must be dropped after cleanup",
        );
    }

    // A score request records only the score blob, so cleanup must not infer an
    // item blob from the request id.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_score_key_only_when_tracked() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-score", "req-score_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-score").await;
        let got = store.deleted.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["req-score_score.bin".to_string()],
            "score cleanup must delete only the tracked score key",
        );
        assert!(!offloaded.contains_key("req-score"));
    }

    // Cleanup is idempotent after a successful cleanup: a second call (both
    // terminal cleanup paths can fire) is a no-op once the first dropped the
    // tracking entry.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_second_call_is_noop() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-twice", "req-twice_0.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-twice").await;
        let after_first = store.deleted.lock().unwrap().len();
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-twice").await;
        let after_second = store.deleted.lock().unwrap().len();
        assert_eq!(
            after_first, after_second,
            "second cleanup must issue no further deletes (entry already removed)",
        );
    }

    // Direct-publish fallback can attempt the same offload key more than once
    // for the same request. Track unique keys so cleanup remains O(unique
    // object keys), not O(publish attempts).
    #[tokio::test]
    async fn cleanup_offloaded_payloads_deduplicates_republish_keys() {
        let store = CountingPayloadStore::default();
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-retry", "req-retry_0.bin");
        record_offloaded_payload_key(&offloaded, "req-retry", "req-retry_0.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-retry").await;
        let got = store.deleted.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["req-retry_0.bin".to_string()],
            "duplicate key records must produce one payload-store delete",
        );
    }

    // A failing object store must not break cleanup: every delete is attempted,
    // failed keys remain tracked, and a later successful pass drains them.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_retains_failed_deletes_for_retry() {
        let store = CountingPayloadStore {
            fail_delete: true,
            ..Default::default()
        };
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-fail", "req-fail_1.bin");
        record_offloaded_payload_key(&offloaded, "req-fail", "req-fail_score.bin");
        cleanup_offloaded_payloads_inner(&store, &offloaded, "req-fail").await;
        let mut attempted = store.deleted.lock().unwrap().clone();
        attempted.sort();
        assert_eq!(
            attempted,
            vec![
                "req-fail_1.bin".to_string(),
                "req-fail_score.bin".to_string(),
            ],
            "all deletes must be attempted even when the store errors",
        );
        {
            let retained = offloaded.get("req-fail").expect("failed keys retained");
            assert_eq!(
                retained.iter().cloned().collect::<Vec<_>>(),
                vec![
                    "req-fail_1.bin".to_string(),
                    "req-fail_score.bin".to_string(),
                ],
                "failed delete keys must remain eligible for retry",
            );
        }

        let retry_store = CountingPayloadStore::default();
        cleanup_offloaded_payloads_inner(&retry_store, &offloaded, "req-fail").await;
        let mut retry_attempted = retry_store.deleted.lock().unwrap().clone();
        retry_attempted.sort();
        assert_eq!(
            retry_attempted,
            vec![
                "req-fail_1.bin".to_string(),
                "req-fail_score.bin".to_string(),
            ],
            "retry cleanup must attempt retained keys",
        );
        assert!(!offloaded.contains_key("req-fail"));
    }

    struct BlockingSecondDeleteStore {
        deleted: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl PayloadStore for BlockingSecondDeleteStore {
        async fn put(&self, _key: &str, _data: &[u8]) -> Result<String, String> {
            Ok(String::new())
        }

        async fn delete(&self, key: &str) -> Result<(), String> {
            self.deleted.lock().unwrap().push(key.to_string());
            if key.ends_with("_1.bin") {
                std::future::pending::<()>().await;
            }
            Ok(())
        }
    }

    // Failure cleanup is bounded by the dispatch-durability monitor. If that
    // timeout cancels this future between object-store deletes, the completed
    // key may retire but the current and unvisited keys must remain tracked for
    // the periodic reconcile instead of leaking permanently.
    #[tokio::test]
    async fn cleanup_offloaded_payloads_is_cancellation_safe() {
        let store = BlockingSecondDeleteStore {
            deleted: std::sync::Mutex::new(Vec::new()),
        };
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "req-cancel", "req-cancel_0.bin");
        record_offloaded_payload_key(&offloaded, "req-cancel", "req-cancel_1.bin");
        record_offloaded_payload_key(&offloaded, "req-cancel", "req-cancel_2.bin");

        let result = tokio::time::timeout(
            Duration::from_millis(20),
            cleanup_offloaded_payloads_inner(&store, &offloaded, "req-cancel"),
        )
        .await;
        assert!(result.is_err(), "second delete must keep cleanup pending");

        let retained = offloaded.get("req-cancel").expect("retry keys retained");
        assert_eq!(
            retained.iter().cloned().collect::<Vec<_>>(),
            vec![
                "req-cancel_1.bin".to_string(),
                "req-cancel_2.bin".to_string(),
            ],
            "the in-flight and unvisited deletes must survive cancellation",
        );
    }

    // --- B4: payload cleanup must not run on the caller's task ---

    /// A payload store whose `delete` never returns, standing in for a slow
    /// object-store round trip.
    struct HangingDeleteStore {
        attempted: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl PayloadStore for HangingDeleteStore {
        async fn put(&self, _key: &str, _data: &[u8]) -> Result<String, String> {
            Ok(String::new())
        }

        async fn delete(&self, key: &str) -> Result<(), String> {
            self.attempted.lock().unwrap().push(key.to_string());
            std::future::pending::<()>().await;
            Ok(())
        }
    }

    fn hanging_cleanup_pool(
        attempted: Arc<std::sync::Mutex<Vec<String>>>,
        shards: usize,
        depth: usize,
    ) -> KeyedWorkerPool<CleanupWork> {
        let offloaded: Arc<DashMap<String, BTreeSet<String>>> = Arc::new(DashMap::new());
        KeyedWorkerPool::new("test-cleanup", shards, depth, move |work: CleanupWork| {
            let store = HangingDeleteStore {
                attempted: Arc::clone(&attempted),
            };
            let offloaded = Arc::clone(&offloaded);
            async move {
                let CleanupWork::Payloads(request_id) = work else {
                    return;
                };
                record_offloaded_payload_key(
                    &offloaded,
                    &request_id,
                    &format!("{request_id}_0.bin"),
                );
                cleanup_offloaded_payloads_inner(&store, &offloaded, &request_id).await;
            }
        })
    }

    /// The second half of B4: completing a request used to await an
    /// object-store delete on the inbox decode task, stalling decode for every
    /// other in-flight request. Handing the cleanup to the pool must return
    /// immediately even while the delete is wedged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_cleanup_does_not_await_the_object_store_delete() {
        let attempted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let pool = hanging_cleanup_pool(Arc::clone(&attempted), 1, 4);

        // A hung delete must not be observable by the caller at all: this
        // returns without ever awaiting the store.
        assert!(
            queue_cleanup(Some(&pool), CleanupWork::Payloads("req-hang".to_string())).is_none(),
            "a live cleanup pool takes ownership of the work"
        );

        // The delete really is in flight on the pool, not on this task.
        let deadline = Instant::now() + Duration::from_secs(5);
        while attempted.lock().unwrap().is_empty() {
            assert!(Instant::now() < deadline, "cleanup never reached the store");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(
            attempted.lock().unwrap().clone(),
            vec!["req-hang_0.bin".to_string()],
            "the queued cleanup must be the one that ran"
        );
    }

    /// Without a pool (a publisher that never started an inbox subscription)
    /// the work comes back so the caller still runs it inline. Cleanup is
    /// never silently skipped.
    #[tokio::test]
    async fn queue_cleanup_returns_work_when_there_is_no_pool() {
        let work = queue_cleanup(None, CleanupWork::Generate("req-inline".to_string()));
        assert!(
            matches!(work, Some(CleanupWork::Generate(id)) if id == "req-inline"),
            "no pool must hand the work back for inline execution"
        );
    }

    /// At the bound the cleanup pool sheds rather than blocking, because the
    /// periodic `cleanup_expired` reconcile is an independent retry path for
    /// exactly these keys.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_cleanup_sheds_at_the_bound_and_the_reconcile_still_deletes() {
        let attempted = Arc::new(std::sync::Mutex::new(Vec::new()));
        // One shard, depth one, and every delete hangs: one item is taken by
        // the worker, one fills the queue, everything after that is shed.
        let pool = hanging_cleanup_pool(Arc::clone(&attempted), 1, 1);
        assert!(queue_cleanup(Some(&pool), CleanupWork::Payloads("a".to_string())).is_none());
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(queue_cleanup(Some(&pool), CleanupWork::Payloads("b".to_string())).is_none());

        // Shed: taken off the caller's hands, but never enqueued. The caller
        // still does not block.
        let shed_started = Instant::now();
        assert!(queue_cleanup(Some(&pool), CleanupWork::Payloads("c".to_string())).is_none());
        assert!(
            shed_started.elapsed() < Duration::from_millis(500),
            "shedding must not block the caller"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            attempted.lock().unwrap().clone(),
            vec!["a_0.bin".to_string()],
            "the shed request must not have reached the store"
        );

        // The backstop: the shed request's keys are still tracked, and the
        // reconcile path deletes them on its next sweep.
        let offloaded: DashMap<String, BTreeSet<String>> = DashMap::new();
        record_offloaded_payload_key(&offloaded, "c", "c_0.bin");
        let orphaned = collect_orphaned_offloaded_payload_requests(&offloaded, |_request_id| false);
        assert_eq!(
            orphaned,
            vec!["c".to_string()],
            "a shed cleanup must still be discoverable by the periodic reconcile"
        );
        let store = CountingPayloadStore::default();
        cleanup_offloaded_payloads_inner(&store, &offloaded, "c").await;
        assert_eq!(
            store.deleted.lock().unwrap().clone(),
            vec!["c_0.bin".to_string()],
            "the reconcile must delete what the bound shed"
        );
    }

    /// Live end-to-end proof of both B4 moves against the real inbox loop.
    ///
    /// One request completes with an object-store delete that never returns.
    /// Before this change that delete was awaited on the single decode task,
    /// so every other in-flight request in the process stopped: the second
    /// request's chunks would never be handled and its outcome would never
    /// arrive. Now the cleanup is queued and decode keeps running, so the
    /// second request completes — with its chunks in order.
    ///
    /// NATS-gated like the other live tests here: a no-op pass when
    /// `NATS_URL` is unset so the hermetic run stays green.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inbox_decode_survives_a_wedged_cleanup_and_keeps_stream_order() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(c)) => c,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = async_nats::jetstream::new(client.clone());
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let router_id = format!("b4router{nanos}");

        let attempted = Arc::new(std::sync::Mutex::new(Vec::new()));
        let publisher = Arc::new(WorkPublisher::new(
            context,
            router_id.clone(),
            Arc::new(HangingDeleteStore {
                attempted: Arc::clone(&attempted),
            }),
            Duration::from_secs(30),
            1024,
            WorkStreamConfig {
                max_age: Duration::from_secs(300),
                storage: jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));
        publisher
            .start_inbox_subscription(&client)
            .await
            .expect("start inbox subscription");

        // `wedged` completes first and its cleanup delete never returns.
        //
        // Both requests are deliberately placed on the SAME inbox shard, so
        // the live request is strictly serialized behind the wedged request's
        // handler. Sharding alone therefore cannot rescue it: it completes
        // only because that handler declines to await the object-store
        // delete. This is the assertion the cleanup move owns.
        let wedged = format!("wedged-{nanos}");
        let shard = crate::queue::keyed_worker_pool::shard_index(&wedged, INBOX_POOL_SHARDS);
        let live = (0..1_000)
            .map(|n| format!("live-{nanos}-{n}"))
            .find(|id| crate::queue::keyed_worker_pool::shard_index(id, INBOX_POOL_SHARDS) == shard)
            .expect("a live request id sharing the wedged request's shard");
        let (wedged_tx, _wedged_rx) = oneshot::channel();
        publisher.pending_streams.insert(
            wedged.clone(),
            StreamCollector::new(wedged_tx, "m".to_string(), "default".to_string()),
        );
        publisher.offloaded_streams.insert(wedged.clone());

        // `live` must still complete, in order, behind it.
        let (live_tx, live_rx) = oneshot::channel();
        publisher.pending_streams.insert(
            live.clone(),
            StreamCollector::new(live_tx, "m".to_string(), "default".to_string()),
        );

        let subject = format!("_INBOX.{router_id}.reply");
        let terminal = |request_id: &str, seq: u32| {
            rmp_serde::to_vec_named(&WireChunk {
                kind: "chunk",
                request_id,
                seq,
                text_delta: "",
                done: true,
            })
            .expect("encode terminal")
        };

        // Wedge the cleanup path first.
        client
            .publish(subject.clone(), terminal(&wedged, 0).into())
            .await
            .expect("publish wedged terminal");

        // Then a long interleaved stream for the live request.
        const CHUNKS: u32 = 64;
        for seq in 0..CHUNKS {
            client
                .publish(subject.clone(), wire_chunk(&live, seq).into())
                .await
                .expect("publish live chunk");
        }
        client
            .publish(subject.clone(), terminal(&live, CHUNKS).into())
            .await
            .expect("publish live terminal");
        client.flush().await.expect("flush inbox publishes");

        let outcome = tokio::time::timeout(Duration::from_secs(10), live_rx)
            .await
            .expect("a wedged cleanup must not stall decode for other requests")
            .expect("live outcome sender must not be dropped");

        // Ordering: `StreamCollector::apply` drops a chunk whose seq is not
        // greater than the watermark and fails the stream on a forward gap, so
        // any reordering shows up as short/failed text here.
        assert_eq!(
            outcome.text,
            "tok".repeat(CHUNKS as usize),
            "every chunk must be applied exactly once, in seq order"
        );
        assert!(
            outcome.error.is_none(),
            "no seq gap may be observed: {:?}",
            outcome.error
        );

        // And the wedged cleanup really did start — off the decode task.
        assert_eq!(
            attempted.lock().unwrap().clone(),
            vec![format!("{wedged}_0.bin")],
            "the wedged request's delete must be in flight on the cleanup pool"
        );
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct WorkItem {
        pub work_item_id: String,
        pub request_id: String,
        pub item_index: u32,
        pub total_items: u32,
        pub operation: String,
        pub model_id: String,
        #[serde(default)]
        pub profile_id: String,
        #[serde(default)]
        pub engine: String,
        pub pool_name: String,
        #[serde(default, skip_serializing_if = "String::is_empty")]
        pub admission_pool: String,
        pub machine_profile: String,
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
        #[serde(default)]
        pub query_item: Option<rmpv::Value>,
        #[serde(default)]
        pub query_payload_ref: Option<String>,
        #[serde(default)]
        pub score_items: Option<Vec<rmpv::Value>>,
        #[serde(default)]
        pub labels: Option<Vec<String>>,
        #[serde(default)]
        pub output_schema: Option<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub generate: Option<GenerateParams>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub routing_key: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prompt_cache_key: Option<String>,
        #[serde(default)]
        pub bundle_config_hash: String,
        #[serde(default)]
        pub router_id: String,
        pub reply_subject: String,
        #[serde(default)]
        pub timestamp: f64,
        #[serde(default)]
        pub accepts_result_chunks: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub traceparent: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub tracestate: Option<String>,
    }

    #[test]
    fn test_stream_name() {
        assert_eq!(stream_name("default"), "WORK_POOL_default");
        assert_eq!(stream_name("eval-l4"), "WORK_POOL_eval-l4");
    }

    #[test]
    fn test_canonical_stream_subjects_replaces_legacy_subject() {
        let observed = vec!["sie.work.*.default".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn test_canonical_stream_subjects_drops_extra_subjects() {
        let observed = vec![
            "sie.work.*.default".to_string(),
            "sie.work.other.*.*.*".to_string(),
        ];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn test_canonical_stream_subjects_noops_when_exact() {
        let observed = vec!["sie.work.default.*.*.*".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            None
        );
    }

    /// Live JetStream reconcile test for the work-stream discard policy
    /// (architecture-review finding B3). A stream created by a pre-fix
    /// gateway carries the async-nats default `DiscardPolicy::Old`, which
    /// silently drops the oldest queued work at the `max_messages` cap;
    /// `ensure_stream` must repair it to `New` (and create fresh streams
    /// with `New`) so a full queue rejects the publish onto the loud 503
    /// path instead. NATS-gated like the other live tests here: a no-op
    /// pass when `NATS_URL` is unset so the hermetic run stays green.
    #[tokio::test]
    async fn test_ensure_stream_reconciles_discard_policy_to_new() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(c)) => c,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = async_nats::jetstream::new(client);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let stream_max_age = Duration::from_secs(300);

        // Pre-create the pool stream exactly as a pre-fix gateway did:
        // no `discard` field, so the async-nats default (`Old`) sticks.
        let legacy_pool = format!("discardold{nanos}");
        let legacy_stream_name = stream_name(&legacy_pool);
        let mut legacy_stream = context
            .get_or_create_stream(jetstream::stream::Config {
                name: legacy_stream_name.clone(),
                subjects: vec![format!("sie.work.{legacy_pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: stream_max_age,
                max_messages: 100_000,
                ..Default::default()
            })
            .await
            .expect("create legacy work stream");
        assert_eq!(
            legacy_stream
                .info()
                .await
                .expect("legacy stream info")
                .config
                .discard,
            jetstream::stream::DiscardPolicy::Old,
            "precondition: the legacy creation path must yield the Old default"
        );

        let publisher = Arc::new(WorkPublisher::new(
            context.clone(),
            "it-router".to_string(),
            Arc::new(crate::queue::payload_store::DisabledPayloadStore),
            Duration::from_secs(5),
            1024,
            WorkStreamConfig {
                max_age: stream_max_age,
                storage: jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));
        publisher
            .ensure_stream(&legacy_pool)
            .await
            .expect("ensure_stream over the legacy stream");
        let mut reconciled = context
            .get_stream(&legacy_stream_name)
            .await
            .expect("get reconciled stream");
        assert_eq!(
            reconciled
                .info()
                .await
                .expect("reconciled stream info")
                .config
                .discard,
            jetstream::stream::DiscardPolicy::New,
            "ensure_stream must repair a pre-existing stream's discard policy to New"
        );

        // Fresh-create path: a pool with no pre-existing stream must get
        // `New` at creation time.
        let fresh_pool = format!("discardnew{nanos}");
        publisher
            .ensure_stream(&fresh_pool)
            .await
            .expect("ensure_stream fresh create");
        let mut fresh = context
            .get_stream(stream_name(&fresh_pool))
            .await
            .expect("get fresh stream");
        assert_eq!(
            fresh
                .info()
                .await
                .expect("fresh stream info")
                .config
                .discard,
            jetstream::stream::DiscardPolicy::New,
            "ensure_stream must create fresh streams with discard New"
        );

        context
            .delete_stream(&legacy_stream_name)
            .await
            .expect("delete legacy test stream");
        context
            .delete_stream(stream_name(&fresh_pool))
            .await
            .expect("delete fresh test stream");
    }

    /// Live JetStream contract test for the work-stream durability knobs
    /// (architecture-review finding B1). Pins three things:
    ///
    /// 1. `SIE_STREAM_STORAGE` / `SIE_STREAM_REPLICAS` reach the created
    ///    stream, so `file` actually produces a file-backed stream.
    /// 2. NATS REJECTS an in-place storage-type change. This is the reason
    ///    `ensure_stream` warns instead of reconciling storage, so the
    ///    behaviour is asserted rather than assumed.
    /// 3. `num_replicas` IS reconciled on an existing stream (the one knob
    ///    of the two that JetStream will change in place).
    ///
    /// NATS-gated like the other live tests here.
    #[tokio::test]
    async fn test_ensure_stream_storage_is_configurable_but_not_reconcilable() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(c)) => c,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = async_nats::jetstream::new(client);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let stream_max_age = Duration::from_secs(300);

        // A file-backed publisher must create a file-backed stream.
        let file_pool = format!("storagefile{nanos}");
        let file_publisher = Arc::new(WorkPublisher::new(
            context.clone(),
            "it-router".to_string(),
            Arc::new(crate::queue::payload_store::DisabledPayloadStore),
            Duration::from_secs(5),
            1024,
            WorkStreamConfig {
                max_age: stream_max_age,
                storage: jetstream::stream::StorageType::File,
                num_replicas: 1,
            },
        ));
        file_publisher
            .ensure_stream(&file_pool)
            .await
            .expect("ensure_stream with File storage");
        let mut file_stream = context
            .get_stream(stream_name(&file_pool))
            .await
            .expect("get file-backed stream");
        let file_config = file_stream
            .info()
            .await
            .expect("file-backed stream info")
            .config
            .clone();
        assert_eq!(
            file_config.storage,
            jetstream::stream::StorageType::File,
            "SIE_STREAM_STORAGE=file must produce a file-backed stream"
        );

        // The core finding: JetStream refuses to change storage in place.
        let mut wants_memory = file_config.clone();
        wants_memory.storage = jetstream::stream::StorageType::Memory;
        let update_err = context.update_stream(wants_memory).await.err();
        // NATS 2.12 answers this with err_code 10052,
        // "stream configuration update can not change storage type".
        assert!(
            update_err.is_some(),
            "JetStream is expected to reject an in-place storage-type change; \
             if this ever starts succeeding, ensure_stream can reconcile storage \
             instead of only warning"
        );
        eprintln!("in-place storage change rejected by NATS: {update_err:?}");

        // So a publisher configured for Memory over that same File stream
        // must NOT fail and must NOT mutate it — it warns and moves on.
        let memory_publisher = Arc::new(WorkPublisher::new(
            context.clone(),
            "it-router".to_string(),
            Arc::new(crate::queue::payload_store::DisabledPayloadStore),
            Duration::from_secs(5),
            1024,
            WorkStreamConfig {
                max_age: stream_max_age,
                storage: jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));
        memory_publisher
            .ensure_stream(&file_pool)
            .await
            .expect("a storage mismatch must not fail ensure_stream");
        assert_eq!(
            context
                .get_stream(stream_name(&file_pool))
                .await
                .expect("re-get stream")
                .info()
                .await
                .expect("stream info after mismatch")
                .config
                .storage,
            jetstream::stream::StorageType::File,
            "ensure_stream must leave a diverging stream's storage untouched"
        );

        // Replicas, unlike storage, ARE reconciled. A single-node NATS caps
        // R at 1, so only assert the no-op direction there; the reconcile
        // path itself is exercised by the R1 -> R1 equality short-circuit.
        assert_eq!(
            file_config.num_replicas, 1,
            "default replica count must stay 1"
        );

        context
            .delete_stream(stream_name(&file_pool))
            .await
            .expect("delete file-backed test stream");
    }

    fn result_collector(
        total_items: usize,
    ) -> (ResultCollector, oneshot::Receiver<Vec<WorkResult>>) {
        result_collector_with_budget(
            total_items,
            Arc::new(ResultChunkBudget::new(
                MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
            )),
            MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
        )
    }

    fn result_collector_with_budget(
        total_items: usize,
        result_chunk_budget: Arc<ResultChunkBudget>,
        result_chunk_reserved_limit: usize,
    ) -> (ResultCollector, oneshot::Receiver<Vec<WorkResult>>) {
        let (tx, rx) = oneshot::channel();
        (
            ResultCollector {
                _total_items: total_items as u32,
                results: vec![None; total_items],
                result_chunk_transfers: BTreeMap::new(),
                result_chunk_buffered_bytes: 0,
                result_chunk_reserved_bytes: 0,
                result_chunk_reserved_limit,
                result_chunk_budget,
                sender: Some(tx),
                deadline: Instant::now() + Duration::from_secs(30),
                operation: "encode".to_string(),
                pool_fallback_subject: None,
                direct_fallback_worker_id: None,
                direct_fallback_republished_indices: BTreeSet::new(),
                direct_fallback_payloads: vec![None; total_items],
                direct_fallback_republished: false,
                _lane_reservation: None,
            },
            rx,
        )
    }

    #[test]
    fn abandonment_has_one_removal_winner_and_releases_reservations() {
        let budget = Arc::new(ResultChunkBudget::new(8192));
        let (mut collector, mut rx) = result_collector_with_budget(1, Arc::clone(&budget), 8192);
        let reserved = collector.reserve_result_chunk_transfer(64).unwrap();
        collector.results[0] = Some(successful_result("req", 0, vec![1]));
        assert!(reserved > 0);
        let pending = Arc::new(DashMap::new());
        pending.insert("req".to_string(), collector);
        let wins = std::thread::scope(|scope| {
            let first = scope.spawn(|| drop_pending_result_collector(&pending, "req"));
            let second = scope.spawn(|| drop_pending_result_collector(&pending, "req"));
            usize::from(first.join().unwrap()) + usize::from(second.join().unwrap())
        });
        assert_eq!(wins, 1);
        assert!(!pending.contains_key("req"));
        assert!(matches!(
            rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Closed)
        ));
        assert_eq!(budget.current(), 0);
    }

    fn successful_result(request_id: &str, item_index: u32, payload: Vec<u8>) -> WorkResult {
        WorkResult {
            work_item_id: canonical_work_item_id(request_id, item_index),
            request_id: request_id.to_string(),
            item_index,
            success: true,
            result_msgpack: payload,
            error: None,
            error_code: None,
            inference_ms: Some(1.0),
            queue_ms: None,
            processing_ms: None,
            worker_id: Some("worker-1".to_string()),
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    fn result_chunks(result: &WorkResult) -> Vec<ResultChunkV1> {
        result_chunks_with_count(result, 3)
    }

    fn result_chunks_with_count(result: &WorkResult, chunk_count: u32) -> Vec<ResultChunkV1> {
        let encoded = rmp_serde::to_vec_named(result).expect("encode WorkResult");
        let digest = Sha256::digest(&encoded).to_vec();
        (0..chunk_count)
            .map(|chunk_index| {
                let start = encoded.len() * chunk_index as usize / chunk_count as usize;
                let end = encoded.len() * (chunk_index as usize + 1) / chunk_count as usize;
                ResultChunkV1 {
                    kind: RESULT_CHUNK_KIND.to_string(),
                    work_item_id: result.work_item_id.clone(),
                    request_id: result.request_id.clone(),
                    item_index: result.item_index,
                    transfer_digest: digest.clone(),
                    chunk_index,
                    chunk_count,
                    total_bytes: encoded.len() as u64,
                    payload: encoded[start..end].to_vec(),
                }
            })
            .collect()
    }

    fn complete_chunked_transfer(
        collector: &mut ResultCollector,
        chunks: Vec<ResultChunkV1>,
    ) -> CompletedResultTransfer {
        for chunk in chunks {
            let applied = apply_result_chunk(collector, chunk).expect("valid result chunk");
            if let ResultChunkStatus::Complete(completed) = applied.status {
                return completed;
            }
        }
        panic!("all chunks must complete the result transfer");
    }

    fn complete_chunked_result(
        collector: &mut ResultCollector,
        chunks: Vec<ResultChunkV1>,
    ) -> WorkResult {
        decode_completed_result(complete_chunked_transfer(collector, chunks))
            .expect("decode completed WorkResult")
    }

    fn assert_result_chunk_rejection_terminates(
        mut collector: ResultCollector,
        mut rx: oneshot::Receiver<Vec<WorkResult>>,
        chunk: ResultChunkV1,
        expected: ResultChunkReject,
    ) {
        let request_id = chunk.request_id.clone();
        assert_eq!(
            apply_result_chunk(&mut collector, chunk).unwrap_err(),
            expected
        );
        let pending = DashMap::new();
        pending.insert(request_id.clone(), collector);
        assert!(fail_pending_result_chunk_request(&pending, &request_id).is_some());
        let results = rx.try_recv().expect("terminal result sent immediately");
        assert!(!results.is_empty());
        assert!(results.iter().all(|result| {
            !result.success
                && result.error.as_deref() == Some("Worker result transport validation failed")
                && result.error_code.as_deref() == Some("transport_failure")
        }));
    }

    #[test]
    fn result_chunk_named_msgpack_uses_binary_digest_and_payload() {
        let chunk = result_chunks(&successful_result("req", 0, vec![1, 2, 3])).remove(0);
        let encoded = rmp_serde::to_vec_named(&chunk).expect("encode chunk");
        let value: rmpv::Value = rmp_serde::from_slice(&encoded).expect("decode value");
        assert_eq!(envelope_kind(&value), Some(RESULT_CHUNK_KIND));
        let rmpv::Value::Map(fields) = &value else {
            panic!("chunk must be a named map");
        };
        for field in ["transfer_digest", "payload"] {
            let value = fields
                .iter()
                .find_map(|(key, value)| (key.as_str() == Some(field)).then_some(value))
                .expect("binary field");
            assert!(matches!(value, rmpv::Value::Binary(_)), "{field}");
        }
        let decoded: ResultChunkV1 = rmp_serde::from_slice(&encoded).expect("chunk round trip");
        assert_eq!(decoded, chunk);
    }

    #[test]
    fn malformed_exact_v1_is_not_accepted_as_a_work_result() {
        let chunk = result_chunks(&successful_result("req", 0, vec![1])).remove(0);
        let encoded = rmp_serde::to_vec_named(&chunk).unwrap();
        let mut value: rmpv::Value = rmp_serde::from_slice(&encoded).unwrap();
        let rmpv::Value::Map(fields) = &mut value else {
            panic!("chunk must be a map");
        };
        fields.retain(|(key, _)| key.as_str() != Some("payload"));

        assert_eq!(envelope_kind(&value), Some(RESULT_CHUNK_KIND));
        assert!(rmpv::ext::from_value::<ResultChunkV1>(value).is_err());
    }

    #[test]
    fn work_result_unknown_total_bytes_field_remains_forward_compatible() {
        let expected = successful_result("req", 0, vec![1, 2, 3]);
        let encoded = rmp_serde::to_vec_named(&expected).unwrap();
        let mut value: rmpv::Value = rmp_serde::from_slice(&encoded).unwrap();
        let rmpv::Value::Map(fields) = &mut value else {
            panic!("named WorkResult must be a map");
        };
        fields.push((
            rmpv::Value::String("total_bytes".into()),
            rmpv::Value::from(123_u64),
        ));

        assert_eq!(envelope_kind(&value), None);
        let decoded: WorkResult = rmpv::ext::from_value(value).unwrap();
        assert_eq!(decoded.work_item_id, expected.work_item_id);
        assert_eq!(decoded.result_msgpack, expected.result_msgpack);
    }

    #[test]
    fn result_chunk_reassembles_out_of_order_and_decodes_legacy_work_result() {
        let expected = successful_result("req", 0, vec![7, 8, 9]);
        let chunks = result_chunks(&expected);
        let (mut collector, _rx) = result_collector(1);

        for index in [2, 0] {
            let applied = apply_result_chunk(&mut collector, chunks[index].clone()).unwrap();
            assert!(matches!(applied.status, ResultChunkStatus::Buffered));
        }
        let applied = apply_result_chunk(&mut collector, chunks[1].clone()).unwrap();
        let ResultChunkStatus::Complete(completed) = applied.status else {
            panic!("last missing chunk must complete the transfer");
        };
        let decoded = decode_completed_result(completed).expect("decode reconstructed result");
        assert_eq!(decoded.work_item_id, expected.work_item_id);
        assert_eq!(decoded.result_msgpack, expected.result_msgpack);
        assert_eq!(collector.result_chunk_buffered_bytes, 0);
        assert!(collector.result_chunk_transfers.is_empty());

        // Rolling compatibility: a legacy one-shot WorkResult is not routed to
        // the chunk decoder and remains directly deserializable.
        let legacy = rmp_serde::to_vec_named(&expected).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&legacy).unwrap();
        let decoded: WorkResult = rmpv::ext::from_value(value).unwrap();
        assert_eq!(decoded.work_item_id, expected.work_item_id);
    }

    #[test]
    fn result_chunk_duplicate_is_idempotent_but_conflict_is_rejected() {
        let chunks = result_chunks(&successful_result("req", 0, vec![1]));
        let (mut collector, _rx) = result_collector(1);
        apply_result_chunk(&mut collector, chunks[0].clone()).unwrap();
        let buffered = collector.result_chunk_buffered_bytes;

        let duplicate = apply_result_chunk(&mut collector, chunks[0].clone()).unwrap();
        assert!(matches!(duplicate.status, ResultChunkStatus::Duplicate));
        assert_eq!(collector.result_chunk_buffered_bytes, buffered);

        let mut conflicting = chunks[0].clone();
        conflicting.payload[0] ^= 0xff;
        assert_eq!(
            apply_result_chunk(&mut collector, conflicting).unwrap_err(),
            ResultChunkReject::DuplicateConflict
        );
        assert_eq!(collector.result_chunk_buffered_bytes, buffered);
    }

    #[test]
    fn result_chunk_rejects_metadata_changes_and_replaces_new_digest_retry() {
        let old = result_chunks(&successful_result("req", 0, vec![1]));
        let replacement_result = successful_result("req", 0, vec![2, 3, 4]);
        let replacement = result_chunks(&replacement_result);
        let (mut collector, _rx) = result_collector(1);
        apply_result_chunk(&mut collector, old[0].clone()).unwrap();

        let mut changed_metadata = old[1].clone();
        changed_metadata.total_bytes += 1;
        assert_eq!(
            apply_result_chunk(&mut collector, changed_metadata).unwrap_err(),
            ResultChunkReject::MetadataConflict
        );

        let applied = apply_result_chunk(&mut collector, replacement[0].clone()).unwrap();
        assert!(applied.retry_replaced);
        assert_eq!(
            collector.result_chunk_buffered_bytes,
            replacement[0].payload.len()
        );

        let active_bytes = collector.result_chunk_buffered_bytes;
        for delayed in [old[1].clone(), old[0].clone()] {
            let ignored = apply_result_chunk(&mut collector, delayed).unwrap();
            assert!(matches!(ignored.status, ResultChunkStatus::StaleRetry));
            assert_eq!(collector.result_chunk_buffered_bytes, active_bytes);
        }
        let mut unknown_nonzero = replacement[1].clone();
        unknown_nonzero.transfer_digest = vec![0x55; 32];
        let ignored = apply_result_chunk(&mut collector, unknown_nonzero).unwrap();
        assert!(matches!(ignored.status, ResultChunkStatus::StaleRetry));
        assert_eq!(collector.result_chunk_buffered_bytes, active_bytes);

        apply_result_chunk(&mut collector, replacement[2].clone()).unwrap();
        let applied = apply_result_chunk(&mut collector, replacement[1].clone()).unwrap();
        let ResultChunkStatus::Complete(completed) = applied.status else {
            panic!("replacement transfer must complete");
        };
        let decoded = decode_completed_result(completed).unwrap();
        assert_eq!(decoded.result_msgpack, replacement_result.result_msgpack);
    }

    #[test]
    fn result_chunk_same_digest_chunk_zero_restarts_layout_without_mixing() {
        let expected = successful_result("req", 0, vec![1, 2, 3, 4]);
        let old_layout = result_chunks_with_count(&expected, 3);
        let replacement_layout = result_chunks_with_count(&expected, 4);
        let (mut collector, _rx) = result_collector(1);

        apply_result_chunk(&mut collector, old_layout[0].clone()).unwrap();
        let replaced = apply_result_chunk(&mut collector, replacement_layout[0].clone()).unwrap();
        assert!(replaced.retry_replaced);

        for delayed in [old_layout[1].clone(), old_layout[0].clone()] {
            let ignored = apply_result_chunk(&mut collector, delayed).unwrap();
            assert!(matches!(ignored.status, ResultChunkStatus::StaleRetry));
        }

        let decoded = complete_chunked_result(&mut collector, replacement_layout[1..].to_vec());
        assert_eq!(decoded.result_msgpack, expected.result_msgpack);
    }

    #[test]
    fn completed_chunked_items_remain_reserved_until_request_completion() {
        let first = successful_result("req", 0, vec![1]);
        let second = successful_result("req", 1, vec![2]);
        let first_chunks = result_chunks(&first);
        let second_chunks = result_chunks(&second);
        let first_reservation =
            result_chunk_reservation_bytes(first_chunks[0].total_bytes as usize).unwrap();
        let second_reservation =
            result_chunk_reservation_bytes(second_chunks[0].total_bytes as usize).unwrap();
        let budget = Arc::new(ResultChunkBudget::new(
            MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
        ));
        let (mut collector, _rx) = result_collector_with_budget(
            2,
            Arc::clone(&budget),
            first_reservation + second_reservation - 1,
        );

        let decoded = complete_chunked_result(&mut collector, first_chunks);
        assert_eq!(
            store_result_if_missing(&mut collector, decoded),
            StoreResultOutcome::Stored
        );
        assert_eq!(collector.result_chunk_reserved_bytes, first_reservation);
        assert_eq!(budget.current(), first_reservation);

        assert_eq!(
            apply_result_chunk(&mut collector, second_chunks[0].clone()).unwrap_err(),
            ResultChunkReject::AggregateSize
        );
        assert_eq!(collector.result_chunk_reserved_bytes, first_reservation);
        drop(collector);
        assert_eq!(budget.current(), 0);
    }

    #[test]
    fn completed_stale_direct_chunk_releases_reservation_for_replacement() {
        let mut stale_direct = successful_result("req", 0, vec![1]);
        stale_direct.worker_direct = true;
        let replacement = successful_result("req", 0, vec![2]);
        let stale_chunks = result_chunks(&stale_direct);
        let replacement_chunks = result_chunks(&replacement);
        let reservation =
            result_chunk_reservation_bytes(stale_chunks[0].total_bytes as usize).unwrap();
        assert_eq!(
            reservation,
            result_chunk_reservation_bytes(replacement_chunks[0].total_bytes as usize).unwrap()
        );

        let budget = Arc::new(ResultChunkBudget::new(reservation));
        let (mut collector, _rx) =
            result_collector_with_budget(1, Arc::clone(&budget), reservation);
        mark_direct_fallback_republished_indices(&mut collector, [0]);

        let completed = complete_chunked_transfer(&mut collector, stale_chunks);
        assert_eq!(completed.reservation_bytes, reservation);
        assert_eq!(collector.result_chunk_reserved_bytes, reservation);
        assert_eq!(budget.current(), reservation);
        let completed_reservation = completed.reservation_bytes;
        let decoded = decode_completed_result(completed).unwrap();
        assert_eq!(
            store_result_with_completed_chunk_reservation(
                &mut collector,
                decoded,
                completed_reservation,
            ),
            StoreResultOutcome::StaleDirectFallback
        );
        assert!(collector.results[0].is_none());
        assert_eq!(collector.result_chunk_reserved_bytes, 0);
        assert_eq!(budget.current(), 0);

        let completed = complete_chunked_transfer(&mut collector, replacement_chunks);
        assert_eq!(completed.reservation_bytes, reservation);
        let completed_reservation = completed.reservation_bytes;
        let decoded = decode_completed_result(completed).unwrap();
        assert_eq!(
            store_result_with_completed_chunk_reservation(
                &mut collector,
                decoded,
                completed_reservation,
            ),
            StoreResultOutcome::Stored
        );
        assert_eq!(collector.result_chunk_reserved_bytes, reservation);
        assert_eq!(budget.current(), reservation);
        assert_eq!(
            collector.results[0]
                .as_ref()
                .expect("replacement stored")
                .result_msgpack,
            replacement.result_msgpack
        );

        drop(collector);
        assert_eq!(budget.current(), 0);
    }

    #[test]
    fn gateway_wide_chunk_budget_rejects_then_raii_cleanup_restores_capacity() {
        let first = successful_result("req-a", 0, vec![1]);
        let second = successful_result("req-b", 0, vec![2]);
        let first_chunks = result_chunks(&first);
        let second_chunks = result_chunks(&second);
        let reservation =
            result_chunk_reservation_bytes(first_chunks[0].total_bytes as usize).unwrap();
        assert_eq!(
            reservation,
            result_chunk_reservation_bytes(second_chunks[0].total_bytes as usize).unwrap()
        );
        let budget = Arc::new(ResultChunkBudget::new(reservation));
        let (mut first_collector, _first_rx) = result_collector_with_budget(
            1,
            Arc::clone(&budget),
            MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
        );
        let (mut second_collector, _second_rx) = result_collector_with_budget(
            1,
            Arc::clone(&budget),
            MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
        );

        let decoded = complete_chunked_result(&mut first_collector, first_chunks);
        assert_eq!(
            store_result_if_missing(&mut first_collector, decoded),
            StoreResultOutcome::Stored
        );
        assert_eq!(budget.current(), reservation);
        assert_eq!(
            apply_result_chunk(&mut second_collector, second_chunks[0].clone()).unwrap_err(),
            ResultChunkReject::GlobalBudget
        );

        drop(first_collector);
        assert_eq!(budget.current(), 0);
        apply_result_chunk(&mut second_collector, second_chunks[0].clone()).unwrap();
        assert_eq!(budget.current(), reservation);
        drop(second_collector);
        assert_eq!(budget.current(), 0);
    }

    #[test]
    fn result_chunk_retired_retry_digests_are_bounded() {
        let mut chunk = result_chunks(&successful_result("req", 0, vec![1]))[0].clone();
        let (mut collector, _rx) = result_collector(1);
        apply_result_chunk(&mut collector, chunk.clone()).unwrap();

        for generation in 1..=(MAX_RETIRED_RESULT_CHUNK_DIGESTS + 4) {
            chunk.transfer_digest = vec![generation as u8; 32];
            let applied = apply_result_chunk(&mut collector, chunk.clone()).unwrap();
            assert!(applied.retry_replaced);
        }
        let active = collector
            .result_chunk_transfers
            .get("req.0")
            .expect("active transfer");
        assert_eq!(
            active.retired_digests.len(),
            MAX_RETIRED_RESULT_CHUNK_DIGESTS
        );
        assert_eq!(
            active.transfer_digest,
            [(MAX_RETIRED_RESULT_CHUNK_DIGESTS + 4) as u8; 32]
        );
    }

    #[test]
    fn result_chunk_enforces_count_item_request_digest_and_length_limits() {
        let valid = result_chunks(&successful_result("req", 0, vec![1]));

        let (collector, rx) = result_collector(1);
        let mut bad_kind = valid[0].clone();
        bad_kind.kind = "result_chunk_v2".to_string();
        assert_result_chunk_rejection_terminates(collector, rx, bad_kind, ResultChunkReject::Kind);

        let (collector, rx) = result_collector(1);
        let mut too_many = valid[0].clone();
        too_many.chunk_count = MAX_RESULT_CHUNKS_PER_ITEM + 1;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            too_many,
            ResultChunkReject::ChunkCount,
        );

        let (collector, rx) = result_collector(1);
        let mut bad_index = valid[0].clone();
        bad_index.chunk_index = bad_index.chunk_count;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            bad_index,
            ResultChunkReject::ChunkIndex,
        );

        let (collector, rx) = result_collector(1);
        let mut too_large = valid[0].clone();
        too_large.total_bytes = MAX_RESULT_CHUNK_ITEM_BYTES as u64 + 1;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            too_large,
            ResultChunkReject::ItemSize,
        );

        let (collector, rx) = result_collector(1);
        let mut oversized_payload = valid[0].clone();
        oversized_payload.total_bytes = oversized_payload.payload.len() as u64 - 1;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            oversized_payload,
            ResultChunkReject::PayloadSize,
        );

        let reservation = result_chunk_reservation_bytes(valid[0].total_bytes as usize).unwrap();
        let (collector, rx) = result_collector_with_budget(
            1,
            Arc::new(ResultChunkBudget::new(
                MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
            )),
            reservation - 1,
        );
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            valid[0].clone(),
            ResultChunkReject::AggregateSize,
        );

        let (collector, rx) = result_collector(1);
        let mut bad_digest_shape = valid[0].clone();
        bad_digest_shape.transfer_digest.pop();
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            bad_digest_shape,
            ResultChunkReject::Digest,
        );

        let encoded = rmp_serde::to_vec_named(&successful_result("req", 0, vec![1])).unwrap();
        let (collector, rx) = result_collector(1);
        let bad_digest = ResultChunkV1 {
            kind: RESULT_CHUNK_KIND.to_string(),
            work_item_id: "req.0".to_string(),
            request_id: "req".to_string(),
            item_index: 0,
            transfer_digest: vec![0; 32],
            chunk_index: 0,
            chunk_count: 1,
            total_bytes: encoded.len() as u64,
            payload: encoded.clone(),
        };
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            bad_digest,
            ResultChunkReject::DigestMismatch,
        );

        let (collector, rx) = result_collector(1);
        let bad_length = ResultChunkV1 {
            kind: RESULT_CHUNK_KIND.to_string(),
            work_item_id: "req.0".to_string(),
            request_id: "req".to_string(),
            item_index: 0,
            transfer_digest: Sha256::digest(&encoded).to_vec(),
            chunk_index: 0,
            chunk_count: 1,
            total_bytes: encoded.len() as u64 + 1,
            payload: encoded,
        };
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            bad_length,
            ResultChunkReject::TotalMismatch,
        );

        let conflict_chunks = result_chunks(&successful_result("req", 0, vec![9]));
        let (mut collector, rx) = result_collector(1);
        apply_result_chunk(&mut collector, conflict_chunks[0].clone()).unwrap();
        let mut conflicting_duplicate = conflict_chunks[0].clone();
        conflicting_duplicate.payload[0] ^= 0xff;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            conflicting_duplicate,
            ResultChunkReject::DuplicateConflict,
        );

        let (mut collector, rx) = result_collector(1);
        apply_result_chunk(&mut collector, conflict_chunks[0].clone()).unwrap();
        let mut conflicting_metadata = conflict_chunks[1].clone();
        conflicting_metadata.total_bytes += 1;
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            conflicting_metadata,
            ResultChunkReject::MetadataConflict,
        );

        // The inbox is an existing trusted-worker boundary: a malformed chunk
        // that names a live request fails closed, but its attacker-controlled
        // identity/detail is replaced by the static transport error above.
        let (collector, rx) = result_collector(1);
        let mut bad_identity = valid[0].clone();
        bad_identity.work_item_id = "attacker-controlled".to_string();
        assert_result_chunk_rejection_terminates(
            collector,
            rx,
            bad_identity,
            ResultChunkReject::Identity,
        );
    }

    #[test]
    fn invalid_result_chunk_completes_pending_request_with_safe_transport_failure() {
        let pending = DashMap::new();
        let (mut collector, mut rx) = result_collector(2);
        // Prove partial buffers and already-completed items do not survive the
        // terminal funnel or produce a partial-success response.
        let chunks = result_chunks(&successful_result("req", 0, vec![1]));
        apply_result_chunk(&mut collector, chunks[0].clone()).unwrap();
        collector.results[1] = Some(successful_result("req", 1, vec![2]));
        pending.insert("req".to_string(), collector);

        let terminal = fail_pending_result_chunk_request(&pending, "req");
        assert!(terminal.is_some());
        assert!(!pending.contains_key("req"));
        let results = rx.try_recv().expect("terminal result sent immediately");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|result| !result.success));
        assert!(results.iter().all(|result| {
            result.error.as_deref() == Some("Worker result transport validation failed")
                && result.error_code.as_deref() == Some("transport_failure")
        }));
    }

    #[test]
    fn test_direct_batch_fallback_payloads_only_include_missing_items_once() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = ResultCollector {
            _total_items: 3,
            results: vec![
                None,
                Some(WorkResult {
                    work_item_id: "req.1".into(),
                    request_id: "req".into(),
                    item_index: 1,
                    success: true,
                    result_msgpack: Vec::new(),
                    error: None,
                    error_code: None,
                    inference_ms: None,
                    queue_ms: None,
                    processing_ms: None,
                    worker_id: None,
                    tokenization_ms: None,
                    postprocessing_ms: None,
                    payload_fetch_ms: None,
                    units: None,
                    worker_direct: false,
                    executed_bundle_config_hash: None,
                    execution_identity_sha256: None,
                    execution_binding_sha256: None,
                }),
                None,
            ],
            result_chunk_transfers: BTreeMap::new(),
            result_chunk_buffered_bytes: 0,
            result_chunk_reserved_bytes: 0,
            result_chunk_reserved_limit: MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
            result_chunk_budget: Arc::new(ResultChunkBudget::new(
                MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
            )),
            sender: Some(tx),
            deadline: Instant::now() + Duration::from_secs(30),
            operation: "encode".into(),
            pool_fallback_subject: Some("sie.work.default.l4.default.BAAI__bge-m3".into()),
            direct_fallback_worker_id: Some("worker-1".into()),
            direct_fallback_republished_indices: BTreeSet::new(),
            direct_fallback_payloads: vec![Some(vec![0]), Some(vec![1]), Some(vec![2])],
            direct_fallback_republished: false,
            _lane_reservation: None,
        };

        let (subject, worker_id, payloads) =
            take_direct_fallback_payloads(&mut collector).expect("fallback payloads");
        assert_eq!(subject, "sie.work.default.l4.default.BAAI__bge-m3");
        assert_eq!(worker_id.as_deref(), Some("worker-1"));
        assert_eq!(payloads, vec![(0, vec![0]), (2, vec![2])]);
        assert!(collector.direct_fallback_republished);
        assert!(take_direct_fallback_payloads(&mut collector).is_none());
    }

    #[test]
    fn test_batch_result_collector_keeps_first_result_for_duplicate_item() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = ResultCollector {
            _total_items: 1,
            results: vec![None],
            result_chunk_transfers: BTreeMap::new(),
            result_chunk_buffered_bytes: 0,
            result_chunk_reserved_bytes: 0,
            result_chunk_reserved_limit: MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
            result_chunk_budget: Arc::new(ResultChunkBudget::new(
                MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
            )),
            sender: Some(tx),
            deadline: Instant::now() + Duration::from_secs(30),
            operation: "encode".into(),
            pool_fallback_subject: None,
            direct_fallback_worker_id: None,
            direct_fallback_republished_indices: BTreeSet::new(),
            direct_fallback_payloads: vec![None],
            direct_fallback_republished: false,
            _lane_reservation: None,
        };

        let first = WorkResult {
            work_item_id: "req.0".into(),
            request_id: "req".into(),
            item_index: 0,
            success: true,
            result_msgpack: vec![1],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: Some("direct-worker".into()),
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let duplicate = WorkResult {
            result_msgpack: vec![2],
            worker_id: Some("pool-worker".into()),
            ..first.clone()
        };

        assert_eq!(
            store_result_if_missing(&mut collector, first),
            StoreResultOutcome::Stored
        );
        assert_eq!(
            store_result_if_missing(&mut collector, duplicate),
            StoreResultOutcome::Duplicate
        );
        let stored = collector.results[0].as_ref().expect("stored result");
        assert_eq!(stored.result_msgpack, vec![1]);
        assert_eq!(stored.worker_id.as_deref(), Some("direct-worker"));
    }

    #[test]
    fn test_batch_result_collector_ignores_stale_direct_after_fallback_confirmed() {
        let (tx, _rx) = oneshot::channel();
        let mut collector = ResultCollector {
            _total_items: 1,
            results: vec![None],
            result_chunk_transfers: BTreeMap::new(),
            result_chunk_buffered_bytes: 0,
            result_chunk_reserved_bytes: 0,
            result_chunk_reserved_limit: MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST,
            result_chunk_budget: Arc::new(ResultChunkBudget::new(
                MAX_RESULT_CHUNK_RESERVED_BYTES_GLOBAL,
            )),
            sender: Some(tx),
            deadline: Instant::now() + Duration::from_secs(30),
            operation: "encode".into(),
            pool_fallback_subject: Some("sie.work.default.l4.default.BAAI__bge-m3".into()),
            direct_fallback_worker_id: Some("direct-worker".into()),
            direct_fallback_republished_indices: BTreeSet::from([0]),
            direct_fallback_payloads: vec![Some(vec![0])],
            direct_fallback_republished: true,
            _lane_reservation: None,
        };

        let stale_direct = WorkResult {
            work_item_id: "req.0".into(),
            request_id: "req".into(),
            item_index: 0,
            success: true,
            result_msgpack: vec![1],
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: Some("direct-worker".into()),
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: true,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let pool_result = WorkResult {
            result_msgpack: vec![2],
            worker_id: Some("pool-worker".into()),
            units: None,
            worker_direct: false,
            ..stale_direct.clone()
        };

        assert_eq!(
            store_result_if_missing(&mut collector, stale_direct),
            StoreResultOutcome::StaleDirectFallback
        );
        assert!(collector.results[0].is_none());
        assert_eq!(
            store_result_if_missing(&mut collector, pool_result),
            StoreResultOutcome::Stored
        );
        let stored = collector.results[0].as_ref().expect("stored result");
        assert_eq!(stored.result_msgpack, vec![2]);
        assert_eq!(stored.worker_id.as_deref(), Some("pool-worker"));
    }

    #[test]
    fn test_pending_generation_snapshot_groups_collectors() {
        let now = Instant::now();
        let mut grouped: BTreeMap<(String, String, String), PendingGenerationGroup> =
            BTreeMap::new();

        let (tx1, _rx1) = oneshot::channel();
        let mut waiting = StreamCollector::new(tx1, "model-a:no-spec".into(), "default".into());
        waiting.display_model = "model-a".into();
        waiting.published_at = now.checked_sub(Duration::from_millis(250)).unwrap_or(now);
        accumulate_pending_generation_group(&mut grouped, &waiting, now);

        let (tx2, _rx2) = oneshot::channel();
        let mut active = StreamCollector::new(tx2, "model-a:no-spec".into(), "default".into());
        active.display_model = "model-a".into();
        active.first_chunk_at = Some(now);
        active.republished = true;
        active.published_at = now.checked_sub(Duration::from_millis(100)).unwrap_or(now);
        accumulate_pending_generation_group(&mut grouped, &active, now);

        let (tx3, _rx3) = oneshot::channel();
        let other = StreamCollector::new(tx3, "model-b".into(), "l4".into());
        accumulate_pending_generation_group(&mut grouped, &other, now);

        let groups: Vec<PendingGenerationGroup> = grouped.into_values().collect();
        let snapshot = PendingGenerationSnapshot {
            total: groups.iter().map(|group| group.count).sum(),
            groups,
        };

        assert_eq!(snapshot.total, 3);
        let model_a = snapshot.for_model("model-a");
        assert_eq!(model_a.total, 2);
        assert_eq!(model_a.groups.len(), 1);
        assert_eq!(model_a.groups[0].count, 2);
        assert_eq!(model_a.groups[0].waiting_first_chunk, 1);
        assert_eq!(model_a.groups[0].active_streams, 1);
        assert_eq!(model_a.groups[0].republished, 1);
        assert!(model_a.groups[0].oldest_request_age_ms >= 250);

        let model_b = snapshot.for_model("model-b");
        assert_eq!(model_b.total, 1);
        assert_eq!(model_b.groups[0].pool, "l4");
    }

    #[test]
    fn client_disconnect_latches_before_the_stream_collector_is_dropped() {
        let pending = DashMap::new();
        let (tx, _rx) = oneshot::channel();
        let collector = StreamCollector::new(tx, "model-a".into(), "default".into());
        let disconnected = collector.client_disconnected_handle();
        pending.insert("req-disconnect".to_string(), collector);

        assert!(drop_disconnected_stream_collector(
            &pending,
            "req-disconnect"
        ));
        assert!(disconnected.load(Ordering::Acquire));
        assert!(!pending.contains_key("req-disconnect"));
        assert!(!drop_disconnected_stream_collector(
            &pending,
            "req-disconnect"
        ));
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

    /// Cross-language fixture for the worker_id normalization contract.
    ///
    /// Mirrors `packages/sie_sdk/tests/test_worker_id_normalization.py` —
    /// both languages share the same NATS subject and the two sides MUST
    /// produce byte-identical tokens or direct-dispatch silently misses
    /// (workstream G-M5).
    ///
    /// Worker IDs flow through `normalize_model_id` (same function as the
    /// model id, since both targets are single NATS subject tokens with
    /// the same legality rules). Case is **preserved** intentionally —
    /// NATS subjects are case-sensitive and operator-set worker names
    /// commonly include mixed case; lowercasing would surprise operators
    /// and force a cross-language migration with no benefit.
    #[test]
    fn test_worker_id_normalization_cross_language() {
        // unchanged: clean ascii with hyphens
        assert_eq!(normalize_model_id("worker-1"), "worker-1");
        // case preserved (NOT lowercased — see doc comment above)
        assert_eq!(normalize_model_id("Worker-1"), "Worker-1");
        assert_eq!(normalize_model_id("WORKER"), "WORKER");
        // dotted Kubernetes pod hostname → each dot → "_dot_"
        assert_eq!(
            normalize_model_id("sie-worker-7d9f-default-0.sie-worker.default.svc"),
            "sie-worker-7d9f-default-0_dot_sie-worker_dot_default_dot_svc"
        );
        // whitespace → "_"
        assert_eq!(normalize_model_id("my worker"), "my_worker");
        // wildcard tokens → "_". Each `.` → `_dot_` and `*` → `_`, so
        // `worker.*.foo` = `worker` + `_dot_` + `_` + `_dot_` + `foo`.
        assert_eq!(normalize_model_id("worker.*.foo"), "worker_dot___dot_foo");
        // leading/trailing whitespace is preserved as `_` (we do NOT trim —
        // Python helper rejects whitespace-only ids upstream; non-empty
        // padding is mapped through the same scrub).
        assert_eq!(normalize_model_id("  worker-1  "), "__worker-1__");
        // consecutive separators are NOT collapsed (no benefit; would
        // diverge from Python and break this fixture)
        assert_eq!(normalize_model_id("worker--1"), "worker--1");
    }

    #[test]
    fn test_work_subject() {
        assert_eq!(
            work_subject("default", "rtx6000", "default", "BAAI/bge-m3"),
            "sie.work.default.rtx6000.default.BAAI__bge-m3"
        );
        assert_eq!(
            work_subject("eval-l4", "l4-spot", "sglang", "my-model"),
            "sie.work.eval-l4.l4-spot.sglang.my-model"
        );
        assert_eq!(
            work_subject_worker(
                "default",
                "rtx6000",
                "sglang",
                "Qwen/Qwen3.6-27B",
                "worker-0.svc"
            ),
            "sie.work.default.rtx6000.sglang.Qwen__Qwen3_dot_6-27B.worker-0_dot_svc"
        );
    }

    /// Regression: model IDs containing `.` must produce exactly 6
    /// subject tokens (`sie`, `work`, `{pool}`, `{machine_profile}`,
    /// `{bundle}`, `{normalized_model}`) so the worker's consumer filter
    /// `sie.work.{pool}.{machine_profile}.{bundle}.*` matches them.
    #[test]
    fn test_work_subject_token_count_with_dotted_model() {
        let subj = work_subject("default", "l4", "default", "vidore/colqwen2.5-v0.2");
        let tokens: Vec<&str> = subj.split('.').collect();
        assert_eq!(
            tokens.len(),
            6,
            "subject {subj} must have 6 tokens to match sie.work.{{pool}}.{{machine_profile}}.{{bundle}}.*"
        );
        assert_eq!(tokens[0], "sie");
        assert_eq!(tokens[1], "work");
        assert_eq!(tokens[2], "default");
        assert_eq!(tokens[3], "l4");
        assert_eq!(tokens[4], "default");
        // Token[5] contains no '.' (it's the normalized model id).
        assert!(!tokens[5].contains('.'));
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
            engine: "pytorch".to_string(),
            pool_name: "default".to_string(),
            admission_pool: "default".to_string(),
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
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "abc123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.r1.req-1".to_string(),
            timestamp: 1700000000.0,
            accepts_result_chunks: true,
            traceparent: None,
            tracestate: None,
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
        assert_eq!(decoded.admission_pool, "default");
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
        assert_eq!(decoded.engine, "pytorch");
        assert!(decoded.accepts_result_chunks);
    }

    #[test]
    fn test_work_item_engine_back_compat() {
        let mut without_engine = serde_json::Map::new();
        without_engine.insert("work_item_id".into(), "req-1.0".into());
        without_engine.insert("request_id".into(), "req-1".into());
        without_engine.insert("item_index".into(), 0.into());
        without_engine.insert("total_items".into(), 1.into());
        without_engine.insert("operation".into(), "encode".into());
        without_engine.insert("model_id".into(), "m".into());
        without_engine.insert("pool_name".into(), "default".into());
        without_engine.insert("machine_profile".into(), "".into());
        without_engine.insert("reply_subject".into(), "_INBOX.x.y".into());
        let encoded = rmp_serde::to_vec_named(&without_engine).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded.engine, "");
        assert_eq!(decoded.operation, "encode");
        assert!(!decoded.accepts_result_chunks);
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
            engine: "pytorch".to_string(),
            pool_name: "default".to_string(),
            admission_pool: "tenant".to_string(),
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
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: "hash123".to_string(),
            router_id: "router-1".to_string(),
            reply_subject: "_INBOX.router-1.req-ref".to_string(),
            timestamp: 1_700_000_000.5,
            accepts_result_chunks: true,
            traceparent: None,
            tracestate: None,
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
            engine: &owned.engine,
            pool_name: &owned.pool_name,
            admission_pool: &owned.admission_pool,
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
            generate: None,
            routing_key: owned.routing_key.as_deref(),
            prompt_cache_key: owned.prompt_cache_key.as_deref(),
            bundle_config_hash: &owned.bundle_config_hash,
            router_id: &owned.router_id,
            reply_subject: &owned.reply_subject,
            timestamp: owned.timestamp,
            accepts_result_chunks: true,
            traceparent: owned.traceparent.as_deref(),
            tracestate: owned.tracestate.as_deref(),
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
            units: Some(UnitCounts {
                input_tokens: None,
                pairs: None,
                pages: None,
                images: None,
                audio_ms: Some(1_001),
                gpu_second: None,
                output_tokens: None,
            }),
            worker_direct: true,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: Some(
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            ),
        };

        let encoded = rmp_serde::to_vec(&result).unwrap();
        let decoded: WorkResult = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.item_index, 2);
        assert!(decoded.success);
        assert_eq!(decoded.result_msgpack, vec![5, 6, 7]);
        assert_eq!(decoded.units.and_then(|units| units.audio_ms), Some(1_001));
        assert!(decoded.worker_direct);
        assert_eq!(
            decoded.execution_binding_sha256.as_deref(),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
        );
    }

    #[test]
    fn test_work_result_audio_ms_units_are_additive_and_unsigned() {
        let encoded = rmp_serde::to_vec_named(&serde_json::json!({
            "request_id": "req-audio",
            "success": true,
            "result_msgpack": [],
            "units": {"audio_ms": 12_345},
        }))
        .unwrap();
        let decoded: WorkResult = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded.units.expect("units").audio_ms, Some(12_345));

        let negative = rmp_serde::to_vec_named(&serde_json::json!({
            "request_id": "req-audio",
            "success": true,
            "result_msgpack": [],
            "units": {"audio_ms": -1},
        }))
        .unwrap();
        assert!(rmp_serde::from_slice::<WorkResult>(&negative).is_err());
    }

    #[test]
    fn test_work_result_decodes_nil_result_msgpack() {
        // The Python worker types `result_msgpack: bytes | None`
        // (sie_sdk.queue_types.WorkResult) and error results carry None —
        // msgpack `nil` on the wire. The decoder must accept it (empty
        // bytes) so a typed per-item failure never becomes an opaque
        // transport-level decode error (see deserialize_optional_bytes).
        let failure = rmpv::Value::Map(vec![
            ("work_item_id".into(), "req-9.0".into()),
            ("request_id".into(), "req-9".into()),
            ("item_index".into(), 0u32.into()),
            ("success".into(), false.into()),
            ("result_msgpack".into(), rmpv::Value::Nil),
            ("error".into(), "unsupported operation".into()),
            ("error_code".into(), "inference_error".into()),
        ]);
        let encoded = rmp_serde::to_vec_named(&failure).unwrap();
        let decoded: WorkResult = rmp_serde::from_slice(&encoded).unwrap();
        assert!(!decoded.success);
        assert!(decoded.result_msgpack.is_empty());
        assert_eq!(decoded.error_code.as_deref(), Some("inference_error"));

        // And a whole array containing a mix of success + nil-failure
        // results (the real lane-batch shape) decodes item-per-item.
        let ok = rmpv::Value::Map(vec![
            ("work_item_id".into(), "req-9.1".into()),
            ("request_id".into(), "req-9".into()),
            ("item_index".into(), 1u32.into()),
            ("success".into(), true.into()),
            ("result_msgpack".into(), rmpv::Value::Binary(vec![1, 2, 3])),
        ]);
        let encoded = rmp_serde::to_vec_named(&vec![failure, ok]).unwrap();
        let decoded: Vec<WorkResult> = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[1].result_msgpack, vec![1, 2, 3]);
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
            engine: String::new(),
            pool_name: "default".to_string(),
            admission_pool: String::new(),
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
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r1.req-2".to_string(),
            timestamp: 0.0,
            accepts_result_chunks: true,
            traceparent: None,
            tracestate: None,
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

    // --- B4: decode-side classification (`inbox_work_for_payload`) ---

    /// Minimal streaming chunk on the wire. Every other `ChunkEnvelope` field
    /// carries a serde default, so this is what a worker's smallest legal
    /// chunk decodes from.
    #[derive(Serialize)]
    struct WireChunk<'a> {
        kind: &'a str,
        request_id: &'a str,
        seq: u32,
        text_delta: &'a str,
        done: bool,
    }

    #[derive(Serialize)]
    struct WireNak<'a> {
        kind: &'a str,
        request_id: &'a str,
        reason: &'a str,
    }

    fn wire_chunk(request_id: &str, seq: u32) -> Vec<u8> {
        rmp_serde::to_vec_named(&WireChunk {
            kind: "chunk",
            request_id,
            seq,
            text_delta: "tok",
            done: false,
        })
        .expect("encode chunk")
    }

    fn wire_work_result(request_id: &str) -> Vec<u8> {
        rmp_serde::to_vec(&WorkResult {
            work_item_id: format!("{request_id}.0"),
            request_id: request_id.to_string(),
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
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        })
        .expect("encode work result")
    }

    /// The fast path is a mitigation this change must preserve: a result for a
    /// request this gateway no longer tracks is dropped without decoding.
    #[test]
    fn inbox_work_skips_untracked_requests_without_decoding() {
        assert!(
            inbox_work_for_payload(&wire_chunk("req-gone", 7), |_| false).is_none(),
            "an untracked request must be dropped on the fast path"
        );
        // Same payload, now tracked — it decodes.
        assert!(matches!(
            inbox_work_for_payload(&wire_chunk("req-gone", 7), |_| true),
            Some(InboxWork::Chunk(_))
        ));
    }

    #[test]
    fn inbox_work_routes_each_envelope_kind_to_its_handler() {
        let chunk = inbox_work_for_payload(&wire_chunk("req-a", 3), |_| true);
        match chunk {
            Some(InboxWork::Chunk(envelope)) => {
                assert_eq!(envelope.request_id, "req-a");
                assert_eq!(envelope.seq, 3);
            }
            other => panic!("expected a stream chunk, got {other:?}"),
        }

        let nak_payload = rmp_serde::to_vec_named(&WireNak {
            kind: "nak",
            request_id: "req-b",
            reason: "kv_budget",
        })
        .expect("encode nak");
        match inbox_work_for_payload(&nak_payload, |_| true) {
            Some(InboxWork::Nak(nak)) => {
                assert_eq!(nak.request_id, "req-b");
                assert_eq!(nak.reason, "kv_budget");
            }
            other => panic!("expected a nak, got {other:?}"),
        }

        let transfer = ResultChunkV1 {
            kind: RESULT_CHUNK_KIND.to_string(),
            work_item_id: "req-c.0".to_string(),
            request_id: "req-c".to_string(),
            item_index: 0,
            transfer_digest: vec![0u8; 32],
            chunk_index: 0,
            chunk_count: 1,
            total_bytes: 3,
            payload: vec![1, 2, 3],
        };
        let transfer_payload = rmp_serde::to_vec_named(&transfer).expect("encode result chunk");
        match inbox_work_for_payload(&transfer_payload, |_| true) {
            Some(InboxWork::ResultChunk(decoded)) => assert_eq!(*decoded, transfer),
            other => panic!("expected a result chunk transfer, got {other:?}"),
        }

        match inbox_work_for_payload(&wire_work_result("req-d"), |_| true) {
            Some(InboxWork::Result(result)) => assert_eq!(result.request_id, "req-d"),
            other => panic!("expected a WorkResult, got {other:?}"),
        }
    }

    /// A malformed exact-v1 transfer still has to fail its request, and that
    /// failure has to be ordered against the request's other inbox work — so
    /// it becomes a work item under the same key rather than running on the
    /// decode task.
    #[test]
    fn inbox_work_turns_a_malformed_transfer_into_keyed_failure_work() {
        // Right discriminator, wrong shape: `chunk_index` is a string.
        #[derive(Serialize)]
        struct BadTransfer<'a> {
            kind: &'a str,
            request_id: &'a str,
            chunk_index: &'a str,
        }
        let payload = rmp_serde::to_vec_named(&BadTransfer {
            kind: RESULT_CHUNK_KIND,
            request_id: "req-bad",
            chunk_index: "not-a-number",
        })
        .expect("encode malformed transfer");

        match inbox_work_for_payload(&payload, |_| true) {
            Some(work @ InboxWork::ResultChunkDecodeFailure { .. }) => {
                assert_eq!(
                    work.shard_key(),
                    "req-bad",
                    "the failure must be keyed to the failing request"
                );
            }
            other => panic!("expected keyed failure work, got {other:?}"),
        }
    }

    #[test]
    fn inbox_work_drops_undecodable_payloads() {
        assert!(inbox_work_for_payload(&[0xc1], |_| true).is_none());
    }

    /// Every variant keys on its own request id, which is what puts all of one
    /// request's inbox work on a single shard.
    #[test]
    fn inbox_work_keys_on_the_request_id() {
        let cases = [
            wire_chunk("req-key", 0),
            wire_work_result("req-key"),
            rmp_serde::to_vec_named(&WireNak {
                kind: "nak",
                request_id: "req-key",
                reason: "kv_budget",
            })
            .expect("encode nak"),
        ];
        for payload in cases {
            let work = inbox_work_for_payload(&payload, |_| true).expect("decoded work");
            assert_eq!(work.shard_key(), "req-key");
        }
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
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        assert_eq!(extracted, Some("abc-123-def"));
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
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
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
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let encoded = rmp_serde::to_vec(&result).unwrap();
        let extracted = extract_request_id_fast(&encoded);
        assert_eq!(extracted, Some(uuid_str));
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

    // ── GenerateInput wire-shape regression ──────────────────────

    /// Original prompt-only wire shape: a flat ``{prompt, max_new_tokens, ...}`` map
    /// must still decode into ``GenerateParams { input: Prompt }``. That
    /// guarantees in-flight prompt-only work items remain readable after
    /// the chat-completions surface deploys the enum.
    #[test]
    fn test_generate_params_decodes_slice02_prompt_shape() {
        let wire = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("prompt"),
                rmpv::Value::from("hello world"),
            ),
            (
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::Integer(32u64.into()),
            ),
            (rmpv::Value::from("temperature"), rmpv::Value::F64(0.7)),
        ]);
        let bytes = rmp_serde::to_vec_named(&wire).unwrap();
        let decoded: GenerateParams = rmp_serde::from_slice(&bytes).unwrap();
        match decoded.input {
            GenerateInput::Prompt { prompt } => assert_eq!(prompt, "hello world"),
            GenerateInput::Messages { .. } => panic!("expected Prompt variant"),
        }
        assert_eq!(decoded.max_new_tokens, 32);
        assert_eq!(decoded.temperature, Some(0.7));
    }

    #[test]
    fn test_generate_params_decodes_messages_shape() {
        let wire = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("messages"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![
                    (rmpv::Value::from("role"), rmpv::Value::from("user")),
                    (rmpv::Value::from("content"), rmpv::Value::from("hi")),
                ])]),
            ),
            (
                rmpv::Value::from("max_new_tokens"),
                rmpv::Value::Integer(8u64.into()),
            ),
        ]);
        let bytes = rmp_serde::to_vec_named(&wire).unwrap();
        let decoded: GenerateParams = rmp_serde::from_slice(&bytes).unwrap();
        match decoded.input {
            GenerateInput::Messages { messages } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].role, "user");
                assert_eq!(messages[0].content, "hi");
            }
            GenerateInput::Prompt { .. } => panic!("expected Messages variant"),
        }
        assert_eq!(decoded.max_new_tokens, 8);
    }

    /// CRITICAL transport invariant: a generate work item carrying images
    /// must decode as ``serde_json::Value`` — that's the type the sidecar's
    /// ``WorkItem.generate`` uses, and ``serde_json::Value`` CANNOT hold a
    /// msgpack ``bin`` (rmp_serde errors "invalid type: byte array"). Image
    /// bytes therefore travel as a base64 *string* in ``ChatImage.data``.
    /// This test pins that contract: had ``ChatImage.data`` been
    /// ``#[serde(with = "serde_bytes")]`` (bin), the serde_json decode below
    /// would fail and the feature would break through the real
    /// NATS→sidecar→worker path.
    #[test]
    fn test_chat_message_images_survive_sidecar_serde_json_value() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "what is this?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![ChatImage {
                data: "aGVsbG8=".to_string(), // base64 of b"hello"
                format: Some("png".to_string()),
            }]),
            content_parts: None,
        };
        let mp = rmp_serde::to_vec_named(&msg).unwrap();
        // (1) sidecar step: decode as serde_json::Value — MUST succeed.
        let as_value: serde_json::Value = rmp_serde::from_slice(&mp)
            .expect("ChatMessage with images must decode as serde_json::Value (no msgpack bin)");
        assert_eq!(as_value["images"][0]["data"], "aGVsbG8=");
        // (2) sidecar re-encode → worker typed decode: base64 survives.
        let mp2 = rmp_serde::to_vec_named(&as_value).unwrap();
        let back: ChatMessage = rmp_serde::from_slice(&mp2).unwrap();
        let imgs = back.images.expect("images survive round-trip");
        assert_eq!(imgs[0].data, "aGVsbG8=");
        assert_eq!(imgs[0].format.as_deref(), Some("png"));
    }

    /// ``content_parts`` (the #1294 interleaving layout) must survive the same
    /// NATS→sidecar(``serde_json::Value``)→worker round-trip, preserving the
    /// text↔image ORDER and the internally-tagged shape the worker parses
    /// (``{"type":"text","text":…}`` / ``{"type":"image"}``).
    #[test]
    fn test_content_parts_ordering_survives_sidecar_round_trip() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Page 1:Page 2:which has a cat?".to_string(),
            tool_calls: None,
            tool_call_id: None,
            images: Some(vec![
                ChatImage {
                    data: "aW1nQQ==".to_string(),
                    format: Some("png".to_string()),
                },
                ChatImage {
                    data: "aW1nQg==".to_string(),
                    format: Some("png".to_string()),
                },
            ]),
            content_parts: Some(vec![
                ContentPart::Text {
                    text: "Page 1:".to_string(),
                },
                ContentPart::Image,
                ContentPart::Text {
                    text: "Page 2:".to_string(),
                },
                ContentPart::Image,
                ContentPart::Text {
                    text: "which has a cat?".to_string(),
                },
            ]),
        };
        let mp = rmp_serde::to_vec_named(&msg).unwrap();
        // (1) sidecar step: decode as serde_json::Value — MUST succeed.
        let as_value: serde_json::Value = rmp_serde::from_slice(&mp).expect(
            "ChatMessage with content_parts must decode as serde_json::Value (no msgpack bin)",
        );
        let parts = as_value["content_parts"]
            .as_array()
            .expect("content_parts is an array");
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "Page 1:");
        assert_eq!(parts[1]["type"], "image");
        assert_eq!(parts[3]["type"], "image");
        assert_eq!(parts[4]["text"], "which has a cat?");
        // (2) sidecar re-encode → worker typed decode: order survives.
        let mp2 = rmp_serde::to_vec_named(&as_value).unwrap();
        let back: ChatMessage = rmp_serde::from_slice(&mp2).unwrap();
        let back_parts = back
            .content_parts
            .expect("content_parts survive round-trip");
        assert!(matches!(&back_parts[0], ContentPart::Text { text } if text == "Page 1:"));
        assert!(matches!(back_parts[1], ContentPart::Image));
        assert!(matches!(back_parts[3], ContentPart::Image));
        assert!(matches!(&back_parts[4], ContentPart::Text { text } if text == "which has a cat?"));
    }

    /// A round-trip through msgpack must preserve the shape — verifies
    /// that the ``flatten`` / ``untagged`` combination encodes the input
    /// arm back into the flat wire shape rather than nesting it under
    /// an ``input:`` key.
    #[test]
    fn test_generate_params_prompt_round_trips_flat() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "hi".to_string(),
            },
            max_new_tokens: 16,
            ..Default::default()
        };
        let bytes = rmp_serde::to_vec_named(&params).unwrap();
        // Decode back as raw value to assert the flat key set.
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(keys.contains(&"prompt"), "missing prompt key in {keys:?}");
        assert!(
            keys.contains(&"max_new_tokens"),
            "missing max_new_tokens in {keys:?}"
        );
        assert!(!keys.contains(&"input"), "input wrapper leaked: {keys:?}");
        assert!(!keys.contains(&"messages"), "messages leaked: {keys:?}");
    }

    #[test]
    fn test_generate_params_preserves_negative_seed_on_msgpack_wire() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "hi".to_string(),
            },
            max_new_tokens: 16,
            seed: Some(-1),
            ..Default::default()
        };
        let bytes = rmp_serde::to_vec_named(&params).unwrap();
        let decoded: GenerateParams = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.seed, Some(-1));
    }

    /// Routing-affinity fields appear on the wire when set; otherwise they
    /// are omitted entirely (``skip_serializing_if = "Option::is_none"``)
    /// so a prompt-only worker decoding the bytes still sees the same key
    /// set it expects.
    #[test]
    fn test_work_item_omits_inert_routing_fields_when_none() {
        let item = WorkItem {
            work_item_id: "req-x.0".to_string(),
            request_id: "req-x".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
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
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-x".to_string(),
            timestamp: 1.0,
            accepts_result_chunks: true,
            traceparent: None,
            tracestate: None,
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(
            !keys.contains(&"routing_key"),
            "routing_key leaked when None"
        );
        assert!(
            !keys.contains(&"prompt_cache_key"),
            "prompt_cache_key leaked when None"
        );
    }

    // ── GrammarSpec wire shape ────────────────────────────────────────

    /// Pin the JSON wire shape for ``GrammarSpec::JsonSchema`` against
    /// the Python ``sie_server.types.grammar.GrammarSpec`` dataclass.
    /// Worker-side deserialization reads ``{kind, value, label?,
    /// strict?}`` — any change here is a wire-break.
    #[test]
    fn test_grammar_spec_json_schema_round_trip() {
        let spec = GrammarSpec::JsonSchema {
            value: serde_json::json!({"type": "object", "properties": {"x": {"type": "number"}}}),
            label: Some("math_response".to_string()),
            strict: Some(true),
        };
        let encoded = serde_json::to_value(&spec).expect("serialize");
        // Wire shape check — these field names are Python-readable.
        assert_eq!(encoded["kind"], "json_schema");
        assert_eq!(encoded["value"]["type"], "object");
        assert_eq!(encoded["label"], "math_response");
        assert_eq!(encoded["strict"], true);
        let decoded: GrammarSpec = serde_json::from_value(encoded).expect("round-trip deserialize");
        assert_eq!(decoded, spec);
    }

    #[test]
    fn test_grammar_spec_regex_round_trip() {
        let spec = GrammarSpec::Regex {
            value: r"[A-Z]{3}-\d{4}".to_string(),
            label: None,
            strict: None,
        };
        let encoded = serde_json::to_value(&spec).expect("serialize");
        assert_eq!(encoded["kind"], "regex");
        assert_eq!(encoded["value"], r"[A-Z]{3}-\d{4}");
        // None fields skip-serialise so the worker doesn't see explicit
        // nulls.
        let obj = encoded.as_object().expect("object");
        assert!(!obj.contains_key("label"));
        assert!(!obj.contains_key("strict"));
        let decoded: GrammarSpec = serde_json::from_value(encoded).expect("round-trip deserialize");
        assert_eq!(decoded, spec);
    }

    /// :class:`GenerateParams` carries the grammar through the work
    /// envelope; absence must serialise as field-omitted (not ``null``)
    /// so a prompt-only worker decoding a grammar-bearing work item does not
    /// trip over an unexpected key.
    #[test]
    fn test_generate_params_omits_absent_grammar() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "Hi".to_string(),
            },
            max_new_tokens: 8,
            ..Default::default()
        };
        let v = serde_json::to_value(&params).expect("serialize");
        let obj = v.as_object().expect("object");
        assert!(
            !obj.contains_key("grammar"),
            "grammar must skip-serialise when None: {v}"
        );
    }

    #[test]
    fn test_generate_params_carries_grammar_when_present() {
        let params = GenerateParams {
            input: GenerateInput::Prompt {
                prompt: "Hi".to_string(),
            },
            max_new_tokens: 8,
            grammar: Some(GrammarSpec::Regex {
                value: r"\d+".to_string(),
                label: None,
                strict: None,
            }),
            ..Default::default()
        };
        let v = serde_json::to_value(&params).expect("serialize");
        assert_eq!(v["grammar"]["kind"], "regex");
        assert_eq!(v["grammar"]["value"], r"\d+");
    }

    // ── M5: W3C Trace Context envelope round-trip ────────────────────

    /// When the gateway has captured a `traceparent` from the inbound
    /// request, it must land on the work envelope verbatim so the
    /// worker can extract it and continue the trace. The two fields
    /// are paired in the wire shape so a single round-trip exercises
    /// both.
    #[test]
    fn test_work_item_carries_traceparent_when_set() {
        let tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        let ts = "vendor=value";
        let item = WorkItem {
            work_item_id: "req-tp.0".to_string(),
            request_id: "req-tp".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
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
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-tp".to_string(),
            timestamp: 1.0,
            accepts_result_chunks: true,
            traceparent: Some(tp.to_string()),
            tracestate: Some(ts.to_string()),
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let decoded: WorkItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.traceparent.as_deref(), Some(tp));
        assert_eq!(decoded.tracestate.as_deref(), Some(ts));
    }

    /// Backward-compat: when both trace fields are absent, the
    /// msgpack must omit them entirely (not encode `null`), so a
    /// pre-M5 worker reading the bytes sees its expected key set.
    #[test]
    fn test_work_item_omits_trace_fields_when_none() {
        let item = WorkItem {
            work_item_id: "req-tp2.0".to_string(),
            request_id: "req-tp2".to_string(),
            item_index: 0,
            total_items: 1,
            operation: "generate".to_string(),
            model_id: "m".to_string(),
            profile_id: "default".to_string(),
            engine: "pytorch".to_string(),
            pool_name: "p".to_string(),
            admission_pool: String::new(),
            machine_profile: "g".to_string(),
            item: None,
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
            bundle_config_hash: String::new(),
            router_id: String::new(),
            reply_subject: "_INBOX.r.req-tp2".to_string(),
            timestamp: 1.0,
            accepts_result_chunks: true,
            traceparent: None,
            tracestate: None,
        };
        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let value: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let map = match &value {
            rmpv::Value::Map(m) => m,
            _ => panic!("expected map"),
        };
        let keys: Vec<&str> = map
            .iter()
            .filter_map(|(k, _)| match k {
                rmpv::Value::String(s) => s.as_str(),
                _ => None,
            })
            .collect();
        assert!(
            !keys.contains(&"traceparent"),
            "traceparent leaked when None: {keys:?}"
        );
        assert!(
            !keys.contains(&"tracestate"),
            "tracestate leaked when None: {keys:?}"
        );
    }

    /// End-to-end gateway round-trip: an active span on the gateway
    /// side must yield a `traceparent` on the envelope. Uses the same
    /// `inject_current_context` helper the publisher calls in
    /// production, so this locks the integration point.
    #[test]
    fn test_inject_current_context_with_active_span_populates_envelope() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::SdkTracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_chat");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        let (tp, _ts) = crate::observability::propagation::inject_current_context();
        let tp = tp.expect("active span must produce a traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(parts[1].len(), 32, "trace_id is 32 hex chars");
        assert_eq!(parts[2].len(), 16, "span_id is 16 hex chars");
    }

    #[test]
    fn test_queue_trace_context_injects_for_non_generation() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::SdkTracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_request");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        // Non-generation endpoints now inject the active W3C context into
        // the queue work-item envelope so the worker span attaches to the
        // gateway span (issue #1500). A real 4-part traceparent proves the
        // context was actually serialized, not just left non-None.
        for endpoint in InferenceEndpoint::NON_GENERATION_QUEUE_LABELS {
            assert!(
                WorkPublisher::should_propagate_queue_trace(endpoint),
                "{endpoint} must inject queue trace context"
            );
            let (tp, _ts) = WorkPublisher::trace_context_for_endpoint(endpoint);
            let tp = tp.unwrap_or_else(|| panic!("{endpoint} should propagate traceparent"));
            let parts: Vec<&str> = tp.split('-').collect();
            assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
            assert_eq!(parts[1].len(), 32, "trace_id is 32 hex chars");
            assert_eq!(parts[2].len(), 16, "span_id is 16 hex chars");
        }
        // Unrecognized endpoints still fail closed (no envelope injection).
        let (tp, ts) = WorkPublisher::trace_context_for_endpoint("unknown");
        assert!(
            tp.is_none(),
            "unknown endpoint unexpectedly injected traceparent"
        );
        assert!(
            ts.is_none(),
            "unknown endpoint unexpectedly injected tracestate"
        );
    }

    #[test]
    fn test_queue_trace_context_propagates_for_generate() {
        use opentelemetry::trace::{TraceContextExt, Tracer, TracerProvider as _};
        use opentelemetry::Context;
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::SdkTracerProvider;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("gateway-test");
        let span = tracer.start("gateway.proxy_generate");
        let cx = Context::current().with_span(span);
        let _guard = cx.attach();

        assert!(WorkPublisher::should_propagate_queue_trace("generate"));
        let (tp, _ts) = WorkPublisher::trace_context_for_endpoint("generate");
        let tp = tp.expect("generate queue fallback should propagate traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
    }

    #[tokio::test]
    async fn test_inbound_context_continues_into_non_generation_envelope() {
        // End-to-end mechanism for #1500 on the non-generation path: the
        // gateway extracts the inbound client `traceparent`, scopes it
        // over the publish via `with_context` (no billed gateway span),
        // and the queue-publish injection re-emits it. The emitted
        // traceparent must CONTINUE the inbound trace — same trace_id and
        // parent == the inbound span_id — so the worker's run_batch span
        // attaches to the client trace instead of rooting a fresh one.
        use opentelemetry::trace::FutureExt;
        use opentelemetry_sdk::propagation::TraceContextPropagator;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

        let inbound_trace_id = "0af7651916cd43dd8448eb211c80319c";
        let inbound_span_id = "b7ad6b7169203331";
        let inbound_tp = format!("00-{inbound_trace_id}-{inbound_span_id}-01");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::HeaderName::from_static("traceparent"),
            axum::http::HeaderValue::from_str(&inbound_tp).unwrap(),
        );
        let cx = crate::observability::propagation::extract_context_from_headers(&headers);

        // Mirror the gateway: the inbound context is current over the
        // publish future, then a non-generation endpoint injects it.
        let (tp, _ts) = async { WorkPublisher::trace_context_for_endpoint("encode") }
            .with_context(cx)
            .await;

        let tp = tp.expect("encode must inject the inbound trace context");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(
            parts[1], inbound_trace_id,
            "must continue the inbound trace_id"
        );
        assert_eq!(
            parts[2], inbound_span_id,
            "worker's parent must be the inbound span"
        );
    }

    #[tokio::test]
    async fn test_gateway_publish_span_context_parents_non_generation_envelope() {
        use opentelemetry::trace::{FutureExt, TraceContextExt, TracerProvider as _};
        use opentelemetry_sdk::propagation::TraceContextPropagator;
        use opentelemetry_sdk::trace::SdkTracerProvider;
        use tracing::Instrument;
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        use tracing_subscriber::layer::SubscriberExt;

        async fn publish_with_gateway_span(
            headers: axum::http::HeaderMap,
        ) -> (String, String, String) {
            let provider = SdkTracerProvider::builder().build();
            let tracer = provider.tracer("gateway-test");
            let subscriber = tracing_subscriber::registry()
                .with(tracing_opentelemetry::layer().with_tracer(tracer));
            let dispatch = tracing::Dispatch::new(subscriber);

            let publish_fut = tracing::dispatcher::with_default(&dispatch, || {
                let inbound_cx =
                    crate::observability::propagation::extract_context_from_headers(&headers);
                let span = tracing::info_span!(
                    "gateway.publish",
                    otel.name = "gateway.publish",
                    sie.endpoint = "encode",
                    sie.model = "BAAI/bge-m3",
                    sie.pool = "default",
                    sie.publish_ms = tracing::field::Empty,
                );
                let _ = span.set_parent(inbound_cx);
                let publish_cx = span.context();
                let span_context = publish_cx.span().span_context().clone();
                let gateway_trace_id = span_context.trace_id().to_string();
                let gateway_span_id = span_context.span_id().to_string();

                async move {
                    let (tp, _ts) = async { WorkPublisher::trace_context_for_endpoint("encode") }
                        .with_context(publish_cx)
                        .instrument(span)
                        .await;
                    (
                        tp.expect("encode envelope must carry gateway.publish context"),
                        gateway_trace_id,
                        gateway_span_id,
                    )
                }
            });

            publish_fut.await
        }

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

        let inbound_trace_id = "0af7651916cd43dd8448eb211c80319c";
        let inbound_span_id = "b7ad6b7169203331";
        let inbound_tp = format!("00-{inbound_trace_id}-{inbound_span_id}-01");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::HeaderName::from_static("traceparent"),
            axum::http::HeaderValue::from_str(&inbound_tp).unwrap(),
        );

        let (tp, gateway_trace_id, gateway_span_id) = publish_with_gateway_span(headers).await;
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(
            parts[1], inbound_trace_id,
            "gateway.publish must continue the inbound trace_id"
        );
        assert_eq!(parts[1], gateway_trace_id);
        assert_eq!(
            parts[2], gateway_span_id,
            "worker's parent must be gateway.publish"
        );
        assert_ne!(
            parts[2], inbound_span_id,
            "worker must not attach directly to the inbound client span"
        );
        assert_eq!(parts[3], "01", "gateway.publish span should be sampled");

        let (root_tp, root_gateway_trace_id, root_gateway_span_id) =
            publish_with_gateway_span(axum::http::HeaderMap::new()).await;
        let root_parts: Vec<&str> = root_tp.split('-').collect();
        assert_eq!(root_parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(
            root_parts[1], root_gateway_trace_id,
            "gateway.publish must be the fresh trace root when no inbound context exists"
        );
        assert_eq!(
            root_parts[2], root_gateway_span_id,
            "worker's parent must be the root gateway.publish span"
        );
        assert_ne!(root_parts[1], "00000000000000000000000000000000");
        assert_ne!(root_parts[2], "0000000000000000");
        assert_eq!(
            root_parts[3], "01",
            "root gateway.publish span should be sampled"
        );
    }

    /// Integration test (issue #1500): drive a real `WorkPublisher` encode
    /// publish over a live NATS/JetStream broker, with the inbound trace
    /// context scoped over the publish via `with_context` exactly as
    /// `proxy_request` does, and assert the published work-item envelope
    /// (decoded off the wire) carries a `traceparent` that CONTINUES the
    /// inbound trace. Unlike the unit tests above this exercises the real
    /// JetStream publish, the real msgpack `WorkItemRef` serialization,
    /// and the context surviving the publish future's own `.await`s.
    ///
    /// NATS-gated, mirroring `tests/nak_republish.rs`: a no-op pass when
    /// `NATS_URL` is unset / unreachable so the hermetic `cargo test` run
    /// (and the gateway CI job, which runs no NATS sidecar) stays green;
    /// the full flow runs when `NATS_URL` points at a JetStream server.
    #[tokio::test]
    async fn test_encode_publish_continues_inbound_trace_over_nats() {
        use futures_util::StreamExt;
        use opentelemetry::trace::FutureExt;
        use opentelemetry_sdk::propagation::TraceContextPropagator;

        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(c)) => c,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

        // Unique pool token so this run's JetStream stream + work subject
        // never collide with a parallel run on a shared broker.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pool = format!("itpool{nanos}");
        let model = "BAAI__bge-m3";

        let publisher = Arc::new(WorkPublisher::new(
            async_nats::jetstream::new(client.clone()),
            "it-router".to_string(),
            Arc::new(crate::queue::payload_store::DisabledPayloadStore),
            Duration::from_secs(5),
            1024,
            WorkStreamConfig {
                max_age: Duration::from_secs(300),
                storage: jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));

        let target = PublishTarget::Pool {
            pool: pool.clone(),
            machine_profile: "l4".to_string(),
            bundle: "default".to_string(),
            model: model.to_string(),
        };
        let subject = target.subject();

        // Pre-create the work stream + a JetStream consumer so the
        // publisher sees a live consumer (`num_consumers > 0`), mirroring a
        // real worker bound to the pool. Without it `publish_work` fails
        // backpressure with "no consumers available for work stream". The
        // stream config matches `ensure_stream` so its later
        // `get_or_create_stream` is a no-op.
        let stream = async_nats::jetstream::new(client.clone())
            .get_or_create_stream(jetstream::stream::Config {
                name: stream_name(&pool),
                subjects: vec![format!("sie.work.{pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(300),
                max_messages: 100_000,
                discard: jetstream::stream::DiscardPolicy::New,
                ..Default::default()
            })
            .await
            .expect("create work stream");
        stream
            .create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some("itworker".to_string()),
                filter_subject: subject.clone(),
                ..Default::default()
            })
            .await
            .expect("create work consumer");

        // Subscribe BEFORE publishing — a core subscriber also receives
        // JetStream-published messages on the subject.
        let mut sub = client.subscribe(subject.clone()).await.expect("subscribe");
        client.flush().await.expect("flush");

        // Inbound client trace context (as a `/v1/encode` caller sends).
        let inbound_trace_id = "0af7651916cd43dd8448eb211c80319c";
        let inbound_span_id = "b7ad6b7169203331";
        let inbound_tp = format!("00-{inbound_trace_id}-{inbound_span_id}-01");
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::HeaderName::from_static("traceparent"),
            axum::http::HeaderValue::from_str(&inbound_tp).unwrap(),
        );
        let cx = crate::observability::propagation::extract_context_from_headers(&headers);

        let params = WorkParams {
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            labels: None,
            output_schema: None,
            query_item: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
        };
        let items = vec![rmpv::Value::Map(vec![(
            rmpv::Value::String("text".into()),
            rmpv::Value::String("hello".into()),
        )])];

        // Mirror the handler: scope the inbound context over the publish.
        let (_request_id, _rx, durability) = publisher
            .publish_work(
                target, &pool, "encode", model, "pytorch", "", items, &params,
            )
            .with_context(cx)
            .await
            .expect("publish_work");
        durability.wait().await.expect("durable publish ACK");

        let msg = tokio::time::timeout(Duration::from_secs(5), sub.next())
            .await
            .expect("timed out waiting for the published work item")
            .expect("subscription closed before a message arrived");
        let work: WorkItem =
            rmp_serde::from_slice(&msg.payload).expect("decode work-item envelope");

        let tp = work
            .traceparent
            .expect("encode envelope must carry the inbound traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "wire shape: version-trace-span-flags");
        assert_eq!(
            parts[1], inbound_trace_id,
            "must continue the inbound trace_id"
        );
        assert_eq!(
            parts[2], inbound_span_id,
            "worker's parent must be the inbound span"
        );

        // Best-effort cleanup so repeated local runs don't accumulate
        // per-run streams on a persistent broker.
        let _ = async_nats::jetstream::new(client.clone())
            .delete_stream(stream_name(&pool))
            .await;
    }

    /// Real publisher regression for native OSS-shaped payload stores. This is
    /// NATS-gated like the adjacent publish-span integration test: point
    /// `NATS_URL` at an ephemeral memory-backed JetStream server to exercise
    /// public `publish_work` and its real terminal cleanup path.
    #[tokio::test]
    async fn publisher_large_oss_payload_emits_plain_ref_and_terminal_delete() {
        use futures_util::StreamExt;

        if std::env::var("SIE_RUN_NATS_PUBLISHER_TEST").as_deref() != Ok("1") {
            eprintln!("skipping: SIE_RUN_NATS_PUBLISHER_TEST=1 was not requested");
            return;
        }
        let url =
            std::env::var("NATS_URL").expect("mandatory publisher regression requires NATS_URL");
        let client = tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url))
            .await
            .expect("timed out connecting mandatory publisher regression to NATS")
            .expect("failed to connect mandatory publisher regression to NATS");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let pool = format!("osspayload{nanos}");
        let model = "BAAI__bge-m3";
        let store = Arc::new(FullReferencePayloadStore {
            full_reference: "oss://sie-wire-fixture/payloads/provider-returned-full-ref.bin"
                .to_owned(),
            puts: std::sync::Mutex::new(Vec::new()),
            deleted: std::sync::Mutex::new(Vec::new()),
        });
        let publisher = Arc::new(WorkPublisher::new(
            async_nats::jetstream::new(client.clone()),
            "oss-publisher-test".to_owned(),
            store.clone(),
            Duration::from_secs(5),
            1024,
            WorkStreamConfig {
                max_age: Duration::from_secs(300),
                storage: jetstream::stream::StorageType::Memory,
                num_replicas: 1,
            },
        ));
        let target = PublishTarget::Pool {
            pool: pool.clone(),
            machine_profile: "a10".to_owned(),
            bundle: "default".to_owned(),
            model: model.to_owned(),
        };
        let subject = target.subject();
        let context = async_nats::jetstream::new(client.clone());
        let stream = context
            .get_or_create_stream(jetstream::stream::Config {
                name: stream_name(&pool),
                subjects: vec![format!("sie.work.{pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(300),
                max_messages: 100_000,
                discard: jetstream::stream::DiscardPolicy::New,
                ..Default::default()
            })
            .await
            .expect("create OSS publisher test stream");
        stream
            .create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some("oss-payload-worker".to_owned()),
                filter_subject: subject.clone(),
                ..Default::default()
            })
            .await
            .expect("create OSS publisher test consumer");
        let mut subscription = client.subscribe(subject).await.expect("subscribe");
        client.flush().await.expect("flush subscription");

        let params = WorkParams {
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            labels: None,
            output_schema: None,
            query_item: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
        };
        let items = vec![rmpv::Value::Map(vec![(
            rmpv::Value::String("image".into()),
            rmpv::Value::Binary(vec![7; PAYLOAD_OFFLOAD_THRESHOLD + 1024]),
        )])];
        let (request_id, _rx, durability) = publisher
            .publish_work(
                target, &pool, "encode", model, "pytorch", "", items, &params,
            )
            .await
            .expect("publish oversized work item");
        durability.wait().await.expect("durable publish ACK");

        let message = tokio::time::timeout(Duration::from_secs(5), subscription.next())
            .await
            .expect("timed out waiting for oversized work item")
            .expect("subscription closed before oversized work item");
        let work: WorkItem = rmp_serde::from_slice(&message.payload).expect("decode work item");
        let expected_key = format!("{request_id}_0.bin");
        assert!(work.item.is_none());
        assert_eq!(work.payload_ref.as_deref(), Some(expected_key.as_str()));
        {
            let puts = store.puts.lock().unwrap();
            assert_eq!(puts.len(), 1);
            assert_eq!(puts[0].0, expected_key);
            assert!(puts[0].1 > PAYLOAD_OFFLOAD_THRESHOLD);
        }

        assert!(publisher.drop_pending_result(&request_id));
        publisher.finish_abandoned_work(&request_id).await;
        assert_eq!(
            store.deleted.lock().unwrap().as_slice(),
            std::slice::from_ref(&expected_key)
        );
        assert!(!publisher.offloaded_payload_keys.contains_key(&request_id));

        context
            .delete_stream(stream_name(&pool))
            .await
            .expect("delete OSS publisher test stream");
    }

    // ----- H9 — first-chunk-fallback rate limit ---------------------

    /// A NAK is dropped as stale only when the collector can identify it as an
    /// abandoned, non-empty attempt id — either by comparing with the latched
    /// current attempt, or with the abandoned attempt recorded during a
    /// NAK-driven republish before the successor's first chunk. See #1601.
    #[test]
    fn test_nak_is_stale_for_latched_or_recorded_abandoned_attempt() {
        // Stale: collector latched gen1, NAK is from the abandoned gen0.
        assert!(nak_is_stale(Some("gen1"), None, "gen0"));
        // Not stale: NAK is from the currently-latched attempt.
        assert!(!nak_is_stale(Some("gen0"), None, "gen0"));
        // Stale: republish happened before any successor chunk latched, but the
        // NAK-driven republish recorded the abandoned attempt id.
        assert!(nak_is_stale(None, Some("gen0"), "gen0"));
        // Not stale by this predicate: before successor latch, a different NAK
        // may be from the live pool attempt. The already-republished handler
        // treats that state as ambiguous and waits instead of failing.
        assert!(!nak_is_stale(None, Some("gen0"), "gen1"));
        // Fail open — no attempt latched or recorded yet: act on the NAK.
        assert!(!nak_is_stale(None, None, "gen0"));
        // Fail open — NAK carries no attempt id: act on the NAK.
        assert!(!nak_is_stale(Some("gen1"), Some("gen0"), ""));
        assert!(!nak_is_stale(None, Some("gen0"), ""));
        assert!(!nak_is_stale(None, None, ""));
    }

    #[test]
    fn test_already_republished_nak_waits_before_successor_latches() {
        use AlreadyRepublishedNakDecision::{DropStale, Fail, WaitForSuccessor};

        // Timeout-driven fallback: no worker attempt id was known when the
        // request republished, so a later NAK before the pool worker's first
        // chunk is ambiguous. Do not synthesize the spurious 429 from #1601.
        assert_eq!(
            already_republished_nak_decision(None, None, "gen0"),
            WaitForSuccessor
        );
        // Even if a previous pre-latch NAK recorded an abandoned id, a
        // different pre-latch NAK is still ambiguous until a successor chunk
        // proves which attempt is live.
        assert_eq!(
            already_republished_nak_decision(None, Some("gen0"), "gen1"),
            WaitForSuccessor
        );
        // Exact abandoned-id matches are still stale drops.
        assert_eq!(
            already_republished_nak_decision(None, Some("gen0"), "gen0"),
            DropStale
        );
        assert_eq!(
            already_republished_nak_decision(Some("gen1"), Some("gen0"), "gen0"),
            DropStale
        );
        // Once an attempt has latched, a NAK from that same attempt is the live
        // retry failure and may surface immediately.
        assert_eq!(
            already_republished_nak_decision(Some("gen1"), Some("gen0"), "gen1"),
            Fail
        );
    }

    /// The bucket admits exactly ``burst`` tokens immediately, then refuses
    /// until the rate refills enough for the next whole token. Time is
    /// injected, so the refill is exercised by advancing a synthetic clock —
    /// deterministic, no ``thread::sleep`` and no CI-timing window.
    #[test]
    fn test_token_bucket_burst_and_refill() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::new(10.0, 3.0, t0); // 10/s, burst 3
                                                          // Burst exhausted in three takes at t0.
        assert!(bucket.try_take(t0));
        assert!(bucket.try_take(t0));
        assert!(bucket.try_take(t0));
        assert!(!bucket.try_take(t0), "burst exceeded — must refuse");

        // Advance 150ms — at 10/s that is >=1 token of refill.
        let t1 = t0 + Duration::from_millis(150);
        assert!(
            bucket.try_take(t1),
            "expected refill to permit one more take"
        );
    }

    /// Regression for the burst cap: the bucket must NOT accumulate
    /// tokens above ``burst`` even when idle for a long stretch. Without
    /// the cap a quiet system would let a single noisy request drain
    /// hours of accrued tokens at once.
    #[test]
    fn test_token_bucket_caps_at_burst() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::new(100.0, 2.0, t0); // very fast refill, tiny burst
                                                           // 50ms at 100/s would refill 5 tokens uncapped; the cap must hold.
        let t1 = t0 + Duration::from_millis(50);
        // Two takes succeed (burst), the third refuses (no overflow stored).
        assert!(bucket.try_take(t1));
        assert!(bucket.try_take(t1));
        assert!(
            !bucket.try_take(t1),
            "burst cap must be enforced even after a long idle window"
        );
    }

    /// Property: drop a flurry of N > burst tries with zero sleep —
    /// exactly ``burst`` succeed. Mirrors the gateway-side scenario where
    /// a cold-start storm fires more first-chunk timeouts than the rate
    /// permits and we want a deterministic, bounded number of republishes.
    #[test]
    fn test_token_bucket_drops_excess_attempts() {
        let t0 = Instant::now();
        let mut bucket = TokenBucket::new(5.0, 4.0, t0); // 5/s, burst 4
        let mut admitted = 0usize;
        // All attempts at the same instant — no refill, so exactly burst admit.
        for _ in 0..20 {
            if bucket.try_take(t0) {
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, 4,
            "exactly burst-many attempts admit when the test runs faster than the refill rate"
        );
    }

    /// Key-isolation test for the per-(model, pool) bucket map: the
    /// rate limit must NOT cross-talk between distinct key tuples.
    /// Models the gateway-side scenario where one pool is in a fallback
    /// storm and a different pool's healthy traffic must be unaffected.
    /// We exercise the keying logic directly via the same DashMap +
    /// TokenBucket types the production code uses — no NATS / JetStream
    /// dependency, no I/O.
    #[test]
    fn test_fallback_rate_limit_isolates_keys_and_drops_excess() {
        let buckets: DashMap<String, std::sync::Mutex<TokenBucket>> = DashMap::new();
        let rate = FALLBACK_RATE_PER_SEC_DEFAULT;
        let burst = FALLBACK_BURST_DEFAULT;

        fn try_take(
            buckets: &DashMap<String, std::sync::Mutex<TokenBucket>>,
            rate: f64,
            burst: f64,
            model: &str,
            pool: &str,
            now: Instant,
        ) -> bool {
            let key = format!("{}|{}", model, pool);
            let entry = buckets
                .entry(key)
                .or_insert_with(|| std::sync::Mutex::new(TokenBucket::new(rate, burst, now)));
            let admitted = entry.value().lock().unwrap().try_take(now);
            admitted
        }

        // All attempts at one instant — no refill, so each key admits exactly
        // burst-many before refusing.
        let t0 = Instant::now();

        // Same key — admit exactly burst, then refuse.
        let mut admitted_a = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-A", "pool-1", t0) {
                admitted_a += 1;
            }
        }
        assert_eq!(admitted_a, burst as usize);

        // Different model on same pool → independent bucket → full burst.
        let mut admitted_b = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-B", "pool-1", t0) {
                admitted_b += 1;
            }
        }
        assert_eq!(admitted_b, burst as usize);

        // Different pool on same model → also independent.
        let mut admitted_c = 0usize;
        for _ in 0..30 {
            if try_take(&buckets, rate, burst, "model-A", "pool-2", t0) {
                admitted_c += 1;
            }
        }
        assert_eq!(admitted_c, burst as usize);
    }
}
