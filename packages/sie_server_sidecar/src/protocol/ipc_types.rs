//! IPC wire types — these MUST stay in lockstep with:
//!
//! 1. `packages/sie_server_sidecar/src/ipc_types.rs` (the **sidecar** —
//!    consumer of this adapter's responses).
//! 2. `packages/sie_server/src/sie_server/ipc_types.py` (the **Python
//!    adapter** — peer adapter speaking the same wire surface).
//!
//! All three are deliberately duplicated rather than extracted to a
//! shared Python/Rust package: adapters are standalone deliverables, and
//! we want adapter authors to vendor the
//! protocol like any other API client. CI checks the three copies are
//! field-compatible (see `tools/ci/check_ipc_types_parity.py`).
//!
//! Wire format: `[4-byte BE length][msgpack body]`, where `body` is a
//! msgpack **map** encoding `RequestEnvelope` / `ResponseEnvelope`.
//! `rmp_serde::to_vec_named` + `serde(default)` for forward-compat
//! field absorption.

use serde::{Deserialize, Serialize};

/// User item payloads are msgpack-native so binary fields (`bin` / `ext`) can
/// reach Python unchanged. Small config fields stay JSON-shaped.
pub type WireValue = rmpv::Value;

pub const IPC_VERSION: u32 = 1;

pub const METHOD_PING: &str = "Ping";
pub const METHOD_ENSURE_MODEL_READY: &str = "EnsureModelReady";
pub const METHOD_PROCESS_ENCODE_BATCH: &str = "ProcessEncodeBatch";
pub const METHOD_PROCESS_SCORE_BATCH: &str = "ProcessScoreBatch";
pub const METHOD_PROCESS_EXTRACT_BATCH: &str = "ProcessExtractBatch";
pub const METHOD_PROCESS_GENERATE: &str = "ProcessGenerate";
pub const METHOD_WORKER_CAPABILITIES: &str = "WorkerCapabilities";
pub const METHOD_SIGNAL_GENERATE_CANCEL: &str = "SignalGenerateCancel";
/// RPC that accepts a whole pre-formed batch (mixed op kinds illegal —
/// one `RunBatchRequest` is a single op) and returns today's
/// [`BatchOutcome`] unchanged. Carries every batch emitted by the
/// Rust-side scheduler; per-op Process* RPCs stay in the wire contract
/// for backends that don't implement `run_batch` and for unit tests
/// that don't wire a scheduler.
pub const METHOD_RUN_BATCH: &str = "RunBatch";
pub const METHOD_APPLY_MODEL_CONFIG: &str = "ApplyModelConfig";
pub const METHOD_REPLACE_MODEL_CONFIGS: &str = "ReplaceModelConfigs";
pub const METHOD_SET_PINNED_MODELS: &str = "SetPinnedModels";
pub const METHOD_DRAIN: &str = "Drain";

// -----------------------------------------------------------------------------
// Envelope
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct RequestEnvelope<'a, B: Serialize> {
    pub version: u32,
    pub method: &'a str,
    pub request_id: String,
    pub body: B,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEnvelope<B> {
    pub version: u32,
    pub request_id: String,
    pub ok: bool,
    #[serde(default = "none_option")]
    pub body: Option<B>,
    #[serde(default)]
    pub error: Option<String>,
}

fn none_option<T>() -> Option<T> {
    None
}

/// Deserialize a msgpack field that is EITHER a byte array OR `nil`. Used
/// for `ItemOutcome::result_msgpack`, which Python types as `bytes | None`;
/// `msgspec` / `msgpack.packb` happily emits `nil` when the field is
/// `None`, and `serde_bytes` alone rejects that with "invalid type: unit
/// value, expected byte array", sinking the whole `BatchOutcome` into a
/// transport-level decode error and triggering a batch NAK.
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

// -----------------------------------------------------------------------------
// Ping
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingRequest {
    pub timestamp_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResponse {
    #[serde(default)]
    pub timestamp_ms: f64,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub bundle_config_hash: String,
    #[serde(default)]
    pub loaded_models: Vec<String>,
}

// -----------------------------------------------------------------------------
// EnsureModelReady
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureModelReadyRequest {
    pub model_id: String,
}

/// Readiness state reported by the Python executor.
/// Wire format is a plain string (see Python `ReadinessState` Literal).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessState {
    Ready,
    LoadingStarted,
    LoadingInProgress,
    RetryLater,
    /// Terminal, non-retryable load failure. The Python executor
    /// reports this when the registry holds a PERMANENT `LoadFailure`
    /// (`cooldown=permanent`: `GATED` / `NOT_FOUND` / `DEPENDENCY` /
    /// `UNKNOWN`) — the same classification the direct-HTTP
    /// `check_not_failed` gate and the Modal lane's
    /// `worker_runtime._terminal_load_failure` use for #1786. It exists
    /// so the sidecar can DISTINGUISH "still loading" (retry) from
    /// "dead on arrival" (dead-letter): on `Failed` the dispatcher stops
    /// re-driving `EnsureModelReady` and publishes a typed
    /// `MODEL_LOAD_FAILED` error `WorkResult` (gateway → 502), matching
    /// the batch/`run_batch` path. Wire format is the plain string
    /// `"failed"` (see Python `ReadinessState` Literal).
    Failed,
}

/// Model descriptor carried on the `EnsureModelReady` handshake.
///
/// Replaces legacy per-model env lists with a per-model handshake:
/// the adapter declares its tokeniser path / id / max-seq-len / output
/// types / `RunBatch` capability on first ready, and the sidecar caches
/// the descriptor for Rust-side tokenisation and output shaping. Every
/// field is optional / defaulted because capabilities are model-dependent;
/// a missing descriptor means "no Rust-side tokenisation for this model".
///
/// See `packages/sie_server_sidecar/docs/architecture-guide.md`
/// for the contract.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelDescriptor {
    /// Absolute path inside a filesystem the sidecar can read
    /// (single-container deploy: HF cache; split-container: shared
    /// `emptyDir` under `/var/run/sie/...`). `None` -> adapter has not
    /// materialised its tokeniser to a sidecar-visible path; Rust
    /// skips Rust-side tokenisation for this model and Python tokenises.
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    /// BLAKE3 (truncated to 32 hex chars) of the canonical tokenizer
    /// JSON. Lets the sidecar verify byte-identity with the adapter's
    /// loaded tokeniser before enabling the fast path.
    #[serde(default)]
    pub tokenizer_id: Option<String>,
    /// Per-model truncation cap. `None` falls back to the registry's
    /// `DEFAULT_MAX_SEQ_LEN`.
    #[serde(default)]
    pub max_seq_len: Option<u32>,
    /// Output keys the adapter knows how to emit (`"dense"`,
    /// `"sparse"`, `"multivector"`, `"score"`). Informational — the
    /// Rust-side framing path is gated per-request by
    /// `_maybe_*_raw_output` shape checks on the Python side, not by
    /// this field.
    #[serde(default)]
    pub output_types: Vec<String>,
    /// True iff the adapter has implemented the `RunBatch` IPC method.
    /// False for adapters still on the per-op `Process*Batch` RPCs.
    #[serde(default)]
    pub supports_run_batch: bool,
    /// Model-default `query_template` (Python side: per-adapter
    /// `query_template` ctor arg, sourced from the model YAML). Used
    /// by the sidecar to apply asymmetric-retrieval prompts in Rust
    /// before tokenisation when an item arrives with `is_query=true`,
    /// matching Python's `_utils.extract_texts` precedence:
    /// template &gt; bare instruction &gt; passthrough. `None` keeps
    /// today's behaviour: Python applies templates and Rust either
    /// skips `prepared_tokens` or matches the unaltered text.
    ///
    /// Per-request `options.query_template` still wins; this is only
    /// the model-default fallback. Wire format is the raw Python
    /// `str.format`-style template (e.g. `"query: {text}"` or
    /// `"Instruct: {instruction}\nQuery: {text}"`) — the sidecar's
    /// [`crate::prep::text_prep::TextPrep`] is byte-identical to Python's
    /// `_utils.extract_texts` for the two known placeholders.
    #[serde(default)]
    pub default_query_template: Option<String>,
    /// Model-default `doc_template` (Python side: per-adapter
    /// `doc_template` ctor arg). Same wire contract as
    /// [`Self::default_query_template`] but selected when
    /// `is_query=false`.
    #[serde(default)]
    pub default_doc_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureModelReadyResponse {
    pub state: ReadinessState,
    /// Per-model batch budget for fair dispatch. Mirrors
    /// `sie_server.ipc_types.EnsureModelReadyResponse.batch_budget`. Only
    /// populated when `state == Ready`; falls back to a default budget on
    /// the Rust side when unset.
    #[serde(default)]
    pub batch_budget: Option<u32>,
    /// Model descriptor. Populated by Python adapters that expose a HF
    /// fast tokeniser + structured `output_types`; absent for older /
    /// image / audio adapters that don't yet emit it.
    /// See [`ModelDescriptor`] for the per-field semantics and the
    /// fallback policy.
    #[serde(default)]
    pub descriptor: Option<ModelDescriptor>,
}

// -----------------------------------------------------------------------------
// ApplyModelConfig
// -----------------------------------------------------------------------------

/// Bundle-scoped config delta applied by the worker-sidecar's
/// `sie.config.models.<bundle>` subscriber.
///
/// Rust owns the NATS subscription, producer validation, epoch gating, and
/// health hash publication. The colocated backend owns schema validation and
/// mutation of its local registry/catalog, so the actual registry write remains
/// in the process that serves inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyModelConfigRequest {
    pub bundle_id: String,
    pub model_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub profiles_added: Vec<String>,
    pub model_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyModelConfigResponse {
    pub applied: bool,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub config_version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigEntry {
    pub model_id: String,
    pub model_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigsRequest {
    pub bundle_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    pub models: Vec<ReplaceModelConfigEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigsResponse {
    pub applied: bool,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub config_version: u64,
    #[serde(default)]
    pub applied_models: Vec<String>,
}

// -----------------------------------------------------------------------------
// SetPinnedModels
// -----------------------------------------------------------------------------

/// Runtime delivery of a pool's pinned-model set from the gateway to the
/// worker. The sidecar polls the pool's `PoolSpec.pinned_models` (the single
/// source of truth, set via the `/v1/pools` API or
/// `SIE_GATEWAY_STATIC_QUEUE_POOLS`) and pushes it here on change. `models`
/// carries gateway-canonical ids (bare `sie_id` or `sie_id:profile`); the
/// backend normalizes them. The set is authoritative and REPLACES the worker's
/// current pinned set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetPinnedModelsRequest {
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetPinnedModelsResponse {
    pub applied: bool,
    #[serde(default)]
    pub pinned_count: u32,
}

// -----------------------------------------------------------------------------
// ProcessGenerate: streaming generation over IPC
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessGenerateRequest {
    pub model_id: String,
    #[serde(with = "serde_bytes")]
    pub work_item_msgpack: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerateEventKind {
    Publish,
    Ack,
    Nak,
    InProgress,
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateEvent {
    pub kind: String,
    #[serde(default)]
    pub reply_subject: String,
    #[serde(default, with = "serde_bytes")]
    pub payload: Vec<u8>,
    #[serde(default)]
    pub delay_ms: Option<u64>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerCapabilitiesRequest {
    // No fields: the method asks the backend to describe its local registry.
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerCapabilitiesResponse {
    #[serde(default)]
    pub has_generation_models: bool,
    #[serde(default)]
    pub generation_models: Vec<String>,
    #[serde(default)]
    pub supported_models: Vec<String>,
    #[serde(default)]
    pub loaded_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGenerateCancelRequest {
    pub request_id: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SignalGenerateCancelResponse {
    #[serde(default)]
    pub matched: bool,
}

// -----------------------------------------------------------------------------
// PreparedTokens: Rust-side tokenisation
// -----------------------------------------------------------------------------

/// Ragged per-item token output, attached by the Rust dispatcher to
/// `EncodeBatchItem` / `ScoreBatchItem` when the sidecar has
/// pre-tokenised the text. The Python adapter compares
/// `tokenizer_id` against its own loaded tokenizer's content hash:
/// match → use the tokens directly, mismatch / absence → tokenise in
/// Python as today. See
/// `packages/sie_server_sidecar/docs/architecture-guide.md`
/// for the full contract, parity rules, and fallback policy.
///
/// Each inner `Vec<u32>` is the `input_ids` for a single text item
/// (ragged, no batch padding — mirrors Python's
/// `padding=False, truncation=True, max_length=N`). `attention_mask`
/// and `token_type_ids` run in lockstep with `input_ids` when
/// populated; empty inner vectors signal "tokenizer didn't emit this
/// field" and the Python adapter falls back to `[1] * len(input_ids)`
/// / all-zero segments.
///
/// The containing `EncodeBatchItem` / `ScoreBatchItem` keeps a
/// `prepared_tokens: Option<PreparedTokens>` with `#[serde(default)]` because
/// the worker-sidecar only attaches it when tokenizer materialization and
/// hashing succeed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedTokens {
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask per item. Same shape as `input_ids`. Empty
    /// outer vec signals "not emitted" (Python fills all-ones).
    #[serde(default)]
    pub attention_mask: Vec<Vec<u32>>,
    /// Segment ids per item. Same shape as `input_ids`. Empty outer
    /// vec signals "tokenizer doesn't produce segments" (Python
    /// treats as all-zero when the adapter needs it).
    #[serde(default)]
    pub token_type_ids: Vec<Vec<u32>>,
    /// Stable content hash of the `tokenizer.json` bytes. Python
    /// recomputes the same hash on its loaded tokenizer; mismatch
    /// triggers fallback to Python tokenise for this request.
    pub tokenizer_id: String,
    /// Per-model truncation bound the Rust side used (same as the
    /// `max_length` kwarg on the Python tokenise call). Carried for
    /// metrics / debugging only — Python does not enforce it.
    #[serde(default)]
    pub max_seq_len: u32,
}

// -----------------------------------------------------------------------------
// RawOutput: Rust-side output shaping / result framing
// -----------------------------------------------------------------------------

/// Typed raw inference output returned by the Python adapter when
/// the worker-sidecar owns final wire framing.
///
/// Each [`ItemOutcome`] carries **either**:
///
/// * a pre-framed `result_msgpack` (legacy path; Python built the
///   full msgpack envelope exactly like today), **or**
/// * a `raw_output` with one of the typed variants populated — in
///   which case the Rust publisher calls the matching output shaper to
///   produce byte-identical `result_msgpack` bytes.
///
/// Wire format is additive and tolerant:
///
/// * All variants are `Option<_>` with `#[serde(default)]` so an old
///   Python adapter process round-trips as `raw_output = None` and the legacy
///   `result_msgpack` takes over.
/// * New variants can add another `Option<_>` field on this struct; old
///   Rust builds ignore them and fall back to the Python-framed bytes.
///
/// See
/// `packages/sie_server_sidecar/docs/architecture-guide.md`
/// for the full contract and fallback matrix.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawOutput {
    /// Dense `[dim]` embedding for a single encode item. One per
    /// `EncodeBatchItem` because the IPC batch is per-item; the
    /// adapter's internal GPU batch is flattened back on the way
    /// out to match today's framing.
    #[serde(default)]
    pub dense: Option<DenseOutput>,
    /// Rerank scores for one score item (`query + N docs`). Rust
    /// performs the sort-by-score-desc + rank assignment and emits
    /// the `[{item_id, score, rank}, ...]` list.
    #[serde(default)]
    pub score: Option<ScoreOutputRaw>,
    /// Sparse `(indices, values)` pair for one encode item —
    /// v1 scope is f32 values + i32 indices only. See
    /// [`crate::output::build_sparse_payload`] for the wire shape
    /// and `_maybe_sparse_raw_output` (Python side) for the safety
    /// gate that keeps float16 / non-i32 / binary variants on the
    /// Python-framed fallback path.
    #[serde(default)]
    pub sparse: Option<SparseOutput>,
    /// Multivector `[num_tokens, token_dims]` matrix for one encode
    /// item — v1 scope is f32 only. `float16` and the bit-packed
    /// binary-multivector variant (token row size `< token_dims`)
    /// stay on the Python path via the same gate.
    #[serde(default)]
    pub multivector: Option<MultivectorOutput>,
}

/// Flat dense-vector payload, dtype is always float32 for v1.
///
/// Shape guarantees: `values.len() == dim`. If this invariant is
/// violated the shaper returns
/// [`crate::output::ShapeError::DenseDimMismatch`] and the
/// publisher emits an `error` `WorkResult` — we never send a
/// truncated / oversized array over the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseOutput {
    pub values: Vec<f32>,
    pub dim: u32,
    /// If `true`, Rust L2-normalizes `values` before packing.
    /// v1: Python adapters normalize themselves (unchanged) and
    /// always emit `normalize = false`; kept on the wire so a later
    /// stage can delegate the normalize step to Rust without a
    /// schema bump.
    #[serde(default)]
    pub normalize: bool,
}

/// Rerank score payload: parallel `scores` + `item_ids` lists.
///
/// The Rust shaper sorts by score **descending** (stable tie-break
/// preserving input order, matching Python's
/// `list.sort(key=..., reverse=True)`) and assigns `rank = 0..`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreOutputRaw {
    /// One score per scored document, in input order. `f32` to
    /// match the `np.float32` origin inside the Python adapter;
    /// the shaper widens each via `f64::from(f32)` before emitting
    /// msgpack float64 — byte-identical to Python's
    /// `float(np.float32(x))` widening path (see golden in
    /// `crate::output::tests`).
    pub scores: Vec<f32>,
    /// Document ids in the same order as `scores`. Empty strings
    /// must be filled in by Python (e.g. synthesised `"item-{i}"`)
    /// before reaching Rust so the wire shape matches the Python
    /// fallback verbatim.
    pub item_ids: Vec<String>,
}

/// Sparse embedding payload: parallel `indices` + `values` lists.
///
/// Matches the adapter-side `SparseVector` shape (`np.int32`
/// indices, `np.float32` values) used by every in-tree sparse
/// encoder (`splade_flash`, `gte_sparse_flash`, `sentence_transformer`,
/// ...).
///
/// Shape guarantees:
///
/// * `indices.len() == values.len()` — the Rust shaper returns
///   [`crate::output::ShapeError::SparseLenMismatch`] otherwise.
/// * `dims` is optional on the wire (matches the Python
///   `{"dims": None, ...}` convention for vocab-less sparse); when
///   `Some`, it's the full sparse vocabulary size the SDK needs to
///   rehydrate the vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseOutput {
    /// Non-zero indices, int32 per adapter-layer convention.
    pub indices: Vec<i32>,
    /// Non-zero values, parallel to `indices`.
    pub values: Vec<f32>,
    /// Full sparse dimension (vocab size); `None` when the model
    /// does not expose a stable vocab size. Packed as msgpack `nil`
    /// in the latter case, matching `msgpack.packb({"dims": None, ...})`.
    #[serde(default)]
    pub dims: Option<u32>,
}

/// Multivector (per-token) embedding payload — `[num_tokens, token_dims]`
/// contiguous f32 matrix in C (row-major) order.
///
/// Shape guarantees: `values.len() == num_tokens * token_dims`. The
/// shaper returns [`crate::output::ShapeError::MultivectorShapeMismatch`]
/// when that invariant is violated, and the publisher converts the
/// outcome into an error `WorkResult` instead of emitting garbage
/// bytes.
///
/// v1 scope is `float32` only. Bit-packed binary multivector
/// (`shape[1] < token_dims`, `dim/8` bytes per token) and `float16`
/// remain on the Python framing path — the Python gate
/// `_maybe_multivector_raw_output` refuses to emit this variant for
/// those dtypes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivectorOutput {
    /// Flattened `[num_tokens, token_dims]` f32 matrix in row-major
    /// (C) order. Matches `np.ndarray.tobytes()` for a contiguous
    /// array.
    pub values: Vec<f32>,
    /// Number of tokens (rows).
    pub num_tokens: u32,
    /// Per-token embedding dimension (columns).
    pub token_dims: u32,
}

// -----------------------------------------------------------------------------
// ProcessEncodeBatch
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub timestamp: f64,
    pub item: WireValue,
    #[serde(default)]
    pub output_types: Option<Vec<String>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub is_query: bool,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub bundle_config_hash: Option<String>,
    #[serde(default)]
    pub payload_fetch_ms: f64,
    /// Rust-side pre-tokenised input. `None` when the model has no
    /// registered tokenizer, when the item is not safe to tokenize in
    /// Rust, or when the registry returned an error. See
    /// [`PreparedTokens`] for the full fallback matrix.
    #[serde(default)]
    pub prepared_tokens: Option<PreparedTokens>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessEncodeBatchRequest {
    pub model_id: String,
    pub items: Vec<EncodeBatchItem>,
}

// -----------------------------------------------------------------------------
// ProcessScoreBatch
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub timestamp: f64,
    pub query_item: WireValue,
    pub score_items: Vec<WireValue>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub payload_fetch_ms: f64,
    /// Rust-side pre-tokenised input, ordering `[query, doc_0, doc_1, ...]`.
    /// `None` for v1 by default: the dispatcher only attaches tokens for
    /// single-text score items (one query + one doc). Cross-encoder
    /// pair-tokenisation lives in the Python adapter for now — see the
    /// architecture guide.
    #[serde(default)]
    pub prepared_tokens: Option<PreparedTokens>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessScoreBatchRequest {
    pub model_id: String,
    pub items: Vec<ScoreBatchItem>,
}

// -----------------------------------------------------------------------------
// ProcessExtractBatch
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub timestamp: f64,
    pub item: WireValue,
    #[serde(default)]
    pub labels: Option<Vec<String>>,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub bundle_config_hash: Option<String>,
    #[serde(default)]
    pub payload_fetch_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessExtractBatchRequest {
    pub model_id: String,
    pub items: Vec<ExtractBatchItem>,
}

// -----------------------------------------------------------------------------
// Batch outcome
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    PublishAndAck,
    PublishErrorAndAck,
    NakRetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemOutcome {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub disposition: Disposition,
    #[serde(default)]
    pub nak_delay_ms: Option<u64>,
    /// `result_msgpack` is the opaque payload a successful item publishes.
    /// For `publish_error_and_ack` / `nak_retry` Python omits it, which
    /// `msgpack` encodes as either key-absent OR key-present-with-nil. The
    /// custom deserializer accepts both (and treats missing/nil as empty)
    /// so per-item error outcomes don't sink the whole batch into a NAK.
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
    pub tokenization_ms: Option<f64>,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
    /// Typed raw inference output. Present only when the Python adapter
    /// defers final wire framing to Rust after per-request safety checks.
    /// When `Some`, `result_msgpack` is expected to be empty and the
    /// publisher runs the Rust-side shapers to produce the final bytes.
    /// On any shaper error the outcome is converted into an error
    /// `WorkResult`, so a misconfigured model cannot silently drop
    /// results. See [`RawOutput`] for the wire-format fallback matrix.
    #[serde(default)]
    pub raw_output: Option<RawOutput>,
    /// Authoritative billable-unit counts (P3.5, design §7.3): real
    /// tokenizer counts emitted by the engine so metering edges never
    /// re-derive units from estimates. Optional; passed through onto the
    /// `WorkResult` wire unchanged.
    #[serde(default)]
    pub units: Option<UnitCounts>,
}

/// Mirror of `sie_server.ipc_types.UnitCounts` — a field is set only when
/// the engine holds an authoritative count for it (§7.3).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitCounts {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub pages: Option<u64>,
    #[serde(default)]
    pub images: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOutcome {
    pub outcomes: Vec<ItemOutcome>,
}

// -----------------------------------------------------------------------------
// RunBatch
// -----------------------------------------------------------------------------

/// One item inside a [`RunBatchRequest`]. Exactly one of the three
/// `Option` fields must be `Some`, matching the `op` discriminator.
///
/// Tagged struct rather than a `serde` enum because:
/// * msgpack → msgspec prefers explicit discriminators over untagged
///   unions (decoder tries each variant in order, which is both slow
///   and brittle when field sets overlap between `EncodeBatchItem`
///   and `ExtractBatchItem`).
/// * All three item types already have `#[serde(default)]` on most
///   fields so the "absent two" serialize as either nil or omitted —
///   no wasted bytes.
/// * Forward-compatible: adding a new op in the future is a new
///   optional field + a new `op` string, no breaking change for
///   readers that don't know it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBatchItem {
    /// Discriminator: `"encode"`, `"score"`, or `"extract"`.
    /// `adapter_call_loop.py` switches on this to pick the right
    /// handler without having to inspect which optional field is
    /// populated.
    pub op: String,
    /// Identity copied from the wrapped per-op item. This is redundant
    /// on the happy path, but it lets Python publish a deterministic
    /// per-item error if the wrapped payload is malformed or absent.
    #[serde(default)]
    pub work_item_id: String,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub item_index: u32,
    #[serde(default)]
    pub encode: Option<EncodeBatchItem>,
    #[serde(default)]
    pub score: Option<ScoreBatchItem>,
    #[serde(default)]
    pub extract: Option<ExtractBatchItem>,
    /// W3C trace context copied off the originating [`crate::work_types::WorkItem`]
    /// so the Python non-streaming worker loop can re-extract the gateway
    /// span and attach `worker.run_batch` as its child. `None` when the
    /// gateway didn't propagate a trace (older builds / untraced request).
    #[serde(default)]
    pub traceparent: Option<String>,
    #[serde(default)]
    pub tracestate: Option<String>,
}

impl RunBatchItem {
    /// Build a RunBatchItem wrapping an encode item, with the
    /// `op` discriminator set to `"encode"`. Exists so callers don't
    /// have to remember to sync the tag and the field.
    #[must_use]
    pub fn encode(item: EncodeBatchItem) -> Self {
        let work_item_id = item.work_item_id.clone();
        let request_id = item.request_id.clone();
        let item_index = item.item_index;
        Self {
            op: "encode".to_owned(),
            work_item_id,
            request_id,
            item_index,
            encode: Some(item),
            score: None,
            extract: None,
            traceparent: None,
            tracestate: None,
        }
    }

    /// Build a RunBatchItem wrapping a score item.
    #[must_use]
    pub fn score(item: ScoreBatchItem) -> Self {
        let work_item_id = item.work_item_id.clone();
        let request_id = item.request_id.clone();
        let item_index = item.item_index;
        Self {
            op: "score".to_owned(),
            work_item_id,
            request_id,
            item_index,
            encode: None,
            score: Some(item),
            extract: None,
            traceparent: None,
            tracestate: None,
        }
    }

    /// Build a RunBatchItem wrapping an extract item.
    #[must_use]
    pub fn extract(item: ExtractBatchItem) -> Self {
        let work_item_id = item.work_item_id.clone();
        let request_id = item.request_id.clone();
        let item_index = item.item_index;
        Self {
            op: "extract".to_owned(),
            work_item_id,
            request_id,
            item_index,
            encode: None,
            score: None,
            extract: Some(item),
            traceparent: None,
            tracestate: None,
        }
    }
}

/// Pre-formed batch dispatched through the Rust scheduler.
///
/// The scheduler forms a batch of *homogeneous* op (FCFS picks one
/// batcher per call; a batcher only holds one op by construction),
/// so every [`RunBatchItem`] in `items` shares the same `op` tag.
/// The Python adapter loop validates this on receipt.
///
/// `total_cost` is sent redundantly so the Python side can log it
/// without walking `items` — matches the current per-RPC logging
/// pattern where batch-level stats are peeled off without descending
/// into the payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBatchRequest {
    pub model_id: String,
    /// Rust-assigned id for cross-log correlation. Unique within the
    /// worker process lifetime; generation strategy is a plain
    /// monotonic `u64` on the dispatcher side.
    pub batch_id: u64,
    /// LoRA adapter the batch targets. Empty string == base model.
    /// Python's `adapter.set_active_lora` accepts `""` to mean base,
    /// matching this convention.
    pub lora_key: String,
    /// Sum of per-item costs (typically `PreparedTokens.seq_len` or
    /// the Python fallback estimate). Informational; Python
    /// recomputes when it needs the exact value.
    pub total_cost: u64,
    pub items: Vec<RunBatchItem>,
}

// -----------------------------------------------------------------------------
// Drain
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainRequest {
    #[serde(default)]
    pub deadline_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainResponse {
    #[serde(default)]
    pub acknowledged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_item(text: &str) -> WireValue {
        WireValue::Map(vec![(WireValue::from("text"), WireValue::from(text))])
    }

    #[test]
    fn readiness_state_serde_matches_python_literal() {
        // Python side uses Literal["ready","loading_started","loading_in_progress","retry_later"].
        let bytes = rmp_serde::to_vec_named(&ReadinessState::LoadingInProgress).unwrap();
        let back: ReadinessState = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, ReadinessState::LoadingInProgress);

        // JSON form — easiest to assert the string value.
        let s = serde_json::to_string(&ReadinessState::LoadingInProgress).unwrap();
        assert_eq!(s, "\"loading_in_progress\"");

        let s = serde_json::to_string(&ReadinessState::RetryLater).unwrap();
        assert_eq!(s, "\"retry_later\"");

        // Terminal load-failure variant — the string the Python executor
        // sends so the sidecar can dead-letter instead of re-driving.
        let s = serde_json::to_string(&ReadinessState::Failed).unwrap();
        assert_eq!(s, "\"failed\"");
        let bytes = rmp_serde::to_vec_named(&ReadinessState::Failed).unwrap();
        let back: ReadinessState = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, ReadinessState::Failed);
    }

    #[test]
    fn disposition_serde_matches_python_literal() {
        let s = serde_json::to_string(&Disposition::PublishAndAck).unwrap();
        assert_eq!(s, "\"publish_and_ack\"");
        let s = serde_json::to_string(&Disposition::PublishErrorAndAck).unwrap();
        assert_eq!(s, "\"publish_error_and_ack\"");
        let s = serde_json::to_string(&Disposition::NakRetry).unwrap();
        assert_eq!(s, "\"nak_retry\"");
    }

    #[test]
    fn item_outcome_roundtrip_with_bytes() {
        let outcome = ItemOutcome {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            disposition: Disposition::PublishAndAck,
            nak_delay_ms: None,
            result_msgpack: vec![1, 2, 3, 4, 5],
            error: None,
            error_code: None,
            inference_ms: Some(42.0),
            tokenization_ms: None,
            postprocessing_ms: None,
            raw_output: None,
            units: None,
        };
        let bytes = rmp_serde::to_vec_named(&outcome).unwrap();
        let back: ItemOutcome = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.request_id, "r");
        assert_eq!(back.result_msgpack, vec![1, 2, 3, 4, 5]);
        assert_eq!(back.inference_ms, Some(42.0));
        assert_eq!(back.disposition, Disposition::PublishAndAck);
    }

    #[test]
    fn item_outcome_decodes_when_python_sends_nil_bytes() {
        // Python (`msgspec.to_builtins` + `msgpack.packb`) can serialise
        // `result_msgpack: bytes | None = None` as either (a) key-absent or
        // (b) key-present with a `nil` value. `#[serde(default, with =
        // "serde_bytes")]` must handle both gracefully, otherwise every
        // per-item error outcome decodes as a transport failure and the
        // worker-sidecar NAKs the whole batch — a latent blocker the
        // pre-deploy audit flagged.
        let mut buf = Vec::new();
        rmp::encode::write_map_len(&mut buf, 9).unwrap();
        rmp::encode::write_str(&mut buf, "work_item_id").unwrap();
        rmp::encode::write_str(&mut buf, "r.0").unwrap();
        rmp::encode::write_str(&mut buf, "request_id").unwrap();
        rmp::encode::write_str(&mut buf, "r").unwrap();
        rmp::encode::write_str(&mut buf, "item_index").unwrap();
        rmp::encode::write_uint(&mut buf, 0).unwrap();
        rmp::encode::write_str(&mut buf, "disposition").unwrap();
        rmp::encode::write_str(&mut buf, "publish_error_and_ack").unwrap();
        rmp::encode::write_str(&mut buf, "nak_delay_ms").unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_str(&mut buf, "result_msgpack").unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();
        rmp::encode::write_str(&mut buf, "error").unwrap();
        rmp::encode::write_str(&mut buf, "boom").unwrap();
        rmp::encode::write_str(&mut buf, "error_code").unwrap();
        rmp::encode::write_str(&mut buf, "inference_error").unwrap();
        rmp::encode::write_str(&mut buf, "inference_ms").unwrap();
        rmp::encode::write_nil(&mut buf).unwrap();

        let back: ItemOutcome = rmp_serde::from_slice(&buf)
            .expect("ItemOutcome must decode when Python sends nil for optional bytes");
        assert_eq!(back.disposition, Disposition::PublishErrorAndAck);
        assert!(back.result_msgpack.is_empty());
        assert_eq!(back.error.as_deref(), Some("boom"));
        assert_eq!(back.error_code.as_deref(), Some("inference_error"));
        assert!(back.nak_delay_ms.is_none());
    }

    #[test]
    fn item_outcome_decodes_when_python_omits_optional_fields() {
        let mut buf = Vec::new();
        rmp::encode::write_map_len(&mut buf, 4).unwrap();
        rmp::encode::write_str(&mut buf, "work_item_id").unwrap();
        rmp::encode::write_str(&mut buf, "r.0").unwrap();
        rmp::encode::write_str(&mut buf, "request_id").unwrap();
        rmp::encode::write_str(&mut buf, "r").unwrap();
        rmp::encode::write_str(&mut buf, "item_index").unwrap();
        rmp::encode::write_uint(&mut buf, 0).unwrap();
        rmp::encode::write_str(&mut buf, "disposition").unwrap();
        rmp::encode::write_str(&mut buf, "publish_and_ack").unwrap();

        let back: ItemOutcome = rmp_serde::from_slice(&buf)
            .expect("ItemOutcome must decode when Python omits optional fields entirely");
        assert!(back.result_msgpack.is_empty());
        assert!(back.error.is_none());
        assert!(back.inference_ms.is_none());
    }

    #[test]
    fn prepared_tokens_roundtrips_msgpack() {
        // Additive field on EncodeBatchItem: both "field present, Some"
        // and "field absent" must roundtrip so Rust ↔ Python can roll
        // out in either order.
        let pt = PreparedTokens {
            input_ids: vec![vec![101, 2023, 2003, 1037, 3231, 102], vec![101, 2061, 102]],
            attention_mask: vec![vec![1, 1, 1, 1, 1, 1], vec![1, 1, 1]],
            token_type_ids: vec![vec![0, 0, 0, 0, 0, 0], vec![0, 0, 0]],
            tokenizer_id: "deadbeefcafef00dba5eba11c0ffee42".into(),
            max_seq_len: 512,
        };
        let bytes = rmp_serde::to_vec_named(&pt).unwrap();
        let back: PreparedTokens = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.input_ids, pt.input_ids);
        assert_eq!(back.attention_mask, pt.attention_mask);
        assert_eq!(back.token_type_ids, pt.token_type_ids);
        assert_eq!(back.tokenizer_id, pt.tokenizer_id);
        assert_eq!(back.max_seq_len, 512);
    }

    #[test]
    fn encode_batch_item_default_prepared_tokens_absent_on_wire() {
        // Old Python sending an EncodeBatchItem without the
        // `prepared_tokens` key — new Rust must decode successfully
        // and see `None`. This guards the additive wire contract.
        let bi = EncodeBatchItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: text_item("hello"),
            output_types: Some(vec!["dense".into()]),
            instruction: None,
            is_query: false,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        };
        let bytes = rmp_serde::to_vec_named(&bi).unwrap();
        let back: EncodeBatchItem = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.prepared_tokens.is_none());
    }

    #[test]
    fn run_batch_item_trace_context_roundtrips_msgpack() {
        // The trace fields must survive the named-msgpack round-trip so
        // the Python non-streaming worker loop can re-extract the
        // gateway span. Build an encode item and stamp the trace
        // context the way `into_run_batch_item_with_trace` would.
        let mut rbi = RunBatchItem::encode(EncodeBatchItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: text_item("hello"),
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        });
        rbi.traceparent = Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".into());
        rbi.tracestate = Some("vendor=opaque".into());

        let bytes = rmp_serde::to_vec_named(&rbi).unwrap();
        let back: RunBatchItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(
            back.traceparent.as_deref(),
            Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        );
        assert_eq!(back.tracestate.as_deref(), Some("vendor=opaque"));
    }

    #[test]
    fn run_batch_item_trace_context_absent_decodes_to_none() {
        // Old Python (or an untraced request) omits the trace keys
        // entirely — new Rust must decode them as `None` via the
        // additive `#[serde(default)]` contract.
        let bytes = rmp_serde::to_vec_named(&serde_json::json!({
            "op": "encode",
            "work_item_id": "r.0",
            "request_id": "r",
            "item_index": 0,
        }))
        .unwrap();
        let back: RunBatchItem = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.op, "encode");
        assert!(back.traceparent.is_none());
        assert!(back.tracestate.is_none());
    }

    #[test]
    fn process_extract_batch_request_preserves_document_bytes() {
        let pdf_bytes = b"%PDF-1.4 tiny".to_vec();
        let req = ProcessExtractBatchRequest {
            model_id: "docling".into(),
            items: vec![ExtractBatchItem {
                work_item_id: "r.0".into(),
                request_id: "r".into(),
                item_index: 0,
                total_items: 1,
                timestamp: 0.0,
                item: WireValue::Map(vec![(
                    WireValue::from("document"),
                    WireValue::Map(vec![
                        (
                            WireValue::from("data"),
                            WireValue::Binary(pdf_bytes.clone()),
                        ),
                        (WireValue::from("format"), WireValue::from("pdf")),
                    ]),
                )]),
                labels: None,
                output_schema: None,
                instruction: None,
                options: None,
                profile_id: None,
                bundle_config_hash: None,
                payload_fetch_ms: 0.0,
            }],
        };

        let bytes = rmp_serde::to_vec_named(&req).unwrap();
        let decoded: rmpv::Value = rmp_serde::from_slice(&bytes).unwrap();
        let rmpv::Value::Map(root) = decoded else {
            panic!("request should be a msgpack map");
        };
        let items = root
            .iter()
            .find_map(|(key, value)| {
                if matches!(key, rmpv::Value::String(s) if s.as_str() == Some("items")) {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("items field");
        let rmpv::Value::Array(items) = items else {
            panic!("items should be an array");
        };
        let Some(rmpv::Value::Map(first)) = items.first() else {
            panic!("first item should be a map");
        };
        let document = first
            .iter()
            .find_map(|(key, value)| {
                if matches!(key, rmpv::Value::String(s) if s.as_str() == Some("item")) {
                    Some(value)
                } else {
                    None
                }
            })
            .and_then(|item| match item {
                rmpv::Value::Map(fields) => fields.iter().find_map(|(key, value)| {
                    if matches!(key, rmpv::Value::String(s) if s.as_str() == Some("document")) {
                        Some(value)
                    } else {
                        None
                    }
                }),
                _ => None,
            })
            .expect("document field");
        let rmpv::Value::Map(document) = document else {
            panic!("document should be a map");
        };
        let data = document
            .iter()
            .find_map(|(key, value)| {
                if matches!(key, rmpv::Value::String(s) if s.as_str() == Some("data")) {
                    Some(value)
                } else {
                    None
                }
            })
            .expect("document.data");
        assert_eq!(data, &rmpv::Value::Binary(pdf_bytes));
    }

    #[test]
    fn ping_response_decodes_from_minimal_map() {
        let bytes = rmp_serde::to_vec_named(&serde_json::json!({
            "timestamp_ms": 123.0,
            "worker_id": "w-1",
        }))
        .unwrap();
        let back: PingResponse = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back.worker_id, "w-1");
        assert_eq!(back.timestamp_ms, 123.0);
        assert!(!back.ready);
        assert_eq!(back.bundle_config_hash, "");
        assert!(back.loaded_models.is_empty());
    }

    #[test]
    fn replace_model_configs_request_requires_models_field() {
        let missing_models = rmp_serde::to_vec_named(&serde_json::json!({
            "bundle_id": "default",
            "epoch": 7,
            "bundle_config_hash": "",
        }))
        .unwrap();
        assert!(
            rmp_serde::from_slice::<ReplaceModelConfigsRequest>(&missing_models).is_err(),
            "missing models must not decode as an authoritative empty snapshot"
        );

        let explicit_empty = rmp_serde::to_vec_named(&serde_json::json!({
            "bundle_id": "default",
            "epoch": 7,
            "bundle_config_hash": "",
            "models": [],
        }))
        .unwrap();
        let decoded: ReplaceModelConfigsRequest = rmp_serde::from_slice(&explicit_empty).unwrap();
        assert!(decoded.models.is_empty());
    }
}
