from __future__ import annotations

from typing import Any, Literal

import msgspec

# -----------------------------------------------------------------------------
# Envelope
# -----------------------------------------------------------------------------

IPC_VERSION: int = 1


# Incoming request frames are parsed directly as dicts in
# ``ipc_server._dispatch_frame`` — a typed ``RequestEnvelope`` struct
# sits in the Rust client (see ``sie_server_sidecar::ipc_types``) but was
# unused on the Python side. Responses DO use a typed struct, below.


class ResponseEnvelope(msgspec.Struct, tag_field="kind", tag="response"):
    version: int
    request_id: str
    ok: bool
    body: dict[str, Any] | None = None
    error: str | None = None


class IpcResponseChunkV1(msgspec.Struct):
    """One negotiated chunk of an oversized serialized ``ResponseEnvelope``.

    ``payload`` is a byte slice of the complete, already-serialized response.
    The sidecar must reconstruct those exact bytes and verify the SHA-256
    ``transfer_digest`` before decoding the ordinary response envelope.
    """

    version: int
    request_id: str
    transfer_digest: bytes
    chunk_index: int
    chunk_count: int
    total_bytes: int
    payload: bytes
    kind: str = "ipc_response_chunk_v1"


# -----------------------------------------------------------------------------
# Methods
# -----------------------------------------------------------------------------

METHOD_PING = "Ping"
METHOD_ENSURE_MODEL_READY = "EnsureModelReady"
METHOD_PROCESS_ENCODE_BATCH = "ProcessEncodeBatch"
METHOD_PROCESS_SCORE_BATCH = "ProcessScoreBatch"
METHOD_PROCESS_EXTRACT_BATCH = "ProcessExtractBatch"
METHOD_PROCESS_GENERATE = "ProcessGenerate"
METHOD_WORKER_CAPABILITIES = "WorkerCapabilities"
METHOD_SIGNAL_GENERATE_CANCEL = "SignalGenerateCancel"
# RPC that accepts a whole pre-formed batch (single-op, mixed items)
# and returns ``BatchOutcome`` unchanged.
# Coexists with the per-op Process* RPCs: the current sie-server-sidecar image
# uses ``RunBatch`` exclusively, while the per-op RPCs stay in the wire
# contract for focused IPC parity tests and Python-framed callers.
# Direct ``sie-server`` HTTP execution does not use this IPC boundary.
METHOD_RUN_BATCH = "RunBatch"
METHOD_APPLY_MODEL_CONFIG = "ApplyModelConfig"
METHOD_REPLACE_MODEL_CONFIGS = "ReplaceModelConfigs"
METHOD_SET_PINNED_MODELS = "SetPinnedModels"
METHOD_DRAIN = "Drain"


# -----------------------------------------------------------------------------
# Ping
# -----------------------------------------------------------------------------


class PingRequest(msgspec.Struct):
    timestamp_ms: float


class PingResponse(msgspec.Struct):
    timestamp_ms: float
    worker_id: str
    ready: bool = False
    bundle_config_hash: str = ""
    loaded_models: list[str] = msgspec.field(default_factory=list)


# -----------------------------------------------------------------------------
# EnsureModelReady
# -----------------------------------------------------------------------------

# ``"failed"`` is the terminal, non-retryable state (the registry holds a
# PERMANENT ``LoadFailure``, ``cooldown=permanent``). The sidecar
# dead-letters a group in this state as ``MODEL_LOAD_FAILED`` (gateway ->
# 502) instead of re-driving ``EnsureModelReady`` forever — the fast-path
# twin of the ``run_batch`` / direct-HTTP #1786 typed-502 mapping.
ReadinessState = Literal["ready", "loading_started", "loading_in_progress", "retry_later", "failed"]


class EnsureModelReadyRequest(msgspec.Struct):
    model_id: str


# ---------------------------------------------------------------------------
# Model descriptor
# ---------------------------------------------------------------------------
#
# Carried on ``EnsureModelReadyResponse`` to let the worker-sidecar discover
# per-model capabilities at runtime instead of from startup env vars. Every
# field is optional/defaulted because capabilities are model-dependent. See
# ``packages/sie_server_sidecar/docs/architecture-guide.md`` for the contract.


class ModelDescriptor(msgspec.Struct):
    # Absolute path to ``tokenizer.json`` on a filesystem the sidecar can
    # read (single-container deploy: HF cache or any sidecar-readable path;
    # split-container: shared ``emptyDir``). ``None`` → adapter has not
    # materialised its tokeniser to a sidecar-visible path; the sidecar
    # skips Rust-side tokenisation for this model and Python tokenises.
    tokenizer_path: str | None = None
    # BLAKE3 (truncated to 32 hex chars) of the canonical tokenizer JSON.
    # Lets the sidecar verify byte-identity with the adapter's loaded
    # tokeniser before enabling the fast path.
    tokenizer_id: str | None = None
    # Per-model truncation cap. ``None`` falls back to the registry's
    # ``DEFAULT_MAX_SEQ_LEN`` on the Rust side.
    max_seq_len: int | None = None
    # Output keys the adapter knows how to emit (``"dense"``, ``"sparse"``,
    # ``"multivector"``, ``"score"``). Informational; per-request shape
    # checks in ``queue_executor._maybe_*_raw_output`` are still the
    # authoritative gate for whether a given request is framed in Rust.
    output_types: list[str] = msgspec.field(default_factory=list)
    # True iff this adapter has implemented the ``RunBatch`` IPC method.
    # False for adapters still on the per-op ``Process*Batch`` RPCs.
    supports_run_batch: bool = False
    # Model-default ``query_template`` (per-adapter ctor arg, sourced
    # from the model YAML). Used by the sidecar to apply asymmetric-
    # retrieval prompts in Rust before tokenisation when an item
    # arrives with ``is_query=True``, matching the precedence in
    # ``_utils.extract_texts`` (template > bare instruction >
    # passthrough). ``None`` keeps the legacy behaviour: Python applies
    # templates and the sidecar either skips ``prepared_tokens`` for
    # the item or matches the unaltered text.
    #
    # Per-request ``options.query_template`` still wins; this is only
    # the model-default fallback. Wire format is the raw Python
    # ``str.format``-style template (e.g. ``"query: {text}"`` or
    # ``"Instruct: {instruction}\nQuery: {text}"``).
    default_query_template: str | None = None
    # Model-default ``doc_template`` (per-adapter ctor arg). Same wire
    # contract as ``default_query_template`` but selected when
    # ``is_query=False``.
    default_doc_template: str | None = None


class EnsureModelReadyResponse(msgspec.Struct):
    state: ReadinessState
    # Per-model batch budget for fair dispatch. The worker-sidecar uses this to
    # cap how many messages for this model it dispatches to the Python side
    # in a single batch (excess are NAK'd for redelivery, possibly to another
    # worker).
    #
    # ``None`` when the model is not yet loaded (state != "ready") — the
    # caller should NAK the group and re-query later.
    batch_budget: int | None = None
    # Model descriptor. Populated by Python adapters that expose a
    # HF fast tokeniser + structured ``output_types``; ``None`` for non-text /
    # image / audio adapters that do not emit it.
    descriptor: ModelDescriptor | None = None


# -----------------------------------------------------------------------------
# ApplyModelConfig
# -----------------------------------------------------------------------------


class ApplyModelConfigRequest(msgspec.Struct):
    bundle_id: str
    model_id: str
    epoch: int
    bundle_config_hash: str
    model_config: str
    profiles_added: list[str] = msgspec.field(default_factory=list)


class ApplyModelConfigResponse(msgspec.Struct):
    applied: bool
    bundle_config_hash: str
    config_version: int = 0


class ReplaceModelConfigEntry(msgspec.Struct):
    model_id: str
    model_config: str


class ReplaceModelConfigsRequest(msgspec.Struct):
    bundle_id: str
    epoch: int
    bundle_config_hash: str
    models: list[ReplaceModelConfigEntry]


class ReplaceModelConfigsResponse(msgspec.Struct):
    applied: bool
    bundle_config_hash: str
    config_version: int = 0
    applied_models: list[str] = msgspec.field(default_factory=list)
    applied_profiles: list[str] = msgspec.field(default_factory=list)


# -----------------------------------------------------------------------------
# SetPinnedModels
# -----------------------------------------------------------------------------
#
# Runtime delivery of a pool's pinned-model set from the gateway to the worker.
# The worker-sidecar polls the pool's ``PoolSpec.pinned_models`` (the single
# source of truth, set via the /v1/pools API or SIE_GATEWAY_STATIC_QUEUE_POOLS)
# and pushes it here on change. ``models`` carries gateway-canonical ids (bare
# ``sie_id`` or profile-qualified ``sie_id:profile``); the registry normalizes
# them. The set is authoritative and REPLACES the worker's current pinned set.


class SetPinnedModelsRequest(msgspec.Struct):
    models: list[str]


class SetPinnedModelsResponse(msgspec.Struct):
    applied: bool
    pinned_count: int = 0


# -----------------------------------------------------------------------------
# ProcessGenerate — streaming generation over IPC
# -----------------------------------------------------------------------------

GenerateEventKind = Literal["publish", "ack", "nak", "in_progress", "done"]


class ProcessGenerateRequest(msgspec.Struct):
    model_id: str
    work_item_msgpack: bytes


class GenerateEvent(msgspec.Struct):
    kind: GenerateEventKind
    reply_subject: str = ""
    payload: bytes = b""
    delay_ms: int | None = None
    error: str | None = None


class WorkerCapabilitiesRequest(msgspec.Struct):
    pass


class WorkerCapabilitiesResponse(msgspec.Struct):
    has_generation_models: bool = False
    generation_models: list[str] = msgspec.field(default_factory=list)
    supported_models: list[str] = msgspec.field(default_factory=list)
    loaded_models: list[str] = msgspec.field(default_factory=list)


class SignalGenerateCancelRequest(msgspec.Struct):
    request_id: str


class SignalGenerateCancelResponse(msgspec.Struct):
    matched: bool = False


# -----------------------------------------------------------------------------
# PreparedTokens — Rust-side fast-path tokenisation
# -----------------------------------------------------------------------------
#
# When the worker-sidecar has learned a tokenizer for the target model
# from ``ModelDescriptor``, it pre-tokenises ``item.text`` and attaches the
# ragged, unpadded token arrays here. The Python adapter's fast-path uses
# them verbatim iff ``tokenizer_id`` matches the BLAKE3 content hash of its
# own ``tokenizer.json`` bytes; otherwise it falls back to
# ``TextPreprocessor.prepare()`` exactly like today.
#
# ``prepared_tokens`` is optional because the worker-sidecar only attaches it
# when tokenizer materialization and hashing succeed; otherwise Python tokenises.
#
# Wire-size policy notes (mirror ``dispatcher.rs::rag_to_wire``):
#   * ``attention_mask`` can be empty → Python rebuilds it as all-ones
#     per sequence (safe because all sequences are truncated at
#     ``max_seq_len`` and contain no padding).
#   * ``token_type_ids`` can be empty → Python treats that as "all
#     zeros" (the BERT convention for single-segment text), which
#     matches what every in-tree adapter currently synthesises.


class PreparedTokens(msgspec.Struct):
    # Ragged batch: outer = batch size, inner = this sequence's length.
    # Already truncated to ``max_seq_len`` on the Rust side; NOT padded.
    input_ids: list[list[int]]
    # Stable content hash of ``tokenizer.json`` bytes (BLAKE3, first
    # 32 hex chars). Python compares against its own loaded-tokenizer
    # hash; on mismatch, Python re-tokenises and ``prepared_tokens``
    # is discarded.
    tokenizer_id: str
    attention_mask: list[list[int]] = msgspec.field(default_factory=list)
    token_type_ids: list[list[int]] = msgspec.field(default_factory=list)
    # The ``max_length`` Rust applied when truncating. Python uses
    # this only for drift detection — not for re-truncation.
    max_seq_len: int = 0


# -----------------------------------------------------------------------------
# ProcessEncodeBatch
# -----------------------------------------------------------------------------


class EncodeBatchItem(msgspec.Struct):
    work_item_id: str
    request_id: str
    item_index: int
    total_items: int
    timestamp: float
    # Resolved payload: either the inline item dict or the payload_ref-resolved dict.
    item: dict[str, Any]
    # Sub-grouping keys. Items that share the same tuple of
    # (output_types, instruction, is_query, options) can be run as a
    # single EncodePipeline call.
    output_types: list[str] | None = None
    instruction: str | None = None
    is_query: bool = False
    options: dict[str, Any] | None = None
    profile_id: str | None = None
    bundle_config_hash: str | None = None
    payload_fetch_ms: float = 0.0
    # Optional pre-tokenised payload populated by the worker-sidecar
    # (see ``PreparedTokens`` above). When present and the tokenizer
    # id matches, the Python adapter skips its own tokenisation.
    prepared_tokens: PreparedTokens | None = None


class ProcessEncodeBatchRequest(msgspec.Struct):
    model_id: str
    items: list[EncodeBatchItem]
    # Opt-in for the additive shared-buffer multivector response. Keeping
    # this false by default preserves a safe per-item fallback while a
    # worker and sidecar are rolled independently.
    accepts_batched_f16_multivectors: bool = False


# -----------------------------------------------------------------------------
# ProcessScoreBatch
# -----------------------------------------------------------------------------


class ScoreBatchItem(msgspec.Struct):
    work_item_id: str
    request_id: str
    item_index: int
    total_items: int
    timestamp: float
    query_item: dict[str, Any]
    score_items: list[dict[str, Any]]
    instruction: str | None = None
    options: dict[str, Any] | None = None
    profile_id: str | None = None
    payload_fetch_ms: float = 0.0
    # Rust-side fast-path tokenisation. Wire layout matches the Rust
    # dispatcher: ``input_ids[0]`` is the query, ``input_ids[1..]`` are
    # the N score docs in the same order as ``score_items``.
    prepared_tokens: PreparedTokens | None = None


class ProcessScoreBatchRequest(msgspec.Struct):
    model_id: str
    items: list[ScoreBatchItem]


# -----------------------------------------------------------------------------
# ProcessExtractBatch
# -----------------------------------------------------------------------------


class PreparedAudioPcm16(msgspec.Struct):
    pcm_s16le: bytes
    sample_rate: int
    sample_count: int
    duration_ms: int
    source_sample_rate: int
    source_sample_count: int
    source_channels: int
    container: str


class ExtractBatchItem(msgspec.Struct):
    work_item_id: str
    request_id: str
    item_index: int
    total_items: int
    timestamp: float
    item: dict[str, Any]
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    instruction: str | None = None
    options: dict[str, Any] | None = None
    profile_id: str | None = None
    bundle_config_hash: str | None = None
    payload_fetch_ms: float = 0.0
    prepared_audio: PreparedAudioPcm16 | None = None


class ProcessExtractBatchRequest(msgspec.Struct):
    model_id: str
    items: list[ExtractBatchItem]


# -----------------------------------------------------------------------------
# Batch outcome — returned by all ProcessXxxBatch RPCs
# -----------------------------------------------------------------------------

Disposition = Literal["publish_and_ack", "publish_error_and_ack", "nak_retry"]


# -----------------------------------------------------------------------------
# RawOutput — Rust-side result framing fast-path
# -----------------------------------------------------------------------------
#
# Mirrors ``sie_server_sidecar::ipc_types::RawOutput``. When the Python
# adapter emits one of these typed shapes, Rust owns final wire framing
# instead of Python packing the final ``msgpack`` envelope itself.
#
# Wire contract (non-destructive, additive):
#   * ``raw_output`` defaults to ``None`` on ``ItemOutcome``.
#   * Every inner variant is ``Optional``; new variants (sparse,
#     multivector, extract JSON, ...) are added as further
#     ``Optional`` fields without breaking clients.
#   * When ``raw_output`` is set, ``result_msgpack`` MUST be empty —
#     the Rust side treats a populated ``result_msgpack`` as
#     authoritative (rolling-deploy safety, see
#     ``publisher::shape_raw_output_for_wire``).
#   * On any shape error the Rust publisher converts the outcome to
#     ``publish_error_and_ack`` with ``error_code = "raw_output_shape_error"``.
#     We never silently drop a request.


class DenseOutput(msgspec.Struct):
    # Flat ``[dim]`` float32 embedding for a single encode item.
    # ``len(values) == dim`` is enforced by the Rust shaper.
    values: list[float]
    dim: int
    # If True, the Rust shaper L2-normalises ``values`` before
    # packing. For v1 the Python adapter continues to normalise
    # itself and sets this to False; kept on the wire so a later
    # stage can delegate the step without a schema bump.
    normalize: bool = False


class ScoreOutputRaw(msgspec.Struct):
    # Parallel lists in input order; the Rust shaper sorts desc by
    # ``scores[i]`` (stable tie-break) and assigns ``rank = 0..``.
    scores: list[float]
    item_ids: list[str]


class SparseOutput(msgspec.Struct):
    # Parallel non-zero ``(indices, values)`` pair for one encode
    # item. Mirrors the adapter-side ``SparseVector`` exactly
    # (``np.int32`` indices, ``np.float32`` values), which every
    # in-tree sparse encoder already produces.
    #
    # Invariants enforced by the Rust shaper:
    #   * ``len(indices) == len(values)``
    #   * indices fit in int32 (adapter convention)
    # Violations surface as ``raw_output_shape_error`` on the wire.
    indices: list[int]
    values: list[float]
    # ``None`` → packed as msgpack ``nil`` to match the legacy
    # ``{"dims": None, ...}`` convention for vocab-less sparse.
    dims: int | None = None


class MultivectorOutput(msgspec.Struct):
    # Flattened ``[num_tokens, token_dims]`` matrix in C (row-major)
    # order. ``dtype`` defaults to ``float32`` for older producers;
    # ``float16`` asks the Rust shaper to round values and write an
    # f16 msgpack-numpy sentinel.
    # ``values_f16`` is an optional little-endian f16 byte buffer for
    # native Rust/Candle producers that already materialized half
    # values; Python producers can keep using ``values``.
    #
    values: list[float]
    num_tokens: int
    token_dims: int
    dtype: str = "float32"
    values_f16: bytes = b""


class RawOutput(msgspec.Struct):
    # At most one variant is populated per outcome — the Python
    # safety gates (``_maybe_*_raw_output``) mint a ``RawOutput``
    # with exactly the one field that matched a fast-path-eligible
    # adapter output. All-None is treated as a typed error
    # (``raw_output_shape_error``) by the Rust publisher — a
    # rolling-deploy forward-compat guard against a future variant
    # the current Rust build doesn't know.
    dense: DenseOutput | None = None
    score: ScoreOutputRaw | None = None
    sparse: SparseOutput | None = None
    multivector: MultivectorOutput | None = None


# -----------------------------------------------------------------------------
# ItemOutcome / BatchOutcome
# -----------------------------------------------------------------------------


class UnitCounts(msgspec.Struct):
    """Authoritative billable-unit counts for one work item.

    Emitted by the engine on the result path so metering edges (gateways,
    usage pipelines) never re-derive units from estimates: ``input_tokens``
    is the REAL tokenizer count taken post-tokenization (``PreparedItem.cost``
    for text preprocessing), not a bytes/4-style approximation.

    All fields are optional: a field is set only when the engine has an
    authoritative count for it. ``pages`` (docling parse/OCR) and ``images``
    (vision extract/encode) are populated by the queue executor when the
    pipeline surfaces per-item counts (see ``queue_executor._with_pages`` /
    ``_with_images`` / ``_per_item_image_counts``). ``audio_ms`` is the exact
    accepted audio duration in integer milliseconds; it is populated only
    after media validation/preprocessing accepts the input. Dimensions stay
    ``None`` for work that carries no such count.
    """

    input_tokens: int | None = None
    pages: int | None = None
    images: int | None = None
    audio_ms: int | None = None
    # Additive tail field: never insert before the four legacy positional slots.
    pairs: int | None = None


class ItemOutcome(msgspec.Struct):
    work_item_id: str
    request_id: str
    item_index: int
    disposition: Disposition
    nak_delay_ms: int | None = None
    # Opaque msgpack bytes (what the SDK expects as WorkResult.result_msgpack).
    # The executor produces this before handing settlement back to the sidecar.
    result_msgpack: bytes | None = None
    error: str | None = None
    error_code: str | None = None
    inference_ms: float | None = None
    tokenization_ms: float | None = None
    postprocessing_ms: float | None = None
    # Typed raw inference output. When set, ``result_msgpack`` MUST be
    # empty and the Rust publisher produces the final wire bytes. See
    # ``RawOutput`` above for the fallback matrix.
    raw_output: RawOutput | None = None
    # Authoritative billable-unit counts (see ``UnitCounts``). Optional and
    # appended last: msgspec encodes Structs as msgpack maps, so decoders
    # that don't know the key (older Rust sidecars) ignore it and older
    # producers simply omit it — the NATS wire contract is unchanged.
    units: UnitCounts | None = None


class BatchedF16MultivectorItem(msgspec.Struct):
    work_item_id: str
    byte_offset: int
    byte_len: int
    num_tokens: int
    token_dims: int


class BatchedF16MultivectorOutput(msgspec.Struct):
    values_f16: bytes
    items: list[BatchedF16MultivectorItem]


class BatchOutcome(msgspec.Struct):
    outcomes: list[ItemOutcome]
    # Native Candle f16 multivectors can share one contiguous buffer across
    # a batch. The sidecar owns slicing and final public wire framing.
    batched_f16_multivectors: list[BatchedF16MultivectorOutput] = msgspec.field(default_factory=list)


# -----------------------------------------------------------------------------
# RunBatch
# -----------------------------------------------------------------------------
#
# Mirrors ``sie_server_sidecar::ipc_types::{RunBatchRequest, RunBatchItem}``.
# See the Rust definitions for the design rationale; this file is the
# decode side.
#
# Invariants enforced by ``adapter_call_loop.handle_run_batch``:
#   * ``items`` is non-empty.
#   * Every item's ``op`` discriminator matches exactly one populated
#     optional field (encode / score / extract) and all other optional
#     fields are ``None``.
#   * All items in a single request share the same ``op`` — the
#     Rust scheduler only flushes one op per batch; a mixed-op
#     request is an invariant violation and is rejected with
#     ``error_code = "run_batch_mixed_op"``.
#
# Unknown ``op`` values surface as ``error_code = "run_batch_unknown_op"``
# (forward-compat: a newer Rust scheduler could ship an op this Python
# doesn't understand, and we want the operator to see a clean error
# rather than a confusing traceback).

RunBatchOp = Literal["encode", "score", "extract"]


class RunBatchItem(msgspec.Struct):
    op: RunBatchOp
    # Identity copied from the wrapped per-op item. Redundant on the
    # happy path, but needed for deterministic per-item errors if a
    # malformed frame omits the wrapped encode / score / extract
    # payload.
    work_item_id: str = ""
    request_id: str = ""
    item_index: int = 0
    encode: EncodeBatchItem | None = None
    score: ScoreBatchItem | None = None
    extract: ExtractBatchItem | None = None
    # W3C trace context copied off the originating WorkItem by the Rust
    # scheduler so the non-streaming worker loop can re-extract the
    # gateway span and attach ``worker.run_batch`` as its child.
    # ``None`` when the gateway didn't propagate a trace.
    traceparent: str | None = None
    tracestate: str | None = None


class RunBatchRequest(msgspec.Struct):
    model_id: str
    # Rust-assigned batch id (u64 on the wire; int here).
    batch_id: int
    # Empty string means base model (no LoRA).
    lora_key: str
    # Sum of per-item costs, as computed by the Rust scheduler.
    total_cost: int
    items: list[RunBatchItem]
    # Same capability gate as ``ProcessEncodeBatchRequest``, carried by the
    # scheduler's production RPC path.
    accepts_batched_f16_multivectors: bool = False


# -----------------------------------------------------------------------------
# Drain
# -----------------------------------------------------------------------------


# DrainRequest only exists over the wire; the Python server reads the
# `deadline_ms` field out of the incoming dict directly (see
# ``ipc_server._handle_drain``). A typed struct parallel lives in the
# Rust client.


class DrainResponse(msgspec.Struct):
    acknowledged: bool = True
