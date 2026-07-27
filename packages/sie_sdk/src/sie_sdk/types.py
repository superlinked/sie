"""Client-side type definitions for SIE SDK.

These types are lightweight TypedDicts for type hints. No validation is performed
client-side - the server validates all inputs.

These types support flexible Python inputs
(file paths, PIL images, numpy arrays, bytes) which the client converts for transport.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Never, NotRequired, Required, TypedDict

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image

# Output dtype options.
DType = Literal["float32", "float16", "bfloat16", "int8", "uint8", "binary", "ubinary"]

# Output dtypes we support for casting
# Note: bfloat16 is compute-only, not returned
OutputDType = Literal["float32", "float16", "int8", "uint8", "binary", "ubinary"]

# Default output dtype
DEFAULT_OUTPUT_DTYPE: OutputDType = "float32"

# Output type options
OutputType = Literal["dense", "sparse", "multivector"]

# Model state (for status messages).
# ``failed`` is the terminal branch added for non-retryable load failures
# (gated repos, missing dependencies, etc.). See sie-test#85 for context.
ModelState = Literal["available", "loading", "loaded", "unloading", "failed"]


def np_to_dtype(arr: np.ndarray) -> DType:
    dt = arr.dtype
    if dt == np.float32:
        return "float32"
    if dt == np.float16:
        return "float16"
    if dt == np.int8:
        return "int8"
    if dt == np.uint8:
        return "uint8"
    return "float32"


class ImageInput(TypedDict, total=False):
    """Image input for multimodal models.

    Accepts various formats that the SDK converts for transport.
    """

    data: Image.Image | NDArray[Any] | bytes | str | Path
    format: str  # "jpeg", "png" - inferred if not provided


class AudioInput(TypedDict, total=False):
    """Audio input for audio models."""

    data: bytes | NDArray[Any] | str | Path
    format: str  # "wav", "mp3" - required if data is bytes
    sample_rate: int  # required if data is np.ndarray


class VideoInput(TypedDict, total=False):
    """Video input for video models."""

    data: bytes | str | Path
    format: str  # "mp4", "webm" - inferred from path


class DocumentInput(TypedDict, total=False):
    """Document input for composite-document extractors (PDF, DOCX, HTML, ...).

    Wraps a raw document payload. The SDK accepts bytes or a path; format is
    inferred from the path suffix when not explicitly provided.
    """

    data: bytes | str | Path
    format: str  # "pdf", "docx", "html", "md", "txt", ... - inferred from path


class Item(TypedDict, total=False):
    """A single item to encode, score, or extract from.

    Items can contain text, images, audio, or video.
    Most models operate on text only, but multimodal models can process images.

    For simple text encoding, just use {"text": "your text here"}.

    For ColBERT/late interaction scoring with pre-encoded multivectors,
    pass the multivector directly to avoid re-encoding:
        {"multivector": query_result["multivector"]}

    Examples:
        # Simple text
        {"text": "Hello world"}

        # With ID for tracking
        {"id": "doc-1", "text": "Document text"}

        # Multimodal (for ColPali, CLIP, etc.)
        {"text": "Description", "images": ["photo.jpg"]}
        {"images": [Image.open("photo.jpg")]}

        # Pre-encoded multivector (for use with sie_sdk.scoring.maxsim)
        {"multivector": np.array([[0.1, 0.2, ...], ...])}
    """

    id: str
    text: str
    images: Sequence[ImageInput | Image.Image | NDArray[Any] | bytes | str | Path]
    audio: AudioInput | bytes | str | Path
    video: VideoInput | bytes | str | Path
    document: DocumentInput | bytes | str | Path
    metadata: dict[str, Any]
    # Pre-encoded multivector (for use with scoring.maxsim).
    multivector: NDArray[np.float16] | NDArray[np.float32]


class SparseResult(TypedDict):
    """Sparse vector result with non-zero indices and values."""

    indices: NDArray[np.int32]
    values: NDArray[np.float32]


class TimingInfo(TypedDict, total=False):
    """Server-side timing breakdown for a request."""

    total_ms: float
    queue_ms: float
    tokenization_ms: float
    inference_ms: float


class RequestUsage(TypedDict, total=False):
    """Authoritative metered units for one completed inference request."""

    input_tokens: int
    pairs: int
    images: int
    pages: int
    output_tokens: int
    audio_ms: int


class RequestMetadata(TypedDict, total=False):
    """Gateway metadata from the successful terminal response.

    For batch encode/extract calls, each item repeats this request-scoped
    metadata; the values describe the whole HTTP request, not one item.
    """

    id: str
    usage: RequestUsage
    credits_debited: int
    execution_identity_sha256: str


class EncodeResult(TypedDict, total=False):
    """Result of encoding a single item.

    Contains the item ID (if provided) and one or more
    output representations depending on what was requested.

    Attributes:
        model: Model identity returned by the server.
        id: Item ID (echoed from request if provided).
        dense: Dense embedding as numpy array, shape [dims].
        sparse: Sparse embedding with indices and values.
        multivector: Multi-vector embedding as a float16 or float32 numpy array,
            shape [num_tokens, token_dims]. The resolved model profile chooses
            the default dtype; an explicit output_dtype request overrides it.
        request: Request-scoped id, metered usage, and settled debit when
            supplied by the gateway.
    """

    model: str
    id: str
    dense: NDArray[np.float32]
    sparse: SparseResult
    multivector: NDArray[np.float16] | NDArray[np.float32]
    timing: TimingInfo
    request: RequestMetadata


class ModelDims(TypedDict, total=False):
    """Model dimension information."""

    dense: int
    sparse: int
    multivector: int


class ModelCapabilities(TypedDict, total=False):
    """Advertised model capabilities.

    Mirrors the gateway ``capabilities`` object on each ``/v1/models``
    entry (``ModelCapabilitiesWire``). All keys are optional; their
    presence depends on what the model config declares. ``grammar`` is
    the list of supported grammar kinds (``json_schema`` | ``regex`` |
    ``ebnf``). ``code``/``sql``/``guard`` are informational flags that
    advertise validated generation jobs and back the ``model="code"`` /
    ``model="sql"`` / ``model="guard"`` aliases.

    These flags mean the model *supports* a task — they are NOT a
    precision-independent quality SLA. A flag is true at the model level even
    when quality is profile/precision-dependent (e.g. ``sql`` quality regresses
    under FP8; route SQL-critical traffic to a BF16 bundle via the ``sql``
    alias). Treat them as "can do this", not "guaranteed to score X".
    """

    grammar: list[str]
    tools: bool
    lora_adapters: list[str]
    profile_lora_adapters: dict[str, list[str]]
    code: bool
    sql: bool
    guard: bool


class ModelInfo(TypedDict, total=False):
    """Information about a model returned by list_models().

    Note: Server returns flat structure with inputs/outputs at top level.
    """

    name: str
    loaded: bool
    inputs: list[str]  # ["text"], ["text", "image"], etc.
    outputs: list[str]  # ["dense"], ["dense", "sparse"], etc.
    dims: ModelDims
    max_sequence_length: int
    revision: str | None
    capabilities: ModelCapabilities


class ScoreEntry(TypedDict):
    """A single score entry from the reranker.

    Attributes:
        item_id: ID of the item (from request or auto-generated).
        score: Relevance score (higher = more relevant).
        rank: Position in sorted order (0 = most relevant).
    """

    item_id: str
    score: float
    rank: int


class ScoreUsage(TypedDict):
    """Authoritative usage reported by a score adapter."""

    input_tokens: Required[int]
    images: NotRequired[int]


class ScoreResult(TypedDict, total=False):
    """Result of scoring items against a query.

    Attributes:
        model: Model used for scoring.
        query_id: Query ID (echoed from request if provided).
        scores: List of score entries, sorted by relevance (descending).
        request: Request-scoped id, metered usage, and settled debit when
            supplied by the gateway.
    """

    model: str
    query_id: str
    scores: list[ScoreEntry]
    usage: ScoreUsage
    request: RequestMetadata


class Entity(TypedDict, total=False):
    """A single extracted entity (NER span or document region).

    Attributes:
        text: The extracted text span.
        label: Entity type/label (e.g., "person", "organization").
        score: Confidence score.
        start: Start character offset in the original text (None for image-based).
        end: End character offset in the original text (None for image-based).
        bbox: Bounding box [x, y, width, height] in pixels for image regions.
    """

    text: str
    label: str
    score: float
    start: int | None
    end: int | None
    bbox: list[int] | None


class Relation(TypedDict):
    """A single extracted relation triple.

    Attributes:
        head: Head entity text.
        tail: Tail entity text.
        relation: Relation type/label.
        score: Confidence score.
    """

    head: str
    tail: str
    relation: str
    score: float


class Classification(TypedDict):
    """A single classification result.

    Attributes:
        label: Classification label.
        score: Confidence score.
    """

    label: str
    score: float


class DetectedObject(TypedDict):
    """A single detected object with bounding box.

    Attributes:
        label: Object class label.
        score: Confidence score.
        bbox: Bounding box [x, y, width, height] in pixels.
    """

    label: str
    score: float
    bbox: list[int]


class ExtractItemErrorDetail(TypedDict):
    """Stable per-item extraction failure."""

    code: str
    message: str


class ExtractResult(TypedDict, total=False):
    """Result of extraction for a single item.

    Attributes:
        model: Model identity returned by the server.
        id: Item ID (echoed from request or auto-generated).
        entities: List of extracted entities (NER spans).
        relations: List of extracted relation triples.
        classifications: List of classification results.
        objects: List of detected objects with bounding boxes.
        data: Additional structured extraction data (if output_schema was provided).
        error: Stable per-item failure when extraction did not complete.
        request: Request-scoped id, metered usage, and settled debit when
            supplied by the gateway.
    """

    model: str
    id: str
    entities: list[Entity]
    relations: list[Relation]
    classifications: list[Classification]
    objects: list[DetectedObject]
    data: dict[str, Any]
    error: ExtractItemErrorDetail
    request: RequestMetadata


# Streaming generation result. Streaming happens inside the gateway/worker;
# the SDK surfaces the aggregated outcome plus SIE-native timing metadata
# (TTFT, TPOT, attempt_id). Future SDK additions may surface chunks as an
# async iterator.
FinishReason = Literal["stop", "length", "cancelled", "content_filter", "error"]


class GenerateImage(TypedDict):
    """One image paired with a native generation prompt."""

    data: Image.Image | NDArray[Any] | bytes | str | Path
    format: NotRequired[str]


class JsonSchemaGrammar(TypedDict):
    """Constrain generation to a JSON Schema."""

    json_schema: dict[str, Any]
    regex: NotRequired[Never]
    ebnf: NotRequired[Never]
    label: NotRequired[str | None]
    strict: NotRequired[bool | None]


class RegexGrammar(TypedDict):
    """Constrain generation to a regular expression."""

    json_schema: NotRequired[Never]
    regex: str
    ebnf: NotRequired[Never]
    label: NotRequired[str | None]
    strict: NotRequired[bool | None]


class EbnfGrammar(TypedDict):
    """Constrain generation to an EBNF grammar."""

    json_schema: NotRequired[Never]
    regex: NotRequired[Never]
    ebnf: str
    label: NotRequired[str | None]
    strict: NotRequired[bool | None]


GenerateGrammar = JsonSchemaGrammar | RegexGrammar | EbnfGrammar
"""Native structured-output grammar. Exactly one grammar variant is set."""


class GenerationUsage(TypedDict):
    """Token usage for a single generation call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResult(TypedDict, total=False):
    """Aggregated generation result returned by :meth:`SIEClient.generate`.

    Attributes:
        model: Model id the gateway dispatched to.
        text: Full generated text (concatenation of all streamed deltas).
        finish_reason: ``stop`` | ``length`` | ``cancelled`` | ``error``.
        usage: Prompt / completion / total token counts.
        attempt_id: Worker-generated id for this attempt; useful to
            correlate gateway logs with worker logs across redelivery.
        ttft_ms: Time-to-first-token in milliseconds (worker-measured).
        tpot_ms: Average time per output token in milliseconds.
        request: Request-scoped id, metered usage, and settled debit when
            supplied by the gateway. This is distinct from the model-native
            token summary in ``usage``.
    """

    model: str
    text: str
    finish_reason: FinishReason
    usage: GenerationUsage
    attempt_id: str | None
    ttft_ms: float | None
    tpot_ms: float | None
    request: RequestMetadata


class GenerateChunk(TypedDict, total=False):
    """One SSE event from :meth:`SIEClient.stream_generate` (SIE-native shape).

    Mirrors ``build_generate_chunk_event`` in
    ``packages/sie_gateway/src/handlers/sse.rs``. ``usage`` / ``ttft_ms`` land
    only on the terminal chunk (``done`` is ``True``); ``error`` is populated
    when the worker/gateway failed mid-stream — the SDK raises
    :class:`~sie_sdk.client.errors.ServerError` rather than yielding it.

    Attributes:
        request_id: Gateway request id (stable across the stream).
        seq: Monotonic per-attempt chunk sequence number.
        text_delta: Incremental text for this chunk.
        logprobs: Per-token log probabilities aligned with ``text_delta``.
        done: ``True`` on the terminal chunk.
        finish_reason: Termination reason (terminal chunk only).
        usage: Prompt / completion / total token counts (terminal chunk only).
        ttft_ms: Time-to-first-token in milliseconds (terminal chunk only).
        error: ``{code, message}`` when generation failed mid-stream.
    """

    request_id: str
    seq: int
    text_delta: str
    logprobs: list[dict[str, Any]]
    done: bool
    finish_reason: FinishReason
    usage: GenerationUsage
    ttft_ms: float
    error: dict[str, str]


# --- Chat completions (OpenAI-compatible) — /v1/chat/completions ------------

ChatRole = Literal["system", "user", "assistant", "tool", "developer"]

# OpenAI chat finish reason (``null`` on non-terminal streaming chunks).
ChatFinishReason = Literal["stop", "length", "tool_calls", "content_filter"] | None


class ChatImageURL(TypedDict, total=False):
    """``image_url`` payload of an image content part — a base64 ``data:`` URI.

    May also be supplied as a bare string (the URI) on ``input_image`` parts.
    """

    url: str


class ChatContentPart(TypedDict, total=False):
    """One part of a multimodal ``messages[*].content`` array.

    Text parts (``text`` / ``input_text``) carry ``text``; image parts
    (``image_url`` / ``input_image``) carry ``image_url`` as a base64 ``data:``
    URI and are accepted for vision-capable generation models.
    """

    type: Literal["text", "input_text", "image_url", "input_image"]
    text: str
    image_url: str | ChatImageURL


class ChatMessage(TypedDict, total=False):
    """A single chat message.

    The gateway accepts ``system | user | assistant | tool`` roles.
    ``content`` is a string OR an array of :class:`ChatContentPart` (text and,
    for vision-capable generation models, base64 ``data:`` image parts);
    ``tool_calls`` / ``tool_call_id`` are honoured on the multi-turn tool-replay
    path. See the gateway schema for the canonical accepted subset.
    """

    role: ChatRole
    content: str | list[ChatContentPart] | None
    name: str
    tool_call_id: str
    tool_calls: list[dict[str, Any]]


class ChatUsage(TypedDict):
    """Token usage block (wire shape)."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(TypedDict, total=False):
    """A single choice in a non-streaming :class:`ChatCompletion`."""

    index: int
    message: ChatMessage
    finish_reason: ChatFinishReason
    logprobs: Any


class ChatCompletion(TypedDict, total=False):
    """Non-streaming response from :meth:`SIEClient.chat_completions`."""

    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None
    choices: list[ChatChoice]
    usage: ChatUsage
    request: RequestMetadata


class ChatDelta(TypedDict, total=False):
    """Incremental delta emitted on each streaming chunk."""

    role: str
    content: str
    tool_calls: list[dict[str, Any]]


class ChatChunkChoice(TypedDict, total=False):
    """A single choice in a streaming :class:`ChatCompletionChunk`."""

    index: int
    delta: ChatDelta
    finish_reason: ChatFinishReason
    logprobs: Any


class ChatCompletionChunk(TypedDict, total=False):
    """One SSE event from :meth:`SIEClient.stream_chat_completions`.

    The terminal usage-only chunk (emitted when
    ``stream_options.include_usage`` is ``True``) sets ``choices: []`` and
    populates ``usage``.
    """

    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None
    choices: list[ChatChunkChoice]
    usage: ChatUsage


# --- Responses (OpenAI-compatible) — /v1/responses -------------------------


class ResponseInputContentPart(TypedDict):
    """One text-only content part accepted by the Responses MVP."""

    type: Literal["text", "input_text"]
    text: str


ResponseInputRole = Literal["system", "user", "assistant", "developer"]


class ResponseInputMessage(TypedDict):
    """One stateless text message accepted by the Responses MVP."""

    role: ResponseInputRole
    content: str | list[ResponseInputContentPart]


class ResponseOutputText(TypedDict):
    """One text content part returned by the Responses MVP."""

    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, Any]]


class ResponseOutputMessage(TypedDict):
    """One completed assistant message returned by the Responses MVP."""

    type: Literal["message"]
    id: str
    role: Literal["assistant"]
    status: Literal["completed"]
    content: list[ResponseOutputText]


class ResponseUsage(TypedDict):
    """Token usage returned by the Responses MVP."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseResult(TypedDict, total=False):
    """Non-streaming result from :meth:`SIEClient.responses`."""

    id: str
    object: Literal["response"]
    created_at: int
    model: str
    status: Literal["completed"]
    output: list[ResponseOutputMessage]
    usage: ResponseUsage
    request: RequestMetadata


class WorkerInfo(TypedDict, total=False):
    """Information about a single worker in the cluster.

    Used in cluster status messages and health responses.

    Attributes:
        name: Worker display name (usually same as url).
        url: Worker base URL.
        gpu: GPU type (e.g., "l4", "a100-80gb").
        gpu_count: Number of GPUs on this worker.
        ready_gpu_slots: Number of worker GPU slots ready to receive work.
        healthy: Whether the worker is healthy.
        queue_depth: Number of items in the worker's queue.
        pending_cost: Worker-local scheduler pending cost.
        inflight_batches: Number of batches currently executing on the worker.
        loaded_models: List of model names loaded on this worker.
        memory_used_bytes: Total GPU memory in use.
        memory_total_bytes: Total GPU memory available.
        bundle: Bundle name this worker serves.
        bundle_config_hash: Config hash for bundle config awareness gating.
    """

    name: str
    url: str
    gpu: str
    gpu_count: int
    ready_gpu_slots: int
    healthy: bool
    queue_depth: int
    pending_cost: int
    inflight_batches: int
    loaded_models: list[str]
    memory_used_bytes: int
    memory_total_bytes: int
    bundle: NotRequired[str]
    bundle_config_hash: str


class CapacityInfo(TypedDict, total=False):
    """Cluster capacity information.

    Returned by get_capacity() to show current cluster state.

    Attributes:
        status: Overall cluster status ("healthy", "degraded", "no_workers").
        worker_count: Number of healthy workers.
        gpu_count: Number of GPUs available.
        models_loaded: Number of unique models loaded across all workers.
        configured_gpu_types: Canonical machine profiles configured in the cluster.
        live_gpu_types: GPU types currently running (subset of configured).
        workers: List of worker details.
    """

    status: str
    worker_count: int
    gpu_count: int
    models_loaded: int
    configured_gpu_types: list[str]
    live_gpu_types: list[str]
    workers: list[WorkerInfo]


class PoolSpec(TypedDict, total=False):
    """Resource pool specification for creating pools.

    Used to reserve capacity in a cluster for exclusive use.
    Re-posting the same pool name updates the readiness requirements/caps and
    renews the lease.

    Attributes:
        name: Pool name (required). Used in GPU param as "pool_name/machine_profile";
            the gateway stores and routes pool names in lowercase.
        queue_pool: Optional Helm/NATS queue namespace backing this logical pool.
            Defaults to "default", so API-created pools can draw from base
            capacity deployed once by operators.
        gpus: Optional GPU requirements for pool readiness, e.g., {"l4": 2, "a100-40gb": 1}.
        gpu_caps: Optional maximum assigned workers per GPU type.
        bundle: Optional bundle filter for worker assignment.
        minimum_worker_count: Per-pool warm floor (minimum machines kept warm). The
            gateway emits canonical ``sie.gateway.pool.warm_floor`` telemetry; the
            collector exposes ``sie_gateway_pool_warm_floor`` to KEDA. Defaults to 0
            (scale to zero).
        pinned_models: Per-pool set of model ids kept loaded so the first request to
            them pays no cold model-load. Each id must be a model the gateway already
            tracks (see ``GET /v1/configs/models``) and may be profile-qualified
            (``model-name:profile_name``); unknown ids are rejected. Defaults to an
            empty list (lazy-loading unchanged).
    """

    name: str
    queue_pool: str
    gpus: dict[str, int]
    gpu_caps: dict[str, int]
    bundle: str
    minimum_worker_count: int
    pinned_models: list[str]


# --- Pool Response Types (nested structure matching gateway API) ---


class AssignedWorkerInfo(TypedDict):
    """Worker assigned to a pool.

    Attributes:
        name: Worker name/identifier.
        url: Worker base URL.
        gpu: GPU type (e.g., "l4", "a100-80gb").
        bundle: Runtime bundle this worker serves.
    """

    name: str
    url: str
    gpu: str
    bundle: str


class PoolStatusInfo(TypedDict, total=False):
    """Pool status information from gateway API.

    Attributes:
        state: Pool state ("pending", "active", "expired").
        assigned_workers: List of workers assigned to this pool.
        created_at: Unix timestamp when pool was created.
        last_renewed: Unix timestamp of last lease renewal.
    """

    state: str
    assigned_workers: list[AssignedWorkerInfo]
    created_at: float
    last_renewed: float


class PoolSpecResponse(TypedDict, total=False):
    """Pool specification in API response.

    Attributes:
        name: Pool name.
        queue_pool: Helm/NATS queue namespace backing this logical pool.
        gpus: GPU requirements for pool readiness, e.g., {"l4": 2, "a100-40gb": 1}.
        gpu_caps: Optional maximum assigned workers per GPU type.
        bundle: Optional bundle filter for worker assignment.
        minimum_worker_count: Per-pool warm floor (minimum machines kept warm via KEDA).
        pinned_models: Per-pool set of model ids kept loaded (no cold model-load).
    """

    name: str
    queue_pool: str
    gpus: dict[str, int]
    gpu_caps: dict[str, int]
    bundle: str | None
    minimum_worker_count: int
    pinned_models: list[str]


class PoolResponse(TypedDict, total=False):
    """Full pool response from gateway API.

    This is the canonical response structure for pool endpoints.
    Uses nested spec/status structure for clear separation.

    Attributes:
        name: Pool name.
        spec: Pool specification (GPU requirements).
        status: Pool status (state, assigned workers, timestamps).
    """

    name: str
    spec: PoolSpecResponse
    status: PoolStatusInfo


class PoolInfo(TypedDict, total=False):
    """Information about a resource pool (nested structure).

    Uses the same nested structure as PoolResponse for consistency.

    Attributes:
        name: Pool name.
        spec: Pool specification (GPU requirements).
        status: Pool status (state, assigned workers, timestamps).
    """

    name: str
    spec: PoolSpecResponse
    status: PoolStatusInfo


class PoolListItem(TypedDict, total=False):
    """Pool summary item in list response.

    Attributes:
        name: Pool name.
        state: Pool state ("pending", "active", "expired").
        gpus: GPU requirements.
        worker_count: Number of workers assigned.
    """

    name: str
    state: str
    gpus: dict[str, int]
    worker_count: int


# --- Cluster/Health Response Types ---


class ClusterSummary(TypedDict):
    """Cluster summary statistics.

    Attributes:
        worker_count: Number of healthy workers.
        gpu_count: Total number of GPUs.
        models_loaded: Number of unique models loaded.
        total_qps: Total queries per second across cluster.
    """

    worker_count: int
    gpu_count: int
    models_loaded: int
    total_qps: float


class ModelSummary(TypedDict, total=False):
    """Model summary in cluster status.

    Attributes:
        name: Model name.
        state: Aggregate state ("loaded" if on any worker, else "available").
        worker_count: Number of workers with this model loaded.
        gpu_types: GPU types running this model.
        total_queue_depth: Total queue depth across workers.
    """

    name: str
    state: ModelState
    worker_count: int
    gpu_types: list[str]
    total_queue_depth: int


class HealthResponse(TypedDict, total=False):
    """Health endpoint response from gateway.

    Attributes:
        status: Overall status ("healthy", "degraded", "no_workers").
        type: Component type ("gateway" or "worker").
        cluster: Cluster summary statistics.
        configured_gpu_types: Canonical machine profiles configured in the cluster.
        live_gpu_types: GPU types currently running.
        workers: List of worker details.
        models: List of model summaries.
    """

    status: str
    type: str
    cluster: ClusterSummary
    configured_gpu_types: list[str]
    live_gpu_types: list[str]
    workers: list[WorkerInfo]
    models: list[ModelSummary]


# Backwards compatibility aliases
EntityResult = Entity
RelationResult = Relation
ClassificationResult = Classification
ObjectResult = DetectedObject

# Alias for cluster model info (same as ModelSummary with state)
ClusterModelInfo = ModelSummary


# =============================================================================
# WebSocket Status Message Types
# =============================================================================
# These types define the wire format for real-time status updates:
# - Worker sends WorkerStatusMessage via /ws/status
# - Gateway sends ClusterStatusMessage via /ws/cluster-status


class ServerInfo(TypedDict):
    """Server metadata included in worker status."""

    version: str
    uptime_seconds: int
    user: str
    working_dir: str
    pid: int


class GPUMetrics(TypedDict, total=False):
    """GPU metrics for a single device.

    Attributes:
        device: CUDA device identifier (e.g., "cuda:0").
        name: Full GPU name (e.g., "NVIDIA L4").
        gpu_type: Normalized GPU type for routing (e.g., "l4", "a100-80gb").
        utilization_pct: GPU compute utilization (0-100).
        memory_used_bytes: GPU memory in use.
        memory_total_bytes: Total GPU memory.
        memory_threshold_pct: Memory pressure threshold (added by server).
    """

    device: str
    name: str
    gpu_type: str
    utilization_pct: int
    memory_used_bytes: int
    memory_total_bytes: int
    memory_threshold_pct: float


class ModelConfig(TypedDict, total=False):
    """Model configuration included in status."""

    hf_id: str | None
    adapter: str | None
    inputs: Sequence[str]
    outputs: Sequence[str]
    dims: dict[str, Any] | None
    max_sequence_length: int | None
    pooling: str | None
    normalize: bool
    adapter_options_loadtime: dict[str, Any] | None
    adapter_options_runtime: dict[str, Any] | None


class AdaptiveBatchingStatus(TypedDict, total=False):
    """Adaptive batching state for a single model on a worker.

    Absent from ModelStatus when adaptive batching is disabled for this model.

    Attributes:
        calibrated: Whether auto-calibration has completed.
        target_p50_ms: Current SLO target (may be auto-calibrated). None before calibration.
        wait_ms: Current dynamic max_batch_wait_ms.
        batch_cost: Current dynamic max_batch_cost.
        p50_ms: Observed rolling p50 latency. None if not enough samples.
        headroom_ms: target - observed. None if either is None.
        fill_ratio: Mean batch fill ratio (actual_cost / max_cost). None if no samples.
    """

    calibrated: bool
    target_p50_ms: float | None
    wait_ms: float
    batch_cost: int
    p50_ms: float | None
    headroom_ms: float | None
    fill_ratio: float | None


class ModelStatus(TypedDict, total=False):
    """Status of a single model on a worker.

    Attributes:
        name: Model identifier (e.g., "BAAI/bge-m3").
        state: Current model state.
        device: Device model is loaded on (None if not loaded).
        memory_bytes: GPU memory used by this model.
        config: Model configuration details.
        queue_depth: Number of requests in queue.
        queue_pending_items: Same as queue_depth (for compatibility).
        adaptive_batching: Adaptive batching state. Absent when disabled.
    """

    name: str
    state: ModelState
    device: str | None
    memory_bytes: int
    config: ModelConfig
    queue_depth: int
    queue_pending_items: int
    adaptive_batching: AdaptiveBatchingStatus


class WorkerStatusMessage(TypedDict, total=False):
    """Complete status message sent by worker on /ws/status.

    This is the canonical format consumed by:
    - Gateway (extracts machine_profile, bundle, loaded_models for routing)
    - sie-top in worker mode (displays full details)

    Request-rate and latency telemetry are intentionally not part of this
    status snapshot; consumers read those signals from the configured OTel
    destination.

    Attributes:
        timestamp: Unix timestamp of this status snapshot.
        ready: True when worker is ready to accept traffic. Gateway only routes to
            ready workers. False during startup until lifespan completes.
        name: Worker name/identifier.
        machine_profile: Machine profile for routing. In K8s: from SIE_MACHINE_PROFILE env var
            (e.g., "l4-spot"). Standalone: detected GPU type (e.g., "l4").
        pool_name: Logical NATS work-queue pool name (from SIE_POOL env var, e.g., "default" or "customer-acme").
            Used by the gateway in queue mode to publish to the correct JetStream subject.
            Empty string when not in queue mode or not set.
        gpu_count: Number of GPUs on this worker.
        bundle: Dependency bundle this worker is running (e.g., "default").
        bundle_config_hash: SHA-256 hash of the serialized model configs/profiles
            assigned to this worker's bundle. Used by gateways to gate routing on
            config awareness. Empty string when not yet computed.
        loaded_models: List of model names currently loaded.
        server: Server metadata (version, uptime, etc.).
        gpus: Per-GPU metrics (includes gpu_type for each GPU).
        models: Per-model status.
        max_batch_requests: Maximum number of requests the worker can batch in a
            single inference call (minimum across loaded models).
        saturated: Admission backpressure signal for direct-dispatch routing. True when the worker is
            at or above its high-water mark and the gateway should temporarily
            exclude it from the HRW direct-dispatch ring. The worker owns
            hysteresis (90/70 thresholds); the gateway just consumes the bool.
            Defaults to False; missing field on legacy workers means "not saturated".
    """

    timestamp: float
    ready: bool
    name: str
    machine_profile: str
    pool_name: str
    gpu_count: int
    bundle: str
    bundle_config_hash: str
    loaded_models: list[str]
    server: ServerInfo
    gpus: list[GPUMetrics]
    models: list[ModelStatus]
    max_batch_requests: int
    saturated: bool


class ClusterStatusMessage(TypedDict, total=False):
    """Complete status message sent by gateway on /ws/cluster-status.

    Consumed by sie-top in cluster mode.

    Attributes:
        timestamp: Unix timestamp of this status snapshot.
        cluster: Aggregated cluster statistics.
        workers: Per-worker information.
        models: Per-model information aggregated across workers.
    """

    timestamp: float
    cluster: ClusterSummary
    workers: list[WorkerInfo]
    models: list[ModelSummary]


StatusMessage = WorkerStatusMessage | ClusterStatusMessage


# ---------------------------------------------------------------------------
# Jobs API (client.jobs) — the gateway's batch class.
# ---------------------------------------------------------------------------

# A job's lifecycle state (queued → running → terminal).
JobState = Literal["queued", "running", "succeeded", "failed", "suspended", "cancelled"]


class JobFieldMap(TypedDict, total=False):
    """The uniform source-mapping slots for ``jobs.submit(field_map=...)``.

    ``id_field`` ≈ ``custom_id``, ``input_field`` ≈ ``body.input``, ``carry`` =
    source fields echoed to the sink keyed by id, ``input_type`` pins the item
    shape (``"text"`` | ``"document"``). The sink slot rides separately as
    ``output_field``. All optional — per-connector params stay as aliases.
    """

    id_field: str
    input_field: str
    carry: list[str]
    input_type: str


class JobPreflight(TypedDict, total=False):
    """The preflight reservation echoed on submit / status."""

    estimated_credits: int
    estimate_basis: str


class JobChunk(TypedDict, total=False):
    """One spawned chunk's settle metadata (``output.chunks[]``; results-as-refs).

    ``ref`` is a TTL'd payload-store location holding the chunk's msgpack
    ``WorkResult`` array — never the payload itself.
    """

    seq: int
    items: int
    state: str
    ref: str | None
    units: int | None
    credits: int | None
    error: Any


class JobSubmitResult(TypedDict, total=False):
    """The ``201`` envelope from ``POST /v1/jobs`` (inline or connector job)."""

    id: str
    object: str
    operation: str
    model: str
    state: JobState
    total_items: int
    chunks: int
    preflight: JobPreflight
    # Connector jobs echo the resolved ends instead of total_items.
    input_source: str
    source: str
    sink: str


class JobStatus(TypedDict, total=False):
    """A job's public status doc from ``GET /v1/jobs/{id}`` (refs, never payloads)."""

    id: str
    object: str
    operation: str
    model: str
    state: JobState
    total_items: int
    completed_items: int
    preflight: JobPreflight
    settled_credits: int
    created_at: float
    finished_at: float | None
    output: dict[str, Any]


class JobList(TypedDict, total=False):
    """The listing envelope from ``GET /v1/jobs`` (the org's jobs, keyed by bearer)."""

    object: str
    data: list[JobStatus]


class JobResultItem(TypedDict, total=False):
    """One decoded per-item result retrieved from a finished job's chunk refs."""

    id: str | None
    success: bool | None
    units: Any
    dims: int | None
    dense: Any


class JobResults(TypedDict, total=False):
    """A finished job's decoded results — the chunk refs read and unpacked."""

    job_id: str
    state: JobState | None
    total_items: int | None
    settled_credits: int | None
    chunks: list[JobChunk]
    retrieved: int
    dims: int | None
    items: list[JobResultItem]


# ---------------------------------------------------------------------------
# Connections API (client.connections) — org-scoped connector auth by name.
# ---------------------------------------------------------------------------


class Connection(TypedDict, total=False):
    """An org-scoped connection (connector auth by name). The secret is never returned."""

    id: int
    type: str
    name: str
    created_at: float


class ConnectionCreated(TypedDict, total=False):
    """The ``201`` envelope from creating a connection (org + connection, no secret)."""

    org: str
    account_id: int
    id: int
    type: str
    name: str
    created_at: float


class ConnectionRevoked(TypedDict, total=False):
    """The envelope from revoking (soft-deleting) a connection."""

    org: str
    account_id: int
    name: str
    state: str


# ---------------------------------------------------------------------------
# Files API (client.files) — the OpenAI-compatible file store. Byte-for-byte the
# OpenAI File object, so switching `openai` → `sie_sdk` is a base_url-swap
# drop-in for the batch flow.
# ---------------------------------------------------------------------------


class File(TypedDict, total=False):
    """An OpenAI-shaped file object (``POST/GET /v1/files``).

    Mirrors OpenAI's ``FileObject``: the gateway emits ``{id, object:"file",
    bytes, created_at, filename, purpose}``. ``status`` /
    ``expires_at`` are the optional OpenAI fields surfaced when the TTL-GC
    lands (batch files expire, matching OpenAI's 30-day window).
    """

    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str
    status: str
    expires_at: int


class FileDeleted(TypedDict, total=False):
    """The envelope from deleting a file (OpenAI ``FileDeleted``)."""

    id: str
    object: str
    deleted: bool


# ---------------------------------------------------------------------------
# Batches API (client.batches) — the OpenAI-compatible batch surface. Runs on
# the same jobs engine as /v1/jobs (`spawn_encode_job`), one engine two front
# doors: a `base_url` swap makes an OpenAI-batch caller work unchanged.
# ---------------------------------------------------------------------------


class BatchRequestCounts(TypedDict, total=False):
    """Per-batch request tally (OpenAI ``request_counts``)."""

    total: int
    completed: int
    failed: int


class Batch(TypedDict, total=False):
    """An OpenAI-shaped batch object (``POST/GET /v1/batches``).

    Mirrors OpenAI's ``Batch``: the gateway emits ``{id, object:"batch",
    endpoint, input_file_id, completion_window, status, output_file_id,
    error_file_id, created_at, finished_at, request_counts, errors}``
    (``batches.rs``). The granular OpenAI timestamps (``completed_at`` etc.) are
    optional — a strict OpenAI SDK reads them when the gateway surfaces them.
    """

    id: str
    object: str
    endpoint: str
    input_file_id: str
    completion_window: str
    status: str
    output_file_id: str | None
    error_file_id: str | None
    created_at: int
    finished_at: int | None
    in_progress_at: int | None
    completed_at: int | None
    failed_at: int | None
    expired_at: int | None
    cancelled_at: int | None
    request_counts: BatchRequestCounts
    errors: Any
    metadata: dict[str, Any] | None


class BatchList(TypedDict, total=False):
    """The listing envelope from ``GET /v1/batches`` (OpenAI cursor page)."""

    object: str
    data: list[Batch]
    first_id: str
    last_id: str
    has_more: bool
