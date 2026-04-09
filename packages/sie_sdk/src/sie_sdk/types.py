"""Client-side type definitions for SIE SDK.

These types are lightweight TypedDicts for type hints. No validation is performed
client-side - the server validates all inputs.

Per DESIGN.md Section 4.1 and 8.2, these types support flexible Python inputs
(file paths, PIL images, numpy arrays, bytes) which the client converts for transport.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from PIL import Image

# Output dtype options (per DESIGN.md Section 4.4)
DType = Literal["float32", "float16", "bfloat16", "int8", "uint8", "binary", "ubinary"]

# Output dtypes we support for casting
# Note: bfloat16 is compute-only, not returned
OutputDType = Literal["float32", "float16", "int8", "uint8", "binary", "ubinary"]

# Default output dtype
DEFAULT_OUTPUT_DTYPE: OutputDType = "float32"

# Output type options
OutputType = Literal["dense", "sparse", "multivector"]

# Model state (for status messages)
ModelState = Literal["available", "loading", "loaded", "unloading"]


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


class Item(TypedDict, total=False):
    """A single item to encode, score, or extract from.

    Per DESIGN.md Section 4.1, items can contain text, images, audio, or video.
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
    metadata: dict[str, Any]
    multivector: NDArray[np.float32]  # Pre-encoded multivector (for use with scoring.maxsim)


class SparseResult(TypedDict):
    """Sparse vector result with non-zero indices and values.

    Per DESIGN.md Section 8.2.
    """

    indices: NDArray[np.int32]
    values: NDArray[np.float32]


class TimingInfo(TypedDict, total=False):
    """Server-side timing breakdown for a request."""

    total_ms: float
    queue_ms: float
    tokenization_ms: float
    inference_ms: float


class EncodeResult(TypedDict, total=False):
    """Result of encoding a single item.

    Per DESIGN.md Section 8.2. Contains the item ID (if provided) and one or more
    output representations depending on what was requested.

    Attributes:
        id: Item ID (echoed from request if provided).
        dense: Dense embedding as numpy array, shape [dims].
        sparse: Sparse embedding with indices and values.
        multivector: Multi-vector embedding as numpy array, shape [num_tokens, token_dims].
    """

    id: str
    dense: NDArray[np.float32]
    sparse: SparseResult
    multivector: NDArray[np.float32]
    timing: TimingInfo


class ModelDims(TypedDict, total=False):
    """Model dimension information."""

    dense: int
    sparse: int
    multivector: int


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


class ScoreResult(TypedDict, total=False):
    """Result of scoring items against a query.

    Attributes:
        model: Model used for scoring.
        query_id: Query ID (echoed from request if provided).
        scores: List of score entries, sorted by relevance (descending).
    """

    model: str
    query_id: str
    scores: list[ScoreEntry]


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


class ExtractResult(TypedDict, total=False):
    """Result of extraction for a single item.

    Attributes:
        id: Item ID (echoed from request or auto-generated).
        entities: List of extracted entities (NER spans).
        relations: List of extracted relation triples.
        classifications: List of classification results.
        objects: List of detected objects with bounding boxes.
        data: Additional structured extraction data (if output_schema was provided).
    """

    id: str
    entities: list[Entity]
    relations: list[Relation]
    classifications: list[Classification]
    objects: list[DetectedObject]
    data: dict[str, Any]


class WorkerInfo(TypedDict, total=False):
    """Information about a single worker in the cluster.

    Used in cluster status messages and health responses.

    Attributes:
        name: Worker display name (usually same as url).
        url: Worker base URL.
        gpu: GPU type (e.g., "l4", "a100-80gb").
        gpu_count: Number of GPUs on this worker.
        healthy: Whether the worker is healthy.
        queue_depth: Number of items in the worker's queue.
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
    healthy: bool
    queue_depth: int
    loaded_models: list[str]
    memory_used_bytes: int
    memory_total_bytes: int
    bundle: str
    bundle_config_hash: str


class CapacityInfo(TypedDict, total=False):
    """Cluster capacity information.

    Returned by get_capacity() to show current cluster state.

    Attributes:
        status: Overall cluster status ("healthy", "degraded", "no_workers").
        worker_count: Number of healthy workers.
        gpu_count: Number of GPUs available.
        models_loaded: Number of unique models loaded across all workers.
        configured_gpu_types: GPU types configured in the cluster (from Helm).
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
    Pools are created lazily on first request and garbage collected after inactivity.

    Attributes:
        name: Pool name (required). Used in GPU param as "pool_name/gpu_type".
        gpus: GPU requirements, e.g., {"l4": 2, "a100-40gb": 1}.
        bundle: Optional bundle filter for worker assignment.
        minimum_worker_count: Desired minimum warm workers in the pool. Stored in pool
            spec and forwarded to the router; enforcement depends on cluster autoscaler
            configuration. Defaults to 0 (scale to zero).
    """

    name: str
    gpus: dict[str, int]
    bundle: str
    minimum_worker_count: int


# --- Pool Response Types (nested structure matching router API) ---


class AssignedWorkerInfo(TypedDict):
    """Worker assigned to a pool.

    Attributes:
        name: Worker name/identifier.
        url: Worker base URL.
        gpu: GPU type (e.g., "l4", "a100-80gb").
    """

    name: str
    url: str
    gpu: str


class PoolStatusInfo(TypedDict, total=False):
    """Pool status information from router API.

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
        gpus: GPU requirements, e.g., {"l4": 2, "a100-40gb": 1}.
        bundle: Optional bundle filter for worker assignment.
        minimum_worker_count: Minimum warm workers in the pool.
    """

    gpus: dict[str, int]
    bundle: str | None
    minimum_worker_count: int


class PoolResponse(TypedDict, total=False):
    """Full pool response from router API.

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
    """Health endpoint response from router.

    Attributes:
        status: Overall status ("healthy", "degraded", "no_workers").
        type: Component type ("router" or "worker").
        cluster: Cluster summary statistics.
        configured_gpu_types: GPU types configured in the cluster.
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
# - Router sends ClusterStatusMessage via /ws/cluster-status


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
    - Router (extracts machine_profile, bundle, loaded_models for routing)
    - sie-top in worker mode (displays full details)

    Attributes:
        timestamp: Unix timestamp of this status snapshot.
        ready: True when worker is ready to accept traffic. Router only routes to
            ready workers. False during startup until lifespan completes.
        name: Worker name/identifier.
        machine_profile: Machine profile for routing. In K8s: from SIE_MACHINE_PROFILE env var
            (e.g., "l4-spot"). Standalone: detected GPU type (e.g., "l4").
        pool_name: NATS work-queue pool name (from SIE_POOL env var, e.g., "l4-spot-default").
            Used by the router in queue mode to publish to the correct JetStream subject.
            Empty string when not in queue mode or not set.
        gpu_count: Number of GPUs on this worker.
        bundle: Dependency bundle this worker is running (e.g., "default").
        bundle_config_hash: SHA-256 hash of the serialized model configs/profiles
            assigned to this worker's bundle. Used by routers to gate routing on
            config awareness. Empty string when not yet computed.
        loaded_models: List of model names currently loaded.
        server: Server metadata (version, uptime, etc.).
        gpus: Per-GPU metrics (includes gpu_type for each GPU).
        models: Per-model status.
        max_batch_requests: Maximum number of requests the worker can batch in a
            single inference call (minimum across loaded models).
        counters: Prometheus counter values for QPS calculation.
        histograms: Prometheus histogram data for latency percentiles.
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
    counters: dict[str, dict[str, float]]
    histograms: dict[str, dict[str, dict[str, Any]]]


class ClusterStatusMessage(TypedDict, total=False):
    """Complete status message sent by router on /ws/cluster-status.

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
