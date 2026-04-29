"""SIE SDK - Python client for Search Inference Engine.

Example:
    >>> from sie_sdk import SIEClient
    >>> client = SIEClient("http://localhost:8080")
    >>> result = client.encode("bge-m3", {"text": "Hello world"})
    >>> result["dense"]  # np.ndarray, shape [1024]

For ColBERT/late interaction models, use the scoring module:
    >>> from sie_sdk.scoring import maxsim
    >>> scores = maxsim(query_multivector, doc_multivectors)
"""

from sie_sdk.client import (
    LoraLoadingError,
    ModelLoadFailedError,
    ModelLoadingError,
    PoolError,
    ProvisioningError,
    RequestError,
    ServerError,
    SIEAsyncClient,
    SIEClient,
    SIEConnectionError,
    SIEError,
)
from sie_sdk.encoding import (
    SparseVector,
    dense_embedding,
    multivector_embedding,
    normalize_sparse_vector,
    sparse_embedding,
    sparse_embedding_dict,
)
from sie_sdk.types import (
    # Response types
    AssignedWorkerInfo,
    CapacityInfo,
    ClusterStatusMessage,
    ClusterSummary,
    EncodeResult,
    Entity,
    ExtractResult,
    HealthResponse,
    Item,
    ModelInfo,
    ModelSummary,
    PoolInfo,
    PoolListItem,
    PoolResponse,
    PoolSpec,
    PoolSpecResponse,
    PoolStatusInfo,
    ScoreResult,
    SparseResult,
    StatusMessage,
    TimingInfo,
    WorkerInfo,
    WorkerStatusMessage,
)

__version__ = "0.1.0"

__all__ = [
    "AssignedWorkerInfo",
    "CapacityInfo",
    "ClusterStatusMessage",
    "ClusterSummary",
    "EncodeResult",
    "Entity",
    "ExtractResult",
    "HealthResponse",
    "Item",
    "LoraLoadingError",
    "ModelInfo",
    "ModelLoadFailedError",
    "ModelLoadingError",
    "ModelSummary",
    "PoolError",
    "PoolInfo",
    "PoolListItem",
    "PoolResponse",
    "PoolSpec",
    "PoolSpecResponse",
    "PoolStatusInfo",
    "ProvisioningError",
    "RequestError",
    "SIEAsyncClient",
    "SIEClient",
    "SIEConnectionError",
    "SIEError",
    "ScoreResult",
    "ServerError",
    "SparseResult",
    "SparseVector",
    "StatusMessage",
    "TimingInfo",
    "WorkerInfo",
    "WorkerStatusMessage",
    "dense_embedding",
    "multivector_embedding",
    "normalize_sparse_vector",
    "sparse_embedding",
    "sparse_embedding_dict",
]
