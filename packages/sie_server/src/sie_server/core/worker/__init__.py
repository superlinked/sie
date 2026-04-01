"""Model worker for async request handling with dynamic batching.

The ModelWorker manages a single model's inference pipeline:
1. Accepts tokenized requests via submit()
2. Batches requests using BatchFormer
3. Runs inference on batches via operation handlers
4. Fans out results to waiting futures

Architecture:
- ModelWorker: Manages lifecycle, batching, FCFS scheduling, stats
- OperationHandler: Abstract interface for operation-specific logic
- EncodeHandler, ExtractHandler, ScoreHandler: Concrete implementations

See DESIGN.md Section 5.3.
"""

from sie_server.core.worker.handlers import (
    EncodeHandler,
    ExtractHandler,
    OperationHandler,
    ScoreHandler,
)
from sie_server.core.worker.model_worker import ModelWorker
from sie_server.core.worker.types import (
    QueueFullError,
    RequestMetadata,
    WorkerConfig,
    WorkerOutput,
    WorkerResult,
    WorkerStats,
)

__all__ = [
    "EncodeHandler",
    "ExtractHandler",
    # Main worker class
    "ModelWorker",
    # Operation handlers
    "OperationHandler",
    # Types
    "QueueFullError",
    "RequestMetadata",
    "ScoreHandler",
    "WorkerConfig",
    "WorkerOutput",
    "WorkerResult",
    "WorkerStats",
]
