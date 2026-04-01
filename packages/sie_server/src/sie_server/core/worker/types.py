"""Types and dataclasses for model worker.

Contains request metadata, configuration, and statistics types.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput
from sie_server.core.timing import RequestTiming

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

# Type alias for worker output union
WorkerOutput = EncodeOutput | ScoreOutput | ExtractOutput


@dataclass
class WorkerResult:
    """Result from a worker inference request.

    Contains typed inference output and timing information.
    The output field contains the raw typed output from the adapter.
    API layer is responsible for formatting to JSON-serializable dicts.
    """

    output: WorkerOutput
    timing: RequestTiming


@dataclass(slots=True)
class RequestMetadata:
    """Metadata for a pending inference request.

    Carries the information needed to run inference and return results.
    Supports encode, extract, and score operations via the `operation` field.

    For encode operations:
        - output_types: Which outputs to return ("dense", "sparse", "multivector")
        - instruction: Optional instruction for instruction-tuned models
        - is_query: Whether items are queries (True) or documents (False)

    For extract operations:
        - labels: Entity types to extract (e.g., ["person", "organization"])
        - output_schema: Optional schema for structured extraction
        - instruction: Optional instruction (reused from encode)

    For score operations:
        - query: The query item to score all items against
        - instruction: Optional instruction for instruction-tuned rerankers
    """

    future: asyncio.Future[WorkerResult]
    items: list[Item]  # Original items for adapter (docs for score)
    timing: RequestTiming  # Tracks timing for this request
    request_id: str | None = None
    # Partial results for sub-batching: maps original_index -> typed output (batch_size=1)
    _partial_results: dict[int, EncodeOutput | ScoreOutput | ExtractOutput] | None = None

    # Operation type determines which adapter method to call
    operation: Literal["encode", "extract", "score"] = "encode"

    # Shared params
    instruction: str | None = None
    options: dict[str, Any] | None = None  # Adapter options to override model config defaults

    # Encode-specific params
    output_types: list[str] = field(default_factory=list)
    is_query: bool = False

    # Extract-specific params
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None

    # Score-specific params
    query: Item | None = None  # Query item for reranking


class QueueFullError(Exception):
    """Raised when the worker queue is full and cannot accept more requests."""


@dataclass
class WorkerConfig:
    """Configuration for ModelWorker."""

    max_batch_tokens: int = 16384
    max_batch_requests: int = 256
    max_batch_wait_ms: int = 10
    max_queue_size: int = 1000  # Maximum pending items in queue (0 = unlimited)
    instrumentation: bool = False


@dataclass
class WorkerStats:
    """Runtime statistics for a ModelWorker."""

    batches_processed: int = 0
    items_processed: int = 0
    total_tokens_processed: int = 0
    inference_errors: int = 0

    # Detailed instrumentation (for performance analysis)
    batch_sizes: list[int] | None = None  # Items per batch
    batch_tokens: list[int] | None = None  # Tokens per batch
    batch_wait_ms: list[float] | None = None  # Time waiting for batch to form
    inference_ms: list[float] | None = None  # GPU inference time per batch
    requests_per_batch: list[int] | None = None  # Unique requests combined per batch

    def enable_instrumentation(self) -> None:
        """Enable detailed instrumentation tracking."""
        self.batch_sizes = []
        self.batch_tokens = []
        self.batch_wait_ms = []
        self.inference_ms = []
        self.requests_per_batch = []

    @property
    def instrumentation_enabled(self) -> bool:
        """Check if instrumentation is enabled."""
        return self.batch_sizes is not None

    def summary(self) -> str:
        """Return a summary of collected statistics."""
        lines = [
            f"Batches processed: {self.batches_processed}",
            f"Items processed: {self.items_processed}",
            f"Tokens processed: {self.total_tokens_processed}",
            f"Inference errors: {self.inference_errors}",
        ]
        if self.instrumentation_enabled and self.batch_sizes:
            import statistics

            # All instrumentation lists are initialized together, so assert they exist
            assert self.batch_tokens is not None
            assert self.batch_wait_ms is not None
            assert self.inference_ms is not None
            assert self.requests_per_batch is not None

            lines.extend(
                [
                    "",
                    "=== Batch Size Stats ===",
                    f"  Items/batch: min={min(self.batch_sizes)}, max={max(self.batch_sizes)}, "
                    f"mean={statistics.mean(self.batch_sizes):.1f}, median={statistics.median(self.batch_sizes):.1f}",
                    f"  Tokens/batch: min={min(self.batch_tokens)}, max={max(self.batch_tokens)}, "
                    f"mean={statistics.mean(self.batch_tokens):.1f}",
                    f"  Requests/batch: min={min(self.requests_per_batch)}, max={max(self.requests_per_batch)}, "
                    f"mean={statistics.mean(self.requests_per_batch):.1f}",
                    "",
                    "=== Timing Stats ===",
                    f"  Batch wait (ms): min={min(self.batch_wait_ms):.1f}, max={max(self.batch_wait_ms):.1f}, "
                    f"mean={statistics.mean(self.batch_wait_ms):.1f}, p50={statistics.median(self.batch_wait_ms):.1f}",
                    f"  Inference (ms): min={min(self.inference_ms):.1f}, max={max(self.inference_ms):.1f}, "
                    f"mean={statistics.mean(self.inference_ms):.1f}, p50={statistics.median(self.inference_ms):.1f}",
                ]
            )
        return "\n".join(lines)
