"""Typed output types for standardized batched inference results.

Three separate output types for three operations:
- EncodeOutput: from adapter.encode() - embeddings (dense/sparse/multivector)
- ScoreOutput: from adapter.score_pairs() - reranking scores
- ExtractOutput: from adapter.extract() - entities/structured data

Design principles:
- Each operation has its own typed output (no field-sniffing to determine type)
- Dense stays batched [batch, dim] for zero overhead
- Sparse/multivector are lists due to variable size per item
- Worker passes typed outputs through; API layer formats to JSON
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sie_server.types.responses import Classification, DetectedObject, Entity, Relation


@dataclass
class SparseVector:
    """Sparse vector with indices and values.

    Represents a sparse embedding where only non-zero dimensions are stored.
    Used by lexical models (SPLADE, BM25) and hybrid models (BGE-M3).
    """

    indices: np.ndarray  # int32, sorted ascending
    values: np.ndarray  # float32, same length as indices


@dataclass
class EncodeOutput:
    """Batched output from adapter.encode() - embeddings.

    This is the exchange format between adapter.encode() and postprocessors.
    Dense embeddings stay in batched format [batch, dim] for zero overhead.
    Sparse and multivector are lists due to variable size per item.

    Attributes:
        dense: Dense embeddings [batch, dim] or None if not requested.
        sparse: List of sparse vectors (len=batch) or None.
        multivector: List of per-token embeddings or None. Each is [seq_len, token_dim].
        batch_size: Number of items in the batch.
        is_query: Whether items are queries (affects some postprocessors).
        dense_dim: Dimension of dense embeddings (for validation).
        multivector_token_dim: Per-token dimension for multivector.
    """

    # Embedding outputs
    dense: np.ndarray | None = None  # [batch, dim] - always batched
    sparse: list[SparseVector] | None = None  # len=batch, variable nnz per item
    multivector: list[np.ndarray] | None = None  # len=batch, each [seq_i, token_dim]

    # Metadata
    batch_size: int = 0
    is_query: bool = False
    dense_dim: int | None = None
    multivector_token_dim: int | None = None

    # Extension point for adapter-specific data
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate consistency of output fields."""
        if self.dense is not None:
            if len(self.dense.shape) != 2:
                msg = f"dense must be 2D [batch, dim], got shape {self.dense.shape}"
                raise ValueError(msg)
            if self.batch_size == 0:
                self.batch_size = self.dense.shape[0]
            elif self.dense.shape[0] != self.batch_size:
                msg = f"dense batch size {self.dense.shape[0]} != {self.batch_size}"
                raise ValueError(msg)
            if self.dense_dim is None:
                self.dense_dim = self.dense.shape[1]

        if self.sparse is not None:
            if self.batch_size == 0:
                self.batch_size = len(self.sparse)
            elif len(self.sparse) != self.batch_size:
                msg = f"sparse list length {len(self.sparse)} != batch_size {self.batch_size}"
                raise ValueError(msg)

        if self.multivector is not None:
            if self.batch_size == 0:
                self.batch_size = len(self.multivector)
            elif len(self.multivector) != self.batch_size:
                msg = f"multivector list length {len(self.multivector)} != batch_size {self.batch_size}"
                raise ValueError(msg)
            if self.multivector_token_dim is None and len(self.multivector) > 0:
                self.multivector_token_dim = self.multivector[0].shape[-1]


@dataclass
class ScoreOutput:
    """Batched output from adapter.score_pairs() - reranking scores.

    Attributes:
        scores: Scores for each (query, doc) pair [batch], float32.
        batch_size: Number of pairs scored.
    """

    scores: np.ndarray  # [batch] float32

    # Metadata
    batch_size: int = 0

    # Unit-meter seam (§7.3): authoritative per-pair input-token counts,
    # aligned 1:1 with ``scores`` — the REAL tokenizer length of each
    # (query, doc) pair the model processed, summed by the queue executor
    # into ``ItemOutcome.units.input_tokens`` for the score work item.
    # ``None`` when the adapter cannot surface a real count (char-proxy
    # rerankers): metering then falls back to its reserve estimate rather
    # than billing an estimate as a count. Sliced/assembled positionally
    # with ``scores`` (see ScoreHandler) so fused cross-request batches
    # keep each pair's count attributed to the right work item.
    input_token_counts: list[int] | None = None

    # Authoritative per-pair counts of images actually processed. A query
    # image is counted once for every candidate pair because the cross-encoder
    # processes that image again for each candidate. The queue executor sums
    # these counts into ``ItemOutcome.units.images``.
    input_image_counts: list[int] | None = None

    def __post_init__(self) -> None:
        """Validate consistency of output fields."""
        if len(self.scores.shape) != 1:
            msg = f"scores must be 1D [batch], got shape {self.scores.shape}"
            raise ValueError(msg)
        if self.batch_size == 0:
            self.batch_size = self.scores.shape[0]
        elif self.scores.shape[0] != self.batch_size:
            msg = f"scores batch size {self.scores.shape[0]} != {self.batch_size}"
            raise ValueError(msg)
        if self.input_token_counts is not None and len(self.input_token_counts) != self.batch_size:
            msg = f"input_token_counts length {len(self.input_token_counts)} != batch_size {self.batch_size}"
            raise ValueError(msg)
        if self.input_token_counts is not None and not all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in self.input_token_counts
        ):
            raise ValueError("input_token_counts must contain only non-negative integers")
        if self.input_image_counts is not None and len(self.input_image_counts) != self.batch_size:
            msg = f"input_image_counts length {len(self.input_image_counts)} != batch_size {self.batch_size}"
            raise ValueError(msg)
        if self.input_image_counts is not None and not all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in self.input_image_counts
        ):
            raise ValueError("input_image_counts must contain only non-negative integers")


@dataclass
class ExtractItemError:
    """Stable per-item extraction failure safe to expose to callers."""

    code: str
    message: str


@dataclass
class ExtractOutput:
    """Batched output from adapter.extract() - entities/relations/classifications/objects.

    Attributes:
        entities: Extracted entities per item. Each is list[Entity].
        classifications: Classification results per item. Each is list[Classification].
            None when the adapter does not produce classifications.
        relations: Extracted relations per item. Each is list[Relation].
            None when the adapter does not produce relations.
        objects: Detected objects per item. Each is list[DetectedObject].
            None when the adapter does not produce object detections.
        data: Structured extraction payload per item (e.g., Docling document JSON).
            None when the adapter does not produce structured data.
        errors: Stable per-item failures aligned with entities.
            None entries identify successful items.
        batch_size: Number of items processed.
    """

    entities: list[list[Entity]]  # len=batch
    classifications: list[list[Classification]] | None = None  # len=batch or None
    relations: list[list[Relation]] | None = None  # len=batch or None
    objects: list[list[DetectedObject]] | None = None  # len=batch or None
    data: list[dict[str, Any]] | None = None  # len=batch or None
    errors: list[ExtractItemError | None] | None = None  # len=batch or None

    # Metadata
    batch_size: int = 0

    # Unit-meter seam (§7.3): authoritative per-item input-token counts,
    # aligned 1:1 with ``entities`` — the REAL tokenizer length of each
    # document as counted by the extractor's own tokenizer, surfaced by
    # the queue executor as ``ItemOutcome.units.input_tokens``. ``None``
    # when the adapter owns tokenization opaquely and cannot cheaply
    # expose a count: metering then falls back to its reserve estimate
    # rather than billing an estimate as a count. Sliced/assembled
    # positionally with ``entities`` (see ExtractHandler).
    input_token_counts: list[int] | None = None

    # Unit-meter seam (§7): authoritative per-item PAGE counts, aligned 1:1
    # with ``entities`` — the canonical parse/OCR billing dimension ("$ per
    # 1k pages", design §7). Document-model parsers (docling) surface the real
    # page count they processed here, which the queue executor folds into
    # ``ItemOutcome.units.pages``. ``None`` when the adapter processes no
    # document pages (text/NER extractors like GLiNER): metering then stays on
    # its token/reserve basis. Sliced/assembled positionally with ``entities``.
    pages: list[int] | None = None

    def __post_init__(self) -> None:
        """Validate consistency of output fields."""
        if self.batch_size == 0:
            self.batch_size = len(self.entities)
        elif len(self.entities) != self.batch_size:
            msg = f"entities list length {len(self.entities)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.classifications is not None and len(self.classifications) != self.batch_size:
            msg = f"classifications list length {len(self.classifications)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.relations is not None and len(self.relations) != self.batch_size:
            msg = f"relations list length {len(self.relations)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.objects is not None and len(self.objects) != self.batch_size:
            msg = f"objects list length {len(self.objects)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.data is not None and len(self.data) != self.batch_size:
            msg = f"data list length {len(self.data)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.errors is not None and len(self.errors) != self.batch_size:
            msg = f"errors list length {len(self.errors)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.pages is not None and len(self.pages) != self.batch_size:
            msg = f"pages list length {len(self.pages)} != batch_size {self.batch_size}"
            raise ValueError(msg)

        if self.input_token_counts is not None and len(self.input_token_counts) != self.batch_size:
            msg = f"input_token_counts length {len(self.input_token_counts)} != batch_size {self.batch_size}"
            raise ValueError(msg)
