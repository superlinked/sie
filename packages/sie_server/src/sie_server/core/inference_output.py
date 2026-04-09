"""Typed output types for standardized batched inference results.

See DESIGN.md Section 7.3 for specification.
See product/roadmap.md Project 10.12 for design rationale.

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
        batch_size: Number of items processed.
    """

    entities: list[list[Entity]]  # len=batch
    classifications: list[list[Classification]] | None = None  # len=batch or None
    relations: list[list[Relation]] | None = None  # len=batch or None
    objects: list[list[DetectedObject]] | None = None  # len=batch or None

    # Metadata
    batch_size: int = 0

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
