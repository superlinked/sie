"""Output types for SIE Server API.

These types define the structure of embedding outputs: dense vectors, sparse vectors,
and multi-vectors (for late-interaction models like ColBERT).

Per DESIGN.md Section 5.9, numpy arrays are preserved for msgpack serialization.
For JSON fallback, arrays are converted to lists on serialization.

Using TypedDict for zero runtime overhead.
"""

from typing import TYPE_CHECKING, TypedDict

from sie_sdk.types import DType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class DenseVector(TypedDict, total=False):
    """Dense embedding vector.

    A fixed-dimension vector where every dimension has a value.
    Used by most embedding models (BERT, E5, GTE, etc.).

    Attributes:
        dims: Vector dimensionality (e.g., 768, 1024).
        dtype: Data type of values.
        values: Vector values as numpy array, shape: [dims].
    """

    dims: int
    dtype: DType
    values: "NDArray[np.float32]"


class SparseVector(TypedDict, total=False):
    """Sparse embedding vector.

    A vector where only non-zero dimensions are stored.
    Used by lexical models (SPLADE, BM25) and hybrid models (BGE-M3).

    Attributes:
        dims: Vocabulary size (None if unknown).
        dtype: Data type of values.
        indices: Non-zero dimension indices as numpy array.
        values: Values at those indices as numpy array.
    """

    dims: int | None
    dtype: DType
    indices: "NDArray[np.int32]"
    values: "NDArray[np.float32]"


class MultiVector(TypedDict, total=False):
    """Multi-vector embedding (per-token embeddings).

    Used by late-interaction models (ColBERT, ColPali) where each token
    gets its own embedding vector. Scoring uses MaxSim between query
    and document token embeddings.

    Attributes:
        token_dims: Per-token embedding dimension (e.g., 128).
        num_tokens: Number of tokens (varies per document).
        dtype: Data type of values.
        values: Token embeddings as numpy array, shape: [num_tokens, token_dims].
    """

    token_dims: int
    num_tokens: int
    dtype: DType
    values: "NDArray[np.float32]"


class EncodeResult(TypedDict, total=False):
    """Result of encoding a single item.

    Contains the item ID (if provided) and one or more output representations
    depending on what was requested and what the model supports.

    Attributes:
        id: Item ID (echoed from request).
        dense: Dense embedding.
        sparse: Sparse embedding.
        multivector: Multi-vector embedding.
    """

    id: str
    dense: DenseVector
    sparse: SparseVector
    multivector: MultiVector
