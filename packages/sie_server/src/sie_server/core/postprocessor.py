"""Postprocessor protocol and implementations for output type coercion.

See DESIGN.md Section 7.3.2 for specification.
See product/roadmap.md Project 10.12 for design rationale.

Postprocessors transform EncodeOutput in-place to add/convert output types:
- MuveraPostprocessor: multivector -> dense (for ColBERT/ColPali)
- Future: Int8Postprocessor, BinaryPostprocessor for quantization

Design principles:
- In-place mutation: postprocessors modify EncodeOutput directly
- Source/target fields: explicit about what they read and write
- Stateless transforms: no per-request state, just configuration
- Deterministic: seeded random for reproducibility

Reference implementation: https://github.com/sionic-ai/muvera-py
Paper: https://arxiv.org/abs/2405.19504
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from sie_server.core.inference_output import EncodeOutput


class Postprocessor(Protocol):
    """Protocol for output postprocessors.

    Postprocessors transform EncodeOutput in-place to add new output types
    or convert between representations (e.g., multivector -> dense via MUVERA).

    Attributes:
        source_field: Which field to read from EncodeOutput.
        target_field: Which field to write to EncodeOutput.
        target_dim: Dimension of the output (None if variable).
    """

    source_field: Literal["dense", "sparse", "multivector"]
    target_field: Literal["dense", "sparse", "multivector"]
    target_dim: int | None

    def transform(self, output: EncodeOutput, *, is_query: bool = False) -> None:
        """Transform output in-place.

        Reads from source_field, writes to target_field.

        Args:
            output: EncodeOutput to transform. Modified in-place.
            is_query: Whether the items are queries (affects aggregation in some algorithms).
        """
        ...


# =============================================================================
# Gray Code utilities (matching C++ reference implementation)
# =============================================================================


def _append_to_gray_code(gray_code: int, bit: bool) -> int:
    """Append a bit to a Gray code value.

    Gray code ensures adjacent indices differ by exactly 1 bit,
    which preserves LSH locality properties.
    """
    return (gray_code << 1) + (int(bit) ^ (gray_code & 1))


def _simhash_partition_index_gray(sketch_vector: np.ndarray) -> int:
    """Convert sketch vector to partition index using Gray code.

    Args:
        sketch_vector: SimHash sketch values [num_projections].

    Returns:
        Partition index in range [0, 2^num_projections).
    """
    partition_index = 0
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index


def _simhash_matrix_from_seed(dimension: int, num_projections: int, seed: int) -> np.ndarray:
    """Generate SimHash random projection matrix.

    Uses Gaussian distribution as in reference implementation.

    Args:
        dimension: Input vector dimension.
        num_projections: Number of random projections.
        seed: Random seed for reproducibility.

    Returns:
        Random matrix [dimension, num_projections] with N(0,1) entries.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(dimension, num_projections)).astype(np.float32)


def _apply_count_sketch(input_vector: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Apply Count Sketch projection to compress a vector.

    Count Sketch is a dimensionality reduction technique that preserves
    dot product in expectation. Each input dimension is hashed to an
    output bucket with a random sign.

    Matches reference: _apply_count_sketch_to_vector()

    Args:
        input_vector: Input vector to compress.
        output_dim: Target output dimension.
        seed: Random seed for reproducibility.

    Returns:
        Compressed vector of shape [output_dim].
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(output_dim, dtype=np.float32)
    indices = rng.integers(0, output_dim, size=input_vector.shape[0])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_vector.shape[0])
    np.add.at(out, indices, signs * input_vector)
    return out


@dataclass
class MuveraConfig:
    """Configuration for MUVERA postprocessor.

    MUVERA (Multi-Vector Retrieval Algorithm) converts variable-length
    multivector embeddings to fixed-dimension dense vectors using:
    1. SimHash partitioning (random Gaussian projections + Gray code)
    2. Per-partition aggregation (sum for queries, average for documents)
    3. Concatenation across repetitions
    4. Optional final Count Sketch compression

    The paper's recommended configuration:
    - num_repetitions: 40
    - num_simhash_projections: 6 (64 partitions)
    - projection_dim: None (identity, use full token dimension)
    - final_projection_dim: 10240 (Count Sketch compression)

    This gives: 40 * 64 * 128 = 327,680 intermediate dims -> 10,240 final

    Alternative (memory-efficient but lower quality):
    - projection_dim: 8 (AMS sketch per-token compression)
    - final_projection_dim: None (no final compression)

    This gives: 20 * 64 * 8 = 10,240 directly

    References:
        - Paper: https://arxiv.org/abs/2405.19504
        - Blog: https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/
        - Reference impl: https://github.com/sionic-ai/muvera-py

    Attributes:
        num_repetitions: Number of independent partitioning runs (R).
        num_simhash_projections: Number of random projections for partitioning.
            Creates 2^k partitions. Default: 6 (64 partitions).
        projection_dim: Dimension of vectors within each partition.
            Set to None for identity projection (paper's approach).
            Set to small value (e.g., 8) for AMS sketch (lower quality).
        final_projection_dim: If set, apply Count Sketch to compress the
            concatenated FDE to this dimension. Paper uses 10240.
        seed: Random seed for reproducibility. Default: 42.
        normalize: Whether to L2 normalize FDE vectors. Default: False.
            Set True for cosine similarity, False for inner product.
    """

    num_repetitions: int = 40  # Paper uses 40
    num_simhash_projections: int = 6  # 2^6 = 64 partitions
    projection_dim: int | None = None  # None = identity (paper's approach)
    final_projection_dim: int | None = 10240  # Count Sketch to this dim (paper uses 10240)
    seed: int = 42
    normalize: bool = False  # True for cosine, False for inner product

    @property
    def num_partitions(self) -> int:
        """Number of partitions per repetition."""
        return 2**self.num_simhash_projections

    def fde_dim(self, token_dim: int) -> int:
        """Calculate FDE output dimension.

        Args:
            token_dim: Original per-token embedding dimension.

        Returns:
            Total FDE dimension (after final projection if configured).
        """
        proj_dim = self.projection_dim or token_dim
        intermediate_dim = self.num_repetitions * self.num_partitions * proj_dim

        if self.final_projection_dim is not None:
            return self.final_projection_dim
        return intermediate_dim

    def intermediate_dim(self, token_dim: int) -> int:
        """Calculate intermediate FDE dimension before final projection.

        Args:
            token_dim: Original per-token embedding dimension.

        Returns:
            Intermediate dimension (before Count Sketch).
        """
        proj_dim = self.projection_dim or token_dim
        return self.num_repetitions * self.num_partitions * proj_dim


class MuveraPostprocessor:
    """MUVERA postprocessor: converts multivector to fixed-dimension dense.

    Implements the FDE (Fixed Dimensional Encoding) algorithm from MUVERA paper.
    Converts variable-length token embeddings (e.g., ColBERT's [seq, 128]) into
    fixed-dimension dense vectors suitable for HNSW search.

    Algorithm (matching reference implementation):
    1. For each repetition r in [0, R):
        a. Generate SimHash matrix with seed = base_seed + r
        b. Project tokens and compute partition indices via Gray code
        c. Aggregate vectors per partition (sum for queries, average for docs)
    2. Concatenate all partition vectors across repetitions

    Performance notes:
    - Uses vectorized numpy operations where possible
    - Gray code computed via vectorized bit manipulation
    - Aggregation uses np.add.at for efficient scatter-add

    Example:
        >>> config = MuveraConfig(num_repetitions=10, num_simhash_projections=6)
        >>> postprocessor = MuveraPostprocessor(token_dim=128, config=config)
        >>> postprocessor.target_dim  # 10 * 64 * 128 = 81920
        81920
    """

    source_field: Literal["dense", "sparse", "multivector"] = "multivector"
    target_field: Literal["dense", "sparse", "multivector"] = "dense"

    def __init__(self, token_dim: int, config: MuveraConfig | None = None) -> None:
        """Initialize MUVERA postprocessor.

        Args:
            token_dim: Dimension of per-token embeddings (e.g., 128 for ColBERT).
            config: MUVERA configuration. Uses defaults if not provided.
        """
        self.token_dim = token_dim
        self.config = config or MuveraConfig()

        # Determine projection dimension (None = identity = use token_dim)
        self._proj_dim = self.config.projection_dim or token_dim
        self._use_identity = self.config.projection_dim is None

        # Calculate dimensions
        self._intermediate_dim = self.config.intermediate_dim(token_dim)
        self.target_dim = self.config.fde_dim(token_dim)

        # Pre-compute Gray code lookup table for fast partition index conversion
        # Gray code: adjacent indices differ by 1 bit (preserves LSH locality)
        self._gray_lut = self._build_gray_lut(self.config.num_simhash_projections)

    def _build_gray_lut(self, num_bits: int) -> np.ndarray:
        """Build lookup table for binary -> Gray code conversion.

        For each binary number b, gray(b) = b XOR (b >> 1).
        We store the reverse mapping: binary value at each index.
        """
        size = 2**num_bits
        # Standard binary to Gray: gray = n ^ (n >> 1)
        # But reference uses append_to_gray_code which builds differently
        # Let's match the reference exactly by computing partition indices
        # the same way the reference does
        return np.arange(size, dtype=np.int32)  # Will compute Gray inline

    def transform(self, output: EncodeOutput, *, is_query: bool = False) -> None:
        """Transform multivector to dense FDE.

        Uses batched processing for efficiency when multiple items present.

        Args:
            output: EncodeOutput with multivector field populated.
            is_query: If True, use sum aggregation. If False, use average.
        """
        if output.multivector is None:
            msg = "MuveraPostprocessor requires multivector field"
            raise ValueError(msg)

        batch_size = len(output.multivector)
        if batch_size == 0:
            output.dense = np.zeros((0, self.target_dim), dtype=np.float32)
            output.dense_dim = self.target_dim
            return

        # Process batch - could be parallelized but numpy is already fast
        fde_batch = np.zeros((batch_size, self.target_dim), dtype=np.float32)

        for i, token_embeddings in enumerate(output.multivector):
            fde_batch[i] = self._compute_fde_single(token_embeddings, is_query=is_query)

        # Optionally L2 normalize FDE vectors for cosine similarity compatibility
        if self.config.normalize:
            norms = np.linalg.norm(fde_batch, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
            fde_batch = fde_batch / norms

        output.dense = fde_batch
        output.dense_dim = self.target_dim

    def _compute_fde_single(self, token_embeddings: np.ndarray, *, is_query: bool) -> np.ndarray:
        """Compute FDE for a single multivector.

        Matches reference implementation algorithm exactly.

        Args:
            token_embeddings: Token embeddings [num_tokens, token_dim].
            is_query: Whether to use sum (True) or average (False) aggregation.

        Returns:
            FDE vector [fde_dim].
        """
        num_tokens, original_dim = token_embeddings.shape
        if num_tokens == 0:
            return np.zeros(self.target_dim, dtype=np.float32)

        num_partitions = self.config.num_partitions
        proj_dim = self._proj_dim
        rep_block_size = num_partitions * proj_dim

        # Pre-allocate intermediate output (before final projection)
        intermediate_fde = np.zeros(self._intermediate_dim, dtype=np.float32)

        for rep_num in range(self.config.num_repetitions):
            # Each repetition uses seed + rep_num (matching reference)
            current_seed = self.config.seed + rep_num

            # Step 1: SimHash projection to get sketches
            simhash_matrix = _simhash_matrix_from_seed(original_dim, self.config.num_simhash_projections, current_seed)
            sketches = token_embeddings @ simhash_matrix  # [num_tokens, num_proj]

            # Step 2: Compute partition indices using Gray code
            partition_indices = self._sketches_to_gray_partitions(sketches)

            # Step 3: Project tokens (identity or AMS sketch)
            if self._use_identity:
                projected = token_embeddings
            else:
                # AMS sketch projection (sparse random projection)
                proj_matrix = self._ams_projection_matrix(original_dim, proj_dim, current_seed)
                projected = token_embeddings @ proj_matrix

            # Step 4: Aggregate per partition (vectorized)
            rep_fde = self._aggregate_partitions_vectorized(
                projected, partition_indices, num_partitions, is_query=is_query
            )

            # Step 5: Store in intermediate output
            rep_start = rep_num * rep_block_size
            intermediate_fde[rep_start : rep_start + rep_block_size] = rep_fde

        # Step 6: Apply final Count Sketch projection if configured
        if self.config.final_projection_dim is not None:
            return _apply_count_sketch(intermediate_fde, self.config.final_projection_dim, self.config.seed)

        return intermediate_fde

    def _sketches_to_gray_partitions(self, sketches: np.ndarray) -> np.ndarray:
        """Convert SimHash sketches to partition indices using Gray code.

        Matches reference: _simhash_partition_index_gray()

        Args:
            sketches: Sketch values [num_tokens, num_projections].

        Returns:
            Partition indices [num_tokens] in range [0, num_partitions).
        """
        num_tokens = sketches.shape[0]

        # Vectorized Gray code computation
        # For each token, compute partition index by iterating through projections
        # gray_code = 0; for bit in bits: gray_code = (gray_code << 1) + (bit ^ (gray_code & 1))

        # This is tricky to vectorize perfectly, but we can do it with cumulative ops
        bits = (sketches > 0).astype(np.int32)  # [num_tokens, num_proj]

        # Compute Gray code indices - need to do this per-token
        # For small num_proj (typically 4-8), loop is fast enough
        partition_indices = np.zeros(num_tokens, dtype=np.int32)
        for i in range(self.config.num_simhash_projections):
            partition_indices = (partition_indices << 1) + (bits[:, i] ^ (partition_indices & 1))

        return partition_indices

    def _ams_projection_matrix(self, input_dim: int, output_dim: int, seed: int) -> np.ndarray:
        """Generate AMS sketch projection matrix.

        Sparse random matrix with one ±1 per row at random column.
        Matches reference: _ams_projection_matrix_from_seed()
        """
        rng = np.random.default_rng(seed)
        out = np.zeros((input_dim, output_dim), dtype=np.float32)
        indices = rng.integers(0, output_dim, size=input_dim)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_dim)
        out[np.arange(input_dim), indices] = signs
        return out

    def _aggregate_partitions_vectorized(
        self,
        projected: np.ndarray,
        partition_indices: np.ndarray,
        num_partitions: int,
        *,
        is_query: bool,
    ) -> np.ndarray:
        """Aggregate projected vectors per partition using vectorized ops.

        Args:
            projected: Projected vectors [num_tokens, proj_dim].
            partition_indices: Partition index per token [num_tokens].
            num_partitions: Number of partitions.
            is_query: If True, sum. If False, average.

        Returns:
            Flattened aggregated vectors [num_partitions * proj_dim].
        """
        proj_dim = projected.shape[1]

        # Initialize accumulators
        sums = np.zeros((num_partitions, proj_dim), dtype=np.float32)
        counts = np.zeros(num_partitions, dtype=np.int32)

        # Vectorized scatter-add
        np.add.at(sums, partition_indices, projected)
        np.add.at(counts, partition_indices, 1)

        if not is_query:
            # Documents: convert sums to averages where count > 0
            mask = counts > 0
            # Vectorized division with broadcasting
            sums[mask] /= counts[mask, np.newaxis]

        return sums.ravel()


# =============================================================================
# Quantization Postprocessor
# =============================================================================


class QuantizePostprocessor:
    """Quantization postprocessor: converts embeddings to target dtype.

    Unlike MUVERA which transforms between fields, quantization transforms
    the dtype of existing fields in-place. Supports:
    - float32: Full precision (default, no-op)
    - float16: Half precision (2x smaller)
    - int8: Symmetric quantization (4x smaller, ~1% quality loss)
    - uint8: Linear quantization (4x smaller, Qdrant format)
    - binary/ubinary: Bit-packed (32x smaller, for Hamming distance)

    Triggered by `output_dtype` in runtime options.

    Example:
        >>> postprocessor = QuantizePostprocessor()
        >>> postprocessor.quantize(output, output_dtype="int8")
    """

    # Quantization applies to all fields, not a source→target transform
    source_field = None
    target_field = None
    target_dim = None

    def quantize(
        self,
        output: EncodeOutput,
        *,
        output_dtype: str = "float32",
    ) -> None:
        """Quantize embeddings to target dtype in-place.

        Args:
            output: EncodeOutput to quantize. Modified in-place.
            output_dtype: Target dtype (float32, float16, int8, uint8, binary).
        """
        if output_dtype == "float32":
            # No transformation needed, but ensure float32
            if output.dense is not None:
                output.dense = output.dense.astype(np.float32)
            if output.sparse is not None:
                for sv in output.sparse:
                    sv.values = sv.values.astype(np.float32)
            if output.multivector is not None:
                output.multivector = [mv.astype(np.float32) for mv in output.multivector]
            return

        if output_dtype == "float16":
            if output.dense is not None:
                output.dense = output.dense.astype(np.float16)
            if output.sparse is not None:
                for sv in output.sparse:
                    sv.values = sv.values.astype(np.float16)
            if output.multivector is not None:
                output.multivector = [mv.astype(np.float16) for mv in output.multivector]
            return

        if output_dtype == "int8":
            if output.dense is not None:
                output.dense = _quantize_int8_batch(output.dense)
            # Sparse: int8 doesn't make sense (indices + values), keep float32
            if output.multivector is not None:
                output.multivector = [_quantize_int8_batch(mv) for mv in output.multivector]
            return

        if output_dtype == "uint8":
            if output.dense is not None:
                output.dense = _quantize_uint8_batch(output.dense)
            # Sparse: uint8 doesn't make sense, keep float32
            if output.multivector is not None:
                output.multivector = [_quantize_uint8_batch(mv) for mv in output.multivector]
            return

        if output_dtype in ("binary", "ubinary"):
            if output.dense is not None:
                output.dense = np.packbits((output.dense > 0).astype(np.uint8), axis=-1)
            # Sparse: binary doesn't make sense, keep float32
            if output.multivector is not None:
                output.multivector = [np.packbits((mv > 0).astype(np.uint8), axis=-1) for mv in output.multivector]
            return

        raise ValueError(f"Unsupported output_dtype: {output_dtype}")


def _quantize_int8_batch(x: np.ndarray) -> np.ndarray:
    """Quantize float embedding to int8 using symmetric scalar quantization.

    Values are mapped to [-127, 127] range per-row.
    """
    x = x.astype(np.float32)
    if x.ndim == 1:
        scale = np.max(np.abs(x))
        if scale == 0:
            return np.zeros_like(x, dtype=np.int8)
        return np.round(x / scale * 127).astype(np.int8)
    # Batch: scale per row
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(scale == 0, 1, scale)
    return np.round(x / scale * 127).astype(np.int8)


def _quantize_uint8_batch(x: np.ndarray) -> np.ndarray:
    """Quantize float embedding to uint8 using linear mapping [0, 255]."""
    x = x.astype(np.float32)
    if x.ndim == 1:
        min_val, max_val = np.min(x), np.max(x)
        range_val = max_val - min_val
        if range_val == 0:
            return np.full_like(x, 128, dtype=np.uint8)
        return np.round((x - min_val) / range_val * 255).astype(np.uint8)
    # Batch: scale per row
    min_val = np.min(x, axis=-1, keepdims=True)
    max_val = np.max(x, axis=-1, keepdims=True)
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)
    return np.round((x - min_val) / range_val * 255).astype(np.uint8)
