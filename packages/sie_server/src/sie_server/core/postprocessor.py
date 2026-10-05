"""Postprocessor protocol and implementations for output type coercion.

Postprocessors transform EncodeOutput in-place to add/convert output types:
- MuveraPostprocessor: multivector -> dense (for ColBERT/ColPali)
- SmvePostprocessor: multivector -> sparse (Sparse Multi-Vector Encoding)
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

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from sie_server.core.inference_output import SparseVector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sie_server.core.inference_output import EncodeOutput

logger = logging.getLogger(__name__)


def _readonly(array: np.ndarray) -> np.ndarray:
    """Mark a cached random structure immutable and return it."""
    array.setflags(write=False)
    return array


def _count_sketch_index_dtype(output_dim: int) -> type[np.unsignedinteger]:
    """Return the narrowest unsigned dtype that can address output_dim."""
    max_index = output_dim - 1
    if max_index <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_index <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_index <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


@dataclass(frozen=True, slots=True)
class _MuveraRandomCache:
    """Immutable, atomically published random structures for one MUVERA instance."""

    simhash_matrices: tuple[np.ndarray, ...]
    ams_projection_matrices: tuple[np.ndarray, ...]
    count_sketch_indices: np.ndarray | None
    count_sketch_signs: np.ndarray | None


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

    Alternative (memory-efficient; the text ColBERT models use this — #1493):
    - projection_dim: 8 (AMS sketch per-token compression)
    - final_projection_dim: None (no final Count Sketch — it is harmful here)

    This gives: 40 * 64 * 8 = 20,480 directly

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
            concatenated FDE to this dimension. Paper uses 10240. A value
            ``>= intermediate_dim`` is ignored (the sketch is skipped), since
            a same-or-larger Count Sketch can only destroy information (#1493).
        seed: Random seed for reproducibility. Default: 42.
        normalize: Whether to L2 normalize FDE vectors. Default: False.
            Set True for cosine similarity, False for inner product.
        center_tokens: If True, subtract the per-multivector mean token before
            SimHash partitioning. A dominant shared DC component otherwise makes
            SimHash bucket all tokens together (near-tied FDEs that collapse
            ranking, while MaxSim stays healthy); centering partitions on the
            discriminative residual. Improves FDE quality for near-collinear
            models. Default: False (#1528).
    """

    num_repetitions: int = 40  # Paper uses 40
    num_simhash_projections: int = 6  # 2^6 = 64 partitions
    projection_dim: int | None = None  # None = identity (paper's approach)
    final_projection_dim: int | None = 10240  # Count Sketch to this dim (paper uses 10240)
    seed: int = 42
    normalize: bool = False  # True for cosine, False for inner product
    center_tokens: bool = False  # subtract per-multivector mean before SimHash (#1528)

    @property
    def num_partitions(self) -> int:
        """Number of partitions per repetition."""
        return 2**self.num_simhash_projections

    def _effective_final_dim(self, token_dim: int) -> int | None:
        """Final Count-Sketch target dim, or None when no sketch should run.

        A Count-Sketch is only ever a *reduction*. When the configured
        final_projection_dim is >= the intermediate dim it can only destroy
        information (same-dim destructive hashing, see #1493), so it is
        skipped and the intermediate FDE is returned unprojected.
        """
        if self.final_projection_dim is None:
            return None
        if self.final_projection_dim >= self.intermediate_dim(token_dim):
            return None
        return self.final_projection_dim

    def fde_dim(self, token_dim: int) -> int:
        """Calculate FDE output dimension.

        Args:
            token_dim: Original per-token embedding dimension.

        Returns:
            Total FDE dimension (after final projection if configured).
        """
        final = self._effective_final_dim(token_dim)
        return final if final is not None else self.intermediate_dim(token_dim)

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

        # Calculate dimensions. ``_final_dim`` is the effective Count-Sketch
        # target dim (None = skip the sketch); ``target_dim`` stays consistent
        # with it via ``fde_dim`` (both route through ``_effective_final_dim``).
        self._intermediate_dim = self.config.intermediate_dim(token_dim)
        self._final_dim = self.config._effective_final_dim(token_dim)
        self.target_dim = self.config.fde_dim(token_dim)

        # Warn once (at construction) when a configured final_projection_dim is
        # ignored because it would not reduce the FDE (#1493 footgun guard).
        if (
            self.config.final_projection_dim is not None
            and self.config.final_projection_dim >= self.config.intermediate_dim(token_dim)
        ):
            logger.warning(
                "MUVERA final Count-Sketch skipped: final_projection_dim=%d >= intermediate_dim=%d "
                "(token_dim=%d) — a same-or-larger sketch only destroys information; "
                "returning unprojected %d-dim FDE (#1493).",
                self.config.final_projection_dim,
                self.config.intermediate_dim(token_dim),
                token_dim,
                self.target_dim,
            )

        # Pre-compute Gray code lookup table for fast partition index conversion
        # Gray code: adjacent indices differ by 1 bit (preserves LSH locality)
        self._gray_lut = self._build_gray_lut(self.config.num_simhash_projections)

        # Configuration-derived random structures are initialized lazily on the
        # first non-empty transform. Building into a local immutable object and
        # publishing it once keeps concurrent first use safe without adding a
        # process-wide executor or charging unused model profiles for the cache.
        self._random_cache: _MuveraRandomCache | None = None
        self._random_cache_lock = threading.Lock()

    def _build_random_cache(self) -> _MuveraRandomCache:
        """Build all random structures locally, preserving legacy RNG calls."""
        simhash_matrices = tuple(
            _readonly(
                _simhash_matrix_from_seed(
                    self.token_dim,
                    self.config.num_simhash_projections,
                    self.config.seed + rep_num,
                )
            )
            for rep_num in range(self.config.num_repetitions)
        )
        ams_projection_matrices = (
            ()
            if self._use_identity
            else tuple(
                _readonly(
                    self._ams_projection_matrix(
                        self.token_dim,
                        self._proj_dim,
                        self.config.seed + rep_num,
                    )
                )
                for rep_num in range(self.config.num_repetitions)
            )
        )

        count_sketch_indices: np.ndarray | None = None
        count_sketch_signs: np.ndarray | None = None
        if self._final_dim is not None:
            # Preserve the legacy RNG calls, their order, and their original
            # dtypes exactly. In particular, asking Generator.integers for a
            # narrower dtype changes how much RNG state it consumes and would
            # therefore change the signs generated by the following call.
            rng = np.random.default_rng(self.config.seed)
            index_dtype = _count_sketch_index_dtype(self._final_dim)
            # Compress the legacy int64 draw before allocating signs. This does
            # not consume RNG state, so the following legacy sign draw remains
            # bit-exact while avoiding overlap between both large raw arrays.
            count_sketch_indices = _readonly(
                rng.integers(0, self._final_dim, size=self._intermediate_dim).astype(index_dtype)
            )
            count_sketch_signs = _readonly(
                rng.choice(
                    np.array([-1.0, 1.0], dtype=np.float32),
                    size=self._intermediate_dim,
                ).astype(np.int8)
            )

        return _MuveraRandomCache(
            simhash_matrices=simhash_matrices,
            ams_projection_matrices=ams_projection_matrices,
            count_sketch_indices=count_sketch_indices,
            count_sketch_signs=count_sketch_signs,
        )

    def _get_random_cache(self) -> _MuveraRandomCache:
        """Return the per-instance cache, publishing one complete build."""
        cache = self._random_cache
        if cache is None:
            with self._random_cache_lock:
                cache = self._random_cache
                if cache is None:
                    cache = self._build_random_cache()
                    self._random_cache = cache
        return cache

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

        # Keep items and repetitions sequential on the existing serialized
        # inference/postprocessing path.
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
        num_tokens = token_embeddings.shape[0]
        if num_tokens == 0:
            return np.zeros(self.target_dim, dtype=np.float32)

        # Subtract the per-multivector mean token before partitioning/projection.
        # A dominant shared DC component makes SimHash bucket all tokens together
        # (near-tied FDEs -> collapsed ranking), so centering partitions on the
        # discriminative residual. MaxSim is DC-shift-tolerant and unaffected (#1528).
        if self.config.center_tokens:
            token_embeddings = token_embeddings - token_embeddings.mean(axis=0, keepdims=True)

        random_cache = self._get_random_cache()
        num_partitions = self.config.num_partitions
        proj_dim = self._proj_dim
        rep_block_size = num_partitions * proj_dim

        if self._final_dim is None:
            fde = np.zeros(self._intermediate_dim, dtype=np.float32)
        else:
            fde = np.zeros(self._final_dim, dtype=np.float32)

        # Compute repetitions in legacy order. For a final Count Sketch,
        # streaming each contiguous repetition block in that same order is
        # bit-equivalent to sketching the fully materialized intermediate
        # vector while avoiding the large per-item intermediate allocation.
        for rep_num in range(self.config.num_repetitions):
            rep_fde = self._compute_repetition(token_embeddings, rep_num, is_query=is_query)
            rep_start = rep_num * rep_block_size
            rep_end = rep_start + rep_block_size
            if self._final_dim is None:
                fde[rep_start:rep_end] = rep_fde
            else:
                if random_cache.count_sketch_indices is None or random_cache.count_sketch_signs is None:
                    raise RuntimeError("MUVERA Count-Sketch cache is not initialized")
                np.add.at(
                    fde,
                    random_cache.count_sketch_indices[rep_start:rep_end],
                    random_cache.count_sketch_signs[rep_start:rep_end] * rep_fde,
                )

        return fde

    def _compute_repetition(
        self,
        token_embeddings: np.ndarray,
        rep_num: int,
        *,
        is_query: bool,
    ) -> np.ndarray:
        """Compute one independent MUVERA repetition from immutable caches."""
        random_cache = self._get_random_cache()
        sketches = token_embeddings @ random_cache.simhash_matrices[rep_num]
        partition_indices = self._sketches_to_gray_partitions(sketches)

        if self._use_identity:
            projected = token_embeddings
        else:
            projected = token_embeddings @ random_cache.ams_projection_matrices[rep_num]

        return self._aggregate_partitions_vectorized(
            projected,
            partition_indices,
            self.config.num_partitions,
            is_query=is_query,
        )

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
# SMVE: Sparse Multi-Vector Encoding
# =============================================================================

# Upper bound on one chunk of token projections ([tokens, width] float32) and on
# one group's accumulators ([items, output_dim] float32, two of them for
# documents), so long pages and large batches stay within a fixed memory budget.
_SMVE_PROJECTION_CHUNK_BYTES = 64 * 1024 * 1024
_SMVE_ACCUMULATOR_BYTES = 256 * 1024 * 1024


@dataclass
class SmveConfig:
    """Configuration for the SMVE postprocessor.

    SMVE (Sparse Multi-Vector Encoding) turns a multivector into one sparse
    vector whose dot product with another approximates MaxSim, so a sparse
    (inverted) index can serve as the first retrieval stage, with MaxSim over
    the stored token vectors re-ranking the candidates:

    1. Project every token onto ``width`` random unit vectors (anchors).
    2. Keep each token's ``k`` largest projections.
    3. Pool the tokens: a query sums them; a document averages the non-zero
       contributions in each dimension.

    ``num_repetitions`` runs the three steps with independent anchors and
    concatenates the results. Storage and compute scale with ``k``, not with
    ``width``: an item of ``n`` tokens has at most ``k * n * num_repetitions``
    non-zeros. ``max_nonzeros`` optionally keeps only the largest values of
    each item, which bounds long documents.

    Queries and documents must use the same settings: the anchors come from
    the seed, so width, seed and repetitions are part of the index.

    Reference: M. Spisak and M. Galovic, "SMVE: Multi-Vector Retrieval That
    Just Works", TopK blog, March 2026,
    https://www.topk.io/blog/20260311-smve-multi-vector-retrieval

    Attributes:
        width: Number of anchors per repetition (dimensions per repetition).
        k: Projections kept per token.
        num_repetitions: Independent anchor sets, concatenated.
        seed: Seed of the first anchor set; repetition ``r`` uses ``seed + r``.
        max_nonzeros: If set, keep only this many largest values per item.
    """

    width: int = 65536
    k: int = 32
    num_repetitions: int = 1
    seed: int = 42
    max_nonzeros: int | None = None

    def __post_init__(self) -> None:
        """Reject settings that cannot produce a valid encoding."""
        if self.width < 1:
            msg = f"SMVE width must be at least 1, got {self.width}"
            raise ValueError(msg)
        if not 1 <= self.k <= self.width:
            msg = f"SMVE k must be in 1..width ({self.width}), got {self.k}"
            raise ValueError(msg)
        if self.num_repetitions < 1:
            msg = f"SMVE num_repetitions must be at least 1, got {self.num_repetitions}"
            raise ValueError(msg)
        if self.max_nonzeros is not None and self.max_nonzeros < 1:
            msg = f"SMVE max_nonzeros must be at least 1 when set, got {self.max_nonzeros}"
            raise ValueError(msg)

    @property
    def output_dim(self) -> int:
        """Dimension of the sparse output: ``width * num_repetitions``."""
        return self.width * self.num_repetitions


class SmvePostprocessor:
    """SMVE postprocessor: converts multivector to sparse.

    The projection is one large matrix multiply (for 2,048-number tokens and a
    65,536 width, a 1,230-token page is about 330 GFLOP), so it runs in torch on
    ``device``, normally the model's own device, with CPU as the fallback.
    Items of a batch are projected together, in chunks of tokens.

    The anchors are drawn with numpy from the seed and normalized to unit
    length, so every process and device builds the same ones. Projections on a
    GPU follow its matmul precision (SIE enables TF32), so values can differ
    from a CPU encoding in the last bits; that is far below what retrieval
    resolves.
    """

    source_field: Literal["dense", "sparse", "multivector"] = "multivector"
    target_field: Literal["dense", "sparse", "multivector"] = "sparse"

    def __init__(self, token_dim: int, config: SmveConfig | None = None, *, device: str | None = None) -> None:
        """Initialize the SMVE postprocessor.

        Args:
            token_dim: Dimension of the per-token embeddings.
            config: SMVE configuration. Uses defaults if not provided.
            device: Torch device for the projections (default ``"cpu"``).
        """
        if token_dim < 1:
            msg = f"SMVE token_dim must be at least 1, got {token_dim}"
            raise ValueError(msg)
        self.token_dim = token_dim
        self.config = config or SmveConfig()
        self.target_dim = self.config.output_dim
        self.device = device or "cpu"
        # Built lazily on first use: a 2,048 x 65,536 anchor set is 512 MiB.
        self._anchors: tuple[Any, ...] | None = None
        self._anchors_lock = threading.Lock()

    def _get_anchors(self) -> tuple[Any, ...]:
        """Return the anchor matrices, building them once."""
        anchors = self._anchors
        if anchors is None:
            with self._anchors_lock:
                anchors = self._anchors
                if anchors is None:
                    anchors = tuple(self._build_anchors(rep) for rep in range(self.config.num_repetitions))
                    self._anchors = anchors
        return anchors

    def _build_anchors(self, rep: int) -> Any:
        """One repetition's anchors: ``[token_dim, width]`` unit columns on the device."""
        import torch

        rng = np.random.default_rng(self.config.seed + rep)
        anchors = rng.standard_normal((self.token_dim, self.config.width), dtype=np.float32)
        anchors /= np.linalg.norm(anchors, axis=0, keepdims=True)
        return torch.from_numpy(anchors).to(self.device)

    def transform(self, output: EncodeOutput, *, is_query: bool = False) -> None:
        """Add the SMVE encoding of every item as ``output.sparse``.

        Args:
            output: EncodeOutput with the multivector field populated.
            is_query: If True, sum the tokens. If False, average them.
        """
        if output.multivector is None:
            msg = "SmvePostprocessor requires multivector field"
            raise ValueError(msg)
        output.sparse = self.encode(output.multivector, is_query=is_query)

    def encode(self, multivectors: Sequence[np.ndarray], *, is_query: bool) -> list[SparseVector]:
        """Encode each item's token vectors as one sparse vector.

        Args:
            multivectors: Per-item token embeddings, each ``[num_tokens, token_dim]``.
            is_query: If True, sum the tokens. If False, average them.

        Returns:
            One sparse vector per item, indices sorted ascending.
        """
        for tokens in multivectors:
            if tokens.ndim != 2 or (tokens.shape[0] and tokens.shape[1] != self.token_dim):
                msg = f"SMVE expects [num_tokens, {self.token_dim}] token embeddings, got shape {tokens.shape}"
                raise ValueError(msg)
        items_per_group = max(1, _SMVE_ACCUMULATOR_BYTES // (8 * self.config.output_dim))
        encoded: list[SparseVector] = []
        for start in range(0, len(multivectors), items_per_group):
            encoded.extend(self._encode_group(multivectors[start : start + items_per_group], is_query=is_query))
        return encoded

    def _encode_group(self, multivectors: Sequence[np.ndarray], *, is_query: bool) -> list[SparseVector]:
        """Encode a group of items together: one projection per chunk of their tokens."""
        import torch

        config = self.config
        lengths = [int(tokens.shape[0]) for tokens in multivectors]
        nonempty = [tokens for tokens in multivectors if tokens.shape[0]]
        if not nonempty:
            return [_empty_sparse_vector() for _ in multivectors]

        batch, output_dim, width = len(multivectors), config.output_dim, config.width
        tokens = torch.from_numpy(np.ascontiguousarray(np.concatenate(nonempty), dtype=np.float32)).to(self.device)
        owners = torch.repeat_interleave(torch.arange(batch), torch.tensor(lengths)).to(self.device)

        # Accumulate per (item, dimension) in one flat buffer.
        sums = torch.zeros(batch * output_dim, dtype=torch.float32, device=self.device)
        counts = None if is_query else torch.zeros_like(sums)
        chunk = max(1, _SMVE_PROJECTION_CHUNK_BYTES // (4 * width))
        for rep, anchors in enumerate(self._get_anchors()):
            for start in range(0, tokens.shape[0], chunk):
                stop = start + chunk
                values, indices = torch.topk(tokens[start:stop] @ anchors, config.k, dim=1)
                flat = (owners[start:stop, None] * output_dim + rep * width + indices).reshape(-1)
                sums.index_add_(0, flat, values.reshape(-1))
                if counts is not None:
                    counts.index_add_(0, flat, (values != 0).to(torch.float32).reshape(-1))

        pooled = (sums if counts is None else sums / counts.clamp_min(1.0)).view(batch, output_dim)
        owner_ids, dims = torch.nonzero(pooled, as_tuple=True)
        values = pooled[owner_ids, dims]
        per_item = torch.bincount(owner_ids, minlength=batch).tolist()
        dims_np = dims.cpu().numpy().astype(np.int32)
        values_np = values.cpu().numpy().astype(np.float32)

        encoded: list[SparseVector] = []
        offset = 0
        for count in per_item:
            item_dims, item_values = dims_np[offset : offset + count], values_np[offset : offset + count]
            offset += count
            encoded.append(self._finish(item_dims, item_values))
        return encoded

    def _finish(self, dims: np.ndarray, values: np.ndarray) -> SparseVector:
        """Apply ``max_nonzeros`` and return the item's vector with sorted indices."""
        limit = self.config.max_nonzeros
        if limit is not None and dims.size > limit:
            keep = np.argpartition(-np.abs(values), limit - 1)[:limit]
            dims, values = dims[keep], values[keep]
        order = np.argsort(dims, kind="stable")
        return SparseVector(indices=dims[order], values=values[order])


def _empty_sparse_vector() -> SparseVector:
    """Encoding of an item with no tokens."""
    return SparseVector(indices=np.zeros(0, dtype=np.int32), values=np.zeros(0, dtype=np.float32))


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
