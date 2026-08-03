"""Tests for postprocessor protocol and implementations."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from threading import Event

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import (
    MuveraConfig,
    MuveraPostprocessor,
    _append_to_gray_code,
    _simhash_partition_index_gray,
)


def _legacy_count_sketch(input_vector: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Faithful serial implementation used before MUVERA streaming."""
    rng = np.random.default_rng(seed)
    output = np.zeros(output_dim, dtype=np.float32)
    indices = rng.integers(0, output_dim, size=input_vector.shape[0])
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=input_vector.shape[0])
    np.add.at(output, indices, signs * input_vector)
    return output


def _legacy_muvera_single(
    token_embeddings: np.ndarray,
    config: MuveraConfig,
    *,
    is_query: bool,
) -> np.ndarray:
    """Faithful copy of the pre-optimization serial MUVERA algorithm."""
    num_tokens, token_dim = token_embeddings.shape
    target_dim = config.fde_dim(token_dim)
    if num_tokens == 0:
        return np.zeros(target_dim, dtype=np.float32)

    if config.center_tokens:
        token_embeddings = token_embeddings - token_embeddings.mean(axis=0, keepdims=True)

    projection_dim = config.projection_dim or token_dim
    num_partitions = config.num_partitions
    repetition_size = num_partitions * projection_dim
    intermediate = np.zeros(config.intermediate_dim(token_dim), dtype=np.float32)

    for repetition in range(config.num_repetitions):
        seed = config.seed + repetition
        rng = np.random.default_rng(seed)
        simhash_matrix = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(token_dim, config.num_simhash_projections),
        ).astype(np.float32)
        sketches = token_embeddings @ simhash_matrix

        bits = (sketches > 0).astype(np.int32)
        partition_indices = np.zeros(num_tokens, dtype=np.int32)
        for projection in range(config.num_simhash_projections):
            partition_indices = (partition_indices << 1) + (bits[:, projection] ^ (partition_indices & 1))

        if config.projection_dim is None:
            projected = token_embeddings
        else:
            rng = np.random.default_rng(seed)
            projection_matrix = np.zeros((token_dim, projection_dim), dtype=np.float32)
            indices = rng.integers(0, projection_dim, size=token_dim)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=token_dim)
            projection_matrix[np.arange(token_dim), indices] = signs
            projected = token_embeddings @ projection_matrix

        sums = np.zeros((num_partitions, projection_dim), dtype=np.float32)
        counts = np.zeros(num_partitions, dtype=np.int32)
        np.add.at(sums, partition_indices, projected)
        np.add.at(counts, partition_indices, 1)
        if not is_query:
            populated = counts > 0
            sums[populated] /= counts[populated, np.newaxis]

        start = repetition * repetition_size
        intermediate[start : start + repetition_size] = sums.ravel()

    final_dim = config._effective_final_dim(token_dim)
    if final_dim is not None:
        return _legacy_count_sketch(intermediate, final_dim, config.seed)
    return intermediate


def _legacy_muvera_batch(
    multivectors: Sequence[np.ndarray],
    token_dim: int,
    config: MuveraConfig,
    *,
    is_query: bool,
) -> np.ndarray:
    """Run the legacy algorithm and normalization in the original order."""
    dense = np.zeros((len(multivectors), config.fde_dim(token_dim)), dtype=np.float32)
    for index, token_embeddings in enumerate(multivectors):
        dense[index] = _legacy_muvera_single(token_embeddings, config, is_query=is_query)
    if config.normalize:
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / np.where(norms > 0, norms, 1.0)
    return dense


class TestGrayCode:
    """Tests for Gray code utilities."""

    def test_append_to_gray_code_first_bit(self) -> None:
        """First bit appended correctly."""
        assert _append_to_gray_code(0, False) == 0  # 0 -> 0
        assert _append_to_gray_code(0, True) == 1  # 0 -> 1

    def test_append_to_gray_code_sequence(self) -> None:
        """Gray code sequence is correct."""
        # Build Gray code by appending bits: 1, 0, 1
        gc = 0
        gc = _append_to_gray_code(gc, True)  # 1
        assert gc == 1
        gc = _append_to_gray_code(gc, False)  # 10 in Gray = 3 in binary? Let's verify
        # (1 << 1) + (0 ^ (1 & 1)) = 2 + (0 ^ 1) = 2 + 1 = 3
        assert gc == 3
        gc = _append_to_gray_code(gc, True)  # 101 in some encoding
        # (3 << 1) + (1 ^ (3 & 1)) = 6 + (1 ^ 1) = 6 + 0 = 6
        assert gc == 6

    def test_simhash_partition_index_gray(self) -> None:
        """Partition index computed correctly via Gray code."""
        # All positive bits
        sketch = np.array([1.0, 1.0, 1.0])
        idx = _simhash_partition_index_gray(sketch)
        # Bits: 1, 1, 1
        # gc = 0 -> append 1 -> 1
        # gc = 1 -> append 1 -> (1<<1) + (1^1) = 2 + 0 = 2
        # gc = 2 -> append 1 -> (2<<1) + (1^0) = 4 + 1 = 5
        assert idx == 5

        # All negative bits
        sketch = np.array([-1.0, -1.0, -1.0])
        idx = _simhash_partition_index_gray(sketch)
        assert idx == 0


class TestMuveraConfig:
    """Tests for MuveraConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration values (paper's recommended config)."""
        config = MuveraConfig()

        assert config.num_repetitions == 40  # Paper uses 40
        assert config.num_simhash_projections == 6  # 64 partitions
        assert config.projection_dim is None  # Identity by default
        assert config.final_projection_dim == 10240  # Count Sketch compression
        assert config.seed == 42

    def test_num_partitions(self) -> None:
        """Number of partitions is 2^num_simhash_projections."""
        config = MuveraConfig(num_simhash_projections=6)
        assert config.num_partitions == 64

        config = MuveraConfig(num_simhash_projections=4)
        assert config.num_partitions == 16

    def test_fde_dim_with_projection(self) -> None:
        """FDE dimension with projection_dim set (no final projection)."""
        config = MuveraConfig(
            num_repetitions=10,
            num_simhash_projections=6,  # 64 partitions
            projection_dim=4,
            final_projection_dim=None,  # No Count Sketch
        )
        # 10 * 64 * 4 = 2560
        assert config.fde_dim(token_dim=128) == 2560

    def test_fde_dim_identity_projection(self) -> None:
        """FDE dimension with identity + Count Sketch (paper's config)."""
        config = MuveraConfig(
            num_repetitions=40,
            num_simhash_projections=6,  # 64 partitions
            projection_dim=None,  # Identity = use token_dim
            final_projection_dim=10240,  # Count Sketch
        )
        # Intermediate: 40 * 64 * 128 = 327680, but final = 10240
        assert config.fde_dim(token_dim=128) == 10240
        assert config.intermediate_dim(token_dim=128) == 327680


class TestMuveraPostprocessor:
    """Tests for MuveraPostprocessor."""

    @pytest.fixture
    def config_with_projection(self) -> MuveraConfig:
        """Config with AMS projection for smaller output (no Count Sketch)."""
        return MuveraConfig(
            num_repetitions=10,
            num_simhash_projections=6,
            projection_dim=4,  # Small projection for testing
            final_projection_dim=None,  # No Count Sketch for fast tests
            seed=42,
        )

    @pytest.fixture
    def postprocessor(self, config_with_projection: MuveraConfig) -> MuveraPostprocessor:
        """Create postprocessor with token_dim=128 and projection."""
        return MuveraPostprocessor(token_dim=128, config=config_with_projection)

    @pytest.fixture
    def identity_postprocessor(self) -> MuveraPostprocessor:
        """Create postprocessor with identity projection (no Count Sketch)."""
        config = MuveraConfig(
            num_repetitions=2,
            num_simhash_projections=4,  # 16 partitions
            projection_dim=None,  # Identity
            final_projection_dim=None,  # No Count Sketch for fast tests
            seed=42,
        )
        return MuveraPostprocessor(token_dim=128, config=config)

    def test_init_target_dim_with_projection(self, postprocessor: MuveraPostprocessor) -> None:
        """Target dimension is correctly computed with projection."""
        # 10 * 64 * 4 = 2560
        assert postprocessor.target_dim == 2560

    def test_init_target_dim_identity(self, identity_postprocessor: MuveraPostprocessor) -> None:
        """Target dimension with identity projection uses token_dim."""
        # 2 * 16 * 128 = 4096
        assert identity_postprocessor.target_dim == 4096

    def test_init_uses_identity_when_no_projection(self, identity_postprocessor: MuveraPostprocessor) -> None:
        """Identity mode is detected correctly."""
        assert identity_postprocessor._use_identity is True
        assert identity_postprocessor._proj_dim == 128

    def test_init_uses_projection_when_set(self, postprocessor: MuveraPostprocessor) -> None:
        """Projection mode is detected correctly."""
        assert postprocessor._use_identity is False
        assert postprocessor._proj_dim == 4

    def test_source_target_fields(self, postprocessor: MuveraPostprocessor) -> None:
        """Source and target fields are correctly set."""
        assert postprocessor.source_field == "multivector"
        assert postprocessor.target_field == "dense"

    def test_transform_basic(self, postprocessor: MuveraPostprocessor) -> None:
        """Transform converts multivector to dense."""
        # Create multivector output
        multivector = [
            np.random.randn(10, 128).astype(np.float32),  # 10 tokens
            np.random.randn(15, 128).astype(np.float32),  # 15 tokens
        ]
        output = EncodeOutput(multivector=multivector, batch_size=2)

        # Transform
        postprocessor.transform(output, is_query=False)

        # Check dense is populated
        assert output.dense is not None
        assert output.dense.shape == (2, 2560)
        assert output.dense.dtype == np.float32
        assert output.dense_dim == 2560

    def test_transform_preserves_multivector(self, postprocessor: MuveraPostprocessor) -> None:
        """Transform preserves original multivector."""
        original = np.random.randn(10, 128).astype(np.float32)
        multivector = [original.copy()]
        output = EncodeOutput(multivector=multivector, batch_size=1)

        postprocessor.transform(output, is_query=False)

        # Multivector should be unchanged
        np.testing.assert_array_equal(output.multivector[0], original)

    def test_transform_query_vs_document(self, postprocessor: MuveraPostprocessor) -> None:
        """Query and document produce different results (sum vs average)."""
        np.random.seed(123)
        multivector = [np.random.randn(10, 128).astype(np.float32)]

        # Query (sum aggregation)
        output_query = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor.transform(output_query, is_query=True)

        # Document (average aggregation)
        output_doc = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor.transform(output_doc, is_query=False)

        # Results should differ
        assert not np.allclose(output_query.dense, output_doc.dense)

    def test_transform_empty_multivector(self, postprocessor: MuveraPostprocessor) -> None:
        """Handle empty token sequence (0 tokens)."""
        multivector = [np.zeros((0, 128), dtype=np.float32)]
        output = EncodeOutput(multivector=multivector, batch_size=1)

        postprocessor.transform(output, is_query=False)

        # Should produce zero vector
        assert output.dense is not None
        assert output.dense.shape == (1, 2560)
        np.testing.assert_array_equal(output.dense[0], np.zeros(2560))

    def test_transform_single_token(self, postprocessor: MuveraPostprocessor) -> None:
        """Handle single token."""
        multivector = [np.random.randn(1, 128).astype(np.float32)]
        output = EncodeOutput(multivector=multivector, batch_size=1)

        postprocessor.transform(output, is_query=False)

        assert output.dense is not None
        assert output.dense.shape == (1, 2560)
        # Should not be all zeros (single token goes to one partition per rep)
        assert not np.allclose(output.dense, 0)

    def test_transform_requires_multivector(self, postprocessor: MuveraPostprocessor) -> None:
        """Raises error if multivector is None."""
        output = EncodeOutput(dense=np.random.randn(2, 128).astype(np.float32), batch_size=2)

        with pytest.raises(ValueError, match="requires multivector"):
            postprocessor.transform(output)

    def test_transform_deterministic(self, postprocessor: MuveraPostprocessor) -> None:
        """Same input produces same output (deterministic)."""
        np.random.seed(456)
        multivector = [np.random.randn(10, 128).astype(np.float32)]

        output1 = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor.transform(output1, is_query=False)

        output2 = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor.transform(output2, is_query=False)

        np.testing.assert_array_equal(output1.dense, output2.dense)

    def test_transform_different_seeds(self) -> None:
        """Different seeds produce different results."""
        config1 = MuveraConfig(seed=42, projection_dim=4, final_projection_dim=None)
        config2 = MuveraConfig(seed=123, projection_dim=4, final_projection_dim=None)

        postprocessor1 = MuveraPostprocessor(token_dim=128, config=config1)
        postprocessor2 = MuveraPostprocessor(token_dim=128, config=config2)

        np.random.seed(789)
        multivector = [np.random.randn(10, 128).astype(np.float32)]

        output1 = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor1.transform(output1, is_query=False)

        output2 = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        postprocessor2.transform(output2, is_query=False)

        assert not np.allclose(output1.dense, output2.dense)

    def test_sketches_to_gray_partitions(self, postprocessor: MuveraPostprocessor) -> None:
        """Sketch values correctly map to partition indices via Gray code."""
        # Test matches reference _simhash_partition_index_gray behavior
        # All positive bits: Gray code sequence 1,1,1,1,1,1
        sketches = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        indices = postprocessor._sketches_to_gray_partitions(sketches)
        # Verify it's consistent (exact value depends on Gray code impl)
        assert 0 <= indices[0] < 64

        # All negative bits: should be 0
        sketches = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
        indices = postprocessor._sketches_to_gray_partitions(sketches)
        assert indices[0] == 0

    def test_aggregate_partitions_sum(self, postprocessor: MuveraPostprocessor) -> None:
        """Sum aggregation for queries."""
        projected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        partition_indices = np.array([0, 0])  # Both in partition 0

        result = postprocessor._aggregate_partitions_vectorized(projected, partition_indices, 64, is_query=True)

        # Sum: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]  # expected values
        np.testing.assert_array_almost_equal(result[:4], [6.0, 8.0, 10.0, 12.0])

    def test_aggregate_partitions_average(self, postprocessor: MuveraPostprocessor) -> None:
        """Average aggregation for documents."""
        projected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        partition_indices = np.array([0, 0])  # Both in partition 0

        result = postprocessor._aggregate_partitions_vectorized(projected, partition_indices, 64, is_query=False)

        # Average: [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
        np.testing.assert_array_almost_equal(result[:4], [3.0, 4.0, 5.0, 6.0])

    def test_aggregate_different_partitions(self, postprocessor: MuveraPostprocessor) -> None:
        """Vectors in different partitions aggregate separately."""
        projected = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        partition_indices = np.array([0, 1])  # Different partitions

        result = postprocessor._aggregate_partitions_vectorized(projected, partition_indices, 64, is_query=False)
        result_2d = result.reshape(64, 4)

        np.testing.assert_array_almost_equal(result_2d[0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result_2d[1], [5.0, 6.0, 7.0, 8.0])


class TestMuveraOptimizedEquivalence:
    """Bit-exact regression coverage for cached and streaming MUVERA."""

    @pytest.mark.parametrize("use_count_sketch", [False, True])
    @pytest.mark.parametrize("is_query", [False, True])
    @pytest.mark.parametrize("center_tokens", [False, True])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_matches_legacy_serial_for_varied_lengths_and_order(
        self,
        use_count_sketch: bool,
        is_query: bool,
        center_tokens: bool,
        normalize: bool,
    ) -> None:
        """Optimization preserves every float32 output bit and batch row."""
        token_dim = 7
        config = MuveraConfig(
            num_repetitions=4,
            num_simhash_projections=3,
            projection_dim=None if use_count_sketch else 3,
            final_projection_dim=17 if use_count_sketch else None,
            seed=29,
            normalize=normalize,
            center_tokens=center_tokens,
        )
        rng = np.random.default_rng(8675309)
        multivectors = [rng.standard_normal((length, token_dim)).astype(np.float32) for length in (7, 1, 0, 4)]

        for ordered in (multivectors, list(reversed(multivectors))):
            expected = _legacy_muvera_batch(ordered, token_dim, config, is_query=is_query)
            output = EncodeOutput(
                multivector=[multivector.copy() for multivector in ordered],
                batch_size=len(ordered),
            )

            MuveraPostprocessor(token_dim=token_dim, config=config).transform(output, is_query=is_query)

            assert output.dense is not None
            np.testing.assert_array_equal(output.dense, expected)
            assert output.dense_dim == expected.shape[1]

    def test_random_cache_is_lazy_for_empty_inputs(self) -> None:
        """Construction and empty transforms do not allocate random caches."""
        token_dim = 9
        config = MuveraConfig(
            num_repetitions=5,
            num_simhash_projections=3,
            projection_dim=None,
            final_projection_dim=11,
            seed=101,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        assert postprocessor._random_cache is None

        empty_batch = EncodeOutput(multivector=[], batch_size=0)
        postprocessor.transform(empty_batch, is_query=False)
        assert postprocessor._random_cache is None

        empty_item = EncodeOutput(
            multivector=[np.empty((0, token_dim), dtype=np.float32)],
            batch_size=1,
        )
        postprocessor.transform(empty_item, is_query=False)
        assert postprocessor._random_cache is None

        nonempty = EncodeOutput(
            multivector=[np.ones((1, token_dim), dtype=np.float32)],
            batch_size=1,
        )
        postprocessor.transform(nonempty, is_query=False)
        assert postprocessor._random_cache is not None

    def test_cached_random_structures_match_legacy_generation(self) -> None:
        """Lazily cached random values retain the legacy draws exactly."""
        token_dim = 9
        count_config = MuveraConfig(
            num_repetitions=5,
            num_simhash_projections=3,
            projection_dim=None,
            final_projection_dim=11,
            seed=101,
        )
        count_postprocessor = MuveraPostprocessor(token_dim=token_dim, config=count_config)
        assert count_postprocessor._random_cache is None
        count_postprocessor._compute_fde_single(
            np.ones((1, token_dim), dtype=np.float32),
            is_query=False,
        )
        count_cache = count_postprocessor._random_cache
        assert count_cache is not None

        for repetition, cached in enumerate(count_cache.simhash_matrices):
            rng = np.random.default_rng(count_config.seed + repetition)
            expected = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(token_dim, count_config.num_simhash_projections),
            ).astype(np.float32)
            np.testing.assert_array_equal(cached, expected)
            assert not cached.flags.writeable
        assert count_cache.ams_projection_matrices == ()

        rng = np.random.default_rng(count_config.seed)
        expected_indices = rng.integers(
            0,
            count_postprocessor._final_dim,
            size=count_postprocessor._intermediate_dim,
        )
        expected_signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=count_postprocessor._intermediate_dim,
        )
        assert count_cache.count_sketch_indices is not None
        assert count_cache.count_sketch_signs is not None
        np.testing.assert_array_equal(count_cache.count_sketch_indices, expected_indices)
        np.testing.assert_array_equal(count_cache.count_sketch_signs, expected_signs)
        assert count_cache.count_sketch_indices.dtype == np.uint8
        assert count_cache.count_sketch_signs.dtype == np.int8
        assert not count_cache.count_sketch_indices.flags.writeable
        assert not count_cache.count_sketch_signs.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            count_cache.count_sketch_indices[0] = 0
        with pytest.raises(FrozenInstanceError):
            count_cache.count_sketch_indices = None  # type: ignore[misc]

        ams_projection_dim = 4
        ams_config = MuveraConfig(
            num_repetitions=5,
            num_simhash_projections=3,
            projection_dim=ams_projection_dim,
            final_projection_dim=None,
            seed=101,
        )
        ams_postprocessor = MuveraPostprocessor(token_dim=token_dim, config=ams_config)
        ams_postprocessor._compute_fde_single(
            np.ones((1, token_dim), dtype=np.float32),
            is_query=False,
        )
        ams_cache = ams_postprocessor._random_cache
        assert ams_cache is not None
        for repetition, cached in enumerate(ams_cache.ams_projection_matrices):
            rng = np.random.default_rng(ams_config.seed + repetition)
            expected = np.zeros((token_dim, ams_projection_dim), dtype=np.float32)
            indices = rng.integers(0, ams_projection_dim, size=token_dim)
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=token_dim)
            expected[np.arange(token_dim), indices] = signs
            np.testing.assert_array_equal(cached, expected)
            assert not cached.flags.writeable
        assert ams_cache.count_sketch_indices is None
        assert ams_cache.count_sketch_signs is None

    def test_count_sketch_cache_uses_uint16_for_production_final_dim(self) -> None:
        """The 10,240-dimensional production sketch uses exact uint16 indices."""
        token_dim = 5
        config = MuveraConfig(
            num_repetitions=40,
            num_simhash_projections=6,
            projection_dim=None,
            final_projection_dim=10_240,
            seed=47,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        token_embeddings = np.arange(15, dtype=np.float32).reshape(3, token_dim)

        actual = postprocessor._compute_fde_single(token_embeddings, is_query=False)
        expected = _legacy_muvera_single(token_embeddings, config, is_query=False)

        np.testing.assert_array_equal(actual, expected)
        cache = postprocessor._random_cache
        assert cache is not None
        assert cache.count_sketch_indices is not None
        assert cache.count_sketch_signs is not None
        assert cache.count_sketch_indices.dtype == np.uint16
        assert cache.count_sketch_signs.dtype == np.int8

    def test_streaming_count_sketch_matches_materialized_legacy_order(self) -> None:
        """Chunked streaming retains legacy global ``np.add.at`` order."""
        token_dim = 5
        config = MuveraConfig(
            num_repetitions=9,
            num_simhash_projections=3,
            projection_dim=None,
            final_projection_dim=3,
            seed=73,
            center_tokens=True,
        )
        tokens = np.random.default_rng(144).standard_normal((11, token_dim)).astype(np.float32)
        expected = _legacy_muvera_single(tokens, config, is_query=False)

        actual = MuveraPostprocessor(token_dim=token_dim, config=config)._compute_fde_single(
            tokens,
            is_query=False,
        )

        np.testing.assert_array_equal(actual, expected)

    def test_single_item_transform_computes_repetitions_in_legacy_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The serial path consumes repetitions in exact legacy order."""
        token_dim = 6
        config = MuveraConfig(
            num_repetitions=7,
            num_simhash_projections=2,
            projection_dim=3,
            final_projection_dim=None,
            seed=9,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        original_compute = postprocessor._compute_repetition
        computed_repetitions: list[int] = []

        def recording_compute(
            token_embeddings: np.ndarray,
            rep_num: int,
            *,
            is_query: bool,
        ) -> np.ndarray:
            computed_repetitions.append(rep_num)
            return original_compute(token_embeddings, rep_num, is_query=is_query)

        monkeypatch.setattr(postprocessor, "_compute_repetition", recording_compute)
        tokens = np.random.default_rng(99).standard_normal((5, token_dim)).astype(np.float32)
        output = EncodeOutput(multivector=[tokens], batch_size=1)
        postprocessor.transform(output, is_query=True)

        assert computed_repetitions == list(range(config.num_repetitions))
        expected = _legacy_muvera_batch([tokens], token_dim, config, is_query=True)
        np.testing.assert_array_equal(output.dense, expected)

    @pytest.mark.parametrize("use_count_sketch", [False, True])
    def test_concurrent_first_use_builds_and_publishes_one_cache(
        self,
        use_count_sketch: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concurrent first transforms share one fully initialized cache."""
        token_dim = 6
        config = MuveraConfig(
            num_repetitions=5,
            num_simhash_projections=3,
            projection_dim=None if use_count_sketch else 3,
            final_projection_dim=7 if use_count_sketch else None,
            seed=41,
            center_tokens=True,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        tokens = np.random.default_rng(42).standard_normal((9, token_dim)).astype(np.float32)
        expected = _legacy_muvera_single(tokens, config, is_query=False)
        original_build = postprocessor._build_random_cache
        build_entered = Event()
        release_build = Event()
        build_calls = 0

        def blocking_build() -> object:
            nonlocal build_calls
            build_calls += 1
            build_entered.set()
            assert release_build.wait(timeout=5)
            return original_build()

        monkeypatch.setattr(postprocessor, "_build_random_cache", blocking_build)

        with ThreadPoolExecutor(max_workers=8) as callers:
            first = callers.submit(postprocessor._compute_fde_single, tokens, is_query=False)
            assert build_entered.wait(timeout=5)
            remaining = [callers.submit(postprocessor._compute_fde_single, tokens, is_query=False) for _ in range(7)]
            release_build.set()
            results = [first.result(), *(future.result() for future in remaining)]

        assert build_calls == 1
        published_cache = postprocessor._random_cache
        assert published_cache is not None
        for result in results:
            np.testing.assert_array_equal(result, expected)

    def test_failed_cache_build_is_not_published_and_can_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed local cache build leaves no partial state or dense output."""
        token_dim = 5
        config = MuveraConfig(
            num_repetitions=6,
            num_simhash_projections=2,
            projection_dim=None,
            final_projection_dim=7,
            seed=12,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        original_build = postprocessor._build_random_cache
        build_calls = 0

        def fail_once() -> object:
            nonlocal build_calls
            build_calls += 1
            cache = original_build()
            assert postprocessor._random_cache is None
            if build_calls == 1:
                raise RuntimeError("cache build failed")
            return cache

        monkeypatch.setattr(postprocessor, "_build_random_cache", fail_once)
        tokens = np.random.default_rng(19).standard_normal((4, token_dim)).astype(np.float32)
        original_dense = np.arange(3, dtype=np.float32).reshape(1, 3)
        output = EncodeOutput(
            dense=original_dense,
            dense_dim=3,
            multivector=[tokens],
            batch_size=1,
        )

        with pytest.raises(RuntimeError, match="cache build failed"):
            postprocessor.transform(output, is_query=False)

        assert postprocessor._random_cache is None
        assert output.dense is original_dense
        assert output.dense_dim == 3

        postprocessor.transform(output, is_query=False)

        assert build_calls == 2
        assert postprocessor._random_cache is not None
        expected = _legacy_muvera_batch([tokens], token_dim, config, is_query=False)
        np.testing.assert_array_equal(output.dense, expected)

    def test_repetition_failure_does_not_publish_partial_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A repetition exception leaves pre-existing dense output untouched."""
        token_dim = 5
        config = MuveraConfig(
            num_repetitions=6,
            num_simhash_projections=2,
            projection_dim=None,
            final_projection_dim=7,
            seed=12,
        )
        postprocessor = MuveraPostprocessor(token_dim=token_dim, config=config)
        original_compute = postprocessor._compute_repetition

        def failing_compute(
            token_embeddings: np.ndarray,
            rep_num: int,
            *,
            is_query: bool,
        ) -> np.ndarray:
            if token_embeddings[0, 0] == 99.0 and rep_num == 1:
                raise RuntimeError("repetition failed")
            return original_compute(token_embeddings, rep_num, is_query=is_query)

        monkeypatch.setattr(postprocessor, "_compute_repetition", failing_compute)
        good = np.ones((4, token_dim), dtype=np.float32)
        bad = np.ones((4, token_dim), dtype=np.float32)
        bad[0, 0] = 99.0
        original_dense = np.arange(6, dtype=np.float32).reshape(2, 3)
        output = EncodeOutput(
            dense=original_dense,
            dense_dim=3,
            multivector=[good, bad],
            batch_size=2,
        )

        with pytest.raises(RuntimeError, match="repetition failed"):
            postprocessor.transform(output, is_query=False)

        assert output.dense is original_dense
        np.testing.assert_array_equal(output.dense, np.arange(6, dtype=np.float32).reshape(2, 3))
        assert output.dense_dim == 3


class TestMuveraSameDimCountSketchGuard:
    """Tests for the #1493 same-or-larger Count-Sketch guard.

    A Count-Sketch is only ever a reduction; when ``final_projection_dim >=
    intermediate_dim`` it is destructive (collapsed @muvera nDCG@10 to ~0.05).
    The guard skips it and returns the unprojected intermediate FDE.
    """

    def test_guard_skips_same_dim_count_sketch(self) -> None:
        """Same-dim count-sketch is skipped; target_dim == intermediate_dim."""
        # The old buggy answerai config: reps=20, proj=8, token_dim=96 ->
        # intermediate = 20 * 64 * 8 = 10240 == final_projection_dim.
        config = MuveraConfig(
            num_repetitions=20,
            num_simhash_projections=6,
            projection_dim=8,
            final_projection_dim=10240,
            seed=42,
        )
        pp = MuveraPostprocessor(token_dim=96, config=config)
        assert config.intermediate_dim(token_dim=96) == 10240
        assert pp.target_dim == 10240
        assert pp._final_dim is None  # sketch skipped

        # Equivalence: a no-sketch postprocessor (final=None) must produce the
        # bit-identical FDE, proving Step 6 was skipped for the guarded config.
        no_sketch_config = MuveraConfig(
            num_repetitions=20,
            num_simhash_projections=6,
            projection_dim=8,
            final_projection_dim=None,
            seed=42,
        )
        no_sketch_pp = MuveraPostprocessor(token_dim=96, config=no_sketch_config)
        assert no_sketch_pp.target_dim == 10240

        rng = np.random.default_rng(42)
        multivector = [rng.standard_normal((12, 96)).astype(np.float32)]

        guarded = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        unsketched = EncodeOutput(multivector=[mv.copy() for mv in multivector], batch_size=1)
        pp.transform(guarded, is_query=False)
        no_sketch_pp.transform(unsketched, is_query=False)

        np.testing.assert_array_equal(guarded.dense, unsketched.dense)

    def test_genuine_reduction_still_applies_count_sketch(self) -> None:
        """A real reduction (final < intermediate) keeps the count-sketch."""
        # reps=20, proj=8, token_dim=128 -> intermediate = 20 * 64 * 8 = 10240.
        config = MuveraConfig(
            num_repetitions=20,
            num_simhash_projections=6,
            projection_dim=8,
            final_projection_dim=4096,
            seed=42,
        )
        pp = MuveraPostprocessor(token_dim=128, config=config)
        assert config.intermediate_dim(token_dim=128) == 10240
        assert pp.target_dim == 4096
        assert pp._final_dim == 4096  # sketch applied

        multivector = [np.random.default_rng(7).standard_normal((10, 128)).astype(np.float32)]
        output = EncodeOutput(multivector=multivector, batch_size=1)
        pp.transform(output, is_query=False)
        assert output.dense is not None
        assert output.dense.shape == (1, 4096)

    def test_separability_recovered_when_sketch_skipped(self) -> None:
        """End-to-end sanity check: the guarded FDE ranks a near-duplicate doc
        above an unrelated doc for the query.

        This is a sanity check, not the regression guard. The load-bearing
        proof that the same-dim sketch is skipped is the bit-exact equivalence
        assertion in ``test_guard_skips_same_dim_count_sketch`` (which fails
        pre-fix). The destructive sketch's damage shows up at corpus scale
        (near-tied scores across thousands of docs), not on a single pair.
        """
        config = MuveraConfig(
            num_repetitions=20,
            num_simhash_projections=6,
            projection_dim=8,
            final_projection_dim=10240,  # same-dim -> guarded (skipped)
            seed=42,
        )
        pp = MuveraPostprocessor(token_dim=96, config=config)

        rng = np.random.default_rng(42)
        query_tokens = rng.standard_normal((12, 96)).astype(np.float32)
        dup_tokens = (query_tokens + rng.standard_normal((12, 96)).astype(np.float32) * 0.01).astype(np.float32)
        unrelated_tokens = rng.standard_normal((20, 96)).astype(np.float32)

        q_out = EncodeOutput(multivector=[query_tokens], batch_size=1)
        dup_out = EncodeOutput(multivector=[dup_tokens], batch_size=1)
        unrelated_out = EncodeOutput(multivector=[unrelated_tokens], batch_size=1)
        pp.transform(q_out, is_query=True)
        pp.transform(dup_out, is_query=False)
        pp.transform(unrelated_out, is_query=False)

        dup_score = float(q_out.dense[0] @ dup_out.dense[0])
        unrelated_score = float(q_out.dense[0] @ unrelated_out.dense[0])
        assert dup_score > unrelated_score


class TestMuveraFDEProperties:
    """Tests for mathematical properties of MUVERA FDE."""

    @pytest.fixture
    def postprocessor(self) -> MuveraPostprocessor:
        """Create postprocessor for property tests (no Count Sketch)."""
        config = MuveraConfig(
            num_repetitions=20,
            num_simhash_projections=5,  # 32 partitions
            projection_dim=8,
            final_projection_dim=None,  # No Count Sketch for tests
            seed=42,
        )
        return MuveraPostprocessor(token_dim=128, config=config)

    def test_similar_inputs_similar_outputs(self, postprocessor: MuveraPostprocessor) -> None:
        """Similar multivectors should produce similar FDEs."""
        np.random.seed(100)
        base = np.random.randn(20, 128).astype(np.float32)

        # Similar: small perturbation
        similar = base + np.random.randn(20, 128).astype(np.float32) * 0.1

        # Different: large perturbation
        different = np.random.randn(20, 128).astype(np.float32)

        output_base = EncodeOutput(multivector=[base], batch_size=1)
        output_similar = EncodeOutput(multivector=[similar], batch_size=1)
        output_different = EncodeOutput(multivector=[different], batch_size=1)

        postprocessor.transform(output_base, is_query=False)
        postprocessor.transform(output_similar, is_query=False)
        postprocessor.transform(output_different, is_query=False)

        # Compute cosine similarities
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_similar = cosine_sim(output_base.dense[0], output_similar.dense[0])
        sim_different = cosine_sim(output_base.dense[0], output_different.dense[0])

        # Similar inputs should have higher FDE similarity
        assert sim_similar > sim_different

    def test_fde_inner_product_approximates_maxsim(self, postprocessor: MuveraPostprocessor) -> None:
        """FDE inner product should roughly correlate with MaxSim.

        This is a weak test - MUVERA approximates Chamfer similarity,
        and exact correlation depends on data distribution.
        """
        np.random.seed(200)

        def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
            """Compute MaxSim (Chamfer similarity)."""
            # For each query token, find max similarity to any doc token
            sims = query @ doc.T  # [query_tokens, doc_tokens]
            return float(sims.max(axis=1).sum())

        # Generate several query-doc pairs
        queries = [np.random.randn(10, 128).astype(np.float32) for _ in range(5)]
        docs = [np.random.randn(20, 128).astype(np.float32) for _ in range(5)]

        maxsim_scores = []
        fde_scores = []

        for query, doc in zip(queries, docs, strict=True):
            # Compute MaxSim
            maxsim_scores.append(maxsim(query, doc))

            # Compute FDE inner product
            output_q = EncodeOutput(multivector=[query], batch_size=1)
            output_d = EncodeOutput(multivector=[doc], batch_size=1)
            postprocessor.transform(output_q, is_query=True)
            postprocessor.transform(output_d, is_query=False)
            fde_scores.append(float(output_q.dense[0] @ output_d.dense[0]))

        # Check positive correlation (Spearman rank correlation)
        from scipy.stats import spearmanr

        corr, _ = spearmanr(maxsim_scores, fde_scores)
        # MUVERA should preserve ranking reasonably well
        assert corr > 0.5, f"Expected positive correlation, got {corr}"


class TestMuveraCenterTokens:
    """Tests for the #1528 ``center_tokens`` fix.

    A dominant shared DC component across a multivector's tokens makes SimHash
    bucket them all into the same partitions, so the FDEs of different docs come
    out near-identical (collapsed ranking) even though MaxSim stays healthy.
    Subtracting the per-multivector mean token before partitioning removes the DC
    component and partitions on the discriminative residual.
    """

    def test_default_is_off(self) -> None:
        """center_tokens defaults False — existing configs/floors are unaffected."""
        assert MuveraConfig().center_tokens is False

    def test_flag_changes_fde_for_dc_heavy_multivector(self) -> None:
        """For a DC-dominated multivector the centered FDE differs from uncentered."""
        common = {"num_repetitions": 8, "projection_dim": None, "final_projection_dim": None, "seed": 42}
        pp_off = MuveraPostprocessor(token_dim=16, config=MuveraConfig(**common, center_tokens=False))
        pp_on = MuveraPostprocessor(token_dim=16, config=MuveraConfig(**common, center_tokens=True))

        rng = np.random.default_rng(0)
        dc = rng.standard_normal(16).astype(np.float32)
        dc /= np.linalg.norm(dc)
        mv = (3.0 * dc + 0.3 * rng.standard_normal((10, 16))).astype(np.float32)

        off = EncodeOutput(multivector=[mv.copy()], batch_size=1)
        on = EncodeOutput(multivector=[mv.copy()], batch_size=1)
        pp_off.transform(off, is_query=False)
        pp_on.transform(on, is_query=False)
        assert not np.allclose(off.dense, on.dense)

    def test_centering_separates_dc_dominated_docs(self) -> None:
        """Load-bearing: two docs sharing a dominant DC direction but differing in
        a small residual have near-tied FDEs without centering; centering pulls
        their FDEs apart (lower pairwise cosine = separable ranking).
        """
        common = {
            "num_repetitions": 20,
            "num_simhash_projections": 6,
            "projection_dim": None,
            "final_projection_dim": None,
            "seed": 42,
        }
        pp_off = MuveraPostprocessor(token_dim=32, config=MuveraConfig(**common, center_tokens=False))
        pp_on = MuveraPostprocessor(token_dim=32, config=MuveraConfig(**common, center_tokens=True))

        rng = np.random.default_rng(1)
        dc = rng.standard_normal(32).astype(np.float32)
        dc /= np.linalg.norm(dc)

        def dc_doc(seed: int) -> np.ndarray:
            # A dominant shared DC direction (15x) plus a small per-doc residual,
            # then unit-normalized as the adapter does. The DC dominance is what
            # tips SimHash into collapse (mirrors answerai's ~0.93 DC on real data).
            r = np.random.default_rng(seed)
            tokens = 15.0 * dc + 0.2 * r.standard_normal((12, 32)).astype(np.float32)
            norms = np.linalg.norm(tokens, axis=1, keepdims=True)
            return (tokens / norms).astype(np.float32)

        doc_a, doc_b = dc_doc(11), dc_doc(22)

        def fde_cos(pp: MuveraPostprocessor) -> float:
            oa = EncodeOutput(multivector=[doc_a.copy()], batch_size=1)
            ob = EncodeOutput(multivector=[doc_b.copy()], batch_size=1)
            pp.transform(oa, is_query=False)
            pp.transform(ob, is_query=False)
            a, b = oa.dense[0], ob.dense[0]
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        cos_off = fde_cos(pp_off)
        cos_on = fde_cos(pp_on)
        # Without centering the DC component dominates -> near-tied FDEs.
        assert cos_off > 0.85
        # Centering removes it -> the two docs become clearly separable.
        assert cos_on < cos_off - 0.2
