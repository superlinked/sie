"""Tests for postprocessor protocol and implementations."""

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import (
    MuveraConfig,
    MuveraPostprocessor,
    _append_to_gray_code,
    _simhash_partition_index_gray,
)


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
