"""Tests for QuantizePostprocessor and quantization quality metrics."""

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor import QuantizePostprocessor


class TestQuantizePostprocessor:
    """Tests for QuantizePostprocessor."""

    @pytest.fixture
    def postprocessor(self) -> QuantizePostprocessor:
        """Create a quantization postprocessor."""
        return QuantizePostprocessor()

    @pytest.fixture
    def dense_output(self) -> EncodeOutput:
        """Create sample dense output."""
        # Random vectors normalized to unit length (typical embedding output)
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(4, 768)).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return EncodeOutput(
            dense=vectors,
            batch_size=4,
        )

    @pytest.fixture
    def multivector_output(self) -> EncodeOutput:
        """Create sample multivector output (ColBERT style)."""
        rng = np.random.default_rng(42)
        # Variable length sequences per item
        multivectors = [rng.normal(size=(seq_len, 128)).astype(np.float32) for seq_len in [12, 8, 15, 10]]
        # Normalize each token
        multivectors = [mv / np.linalg.norm(mv, axis=1, keepdims=True) for mv in multivectors]
        return EncodeOutput(
            multivector=multivectors,
            batch_size=4,
            multivector_token_dim=128,
        )

    # =========================================================================
    # Basic functionality tests
    # =========================================================================

    def test_float32_no_op(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """float32 output_dtype preserves original values."""
        original = dense_output.dense.copy()
        postprocessor.quantize(dense_output, output_dtype="float32")
        np.testing.assert_array_equal(dense_output.dense, original)
        assert dense_output.dense.dtype == np.float32

    def test_float16_dtype(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """float16 converts dtype correctly."""
        postprocessor.quantize(dense_output, output_dtype="float16")
        assert dense_output.dense.dtype == np.float16

    def test_int8_dtype(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """int8 converts to correct dtype and range."""
        postprocessor.quantize(dense_output, output_dtype="int8")
        assert dense_output.dense.dtype == np.int8
        assert dense_output.dense.min() >= -127
        assert dense_output.dense.max() <= 127

    def test_uint8_dtype(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """uint8 converts to correct dtype and range."""
        postprocessor.quantize(dense_output, output_dtype="uint8")
        assert dense_output.dense.dtype == np.uint8
        assert dense_output.dense.min() >= 0
        assert dense_output.dense.max() <= 255

    def test_binary_dtype(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """Binary packs bits correctly."""
        original_shape = dense_output.dense.shape
        postprocessor.quantize(dense_output, output_dtype="binary")
        assert dense_output.dense.dtype == np.uint8
        # Packed size should be dim // 8
        assert dense_output.dense.shape == (original_shape[0], original_shape[1] // 8)

    def test_multivector_int8(self, postprocessor: QuantizePostprocessor, multivector_output: EncodeOutput) -> None:
        """int8 works on multivector outputs."""
        postprocessor.quantize(multivector_output, output_dtype="int8")
        for mv in multivector_output.multivector:
            assert mv.dtype == np.int8
            assert mv.min() >= -127
            assert mv.max() <= 127

    def test_multivector_binary(self, postprocessor: QuantizePostprocessor, multivector_output: EncodeOutput) -> None:
        """Binary works on multivector outputs."""
        original_dims = [(mv.shape[0], mv.shape[1]) for mv in multivector_output.multivector]
        postprocessor.quantize(multivector_output, output_dtype="binary")
        for i, mv in enumerate(multivector_output.multivector):
            assert mv.dtype == np.uint8
            assert mv.shape == (original_dims[i][0], original_dims[i][1] // 8)

    def test_unsupported_dtype_raises(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """Unsupported output_dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported output_dtype"):
            postprocessor.quantize(dense_output, output_dtype="invalid")

    # =========================================================================
    # Quality preservation tests (similarity correlation)
    # =========================================================================

    def test_int8_preserves_similarity_ranking(self, postprocessor: QuantizePostprocessor) -> None:
        """int8 quantization preserves similarity ranking (>99% correlation)."""
        rng = np.random.default_rng(42)

        # Create normalized vectors
        query = rng.normal(size=(1, 768)).astype(np.float32)
        query = query / np.linalg.norm(query)

        docs = rng.normal(size=(100, 768)).astype(np.float32)
        docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

        # Float32 similarities
        float32_sims = (docs @ query.T).flatten()

        # Create outputs and quantize
        doc_output = EncodeOutput(dense=docs.copy(), batch_size=100)
        query_output = EncodeOutput(dense=query.copy(), batch_size=1)

        postprocessor.quantize(doc_output, output_dtype="int8")
        postprocessor.quantize(query_output, output_dtype="int8")

        # int8 similarities (using float for dot product)
        int8_sims = (doc_output.dense.astype(np.float32) @ query_output.dense.T.astype(np.float32)).flatten()

        # Correlation should be very high
        correlation = np.corrcoef(float32_sims, int8_sims)[0, 1]
        assert correlation > 0.99, f"int8 correlation {correlation:.4f} < 0.99"

    def test_uint8_preserves_similarity_ranking(self, postprocessor: QuantizePostprocessor) -> None:
        """uint8 quantization preserves similarity ranking (>99% correlation)."""
        rng = np.random.default_rng(42)

        # Create normalized vectors
        query = rng.normal(size=(1, 768)).astype(np.float32)
        query = query / np.linalg.norm(query)

        docs = rng.normal(size=(100, 768)).astype(np.float32)
        docs = docs / np.linalg.norm(docs, axis=1, keepdims=True)

        # Float32 similarities
        float32_sims = (docs @ query.T).flatten()

        # Create outputs and quantize
        doc_output = EncodeOutput(dense=docs.copy(), batch_size=100)
        query_output = EncodeOutput(dense=query.copy(), batch_size=1)

        postprocessor.quantize(doc_output, output_dtype="uint8")
        postprocessor.quantize(query_output, output_dtype="uint8")

        # uint8 similarities (center around 128 for proper correlation)
        docs_centered = doc_output.dense.astype(np.float32) - 128
        query_centered = query_output.dense.astype(np.float32) - 128
        uint8_sims = (docs_centered @ query_centered.T).flatten()

        # Correlation should be high
        correlation = np.corrcoef(float32_sims, uint8_sims)[0, 1]
        assert correlation > 0.98, f"uint8 correlation {correlation:.4f} < 0.98"

    def test_binary_produces_packed_output(self, postprocessor: QuantizePostprocessor) -> None:
        """Binary quantization produces correctly packed bits."""
        rng = np.random.default_rng(42)

        # Create vectors with known sign pattern (use 512 dims for clean packing)
        docs = rng.normal(size=(10, 512)).astype(np.float32)

        # Create output and quantize
        doc_output = EncodeOutput(dense=docs.copy(), batch_size=10)
        postprocessor.quantize(doc_output, output_dtype="binary")

        # Should be packed to 64 bytes per vector (512 bits / 8)
        assert doc_output.dense.shape == (10, 64)
        assert doc_output.dense.dtype == np.uint8

        # Verify packing: unpack and compare sign
        for i in range(10):
            unpacked = np.unpackbits(doc_output.dense[i])
            expected_signs = (docs[i] > 0).astype(np.uint8)
            np.testing.assert_array_equal(unpacked, expected_signs)

    # =========================================================================
    # Size reduction tests
    # =========================================================================

    def test_int8_size_reduction(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """int8 achieves 4x size reduction."""
        original_size = dense_output.dense.nbytes
        postprocessor.quantize(dense_output, output_dtype="int8")
        quantized_size = dense_output.dense.nbytes
        assert quantized_size == original_size // 4

    def test_float16_size_reduction(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """float16 achieves 2x size reduction."""
        original_size = dense_output.dense.nbytes
        postprocessor.quantize(dense_output, output_dtype="float16")
        quantized_size = dense_output.dense.nbytes
        assert quantized_size == original_size // 2

    def test_binary_size_reduction(self, postprocessor: QuantizePostprocessor, dense_output: EncodeOutput) -> None:
        """Binary achieves 32x size reduction."""
        original_size = dense_output.dense.nbytes
        postprocessor.quantize(dense_output, output_dtype="binary")
        quantized_size = dense_output.dense.nbytes
        assert quantized_size == original_size // 32


class TestQuantizationEdgeCases:
    """Edge case tests for quantization."""

    @pytest.fixture
    def postprocessor(self) -> QuantizePostprocessor:
        return QuantizePostprocessor()

    def test_zero_vector_int8(self, postprocessor: QuantizePostprocessor) -> None:
        """Zero vectors quantize without division by zero."""
        output = EncodeOutput(
            dense=np.zeros((2, 768), dtype=np.float32),
            batch_size=2,
        )
        postprocessor.quantize(output, output_dtype="int8")
        assert output.dense.dtype == np.int8
        np.testing.assert_array_equal(output.dense, 0)

    def test_constant_vector_uint8(self, postprocessor: QuantizePostprocessor) -> None:
        """Constant vectors quantize without error (range=0 case handled)."""
        output = EncodeOutput(
            dense=np.full((2, 768), 0.5, dtype=np.float32),
            batch_size=2,
        )
        postprocessor.quantize(output, output_dtype="uint8")
        assert output.dense.dtype == np.uint8
        # When all values are same (range=0), implementation uses range=1
        # so (0.5 - 0.5) / 1 * 255 = 0
        np.testing.assert_array_equal(output.dense, 0)

    def test_single_batch_vector(self, postprocessor: QuantizePostprocessor) -> None:
        """Single vector (as batch of 1) quantizes correctly."""
        rng = np.random.default_rng(42)
        vec = rng.normal(size=(1, 768)).astype(np.float32)
        output = EncodeOutput(
            dense=vec,
            batch_size=1,
        )
        postprocessor.quantize(output, output_dtype="int8")
        assert output.dense.dtype == np.int8
        assert output.dense.shape == (1, 768)
