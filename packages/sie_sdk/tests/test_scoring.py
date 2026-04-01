"""Tests for SDK scoring module (maxsim / maxsim_batch utilities)."""

import numpy as np
from sie_sdk.scoring import maxsim, maxsim_batch

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestMaxSim:
    """Tests for maxsim() function."""

    def test_maxsim_basic(self) -> None:
        """MaxSim returns correct number of scores."""
        query = _RNG.standard_normal((5, 128)).astype(np.float32)  # 5 tokens
        docs = [
            _RNG.standard_normal((10, 128)).astype(np.float32),  # doc 1: 10 tokens
            _RNG.standard_normal((8, 128)).astype(np.float32),  # doc 2: 8 tokens
        ]

        scores = maxsim(query, docs)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_maxsim_single_document(self) -> None:
        """MaxSim works with a single document as 2D array."""
        query = _RNG.standard_normal((3, 64)).astype(np.float32)
        doc = _RNG.standard_normal((5, 64)).astype(np.float32)

        scores = maxsim(query, doc)

        assert len(scores) == 1

    def test_maxsim_identical_vectors(self) -> None:
        """MaxSim of identical normalized vectors equals num_query_tokens."""
        # Create normalized vectors
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        scores = maxsim(query, [doc])

        # Each query token has max sim of 1.0 with matching doc token, so total is 2.0
        assert abs(scores[0] - 2.0) < 1e-5

    def test_maxsim_orthogonal_vectors(self) -> None:
        """MaxSim of orthogonal vectors is lower."""
        query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 1: identical to query
        doc1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 2: rotated 45 degrees
        doc2 = np.array([[0.707, 0.707], [-0.707, 0.707]], dtype=np.float32)

        scores = maxsim(query, [doc1, doc2])

        # Doc 1 should have higher score (perfect match)
        assert scores[0] > scores[1]

    def test_maxsim_ranking_order(self) -> None:
        """MaxSim correctly ranks documents by similarity."""
        # Single-token query pointing in x direction
        query = np.array([[1.0, 0.0]], dtype=np.float32)

        # Docs with decreasing similarity to query
        doc_high = np.array([[1.0, 0.0]], dtype=np.float32)  # perfect match
        doc_mid = np.array([[0.707, 0.707]], dtype=np.float32)  # 45 degree
        doc_low = np.array([[0.0, 1.0]], dtype=np.float32)  # orthogonal

        scores = maxsim(query, [doc_high, doc_mid, doc_low])

        assert scores[0] > scores[1] > scores[2]

    def test_maxsim_with_variable_doc_lengths(self) -> None:
        """MaxSim handles documents with different token counts."""
        query = _RNG.standard_normal((4, 32)).astype(np.float32)
        docs = [
            _RNG.standard_normal((1, 32)).astype(np.float32),  # 1 token
            _RNG.standard_normal((10, 32)).astype(np.float32),  # 10 tokens
            _RNG.standard_normal((100, 32)).astype(np.float32),  # 100 tokens
        ]

        scores = maxsim(query, docs)

        assert len(scores) == 3
        # All scores should be finite
        assert all(np.isfinite(s) for s in scores)


class TestMaxSimBatch:
    """Tests for maxsim_batch() function."""

    def test_batch_shape(self) -> None:
        """Batch maxsim returns correct shape."""
        queries = [
            _RNG.standard_normal((3, 64)).astype(np.float32),
            _RNG.standard_normal((5, 64)).astype(np.float32),
        ]
        docs = [
            _RNG.standard_normal((4, 64)).astype(np.float32),
            _RNG.standard_normal((6, 64)).astype(np.float32),
            _RNG.standard_normal((8, 64)).astype(np.float32),
        ]

        scores = maxsim_batch(queries, docs)

        assert scores.shape == (2, 3)  # 2 queries x 3 docs

    def test_batch_matches_individual(self) -> None:
        """Batch results match individual maxsim calls."""
        queries = [
            _RNG.standard_normal((3, 32)).astype(np.float32),
            _RNG.standard_normal((4, 32)).astype(np.float32),
        ]
        docs = [
            _RNG.standard_normal((5, 32)).astype(np.float32),
            _RNG.standard_normal((6, 32)).astype(np.float32),
        ]

        batch_scores = maxsim_batch(queries, docs)

        # Compare with individual calls
        for i, query in enumerate(queries):
            individual_scores = maxsim(query, docs)
            for j, score in enumerate(individual_scores):
                assert abs(batch_scores[i, j] - score) < 1e-5
