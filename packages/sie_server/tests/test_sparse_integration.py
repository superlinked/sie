"""Integration tests for sparse model correctness.

These tests validate that SIE's sparse adapter produces outputs that match
sentence-transformers SparseEncoder directly.

Requires: Real model weights (downloaded on first run)
Mark: integration (run with `mise run test -m integration`)
"""

import numpy as np
import pytest
from sentence_transformers import SparseEncoder
from sie_server.adapters.sentence_transformer import SentenceTransformerSparseAdapter
from sie_server.types.inputs import Item


@pytest.mark.integration
class TestSparseAdapterCorrectness:
    """Validate SIE sparse outputs match sentence-transformers."""

    MODEL_NAME = "rasyosef/splade-mini"  # Small, ungated model for testing

    @pytest.fixture(scope="class")
    def reference_encoder(self, device: str) -> SparseEncoder:
        """Load reference SparseEncoder on detected device."""
        return SparseEncoder(self.MODEL_NAME, device=device)

    @pytest.fixture(scope="class")
    def sie_adapter(self, device: str) -> SentenceTransformerSparseAdapter:
        """Load SIE adapter on detected device."""
        adapter = SentenceTransformerSparseAdapter(self.MODEL_NAME)
        adapter.load(device)
        yield adapter
        adapter.unload()

    def test_document_encoding_matches(
        self,
        reference_encoder: SparseEncoder,
        sie_adapter: SentenceTransformerSparseAdapter,
    ) -> None:
        """Document encoding produces identical outputs."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks can learn complex patterns from data.",
        ]

        # Get reference output
        ref_output = reference_encoder.encode_document(
            texts,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
        )

        # Get SIE output
        items = [Item(text=t) for t in texts]
        sie_output = sie_adapter.encode(items, output_types=["sparse"], is_query=False)

        # Compare each item
        for i, text in enumerate(texts):
            # Extract reference sparse values for this item
            ref_indices = ref_output._indices()
            ref_values = ref_output._values()
            ref_mask = ref_indices[0] == i
            ref_cols = ref_indices[1][ref_mask].cpu().numpy()
            ref_vals = ref_values[ref_mask].cpu().numpy()

            # Get SIE values (EncodeOutput.sparse is list[SparseVector])
            assert sie_output.sparse is not None
            sie_cols = sie_output.sparse[i].indices
            sie_vals = sie_output.sparse[i].values

            # Sort both by index for comparison
            ref_order = np.argsort(ref_cols)
            sie_order = np.argsort(sie_cols)

            np.testing.assert_array_equal(
                ref_cols[ref_order],
                sie_cols[sie_order],
                err_msg=f"Indices mismatch for item {i}: {text[:50]}...",
            )
            np.testing.assert_allclose(
                ref_vals[ref_order],
                sie_vals[sie_order],
                rtol=1e-5,
                err_msg=f"Values mismatch for item {i}: {text[:50]}...",
            )

    def test_query_encoding_matches(
        self,
        reference_encoder: SparseEncoder,
        sie_adapter: SentenceTransformerSparseAdapter,
    ) -> None:
        """Query encoding produces identical outputs."""
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
        ]

        # Get reference output
        ref_output = reference_encoder.encode_query(
            queries,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
        )

        # Get SIE output
        items = [Item(text=q) for q in queries]
        sie_output = sie_adapter.encode(items, output_types=["sparse"], is_query=True)

        # Compare each item
        for i, query in enumerate(queries):
            ref_indices = ref_output._indices()
            ref_values = ref_output._values()
            ref_mask = ref_indices[0] == i
            ref_cols = ref_indices[1][ref_mask].cpu().numpy()
            ref_vals = ref_values[ref_mask].cpu().numpy()

            assert sie_output.sparse is not None
            sie_cols = sie_output.sparse[i].indices
            sie_vals = sie_output.sparse[i].values

            ref_order = np.argsort(ref_cols)
            sie_order = np.argsort(sie_cols)

            np.testing.assert_array_equal(
                ref_cols[ref_order],
                sie_cols[sie_order],
                err_msg=f"Query indices mismatch for: {query}",
            )
            np.testing.assert_allclose(
                ref_vals[ref_order],
                sie_vals[sie_order],
                rtol=1e-5,
                err_msg=f"Query values mismatch for: {query}",
            )

    def test_sparse_similarity_matches(
        self,
        reference_encoder: SparseEncoder,
        sie_adapter: SentenceTransformerSparseAdapter,
    ) -> None:
        """Sparse dot-product similarity matches between SIE and reference."""
        query = "What is deep learning?"
        docs = [
            "Deep learning uses neural networks with many layers.",
            "Python is a programming language.",
            "Transformers revolutionized natural language processing.",
        ]

        # Get reference embeddings
        ref_query = reference_encoder.encode_query([query], convert_to_tensor=True, convert_to_sparse_tensor=True)
        ref_docs = reference_encoder.encode_document(docs, convert_to_tensor=True, convert_to_sparse_tensor=True)
        ref_sim = reference_encoder.similarity(ref_query, ref_docs)

        # Get SIE embeddings (encode returns EncodeOutput, sparse is list[SparseVector])
        sie_query_output = sie_adapter.encode([Item(text=query)], output_types=["sparse"], is_query=True)
        sie_docs_output = sie_adapter.encode([Item(text=d) for d in docs], output_types=["sparse"], is_query=False)

        assert sie_query_output.sparse is not None
        assert sie_docs_output.sparse is not None
        sie_query = sie_query_output.sparse[0]

        # Compute similarity manually (sparse dot product)
        sie_sims = []
        for doc_sparse in sie_docs_output.sparse:
            # Find common indices
            common_mask = np.isin(sie_query.indices, doc_sparse.indices)
            if common_mask.any():
                query_indices = sie_query.indices[common_mask]
                query_vals = sie_query.values[common_mask]
                # Get corresponding doc values
                doc_mask = np.isin(doc_sparse.indices, query_indices)
                doc_vals = doc_sparse.values[doc_mask]
                # Sort both by index to align
                q_order = np.argsort(query_indices)
                d_order = np.argsort(doc_sparse.indices[doc_mask])
                sim = np.dot(query_vals[q_order], doc_vals[d_order])
            else:
                sim = 0.0
            sie_sims.append(sim)

        np.testing.assert_allclose(
            ref_sim.cpu().numpy().flatten(),
            np.array(sie_sims),
            rtol=1e-4,
            err_msg="Similarity scores don't match",
        )
