"""Integration tests for sie-dspy.

These tests require a running SIE server and serve as runnable examples
of DSPy workflows using SIE components.

Run with: pytest -m integration integrations/sie_dspy/tests/

Prerequisites:
    mise run serve -d cpu -p 8080

DSPy Use Cases Demonstrated:
- RAG pipelines with optimizable components
- Program synthesis and auto-prompting
- Few-shot learning with semantic example selection
- Research and academic text analysis
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this module if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    """Get SIE server URL from environment or default."""
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


class TestRAGPipeline:
    """Integration tests demonstrating RAG pipeline use case.

    DSPy excels at building optimizable RAG pipelines where:
    1. Retriever fetches candidate passages
    2. Reranker improves precision
    3. Generator produces the answer
    4. The entire pipeline can be optimized end-to-end
    """

    def test_embedder_with_faiss_retriever(self, sie_url: str) -> None:
        """Example: Using SIEEmbedder with DSPy's built-in FAISS retriever."""
        import dspy
        from sie_dspy import SIEEmbedder

        embedder = SIEEmbedder(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        # Sample corpus about ML concepts
        corpus = [
            "Machine learning is a subset of AI that learns from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing analyzes human language.",
            "Computer vision enables machines to interpret images.",
            "Reinforcement learning trains agents through rewards.",
        ]

        retriever = dspy.retrievers.Embeddings(
            corpus=corpus,
            embedder=embedder,
            k=2,
        )

        result = retriever("What is deep learning?")

        assert hasattr(result, "passages")
        assert len(result.passages) == 2

    def test_reranker_improves_retrieval(self, sie_url: str) -> None:
        """Example: Reranking retrieval results for higher precision."""
        from sie_dspy import SIEReranker

        reranker = SIEReranker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        # Simulated retrieval results (mixed relevance)
        retrieved_passages = [
            "The stock market showed volatility today.",
            "Neural networks learn through backpropagation algorithms.",
            "Weather patterns are changing due to climate factors.",
            "Deep learning models require large amounts of training data.",
            "Machine learning enables pattern recognition from data.",
        ]

        result = reranker(
            query="How do neural networks learn?",
            passages=retrieved_passages,
            k=2,
        )

        assert len(result.passages) == 2
        assert len(result.scores) == 2
        # Top results should be about ML/neural networks


class TestProgramOptimization:
    """Integration tests demonstrating DSPy program optimization.

    DSPy's key feature is optimizing LM programs. These tests show
    how SIE components integrate with optimization workflows.
    """

    def test_embedder_for_knn_fewshot(self, sie_url: str) -> None:
        """Example: Using embedder for semantic example selection."""
        from sie_dspy import SIEEmbedder

        embedder = SIEEmbedder(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        # Embed training examples for KNN-based few-shot
        examples = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War II end?",
        ]

        embeddings = embedder(examples)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0  # Has embedding dimensions


class TestResearchWorkflows:
    """Integration tests demonstrating research/academic use cases.

    DSPy is popular in research for:
    1. Building reproducible NLP pipelines
    2. Extracting structured information from papers
    3. Constructing knowledge graphs
    """

    def test_entity_extraction_for_knowledge_graph(self, sie_url: str) -> None:
        """Example: Extracting entities for knowledge graph construction."""
        from sie_dspy import SIEExtractor

        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["person", "organization", "research_topic"],
        )

        paper_abstract = """
        Dr. Yoshua Bengio at Mila Research Institute presented new findings
        on attention mechanisms in transformer architectures. The work builds
        on previous research by Google Brain and OpenAI on large language models.
        """

        result = extractor(text=paper_abstract)

        assert hasattr(result, "entities")
        assert hasattr(result, "entities_dict")

    def test_combined_retrieval_and_extraction(self, sie_url: str) -> None:
        """Example: RAG + extraction for structured research output."""
        import dspy
        from sie_dspy import SIEEmbedder, SIEExtractor, SIEReranker

        # Step 1: Retrieve relevant papers
        embedder = SIEEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        papers = [
            "Attention Is All You Need introduces the transformer architecture.",
            "BERT uses bidirectional training for language understanding.",
            "GPT models use autoregressive pretraining for text generation.",
            "The weather today is particularly nice for outdoor activities.",
        ]

        retriever = dspy.retrievers.Embeddings(
            corpus=papers,
            embedder=embedder,
            k=3,
        )

        retrieved = retriever("transformer architecture papers")
        assert len(retrieved.passages) == 3

        # Step 2: Rerank for precision
        reranker = SIEReranker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        reranked = reranker(
            query="transformer architecture papers",
            passages=retrieved.passages,
            k=2,
        )
        assert len(reranked.passages) == 2

        # Step 3: Extract key information
        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["model_name", "architecture", "technique"],
        )

        for passage in reranked.passages:
            result = extractor(text=passage)
            assert hasattr(result, "entities")


class TestModuleComposition:
    """Integration tests demonstrating DSPy module composition.

    DSPy modules can be composed into larger programs. These tests
    show how SIE components work within composed programs.
    """

    def test_custom_rag_module(self, sie_url: str) -> None:
        """Example: Custom RAG module using SIE components."""
        import dspy
        from sie_dspy import SIEEmbedder, SIEReranker

        class SIERAGModule(dspy.Module):
            """Custom RAG module with SIE retrieval and reranking."""

            def __init__(
                self,
                corpus: list[str],
                embedder: SIEEmbedder,
                reranker: SIEReranker,
                k: int = 3,
            ) -> None:
                super().__init__()
                self.retriever = dspy.retrievers.Embeddings(
                    corpus=corpus,
                    embedder=embedder,
                    k=k,
                )
                self.reranker = reranker

            def forward(self, question: str) -> dspy.Prediction:
                # Retrieve candidates
                retrieved = self.retriever(question)

                # Rerank for precision
                reranked = self.reranker(
                    query=question,
                    passages=retrieved.passages,
                    k=2,
                )

                return dspy.Prediction(
                    passages=reranked.passages,
                    scores=reranked.scores,
                )

        # Create module
        embedder = SIEEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        reranker = SIEReranker(base_url=sie_url)

        corpus = [
            "Python is great for machine learning.",
            "JavaScript powers the web.",
            "TensorFlow and PyTorch are deep learning frameworks.",
            "SQL is used for database queries.",
        ]

        rag = SIERAGModule(
            corpus=corpus,
            embedder=embedder,
            reranker=reranker,
            k=3,
        )

        result = rag("What tools are used for deep learning?")

        assert len(result.passages) == 2
        assert len(result.scores) == 2
