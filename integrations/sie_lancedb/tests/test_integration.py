"""Integration tests for sie-lancedb.

These tests require a running SIE server and serve as runnable examples
of LanceDB workflows using SIE embeddings, reranking, and extraction.

Run with: pytest -m integration integrations/sie_lancedb/tests/

Prerequisites:
    mise run serve -d cpu -p 8080
"""

from __future__ import annotations

import os

import lancedb
import pytest
import sie_lancedb  # noqa: F401 — registers "sie" and "sie-multivector"
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

# Skip all tests in this module if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    """Get SIE server URL from environment or default."""
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


@pytest.fixture
def db(tmp_path):
    """Ephemeral in-process LanceDB instance."""
    return lancedb.connect(tmp_path / "integration_test.lance")


class TestAutoEmbedding:
    """Integration tests for auto-embedding via LanceDB registry."""

    def test_auto_embed_and_search(self, sie_url: str, db) -> None:
        """Full round-trip: register SIE, add with auto-embed, search."""
        sie = (
            get_registry()
            .get("sie")
            .create(
                model="BAAI/bge-m3",
                base_url=sie_url,
            )
        )

        class Documents(LanceModel):
            text: str = sie.SourceField()
            vector: Vector(sie.ndims()) = sie.VectorField()

        table = db.create_table("auto_embed", schema=Documents, mode="overwrite")
        table.add(
            [
                {"text": "Machine learning is a subset of artificial intelligence."},
                {"text": "The weather forecast predicts rain tomorrow."},
                {"text": "Deep learning uses neural networks with multiple layers."},
                {"text": "Stock prices fluctuated significantly today."},
            ]
        )

        assert table.count_rows() == 4

        # Search auto-embeds the query
        results = table.search("How do neural networks work?").limit(2).to_list()

        assert len(results) == 2
        # ML-related docs should be most relevant
        contents = [r["text"] for r in results]
        assert any("neural" in c.lower() or "learning" in c.lower() for c in contents)

    def test_ndims_from_server(self, sie_url: str) -> None:
        """ndims() queries real server metadata without loading model."""
        sie = (
            get_registry()
            .get("sie")
            .create(
                model="BAAI/bge-m3",
                base_url=sie_url,
            )
        )

        dim = sie.ndims()

        assert dim == 1024  # BGE-M3 is 1024-dim


class TestHybridSearchWithReranker:
    """Integration tests for hybrid search + SIE reranking."""

    def test_hybrid_search_with_reranker(self, sie_url: str, db) -> None:
        """Hybrid search (vector + FTS) with SIE cross-encoder reranking."""
        from sie_lancedb import SIEReranker

        sie = (
            get_registry()
            .get("sie")
            .create(
                model="BAAI/bge-m3",
                base_url=sie_url,
            )
        )

        class Documents(LanceModel):
            text: str = sie.SourceField()
            vector: Vector(sie.ndims()) = sie.VectorField()

        table = db.create_table("hybrid_search", schema=Documents, mode="overwrite")
        table.add(
            [
                {"text": "SIE provides fast GPU inference for embeddings."},
                {"text": "LanceDB is a multimodal lakehouse for AI."},
                {"text": "Cross-encoder rerankers improve search precision."},
                {"text": "The weather is sunny with clear skies."},
                {"text": "Hybrid search combines vector and full-text search."},
            ]
        )

        table.create_fts_index("text", replace=True)

        reranker = SIEReranker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
        )

        results = (
            table.search("How does hybrid search improve results?", query_type="hybrid")
            .rerank(reranker)
            .limit(3)
            .to_list()
        )

        assert len(results) == 3
        assert "_relevance_score" in results[0]
        # Scores should be descending
        scores = [r["_relevance_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestEntityExtraction:
    """Integration tests for entity extraction and table enrichment."""

    def test_extract_entities(self, sie_url: str) -> None:
        """Direct entity extraction from text."""
        from sie_lancedb import SIEExtractor

        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
        )

        results = extractor.extract(
            ["Tim Cook leads Apple Inc. in Cupertino, California."],
            labels=["person", "organization", "location"],
        )

        assert len(results) == 1
        entities = results[0]
        assert len(entities) > 0

        labels_found = {e["label"] for e in entities}
        assert "person" in labels_found or "organization" in labels_found

    def test_enrich_table(self, sie_url: str, db) -> None:
        """Enrich a LanceDB table with extracted entities."""
        from sie_lancedb import SIEExtractor

        table = db.create_table(
            "enrich_test",
            data=[
                {"id": 0, "text": "Tim Cook is the CEO of Apple Inc."},
                {"id": 1, "text": "Sundar Pichai leads Google in Mountain View."},
                {"id": 2, "text": "Jensen Huang founded NVIDIA Corporation."},
            ],
            mode="overwrite",
        )

        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
        )

        extractor.enrich_table(
            table,
            source_column="text",
            target_column="entities",
            labels=["person", "organization", "location"],
            id_column="id",
        )

        df = table.to_pandas()
        assert "entities" in df.columns
        assert len(df) == 3

        # Each row should have extracted entities
        for entities in df["entities"]:
            assert isinstance(entities, list)
