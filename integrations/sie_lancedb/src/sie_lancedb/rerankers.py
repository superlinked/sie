"""SIE reranker for LanceDB hybrid search.

Provides a LanceDB-native reranker that uses SIE's score endpoint for
cross-encoder reranking in hybrid (vector + FTS) search pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
from lancedb.rerankers import Reranker

if TYPE_CHECKING:
    from sie_sdk import SIEClient


class SIEReranker(Reranker):
    """LanceDB reranker using SIE's cross-encoder scoring.

    Plugs into LanceDB's hybrid search pipeline via the .rerank() method.
    Uses SIEClient.score() to compute relevance scores for query-document
    pairs.

    Example:
        >>> from sie_lancedb import SIEReranker
        >>>
        >>> table.create_fts_index("text")
        >>> results = (
        ...     table.search("flower moon", query_type="hybrid")
        ...     .rerank(SIEReranker(model="jinaai/jina-reranker-v2-base-multilingual"))
        ...     .limit(10)
        ...     .to_list()
        ... )

    Args:
        base_url: URL of the SIE server.
        model: Reranker model name/ID.
        column: Name of the text column to rerank on.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        options: Runtime options dict for model adapter overrides.
        timeout_s: Request timeout in seconds.
        return_score: "relevance" (default) returns only _relevance_score,
            "all" also keeps _distance and _score columns.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        column: str = "text",
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
        return_score: str = "relevance",
    ) -> None:
        """Initialize SIE reranker."""
        super().__init__(return_score)
        self._base_url = base_url
        self._model = model
        self._column = column
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._client: SIEClient | None = None

    @property
    def client(self) -> SIEClient:
        """Lazily initialize the SIE client."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ) -> pa.Table:
        """Rerank combined vector + FTS results.

        Args:
            query: Search query text.
            vector_results: Results from vector search.
            fts_results: Results from full-text search.

        Returns:
            Merged and reranked table with _relevance_score column.
        """
        combined = self.merge_results(vector_results, fts_results)
        return self._rerank_table(query, combined)

    def rerank_vector(
        self,
        query: str,
        vector_results: pa.Table,
    ) -> pa.Table:
        """Rerank pure vector search results.

        Args:
            query: Search query text.
            vector_results: Results from vector search.

        Returns:
            Reranked table with _relevance_score column.
        """
        return self._rerank_table(query, vector_results)

    def rerank_fts(
        self,
        query: str,
        fts_results: pa.Table,
    ) -> pa.Table:
        """Rerank pure full-text search results.

        Args:
            query: Search query text.
            fts_results: Results from full-text search.

        Returns:
            Reranked table with _relevance_score column.
        """
        return self._rerank_table(query, fts_results)

    def _rerank_table(self, query: str, table: pa.Table) -> pa.Table:
        """Score all rows against the query and add _relevance_score.

        Args:
            query: Search query text.
            table: Table with a text column to score against.

        Returns:
            Table sorted by _relevance_score descending.
        """
        if table.num_rows == 0:
            return self._handle_empty_results(table)

        from sie_sdk.types import Item

        texts = table.column(self._column).to_pylist()
        query_item = Item(text=query)
        doc_items = [Item(text=str(t)) for t in texts]

        results = self.client.score(self._model, query_item, doc_items)

        # Build score array indexed by input position
        scores = [0.0] * len(texts)
        for result in results["scores"]:
            if isinstance(result, dict):
                idx = result.get("item_id", result.get("index"))
                score = result.get("score", 0.0)
            else:
                idx = getattr(result, "item_id", getattr(result, "index", None))
                score = getattr(result, "score", 0.0)

            if idx is None:
                continue
            if isinstance(idx, str):
                idx = int(idx)
            if idx < len(scores):
                scores[idx] = float(score)

        # Add _relevance_score column and sort
        table = table.append_column(
            "_relevance_score",
            pa.array(scores, type=pa.float32()),
        )
        table = table.sort_by([("_relevance_score", "descending")])

        if self.score == "relevance":
            table = self._keep_relevance_score(table)

        return table
