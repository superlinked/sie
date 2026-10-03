"""SIE ranker component for Haystack.

Provides SIERanker for reranking documents by relevance to a query.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from haystack import Document, component


def _scores_by_index(results: Mapping[str, Any], count: int) -> list[float]:
    """Map ScoreResult entries back to input positions by item_id.

    Each input is sent with ``id=str(position)``, which the server echoes as
    the entry's ``item_id``. Only an exact echo of a sent id is used: an entry
    whose ``item_id`` is missing, not a string, or not a sent id is skipped
    (that document keeps its 0.0 default), so a malformed entry can neither
    crash the rerank nor mis-assign a score to the wrong document.

    Args:
        results: ScoreResult envelope from ``SIEClient.score()``.
        count: Number of input documents.

    Returns:
        Scores indexed by input position (0.0 for any unscored/invalid item).
    """
    positions = {str(index): index for index in range(count)}
    scores = [0.0] * count
    for entry in results.get("scores", []):
        item_id = entry.get("item_id")
        if isinstance(item_id, str) and item_id in positions:
            scores[positions[item_id]] = float(entry.get("score", 0.0))
    return scores


@component
class SIERanker:
    """Reranks documents by relevance to a query using SIE.

    Use this component to improve retrieval precision by reranking
    candidate documents with a cross-encoder model.

    Example:
        >>> from haystack import Document
        >>> ranker = SIERanker(
        ...     base_url="http://localhost:8080",
        ...     model="jinaai/jina-reranker-v2-base-multilingual",
        ...     top_k=3,
        ... )
        >>> docs = [
        ...     Document(content="Python is a programming language."),
        ...     Document(content="The weather is sunny today."),
        ...     Document(content="Machine learning uses statistical models."),
        ... ]
        >>> result = ranker.run(query="What is Python?", documents=docs)
        >>> ranked_docs = result["documents"]  # Top 3 most relevant
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        *,
        top_k: int | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize the ranker.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for scoring.
            top_k: Maximum number of documents to return. If None, returns all.
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
            timeout_s: Request timeout in seconds.
        """
        self._base_url = base_url
        self._model = model
        self._top_k = top_k
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._client: Any = None

    @property
    def client(self) -> Any:
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

    def warm_up(self) -> None:
        """Warm up the component by initializing the client."""
        _ = self.client

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The query string to rank against.
            documents: List of documents to rerank.
            top_k: Override the configured top_k for this call.

        Returns:
            Dictionary with "documents" key containing ranked documents.
        """
        if not documents:
            return {"documents": []}

        from sie_sdk.types import Item

        # Prepare items
        query_item = Item(text=query)
        doc_items = [Item(text=doc.content or "", id=str(idx)) for idx, doc in enumerate(documents)]

        # Score documents
        results = self.client.score(self._model, query_item, doc_items)

        # ``results`` is a ScoreResult envelope; ranked entries are under
        # ``results["scores"]`` (each a ScoreEntry whose ``item_id`` echoes the
        # input position sent as the item ``id``, plus ``score``). Map scores
        # back to input order by item_id rather than zipping positionally — the
        # entries are sorted by relevance, not by input order. The envelope also
        # carries ``results["request"]`` (request id) and ``results["usage"]``
        # (token usage); those are available but intentionally not surfaced
        # through the Haystack contract.
        score_by_index = _scores_by_index(results, len(documents))

        # Build scored documents
        scored_docs = []
        for idx, doc in enumerate(documents):
            score = score_by_index[idx]
            # Store score in document metadata
            doc_with_score = Document(
                id=doc.id,
                content=doc.content,
                meta={**doc.meta, "score": score},
                embedding=doc.embedding,
            )
            scored_docs.append((score, doc_with_score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k
        effective_top_k = top_k if top_k is not None else self._top_k
        ranked_docs = [doc for _, doc in scored_docs]
        if effective_top_k is not None:
            ranked_docs = ranked_docs[:effective_top_k]

        return {"documents": ranked_docs}
