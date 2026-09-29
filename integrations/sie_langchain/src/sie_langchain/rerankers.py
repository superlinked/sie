"""SIE reranker integration for LangChain.

Provides document reranking using SIE's score endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import ConfigDict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sie_sdk import SIEAsyncClient, SIEClient


def _scores_by_index(results: Mapping[str, Any], count: int) -> list[float]:
    """Map ScoreResult entries back to input positions by item_id.

    Each input is sent with ``id=str(position)``, which the server echoes as
    the entry's ``item_id``. Only an exact echo of a sent id is used: an entry
    whose ``item_id`` is missing, not a string, or not a sent id is skipped
    (that input keeps its 0.0 default), so a malformed entry can neither
    crash the rerank nor mis-assign a score to the wrong input.

    Args:
        results: ScoreResult envelope from ``SIEClient.score()``.
        count: Number of input items.

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


class SIEReranker(BaseDocumentCompressor):
    """LangChain document compressor using SIE's reranking.

    Wraps SIEClient.score() to implement BaseDocumentCompressor.

    Example:
        >>> reranker = SIEReranker(
        ...     base_url="http://localhost:8080", model="jinaai/jina-reranker-v2-base-multilingual", top_k=3
        ... )
        >>> reranked = reranker.compress_documents(documents, query)

    Args:
        base_url: URL of the SIE server.
        model: Reranker model name/ID.
        top_k: Number of top documents to return (default: all).
        client: Optional pre-configured SIEClient instance.
        async_client: Optional pre-configured SIEAsyncClient instance.
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    # Pydantic v2 config
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Pydantic model configuration
    base_url: str = "http://localhost:8080"
    model: str = "jinaai/jina-reranker-v2-base-multilingual"
    top_k: int | None = None
    options: dict[str, Any] | None = None
    gpu: str | None = None
    timeout_s: float = 180.0

    # Private attributes for clients (Pydantic v2 style)
    _client: Any = None
    _async_client: Any = None

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        top_k: int | None = None,
        client: SIEClient | None = None,
        async_client: SIEAsyncClient | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE reranker."""
        super().__init__(base_url=base_url, model=model, top_k=top_k, options=options, gpu=gpu, timeout_s=timeout_s)
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._client

    @property
    def async_client(self) -> SIEAsyncClient:
        """Get or create the async SIEClient."""
        if self._async_client is None:
            from sie_sdk import SIEAsyncClient

            self._async_client = SIEAsyncClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._async_client

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,  # noqa: ARG002
    ) -> Sequence[Document]:
        """Rerank documents by relevance to query.

        Args:
            documents: Documents to rerank.
            query: Query to rank documents against.
            callbacks: Optional callbacks (not used).

        Returns:
            Reranked documents with scores in metadata.
        """
        if not documents:
            return []

        from sie_sdk.types import Item

        query_item = Item(text=query)
        doc_items = [Item(text=doc.page_content, id=str(idx)) for idx, doc in enumerate(documents)]

        results = self.client.score(
            self.model,
            query_item,
            doc_items,
        )

        reranked = self._build_reranked_documents(documents, results)

        # Apply top_k limit if specified
        if self.top_k is not None:
            return reranked[: self.top_k]
        return reranked

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks = None,  # noqa: ARG002
    ) -> Sequence[Document]:
        """Async rerank documents by relevance to query.

        Args:
            documents: Documents to rerank.
            query: Query to rank documents against.
            callbacks: Optional callbacks (not used).

        Returns:
            Reranked documents with scores in metadata.
        """
        if not documents:
            return []

        from sie_sdk.types import Item

        query_item = Item(text=query)
        doc_items = [Item(text=doc.page_content, id=str(idx)) for idx, doc in enumerate(documents)]

        results = await self.async_client.score(
            self.model,
            query_item,
            doc_items,
        )

        reranked = self._build_reranked_documents(documents, results)

        # Apply top_k limit if specified
        if self.top_k is not None:
            return reranked[: self.top_k]
        return reranked

    def _build_reranked_documents(self, documents: Sequence[Document], results: Mapping[str, Any]) -> list[Document]:
        """Build reranked documents from score results.

        Args:
            documents: Original documents.
            results: ScoreResult envelope from ``SIEClient.score()``. Ranked
                entries live under ``results["scores"]`` (each a ScoreEntry whose
                ``item_id`` echoes the input position sent as the item ``id``,
                plus ``score``), already sorted by relevance descending. The
                envelope also carries ``results["request"]`` (request id) and
                ``results["usage"]`` (token usage); those are available but
                intentionally not plumbed through the LangChain contract.

        Returns:
            Reranked documents (relevance descending) with scores in metadata.
        """
        scores = _scores_by_index(results, len(documents))
        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)
        return [
            Document(
                page_content=documents[i].page_content,
                metadata={**documents[i].metadata, "relevance_score": scores[i]},
            )
            for i in order
        ]
