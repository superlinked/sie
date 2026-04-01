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
    from collections.abc import Sequence

    from sie_sdk import SIEAsyncClient, SIEClient


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
        doc_items = [Item(text=doc.page_content) for doc in documents]

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
        doc_items = [Item(text=doc.page_content) for doc in documents]

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

    def _build_reranked_documents(self, documents: Sequence[Document], results: list[Any]) -> list[Document]:
        """Build reranked documents from score results.

        Args:
            documents: Original documents.
            results: Score results from SIE.

        Returns:
            Reranked documents with scores.
        """
        reranked = []

        for result in results:
            # Handle dict or object result
            if isinstance(result, dict):
                idx = result.get("item_id", result.get("index", 0))
                score = result.get("score", 0.0)
            else:
                idx = getattr(result, "item_id", getattr(result, "index", 0))
                score = getattr(result, "score", 0.0)

            # Parse index if it's a string
            if isinstance(idx, str):
                idx = int(idx)

            if idx < len(documents):
                doc = documents[idx]
                # Create new document with score in metadata
                reranked.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "relevance_score": float(score)},
                    )
                )

        return reranked
