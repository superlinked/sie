"""SIE reranker integration for LlamaIndex.

Provides document reranking using SIE's score endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

if TYPE_CHECKING:
    from sie_sdk import SIEAsyncClient, SIEClient


class SIENodePostprocessor(BaseNodePostprocessor):
    """LlamaIndex node postprocessor using SIE's reranking.

    Wraps SIEClient.score() to implement BaseNodePostprocessor.

    Example:
        >>> from llama_index.core.query_engine import RetrieverQueryEngine
        >>> from sie_llamaindex import SIENodePostprocessor
        >>>
        >>> reranker = SIENodePostprocessor(
        ...     base_url="http://localhost:8080", model="jinaai/jina-reranker-v2-base-multilingual", top_n=3
        ... )
        >>>
        >>> query_engine = RetrieverQueryEngine.from_args(retriever=retriever, node_postprocessors=[reranker])

    Args:
        base_url: URL of the SIE server.
        model: Reranker model name/ID.
        top_n: Number of top nodes to return (default: all).
        options: Runtime options dict for model adapter overrides.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    # Pydantic fields
    base_url: str = Field(default="http://localhost:8080", description="SIE server URL")
    model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual",
        description="Reranker model name/ID",
    )
    top_n: int | None = Field(default=None, description="Number of top nodes to return")
    options: dict[str, Any] | None = Field(default=None, description="Runtime options")
    gpu: str | None = Field(default=None, description="GPU type for routing")
    timeout_s: float = Field(default=180.0, description="Request timeout")

    # Private attributes for clients
    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        top_n: int | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
        **kwargs: Any,
    ) -> None:
        """Initialize SIE node postprocessor."""
        super().__init__(
            base_url=base_url,
            model=model,
            top_n=top_n,
            options=options,
            gpu=gpu,
            timeout_s=timeout_s,
            **kwargs,
        )
        self._client = None
        self._async_client = None

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "SIENodePostprocessor"

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

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Rerank nodes by relevance to query.

        Args:
            nodes: List of nodes with scores to rerank.
            query_bundle: Query bundle containing query string.

        Returns:
            Reranked nodes with updated scores.
        """
        if not nodes:
            return []

        if query_bundle is None:
            # No query to rerank against, return as-is
            return nodes

        from sie_sdk.types import Item

        query_text = query_bundle.query_str
        query_item = Item(text=query_text)
        doc_items = [Item(text=node.node.get_content()) for node in nodes]

        results = self.client.score(
            self.model,
            query_item,
            doc_items,
        )

        # Build reranked nodes
        reranked = self._build_reranked_nodes(nodes, results)

        # Apply top_n limit if specified
        if self.top_n is not None:
            return reranked[: self.top_n]
        return reranked

    def _build_reranked_nodes(
        self,
        nodes: list[NodeWithScore],
        results: list[Any],
    ) -> list[NodeWithScore]:
        """Build reranked nodes from score results.

        Args:
            nodes: Original nodes.
            results: Score results from SIE.

        Returns:
            Reranked nodes with new scores.
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

            if idx < len(nodes):
                original_node = nodes[idx]
                # Create new NodeWithScore with updated score
                reranked.append(
                    NodeWithScore(
                        node=original_node.node,
                        score=float(score),
                    )
                )

        return reranked
