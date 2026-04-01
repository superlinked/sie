"""SIE embedders for CrewAI.

Provides sparse embedding function for hybrid search workflows.
For dense embeddings, use SIE's OpenAI-compatible API with CrewAI's embedder config.
"""

from __future__ import annotations

from typing import Any


class SIESparseEmbedder:
    """Sparse embedding function using SIE.

    Use alongside SIE's OpenAI-compatible API (for dense) in hybrid search workflows.
    Users handle storage (Qdrant, Weaviate, etc.) themselves.

    Example:
        >>> from sie_crewai import SIESparseEmbedder
        >>>
        >>> sparse_embedder = SIESparseEmbedder(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ... )
        >>>
        >>> # For dense, use OpenAI-compatible API in CrewAI config
        >>> # embedder={"provider": "openai", "config": {"api_base": "http://localhost:8080/v1"}}
        >>>
        >>> # Get sparse embeddings for your corpus
        >>> sparse_vecs = sparse_embedder.embed_documents(corpus)
        >>> # Store in your vector DB alongside dense embeddings
        >>>
        >>> # Query
        >>> sparse_query = sparse_embedder.embed_query(query)

    Args:
        base_url: SIE server URL.
        model: Model name for sparse embedding.
        gpu: GPU type for routing.
        options: Model-specific options.
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        *,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE sparse embedder."""
        self._base_url = base_url
        self._model = model
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

    def embed_documents(self, texts: list[str]) -> list[dict[str, list]]:
        """Embed documents, returning sparse vectors.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of {"indices": [...], "values": [...]} dicts.
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model,
            items,
            output_types=["sparse"],
        )

        return [self._extract_sparse(r) for r in results]

    def embed_query(self, text: str) -> dict[str, list]:
        """Embed a query, returning sparse vector.

        Uses is_query=True for asymmetric models.

        Args:
            text: Query text to embed.

        Returns:
            {"indices": [...], "values": [...]} dict.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=["sparse"],
            options={"is_query": True},
        )

        return self._extract_sparse(result)

    def _extract_sparse(self, result: Any) -> dict[str, list]:
        """Extract sparse embedding from SDK result."""
        sparse = result.get("sparse") if isinstance(result, dict) else getattr(result, "sparse", None)
        if sparse is None:
            return {"indices": [], "values": []}

        indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
        values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

        return {
            "indices": indices.tolist() if hasattr(indices, "tolist") else list(indices or []),
            "values": values.tolist() if hasattr(values, "tolist") else list(values or []),
        }
