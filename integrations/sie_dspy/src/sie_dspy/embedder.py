"""SIE embedder for DSPy.

Provides an embedding function compatible with dspy.Embedder and
dspy.retrievers.Embeddings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class SIEEmbedder:
    """Embedding function using SIE.

    This class is callable and can be used directly with dspy.retrievers.Embeddings
    or wrapped with dspy.Embedder for other DSPy components.

    Example:
        >>> embedder = SIEEmbedder(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ... )
        >>> # Use with DSPy's FAISS retriever
        >>> retriever = dspy.retrievers.Embeddings(
        ...     corpus=["doc1", "doc2"],
        ...     embedder=embedder,
        ...     k=2,
        ... )
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
        """Initialize SIE embedder.

        Args:
            base_url: SIE server URL.
            model: Model name for embedding.
            gpu: GPU type for routing (e.g., "l4", "a100").
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
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

    def __call__(self, texts: str | list[str]) -> np.ndarray:
        """Embed text(s) and return numpy array.

        DSPy's Embeddings retriever calls the embedder with either:
        - A single query string
        - A list of corpus documents

        Args:
            texts: Single text or list of texts to embed.

        Returns:
            Numpy array of shape (n_texts, embedding_dim) or (embedding_dim,)
            for single text.
        """
        import numpy as np
        from sie_sdk.types import Item

        # Handle single text
        if isinstance(texts, str):
            result = self.client.encode(
                self._model,
                Item(text=texts),
                output_types=["dense"],
                options={"is_query": True},
            )
            return np.array(self._extract_dense(result), dtype=np.float32)

        # Handle list of texts (corpus)
        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model,
            items,
            output_types=["dense"],
        )

        embeddings = [self._extract_dense(r) for r in results]
        return np.array(embeddings, dtype=np.float32)

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding values from SDK result."""
        # SDK returns {"dense": np.ndarray, ...}
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            return []
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


class SIESparseEmbedder:
    """Sparse embedding function using SIE.

    Use alongside SIEEmbedder for hybrid search workflows.
    Users handle storage (Qdrant, Weaviate, etc.) themselves.

    Example:
        >>> from sie_dspy import SIEEmbedder, SIESparseEmbedder
        >>>
        >>> dense_embedder = SIEEmbedder(model="BAAI/bge-m3")
        >>> sparse_embedder = SIESparseEmbedder(model="BAAI/bge-m3")
        >>>
        >>> # Index corpus
        >>> dense_vecs = dense_embedder(corpus)
        >>> sparse_vecs = sparse_embedder.embed_documents(corpus)
        >>> # Store in your vector DB (Qdrant, Weaviate, etc.)
        >>>
        >>> # Query
        >>> dense_query = dense_embedder(query)
        >>> sparse_query = sparse_embedder.embed_query(query)
        >>> # Search your vector DB with both

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
