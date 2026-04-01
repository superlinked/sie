"""SIE embedding functions for ChromaDB.

Provides custom embedding functions that use SIE for generating embeddings:
- SIEEmbeddingFunction: Dense embeddings for standard Chroma collections
- SIESparseEmbeddingFunction: Sparse embeddings for Chroma Cloud hybrid search
"""

from __future__ import annotations

from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings

# Type alias for sparse embeddings: list of {token_id: weight} dicts
SparseEmbeddings = list[dict[int, float]]


class SIEEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function using SIE for ChromaDB collections.

    This class implements ChromaDB's EmbeddingFunction protocol,
    allowing SIE to generate embeddings for document storage and retrieval.

    Example:
        >>> embedding_function = SIEEmbeddingFunction(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ... )
        >>> collection = client.create_collection(
        ...     name="my_collection",
        ...     embedding_function=embedding_function,
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
        """Initialize SIE embedding function.

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

    def __call__(self, documents: Documents) -> Embeddings:
        """Generate embeddings for documents.

        Args:
            documents: List of document texts to embed.

        Returns:
            List of embedding vectors as numpy arrays (ChromaDB's expected format).
        """
        if not documents:
            return []

        import numpy as np
        from sie_sdk.types import Item

        items = [Item(text=text) for text in documents]

        results = self.client.encode(
            self._model,
            items,
            output_types=["dense"],
        )

        embeddings: Embeddings = []
        for result in results:
            dense = self._extract_dense(result)
            # ChromaDB expects numpy arrays (Embeddings = List[numpy.ndarray])
            embeddings.append(np.array(dense, dtype=np.float32))

        return embeddings

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding values from SDK result."""
        # SDK returns {"dense": np.ndarray, ...}
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            return []
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


class SIESparseEmbeddingFunction:
    """Sparse embedding function using SIE for ChromaDB Cloud hybrid search.

    This class generates sparse embeddings for use with Chroma Cloud's
    hybrid search feature. Use with SparseVectorIndexConfig.

    Example:
        >>> from chromadb.cloud import SparseVectorIndexConfig, K
        >>> sparse_ef = SIESparseEmbeddingFunction(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ... )
        >>> # Configure sparse index on collection schema
        >>> schema.create_index(
        ...     key="sparse_embedding",
        ...     config=SparseVectorIndexConfig(source_key=K.DOCUMENT, embedding_function=sparse_ef),
        ... )

    For hybrid search, combine with dense embeddings using Rrf:
        >>> from chromadb.cloud import Rrf, Knn
        >>> hybrid_rank = Rrf(
        ...     ranks=[
        ...         Knn(query="search text", return_rank=True),  # dense
        ...         Knn(query="search text", key="sparse_embedding", return_rank=True),
        ...     ],
        ...     weights=[0.7, 0.3],
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
        """Initialize SIE sparse embedding function.

        Args:
            base_url: SIE server URL.
            model: Model name for embedding. Must support sparse output (e.g., BAAI/bge-m3).
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

    def __call__(self, documents: Documents) -> SparseEmbeddings:
        """Generate sparse embeddings for documents.

        Args:
            documents: List of document texts to embed.

        Returns:
            List of sparse embeddings. Each sparse embedding is a dict
            mapping token indices (int) to weights (float).
        """
        if not documents:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in documents]

        results = self.client.encode(
            self._model,
            items,
            output_types=["sparse"],
        )

        return [self._extract_sparse(result) for result in results]

    def _extract_sparse(self, result: Any) -> dict[int, float]:
        """Extract sparse embedding as token_id → weight dict.

        Chroma Cloud expects sparse embeddings as dict[int, float]
        where keys are token indices and values are weights.
        """
        # SDK returns {"sparse": {"indices": np.ndarray, "values": np.ndarray}, ...}
        sparse = result.get("sparse") if isinstance(result, dict) else getattr(result, "sparse", None)
        if sparse is None:
            return {}

        indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
        values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

        if indices is None or values is None:
            return {}

        # Convert to lists if numpy arrays
        indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        values_list = values.tolist() if hasattr(values, "tolist") else list(values)

        # Create dict mapping token_id → weight
        return dict(zip(indices_list, values_list))
