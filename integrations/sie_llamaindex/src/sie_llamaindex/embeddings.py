"""SIE embeddings integration for LlamaIndex.

Provides embedding generation using SIE's encode endpoint:
- SIEEmbedding: Dense embeddings implementing BaseEmbedding
- SIESparseEmbeddingFunction: Sparse embeddings for hybrid search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

if TYPE_CHECKING:
    from sie_sdk import SIEAsyncClient, SIEClient


class SIEEmbedding(BaseEmbedding):
    """LlamaIndex BaseEmbedding implementation using SIE.

    Wraps SIEClient.encode() to implement the BaseEmbedding interface.

    Example:
        >>> from llama_index.core import Settings
        >>> from sie_llamaindex import SIEEmbedding
        >>>
        >>> # Set as default embedding model
        >>> Settings.embed_model = SIEEmbedding(base_url="http://localhost:8080", model_name="BAAI/bge-m3")

        >>> # With GPU routing for multi-GPU clusters
        >>> embed_model = SIEEmbedding(
        ...     base_url="https://cluster.example.com", model_name="BAAI/bge-m3", gpu="a100-80gb"
        ... )

    Args:
        base_url: URL of the SIE server.
        model_name: Model name/ID to use for encoding.
        instruction: Optional instruction prefix for embedding (model-dependent).
        output_dtype: Output dtype: "float32" (default), "float16", "int8", "binary".
        options: Runtime options dict passed to the model adapter.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
        embed_batch_size: Batch size for embedding multiple texts.
    """

    # Pydantic fields (LlamaIndex uses Pydantic v2)
    base_url: str = Field(default="http://localhost:8080", description="SIE server URL")
    model_name: str = Field(default="BAAI/bge-m3", description="Model name/ID")
    instruction: str | None = Field(default=None, description="Instruction prefix")
    output_dtype: str | None = Field(default=None, description="Output dtype")
    options: dict[str, Any] | None = Field(default=None, description="Runtime options")
    gpu: str | None = Field(default=None, description="GPU type for routing")
    timeout_s: float = Field(default=180.0, description="Request timeout")

    # Private attributes for clients
    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model_name: str = "BAAI/bge-m3",
        instruction: str | None = None,
        output_dtype: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize SIE embedding model."""
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            instruction=instruction,
            output_dtype=output_dtype,
            options=options,
            gpu=gpu,
            timeout_s=timeout_s,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._client = None
        self._async_client = None

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "SIEEmbedding"

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

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )

        return self._extract_dense(result)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return self._extract_dense(result)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self.model_name,
            items,
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return [self._extract_dense(result) for result in results]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = await self.async_client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )

        return self._extract_dense(result)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = await self.async_client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return self._extract_dense(result)

    def _extract_dense(self, result: object) -> list[float]:
        """Extract dense embedding from encode result.

        Args:
            result: EncodeResult from SIE client (dict with "dense" key containing numpy array).

        Returns:
            Dense embedding as list of floats.
        """
        # SDK returns {"dense": np.ndarray, ...}
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            msg = "Encode result missing dense embedding"
            raise ValueError(msg)
        # Convert numpy array to list
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


class SIESparseEmbeddingFunction:
    """Sparse embedding function for LlamaIndex hybrid search.

    Compatible with LlamaIndex vector stores that support hybrid search,
    such as QdrantVectorStore with enable_hybrid=True.

    Example:
        >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
        >>> from sie_llamaindex import SIESparseEmbeddingFunction
        >>>
        >>> sparse_embed_fn = SIESparseEmbeddingFunction(base_url="http://localhost:8080", model_name="BAAI/bge-m3")
        >>>
        >>> vector_store = QdrantVectorStore(
        ...     client=qdrant_client,
        ...     collection_name="hybrid_docs",
        ...     enable_hybrid=True,
        ...     sparse_embedding_function=sparse_embed_fn,
        ... )

    Args:
        base_url: URL of the SIE server.
        model_name: Model name/ID to use for encoding. Must support sparse output.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model_name: str = "BAAI/bge-m3",
        gpu: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE sparse embedding function."""
        self._base_url = base_url
        self._model_name = model_name
        self._gpu = gpu
        self._timeout_s = timeout_s
        self._client: Any = None

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
            )
        return self._client

    def encode_queries(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        """Encode query texts to sparse vectors.

        Args:
            texts: List of query texts to encode.

        Returns:
            Tuple of (indices_list, values_list) where each is a list of lists.
        """
        if not texts:
            return [], []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model_name,
            items,
            output_types=["sparse"],
            options={"is_query": True},
        )

        indices_list = []
        values_list = []
        for result in results:
            sparse = self._extract_sparse(result)
            indices_list.append(sparse["indices"])
            values_list.append(sparse["values"])

        return indices_list, values_list

    def encode_documents(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        """Encode document texts to sparse vectors.

        Args:
            texts: List of document texts to encode.

        Returns:
            Tuple of (indices_list, values_list) where each is a list of lists.
        """
        if not texts:
            return [], []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model_name,
            items,
            output_types=["sparse"],
        )

        indices_list = []
        values_list = []
        for result in results:
            sparse = self._extract_sparse(result)
            indices_list.append(sparse["indices"])
            values_list.append(sparse["values"])

        return indices_list, values_list

    def _extract_sparse(self, result: object) -> dict[str, list]:
        """Extract sparse embedding from encode result.

        Args:
            result: EncodeResult from SIE client with "sparse" key.

        Returns:
            Dict with "indices" and "values" lists.
        """
        # SDK returns {"sparse": {"indices": np.ndarray, "values": np.ndarray}, ...}
        sparse = result.get("sparse") if isinstance(result, dict) else getattr(result, "sparse", None)
        if sparse is None:
            return {"indices": [], "values": []}

        indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
        values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

        return {
            "indices": indices.tolist() if hasattr(indices, "tolist") else list(indices or []),
            "values": values.tolist() if hasattr(values, "tolist") else list(values or []),
        }
