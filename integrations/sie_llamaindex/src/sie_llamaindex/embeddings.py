"""SIE embeddings integration for LlamaIndex.

Provides embedding generation using SIE's encode endpoint:
- SIEEmbedding: Dense embeddings implementing BaseEmbedding
- SIEMultiModalEmbedding: Dense embeddings for text and images implementing MultiModalEmbedding
- SIESparseEmbeddingFunction: Sparse embeddings for hybrid search
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.schema import ImageType
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.encoding import dense_embedding, sparse_embedding
from sie_sdk.types import Item


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
        result = self.client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )

        return dense_embedding(result)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        result = self.client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return dense_embedding(result)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self.model_name,
            items,
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return [dense_embedding(result) for result in results]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        result = await self.async_client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )

        return dense_embedding(result)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get embedding for a text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        result = await self.async_client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return dense_embedding(result)


class SIEMultiModalEmbedding(MultiModalEmbedding):
    """LlamaIndex MultiModalEmbedding implementation using SIE.

    Supports both text and image embedding, plugging into LlamaIndex's
    multimodal RAG pipelines (e.g. MultiModalVectorStoreIndex).

    Example:
        >>> from llama_index.core import Settings
        >>> from sie_llamaindex import SIEMultiModalEmbedding
        >>>
        >>> Settings.embed_model = SIEMultiModalEmbedding(
        ...     model_name="openai/clip-vit-large-patch14",
        ... )

    Args:
        base_url: URL of the SIE server.
        model_name: Model name/ID to use for encoding. Must support image input.
        instruction: Optional instruction prefix for embedding (model-dependent).
        output_dtype: Output dtype: "float32" (default), "float16", "int8", "binary".
        options: Runtime options dict passed to the model adapter.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
        embed_batch_size: Batch size for embedding multiple items.
    """

    base_url: str = Field(default="http://localhost:8080", description="SIE server URL")
    model_name: str = Field(default="openai/clip-vit-large-patch14", description="Model name/ID")
    instruction: str | None = Field(default=None, description="Instruction prefix")
    output_dtype: str | None = Field(default=None, description="Output dtype")
    options: dict[str, Any] | None = Field(default=None, description="Runtime options")
    gpu: str | None = Field(default=None, description="GPU type for routing")
    timeout_s: float = Field(default=180.0, description="Request timeout")

    _client: Any = PrivateAttr(default=None)
    _async_client: Any = PrivateAttr(default=None)

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model_name: str = "openai/clip-vit-large-patch14",
        instruction: str | None = None,
        output_dtype: str | None = None,
        options: dict[str, Any] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize SIE multimodal embedding model."""
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
        return "SIEMultiModalEmbedding"

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
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
            self._async_client = SIEAsyncClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._async_client

    # -- Text embedding methods (same as SIEEmbedding) --

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query string."""
        result = self.client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )
        return dense_embedding(result)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a text string."""
        result = self.client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )
        return dense_embedding(result)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        if not texts:
            return []

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self.model_name,
            items,
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )
        return [dense_embedding(result) for result in results]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Async get embedding for a query string."""
        result = await self.async_client.encode(
            self.model_name,
            Item(text=query),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
            options={"is_query": True},
        )
        return dense_embedding(result)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Async get embedding for a text string."""
        result = await self.async_client.encode(
            self.model_name,
            Item(text=text),
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )
        return dense_embedding(result)

    # -- Image embedding methods (MultiModalEmbedding interface) --

    def _get_image_embedding(self, img_file_path: ImageType) -> list[float]:
        """Get embedding for a single image.

        Args:
            img_file_path: File path (str) or BytesIO of the image.

        Returns:
            Embedding vector as list of floats.
        """
        img = img_file_path.getvalue() if isinstance(img_file_path, BytesIO) else img_file_path
        result = self.client.encode(
            self.model_name,
            Item(images=[img]),
            output_types=["dense"],
            output_dtype=self.output_dtype,
        )
        return dense_embedding(result)

    async def _aget_image_embedding(self, img_file_path: ImageType) -> list[float]:
        """Async get embedding for a single image.

        Args:
            img_file_path: File path (str) or BytesIO of the image.

        Returns:
            Embedding vector as list of floats.
        """
        img = img_file_path.getvalue() if isinstance(img_file_path, BytesIO) else img_file_path
        result = await self.async_client.encode(
            self.model_name,
            Item(images=[img]),
            output_types=["dense"],
            output_dtype=self.output_dtype,
        )
        return dense_embedding(result)

    def _get_image_embeddings(self, img_file_paths: list[ImageType]) -> list[list[float]]:
        """Get embeddings for multiple images in a single batch.

        Args:
            img_file_paths: List of file paths or BytesIO objects.

        Returns:
            List of embedding vectors.
        """
        if not img_file_paths:
            return []

        items = [Item(images=[img.getvalue() if isinstance(img, BytesIO) else img]) for img in img_file_paths]
        results = self.client.encode(
            self.model_name,
            items,
            output_types=["dense"],
            output_dtype=self.output_dtype,
        )
        return [dense_embedding(result) for result in results]


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
            sparse = sparse_embedding(result)
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

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model_name,
            items,
            output_types=["sparse"],
        )

        indices_list = []
        values_list = []
        for result in results:
            sparse = sparse_embedding(result)
            indices_list.append(sparse["indices"])
            values_list.append(sparse["values"])

        return indices_list, values_list
