"""SIE embeddings integration for LangChain.

Provides drop-in replacement for OpenAI embeddings using SIE's inference server:
- SIEEmbeddings: Dense embeddings for vector stores
- SIESparseEncoder: Sparse encoder for hybrid search with PineconeHybridSearchRetriever
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.encoding import dense_embedding, sparse_embedding


class SIEEmbeddings(Embeddings):
    """LangChain Embeddings implementation using SIE.

    Wraps SIEClient.encode() to implement the Embeddings interface.

    Example:
        >>> # Basic usage
        >>> embeddings = SIEEmbeddings(base_url="http://localhost:8080", model="BAAI/bge-m3")
        >>> vectors = embeddings.embed_documents(["Hello world"])

        >>> # With GPU routing for multi-GPU clusters
        >>> embeddings = SIEEmbeddings(base_url="https://cluster.example.com", model="BAAI/bge-m3", gpu="a100-80gb")

    Args:
        base_url: URL of the SIE server.
        model: Model name/ID to use for encoding.
        client: Optional pre-configured SIEClient instance.
        async_client: Optional pre-configured SIEAsyncClient instance.
        instruction: Optional instruction prefix for embedding (model-dependent).
        output_dtype: Output dtype: "float32" (default), "float16", "int8", "binary".
        options: Runtime options dict passed to the model adapter. Available options
            depend on the model - see model documentation for details.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        client: SIEClient | None = None,
        async_client: SIEAsyncClient | None = None,
        instruction: str | None = None,
        output_dtype: str | None = None,
        options: dict[str, object] | None = None,
        gpu: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE embeddings."""
        self._base_url = base_url
        self._model = model
        self._instruction = instruction
        self._output_dtype = output_dtype
        self._options = options
        self._gpu = gpu
        self._timeout_s = timeout_s

        # Store provided clients or create lazily
        self._client = client
        self._async_client = async_client

    @property
    def client(self) -> SIEClient:
        """Get or create the sync SIEClient."""
        if self._client is None:
            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    @property
    def async_client(self) -> SIEAsyncClient:
        """Get or create the async SIEClient."""
        if self._async_client is None:
            self._async_client = SIEAsyncClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._async_client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors (as lists of floats).
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model,
            items,
            output_types=["dense"],
            instruction=self._instruction,
            output_dtype=self._output_dtype,
        )

        return [dense_embedding(result) for result in results]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=["dense"],
            instruction=self._instruction,
            output_dtype=self._output_dtype,
            options={"is_query": True},
        )

        return dense_embedding(result)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed a list of documents.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors (as lists of floats).
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = await self.async_client.encode(
            self._model,
            items,
            output_types=["dense"],
            instruction=self._instruction,
            output_dtype=self._output_dtype,
        )

        return [dense_embedding(result) for result in results]

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed a single query text.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        from sie_sdk.types import Item

        result = await self.async_client.encode(
            self._model,
            Item(text=text),
            output_types=["dense"],
            instruction=self._instruction,
            output_dtype=self._output_dtype,
            options={"is_query": True},
        )

        return dense_embedding(result)


class SIESparseEncoder:
    """Sparse encoder for LangChain hybrid search.

    Compatible with PineconeHybridSearchRetriever's sparse_encoder interface.
    Provides encode_queries() and encode_documents() methods.

    Example:
        >>> from langchain_pinecone import PineconeHybridSearchRetriever
        >>> from sie_langchain import SIEEmbeddings, SIESparseEncoder
        >>>
        >>> retriever = PineconeHybridSearchRetriever(
        ...     embeddings=SIEEmbeddings(model="BAAI/bge-m3"),
        ...     sparse_encoder=SIESparseEncoder(model="BAAI/bge-m3"),
        ...     index=pinecone_index,
        ... )

    Args:
        base_url: URL of the SIE server.
        model: Model name/ID to use for encoding. Must support sparse output.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        gpu: str | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE sparse encoder."""
        self._base_url = base_url
        self._model = model
        self._gpu = gpu
        self._timeout_s = timeout_s
        self._client: SIEClient | None = None

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

    def encode_queries(self, texts: list[str]) -> list[dict[str, list]]:
        """Encode query texts to sparse vectors.

        Args:
            texts: List of query texts to encode.

        Returns:
            List of dicts with "indices" and "values" keys.
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model,
            items,
            output_types=["sparse"],
            options={"is_query": True},
        )

        return [sparse_embedding(result) for result in results]

    def encode_documents(self, texts: list[str]) -> list[dict[str, list]]:
        """Encode document texts to sparse vectors.

        Args:
            texts: List of document texts to encode.

        Returns:
            List of dicts with "indices" and "values" keys.
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

        return [sparse_embedding(result) for result in results]
