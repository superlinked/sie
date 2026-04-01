"""SIE embedding components for Haystack.

Provides embedder components following Haystack's conventions:
- SIETextEmbedder: For embedding single text strings (queries) - dense embeddings
- SIEDocumentEmbedder: For embedding documents - dense embeddings
- SIESparseTextEmbedder: For sparse embeddings of queries (hybrid search)
- SIESparseDocumentEmbedder: For sparse embeddings of documents (hybrid search)
"""

from __future__ import annotations

from typing import Any

from haystack import Document, component


@component
class SIETextEmbedder:
    """Embeds a single text string using SIE.

    Use this component for embedding queries in retrieval pipelines.

    Example:
        >>> embedder = SIETextEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
        >>> result = embedder.run(text="What is vector search?")
        >>> embedding = result["embedding"]  # list[float]
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
        """Initialize the text embedder.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for encoding.
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
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

    def warm_up(self) -> None:
        """Warm up the component by initializing the client."""
        _ = self.client

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Dictionary with "embedding" key containing the embedding vector.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=["dense"],
            options={"is_query": True},
        )
        embedding = self._extract_dense(result)
        return {"embedding": embedding}

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding from SDK result."""
        # SDK returns {"dense": np.ndarray, ...}
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            return []
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


@component
class SIEDocumentEmbedder:
    """Embeds documents using SIE and stores embeddings on each document.

    Use this component for embedding documents before indexing.

    Example:
        >>> from haystack import Document
        >>> embedder = SIEDocumentEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
        >>> docs = [Document(content="Python is a programming language.")]
        >>> result = embedder.run(documents=docs)
        >>> embedded_docs = result["documents"]
        >>> print(embedded_docs[0].embedding)  # list[float]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        *,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
        meta_fields_to_embed: list[str] | None = None,
    ) -> None:
        """Initialize the document embedder.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for encoding.
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
            timeout_s: Request timeout in seconds.
            meta_fields_to_embed: List of metadata fields to include in embedding.
        """
        self._base_url = base_url
        self._model = model
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._meta_fields_to_embed = meta_fields_to_embed or []
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
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Embed documents and store embeddings on each document.

        Args:
            documents: List of documents to embed.

        Returns:
            Dictionary with "documents" key containing documents with embeddings.
        """
        if not documents:
            return {"documents": []}

        from sie_sdk.types import Item

        # Build text to embed for each document
        texts = [self._build_text(doc) for doc in documents]
        items = [Item(text=text) for text in texts]

        # Batch encode
        results = self.client.encode(
            self._model,
            items,
            output_types=["dense"],
        )

        # Store embeddings on documents
        for doc, result in zip(documents, results, strict=True):
            doc.embedding = self._extract_dense(result)

        return {"documents": documents}

    def _build_text(self, doc: Document) -> str:
        """Build the text to embed for a document.

        Optionally includes metadata fields.
        """
        parts = [str(doc.meta[field]) for field in self._meta_fields_to_embed if field in doc.meta]
        parts.append(doc.content or "")
        return " ".join(parts)

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding from SDK result."""
        # SDK returns {"dense": np.ndarray, ...}
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            return []
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


@component
class SIESparseTextEmbedder:
    """Embeds a single text string using SIE sparse embeddings.

    Use this component for embedding queries in hybrid search pipelines.
    Works with QdrantHybridRetriever and other hybrid retrievers.

    Example:
        >>> embedder = SIESparseTextEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
        >>> result = embedder.run(text="What is vector search?")
        >>> sparse_embedding = result["sparse_embedding"]  # dict with indices/values
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
        """Initialize the sparse text embedder.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for encoding. Must support sparse output (e.g., BAAI/bge-m3).
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
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

    def warm_up(self) -> None:
        """Warm up the component by initializing the client."""
        _ = self.client

    @component.output_types(sparse_embedding=dict)
    def run(self, text: str) -> dict[str, dict[str, list]]:
        """Embed a single text string with sparse embeddings.

        Args:
            text: The text to embed.

        Returns:
            Dictionary with "sparse_embedding" key containing dict with "indices" and "values" lists.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=["sparse"],
            options={"is_query": True},
        )
        sparse_embedding = self._extract_sparse(result)
        return {"sparse_embedding": sparse_embedding}

    def _extract_sparse(self, result: Any) -> dict[str, list]:
        """Extract sparse embedding from SDK result."""
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


@component
class SIESparseDocumentEmbedder:
    """Embeds documents using SIE sparse embeddings and stores them on each document.

    Use this component for embedding documents before indexing in hybrid search pipelines.
    Works with QdrantDocumentStore(use_sparse_embeddings=True).

    Example:
        >>> from haystack import Document
        >>> embedder = SIESparseDocumentEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
        >>> docs = [Document(content="Python is a programming language.")]
        >>> result = embedder.run(documents=docs)
        >>> embedded_docs = result["documents"]
        >>> print(embedded_docs[0].meta["_sparse_embedding"])  # dict with indices/values
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        *,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
        meta_fields_to_embed: list[str] | None = None,
    ) -> None:
        """Initialize the sparse document embedder.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for encoding. Must support sparse output (e.g., BAAI/bge-m3).
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
            timeout_s: Request timeout in seconds.
            meta_fields_to_embed: List of metadata fields to include in embedding.
        """
        self._base_url = base_url
        self._model = model
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._meta_fields_to_embed = meta_fields_to_embed or []
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
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Embed documents with sparse embeddings and store on each document.

        Args:
            documents: List of documents to embed.

        Returns:
            Dictionary with "documents" key containing documents with sparse embeddings.
        """
        if not documents:
            return {"documents": []}

        from sie_sdk.types import Item

        # Build text to embed for each document
        texts = [self._build_text(doc) for doc in documents]
        items = [Item(text=text) for text in texts]

        # Batch encode with sparse output
        results = self.client.encode(
            self._model,
            items,
            output_types=["sparse"],
        )

        # Store sparse embeddings on documents in meta
        for doc, result in zip(documents, results, strict=True):
            doc.meta["_sparse_embedding"] = self._extract_sparse(result)

        return {"documents": documents}

    def _build_text(self, doc: Document) -> str:
        """Build the text to embed for a document.

        Optionally includes metadata fields.
        """
        parts = [str(doc.meta[field]) for field in self._meta_fields_to_embed if field in doc.meta]
        parts.append(doc.content or "")
        return " ".join(parts)

    def _extract_sparse(self, result: Any) -> dict[str, list]:
        """Extract sparse embedding from SDK result."""
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
