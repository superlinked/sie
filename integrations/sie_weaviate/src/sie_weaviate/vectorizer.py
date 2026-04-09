"""SIE vectorizer helpers for Weaviate.

Weaviate v4 does not have a client-side embedding function protocol.
Collections are configured with ``Configure.Vectors.self_provided()``
and vectors must be passed explicitly on insert and query.

These helpers wrap SIE's encode() to produce vectors in the format
Weaviate expects: ``list[float]`` for dense, ``dict[str, list]`` for
named vectors.
"""

from __future__ import annotations

from typing import Any

from sie_sdk import SIEClient
from sie_sdk.encoding import dense_embedding, multivector_embedding


class SIEVectorizer:
    """Compute dense embeddings via SIE for Weaviate collections.

    Wraps ``SIEClient.encode()`` with ``output_types=["dense"]`` and
    converts results to ``list[float]`` — the format Weaviate's
    ``DataObject(vector=...)`` and ``query.near_vector()`` expect.

    Example:
        >>> vectorizer = SIEVectorizer(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ... )
        >>> vectors = vectorizer.embed_documents(["hello", "world"])
        >>> query_vec = vectorizer.embed_query("search text")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        *,
        instruction: str | None = None,
        output_dtype: str | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE vectorizer.

        Args:
            base_url: SIE server URL.
            model: Model name for embedding.
            instruction: Instruction prefix for instruction-tuned models (e.g., E5).
            output_dtype: Output data type (e.g., "float32", "float16", "int8", "binary").
            gpu: GPU type for routing (e.g., "l4", "a100").
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
        self._base_url = base_url
        self._model = model
        self._instruction = instruction
        self._output_dtype = output_dtype
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._client is None:
            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            texts: Document texts to embed.

        Returns:
            List of dense vectors, one per document. Each vector is a
            ``list[float]`` ready for ``DataObject(vector=...)``.
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

        Passes ``is_query=True`` so instruction-tuned models (e.g., E5)
        apply query-specific prefixes or behavior.

        Args:
            text: Query text to embed.

        Returns:
            Dense vector as ``list[float]``, ready for
            ``collection.query.near_vector()``.
        """
        from sie_sdk.types import Item

        # Note: ``self._options`` is passed to the SIEClient constructor
        # (for client-level config like GPU routing).  Per-call options here
        # are kept separate — only ``is_query`` is needed at call time.
        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=["dense"],
            instruction=self._instruction,
            output_dtype=self._output_dtype,
            options={"is_query": True},
        )

        return dense_embedding(result)


class SIENamedVectorizer:
    """Compute multiple vector types via SIE for Weaviate named vectors.

    Uses a single SIE encode() call to produce multiple output types
    (e.g., dense + multivector) for Weaviate's named vector feature.
    This maps to collections configured with multiple
    ``Configure.Vectors.self_provided(name=...)`` entries.

    Supported output types: ``"dense"``, ``"multivector"`` (ColBERT).

    Example:
        >>> vectorizer = SIENamedVectorizer(
        ...     base_url="http://localhost:8080",
        ...     model="jinaai/jina-colbert-v2",
        ...     output_types=["dense", "multivector"],
        ... )
        >>> named_vectors = vectorizer.embed_documents(["hello", "world"])
        >>> # [{"dense": [0.1, ...], "multivector": [[0.2, ...], ...]}, ...]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "BAAI/bge-m3",
        *,
        output_types: list[str] | None = None,
        instruction: str | None = None,
        output_dtype: str | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE named vectorizer.

        Args:
            base_url: SIE server URL.
            model: Model name for embedding.
            output_types: Vector types to produce (default: ["dense"]).
                Supports "dense" and "multivector" (ColBERT).
                Must match the named vector config on the Weaviate collection.
            instruction: Instruction prefix for instruction-tuned models (e.g., E5).
            output_dtype: Output data type (e.g., "float32", "float16", "int8", "binary").
            gpu: GPU type for routing (e.g., "l4", "a100").
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
        self._base_url = base_url
        self._model = model
        self._output_types = output_types or ["dense"]
        self._instruction = instruction
        self._output_dtype = output_dtype
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._client is None:
            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[dict[str, list]]:
        """Embed documents with multiple vector types.

        Args:
            texts: Document texts to embed.

        Returns:
            List of dicts mapping vector name to vector values.
            Ready for ``DataObject(vector={"dense": ..., "multivector": ...})``.
        """
        if not texts:
            return []

        from sie_sdk.types import Item

        items = [Item(text=text) for text in texts]
        results = self.client.encode(
            self._model,
            items,
            output_types=self._output_types,
            instruction=self._instruction,
            output_dtype=self._output_dtype,
        )

        return [self._extract_named(result) for result in results]

    def embed_query(self, text: str) -> dict[str, list]:
        """Embed a query with multiple vector types.

        Passes ``is_query=True`` so instruction-tuned models apply
        query-specific behavior.

        Args:
            text: Query text to embed.

        Returns:
            Dict mapping vector name to vector values.
        """
        from sie_sdk.types import Item

        # Note: ``self._options`` is passed to the SIEClient constructor
        # (for client-level config like GPU routing).  Per-call options here
        # are kept separate — only ``is_query`` is needed at call time.
        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=self._output_types,
            instruction=self._instruction,
            output_dtype=self._output_dtype,
            options={"is_query": True},
        )

        return self._extract_named(result)

    def _extract_named(self, result: Any) -> dict[str, list]:
        """Extract all requested vector types from SDK result."""
        named: dict[str, list] = {}
        for output_type in self._output_types:
            raw = result.get(output_type) if isinstance(result, dict) else getattr(result, output_type, None)
            if raw is None:
                continue
            if output_type == "multivector":
                named[output_type] = multivector_embedding(raw)
            else:
                named[output_type] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return named
