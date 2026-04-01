"""SIE vectorizer helpers for Qdrant.

Qdrant supports both dense and sparse vectors natively. Dense vectors are
stored as ``list[float]``, and sparse vectors use Qdrant's native
``SparseVector(indices=..., values=...)`` format — no expansion needed.

These helpers wrap SIE's encode() to produce vectors in the format
Qdrant expects, handling Item creation and output type selection.
"""

from __future__ import annotations

from typing import Any


class SIEVectorizer:
    """Compute dense embeddings via SIE for Qdrant collections.

    Wraps ``SIEClient.encode()`` with ``output_types=["dense"]`` and
    converts results to ``list[float]`` — the format Qdrant's
    ``PointStruct(vector=...)`` expects for flat vector collections.

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
            from sie_sdk import SIEClient

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
            ``list[float]`` ready for ``PointStruct(vector=...)``.
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

        return [self._extract_dense(result) for result in results]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Passes ``is_query=True`` so instruction-tuned models (e.g., E5)
        apply query-specific prefixes or behavior.

        Args:
            text: Query text to embed.

        Returns:
            Dense vector as ``list[float]``, ready for
            ``client.query_points()``.
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

        return self._extract_dense(result)

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding values from SDK result."""
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            msg = "Encode result missing dense embedding"
            raise ValueError(msg)
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


class SIENamedVectorizer:
    """Compute multiple vector types via SIE for Qdrant named vectors.

    Uses a single SIE encode() call to produce multiple output types
    (e.g., dense + sparse) for Qdrant's named vector feature. This
    maps to collections configured with multiple vector configs.

    Unlike Weaviate, Qdrant supports sparse vectors natively via
    ``SparseVector(indices=..., values=...)``, so sparse output is
    returned in its native dict format without expansion.

    Example:
        >>> vectorizer = SIENamedVectorizer(
        ...     base_url="http://localhost:8080",
        ...     model="BAAI/bge-m3",
        ...     output_types=["dense", "sparse"],
        ... )
        >>> named_vectors = vectorizer.embed_documents(["hello", "world"])
        >>> # [{"dense": [0.1, ...], "sparse": {"indices": [...], "values": [...]}}, ...]
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
            output_types: Vector types to produce (default: ["dense", "sparse"]).
                Must match the named vector config on the Qdrant collection.
            instruction: Instruction prefix for instruction-tuned models (e.g., E5).
            output_dtype: Output data type (e.g., "float32", "float16", "int8", "binary").
            gpu: GPU type for routing (e.g., "l4", "a100").
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
        self._base_url = base_url
        self._model = model
        self._output_types = output_types or ["dense", "sparse"]
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
            from sie_sdk import SIEClient

            self._client = SIEClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._client

    def embed_documents(self, texts: list[str]) -> list[dict[str, Any]]:
        """Embed documents with multiple vector types.

        Args:
            texts: Document texts to embed.

        Returns:
            List of dicts mapping vector name to vector values.
            Dense vectors are ``list[float]``, sparse vectors are
            ``{"indices": list[int], "values": list[float]}``.
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

    def embed_query(self, text: str) -> dict[str, Any]:
        """Embed a query with multiple vector types.

        Passes ``is_query=True`` so instruction-tuned models apply
        query-specific behavior.

        Args:
            text: Query text to embed.

        Returns:
            Dict mapping vector name to vector values.
        """
        from sie_sdk.types import Item

        result = self.client.encode(
            self._model,
            Item(text=text),
            output_types=self._output_types,
            instruction=self._instruction,
            output_dtype=self._output_dtype,
            options={"is_query": True},
        )

        return self._extract_named(result)

    def _extract_named(self, result: Any) -> dict[str, Any]:
        """Extract all requested vector types from SDK result.

        Dense vectors are converted to ``list[float]``.
        Sparse vectors are returned as ``{"indices": list[int], "values": list[float]}``
        matching Qdrant's native ``SparseVector`` format.
        """
        named: dict[str, Any] = {}
        for output_type in self._output_types:
            raw = result.get(output_type) if isinstance(result, dict) else getattr(result, output_type, None)
            if raw is None:
                continue
            if output_type == "sparse":
                named[output_type] = self._format_sparse(raw)
            else:
                named[output_type] = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        return named

    @staticmethod
    def _format_sparse(sparse: Any) -> dict[str, list]:
        """Format SIE sparse output for Qdrant's SparseVector.

        SIE returns sparse vectors as ``{"indices": [...], "values": [...]}``.
        Qdrant's ``SparseVector`` accepts the same format natively, so we
        just convert numpy arrays to Python lists.

        Unlike Weaviate (which requires expanding sparse to full vocabulary
        length), Qdrant stores sparse vectors efficiently in their native
        indices+values form.
        """
        indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
        values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

        if indices is None or values is None:
            return {"indices": [], "values": []}

        indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
        values_list = values.tolist() if hasattr(values, "tolist") else list(values)

        return {"indices": indices_list, "values": values_list}
