"""SIE embedding functions for LanceDB.

Provides LanceDB-native embedding functions registered in the global registry:
- SIEEmbeddingFunction: Dense text embeddings via "sie"
- SIEMultiVectorEmbeddingFunction: ColBERT/ColPali multi-vector via "sie-multivector"

These integrate with LanceDB's auto-embedding system: embeddings are computed
automatically on table.add() and table.search() when the schema uses
SourceField/VectorField annotations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lancedb.embeddings import EmbeddingFunction, TextEmbeddingFunction, register
from pydantic import PrivateAttr


@register("sie")
class SIEEmbeddingFunction(TextEmbeddingFunction):
    """Dense text embeddings via SIE, registered as "sie" in LanceDB.

    Wraps SIEClient.encode() with output_types=["dense"] to implement
    LanceDB's TextEmbeddingFunction interface. Embeddings are computed
    automatically when using SourceField/VectorField annotations.

    Example:
        >>> from lancedb.embeddings import get_registry
        >>> from lancedb.pydantic import LanceModel, Vector
        >>> import sie_lancedb  # registers "sie"
        >>>
        >>> sie = (
        ...     get_registry()
        ...     .get("sie")
        ...     .create(
        ...         model="BAAI/bge-m3",
        ...         base_url="http://localhost:8080",
        ...     )
        ... )
        >>>
        >>> class Documents(LanceModel):
        ...     text: str = sie.SourceField()
        ...     vector: Vector(sie.ndims()) = sie.VectorField()

    Args:
        base_url: URL of the SIE server.
        model: Model name/ID to use for encoding.
        instruction: Optional instruction prefix for embedding (model-dependent).
        output_dtype: Output dtype: "float32" (default), "float16", "int8", "binary".
        options: Runtime options dict passed to the model adapter.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    base_url: str = "http://localhost:8080"
    model: str = "BAAI/bge-m3"
    instruction: str | None = None
    output_dtype: str | None = None
    options: dict[str, Any] | None = None
    gpu: str | None = None
    timeout_s: float = 180.0

    _sie_client: Any = PrivateAttr(default=None)
    _ndims: int | None = PrivateAttr(default=None)

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._sie_client is None:
            from sie_sdk import SIEClient

            self._sie_client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._sie_client

    def ndims(self) -> int:
        """Return embedding dimensionality.

        Queries ``GET /v1/models/{model}`` on first call (lightweight,
        reads from model config — no model loading or inference).
        Cached after the first call per LanceDB's base class convention.
        """
        if self._ndims is None:
            info = self.client.get_model(self.model)
            dims = info.get("dims") or {}
            if "dense" not in dims:
                msg = f"Model '{self.model}' does not support dense embeddings"
                raise ValueError(msg)
            self._ndims = dims["dense"]
        return self._ndims

    def generate_embeddings(
        self,
        texts: list[str] | np.ndarray,
    ) -> list[np.ndarray | None]:
        """Generate dense embeddings for a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding arrays, one per text.
        """
        if not len(texts):
            return []

        from sie_sdk.types import Item

        items = [Item(text=t) for t in texts]
        results = self.client.encode(
            self.model,
            items,
            output_types=["dense"],
            instruction=self.instruction,
            output_dtype=self.output_dtype,
        )

        return [self._extract_dense(result) for result in results]

    def _extract_dense(self, result: Any) -> list[float]:
        """Extract dense embedding from encode result."""
        dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
        if dense is None:
            msg = "Encode result missing dense embedding"
            raise ValueError(msg)
        return dense.tolist() if hasattr(dense, "tolist") else list(dense)


@register("sie-multivector")
class SIEMultiVectorEmbeddingFunction(EmbeddingFunction):
    """Multi-vector embeddings via SIE, registered as "sie-multivector" in LanceDB.

    Wraps SIEClient.encode() with output_types=["multivector"] for ColBERT
    and ColPali models. Works with LanceDB's MultiVector type and native
    MaxSim scoring.

    Example:
        >>> from lancedb.embeddings import get_registry
        >>> from lancedb.pydantic import LanceModel, MultiVector
        >>> import sie_lancedb  # registers "sie-multivector"
        >>>
        >>> sie_colbert = (
        ...     get_registry()
        ...     .get("sie-multivector")
        ...     .create(
        ...         model="jinaai/jina-colbert-v2",
        ...         base_url="http://localhost:8080",
        ...     )
        ... )
        >>>
        >>> class Documents(LanceModel):
        ...     text: str = sie_colbert.SourceField()
        ...     vector: MultiVector(sie_colbert.ndims()) = sie_colbert.VectorField()

    Args:
        base_url: URL of the SIE server.
        model: Multi-vector model name/ID (e.g., ColBERT, ColPali).
        instruction: Optional instruction prefix for embedding.
        options: Runtime options dict passed to the model adapter.
        gpu: Target GPU type for routing (e.g., "l4", "a100-80gb").
        timeout_s: Request timeout in seconds.
    """

    base_url: str = "http://localhost:8080"
    model: str = "jinaai/jina-colbert-v2"
    instruction: str | None = None
    options: dict[str, Any] | None = None
    gpu: str | None = None
    timeout_s: float = 180.0

    _sie_client: Any = PrivateAttr(default=None)
    _ndims: int | None = PrivateAttr(default=None)

    @property
    def client(self) -> Any:
        """Lazily initialize the SIE client."""
        if self._sie_client is None:
            from sie_sdk import SIEClient

            self._sie_client = SIEClient(
                self.base_url,
                timeout_s=self.timeout_s,
                gpu=self.gpu,
                options=self.options,
            )
        return self._sie_client

    def ndims(self) -> int:
        """Return per-token embedding dimensionality.

        Queries ``GET /v1/models/{model}`` on first call (lightweight,
        reads from model config — no model loading or inference).
        Cached after the first call per LanceDB's base class convention.
        """
        if self._ndims is None:
            info = self.client.get_model(self.model)
            dims = info.get("dims") or {}
            if "multivector" not in dims:
                msg = f"Model '{self.model}' does not support multivector embeddings"
                raise ValueError(msg)
            self._ndims = dims["multivector"]
        return self._ndims

    def compute_source_embeddings(
        self,
        *args: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[list[list[float]] | None]:
        """Compute multi-vector embeddings for source documents.

        Args:
            *args: First positional arg is texts (str, list, pa.Array, etc.).

        Returns:
            List of 2D embeddings (tokens x dims), one per document.
        """
        texts = self.sanitize_input(args[0])

        from sie_sdk.types import Item

        items = [Item(text=t) for t in texts]
        results = self.client.encode(
            self.model,
            items,
            output_types=["multivector"],
            instruction=self.instruction,
        )

        return [self._extract_multivector(result) for result in results]

    def compute_query_embeddings(
        self,
        *args: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[list[list[float]] | None]:
        """Compute multi-vector embeddings for queries.

        Passes is_query=True for models that differentiate query/document encoding.

        Args:
            *args: First positional arg is query text(s).

        Returns:
            List of 2D embeddings (tokens x dims), one per query.
        """
        query = args[0]
        if isinstance(query, str):
            query = [query]
        texts = self.sanitize_input(query)

        from sie_sdk.types import Item

        items = [Item(text=t) for t in texts]
        results = self.client.encode(
            self.model,
            items,
            output_types=["multivector"],
            instruction=self.instruction,
            options={"is_query": True},
        )

        return [self._extract_multivector(result) for result in results]

    def _extract_multivector(self, result: Any) -> list[list[float]]:
        """Extract multi-vector embedding from encode result."""
        mv = result.get("multivector") if isinstance(result, dict) else getattr(result, "multivector", None)
        if mv is None:
            msg = "Encode result missing multivector embedding"
            raise ValueError(msg)
        return mv.tolist() if hasattr(mv, "tolist") else list(mv)
