"""Document enrichment for Weaviate via SIE encode + extract.

Combines SIE's embedding and extraction pipelines to produce documents
with both dense vectors and structured metadata (entities,
classifications) that Weaviate's Query Agent can filter on.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Iterator
from typing import Any, Self

from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.encoding import dense_embedding
from sie_sdk.types import Item


@dataclasses.dataclass
class EnrichedDocument:
    """A document enriched with SIE embeddings and extracted properties.

    Attributes:
        vector: Dense embedding ready for ``DataObject(vector=...)``.
        properties: Text plus extracted entity arrays and classifications,
            ready for ``DataObject(properties=...)``.
        entities: Raw entity extraction results for custom processing.
        classifications: Raw classification results for custom processing.
    """

    vector: list[float]
    properties: dict[str, Any]
    entities: list[dict[str, Any]] | None = None
    classifications: list[dict[str, Any]] | None = None


class SIEDocumentEnricher:
    """Enrich documents with SIE embeddings and extracted properties.

    Combines ``SIEClient.encode()`` and ``SIEClient.extract()`` to
    produce documents with dense vectors and structured metadata that
    Weaviate's Query Agent can filter on.  Extracted entities are grouped
    by label into ``list[str]`` properties (e.g., ``persons``,
    ``organizations``).

    Supports four modes of operation:

    - **Sync bulk** (``enrich``): returns all results at once.
    - **Sync streaming** (``enrich_iter``): yields per-chunk results.
    - **Async bulk** (``aenrich``): concurrent API calls via ``asyncio.gather``.
    - **Async streaming** (``aenrich_iter``): concurrent + streaming.

    Example:
        >>> enricher = SIEDocumentEnricher(
        ...     base_url="http://localhost:8080",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> docs = enricher.enrich(["John Smith works at Google in NYC."])
        >>> docs[0].properties
        {'text': 'John Smith works at Google in NYC.',
         'person': ['John Smith'], 'organization': ['Google'],
         'location': ['NYC']}
        >>> docs[0].vector  # dense embedding
        [0.12, -0.34, ...]
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        embed_model: str = "BAAI/bge-m3",
        *,
        extract_model: str | None = "urchade/gliner_medium-v2.1",
        labels: list[str] | None = None,
        classify_model: str | None = None,
        classify_labels: list[str] | None = None,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
        batch_size: int = 256,
    ) -> None:
        """Initialize document enricher.

        Args:
            base_url: SIE server URL.
            embed_model: Model for dense embeddings.
            extract_model: Model for entity extraction (e.g., GLiNER).
                Set to ``None`` to skip extraction (embed only).
            labels: Entity labels for extraction
                (e.g., ``["person", "organization", "location"]``).
            classify_model: Optional model for classification (e.g., GLiClass).
            classify_labels: Classification categories
                (e.g., ``["technical", "business", "legal"]``).
            gpu: GPU type for routing.
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
            batch_size: Number of texts per API call. Larger batches are
                more efficient but use more memory. Default 256.
        """
        self._base_url = base_url
        self._embed_model = embed_model
        self._extract_model = extract_model
        self._labels = labels
        self._classify_model = classify_model
        self._classify_labels = classify_labels
        if (self._classify_model is None) != (self._classify_labels is None):
            msg = "classify_model and classify_labels must both be provided or both be None"
            raise ValueError(msg)
        self._gpu = gpu
        self._options = options
        self._timeout_s = timeout_s
        self._batch_size = batch_size
        self._client: SIEClient | None = None
        self._async_client: SIEAsyncClient | None = None

    # -- Client properties --------------------------------------------------

    @property
    def client(self) -> SIEClient:
        """Lazily initialize the sync SIE client."""
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
        """Lazily initialize the async SIE client."""
        if self._async_client is None:
            self._async_client = SIEAsyncClient(
                self._base_url,
                timeout_s=self._timeout_s,
                gpu=self._gpu,
                options=self._options,
            )
        return self._async_client

    # -- Lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the sync client if initialized."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close the async client if initialized."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    # -- Sync API ------------------------------------------------------------

    def enrich(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[EnrichedDocument]:
        """Embed and extract structured properties from documents.

        Args:
            texts: Document texts to enrich.
            batch_size: Override the instance ``batch_size`` for this call.

        Returns:
            List of :class:`EnrichedDocument` with vectors and properties
            ready for Weaviate ``DataObject`` construction.
        """
        result: list[EnrichedDocument] = []
        for chunk_docs in self.enrich_iter(texts, batch_size=batch_size):
            result.extend(chunk_docs)
        return result

    def enrich_iter(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> Iterator[list[EnrichedDocument]]:
        """Embed and extract in chunks, yielding results per chunk.

        Enables pipelining: start inserting into Weaviate while the next
        chunk is being processed.

        Args:
            texts: Document texts to enrich.
            batch_size: Override the instance ``batch_size`` for this call.

        Yields:
            List of :class:`EnrichedDocument` for each chunk.
        """
        if not texts:
            return

        for chunk_texts in self._iter_chunks(texts, batch_size):
            items = [Item(text=t) for t in chunk_texts]

            encode_results = self.client.encode(
                self._embed_model,
                items,
                output_types=["dense"],
            )

            extract_results = None
            if self._extract_model is not None:
                extract_results = self.client.extract(
                    self._extract_model,
                    items,
                    labels=self._labels,
                )

            classify_results = None
            if self._classify_model is not None:
                classify_results = self.client.extract(
                    self._classify_model,
                    items,
                    labels=self._classify_labels,
                )

            yield self._assemble_chunk(chunk_texts, encode_results, extract_results, classify_results)

    def enrich_query(self, text: str) -> list[float]:
        """Embed a query (no extraction needed for queries).

        Args:
            text: Query text to embed.

        Returns:
            Dense vector as ``list[float]``.
        """
        result = self.client.encode(
            self._embed_model,
            Item(text=text),
            output_types=["dense"],
            options={"is_query": True},
        )
        return dense_embedding(result)

    # -- Async API -----------------------------------------------------------

    async def aenrich(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[EnrichedDocument]:
        """Async version of :meth:`enrich` with concurrent API calls.

        Runs encode, extract, and classify concurrently via
        ``asyncio.gather`` within each chunk.

        Args:
            texts: Document texts to enrich.
            batch_size: Override the instance ``batch_size`` for this call.

        Returns:
            List of :class:`EnrichedDocument`.
        """
        result: list[EnrichedDocument] = []
        async for chunk_docs in self.aenrich_iter(texts, batch_size=batch_size):
            result.extend(chunk_docs)
        return result

    async def aenrich_iter(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> AsyncIterator[list[EnrichedDocument]]:
        """Async generator with concurrent API calls per chunk.

        Within each chunk, encode/extract/classify run concurrently via
        ``asyncio.gather``, then results are assembled and yielded.

        Args:
            texts: Document texts to enrich.
            batch_size: Override the instance ``batch_size`` for this call.

        Yields:
            List of :class:`EnrichedDocument` for each chunk.
        """
        if not texts:
            return

        for chunk_texts in self._iter_chunks(texts, batch_size):
            items = [Item(text=t) for t in chunk_texts]

            has_extract = self._extract_model is not None
            has_classify = self._classify_model is not None and self._classify_labels is not None

            # Build concurrent coroutines
            coros: list[Any] = [self.async_client.encode(self._embed_model, items, output_types=["dense"])]
            if has_extract:
                coros.append(self.async_client.extract(self._extract_model, items, labels=self._labels))
            if has_classify:
                coros.append(self.async_client.extract(self._classify_model, items, labels=self._classify_labels))

            results = await asyncio.gather(*coros)

            # Unpack positionally based on which coros were added
            encode_results = results[0]
            idx = 1
            extract_results = results[idx] if has_extract else None
            if has_extract:
                idx += 1
            classify_results = results[idx] if has_classify else None

            yield self._assemble_chunk(chunk_texts, encode_results, extract_results, classify_results)

    async def aenrich_query(self, text: str) -> list[float]:
        """Async version of :meth:`enrich_query`.

        Args:
            text: Query text to embed.

        Returns:
            Dense vector as ``list[float]``.
        """
        result = await self.async_client.encode(
            self._embed_model,
            Item(text=text),
            output_types=["dense"],
            options={"is_query": True},
        )
        return dense_embedding(result)

    # -- Internal helpers ----------------------------------------------------

    def _iter_chunks(self, texts: list[str], batch_size: int | None) -> Iterator[list[str]]:
        """Split texts into chunks of ``batch_size``."""
        size = batch_size if batch_size is not None else self._batch_size
        for i in range(0, len(texts), size):
            yield texts[i : i + size]

    def _assemble_chunk(
        self,
        texts: list[str],
        encode_results: list[Any],
        extract_results: list[Any] | None,
        classify_results: list[Any] | None,
    ) -> list[EnrichedDocument]:
        """Assemble EnrichedDocument list from raw API results."""
        docs = []
        for i, text in enumerate(texts):
            vector = dense_embedding(encode_results[i])

            properties: dict[str, Any] = {"text": text}
            raw_entities: list[dict[str, Any]] | None = None
            raw_classifications: list[dict[str, Any]] | None = None

            if extract_results is not None:
                raw_entities = self._get_entities(extract_results[i])
                self._merge_entity_properties(properties, raw_entities)

            if classify_results is not None:
                raw_classifications = self._get_classifications(classify_results[i])
                self._merge_classification_properties(properties, raw_classifications)

            docs.append(
                EnrichedDocument(
                    vector=vector,
                    properties=properties,
                    entities=raw_entities,
                    classifications=raw_classifications,
                )
            )

        return docs

    @staticmethod
    def _get_entities(result: Any) -> list[dict[str, Any]]:
        """Extract entity list from SDK extract result."""
        entities = result.get("entities") if isinstance(result, dict) else getattr(result, "entities", None)
        return list(entities) if entities else []

    @staticmethod
    def _get_classifications(result: Any) -> list[dict[str, Any]]:
        """Extract classification list from SDK extract result."""
        classifications = (
            result.get("classifications") if isinstance(result, dict) else getattr(result, "classifications", None)
        )
        return list(classifications) if classifications else []

    @staticmethod
    def _merge_entity_properties(
        properties: dict[str, Any],
        entities: list[dict[str, Any]],
    ) -> None:
        """Group entities by label and merge into properties dict.

        Each label becomes a ``list[str]`` property with deduplicated
        entity texts.  For example, entities
        ``[{"label": "person", "text": "John"}, {"label": "person", "text": "Jane"}]``
        become ``{"person": ["John", "Jane"]}``.
        """
        seen: dict[str, set[str]] = {}
        by_label: dict[str, list[str]] = {}
        for entity in entities:
            label = entity.get("label") if isinstance(entity, dict) else getattr(entity, "label", None)
            text = entity.get("text") if isinstance(entity, dict) else getattr(entity, "text", None)
            if label and text:
                if label not in seen:
                    seen[label] = set()
                    by_label[label] = []
                if text not in seen[label]:
                    seen[label].add(text)
                    by_label[label].append(text)
        properties.update(by_label)

    @staticmethod
    def _merge_classification_properties(
        properties: dict[str, Any],
        classifications: list[dict[str, Any]],
    ) -> None:
        """Add top classification as a property.

        Sets ``classification`` to the highest-scoring label and
        ``classification_score`` to its confidence.
        """
        if not classifications:
            return
        best = max(
            classifications,
            key=lambda c: (c.get("score") if isinstance(c, dict) else getattr(c, "score", 0)) or 0,
        )
        label = best.get("label") if isinstance(best, dict) else getattr(best, "label", None)
        score = best.get("score") if isinstance(best, dict) else getattr(best, "score", None)
        if label:
            properties["classification"] = label
        if score is not None:
            properties["classification_score"] = float(score)
