"""SIE modules for DSPy.

Provides reranking and extraction modules compatible with DSPy pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import dspy


@dataclass
class Entity:
    """Extracted entity with metadata."""

    text: str
    label: str
    score: float
    start: int
    end: int


@dataclass
class Relation:
    """Extracted relation triple."""

    head: str
    tail: str
    relation: str
    score: float


@dataclass
class Classification:
    """Text classification result."""

    label: str
    score: float


@dataclass
class DetectedObject:
    """Detected object with bounding box."""

    label: str
    score: float
    bbox: list[int]


class SIEReranker(dspy.Module):
    """Rerank passages by relevance to a query using SIE.

    This module can be used in DSPy pipelines to improve retrieval quality
    by reranking initial retrieval results with a cross-encoder model.

    Example:
        >>> reranker = SIEReranker(
        ...     base_url="http://localhost:8080",
        ...     model="jinaai/jina-reranker-v2-base-multilingual",
        ... )
        >>> result = reranker(
        ...     query="What is machine learning?",
        ...     passages=["ML learns from data.", "Weather is sunny."],
        ...     k=1,
        ... )
        >>> print(result.passages)  # Most relevant passage
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "jinaai/jina-reranker-v2-base-multilingual",
        *,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE reranker.

        Args:
            base_url: SIE server URL.
            model: Model name for reranking.
            gpu: GPU type for routing.
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
        super().__init__()
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

    def forward(
        self,
        query: str,
        passages: list[str],
        k: int | None = None,
    ) -> dspy.Prediction:
        """Rerank passages by relevance to the query.

        Args:
            query: The query to rank against.
            passages: List of passages to rerank.
            k: Number of top passages to return. If None, returns all.

        Returns:
            Prediction with 'passages' (reranked list) and 'scores' (relevance scores).
        """
        if not passages:
            return dspy.Prediction(passages=[], scores=[])

        from sie_sdk.types import Item

        query_item = Item(text=query)
        passage_items = [Item(text=p) for p in passages]

        results = self.client.score(self._model, query_item, passage_items)

        # Build scored passages
        scored = []
        for passage, result in zip(passages, results, strict=True):
            score = result.get("score", 0.0) if isinstance(result, dict) else getattr(result, "score", 0.0)
            scored.append((float(score), passage))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply top_k
        if k is not None:
            scored = scored[:k]

        scores = [s for s, _ in scored]
        passages_out = [p for _, p in scored]

        return dspy.Prediction(passages=passages_out, scores=scores)


class SIEExtractor(dspy.Module):
    """Extract entities from text using SIE.

    This module uses a GLiNER or similar model to identify and extract
    named entities from text for use in DSPy pipelines.

    Example:
        >>> extractor = SIEExtractor(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> result = extractor(text="John works at Google in NYC.")
        >>> print(result.entities)  # List of Entity objects
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "urchade/gliner_multi-v2.1",
        labels: list[str] | None = None,
        *,
        gpu: str | None = None,
        options: dict[str, Any] | None = None,
        timeout_s: float = 180.0,
    ) -> None:
        """Initialize SIE extractor.

        Args:
            base_url: SIE server URL.
            model: Model name for extraction.
            labels: Default entity labels to extract.
            gpu: GPU type for routing.
            options: Model-specific options.
            timeout_s: Request timeout in seconds.
        """
        super().__init__()
        self._base_url = base_url
        self._model = model
        self._labels = labels or ["person", "organization", "location"]
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

    def forward(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> dspy.Prediction:
        """Extract structured information from text.

        Args:
            text: The text to extract from.
            labels: Labels to extract (overrides default labels).

        Returns:
            Prediction with entities, relations, classifications, objects
            (as dataclass objects) and _dict variants for serialization.
        """
        from sie_sdk.types import Item

        effective_labels = labels if labels is not None else self._labels

        result = self.client.extract(
            self._model,
            Item(text=text),
            labels=effective_labels,
        )

        entities = self._parse_entities(result)
        relations = self._parse_relations(result)
        classifications = self._parse_classifications(result)
        objects = self._parse_objects(result)

        return dspy.Prediction(
            entities=entities,
            relations=relations,
            classifications=classifications,
            objects=objects,
            entities_dict=[
                {"text": e.text, "label": e.label, "score": e.score, "start": e.start, "end": e.end} for e in entities
            ],
            relations_dict=[
                {"head": r.head, "tail": r.tail, "relation": r.relation, "score": r.score} for r in relations
            ],
            classifications_dict=[{"label": c.label, "score": c.score} for c in classifications],
            objects_dict=[{"label": o.label, "score": o.score, "bbox": o.bbox} for o in objects],
        )

    def _parse_entities(self, result: Any) -> list[Entity]:
        """Parse entities from SDK result."""
        entities = []

        # Result could be a dict with 'entities' key or a list directly
        if isinstance(result, dict):
            items = result.get("entities", [])
        elif isinstance(result, list):
            items = result
        else:
            items = []

        for item in items:
            if isinstance(item, dict):
                entities.append(
                    Entity(
                        text=item.get("text", ""),
                        label=item.get("label", ""),
                        score=float(item.get("score", 0.0)),
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                    )
                )
            else:
                entities.append(
                    Entity(
                        text=getattr(item, "text", ""),
                        label=getattr(item, "label", ""),
                        score=float(getattr(item, "score", 0.0)),
                        start=getattr(item, "start", 0),
                        end=getattr(item, "end", 0),
                    )
                )

        return entities

    def _parse_relations(self, result: Any) -> list[Relation]:
        """Parse relations from SDK result."""
        items = result.get("relations", []) if isinstance(result, dict) else getattr(result, "relations", [])
        relations = []
        for item in items:
            if isinstance(item, dict):
                relations.append(
                    Relation(
                        head=item.get("head", ""),
                        tail=item.get("tail", ""),
                        relation=item.get("relation", ""),
                        score=float(item.get("score", 0.0)),
                    )
                )
            else:
                relations.append(
                    Relation(
                        head=getattr(item, "head", ""),
                        tail=getattr(item, "tail", ""),
                        relation=getattr(item, "relation", ""),
                        score=float(getattr(item, "score", 0.0)),
                    )
                )
        return relations

    def _parse_classifications(self, result: Any) -> list[Classification]:
        """Parse classifications from SDK result."""
        items = (
            result.get("classifications", []) if isinstance(result, dict) else getattr(result, "classifications", [])
        )
        return [
            Classification(
                label=c.get("label", "") if isinstance(c, dict) else getattr(c, "label", ""),
                score=float(c.get("score", 0.0) if isinstance(c, dict) else getattr(c, "score", 0.0)),
            )
            for c in items
        ]

    def _parse_objects(self, result: Any) -> list[DetectedObject]:
        """Parse detected objects from SDK result."""
        items = result.get("objects", []) if isinstance(result, dict) else getattr(result, "objects", [])
        return [
            DetectedObject(
                label=o.get("label", "") if isinstance(o, dict) else getattr(o, "label", ""),
                score=float(o.get("score", 0.0) if isinstance(o, dict) else getattr(o, "score", 0.0)),
                bbox=o.get("bbox", []) if isinstance(o, dict) else getattr(o, "bbox", []),
            )
            for o in items
        ]
