"""SIE extractor component for Haystack.

Provides SIEExtractor for extracting entities, relations, classifications,
and detected objects from text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from haystack import component


@dataclass
class Entity:
    """Extracted entity with position and label information."""

    text: str
    label: str
    score: float
    start: int
    end: int


@dataclass
class Relation:
    """Extracted relation triple between two entities."""

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


@component
class SIEExtractor:
    """Extracts structured information from text using SIE.

    Use this component to extract named entities, relations, classifications,
    and detected objects using GLiNER, GLiREL, GLiClass, or similar models.

    Example:
        >>> extractor = SIEExtractor(
        ...     base_url="http://localhost:8080",
        ...     model="urchade/gliner_multi-v2.1",
        ...     labels=["person", "organization", "location"],
        ... )
        >>> result = extractor.run(text="John Smith works at Google in New York.")
        >>> entities = result["entities"]
        >>> for entity in entities:
        ...     print(f"{entity.text} ({entity.label}): {entity.score:.2f}")
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
        """Initialize the extractor.

        Args:
            base_url: URL of the SIE server.
            model: Model name to use for extraction.
            labels: Labels to extract (entity types, relation types, or classification labels).
            gpu: GPU type to use (e.g., "l4", "a100"). Passed to SDK as default.
            options: Model-specific options. Passed to SDK as default.
            timeout_s: Request timeout in seconds.
        """
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

    def warm_up(self) -> None:
        """Warm up the component by initializing the client."""
        _ = self.client

    @component.output_types(
        entities=list[Entity],
        relations=list[Relation],
        classifications=list[Classification],
        objects=list[DetectedObject],
    )
    def run(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> dict[str, list]:
        """Extract structured information from text.

        Args:
            text: The text to extract from.
            labels: Override the configured labels for this call.

        Returns:
            Dictionary with entities, relations, classifications, and objects.
        """
        from sie_sdk.types import Item

        effective_labels = labels if labels is not None else self._labels

        result = self.client.extract(
            self._model,
            Item(text=text),
            labels=effective_labels,
        )

        return {
            "entities": self._build_entities(result),
            "relations": self._build_relations(result),
            "classifications": self._build_classifications(result),
            "objects": self._build_objects(result),
        }

    def _build_entities(self, result: Any) -> list[Entity]:
        """Build Entity objects from SDK result."""
        items = self._get_field(result, "entities")
        entities = []
        for item in items:
            if isinstance(item, dict):
                entities.append(
                    Entity(
                        text=item.get("text", ""),
                        label=item.get("label", ""),
                        score=float(item.get("score") or 0.0),
                        start=int(item.get("start") or 0),
                        end=int(item.get("end") or 0),
                    )
                )
            else:
                entities.append(
                    Entity(
                        text=getattr(item, "text", ""),
                        label=getattr(item, "label", ""),
                        score=float(getattr(item, "score", None) or 0.0),
                        start=int(getattr(item, "start", None) or 0),
                        end=int(getattr(item, "end", None) or 0),
                    )
                )
        return entities

    def _build_relations(self, result: Any) -> list[Relation]:
        """Build Relation objects from SDK result."""
        items = self._get_field(result, "relations")
        relations = []
        for item in items:
            if isinstance(item, dict):
                relations.append(
                    Relation(
                        head=item.get("head", ""),
                        tail=item.get("tail", ""),
                        relation=item.get("relation", ""),
                        score=float(item.get("score") or 0.0),
                    )
                )
            else:
                relations.append(
                    Relation(
                        head=getattr(item, "head", ""),
                        tail=getattr(item, "tail", ""),
                        relation=getattr(item, "relation", ""),
                        score=float(getattr(item, "score", None) or 0.0),
                    )
                )
        return relations

    def _build_classifications(self, result: Any) -> list[Classification]:
        """Build Classification objects from SDK result."""
        items = self._get_field(result, "classifications")
        classifications = []
        for item in items:
            if isinstance(item, dict):
                classifications.append(
                    Classification(
                        label=item.get("label", ""),
                        score=float(item.get("score") or 0.0),
                    )
                )
            else:
                classifications.append(
                    Classification(
                        label=getattr(item, "label", ""),
                        score=float(getattr(item, "score", None) or 0.0),
                    )
                )
        return classifications

    def _build_objects(self, result: Any) -> list[DetectedObject]:
        """Build DetectedObject objects from SDK result."""
        items = self._get_field(result, "objects")
        objects = []
        for item in items:
            if isinstance(item, dict):
                objects.append(
                    DetectedObject(
                        label=item.get("label", ""),
                        score=float(item.get("score") or 0.0),
                        bbox=item.get("bbox", []),
                    )
                )
            else:
                objects.append(
                    DetectedObject(
                        label=getattr(item, "label", ""),
                        score=float(getattr(item, "score", None) or 0.0),
                        bbox=getattr(item, "bbox", []),
                    )
                )
        return objects

    def _get_field(self, result: Any, field: str) -> list:
        """Extract a field from the SDK result."""
        if isinstance(result, dict):
            return result.get(field, [])
        return getattr(result, field, [])
