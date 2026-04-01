"""SIE extractor component for Haystack.

Provides SIEExtractor for extracting entities from text.
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


@component
class SIEExtractor:
    """Extracts entities from text using SIE.

    Use this component to extract named entities or custom entity types
    from text using GLiNER or similar extraction models.

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
            labels: Entity labels to extract (e.g., ["person", "organization"]).
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

    @component.output_types(entities=list[Entity])
    def run(
        self,
        text: str,
        labels: list[str] | None = None,
    ) -> dict[str, list[Entity]]:
        """Extract entities from text.

        Args:
            text: The text to extract entities from.
            labels: Override the configured labels for this call.

        Returns:
            Dictionary with "entities" key containing extracted entities.
        """
        from sie_sdk.types import Item

        effective_labels = labels if labels is not None else self._labels

        result = self.client.extract(
            self._model,
            Item(text=text),
            labels=effective_labels,
        )

        entities = self._build_entities(result)
        return {"entities": entities}

    def _build_entities(self, result: Any) -> list[Entity]:
        """Build Entity objects from SDK result."""
        entities = []

        # Result could be a list of entities
        items = result if isinstance(result, list) else []

        for item in items:
            if isinstance(item, dict):
                entity = Entity(
                    text=item.get("text", ""),
                    label=item.get("label", ""),
                    score=float(item.get("score", 0.0)),
                    start=int(item.get("start", 0)),
                    end=int(item.get("end", 0)),
                )
            else:
                entity = Entity(
                    text=getattr(item, "text", ""),
                    label=getattr(item, "label", ""),
                    score=float(getattr(item, "score", 0.0)),
                    start=int(getattr(item, "start", 0)),
                    end=int(getattr(item, "end", 0)),
                )
            entities.append(entity)

        return entities
