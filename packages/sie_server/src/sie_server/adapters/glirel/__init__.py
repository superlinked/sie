"""GLiREL adapter for zero-shot relation extraction.

GLiREL (Generalized Relation Extraction) models extract relations between
entities without fine-tuning. Given text, entities, and relation labels,
they output (head, relation, tail) triples with confidence scores.

Reference models:
- jackboyla/glirel-large-v0 (zero-shot relation extraction)
- jackboyla/glirel_re_large-v0 (relation-focused variant)

See DESIGN.md Section 7.6 for adapter specification.
"""

from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity, Relation

# Error messages
_ERR_REQUIRES_LABELS = "GLiREL requires labels parameter for relation extraction"
_ERR_REQUIRES_ENTITIES = "GLiREL requires entities in metadata for relation extraction"


class GLiRELAdapter(BaseAdapter):
    """Adapter for GLiREL zero-shot relation extraction models.

    GLiREL extracts relations between entities. You provide:
    - Text to analyze
    - Entity spans (in item metadata or auto-detected)
    - Relation labels to look for (e.g., ["founded_by", "works_at"])

    Example usage:
        adapter = GLiRELAdapter("jackboyla/glirel-large-v0")
        adapter.load("cuda:0")
        results = adapter.extract(
            [Item(
                text="Apple Inc. was founded by Steve Jobs.",
                metadata={
                    "entities": [
                        {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                        {"text": "Steve Jobs", "label": "PER", "start": 26, "end": 36},
                    ]
                }
            )],
            labels=["founded_by", "works_at", "headquartered_in"],
        )
        # Returns: [{"relations": [
        #   {"head": "Apple Inc.", "tail": "Steve Jobs", "relation": "founded_by", "score": 0.92},
        # ]}]
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model",),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        threshold: float = 0.3,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,  # Accept extra args from loader
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path to GLiREL model.
            threshold: Minimum confidence score for relation extraction (0-1).
            compute_precision: Compute precision for inference.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._compute_precision = compute_precision

        self._model: Any = None  # GLiREL model type
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        # Import here to avoid dependency issues if glirel isn't installed
        from glirel import GLiREL  # ty:ignore[unresolved-import]

        self._device = device

        # Load model
        self._model = GLiREL.from_pretrained(self._model_name_or_path)

        # Move to device
        self._model = self._model.to(device)

        # Set eval mode
        self._model.eval()

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        """Extract relations from items.

        Args:
            items: List of items to extract from. Each item must have:
                - text: The text to analyze
                - metadata.entities: List of entity dicts with text, label, start, end
            labels: Relation types to extract (e.g., ["founded_by", "works_at"]).
                   Required for GLiREL models.
            output_schema: Unused for GLiREL (included for interface compatibility).
            instruction: Unused for GLiREL (included for interface compatibility).
            options: Adapter options to override model config defaults.
                    Supported: threshold (float), top_k (int).

        Returns:
            List of dicts, one per item, each containing:
                - "relations": List of extracted relations, each with:
                    - "head": Head entity text
                    - "tail": Tail entity text
                    - "relation": Relation type label
                    - "score": Confidence score (0-1)
                - "entities": Echo of input entities (if provided)
                - "data": Empty dict

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack entities.
        """
        self._check_loaded()

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        all_entities = []
        all_relations = []
        for item in items:
            text = self._extract_text(item)
            entities = self._extract_entities(item)

            if not entities:
                # No entities provided - return empty lists
                all_entities.append([])
                all_relations.append([])
                continue

            # Convert entities to GLiREL format: [[start_char, end_char, type, text], ...]
            # GLiREL v1.0+ uses character offsets, not token indices
            ner_input = []
            for ent in entities:
                start_char = ent.get("start", 0)
                end_char = ent.get("end", len(ent.get("text", "")))
                ner_input.append([start_char, end_char, ent.get("label", "ENTITY"), ent.get("text", "")])

            # Get options with fallback to model defaults
            opts = options or {}
            effective_threshold = opts.get("threshold", self._threshold)
            effective_top_k = opts.get("top_k", 10)

            # GLiREL prediction - v1.0+ API takes text directly
            with torch.inference_mode():
                raw_relations = self._model.predict_relations(
                    text=text,
                    labels=labels,
                    threshold=effective_threshold,
                    ner=ner_input,
                    top_k=effective_top_k,
                )

            # Convert to proper Relation objects
            item_relations = []
            for rel in raw_relations:
                # Extract head/tail text (GLiREL v1.0+ returns strings, strip extra whitespace)
                head_text = rel.get("head_text", "")
                tail_text = rel.get("tail_text", "")
                if isinstance(head_text, list):
                    head_text = " ".join(head_text)
                if isinstance(tail_text, list):
                    tail_text = " ".join(tail_text)

                item_relations.append(
                    Relation(
                        head=head_text.strip(),
                        tail=tail_text.strip(),
                        relation=rel.get("label", ""),
                        score=float(rel.get("score", 0.0)),
                    )
                )

            # Echo input entities
            item_entities = []
            for ent in entities:
                item_entities.append(
                    Entity(
                        text=ent.get("text", ""),
                        label=ent.get("label", ""),
                        score=ent.get("score", 1.0),
                        start=ent.get("start"),
                        end=ent.get("end"),
                    )
                )

            all_entities.append(item_entities)
            all_relations.append(item_relations)

        return ExtractOutput(entities=all_entities, relations=all_relations)

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="GLiREL adapter"))
        return item.text

    def _extract_entities(self, item: Item) -> list[dict[str, Any]]:
        """Extract entities from item metadata."""
        metadata = item.metadata
        if metadata is None:
            return []
        return metadata.get("entities", [])
