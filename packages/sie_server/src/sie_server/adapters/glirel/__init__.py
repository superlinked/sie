"""GLiREL adapter for zero-shot relation extraction.

GLiREL (Generalized Relation Extraction) models extract relations between
entities without fine-tuning. Given text, entities, and relation labels,
they output (head, relation, tail) triples with confidence scores.

Reference models:
- jackboyla/glirel-large-v0 (zero-shot relation extraction)
- jackboyla/glirel_re_large-v0 (relation-focused variant)
"""

import re
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
_ERR_REQUIRES_ENTITIES = "GLiREL requires entities in item metadata for relation extraction"
_TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


class GLiRELAdapter(BaseAdapter):
    """Adapter for GLiREL zero-shot relation extraction models.

    GLiREL extracts relations between entities. You provide:
    - Text to analyze
    - Entity spans in item metadata
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
        revision: str | None = None,
        **kwargs: Any,  # Accept extra args from loader
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path to GLiREL model.
            threshold: Minimum confidence score for relation extraction (0-1).
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._compute_precision = compute_precision
        self._revision = revision

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
        load_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            load_kwargs["revision"] = self._revision
        self._model = GLiREL.from_pretrained(self._model_name_or_path, **load_kwargs)

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
                raise ValueError(_ERR_REQUIRES_ENTITIES)

            tokens, token_offsets = self._tokenize(text)
            ner_input = self._to_glirel_ner(entities, token_offsets)

            # Get options with fallback to model defaults
            opts = options or {}
            effective_threshold = opts.get("threshold", self._threshold)
            effective_top_k = opts.get("top_k", 10)

            # GLiREL expects pre-tokenized text and inclusive token offsets.
            with torch.inference_mode():
                raw_relations = self._model.predict_relations(
                    text=tokens,
                    labels=labels,
                    threshold=effective_threshold,
                    ner=ner_input,
                    top_k=effective_top_k,
                )

            # Convert to proper Relation objects
            item_relations = []
            for rel in raw_relations:
                head_text = self._relation_entity_text(rel, "head", entities, ner_input)
                tail_text = self._relation_entity_text(rel, "tail", entities, ner_input)

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

    @staticmethod
    def _tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Tokenize text exactly like GLiREL and retain character offsets."""
        matches = list(_TOKEN_PATTERN.finditer(text))
        return [match.group() for match in matches], [(match.start(), match.end()) for match in matches]

    @staticmethod
    def _to_glirel_ner(
        entities: list[dict[str, Any]],
        token_offsets: list[tuple[int, int]],
    ) -> list[list[Any]]:
        """Convert SIE's character-offset entities to GLiREL token spans."""
        ner_input: list[list[Any]] = []
        for entity in entities:
            start_char = entity.get("start")
            end_char = entity.get("end")
            if not isinstance(start_char, int) or not isinstance(end_char, int) or start_char >= end_char:
                msg = "GLiREL entity metadata requires integer character offsets with start < end"
                raise ValueError(msg)

            covered_tokens = [
                index
                for index, (token_start, token_end) in enumerate(token_offsets)
                if token_end > start_char and token_start < end_char
            ]
            if not covered_tokens:
                msg = f"GLiREL entity span [{start_char}, {end_char}) does not cover any text token"
                raise ValueError(msg)

            ner_input.append(
                [
                    covered_tokens[0],
                    covered_tokens[-1],
                    entity.get("label", "ENTITY"),
                    entity.get("text", ""),
                ]
            )
        return ner_input

    @staticmethod
    def _relation_entity_text(
        relation: dict[str, Any],
        role: str,
        entities: list[dict[str, Any]],
        ner_input: list[list[Any]],
    ) -> str:
        """Use the caller's original entity text when GLiREL identifies its span."""
        position = relation.get(f"{role}_pos")
        if isinstance(position, list) and len(position) == 2:
            for entity, ner_span in zip(entities, ner_input):
                if position == [ner_span[0], ner_span[1] + 1]:
                    return str(entity.get("text", ""))

        relation_text = relation.get(f"{role}_text", "")
        if isinstance(relation_text, list):
            relation_text = " ".join(str(token) for token in relation_text)
            relation_text = re.sub(r"\s+([,.;:!?%])", r"\1", relation_text)
        return str(relation_text)
