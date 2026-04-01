"""GLiNER adapter for zero-shot NER extraction.

GLiNER (Generalized NER) models perform named entity recognition on arbitrary
entity types without fine-tuning. Given text and a list of entity labels,
they extract spans matching those labels.

Also supports NuNER models (token-based) with merge_adjacent_entities option.

Reference models:
- urchade/gliner_multi-v2.1 (multilingual, recommended)
- urchade/gliner_large-v2.1 (English, larger)
- urchade/gliner_small-v2.1 (English, smaller/faster)
- numind/NuNER_Zero (token-based, requires merge_adjacent_entities=True)
- numind/NuNER_Zero-span (span-based, works without merging)

See DESIGN.md Section 7.6 for adapter specification.
"""

from pathlib import Path
from typing import Any, Literal

import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity

# Compute precision type (for interface compatibility)
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "GLiNER adapter requires text input"
_ERR_REQUIRES_LABELS = "GLiNER requires labels parameter for extraction"


class GLiNERAdapter(ModelAdapter):
    """Adapter for GLiNER zero-shot NER models.

    GLiNER extracts entities of any specified type from text without
    model fine-tuning. You provide entity labels (e.g., ["person", "organization"])
    and the model returns matching spans with confidence scores.

    Example usage:
        adapter = GLiNERAdapter("urchade/gliner_multi-v2.1")
        adapter.load("cuda:0")
        results = adapter.extract(
            [Item(text="Apple Inc. was founded by Steve Jobs.")],
            labels=["person", "organization"],
        )
        # Returns: [{"entities": [
        #   {"text": "Apple Inc.", "label": "organization", "score": 0.95, "start": 0, "end": 10},
        #   {"text": "Steve Jobs", "label": "person", "score": 0.92, "start": 26, "end": 36},
        # ]}]
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        merge_adjacent_entities: bool = False,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path to GLiNER model.
            threshold: Minimum confidence score for entity extraction (0-1).
            flat_ner: If True, enforce non-overlapping entities (recommended).
            multi_label: If True, allow same span to have multiple labels.
            merge_adjacent_entities: If True, merge adjacent entities with same label.
                Required for token-based models like numind/NuNER_Zero.
            compute_precision: Compute precision for inference.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._flat_ner = flat_ner
        self._multi_label = multi_label
        self._merge_adjacent_entities = merge_adjacent_entities
        self._compute_precision = compute_precision

        self._model: Any = None  # GLiNER model type
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities.

        GLiNER models support extraction but not encoding or scoring.
        """
        return ModelCapabilities(
            inputs=["text"],
            outputs=["json"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions.

        GLiNER models don't produce embeddings, so all dims are None.
        """
        return ModelDims()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        # Import here to avoid dependency issues if gliner isn't installed
        from gliner import GLiNER  # ty:ignore[unresolved-import]

        self._device = device

        # Determine torch dtype from precision
        dtype = torch.float32
        if device != "cpu":
            if self._compute_precision == "float16":
                dtype = torch.float16
            elif self._compute_precision == "bfloat16":
                dtype = torch.bfloat16

        # Load model
        self._model = GLiNER.from_pretrained(
            self._model_name_or_path,
        )

        # Move to device with precision
        if device == "cpu":
            self._model = self._model.to(device)
        else:
            self._model = self._model.to(device, dtype=dtype)

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        self._device = None

        # Release GPU memory per the memory management contract in base.py
        import gc

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

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
        """Extract entities from items.

        Args:
            items: List of items to extract from (must have text).
            labels: Entity types to extract (e.g., ["person", "organization"]).
                   Required for GLiNER models.
            output_schema: Unused for GLiNER (included for interface compatibility).
            instruction: Unused for GLiNER (included for interface compatibility).
            options: Adapter options to override model config defaults.
                    Supported: threshold (float), flat_ner (bool), multi_label (bool),
                    merge_adjacent_entities (bool).

        Returns:
            List of dicts, one per item, each containing:
                - "entities": List of extracted entities, each with:
                    - "text": The extracted text span
                    - "label": Entity type label
                    - "score": Confidence score (0-1)
                    - "start": Start character offset
                    - "end": End character offset
                - "data": Empty dict (GLiNER doesn't produce structured data)

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack text.
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        # Extract texts from all items
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults
        opts = options or {}
        effective_threshold = opts.get("threshold", self._threshold)
        effective_flat_ner = opts.get("flat_ner", self._flat_ner)
        effective_multi_label = opts.get("multi_label", self._multi_label)
        merge_adjacent = opts.get("merge_adjacent_entities", self._merge_adjacent_entities)

        # Use batch prediction for efficiency (24x speedup vs single item loop)
        with torch.inference_mode():
            batch_entities = self._model.inference(
                texts,
                labels,
                threshold=effective_threshold,
                flat_ner=effective_flat_ner,
                multi_label=effective_multi_label,
            )

        # Convert to our format
        all_entities = []
        for text, entities in zip(texts, batch_entities):
            entity_results = []
            for entity in entities:
                entity_results.append(
                    Entity(
                        text=entity["text"],
                        label=entity["label"],
                        score=float(entity["score"]),
                        start=entity["start"],
                        end=entity["end"],
                    )
                )

            # Merge adjacent entities if enabled (for token-based models like NuNER)
            if merge_adjacent:
                entity_results = self._merge_entities(entity_results, text)

            all_entities.append(entity_results)

        return ExtractOutput(entities=all_entities)

    def _merge_entities(self, entities: list[Entity], text: str) -> list[Entity]:
        """Merge adjacent entities with the same label.

        Token-based models like NuNER_Zero output per-token predictions that need
        to be merged into contiguous spans. For example:
            [Entity(text="Steve", label="person"), Entity(text="Jobs", label="person")]
        becomes:
            [Entity(text="Steve Jobs", label="person")]

        Args:
            entities: List of Entity objects (TypedDict, i.e. dict).
            text: Original text (needed to extract merged spans).

        Returns:
            List of merged entities.
        """
        if not entities:
            return []

        # Sort by start position to ensure correct merging order
        # Entity is a TypedDict (dict), so use dict access
        sorted_entities = sorted(entities, key=lambda e: e.get("start") or 0)

        merged = []
        current = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check if adjacent (touching or 1 char gap) and same label
            is_adjacent = (next_entity.get("start") or 0) <= (current.get("end") or 0) + 1
            same_label = next_entity["label"] == current["label"]

            if is_adjacent and same_label:
                # Merge: extend current entity to include next
                new_end = next_entity.get("end")
                new_start = current.get("start")
                new_text = text[new_start:new_end] if new_start is not None and new_end is not None else current["text"]
                new_score = max(current["score"], next_entity["score"])
                current = Entity(
                    text=new_text,
                    label=current["label"],
                    score=new_score,
                    start=new_start,
                    end=new_end,
                )
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
