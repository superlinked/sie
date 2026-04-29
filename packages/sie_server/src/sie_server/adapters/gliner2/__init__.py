from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity

_ERR_REQUIRES_LABELS = "GLiNER2 requires labels parameter for extraction"


class GLiNER2Adapter(BaseAdapter):
    """Adapter for GLiNER2 zero-shot NER models.

    GLiNER2 uses the separate ``gliner2`` pip package (NOT ``gliner``). It
    performs named entity recognition on arbitrary entity types and returns
    spans with confidence scores.

    Key API differences from GLiNER v1:
    - ``GLiNER2.from_pretrained(name, map_location=device, quantize=True)``
    - ``extract_entities()`` returns nested-by-label dict, not a flat list
    - Batch method is ``batch_extract_entities()``

    Reference models:
    - fastino/gliner2-base-v1

    See plan .kilo/plans/1776678677227-glowing-moon.md for design details.
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
        threshold: float = 0.5,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Minimum confidence score for entity extraction (0-1).
            compute_precision: Compute precision for inference.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._compute_precision = compute_precision

        self._model: Any = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        from gliner2 import GLiNER2  # ty:ignore[unresolved-import]

        self._device = device

        # GLiNER2 uses map_location for device placement and quantize for fp16
        use_quantize = device != "cpu" and self._compute_precision == "float16"
        self._model = GLiNER2.from_pretrained(
            self._model_name_or_path,
            map_location=device,
            quantize=use_quantize,
        )

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
            output_schema: Unused (interface compatibility).
            instruction: Unused (interface compatibility).
            options: Adapter options to override defaults.
                Supported: threshold (float).
            prepared_items: Unused (interface compatibility).

        Returns:
            ExtractOutput with entities per item.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack text.
        """
        self._check_loaded()

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        texts = [self._extract_text(item) for item in items]

        opts = options or {}
        effective_threshold = opts.get("threshold", self._threshold)

        with torch.inference_mode():
            if len(texts) == 1:
                # Single-item path
                raw_results = [
                    self._model.extract_entities(
                        texts[0],
                        labels,
                        threshold=effective_threshold,
                        include_confidence=True,
                        include_spans=True,
                    )
                ]
            else:
                # Batch path
                raw_results = self._model.batch_extract_entities(
                    texts,
                    labels,
                    threshold=effective_threshold,
                    include_confidence=True,
                    include_spans=True,
                )

        # Convert GLiNER2 nested-by-label format to flat SIE Entity list
        all_entities: list[list[Entity]] = []
        for result in raw_results:
            all_entities.append(self._flatten_entities(result))

        return ExtractOutput(entities=all_entities)

    @staticmethod
    def _flatten_entities(result: dict[str, Any]) -> list[Entity]:
        """Flatten GLiNER2 nested entity dict to SIE Entity list.

        GLiNER2 returns::

            {"entities": {"label_name": [{"text": ..., "confidence": ..., "start": ..., "end": ...}]}}

        This flattens to a list of Entity TypedDicts sorted by start offset.
        """
        entity_list: list[Entity] = []

        entities_dict = result.get("entities", {})
        for label_name, spans in entities_dict.items():
            for span in spans:
                entity_list.append(
                    Entity(
                        text=span["text"],
                        label=label_name,
                        score=float(span.get("confidence", 0.0)),
                        start=span.get("start"),
                        end=span.get("end"),
                    )
                )

        # Sort by start offset for consistent ordering
        entity_list.sort(key=lambda e: e.get("start") or 0)
        return entity_list

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER2 adapter"))
        return item.text
