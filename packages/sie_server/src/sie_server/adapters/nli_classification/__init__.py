"""NLI-based zero-shot classification adapter.

Uses HuggingFace transformers zero-shot-classification pipeline with NLI models.
Compatible with MoritzLaurer's deberta-v3-zeroshot models and similar NLI classifiers.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.config.model import ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.responses import Classification

if TYPE_CHECKING:
    from transformers import Pipeline

    from sie_server.types.inputs import Item

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_LABELS = "Zero-shot classification requires labels parameter."


class NLIClassificationAdapter(ModelAdapter):
    """Adapter for NLI-based zero-shot classification models.

    Uses the HuggingFace transformers zero-shot-classification pipeline.
    Works with models like MoritzLaurer/deberta-v3-base-zeroshot-v2.0.

    The pipeline converts classification into NLI: for each label, it creates
    a hypothesis like "This text is about {label}" and scores entailment.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize NLI classification adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            hypothesis_template: Template for converting labels to hypotheses.
                Must contain {} placeholder for the label.
                Default: "This text is about {}."
            multi_label: If True, allows multiple labels per text.
                If False, forces single-label classification.
            compute_precision: Precision for inference (float16, float32, bfloat16).
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._hypothesis_template = hypothesis_template
        self._multi_label = multi_label
        self._compute_precision = compute_precision

        self._pipeline: Pipeline | None = None
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["json"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return output dimensions (none for classification)."""
        if self._pipeline is None:
            msg = "Dimensions not available until model is loaded"
            raise RuntimeError(msg)
        return ModelDims()  # No embedding dimensions

    def load(self, device: str) -> None:
        """Load model onto specified device.

        Args:
            device: Target device (cuda:0, cuda:1, cpu, mps).
        """
        from transformers import pipeline

        self._device = device

        # Determine device index for pipeline
        if device.startswith("cuda"):
            device_idx = int(device.rsplit(":", maxsplit=1)[-1]) if ":" in device else 0
        elif device == "mps":
            device_idx = "mps"
        else:
            device_idx = -1  # CPU

        # Determine torch dtype
        if device == "cpu":
            torch_dtype = torch.float32
        elif self._compute_precision == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self._compute_precision == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self._model_name_or_path,
            device=device_idx,
            dtype=torch_dtype,
        )

    def unload(self) -> None:
        """Release model and free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        gc.collect()

        device = self._device
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item.

        Args:
            item: Input item with text field.

        Returns:
            Text string.

        Raises:
            ValueError: If item has no text.
        """
        if not item.text:
            msg = "Item must have text for classification"
            raise ValueError(msg)
        return item.text

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
        """Classify texts with zero-shot labels.

        Args:
            items: List of items to classify (must have text).
            labels: Classification labels (e.g., ["positive", "negative", "neutral"]).
                Required for zero-shot classification.
            output_schema: Unused (included for interface compatibility).
            instruction: Unused (included for interface compatibility).
            options: Adapter options to override model config defaults.
                Supported: hypothesis_template (str), multi_label (bool).

        Returns:
            List of dicts, one per item, each containing:
                - "classifications": List of classification results, each with:
                    - "label": Classification label
                    - "score": Confidence score (0-1)
                - "entities": Empty list (for interface compatibility)
                - "data": Empty dict

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack text.
        """
        if self._pipeline is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        # Extract texts from all items
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults
        opts = options or {}
        effective_template = opts.get("hypothesis_template", self._hypothesis_template)
        effective_multi_label = opts.get("multi_label", self._multi_label)

        # Run classification pipeline
        # The pipeline handles batching internally
        with torch.inference_mode():
            pipeline_results = self._pipeline(
                texts,
                candidate_labels=labels,
                hypothesis_template=effective_template,
                multi_label=effective_multi_label,
            )

        # Handle single-item case (pipeline returns dict instead of list)
        if isinstance(pipeline_results, dict):
            pipeline_results = [pipeline_results]

        # Convert to our format
        all_classifications: list[list[Classification]] = []
        for pipeline_result in pipeline_results:
            # Pipeline returns {"labels": [...], "scores": [...]}
            classifications: list[Classification] = []
            for label, score in zip(
                pipeline_result["labels"],
                pipeline_result["scores"],
                strict=True,
            ):
                classifications.append(
                    Classification(
                        label=label,
                        score=float(score),
                    )
                )

            classifications.sort(key=lambda x: x["score"], reverse=True)
            all_classifications.append(classifications)

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=all_classifications,
        )

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
