"""GLiClass zero-shot classification adapter.

Uses Knowledgator's GLiClass library for efficient zero-shot text classification.
GLiClass is inspired by GLiNER but optimized for classification tasks.
Up to 50x faster than cross-encoders with similar accuracy.

Performance note (Dec 2025):
    Benchmarked GLiClass library at 496 texts/sec vs NLI flash adapter at 494 texts/sec
    (100 texts, 5 labels). GLiClass is a single-pass architecture (not N×M expansion
    like NLI cross-encoders), so the gliclass library pipeline has minimal overhead.
    No separate "GLiClassFlashAdapter" is needed - the library is already efficient.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.config.model import ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.responses import Classification

if TYPE_CHECKING:
    from gliclass import ZeroShotClassificationPipeline  # ty:ignore[unresolved-import]

    from sie_server.types.inputs import Item

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_LABELS = "Zero-shot classification requires labels parameter."

ClassificationType = Literal["single-label", "multi-label"]


class GLiClassAdapter(ModelAdapter):
    """Adapter for GLiClass zero-shot classification models.

    Uses the gliclass library's ZeroShotClassificationPipeline.
    Works with models like knowledgator/gliclass-base-v1.0.

    GLiClass performs classification in a single forward pass (not NLI-based),
    making it much faster than cross-encoder approaches.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        classification_type: ClassificationType = "single-label",
        threshold: float = 0.5,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize GLiClass adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            classification_type: "single-label" for mutually exclusive classes,
                "multi-label" for multiple classes per text.
            threshold: Confidence threshold for multi-label classification (0-1).
            compute_precision: Precision for inference (float16, float32, bfloat16).
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._classification_type = classification_type
        self._threshold = threshold
        self._compute_precision = compute_precision

        self._pipeline: ZeroShotClassificationPipeline | None = None
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
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline  # ty:ignore[unresolved-import]
        from transformers import AutoTokenizer

        self._device = device

        # Determine torch dtype
        if device == "cpu":
            torch_dtype = torch.float32
        elif self._compute_precision == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self._compute_precision == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model and tokenizer
        model = GLiClassModel.from_pretrained(self._model_name_or_path)
        model = model.to(device, dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Create pipeline
        self._pipeline = ZeroShotClassificationPipeline(
            model=model,
            tokenizer=tokenizer,
            classification_type=self._classification_type,
            device=device,
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
                Supported: threshold (float), classification_type (str).

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

        # Extract texts from all items (batch processing)
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults
        opts = options or {}
        effective_threshold = opts.get("threshold", self._threshold)

        # Run batch classification
        # GLiClass pipeline supports batch input via 'texts' parameter
        with torch.inference_mode():
            batch_results = self._pipeline(
                texts,
                labels,
                threshold=effective_threshold,
            )

        # Convert to our format
        all_classifications: list[list[Classification]] = []
        for item_results in batch_results:
            # Each item's results is a list of {label, score} dicts
            classifications: list[Classification] = []
            for result in item_results:
                classifications.append(
                    Classification(
                        label=result["label"],
                        score=float(result["score"]),
                    )
                )

            # Sort by score descending
            classifications.sort(key=lambda x: x["score"], reverse=True)

            all_classifications.append(classifications)

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=all_classifications,
        )

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
