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

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.responses import Classification

if TYPE_CHECKING:
    from gliclass import ZeroShotClassificationPipeline  # ty:ignore[unresolved-import]

    from sie_server.types.inputs import Item

_ERR_REQUIRES_LABELS = "Zero-shot classification requires labels parameter."
_ERR_INPUT_TOO_LONG = (
    "Input produced an empty tensor inside the gliclass pipeline; "
    "this typically indicates the input exceeds the model's max sequence length "
    "even after truncation. Reduce input length or split into chunks."
)

ClassificationType = Literal["single-label", "multi-label"]


class GLiClassAdapter(BaseAdapter):
    """Adapter for GLiClass zero-shot classification models.

    Uses the gliclass library's ZeroShotClassificationPipeline.
    Works with models like knowledgator/gliclass-base-v1.0.

    GLiClass performs classification in a single forward pass (not NLI-based),
    making it much faster than cross-encoder approaches.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_pipeline",),
    )

    def _check_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        classification_type: ClassificationType = "single-label",
        threshold: float = 0.0,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize GLiClass adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            classification_type: "single-label" for mutually exclusive classes,
                "multi-label" for multiple classes per text.
            threshold: Default server-side post-filter threshold (0-1). Defaults
                to 0.0 so all requested labels are returned with their scores.
                Callers can override per-request via ``options={"threshold": ...}``.
            max_seq_length: Maximum input sequence length in tokens. Used to bound
                tokenization inside the gliclass pipeline so inputs cannot exceed
                the model's position-embedding capacity.
            compute_precision: Precision for inference (float16, float32, bfloat16).
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._classification_type = classification_type
        self._threshold = threshold
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        self._pipeline: ZeroShotClassificationPipeline | None = None
        self._device: str | None = None

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

        # Bound the tokenizer's max length so any internal tokenization in the
        # gliclass library auto-truncates to the model's actual capacity.
        if self._max_seq_length is not None:
            tokenizer.model_max_length = self._max_seq_length

        # Create pipeline. Pass max_length explicitly so the pipeline's
        # ``tokenizer(..., truncation=True, max_length=self.max_length)`` calls
        # cap inputs at the model's position-embedding limit. Without this the
        # library defaults to 1024, which exceeds the 512-token capacity of the
        # current GLiClass models and causes argmax-on-empty-tensor crashes for
        # long inputs (see sie-test#88, sie-test#89).
        pipeline_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "classification_type": self._classification_type,
            "device": device,
        }
        if self._max_seq_length is not None:
            pipeline_kwargs["max_length"] = self._max_seq_length

        self._pipeline = ZeroShotClassificationPipeline(**pipeline_kwargs)

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

        Returns scores for *every* requested label (sorted by score descending).
        If callers pass ``options={"threshold": <float>}``, labels scoring below
        that threshold are filtered out server-side before returning.

        Args:
            items: List of items to classify (must have text).
            labels: Classification labels (e.g., ["positive", "negative", "neutral"]).
                Required for zero-shot classification.
            output_schema: Unused (included for interface compatibility).
            instruction: Unused (included for interface compatibility).
            options: Adapter options to override model config defaults.
                Supported: threshold (float), classification_type (str).

        Returns:
            ExtractOutput where ``classifications[i]`` is the list of
            ``Classification(label, score)`` for ``items[i]``, sorted by score
            descending.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack text, or if the
                input produced an empty tensor inside the gliclass pipeline.
        """
        self._check_loaded()
        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        # Extract texts from all items (batch processing)
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults. The threshold is applied
        # server-side as a post-filter so we always get all label scores from the
        # underlying pipeline regardless of caller preferences.
        opts = options or {}
        effective_threshold = float(opts.get("threshold", self._threshold))

        # Run batch classification.
        # - threshold=0.0: never let the gliclass library drop labels for us
        #   (in single-label mode the lib returns only argmax anyway, so we
        #   need return_hierarchical=True to recover all label scores).
        # - return_hierarchical=True with a flat ``labels`` list yields a list
        #   of ``{label: score}`` dicts with every requested label present.
        try:
            with torch.inference_mode():
                batch_results = self._pipeline(
                    texts,
                    labels,
                    threshold=0.0,
                    return_hierarchical=True,
                )
        except RuntimeError as exc:
            # The gliclass library calls torch.argmax on potentially-empty
            # tensors when inputs blow past the model's position-embedding
            # capacity. Surface this as a clear validation error rather than a
            # 500 INFERENCE_ERROR. Match on BOTH phrases to avoid catching
            # unrelated runtime errors.
            msg = str(exc)
            if "numel() == 0" in msg and "argmax" in msg:
                raise ValueError(_ERR_INPUT_TOO_LONG) from exc
            raise

        all_classifications: list[list[Classification]] = []
        for item_results in batch_results:
            # With return_hierarchical=True and a flat label list the library
            # returns a dict {label: score}. Anything else (e.g. None for an
            # empty input) yields no classifications rather than crashing.
            if isinstance(item_results, dict):
                pairs: list[tuple[str, float]] = [(str(k), float(v)) for k, v in item_results.items()]
            else:
                pairs = []

            classifications: list[Classification] = [Classification(label=label, score=score) for label, score in pairs]

            # Server-side post-filter when the caller explicitly requested one.
            if effective_threshold > 0.0:
                classifications = [c for c in classifications if c["score"] >= effective_threshold]

            # Sort by score descending
            classifications.sort(key=lambda x: x["score"], reverse=True)

            all_classifications.append(classifications)

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=all_classifications,
        )
