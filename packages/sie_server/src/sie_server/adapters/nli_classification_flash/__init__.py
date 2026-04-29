from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import torch.nn.functional as F

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item
from sie_server.types.responses import Classification

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_REQUIRES_LABELS = "Zero-shot classification requires labels parameter."


class NLIClassificationFlashAdapter(FlashBaseAdapter):
    """Native NLI-based zero-shot classification adapter.

    Uses direct model inference instead of transformers pipeline for better
    throughput. Compatible with MoritzLaurer's deberta-v3-zeroshot models.
    """

    fallback_adapter_path: ClassVar[str | None] = None

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer", "_dtype"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        hypothesis_template: str = "This text is about {}.",
        multi_label: bool = False,
        max_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize native NLI classification adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            hypothesis_template: Template for converting labels to hypotheses.
                Must contain {} placeholder for the label.
            multi_label: If True, use sigmoid for independent label scores.
                If False, use softmax for mutually exclusive labels.
            max_length: Maximum sequence length for tokenization.
            compute_precision: Precision for inference (float16, bfloat16, float32).
            **kwargs: Additional arguments (ignored for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._hypothesis_template = hypothesis_template
        self._multi_label = multi_label
        self._max_length = max_length
        self._compute_precision = compute_precision

        # Loaded state
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

        # NLI label indices (standard for most NLI models)
        # entailment=0 or 2 depending on model, we'll detect at load time
        self._entailment_idx: int = 0

    def load(self, device: str) -> None:
        """Load model weights onto the specified device.

        Args:
            device: Target device (cuda:0, cuda:1, cpu, mps).
        """
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()

        logger.info(
            "Loading %s (dtype=%s, device=%s)",
            self._model_name_or_path,
            self._dtype,
            device,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load model (use 'dtype' not deprecated 'torch_dtype')
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
        )
        self._model = self._model.to(dtype=self._dtype)
        self._model.to(device)
        self._model.eval()

        # Detect entailment index from model config
        # MoritzLaurer models use: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        # Some models use: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
        id2label = getattr(self._model.config, "id2label", {})
        for idx, label in id2label.items():
            if label.lower() == "entailment":
                self._entailment_idx = int(idx)
                break

        logger.info(
            "Loaded: hidden=%d, num_labels=%d, entailment_idx=%d",
            self._model.config.hidden_size,
            self._model.config.num_labels,
            self._entailment_idx,
        )

        # Clamp configured max_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_length,
        )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype based on device and precision setting."""
        if self._device == "cpu":
            return torch.float32
        return super()._resolve_dtype()

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if not item.text:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="NLIClassificationFlashAdapter"))
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
        """Classify texts with zero-shot labels using native inference.

        Args:
            items: List of items to classify (must have text).
            labels: Classification labels (e.g., ["positive", "negative", "neutral"]).
            output_schema: Unused (included for interface compatibility).
            instruction: Unused (included for interface compatibility).
            options: Adapter options to override model config defaults.
                Supported: hypothesis_template (str), multi_label (bool).

        Returns:
            List of dicts, one per item, each containing:
                - "classifications": List of {label, score} sorted by score descending
                - "entities": Empty list
                - "data": Empty dict
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        # Get options with fallback to model defaults
        opts = options or {}
        effective_template = opts.get("hypothesis_template", self._hypothesis_template)
        effective_multi_label = opts.get("multi_label", self._multi_label)

        # Extract texts
        texts = [self._extract_text(item) for item in items]
        n_texts = len(texts)
        n_labels = len(labels)

        # Create hypotheses for each label
        hypotheses = [effective_template.format(label) for label in labels]

        # Expand to all (text, hypothesis) pairs
        # Shape: n_texts * n_labels pairs
        all_texts = []
        all_hypotheses = []
        for text in texts:
            for hypothesis in hypotheses:
                all_texts.append(text)
                all_hypotheses.append(hypothesis)

        # Batch tokenize all pairs
        encodings = self._tokenizer(
            all_texts,
            all_hypotheses,
            max_length=self._max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encodings = {k: v.to(self._device) for k, v in encodings.items()}

        # Single forward pass for all pairs
        with torch.inference_mode():
            outputs = self._model(**encodings)
            logits = outputs.logits  # [n_texts * n_labels, num_classes]

            # Extract entailment scores
            entailment_logits = logits[:, self._entailment_idx]  # [n_texts * n_labels]

            # Reshape to [n_texts, n_labels]
            entailment_logits = entailment_logits.view(n_texts, n_labels)

            # Normalize scores
            if effective_multi_label:
                # Independent scores per label
                scores = torch.sigmoid(entailment_logits)
            else:
                # Mutually exclusive labels
                scores = F.softmax(entailment_logits, dim=-1)

            scores = scores.cpu().tolist()

        # Convert to output format
        all_classifications: list[list[Classification]] = []
        for i in range(n_texts):
            classifications: list[Classification] = []
            for j, label in enumerate(labels):
                classifications.append(
                    Classification(
                        label=label,
                        score=float(scores[i][j]),
                    )
                )

            # Sort by score descending
            classifications.sort(key=lambda x: x["score"], reverse=True)

            all_classifications.append(classifications)

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=all_classifications,
        )
