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

import math
import re
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.errors import InputTooLongError
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.overflow_policy import DEFAULT_OVERFLOW_POLICY, OverflowPolicy
from sie_server.types.responses import Classification

if TYPE_CHECKING:
    from gliclass import ZeroShotClassificationPipeline  # ty:ignore[unresolved-import]
    from transformers import PreTrainedTokenizerBase  # ty:ignore[unresolved-import]

    from sie_server.types.inputs import Item

_ERR_REQUIRES_LABELS = "Zero-shot classification requires labels parameter."
_ERR_INPUT_TOO_LONG = (
    "Input overflowed the gliclass model's max sequence length (the label/text "
    "window could not fit after truncation). Reduce the number of labels or the "
    "input length, or split into chunks."
)
# Captures the index and size from the torch IndexError shape "index N is out of
# bounds for dimension D with size M". The dispatch below only treats it as an
# overflow when N == M (the off-by-one / exhausted-dimension shape); a generic
# out-of-range bug (N > M) keeps propagating. Anchored to the "out of bounds for
# dimension" shape so an unrelated IndexError such as "list index out of range"
# never matches at all.
_INDEX_OOB_RE = re.compile(r"index (\d+) is out of bounds for dimension \d+ with size (\d+)")
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
        unload_fields=("_pipeline", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        classification_type: ClassificationType = "single-label",
        threshold: float = 0.0,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
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
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._classification_type = classification_type
        self._threshold = threshold
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

        self._pipeline: ZeroShotClassificationPipeline | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._special_count: int = 0
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
        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision
        model = GLiClassModel.from_pretrained(self._model_name_or_path, **shared_kwargs)
        model = model.to(device, dtype=torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path, **shared_kwargs)

        # Bound the tokenizer's max length so any internal tokenization in the
        # gliclass library auto-truncates to the model's actual capacity.
        if self._max_seq_length is not None:
            self._tokenizer.model_max_length = self._max_seq_length

        # Create pipeline. Pass max_length explicitly so the pipeline's
        # ``tokenizer(..., truncation=True, max_length=self.max_length)`` calls
        # cap inputs at the model's position-embedding limit. Without this the
        # library defaults to 1024, which exceeds the 512-token capacity of the
        # current GLiClass models and causes argmax-on-empty-tensor crashes for
        # long inputs (see sie-test#88, sie-test#89).
        pipeline_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": self._tokenizer,
            "classification_type": self._classification_type,
            "device": device,
        }
        if self._max_seq_length is not None:
            pipeline_kwargs["max_length"] = self._max_seq_length

        self._special_count = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        self._pipeline = ZeroShotClassificationPipeline(**pipeline_kwargs)

    def _extract_text(self, item: Item) -> str:
        if not item.text:
            msg = "Item must have text for classification"
            raise ValueError(msg)
        return item.text

    def _apply_overflow_policy(
        self,
        texts: list[str],
        labels: list[str],
        policy: OverflowPolicy = DEFAULT_OVERFLOW_POLICY,
    ) -> list[str]:
        """Enforce overflow_policy by pre-tokenizing text and label_prompt separately.

        At inference the gliclass pipeline tokenizes the fused string with
        ``add_special_tokens=True``, so the model sees
        ``observed = text_tokens + label_prompt_tokens + special_count``, where
        ``special_count`` is the BERT-style ``[CLS]``/``[SEP]`` wrap (2 for all
        current gliclass models). We recover the same total without running the
        model by tokenizing each part with ``add_special_tokens=False``.

        On overflow:
        - ``default`` returns texts unchanged (upstream as-is — may crash inside
          the pipeline; the ``c0ce823c`` ``try/except`` in ``extract`` is the
          defense-in-depth backstop).
        - ``error`` raises ``InputTooLongError`` (whole batch fails, no partial
          responses).
        - ``truncate_text`` slices text to
          ``budget = max_sequence_length - label_prompt_tokens - special_count``.

        Under ``truncate_text`` and ``error``, ``label_prompt`` alone exceeding
        the cap always raises.
        """
        if policy == "default":
            return texts

        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if self._max_seq_length is None:
            raise RuntimeError(ERR_NOT_LOADED)

        label_prompt = self._pipeline.pipe.prepare_input(text="", labels=labels)  # ty:ignore[unresolved-attribute]
        label_prompt_tokens = len(self._tokenizer(label_prompt, add_special_tokens=False)["input_ids"])
        overhead = label_prompt_tokens + self._special_count
        budget = self._max_seq_length - overhead

        if budget <= 0:
            raise InputTooLongError(
                f"label_prompt ({label_prompt_tokens} tokens) + special ({self._special_count}) "
                f"exceeds max_sequence_length ({self._max_seq_length}); reduce the number or length of labels"
            )

        new_texts: list[str] = []
        for i, text in enumerate(texts):
            text_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
            text_tokens = len(text_ids)
            observed = text_tokens + overhead
            if observed <= self._max_seq_length:
                new_texts.append(text)
                continue
            if policy == "error":
                raise InputTooLongError(
                    f"items[{i}] observed_tokens={observed} exceeds max_sequence_length ({self._max_seq_length}) "
                    f"(text={text_tokens}, label_prompt={label_prompt_tokens}, special={self._special_count})"
                )
            new_texts.append(self._tokenizer.decode(text_ids[:budget], skip_special_tokens=True))
        return new_texts

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
        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)

        normalized_labels = self._validate_labels(labels)

        # Extract texts from all items (batch processing)
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults. The threshold is applied
        # server-side as a post-filter so we always get all label scores from the
        # underlying pipeline regardless of caller preferences.
        opts = options or {}
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold))

        overflow_policy = opts.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        texts = self._apply_overflow_policy(texts, normalized_labels, overflow_policy)
        input_token_counts = self._doc_input_token_counts(texts)

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
                    normalized_labels,
                    threshold=0.0,
                    return_hierarchical=True,
                )
        except (RuntimeError, IndexError) as exc:
            # The gliclass library crashes inside the pipeline when inputs exceed
            # the model's position-embedding capacity, producing empty intermediate
            # tensors that downstream ops then operate on. Surface the known crash
            # signatures as InputTooLongError (validation) instead of leaking as
            # 500 INFERENCE_ERROR. Match must be specific to avoid swallowing
            # unrelated errors. Catalog of caught signatures:
            #   - RuntimeError "argmax(): ... numel() == 0"  (sie-test#89 / #848)
            #     torch.argmax on an empty tensor inside the classification head.
            #   - IndexError  "index N is out of bounds for dimension D with size N"
            #     (index == size: off-by-one / exhausted dimension). Covers both the
            #     empty-tensor case (#860, index 0 / size 0) and the label-window
            #     overflow (#1434, e.g. index 79 / size 79): when too many labels
            #     overflow the shared 512-token window only some survive truncation
            #     and the single-label hierarchical decode then indexes exactly one
            #     past the shrunk label window.
            msg = str(exc)
            if isinstance(exc, RuntimeError) and "numel() == 0" in msg and "argmax" in msg:
                raise InputTooLongError(_ERR_INPUT_TOO_LONG) from exc
            oob = _INDEX_OOB_RE.search(msg)
            if isinstance(exc, IndexError) and oob is not None and oob.group(1) == oob.group(2):
                # Off-by-one: the gliclass single-label hierarchical decode indexes
                # exactly one past an exhausted dimension (index == size) when the
                # label/text window overflows max_sequence_length. Covers both the
                # empty-tensor case (#860, index 0 / size 0) and the label-window
                # overflow (#1434, e.g. index 79 / size 79). A generic out-of-range
                # bug (index > size, e.g. index 5 / size 3) is NOT this and must keep
                # propagating via the bare ``raise`` below.
                raise InputTooLongError(_ERR_INPUT_TOO_LONG) from exc
            raise

        all_classifications: list[list[Classification]] = []
        for item_results in batch_results:
            # With return_hierarchical=True and a flat label list the library
            # returns a dict {label: score}. Anything else (e.g. None for an
            # empty input) yields no classifications rather than crashing.
            if isinstance(item_results, dict):
                if set(item_results) != set(normalized_labels):
                    raise ValueError("GLiClass returned classifications outside the requested label set")
                pairs = [(label, self._validate_score(item_results[label])) for label in normalized_labels]
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
            input_token_counts=input_token_counts,
        )

    @staticmethod
    def _validate_labels(labels: list[str] | None) -> list[str]:
        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("GLiClass labels must be non-empty strings")
        normalized = [label.strip() for label in labels]
        if len(set(normalized)) != len(normalized):
            raise ValueError("GLiClass labels must be unique")
        return normalized

    @staticmethod
    def _validate_threshold(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("GLiClass threshold must be a finite number between 0 and 1")
        try:
            threshold = float(value)
        except OverflowError as exc:
            raise ValueError("GLiClass threshold must be a finite number between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("GLiClass threshold must be a finite number between 0 and 1")
        return threshold

    @staticmethod
    def _validate_score(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("GLiClass returned an invalid classification score")
        try:
            score = float(value)
        except OverflowError as exc:
            raise ValueError("GLiClass returned an invalid classification score") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("GLiClass returned an invalid classification score")
        return score

    def _doc_input_token_counts(self, texts: list[str]) -> list[int] | None:
        """Count document-only model-tokenizer input units for billing.

        GLiClass fuses the request's label schema with every document. The
        label prompt is reusable request schema rather than billed content, so
        this mirrors the GLiNER family contract and counts each post-policy
        document with the model tokenizer, including its normal special tokens.
        """
        if self._tokenizer is None:
            return None
        try:
            encoded = self._tokenizer(
                texts,
                add_special_tokens=True,
                truncation=self._max_seq_length is not None,
                max_length=self._max_seq_length,
            )
            counts = [len(input_ids) for input_ids in encoded["input_ids"]]
        except Exception:  # noqa: BLE001 -- metering must not fail classification
            return None
        return counts if len(counts) == len(texts) else None
