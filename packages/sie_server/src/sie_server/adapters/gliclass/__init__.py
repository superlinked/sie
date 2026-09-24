"""GLiClass zero-shot classification adapter.

Uses Knowledgator's GLiClass library for efficient zero-shot text classification.
GLiClass is inspired by GLiNER but optimized for classification tasks.
Up to 50x faster than cross-encoders with similar accuracy.

Performance note (Dec 2025):
    Benchmarked GLiClass library at 496 texts/sec vs NLI flash adapter at 494 texts/sec
    (100 texts, 5 labels). GLiClass is a single-pass architecture (not N×M expansion
    like NLI cross-encoders), so the gliclass library pipeline has minimal overhead.
    No separate "GLiClassFlashAdapter" is needed - the library is already efficient.

Request surface (all optional; a request that sets none of them runs exactly the
pipeline call used before these fields existed):

- ``instruction``: task description passed to the pipeline as ``prompt``.
- ``options.examples``: few-shot examples, ``[{"text": ..., "labels": [...]}]``.
- ``options.classification_type``: ``"single-label"`` or ``"multi-label"`` for
  this request, overriding the load-time default.
- ``options.label_groups``: named label groups, ``{"urgency": ["low", "high"],
  ...}``, used instead of ``labels``. Single-label scores are normalized within
  each group, and ``data`` holds one answer per group.

Usage counts each item's document tokens plus the instruction and example
texts encoded with it; label names are not counted.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.errors import InputTooLongError
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError
from sie_server.types.overflow_policy import DEFAULT_OVERFLOW_POLICY, OverflowPolicy
from sie_server.types.responses import Classification, ErrorCode

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
_CLASSIFICATION_TYPES: tuple[ClassificationType, ...] = ("single-label", "multi-label")
# gliclass joins a label group and its labels with this separator (the
# pipeline's default ``label_separator``); the model sees "urgency.high".
_LABEL_GROUP_SEPARATOR = "."
# ``ZeroShotClassificationPipeline.__call__`` default. The grouped path runs the
# model directly and mirrors the pipeline's sub-batching so padding matches.
_PIPELINE_BATCH_SIZE = 8
_MAX_EXAMPLES = 32
_EXAMPLE_KEYS = frozenset({"text", "labels"})
_DEFAULT_MULTI_LABEL_THRESHOLD = 0.5
# Free-form request text is fused into every item's input, so it is capped
# before any tokenization. The model window (512-1024 tokens) holds far less.
_MAX_INSTRUCTION_CHARS = 2048
_MAX_EXAMPLE_TEXT_CHARS = 2048
_MAX_CONTEXT_CHARS = 8192
# Used when the tokenizer's vocabulary cannot be inspected.
_DEFAULT_MAX_TOKEN_CHARS = 32
# Label names are natural language. Some vocabularies hold long whitespace or
# symbol-run tokens (hundreds of characters), but no label prompt averages more
# than this many characters per token.
_MAX_LABEL_CHARS_PER_TOKEN = 16
# Items whose estimated fit is this close to the window edge get an exact check
# on the fused encoding: tokenizing the document alone can differ by a token
# or two from tokenizing it next to the label prompt.
_FIT_MARGIN_TOKENS = 8
_ERR_ITEM_LABELS_TRUNCATED = (
    "The document pushes the labels out of the gliclass model's max sequence length. "
    "Shorten the document, or send options.overflow_policy='truncate_text'."
)


@dataclass(frozen=True)
class _RequestLayout:
    """Token layout shared by every item of a request, tokenized once.

    ``window`` is the number of non-special tokens the model keeps. The label
    prompt (markers, label names, separator) is ``label_tokens`` long and its
    last marker sits at ``last_marker``. ``context_tokens`` counts the
    billable instruction and example texts.
    """

    prompt_first: bool
    window: int
    label_tokens: int
    last_marker: int
    context_tokens: int

    def labels_fit(self, document_tokens: int) -> bool:
        """Whether every label marker survives truncation next to this document."""
        if self.prompt_first:
            return self.last_marker < self.window
        return document_tokens + self.last_marker < self.window


def _choice_confidence(probabilities: list[float]) -> float:
    """``1 - H(p) / log(k)``: 1.0 for a certain answer, 0.0 for a uniform one."""
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def _pipeline_context(prompt: str | None, examples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Pipeline keyword arguments for the fields a request actually set.

    Leaving unset fields out keeps a request without them on exactly the
    pipeline call the adapter has always made.
    """
    context: dict[str, Any] = {}
    if prompt is not None:
        context["prompt"] = prompt
    if examples is not None:
        context["examples"] = examples
    return context


def _longest_token_chars(tokenizer: Any) -> int:
    """Length in characters of the longest entry in the tokenizer's vocabulary."""
    try:
        return max(len(token) for token in tokenizer.get_vocab())
    except Exception:  # noqa: BLE001 -- fall back to a generous default
        return _DEFAULT_MAX_TOKEN_CHARS


def _string_list(value: Any) -> list[str] | None:
    """Return ``value`` as a list of strings, or None when it is not one."""
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast("list[str]", value)
    return None


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
        unload_fields=("_pipeline", "_pipelines", "_tokenizer"),
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
                "multi-label" for multiple classes per text. Callers can
                override it per request via
                ``options={"classification_type": ...}``.
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
        self._classification_type = self._validate_classification_type(classification_type)
        self._threshold = threshold
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

        self._pipeline: ZeroShotClassificationPipeline | None = None
        # One pipeline per classification type, sharing the model and
        # tokenizer; ``_pipeline`` is the load-time type's entry.
        self._pipelines: dict[ClassificationType, ZeroShotClassificationPipeline] | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._special_count: int = 0
        self._max_token_chars: int = _DEFAULT_MAX_TOKEN_CHARS
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
        # long inputs.
        pipeline_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": self._tokenizer,
            "device": device,
        }
        if self._max_seq_length is not None:
            pipeline_kwargs["max_length"] = self._max_seq_length

        self._special_count = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        self._max_token_chars = _longest_token_chars(self._tokenizer)
        self._pipelines = {
            classification_type: ZeroShotClassificationPipeline(
                **pipeline_kwargs, classification_type=classification_type
            )
            for classification_type in _CLASSIFICATION_TYPES
        }
        self._pipeline = self._pipelines[self._classification_type]

    def _extract_text(self, item: Item) -> str:
        if not item.text:
            msg = "Item must have text for classification"
            raise InvalidInputError(msg)
        return item.text

    def _apply_overflow_policy(
        self,
        texts: list[str],
        labels: list[str],
        policy: OverflowPolicy = DEFAULT_OVERFLOW_POLICY,
        *,
        prompt: str | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Enforce overflow_policy by pre-tokenizing text and label_prompt separately.

        At inference the gliclass pipeline tokenizes the fused string with
        ``add_special_tokens=True``, so the model sees
        ``observed = text_tokens + label_prompt_tokens + special_count``, where
        ``special_count`` is the BERT-style ``[CLS]``/``[SEP]`` wrap (2 for all
        current gliclass models). We recover the same total without running the
        model by tokenizing each part with ``add_special_tokens=False``. The
        label prompt includes the task prompt and few-shot examples when the
        request sets them, so ``truncate_text`` shortens only the document.

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

        context = _pipeline_context(prompt, examples)
        label_prompt = self._pipeline.pipe.prepare_input(text="", labels=labels, **context)  # ty:ignore[unresolved-attribute]
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
                Required unless ``options["label_groups"]`` is set.
            output_schema: Unused (included for interface compatibility).
            instruction: Optional task description, passed to the gliclass
                pipeline as its ``prompt``.
            options: Adapter options to override model config defaults.
                Supported: ``threshold`` (float), ``classification_type``
                ("single-label" or "multi-label"), ``examples`` (few-shot
                ``[{"text": str, "labels": [str, ...]}]``), ``label_groups``
                (``{group: [label, ...]}``, used instead of ``labels``), and
                ``overflow_policy``.

        Returns:
            ExtractOutput where ``classifications[i]`` is the list of
            ``Classification(label, score)`` for ``items[i]``, sorted by score
            descending. With ``label_groups``, labels read ``"group.label"``
            and ``data[i]`` maps each group to its answer: ``{"type":
            "choice", "choice", "probabilities", "confidence"}`` for
            single-label groups, ``{"labels", "probabilities"}`` for
            multi-label ones. Probabilities are never threshold-filtered.

        Raises:
            RuntimeError: If model not loaded.
            InvalidInputError: If labels are missing or options are malformed.
            InputTooLongError: If the labels do not fit in the model window.
            ValueError: If items lack text or the pipeline returns malformed
                scores.
        """
        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)

        opts = options or {}
        label_groups = self._validate_label_groups(opts.get("label_groups"))
        classification_type = self._validate_classification_type(
            opts.get("classification_type", self._classification_type)
        )
        if label_groups is None:
            normalized_labels = self._validate_labels(labels)
        elif labels:
            raise InvalidInputError("GLiClass accepts either labels or options.label_groups, not both")
        else:
            normalized_labels = self._flatten_label_groups(label_groups, classification_type)
        self._check_label_size(normalized_labels)
        prompt = self._validate_instruction(instruction)
        examples = self._validate_examples(opts.get("examples"), normalized_labels, label_groups)
        self._check_context_size(prompt, examples)

        # Extract texts from all items (batch processing)
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults. The threshold is applied
        # server-side as a post-filter so we always get all label scores from the
        # underlying pipeline regardless of caller preferences.
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold))

        overflow_policy = opts.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        texts = self._apply_overflow_policy(texts, normalized_labels, overflow_policy, prompt=prompt, examples=examples)
        layout = self._request_layout(normalized_labels, prompt, examples)
        fits = self._items_fit(texts, layout, normalized_labels, prompt, examples)
        input_token_counts = self._input_token_counts(texts, prompt, examples, layout, fits)
        errors = self._item_errors(fits)
        kept = [index for index, ok in enumerate(fits) if ok]
        kept_texts = [texts[index] for index in kept]

        if label_groups is not None:
            return self._extract_grouped(
                kept_texts,
                label_groups,
                normalized_labels,
                classification_type=classification_type,
                prompt=prompt,
                examples=examples,
                threshold=effective_threshold,
                input_token_counts=input_token_counts,
                kept=kept,
                errors=errors,
            )

        if not kept_texts:
            return ExtractOutput(
                entities=[[] for _ in items],
                classifications=[[] for _ in items],
                errors=errors,
                input_token_counts=input_token_counts,
            )

        context = _pipeline_context(prompt, examples)
        pipeline = self._select_pipeline(classification_type)

        # Run batch classification.
        # - threshold=0.0: never let the gliclass library drop labels for us
        #   (in single-label mode the lib returns only argmax anyway, so we
        #   need return_hierarchical=True to recover all label scores).
        # - return_hierarchical=True with a flat ``labels`` list yields a list
        #   of ``{label: score}`` dicts with every requested label present.
        try:
            with torch.inference_mode():
                batch_results = pipeline(
                    kept_texts,
                    normalized_labels,
                    threshold=0.0,
                    return_hierarchical=True,
                    **context,
                )
        except (RuntimeError, IndexError) as exc:
            # The gliclass library crashes inside the pipeline when inputs exceed
            # the model's position-embedding capacity, producing empty intermediate
            # tensors that downstream ops then operate on. Surface the known crash
            # signatures as InputTooLongError (validation) instead of leaking as
            # 500 INFERENCE_ERROR. Match must be specific to avoid swallowing
            # unrelated errors. Catalog of caught signatures:
            #   - RuntimeError "argmax(): ... numel() == 0"
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

        all_classifications: list[list[Classification]] = [[] for _ in items]
        for index, item_results in zip(kept, batch_results, strict=True):
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

            all_classifications[index] = classifications

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=all_classifications,
            errors=errors,
            input_token_counts=input_token_counts,
        )

    def _select_pipeline(self, classification_type: ClassificationType) -> ZeroShotClassificationPipeline:
        if classification_type == self._classification_type:
            if self._pipeline is None:
                raise RuntimeError(ERR_NOT_LOADED)
            return self._pipeline
        if self._pipelines is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._pipelines[classification_type]

    def _extract_grouped(
        self,
        texts: list[str],
        label_groups: list[tuple[str, list[str]]],
        flat_labels: list[str],
        *,
        classification_type: ClassificationType,
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        threshold: float,
        input_token_counts: list[int] | None,
        kept: list[int],
        errors: list[ExtractItemError | None] | None,
    ) -> ExtractOutput:
        """Score grouped labels and return one answer per group.

        All groups share one forward pass per text, exactly as the gliclass
        pipeline encodes a dict of labels ("group.label" after flattening).
        The pipeline's own single-label mode applies one softmax across every
        flattened label, which makes a group's scores depend on the other
        groups. Here each group gets its own softmax over the raw logits
        instead; multi-label scores are independent sigmoids either way.

        A single-label group answers like a choice question:
        ``{"type": "choice", "choice", "probabilities", "confidence"}`` with
        ``confidence = 1 - H(p) / log(k)``. A multi-label group answers
        ``{"labels", "probabilities"}``.
        """
        rows = (
            self._grouped_scores(texts, label_groups, flat_labels, classification_type, prompt, examples)
            if texts
            else []
        )
        item_count = len(errors) if errors is not None else len(texts)
        # Multi-label groups list the labels at or above the request threshold,
        # or at or above 0.5 (an even sigmoid) when no threshold is set.
        selection_threshold = threshold if threshold > 0.0 else _DEFAULT_MULTI_LABEL_THRESHOLD

        all_classifications: list[list[Classification]] = [[] for _ in range(item_count)]
        all_data: list[dict[str, Any]] = [{} for _ in range(item_count)]
        for index, row in zip(kept, rows, strict=True):
            position = 0
            answers: dict[str, dict[str, Any]] = {}
            classifications: list[Classification] = []
            for group, group_labels in label_groups:
                probabilities: dict[str, float] = {}
                for label in group_labels:
                    score = self._validate_score(row[position])
                    position += 1
                    probabilities[label] = score
                    classifications.append(Classification(label=f"{group}{_LABEL_GROUP_SEPARATOR}{label}", score=score))
                if classification_type == "single-label":
                    answers[group] = {
                        "type": "choice",
                        "choice": max(probabilities, key=probabilities.__getitem__),
                        "probabilities": probabilities,
                        "confidence": _choice_confidence(list(probabilities.values())),
                    }
                else:
                    answers[group] = {
                        "labels": [label for label, score in probabilities.items() if score >= selection_threshold],
                        "probabilities": probabilities,
                    }
            if threshold > 0.0:
                classifications = [c for c in classifications if c["score"] >= threshold]
            classifications.sort(key=lambda x: x["score"], reverse=True)
            all_classifications[index] = classifications
            all_data[index] = answers

        return ExtractOutput(
            entities=[[] for _ in range(item_count)],
            classifications=all_classifications,
            data=all_data,
            errors=errors,
            input_token_counts=input_token_counts,
        )

    def _grouped_scores(
        self,
        texts: list[str],
        label_groups: list[tuple[str, list[str]]],
        flat_labels: list[str],
        classification_type: ClassificationType,
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> list[list[float]]:
        """Run the model on flattened group labels and normalize per group."""
        if self._pipeline is None:
            raise RuntimeError(ERR_NOT_LOADED)
        pipe = self._pipeline.pipe  # ty:ignore[unresolved-attribute]
        model = pipe.model
        num_labels = len(flat_labels)
        forward_kwargs: dict[str, Any] = {}
        resolve_max_num_classes = getattr(pipe, "_resolve_max_num_classes", None)
        if resolve_max_num_classes is not None:
            forward_kwargs["max_num_classes"] = resolve_max_num_classes(flat_labels, True)

        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(texts), _PIPELINE_BATCH_SIZE):
                inputs = pipe.prepare_inputs(
                    texts[start : start + _PIPELINE_BATCH_SIZE],
                    flat_labels,
                    same_labels=True,
                    examples=examples,
                    prompt=prompt,
                )
                logits = model(**inputs, **forward_kwargs).logits
                if logits.shape[-1] < num_labels:
                    raise InputTooLongError(_ERR_INPUT_TOO_LONG)
                chunks.append(logits[:, :num_labels].float())

            logits = torch.cat(chunks)
            if classification_type == "multi-label":
                scores = torch.sigmoid(logits)
            else:
                scores = torch.empty_like(logits)
                start = 0
                for _, group_labels in label_groups:
                    end = start + len(group_labels)
                    scores[:, start:end] = torch.softmax(logits[:, start:end], dim=-1)
                    start = end
        return scores.cpu().tolist()

    @staticmethod
    def _validate_classification_type(value: object) -> ClassificationType:
        if value == "single-label":
            return "single-label"
        if value == "multi-label":
            return "multi-label"
        raise InvalidInputError("GLiClass classification_type must be 'single-label' or 'multi-label'")

    @staticmethod
    def _validate_instruction(instruction: object) -> str | None:
        if instruction is None:
            return None
        if not isinstance(instruction, str):
            raise InvalidInputError("GLiClass instruction must be a string")
        if len(instruction) > _MAX_INSTRUCTION_CHARS:
            raise InvalidInputError(f"GLiClass instruction must be at most {_MAX_INSTRUCTION_CHARS} characters")
        return instruction if instruction.strip() else None

    @classmethod
    def _validate_label_groups(cls, value: Any) -> list[tuple[str, list[str]]] | None:
        if value is None:
            return None
        if not isinstance(value, dict) or not value:
            raise InvalidInputError(
                "GLiClass label_groups must be a non-empty object mapping group names to label lists"
            )
        groups: list[tuple[str, list[str]]] = []
        for name, group_labels in value.items():
            if not isinstance(name, str) or not name.strip():
                raise InvalidInputError("GLiClass label_groups names must be non-empty strings")
            if not isinstance(group_labels, list) or not group_labels:
                raise InvalidInputError(f"GLiClass label_groups[{name!r}] must be a non-empty list of labels")
            groups.append((name.strip(), cls._validate_labels(group_labels)))
        if len({name for name, _ in groups}) != len(groups):
            raise InvalidInputError("GLiClass label_groups names must be unique")
        return groups

    @staticmethod
    def _flatten_label_groups(
        label_groups: list[tuple[str, list[str]]],
        classification_type: ClassificationType,
    ) -> list[str]:
        flat: list[str] = []
        for name, group_labels in label_groups:
            if classification_type == "single-label" and len(group_labels) < 2:
                raise InvalidInputError(f"GLiClass single-label group {name!r} needs at least two labels")
            flat.extend(f"{name}{_LABEL_GROUP_SEPARATOR}{label}" for label in group_labels)
        if len(flat) > MAX_EXTRACT_LABELS:
            raise InvalidInputError(f"GLiClass label_groups must contain at most {MAX_EXTRACT_LABELS} labels in total")
        if len(set(flat)) != len(flat):
            raise InvalidInputError(
                f"GLiClass label_groups repeat a label once group and label names are joined with "
                f"{_LABEL_GROUP_SEPARATOR!r}"
            )
        return flat

    @staticmethod
    def _validate_examples(
        value: Any,
        labels: list[str],
        label_groups: list[tuple[str, list[str]]] | None,
    ) -> list[dict[str, Any]] | None:
        """Normalize few-shot examples to the pipeline's ``{"text", "labels"}`` form.

        Example labels must come from the request's own label set. With
        ``label_groups`` they may be written as ``"group.label"`` strings or
        as an object mapping a group to one label or a list of labels.
        """
        if value is None:
            return None
        if not isinstance(value, list):
            raise InvalidInputError("GLiClass examples must be a list of {text, labels} objects")
        if not value:
            return None
        if len(value) > _MAX_EXAMPLES:
            raise InvalidInputError(f"GLiClass examples must contain at most {_MAX_EXAMPLES} entries")
        allowed = set(labels)
        groups = dict(label_groups) if label_groups is not None else None
        normalized: list[dict[str, Any]] = []
        for index, entry in enumerate(value):
            where = f"GLiClass examples[{index}]"
            if not isinstance(entry, dict) or set(entry) != _EXAMPLE_KEYS:
                raise InvalidInputError(f"{where} must be an object with exactly 'text' and 'labels'")
            example = cast("dict[str, Any]", entry)
            text = example["text"]
            if not isinstance(text, str) or not text.strip():
                raise InvalidInputError(f"{where}.text must be a non-empty string")
            if len(text) > _MAX_EXAMPLE_TEXT_CHARS:
                raise InvalidInputError(f"{where}.text must be at most {_MAX_EXAMPLE_TEXT_CHARS} characters")
            raw_labels = example["labels"]
            example_labels: list[str] = []
            # An example can name each requested label at most once; a longer
            # list is refused before it is walked.
            if isinstance(raw_labels, (dict, list)) and len(raw_labels) > len(labels):
                raise InvalidInputError(f"{where}.labels names more labels than the request has")
            if isinstance(raw_labels, dict):
                if groups is None:
                    raise InvalidInputError(f"{where}.labels can be an object only when label_groups is set")
                for group, chosen in cast("dict[Any, Any]", raw_labels).items():
                    if not isinstance(group, str) or group.strip() not in groups:
                        raise InvalidInputError(f"{where}.labels names an unknown group {group!r}")
                    chosen_labels = _string_list([chosen] if isinstance(chosen, str) else chosen)
                    if chosen_labels is None:
                        raise InvalidInputError(f"{where}.labels[{group!r}] must be a label or a list of labels")
                    if len(chosen_labels) > len(groups[group.strip()]):
                        raise InvalidInputError(f"{where}.labels[{group!r}] names more labels than the group has")
                    for label in chosen_labels:
                        if label.strip() not in groups[group.strip()]:
                            raise InvalidInputError(f"{where} uses {label!r}, which is not a label of group {group!r}")
                        example_labels.append(f"{group.strip()}{_LABEL_GROUP_SEPARATOR}{label.strip()}")
            elif isinstance(raw_labels, list):
                flat_labels = _string_list(raw_labels)
                if flat_labels is None:
                    raise InvalidInputError(f"{where}.labels must be a list of strings")
                example_labels = [label.strip() for label in flat_labels]
                unknown = [label for label in example_labels if label not in allowed]
                if unknown:
                    raise InvalidInputError(f"{where} uses labels outside the requested label set: {unknown}")
            else:
                raise InvalidInputError(f"{where}.labels must be a list of labels")
            normalized.append({"text": text, "labels": list(dict.fromkeys(example_labels))})
        return normalized

    @staticmethod
    def _validate_labels(labels: list[str] | None) -> list[str]:
        if not labels:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise InvalidInputError("GLiClass labels must be non-empty strings")
        normalized = [label.strip() for label in labels]
        if len(set(normalized)) != len(normalized):
            raise InvalidInputError("GLiClass labels must be unique")
        return normalized

    @staticmethod
    def _validate_threshold(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidInputError("GLiClass threshold must be a finite number between 0 and 1")
        try:
            threshold = float(value)
        except OverflowError as exc:
            raise InvalidInputError("GLiClass threshold must be a finite number between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise InvalidInputError("GLiClass threshold must be a finite number between 0 and 1")
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

    def _request_layout(
        self,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> _RequestLayout | None:
        """Tokenize the parts every item shares, once per request.

        Returns None when the model does not put label markers in its input
        (only uni-encoder GLiClass models do), so there is nothing to check.

        Raises:
            InputTooLongError: If the label prompt alone overflows the window.
            InvalidInputError: If the instruction and examples leave no room
                for the document.
        """
        pipe = getattr(self._pipeline, "pipe", None)
        config = getattr(getattr(pipe, "model", None), "config", None)
        tokenizer = self._tokenizer
        if (
            pipe is None
            or config is None
            or tokenizer is None
            or getattr(config, "architecture_type", None) != "uni-encoder"
        ):
            return None

        def count(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        label_prompt = "".join(f"{pipe.label_token}{label}" for label in labels) + pipe.sep_token
        label_ids = tokenizer(label_prompt, add_special_tokens=False)["input_ids"]
        markers = [position for position, token in enumerate(label_ids) if token == config.class_token_index]
        if len(markers) < len(labels):
            return None
        window = pipe.max_length - self._special_count
        if markers[-1] >= window:
            raise InputTooLongError(_ERR_INPUT_TOO_LONG)
        prompt_tokens = count(prompt) if prompt else 0
        example_text_tokens = sum(count(example["text"]) for example in examples or [])
        if prompt or examples:
            examples_tokens = count(pipe._format_examples_for_input(examples)) if examples else 0
            if len(label_ids) + prompt_tokens + examples_tokens >= window:
                raise InvalidInputError(
                    "GLiClass instruction, examples and labels leave no room for the document in the "
                    f"model's {window}-token window; shorten the instruction or send fewer examples"
                )
        return _RequestLayout(
            prompt_first=bool(getattr(config, "prompt_first", False)),
            window=window,
            label_tokens=len(label_ids),
            last_marker=markers[-1],
            context_tokens=prompt_tokens + example_text_tokens,
        )

    def _items_fit(
        self,
        texts: list[str],
        layout: _RequestLayout | None,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> list[bool]:
        """Whether each item's label markers survive truncation.

        Documents are tokenized once, on their own. Tokenizing a document next
        to the label prompt can differ from that by a token or two (a trailing
        space before a marker, for example), so items within a few tokens of
        the window edge are checked exactly on their fused encoding.
        """
        if layout is None or layout.prompt_first or self._tokenizer is None:
            return [True] * len(texts)
        encoded = self._tokenizer(texts, add_special_tokens=False, truncation=True, max_length=layout.window)
        fits: list[bool] = []
        for text, ids in zip(texts, encoded["input_ids"], strict=True):
            estimate = len(ids) + layout.last_marker
            if estimate < layout.window - _FIT_MARGIN_TOKENS:
                fits.append(True)
            elif estimate >= layout.window + _FIT_MARGIN_TOKENS:
                fits.append(False)
            else:
                fits.append(self._labels_survive(text, list(ids), labels, prompt, examples))
        return fits

    def _labels_survive(
        self,
        text: str,
        text_ids: list[int],
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> bool:
        """Exact check: count the label markers left in the truncated fused input."""
        pipe = getattr(self._pipeline, "pipe", None)
        tokenizer = self._tokenizer
        if pipe is None or tokenizer is None:
            return False
        marker = pipe.model.config.class_token_index
        fused = pipe.prepare_input(text, labels, examples, prompt)
        ids = tokenizer(fused, truncation=True, max_length=pipe.max_length)["input_ids"]
        # Markers written in the document itself come first and are not labels.
        return ids.count(marker) >= text_ids.count(marker) + len(labels)

    @staticmethod
    def _item_errors(fits: list[bool]) -> list[ExtractItemError | None] | None:
        """Per-item INPUT_TOO_LONG for items whose labels would be cut off.

        Reporting these per item keeps one oversized document from failing
        every request batched with it.
        """
        if all(fits):
            return None
        return [
            None if ok else ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=_ERR_ITEM_LABELS_TRUNCATED)
            for ok in fits
        ]

    def _input_token_counts(
        self,
        texts: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        layout: _RequestLayout | None,
        fits: list[bool],
    ) -> list[int] | None:
        """Billable input tokens per item.

        Each item is billed for its document and for the free-form request
        text encoded with it: the instruction and the few-shot example texts.
        Label names (including the labels attached to examples) and the
        pipeline's marker tokens are not billed. With an instruction or
        examples, an item's total is capped at the model window minus the
        label prompt, the most free-form text it can encode, unless the
        document count alone is already higher. Items refused because their
        labels would be cut off are billed nothing.
        """
        counts = self._doc_input_token_counts(texts)
        if counts is None:
            return None
        if prompt or examples:
            if layout is not None:
                context_tokens = layout.context_tokens
                cap = layout.window + self._special_count - layout.label_tokens
            else:
                context_tokens = self._context_token_count(prompt, examples)
                if context_tokens is None:
                    return None
                cap = self._max_seq_length
            counts = [count if cap is None else max(count, min(count + context_tokens, cap)) for count in counts]
            if cap is None:
                counts = [count + context_tokens for count in counts]
        return [count if ok else 0 for count, ok in zip(counts, fits, strict=True)]

    def _context_token_count(self, prompt: str | None, examples: list[dict[str, Any]] | None) -> int | None:
        if self._tokenizer is None:
            return None
        parts = ([prompt] if prompt else []) + [example["text"] for example in examples or []]
        try:
            return sum(len(ids) for ids in self._tokenizer(parts, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001 -- metering must not fail classification
            return None

    def _check_label_size(self, labels: list[str]) -> None:
        """Refuse label sets whose text cannot fit the model window, before tokenizing.

        A token covers at most ``_max_token_chars`` characters (the longest
        vocabulary entry), and label text averages far fewer: at most
        ``_MAX_LABEL_CHARS_PER_TOKEN``. A label prompt longer than that many
        characters per token of the window needs more tokens than the window
        holds, so it can never be scored correctly, and only tokenizing it
        would take time proportional to its size.
        """
        window = self._max_seq_length or getattr(getattr(self._pipeline, "pipe", None), "max_length", None)
        if not isinstance(window, int):
            return
        limit = min(self._max_token_chars, _MAX_LABEL_CHARS_PER_TOKEN) * window
        total = sum(len(label) for label in labels)
        if total > limit:
            raise InvalidInputError(
                f"GLiClass labels total {total} characters; at most {limit} can fit in the model's "
                f"{window}-token window"
            )

    @staticmethod
    def _check_context_size(prompt: str | None, examples: list[dict[str, Any]] | None) -> None:
        """Bound the free-form text fused into every item, before any tokenization."""
        size = len(prompt or "")
        for example in examples or []:
            size += len(example["text"]) + sum(len(label) for label in example["labels"])
        if size > _MAX_CONTEXT_CHARS:
            raise InvalidInputError(
                f"GLiClass instruction and examples must total at most {_MAX_CONTEXT_CHARS} characters"
            )

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
