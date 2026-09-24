"""GLiClass zero-shot classification adapter.

Uses Knowledgator's GLiClass library for efficient zero-shot text classification.
GLiClass is inspired by GLiNER but optimized for classification tasks.
Up to 50x faster than cross-encoders with similar accuracy.

Performance note (Dec 2025):
    Benchmarked GLiClass library at 496 texts/sec vs NLI flash adapter at 494 texts/sec
    (100 texts, 5 labels). GLiClass is a single-pass architecture (not N×M expansion
    like NLI cross-encoders), so the gliclass library pipeline has minimal overhead.
    No separate "GLiClassFlashAdapter" is needed - the library is already efficient.

Scoring runs the gliclass model directly with the library's own prompt assembly
and tokenization, in the pipeline's sub-batches of eight rows, and applies the
pipeline's softmax or sigmoid. The scores equal the pipeline's. The adapter skips
the pipeline's progress bar and its per-label reads from the device, and it
tokenizes each document once for its length checks and metering.

Request surface (all optional; a request that sets none of them scores exactly
like the pipeline call used before these fields existed):

- ``instruction``: task description passed to the model as the pipeline's ``prompt``.
- ``options.examples``: few-shot examples, ``[{"text": ..., "labels": [...]}]``.
- ``options.classification_type``: ``"single-label"`` or ``"multi-label"`` for
  this request, overriding the load-time default.
- ``options.label_groups``: named label groups, ``{"urgency": ["low", "high"],
  ...}``, used instead of ``labels``. ``data`` holds one answer per group.
- ``options.group_encoding``: how label groups are encoded. ``"separate"`` (the
  default) encodes the document once per group, with only that group's labels.
  Each group scores like a request whose ``labels`` are that group's labels.
  The rows of a request share forward passes, and padding them together can
  move fp16 probabilities by a few thousandths. ``"joint"`` encodes every
  group's labels, as ``group.label``, next to one copy of the document, and
  normalizes single-label scores within each group.

Usage counts, for every encoded row, the document tokens plus the instruction
and example texts encoded with it. Label names are not counted. A separate-group
request encodes the document once per group, so it counts it once per group.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

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
GroupEncoding = Literal["separate", "joint"]
_DEFAULT_GROUP_ENCODING: GroupEncoding = "separate"
# gliclass joins a label group and its labels with this separator (the
# pipeline's default ``label_separator``); the model sees "urgency.high".
_LABEL_GROUP_SEPARATOR = "."
# ``ZeroShotClassificationPipeline.__call__`` default: rows per forward pass.
# The flat path keeps it so that items share padding exactly as before.
_PIPELINE_BATCH_SIZE = 8
# Separate group encoding runs one row of up to the model window per (item,
# group). An item may occupy at most this many row tokens: 64 groups at a
# 512-token window, 32 at 1024.
_MAX_ITEM_ROW_TOKENS = 32_768
_MAX_LABEL_GROUPS = 64
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
    billable instruction and example texts. ``overhead_tokens`` is everything
    a row encodes besides its document and special tokens: the label prompt,
    the instruction and the formatted examples.
    """

    prompt_first: bool
    window: int
    label_tokens: int
    last_marker: int
    context_tokens: int
    overhead_tokens: int

    def labels_fit(self, document_tokens: int) -> bool:
        """Whether every label marker survives truncation next to this document."""
        if self.prompt_first:
            return self.last_marker < self.window
        return document_tokens + self.last_marker < self.window


@dataclass
class _Context:
    """One (labels, instruction, examples) context of a request, tokenized once.

    ``parts`` holds the token ids of the label prompt, the instruction, each
    example text and the formatted examples; it stays empty for models
    without in-sequence label markers. ``overflow_tokens`` is the length of
    the pipeline input with an empty document. The layout is evaluated from
    ``parts`` when a request first needs it, so errors keep their order.
    """

    parts: list[list[int]] | None = None
    overflow_tokens: int | None = None
    layout: _RequestLayout | None = None
    has_layout: bool = False


class _RequestTokens:
    """What one request tokenizes, each distinct text at most once.

    ``full`` keeps every token of a document (the overflow policies need them
    all). ``cut`` returns ids truncated the way the tokenizer truncates to a
    fixed length, sliced from ``full`` when a text has already been tokenized
    in full. ``contexts`` holds the request's label contexts. Nothing is kept
    across requests, so one request's timing cannot reveal what another sent.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._full: dict[str, list[int]] = {}
        self._cut: dict[str, list[int]] = {}
        self._cut_length: int | None = None
        self.contexts: dict[tuple[Any, ...], _Context] = {}

    def full(self, texts: list[str]) -> list[list[int]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._full]
        if missing:
            encoded = self._tokenizer(missing, add_special_tokens=False)["input_ids"]
            self._full.update(zip(missing, encoded, strict=True))
        return [self._full[text] for text in texts]

    def cut(self, texts: list[str], length: int) -> list[list[int]]:
        if length != self._cut_length:
            self._cut = {}
            self._cut_length = length
        missing = [text for text in dict.fromkeys(texts) if text not in self._cut and text not in self._full]
        if missing:
            encoded = self._tokenizer(missing, add_special_tokens=False, truncation=True, max_length=length)
            self._cut.update(zip(missing, encoded["input_ids"], strict=True))
        left = getattr(self._tokenizer, "truncation_side", "right") == "left"
        ids: list[list[int]] = []
        for text in texts:
            cached = self._cut.get(text)
            if cached is None:
                full = self._full[text]
                cached = full[len(full) - length :] if left and len(full) > length else full[:length]
            ids.append(cached)
        return ids


def _choice_confidence(probabilities: list[float]) -> float:
    """``1 - H(p) / log(k)``: 1.0 for a certain answer, 0.0 for a uniform one."""
    if len(probabilities) < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def _pipeline_context(prompt: str | None, examples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Pipeline keyword arguments for the fields a request actually set.

    Leaving unset fields out keeps a request without them on exactly the
    pipeline input the adapter has always built.
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


@contextmanager
def _overflow_errors_as_input_too_long() -> Iterator[None]:
    """Report the known gliclass overflow crash signatures as ``InputTooLongError``.

    Inputs past the model's position capacity used to crash inside the
    gliclass pipeline with empty intermediate tensors. The match is specific,
    so unrelated errors keep propagating. Caught signatures:

    - RuntimeError "argmax(): ... numel() == 0": argmax on an empty tensor.
    - IndexError "index N is out of bounds for dimension D with size N"
      (index == size: an exhausted dimension), both the empty-tensor case
      (index 0 / size 0) and a label window that shrank under truncation
      (e.g. index 79 / size 79). A generic out-of-range bug (index > size,
      e.g. index 5 / size 3) is not this and propagates.
    """
    try:
        yield
    except (RuntimeError, IndexError) as exc:
        msg = str(exc)
        if isinstance(exc, RuntimeError) and "numel() == 0" in msg and "argmax" in msg:
            raise InputTooLongError(_ERR_INPUT_TOO_LONG) from exc
        oob = _INDEX_OOB_RE.search(msg)
        if isinstance(exc, IndexError) and oob is not None and oob.group(1) == oob.group(2):
            raise InputTooLongError(_ERR_INPUT_TOO_LONG) from exc
        raise


def _restore_modernbert_rope_fields(encoder_config: Any) -> bool:
    """Carry a transformers-5 ModernBERT RoPE config over to transformers 4.

    transformers 5 saves ModernBERT RoPE bases per layer type in
    ``rope_parameters`` and no longer writes ``global_rope_theta`` or
    ``local_rope_theta``. transformers 4 reads only the latter, so a checkpoint
    saved with 5 would silently run its sliding-window layers with the 4.x
    default base of 10000. Returns True when the config was changed.
    """
    rope_parameters = getattr(encoder_config, "rope_parameters", None)
    if (
        getattr(encoder_config, "model_type", None) != "modernbert"
        or not isinstance(rope_parameters, dict)
        or not hasattr(encoder_config, "local_rope_theta")  # transformers 5 has no legacy fields
    ):
        return False
    fields = {"full_attention": "global_rope_theta", "sliding_attention": "local_rope_theta"}
    if not set(rope_parameters) <= set(fields):
        raise ValueError(
            f"ModernBERT rope_parameters keys {sorted(rope_parameters)} are not supported by transformers 4"
        )
    for layer_type, field in fields.items():
        params = rope_parameters.get(layer_type)
        if params is None:
            continue
        if params.get("rope_type", "default") != "default" or "rope_theta" not in params:
            raise ValueError(f"ModernBERT {layer_type} RoPE parameters {params!r} are not supported by transformers 4")
        setattr(encoder_config, field, float(params["rope_theta"]))
    layer_types = getattr(encoder_config, "layer_types", None)
    if layer_types is not None:
        every = encoder_config.global_attn_every_n_layers
        expected = [
            "full_attention" if index % every == 0 else "sliding_attention" for index in range(len(layer_types))
        ]
        if list(layer_types) != expected:
            raise ValueError(
                "ModernBERT layer_types do not follow global_attn_every_n_layers; transformers 4 cannot load them"
            )
    return True


class GLiClassAdapter(BaseAdapter):
    """Adapter for GLiClass zero-shot classification models.

    Runs gliclass models with the library's input format. Works with models
    like knowledgator/gliclass-base-v1.0.

    GLiClass performs classification in a single forward pass (not NLI-based),
    making it much faster than cross-encoder approaches.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_pipe", "_tokenizer"),
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
                tokenization so inputs cannot exceed the model's
                position-embedding capacity.
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

        # The gliclass pipe for the model's architecture: it assembles and
        # tokenizes model inputs and holds the model.
        self._pipe: Any | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._special_count: int = 0
        self._max_token_chars: int = _DEFAULT_MAX_TOKEN_CHARS
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load model onto specified device.

        Args:
            device: Target device (cuda:0, cuda:1, cpu, mps).
        """
        from gliclass import (  # ty:ignore[unresolved-import]
            GLiClassModel,
            GLiClassModelConfig,
            ZeroShotClassificationPipeline,
        )

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
        config = GLiClassModelConfig.from_pretrained(self._model_name_or_path, **shared_kwargs)
        if _restore_modernbert_rope_fields(config.encoder_config):
            model = GLiClassModel.from_pretrained(self._model_name_or_path, config=config, **shared_kwargs)
        else:
            model = GLiClassModel.from_pretrained(self._model_name_or_path, **shared_kwargs)
        model = model.to(device, dtype=torch_dtype)
        tokenizer = self._load_tokenizer(shared_kwargs)

        # Bound the tokenizer's max length so any internal tokenization in the
        # gliclass library auto-truncates to the model's actual capacity.
        if self._max_seq_length is not None:
            tokenizer.model_max_length = self._max_seq_length

        # Pass max_length explicitly so the pipe's
        # ``tokenizer(..., truncation=True, max_length=self.max_length)`` calls
        # cap inputs at the model's position-embedding limit. Without this the
        # library defaults to 1024, which exceeds the 512-token capacity of the
        # current GLiClass models and causes argmax-on-empty-tensor crashes for
        # long inputs. The classification type only matters to the pipeline's
        # own ``__call__``, which the adapter does not use.
        pipeline_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "classification_type": self._classification_type,
            "progress_bar": False,
        }
        if self._max_seq_length is not None:
            pipeline_kwargs["max_length"] = self._max_seq_length
        self._attach(ZeroShotClassificationPipeline(**pipeline_kwargs).pipe, tokenizer)

    def _attach(self, pipe: Any, tokenizer: PreTrainedTokenizerBase) -> None:
        """Use ``pipe`` (a gliclass pipe holding the model) and its tokenizer."""
        self._pipe = pipe
        self._tokenizer = tokenizer
        self._special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        self._max_token_chars = _longest_token_chars(tokenizer)

    def _load_tokenizer(self, shared_kwargs: dict[str, Any]) -> PreTrainedTokenizerBase:
        try:
            return AutoTokenizer.from_pretrained(self._model_name_or_path, **shared_kwargs)
        except ValueError as exc:
            # Checkpoints saved with transformers 5 record the generic fast
            # tokenizer as "TokenizersBackend", a class transformers 4 lacks.
            # Their tokenizer.json loads unchanged as a PreTrainedTokenizerFast.
            if "TokenizersBackend" not in str(exc):
                raise
            return PreTrainedTokenizerFast.from_pretrained(self._model_name_or_path, **shared_kwargs)

    def _require_pipe(self) -> Any:
        if self._pipe is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._pipe

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
        tokens: _RequestTokens | None = None,
    ) -> list[str]:
        """Enforce overflow_policy by pre-tokenizing text and label_prompt separately.

        The model sees ``observed = text_tokens + label_prompt_tokens +
        special_count``, where ``special_count`` is the BERT-style
        ``[CLS]``/``[SEP]`` wrap (2 for all current gliclass models). We
        recover the same total without running the model by tokenizing each
        part with ``add_special_tokens=False``. The label prompt includes the
        task prompt and few-shot examples when the request sets them, so
        ``truncate_text`` shortens only the document.

        On overflow:
        - ``default`` returns texts unchanged (upstream as-is — may crash inside
          the model; ``_overflow_errors_as_input_too_long`` in ``extract`` is
          the defense-in-depth backstop).
        - ``error`` raises ``InputTooLongError`` (whole batch fails, no partial
          responses).
        - ``truncate_text`` slices text to
          ``budget = max_sequence_length - label_prompt_tokens - special_count``.

        Under ``truncate_text`` and ``error``, ``label_prompt`` alone exceeding
        the cap always raises.
        """
        if policy == "default":
            return texts

        if self._pipe is None or self._tokenizer is None or self._max_seq_length is None:
            raise RuntimeError(ERR_NOT_LOADED)

        tokens = tokens or _RequestTokens(self._tokenizer)
        label_prompt_tokens = self._overflow_label_tokens(labels, prompt, examples, tokens)
        overhead = label_prompt_tokens + self._special_count
        budget = self._max_seq_length - overhead

        if budget <= 0:
            raise InputTooLongError(
                f"label_prompt ({label_prompt_tokens} tokens) + special ({self._special_count}) "
                f"exceeds max_sequence_length ({self._max_seq_length}); reduce the number or length of labels"
            )

        new_texts: list[str] = []
        for i, (text, text_ids) in enumerate(zip(texts, tokens.full(texts), strict=True)):
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
            instruction: Optional task description, passed to the model as the
                gliclass pipeline's ``prompt``.
            options: Adapter options to override model config defaults.
                Supported: ``threshold`` (float), ``classification_type``
                ("single-label" or "multi-label"), ``examples`` (few-shot
                ``[{"text": str, "labels": [str, ...]}]``), ``label_groups``
                (``{group: [label, ...]}``, used instead of ``labels``),
                ``group_encoding`` ("separate" or "joint") and
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
            ValueError: If items lack text or the model returns invalid scores.
        """
        self._require_pipe()

        opts = options or {}
        label_groups = self._validate_label_groups(opts.get("label_groups"))
        group_encoding = self._validate_group_encoding(opts.get("group_encoding"), label_groups)
        classification_type = self._validate_classification_type(
            opts.get("classification_type", self._classification_type)
        )
        if label_groups is None:
            normalized_labels = self._validate_labels(labels)
        elif labels:
            raise InvalidInputError("GLiClass accepts either labels or options.label_groups, not both")
        else:
            normalized_labels = self._flatten_label_groups(label_groups, classification_type)
        if label_groups is not None and group_encoding == "separate":
            self._check_group_count(label_groups)
            for _, group_labels in label_groups:
                self._check_label_size(group_labels)
        else:
            self._check_label_size(normalized_labels)
        prompt = self._validate_instruction(instruction)
        examples = self._validate_examples(opts.get("examples"), normalized_labels, label_groups)
        self._check_context_size(prompt, examples)

        # Extract texts from all items (batch processing)
        texts = [self._extract_text(item) for item in items]

        # Get options with fallback to model defaults. The threshold is applied
        # server-side as a post-filter so every label's score is computed
        # regardless of caller preferences.
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold))

        overflow_policy = opts.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        tokens = _RequestTokens(self._tokenizer)

        if label_groups is not None and group_encoding == "separate":
            return self._extract_separate(
                texts,
                label_groups,
                classification_type=classification_type,
                prompt=prompt,
                examples=examples,
                threshold=effective_threshold,
                overflow_policy=overflow_policy,
                tokens=tokens,
            )

        texts = self._apply_overflow_policy(
            texts, normalized_labels, overflow_policy, prompt=prompt, examples=examples, tokens=tokens
        )
        layout = self._request_layout(normalized_labels, prompt, examples, tokens)
        fits = self._items_fit(texts, layout, normalized_labels, prompt, examples, tokens=tokens)
        input_token_counts = self._input_token_counts(texts, prompt, examples, layout, fits, tokens=tokens)
        errors = self._item_errors(fits)
        kept = [index for index, ok in enumerate(fits) if ok]
        kept_texts = [texts[index] for index in kept]

        if label_groups is not None:
            rows = (
                self._joint_scores(kept_texts, label_groups, normalized_labels, classification_type, prompt, examples)
                if kept_texts
                else []
            )
            return self._grouped_output(
                rows,
                label_groups,
                classification_type=classification_type,
                threshold=effective_threshold,
                kept=kept,
                item_count=len(items),
                errors=errors,
                input_token_counts=input_token_counts,
            )

        if not kept_texts:
            return ExtractOutput(
                entities=[[] for _ in items],
                classifications=[[] for _ in items],
                errors=errors,
                input_token_counts=input_token_counts,
            )

        with _overflow_errors_as_input_too_long():
            batch_scores = self._score_rows(
                kept_texts,
                normalized_labels,
                classification_type=classification_type,
                prompt=prompt,
                examples=examples,
            )

        all_classifications: list[list[Classification]] = [[] for _ in items]
        for index, scores in zip(kept, batch_scores, strict=True):
            classifications: list[Classification] = [
                Classification(label=label, score=self._validate_score(score))
                for label, score in zip(normalized_labels, scores, strict=True)
            ]

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

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        """Batching cost per item: the characters of every row it encodes.

        Separate label groups encode each document once per group, next to
        that group's labels and the request's instruction and examples, so
        the default per-item character count would undercount such a request
        by its group count. Other requests keep the default (None). This runs
        before batching and before validation, so it walks at most
        ``MAX_EXTRACT_LABELS`` labels and ``_MAX_EXAMPLES`` examples.
        Best-effort: never raises; malformed requests fail in ``extract``.
        """
        _ = labels, output_schema
        try:
            opts = options or {}
            groups = opts.get("label_groups")
            if not isinstance(groups, dict) or not groups or len(groups) > _MAX_LABEL_GROUPS:
                return None
            if opts.get("group_encoding", _DEFAULT_GROUP_ENCODING) != "separate":
                return None
            group_labels = (label for values in groups.values() if isinstance(values, list) for label in values)
            label_chars = sum(
                len(label) for label in islice(group_labels, MAX_EXTRACT_LABELS) if isinstance(label, str)
            )
            context_chars = len(instruction) if isinstance(instruction, str) else 0
            examples = opts.get("examples")
            if isinstance(examples, list):
                for example in islice(cast("list[Any]", examples), _MAX_EXAMPLES):
                    text = cast("dict[str, Any]", example).get("text") if isinstance(example, dict) else None
                    context_chars += len(text) if isinstance(text, str) else 0
            return [len(groups) * (len(item.text or "") + context_chars) + label_chars for item in items]
        except Exception:  # noqa: BLE001 -- a cost estimate must never fail the request
            return None

    def _score_rows(
        self,
        texts: list[str],
        labels: list[str] | list[list[str]],
        *,
        classification_type: ClassificationType,
        prompt: str | None,
        examples: list[dict[str, Any]] | list[list[dict[str, Any]]] | None,
        batch_size: int = _PIPELINE_BATCH_SIZE,
    ) -> list[list[float]]:
        """Each row's label scores, computed as the gliclass pipeline computes them.

        ``labels`` is one label list shared by every row, or one list per row
        (then ``examples`` is also one list per row, or None). Rows run in
        forward passes of at most ``batch_size``, by default the pipeline's
        sub-batches. A single-label row gets a softmax over the label slots a
        call with only its labels would have; a multi-label row a sigmoid per
        label. Each forward's scores come back in one device-to-host copy.
        """
        pipe = self._require_pipe()
        shared = isinstance(labels[0], str)
        scores: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                end = start + len(batch_texts)
                batch_labels = labels if shared else labels[start:end]
                batch_examples = examples if shared or examples is None else examples[start:end]
                inputs = pipe.prepare_inputs(
                    batch_texts, batch_labels, same_labels=shared, examples=batch_examples, prompt=prompt
                )
                logits = self._forward(pipe, inputs, batch_labels, same_labels=shared)
                row_labels = cast("list[list[str]]", [batch_labels] * len(batch_texts) if shared else batch_labels)
                probs = torch.sigmoid(logits) if classification_type == "multi-label" else None
                rows: list[torch.Tensor] = []
                for row, row_label_list in enumerate(row_labels):
                    count = len(row_label_list)
                    width = self._row_width(pipe, row_label_list, logits.shape[-1])
                    if width < count:
                        raise InputTooLongError(_ERR_INPUT_TOO_LONG)
                    if probs is not None:
                        rows.append(probs[row, :count])
                    elif width == logits.shape[-1]:
                        rows.append(torch.softmax(logits[row], dim=-1)[:count])
                    else:
                        rows.append(torch.softmax(logits[row, :width], dim=-1)[:count])
                values = torch.cat(rows).tolist()
                position = 0
                for row_label_list in row_labels:
                    scores.append(values[position : position + len(row_label_list)])
                    position += len(row_label_list)
        return scores

    @staticmethod
    def _forward(pipe: Any, inputs: Any, labels: list[str] | list[list[str]], *, same_labels: bool) -> torch.Tensor:
        """Run the model on tokenized rows, passing the class-slot count the pipeline passes."""
        forward_kwargs: dict[str, Any] = {}
        resolve_max_num_classes = getattr(pipe, "_resolve_max_num_classes", None)
        if resolve_max_num_classes is not None:
            forward_kwargs["max_num_classes"] = resolve_max_num_classes(labels, same_labels)
        return pipe.model(**inputs, **forward_kwargs).logits

    @staticmethod
    def _row_width(pipe: Any, labels: list[str], batch_width: int) -> int:
        """Class slots a forward over only this row's labels would score.

        With the default dynamic allocation that is one slot per label, fewer
        than a batch of rows with more labels gets. Fixed allocations give
        every row the batch's width.
        """
        resolve_max_num_classes = getattr(pipe, "_resolve_max_num_classes", None)
        width = resolve_max_num_classes(labels, True) if resolve_max_num_classes is not None else None
        return batch_width if width is None else min(int(width), batch_width)

    def _joint_scores(
        self,
        texts: list[str],
        label_groups: list[tuple[str, list[str]]],
        flat_labels: list[str],
        classification_type: ClassificationType,
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> list[list[float]]:
        """Score all groups' labels in one row per text and normalize per group.

        All groups share one forward pass per text, as the gliclass pipeline
        encodes a dict of labels ("group.label" after flattening). The
        pipeline's own single-label mode applies one softmax across every
        flattened label, which makes a group's scores depend on the other
        groups. Here each group gets its own softmax over the raw logits
        instead; multi-label scores are independent sigmoids either way.
        """
        pipe = self._require_pipe()
        num_labels = len(flat_labels)

        chunks: list[torch.Tensor] = []
        with torch.inference_mode(), _overflow_errors_as_input_too_long():
            for start in range(0, len(texts), _PIPELINE_BATCH_SIZE):
                inputs = pipe.prepare_inputs(
                    texts[start : start + _PIPELINE_BATCH_SIZE],
                    flat_labels,
                    same_labels=True,
                    examples=examples,
                    prompt=prompt,
                )
                logits = self._forward(pipe, inputs, flat_labels, same_labels=True)
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

    def _extract_separate(
        self,
        texts: list[str],
        label_groups: list[tuple[str, list[str]]],
        *,
        classification_type: ClassificationType,
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        threshold: float,
        overflow_policy: OverflowPolicy,
        tokens: _RequestTokens,
    ) -> ExtractOutput:
        """Encode the document once per group, with only that group's labels.

        Each row is what a request with ``labels`` set to the group's labels
        encodes: its overflow policy, window check and metering apply per
        row. An item whose labels do not fit in one of its rows fails with
        ``INPUT_TOO_LONG`` and is billed nothing; the other items still run.
        """
        group_examples = self._examples_by_group(examples, label_groups)
        self._contexts(
            tokens,
            [
                (group_labels, prompt, group_examples[group] if group_examples is not None else None)
                for group, (_, group_labels) in enumerate(label_groups)
            ],
            overflow=overflow_policy != "default",
        )
        item_count = len(texts)
        fits = [True] * item_count
        counts: list[int] | None = [0] * item_count
        group_texts: list[list[str]] = []
        group_lengths: list[list[int]] | None = []
        for group, (_, group_labels) in enumerate(label_groups):
            row_examples = group_examples[group] if group_examples is not None else None
            row_texts = self._apply_overflow_policy(
                texts, group_labels, overflow_policy, prompt=prompt, examples=row_examples, tokens=tokens
            )
            layout = self._request_layout(group_labels, prompt, row_examples, tokens)
            row_fits = self._items_fit(row_texts, layout, group_labels, prompt, row_examples, tokens=tokens)
            row_counts = self._input_token_counts(
                row_texts, prompt, row_examples, layout, [True] * item_count, tokens=tokens
            )
            fits = [ok and row_ok for ok, row_ok in zip(fits, row_fits, strict=True)]
            counts = (
                None
                if counts is None or row_counts is None
                else [total + count for total, count in zip(counts, row_counts, strict=True)]
            )
            group_texts.append(row_texts)
            group_lengths = (
                None
                if group_lengths is None or layout is None
                else [*group_lengths, self._row_lengths(row_texts, layout, tokens)]
            )
        if counts is not None:
            counts = [count if ok else 0 for count, ok in zip(counts, fits, strict=True)]
        errors = self._item_errors(fits)
        kept = [index for index, ok in enumerate(fits) if ok]
        rows = self._separate_scores(
            [[group_texts[group][index] for group in range(len(label_groups))] for index in kept],
            label_groups,
            classification_type=classification_type,
            prompt=prompt,
            group_examples=group_examples,
            row_lengths=(
                None
                if group_lengths is None
                else [[group_lengths[group][index] for group in range(len(label_groups))] for index in kept]
            ),
        )
        return self._grouped_output(
            rows,
            label_groups,
            classification_type=classification_type,
            threshold=threshold,
            kept=kept,
            item_count=item_count,
            errors=errors,
            input_token_counts=counts,
        )

    def _separate_scores(
        self,
        item_rows: list[list[str]],
        label_groups: list[tuple[str, list[str]]],
        *,
        classification_type: ClassificationType,
        prompt: str | None,
        group_examples: list[list[dict[str, Any]]] | None,
        row_lengths: list[list[int]] | None = None,
    ) -> list[list[float]]:
        """Scores of every (item, group) row, flattened per item in group order.

        ``item_rows[i][g]`` is item ``i``'s document as group ``g`` encodes it,
        and ``row_lengths[i][g]`` that row's estimated token length.
        Uni-encoder rows of different groups share forward passes, packed by
        ``_row_chunks``. Other architectures take one label set per forward,
        so their rows run group by group.
        """
        if not item_rows:
            return []
        group_count = len(label_groups)
        pipe = self._require_pipe()
        config = getattr(getattr(pipe, "model", None), "config", None)
        per_row: list[list[float]] = []
        with _overflow_errors_as_input_too_long():
            if getattr(config, "architecture_type", None) == "uni-encoder":
                row_texts = [text for texts in item_rows for text in texts]
                row_labels = [group_labels for _ in item_rows for _, group_labels in label_groups]
                row_examples = (
                    [group_examples[group] for _ in item_rows for group in range(group_count)]
                    if group_examples is not None
                    else None
                )
                lengths = None if row_lengths is None else [length for item in row_lengths for length in item]
                per_row = [[] for _ in row_texts]
                for chunk in self._row_chunks(lengths, len(row_texts), pipe.max_length):
                    scored = self._score_rows(
                        [row_texts[row] for row in chunk],
                        [row_labels[row] for row in chunk],
                        classification_type=classification_type,
                        prompt=prompt,
                        examples=None if row_examples is None else [row_examples[row] for row in chunk],
                        batch_size=len(chunk),
                    )
                    for row, row_scores in zip(chunk, scored, strict=True):
                        per_row[row] = row_scores
            else:
                by_group = [
                    self._score_rows(
                        [texts[group] for texts in item_rows],
                        group_labels,
                        classification_type=classification_type,
                        prompt=prompt,
                        examples=group_examples[group] if group_examples is not None else None,
                    )
                    for group, (_, group_labels) in enumerate(label_groups)
                ]
                per_row = [by_group[group][item] for item in range(len(item_rows)) for group in range(group_count)]
        return [
            [score for row in per_row[item * group_count : (item + 1) * group_count] for score in row]
            for item in range(len(item_rows))
        ]

    def _row_lengths(self, texts: list[str], layout: _RequestLayout, tokens: _RequestTokens) -> list[int]:
        """Estimated token length of each row: its document, label prompt, context and special tokens.

        The parts are tokenized apart, so a row can differ from its estimate
        by a token or two where they meet.
        """
        pipe = self._require_pipe()
        overhead = layout.overhead_tokens + self._special_count
        return [min(pipe.max_length, len(ids) + overhead) for ids in tokens.cut(texts, layout.window)]

    @staticmethod
    def _row_chunks(lengths: list[int] | None, count: int, max_length: int) -> list[list[int]]:
        """Group row indices into forward passes of at most ``_PIPELINE_BATCH_SIZE * max_length`` padded tokens.

        That is the most a labels request's sub-batch of eight full-window
        rows pads to. Rows are packed longest first, so rows of similar
        length share padding and many short rows share one forward pass.
        Without length estimates, rows run in order, eight per forward.
        """
        if lengths is None:
            return [
                list(range(start, min(start + _PIPELINE_BATCH_SIZE, count)))
                for start in range(0, count, _PIPELINE_BATCH_SIZE)
            ]
        budget = _PIPELINE_BATCH_SIZE * max_length
        order = sorted(range(count), key=lambda row: lengths[row], reverse=True)
        chunks: list[list[int]] = []
        start = 0
        while start < count:
            size = max(1, min(count - start, budget // max(1, lengths[order[start]])))
            chunks.append(order[start : start + size])
            start += size
        return chunks

    def _grouped_output(
        self,
        rows: list[list[float]],
        label_groups: list[tuple[str, list[str]]],
        *,
        classification_type: ClassificationType,
        threshold: float,
        kept: list[int],
        item_count: int,
        errors: list[ExtractItemError | None] | None,
        input_token_counts: list[int] | None,
    ) -> ExtractOutput:
        """One answer per group from each kept item's scores, in flattened group order.

        A single-label group answers like a choice question:
        ``{"type": "choice", "choice", "probabilities", "confidence"}`` with
        ``confidence = 1 - H(p) / log(k)``. A multi-label group answers
        ``{"labels", "probabilities"}``.
        """
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

    @staticmethod
    def _examples_by_group(
        examples: list[dict[str, Any]] | None,
        label_groups: list[tuple[str, list[str]]],
    ) -> list[list[dict[str, Any]]] | None:
        """Each group's view of the examples: every example, with only that group's labels.

        Example labels arrive flattened (``"group.label"``); the flattened
        names are unique, so each maps back to one group.
        """
        if examples is None:
            return None
        owner = {
            f"{group}{_LABEL_GROUP_SEPARATOR}{label}": (index, label)
            for index, (group, group_labels) in enumerate(label_groups)
            for label in group_labels
        }
        by_group: list[list[dict[str, Any]]] = [[] for _ in label_groups]
        for example in examples:
            chosen: list[list[str]] = [[] for _ in label_groups]
            for flat in example["labels"]:
                group, label = owner[flat]
                chosen[group].append(label)
            for group, group_labels in enumerate(chosen):
                by_group[group].append({"text": example["text"], "labels": group_labels})
        return by_group

    @staticmethod
    def _validate_classification_type(value: object) -> ClassificationType:
        if value == "single-label":
            return "single-label"
        if value == "multi-label":
            return "multi-label"
        raise InvalidInputError("GLiClass classification_type must be 'single-label' or 'multi-label'")

    @staticmethod
    def _validate_group_encoding(value: object, label_groups: list[tuple[str, list[str]]] | None) -> GroupEncoding:
        if value is None:
            return _DEFAULT_GROUP_ENCODING
        if value not in ("separate", "joint"):
            raise InvalidInputError("GLiClass group_encoding must be 'separate' or 'joint'")
        if label_groups is None:
            raise InvalidInputError("GLiClass group_encoding applies only with options.label_groups")
        return cast("GroupEncoding", value)

    def _check_group_count(self, label_groups: list[tuple[str, list[str]]]) -> None:
        """Bound the rows a separate-group request adds per item: one full row per group."""
        window = self._max_seq_length or getattr(self._pipe, "max_length", None)
        limit = _MAX_LABEL_GROUPS
        if isinstance(window, int) and window > 0:
            limit = max(1, min(limit, _MAX_ITEM_ROW_TOKENS // window))
        if len(label_groups) > limit:
            raise InvalidInputError(
                f"GLiClass encodes the document once per label group and accepts at most {limit} groups "
                f"per request (got {len(label_groups)}); send fewer groups, or options.group_encoding='joint'"
            )

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

    def _contexts(
        self,
        tokens: _RequestTokens,
        contexts: list[tuple[list[str], str | None, list[dict[str, Any]] | None]],
        *,
        overflow: bool,
    ) -> list[_Context]:
        """Tokenize the (labels, instruction, examples) contexts a request encodes, in one call.

        ``overflow`` also tokenizes each context's pipeline input with an
        empty document, which the overflow policies measure. Contexts the
        request already tokenized are reused.
        """
        entries: list[_Context] = []
        strings: list[str] = []
        pending: list[tuple[_Context, int, int, bool]] = []
        for labels, prompt, examples in contexts:
            key = (
                tuple(labels),
                prompt,
                None if examples is None else tuple((e["text"], tuple(e["labels"])) for e in examples),
            )
            context = tokens.contexts.setdefault(key, _Context())
            entries.append(context)
            first = len(strings)
            if context.parts is None:
                strings.extend(self._layout_strings(labels, prompt, examples))
            layout_count = len(strings) - first if context.parts is None else -1
            with_overflow = overflow and context.overflow_tokens is None
            if with_overflow:
                pipe = self._require_pipe()
                strings.append(pipe.prepare_input(text="", labels=labels, **_pipeline_context(prompt, examples)))
            pending.append((context, first, layout_count, with_overflow))
        encoded: list[list[int]] = []
        if strings:
            if self._tokenizer is None:
                raise RuntimeError(ERR_NOT_LOADED)
            encoded = self._tokenizer(strings, add_special_tokens=False)["input_ids"]
        for context, first, layout_count, with_overflow in pending:
            if layout_count >= 0:
                context.parts = encoded[first : first + layout_count]
            if with_overflow:
                context.overflow_tokens = len(encoded[first + max(layout_count, 0)])
        return entries

    def _layout_strings(
        self,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> list[str]:
        """The parts of a context the label-fit check measures; none unless the model is a uni-encoder.

        Only uni-encoder GLiClass models put label markers in their input.
        """
        pipe = self._pipe
        config = getattr(getattr(pipe, "model", None), "config", None)
        if (
            pipe is None
            or config is None
            or self._tokenizer is None
            or getattr(config, "architecture_type", None) != "uni-encoder"
        ):
            return []
        label_prompt = "".join(f"{pipe.label_token}{label}" for label in labels) + pipe.sep_token
        parts = [label_prompt, *([prompt] if prompt else []), *(example["text"] for example in examples or [])]
        if examples:
            parts.append(pipe._format_examples_for_input(examples))
        return parts

    def _request_layout(
        self,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        tokens: _RequestTokens | None = None,
    ) -> _RequestLayout | None:
        """The token layout every item of a request shares (see ``_compute_layout``)."""
        tokens = tokens or _RequestTokens(self._tokenizer)
        (context,) = self._contexts(tokens, [(labels, prompt, examples)], overflow=False)
        if not context.has_layout:
            context.layout = self._compute_layout(context.parts or [], labels, prompt, examples)
            context.has_layout = True
        return context.layout

    def _overflow_label_tokens(
        self,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        tokens: _RequestTokens,
    ) -> int:
        """Tokens of the pipeline input with an empty document: labels, instruction and examples."""
        (context,) = self._contexts(tokens, [(labels, prompt, examples)], overflow=True)
        if context.overflow_tokens is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return context.overflow_tokens

    def _compute_layout(
        self,
        parts: list[list[int]],
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
    ) -> _RequestLayout | None:
        """The layout of a context from its tokenized parts (see ``_layout_strings``).

        Returns None when the model does not put label markers in its input
        (only uni-encoder GLiClass models do), so there is nothing to check.

        Raises:
            InputTooLongError: If the label prompt alone overflows the window.
            InvalidInputError: If the instruction and examples leave no room
                for the document.
        """
        pipe = self._pipe
        config = getattr(getattr(pipe, "model", None), "config", None)
        if not parts or pipe is None or config is None:
            return None
        label_ids = parts[0]
        markers = [position for position, token in enumerate(label_ids) if token == config.class_token_index]
        if len(markers) < len(labels):
            return None
        window = pipe.max_length - self._special_count
        if markers[-1] >= window:
            raise InputTooLongError(_ERR_INPUT_TOO_LONG)
        prompt_tokens = len(parts[1]) if prompt else 0
        first_example = 2 if prompt else 1
        example_count = len(examples or [])
        example_text_tokens = sum(len(ids) for ids in parts[first_example : first_example + example_count])
        examples_tokens = len(parts[-1]) if examples else 0
        if (prompt or examples) and len(label_ids) + prompt_tokens + examples_tokens >= window:
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
            overhead_tokens=len(label_ids) + prompt_tokens + examples_tokens,
        )

    def _items_fit(
        self,
        texts: list[str],
        layout: _RequestLayout | None,
        labels: list[str],
        prompt: str | None,
        examples: list[dict[str, Any]] | None,
        *,
        tokens: _RequestTokens | None = None,
    ) -> list[bool]:
        """Whether each item's label markers survive truncation.

        Documents are tokenized once, on their own. Tokenizing a document next
        to the label prompt can differ from that by a token or two (a trailing
        space before a marker, for example), so items within a few tokens of
        the window edge are checked exactly on their fused encoding.
        """
        if layout is None or layout.prompt_first or self._tokenizer is None:
            return [True] * len(texts)
        tokens = tokens or _RequestTokens(self._tokenizer)
        fits: list[bool] = []
        for text, ids in zip(texts, tokens.cut(texts, layout.window), strict=True):
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
        pipe = self._pipe
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
        *,
        tokens: _RequestTokens | None = None,
    ) -> list[int] | None:
        """Billable input tokens per item, for one encoded row per item.

        Each row is billed for its document and for the free-form request
        text encoded with it: the instruction and the few-shot example texts.
        Label names (including the labels attached to examples) and the
        pipeline's marker tokens are not billed. With an instruction or
        examples, a row's total is capped at the model window minus the
        label prompt, the most free-form text it can encode, unless the
        document count alone is already higher. Items refused because their
        labels would be cut off are billed nothing.
        """
        counts = self._doc_input_token_counts(texts, tokens)
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
        window = self._max_seq_length or getattr(self._pipe, "max_length", None)
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

    def _doc_input_token_counts(self, texts: list[str], tokens: _RequestTokens | None = None) -> list[int] | None:
        """Count document-only model-tokenizer input units for billing.

        GLiClass fuses the request's label schema with every document. The
        label prompt is reusable request schema rather than billed content, so
        this mirrors the GLiNER family contract and counts each post-policy
        document with the model tokenizer, including its normal special tokens.
        The tokenizer truncates content to ``max_seq_length`` minus the
        special tokens and then adds them, so the count is the document's
        content tokens cut to that length, plus the special tokens.
        """
        if self._tokenizer is None:
            return None
        special = self._special_count
        try:
            if self._max_seq_length is None:
                content = tokens.full(texts) if tokens is not None else None
                if content is None:
                    encoded = self._tokenizer(texts, add_special_tokens=True)
                    return [len(input_ids) for input_ids in encoded["input_ids"]]
                return [len(ids) + special for ids in content]
            if tokens is None or self._max_seq_length <= special:
                encoded = self._tokenizer(
                    texts, add_special_tokens=True, truncation=True, max_length=self._max_seq_length
                )
                counts = [len(input_ids) for input_ids in encoded["input_ids"]]
                return counts if len(counts) == len(texts) else None
            return [len(ids) + special for ids in tokens.cut(texts, self._max_seq_length - special)]
        except Exception:  # noqa: BLE001 -- metering must not fail classification
            return None
