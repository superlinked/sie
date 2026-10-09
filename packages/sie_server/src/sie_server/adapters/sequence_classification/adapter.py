"""Fixed-head text classifiers: Hugging Face ``AutoModelForSequenceClassification`` checkpoints.

The classes are the checkpoint's ``config.id2label``; a request sends no labels
to define them. Scores follow transformers' ``TextClassificationPipeline``: a
softmax over every class for a single-label head, an independent sigmoid per
class for a ``multi_label_classification`` head or a head with one class.
Regression heads are refused at load because their outputs are not class
probabilities.

Request surface (all optional):

- ``labels``: a subset of the model's classes to return. Scores are still
  computed over every class (a softmax is not renormalized over the subset).
- ``options.multi_label``: ``true`` scores each class with a sigmoid, ``false``
  with a softmax over all classes, overriding the checkpoint's ``problem_type``.
- ``options.top_k``: return at most this many classes, best first.
- ``options.threshold``: drop classes scoring below this value in [0, 1].
- ``options.overflow_policy``: ``default`` and ``truncate_text`` keep the first
  ``max_seq_length`` tokens, as the pipeline does with ``truncation=True``;
  ``error`` fails an item that does not fit whole with ``INPUT_TOO_LONG``.

A long text is cut at a whitespace boundary before it is tokenized, to the
characters that hold more tokens than the window keeps. With a tokenizer that
splits words at whitespace (checked at load), the tokens of the cut text are a
prefix of the whole text's tokens, so scores and usage are unchanged; any other
tokenizer reads whole texts. ``usage.input_tokens`` counts the tokens the model
reads, special tokens included; failed items count nothing.
"""

from __future__ import annotations

import logging
import math
import re
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError
from sie_server.types.overflow_policy import DEFAULT_OVERFLOW_POLICY
from sie_server.types.responses import Classification, ErrorCode

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_OPTIONS = frozenset({"multi_label", "top_k", "threshold", "overflow_policy"})
_OFFSET_POSITION_MODEL_TYPES = frozenset(
    {"roberta", "xlm-roberta", "xlm-roberta-xl", "camembert", "roberta-prelayernorm", "data2vec-text"}
)
_NO_LIMIT = 1 << 40
# A forward pass stops taking shorter rows once it holds this many real tokens
# and the next row would make more than this share of it padding.
_MIN_CHUNK_TOKENS = 2048
_MAX_PADDING = 0.25
# A text longer than this many characters per window token is cut before
# tokenizing; the cut doubles up to the maximum when the cut part holds too few
# tokens (whitespace runs, which most tokenizers read as nothing).
_CUT_CHARS_PER_TOKEN = 16
_MAX_CUT_CHARS_PER_TOKEN = 64
_CUT_SLACK_TOKENS = 16
_WHITESPACE = re.compile(r"\s")
_MAX_QUOTED_CHARS = 64
_MAX_LISTED_LABELS = 32


class SequenceClassificationAdapter(FlashBaseAdapter):
    """Serve a fine-tuned classification head through ``extract``."""

    fallback_adapter_path: ClassVar[str | None] = None

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer", "_dtype"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int | None = None,
        max_forward_tokens: int = 16384,
        compute_precision: ComputePrecision = "float16",
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: Hugging Face repo id or local checkpoint directory.
            max_seq_length: Tokens per text, special tokens included. Clamped to
                the tokenizer's and position table's limits; defaults to them.
            max_forward_tokens: Padded-token budget per forward pass. Texts are
                sorted by length and run in chunks within it.
            compute_precision: Weight and activation precision on accelerators.
                CPU always runs float32. Probabilities are computed in float32.
            attn_implementation: ``eager``, ``sdpa`` or ``flash_attention_2``;
                ``None`` keeps the transformers default (SDPA where the model
                supports it).
            trust_remote_code: Load custom modeling code from the checkpoint.
            revision: Hugging Face revision (commit SHA) to pin.
            **kwargs: Additional loader arguments (ignored).
        """
        _ = kwargs
        if max_seq_length is not None and (
            isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length < 2
        ):
            raise ValueError("max_seq_length must be an integer of at least 2")
        if isinstance(max_forward_tokens, bool) or not isinstance(max_forward_tokens, int) or max_forward_tokens < 1:
            raise ValueError("max_forward_tokens must be a positive integer")
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._max_forward_tokens = max_forward_tokens
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._trust_remote_code = trust_remote_code
        self._revision = revision

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._labels: tuple[str, ...] = ()
        self._label_index: dict[str, int] = {}
        self._default_multi_label = False
        self._sigmoid_only = False
        self._max_length = 0
        self._cut_text = False
        self._pad_token_id = 0

    # ------------------------------------------------------------------ loading

    def _resolve_dtype(self) -> torch.dtype:
        if self._device is None or not self._device.startswith(("cuda", "mps")):
            return torch.float32
        return super()._resolve_dtype()

    def load(self, device: str) -> None:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()
        shared: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared["revision"] = self._revision

        config = AutoConfig.from_pretrained(self._model_name_or_path, **shared)
        labels, multi_label, sigmoid_only = self._read_head(config)
        if hasattr(config, "reference_compile"):
            # ModernBERT otherwise torch.compiles its embeddings on the first CUDA call.
            config.reference_compile = False

        model_kwargs: dict[str, Any] = {"config": config, "dtype": self._dtype}
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path, **shared)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name_or_path, **shared, **model_kwargs)
        model.to(device)
        model.eval()

        capacity = self._capacity(tokenizer, config)
        if self._max_seq_length is not None:
            max_length = min(self._resolve_tokenizer_ceiling(tokenizer, model, self._max_seq_length), capacity)
        elif capacity < _NO_LIMIT:
            max_length = capacity
        else:
            raise ValueError("Set max_seq_length: the checkpoint declares no sequence length limit")
        specials = tokenizer.num_special_tokens_to_add(pair=False)
        if max_length <= specials:
            raise ValueError("max_seq_length leaves no room for text next to the special tokens")
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = getattr(config, "pad_token_id", None)
        if pad is None:
            raise ValueError("The tokenizer defines no padding token")

        self._tokenizer = tokenizer
        self._model = model
        self._labels = labels
        self._label_index = {label: index for index, label in enumerate(labels)}
        self._default_multi_label = multi_label
        self._sigmoid_only = sigmoid_only
        self._max_length = max_length
        self._cut_text = self._splits_at_whitespace(tokenizer)
        self._pad_token_id = int(pad)
        logger.info(
            "Loaded %s: %d classes, %s, max_length=%d, attention=%s, dtype=%s",
            self._model_name_or_path,
            len(labels),
            "sigmoid" if multi_label or sigmoid_only else "softmax",
            max_length,
            getattr(model.config, "_attn_implementation", None),
            self._dtype,
        )

    @staticmethod
    def _capacity(tokenizer: Any, config: Any) -> int:
        """The most tokens the tokenizer and the position table allow, or ``_NO_LIMIT``."""
        caps = [_NO_LIMIT]
        declared = getattr(tokenizer, "model_max_length", None)
        if isinstance(declared, int) and 0 < declared < _NO_LIMIT:
            caps.append(declared)
        positions = getattr(config, "max_position_embeddings", None)
        if isinstance(positions, int) and positions > 0:
            if getattr(config, "model_type", None) in _OFFSET_POSITION_MODEL_TYPES:
                # RoBERTa-style embeddings number positions from padding_idx + 1.
                positions -= int(getattr(config, "pad_token_id", 1) or 0) + 1
            caps.append(positions)
        return min(caps)

    @staticmethod
    def _read_head(config: Any) -> tuple[tuple[str, ...], bool, bool]:
        """Return the class names, whether the head is multi-label, and whether it has one class."""
        problem_type = getattr(config, "problem_type", None)
        if problem_type == "regression":
            raise ValueError("Regression heads are not supported: their outputs are not class probabilities")
        if problem_type not in (None, "single_label_classification", "multi_label_classification"):
            raise ValueError(f"Unsupported problem_type {problem_type!r}")
        num_labels = int(getattr(config, "num_labels", 0) or 0)
        id2label = getattr(config, "id2label", None) or {}
        names: list[str] = []
        for index in range(num_labels):
            name = id2label.get(index, id2label.get(str(index)))
            if not isinstance(name, str) or not name:
                raise ValueError(f"config.id2label has no name for class {index}")
            names.append(name)
        if not names:
            raise ValueError("The checkpoint declares no classes (config.num_labels)")
        if len(set(names)) != len(names):
            raise ValueError("config.id2label names must be unique")
        return tuple(names), problem_type == "multi_label_classification", num_labels == 1

    @staticmethod
    def _splits_at_whitespace(tokenizer: Any) -> bool:
        """Whether text cut before whitespace tokenizes to a prefix of the whole text's tokens."""
        backend = getattr(tokenizer, "backend_tokenizer", None)
        pre_tokenizer = getattr(backend, "pre_tokenizer", None)
        if pre_tokenizer is None:
            return False
        try:
            pieces = pre_tokenizer.pre_tokenize_str("ab cd")
        except Exception:  # noqa: BLE001 -- an unusual pre-tokenizer just reads whole texts
            return False
        return len(pieces) == 2 and pieces[0][1][1] <= 2

    def _metering_max_length(self) -> int | None:
        return self._max_length or None

    # ------------------------------------------------------------------ requests

    def _check_ready(self) -> tuple[Any, PreTrainedTokenizerBase]:
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        return self._model, self._tokenizer

    def _selected_classes(self, labels: list[str] | None) -> list[int] | None:
        if labels is None:
            return None
        if not isinstance(labels, list) or not labels:
            raise InvalidInputError("labels must be a non-empty list of the model's class names")
        if len(labels) > MAX_EXTRACT_LABELS:
            raise InvalidInputError(f"A request may carry at most {MAX_EXTRACT_LABELS} labels")
        selected: list[int] = []
        for label in labels:
            index = self._label_index.get(label) if isinstance(label, str) else None
            if index is None:
                known = ", ".join(_quoted(name) for name in self._labels[:_MAX_LISTED_LABELS])
                more = (
                    f" and {len(self._labels) - _MAX_LISTED_LABELS} more"
                    if len(self._labels) > _MAX_LISTED_LABELS
                    else ""
                )
                raise InvalidInputError(
                    f"Label {_quoted(label)} is not one of this model's classes ({known}{more}). "
                    "Fixed-head classifiers score only the classes they were trained on."
                )
            selected.append(index)
        if len(set(selected)) != len(selected):
            raise InvalidInputError("labels must be unique")
        return selected

    def _request_options(self, options: dict[str, Any] | None) -> tuple[bool, int | None, float, bool]:
        opts = options or {}
        unknown = sorted(str(key) for key in opts if key not in _OPTIONS)
        if unknown:
            raise InvalidInputError(
                f"Unsupported option(s) {', '.join(_quoted(key) for key in unknown[:8])}; "
                f"this classifier takes {', '.join(sorted(_OPTIONS))}"
            )
        multi_label = opts.get("multi_label")
        if multi_label is None:
            multi_label = self._default_multi_label
        elif not isinstance(multi_label, bool):
            raise InvalidInputError("multi_label must be a boolean")
        top_k = opts.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1):
            raise InvalidInputError("top_k must be a positive integer")
        threshold = opts.get("threshold", 0.0)
        if threshold is None:
            threshold = 0.0
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise InvalidInputError("threshold must be a number between 0 and 1")
        try:
            threshold = float(threshold)
        except OverflowError as exc:
            raise InvalidInputError("threshold must be a number between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise InvalidInputError("threshold must be a number between 0 and 1")
        policy = opts.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        return multi_label or self._sigmoid_only, None if top_k is None else int(top_k), threshold, policy == "error"

    def _cut(self, text: str, chars_per_token: int) -> str:
        """The text up to the last whitespace within its first ``chars_per_token`` characters per window token."""
        limit = self._max_length * chars_per_token
        if not self._cut_text or len(text) <= limit:
            return text
        window = text[limit // 2 : limit]
        last = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"), window.rfind("\r"))
        if last < 0:
            for match in _WHITESPACE.finditer(window):
                last = match.start()
        return text if last < 0 else text[: limit // 2 + last]

    def _encode(self, texts: list[str]) -> tuple[list[list[int]], list[bool]]:
        """Token ids truncated to the window, and whether each whole text overflowed it."""
        _, tokenizer = self._check_ready()
        window = self._max_length
        ids: list[list[int] | None] = [None] * len(texts)
        overflowed = [False] * len(texts)
        pending = list(range(len(texts)))
        chars_per_token = _CUT_CHARS_PER_TOKEN
        while pending:
            cut = [self._cut(texts[i], chars_per_token) for i in pending]
            # Tokens past the window show whether the text overflows it; a cut
            # text must reach past them, so that the tokens at its cut are not
            # among those kept.
            with self._tokenizer_guard():
                encoded = tokenizer(
                    cut,
                    truncation=True,
                    max_length=window + _CUT_SLACK_TOKENS,
                    return_attention_mask=False,
                    return_special_tokens_mask=True,
                )
            retry: list[int] = []
            for row, i in enumerate(pending):
                row_ids = list(encoded["input_ids"][row])
                was_cut = len(cut[row]) < len(texts[i])
                if was_cut and len(row_ids) < window + _CUT_SLACK_TOKENS:
                    retry.append(i)
                    continue
                if len(row_ids) > window:
                    # Truncating to the window drops content tokens from the end.
                    content = [j for j, special in enumerate(encoded["special_tokens_mask"][row]) if not special]
                    dropped = set(content[window - len(row_ids) :])
                    row_ids = [token for j, token in enumerate(row_ids) if j not in dropped]
                    overflowed[i] = True
                ids[i] = row_ids
            pending = retry
            if chars_per_token >= _MAX_CUT_CHARS_PER_TOKEN:
                chars_per_token = 1 << 62
            else:
                chars_per_token *= 2
        return [row for row in ids if row is not None], overflowed

    def _forward(self, rows: list[list[int]]) -> torch.Tensor:
        """Logits for every row, in float32, run longest first in padded-token-budgeted chunks."""
        model, _ = self._check_ready()
        order = sorted(range(len(rows)), key=lambda r: len(rows[r]), reverse=True)
        logits = torch.empty((len(rows), len(self._labels)), dtype=torch.float32)
        for start, count in _chunks([len(rows[r]) for r in order], self._max_forward_tokens):
            longest = len(rows[order[start]])
            chunk = order[start : start + count]
            input_ids = torch.full((count, longest), self._pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((count, longest), dtype=torch.long)
            for j, r in enumerate(chunk):
                input_ids[j, : len(rows[r])] = torch.tensor(rows[r], dtype=torch.long)
                attention_mask[j, : len(rows[r])] = 1
            with torch.inference_mode():
                out = model(input_ids=input_ids.to(self._device), attention_mask=attention_mask.to(self._device))
            logits[chunk] = out.logits.float().cpu()
        return logits

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
        """Score every item against the checkpoint's classes."""
        _ = prepared_items
        self._check_ready()
        if output_schema is not None or instruction is not None:
            raise InvalidInputError("A fixed-head classifier takes no output_schema or instruction")
        selected = self._selected_classes(labels)
        multi_label, top_k, threshold, error_on_overflow = self._request_options(options)

        errors: list[ExtractItemError | None] = [None] * len(items)
        readable: list[int] = []
        for i, item in enumerate(items):
            if isinstance(item.text, str):
                readable.append(i)
            else:
                errors[i] = ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message="Item must have text")
        rows, overflowed = self._encode([items[i].text or "" for i in readable])
        kept: list[int] = []
        kept_rows: list[list[int]] = []
        for i, row, overflow in zip(readable, rows, overflowed, strict=True):
            if overflow and error_on_overflow:
                errors[i] = ExtractItemError(
                    code=ErrorCode.INPUT_TOO_LONG.value,
                    message=(
                        f"The text does not fit whole in the model's {self._max_length}-token window and "
                        "options.overflow_policy is 'error'. Shorten the text or send overflow_policy 'truncate_text'."
                    ),
                )
                continue
            kept.append(i)
            kept_rows.append(row)

        classifications: list[list[Classification]] = [[] for _ in items]
        counts = [0] * len(items)
        if kept_rows:
            logits = self._forward(kept_rows)
            scores = torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)
            for i, row, item_scores in zip(kept, kept_rows, scores.tolist(), strict=True):
                classifications[i] = self._rank(item_scores, selected, top_k, threshold)
                counts[i] = len(row)
        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=classifications,
            errors=errors if any(error is not None for error in errors) else None,
            input_token_counts=counts,
        )

    def _rank(
        self, scores: list[float], selected: list[int] | None, top_k: int | None, threshold: float
    ) -> list[Classification]:
        indices = range(len(self._labels)) if selected is None else selected
        ranked = sorted((i for i in indices if scores[i] >= threshold), key=lambda i: (-scores[i], i))
        if top_k is not None:
            ranked = ranked[:top_k]
        return [Classification(label=self._labels[i], score=float(scores[i])) for i in ranked]

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        """Batching cost per item: the characters the tokenizer reads, after the cut. Never raises."""
        _ = labels, output_schema, instruction, options
        if not self._cut_text or self._max_length <= 0:
            return None
        try:
            return [
                len(self._cut(item.text, _CUT_CHARS_PER_TOKEN)) if isinstance(item.text, str) else 0 for item in items
            ]
        except Exception:  # noqa: BLE001 -- a cost estimate must never fail a request
            return None


def _chunks(lengths: list[int], budget: int) -> list[tuple[int, int]]:
    """Split rows sorted longest first into forward passes: (start, count) pairs.

    A pass holds at most ``budget`` padded tokens. Once it holds
    ``_MIN_CHUNK_TOKENS`` real tokens, it also ends before a row that would
    make more than ``_MAX_PADDING`` of it padding, so short rows are not padded
    to the length of long ones.
    """
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(lengths):
        longest = max(1, lengths[start])
        count, real = 1, lengths[start]
        while start + count < len(lengths):
            following = lengths[start + count]
            padded = longest * (count + 1)
            if padded > budget:
                break
            if real >= _MIN_CHUNK_TOKENS and real + following < (1.0 - _MAX_PADDING) * padded:
                break
            count += 1
            real += following
        chunks.append((start, count))
        start += count
    return chunks


def _quoted(value: object) -> str:
    text = repr(value)
    return text if len(text) <= _MAX_QUOTED_CHARS else f"{text[: _MAX_QUOTED_CHARS - 3]}..."
