"""Laya typed-decision adapter (``extract``).

Laya is a non-generative decision model: given a *state* (text, or a JSON dict
or conversation list) and typed questions — ``choice`` (pick one of N options),
``noul`` (yes/no), ``score`` (ordinal rubric) — it returns answer distributions
from one encoder row per (state, question) pair. They are temperature-calibrated
when the checkpoint ships calibration temperatures; ``laya-multilingual`` ships
none, so its probabilities are the raw softmax.

Request contract:

* Each extract item is one state: ``Item(text=...)``, or
  ``Item(metadata={"state": <dict | list | str>})`` for structured states. Dict
  states are serialized with ``json.dumps(ensure_ascii=False)``; list states
  (chronological turns) are truncated from the left so the newest turn survives.
* ``output_schema`` is Laya's native question mapping
  ``{question_id: {"type", "instructions", "criteria", "labels"?}}``; answers come
  back in ``data[question_id]``.
* Alternatively ``labels=[...]`` (with an optional ``instruction`` as the question
  text) runs one ``choice`` question over the labels and returns every label's
  probability in ``classifications``, like the other zero-shot classifiers.
* Limits, checked before anything is tokenized: at most ``MAX_QUESTIONS`` (64)
  questions and ``MAX_OPTIONS`` (1024) answer options or labels per request, and
  at most ``MAX_ITEM_TOKENS`` (32768) tokens per item, counted as questions x
  ``max_len``: 32 questions at ``max_len`` 1024, 64 at 512. A caller-set
  ``head_max_len`` must not exceed ``max_len``.
* Long inputs: a row reads at most ``max_len`` tokens, so before tokenizing,
  states are cut to 32 UTF-8 bytes per token the row can use and instructions
  and options to 16 (JSON states are rendered only that far). Rows equal the
  reference's for inputs within those budgets and for longer text in any
  script. A longer text that averages more bytes per token than its budget over
  the cut can keep fewer tokens than the reference would; that takes long runs
  of one character, which ModernBERT merges into tokens of up to 512 characters.

Metering counts every row the model encodes: an item's input tokens are the sum
of its (state, question) row lengths, so cost scales with items x questions.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.laya.model import LayaDecisionHead, LayaModel, apply_rope_parameters, split_checkpoint
from sie_server.adapters.laya.questions import (
    QTYPES,
    STATE_BYTES_PER_TOKEN,
    Calibration,
    Question,
    QuestionPrefix,
    assemble_row,
    build_prefixes,
    calibrated_probs,
    clip_text,
    decode_answer,
    parse_questions,
    serialize_state,
    shown,
)
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Classification, ErrorCode

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Files a Laya checkpoint needs; the Hub repos also carry README assets and eval
# artifacts that serving never reads.
_CHECKPOINT_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")
# Token-embedding table in the encoder state dict.
_EMBEDDING_KEY = "embeddings.tok_embeddings.weight"

# Batching-cost estimate: characters per row token. A row holds at most max_len
# tokens, so state or question text beyond max_len * this many characters is
# truncated away and adds no work.
_COST_CHARS_PER_TOKEN = 4

# Upper bound on questions per request. Every question adds one full encoder row
# per state, so an unbounded mapping is an uncapped compute vector (the upstream
# laya HTTP server applies the same limit).
MAX_QUESTIONS = 64
# Upper bound on answer options (or labels) across a request's questions, checked
# before anything is tokenized.
MAX_OPTIONS = 1024
# Upper bound on the row tokens one item can occupy: questions x max_len (one row
# of up to max_len tokens per question). A queue batch of items has to finish well
# inside the work queue's acknowledgement window; this allows 32 questions per
# item at max_len 1024, and all 64 at 512.
MAX_ITEM_TOKENS = 32_768

DEFAULT_LABELS_INSTRUCTION = "Which category does this text belong to?"
_MIN_BF16_MAJOR = 8  # Ampere
_LABELS_QUESTION_ID = "label"

_ERR_NO_QUESTIONS = (
    "Laya requires typed questions in output_schema ({question_id: {type, instructions, criteria}}) "
    "or classification labels"
)
_ERR_SCHEMA_AND_LABELS = "Laya takes either output_schema (typed questions) or labels, not both"
_ERR_SCHEMA_AND_INSTRUCTION = (
    "Laya reads each question's own 'instructions' from output_schema; the request-level instruction "
    "applies only to labels"
)
_ERR_STATE = (
    "Laya items need a state: text, or metadata.state holding a string, a JSON object, or a list of turns (not both)"
)
_ERR_STATE_JSON = "Laya metadata.state must be JSON-serializable"
_ERR_TOO_MANY_OPTIONS = "Laya accepts at most {limit} {what} per request, got {count}"


class LayaAdapter(BaseAdapter):
    """Adapter for Laya decision models (``convaiinnovations/laya`` family)."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int | None = None,
        max_forward_tokens: int = 16384,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: Hugging Face repo id or local checkpoint directory
                (``rl_agent_config.json``, ``model.safetensors``, ``tokenizer/``, ``encoder/``).
            max_seq_length: Upper bound on tokens per (state, question) row. Defaults
                to the checkpoint's trained ``max_len``.
            max_forward_tokens: Padded-token budget per forward pass; rows beyond it
                run in further passes.
            compute_precision: Projection precision on CUDA (``bfloat16`` matches
                the reference autocast). CPU and MPS always run float32.
            revision: Hugging Face revision (commit SHA) to pin.
            **kwargs: Additional loader arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._max_seq_length = max_seq_length
        self._max_forward_tokens = max(1, int(max_forward_tokens))
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: LayaModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._calibration = Calibration(temperature=(1.0, 1.0, 1.0), temperature_by_options={})
        self._max_len = 512
        self._head_max_len = 192

    # ------------------------------------------------------------------ loading

    def _checkpoint_dir(self) -> Path:
        path = Path(self._model_name_or_path)
        if path.is_dir():
            return path
        return Path(
            snapshot_download(
                repo_id=self._model_name_or_path,
                revision=self._revision,
                allow_patterns=list(_CHECKPOINT_FILES),
            )
        )

    @staticmethod
    def _load_tokenizer(tokenizer_dir: Path) -> PreTrainedTokenizerFast:
        """Load the checkpoint tokenizer without rewriting any cached file.

        The checkpoints were saved by transformers 5. Their configs name
        ``PreTrainedTokenizerFast`` (so the class is loaded directly), and the
        mmBERT tokenizer stores ``extra_special_tokens`` as a list, which
        transformers 4.x cannot read; it is passed as the equivalent mapping.
        """
        from transformers import PreTrainedTokenizerFast

        overrides: dict[str, Any] = {}
        with (tokenizer_dir / "tokenizer_config.json").open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        extra = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra, list):
            overrides["extra_special_tokens"] = {f"extra_{i}": token for i, token in enumerate(extra)}
        return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir), **overrides)

    def _resolve_compute(self, device: str) -> tuple[bool, torch.dtype]:
        """Pick the backbone path and projection dtype for ``device``."""
        if not device.startswith("cuda"):
            # The reference runs float32 off-CUDA; reduced precision on CPU is slower and lossy.
            return False, torch.float32
        dtype = self._resolve_dtype()
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability(torch.device(device))[0] < _MIN_BF16_MAJOR:
            dtype = torch.float16  # pre-Ampere GPUs have no fast bf16 (the reference does the same)
        if dtype == torch.float32:
            return False, dtype  # flash attention has no float32 kernels
        from sie_server.core.inference import is_flash_attention_available

        return is_flash_attention_available(device), dtype

    def load(self, device: str) -> None:
        """Load the checkpoint onto ``device``."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel

        try:
            from transformers.initialization import no_init_weights  # ty: ignore[unresolved-import]
        except ImportError:  # transformers 4.x
            from transformers.modeling_utils import no_init_weights

        checkpoint_dir = self._checkpoint_dir()
        with (checkpoint_dir / "rl_agent_config.json").open(encoding="utf-8") as f:
            cfg = json.load(f)

        trained_max_len = int(cfg.get("max_len", 512))
        self._max_len = min(trained_max_len, self._max_seq_length) if self._max_seq_length else trained_max_len
        self._head_max_len = int(cfg.get("head_max_len", 192))
        self._calibration = Calibration.from_config(cfg)

        tokenizer = self._load_tokenizer(checkpoint_dir / "tokenizer")

        encoder_config = AutoConfig.from_pretrained(str(checkpoint_dir / "encoder"))
        apply_rope_parameters(encoder_config)
        encoder_config.reference_compile = False
        with no_init_weights():
            encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
            head = LayaDecisionHead(encoder_config.hidden_size, int(cfg.get("head_layers", 2)))

        encoder_weights, head_weights = split_checkpoint(load_file(str(checkpoint_dir / "model.safetensors")))
        encoder.load_state_dict(encoder_weights, strict=True)
        head.load_state_dict(head_weights, strict=True)
        embedding_dtype = encoder_weights[_EMBEDDING_KEY].dtype
        del encoder_weights, head_weights

        use_flash, compute_dtype = self._resolve_compute(device)
        model = LayaModel(encoder, head)
        model.prepare(
            device,
            use_flash=use_flash,
            compute_dtype=compute_dtype,
            max_positions=self._max_len,
            embedding_dtype=embedding_dtype,
        )

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        logger.info(
            "Loaded Laya %s on %s (backbone=%s, compute=%s, max_len=%d, head_max_len=%d)",
            self._model_name_or_path,
            device,
            "flash-varlen" if use_flash else "sdpa",
            compute_dtype,
            self._max_len,
            self._head_max_len,
        )

    def warmup(self) -> None:
        """Run one tiny decision to initialize kernels."""
        self.extract(
            [Item(text="warmup")],
            output_schema={"warmup": {"type": "noul", "instructions": "Is this a warmup?"}},
        )

    # ------------------------------------------------------------------ request parsing

    def _resolve_questions(
        self,
        labels: list[str] | None,
        output_schema: dict[str, Any] | None,
        instruction: str | None,
    ) -> tuple[list[Question], list[str] | None]:
        """Return (questions, caller labels); the labels are None unless ``labels`` drives the request.

        In labels mode the model reads each label with surrounding whitespace
        stripped, while ``classifications`` echo the caller's labels unchanged.
        """
        if output_schema is not None:
            if labels:
                raise InvalidInputError(_ERR_SCHEMA_AND_LABELS)
            if instruction is not None:
                raise InvalidInputError(_ERR_SCHEMA_AND_INSTRUCTION)
            if not isinstance(output_schema, Mapping):
                raise InvalidInputError("Laya output_schema must map question ids to question definitions")
            if not output_schema:
                raise InvalidInputError(_ERR_NO_QUESTIONS)
            if len(output_schema) > MAX_QUESTIONS:
                raise InvalidInputError(
                    f"Laya accepts at most {MAX_QUESTIONS} questions per request, got {len(output_schema)}"
                )
            num_options = sum(_declared_options(qdef) for qdef in output_schema.values())
            if num_options > MAX_OPTIONS:
                raise InvalidInputError(
                    _ERR_TOO_MANY_OPTIONS.format(limit=MAX_OPTIONS, what="answer options", count=num_options)
                )
            return parse_questions(output_schema), None
        if not labels:
            raise InvalidInputError(_ERR_NO_QUESTIONS)
        if not isinstance(labels, list):
            raise InvalidInputError("Laya labels must be non-empty strings")
        if len(labels) > MAX_OPTIONS:
            raise InvalidInputError(_ERR_TOO_MANY_OPTIONS.format(limit=MAX_OPTIONS, what="labels", count=len(labels)))
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise InvalidInputError("Laya labels must be non-empty strings")
        normalized = [label.strip() for label in labels]
        if len(set(normalized)) != len(normalized):
            raise InvalidInputError("Laya labels must be unique")
        question_text = instruction if instruction is not None else DEFAULT_LABELS_INSTRUCTION
        question = {"type": "choice", "instructions": question_text, "criteria": normalized}
        return parse_questions({_LABELS_QUESTION_ID: question}), list(labels)

    def _token_budget(self, options: Mapping[str, Any]) -> tuple[int, int]:
        """Resolve (max_len, head_max_len) from runtime options."""
        max_len = _positive_int_option(options, "max_len", self._max_len)
        if max_len > self._max_len:
            raise InvalidInputError(
                f"Laya max_len must be at most {self._max_len} for this model, got {shown(max_len)}"
            )
        head_max_len = _positive_int_option(options, "head_max_len", self._head_max_len)
        if options.get("head_max_len") is not None and head_max_len > max_len:
            raise InvalidInputError(f"Laya head_max_len must be at most max_len ({max_len}), got {shown(head_max_len)}")
        return max_len, head_max_len

    @staticmethod
    def _check_item_tokens(num_questions: int, max_len: int) -> None:
        """Reject requests whose rows could exceed ``MAX_ITEM_TOKENS`` for one item."""
        if num_questions * max_len > MAX_ITEM_TOKENS:
            raise InvalidInputError(
                f"Laya encodes each item once per question, up to max_len={max_len} tokens per row, "
                f"and accepts at most {MAX_ITEM_TOKENS} tokens per item: send at most "
                f"{MAX_ITEM_TOKENS // max_len} questions, or a smaller max_len option (got {num_questions} questions)"
            )

    @staticmethod
    def _item_state(item: Item, limit: int | None = None) -> tuple[str, bool]:
        """Return (serialized state, truncate_left) for an item.

        With ``limit``, only the part of the state a row can read is rendered:
        its first ``limit`` UTF-8 bytes, or its last for a conversation list.

        Raises:
            InvalidInputError: When the item has no usable state.
        """
        metadata = item.metadata or {}
        has_state = "state" in metadata
        if has_state == (item.text is not None):
            raise InvalidInputError(_ERR_STATE)
        if not has_state:
            text = item.text or ""
            return (text if limit is None else clip_text(text, limit)), False
        state = metadata["state"]
        if not isinstance(state, (str, Mapping, list)):
            raise InvalidInputError(_ERR_STATE)
        # Conversation turns are chronological: keep the newest on truncation.
        truncate_left = isinstance(state, list)
        try:
            return serialize_state(state, limit, from_end=truncate_left), truncate_left
        except Exception as exc:  # caller data that cannot be rendered fails only its item
            raise InvalidInputError(_ERR_STATE_JSON) from exc

    # ------------------------------------------------------------------ extract

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
        """Answer the typed questions (or classify over ``labels``) for every item."""
        _ = prepared_items
        model, tokenizer = self._model, self._tokenizer
        if model is None or tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        opts = options or {}
        questions, caller_labels = self._resolve_questions(labels, output_schema, instruction)
        labels_mode = caller_labels is not None
        threshold = _threshold_option(opts) if labels_mode else 0.0
        max_len, head_max_len = self._token_budget(opts)
        self._check_item_tokens(len(questions), max_len)

        state_bytes = self._max_len * STATE_BYTES_PER_TOKEN
        states: list[tuple[str, bool] | None] = []
        errors: list[ExtractItemError | None] = []
        for item in items:
            try:
                states.append(self._item_state(item, state_bytes))
                errors.append(None)
            except InvalidInputError as exc:
                states.append(None)
                errors.append(ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=str(exc)))

        valid = [i for i, s in enumerate(states) if s is not None]
        prefixes, rows = self._build_rows(
            [s for s in states if s is not None], questions, max_len=max_len, head_max_len=head_max_len
        )
        logits = self._run_rows(rows, [prefixes[r % len(prefixes)] for r in range(len(rows))]) if rows else []

        data: list[dict[str, Any]] = [{} for _ in items]
        classifications: list[list[Classification]] = [[] for _ in items]
        token_counts = [0] * len(items)
        row = 0
        for i in valid:
            answers: dict[str, Any] = {}
            for prefix in prefixes:
                q = prefix.question
                token_counts[i] += len(rows[row])
                if caller_labels is not None:
                    classifications[i] = _classifications(caller_labels, q, logits[row], self._calibration, threshold)
                else:
                    answers[q.qid] = decode_answer(q, logits[row], self._calibration)
                row += 1
            if not labels_mode:
                data[i] = answers

        has_errors = any(e is not None for e in errors)
        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=classifications if labels_mode else None,
            data=None if labels_mode else data,
            errors=errors if has_errors else None,
            input_token_counts=token_counts,
        )

    def _build_rows(
        self,
        states: list[tuple[str, bool]],
        questions: list[Question],
        *,
        max_len: int,
        head_max_len: int,
    ) -> tuple[list[QuestionPrefix], list[list[int]]]:
        """Tokenize each question once and each state once, then assemble the rows.

        Returns the question prefixes and one row per (state, question) pair,
        state-major (row ``s * len(questions) + q``), like the reference batch.
        """
        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        mask_token = tokenizer.mask_token
        state_bytes = self._max_len * STATE_BYTES_PER_TOKEN
        with self._tokenizer_guard():
            prefixes = build_prefixes(tokenizer, questions, max_len=max_len, head_max_len=head_max_len)
            if not prefixes or not states:
                return prefixes, []
            texts = [
                clip_text(text, state_bytes, from_end=truncate_left).replace(mask_token, " ")
                for text, truncate_left in states
            ]
            state_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        rows = [
            assemble_row(
                prefix,
                ids,
                sep_token_id=tokenizer.sep_token_id,
                max_len=max_len,
                truncate_left=truncate_left,
            )
            for ids, (_, truncate_left) in zip(state_ids, states, strict=True)
            for prefix in prefixes
        ]
        return prefixes, rows

    def _run_rows(self, rows: list[list[int]], row_prefix: list[QuestionPrefix]) -> list[np.ndarray]:
        """Score every row's option markers, in token-budgeted chunks of similar length."""
        model, device, tokenizer = self._model, self._device, self._tokenizer
        if model is None or device is None or tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        order = sorted(range(len(rows)), key=lambda r: len(rows[r]), reverse=True)
        results: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * len(rows)
        start = 0
        while start < len(order):
            # Longest row first: the chunk's padded size is len(first) * n_rows.
            longest = len(rows[order[start]])
            n = max(1, min(len(order) - start, self._max_forward_tokens // max(1, longest)))
            chunk = order[start : start + n]
            chunk_rows = [rows[r] for r in chunk]
            markers = [row_prefix[r].markers for r in chunk]
            max_options = max(len(m) for m in markers)
            marker_pos = torch.zeros((len(chunk), max_options), dtype=torch.long)
            for j, m in enumerate(markers):
                marker_pos[j, : len(m)] = torch.tensor(m, dtype=torch.long)
            qtype = torch.tensor([QTYPES[row_prefix[r].question.qtype] for r in chunk], dtype=torch.long)
            marker_pos, qtype = marker_pos.to(device), qtype.to(device)
            with torch.inference_mode():
                if model.use_flash:
                    out = model.forward_flash(chunk_rows, marker_pos, qtype, device)
                else:
                    out = model.forward_padded(
                        chunk_rows,
                        marker_pos,
                        qtype,
                        device,
                        pad_token_id=int(tokenizer.pad_token_id),
                    )
            out_np = out.float().cpu().numpy()
            for j, r in enumerate(chunk):
                results[r] = out_np[j, : len(markers[j])]
            start += n
        return results

    # ------------------------------------------------------------------ cost

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        """Batching cost per item: its state characters once per question, plus the question text.

        The model encodes the state again for every question, so the generic
        per-item character count would undercount a request by its question
        count. This runs before batching, on the event loop, so it only
        estimates: structured states and questions are sized by a walk that
        stops at the characters a row can hold (``max_len`` tokens), without
        serializing them. Metered ``usage`` counts the rows extract() encodes.
        Best-effort: never raises (malformed requests fail in extract()).
        """
        _ = options
        row_chars = self._max_len * _COST_CHARS_PER_TOKEN
        try:
            if isinstance(output_schema, Mapping):
                if len(output_schema) > MAX_QUESTIONS:
                    return None  # extract() rejects the request
                num_rows = max(1, len(output_schema))
                question_chars = _approx_json_chars(output_schema, num_rows * row_chars)
            else:
                num_rows = 1
                label_chars = sum(len(label) for label in islice(labels or [], MAX_OPTIONS) if isinstance(label, str))
                question_chars = min(label_chars + len(instruction or DEFAULT_LABELS_INSTRUCTION), row_chars)
            return [num_rows * _state_chars(item, row_chars) + question_chars for item in items]
        except Exception:  # noqa: BLE001 — cost estimation must never fail a request
            return None


_END = object()


def _declared_options(qdef: Any) -> int:
    """Options a raw question definition declares, counted without walking its criteria."""
    if not isinstance(qdef, Mapping):
        return 0
    if qdef.get("type") == "noul":
        return 2
    criteria = qdef.get("criteria")
    return len(criteria) if isinstance(criteria, (Mapping, list)) else 0


def _approx_json_chars(value: Any, limit: int) -> int:
    """Approximate ``len(json.dumps(value))``, capped at ``limit``.

    Visits nested values lazily and stops once the running total reaches
    ``limit``; every visit adds at least two characters, so the work is bounded
    by the cap rather than by the size of ``value``.
    """
    total = 0
    stack: list[Iterator[Any]] = [iter((value,))]
    while stack and total < limit:
        node = next(stack[-1], _END)
        if node is _END:
            stack.pop()
        elif isinstance(node, str):
            total += len(node) + 4  # quotes and a separator
        elif isinstance(node, Mapping):
            total += 2
            stack.append(chain.from_iterable(node.items()))
        elif isinstance(node, (list, tuple)):
            total += 2
            stack.append(iter(node))
        else:
            total += 6  # a number, boolean, or null
    return min(total, limit)


def _state_chars(item: Item, limit: int) -> int:
    """Approximate characters of an item's state (text or ``metadata.state``), capped at ``limit``."""
    metadata = item.metadata or {}
    if "state" in metadata:
        state = metadata["state"]
        return min(len(state), limit) if isinstance(state, str) else _approx_json_chars(state, limit)
    return min(len(item.text or ""), limit)


def _positive_int_option(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidInputError(f"Laya option {name!r} must be a positive integer")
    return value


def _threshold_option(options: Mapping[str, Any]) -> float:
    value = options.get("threshold", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError("Laya threshold must be a finite number between 0 and 1")
    try:
        threshold = float(value)
    except OverflowError:  # an int too large for a float
        threshold = math.inf
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InvalidInputError("Laya threshold must be a finite number between 0 and 1")
    return threshold


def _classifications(
    labels: Sequence[str],
    q: Question,
    logits: Sequence[float] | np.ndarray,
    calibration: Calibration,
    threshold: float,
) -> list[Classification]:
    """Every caller label with its probability, highest first (zero-shot classifier shape).

    ``labels`` are the caller's labels in the order of ``q``'s options.
    """
    p = calibrated_probs(np.asarray(logits), q.qtype, calibration)
    result = [
        Classification(label=label, score=float(score))
        for label, score in zip(labels, p, strict=True)
        if float(score) >= threshold
    ]
    result.sort(key=lambda c: c["score"], reverse=True)
    return result
