"""GLiNER2.5-Decide typed-decision adapter (``extract``).

Serves Fastino's GLiNER2.5-Decide classifiers through the ``gliner2`` 2.x
package: ``AutoExtractor`` loads the checkpoint (span or boundary architecture),
and the package's own processor builds the task prompt. Every label of every
question gets a probability.

Request contract (see ``decisions.py``): Laya's typed question mapping in
``output_schema`` (answers in ``data[question_id]``), GLiClass's
``options.label_groups`` (answers in ``data[group]`` and ``"group.label"``
``classifications``), or plain ``labels`` (every label in ``classifications``).
Items are texts, or Laya states in ``metadata.state`` (a string, a JSON object,
or a list of conversation turns, of which the newest are kept).

One row per item. The model reads all of a request's tasks and the document in
one sequence, ``(task prompts and labels) [SEP_TEXT] document``, and scores every
label from its ``[L]`` marker in a single forward pass, as
``gliner2.classification.Classifier`` does. A task's probabilities therefore
depend on the other tasks sent with it. The task prompt is built once per
request by the package's processor; each document's words are appended the way
the processor appends them (lowercased words, each tokenized on its own).

Window: the task prompt may take at most ``max_prompt_tokens`` (default 512, or
half of ``max_seq_length`` if less), and the document is cut to the whole words
that fit in the rest. Words are read lazily, so work stops at the window: a word
longer than ``_MAX_WORD_CHARS`` characters, or reaching past
``_MAX_CHARS_PER_TOKEN`` characters per token of the window, also ends the part
the model reads. An item none of whose words fits returns a per-item
``INPUT_TOO_LONG`` error, as does any item that does not fit whole when
``options.overflow_policy`` is ``"error"``.

Usage: an item's input tokens are its document tokens the model reads plus the
tokens of the caller's free text encoded with it (question instructions and
label descriptions), as the Laya and GLiClass adapters count instructions and
criteria. Task names and label names are not counted. Errored items count
nothing.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.gliner2.decisions import (
    DecisionRequest,
    answer,
    approx_json_chars,
    classifications,
    parse_request,
    probabilities,
    schema_chars,
    split_logits,
)
from sie_server.adapters.gliner2.words import linear_equivalent
from sie_server.adapters.laya.questions import STATE_BYTES_PER_TOKEN, serialize_state
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.overflow_policy import DEFAULT_OVERFLOW_POLICY, VALID_OVERFLOW_POLICIES
from sie_server.types.responses import Classification, ErrorCode

logger = logging.getLogger(__name__)

# Files a GLiNER2.5-Decide checkpoint needs; the Hub repos also carry a banner image and an agent skill file.
_CHECKPOINT_FILES = ("config.json", "encoder_config/*", "tokenizer*", "special_tokens_map.json", "model.safetensors")
_DEFAULT_WINDOW = 512
# Tokens the task prompt may take by default (at most half the window). Task
# and label names are not billed, so this bounds the unbilled work a request
# attaches to each item, as GLiFormer's max_prompt_tokens does.
_DEFAULT_MAX_PROMPT_TOKENS = 512
# Padded tokens per forward pass; larger requests run in several passes.
_DEFAULT_INFERENCE_BATCH_TOKENS = 16384
# A document word longer than this ends the part of the document the model reads.
_MAX_WORD_CHARS = 4096
# ...as does reaching past this many characters per token of the window. Prose
# spends about 5; text laid out with runs of spaces about 40.
_MAX_CHARS_PER_TOKEN = 64
# ...or reading this many words per token of the window: a word of characters
# the tokenizer drops takes no room.
_MAX_WORDS_PER_TOKEN = 4
# A run of non-space characters. gliner2's words never cross a space.
_RUN = re.compile(r"\S+")
# Batching-cost estimate: characters per window token.
_COST_CHARS_PER_TOKEN = 4
# Token ids of recent words, kept across requests. Only words of at most
# _CACHED_WORD_CHARS characters are kept, so the cache holds a few tens of MB at
# most; a longer word (a hash, an id, a base64 run) is tokenized each time.
_WORD_CACHE_SIZE = 16384
_CACHED_WORD_CHARS = 32
_SENTENCE_END = (".", "!", "?")

_ERR_ITEM = (
    "GLiNER2.5-Decide items need a non-blank text, or metadata.state holding a string, a JSON object, "
    "or a list of turns (not both)"
)
_ERR_STATE_JSON = "GLiNER2.5-Decide metadata.state must be JSON-serializable"
_ERR_NO_WORD_FITS = (
    "No word of the document fits in the {room} tokens the questions leave in the model's {window}-token window "
    "(a single word longer than {max_word} characters is not read)"
)
_ERR_DOES_NOT_FIT = (
    "The document does not fit in the {room} tokens the questions leave in the model's {window}-token window "
    "(overflow_policy 'error')"
)
_ERR_NON_FINITE = "GLiNER2.5-Decide returned non-finite scores for this item"

# ModernBERT RoPE bases: transformers 5 writes them per layer type in
# ``rope_parameters``; transformers 4 reads only these flat fields, with these defaults.
_ROPE_FIELDS = {"full_attention": "global_rope_theta", "sliding_attention": "local_rope_theta"}
_TRANSFORMERS4_ROPE_DEFAULTS = {"full_attention": 160000.0, "sliding_attention": 10000.0}
_ROPE_RELATIVE_TOLERANCE = 1e-3
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# Checkpoint compatibility
# ---------------------------------------------------------------------------


def _transformers_major() -> int:
    return int(transformers.__version__.split(".", 1)[0])


def declared_rope_thetas(encoder_config: Mapping[str, Any]) -> dict[str, float] | None:
    """The RoPE base a ModernBERT checkpoint declares for each layer type, or None for other encoders.

    Raises:
        ValueError: The checkpoint declares RoPE parameters this adapter cannot verify.
    """
    if encoder_config.get("model_type") != "modernbert":
        return None
    rope = encoder_config.get("rope_parameters")
    if rope is None:
        return {
            layer_type: float(encoder_config.get(field, _TRANSFORMERS4_ROPE_DEFAULTS[layer_type]))
            for layer_type, field in _ROPE_FIELDS.items()
        }
    if not isinstance(rope, Mapping) or not set(rope) <= set(_ROPE_FIELDS):
        raise ValueError(f"Unsupported ModernBERT rope_parameters {rope!r}")
    thetas: dict[str, float] = {}
    for layer_type in _ROPE_FIELDS:
        params = rope.get(layer_type)
        if params is None:
            continue
        theta = params.get("rope_theta") if isinstance(params, Mapping) else None
        if (
            not isinstance(params, Mapping)
            or params.get("rope_type", "default") != "default"
            or isinstance(theta, bool)
            or not isinstance(theta, (int, float))
        ):
            raise ValueError(f"Unsupported ModernBERT {layer_type} RoPE parameters {params!r}")
        thetas[layer_type] = float(theta)
    return thetas


def transformers4_encoder_config(encoder_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The encoder config with its RoPE bases where transformers 4 reads them, or None when nothing changes.

    transformers 5 saves ModernBERT's RoPE bases only in ``rope_parameters``;
    transformers 4 ignores that and runs sliding-window layers at its default
    base of 10000. GLiNER2.5-Decide-1B's Ettin encoder uses 160000 for both.

    Raises:
        ValueError: The layer layout cannot be expressed in transformers 4.
    """
    if _transformers_major() >= 5 or encoder_config.get("rope_parameters") is None:
        return None
    thetas = declared_rope_thetas(encoder_config)
    if not thetas:
        return None
    every = encoder_config.get("global_attn_every_n_layers", 3)
    layer_types = encoder_config.get("layer_types")
    if layer_types is not None:
        expected = [
            "full_attention" if index % every == 0 else "sliding_attention" for index in range(len(layer_types))
        ]
        if list(layer_types) != expected:
            raise ValueError(
                "ModernBERT layer_types do not follow global_attn_every_n_layers; transformers 4 cannot load them"
            )
    patched = dict(encoder_config)
    for layer_type, theta in thetas.items():
        patched[_ROPE_FIELDS[layer_type]] = theta
    return patched if patched != dict(encoder_config) else None


def transformers4_tokenizer_config(tokenizer_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The tokenizer config naming a class transformers 4 has, or None when nothing changes.

    transformers 5 records a generic fast tokenizer as ``TokenizersBackend``; its
    ``tokenizer.json`` loads unchanged as a ``PreTrainedTokenizerFast``.
    """
    if _transformers_major() >= 5 or tokenizer_config.get("tokenizer_class") != "TokenizersBackend":
        return None
    return {**tokenizer_config, "tokenizer_class": "PreTrainedTokenizerFast"}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def loadable_checkpoint(checkpoint: Path) -> Iterator[Path]:
    """``checkpoint``, or a temporary overlay of it with the configs transformers 4 needs.

    The overlay links every file and rewrites only the patched JSON files, so
    the cached checkpoint is never modified.
    """
    patches: dict[str, dict[str, Any]] = {}
    encoder = transformers4_encoder_config(_read_json(checkpoint / "encoder_config" / "config.json"))
    if encoder is not None:
        patches["encoder_config/config.json"] = encoder
    tokenizer_path = checkpoint / "tokenizer_config.json"
    if tokenizer_path.is_file():
        tokenizer = transformers4_tokenizer_config(_read_json(tokenizer_path))
        if tokenizer is not None:
            patches["tokenizer_config.json"] = tokenizer
    if not patches:
        yield checkpoint
        return
    with tempfile.TemporaryDirectory(prefix="sie-gliner2-decide-") as tmp:
        root = Path(tmp)
        for path in checkpoint.rglob("*"):
            if path.is_dir():
                continue
            relative = path.relative_to(checkpoint).as_posix()
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in patches:
                target.write_text(json.dumps(patches[relative], indent=2), encoding="utf-8")
            else:
                target.symlink_to(path.resolve())
        logger.info("Loading %s with transformers 4 compatible %s", checkpoint, ", ".join(sorted(patches)))
        yield root


def loaded_rope_thetas(encoder: torch.nn.Module, every: int) -> dict[str, list[float]]:
    """The RoPE base each rotary table of a loaded encoder encodes, by layer type.

    ``inv_freq[i] = theta ** (-i / n)`` for a table of ``n`` frequencies, so
    ``theta = inv_freq[1] ** -n``. Layers are typed by name (transformers 5) or
    by index (transformers 4: every ``every``-th layer is full attention).
    """
    observed: dict[str, list[float]] = {}
    for name, buffer in encoder.named_buffers():
        if not name.endswith("inv_freq") or name.endswith("original_inv_freq") or buffer.numel() < 2:
            continue
        if "full_attention" in name:
            layer_type = "full_attention"
        elif "sliding_attention" in name:
            layer_type = "sliding_attention"
        elif (match := _LAYER_INDEX.search(name)) is not None:
            layer_type = "full_attention" if int(match.group(1)) % every == 0 else "sliding_attention"
        else:
            continue
        theta = float(buffer[1].double().item() ** -buffer.numel())
        observed.setdefault(layer_type, []).append(theta)
    return observed


def verify_encoder_rope(encoder: torch.nn.Module, encoder_config: Mapping[str, Any]) -> dict[str, float] | None:
    """Check that a loaded ModernBERT encoder runs the RoPE bases its checkpoint declares.

    Run in float32, before any precision cast. Returns the verified bases, or
    None for encoders without rotary tables.

    Raises:
        RuntimeError: A layer type's rotary tables are missing or encode another base.
    """
    expected = declared_rope_thetas(encoder_config)
    if not expected:
        return None
    observed = loaded_rope_thetas(encoder, int(encoder_config.get("global_attn_every_n_layers", 3)))
    for layer_type, theta in expected.items():
        values = observed.get(layer_type)
        if not values:
            raise RuntimeError(f"Cannot find the {layer_type} RoPE tables of the loaded encoder to verify them")
        wrong = [value for value in values if abs(value - theta) > _ROPE_RELATIVE_TOLERANCE * theta]
        if wrong:
            raise RuntimeError(
                f"The loaded encoder's {layer_type} layers run RoPE base {wrong[0]:.1f}, but the checkpoint "
                f"declares {theta:.1f}"
            )
    return expected


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Prefix:
    """A request's task prompt: token ids up to and including ``[SEP_TEXT]``, and each label's marker."""

    ids: list[int]
    label_positions: list[int]
    billed_tokens: int


@dataclass(frozen=True, slots=True)
class _Document:
    """The part of an item's document the model reads."""

    ids: list[int]
    billed_tokens: int
    complete: bool


class GLiNER2DecideAdapter(BaseAdapter):
    """Typed decisions with GLiNER2.5-Decide checkpoints (``gliner2`` 2.x).

    Reference models:
    - fastino/GLiNER2.5-Decide
    - fastino/GLiNER2.5-multi-Decide
    - fastino/GLiNER2.5-Decide-1B
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_tokenizer", "_word_splitter", "_word_ids", "_word_cache"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int | None = None,
        max_prompt_tokens: int | None = None,
        inference_batch_tokens: int = _DEFAULT_INFERENCE_BATCH_TOKENS,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: Hugging Face repo id or local checkpoint directory.
            max_seq_length: Tokens per row: task prompt, ``[SEP_TEXT]``, and document.
            max_prompt_tokens: Most tokens the task prompt may take (default: 512,
                or half of ``max_seq_length`` when that is less); a larger request
                fails with ``INPUT_TOO_LONG``.
            inference_batch_tokens: Padded tokens per forward pass.
            compute_precision: Accelerator precision (``float16`` is the package's
                ``quantize=True``). CPU always runs float32.
            revision: Hugging Face revision (commit SHA) to pin.
            **kwargs: Additional loader arguments (ignored).
        """
        _ = kwargs
        window = _DEFAULT_WINDOW if max_seq_length is None else int(max_seq_length)
        prompt = min(window // 2, _DEFAULT_MAX_PROMPT_TOKENS) if max_prompt_tokens is None else int(max_prompt_tokens)
        if window < 2 or not 0 < prompt < window:
            raise ValueError("GLiNER2.5-Decide needs 0 < max_prompt_tokens < max_seq_length")
        if int(inference_batch_tokens) < window:
            raise ValueError("GLiNER2.5-Decide inference_batch_tokens must hold at least one full row")
        self._model_name_or_path = str(model_name_or_path)
        self._window = window
        self._max_prompt_tokens = prompt
        self._inference_batch_tokens = int(inference_batch_tokens)
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._word_splitter: Any = None
        self._word_ids: Any = None
        self._word_cache: Any = None
        self._device: str | None = None

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

    def load(self, device: str) -> None:
        """Load the checkpoint with ``gliner2.AutoExtractor`` onto ``device``."""
        try:
            from gliner2 import AutoExtractor  # ty:ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError("GLiNER2.5-Decide needs gliner2 2.x (AutoExtractor)") from exc

        checkpoint = self._checkpoint_dir()
        encoder_config = _read_json(checkpoint / "encoder_config" / "config.json")
        with loadable_checkpoint(checkpoint) as path:
            model = AutoExtractor.from_pretrained(str(path), map_location=device)
        if not hasattr(model, "classifier") or not hasattr(model, "encoder"):
            raise RuntimeError(f"{self._model_name_or_path} has no classification head")
        rope = verify_encoder_rope(model.encoder, encoder_config)
        model.eval()
        processor = model.processor
        processor.change_mode(is_training=False)

        dtype = self._resolve_dtype() if device.startswith("cuda") else torch.float32
        if dtype == torch.float16:
            model.quantize()  # the package's own half-precision path
        elif dtype == torch.bfloat16:
            model.to(dtype=torch.bfloat16)

        self._attach(model, processor, device)
        logger.info(
            "Loaded GLiNER2.5-Decide %s (%s) on %s: %s, window %d tokens, prompt at most %d, RoPE %s",
            self._model_name_or_path,
            type(model).__name__,
            device,
            dtype,
            self._window,
            self._max_prompt_tokens,
            rope,
        )

    def _attach(self, model: Any, processor: Any, device: str) -> None:
        """Use ``model`` (encoder and classification head) and ``processor`` (the task prompt builder).

        Raises:
            RuntimeError: The processor splits words with a splitter this adapter
                has no linear-time equivalent for.
        """
        splitter = linear_equivalent(processor.word_splitter)
        if splitter is None:
            raise RuntimeError(
                f"GLiNER2.5-Decide has no linear-time equivalent of {type(processor.word_splitter).__name__}; "
                "gliner2's word splitter changed"
            )
        processor.word_splitter = splitter
        tokenizer = processor.tokenizer

        def tokenize(word: str) -> tuple[int, ...]:
            return tuple(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)))

        cached = lru_cache(maxsize=_WORD_CACHE_SIZE)(tokenize)

        def word_ids(word: str) -> tuple[int, ...]:
            return cached(word) if len(word) <= _CACHED_WORD_CHARS else tokenize(word)

        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._word_splitter = splitter
        self._word_ids = word_ids
        self._word_cache = cached
        self._device = device

    def warmup(self) -> None:
        """Run one tiny decision to initialize kernels."""
        self.extract([Item(text="warmup")], labels=["yes", "no"])

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
        """Answer every task for every item. See the module docstring for the contract."""
        _ = prepared_items
        if self._model is None:
            raise RuntimeError(ERR_NOT_LOADED)
        opts = options or {}
        request = parse_request(labels=labels, output_schema=output_schema, instruction=instruction, options=opts)
        overflow_policy = opts.get("overflow_policy", DEFAULT_OVERFLOW_POLICY)
        if not isinstance(overflow_policy, str) or overflow_policy not in VALID_OVERFLOW_POLICIES:
            raise InvalidInputError(
                f"GLiNER2.5-Decide overflow_policy must be one of {sorted(VALID_OVERFLOW_POLICIES)}"
            )

        with self._tokenizer_guard():
            prefix = self._prefix(request)
            room = self._window - len(prefix.ids)
            documents: list[_Document | None] = []
            errors: list[ExtractItemError | None] = []
            for item in items:
                document, error = self._document(item, room, overflow_policy)
                documents.append(document)
                errors.append(error)

        kept = [(index, document) for index, document in enumerate(documents) if document is not None]
        rows = [prefix.ids + document.ids for _, document in kept]
        logits = self._score(rows, prefix.label_positions) if rows else np.empty((0, 0), dtype=np.float32)

        data: list[dict[str, Any]] = [{} for _ in items]
        item_classifications: list[list[Classification]] = [[] for _ in items]
        token_counts = [0] * len(items)
        for row, (index, document) in enumerate(kept):
            if not np.isfinite(logits[row]).all():
                errors[index] = ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_NON_FINITE)
                continue
            per_task = [
                probabilities(task, part)
                for task, part in zip(request.tasks, split_logits(request, logits[row]), strict=True)
            ]
            if request.mode != "labels":
                data[index] = {
                    task.key: answer(task, p, threshold=request.threshold)
                    for task, p in zip(request.tasks, per_task, strict=True)
                }
            if request.mode != "questions":
                item_classifications[index] = classifications(request, per_task, grouped=request.mode == "groups")
            token_counts[index] = document.billed_tokens + prefix.billed_tokens

        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=None if request.mode == "questions" else item_classifications,
            data=None if request.mode == "labels" else data,
            errors=errors if any(error is not None for error in errors) else None,
            input_token_counts=token_counts,
        )

    def _prefix(self, request: DecisionRequest) -> _Prefix:
        """The task prompt as the processor builds it, once per request.

        Raises:
            InputTooLongError: The task prompt takes more than ``max_prompt_tokens``.
        """
        record = self._processor.transform_and_format(".", request.model_schema())
        # The processor keeps every string it tokenizes in an LRU of 50,000
        # entries; task prompts with descriptions run to tens of KB, so drop
        # them rather than keep each request's prompt for the adapter's lifetime.
        cache_clear = getattr(getattr(self._processor, "_tokenize_cached", None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
        text_start = record.text_word_first_positions[0]
        ids = list(record.input_ids[:text_start])
        if len(ids) > self._max_prompt_tokens:
            raise InputTooLongError(
                f"GLiNER2.5-Decide questions and labels take {len(ids)} tokens; this model takes at most "
                f"{self._max_prompt_tokens} of its {self._window}-token window. Send fewer or shorter questions, "
                "labels, instructions, or descriptions."
            )
        positions: list[list[int]] = record.schema_special_positions
        if len(positions) != len(request.tasks) or any(
            len(task_positions) != len(task.labels) + 1
            for task_positions, task in zip(positions, request.tasks, strict=True)
        ):
            raise RuntimeError("GLiNER2.5-Decide prompt markers do not match the request's labels")
        billed = sum(len(self._tokenizer.tokenize(text)) for task in request.tasks for text in task.free_texts())
        return _Prefix(
            ids=ids,
            label_positions=[position for task_positions in positions for position in task_positions[1:]],
            billed_tokens=billed,
        )

    def _document(
        self, item: Item, room: int, overflow_policy: str
    ) -> tuple[_Document | None, ExtractItemError | None]:
        """The words of an item the model reads, or the item's error."""
        try:
            text, from_end, whole = self._item_text(item, room)
        except InvalidInputError as exc:
            return None, ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=str(exc))
        document = self._read(text, room, from_end=from_end, whole=whole)
        if not document.ids:
            message = _ERR_NO_WORD_FITS.format(room=room, window=self._window, max_word=_MAX_WORD_CHARS)
            return None, ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=message)
        if not document.complete and overflow_policy == "error":
            message = _ERR_DOES_NOT_FIT.format(room=room, window=self._window)
            return None, ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=message)
        return document, None

    @staticmethod
    def _item_text(item: Item, room: int) -> tuple[str, bool, bool]:
        """(text, read from the end, text is the whole document) for an item.

        A state is rendered only as far as a row can read: its first (for a list
        of turns, its last) ``STATE_BYTES_PER_TOKEN`` UTF-8 bytes per window token.

        Raises:
            InvalidInputError: The item has no usable text or state.
        """
        metadata = item.metadata or {}
        has_state = "state" in metadata
        if has_state == (item.text is not None):
            raise InvalidInputError(_ERR_ITEM)
        if not has_state:
            text = item.text or ""
            from_end, whole = False, True
        else:
            state = metadata["state"]
            if not isinstance(state, (str, Mapping, list)):
                raise InvalidInputError(_ERR_ITEM)
            from_end = isinstance(state, list)
            limit = room * STATE_BYTES_PER_TOKEN
            try:
                text = serialize_state(state, limit, from_end=from_end)
            except Exception as exc:  # caller data that cannot be rendered fails only its item
                raise InvalidInputError(_ERR_STATE_JSON) from exc
            whole = len(text.encode("utf-8", "surrogatepass")) < limit
        if not text or text.isspace():
            raise InvalidInputError(_ERR_ITEM)
        return text, from_end, whole

    def _read(self, text: str, room: int, *, from_end: bool, whole: bool) -> _Document:
        """The whole words of ``text`` whose tokens fit in ``room``, as the processor tokenizes them.

        The processor ends a text that does not end a sentence with ``"."`` and
        reads it as one more word; that word is read when it fits but not billed.
        Words come from a linear-time splitter, read lazily from the start; from
        the end (conversation turns), run by run of non-space characters, each
        split on its own (no word crosses a space), keeping only the last
        ``_MAX_WORD_CHARS`` characters of a longer run and stopping after it.
        """
        normalized = text if text.endswith(_SENTENCE_END) else text + "."
        char_limit = room * _MAX_CHARS_PER_TOKEN
        max_words = room * _MAX_WORDS_PER_TOKEN
        words = self._words_from_end(normalized) if from_end else self._word_splitter(normalized, lower=True)
        kept: list[tuple[tuple[int, ...], int]] = []
        used = 0
        complete = whole
        for word, start, end in words:
            if not word:  # _words_from_end cut an overlong run here: older text is not read
                complete = False
                break
            reach = len(normalized) - start if from_end else end
            if end - start > _MAX_WORD_CHARS or reach > char_limit or len(kept) >= max_words:
                complete = False
                break
            ids = self._word_ids(word)
            if used + len(ids) > room:
                # Missing only the processor's "." still reads the caller's whole text.
                complete = complete and not from_end and start >= len(text)
                break
            kept.append((ids, start))
            used += len(ids)
        if from_end:
            kept.reverse()
        return _Document(
            ids=[token for ids, _ in kept for token in ids],
            billed_tokens=sum(len(ids) for ids, start in kept if start < len(text)),
            complete=complete,
        )

    def _words_from_end(self, text: str) -> Iterator[tuple[str, int, int]]:
        """The words of ``text``, last first.

        A run longer than ``_MAX_WORD_CHARS`` yields only its tail's words, then
        an empty word marking the cut (the splitter never yields an empty word).
        """
        for run in reversed([match.span() for match in _RUN.finditer(text)]):
            begin, end = run
            cut = max(begin, end - _MAX_WORD_CHARS)
            words = list(self._word_splitter(text[cut:end], lower=True))
            for word, start, stop in reversed(words):
                yield word, cut + start, cut + stop
            if cut > begin:
                yield "", cut, cut
                return

    def _score(self, rows: list[list[int]], label_positions: list[int]) -> np.ndarray:
        """Every row's label logits, ``[rows, labels]``, in padded chunks of similar length."""
        model, device = self._model, self._device
        if model is None or device is None:
            raise RuntimeError(ERR_NOT_LOADED)
        order = sorted(range(len(rows)), key=lambda row: len(rows[row]), reverse=True)
        out = np.empty((len(rows), len(label_positions)), dtype=np.float32)
        markers = torch.tensor(label_positions, dtype=torch.long, device=device)
        start = 0
        while start < len(order):
            longest = len(rows[order[start]])
            count = max(1, min(len(order) - start, self._inference_batch_tokens // longest))
            chunk = order[start : start + count]
            input_ids = torch.zeros((count, longest), dtype=torch.long)
            attention_mask = torch.zeros((count, longest), dtype=torch.long)
            for j, row in enumerate(chunk):
                input_ids[j, : len(rows[row])] = torch.tensor(rows[row], dtype=torch.long)
                attention_mask[j, : len(rows[row])] = 1
            with torch.inference_mode():
                hidden = model.encoder(
                    input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
                ).last_hidden_state
                logits = model.classifier(hidden[:, markers, :]).squeeze(-1)
            out[chunk] = logits.float().cpu().numpy()
            start += count
        return out

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
        """Batching cost per item: the characters of its one row, the task prompt included.

        A document is read only as far as the window, so its cost is capped
        there, and the task prompt is encoded again with every item. Runs before
        batching and validation; best-effort, never raises.
        """
        try:
            limit = self._window * _COST_CHARS_PER_TOKEN
            prompt = schema_chars(output_schema, labels, options or {}, limit)
            prompt = min(limit, prompt + (len(instruction) if isinstance(instruction, str) else 0))
            return [_item_chars(item, limit) + prompt for item in items]
        except Exception:  # noqa: BLE001 -- a cost estimate must never fail a request
            return None


def _item_chars(item: Item, limit: int) -> int:
    state = (item.metadata or {}).get("state")
    if state is not None and not isinstance(state, str):
        return approx_json_chars(state, limit)
    text = state if isinstance(state, str) else item.text
    return min(len(text or ""), limit)
