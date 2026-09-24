"""Adapter for GLiFormer multi-task extraction and embedding models."""

from __future__ import annotations

import copy
import logging
import math
import threading
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import Qwen3Config, Qwen3Model
from transformers.models.auto import modeling_auto

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters.gliformer.output_schema import (
    MAX_LABEL_CHARS,
    SchemaField,
    StructuredPlan,
    clip,
    compile_output_schema,
    shape_structured_output,
)
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS
from sie_server.core.inference_output import EncodeOutput, ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Classification, Entity, ErrorCode, Relation

logger = logging.getLogger(__name__)

_ERR_REQUIRES_TASK = "GLiFormer requires labels, label_groups, or output_schema for extraction"
_ERR_BLANK_TEXT = "GLiFormer requires non-blank text"
_ERR_PROMPT_EXHAUSTS_DOCUMENT = "GLiFormer task prompt leaves no document tokens within max_sequence_length"
_ERR_MALFORMED = "GLiFormer returned malformed {output}"
_ERR_ITEM_OUTPUT = "GLiFormer returned malformed or non-finite output for this item"
_ERR_RELATIONS_OPTION = "GLiFormer takes relation types as options.relation_labels, not options.relations"

# The model repositories ship a multi-megabyte demo animation next to the weights.
_SNAPSHOT_IGNORE_PATTERNS = ["*.gif"]

# Lowest threshold for requests that decode spans (entities, relations,
# output_schema fields). GLiFormer's span decoder pairs every start above the
# threshold with every end above it in Python, so a near-zero threshold turns
# decoding quadratic in words x labels (seconds at a few hundred words); the
# relation head decodes its entities the same way. Classification decoding is
# a sigmoid and a filter per label, so classification-only requests accept any
# threshold.
_MIN_SPAN_THRESHOLD = 0.1
# GLiFormer's classification decoder reads a threshold of 0 as "unset" and
# falls back to 0.5, so 0 is passed on as the smallest positive threshold.
_MIN_DECODER_THRESHOLD = 1e-6

# Padded prompt + document tokens per forward pass. Eager DeBERTa attention is
# quadratic in the sequence length, so a request is split into chunks whose
# padded size stays within this budget.
_DEFAULT_INFERENCE_BATCH_TOKENS = 16384

# Model outputs that the decoders turn into scores. Masked positions hold -inf.
_SCORE_OUTPUT_SUFFIXES = ("_logits", "_scores")

# Tokens the task prompt (labels, relation types, class labels, schema fields)
# may take. The prompt is not billed, so this bounds how much unbilled work a
# request can attach to each document: at most (512 + d) / d times the billed
# d document tokens, about 172x for a one-word document (d = 3 with the two
# special tokens).
_DEFAULT_MAX_PROMPT_TOKENS = 512

# Joint relation extraction scores every ordered pair of the entities it keeps,
# through a projection four times the hidden width. Keep the most confident 100
# entities per document (as the GLiNER relation adapter does) and at most
# 131072 candidate pairs per forward pass, about 2.4 GB of pair activations for
# the large model in float16.
_MAX_RELATION_ENTITIES = 100
_RELATION_PAIR_BUDGET = 131072
# The package decodes relations cell by cell over pairs x relation types.
_MAX_RELATION_TYPES = 20

# Entity spans a caller may supply per item for relation extraction.
_MAX_SUPPLIED_ENTITIES = MAX_EXTRACT_LABELS
# Distinct entity-type sets across a request's supplied entities. Each one is
# its own task prompt and its own forward passes.
_MAX_TYPE_GROUPS = 64

# Marks an item whose model output could not be used.
_ITEM_ERROR = "__gliformer_item_error__"

_IMPORT_LOCK = threading.Lock()


def _import_gliformer() -> ModuleType:
    """Import the ``gliformer`` package without its Auto-class side effects.

    Importing ``gliformer`` registers its bidirectional Qwen3 backbones as the
    ``AutoModel`` implementation for transformers' own Qwen3 configs. That
    registration is process-wide, so any other model loaded through
    ``AutoModel`` in the same worker would silently get the wrong architecture.
    GLiFormer resolves its backbones through its own registry, so only the
    registrations for configs that are new to transformers are kept.
    """
    with _IMPORT_LOCK:
        import gliformer  # ty:ignore[unresolved-import]

        _drop_builtin_auto_model_overrides()
    return gliformer


def _drop_builtin_auto_model_overrides() -> None:
    """Remove gliformer's ``AutoModel`` overrides of built-in configs, or fail.

    Raises:
        RuntimeError: The registrations cannot be inspected on this
            transformers version, or a built-in config still resolves to a
            gliformer class afterwards.
    """
    mapping = modeling_auto.MODEL_MAPPING
    registered = getattr(mapping, "_extra_content", None)
    reverse_config = getattr(mapping, "_reverse_config_mapping", None)
    builtin_models = getattr(mapping, "_model_mapping", None)
    if not isinstance(registered, dict) or not isinstance(reverse_config, Mapping) or builtin_models is None:
        raise RuntimeError("Cannot inspect the AutoModel registrations made by gliformer on this transformers version")
    overridden = [
        config_class
        for config_class, model_class in registered.items()
        if getattr(model_class, "__module__", "").startswith("gliformer.")
        and reverse_config.get(getattr(config_class, "__name__", "")) in builtin_models
    ]
    for config_class in overridden:
        del registered[config_class]

    expected = [(Qwen3Config, Qwen3Model)]
    qwen3_5_config = getattr(transformers, "Qwen3_5Config", None)
    qwen3_5_model = getattr(transformers, "Qwen3_5Model", None)
    if qwen3_5_config is not None and qwen3_5_model is not None:
        expected.append((qwen3_5_config, qwen3_5_model))
    for config_class, model_class in expected:
        if mapping[config_class] is not model_class:
            raise RuntimeError(f"AutoModel for {config_class.__name__} no longer resolves to {model_class.__name__}")
    for config_class in overridden:
        if getattr(mapping[config_class], "__module__", "").startswith("gliformer."):
            raise RuntimeError(f"AutoModel for {config_class.__name__} still resolves to a gliformer class")


class _NonFiniteScoresError(RuntimeError):
    """The model produced NaN or +inf scores for at least one document in a pass."""


def _upcast_score_outputs(_module: torch.nn.Module, _args: Any, output: Any) -> Any:
    """Forward hook: decode scores in float32 from host memory; reject NaN or +inf.

    The decoders apply a sigmoid to these logits. In float16 every logit
    above about 8.3 rounds to 1.0, which turns single-label classification
    into "first label wins". Upcasting before decoding keeps the ranking.
    A NaN would otherwise fail every threshold comparison and silently drop
    predictions. The decoders read span and relation scores one element at a
    time, so outputs are copied to host memory once, as on CPU inference,
    instead of synchronizing with the device for every element.
    """
    if not isinstance(output, Mapping):
        return output
    flags = []
    for key in list(output.keys()):
        value = output[key]
        if not (isinstance(key, str) and key.endswith(_SCORE_OUTPUT_SUFFIXES)):
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            continue
        if value.dtype != torch.float32:
            value = value.float()
            output[key] = value
        flags.append(torch.isnan(value).any() | torch.isposinf(value).any())
    if flags and bool(torch.stack(flags).any()):
        raise _NonFiniteScoresError("GLiFormer produced non-finite scores")
    for key in list(output.keys()):
        value = output[key]
        if isinstance(value, torch.Tensor):
            output[key] = value.cpu()
    return output


class GLiFormerAdapter(BaseAdapter):
    """Adapter for GLiFormer checkpoints (``gliformer`` pip package).

    One GLiFormer prompt carries every task in a request, so a single
    ``extract`` call can combine:

    - named entity recognition over ``labels``;
    - joint relation extraction with ``options["relation_labels"]`` (relation
      types, with ``labels`` as the entity types, as in the GLiNER adapter);
    - classification of ``labels`` under ``options["classification_task"]``;
    - several named classification questions with ``options["label_groups"]``,
      reported as ``"group.label"`` classifications like the GLiClass adapter
      (GLiFormer scores only the labels that pass the threshold, so no
      per-group distribution is returned in ``data``);
    - structured extraction into ``output_schema``: span-valued properties
      are filled by the structuring head (nested objects and record arrays
      included) and root string ``enum`` properties by classification groups.

    For compatibility with the other relation extractors, items whose
    ``metadata["entities"]`` carry entity spans switch ``labels`` to relation
    types, and relations are reported only between the supplied entities.

    ``options["threshold"]`` (default 0.5) applies to every task in the
    request. Requests that extract entities, relations, or span-valued
    ``output_schema`` fields need at least 0.1: below that, GLiFormer's span
    decoding grows quadratically with the document and label count.
    Classification-only requests (``classification_task``, ``label_groups``,
    or an ``output_schema`` of root enums only) accept any threshold from 0,
    where every question gets its best answer.

    ``options["relation_threshold"]`` raises the minimum score for relations
    only. GLiFormer decodes entities and relations with one threshold, so it
    cannot be lower than ``threshold``.

    Relations are scored among the 100 most confident entities of each
    document; pairs involving other entities are not reported. A request may
    ask for at most 20 relation types. Supplied ``metadata.entities`` may use
    at most 64 distinct sets of entity labels per request.

    Structured output follows GLiFormer's own formatting: a string property
    keeps the best of several extracted values, and array-of-string values
    are split on commas. A root ``required`` property that the model did not
    extract becomes a per-item error (GLiNER2 rejects the whole request),
    so the rest of the batch still completes. Malformed or non-finite model
    output for one document is also a per-item error. An errored item
    returns no entities, relations, classifications, or data.

    Billed input tokens are the document subwords that survive truncation;
    errored items bill nothing. The task prompt (labels, relation types,
    schema fields) is not billed, matching the GLiNER family, so it is
    bounded instead: labels, field names, and choices have at most 128
    characters, a request carries at most 1000 of them, and the prompt takes
    at most ``max_prompt_tokens`` (default 512) tokens.

    ``encode`` returns the checkpoint's text embedding head output.

    Reference models:
    - knowledgator/gliformer-base-v1
    - knowledgator/gliformer-large-v1
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json", "dense"),
        unload_fields=("_model", "_tokenizer", "_normalize_structures", "_build_formatter"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        normalize: bool = True,
        max_seq_length: int | None = None,
        inference_batch_tokens: int = _DEFAULT_INFERENCE_BATCH_TOKENS,
        max_prompt_tokens: int = _DEFAULT_MAX_PROMPT_TOKENS,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        dense_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Minimum confidence for every task, from 0 to 1 (at least
                0.1 for requests that extract entities, relations, or
                output_schema fields).
            flat_ner: Keep entity spans non-overlapping.
            multi_label: Allow several labels per span and per classification.
            normalize: L2-normalize ``encode`` embeddings.
            max_seq_length: Token budget shared by the task prompt and the
                document; longer documents are truncated.
            inference_batch_tokens: Padded tokens per forward pass (prompt +
                document for ``extract``); larger requests are split into
                chunks, for ``encode`` too.
            max_prompt_tokens: Most tokens the unbilled task prompt may take.
            compute_precision: Accelerator weight precision. Scores are always
                decoded in float32. CPU runs float32.
            revision: HuggingFace revision to pin when downloading artifacts.
            dense_dim: Catalog-declared embedding dimension, checked at load.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        for name, value in (
            ("inference_batch_tokens", inference_batch_tokens),
            ("max_prompt_tokens", max_prompt_tokens),
        ):
            if not _is_int(value) or value <= 0:
                raise ValueError(f"GLiFormer {name} must be a positive integer")
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = _validate_threshold(threshold)
        self._flat_ner = _validate_flag(flat_ner, "flat_ner")
        self._multi_label = _validate_flag(multi_label, "multi_label")
        self._normalize = _validate_flag(normalize, "normalize")
        self._max_seq_length = max_seq_length
        self._inference_batch_tokens = inference_batch_tokens
        self._max_prompt_tokens = max_prompt_tokens
        self._compute_precision = compute_precision
        self._revision = revision
        self._dense_dim = dense_dim

        self._model: Any = None
        self._tokenizer: Any = None
        self._normalize_structures: Callable[[Any], Any] | None = None
        self._build_formatter: Callable[[Any], Any] | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the checkpoint onto ``device``."""
        gliformer = _import_gliformer()

        model_path = self._model_name_or_path
        if not Path(model_path).is_dir():
            model_path = snapshot_download(
                repo_id=model_path,
                revision=self._revision,
                ignore_patterns=_SNAPSHOT_IGNORE_PATTERNS,
            )

        with warnings.catch_warnings():
            # The checkpoints request the optional flashdeberta kernels. Without
            # them GLiFormer uses its eager attention, which is what SIE serves.
            warnings.filterwarnings("ignore", message=".*flashdeberta.*")
            model = gliformer.GLiFormer.from_pretrained(
                str(model_path),
                load_tokenizer=True,
                map_location="cpu",
                max_length=self._max_seq_length,
            )

        # Reduced precision only pays off on accelerators; CPU keeps the
        # reference float32 numerics.
        dtype = torch.float32 if device == "cpu" else self._resolve_dtype()
        model.to(device=device, dtype=dtype)
        model.eval()
        model.model.register_forward_hook(_upcast_score_outputs)
        _limit_relation_entities(model.model)

        embedding_config = getattr(model.config, "embedding_config", None)
        embedding_dim = getattr(embedding_config, "projection_dim", None)
        if self._dense_dim is not None and embedding_dim != self._dense_dim:
            raise ValueError(
                f"GLiFormer embedding dimension mismatch: configured dense_dim={self._dense_dim}, "
                f"checkpoint projection_dim={embedding_dim}"
            )

        tokenizer = model.data_processor.transformer_tokenizer
        # ``embed_text`` truncates at the tokenizer limit, which the saved
        # tokenizer leaves unbounded; share the extraction budget instead.
        tokenizer.model_max_length = int(model.config.max_len)

        self._model = model
        self._tokenizer = tokenizer
        self._normalize_structures = gliformer.processing.schema.normalize_structuring_schemas
        self._build_formatter = gliformer.processing.schema.build_structuring_output_formatter
        self._device = device

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
        """Run every requested GLiFormer task over ``items`` in one prompt."""
        _ = instruction, prepared_items
        self._check_loaded()
        texts = [self._extract_text(item) for item in items]
        opts = options or {}
        if "relations" in opts:
            raise InvalidInputError(_ERR_RELATIONS_OPTION)
        threshold = _validate_threshold(opts.get("threshold", self._threshold))
        relation_threshold = opts.get("relation_threshold")
        if relation_threshold is not None:
            relation_threshold = _validate_threshold(relation_threshold, "relation_threshold")
        relation_labels = opts.get("relation_labels")
        flat_ner = _validate_flag(opts.get("flat_ner", self._flat_ner), "flat_ner")
        multi_label = _resolve_multi_label(opts, default=self._multi_label)
        request = _plan_request(
            labels=_validate_labels(labels, "labels") if labels else None,
            plan=compile_output_schema(output_schema) if output_schema is not None else None,
            classification_task=_validate_task(opts.get("classification_task")),
            label_groups=_validate_label_groups(opts.get("label_groups")),
            # An empty list asks for no relations, as in the GLiNER adapter.
            relation_types=(
                None if relation_labels in (None, []) else _validate_labels(relation_labels, "relation_labels")
            ),
            supplied_entities=self._supplied_relation_entities(items),
            relation_threshold=relation_threshold,
        )
        if request.decodes_spans and threshold < _MIN_SPAN_THRESHOLD:
            raise InvalidInputError(
                f"GLiFormer threshold must be at least {_MIN_SPAN_THRESHOLD} when extracting entities, relations, "
                "or output_schema fields; classification-only requests accept any threshold"
            )
        threshold = max(threshold, _MIN_DECODER_THRESHOLD)
        max_items = None
        if request.relations_requested:
            if len(request.relation_types or []) > _MAX_RELATION_TYPES:
                raise InvalidInputError(
                    f"GLiFormer relation extraction accepts at most {_MAX_RELATION_TYPES} relation types"
                )
            if relation_threshold is not None and relation_threshold < threshold:
                raise InvalidInputError(
                    "GLiFormer relation_threshold must be at least threshold: entities and relations are "
                    "decoded with one threshold, so relation_threshold can only raise it for relations"
                )
            max_items = max(1, _RELATION_PAIR_BUDGET // (_MAX_RELATION_ENTITIES * (_MAX_RELATION_ENTITIES - 1)))

        # Items are grouped by their entity types: relation extraction over
        # supplied entities prompts each item with the types it carries, in a
        # canonical order so that label order alone never adds a group.
        batches: dict[tuple[str, ...], list[int]] = {}
        for index in range(len(items)):
            types = (
                tuple(sorted({entity["label"] for entity in request.supplied_entities[index]}))
                if request.supplied_entities is not None
                else tuple(request.entity_types or ())
            )
            batches.setdefault(types, []).append(index)
            if len(batches) > _MAX_TYPE_GROUPS:
                raise InvalidInputError(
                    f"GLiFormer item metadata may use at most {_MAX_TYPE_GROUPS} distinct sets of entity labels "
                    "per request"
                )
        task_kwargs = {types: request.task_kwargs(list(types) if types else None) for types in batches}

        with self._tokenizer_guard():
            # Every group's prompt is built and measured before any per-item
            # work, so an over-budget request is rejected cheaply and before
            # any inference has run.
            prompt_tokens = {types: self._prompt_tokens(kwargs) for types, kwargs in task_kwargs.items()}
            document_tokens = self._document_tokens(texts)
        specials = int(self._tokenizer.num_special_tokens_to_add(pair=False))
        window = int(self._model.config.max_len)
        counts = [0] * len(items)
        lengths = [0] * len(items)
        for types, indices in batches.items():
            room = window - specials - prompt_tokens[types]
            for index in indices:
                counts[index] = min(document_tokens[index], room) + specials
                lengths[index] = prompt_tokens[types] + counts[index]

        raw_results: list[dict[str, Any]] = [{} for _ in items]
        structures = request.plan.structures if request.plan is not None else None
        for types, indices in batches.items():
            rows = self._run(
                [texts[index] for index in indices],
                task_kwargs[types],
                [lengths[index] for index in indices],
                structures=structures,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
                max_items=max_items,
            )
            for index, row in zip(indices, rows, strict=True):
                raw_results[index] = row

        return _assemble_output(texts, raw_results, request, input_token_counts=counts)

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Embed ``items`` with the checkpoint's embedding head."""
        _ = instruction, prepared_items
        self._check_loaded()
        unsupported = sorted(set(output_types) - {"dense"})
        if unsupported or "dense" not in output_types:
            raise InvalidInputError(
                f"GLiFormer encode supports only dense output, got {clip(str(sorted(output_types)))}"
            )
        texts = [self._extract_text(item) for item in items]
        normalize = _validate_flag((options or {}).get("normalize", self._normalize), "normalize")

        with self._tokenizer_guard():
            counts = [len(ids) for ids in self._tokenizer(texts, truncation=True)["input_ids"]]
            parts = []
            for chunk in _token_budget_chunks(counts, self._inference_batch_tokens):
                with torch.inference_mode():
                    parts.append(self._model.embed_text([texts[i] for i in chunk], batch_size=len(chunk)).float())
            embeddings = torch.cat(parts)
        if not bool(torch.isfinite(embeddings).all()):
            raise RuntimeError("GLiFormer produced non-finite embeddings")
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        dense = embeddings.numpy()
        output = EncodeOutput(dense=dense, batch_size=len(items), is_query=is_query, dense_dim=dense.shape[1])
        output.extra["input_token_counts"] = counts
        return output

    def _run(
        self,
        texts: list[str],
        task_kwargs: dict[str, Any],
        lengths: list[int],
        *,
        structures: dict[str, Any] | None,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """Run ``GLiFormer.inference`` in bounded chunks; one raw result per text."""
        formatter = self._build_formatter(structures) if structures and self._build_formatter else None
        rows: list[dict[str, Any]] = [{} for _ in texts]
        with self._tokenizer_guard():
            for chunk in _token_budget_chunks(lengths, self._inference_batch_tokens, max_items=max_items):
                chunk_rows = self._infer(
                    texts,
                    chunk,
                    task_kwargs,
                    threshold=threshold,
                    flat_ner=flat_ner,
                    multi_label=multi_label,
                )
                for index, row in zip(chunk, chunk_rows, strict=True):
                    rows[index] = _format_row(row, formatter)
        return rows

    def _infer(
        self,
        texts: list[str],
        chunk: list[int],
        task_kwargs: dict[str, Any],
        **inference_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run ``chunk``, isolating documents whose scores are non-finite.

        A non-finite score fails the whole pass. The chunk is then split in
        halves, following only the failing half, until the failing document
        runs alone: about 2 log2(n) extra passes for one bad document in n.
        When both halves fail, the failures are spread out, so each document
        is tried alone instead: at most n + 3 passes in all.
        """

        def attempt(part: list[int]) -> list[dict[str, Any]] | None:
            return self._forward(texts, part, task_kwargs, **inference_kwargs)

        def isolate(part: list[int]) -> list[dict[str, Any]]:
            if len(part) == 1:
                return [{_ITEM_ERROR: True}]
            middle = len(part) // 2
            left, right = part[:middle], part[middle:]
            left_rows, right_rows = attempt(left), attempt(right)
            if left_rows is None and right_rows is None:
                return [row for index in part for row in attempt([index]) or [{_ITEM_ERROR: True}]]
            return [*(left_rows or isolate(left)), *(right_rows or isolate(right))]

        rows = attempt(chunk)
        return rows if rows is not None else isolate(chunk)

    def _forward(
        self,
        texts: list[str],
        chunk: list[int],
        task_kwargs: dict[str, Any],
        **inference_kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        """One forward pass; ``None`` when it produced non-finite scores."""
        try:
            with torch.inference_mode():
                results = self._model.inference(
                    [texts[index] for index in chunk],
                    **task_kwargs,
                    **inference_kwargs,
                    batch_size=len(chunk),
                )
        except _NonFiniteScoresError:
            return None
        rows: list[dict[str, Any]] = [{} for _ in chunk]
        for task_name in ("ner", "classification", "joint_relex", "structuring"):
            task_results = results.get(task_name)
            if task_results is None:
                continue
            if not isinstance(task_results, list) or len(task_results) != len(chunk):
                raise RuntimeError(_ERR_MALFORMED.format(output=task_name))
            for row, value in zip(rows, task_results, strict=True):
                row[task_name] = value
        return rows

    def _prompt_tokens(self, task_kwargs: dict[str, Any]) -> int:
        """Build one task prompt with the package's processor and count its tokens.

        GLiFormer prepends the same prompt to every document of a task group
        and truncates the combined sequence to ``max_len``, so the prompt is
        measured once per group, not per document.

        Raises:
            InvalidInputError: The prompt exceeds ``max_prompt_tokens`` or
                leaves no room for a document.
            RuntimeError: The package's processor could not build the prompt.
        """
        model = self._model
        processor = model.data_processor
        try:
            kwargs = copy.deepcopy(task_kwargs)
            if kwargs.get("structures") is not None:
                # ``inference`` compiles templates into the processor's wire form first.
                if self._normalize_structures is None:
                    raise RuntimeError("structuring templates cannot be normalized")
                kwargs["structures"] = self._normalize_structures(kwargs["structures"])
            raw_batch = processor.collate_raw_batch(model._build_inference_input([["x"]], **kwargs))
            sequences, prompt_lengths = processor.prepare_inputs(raw_batch["tokens"], raw_batch["classes_mapping"])
            prompt_words = list(sequences[0][: prompt_lengths[0]])
            encoded = processor.transformer_tokenizer(
                [prompt_words], is_split_into_words=True, add_special_tokens=False
            )
            count = len(encoded["input_ids"][0])
        except Exception as exc:
            raise RuntimeError("GLiFormer could not build the task prompt") from exc
        if count > self._max_prompt_tokens:
            raise InvalidInputError(
                f"GLiFormer task prompt needs {count} tokens; labels, relation types, class labels, and "
                f"schema fields may take at most {self._max_prompt_tokens}"
            )
        specials = int(processor.transformer_tokenizer.num_special_tokens_to_add(pair=False))
        if count + specials >= int(model.config.max_len):
            raise InvalidInputError(_ERR_PROMPT_EXHAUSTS_DOCUMENT)
        return count

    def _document_tokens(self, texts: list[str]) -> list[int]:
        """Document subwords per text before the prompt's share of the window.

        Words are tokenized independently of each other, so the document's
        tokens are the same with or without the prompt in front of it.
        """
        model = self._model
        max_words = int(model.config.max_len)
        words, _, _ = model.prepare_inputs(texts)
        encoded = model.data_processor.transformer_tokenizer(
            [list(text_words[:max_words]) for text_words in words],
            is_split_into_words=True,
            add_special_tokens=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def _extract_text(self, item: Item) -> str:
        if item.text is None:
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="GLiFormer adapter"))
        if not item.text.strip():
            raise InvalidInputError(_ERR_BLANK_TEXT)
        return item.text

    @staticmethod
    def _supplied_relation_entities(items: list[Item]) -> list[list[Entity]] | None:
        """Return per-item entities from ``metadata["entities"]``, if supplied."""
        supplied = [item.metadata.get("entities") if item.metadata else None for item in items]
        if all(entities is None for entities in supplied):
            return None
        if not all(isinstance(entities, list) and entities for entities in supplied):
            raise InvalidInputError("GLiFormer relation extraction requires non-empty entities in every item metadata")
        if any(len(entities or []) > _MAX_SUPPLIED_ENTITIES for entities in supplied):
            raise InvalidInputError(f"GLiFormer item metadata may carry at most {_MAX_SUPPLIED_ENTITIES} entities")
        return [
            [_normalize_input_entity(item.text or "", entity) for entity in entities or []]
            for item, entities in zip(items, supplied, strict=True)
        ]


def _limit_relation_entities(model: Any) -> None:
    """Keep only the most confident entities as relation candidates.

    The package ranks the decoded entities and slices the kept ones into
    compact tensors before it builds entity pairs, so the pair tensors never
    grow past this many entities per document.
    """
    heads = getattr(model, "heads", None)
    if heads is None or "joint_relex" not in heads:
        raise RuntimeError("GLiFormer checkpoint has no joint relation head to bound")
    head = heads["joint_relex"]
    current = getattr(head, "max_relation_entities", None)
    head.max_relation_entities = _MAX_RELATION_ENTITIES if current is None else min(current, _MAX_RELATION_ENTITIES)


def _format_row(row: dict[str, Any], formatter: Any) -> dict[str, Any]:
    """Apply GLiFormer's typed output formatting, as ``GLiFormer.structure`` does.

    ``inference`` returns raw decoder values: a scalar field for which
    several spans were found holds all of them, best first. The formatter
    keeps the best value for scalar fields and normalizes list fields.
    """
    if formatter is None or "structuring" not in row or _ITEM_ERROR in row:
        return row
    try:
        return {**row, "structuring": formatter.format_batch([row["structuring"]])[0]}
    except Exception:  # noqa: BLE001 -- one document's output must not fail the batch
        logger.warning("GLiFormer structured output could not be formatted", exc_info=True)
        return {_ITEM_ERROR: True}


def _token_budget_chunks(lengths: list[int], budget: int, max_items: int | None = None) -> list[list[int]]:
    """Split positions into in-order chunks whose padded size fits ``budget``.

    A chunk is padded to its longest sequence, so its cost is
    ``len(chunk) * max(length)``. A single sequence longer than the budget
    runs alone. ``max_items`` additionally caps the documents per chunk.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index, length in enumerate(lengths):
        full = max_items is not None and len(current) >= max_items
        if current and (full or (len(current) + 1) * max(longest, length) > budget):
            chunks.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, length)
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class _RequestPlan:
    """How one extract request maps onto GLiFormer's task heads."""

    entity_types: list[str] | None
    relation_types: list[str] | None
    classes: dict[str, list[str]]
    classification_task: str | None
    label_groups: dict[str, list[str]] | None
    plan: StructuredPlan | None
    supplied_entities: list[list[Entity]] | None
    relation_threshold: float | None

    @property
    def relations_requested(self) -> bool:
        return self.relation_types is not None

    @property
    def decodes_spans(self) -> bool:
        """Whether any task decodes text spans (everything but classification)."""
        return (
            self.entity_types is not None
            or self.relations_requested
            or (self.plan is not None and self.plan.structures is not None)
        )

    def task_kwargs(self, entity_types: list[str] | None) -> dict[str, Any]:
        """``GLiFormer.inference`` task arguments for one entity-type group."""
        kwargs: dict[str, Any] = {
            "entities": None,
            "classes": self.classes or None,
            "joint_relations": None,
            "structures": self.plan.structures if self.plan is not None else None,
        }
        if self.relation_types is not None:
            # An unnamed group shares the plain NER prompt, so the joint head
            # returns the entity spans and the relations between them.
            kwargs["joint_relations"] = {None: {"entities": entity_types, "relations": self.relation_types}}
        else:
            kwargs["entities"] = entity_types
        return kwargs


def _plan_request(
    *,
    labels: list[str] | None,
    plan: StructuredPlan | None,
    classification_task: str | None,
    label_groups: dict[str, list[str]] | None,
    relation_types: list[str] | None,
    supplied_entities: list[list[Entity]] | None,
    relation_threshold: float | None,
) -> _RequestPlan:
    """Resolve what ``labels`` mean and which classification groups to run.

    ``labels`` are entity types unless ``classification_task`` makes them
    class labels or supplied ``metadata.entities`` make them relation types.
    """
    classes: dict[str, list[str]] = {}
    if classification_task is not None:
        if labels is None:
            raise InvalidInputError("GLiFormer classification_task requires labels")
        if label_groups is not None:
            raise InvalidInputError("GLiFormer label_groups cannot be combined with classification_task")
        classes[classification_task] = labels
    if label_groups is not None:
        if plan is not None:
            raise InvalidInputError(
                "GLiFormer label_groups cannot be combined with output_schema; declare enum properties instead"
            )
        classes.update(label_groups)
    if plan is not None:
        for name, choices in plan.choice_groups.items():
            if name in classes:
                raise InvalidInputError(
                    "GLiFormer classification_task must differ from output_schema enum property names"
                )
            classes[name] = choices

    entity_types: list[str] | None = None
    if supplied_entities is not None:
        if classification_task is not None or relation_types is not None:
            raise InvalidInputError(
                "GLiFormer item metadata.entities cannot be combined with classification_task or relation_labels"
            )
        if labels is None:
            raise InvalidInputError("GLiFormer relation extraction requires relation labels")
        relation_types = labels
    elif classification_task is None:
        entity_types = labels
        if relation_types is not None and entity_types is None:
            raise InvalidInputError("GLiFormer relation_labels require labels as entity types")
    elif relation_types is not None:
        raise InvalidInputError("GLiFormer relation_labels cannot be combined with classification_task")
    if entity_types is None and supplied_entities is None and not classes and plan is None:
        raise InvalidInputError(_ERR_REQUIRES_TASK)

    # Everything that becomes prompt text counts against one budget,
    # including the entity types carried by supplied entities.
    prompt_labels = (
        len(entity_types or [])
        + len(relation_types or [])
        + sum(len(group) + 1 for group in classes.values())
        + (_schema_field_count(plan.root) if plan is not None else 0)
        + len({entity["label"] for entities in supplied_entities or [] for entity in entities})
    )
    if prompt_labels > MAX_EXTRACT_LABELS:
        raise InvalidInputError(
            f"GLiFormer requests may carry at most {MAX_EXTRACT_LABELS} labels, relation types, class labels, "
            "and schema fields in total"
        )
    return _RequestPlan(
        entity_types=entity_types,
        relation_types=relation_types,
        classes=classes,
        classification_task=classification_task,
        label_groups=label_groups,
        plan=plan,
        supplied_entities=supplied_entities,
        relation_threshold=relation_threshold,
    )


def _schema_field_count(node: SchemaField) -> int:
    return sum(1 + _schema_field_count(child) for _, child in node.properties if child.kind != "choice")


def _assemble_output(
    texts: list[str],
    raw_results: list[dict[str, Any]],
    request: _RequestPlan,
    *,
    input_token_counts: list[int],
) -> ExtractOutput:
    """Map raw results to SIE fields. Errored items get empty results and bill 0."""
    plan = request.plan
    classified = request.classification_task is not None or request.label_groups is not None
    all_entities: list[list[Entity]] = []
    all_classifications: list[list[Classification]] = []
    all_relations: list[list[Relation]] = []
    all_data: list[dict[str, Any]] = []
    errors: list[ExtractItemError | None] = []
    counts = list(input_token_counts)

    for index, (text, raw) in enumerate(zip(texts, raw_results, strict=True)):
        try:
            if _ITEM_ERROR in raw:
                raise RuntimeError(_ERR_ITEM_OUTPUT)
            entities, relations, classifications, data, error = _assemble_item(text, raw, request, index)
        except RuntimeError as exc:
            logger.warning("GLiFormer output for one item could not be used: %s", exc)
            entities, relations, classifications, data = [], [], [], {}
            error = ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_ITEM_OUTPUT)
        if error is not None:
            # An errored item returns nothing and bills nothing.
            entities, relations, classifications, data = [], [], [], {}
            counts[index] = 0
        all_entities.append(entities)
        all_relations.append(relations)
        if classified:
            all_classifications.append(classifications)
        if plan is not None:
            all_data.append(data)
        errors.append(error)

    return ExtractOutput(
        entities=all_entities,
        classifications=all_classifications if classified else None,
        relations=all_relations if request.relations_requested else None,
        data=all_data if plan is not None else None,
        errors=errors if any(error is not None for error in errors) else None,
        input_token_counts=counts,
    )


def _assemble_item(
    text: str,
    raw: dict[str, Any],
    request: _RequestPlan,
    index: int,
) -> tuple[list[Entity], list[Relation], list[Classification], dict[str, Any], ExtractItemError | None]:
    """One document's SIE fields; raises RuntimeError on malformed model output."""
    if request.supplied_entities is not None:
        entities = request.supplied_entities[index]
        allowed_endpoints: set[str] | None = {entity["text"] for entity in entities}
    else:
        entities = _to_entities(text, raw.get("ner", []))
        allowed_endpoints = None
    relations = _to_relations(raw.get("joint_relex", []), allowed_endpoints, request.relation_threshold)

    groups = _classification_groups(raw.get("classification"), request.classes)
    classifications: list[Classification] = []
    if request.classification_task is not None:
        classifications = groups[request.classification_task]
    if request.label_groups is not None:
        # Same "group.label" naming as the GLiClass adapter's label groups.
        flattened = [
            Classification(label=f"{name}.{prediction['label']}", score=prediction["score"])
            for name in request.label_groups
            for prediction in groups[name]
        ]
        classifications = sorted(flattened, key=lambda item: item["score"], reverse=True)

    data: dict[str, Any] = {}
    error: ExtractItemError | None = None
    plan = request.plan
    if plan is not None:
        choices = {name: groups[name][0]["label"] for name in plan.choice_groups if groups[name]}
        data, missing = shape_structured_output(
            plan,
            raw.get("structuring", {}) if plan.structures is not None else None,
            choices,
        )
        if missing:
            data = {}
            error = ExtractItemError(
                code=ErrorCode.INFERENCE_ERROR.value,
                message=f"GLiFormer did not extract required output_schema properties: {clip(str(missing))}",
            )
    return entities, relations, classifications, data, error


def _normalize_input_entity(text: str, entity: Any) -> Entity:
    if not isinstance(entity, dict):
        raise InvalidInputError("GLiFormer relation entities must be objects")
    start = entity.get("start")
    end = entity.get("end")
    entity_text = entity.get("text")
    label = entity.get("label", "ENTITY")
    if (
        not _is_int(start)
        or not _is_int(end)
        or not isinstance(entity_text, str)
        or not isinstance(label, str)
        or not label.strip()
        or not 0 <= start < end <= len(text)
        or text[start:end] != entity_text
    ):
        raise InvalidInputError("GLiFormer relation entities require valid character offsets")
    if len(label.strip()) > MAX_LABEL_CHARS:
        raise InvalidInputError(f"GLiFormer relation entity labels may have at most {MAX_LABEL_CHARS} characters")
    score = entity.get("score", 1.0)
    # The range check comes first: math.isfinite cannot convert a huge integer.
    if isinstance(score, bool) or not isinstance(score, Real) or not 0 <= score <= 1 or not math.isfinite(score):
        raise InvalidInputError("GLiFormer relation entity score must be a number between 0 and 1")
    return Entity(text=entity_text, label=label.strip(), score=float(score), start=start, end=end)


def _to_entities(text: str, raw_entities: Any) -> list[Entity]:
    if not isinstance(raw_entities, list):
        raise RuntimeError(_ERR_MALFORMED.format(output="entities"))
    entities: list[Entity] = []
    for raw in raw_entities:
        if not isinstance(raw, dict):
            raise RuntimeError(_ERR_MALFORMED.format(output="entities"))
        start, end, span, label = raw.get("start"), raw.get("end"), raw.get("text"), raw.get("label")
        if (
            not _is_int(start)
            or not _is_int(end)
            or not 0 <= start < end <= len(text)
            or span != text[start:end]
            or not isinstance(label, str)
        ):
            raise RuntimeError(_ERR_MALFORMED.format(output="entity offsets"))
        score = _validate_score(raw.get("score"), "entity")
        entities.append(Entity(text=span, label=label, score=score, start=start, end=end))
    entities.sort(key=lambda entity: (entity.get("start") or 0, entity.get("end") or 0))
    return entities


def _to_relations(
    raw_relations: Any, allowed_endpoints: set[str] | None, relation_threshold: float | None
) -> list[Relation]:
    if not isinstance(raw_relations, list):
        raise RuntimeError(_ERR_MALFORMED.format(output="relations"))
    relations: list[Relation] = []
    for raw in raw_relations:
        if not isinstance(raw, dict) or not isinstance(raw.get("head"), dict) or not isinstance(raw.get("tail"), dict):
            raise RuntimeError(_ERR_MALFORMED.format(output="relations"))
        head, tail, relation = raw["head"].get("text"), raw["tail"].get("text"), raw.get("relation")
        if not isinstance(head, str) or not isinstance(tail, str) or not isinstance(relation, str):
            raise RuntimeError(_ERR_MALFORMED.format(output="relations"))
        # The joint head finds endpoints in the text itself; with supplied
        # entities, anything outside that set is not a valid answer.
        if allowed_endpoints is not None and (head not in allowed_endpoints or tail not in allowed_endpoints):
            continue
        score = _validate_score(raw.get("score"), "relation")
        # Same comparison as the package's decoder applies to ``threshold``.
        if relation_threshold is not None and score <= relation_threshold:
            continue
        relations.append(Relation(head=head, tail=tail, relation=relation, score=score))
    relations.sort(key=lambda item: (-item["score"], item["relation"], item["head"], item["tail"]))
    return relations


def _classification_groups(raw_groups: Any, classes: dict[str, list[str]]) -> dict[str, list[Classification]]:
    """Name GLiFormer's positional per-group predictions, best first."""
    if not classes:
        return {}
    groups = raw_groups if raw_groups is not None else [[] for _ in classes]
    if not isinstance(groups, list) or len(groups) != len(classes):
        raise RuntimeError(_ERR_MALFORMED.format(output="classifications"))
    mapped: dict[str, list[Classification]] = {}
    for name, predictions in zip(classes, groups, strict=True):
        if not isinstance(predictions, list):
            raise RuntimeError(_ERR_MALFORMED.format(output="classifications"))
        group: list[Classification] = []
        for prediction in predictions:
            label = prediction.get("class_name") if isinstance(prediction, dict) else None
            if not isinstance(label, str) or label not in classes[name]:
                raise RuntimeError(_ERR_MALFORMED.format(output="classifications"))
            group.append(Classification(label=label, score=_validate_score(prediction.get("score"), "classification")))
        group.sort(key=lambda classification: classification["score"], reverse=True)
        mapped[name] = group
    return mapped


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_threshold(value: object, name: str = "threshold") -> float:
    message = f"GLiFormer {name} must be a number between 0 and 1"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(message)
    try:
        threshold = float(value)
    except OverflowError as exc:
        raise InvalidInputError(message) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InvalidInputError(message)
    return threshold


def _validate_flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidInputError(f"GLiFormer {name} must be boolean")
    return value


def _validate_task(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_LABEL_CHARS:
        raise InvalidInputError(
            f"GLiFormer classification_task must be a non-empty string of at most {MAX_LABEL_CHARS} characters"
        )
    return value.strip()


def _resolve_multi_label(options: dict[str, Any], *, default: bool) -> bool:
    """Read ``multi_label``, also accepting GLiClass's ``classification_type`` spelling."""
    value = options.get("multi_label")
    classification_type = options.get("classification_type")
    if classification_type is None:
        return _validate_flag(default if value is None else value, "multi_label")
    if classification_type not in ("single-label", "multi-label"):
        raise InvalidInputError("GLiFormer classification_type must be 'single-label' or 'multi-label'")
    multi_label = classification_type == "multi-label"
    if value is not None and _validate_flag(value, "multi_label") != multi_label:
        raise InvalidInputError("GLiFormer multi_label contradicts classification_type")
    return multi_label


def _validate_label_groups(groups: Any) -> dict[str, list[str]] | None:
    if groups is None:
        return None
    if not isinstance(groups, dict) or not groups:
        raise InvalidInputError("GLiFormer label_groups must be a non-empty object of label lists")
    if sum(len(labels) if isinstance(labels, list) else 1 for labels in groups.values()) > MAX_EXTRACT_LABELS:
        raise InvalidInputError(f"GLiFormer label_groups must contain at most {MAX_EXTRACT_LABELS} labels")
    validated: dict[str, list[str]] = {}
    for name, labels in groups.items():
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_LABEL_CHARS:
            raise InvalidInputError(
                f"GLiFormer label_groups names must be non-empty strings of at most {MAX_LABEL_CHARS} characters"
            )
        if name.strip() in validated:
            raise InvalidInputError("GLiFormer label_groups names must be unique")
        validated[name.strip()] = _validate_labels(labels, f"label_groups[{clip(repr(name.strip()))}]")
    return validated


def _validate_labels(labels: Any, name: str) -> list[str]:
    if not isinstance(labels, list) or not labels:
        raise InvalidInputError(f"GLiFormer {name} must be a non-empty list")
    if len(labels) > MAX_EXTRACT_LABELS:
        raise InvalidInputError(f"GLiFormer {name} must contain at most {MAX_EXTRACT_LABELS} entries")
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise InvalidInputError(f"GLiFormer {name} must be non-empty strings")
    normalized = [label.strip() for label in labels]
    if any(len(label) > MAX_LABEL_CHARS for label in normalized):
        raise InvalidInputError(f"GLiFormer {name} may have at most {MAX_LABEL_CHARS} characters each")
    if len(set(normalized)) != len(normalized):
        raise InvalidInputError(f"GLiFormer {name} must be unique")
    return normalized


def _validate_score(value: object, output_name: str) -> float:
    """Validate a score the model produced."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"GLiFormer returned an invalid {output_name} score")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"GLiFormer returned an invalid {output_name} score")
    return score
