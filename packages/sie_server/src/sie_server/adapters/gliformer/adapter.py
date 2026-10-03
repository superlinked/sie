"""Adapter for GLiFormer multi-task extraction and embedding models."""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import logging
import math
import operator
import sys
import threading
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from sie_server.adapters.gliformer.relation_decoding import make_relation_decode
from sie_server.adapters.gliformer.span_decoding import (
    PROPOSAL_UNITS,
    candidate_units,
    decoding_row,
    document_allowances,
    pair_spans,
    propose_spans,
    select_spans,
)
from sie_server.adapters.gliformer.structuring_decoding import make_structuring_decode
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
# output_schema fields). Span decoding keeps at most MAX_SPAN_CANDIDATES
# (span_decoding) candidates per document, but candidates multiply as the
# threshold falls (an entity-dense 2048-word document with 64 labels has 691
# at 0.5 and 1334 at 0.1), and the floor keeps ordinary documents well inside
# that bound. Classification decoding is a sigmoid and a filter per label, so
# classification-only requests accept any threshold.
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
# Span decoding work each document may do, in span_decoding's allowance units
# (at most about 7 microseconds of host work each on an L4 host, 1.3 at the
# median): a floor plus an allowance per billed token of that document. A
# document that needs more keeps the best-first prefix of each stage it can
# afford (see span_decoding).
# Measured with both checkpoints on 4,416 documents (short records, 2048-word
# entity-dense text, prose, a list of names, repeated text, and 64-word
# windows of them) and 12 tasks up to 64 labels, 20 relation types, and
# records schemas with 16 fields or two record types: at threshold 0.5 every
# document used at most 21% of its allowance; at 0.1 the heaviest, 64 words
# of names with two record types, used 29,739 of 29,952 units.
_DECODE_FLOOR = 4096
_DECODE_UNITS_PER_TOKEN = 256

# Distinct task prompts whose token counts are kept. A prompt depends only on
# the request's task arguments, so a repeated task skips rebuilding and
# re-tokenizing it.
_PROMPT_CACHE_SIZE = 256

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
        _bound_span_decoding()
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


def _bound_span_decoding() -> None:
    """Replace GLiFormer's span decoding hot spots with bounded, equivalent code, or fail.

    Every BIO span decoder in the package (entities, the relation head's
    entity candidates, structuring fields) pairs starts with ends in
    ``SpanDecoder._calculate_span_score``, one document row at a time from
    ``SpanDecoder.decode_bio_spans_batch``, and removes overlaps in
    ``SpanDecoder.greedy_search``; the structuring head proposes spans with
    GLiNER's ``extract_spans_from_tokens`` and turns them into records in
    ``StructuringDecoder.decode``, and the relation decoder reads its scores
    cell by cell in ``JointRelexDecoder.decode``. All of these work pair by
    pair in Python. The
    replacements return the same spans in the same order with the same
    scores, and bound how many candidates a document row can produce (see
    ``span_decoding`` and ``structuring_decoding``). The relation head builds
    a new span decoder for every forward pass, so methods are replaced on the
    classes.

    Raises:
        RuntimeError: A replaced function is missing, or no longer has the
            parameters (and, in the exactly pinned gliformer package, the
            source) that its replacement was verified against.
    """
    span_decoder = importlib.import_module("gliformer.tasks.span_decoder")
    structuring_model = importlib.import_module("gliformer.tasks.structuring.model")
    structuring_decoder = importlib.import_module("gliformer.tasks.structuring.decoder")
    span_class = span_decoder.Span

    def calculate_span_score(
        self: Any,
        start_idx: Any,
        end_idx: Any,
        scores_inside: torch.Tensor,
        scores_start: torch.Tensor,
        scores_end: torch.Tensor,
        id_to_classes: Mapping[int, Any],
        threshold: float,
        max_width: int | None = None,
    ) -> list[Any]:
        _ = self
        return pair_spans(
            start_idx,
            end_idx,
            scores_inside,
            scores_start,
            scores_end,
            id_to_classes,
            threshold,
            make_span=span_class,
            max_width=max_width,
        )

    def greedy_search(self: Any, spans: list[Any], flat_ner: bool = True, multi_label: bool = False) -> list[Any]:
        _ = self
        return select_spans(spans, flat_ner, multi_label)

    def decode_bio_spans_batch(
        self: Any,
        logits: torch.Tensor,
        id_to_classes: Any,
        batch_size: int,
        threshold: float,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> list[list[Any]]:
        # The package's loop, with each row marked so that its pairing draws
        # on that document's allowance.
        all_spans = []
        for i in range(batch_size):
            id_to_class_i = self._get_id_to_class(id_to_classes, i)
            with decoding_row(i):
                spans = self.decode_bio_spans(logits[i], id_to_class_i, threshold, flat_ner, multi_label)
            all_spans.append(spans)
        return all_spans

    _replace_package_function(
        span_decoder.SpanDecoder,
        "_calculate_span_score",
        calculate_span_score,
        source_sha256="5e05a885577dbedbf52f3d98ad391e7cb37b7762d518b12d82a338cd2976f546",
    )
    _replace_package_function(
        span_decoder.SpanDecoder,
        "greedy_search",
        greedy_search,
        source_sha256="8a08975bc767870ef5e3ed9bc9ec1d2284c19c0604a9eed48480b922dbc099ad",
    )
    _replace_package_function(
        span_decoder.SpanDecoder,
        "decode_bio_spans_batch",
        decode_bio_spans_batch,
        source_sha256="fc2eab7aa5798d5ce07763ac451528b64fd9f4b670f684b9fd7a4813969ceba6",
    )
    _replace_package_function(
        structuring_decoder.StructuringDecoder,
        "decode",
        make_structuring_decode(span_class, structuring_decoder.unflatten_by_batch_origin),
        source_sha256="079bfd5f322bc10981c4150b708da41fbba6cb3742ae5e0daa4e9a4e3f31859c",
    )
    joint_relex_decoder = importlib.import_module("gliformer.tasks.joint_relex.decoder")
    ner_decoder = importlib.import_module("gliformer.tasks.ner.decoder")
    _replace_package_function(
        joint_relex_decoder.JointRelexDecoder,
        "decode",
        make_relation_decode(ner_decoder.NERDecoder.decode, joint_relex_decoder.unflatten_by_batch_origin),
        source_sha256="37d5f8ca90c546ef639e58ede17778565c233c87e1576ed9c65e2f4891c69948",
    )
    original_proposals = getattr(structuring_model, "extract_spans_from_tokens", None)
    if getattr(original_proposals, "_sie_bounded", False):
        original_proposals = original_proposals._sie_original  # ty:ignore[unresolved-attribute]
    if not callable(original_proposals):
        raise RuntimeError("GLiFormer's structuring head has no span proposal function to bound")

    def extract_spans_from_tokens(
        scores: torch.Tensor, labels: torch.Tensor | None = None, threshold: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return propose_spans(scores, labels, threshold, original=original_proposals)

    # Every gliformer module that imported GLiNER's proposal function gets the
    # bounded one; only the structuring head calls it in gliformer 0.1.2.
    # gliner is pinned by range, and images resolve it from the index, so its
    # source is not hashed: loading checks the replacement's behaviour
    # against it instead (see _verify_bounded_decoding).
    for module_name in _PROPOSAL_MODULES:
        module = importlib.import_module(module_name)
        if getattr(module, "extract_spans_from_tokens", None) is not None:
            _replace_package_function(module, "extract_spans_from_tokens", extract_spans_from_tokens)
    _assert_bounded_decoding()


# gliformer modules that bind GLiNER's span proposal function at import.
_PROPOSAL_MODULES = (
    "gliformer.tasks.structuring.model",
    "gliformer.tasks.joint_relex.model",
    "gliformer.tasks.anchored_extraction",
    "gliformer.tasks.open_relex.model",
)
# gliformer's task decoders, imported so that their classes can be checked.
_DECODER_MODULES = (
    "gliformer.tasks.ner.decoder",
    "gliformer.tasks.joint_relex.decoder",
    "gliformer.tasks.open_relex.decoder",
    "gliformer.tasks.structuring.decoder",
)


def _assert_bounded_decoding() -> None:
    """Fail unless nothing in gliformer can still reach the unbounded functions.

    Raises:
        RuntimeError: A span decoder subclass overrides a replaced method, a
            structuring decoder subclass overrides ``decode``, or a gliformer
            module still binds GLiNER's own proposal function.
    """
    span_decoder = importlib.import_module("gliformer.tasks.span_decoder")
    structuring_decoder = importlib.import_module("gliformer.tasks.structuring.decoder")
    joint_relex_decoder = importlib.import_module("gliformer.tasks.joint_relex.decoder")
    for module_name in _DECODER_MODULES:
        importlib.import_module(module_name)
    for base, names in (
        (span_decoder.SpanDecoder, ("_calculate_span_score", "greedy_search", "decode_bio_spans_batch")),
        (structuring_decoder.StructuringDecoder, ("decode",)),
        (joint_relex_decoder.JointRelexDecoder, ("decode",)),
    ):
        for subclass in _subclasses(base):
            overridden = [name for name in names if name in vars(subclass)]
            if overridden:
                raise RuntimeError(
                    f"GLiFormer's {subclass.__module__}.{subclass.__qualname__} overrides {overridden}, "
                    "which the adapter bounds"
                )
    proposals: Any = importlib.import_module("gliformer.tasks.structuring.model").extract_spans_from_tokens
    original = getattr(proposals, "_sie_original", None)
    for module_name, module in list(sys.modules.items()):
        if module is None or not (module_name == "gliformer" or module_name.startswith("gliformer.")):
            continue
        if any(value is original for value in vars(module).values()):
            raise RuntimeError(f"GLiFormer's {module_name} still binds GLiNER's unbounded span proposal function")


def _verify_bounded_decoding(model: Any) -> None:
    """Fail unless each replacement matches the function it replaces on a fixed input.

    The input is small, so no bound applies, and it has ties, padding, nested
    and overlapping spans, and scores exactly at the threshold. This is what
    keeps a gliner release that changes proposal semantics from loading, since
    gliner is pinned by range and not hashed.

    Raises:
        RuntimeError: A replacement's output differs from the original's.
    """
    span_decoder = importlib.import_module("gliformer.tasks.span_decoder")
    structuring_model = importlib.import_module("gliformer.tasks.structuring.model")
    structuring_decoder = importlib.import_module("gliformer.tasks.structuring.decoder")
    _ = model
    # The replaced code reads no configuration, so a blank one keeps the check
    # independent of the checkpoint.
    decoder = span_decoder.SpanDecoder(SimpleNamespace())
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 24, 3, 3, generator=generator) * 3
    logits[:, 2:6, 0, :] = 4.0  # a run of saturated, tied scores
    logits[:, 4, 1, 2] = torch.tensor(0.3).logit()  # inside exactly at the threshold
    logits[1, 18:] = float("-inf")  # padding

    def check(name: str, original: Any, replacement: Any, same: Callable[[Any, Any], bool] = operator.eq) -> None:
        if not same(original, replacement):
            raise RuntimeError(f"GLiFormer's {name} no longer matches its bounded replacement")

    names = {0: "a", 1: "b", 2: "c"}
    for threshold in (0.3, 0.5):
        start, end, inside = logits[0].permute(2, 0, 1)
        args = (
            decoder._get_indices_above_threshold(start, threshold),
            decoder._get_indices_above_threshold(end, threshold),
            torch.sigmoid(inside),
            torch.sigmoid(start),
            torch.sigmoid(end),
            names,
            threshold,
        )
        pairing = vars(span_decoder.SpanDecoder)["_calculate_span_score"]
        spans = pairing(decoder, *args)
        check("span pairing", pairing._sie_original(decoder, *args), spans)
        greedy = vars(span_decoder.SpanDecoder)["greedy_search"]
        for flat_ner in (True, False):
            for multi_label in (True, False):
                check(
                    "overlap removal",
                    greedy._sie_original(decoder, list(spans), flat_ner, multi_label),
                    greedy(decoder, list(spans), flat_ner, multi_label),
                )
        batch = vars(span_decoder.SpanDecoder)["decode_bio_spans_batch"]
        for flat_ner in (True, False):
            check(
                "batch span decoding",
                batch._sie_original(decoder, logits, names, 2, threshold, flat_ner, False),
                batch(decoder, logits, names, 2, threshold, flat_ner, False),
            )
        proposals = vars(structuring_model)["extract_spans_from_tokens"]
        check(
            "span proposal",
            proposals._sie_original(logits, None, threshold),
            proposals(logits, None, threshold),
            lambda a, b: all(torch.equal(x, y) for x, y in zip(a, b, strict=True)),
        )
    records = SimpleNamespace(
        structuring_logits=torch.randn(1, 4, 6, generator=generator) * 2,
        structuring_field_logits=torch.randn(1, 6, 3, generator=generator) * 2,
        structuring_span_idx=torch.tensor([[[0, 1], [0, 3], [2, 2], [2, 5], [4, 4], [0, 1]]]),
        structuring_span_mask=torch.tensor([[True, True, True, True, True, False]]),
        structuring_anchor_mask=torch.tensor([[True, True, False, True]]),
        structuring_batch_origin=torch.arange(1),
        batch_size=1,
    )
    joint_relex_decoder = importlib.import_module("gliformer.tasks.joint_relex.decoder")
    relation_decoder = joint_relex_decoder.JointRelexDecoder(SimpleNamespace())
    relations = SimpleNamespace(
        ner_logits=logits,
        ner_batch_origin=torch.arange(2),
        span_logits=None,
        span_idx=None,
        span_mask=None,
        joint_rel_logits=torch.randn(2, 6, 3, generator=generator) * 2,
        joint_rel_idx=torch.tensor([[[0, 1], [1, 0], [0, 2], [2, 1], [1, 2], [5, 0]]] * 2),
        joint_rel_mask=torch.tensor([[True, True, True, True, True, False], [True, False, True, True, True, True]]),
        joint_rel_batch_origin=torch.arange(2),
        batch_size=2,
    )
    relation_decode = vars(joint_relex_decoder.JointRelexDecoder)["decode"]
    words = [[f"w{i}" for i in range(24)]] * 2
    for threshold in (0.3, 0.5):
        kwargs = {"threshold": threshold, "texts": words}
        check(
            "relation decoding",
            relation_decode._sie_original(relation_decoder, relations, **kwargs),
            relation_decode(relation_decoder, relations, **kwargs),
        )
    record_decoder = structuring_decoder.StructuringDecoder(SimpleNamespace())
    decode = vars(structuring_decoder.StructuringDecoder)["decode"]
    for flat_ner in (True, False):
        kwargs = {"threshold": 0.3, "flat_ner": flat_ner, "texts": [[f"w{i}" for i in range(8)]]}
        check(
            "record decoding",
            decode._sie_original(record_decoder, records, **kwargs),
            decode(record_decoder, records, **kwargs),
        )


def _subclasses(cls: type) -> list[type]:
    found = []
    for subclass in cls.__subclasses__():
        found.append(subclass)
        found.extend(_subclasses(subclass))
    return found


def _replace_package_function(
    owner: Any, name: str, replacement: Callable[..., Any], *, source_sha256: str | None = None
) -> None:
    """Set ``owner.name`` to ``replacement`` once, if the original still matches.

    Raises:
        RuntimeError: ``owner.name`` is missing, takes other parameters than
            ``replacement``, or its source no longer hashes to ``source_sha256``.
    """
    original: Any = getattr(owner, name, None)
    label = f"{getattr(owner, '__name__', owner)}.{name}"
    if not callable(original):
        raise RuntimeError(f"GLiFormer's {label} is missing")
    if getattr(original, "_sie_bounded", False):
        return
    if tuple(inspect.signature(original).parameters) != tuple(inspect.signature(replacement).parameters):
        raise RuntimeError(f"GLiFormer's {label} changed its parameters")
    if source_sha256 is not None:
        try:
            source = inspect.getsource(original)
        except (OSError, TypeError) as exc:
            raise RuntimeError(f"GLiFormer's {label} source cannot be verified") from exc
        if hashlib.sha256(source.encode()).hexdigest() != source_sha256:
            raise RuntimeError(f"GLiFormer's {label} changed; its bounded replacement must be verified again")
    vars(replacement).update(_sie_bounded=True, _sie_original=original)
    setattr(owner, name, replacement)


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
    ``output_schema`` fields need at least 0.1. Classification-only requests
    (``classification_task``, ``label_groups``, or an ``output_schema`` of
    root enums only) accept any threshold from 0, where every question gets
    its best answer.

    Span decoding is bounded per document: at most 4096 candidate spans for
    entities and relation endpoints, 2048 structuring proposals, 65,536
    record field spans across all record slots, and 65,536 relations whose
    heads and tails span at most 262,144 words. Beyond that a document keeps
    the candidates overlap removal would take first (best score first, and
    among equal scores in GLiFormer's own order), so it loses only results
    that rank below every kept one; nothing in the output marks such a
    document. The measured documents stay below every bound, so their results
    are unchanged. Each document of a batch decodes as it would alone: the
    padding of shorter documents never forms spans or lowers scores.

    Each document's span decoding also draws on its own allowance: 4,096
    work units plus 256 per billed token of that document. A unit is at most
    about 7 microseconds of host work: one candidate span or record field
    span, two per relation, 64 per structuring proposal or relation entity
    candidate, and a fraction of a unit per score cell of the document that
    a stage reads (a whole unit when it searches for a cut). A stage that
    cannot afford everything keeps the best-first prefix it can afford, as
    at the fixed bounds; the document still succeeds and bills normally, and
    other documents are unaffected.

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
        self._prompt_counts: OrderedDict[bytes, int] = OrderedDict()
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
        _skip_redundant_eval(model)
        model.model.register_forward_hook(_upcast_score_outputs)
        _mask_padded_words(model.model)
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

        # Loading the checkpoint imports the task decoders; check them too.
        _assert_bounded_decoding()
        _verify_bounded_decoding(model)
        _probe_forward(model)

        self._model = model
        self._tokenizer = tokenizer
        self._normalize_structures = gliformer.processing.schema.normalize_structuring_schemas
        self._build_formatter = gliformer.processing.schema.build_structuring_output_formatter
        self._prompt_counts = OrderedDict()
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
                [_DECODE_FLOOR + _DECODE_UNITS_PER_TOKEN * counts[index] for index in indices],
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
        allowances: list[int],
        *,
        structures: dict[str, Any] | None,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
        max_items: int | None,
    ) -> list[dict[str, Any]]:
        """Run ``GLiFormer.inference`` in bounded chunks; one raw result per text.

        ``allowances`` holds each text's span decoding allowance, in units.
        """
        formatter = self._build_formatter(structures) if structures and self._build_formatter else None
        rows: list[dict[str, Any]] = [{} for _ in texts]
        with self._tokenizer_guard():
            for chunk in _token_budget_chunks(lengths, self._inference_batch_tokens, max_items=max_items):
                chunk_rows = self._infer(
                    texts,
                    chunk,
                    task_kwargs,
                    allowances,
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
        allowances: list[int],
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
            return self._forward(texts, part, task_kwargs, allowances, **inference_kwargs)

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
        allowances: list[int],
        **inference_kwargs: Any,
    ) -> list[dict[str, Any]] | None:
        """One forward pass; ``None`` when it produced non-finite scores.

        Each document starts the pass with its full allowance, so what it
        keeps depends only on the document itself.
        """
        try:
            with torch.inference_mode(), document_allowances([allowances[index] for index in chunk]):
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
        """Count the tokens of one task prompt, measuring each distinct prompt once.

        GLiFormer prepends the same prompt to every document of a task group
        and truncates the combined sequence to ``max_len``, so the prompt is
        measured once per group, not per document. The task arguments hold
        only strings, lists, dicts and ``None``, so their ``repr`` identifies
        the prompt, in order. The limits are checked on every request.

        Raises:
            InvalidInputError: The prompt exceeds ``max_prompt_tokens`` or
                leaves no room for a document.
            RuntimeError: The package's processor could not build the prompt.
        """
        cache = self._prompt_counts
        key = hashlib.sha256(repr(task_kwargs).encode()).digest()
        count = cache.get(key)
        if count is None:
            count = self._measure_prompt(task_kwargs)
            cache[key] = count
            if len(cache) > _PROMPT_CACHE_SIZE:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        if count > self._max_prompt_tokens:
            raise InvalidInputError(
                f"GLiFormer task prompt needs {count} tokens; labels, relation types, class labels, and "
                f"schema fields may take at most {self._max_prompt_tokens}"
            )
        model = self._model
        specials = int(model.data_processor.transformer_tokenizer.num_special_tokens_to_add(pair=False))
        if count + specials >= int(model.config.max_len):
            raise InvalidInputError(_ERR_PROMPT_EXHAUSTS_DOCUMENT)
        return count

    def _measure_prompt(self, task_kwargs: dict[str, Any]) -> int:
        """Build one task prompt with the package's processor and count its tokens.

        Raises:
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
            return len(encoded["input_ids"][0])
        except Exception as exc:
            raise RuntimeError("GLiFormer could not build the task prompt") from exc

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


def _skip_redundant_eval(model: Any) -> None:
    """Make ``model.eval()`` return at once while the model is in eval mode.

    ``GLiFormer.inference`` and ``embed_text`` call ``eval()`` on every
    request, which walks all of the model's ~600 modules to clear training
    flags that are already clear. The adapter never switches the model to
    training, so only a model left in training mode still needs the walk,
    which is what ``torch.nn.Module.eval`` does: ``train(False)``. The model
    is held weakly, so the method stored on it adds no reference cycle.
    """
    model_ref = weakref.ref(model)

    def eval_if_training() -> Any:
        current = model_ref()
        if current is not None and current.training:
            current.train(False)
        return current

    model.eval = eval_if_training


# Heads that run an NER pass for their own entity candidates, and the method
# that returns it.
_NER_CONSUMERS = {"joint_relex": "_forward_ner", "structuring": "_forward_entity_ner"}


# A short batch that runs every head whose output the hooks check: entities
# and relations through the relation head's NER pass, and a structuring field
# through the structuring head's. Two lengths, so the padding mask runs too.
_PROBE_TEXTS = ["Ada Lovelace worked with Charles Babbage in London.", "Ada lived in London."]
_PROBE_TASKS = {
    "entities": None,
    "classes": None,
    "joint_relations": {None: {"entities": ["person", "city"], "relations": ["lives in"]}},
    "structures": {"$root": {"name": "str"}},
}


def _probe_forward(model: Any) -> None:
    """Run one small extraction so a checkpoint the hooks cannot handle fails at load.

    Raises:
        RuntimeError: A hook rejected the checkpoint's outputs, for example
            NER logits of an unexpected shape.
    """
    with torch.inference_mode():
        model.inference(list(_PROBE_TEXTS), **_PROBE_TASKS, threshold=0.5, batch_size=len(_PROBE_TEXTS))


def _mask_padded_words(model: Any) -> None:
    """Keep the padding of shorter documents out of span decoding.

    A forward pass pads every document to the longest one in its chunk, and
    the NER head scores padded word positions with logit 0: probability 0.5
    for start, end, and inside. Below a 0.5 threshold the span decoder then
    pairs every padded position with every other, quadratic in the padding
    (about 100 s for 40 short records at threshold 0.3), only for the mapping
    step to drop those spans. At any threshold, a span that ends a shorter
    document took the padding's 0.5 as its right neighbour and was reported
    with score 0.5. Padded positions get logit -inf instead, so each document
    decodes as it would alone.

    The relation and structuring heads run an NER pass for their own entity
    candidates. When they reuse the standalone NER head, its hook covers
    them; a head with an NER head of its own gets its NER pass masked too.

    Raises:
        RuntimeError: The checkpoint has no NER head, or a relation or
            structuring head runs an NER pass this cannot mask.
    """
    heads = getattr(model, "heads", None)
    if heads is None or "ner" not in heads:
        raise RuntimeError("GLiFormer checkpoint has no NER head to mask")
    ner_head = heads["ner"]
    ner_head.register_forward_hook(_mask_padded_ner_logits)
    for name, method in _NER_CONSUMERS.items():
        if name not in heads:
            continue
        head = heads[name]
        if getattr(head, "_owns_ner_head", None) is False and vars(head).get("_reused_ner_head") is ner_head:
            continue
        ner_pass = getattr(head, method, None)
        if getattr(head, "_owns_ner_head", None) is not True or not callable(ner_pass):
            raise RuntimeError(f"GLiFormer {name} head runs an NER pass the adapter cannot mask")
        vars(head)[method] = _masked_ner_pass(ner_pass)


def _masked_ner_pass(ner_pass: Callable[..., Any]) -> Callable[..., Any]:
    """``ner_pass`` with its output's padded positions masked."""

    def masked(*args: Any, **kwargs: Any) -> Any:
        return _mask_padded_ner_logits(None, args, ner_pass(*args, **kwargs))

    return masked


def _mask_padded_ner_logits(_module: Any, _args: Any, output: Any) -> Any:
    """Forward hook: set the NER logits of padded word positions to -inf.

    Raises:
        RuntimeError: The head returned logits without a word mask of the
            same batch and length.
    """
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        return output
    extra = getattr(output, "extra", None)
    mask = extra.get("mask") if isinstance(extra, Mapping) else None
    if not isinstance(mask, torch.Tensor) or logits.dim() != 4 or tuple(mask.shape) != tuple(logits.shape[:2]):
        raise RuntimeError("GLiFormer NER logits came without a matching word mask")
    padded = ~mask.bool()
    if bool(padded.any()):
        output.logits = logits.masked_fill(padded[:, :, None, None], float("-inf"))
    return output


def _limit_relation_entities(model: Any) -> None:
    """Bound the relation head's entity candidates, and budget their cost.

    - Only the most confident entities are relation candidates. The package
      ranks the decoded entities and slices the kept ones into compact
      tensors before it builds entity pairs, so the pair tensors never grow
      past this many entities per document.
    - The head ranks each decoded entity with a few small tensor operations
      in Python before it applies these bounds, so each candidate span it
      pairs costs ``PROPOSAL_UNITS`` of its document's allowance.

    Raises:
        RuntimeError: The checkpoint has no joint relation head, or it no
            longer decodes its entity candidates in the method this charges.
    """
    heads = getattr(model, "heads", None)
    if heads is None or "joint_relex" not in heads:
        raise RuntimeError("GLiFormer checkpoint has no joint relation head to bound")
    head = heads["joint_relex"]
    current = getattr(head, "max_relation_entities", None)
    head.max_relation_entities = _MAX_RELATION_ENTITIES if current is None else min(current, _MAX_RELATION_ENTITIES)
    decode = getattr(head, "_decode_relation_entity_spans", None)
    if not callable(decode):
        raise RuntimeError("GLiFormer's relation head no longer decodes entity candidates where the adapter expects")

    def charged(*args: Any, **kwargs: Any) -> Any:
        with candidate_units(PROPOSAL_UNITS):
            return decode(*args, **kwargs)

    vars(head)["_decode_relation_entity_spans"] = charged


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
