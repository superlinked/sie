"""Union of GLiNER-family models, served as one extract model.

One GLiNER checkpoint can miss spans another finds. This adapter serves the
union of several under one model id, composed exactly as SIE's PII redaction
study composes ``urchade/gliner_multi_pii-v1`` and ``numind/NuNER_Zero``
(``examples/redact/score.py``, ``compose``):

1. Windows. A document of more than ``window_words`` words, counted the way
   GLiNER splits words, is read as windows of ``window_words`` words, each
   sharing ``window_overlap`` words with the next. A shorter document is read
   whole. Every window's spans are offset back to the document.
2. Members. Every member reads every window with the request's labels at its
   own defaults. Spans scoring below ``score_floor`` are dropped.
3. Union. Spans with the same start, end and label are one span at the higher
   score. Overlapping spans with different extents or labels are all kept.
4. Names. With ``propagate_label`` set, every other whole-word, case-sensitive
   occurrence of a token of at least ``name_min_letters`` letters from a span
   of that label, not already inside a span, becomes a span of that label. It
   scores as the best span the token came from.

The first member is the model config's own ``hf_id`` at its pinned revision.
The others are named by the ``members`` load-time option, each with a pinned
revision and any ``GLiNERAdapter`` load-time options. A request without labels
uses ``default_labels``. Every member's input tokens are counted for every
window it reads, so a document is metered once per member, and the words two
windows share are counted in both.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.config.model import is_immutable_revision
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity

# GLiNER's WhitespaceTokenSplitter: a run of word characters, or one other character.
_GLINER_WORD = re.compile(r"\w+(?:[-_]\w+)*|\S")
# A letter: a word character that is neither a digit nor an underscore.
_NOT_LETTER = r"[^\W\d_]"

# GLiNERAdapter load-time options a member may set.
_MEMBER_OPTIONS = frozenset(
    {"threshold", "flat_ner", "multi_label", "merge_adjacent_entities", "max_prompt_tokens", "batch_size"}
)
# Runtime options this adapter reads. Members always run at their own defaults.
_RUNTIME_OPTIONS = frozenset(
    {"score_floor", "window_words", "window_overlap", "propagate_label", "name_min_letters", "default_labels"}
)

_ERR_REQUIRES_LABELS = "This model needs labels: pass labels, or set the default_labels option"


def word_windows(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """``size``-word windows sharing ``overlap`` words, as (start, end) character ranges.

    A text of at most ``size`` words is one range covering all of it.
    """
    words = [match.span() for match in _GLINER_WORD.finditer(text)]
    if len(words) <= size:
        return [(0, len(text))]
    windows = []
    for first in range(0, len(words), size - overlap):
        last = min(first + size, len(words)) - 1
        windows.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return windows


class Span(NamedTuple):
    """A span found in a document, in document offsets."""

    start: int
    end: int
    label: str
    score: float


def union_spans(member_spans: Sequence[Sequence[Span]], floor: float) -> list[Span]:
    """Every member's spans at ``floor`` or above, one per (start, end, label) at its best score."""
    merged: dict[tuple[int, int, str], Span] = {}
    for spans in member_spans:
        for span in spans:
            if span.score < floor:
                continue
            key = (span.start, span.end, span.label)
            if key not in merged or span.score > merged[key].score:
                merged[key] = span
    return sorted(merged.values(), key=lambda span: (span.start, span.end))


def propagate_names(text: str, spans: Sequence[Span], label: str, min_letters: int) -> list[Span]:
    """Spans for every other whole-word, case-sensitive occurrence of a name token of ``label`` spans.

    A token is a run of at least ``min_letters`` letters inside a ``label``
    span. An occurrence already inside any span is not added again.
    """
    token_pattern = re.compile(f"{_NOT_LETTER}{{{min_letters},}}")
    best: dict[str, float] = {}
    for span in spans:
        if span.label != label:
            continue
        for token in token_pattern.findall(text[span.start : span.end]):
            best[token] = max(best.get(token, span.score), span.score)
    added: list[Span] = []
    for token in sorted(best):
        occurrence = re.compile(f"(?<!{_NOT_LETTER}){re.escape(token)}(?!{_NOT_LETTER})")
        for match in occurrence.finditer(text):
            start, end = match.span()
            if any(span.start <= start and span.end >= end for span in spans):
                continue
            added.append(Span(start, end, label, best[token]))
    return added


def compose(text: str, member_spans: Sequence[Sequence[Span]], options: Mapping[str, Any]) -> list[Entity]:
    """One document's union of member spans, then name propagation, sorted by (start, end)."""
    spans = union_spans(member_spans, options["score_floor"])
    label = options["propagate_label"]
    if label:
        spans = spans + propagate_names(text, spans, label, options["name_min_letters"])
    return [
        Entity(text=text[span.start : span.end], label=span.label, score=span.score, start=span.start, end=span.end)
        for span in sorted(spans, key=lambda span: (span.start, span.end))
    ]


class GLiNERUnionAdapter(BaseAdapter):
    """Serve the union of several GLiNER-family models as one extract model."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        members: Sequence[Mapping[str, Any]],
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        score_floor: float = 0.0,
        window_words: int = 300,
        window_overlap: int = 50,
        propagate_label: str | None = None,
        name_min_letters: int = 3,
        default_labels: Sequence[str] | None = None,
        **kwargs: Any,  # Accept extra args from the loader (e.g. max_seq_length)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path of the first member.
            members: The other members, each a mapping with ``model`` (a
                HuggingFace model ID), ``revision`` (a full commit SHA) and
                optional ``options`` (GLiNERAdapter load-time options).
            compute_precision: Compute precision for every member.
            revision: Pinned revision of the first member.
            score_floor: Lowest score a member's span may have to be kept (0-1).
            window_words: Words in one window of a long document.
            window_overlap: Words a window shares with the next.
            propagate_label: Label whose name tokens are masked wherever they
                recur; None or "" turns propagation off.
            name_min_letters: Fewest letters in a propagated token.
            default_labels: Labels for a request that names none.
            **kwargs: Additional loader arguments (ignored).
        """
        _ = kwargs
        if not members:
            raise ValueError("GLiNERUnionAdapter needs at least one other member in 'members'")
        self._model_name_or_path = str(model_name_or_path)
        self._defaults = self._resolve_options(
            {
                "score_floor": score_floor,
                "window_words": window_words,
                "window_overlap": window_overlap,
                "propagate_label": propagate_label,
                "name_min_letters": name_min_letters,
                "default_labels": list(default_labels) if default_labels is not None else None,
            },
            error=ValueError,
        )
        self._members = [
            GLiNERAdapter(model_name_or_path, compute_precision=compute_precision, revision=revision),
            *(self._member(spec, compute_precision) for spec in members),
        ]
        self._device: str | None = None

    @staticmethod
    def _member(spec: Mapping[str, Any], compute_precision: ComputePrecision) -> GLiNERAdapter:
        if not isinstance(spec, Mapping) or set(spec) - {"model", "revision", "options"}:
            raise ValueError("each GLiNERUnionAdapter member is a mapping of model, revision and options")
        model = spec.get("model")
        revision = spec.get("revision")
        if not isinstance(model, str) or not model:
            raise ValueError("a GLiNERUnionAdapter member needs a model")
        if not is_immutable_revision(revision):
            raise ValueError(f"member {model} needs a revision pinned to a full commit SHA")
        options = dict(spec.get("options") or {})
        unknown = sorted(set(options) - _MEMBER_OPTIONS)
        if unknown:
            raise ValueError(f"member {model} sets unsupported option(s) {', '.join(unknown)}")
        return GLiNERAdapter(model, compute_precision=compute_precision, revision=revision, **options)

    @staticmethod
    def _resolve_options(values: Mapping[str, Any], *, error: type[Exception]) -> dict[str, Any]:
        """Validate the composition options, raising ``error`` on a bad value."""
        floor = values["score_floor"]
        if isinstance(floor, bool) or not isinstance(floor, Real) or not math.isfinite(floor) or not 0 <= floor <= 1:
            raise error("score_floor must be a number between 0 and 1")
        size, overlap, letters = values["window_words"], values["window_overlap"], values["name_min_letters"]
        for name, value, low in (
            ("window_words", size, 2),
            ("window_overlap", overlap, 0),
            ("name_min_letters", letters, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise error(f"{name} must be an integer of at least {low}")
        if overlap >= size:
            raise error("window_overlap must be less than window_words")
        label = values["propagate_label"]
        if label is not None and not isinstance(label, str):
            raise error("propagate_label must be a label or null")
        labels = values["default_labels"]
        if labels is not None and (
            not isinstance(labels, list) or not labels or not all(isinstance(x, str) and x for x in labels)
        ):
            raise error("default_labels must be a non-empty list of labels")
        return {
            "score_floor": float(floor),
            "window_words": size,
            "window_overlap": overlap,
            "propagate_label": label or None,
            "name_min_letters": letters,
            "default_labels": labels,
        }

    def load(self, device: str) -> None:
        """Load every member onto ``device``."""
        for member in self._members:
            member.load(device)
        self._device = device

    def unload(self) -> None:
        """Unload every member."""
        for member in self._members:
            member.unload()
        self._device = None

    def memory_footprint(self) -> int:
        """Bytes every member's weights take on the device."""
        return sum(member.memory_footprint() for member in self._members)

    def _check_loaded(self) -> None:
        if self._device is None:
            raise RuntimeError("Model not loaded. Call load() first.")

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
        """Extract the union of every member's spans from each item.

        Args:
            items: Items to extract from (each needs text).
            labels: Entity types to extract; ``default_labels`` when omitted.
            output_schema: Unused (interface compatibility).
            instruction: Unused (interface compatibility).
            options: Overrides of score_floor, window_words, window_overlap,
                propagate_label, name_min_letters and default_labels. Members
                always run at their own defaults.
            prepared_items: Unused (interface compatibility).

        Returns:
            One list of entities per item, each with text, label, score,
            start and end, sorted by (start, end).
        """
        _ = (output_schema, instruction, prepared_items)
        self._check_loaded()
        opts = self._resolve_options(
            {**self._defaults, **{k: v for k, v in (options or {}).items() if k in _RUNTIME_OPTIONS}},
            error=InvalidInputError,
        )
        labels = labels or opts["default_labels"]
        if not labels:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)

        texts = [item.text if isinstance(item.text, str) else "" for item in items]
        windows = [word_windows(text, opts["window_words"], opts["window_overlap"]) for text in texts]
        rows = [Item(text=text[start:end]) for text, spans in zip(texts, windows, strict=True) for start, end in spans]
        owners = [index for index, spans in enumerate(windows) for _ in spans]
        offsets = [start for spans in windows for start, _ in spans]

        member_spans: list[list[list[Span]]] = [[[] for _ in texts] for _ in self._members]
        errors: list[ExtractItemError | None] = [None] * len(texts)
        counts: list[int] | None = [0] * len(texts)
        for position, member in enumerate(self._members):
            output = member.extract(rows, labels=labels)
            for row, entities in enumerate(output.entities):
                owner, offset = owners[row], offsets[row]
                for entity in entities:
                    start, end = entity.get("start"), entity.get("end")
                    if start is not None and end is not None:
                        member_spans[position][owner].append(
                            Span(start + offset, end + offset, entity["label"], float(entity["score"]))
                        )
                error = output.errors[row] if output.errors else None
                if error is not None and errors[owner] is None:
                    errors[owner] = error
            if output.input_token_counts is None:
                counts = None
            elif counts is not None:
                for row, count in enumerate(output.input_token_counts):
                    counts[owners[row]] += count

        entities = [
            [] if errors[index] is not None else compose(text, [spans[index] for spans in member_spans], opts)
            for index, text in enumerate(texts)
        ]
        if counts is not None:
            counts = [0 if errors[index] is not None else count for index, count in enumerate(counts)]
        return ExtractOutput(
            entities=entities,
            errors=errors if any(error is not None for error in errors) else None,
            input_token_counts=counts,
        )
