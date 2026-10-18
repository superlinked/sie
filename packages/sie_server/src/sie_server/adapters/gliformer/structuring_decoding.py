"""Bounded record decoding for GLiFormer's structuring head.

GLiFormer's ``StructuringDecoder.decode`` turns the structuring head's
outputs into records: for every record slot (anchor) it keeps the entity
proposals whose membership probability passes the threshold and, for each,
the fields whose probability passes it, then removes overlapping spans. It
does this one slot, proposal, and field at a time with tensor indexing in
Python: about 30 microseconds per slot and proposal on CPU, so a dense
document with a few hundred proposals across the checkpoint's 100 slots
takes seconds.

:func:`make_structuring_decode` builds a drop-in ``decode`` that selects the
same spans with array operations, in the same order and with the same
scores, and keeps at most ``max_spans`` of them per document across all
slots: the ones each slot's overlap removal would take first (best score
first, and among equal scores in the package's order). Overlap removal
keeps spans best first, so a cut changes a slot's fields only by leaving out
spans that rank below every kept one. Everything else is the package's own
code, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

from sie_server.adapters.gliformer.span_decoding import _FLOAT32_BITS_ONE, Allowance, _score_cut, row_allowance

# Record-slot field spans one document keeps across all slots. Each open slot
# pairs every member proposal with every field, so real documents produce far
# more of these than entity spans: measured with both checkpoints on
# 2048-word documents, up to 15,291 at threshold 0.5 and 58,890 at 0.1 (a
# list of names with a records schema).
MAX_RECORD_SPANS = 65536
# Words of field text one document's records may hold after overlap removal.
# Each field copies its span's words, and nested overlap removal
# (flat_ner=false) can keep every span of a nested chain, so without a bound
# the text alone grows with spans x document length. Flat single-label
# removal keeps disjoint spans per slot: at most 100 slots x 2048 words.
MAX_RECORD_WORDS = 262144
# Characters of that field text (#378). The constant is MAX_RECORD_WORDS * 32:
# 32 characters per counted word, including the joining space. The bound is
# per document, so text averaging under 31 characters a word reaches the word
# cap first, and text made of very long tokens reaches this cap.
MAX_RECORD_CHARS = MAX_RECORD_WORDS * 32


def make_structuring_decode(
    make_span: Callable[..., Any],
    unflatten_by_batch_origin: Callable[..., Any],
    *,
    max_spans: int = MAX_RECORD_SPANS,
    max_words: int = MAX_RECORD_WORDS,
    max_chars: int = MAX_RECORD_CHARS,
) -> Callable[..., Any]:
    """A replacement for ``gliformer.tasks.structuring.decoder.StructuringDecoder.decode``.

    Args:
        make_span: The package's ``Span`` class.
        unflatten_by_batch_origin: The package's helper of that name.
        max_spans: Most field spans per document, across its record slots.
        max_words: Most words of field text per document, across its slots.
        max_chars: Most characters of decoded field text per document.

    Returns:
        The ``decode`` method.
    """

    def decode(
        self: Any,
        model_output: Any,
        classes_mapping: Any = None,
        threshold: float | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        texts: Any = None,
        objectness_threshold: float | None = None,
        preserve_empty_records: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Decode typed NER spans into unordered record-anchor groups.

        Field logits have shape ``(BN, E, C)`` and independent
        anchor-membership logits have shape ``(BN, A, E)``.
        """
        _ = kwargs
        membership_logits = getattr(model_output, self.span_logits_attr, None)
        field_logits = getattr(model_output, self.field_logits_attr, None)
        span_idx = getattr(model_output, self.span_idx_attr, None)
        span_mask = getattr(model_output, self.span_mask_attr, None)
        if membership_logits is None or span_idx is None or span_mask is None:
            return []

        if membership_logits.dim() != 3:
            raise ValueError(f"structuring_logits must have shape (BN, A, E), got {tuple(membership_logits.shape)}")
        if field_logits is None or field_logits.dim() != 3:
            raise ValueError("structuring_field_logits must have shape (BN, E, C)")

        batch_groups, anchor_count, entity_count = membership_logits.shape
        if field_logits.shape[:2] != (batch_groups, entity_count):
            raise ValueError(
                "structuring field logits must share the membership "
                f"(BN, E) axes; got {tuple(field_logits.shape[:2])} and "
                f"{(batch_groups, entity_count)}"
            )
        if span_idx.dim() != 3 or span_idx.shape != (batch_groups, entity_count, 2):
            raise ValueError(f"structuring_span_idx must have shape (BN, E, 2), got {tuple(span_idx.shape)}")
        if span_mask.shape != (batch_groups, entity_count):
            raise ValueError(f"structuring_span_mask must have shape (BN, E), got {tuple(span_mask.shape)}")

        class_count = field_logits.shape[2]
        membership_probs = torch.sigmoid(membership_logits)
        field_probs = torch.sigmoid(field_logits)
        context = self._prepare_decode_context(
            model_output,
            classes_mapping,
            batch_groups=batch_groups,
            anchor_count=anchor_count,
            device=membership_logits.device,
            threshold=threshold,
            objectness_threshold=objectness_threshold,
        )

        flat_results = []
        for batch_idx in range(batch_groups):
            text_idx = int(context.batch_origin[batch_idx].item())
            field_id_to_class = (
                context.id_to_fields[batch_idx]
                if batch_idx < len(context.id_to_fields) and context.id_to_fields[batch_idx]
                else {idx: str(idx) for idx in range(class_count)}
            )
            valid_entities = torch.where(span_mask[batch_idx, :entity_count].bool())[0]
            anchor_entries = []
            group_context = (
                context.multi_level_contexts[batch_idx] if batch_idx < len(context.multi_level_contexts) else None
            )
            slot_spans = _slot_spans(
                membership_probs[batch_idx],
                field_probs[batch_idx],
                span_idx[batch_idx],
                valid_entities,
                None if context.anchor_mask is None else context.anchor_mask[batch_idx],
                context.threshold,
                field_id_to_class,
                make_span=make_span,
                max_spans=max_spans,
                allowance=row_allowance(batch_idx),
            )
            kept = _within_word_limit(
                [(anchor_idx, self.greedy_search(spans, flat_ner, multi_label)) for anchor_idx, spans in slot_spans],
                max_words,
                tokens=None if texts is None or text_idx >= len(texts) else texts[text_idx],
                max_chars=max_chars,
            )
            for anchor_idx, spans in kept:
                fields = self._spans_to_fields(spans, texts, text_idx)
                anchor_entries.append(
                    {
                        "anchor_index": anchor_idx,
                        "fields": fields,
                        "presence_is_reliable": bool(
                            context.reliable_presence_mask is not None
                            and context.reliable_presence_mask[batch_idx, anchor_idx]
                        ),
                    }
                )
            flat_results.append(
                self._finalize_anchor_group(
                    anchor_entries,
                    relation_scores=(
                        context.relation_scores[batch_idx] if context.relation_scores is not None else None
                    ),
                    context=group_context,
                    preserve_empty_records=preserve_empty_records,
                )
            )

        return unflatten_by_batch_origin(flat_results, context.batch_origin, context.batch_size)

    return decode


def _slot_spans(
    membership_probs: torch.Tensor,
    field_probs: torch.Tensor,
    span_idx: torch.Tensor,
    valid_entities: torch.Tensor,
    anchor_mask: torch.Tensor | None,
    threshold: float,
    field_id_to_class: dict[int, str],
    *,
    make_span: Callable[..., Any],
    max_spans: int,
    allowance: Allowance | None = None,
) -> list[tuple[int, list[Any]]]:
    """Each open slot's candidate field spans, in the package's order.

    For one document row: ``membership_probs`` is ``(A, E)``, ``field_probs``
    ``(E, C)`` and ``span_idx`` ``(E, 2)``. A slot takes a proposal whose
    membership is not at or below the threshold, and a field of it whose
    probability is above the threshold and that has a name; the span's score
    is the smaller of the two probabilities. Spans come by slot, then by
    proposal, then by field id, as the package builds them.

    When more than ``max_spans`` spans qualify, the document keeps the first
    ``max_spans`` of a stable best-score-first ordering: every span above a
    score cut, then spans scoring just below it in their order. With an
    ``allowance``, each kept span costs a unit and finding a cut one unit
    per four membership and field cells, and the document keeps only as many
    spans as its allowance affords, the same best-first prefix.
    """
    anchor_count = membership_probs.shape[0]
    class_count = field_probs.shape[1]
    # The same comparisons as the package's decoder, on the same tensors.
    member = (~(membership_probs[:, valid_entities] <= threshold)).cpu().numpy()
    named = np.fromiter((idx in field_id_to_class for idx in range(class_count)), dtype=bool, count=class_count)
    fields = (field_probs[valid_entities] > threshold).cpu().numpy() & named
    membership = membership_probs[:, valid_entities].detach().to(device="cpu", dtype=torch.float32).numpy()
    probability = field_probs[valid_entities].detach().to(device="cpu", dtype=torch.float32).numpy()
    bounds = span_idx[valid_entities].cpu().numpy()
    open_slots = np.ones(anchor_count, dtype=bool) if anchor_mask is None else anchor_mask.bool().cpu().numpy()
    slots = np.flatnonzero(open_slots)
    member, membership = member[slots], membership[slots]
    names = [field_id_to_class.get(idx) for idx in range(class_count)]

    def masks(cut: np.float32 | None) -> tuple[np.ndarray, np.ndarray]:
        if cut is None:
            return member, fields
        return member & (membership >= cut), fields & (probability >= cut)

    def per_slot(cut: np.float32 | None) -> np.ndarray:
        """Spans per open slot at score ``cut``: members times their passing fields."""
        slot_members, entity_fields = masks(cut)
        return slot_members.astype(np.int64) @ entity_fields.sum(axis=1, dtype=np.int64)

    def slot_spans(position: int, cut: np.float32 | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slot_members, entity_fields = masks(cut)
        members = np.flatnonzero(slot_members[position])
        entity, field = np.nonzero(entity_fields[members])
        entity = members[entity]
        return entity, field, np.minimum(membership[position, entity], probability[entity, field])

    counts = per_slot(None)
    total = int(counts.sum())
    limit = max_spans if allowance is None else min(max_spans, allowance.affordable(1))
    cut: np.float32 | None = None
    quota = np.full(slots.size, -1, dtype=np.int64)  # spans tied just below the cut each slot keeps
    if total > limit:
        cut_cost = (member.size + fields.size) // 4
        if allowance is not None:
            limit = min(max_spans, allowance.affordable(1, reserve=cut_cost))
            if limit == 0:
                return [(anchor_idx, []) for anchor_idx in slots.tolist()]
            allowance.spend(cut_cost)
        cut = _score_cut(lambda value: int(per_slot(value).sum()), limit)
        above = per_slot(cut)
        missing = limit - int(above.sum())
        cut_bits = int(np.float32(cut).view(np.int32))
        quota[:] = 0
        if missing > 0 and 0 < cut_bits <= _FLOAT32_BITS_ONE + 1:
            ties = per_slot(np.int32(cut_bits - 1).view(np.float32)) - above
            before = np.cumsum(ties) - ties
            quota = np.clip(missing - before, 0, ties)

    result = []
    for position, anchor_idx in enumerate(slots.tolist()):
        if cut is None:
            entity, field, score = slot_spans(position, None)
        else:
            entity, field, score = slot_spans(position, cut)
            if quota[position] > 0:
                below = np.int32(int(np.float32(cut).view(np.int32)) - 1).view(np.float32)
                tie_entity, tie_field, tie_score = slot_spans(position, below)
                tied = np.flatnonzero(tie_score < cut)[: quota[position]]
                entity = np.concatenate([entity, tie_entity[tied]])
                field = np.concatenate([field, tie_field[tied]])
                score = np.concatenate([score, tie_score[tied]])
                order = np.lexsort((field, entity))
                entity, field, score = entity[order], field[order], score[order]
        spans = [
            make_span(start=start, end=end, entity_type=names[label], score=value)
            for start, end, label, value in zip(
                bounds[entity, 0].tolist(), bounds[entity, 1].tolist(), field.tolist(), score.tolist(), strict=True
            )
        ]
        result.append((anchor_idx, spans))
    if allowance is not None:
        allowance.spend(sum(len(spans) for _, spans in result))
    return result


def _span_chars(tokens: Sequence[str] | None, start: int, end: int) -> int:
    """Characters of one span's decoded text.

    The package copies ``" ".join(tokens[start:end + 1])``. With no tokens it
    copies nothing, so the character cap does not apply.
    """
    if not tokens or start >= len(tokens):
        return 0
    span = tokens[start : end + 1]
    if not span:
        return 0
    return sum(len(token) for token in span) + len(span) - 1


def _within_word_limit(
    slots: list[tuple[int, list[Any]]],
    max_words: int,
    *,
    tokens: Sequence[str] | None = None,
    max_chars: int = MAX_RECORD_CHARS,
) -> list[tuple[int, list[Any]]]:
    """Drop the lowest-scoring kept fields until their text fits both bounds.

    Fields are ranked best score first, and among equal scores by slot and
    then position. Taking stops at the first field that would exceed
    ``max_words`` words or ``max_chars`` characters of decoded text. The ones
    taken stay in their original order.

    When the word total is already over ``max_words``, token text is read
    only for the best-first prefix that still fits in that word budget. The
    tail the word cap rejects is not walked.
    """
    words = sum(span.end - span.start + 1 for _, spans in slots for span in spans)
    if words <= max_words:
        chars = sum(_span_chars(tokens, span.start, span.end) for _, spans in slots for span in spans)
        if chars <= max_chars:
            return slots
    ranked = sorted(
        ((slot, index, span) for slot, (_, spans) in enumerate(slots) for index, span in enumerate(spans)),
        key=lambda entry: -entry[2].score,
    )
    keep: set[tuple[int, int]] = set()
    used_words = 0
    used_chars = 0
    for slot, index, span in ranked:
        width = span.end - span.start + 1
        if used_words + width > max_words:
            break
        span_chars = _span_chars(tokens, span.start, span.end)
        if used_chars + span_chars > max_chars:
            break
        used_words += width
        used_chars += span_chars
        keep.add((slot, index))
    return [
        (anchor_idx, [span for index, span in enumerate(spans) if (slot, index) in keep])
        for slot, (anchor_idx, spans) in enumerate(slots)
    ]
