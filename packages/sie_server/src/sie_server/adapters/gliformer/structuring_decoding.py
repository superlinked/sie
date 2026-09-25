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
scores, and keeps at most ``max_candidates`` of them per slot: the
highest-scoring, cut at a score so that tied spans are kept or dropped
together. Overlap removal takes spans best score first, so a cut changes a
slot's fields only by leaving out spans that rank below it. Everything else
is the package's own code, unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from sie_server.adapters.gliformer.span_decoding import MAX_SPAN_CANDIDATES


def make_structuring_decode(
    make_span: Callable[..., Any],
    unflatten_by_batch_origin: Callable[..., Any],
    *,
    max_candidates: int = MAX_SPAN_CANDIDATES,
) -> Callable[..., Any]:
    """A replacement for ``gliformer.tasks.structuring.decoder.StructuringDecoder.decode``.

    Args:
        make_span: The package's ``Span`` class.
        unflatten_by_batch_origin: The package's helper of that name.
        max_candidates: Most spans per record slot passed to overlap removal.

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
                max_candidates=max_candidates,
            )
            for anchor_idx, spans in slot_spans:
                spans = self.greedy_search(spans, flat_ner, multi_label)
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
    max_candidates: int,
) -> list[tuple[int, list[Any]]]:
    """Each open slot's candidate field spans, in the package's order.

    For one document row: ``membership_probs`` is ``(A, E)``, ``field_probs``
    ``(E, C)`` and ``span_idx`` ``(E, 2)``. A slot takes a proposal whose
    membership is not at or below the threshold, and a field of it whose
    probability is above the threshold and that has a name; the span's score
    is the smaller of the two probabilities. Spans come by proposal, then by
    field id, as the package builds them.
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
    names = [field_id_to_class.get(idx) for idx in range(class_count)]
    slots = []
    for anchor_idx in np.flatnonzero(open_slots).tolist():
        members = np.flatnonzero(member[anchor_idx])
        entity, field = np.nonzero(fields[members])
        entity = members[entity]
        score = np.minimum(membership[anchor_idx, entity], probability[entity, field])
        if score.size > max_candidates:
            keep = score >= _tie_safe_cut(score, max_candidates)
            entity, field, score = entity[keep], field[keep], score[keep]
        spans = [
            make_span(start=start, end=end, entity_type=names[label], score=value)
            for start, end, label, value in zip(
                bounds[entity, 0].tolist(), bounds[entity, 1].tolist(), field.tolist(), score.tolist(), strict=True
            )
        ]
        slots.append((anchor_idx, spans))
    return slots


def _tie_safe_cut(scores: np.ndarray, limit: int) -> np.float32:
    """Lowest score to keep so that at most ``limit`` scores reach it, ties kept together."""
    edge = np.partition(scores, scores.size - limit)[scores.size - limit]  # the limit-th highest
    if int((scores >= edge).sum()) <= limit:
        return edge
    above = scores[scores > edge]
    return above.min() if above.size else np.float32(np.inf)
