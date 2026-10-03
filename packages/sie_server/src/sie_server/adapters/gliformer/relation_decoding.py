"""Bounded relation decoding for GLiFormer's joint relation head.

GLiFormer's ``JointRelexDecoder.decode`` reads each document's relation
scores one cell at a time in Python, over every entity pair and relation
type: 100 entities and 20 types are 198,000 cells, about half a second per
document even when no cell passes the threshold. Every relation it keeps
also copies its head and tail text.

:func:`make_relation_decode` builds a drop-in ``decode`` that finds the
passing cells with array operations and returns the same relations, in the
same order, with the same scores. It also bounds each document's relations:
at most ``max_relations`` of them, whose heads and tails together span at
most ``max_words`` words, and no more than the document's allowance affords.
Past a bound, a document keeps the best-first prefix: the highest scores
first, and among equal scores the package's own order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from sie_server.adapters.gliformer.span_decoding import Allowance, row_allowance

# Relations one document keeps. The joint head scores at most 100 entities, so
# 100 x 99 ordered pairs x 20 relation types = 198,000 cells can pass.
MAX_RELATIONS = 65536
# Words of head and tail text across one document's relations. Each relation
# copies both endpoints' text, and endpoints can be long.
MAX_RELATION_WORDS = 262144
# Allowance units per kept relation (building it) and per scored cell.
RELATION_UNITS = 2
_CELLS_PER_UNIT = 1024


def make_relation_decode(
    ner_decode: Callable[..., Any],
    unflatten_by_batch_origin: Callable[..., Any],
    *,
    max_relations: int = MAX_RELATIONS,
    max_words: int = MAX_RELATION_WORDS,
) -> Callable[..., Any]:
    """A replacement for ``gliformer.tasks.joint_relex.decoder.JointRelexDecoder.decode``.

    Args:
        ner_decode: ``NERDecoder.decode``, which the relation decoder extends.
        unflatten_by_batch_origin: The package's helper of that name.
        max_relations: Most relations per document.
        max_words: Most head and tail words across a document's relations.

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
        **kwargs: Any,
    ) -> list[list[dict]]:
        """Decode joint relex predictions into relation triples with entity spans."""
        if model_output.joint_rel_logits is None or model_output.joint_rel_idx is None:
            return []

        if threshold is None:
            threshold = self.threshold

        # 1. Decode NER entities first (returns B x groups x spans)
        ner_id_to_classes = self._get_ner_id_to_classes(classes_mapping)
        entities = ner_decode(
            self,
            model_output,
            classes_mapping=ner_id_to_classes,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
            **kwargs,
        )

        # Flatten back to BN-indexed list of span lists for per-group entity lookup
        flat_entities = [spans for batch_groups in entities for spans in batch_groups]

        # 2. Build relation class mappings
        rel_id_to_classes = self._get_rel_id_to_classes(classes_mapping)
        entity_index_maps = self._build_entity_index_maps(
            flat_entities,
            getattr(model_output, "joint_rel_entity_spans", None),
            getattr(model_output, "joint_rel_entity_class_idx", None),
            ner_id_to_classes,
        )

        # 3. Decode relation triples
        probs = torch.sigmoid(model_output.joint_rel_logits)
        pair_idx = model_output.joint_rel_idx
        pair_mask = model_output.joint_rel_mask
        batch_origin = getattr(model_output, "joint_rel_batch_origin", None)
        flat_triples = []
        for bn in range(probs.shape[0]):
            batch_entities = flat_entities[bn] if bn < len(flat_entities) else []
            source_batch_idx = bn
            if batch_origin is not None and bn < batch_origin.numel():
                source_batch_idx = int(batch_origin[bn].item())
            rel_map = (
                rel_id_to_classes[bn]
                if isinstance(rel_id_to_classes, list) and bn < len(rel_id_to_classes)
                else rel_id_to_classes
            )
            entity_index_map = entity_index_maps[bn] if bn < len(entity_index_maps) else None
            flat_triples.append(
                _row_relations(
                    self,
                    probs[bn],
                    pair_idx[bn],
                    None if pair_mask is None else pair_mask[bn],
                    threshold,
                    rel_map,
                    entity_index_map,
                    batch_entities,
                    texts,
                    source_batch_idx,
                    allowance=row_allowance(bn),
                    max_relations=max_relations,
                    max_words=max_words,
                )
            )

        # Unflatten BN -> B
        return unflatten_by_batch_origin(
            flat_triples,
            model_output.joint_rel_batch_origin,
            model_output.batch_size,
        )

    return decode


def _row_relations(
    decoder: Any,
    probs: torch.Tensor,
    pair_idx: torch.Tensor,
    pair_mask: torch.Tensor | None,
    threshold: float,
    rel_map: Any,
    entity_index_map: dict[int, int] | None,
    entities: list[Any],
    texts: Any,
    source_batch_idx: int,
    *,
    allowance: Allowance | None,
    max_relations: int,
    max_words: int,
) -> list[dict]:
    """One document row's relation triples, in the package's order (pair, then type).

    A cell passes when its probability is not at or below the threshold,
    compared in double precision as the package compares ``.item()`` values;
    its pair is not masked out; and its type is named. Pairs whose endpoints
    do not map to decoded entities are dropped, as in the package.
    """
    pair_count, type_count = probs.shape
    if allowance is not None:
        # Only this document's own pairs are charged: the pair axis is padded
        # to the document with the most pairs in the batch.
        pairs_here = pair_count if pair_mask is None else int(pair_mask.sum())
        allowance.spend(-(-(pairs_here * type_count) // _CELLS_PER_UNIT))
    passing = ~(probs.detach().to(device="cpu", dtype=torch.float64) <= threshold)
    if pair_mask is not None:
        passing &= pair_mask.detach().cpu().bool()[:, None]
    if rel_map:
        named = torch.tensor([index in rel_map for index in range(type_count)], dtype=torch.bool)
        passing &= named[None, :]
    pairs, types = (part.numpy() for part in passing.nonzero(as_tuple=True))
    if pairs.size == 0:
        return []
    scores = probs.detach().to(device="cpu", dtype=torch.float32).numpy()[pairs, types]
    ends = pair_idx.detach().cpu().numpy()[pairs]
    heads, tails = ends[:, 0].astype(np.int64), ends[:, 1].astype(np.int64)
    if entity_index_map is not None:
        size = max(1, int(max(heads.max(), tails.max(), max(entity_index_map, default=-1))) + 1)
        lookup = np.full(size, -1, dtype=np.int64)
        for model_idx, decoded_idx in entity_index_map.items():
            if 0 <= model_idx < size:
                lookup[model_idx] = decoded_idx
        heads = np.where(heads >= 0, lookup[np.clip(heads, 0, size - 1)], -1)
        tails = np.where(tails >= 0, lookup[np.clip(tails, 0, size - 1)], -1)
        mapped = (heads >= 0) & (tails >= 0)
        pairs, types, scores, heads, tails = pairs[mapped], types[mapped], scores[mapped], heads[mapped], tails[mapped]
    if pairs.size == 0:
        return []

    # Endpoint widths, for the text bound; ids outside the decoded list
    # resolve to empty endpoints, as in the package.
    widths = np.asarray([span.end - span.start + 1 for span in entities] + [0], dtype=np.int64)
    head_words = widths[np.where((heads >= 0) & (heads < len(entities)), heads, len(entities))]
    tail_words = widths[np.where((tails >= 0) & (tails < len(entities)), tails, len(entities))]
    limit = max_relations if allowance is None else min(max_relations, allowance.affordable(RELATION_UNITS))
    keep = _best_first(scores, head_words + tail_words, limit, max_words)
    if allowance is not None:
        allowance.spend(RELATION_UNITS * int(keep.size))

    resolved: dict[int, dict[str, Any]] = {}

    def endpoint(entity_id: int) -> dict[str, Any]:
        base = resolved.get(entity_id)
        if base is None:
            if 0 <= entity_id < len(entities):
                entity = entities[entity_id]
                base = {
                    "start": entity.start,
                    "end": entity.end,
                    "text": decoder.resolve_span_text(texts, source_batch_idx, entity.start, entity.end),
                    "type": entity.entity_type,
                    "entity_idx": entity_id,
                }
            else:
                base = {"start": -1, "end": -1, "text": "", "type": "", "entity_idx": entity_id}
            resolved[entity_id] = base
        return dict(base)

    triples = []
    for index in keep.tolist():
        rel_name = rel_map.get(int(types[index]), str(int(types[index]))) if rel_map else str(int(types[index]))
        triples.append(
            {
                "head": endpoint(int(heads[index])),
                "tail": endpoint(int(tails[index])),
                "relation": rel_name,
                "score": float(scores[index]),
            }
        )
    return triples


def _best_first(scores: np.ndarray, words: np.ndarray, limit: int, max_words: int) -> np.ndarray:
    """Positions of the kept relations, in their original order.

    Relations are taken best score first (stable), and taking stops at the
    first one that would exceed ``limit`` relations or ``max_words`` words.
    """
    if scores.size <= limit and int(words.sum()) <= max_words:
        return np.arange(scores.size)
    ranked = np.argsort(-scores, kind="stable")[:limit]
    fits = np.cumsum(words[ranked]) <= max_words
    taken = ranked[: int(np.argmin(fits)) if not fits.all() else ranked.size]
    return np.sort(taken)
