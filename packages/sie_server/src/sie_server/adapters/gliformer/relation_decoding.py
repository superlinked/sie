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
most ``max_words`` words and ``max_chars`` characters, and no more than the
document's allowance affords.
Past a bound, a document keeps the best-first prefix: the highest scores
first, and among equal scores the package's own order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
# Characters of that head and tail text (#378). Same rule as the record cap.
# The constant is MAX_RELATION_WORDS * 32: 32 characters per counted word,
# including the joining space. The bound is per document, so text averaging
# under 31 characters a word reaches the word cap first, and text made of
# very long tokens reaches this cap.
MAX_RELATION_CHARS = MAX_RELATION_WORDS * 32
# Allowance units per kept relation (building it) and per scored cell.
RELATION_UNITS = 2
_CELLS_PER_UNIT = 1024


def make_relation_decode(
    ner_decode: Callable[..., Any],
    unflatten_by_batch_origin: Callable[..., Any],
    *,
    max_relations: int = MAX_RELATIONS,
    max_words: int = MAX_RELATION_WORDS,
    max_chars: int = MAX_RELATION_CHARS,
) -> Callable[..., Any]:
    """A replacement for ``gliformer.tasks.joint_relex.decoder.JointRelexDecoder.decode``.

    Args:
        ner_decode: ``NERDecoder.decode``, which the relation decoder extends.
        unflatten_by_batch_origin: The package's helper of that name.
        max_relations: Most relations per document.
        max_words: Most head and tail words across a document's relations.
        max_chars: Most characters of decoded head and tail text per document.

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
                    max_chars=max_chars,
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
    max_chars: int = MAX_RELATION_CHARS,
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

    # Endpoint word widths are arithmetic on the span bounds. Character widths
    # read token text, so they are filled for every endpoint only when the
    # word total still fits. When the word cap already rejects the row, text
    # is read only for the best-first prefix that fits in the word budget.
    tokens = None if texts is None or source_batch_idx >= len(texts) else texts[source_batch_idx]
    widths = np.asarray([span.end - span.start + 1 for span in entities] + [0], dtype=np.int64)
    head_at = np.where((heads >= 0) & (heads < len(entities)), heads, len(entities))
    tail_at = np.where((tails >= 0) & (tails < len(entities)), tails, len(entities))
    word_cost = widths[head_at] + widths[tail_at]
    limit = max_relations if allowance is None else min(max_relations, allowance.affordable(RELATION_UNITS))
    word_total = int(word_cost.sum()) if word_cost.size else 0
    chars: np.ndarray | None = None
    char_of: Callable[[int], int] | None = None
    if pairs.size <= limit and word_total <= max_words:
        char_widths = np.asarray(
            [_span_chars(tokens, span.start, span.end) for span in entities] + [0],
            dtype=np.int64,
        )
        chars = char_widths[head_at] + char_widths[tail_at]
    else:
        cache: dict[int, int] = {}

        def char_of(index: int) -> int:
            total = 0
            for entity_id in (int(heads[index]), int(tails[index])):
                cached = cache.get(entity_id)
                if cached is None:
                    if 0 <= entity_id < len(entities):
                        entity = entities[entity_id]
                        cached = _span_chars(tokens, entity.start, entity.end)
                    else:
                        cached = 0
                    cache[entity_id] = cached
                total += cached
            return total

    keep = _best_first(scores, word_cost, limit, max_words, chars, max_chars, char_of=char_of)
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


def _best_first(
    scores: np.ndarray,
    words: np.ndarray,
    limit: int,
    max_words: int,
    chars: np.ndarray | None,
    max_chars: int,
    *,
    char_of: Callable[[int], int] | None = None,
) -> np.ndarray:
    """Positions of the kept relations, in their original order.

    Relations are taken best score first (stable), and taking stops at the
    first one that would exceed ``limit`` relations, ``max_words`` words, or
    ``max_chars`` characters of decoded text.

    ``chars`` is the per-relation character cost when it is already known.
    When the word total is over the cap, pass ``chars=None`` and ``char_of``
    so token text is read only for the prefix that still fits in the word
    budget.
    """
    count = int(scores.size)
    if count == 0:
        return np.arange(0)
    word_total = int(words.sum())
    if count <= limit and word_total <= max_words:
        if chars is None:
            if char_of is None:
                msg = "char_of is required when chars is omitted"
                raise ValueError(msg)
            chars = np.fromiter((char_of(index) for index in range(count)), dtype=np.int64, count=count)
        if int(chars.sum()) <= max_chars:
            return np.arange(count)
    ranked = np.argsort(-scores, kind="stable")[:limit]
    if ranked.size == 0:
        return ranked
    word_fits = np.cumsum(words[ranked]) <= max_words
    word_count = int(np.argmin(word_fits)) if not bool(word_fits.all()) else int(ranked.size)
    if word_count == 0:
        return np.empty(0, dtype=ranked.dtype)
    prefix = ranked[:word_count]
    if chars is None:
        if char_of is None:
            msg = "char_of is required when chars is omitted"
            raise ValueError(msg)
        prefix_chars = np.fromiter((char_of(int(index)) for index in prefix), dtype=np.int64, count=word_count)
    else:
        prefix_chars = chars[prefix]
    char_fits = np.cumsum(prefix_chars) <= max_chars
    kept = int(np.argmin(char_fits)) if not bool(char_fits.all()) else word_count
    return np.sort(prefix[:kept])
