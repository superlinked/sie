"""GLiNER adapter for zero-shot NER extraction.

GLiNER (Generalized NER) models perform named entity recognition on arbitrary
entity types without fine-tuning. Given text and a list of entity labels,
they extract spans matching those labels.

Also supports NuNER models (token-based) with merge_adjacent_entities option.

Reference models:
- urchade/gliner_multi-v2.1 (multilingual, recommended)
- urchade/gliner_large-v2.1 (English, larger)
- urchade/gliner_small-v2.1 (English, smaller/faster)
- numind/NuNER_Zero (token-based, requires merge_adjacent_entities=True)
- numind/NuNER_Zero-span (span-based, works without merging)
- knowledgator/gliner-relex-large-v1.0 (joint entities and relations)

Joint entity-relation ("relex") models also extract relations between the
entities they find when a request names relation types in
``options["relation_labels"]``. Without it they return entities only.
"""

import math
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.extract_cost import MAX_EXTRACT_LABELS
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity, Relation

# Error messages
_ERR_REQUIRES_LABELS = "GLiNER requires labels parameter for extraction"
_ERR_REQUIRES_NON_BLANK_TEXT = "GLiNER requires non-blank text for extraction"
_ERR_PROMPT_EXHAUSTS_DOCUMENT = "GLiNER label prompt leaves no document tokens for extraction"
# Joint entity-relation models score every ordered pair of entity candidates
# inside the forward pass, so memory grows with the square of the candidate
# count. Candidates are the spans above the entity threshold, which a caller
# controls, and a long document can have thousands. Keep at most this many per
# item (in document order) for relation scoring; entity output is unaffected.
_MAX_RELATION_CANDIDATES = 100
_ERR_CANDIDATE_LAYOUT = "gliner returned an unsupported span candidate layout; the relation candidate cap cannot apply"
# Models with an adjacency layer use this threshold to pick entity pairs; never
# let it fall below the library's usual value because of a low entity threshold.
_MIN_ADJACENCY_THRESHOLD = 0.5
# Relex models turn every span above the entity threshold into a relation
# candidate inside the forward pass. Near zero that is nearly every span of the
# document, so relex requests need at least this entity threshold.
_MIN_RELEX_THRESHOLD = 0.1
_ERR_NO_RELATIONS = (
    "This GLiNER model does not extract relations; options.relation_labels needs a joint "
    "entity-relation model such as knowledgator/gliner-relex-large-v1.0"
)


class GLiNERAdapter(BaseAdapter):
    """Adapter for GLiNER zero-shot NER models.

    GLiNER extracts entities of any specified type from text without
    model fine-tuning. You provide entity labels (e.g., ["person", "organization"])
    and the model returns matching spans with confidence scores.

    Example usage:
        adapter = GLiNERAdapter("urchade/gliner_multi-v2.1")
        adapter.load("cuda:0")
        results = adapter.extract(
            [Item(text="Apple Inc. was founded by Steve Jobs.")],
            labels=["person", "organization"],
        )
        # Returns: [{"entities": [
        #   {"text": "Apple Inc.", "label": "organization", "score": 0.95, "start": 0, "end": 10},
        #   {"text": "Steve Jobs", "label": "person", "score": 0.92, "start": 26, "end": 36},
        # ]}]
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model",),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        merge_adjacent_entities: bool = False,
        relation_threshold: float | None = None,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path to GLiNER model.
            threshold: Minimum confidence score for entity extraction (0-1).
            flat_ner: If True, enforce non-overlapping entities (recommended).
            multi_label: If True, allow same span to have multiple labels.
            merge_adjacent_entities: If True, merge adjacent entities with same label.
                Required for token-based models like numind/NuNER_Zero.
            relation_threshold: Minimum relation score (0-1) for joint
                entity-relation models. None uses the entity threshold, as the
                gliner library does.
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._flat_ner = flat_ner
        self._multi_label = multi_label
        self._merge_adjacent_entities = merge_adjacent_entities
        self._relation_threshold = relation_threshold
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: Any = None  # GLiNER model type
        self._device: str | None = None
        # True for joint entity-relation models, whose config names a relations layer.
        self._extracts_relations = False

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        # Import here to avoid dependency issues if gliner isn't installed
        from gliner import GLiNER  # ty:ignore[unresolved-import]

        self._device = device

        # Determine torch dtype from precision
        dtype = torch.float32
        if device != "cpu":
            if self._compute_precision == "float16":
                dtype = torch.float16
            elif self._compute_precision == "bfloat16":
                dtype = torch.bfloat16

        # Load model
        load_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            load_kwargs["revision"] = self._revision
        self._model = GLiNER.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )

        # Move to device with precision
        if device == "cpu":
            self._model = self._model.to(device)
        else:
            self._model = self._model.to(device, dtype=dtype)
        self._extracts_relations = getattr(self._model.config, "relations_layer", None) is not None
        if self._extracts_relations:
            _cap_relation_candidates(self._model.model, _MAX_RELATION_CANDIDATES)

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
        """Extract entities from items.

        Args:
            items: List of items to extract from (must have text).
            labels: Entity types to extract (e.g., ["person", "organization"]).
                   Required for GLiNER models.
            output_schema: Unused for GLiNER (included for interface compatibility).
            instruction: Unused for GLiNER (included for interface compatibility).
            options: Adapter options to override model config defaults.
                    Supported: threshold (float), flat_ner (bool), multi_label (bool),
                    merge_adjacent_entities (bool). Joint entity-relation models
                    also take relation_labels (list of relation types to extract
                    between the found entities) and relation_threshold (float).

        Returns:
            List of dicts, one per item, each containing:
                - "entities": List of extracted entities, each with:
                    - "text": The extracted text span
                    - "label": Entity type label
                    - "score": Confidence score (0-1)
                    - "start": Start character offset
                    - "end": End character offset
                - "relations": With relation_labels, relation triples between the
                  extracted entities (head and tail entity text, relation type, score).
                - "data": Empty dict (GLiNER doesn't produce structured data)

        Raises:
            RuntimeError: If model not loaded.
            InvalidInputError: If labels are missing, items lack text, options
                are malformed, relation options are sent to a model without
                relations, or the label and relation prompt leaves no room
                for the document.
        """
        self._check_loaded()

        if not labels:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)

        opts = options or {}
        relation_labels = self._validate_relation_labels(opts.get("relation_labels"), labels)
        if relation_labels and not self._extracts_relations:
            raise InvalidInputError(_ERR_NO_RELATIONS)

        # Extract texts from all items
        texts = [self._extract_text(item) for item in items]
        if any(not text.strip() for text in texts):
            raise InvalidInputError(_ERR_REQUIRES_NON_BLANK_TEXT)

        # Meter the exact post-word-truncation document window before GPU work.
        # Besides producing the authoritative terminal counts, this rejects a
        # finite-tokenizer prompt that leaves no represented document subword.
        input_token_counts = self._doc_input_token_counts(texts, labels, relation_labels)

        # Get options with fallback to model defaults
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold), "threshold")
        if self._extracts_relations and effective_threshold < _MIN_RELEX_THRESHOLD:
            raise InvalidInputError(
                f"GLiNER entity-relation models need a threshold of at least {_MIN_RELEX_THRESHOLD}"
            )
        effective_flat_ner = opts.get("flat_ner", self._flat_ner)
        effective_multi_label = opts.get("multi_label", self._multi_label)
        merge_adjacent = opts.get("merge_adjacent_entities", self._merge_adjacent_entities)

        # Joint entity-relation models return (entities, relations) unless told
        # otherwise, so always say which one is wanted. Other GLiNER models keep
        # exactly the call they have always had.
        relation_kwargs: dict[str, Any] = {}
        if self._extracts_relations:
            relation_kwargs["relations"] = relation_labels
            relation_kwargs["return_relations"] = bool(relation_labels)
            relation_kwargs["adjacency_threshold"] = max(effective_threshold, _MIN_ADJACENCY_THRESHOLD)
            relation_threshold = self._validate_relation_threshold(
                opts.get("relation_threshold", self._relation_threshold)
            )
            if relation_labels and relation_threshold is not None:
                relation_kwargs["relation_threshold"] = relation_threshold

        # Use batch prediction for efficiency (24x speedup vs single item loop)
        with torch.inference_mode():
            prediction = self._model.inference(
                texts,
                labels,
                threshold=effective_threshold,
                flat_ner=effective_flat_ner,
                multi_label=effective_multi_label,
                **relation_kwargs,
            )
        if relation_labels:
            batch_entities, batch_relations = prediction
        else:
            batch_entities, batch_relations = prediction, None

        # Convert to our format
        all_entities = []
        for text, entities in zip(texts, batch_entities):
            entity_results = []
            for entity in entities:
                entity_results.append(
                    Entity(
                        text=entity["text"],
                        label=entity["label"],
                        score=float(entity["score"]),
                        start=entity["start"],
                        end=entity["end"],
                    )
                )

            # Merge adjacent entities if enabled (for token-based models like NuNER)
            if merge_adjacent:
                entity_results = self._merge_entities(entity_results, text)

            all_entities.append(entity_results)

        all_relations = None
        if batch_relations is not None:
            if len(batch_relations) != len(texts):
                raise ValueError("GLiNER returned relations for a different number of items")
            all_relations = [self._format_relations(relations) for relations in batch_relations]

        return ExtractOutput(entities=all_entities, relations=all_relations, input_token_counts=input_token_counts)

    @staticmethod
    def _validate_relation_labels(value: Any, labels: list[str]) -> list[str]:
        """Return the requested relation types (empty when none were asked for)."""
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise InvalidInputError("GLiNER relation_labels must be a list of relation types")
        if any(not isinstance(label, str) or not label.strip() for label in value):
            raise InvalidInputError("GLiNER relation_labels must be non-empty strings")
        relation_labels = [label.strip() for label in value]
        if len(set(relation_labels)) != len(relation_labels):
            raise InvalidInputError("GLiNER relation_labels must be unique")
        if len(relation_labels) + len(labels) > MAX_EXTRACT_LABELS:
            raise InvalidInputError(
                f"GLiNER labels and relation_labels must contain at most {MAX_EXTRACT_LABELS} entries together"
            )
        return relation_labels

    @classmethod
    def _validate_relation_threshold(cls, value: Any) -> float | None:
        if value is None:
            return None
        return cls._validate_threshold(value, "relation_threshold")

    @staticmethod
    def _validate_threshold(value: Any, name: str) -> float:
        message = f"GLiNER {name} must be a finite number between 0 and 1"
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidInputError(message)
        try:
            threshold = float(value)
        except OverflowError as exc:
            raise InvalidInputError(message) from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise InvalidInputError(message)
        return threshold

    @staticmethod
    def _format_relations(relations: list[dict[str, Any]] | None) -> list[Relation]:
        """Convert gliner relation dicts to SIE relations, highest score first."""
        formatted = [
            Relation(
                head=relation["head"]["text"],
                tail=relation["tail"]["text"],
                relation=relation["relation"],
                score=float(relation["score"]),
            )
            for relation in relations or []
        ]
        formatted.sort(key=lambda r: (-r["score"], r["relation"], r["head"], r["tail"]))
        return formatted

    def _doc_input_token_counts(
        self,
        texts: list[str],
        labels: list[str],
        relation_labels: list[str] | None = None,
    ) -> list[int] | None:
        """Count the document subwords represented by GLiNER's real processor.

        Classic GLiNER first splits and truncates each document in WORDS, then
        prepends the label prompt and transformer-tokenizes that retained word
        window. Tokenizing the original string separately therefore counts
        discarded tail text and is not an authoritative execution meter.

        Delegate splitting, processor-specific truncation, label mapping, and
        prompt construction to the pinned GLiNER processor. ``word_ids`` then
        identifies every represented document subword while excluding prompt
        words. Attention masking excludes padding; tokenizer specials remain
        billable, matching the existing document-token contract. Batches match
        GLiNER inference's default batch size so a finite tokenizer cap is
        observed identically. Joint entity-relation models also put the
        relation types in the prompt, so they are passed through as well.
        """
        processor = getattr(self._model, "data_processor", None)
        prepare_inputs = getattr(self._model, "prepare_inputs", None)
        prepare_base_input = getattr(self._model, "prepare_base_input", None)
        if processor is None or prepare_inputs is None or prepare_base_input is None:
            return None

        try:
            split_texts, _, _ = prepare_inputs(texts)
            raw_items = prepare_base_input(split_texts)
            counts: list[int] = []
            has_document_subwords: list[bool] = []
            for start in range(0, len(raw_items), 8):
                if self._extracts_relations:
                    raw_batch = processor.collate_raw_batch(
                        raw_items[start : start + 8],
                        entity_types=labels,
                        relation_types=relation_labels or [],
                    )
                    retained_words = raw_batch["tokens"]
                    entity_mappings = raw_batch["classes_to_id"]
                    encoded = processor.tokenize_inputs(
                        retained_words, entity_mappings, relations=raw_batch["rel_class_to_ids"]
                    )
                else:
                    raw_batch = processor.collate_raw_batch(
                        raw_items[start : start + 8],
                        entity_types=labels,
                    )
                    retained_words = raw_batch["tokens"]
                    entity_mappings = raw_batch["classes_to_id"]
                    encoded = processor.tokenize_inputs(retained_words, entity_mappings)
                for batch_index in range(len(retained_words)):
                    word_ids = encoded.word_ids(batch_index)
                    attention_mask = encoded["attention_mask"][batch_index].tolist()
                    words_mask = encoded["words_mask"][batch_index].tolist()
                    if len(word_ids) != len(attention_mask) or len(word_ids) != len(words_mask):
                        return None
                    first_document_index = next(
                        (index for index, word_mask in enumerate(words_mask) if word_mask > 0),
                        None,
                    )
                    first_document_word_id = (
                        word_ids[first_document_index] if first_document_index is not None else None
                    )
                    represented_document = [
                        bool(
                            attended
                            and first_document_word_id is not None
                            and word_id is not None
                            and word_id >= first_document_word_id
                        )
                        for attended, word_id in zip(attention_mask, word_ids)
                    ]
                    has_document_subwords.append(any(represented_document))
                    counts.append(
                        sum(
                            bool(attended)
                            and (
                                word_id is None
                                or (first_document_word_id is not None and word_id >= first_document_word_id)
                            )
                            for attended, word_id in zip(attention_mask, word_ids)
                        )
                    )
        except Exception:  # noqa: BLE001 — metering must never fail an extraction
            return None
        if len(counts) != len(texts) or len(has_document_subwords) != len(texts):
            return None
        if not all(has_document_subwords):
            raise InvalidInputError(_ERR_PROMPT_EXHAUSTS_DOCUMENT)
        return counts

    def _merge_entities(self, entities: list[Entity], text: str) -> list[Entity]:
        """Merge adjacent entities with the same label.

        Token-based models like NuNER_Zero output per-token predictions that need
        to be merged into contiguous spans. For example:
            [Entity(text="Steve", label="person"), Entity(text="Jobs", label="person")]
        becomes:
            [Entity(text="Steve Jobs", label="person")]

        Args:
            entities: List of Entity objects (TypedDict, i.e. dict).
            text: Original text (needed to extract merged spans).

        Returns:
            List of merged entities.
        """
        if not entities:
            return []

        # Sort by start position to ensure correct merging order
        # Entity is a TypedDict (dict), so use dict access
        sorted_entities = sorted(entities, key=lambda e: e.get("start") or 0)

        merged = []
        current = sorted_entities[0]

        for next_entity in sorted_entities[1:]:
            # Check if adjacent (touching or 1 char gap) and same label
            is_adjacent = (next_entity.get("start") or 0) <= (current.get("end") or 0) + 1
            same_label = next_entity["label"] == current["label"]

            if is_adjacent and same_label:
                # Merge: extend current entity to include next
                next_end = next_entity.get("end")
                current_end = current.get("end")
                if current_end is None:
                    new_end = next_end
                elif next_end is None:
                    new_end = current_end
                else:
                    new_end = max(current_end, next_end)
                new_start = current.get("start")
                new_text = text[new_start:new_end] if new_start is not None and new_end is not None else current["text"]
                new_score = max(current["score"], next_entity["score"])
                current = Entity(
                    text=new_text,
                    label=current["label"],
                    score=new_score,
                    start=new_start,
                    end=new_end,
                )
            else:
                merged.append(current)
                current = next_entity

        merged.append(current)
        return merged

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER adapter"))
        return item.text


def _cap_relation_candidates(model: Any, limit: int) -> None:
    """Keep at most ``limit`` entity candidates per item for relation scoring.

    gliner's relex models pass every span above the entity threshold to the
    relation layers, which build every ordered pair of candidates.
    ``represent_spans`` returns the candidate representations, validity mask,
    and span boundaries packed at the front of each row, so slicing those
    tensors to their first ``limit`` columns bounds everything downstream to
    ``limit`` candidates (and ``limit**2`` pairs) per item. Entities are
    decoded from the span scores, so entity output does not change; relations
    among candidates past the first ``limit`` (in document order) are dropped.
    Any other output layout raises instead of running uncapped.
    """
    represent_spans = model.represent_spans

    def capped(*args: Any, **kwargs: Any) -> Any:
        outputs = represent_spans(*args, **kwargs)
        if not _has_candidate_layout(outputs):
            raise RuntimeError(_ERR_CANDIDATE_LAYOUT)
        width = outputs[2].size(1)
        if width <= limit:
            return outputs
        sliced = list(outputs)
        for index in range(1, len(sliced)):
            value = sliced[index]
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.size(1) == width:
                sliced[index] = value[:, :limit]
        return tuple(sliced)

    model.represent_spans = capped


def _has_candidate_layout(outputs: Any) -> bool:
    """Whether ``represent_spans`` returned (scores, reps, mask, spans, ...) as the cap expects."""
    if not isinstance(outputs, tuple) or len(outputs) < 4:
        return False
    reps, mask, spans = outputs[1:4]
    return (
        isinstance(mask, torch.Tensor)
        and mask.dim() == 2
        and isinstance(reps, torch.Tensor)
        and reps.dim() == 3
        and isinstance(spans, torch.Tensor)
        and spans.dim() == 3
        and reps.shape[:2] == mask.shape == spans.shape[:2]
    )
