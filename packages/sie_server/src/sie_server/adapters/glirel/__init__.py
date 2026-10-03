"""GLiREL adapter for zero-shot relation extraction.

GLiREL (Generalized Relation Extraction) models extract relations between
entities without fine-tuning. Given text, entities, and relation labels,
they output (head, relation, tail) triples with confidence scores.

Reference models:
- jackboyla/glirel-large-v0 (zero-shot relation extraction)
- jackboyla/glirel_re_large-v0 (relation-focused variant)

A request's relation types, which GLiREL encodes with every text, may have at
most 128 characters each and take at most ``max_prompt_tokens`` tokens together
(default 1024), and an item may carry at most ``MAX_ENTITIES`` (256) entities;
other requests are rejected with ``INVALID_INPUT``.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._prompt_limit import DEFAULT_MAX_PROMPT_TOKENS, PromptLimit, check_label_chars
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._word_window import SubwordCounter, WindowedSplitter, split_word_counter, subword_budget
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity, Relation

# Error messages
_ERR_REQUIRES_LABELS = "GLiREL requires labels parameter for relation extraction"
_ERR_REQUIRES_ENTITIES = "GLiREL requires entities in item metadata for relation extraction"
# GLiREL scores every pair of an item's entities (in Python while preparing
# the batch, and on the GPU), so an item may carry at most this many.
MAX_ENTITIES = 256
_TOKEN_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")
# Subword tokens a text may take per word GLiREL reads: its checkpoints read
# English, whose prose, code and logs run at up to 2.4 subwords per word.
_SUBWORDS_PER_WORD = 4


class GLiRELAdapter(BaseAdapter):
    """Adapter for GLiREL zero-shot relation extraction models.

    GLiREL extracts relations between entities. You provide:
    - Text to analyze
    - Entity spans in item metadata
    - Relation labels to look for (e.g., ["founded_by", "works_at"])

    Example usage:
        adapter = GLiRELAdapter("jackboyla/glirel-large-v0")
        adapter.load("cuda:0")
        results = adapter.extract(
            [Item(
                text="Apple Inc. was founded by Steve Jobs.",
                metadata={
                    "entities": [
                        {"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10},
                        {"text": "Steve Jobs", "label": "PER", "start": 26, "end": 36},
                    ]
                }
            )],
            labels=["founded_by", "works_at", "headquartered_in"],
        )
        # Returns: [{"relations": [
        #   {"head": "Apple Inc.", "tail": "Steve Jobs", "relation": "founded_by", "score": 0.92},
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
        threshold: float = 0.3,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,  # Accept extra args from loader
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path to GLiREL model.
            threshold: Minimum confidence score for relation extraction (0-1).
            max_prompt_tokens: Most tokens a request's relation types may take
                in the prompt encoded with each text (see ``_prompt_limit``).
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: Any = None  # GLiREL model type
        self._device: str | None = None
        # The words of a text GLiREL reads (see ``_tokenize``); None until loaded.
        self._words: WindowedSplitter | None = None
        self._prompt_limit = PromptLimit("GLiREL", max_prompt_tokens)
        # Tokens of the relation-type prompt, as the loaded model builds it; None until loaded.
        self._count_prompt: Any = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        # Import here to avoid dependency issues if glirel isn't installed
        from glirel import GLiREL  # ty:ignore[unresolved-import]

        self._device = device

        # Load model
        load_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            load_kwargs["revision"] = self._revision
        self._model = GLiREL.from_pretrained(self._model_name_or_path, **load_kwargs)

        # Move to device
        self._model = self._model.to(device)

        # Set eval mode
        self._model.eval()

        embeddings = self._model.token_rep_layer.bert_layer
        self._bound_words(
            embeddings.tokenizer,
            int(self._model.base_config.max_len),
            getattr(embeddings.model, "config", None),
        )
        self._count_prompt = _prompt_counter(
            embeddings.tokenizer, str(self._model.rel_token), str(self._model.sep_token)
        )

    def _bound_words(self, tokenizer: Any, max_words: int, encoder_config: Any = None) -> None:
        """Read at most ``max_words`` words of a text, and a bounded number of their subwords.

        GLiREL keeps its first ``max_len`` words and encodes every subword of
        each, without truncating; ``_word_window.read_window`` also reads a
        word longer than 256 characters in pieces and stops at the subword
        budget.
        """
        self._words = WindowedSplitter(
            _words,
            max_words=max_words,
            max_subwords=subword_budget(max_words, encoder_config, per_word=_SUBWORDS_PER_WORD),
            count_subwords=SubwordCounter(split_word_counter(tokenizer)),
        )

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
        """Extract relations from items.

        Args:
            items: List of items to extract from. Each item must have:
                - text: The text to analyze
                - metadata.entities: List of entity dicts with text, label, start, end
            labels: Relation types to extract (e.g., ["founded_by", "works_at"]).
                   Required for GLiREL models.
            output_schema: Unused for GLiREL (included for interface compatibility).
            instruction: Unused for GLiREL (included for interface compatibility).
            options: Adapter options to override model config defaults.
                    Supported: threshold (float), top_k (int).

        Returns:
            List of dicts, one per item, each containing:
                - "relations": List of extracted relations, each with:
                    - "head": Head entity text
                    - "tail": Tail entity text
                    - "relation": Relation type label
                    - "score": Confidence score (0-1)
                - "entities": Echo of input entities (if provided)
                - "data": Empty dict

        Raises:
            RuntimeError: If model not loaded.
            InvalidInputError: If labels are missing or too long, or an item lacks
                text or entities, carries more than ``MAX_ENTITIES`` entities, or
                has invalid entity offsets.
        """
        self._check_loaded()

        if not labels:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)
        self._check_prompt(labels)

        # Check every item before running any, so bad input fails as a 400 before model work.
        inputs: list[tuple[str, list[dict[str, Any]]]] = []
        for item in items:
            text = self._extract_text(item)
            entities = self._extract_entities(item)
            if not entities:
                raise InvalidInputError(_ERR_REQUIRES_ENTITIES)
            if len(entities) > MAX_ENTITIES:
                raise InvalidInputError(f"GLiREL items may carry at most {MAX_ENTITIES} entities in metadata")
            for entity in entities:
                self._validate_entity_span(entity, text)
            inputs.append((text, entities))

        all_entities = []
        all_relations = []
        for text, entities in inputs:
            tokens, token_offsets, read_end = self._tokenize(text)
            # Entities past the words GLiREL reads take no part, as when it truncates them itself.
            read_entities = [entity for entity in entities if self._is_read(entity, text, read_end)]
            ner_input = self._to_glirel_ner(read_entities, token_offsets)

            # Get options with fallback to model defaults
            opts = options or {}
            effective_threshold = opts.get("threshold", self._threshold)
            effective_top_k = opts.get("top_k", 10)

            # GLiREL expects pre-tokenized text and inclusive token offsets.
            raw_relations = []
            if ner_input:
                with torch.inference_mode():
                    raw_relations = self._model.predict_relations(
                        text=tokens,
                        labels=labels,
                        threshold=effective_threshold,
                        ner=ner_input,
                        top_k=effective_top_k,
                    )

            # Convert to proper Relation objects
            item_relations = []
            for rel in raw_relations:
                head_text = self._relation_entity_text(rel, "head", read_entities, ner_input)
                tail_text = self._relation_entity_text(rel, "tail", read_entities, ner_input)

                item_relations.append(
                    Relation(
                        head=head_text.strip(),
                        tail=tail_text.strip(),
                        relation=rel.get("label", ""),
                        score=float(rel.get("score", 0.0)),
                    )
                )

            # Echo input entities
            item_entities = []
            for ent in entities:
                item_entities.append(
                    Entity(
                        text=ent.get("text", ""),
                        label=ent.get("label", ""),
                        score=ent.get("score", 1.0),
                        start=ent.get("start"),
                        end=ent.get("end"),
                    )
                )

            all_entities.append(item_entities)
            all_relations.append(item_relations)

        return ExtractOutput(entities=all_entities, relations=all_relations)

    def _check_prompt(self, labels: list[str]) -> None:
        """Reject a request whose relation types take more than ``max_prompt_tokens``.

        GLiREL encodes the relation types with every text, and GLiREL output
        carries no input token count.

        Raises:
            InvalidInputError: The prompt is too long, or a relation type is not a string.
        """
        check_label_chars("GLiREL", "labels", labels)
        count = self._count_prompt

        def tokens() -> int:
            return count(labels) if count is not None else 0

        self._prompt_limit.check(labels, tokens, tuple(labels))

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="GLiREL adapter"))
        return item.text

    def _extract_entities(self, item: Item) -> list[dict[str, Any]]:
        """Extract entities from item metadata."""
        metadata = item.metadata
        if metadata is None:
            return []
        entities = metadata.get("entities", [])
        if not isinstance(entities, list):
            raise InvalidInputError("GLiREL item metadata.entities must be a list")
        return entities

    def _tokenize(self, text: str) -> tuple[list[str], list[tuple[int, int]], int]:
        """Tokenize text like GLiREL, keeping the words it reads: ``(words, offsets, end of the text read)``.

        Once loaded, the words are the bounded window of ``_bound_words``;
        before that, every word of the text.
        """
        if self._words is None:
            words, read_end = list(_words(text)), len(text)
        else:
            window = self._words.window(text)
            words = window.words
            read_end = len(text) if window.cut is None else words[-1][2]
        return [word for word, _, _ in words], [(start, end) for _, start, end in words], read_end

    @staticmethod
    def _validate_entity_span(entity: Any, text: str) -> None:
        """Check an entity's character offsets wherever it lies in the text.

        Raises:
            InvalidInputError: The entity is not an object, its offsets are not
                integers with ``0 <= start < end <= len(text)``, or they cover no
                text token.
        """
        if not isinstance(entity, dict):
            raise InvalidInputError("GLiREL entities must be objects")
        start = entity.get("start")
        end = entity.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            msg = (
                "GLiREL entity metadata requires integer character offsets with "
                f"0 <= start < end <= {len(text)} (the text's length)"
            )
            raise InvalidInputError(msg)
        # Every non-space character is part of a GLiREL token.
        if not text[start:end].strip():
            msg = f"GLiREL entity span [{start}, {end}) does not cover any text token"
            raise InvalidInputError(msg)

    @staticmethod
    def _is_read(entity: dict[str, Any], text: str, read_end: int) -> bool:
        """Whether an entity lies in the text GLiREL reads (anything past it but spaces counts against it)."""
        start = entity["start"]
        end = entity["end"]
        if end <= read_end:
            return True
        return start < read_end and not text[read_end:end].strip()

    @staticmethod
    def _to_glirel_ner(
        entities: list[dict[str, Any]],
        token_offsets: list[tuple[int, int]],
    ) -> list[list[Any]]:
        """Convert SIE's character-offset entities to GLiREL token spans."""
        ner_input: list[list[Any]] = []
        for entity in entities:
            start_char = entity.get("start")
            end_char = entity.get("end")
            if not isinstance(start_char, int) or not isinstance(end_char, int) or start_char >= end_char:
                msg = "GLiREL entity metadata requires integer character offsets with start < end"
                raise InvalidInputError(msg)

            covered_tokens = [
                index
                for index, (token_start, token_end) in enumerate(token_offsets)
                if token_end > start_char and token_start < end_char
            ]
            if not covered_tokens:
                msg = f"GLiREL entity span [{start_char}, {end_char}) does not cover any text token"
                raise InvalidInputError(msg)

            ner_input.append(
                [
                    covered_tokens[0],
                    covered_tokens[-1],
                    entity.get("label", "ENTITY"),
                    entity.get("text", ""),
                ]
            )
        return ner_input

    @staticmethod
    def _relation_entity_text(
        relation: dict[str, Any],
        role: str,
        entities: list[dict[str, Any]],
        ner_input: list[list[Any]],
    ) -> str:
        """Use the caller's original entity text when GLiREL identifies its span."""
        position = relation.get(f"{role}_pos")
        if isinstance(position, list) and len(position) == 2:
            for entity, ner_span in zip(entities, ner_input):
                if position == [ner_span[0], ner_span[1] + 1]:
                    return str(entity.get("text", ""))

        relation_text = relation.get(f"{role}_text", "")
        if isinstance(relation_text, list):
            relation_text = " ".join(str(token) for token in relation_text)
            relation_text = re.sub(r"\s+([,.;:!?%])", r"\1", relation_text)
        return str(relation_text)


def _prompt_counter(tokenizer: Any, rel_token: str, sep_token: str) -> Any:
    """Tokens of GLiREL's prompt for a list of relation types: ``[REL] type ... [REL] type [SEP]``."""

    def count(labels: list[str]) -> int:
        words = [word for label in labels for word in (rel_token, label)] + [sep_token]
        encoding = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        return len(encoding["input_ids"])

    return count


def _words(text: str) -> Iterator[tuple[str, int, int]]:
    """GLiREL's words of ``text`` with their character offsets."""
    for match in _TOKEN_PATTERN.finditer(text):
        yield match.group(), match.start(), match.end()
