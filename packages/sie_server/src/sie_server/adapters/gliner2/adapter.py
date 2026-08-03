from __future__ import annotations

import math
import unicodedata
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item
from sie_server.types.responses import Classification, Entity, Relation

_ERR_REQUIRES_LABELS = "GLiNER2 requires labels parameter for extraction"
_STRUCTURE_NAME = "_sie_root"
_STRUCTURE_DELIMITERS = ("::", "|", "[", "]")


class GLiNER2Adapter(BaseAdapter):
    """Adapter for GLiNER2 zero-shot extraction and classification models.

    GLiNER2 uses the separate ``gliner2`` pip package (NOT ``gliner``). It
    performs named entity recognition, relation extraction, flat structured
    extraction, and schema-conditioned classification.

    Key API differences from GLiNER v1:
    - ``GLiNER2.from_pretrained(name, map_location=device, quantize=True)``
    - ``extract_entities()`` returns nested-by-label dict, not a flat list
    - Batch methods cover entities, relations, structured data, and classification
    - Classification uses ``classify_text()`` / ``batch_classify_text()``

    Reference models:
    - fastino/gliner2-base-v1
    - fastino/gliner2-large-v1

    See plan .kilo/plans/1776678677227-glowing-moon.md for design details.
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
        classification_task: str | None = None,
        default_labels: list[str] | None = None,
        multi_label: bool = False,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Minimum confidence score for extraction/classification (0-1).
            classification_task: Optional schema task name. When set, ``extract``
                returns classifications instead of entities. The task may be
                overridden per request through runtime options.
            default_labels: Optional labels used when a request omits ``labels``.
                Request-provided labels take precedence.
            multi_label: Whether the configured classification task may return
                multiple labels.
            max_seq_length: Maximum document and schema input length.
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._classification_task = classification_task
        self._default_labels = self._validate_labels(default_labels) if default_labels is not None else None
        self._multi_label = multi_label
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: Any = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        from gliner2 import GLiNER2  # ty:ignore[unresolved-import]

        self._device = device

        # GLiNER2 does not forward arbitrary kwargs to Hugging Face downloads.
        # Resolve a pinned snapshot ourselves so every model file comes from the
        # configured immutable revision. Pre-staged weights remain local paths.
        model_path = self._model_name_or_path
        if self._revision is not None and not Path(model_path).is_dir():
            model_path = snapshot_download(repo_id=model_path, revision=self._revision)

        use_quantize = device != "cpu" and self._compute_precision == "float16"
        self._model = GLiNER2.from_pretrained(
            model_path,
            map_location=device,
            quantize=use_quantize,
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
        """Extract entities, relations, classifications, or flat structured data."""
        self._check_loaded()
        texts = [self._extract_text(item) for item in items]
        opts = options or {}
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold))
        classification_task = opts.get("classification_task", self._classification_task)
        multi_label = opts.get("multi_label", self._multi_label)
        if not isinstance(multi_label, bool):
            raise ValueError("GLiNER2 multi_label must be boolean")
        input_token_counts = self._doc_input_token_counts(texts)

        if output_schema is not None:
            if labels:
                raise ValueError("GLiNER2 structured extraction does not accept labels")
            if classification_task is not None:
                raise ValueError("GLiNER2 structured extraction does not accept classification_task")
            structures = self._json_schema_to_structures(output_schema)
            with torch.inference_mode():
                raw_results = self._model.batch_extract_json(
                    texts,
                    structures,
                    batch_size=len(texts),
                    threshold=effective_threshold,
                    include_confidence=False,
                    include_spans=False,
                    max_len=self._max_seq_length,
                )
            return ExtractOutput(
                entities=[[] for _ in texts],
                data=[self._flatten_structured_result(result, output_schema=output_schema) for result in raw_results],
                input_token_counts=input_token_counts,
            )

        effective_labels = labels if labels is not None else self._default_labels
        normalized_labels = self._validate_labels(effective_labels)
        relation_entities = [self._extract_relation_entities(item) for item in items]
        if any(entities is not None for entities in relation_entities):
            if not all(entities for entities in relation_entities):
                raise ValueError("GLiNER2 relation extraction requires non-empty entities in every item metadata")
            normalized_entities = [
                self._normalize_input_entities(item, entities or []) for item, entities in zip(items, relation_entities)
            ]
            with torch.inference_mode():
                raw_results = self._model.batch_extract_relations(
                    texts,
                    normalized_labels,
                    batch_size=len(texts),
                    threshold=effective_threshold,
                    include_confidence=True,
                    include_spans=True,
                    max_len=self._max_seq_length,
                )
            return ExtractOutput(
                entities=normalized_entities,
                relations=[
                    self._flatten_relations(result, entities=entities)
                    for result, entities in zip(raw_results, normalized_entities)
                ],
                input_token_counts=input_token_counts,
            )

        if classification_task is not None:
            if not isinstance(classification_task, str) or not classification_task.strip():
                raise ValueError("GLiNER2 classification_task must be a non-empty string")
            return self._classify(
                texts,
                normalized_labels,
                task=classification_task,
                multi_label=multi_label,
                threshold=effective_threshold,
                input_token_counts=input_token_counts,
            )

        with torch.inference_mode():
            if len(texts) == 1:
                raw_results = [
                    self._model.extract_entities(
                        texts[0],
                        normalized_labels,
                        threshold=effective_threshold,
                        include_confidence=True,
                        include_spans=True,
                        max_len=self._max_seq_length,
                    )
                ]
            else:
                raw_results = self._model.batch_extract_entities(
                    texts,
                    normalized_labels,
                    threshold=effective_threshold,
                    include_confidence=True,
                    include_spans=True,
                    max_len=self._max_seq_length,
                )

        all_entities = [self._flatten_entities(result, text=text) for text, result in zip(texts, raw_results)]
        return ExtractOutput(entities=all_entities, input_token_counts=input_token_counts)

    def _classify(
        self,
        texts: list[str],
        labels: list[str],
        *,
        task: str,
        multi_label: bool,
        threshold: float,
        input_token_counts: list[int] | None,
    ) -> ExtractOutput:
        """Run one GLiNER2 classification schema and normalize its results."""
        tasks = {
            task: {
                "labels": labels,
                "multi_label": multi_label,
                "cls_threshold": threshold,
            }
        }

        with torch.inference_mode():
            if len(texts) == 1:
                raw_results = [
                    self._model.classify_text(
                        texts[0],
                        tasks,
                        threshold=threshold,
                        include_confidence=True,
                        max_len=self._max_seq_length,
                    )
                ]
            else:
                raw_results = self._model.batch_classify_text(
                    texts,
                    tasks,
                    threshold=threshold,
                    include_confidence=True,
                    max_len=self._max_seq_length,
                )

        all_classifications = [
            self._flatten_classifications(result, task=task, threshold=threshold) for result in raw_results
        ]
        return ExtractOutput(
            entities=[[] for _ in texts],
            classifications=all_classifications,
            input_token_counts=input_token_counts,
        )

    @classmethod
    def _flatten_classifications(
        cls,
        result: dict[str, Any],
        *,
        task: str,
        threshold: float,
    ) -> list[Classification]:
        """Convert confidence-bearing GLiNER2 task output to SIE classifications."""
        task_result = result.get(task)
        if task_result is None:
            return []
        candidates = [task_result] if isinstance(task_result, dict) else task_result
        if not isinstance(candidates, list):
            raise ValueError("GLiNER2 returned malformed classifications")

        classifications: list[Classification] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("GLiNER2 returned malformed classification")
            label = candidate.get("label")
            if not isinstance(label, str) or not label.strip() or "confidence" not in candidate:
                raise ValueError("GLiNER2 returned malformed classification")
            score = cls._validate_score(candidate["confidence"], "classification")
            if score >= threshold:
                classifications.append(Classification(label=label.strip(), score=score))
        classifications.sort(key=lambda classification: classification["score"], reverse=True)
        return classifications

    @classmethod
    def _flatten_entities(cls, result: dict[str, Any], *, text: str) -> list[Entity]:
        """Normalize confidence-bearing spans and verify character offsets."""
        entity_list: list[Entity] = []
        entities_dict = result.get("entities", {})
        if not isinstance(entities_dict, dict):
            raise ValueError("GLiNER2 returned malformed entities")
        for label_name, spans in entities_dict.items():
            if not isinstance(label_name, str) or not label_name.strip() or not isinstance(spans, list):
                raise ValueError("GLiNER2 returned malformed entities")
            for span in spans:
                if not isinstance(span, dict):
                    raise ValueError("GLiNER2 returned malformed entity span")
                start, end, span_text = cls._normalize_entity_span(
                    text,
                    span.get("start"),
                    span.get("end"),
                    span.get("text"),
                )
                score = cls._validate_score(span.get("confidence", 0.0), "entity")
                entity_list.append(Entity(text=span_text, label=label_name.strip(), score=score, start=start, end=end))
        entity_list.sort(key=lambda entity: entity.get("start") or 0)
        return entity_list

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER2 adapter"))
        return item.text

    @staticmethod
    def _validate_threshold(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("GLiNER2 threshold must be a finite number between 0 and 1")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("GLiNER2 threshold must be a finite number between 0 and 1")
        return threshold

    @staticmethod
    def _validate_score(value: object, output_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"GLiNER2 returned invalid {output_name} confidence")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"GLiNER2 returned invalid {output_name} confidence")
        return score

    @staticmethod
    def _normalize_entity_span(
        text: str,
        start_value: object,
        end_value: object,
        span_text_value: object,
    ) -> tuple[int, int, str]:
        """Return exact source offsets, clipping only verified boundary punctuation."""

        def invalid_offsets() -> ValueError:
            def safe_offset(value: object) -> str:
                if type(value) in {bool, float, int, type(None)}:
                    return repr(value)
                return "<redacted>"

            span_text_length = len(span_text_value) if isinstance(span_text_value, str) else None
            return ValueError(
                "GLiNER2 returned invalid character offsets "
                f"(start={safe_offset(start_value)} [{type(start_value).__name__}], "
                f"end={safe_offset(end_value)} [{type(end_value).__name__}], "
                f"text_length={len(text)}, span_text_length={span_text_length}, "
                f"span_text_type={type(span_text_value).__name__})"
            )

        if (
            not isinstance(start_value, int)
            or isinstance(start_value, bool)
            or not isinstance(end_value, int)
            or isinstance(end_value, bool)
            or not isinstance(span_text_value, str)
        ):
            raise invalid_offsets()

        start = start_value
        end = end_value
        span_text = span_text_value
        if 0 <= start < end <= len(text) and text[start:end] == span_text:
            return start, end, span_text

        if start < 0 or start >= len(text) or end <= start or len(span_text) != end - start or end <= len(text):
            raise invalid_offsets()
        source_prefix = text[start:]
        overflow = span_text[len(source_prefix) :]
        if (
            not overflow
            or span_text[: len(source_prefix)] != source_prefix
            or any(not char.isspace() and not unicodedata.category(char).startswith("P") for char in overflow)
        ):
            raise invalid_offsets()
        return start, len(text), source_prefix

    @staticmethod
    def _validate_labels(labels: list[str] | None) -> list[str]:
        if labels is None:
            raise ValueError(_ERR_REQUIRES_LABELS)
        if not isinstance(labels, list):
            raise ValueError("GLiNER2 labels must be a list")
        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("GLiNER2 labels must be non-empty strings")
        normalized = [label.strip() for label in labels]
        if len(set(normalized)) != len(normalized):
            raise ValueError("GLiNER2 labels must be unique")
        return normalized

    @staticmethod
    def _extract_relation_entities(item: Item) -> list[dict[str, Any]] | None:
        if item.metadata is None or "entities" not in item.metadata:
            return None
        entities = item.metadata["entities"]
        if not isinstance(entities, list):
            raise ValueError("GLiNER2 item metadata.entities must be a list")
        return entities

    @classmethod
    def _normalize_input_entities(cls, item: Item, entities: list[dict[str, Any]]) -> list[Entity]:
        text = item.text or ""
        normalized: list[Entity] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("GLiNER2 relation entities must be objects")
            start = entity.get("start")
            end = entity.get("end")
            entity_text = entity.get("text")
            label = entity.get("label", "ENTITY")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not isinstance(entity_text, str)
                or not isinstance(label, str)
                or not label.strip()
                or start < 0
                or end <= start
                or end > len(text)
                or text[start:end] != entity_text
            ):
                raise ValueError("GLiNER2 relation entities require valid character offsets")
            score = cls._validate_score(entity.get("score", 1.0), "relation entity")
            normalized.append(Entity(text=entity_text, label=label.strip(), score=score, start=start, end=end))
        return normalized

    @classmethod
    def _flatten_relations(
        cls,
        result: dict[str, Any],
        *,
        entities: list[Entity],
    ) -> list[Relation]:
        by_type = result.get("relation_extraction", {})
        if not isinstance(by_type, dict):
            raise ValueError("GLiNER2 returned malformed relations")
        relations: list[Relation] = []
        entity_texts = {entity["text"] for entity in entities}
        for relation_type, candidates in by_type.items():
            if not isinstance(relation_type, str) or not relation_type.strip() or not isinstance(candidates, list):
                raise ValueError("GLiNER2 returned malformed relations")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("GLiNER2 returned malformed relation")
                head = candidate.get("head")
                tail = candidate.get("tail")
                if not isinstance(head, dict) or not isinstance(tail, dict):
                    raise ValueError("GLiNER2 returned malformed relation endpoints")
                head_text = head.get("text")
                tail_text = tail.get("text")
                if not isinstance(head_text, str) or not isinstance(tail_text, str):
                    raise ValueError("GLiNER2 returned malformed relation endpoints")
                # Upstream discovers relation endpoints from the text; it does
                # not accept the caller-supplied entity set as a constraint.
                # Treat out-of-set candidates as negative predictions so they
                # never escape the supplied-entity boundary or fail the whole
                # batch when another candidate is valid.
                if head_text not in entity_texts or tail_text not in entity_texts:
                    continue
                head_score = cls._validate_score(head.get("confidence", 0.0), "relation head")
                tail_score = cls._validate_score(tail.get("confidence", 0.0), "relation tail")
                relations.append(
                    Relation(
                        head=head_text,
                        tail=tail_text,
                        relation=relation_type.strip(),
                        score=min(head_score, tail_score),
                    )
                )
        relations.sort(
            key=lambda relation: (
                -relation["score"],
                relation["relation"],
                relation["head"],
                relation["tail"],
            )
        )
        return relations

    @staticmethod
    def _json_schema_to_structures(output_schema: dict[str, Any]) -> dict[str, list[str]]:
        if output_schema.get("type") != "object":
            raise ValueError("GLiNER2 output_schema root type must be object")
        properties = output_schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise ValueError("GLiNER2 output_schema requires non-empty properties")
        allowed_root = {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
        }
        unsupported_root = set(output_schema) - allowed_root
        if unsupported_root:
            raise ValueError(f"GLiNER2 output_schema has unsupported root keywords: {sorted(unsupported_root)}")
        required = output_schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) or name not in properties for name in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError("GLiNER2 output_schema required must contain unique property names")
        additional_properties = output_schema.get("additionalProperties", True)
        if not isinstance(additional_properties, bool):
            raise ValueError("GLiNER2 output_schema additionalProperties must be boolean")

        fields: list[str] = []
        for name, definition in properties.items():
            if not isinstance(name, str) or not name or any(delimiter in name for delimiter in _STRUCTURE_DELIMITERS):
                raise ValueError("GLiNER2 output_schema property names contain unsupported delimiters")
            if not isinstance(definition, dict):
                raise ValueError(f"GLiNER2 output_schema property {name!r} must be an object")
            unsupported = set(definition) - {"type", "description", "enum", "items", "title"}
            if unsupported:
                raise ValueError(
                    f"GLiNER2 output_schema property {name!r} has unsupported keywords: {sorted(unsupported)}"
                )
            description = definition.get("description") or definition.get("title")
            if description is not None and not isinstance(description, str):
                raise ValueError(f"GLiNER2 output_schema property {name!r} description must be a string")
            description = description.replace("::", ":") if description else None

            field_type = definition.get("type")
            enum = definition.get("enum")
            if enum is not None:
                if field_type != "string" or not isinstance(enum, list) or not enum:
                    raise ValueError(f"GLiNER2 output_schema property {name!r} has invalid enum")
                if any(
                    not isinstance(choice, str)
                    or not choice
                    or any(delimiter in choice for delimiter in _STRUCTURE_DELIMITERS)
                    for choice in enum
                ):
                    raise ValueError(
                        f"GLiNER2 output_schema property {name!r} enum values contain unsupported delimiters"
                    )
                choices = "|".join(enum)
                spec = f"{name}::[{choices}]::str"
            elif field_type == "string":
                spec = f"{name}::str"
            elif field_type == "array" and definition.get("items") == {"type": "string"}:
                spec = f"{name}::list"
            else:
                raise ValueError(
                    f"GLiNER2 output_schema property {name!r} supports only string, string enum, or array of strings"
                )
            if description:
                spec = f"{spec}::{description}"
            fields.append(spec)
        return {_STRUCTURE_NAME: fields}

    @classmethod
    def _flatten_structured_result(
        cls,
        result: dict[str, Any],
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        values = result.get(_STRUCTURE_NAME, [])
        if values in (None, []):
            data: dict[str, Any] = {}
        elif not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("GLiNER2 structured extraction did not return exactly one root object")
        else:
            data = dict(values[0])
        cls._validate_structured_result(data, output_schema)
        return data

    @staticmethod
    def _validate_structured_result(data: dict[str, Any], output_schema: dict[str, Any]) -> None:
        properties = output_schema["properties"]
        required = output_schema.get("required", [])
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"GLiNER2 structured extraction omitted required properties: {missing}")

        if output_schema.get("additionalProperties", True) is False:
            unexpected = sorted(set(data) - set(properties))
            if unexpected:
                raise ValueError(f"GLiNER2 structured extraction returned unexpected properties: {unexpected}")

        for name, value in data.items():
            definition = properties.get(name)
            if definition is None:
                continue
            expected_type = definition.get("type")
            if expected_type == "string":
                if not isinstance(value, str):
                    raise ValueError(f"GLiNER2 structured extraction property {name!r} must be a string")
                choices = definition.get("enum")
                if choices is not None and value not in choices:
                    raise ValueError(f"GLiNER2 structured extraction property {name!r} is outside its enum")
            elif not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"GLiNER2 structured extraction property {name!r} must be an array of strings")

    def _doc_input_token_counts(self, texts: list[str]) -> list[int] | None:
        processor = getattr(self._model, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=self._max_seq_length is not None,
                max_length=self._max_seq_length,
            )
            counts = [len(ids) for ids in encoded["input_ids"]]
        except Exception:  # noqa: BLE001 -- metering must not fail extraction
            return None
        return counts if len(counts) == len(texts) else None
