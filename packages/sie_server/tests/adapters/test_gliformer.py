"""Contract tests for the GLiFormer adapter (no model weights)."""

from __future__ import annotations

import subprocess
import sys
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml
from pydantic import BaseModel
from sie_server.adapters.gliformer import adapter as adapter_module
from sie_server.adapters.gliformer import span_decoding
from sie_server.adapters.gliformer.adapter import GLiFormerAdapter
from sie_server.adapters.gliformer.output_schema import compile_output_schema, shape_structured_output
from sie_server.core.inference_output import ExtractItemError
from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import InvalidInputError, Item

_SERVER_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _SERVER_ROOT / "models"
_MODEL_IDS = {"knowledgator/gliformer-base-v1": 768, "knowledgator/gliformer-large-v1": 1024}


class _FakeTokenizer:
    """One subword per word; two special tokens per sequence."""

    def __init__(self) -> None:
        self.calls: list[list[list[str]]] = []

    def __call__(self, sequences: list[list[str]], **_: Any) -> dict[str, Any]:
        self.calls.append(sequences)
        return {"input_ids": [[1] * len(words) for words in sequences]}

    @staticmethod
    def num_special_tokens_to_add(pair: bool = False) -> int:
        return 2


def _fake_model(results: dict[str, Any], *, prompt_words: int = 3, max_len: int = 2048) -> MagicMock:
    """A stand-in for a loaded GLiFormer whose task prompt is ``prompt_words`` words."""
    model = MagicMock()
    model.inference.return_value = results
    model.config = SimpleNamespace(max_len=max_len)
    model.prepare_inputs.side_effect = lambda texts: ([text.split() for text in texts], [], [])
    model._build_inference_input.side_effect = lambda words, **_: [{"tokenized_text": w} for w in words]
    processor = model.data_processor
    processor.collate_raw_batch.side_effect = lambda items: {
        "tokens": [item["tokenized_text"] for item in items],
        "classes_mapping": None,
    }
    processor.prepare_inputs.side_effect = lambda tokens, _mapping: (
        [["P"] * prompt_words + list(words) for words in tokens],
        [prompt_words] * len(tokens),
    )
    processor.transformer_tokenizer = _FakeTokenizer()
    return model


def _adapter(results: dict[str, Any], *, max_len: int = 2048, **kwargs: Any) -> tuple[GLiFormerAdapter, MagicMock]:
    adapter = GLiFormerAdapter("test-model", **kwargs)
    adapter._model = _fake_model(results, max_len=max_len)
    adapter._tokenizer = adapter._model.data_processor.transformer_tokenizer
    adapter._normalize_structures = lambda structures: structures
    adapter._build_formatter = lambda structures: None
    return adapter, adapter._model


def _ner(text: str, span: str, label: str, score: float = 0.9) -> dict[str, Any]:
    start = text.index(span)
    return {"start": start, "end": start + len(span), "text": span, "label": label, "score": score}


def _relation(head: str, relation: str, tail: str, score: float) -> dict[str, Any]:
    return {"head": {"text": head}, "tail": {"text": tail}, "relation": relation, "score": score}


def _inference_kwargs(model: MagicMock) -> dict[str, Any]:
    return model.inference.call_args.kwargs


# -- Request routing -------------------------------------------------------------


def test_labels_run_entity_recognition_with_document_token_counts() -> None:
    text = "Alice works at Acme"
    adapter, model = _adapter({"ner": [[_ner(text, "Acme", "organization", 0.8), _ner(text, "Alice", "person")]]})

    output = adapter.extract([Item(text=text)], labels=["person", " organization "])

    assert output.entities == [
        [
            {"text": "Alice", "label": "person", "score": 0.9, "start": 0, "end": 5},
            {"text": "Acme", "label": "organization", "score": 0.8, "start": 15, "end": 19},
        ]
    ]
    assert output.classifications is None
    assert output.relations is None
    assert output.data is None
    # [CLS] + four document words + [SEP]; the three prompt words are not billed.
    assert output.input_token_counts == [6]
    model.inference.assert_called_once_with(
        [text],
        entities=["person", "organization"],
        classes=None,
        joint_relations=None,
        structures=None,
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
        batch_size=1,
    )


def test_classification_task_matches_gliner2_contract() -> None:
    adapter, model = _adapter(
        {
            "classification": [
                [[{"class_name": "review", "score": 0.4}, {"class_name": "block", "score": 0.7}]],
                [[]],
            ]
        }
    )

    output = adapter.extract(
        [Item(text="first"), Item(text="second")],
        labels=["allow", "review", "block"],
        options={"classification_task": "moderation", "multi_label": True, "threshold": 0.3},
    )

    assert output.entities == [[], []]
    assert output.classifications == [
        [{"label": "block", "score": 0.7}, {"label": "review", "score": 0.4}],
        [],
    ]
    kwargs = _inference_kwargs(model)
    assert kwargs["classes"] == {"moderation": ["allow", "review", "block"]}
    assert kwargs["entities"] is None
    assert (kwargs["threshold"], kwargs["multi_label"]) == (0.3, True)


def test_relation_labels_run_joint_extraction_over_label_entity_types() -> None:
    text = "Alice works at Acme in London"
    adapter, model = _adapter(
        {
            "ner": [[_ner(text, "Alice", "person"), _ner(text, "Acme", "organization")]],
            "joint_relex": [
                [
                    _relation("Acme", "located in", "London", 0.6),
                    _relation("Alice", "works at", "Acme", 0.9),
                ]
            ],
        }
    )

    output = adapter.extract(
        [Item(text=text)], labels=["person", "organization"], options={"relation_labels": ["works at"]}
    )

    assert [entity["text"] for entity in output.entities[0]] == ["Alice", "Acme"]
    assert output.relations == [
        [
            {"head": "Alice", "tail": "Acme", "relation": "works at", "score": 0.9},
            {"head": "Acme", "tail": "London", "relation": "located in", "score": 0.6},
        ]
    ]
    kwargs = _inference_kwargs(model)
    assert kwargs["entities"] is None
    assert kwargs["joint_relations"] == {None: {"entities": ["person", "organization"], "relations": ["works at"]}}


@pytest.mark.parametrize("supplied", [False, True])
def test_relation_threshold_filters_relations_only(supplied: bool) -> None:
    text = "Alice works at Acme in London"
    alice, acme = _ner(text, "Alice", "person", 0.6), _ner(text, "Acme", "organization", 0.6)
    adapter, model = _adapter(
        {
            "ner": [[alice, acme]],
            "joint_relex": [
                [
                    _relation("Alice", "works at", "Acme", 0.9),
                    _relation("Alice", "located in", "Acme", 0.7),
                    _relation("Acme", "works at", "Alice", 0.6),
                ]
            ],
        }
    )
    relation_types = ["works at", "located in"]
    options = {"relation_threshold": 0.7}

    if supplied:
        item = Item(text=text, metadata={"entities": [alice, acme]})
        output = adapter.extract([item], labels=relation_types, options=options)
    else:
        output = adapter.extract(
            [Item(text=text)],
            labels=["person", "organization"],
            options={**options, "relation_labels": relation_types},
        )

    # Kept only above relation_threshold, as the decoder keeps scores above threshold.
    assert output.relations == [[{"head": "Alice", "tail": "Acme", "relation": "works at", "score": 0.9}]]
    assert [entity["text"] for entity in output.entities[0]] == ["Alice", "Acme"]
    assert _inference_kwargs(model)["threshold"] == 0.5


@pytest.mark.parametrize("relation_labels", [None, []])
def test_no_relation_labels_run_entity_recognition_only(relation_labels: list[str] | None) -> None:
    text = "Alice works at Acme"
    adapter, model = _adapter({"ner": [[_ner(text, "Alice", "person")]]})

    output = adapter.extract([Item(text=text)], labels=["person"], options={"relation_labels": relation_labels})

    assert output.relations is None
    assert [entity["text"] for entity in output.entities[0]] == ["Alice"]
    assert _inference_kwargs(model)["entities"] == ["person"]


def test_relation_threshold_cannot_lower_the_decoding_threshold() -> None:
    adapter, model = _adapter({"ner": [[]], "joint_relex": [[]]})
    options = {"relation_labels": ["knows"], "threshold": 0.5}

    with pytest.raises(InvalidInputError, match="relation_threshold must be at least threshold"):
        adapter.extract([Item(text="Ada met Bo")], labels=["person"], options={**options, "relation_threshold": 0.3})
    model.inference.assert_not_called()

    adapter.extract([Item(text="Ada met Bo")], labels=["person"], options={**options, "relation_threshold": 0.5})
    model.inference.assert_called_once()


def test_metadata_entities_use_the_shared_relation_contract() -> None:
    first = "Ada founded Acme"
    second = "Grace joined Beta in Paris"
    adapter, model = _adapter({})
    model.inference.side_effect = [
        {
            "ner": [[]],
            "joint_relex": [[_relation("Ada", "founded", "Acme", 0.8), _relation("Ada", "founded", "X", 0.9)]],
        },
        {"ner": [[]], "joint_relex": [[_relation("Grace", "joined", "Beta", 0.7)]]},
    ]
    items = [
        Item(
            text=first,
            metadata={
                "entities": [
                    {"text": "Ada", "label": "person", "start": 0, "end": 3},
                    {"text": "Acme", "label": "company", "start": 12, "end": 16},
                ]
            },
        ),
        Item(
            text=second,
            metadata={
                "entities": [
                    {"text": "Grace", "label": "person", "start": 0, "end": 5},
                    {"text": "Paris", "label": "city", "start": 21, "end": 26},
                ]
            },
        ),
    ]

    output = adapter.extract(items, labels=["founded", "joined"])

    assert output.entities[0][1] == {"text": "Acme", "label": "company", "score": 1.0, "start": 12, "end": 16}
    # Endpoints outside the supplied entities are not answers.
    assert output.relations == [[{"head": "Ada", "tail": "Acme", "relation": "founded", "score": 0.8}], []]
    calls = [call.kwargs["joint_relations"] for call in model.inference.call_args_list]
    # Each item's entity types, in a canonical order.
    assert calls == [
        {None: {"entities": ["company", "person"], "relations": ["founded", "joined"]}},
        {None: {"entities": ["city", "person"], "relations": ["founded", "joined"]}},
    ]


def test_output_schema_combines_structuring_and_enum_classification_in_one_pass() -> None:
    text = "Refund for Maria Lopez on order 4471 was denied."
    schema = {
        "type": "object",
        "properties": {
            "customer": {"type": "string"},
            "decision": {"type": "string", "enum": ["approved", "denied"]},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "$amount": {"type": "string"},
        },
        "required": ["decision"],
    }
    adapter, model = _adapter(
        {
            "ner": [[_ner(text, "Maria Lopez", "person")]],
            "classification": [[[{"class_name": "denied", "score": 0.95}]]],
            "structuring": [{"customer": "Maria Lopez", "reasons": [], "$amount": None}],
        }
    )

    output = adapter.extract([Item(text=text)], labels=["person"], output_schema=schema)

    assert output.data == [{"customer": "Maria Lopez", "decision": "denied"}]
    assert output.errors is None
    assert [entity["text"] for entity in output.entities[0]] == ["Maria Lopez"]
    model.inference.assert_called_once()
    kwargs = _inference_kwargs(model)
    assert kwargs["classes"] == {"decision": ["approved", "denied"]}
    assert kwargs["structures"] == {"$root": {"customer": "str", "reasons": ["str"], "$$amount": "str"}}
    assert kwargs["entities"] == ["person"]


def test_structuring_output_goes_through_the_upstream_formatter() -> None:
    schema = {"type": "object", "properties": {"customer": {"type": "string"}}}
    adapter, _ = _adapter({"structuring": [{"customer": ["Maria Lopez", "Tim Cook"]}]})
    formatter = MagicMock()
    formatter.format_batch.return_value = [{"customer": "Maria Lopez"}]
    adapter._build_formatter = MagicMock(return_value=formatter)

    output = adapter.extract([Item(text="Maria Lopez and Tim Cook")], output_schema=schema)

    assert output.data == [{"customer": "Maria Lopez"}]
    adapter._build_formatter.assert_called_once_with({"$root": {"customer": "str"}})
    formatter.format_batch.assert_called_once_with([{"customer": ["Maria Lopez", "Tim Cook"]}])


def test_missing_required_property_is_a_per_item_error() -> None:
    schema = {
        "type": "object",
        "properties": {"customer": {"type": "string"}, "decision": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["decision"],
    }
    adapter, _ = _adapter(
        {
            "classification": [[[]], [[{"class_name": "yes", "score": 0.9}]]],
            "structuring": [{"customer": "Ann"}, {}],
        }
    )

    output = adapter.extract([Item(text="Ann asked"), Item(text="Approved")], output_schema=schema)

    assert output.data == [{}, {"decision": "yes"}]
    assert output.errors == [
        ExtractItemError(
            code="INFERENCE_ERROR",
            message="GLiFormer did not extract required output_schema properties: ['decision']",
        ),
        None,
    ]
    # The errored item bills nothing; "Approved" is one word plus two specials.
    assert output.input_token_counts == [0, 3]


def test_missing_required_property_blanks_every_task_output_and_bills_nothing() -> None:
    text = "Tim Cook runs Apple"
    schema = {
        "type": "object",
        "properties": {"ceo": {"type": "string"}, "tone": {"type": "string", "enum": ["calm", "tense"]}},
        "required": ["ceo"],
    }
    adapter, _ = _adapter(
        {
            "ner": [[_ner(text, "Tim Cook", "person")], [_ner(text, "Apple", "organization")]],
            "joint_relex": [
                [_relation("Tim Cook", "runs", "Apple", 0.9)],
                [_relation("Tim Cook", "runs", "Apple", 0.8)],
            ],
            "classification": [[[{"class_name": "calm", "score": 0.9}]], [[{"class_name": "tense", "score": 0.7}]]],
            "structuring": [{}, {"ceo": "Tim Cook"}],
        }
    )

    output = adapter.extract(
        [Item(text=text), Item(text=text)],
        labels=["person", "organization"],
        output_schema=schema,
        options={"relation_labels": ["runs"]},
    )

    assert output.errors is not None
    assert [error is not None for error in output.errors] == [True, False]
    assert output.entities[0] == []
    assert output.relations is not None
    assert output.relations[0] == []
    assert output.data == [{}, {"ceo": "Tim Cook", "tone": "tense"}]
    assert output.entities[1] != []
    assert output.relations[1] != []
    assert output.input_token_counts == [0, 6]


def test_classification_task_and_enum_properties_share_one_forward_pass() -> None:
    schema = {"type": "object", "properties": {"urgency": {"type": "string", "enum": ["high", "low"]}}}
    adapter, model = _adapter(
        {
            "classification": [
                [[{"class_name": "negative", "score": 0.8}], [{"class_name": "high", "score": 0.7}]],
            ]
        }
    )

    output = adapter.extract(
        [Item(text="The server is down again!")],
        labels=["positive", "negative"],
        output_schema=schema,
        options={"classification_task": "sentiment"},
    )

    assert output.classifications == [[{"label": "negative", "score": 0.8}]]
    assert output.data == [{"urgency": "high"}]
    assert _inference_kwargs(model)["classes"] == {"sentiment": ["positive", "negative"], "urgency": ["high", "low"]}
    assert _inference_kwargs(model)["structures"] is None


def test_label_groups_return_gliclass_style_classifications_only() -> None:
    adapter, model = _adapter(
        {
            "ner": [[]],
            "classification": [
                [
                    [{"class_name": "billing", "score": 0.9}],
                    [{"class_name": "high", "score": 0.6}, {"class_name": "low", "score": 0.55}],
                    [],
                ]
            ],
        }
    )

    output = adapter.extract(
        [Item(text="I was charged twice")],
        labels=["person"],
        options={
            "label_groups": {
                "topic": ["billing", "bug report"],
                "urgency": ["low", "high"],
                "needs human": ["yes", "no"],
            },
            "classification_type": "multi-label",
        },
    )

    # GLiFormer scores only labels that pass the threshold, so there is no
    # per-group distribution to report in ``data``.
    assert output.data is None
    assert output.classifications == [
        [
            {"label": "topic.billing", "score": 0.9},
            {"label": "urgency.high", "score": 0.6},
            {"label": "urgency.low", "score": 0.55},
        ]
    ]
    kwargs = _inference_kwargs(model)
    assert kwargs["classes"] == {
        "topic": ["billing", "bug report"],
        "urgency": ["low", "high"],
        "needs human": ["yes", "no"],
    }
    assert kwargs["entities"] == ["person"]
    assert kwargs["multi_label"] is True


def test_classification_output_outside_the_requested_labels_is_a_per_item_error() -> None:
    adapter, _ = _adapter({"classification": [[[{"class_name": "other", "score": 0.9}]]]})
    output = adapter.extract([Item(text="hello")], labels=["a", "b"], options={"classification_task": "t"})
    assert output.classifications == [[]]
    assert output.errors == [ExtractItemError(code="INFERENCE_ERROR", message=adapter_module._ERR_ITEM_OUTPUT)]
    assert output.input_token_counts == [0]


def test_options_passed_through_as_sent_keep_their_order() -> None:
    adapter, model = _adapter({"classification": [[[], []]]})
    options = {"label_groups": {"urgency": ["low", "high"], "topic": ["jobs"]}}
    adapter.extract([Item(text="hello")], options=options)
    assert list(_inference_kwargs(model)["classes"]) == ["urgency", "topic"]


def test_instruction_is_accepted_and_ignored() -> None:
    adapter, model = _adapter({"ner": [[]]})
    adapter.extract([Item(text="hello world")], labels=["person"], instruction="ignored")
    assert "instruction" not in _inference_kwargs(model)


# -- Request validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "requires labels, label_groups, or output_schema"),
        ({"options": {"label_groups": {}}}, "label_groups must be a non-empty object"),
        ({"options": {"label_groups": {"topic": []}}}, "must be a non-empty list"),
        ({"options": {"label_groups": {"topic": ["a", "a"]}}}, "must be unique"),
        (
            {"labels": ["a"], "options": {"classification_task": "t", "label_groups": {"g": ["x"]}}},
            "label_groups cannot be combined with classification_task",
        ),
        (
            {
                "options": {"label_groups": {"g": ["x"]}},
                "output_schema": {"type": "object", "properties": {"n": {"type": "string"}}},
            },
            "label_groups cannot be combined with output_schema",
        ),
        ({"labels": ["a"], "options": {"classification_type": "softmax"}}, "classification_type must be"),
        (
            {"labels": ["a"], "options": {"classification_type": "multi-label", "multi_label": False}},
            "contradicts classification_type",
        ),
        ({"options": {"classification_task": "topic"}}, "classification_task requires labels"),
        ({"options": {"relation_labels": ["works at"]}}, "relation_labels require labels"),
        (
            {"labels": ["a"], "options": {"classification_task": "topic", "relation_labels": ["r"]}},
            "cannot be combined",
        ),
        ({"labels": ["a"], "options": {"threshold": 1.5}}, "between 0 and 1"),
        ({"labels": ["a"], "options": {"threshold": -0.1}}, "between 0 and 1"),
        ({"labels": ["a"], "options": {"threshold": True}}, "between 0 and 1"),
        ({"labels": ["a"], "options": {"threshold": 0}}, "at least 0.1"),
        ({"labels": ["a"], "options": {"threshold": 0.0999}}, "at least 0.1"),
        ({"labels": ["a"], "options": {"multi_label": "yes"}}, "multi_label must be boolean"),
        ({"labels": ["a"], "options": {"flat_ner": 1}}, "flat_ner must be boolean"),
        ({"labels": ["a", "a"]}, "labels must be unique"),
        ({"labels": ["a", " "]}, "labels must be non-empty strings"),
        ({"labels": ["a"], "options": {"relation_labels": "works at"}}, "relation_labels must be a non-empty list"),
        ({"labels": ["a"], "options": {"relation_labels": ""}}, "relation_labels must be a non-empty list"),
        ({"labels": ["a"], "options": {"relation_labels": False}}, "relation_labels must be a non-empty list"),
        ({"labels": ["a"], "options": {"relations": ["works at"]}}, "options.relation_labels, not options.relations"),
        ({"labels": ["a"], "options": {"relations": []}}, "options.relation_labels, not options.relations"),
        ({"labels": ["a"], "options": {"relation_threshold": 1.5}}, "relation_threshold must be a number between"),
        ({"labels": ["a"], "options": {"relation_threshold": False}}, "relation_threshold must be a number between"),
        ({"labels": ["a"], "options": {"threshold": 10**1000}}, "threshold must be a number between"),
        ({"labels": ["a"], "options": {"relation_threshold": 10**1000}}, "relation_threshold must be a number between"),
        ({"labels": ["a"], "options": {"classification_task": " "}}, "classification_task must be a non-empty string"),
        (
            {
                "labels": ["x"],
                "options": {"classification_task": "urgency"},
                "output_schema": {"type": "object", "properties": {"urgency": {"enum": ["high", "low"]}}},
            },
            "must differ from output_schema enum",
        ),
    ],
)
def test_invalid_requests_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    adapter, model = _adapter({})
    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text="hello world")], **kwargs)
    model.inference.assert_not_called()


@pytest.mark.parametrize(("text", "match"), [(None, "requires text"), ("   ", "non-blank text")])
def test_items_require_non_blank_text(text: str | None, match: str) -> None:
    adapter, _ = _adapter({})
    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text=text)], labels=["person"])


def test_metadata_entities_must_cover_every_item_and_exclude_other_relation_modes() -> None:
    adapter, _ = _adapter({})
    tagged = Item(text="Ada founded Acme", metadata={"entities": [{"text": "Ada", "start": 0, "end": 3}]})
    with pytest.raises(InvalidInputError, match="every item metadata"):
        adapter.extract([tagged, Item(text="Grace founded Beta")], labels=["founded"])
    with pytest.raises(InvalidInputError, match="cannot be combined"):
        adapter.extract([tagged], labels=["founded"], options={"relation_labels": ["founded"]})
    bad_offsets = Item(text="Ada founded Acme", metadata={"entities": [{"text": "Ada", "start": 1, "end": 4}]})
    with pytest.raises(InvalidInputError, match="valid character offsets"):
        adapter.extract([bad_offsets], labels=["founded"])
    for score in (2.0, float("nan"), 10**1000):
        bad_score = Item(
            text="Ada founded Acme", metadata={"entities": [{"text": "Ada", "start": 0, "end": 3, "score": score}]}
        )
        with pytest.raises(InvalidInputError, match="relation entity score must be"):
            adapter.extract([bad_score], labels=["founded"])


# -- Model output validation and metering -----------------------------------------


@pytest.mark.parametrize(
    "bad_entity",
    [
        {"start": 0, "end": 4, "text": "Bob", "label": "person", "score": 0.9},
        {"start": 0, "end": 5, "text": "Alice", "label": "person", "score": 1.5},
        "not an entity",
    ],
)
def test_malformed_output_for_one_item_fails_only_that_item(bad_entity: Any) -> None:
    texts = ["Alice works here", "Bob works there"]
    adapter, _ = _adapter({"ner": [[bad_entity], [_ner(texts[1], "Bob", "person")]]})

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    assert output.entities[0] == []
    assert [entity["text"] for entity in output.entities[1]] == ["Bob"]
    assert output.errors == [ExtractItemError(code="INFERENCE_ERROR", message=adapter_module._ERR_ITEM_OUTPUT), None]
    # Errored items bill nothing.
    assert output.input_token_counts == [0, 5]


def test_output_not_aligned_with_the_batch_is_an_internal_error() -> None:
    adapter, _ = _adapter({"ner": [[], []]})
    with pytest.raises(RuntimeError, match="malformed ner") as info:
        adapter.extract([Item(text="Alice works here")], labels=["person"])
    # A ValueError would be reported to the caller as a 400.
    assert not isinstance(info.value, ValueError)


def test_non_finite_scores_fail_only_the_documents_that_produce_them() -> None:
    texts = ["Alice works here", "poison pill text", "Bob works there"]
    adapter, model = _adapter({})

    def inference(batch: list[str], **_: Any) -> dict[str, Any]:
        if "poison pill text" in batch:
            raise adapter_module._NonFiniteScoresError("GLiFormer produced non-finite scores")
        return {"ner": [[_ner(text, text.split()[0], "person")] for text in batch]}

    model.inference.side_effect = inference

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    # The shared pass fails, then the chunk is split in halves until the
    # failing document runs alone.
    assert [call.args[0] for call in model.inference.call_args_list] == [
        texts,
        texts[:1],
        texts[1:],
        texts[1:2],
        texts[2:],
    ]
    assert [[entity["text"] for entity in entities] for entities in output.entities] == [["Alice"], [], ["Bob"]]
    assert output.errors is not None
    assert [error is not None for error in output.errors] == [False, True, False]
    assert output.input_token_counts == [5, 0, 5]


def test_structured_output_that_cannot_be_formatted_fails_only_its_item() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    adapter, _ = _adapter({"structuring": [{"name": "Ann"}, {"name": "Bob"}]})
    formatter = MagicMock()
    formatter.format_batch.side_effect = lambda rows: (
        [{"name": "Bob"}] if rows == [{"name": "Bob"}] else (_ for _ in ()).throw(ValueError("bad value"))
    )
    adapter._build_formatter = MagicMock(return_value=formatter)

    output = adapter.extract([Item(text="Ann asked"), Item(text="Bob asked")], output_schema=schema)

    assert output.data == [{}, {"name": "Bob"}]
    assert output.errors is not None
    assert [error is not None for error in output.errors] == [True, False]
    assert output.input_token_counts == [0, 4]


def test_score_outputs_are_decoded_in_float32() -> None:
    logits = torch.tensor([[9.0, 8.5]], dtype=torch.float16)
    # The problem being fixed: in float16 both labels saturate to 1.0, and
    # single-label argmax then picks whichever comes first.
    assert torch.sigmoid(logits.float()).half().tolist() == [[1.0, 1.0]]
    embedding = torch.zeros(2, dtype=torch.float16)
    output = {"cat_logits": logits, "words_embedding": embedding, "span_mask": torch.tensor([True]), "batch_size": 1}

    result = adapter_module._upcast_score_outputs(MagicMock(), (), output)

    assert result["cat_logits"].dtype == torch.float32
    probabilities = torch.sigmoid(result["cat_logits"])[0]
    assert probabilities[0] > probabilities[1]
    assert result["words_embedding"] is embedding
    assert result["batch_size"] == 1


def test_masked_scores_pass_but_nan_and_positive_infinity_fail() -> None:
    masked = {"joint_rel_logits": torch.tensor([[float("-inf"), 1.0]], dtype=torch.float16)}
    assert adapter_module._upcast_score_outputs(MagicMock(), (), masked) is masked
    for bad in (float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="non-finite scores"):
            adapter_module._upcast_score_outputs(MagicMock(), (), {"span_logits": torch.tensor([bad, 0.0])})
    assert adapter_module._upcast_score_outputs(MagicMock(), (), (1, 2)) == (1, 2)


_SPAN_FIELD_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}
_ENUM_ONLY_SCHEMA = {"type": "object", "properties": {"decision": {"type": "string", "enum": ["approve", "deny"]}}}
_MIXED_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "decision": {"type": "string", "enum": ["approve", "deny"]}},
}


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"labels": ["a", "b"], "options": {"classification_task": "t"}},
        {"options": {"label_groups": {"g": ["a", "b"], "h": ["c", "d"]}}},
        {"output_schema": _ENUM_ONLY_SCHEMA},
        {"options": {"label_groups": {"g": ["a", "b"]}, "multi_label": True}},
    ],
)
@pytest.mark.parametrize("threshold", [0, 0.0, 0.05])
def test_classification_only_requests_accept_any_threshold(request_kwargs: dict[str, Any], threshold: float) -> None:
    adapter, model = _adapter({"classification": [[[{"class_name": "a", "score": 0.02}], []]]})
    options = {**request_kwargs.get("options", {}), "threshold": threshold}

    adapter.extract([Item(text="hello")], **{**request_kwargs, "options": options})

    # 0 reaches the package as the smallest positive threshold: it reads 0 as
    # "unset" and would fall back to 0.5.
    assert _inference_kwargs(model)["threshold"] == pytest.approx(max(threshold, 1e-6))


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"labels": ["person"]},
        {"labels": ["person"], "options": {"relation_labels": ["knows"]}},
        {"output_schema": _SPAN_FIELD_SCHEMA},
        {"labels": ["knows"], "items_metadata": True},
    ],
)
def test_span_requests_need_the_threshold_floor(request_kwargs: dict[str, Any]) -> None:
    adapter, model = _adapter({"ner": [[]], "joint_relex": [[]], "structuring": [{}]})
    kwargs = {key: value for key, value in request_kwargs.items() if key != "items_metadata"}
    metadata = {"entities": [{"text": "Ada", "label": "person", "start": 0, "end": 3}]}
    item = Item(text="Ada met Bo", metadata=metadata if "items_metadata" in request_kwargs else None)

    with pytest.raises(InvalidInputError, match=r"at least 0\.1"):
        adapter.extract([item], **{**kwargs, "options": {**kwargs.get("options", {}), "threshold": 0.05}})
    model.inference.assert_not_called()

    adapter.extract([item], **{**kwargs, "options": {**kwargs.get("options", {}), "threshold": 0.1}})
    assert _inference_kwargs(model)["threshold"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"labels": ["person"], "options": {"label_groups": {"g": ["a", "b"]}}},
        {"output_schema": _MIXED_SCHEMA},
        {"labels": ["person"], "output_schema": _ENUM_ONLY_SCHEMA},
    ],
)
def test_mixed_requests_apply_the_floor_to_their_single_threshold(request_kwargs: dict[str, Any]) -> None:
    adapter, model = _adapter({})
    options = {**request_kwargs.get("options", {}), "threshold": 0.0}
    with pytest.raises(InvalidInputError, match=r"at least 0\.1"):
        adapter.extract([Item(text="hello")], **{**request_kwargs, "options": options})
    model.inference.assert_not_called()


def test_configured_threshold_is_range_checked() -> None:
    assert GLiFormerAdapter("test-model", threshold=0.0)._threshold == 0.0
    with pytest.raises(InvalidInputError, match="between 0 and 1"):
        GLiFormerAdapter("test-model", threshold=1.5)


def test_nan_bisection_costs_logarithmic_passes() -> None:
    texts = [f"document {index}" for index in range(32)]
    adapter, model = _adapter({})

    def inference(batch: list[str], **_: Any) -> dict[str, Any]:
        if "document 17" in batch:
            raise adapter_module._NonFiniteScoresError("GLiFormer produced non-finite scores")
        return {"ner": [[] for _ in batch]}

    model.inference.side_effect = inference

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    # 1 failed pass + 2 per halving level (log2 32 = 5): 11 passes, not 33.
    assert model.inference.call_count == 11
    assert output.errors is not None
    assert [index for index, error in enumerate(output.errors) if error is not None] == [17]


def test_widespread_non_finite_scores_cost_at_most_n_plus_three_passes() -> None:
    texts = [f"document {index}" for index in range(32)]
    adapter, model = _adapter({})

    def inference(batch: list[str], **_: Any) -> dict[str, Any]:
        raise adapter_module._NonFiniteScoresError("GLiFormer produced non-finite scores")

    model.inference.side_effect = inference

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    # The full chunk, both halves, then each document alone.
    assert model.inference.call_count == 32 + 3
    assert output.errors is not None
    assert all(error is not None for error in output.errors)
    assert output.input_token_counts == [0] * 32


@pytest.mark.parametrize(
    ("lengths", "budget", "chunks"),
    [
        ([5, 5, 20, 3], 20, [[0, 1], [2], [3]]),
        ([50], 20, [[0]]),
        ([4, 4, 4, 4], 16, [[0, 1, 2, 3]]),
        ([], 16, []),
    ],
)
def test_token_budget_chunks_bound_padded_tokens(lengths: list[int], budget: int, chunks: list[list[int]]) -> None:
    assert adapter_module._token_budget_chunks(lengths, budget) == chunks


def test_inference_is_split_into_padded_token_chunks() -> None:
    texts = ["Alice works here", "Bob works there", "Carol stays home"]
    adapter, model = _adapter({}, inference_batch_tokens=16)
    model.inference.side_effect = lambda batch, **_: {
        "ner": [[_ner(text, text.split()[0], "person")] for text in batch]
    }

    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    # Every fake sequence is 8 tokens: 3 prompt + special + 3 document + special.
    assert [call.args[0] for call in model.inference.call_args_list] == [texts[:2], texts[2:]]
    assert [call.kwargs["batch_size"] for call in model.inference.call_args_list] == [2, 1]
    assert [[entity["text"] for entity in entities] for entities in output.entities] == [["Alice"], ["Bob"], ["Carol"]]
    assert output.input_token_counts == [5, 5, 5]


def test_each_document_decodes_within_its_own_allowance() -> None:
    texts = ["Alice works here", "a much longer text about Bob who works there", "Carol stays home"]
    adapter, model = _adapter({})
    seen = []

    def inference(batch: list[str], **_: Any) -> dict[str, Any]:
        allowances = span_decoding._ALLOWANCES.get()
        seen.append([allowance.remaining for allowance in allowances])
        # A hostile first document spends its whole allowance; the others
        # are unaffected.
        span_decoding.row_allowance(0).spend(10**9)
        return {"ner": [[_ner(text, text.split()[0], "person")] for text in batch]}

    model.inference.side_effect = inference
    output = adapter.extract([Item(text=text) for text in texts], labels=["person"])

    counts = output.input_token_counts
    floor, rate = adapter_module._DECODE_FLOOR, adapter_module._DECODE_UNITS_PER_TOKEN
    assert seen == [[floor + rate * count for count in counts]]
    assert output.errors is None
    assert [[entity["text"] for entity in entities] for entities in output.entities] == [["Alice"], ["a"], ["Carol"]]
    assert span_decoding._ALLOWANCES.get() is None


def test_a_documents_allowance_does_not_depend_on_its_batch() -> None:
    adapter, model = _adapter({}, inference_batch_tokens=16)
    seen: dict[str, int] = {}

    def inference(batch: list[str], **_: Any) -> dict[str, Any]:
        for text, allowance in zip(batch, span_decoding._ALLOWANCES.get(), strict=True):
            seen.setdefault(text, allowance.remaining)
            assert seen[text] == allowance.remaining
        return {"ner": [[] for _ in batch]}

    model.inference.side_effect = inference
    adapter.extract([Item(text="Alice works here")], labels=["person"])
    adapter.extract([Item(text="Bob works there"), Item(text="Alice works here")], labels=["person"])
    adapter.extract([Item(text=t) for t in ["one two three", "Alice works here", "x y z"]], labels=["person"])
    assert len(seen) == 4


def test_documents_are_billed_up_to_the_window_left_by_the_prompt() -> None:
    adapter, _ = _adapter({"ner": [[], []]}, max_len=10)

    output = adapter.extract([Item(text="one two"), Item(text=" ".join(["word"] * 50))], labels=["person"])

    # Window 10 = 3 prompt + 2 special + at most 5 document tokens.
    assert output.input_token_counts == [4, 7]


def test_relation_requests_bound_candidate_pairs_per_pass() -> None:
    adapter, model = _adapter({})
    model.inference.side_effect = lambda batch, **_: {"ner": [[] for _ in batch], "joint_relex": [[] for _ in batch]}

    adapter.extract(
        [Item(text=f"text number {index}") for index in range(20)],
        labels=["person"],
        options={"relation_labels": ["knows"]},
    )

    # 100 relation entities per document -> 9900 pairs; 131072 pairs per pass.
    assert [call.kwargs["batch_size"] for call in model.inference.call_args_list] == [13, 7]


def test_relation_types_are_bounded() -> None:
    adapter, model = _adapter({})
    with pytest.raises(InvalidInputError, match="at most 20 relation types"):
        adapter.extract(
            [Item(text="hello world")], labels=["person"], options={"relation_labels": [f"r{i}" for i in range(21)]}
        )
    supplied = Item(
        text="Ada met Bo", metadata={"entities": [{"text": "Ada", "label": "person", "start": 0, "end": 3}]}
    )
    with pytest.raises(InvalidInputError, match="at most 20 relation types"):
        adapter.extract([supplied], labels=[f"r{i}" for i in range(21)])
    model.inference.assert_not_called()


def test_outputs_are_decoded_from_host_memory() -> None:
    logits = torch.zeros(1, 2, 3, dtype=torch.float16)
    output = {
        "joint_rel_logits": logits,
        "joint_rel_idx": torch.zeros(1, 2, 2, dtype=torch.long),
        "words_embedding": torch.zeros(1, 2, dtype=torch.float16),
        "batch_size": 1,
    }

    result = adapter_module._upcast_score_outputs(MagicMock(), (), output)

    assert all(value.device.type == "cpu" for value in result.values() if isinstance(value, torch.Tensor))
    assert result["joint_rel_logits"].dtype == torch.float32
    assert result["words_embedding"].dtype == torch.float16
    assert result["batch_size"] == 1


def _relation_head(**kwargs: Any) -> SimpleNamespace:
    def decode(ner_scores: Any, *_: Any, **__: Any) -> Any:
        mask = torch.tensor([[True, True, False]])
        return torch.zeros(1, 3, 2, dtype=torch.long), mask, torch.zeros(1, 3, dtype=torch.long)

    return SimpleNamespace(**kwargs, _decode_relation_entity_spans=decode)


@pytest.mark.parametrize(("current", "expected"), [(None, 100), (40, 40), (500, 100)])
def test_relation_candidates_are_limited_at_load(current: int | None, expected: int) -> None:
    head = _relation_head(max_relation_entities=current)
    adapter_module._limit_relation_entities(SimpleNamespace(heads={"joint_relex": head, "ner": object()}))
    assert head.max_relation_entities == expected


def test_relation_candidates_cost_proposal_units() -> None:
    head = _relation_head(max_relation_entities=None)
    units = []

    def decode(ner_scores: Any, *_: Any, **__: Any) -> Any:
        units.append(span_decoding._CANDIDATE_UNITS.get())
        return None

    head._decode_relation_entity_spans = decode
    adapter_module._limit_relation_entities(SimpleNamespace(heads={"joint_relex": head, "ner": object()}))
    head._decode_relation_entity_spans(torch.zeros(1))
    assert units == [span_decoding.PROPOSAL_UNITS]
    assert span_decoding._CANDIDATE_UNITS.get() == 1


def test_relation_limit_fails_closed_without_a_relation_head() -> None:
    for model in (SimpleNamespace(heads={"ner": object()}), SimpleNamespace()):
        with pytest.raises(RuntimeError, match="no joint relation head"):
            adapter_module._limit_relation_entities(model)
    with pytest.raises(RuntimeError, match="no longer decodes entity candidates"):
        adapter_module._limit_relation_entities(
            SimpleNamespace(heads={"joint_relex": SimpleNamespace(max_relation_entities=None)})
        )


def test_supplied_entity_label_order_does_not_create_groups() -> None:
    adapter, model = _adapter({})
    model.inference.side_effect = lambda batch, **_: {"ner": [[] for _ in batch], "joint_relex": [[] for _ in batch]}
    orders = (["person", "company"], ["company", "person"])
    items = [
        Item(
            text="Ada founded Acme",
            metadata={
                "entities": [
                    {
                        "text": "Ada" if label == "person" else "Acme",
                        "label": label,
                        "start": 0 if label == "person" else 12,
                        "end": 3 if label == "person" else 16,
                    }
                    for label in order
                ]
            },
        )
        for order in orders
    ]

    adapter.extract(items, labels=["founded"])

    assert model.inference.call_count == 1
    assert model.inference.call_args.kwargs["batch_size"] == 2


def test_distinct_supplied_label_sets_are_capped() -> None:
    adapter, model = _adapter({})
    items = [
        Item(text="Ada met Bo", metadata={"entities": [{"text": "Ada", "label": f"t{index}", "start": 0, "end": 3}]})
        for index in range(65)
    ]
    with pytest.raises(InvalidInputError, match="at most 64 distinct sets"):
        adapter.extract(items, labels=["met"])
    model.inference.assert_not_called()
    model.data_processor.collate_raw_batch.assert_not_called()


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_inference_batch_tokens_must_be_a_positive_integer(budget: object) -> None:
    with pytest.raises(ValueError, match="inference_batch_tokens"):
        GLiFormerAdapter("test-model", inference_batch_tokens=budget)  # type: ignore[arg-type]


def test_prompt_that_exhausts_the_window_is_rejected() -> None:
    adapter, model = _adapter({"ner": [[]]}, max_len=5)
    with pytest.raises(InvalidInputError, match="leaves no document tokens"):
        adapter.extract([Item(text="Alice works here")], labels=["person"])
    model.inference.assert_not_called()


def test_prompt_over_budget_is_rejected_before_any_document_is_tokenized() -> None:
    adapter, model = _adapter({"ner": [[]]}, max_prompt_tokens=2)
    tokenizer = model.data_processor.transformer_tokenizer

    with pytest.raises(InvalidInputError, match="task prompt needs 3 tokens"):
        adapter.extract([Item(text="Alice works here")] * 64, labels=["person"])

    # Only the prompt itself was tokenized, once.
    assert tokenizer.calls == [[["P", "P", "P"]]]
    model.inference.assert_not_called()


def test_every_task_group_is_measured_before_inference_runs() -> None:
    adapter, model = _adapter({}, max_prompt_tokens=3)
    prompt_words = {("a",): 3, ("b",): 4}

    def prepare_inputs(tokens: list[list[str]], _mapping: Any) -> tuple[list[list[str]], list[int]]:
        types = model._build_inference_input.call_args.kwargs["joint_relations"][None]["entities"]
        count = prompt_words[tuple(types)]
        return [["P"] * count + list(words) for words in tokens], [count] * len(tokens)

    model.data_processor.prepare_inputs.side_effect = prepare_inputs
    items = [
        Item(text="Ada met Bo", metadata={"entities": [{"text": "Ada", "label": "a", "start": 0, "end": 3}]}),
        Item(text="Ada met Bo", metadata={"entities": [{"text": "Bo", "label": "b", "start": 8, "end": 10}]}),
    ]

    with pytest.raises(InvalidInputError, match="task prompt needs 4 tokens"):
        adapter.extract(items, labels=["met"])
    model.inference.assert_not_called()


def _prompt_measurements(model: MagicMock) -> int:
    return sum(sequences == [["P", "P", "P"]] for sequences in model.data_processor.transformer_tokenizer.calls)


def test_a_repeated_task_prompt_is_measured_once() -> None:
    adapter, model = _adapter({"ner": [[]]})

    for text in ("Alice works here", "Bob works there", "Alice works here"):
        adapter.extract([Item(text=text)], labels=["person", "place"])

    assert _prompt_measurements(model) == 1
    assert model.inference.call_count == 3


def test_task_prompts_are_cached_by_their_exact_task_arguments() -> None:
    adapter, model = _adapter({"ner": [[]]})
    requests: list[dict[str, Any]] = [
        {"labels": ["person", "place"]},
        {"labels": ["place", "person"]},
        {"labels": ["person"], "options": {"relation_labels": ["works at"]}},
        {"options": {"label_groups": {"a": ["x", "y"]}, "threshold": 0.0}},
        {"options": {"label_groups": {"a": ["y", "x"]}, "threshold": 0.0}},
        {"options": {"label_groups": {"b": ["x", "y"]}, "threshold": 0.0}},
    ]

    for request in requests:
        adapter.extract([Item(text="Alice works here")], **request)
    assert _prompt_measurements(model) == len(requests)

    for request in requests:
        adapter.extract([Item(text="Alice works here")], **request)
    assert _prompt_measurements(model) == len(requests)


def test_prompt_limits_are_checked_on_every_request() -> None:
    adapter, model = _adapter({"ner": [[]]}, max_prompt_tokens=2)

    for _ in range(2):
        with pytest.raises(InvalidInputError, match="task prompt needs 3 tokens"):
            adapter.extract([Item(text="Alice works here")], labels=["person"])

    assert _prompt_measurements(model) == 1
    model.inference.assert_not_called()


def test_prompt_cache_keeps_the_most_recently_used_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module, "_PROMPT_CACHE_SIZE", 2)
    adapter, model = _adapter({"ner": [[]]})

    def request(label: str) -> None:
        adapter.extract([Item(text="Alice works here")], labels=[label])

    for label in ("a", "b", "a", "c"):
        request(label)
    assert len(adapter._prompt_counts) == 2
    assert _prompt_measurements(model) == 3

    request("a")
    assert _prompt_measurements(model) == 3
    request("b")
    assert _prompt_measurements(model) == 4


def test_prompt_build_failure_is_an_internal_error() -> None:
    adapter, model = _adapter({"ner": [[]]})
    model.data_processor.collate_raw_batch.side_effect = KeyError("processor drift")
    with pytest.raises(RuntimeError, match="could not build the task prompt") as info:
        adapter.extract([Item(text="Alice works here")], labels=["person"])
    assert not isinstance(info.value, ValueError)
    model.inference.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"labels": ["x" * 129]}, "at most 128 characters"),
        ({"labels": ["a"], "options": {"relation_labels": ["r" * 129]}}, "at most 128 characters"),
        ({"labels": ["a"], "options": {"classification_task": "t" * 129}}, "at most 128 characters"),
        ({"options": {"label_groups": {"g" * 129: ["a"]}}}, "at most 128 characters"),
        ({"options": {"label_groups": {"g": ["a" * 129]}}}, "at most 128 characters"),
        (
            {"labels": [f"e{i}" for i in range(600)], "options": {"relation_labels": [f"r{i}" for i in range(401)]}},
            "in total",
        ),
        ({"options": {"label_groups": {f"g{i}": ["a", "b"] for i in range(334)}}}, "in total"),
        (
            {"options": {"label_groups": {f"g{i}": [f"l{j}" for j in range(10)] for i in range(101)}}},
            "at most 1000 labels",
        ),
    ],
)
def test_prompt_labels_are_bounded(kwargs: dict[str, Any], match: str) -> None:
    adapter, model = _adapter({})
    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text="hello world")], **kwargs)
    model.inference.assert_not_called()


def test_supplied_entities_are_bounded() -> None:
    adapter, model = _adapter({})
    text = "Ada founded Acme"
    too_long = Item(text=text, metadata={"entities": [{"text": "Ada", "label": "p" * 129, "start": 0, "end": 3}]})
    with pytest.raises(InvalidInputError, match="at most 128 characters"):
        adapter.extract([too_long], labels=["founded"])
    too_many = Item(
        text=text,
        metadata={"entities": [{"text": "Ada", "label": "person", "start": 0, "end": 3}] * 1001},
    )
    with pytest.raises(InvalidInputError, match="at most 1000 entities"):
        adapter.extract([too_many], labels=["founded"])
    distinct = [
        Item(text=text, metadata={"entities": [{"text": "Ada", "label": f"t{i}-{j}", "start": 0, "end": 3}]})
        for i in range(2)
        for j in range(500)
    ]
    with pytest.raises(InvalidInputError, match="in total"):
        adapter.extract(distinct, labels=["founded"])
    model.inference.assert_not_called()


def test_caller_text_echoed_in_errors_is_clipped() -> None:
    adapter, _ = _adapter({})
    name = "n" * 10_000
    with pytest.raises(InvalidInputError) as info:
        adapter.extract([Item(text="hello")], options={"label_groups": {"ok": ["a", "a"], name: ["b"]}})
    assert len(str(info.value)) < 300
    schema = {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"format": "x" * 10_000}}}}}
    with pytest.raises(InvalidInputError) as info:
        compile_output_schema(
            {"type": "object", "properties": {"p" * 100: schema, "q": {"$ref": "#/$defs/" + "z" * 5000}}}
        )
    assert len(str(info.value)) < 400


def test_metering_normalizes_structuring_templates_like_inference() -> None:
    adapter, model = _adapter({"structuring": [{"name": "Ann"}]})
    adapter._normalize_structures = MagicMock(side_effect=lambda structures: {"normalized": True})
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}

    adapter.extract([Item(text="Ann asked")], output_schema=schema)

    adapter._normalize_structures.assert_called_once_with({"$root": {"name": "str"}})
    assert model._build_inference_input.call_args.kwargs["structures"] == {"normalized": True}
    assert _inference_kwargs(model)["structures"] == {"$root": {"name": "str"}}


# -- Encode --------------------------------------------------------------------------


def test_encode_returns_normalized_dense_embeddings_with_token_counts() -> None:
    adapter, model = _adapter({})
    model.embed_text.return_value = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    adapter._tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3], [1, 2]]})

    output = adapter.encode([Item(text="a b"), Item(text="c")], ["dense"], is_query=True)

    np.testing.assert_allclose(output.dense, [[0.6, 0.8], [0.0, 1.0]])
    assert output.is_query is True
    assert output.extra["input_token_counts"] == [3, 2]
    model.embed_text.assert_called_once_with(["a b", "c"], batch_size=2)
    adapter._tokenizer.assert_called_once_with(["a b", "c"], truncation=True)

    raw = adapter.encode([Item(text="a b"), Item(text="c")], ["dense"], options={"normalize": False})
    np.testing.assert_allclose(raw.dense, [[3.0, 4.0], [0.0, 2.0]])


def test_encode_is_split_into_padded_token_chunks() -> None:
    adapter, model = _adapter({}, inference_batch_tokens=6)
    model.embed_text.side_effect = lambda texts, batch_size: torch.ones(len(texts), 2)
    adapter._tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3], [1, 2], [1, 2, 3, 4]]})

    output = adapter.encode([Item(text="a b"), Item(text="c"), Item(text="d e f")], ["dense"])

    assert [call.kwargs["batch_size"] for call in model.embed_text.call_args_list] == [2, 1]
    assert output.dense.shape == (3, 2)
    assert output.extra["input_token_counts"] == [3, 2, 4]


def test_encode_rejects_non_finite_embeddings() -> None:
    adapter, model = _adapter({})
    model.embed_text.return_value = torch.tensor([[float("nan"), 1.0]], dtype=torch.float16)
    adapter._tokenizer = MagicMock(return_value={"input_ids": [[1, 2]]})
    with pytest.raises(RuntimeError, match="non-finite embeddings"):
        adapter.encode([Item(text="a b")], ["dense"])


def test_encode_rejects_non_dense_outputs() -> None:
    adapter, _ = _adapter({})
    with pytest.raises(InvalidInputError, match="only dense"):
        adapter.encode([Item(text="a")], ["dense", "sparse"])


# -- Loading ---------------------------------------------------------------------------


def _fake_heads() -> dict[str, Any]:
    ner_head = MagicMock()
    relation_head = _relation_head(max_relation_entities=None, _owns_ner_head=False, _reused_ner_head=ner_head)
    return {"ner": ner_head, "joint_relex": relation_head}


def _fake_gliformer_module(model: MagicMock) -> ModuleType:
    module = ModuleType("gliformer")
    module.GLiFormer = MagicMock()  # type: ignore[attr-defined]
    module.GLiFormer.from_pretrained.return_value = model
    module.processing = SimpleNamespace(  # type: ignore[attr-defined]
        schema=SimpleNamespace(
            normalize_structuring_schemas="normalizer", build_structuring_output_formatter="formatter"
        )
    )
    return module


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        ("cpu", None, torch.float32),
        ("cuda:0", None, torch.float16),
        ("cuda:0", "bfloat16", torch.bfloat16),
    ],
)
def test_load_pins_snapshot_and_places_model(device: str, precision: str | None, dtype: torch.dtype) -> None:
    model = MagicMock()
    model.config = SimpleNamespace(max_len=2048, embedding_config=SimpleNamespace(projection_dim=768))
    ner_head = MagicMock()
    relation_head = _relation_head(max_relation_entities=None, _owns_ner_head=False, _reused_ner_head=ner_head)
    model.model.heads = {"joint_relex": relation_head, "ner": ner_head}
    set_eval_mode = model.eval
    tokenizer = model.data_processor.transformer_tokenizer
    precision_kwargs = {} if precision is None else {"compute_precision": precision}
    adapter = GLiFormerAdapter(
        "knowledgator/gliformer-base-v1",
        revision="a" * 40,
        max_seq_length=2048,
        dense_dim=768,
        **precision_kwargs,  # type: ignore[arg-type]
    )
    module = _fake_gliformer_module(model)

    with (
        patch.dict(sys.modules, {"gliformer": module}),
        patch.object(adapter_module, "_bound_span_decoding") as bound_span_decoding,
        patch.object(adapter_module, "_assert_bounded_decoding") as assert_bounded,
        patch.object(adapter_module, "_verify_bounded_decoding") as verify_bounded,
        patch.object(adapter_module, "snapshot_download", return_value="/staged/gliformer") as download,
    ):
        adapter.load(device)

    download.assert_called_once_with(
        repo_id="knowledgator/gliformer-base-v1",
        revision="a" * 40,
        ignore_patterns=["*.gif"],
    )
    module.GLiFormer.from_pretrained.assert_called_once_with(
        "/staged/gliformer",
        load_tokenizer=True,
        map_location="cpu",
        max_length=2048,
    )
    model.to.assert_called_once_with(device=device, dtype=dtype)
    set_eval_mode.assert_called_once_with()
    model.model.register_forward_hook.assert_called_once_with(adapter_module._upcast_score_outputs)
    ner_head.register_forward_hook.assert_called_once_with(adapter_module._mask_padded_ner_logits)
    bound_span_decoding.assert_called_once_with()
    assert_bounded.assert_called_once_with()
    verify_bounded.assert_called_once_with(model)
    # One small extraction at load, through every head the hooks check.
    probe = model.inference.call_args
    assert probe.args[0] == adapter_module._PROBE_TEXTS
    assert probe.kwargs["joint_relations"] is not None
    assert probe.kwargs["structures"] is not None
    assert relation_head.max_relation_entities == 100
    assert tokenizer.model_max_length == 2048
    assert adapter._tokenizer is tokenizer
    assert adapter._normalize_structures == "normalizer"
    assert adapter._build_formatter == "formatter"

    adapter.unload()
    assert adapter._model is None
    assert adapter._tokenizer is None


def test_per_request_eval_walks_the_model_only_when_it_is_training() -> None:
    model = MagicMock()
    model.config = SimpleNamespace(max_len=2048, embedding_config=SimpleNamespace(projection_dim=768))
    model.model.heads = _fake_heads()
    set_eval_mode = model.eval
    set_eval_mode.side_effect = lambda: setattr(model, "training", False) or model
    model.train.side_effect = lambda mode: setattr(model, "training", mode) or model
    model.training = True
    adapter = GLiFormerAdapter("/local/checkpoint")
    with (
        patch.dict(sys.modules, {"gliformer": _fake_gliformer_module(model)}),
        patch.object(adapter_module, "_bound_span_decoding"),
        patch.object(adapter_module, "_assert_bounded_decoding"),
        patch.object(adapter_module, "_verify_bounded_decoding"),
        patch.object(adapter_module.Path, "is_dir", return_value=True),
    ):
        adapter.load("cpu")
    set_eval_mode.assert_called_once_with()

    # GLiFormer.inference calls eval() on every request.
    assert model.eval() is model
    assert model.eval() is model
    model.train.assert_not_called()

    model.training = True
    assert model.eval() is model
    model.train.assert_called_once_with(False)
    assert model.training is False


def test_per_request_eval_shortcut_does_not_keep_the_model_alive() -> None:
    class _Model:
        training = False

    model = _Model()
    adapter_module._skip_redundant_eval(model)
    released = weakref.ref(model)

    del model

    assert released() is None


def test_load_fails_when_the_probe_forward_is_rejected() -> None:
    model = MagicMock()
    model.config = SimpleNamespace(max_len=2048, embedding_config=SimpleNamespace(projection_dim=768))
    model.model.heads = _fake_heads()
    model.inference.side_effect = RuntimeError("GLiFormer NER logits came without a matching word mask")
    adapter = GLiFormerAdapter("/local/checkpoint")
    with (
        patch.dict(sys.modules, {"gliformer": _fake_gliformer_module(model)}),
        patch.object(adapter_module, "_bound_span_decoding"),
        patch.object(adapter_module, "_assert_bounded_decoding"),
        patch.object(adapter_module, "_verify_bounded_decoding"),
        patch.object(adapter_module.Path, "is_dir", return_value=True),
        pytest.raises(RuntimeError, match="word mask"),
    ):
        adapter.load("cpu")
    assert adapter._model is None


def test_load_rejects_embedding_dimension_mismatch() -> None:
    model = MagicMock()
    model.config = SimpleNamespace(max_len=2048, embedding_config=SimpleNamespace(projection_dim=768))
    model.model.heads = _fake_heads()
    adapter = GLiFormerAdapter("/local/checkpoint", dense_dim=1024)
    with (
        patch.dict(sys.modules, {"gliformer": _fake_gliformer_module(model)}),
        patch.object(adapter_module, "_bound_span_decoding"),
        patch.object(adapter_module, "_assert_bounded_decoding"),
        patch.object(adapter_module, "_verify_bounded_decoding"),
        patch.object(adapter_module.Path, "is_dir", return_value=True),
        pytest.raises(ValueError, match="dimension mismatch"),
    ):
        adapter.load("cpu")


class _FakeAutoMapping:
    """The parts of transformers' lazy ``MODEL_MAPPING`` the adapter inspects."""

    def __init__(self, extra: dict[Any, Any], builtin: dict[Any, Any]) -> None:
        self._extra_content = extra
        self._reverse_config_mapping = {config.__name__: config.__name__.lower() for config in builtin}
        self._model_mapping = {config.__name__.lower(): model.__name__ for config, model in builtin.items()}
        self._builtin = builtin

    def __getitem__(self, config: Any) -> Any:
        return self._extra_content.get(config) or self._builtin[config]


_GLIFORMER_QWEN3 = type("GLiFormerQwen3Model", (), {"__module__": "gliformer.backbones.qwen3"})


def test_auto_model_overrides_of_builtin_configs_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    new_config = type("LayoutDebertaConfig", (), {})
    gliformer_layout = type("Layout", (), {"__module__": "gliformer.backbones.deberta_2d"})
    other_config = type("Qwen2Config", (), {})
    other_override = type("Custom", (), {"__module__": "someone.else"})
    mapping = _FakeAutoMapping(
        extra={
            adapter_module.Qwen3Config: _GLIFORMER_QWEN3,
            new_config: gliformer_layout,
            other_config: other_override,
        },
        builtin={adapter_module.Qwen3Config: adapter_module.Qwen3Model, other_config: object},
    )
    monkeypatch.setattr(adapter_module.modeling_auto, "MODEL_MAPPING", mapping)

    adapter_module._drop_builtin_auto_model_overrides()

    assert mapping._extra_content == {new_config: gliformer_layout, other_config: other_override}
    assert mapping[adapter_module.Qwen3Config] is adapter_module.Qwen3Model


def test_auto_model_restore_fails_closed_when_qwen3_is_still_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = _FakeAutoMapping(extra={}, builtin={adapter_module.Qwen3Config: _GLIFORMER_QWEN3})
    monkeypatch.setattr(adapter_module.modeling_auto, "MODEL_MAPPING", mapping)
    with pytest.raises(RuntimeError, match="no longer resolves to Qwen3Model"):
        adapter_module._drop_builtin_auto_model_overrides()


def test_auto_model_restore_fails_closed_on_unknown_mapping_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_module.modeling_auto, "MODEL_MAPPING", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        adapter_module._drop_builtin_auto_model_overrides()


def test_importing_gliformer_through_the_adapter_keeps_transformers_auto_models() -> None:
    # Fresh interpreter: the package registers its classes once per process.
    script = (
        "from sie_server.adapters.gliformer.adapter import _import_gliformer\n"
        "_import_gliformer()\n"
        "from transformers import Qwen3Config, Qwen3Model\n"
        "from transformers.models.auto.modeling_auto import MODEL_MAPPING\n"
        "from gliformer.backbones import LayoutDebertaConfig, LayoutDebertaModel\n"
        "assert MODEL_MAPPING[Qwen3Config] is Qwen3Model, MODEL_MAPPING[Qwen3Config]\n"
        "assert MODEL_MAPPING[LayoutDebertaConfig] is LayoutDebertaModel\n"
        "from gliformer.tasks.span_decoder import SpanDecoder\n"
        "assert SpanDecoder._calculate_span_score._sie_bounded is True\n"
        "import importlib\n"
        "from sie_server.adapters.gliformer import adapter\n"
        "for name in adapter._PROPOSAL_MODULES:\n"
        "    module = importlib.import_module(name)\n"
        "    assert module.extract_spans_from_tokens._sie_bounded is True, name\n"
        "adapter._assert_bounded_decoding()\n"
    )
    result = subprocess.run(  # noqa: S603 — fixed interpreter and script
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stderr[-4000:]


# -- output_schema compilation -----------------------------------------------------------


class _Employee(BaseModel):
    name: str
    role: str | None = None


class _Company(BaseModel):
    name: str
    sector: str | None = None
    employees: list[_Employee]
    status: Literal["active", "closed"]


def test_pydantic_json_schema_compiles_to_nested_template_and_choice_group() -> None:
    plan = compile_output_schema(_Company.model_json_schema())

    assert plan.structures == {
        "$root": {"name": "str", "sector": "str", "employees": [{"name": "str", "role": "str"}]},
    }
    assert plan.choice_groups == {"status": ["active", "closed"]}


def test_structured_output_drops_unextracted_values_and_incomplete_records() -> None:
    plan = compile_output_schema(_Company.model_json_schema())
    raw = {
        "name": "Acme",
        "sector": None,
        "employees": [{"name": "Ann", "role": None}, {"name": None, "role": "cook"}, {}],
        "unexpected": "value",
    }

    data, missing = shape_structured_output(plan, raw, {"status": "active"})

    assert data == {"name": "Acme", "employees": [{"name": "Ann"}], "status": "active"}
    assert missing == []
    data, missing = shape_structured_output(plan, {}, {})
    assert (data, missing) == ({}, ["name", "employees", "status"])


def test_nested_objects_lacking_required_values_are_discarded() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {"street": {"type": "string"}, "city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    plan = compile_output_schema(schema)
    assert plan.structures == {"$root": {"title": "str", "address": {"street": "str", "city": "str"}}}
    assert shape_structured_output(plan, {"title": "HQ", "address": {"street": "1 Main St"}}, {}) == (
        {"title": "HQ"},
        [],
    )


def test_single_valued_fields_keep_the_best_of_several_candidates() -> None:
    schema = {
        "type": "object",
        "properties": {
            "customer": {"type": "string"},
            "address": {"type": "object", "properties": {"city": {"type": "string"}}},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    }
    plan = compile_output_schema(schema)
    raw = {
        "customer": [None, "Maria Lopez", "Tim Cook"],
        "address": [{"city": "Boston"}, {"city": "London"}],
        "tags": ["refund", "late"],
    }

    assert shape_structured_output(plan, raw, {}) == (
        {"customer": "Maria Lopez", "address": {"city": "Boston"}, "tags": ["refund", "late"]},
        [],
    )


@pytest.mark.parametrize("value", [[3], 3, {"a": "b"}])
def test_structured_output_with_wrong_shapes_is_rejected(value: object) -> None:
    plan = compile_output_schema({"type": "object", "properties": {"name": {"type": "string"}}})
    with pytest.raises(RuntimeError, match="malformed structured output"):
        shape_structured_output(plan, {"name": value}, {})


def test_optional_types_and_annotations_are_accepted() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "title": "Ticket",
        "description": "A support ticket",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": ["string", "null"], "description": "One line", "default": None},
            "tags": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
            "!urgent": {"type": "string", "examples": ["yes"]},
        },
    }
    plan = compile_output_schema(schema)
    assert plan.structures == {"$root": {"summary": "str", "tags": ["str"], "!!urgent": "str"}}


@pytest.mark.parametrize(
    ("schema", "match"),
    [
        ({"type": "array", "items": {"type": "string"}}, "root type must be object"),
        ({"type": "object", "properties": {}}, "non-empty properties"),
        ({"type": "object", "properties": {"age": {"type": "integer"}}}, "supports string"),
        ({"type": "object", "properties": {"n": {"type": "string", "format": "date"}}}, "unsupported keywords"),
        ({"type": "object", "properties": {"a.b": {"type": "string"}}}, "contain no '.'"),
        ({"type": "object", "properties": {"n": {"type": "string"}}, "required": ["m"]}, "required must list"),
        (
            {"type": "object", "properties": {"n": {"type": "string"}}, "additionalProperties": {"type": "string"}},
            "additionalProperties must be boolean",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "p": {"type": "object", "properties": {"s": {"type": "string", "enum": ["a", "b"]}}},
                },
            },
            "only at the schema root",
        ),
        (
            {"type": "object", "properties": {"t": {"type": "array", "items": {"type": "string", "enum": ["a"]}}}},
            "arrays of enum values",
        ),
        ({"type": "object", "properties": {"s": {"type": "string", "enum": ["a", "a"]}}}, "unique non-empty"),
        ({"type": "object", "properties": {"s": {"enum": [1, 2]}}}, "unique non-empty strings"),
        ({"type": "object", "properties": {"s": {"$ref": "#/$defs/Missing"}}}, "unresolvable \\$ref"),
        (
            {
                "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/Node"}},
            },
            "recursive",
        ),
        (
            {"type": "object", "properties": {"s": {"anyOf": [{"type": "string"}, {"type": "object"}]}}},
            "optional \\(nullable\\) type",
        ),
    ],
)
def test_unsupported_output_schemas_are_rejected(schema: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInputError, match=match):
        compile_output_schema(schema)


@pytest.mark.parametrize(
    "properties",
    [
        {"fields": {"type": "string"}, "description": {"type": "string"}},
        {"children": {"type": "string"}},
        {
            "people": {
                "type": "array",
                "items": {"type": "object", "properties": {"required_fields": {"type": "string"}}},
            }
        },
    ],
)
def test_objects_that_read_as_legacy_descriptors_are_rejected(properties: dict[str, Any]) -> None:
    with pytest.raises(InvalidInputError, match="cannot consist only of properties named"):
        compile_output_schema({"type": "object", "properties": properties})


def test_descriptor_words_mixed_with_other_properties_are_accepted() -> None:
    schema = {"type": "object", "properties": {"fields": {"type": "string"}, "name": {"type": "string"}}}
    assert compile_output_schema(schema).structures == {"$root": {"fields": "str", "name": "str"}}
    lone = {"type": "object", "properties": {"description": {"type": "string"}}}
    assert compile_output_schema(lone).structures == {"$root": {"description": "str"}}


def _fan_out_schema(depth: int, fan: int = 2) -> dict[str, Any]:
    defs: dict[str, Any] = {}
    for level in range(depth):
        defs[f"A{level}"] = {
            "type": "object",
            "properties": {f"p{j}": {"$ref": f"#/$defs/A{level + 1}"} for j in range(fan)},
        }
    defs[f"A{depth}"] = {"type": "object", "properties": {"leaf": {"type": "string"}}}
    return {"type": "object", "properties": {"root": {"$ref": "#/$defs/A0"}}, "$defs": defs}


def test_shared_ref_fan_out_is_rejected_while_compiling() -> None:
    started = time.perf_counter()
    with pytest.raises(InvalidInputError, match="more than 1000 properties"):
        compile_output_schema(_fan_out_schema(20))
    assert time.perf_counter() - started < 0.5


def test_ref_expansions_are_capped() -> None:
    # 40 properties, each resolving a 30-step alias chain: 1200 expansions.
    defs: dict[str, Any] = {f"C{i}": {"$ref": f"#/$defs/C{i + 1}"} for i in range(29)}
    defs["C29"] = {"type": "string"}
    schema = {
        "type": "object",
        "properties": {f"p{i}": {"$ref": "#/$defs/C0"} for i in range(40)},
        "$defs": defs,
    }
    with pytest.raises(InvalidInputError, match=r"expands more than 1000 \$ref values"):
        compile_output_schema(schema)


def test_ref_chains_are_capped() -> None:
    defs: dict[str, Any] = {f"A{i}": {"$ref": f"#/$defs/A{i + 1}"} for i in range(40)}
    defs["A40"] = {"type": "string"}
    schema = {"type": "object", "properties": {"x": {"$ref": "#/$defs/A0"}}, "$defs": defs}
    with pytest.raises(InvalidInputError, match="chains more than 32"):
        compile_output_schema(schema)


def test_object_nesting_is_capped() -> None:
    def nested(levels: int) -> dict[str, Any]:
        node: dict[str, Any] = {"type": "string"}
        for _ in range(levels):
            node = {"type": "object", "properties": {"a": node}}
        return node

    compile_output_schema(nested(32))
    with pytest.raises(InvalidInputError, match="more than 32 levels"):
        compile_output_schema(nested(33))
    with pytest.raises(InvalidInputError, match="nest objects and arrays at most"):
        compile_output_schema(nested(500))


@pytest.mark.parametrize(
    "properties",
    [
        {"n" * 129: {"type": "string"}},
        {"status": {"type": "string", "enum": ["ok", "c" * 129]}},
    ],
)
def test_schema_names_and_choices_are_bounded(properties: dict[str, Any]) -> None:
    with pytest.raises(InvalidInputError, match="at most 128 characters"):
        compile_output_schema({"type": "object", "properties": properties})


def test_output_schema_property_count_is_capped() -> None:
    schema = {"type": "object", "properties": {f"field {index}": {"type": "string"} for index in range(1001)}}
    with pytest.raises(InvalidInputError, match="more than 1000 properties"):
        compile_output_schema(schema)


# -- Catalog and bundle wiring --------------------------------------------------------------


@pytest.mark.parametrize(("model_id", "dim"), sorted(_MODEL_IDS.items()))
def test_catalog_configs_route_to_the_adapter(model_id: str, dim: int) -> None:
    config = load_model_configs(_MODELS_DIR)[model_id]
    assert config.hf_revision is not None
    assert len(config.hf_revision) == 40
    assert config.tasks.extract is not None
    assert config.tasks.encode is not None
    assert config.tasks.encode.dense is not None
    assert config.tasks.encode.dense.dim == dim
    adapter = load_adapter(config, _MODELS_DIR, device="cpu")
    assert isinstance(adapter, GLiFormerAdapter)
    assert adapter._revision == config.hf_revision
    assert adapter._max_seq_length == config.max_sequence_length
    assert adapter._compute_precision == "float16"
    assert adapter.dims.dense == dim


def test_default_bundle_ships_the_adapter_and_its_pinned_package() -> None:
    bundle = yaml.safe_load((_SERVER_ROOT / "bundles" / "default.yaml").read_text())
    pyproject = (_SERVER_ROOT / "pyproject.toml").read_text()
    assert "sie_server.adapters.gliformer.adapter" in bundle["adapters"]
    assert bundle["deps"]["gliformer"] == "==0.1.2"
    assert '"gliformer==0.1.2"' in pyproject
