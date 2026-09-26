"""The limit on the task prompt (labels, relation types, class labels, schema fields) of GLiNER-family adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.adapters._prompt_limit import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_SCHEMA_PROMPT_TOKENS,
    MAX_LABEL_CHARS,
    MAX_PROMPT_CHARS_PER_TOKEN,
    PromptLimit,
)
from sie_server.adapters.gliner import GLiNERAdapter
from sie_server.adapters.gliner2.classification import GLiNER2ClassificationAdapter
from sie_server.adapters.glirel import MAX_ENTITIES, GLiRELAdapter
from sie_server.types.inputs import InvalidInputError, Item

from .test_gliner2_long_text import make_adapter as make_gliner2_adapter
from .test_gliner_long_words import DEBERTA, FakeGLiNER, char_tokenizer, encoded, load

# The largest label set among the repository's examples (12 labels). The stand-in
# tokenizer takes a token per character, so these take about 200 tokens.
EXAMPLE_LABELS = [
    "prompt injection",
    "jailbreak",
    "role play",
    "system prompt leak",
    "harmful request",
    "data exfiltration",
    "instruction override",
    "obfuscation",
    "social engineering",
    "policy evasion",
    "privilege escalation",
    "benign",
]
# Sixty short types (about 300 tokens with the stand-in tokenizer).
PII_LABELS = [f"pii{index}" for index in range(60)]


class Counting:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return self.tokens


def test_a_prompt_within_the_limit_passes_and_is_counted_once() -> None:
    limit = PromptLimit("Model", 100)
    count = Counting(40)

    assert limit.check(["a", "b"], count, ("a", "b")) == 40
    assert limit.check(["a", "b"], count, ("a", "b")) == 40
    assert count.calls == 1


def test_a_prompt_over_the_limit_is_rejected_every_time() -> None:
    limit = PromptLimit("Model", 100)
    count = Counting(101)

    for _ in range(2):
        with pytest.raises(InvalidInputError, match="take 101 tokens; a request may use at most 100"):
            limit.check(["a"], count, ("a",))
    assert count.calls == 1


def test_a_prompt_of_too_many_characters_is_rejected_before_it_is_tokenized() -> None:
    limit = PromptLimit("Model", 10)
    count = Counting(1)

    with pytest.raises(InvalidInputError, match="characters"):
        limit.check(["x" * (10 * MAX_PROMPT_CHARS_PER_TOKEN + 1)], count, ("long",))
    assert count.calls == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_the_limit_must_be_a_positive_integer(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        PromptLimit("Model", value)


def test_gliner_counts_the_prompt_its_processor_builds() -> None:
    model = FakeGLiNER(DEBERTA)
    adapter = load(GLiNERAdapter, model)
    labels = ["person", "organization"]

    tokens = adapter._count_prompt(labels, [])

    # The encoded row is [CLS], the prompt, the document's subwords, [SEP].
    (row,) = encoded(model, ["abc"], labels)
    assert tokens == len(row) - 3 - 2


@pytest.mark.parametrize("labels", [EXAMPLE_LABELS, PII_LABELS], ids=["example-labels", "pii-list"])
def test_gliner_accepts_ordinary_label_sets(labels: list[str]) -> None:
    model = FakeGLiNER(DEBERTA)
    adapter = load(GLiNERAdapter, model)

    adapter.extract([Item(text="Priya Raman works at Novartis.")], labels=labels)

    assert model.calls == [["Priya Raman works at Novartis."]]


@pytest.mark.parametrize(
    "labels",
    [[f"{index}" + "x" * 120 for index in range(12)], [f"type number {index}" for index in range(100)]],
    ids=["long-labels", "many-labels"],
)
def test_gliner_rejects_a_prompt_over_the_limit_before_inference(labels: list[str]) -> None:
    model = FakeGLiNER(DEBERTA)
    adapter = load(GLiNERAdapter, model)

    with pytest.raises(InvalidInputError, match=f"at most {DEFAULT_MAX_PROMPT_TOKENS} tokens"):
        adapter.extract([Item(text="Priya Raman works at Novartis.")], labels=labels)
    assert model.calls == []


def test_gliner_limit_is_configurable() -> None:
    model = FakeGLiNER(DEBERTA)
    adapter = GLiNERAdapter("fake/gliner", max_prompt_tokens=20)
    adapter._model = model
    adapter._count_prompt = load(GLiNERAdapter, FakeGLiNER(DEBERTA))._count_prompt

    with pytest.raises(InvalidInputError, match="at most 20 tokens"):
        adapter.extract([Item(text="Priya Raman works at Novartis.")], labels=["organization", "location"])


def glirel_adapter() -> tuple[GLiRELAdapter, MagicMock]:
    adapter = GLiRELAdapter("fake/glirel")
    adapter._model = MagicMock()
    adapter._model.predict_relations.return_value = []
    adapter._bound_words(char_tokenizer(), 64, DEBERTA)
    from sie_server.adapters.glirel import _prompt_counter

    adapter._count_prompt = _prompt_counter(char_tokenizer(), "[", "]")
    return adapter, adapter._model.predict_relations


def test_glirel_rejects_relation_types_over_the_limit() -> None:
    adapter, predict = glirel_adapter()
    item = Item(
        text="Tim Cook leads Apple",
        metadata={"entities": [{"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8}]},
    )

    adapter.extract([item], labels=["ceo of", "works for"])
    with pytest.raises(InvalidInputError, match="GLiREL"):
        adapter.extract([item], labels=[f"{i}" + "z" * 500 for i in range(3)])
    assert predict.call_count == 1


def test_gliner2_accepts_ordinary_label_sets_and_schemas() -> None:
    adapter, model = make_gliner2_adapter()
    schema = {
        "type": "object",
        "properties": {
            f"field{index}": {"type": "string", "description": "the value of a field on an insurance claim form"}
            for index in range(30)
        },
    }

    adapter.extract([Item(text="Priya Raman works at Novartis")], labels=PII_LABELS)
    adapter._model.batch_extract_json = MagicMock(return_value=[{}])
    adapter.extract([Item(text="Priya Raman works at Novartis")], output_schema=schema)

    assert len(model.inputs) == 1
    adapter._model.batch_extract_json.assert_called_once()


@pytest.mark.parametrize(
    ("labels", "output_schema"),
    [
        ([f"{index:03d}" + "w" * 117 for index in range(60)], None),
        (None, {"type": "object", "properties": {"f": {"type": "string", "description": "d" * 70_000}}}),
        (
            None,
            {
                "type": "object",
                "properties": {
                    f"field{index}": {"type": "string", "enum": [f"choice {index} {option}" for option in range(40)]}
                    for index in range(40)
                },
            },
        ),
    ],
    ids=["long-labels", "long-description", "many-choices"],
)
def test_gliner2_rejects_a_prompt_over_the_limit_before_inference(
    labels: list[str] | None, output_schema: dict[str, Any] | None
) -> None:
    adapter, model = make_gliner2_adapter()
    adapter._model.batch_extract_json = MagicMock(return_value=[{}])

    with pytest.raises(InvalidInputError, match=f"at most {DEFAULT_MAX_SCHEMA_PROMPT_TOKENS} tokens"):
        adapter.extract([Item(text="Priya Raman works at Novartis")], labels=labels, output_schema=output_schema)
    assert model.inputs == []
    adapter._model.batch_extract_json.assert_not_called()


def test_gliner2_rejects_a_classification_prompt_over_the_limit() -> None:
    adapter, model = make_gliner2_adapter(classification_task="topic")

    with pytest.raises(InvalidInputError, match="GLiNER2"):
        adapter.extract([Item(text="Priya Raman works at Novartis")], labels=[f"t{index}" * 40 for index in range(80)])
    assert model.inputs == []


class NoCounting:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a label over the character limit must be rejected before it is tokenized")


def test_gliner_rejects_a_long_label_before_tokenizing_it() -> None:
    model = FakeGLiNER(DEBERTA)
    adapter = load(GLiNERAdapter, model)
    adapter._count_prompt = NoCounting()

    with pytest.raises(InvalidInputError, match=f"at most {MAX_LABEL_CHARS} characters"):
        adapter.extract([Item(text="Priya Raman works at Novartis.")], labels=["q" * (4 * 1024 * 1024)])
    assert model.calls == []


def test_gliner_rejects_long_relation_types() -> None:
    adapter = load(GLiNERAdapter, FakeGLiNER(DEBERTA))
    adapter._extracts_relations = True

    with pytest.raises(InvalidInputError, match="relation_labels may have at most"):
        adapter.extract(
            [Item(text="Priya Raman works at Novartis.")],
            labels=["person"],
            options={"relation_labels": ["r" * (MAX_LABEL_CHARS + 1)]},
        )


@pytest.mark.parametrize(
    ("labels", "output_schema", "match"),
    [
        (["x" * (MAX_LABEL_CHARS + 1)], None, "labels may have at most"),
        (None, {"type": "object", "properties": {"n" * (MAX_LABEL_CHARS + 1): {"type": "string"}}}, "property names"),
        (
            None,
            {"type": "object", "properties": {"f": {"type": "string", "enum": ["c" * (MAX_LABEL_CHARS + 1)]}}},
            "enum values",
        ),
    ],
    ids=["label", "field-name", "choice"],
)
def test_gliner2_rejects_long_labels_field_names_and_choices(
    labels: list[str] | None, output_schema: dict[str, Any] | None, match: str
) -> None:
    adapter, model = make_gliner2_adapter()
    adapter._count_subwords = NoCounting()

    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text="Priya Raman works at Novartis")], labels=labels, output_schema=output_schema)
    assert model.inputs == []


def test_gliner2_classifier_rejects_a_long_task_name_and_long_class_labels() -> None:
    # The GLiGuard route (gliner2 2.x in the transformers5 bundle) uses the same checks.
    adapter, model = make_gliner2_adapter(GLiNER2ClassificationAdapter, classification_task="prompt_safety")

    with pytest.raises(InvalidInputError, match="classification_task may have at most"):
        adapter.extract([Item(text="Hello")], labels=["safe", "unsafe"], options={"classification_task": "t" * 200})
    with pytest.raises(InvalidInputError, match="GLiNER2"):
        adapter.extract([Item(text="Hello")], labels=[f"class {index:03d} " + "k" * 110 for index in range(60)])
    adapter.extract([Item(text="Hello")], labels=["safe", "unsafe"])
    assert len(model.inputs) == 1


def glirel_item(entities: list[Any], text: str = "Tim Cook leads Apple") -> Item:
    return Item(text=text, metadata={"entities": entities})


def test_glirel_caps_the_entities_of_an_item() -> None:
    adapter, predict = glirel_adapter()
    text = "word " * (MAX_ENTITIES + 1)
    entities = [
        {"text": "word", "label": "X", "start": 5 * index, "end": 5 * index + 4} for index in range(MAX_ENTITIES + 1)
    ]

    with pytest.raises(InvalidInputError, match=f"at most {MAX_ENTITIES} entities"):
        adapter.extract([glirel_item(entities, text)], labels=["ceo of"])
    adapter.extract([glirel_item(entities[:MAX_ENTITIES], text)], labels=["ceo of"])
    assert predict.call_count == 1


@pytest.mark.parametrize(
    "entity",
    [
        {"text": "x", "label": "ORG", "start": 5, "end": 2},
        {"text": "x", "label": "ORG", "start": "0", "end": 3},
        {"text": "x", "label": "ORG", "start": 500, "end": 505},
        "Tim Cook",
    ],
    ids=["reversed", "non-integer", "past-the-text", "not-an-object"],
)
def test_glirel_rejects_bad_entities_as_invalid_input_before_any_model_work(entity: Any) -> None:
    adapter, predict = glirel_adapter()
    good = glirel_item([{"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8}])

    with pytest.raises(InvalidInputError):
        adapter.extract([good, glirel_item([entity])], labels=["ceo of"])
    predict.assert_not_called()


def test_glirel_rejects_missing_input_as_invalid_input() -> None:
    adapter, _ = glirel_adapter()

    with pytest.raises(InvalidInputError, match="requires labels"):
        adapter.extract([glirel_item([{"text": "Tim", "label": "P", "start": 0, "end": 3}])], labels=[])
    with pytest.raises(InvalidInputError, match="requires entities"):
        adapter.extract([Item(text="Tim Cook leads Apple")], labels=["ceo of"])
    with pytest.raises(InvalidInputError, match="must be a list"):
        adapter.extract([Item(text="Tim Cook", metadata={"entities": "Tim"})], labels=["ceo of"])
    with pytest.raises(InvalidInputError, match="labels may have at most"):
        adapter.extract([glirel_item([{"text": "Tim", "label": "P", "start": 0, "end": 3}])], labels=["r" * 200])
