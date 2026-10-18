from __future__ import annotations

from threading import RLock
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call

import pytest
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.gliner2.classification import GLiNER2ClassificationAdapter
from sie_server.adapters.gliner2.entity_lora import GLiNER2EntityLoraAdapter
from sie_server.types.inputs import InvalidInputError, Item

_TEXT = "Alice Acme"
_ENTITIES = [{"text": "Alice", "start": 0, "end": 5}, {"text": "Acme", "start": 6, "end": 10}]
_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def _adapter(*, max_prompt_tokens: int = 2048) -> GLiNER2Adapter:
    adapter = GLiNER2Adapter("test", max_seq_length=8, max_prompt_tokens=max_prompt_tokens)
    adapter._model = MagicMock()
    cast("Any", adapter)._read = lambda text: (text, 4, None)
    adapter._model._parse_field_spec.return_value = ("name", "str", None, None)
    adapter._count_prompt = MagicMock(return_value=8)
    return adapter


def _forbid_document_work(monkeypatch: pytest.MonkeyPatch, adapter: GLiNER2Adapter) -> list[MagicMock]:
    guards = []
    for name in ("_read", "_doc_input_token_counts", "_classification_windows", "_windowed_input_token_counts"):
        guard = MagicMock(side_effect=AssertionError(f"Rejected prompt reached {name}"))
        monkeypatch.setattr(adapter, name, guard)
        guards.append(guard)
    return guards


def _request(task: str) -> tuple[list[Item], dict[str, Any]]:
    if task == "structured":
        return [Item(text=_TEXT)], {"output_schema": _SCHEMA}
    if task == "relations":
        return [Item(text=_TEXT, metadata={"entities": _ENTITIES})], {"labels": ["knows"]}
    if task == "classification":
        return [Item(text=_TEXT)], {"labels": ["yes", "no"], "options": {"classification_task": "topic"}}
    return [Item(text=_TEXT)], {"labels": ["person"]}


@pytest.mark.parametrize(
    ("kwargs", "metadata", "match"),
    [
        ({}, None, "requires labels"),
        ({"labels": ["person", " person "]}, None, "unique"),
        ({"labels": ["x" * 129]}, None, "labels may have at most"),
        ({"output_schema": {"type": "array"}}, None, "root type"),
        ({"output_schema": {"type": "object", "properties": {}}}, None, "non-empty properties"),
        ({"output_schema": {"type": "object", "properties": {"x" * 129: {"type": "string"}}}}, None, "property names"),
        (
            {"output_schema": {"type": "object", "properties": {"name": {"type": "string", "enum": ["x" * 129]}}}},
            None,
            "enum values",
        ),
        ({"output_schema": _SCHEMA, "labels": ["person"]}, None, "does not accept labels"),
        (
            {"output_schema": _SCHEMA, "options": {"classification_task": ""}},
            None,
            "does not accept classification_task",
        ),
        ({"labels": ["knows"]}, {"entities": "Alice"}, "metadata.entities must be a list"),
        ({"labels": ["knows"]}, {"entities": []}, "non-empty entities"),
        ({"labels": ["knows"]}, {"entities": [{"text": "Alice", "start": 0, "end": 99}]}, "valid character offsets"),
        ({"labels": ["knows"], "options": {"classification_task": ""}}, {"entities": "Alice"}, "metadata.entities"),
        ({"labels": ["yes"], "options": {"classification_task": ""}}, None, "non-empty string"),
        ({"labels": ["yes"], "options": {"classification_task": "x" * 129}}, None, "classification_task may have"),
        (
            {"labels": ["yes"], "options": {"classification_task": "topic", "positive_label": "missing"}},
            None,
            "positive_label",
        ),
    ],
)
def test_invalid_task_prompt_never_reads_or_meters_documents(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], metadata: dict[str, Any] | None, match: str
) -> None:
    adapter = _adapter()
    guards = _forbid_document_work(monkeypatch, adapter)

    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text=_TEXT, metadata=metadata)], **kwargs)

    for guard in guards:
        guard.assert_not_called()
    assert adapter._model.mock_calls == []


@pytest.mark.parametrize("task", ["entities", "relations", "structured", "classification"])
@pytest.mark.parametrize(("option", "value"), [("threshold", float("nan")), ("multi_label", "yes")])
def test_invalid_shared_options_never_read_or_meter_documents(
    monkeypatch: pytest.MonkeyPatch, task: str, option: str, value: Any
) -> None:
    adapter = _adapter()
    guards = _forbid_document_work(monkeypatch, adapter)
    items, kwargs = _request(task)
    kwargs.setdefault("options", {})[option] = value

    with pytest.raises(InvalidInputError, match=option):
        adapter.extract(items, **kwargs)

    for guard in guards:
        guard.assert_not_called()


@pytest.mark.parametrize("task", ["entities", "relations", "structured", "classification"])
def test_token_limit_checks_only_the_prompt_and_reuses_its_cached_rejection(
    monkeypatch: pytest.MonkeyPatch, task: str
) -> None:
    adapter = _adapter(max_prompt_tokens=16)
    count = MagicMock(return_value=100)
    adapter._count_prompt = count
    guards = _forbid_document_work(monkeypatch, adapter)
    items, kwargs = _request(task)

    for _ in range(2):
        with pytest.raises(InvalidInputError, match="at most 16 tokens"):
            adapter.extract(items, **kwargs)

    count.assert_called_once()
    assert _TEXT not in repr(count.call_args)
    for guard in guards:
        guard.assert_not_called()


@pytest.mark.parametrize(
    ("text", "kwargs", "match"),
    [
        (None, {"options": {"threshold": False}}, "requires text"),
        (_TEXT, {"options": {"threshold": False, "multi_label": "bad"}}, "threshold"),
        (_TEXT, {"labels": [], "output_schema": {"type": "array"}, "options": {"multi_label": "bad"}}, "multi_label"),
        (_TEXT, {"labels": ["person"], "output_schema": {"type": "array"}}, "does not accept labels"),
    ],
)
def test_existing_text_option_and_task_error_precedence(
    monkeypatch: pytest.MonkeyPatch, text: str | None, kwargs: dict[str, Any], match: str
) -> None:
    adapter = _adapter()
    _forbid_document_work(monkeypatch, adapter)

    with pytest.raises(InvalidInputError, match=match):
        adapter.extract([Item(text=text)], **kwargs)


def _responding_model() -> MagicMock:
    model = MagicMock()
    model._parse_field_spec.return_value = ("name", "str", None, None)
    model.processor.tokenizer.side_effect = lambda texts, **_kwargs: {
        "input_ids": [[1] * (len(text.split()) + 2) for text in texts]
    }
    model.extract_entities.return_value = {
        "entities": {"person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.9}]}
    }
    model.batch_extract_json.return_value = [{"_sie_root": [{"name": "Alice"}]}]
    model.batch_extract_relations.return_value = [
        {
            "relation_extraction": {
                "knows": [{"head": {"text": "Alice", "confidence": 0.9}, "tail": {"text": "Acme", "confidence": 0.8}}]
            }
        }
    ]
    model.classify_text.return_value = {"topic": {"label": "yes", "confidence": 0.9}}
    return model


@pytest.mark.parametrize("task", ["entities", "relations", "structured", "classification"])
def test_accepted_task_routing_output_and_usage_are_preserved(task: str) -> None:
    adapter = _adapter()
    model = _responding_model()
    adapter._model = model
    items, kwargs = _request(task)
    # These options are intentionally validated only on the classification path.
    if task != "classification":
        kwargs["options"] = {"positive_label": ""}
    if task == "relations":
        kwargs["options"]["classification_task"] = ""
    if task == "structured":
        items[0].metadata = {"entities": "ignored by structured routing"}

    output = adapter.extract(items, **kwargs)

    assert output.input_token_counts == [4]
    assert output.errors is None
    if task == "entities":
        assert output.entities == [[{"text": "Alice", "label": "person", "score": 0.9, "start": 0, "end": 5}]]
        model.extract_entities.assert_called_once_with(
            _TEXT, ["person"], threshold=0.5, include_confidence=True, include_spans=True, max_len=8
        )
    elif task == "relations":
        assert output.entities == [
            [
                {"text": "Alice", "label": "ENTITY", "score": 1.0, "start": 0, "end": 5},
                {"text": "Acme", "label": "ENTITY", "score": 1.0, "start": 6, "end": 10},
            ]
        ]
        assert output.relations == [[{"head": "Alice", "tail": "Acme", "relation": "knows", "score": 0.8}]]
        model.batch_extract_relations.assert_called_once_with(
            [_TEXT], ["knows"], batch_size=1, threshold=0.5, include_confidence=True, include_spans=True, max_len=8
        )
        model.classify_text.assert_not_called()
    elif task == "structured":
        assert output.entities == [[]]
        assert output.data == [{"name": "Alice"}]
        model.batch_extract_json.assert_called_once_with(
            [_TEXT],
            {"_sie_root": ["name::str"]},
            batch_size=1,
            threshold=0.5,
            include_confidence=False,
            include_spans=False,
            max_len=8,
        )
    else:
        assert output.entities == [[]]
        assert output.classifications == [[{"label": "yes", "score": 0.9}]]
        model.classify_text.assert_called_once_with(
            _TEXT,
            {"topic": {"labels": ["yes", "no"], "multi_label": False, "cls_threshold": 0.5}},
            threshold=0.5,
            include_confidence=True,
            max_len=8,
        )


@pytest.mark.parametrize("reject", [False, True])
def test_classification_subclass_preserves_defaults_and_early_prompt_validation(
    monkeypatch: pytest.MonkeyPatch, reject: bool
) -> None:
    adapter = GLiNER2ClassificationAdapter(
        "test", classification_task="topic", default_labels=["yes", "no"], max_seq_length=8, max_prompt_tokens=16
    )
    adapter._model = _responding_model()
    cast("Any", adapter)._read = lambda text: (text, 4, None)
    adapter._count_prompt = lambda _schema: 8
    if reject:
        adapter._count_prompt = lambda _schema: 100
        guards = _forbid_document_work(monkeypatch, adapter)
        with pytest.raises(InvalidInputError, match="at most 16 tokens"):
            adapter.extract([Item(text=_TEXT)])
        for guard in guards:
            guard.assert_not_called()
    else:
        output = adapter.extract([Item(text=_TEXT)])
        assert output.classifications == [[{"label": "yes", "score": 0.9}]]
        assert output.input_token_counts == [4]


@pytest.mark.parametrize("reject", [False, True])
def test_entity_lora_subclass_keeps_selection_and_final_base_restoration(
    monkeypatch: pytest.MonkeyPatch, reject: bool
) -> None:
    # Exercise the real guard/super/finally seam without loading optional weights.
    adapter = GLiNER2EntityLoraAdapter.__new__(GLiNER2EntityLoraAdapter)
    GLiNER2Adapter.__init__(adapter, "test", max_seq_length=8, max_prompt_tokens=16)
    controller = SimpleNamespace(lock=RLock(), select=MagicMock(), source=MagicMock(return_value="org/a"))
    monkeypatch.setattr(adapter, "_lora", controller, raising=False)
    adapter._model = _responding_model()
    cast("Any", adapter)._read = lambda text: (text, 4, None)
    adapter._count_prompt = lambda _schema: 8
    if reject:
        adapter._count_prompt = lambda _schema: 100
        guards = _forbid_document_work(monkeypatch, adapter)
        with pytest.raises(InvalidInputError, match="at most 16 tokens"):
            adapter.extract([Item(text=_TEXT)], labels=["person"], options={"lora": "a"})
        for guard in guards:
            guard.assert_not_called()
    else:
        output = adapter.extract([Item(text=_TEXT)], labels=["person"], options={"lora": "a"})
        assert output.entities == [[{"text": "Alice", "label": "person", "score": 0.9, "start": 0, "end": 5}]]
        assert output.input_token_counts == [4]
    controller.source.assert_called_once_with("a")
    assert controller.select.call_args_list == [call("a"), call(None)]
