from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.adapters._word_window import WindowedSplitter
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.gliner2.words import LinearWordSplitter
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.types.inputs import InvalidInputError, Item


class _Tokenizer:
    def tokenize(self, text: str) -> list[str]:
        return [text] * max(1, math.ceil(len(text) / 3))

    def __call__(self, texts: list[str], *, max_length: int | None, **_: Any) -> dict[str, Any]:
        return {"input_ids": [list(range(min(len(text.split()) + 2, max_length or 512))) for text in texts]}


def _adapter(*, lower: bool = True, max_words: int = 8, max_subwords: int = 1000) -> tuple[GLiNER2Adapter, MagicMock]:
    adapter = GLiNER2Adapter("test", max_seq_length=max_words)
    model = MagicMock()
    model.processor.tokenizer = _Tokenizer()
    adapter._model = model
    adapter._lower_text_first = lower
    adapter._word_splitter = WindowedSplitter(
        LinearWordSplitter(lower_text_first=lower),
        max_words=max_words,
        max_subwords=max_subwords,
        count_subwords=lambda words: [len(model.processor.tokenizer.tokenize(word)) for word in words],
    )
    model.extract_entities.side_effect = lambda text, *_a, **_k: {
        "entities": {"person": [{"text": text.split()[0], "start": 0, "end": len(text.split()[0]), "confidence": 0.9}]}
    }
    model.batch_extract_entities.side_effect = lambda texts, *_a, **_k: [model.extract_entities(text) for text in texts]
    model.batch_extract_json.side_effect = lambda texts, *_a, **_k: [
        {"_sie_root": [{"name": text.split()[0]}]} for text in texts
    ]
    model.batch_extract_relations.side_effect = lambda texts, *_a, **_k: [
        {
            "relation_extraction": {
                "knows": [
                    {"head": {"text": text.split()[0], "confidence": 0.9}, "tail": {"text": "Acme", "confidence": 0.8}}
                ]
            }
        }
        for text in texts
    ]
    return adapter, model


def _request(task: str, texts: list[str]) -> tuple[list[Item], dict[str, Any]]:
    items = [Item(text=text) for text in texts]
    kwargs: dict[str, Any] = {"labels": ["person"]}
    if task == "structured":
        kwargs = {"output_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}
    elif task == "relations":
        kwargs = {"labels": ["knows"]}
        items = [
            Item(
                text=text,
                metadata={
                    "entities": [
                        {"text": text.split()[0], "start": 0, "end": len(text.split()[0])},
                        {"text": "Acme", "start": text.index("Acme"), "end": text.index("Acme") + 4},
                    ]
                },
            )
            for text in texts
        ]
    return items, kwargs


@pytest.mark.parametrize("task", ["entities", "relations", "structured"])
def test_overlong_items_fail_without_losing_short_items_or_their_positions(task: str) -> None:
    adapter, model = _adapter()
    long = "Alice Acme " + "word " * 20
    items, kwargs = _request(task, ["Alice Acme", long, "Bob Acme", long])
    output = adapter.extract(items, **kwargs)

    assert output.errors is not None
    assert [error.code if error else None for error in output.errors] == [
        None,
        "INPUT_TOO_LONG",
        None,
        "INPUT_TOO_LONG",
    ]
    assert output.input_token_counts == [4, 0, 4, 0]
    assert output.entities[1] == output.entities[3] == []
    if task == "entities":
        assert [output.entities[i][0]["text"] for i in [0, 2]] == ["Alice", "Bob"]
        method = model.batch_extract_entities
    elif task == "structured":
        assert output.data == [{"name": "Alice"}, {}, {"name": "Bob"}, {}]
        method = model.batch_extract_json
    else:
        assert output.relations is not None
        assert [relations[0]["head"] if relations else None for relations in output.relations] == [
            "Alice",
            None,
            "Bob",
            None,
        ]
        method = model.batch_extract_relations
    assert method.call_args.args[0] == ["Alice Acme", "Bob Acme"]

    handler = ExtractHandler()
    assembled = handler.assemble_output({i: handler.slice_output(output, i) for i in range(4)}, batch_size=4)
    assert assembled.input_token_counts == [4, 0, 4, 0]
    formatted = handler.format_output(assembled)
    assert "error" not in formatted[0]
    assert "error" not in formatted[2]
    assert formatted[1]["error"]["code"] == formatted[3]["error"]["code"] == "INPUT_TOO_LONG"


@pytest.mark.parametrize("task", ["entities", "relations", "structured"])
def test_an_all_overlong_batch_never_calls_the_model(task: str) -> None:
    adapter, model = _adapter()
    items, kwargs = _request(task, ["Alice Acme " + "word " * 20] * 2)
    output = adapter.extract(items, **kwargs)
    assert output.errors is not None
    assert all(error and error.code == "INPUT_TOO_LONG" for error in output.errors)
    assert output.input_token_counts == [0, 0]
    assert not model.extract_entities.called
    assert not model.batch_extract_entities.called
    assert not model.batch_extract_relations.called
    assert not model.batch_extract_json.called


@pytest.mark.parametrize("lower", [False, True])
@pytest.mark.parametrize(
    ("text", "max_words", "max_subwords", "too_long"),
    [
        ("one two three", 3, 1000, False),  # only the package's synthetic dot is unread
        ("one two three   \t\n", 3, 1000, False),
        ("one two three.", 3, 1000, True),  # the caller's punctuation is real input
        ("one two three four", 3, 1000, True),
        ("\u0130stanbul Ankara Izmir", 2, 1000, True),  # lowercase expands offsets
        ("\u039f\u0394\u039f\u03a3'\u0391 next", 1, 1000, True),
        ("x" * 513, 2, 1000, True),  # cut is the original word end, beyond the last read piece
        ("abc " * 7, 8, 3, True),  # fewer than max_words, but the subword budget is full
    ],
)
def test_completeness_uses_the_actual_word_window(
    text: str, max_words: int, max_subwords: int, too_long: bool, lower: bool
) -> None:
    adapter, model = _adapter(lower=lower, max_words=max_words, max_subwords=max_subwords)
    model.extract_entities.side_effect = None
    model.extract_entities.return_value = {"entities": {}}
    output = adapter.extract([Item(text=text)], labels=["person"])
    if too_long:
        assert output.errors is not None
        assert output.errors[0] is not None
        assert output.errors[0].code == "INPUT_TOO_LONG"
        assert output.input_token_counts == [0]
        model.extract_entities.assert_not_called()
    else:
        assert output.errors is None
        model.extract_entities.assert_called_once()


def test_invalid_request_parameters_are_still_validated_before_item_length_errors() -> None:
    adapter, model = _adapter()
    with pytest.raises(InvalidInputError, match="requires labels"):
        adapter.extract([Item(text="word " * 20)])
    model.extract_entities.assert_not_called()


@pytest.mark.parametrize("index", [0, 1])
def test_queue_outcome_preserves_length_error_and_zero_billing(index: int) -> None:
    from types import SimpleNamespace

    import msgpack
    from sie_server.ipc_types import ExtractBatchItem
    from sie_server.queue_executor import _extract_success_outcome

    adapter, _ = _adapter()
    items = [Item(text="word " * 20), Item(text="Alice Acme")]
    output = adapter.extract(items, labels=["person"])
    worker_result = SimpleNamespace(
        output=ExtractHandler().slice_output(output, index),
        timing=SimpleNamespace(inference_ms=0, tokenization_ms=0, postprocessing_ms=0),
    )
    batch_item = ExtractBatchItem(
        work_item_id=f"req.{index}",
        request_id="req",
        item_index=index,
        total_items=2,
        timestamp=0,
        item={"text": items[index].text},
        labels=["person"],
    )
    outcome = _extract_success_outcome(adapter, batch_item, items[index], worker_result)
    result = msgpack.unpackb(outcome.result_msgpack, raw=False)
    assert outcome.units is not None
    if index == 0:
        assert result["error"]["code"] == "INPUT_TOO_LONG"
        assert result["entities"] == []
        assert outcome.units.input_tokens == 0
    else:
        assert "error" not in result
        assert result["entities"][0]["text"] == "Alice"
        assert outcome.units.input_tokens == 4


def test_queue_length_error_keeps_zero_billing_when_sibling_metering_fails() -> None:
    from types import SimpleNamespace

    import msgpack
    from sie_server.ipc_types import ExtractBatchItem
    from sie_server.queue_executor import _extract_success_outcome

    adapter, _ = _adapter()
    adapter._doc_input_token_counts = lambda _texts: None
    items = [Item(text="word " * 20), Item(text="Alice Acme")]
    output = adapter.extract(items, labels=["person"])
    assert output.input_token_counts is None
    assert output.errors is not None
    assert output.errors[0] is not None

    batch_item = ExtractBatchItem(
        work_item_id="req.0",
        request_id="req",
        item_index=0,
        total_items=2,
        timestamp=0,
        item={"text": items[0].text},
        labels=["person"],
    )
    worker_result = SimpleNamespace(
        output=ExtractHandler().slice_output(output, 0),
        timing=SimpleNamespace(inference_ms=0, tokenization_ms=0, postprocessing_ms=0),
    )
    outcome = _extract_success_outcome(adapter, batch_item, items[0], worker_result)
    result = msgpack.unpackb(outcome.result_msgpack, raw=False)

    assert result["error"]["code"] == "INPUT_TOO_LONG"
    assert outcome.units is not None
    assert outcome.units.input_tokens == 0


@pytest.mark.parametrize(("error_code", "expected_input_tokens"), [("INPUT_TOO_LONG", 0), ("INFERENCE_ERROR", 7)])
def test_queue_length_error_overrides_reported_positive_billing(error_code: str, expected_input_tokens: int) -> None:
    from types import SimpleNamespace

    import msgpack
    from sie_server.ipc_types import ExtractBatchItem
    from sie_server.queue_executor import _extract_success_outcome

    adapter, _ = _adapter()
    items = [Item(text="word " * 20), Item(text="Alice Acme")]
    output = adapter.extract(items, labels=["person"])
    output.input_token_counts = [7, 4]
    assert output.errors is not None
    assert output.errors[0] is not None
    output.errors[0].code = error_code

    batch_item = ExtractBatchItem(
        work_item_id="req.0",
        request_id="req",
        item_index=0,
        total_items=2,
        timestamp=0,
        item={"text": items[0].text},
        labels=["person"],
    )
    worker_result = SimpleNamespace(
        output=ExtractHandler().slice_output(output, 0),
        timing=SimpleNamespace(inference_ms=0, tokenization_ms=0, postprocessing_ms=0),
    )
    outcome = _extract_success_outcome(adapter, batch_item, items[0], worker_result)
    result = msgpack.unpackb(outcome.result_msgpack, raw=False)

    assert result["error"]["code"] == error_code
    assert outcome.units is not None
    assert outcome.units.input_tokens == expected_input_tokens
