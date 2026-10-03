from typing import Any
from unittest.mock import MagicMock

import pytest
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.ipc_types import ExtractBatchItem
from sie_server.queue_executor import _inference_exception_outcome
from sie_server.types.inputs import Item


@pytest.mark.parametrize(
    ("labels", "output_schema", "options", "metadata"),
    [
        (None, None, None, None),
        ([], None, None, None),
        (["PERSON", " PERSON "], None, None, None),
        ([""], None, None, None),
        (["PERSON"], None, {"threshold": float("nan")}, None),
        (["PERSON"], None, {"threshold": True}, None),
        (["PERSON"], None, {"multi_label": "yes"}, None),
        (["PERSON"], None, {"classification_task": ""}, None),
        (["PERSON"], None, {"classification_task": "task", "positive_label": ""}, None),
        (["PERSON"], None, {"classification_task": "task", "positive_label": "absent"}, None),
        (None, {"type": "array"}, None, None),
        (None, {"type": "object", "properties": {}}, None, None),
        (None, {"type": "object", "properties": {"age": {"type": "number"}}}, None, None),
        (["PERSON"], {"type": "object", "properties": {"name": {"type": "string"}}}, None, None),
        (["WORKS_AT"], None, None, {"entities": "Alice"}),
        (["WORKS_AT"], None, None, {"entities": []}),
        (["WORKS_AT"], None, None, {"entities": ["Alice"]}),
        (["WORKS_AT"], None, None, {"entities": [{"text": "Alice", "start": 0, "end": 99}]}),
        (["WORKS_AT"], None, None, {"entities": [{"text": "Alice", "start": 0, "end": 5, "score": "bad"}]}),
    ],
)
def test_invalid_request_is_invalid_input_on_the_queue_path(
    labels: list[str] | None,
    output_schema: dict[str, Any] | None,
    options: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> None:
    adapter = GLiNER2Adapter("test-model")
    adapter._model = MagicMock()
    item = Item(text="Alice works at Acme.", metadata=metadata)

    with pytest.raises(ValueError, match="GLiNER2") as error:
        adapter.extract([item], labels=labels, output_schema=output_schema, options=options)

    batch_item = ExtractBatchItem(
        work_item_id="req.0",
        request_id="req",
        item_index=0,
        total_items=1,
        timestamp=0,
        item={"text": item.text},
        labels=labels,
        output_schema=output_schema,
        options=options,
    )
    outcome = _inference_exception_outcome(batch_item, error.value)

    assert outcome.disposition == "publish_error_and_ack"
    assert outcome.error_code == "INVALID_INPUT"
    assert outcome.error == str(error.value)
    adapter._model.extract_entities.assert_not_called()
    adapter._model.batch_extract_entities.assert_not_called()
    adapter._model.batch_extract_relations.assert_not_called()
    adapter._model.batch_extract_json.assert_not_called()
    adapter._model.classify_text.assert_not_called()


def test_invalid_model_result_remains_an_inference_error_on_the_queue_path() -> None:
    adapter = GLiNER2Adapter("test-model")
    adapter._model = MagicMock()
    adapter._model.extract_entities.return_value = {
        "entities": {"PERSON": [{"text": "Alice", "start": 0, "end": 5, "confidence": "bad"}]},
    }

    with pytest.raises(ValueError, match="invalid entity confidence") as error:
        adapter.extract([Item(text="Alice")], labels=["PERSON"])

    batch_item = ExtractBatchItem(
        work_item_id="req.0",
        request_id="req",
        item_index=0,
        total_items=1,
        timestamp=0,
        item={"text": "Alice"},
        labels=["PERSON"],
    )
    outcome = _inference_exception_outcome(batch_item, error.value)

    assert outcome.disposition == "publish_error_and_ack"
    assert outcome.error_code == "inference_error"
