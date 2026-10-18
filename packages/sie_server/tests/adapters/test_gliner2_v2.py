"""GLiNER2 output compatibility, including per-item structured failures."""

from contextlib import asynccontextmanager
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import msgpack
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gliner2 import GLiNER2
from sie_sdk import SIEClient
from sie_server.adapters._word_window import WindowedSplitter
from sie_server.adapters.gliner2.adapter import GLiNER2Adapter
from sie_server.adapters.gliner2.words import LinearWordSplitter
from sie_server.api.extract import router
from sie_server.config.model import ModelConfig
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.ipc_types import ExtractBatchItem
from sie_server.queue_executor import _extract_success_outcome
from sie_server.types.inputs import InvalidInputError, Item

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "status": {"type": "string", "enum": ["active", "inactive"]},
    },
    "required": ["name"],
    "additionalProperties": False,
}
ERROR = "GLiNER2 returned structured output that does not match output_schema"


def adapter_with_results(results: list[Any]) -> tuple[GLiNER2Adapter, MagicMock]:
    adapter = GLiNER2Adapter("generated", max_seq_length=512)
    model = MagicMock()
    model._parse_field_spec.return_value = ("name", "str", None, None)
    model.batch_extract_json.return_value = results
    adapter._model = model
    adapter._count_prompt = lambda _schema: 8
    cast("Any", adapter)._read = lambda text: (text, len(text), None)
    return adapter, model


@pytest.mark.parametrize("field", ["tags", "status"])
def test_known_optional_null_is_omitted_without_mutating_model_result(field):
    raw = {"_sie_root": [{"name": "Alice", field: None}]}
    adapter, model = adapter_with_results([raw])
    result = adapter.extract([Item(text="Alice")], output_schema=SCHEMA)
    assert result.data == [{"name": "Alice"}]
    assert result.errors is None
    assert result.input_token_counts == [5]
    assert raw == {"_sie_root": [{"name": "Alice", field: None}]}
    model.batch_extract_json.assert_called_once()


@pytest.mark.parametrize("additional", [True, False])
def test_unknown_null_is_not_silently_dropped(additional):
    schema = {**SCHEMA, "additionalProperties": additional}
    adapter, _ = adapter_with_results([{"_sie_root": [{"name": "Alice", "unknown": None}]}])
    result = adapter.extract([Item(text="Alice")], output_schema=schema)
    if additional:
        assert result.data == [{"name": "Alice", "unknown": None}]
        assert result.errors is None
        assert result.input_token_counts == [5]
    else:
        assert result.data == [{}]
        assert result.errors is not None
        assert result.errors[0] is not None
        assert result.errors[0].message == ERROR
        assert result.input_token_counts == [0]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "private output",
        {"_sie_root": [None]},
        {"_sie_root": [{"name": None}]},
        {"_sie_root": [{"name": ["Alice"]}]},
        {"_sie_root": [{"name": "Alice", "tags": [None]}]},
        {"_sie_root": [{"name": "Alice", "status": "private unknown value"}]},
        {"_sie_root": [{"name": "Alice", "private unknown field": None}]},
        {"_sie_root": [{"name": "Alice", 7: None}]},
        {"_sie_root": [{"name": "Alice"}, {"name": "Bob"}]},
        {"_sie_root": None},
        {},
    ],
)
def test_invalid_structured_item_preserves_successful_sibling_and_queue_billing(bad):
    adapter, model = adapter_with_results([bad, {"_sie_root": [{"name": "Bob", "tags": []}]}])
    items = [Item(text="Alice"), Item(text="Bob")]
    output = adapter.extract(items, output_schema=SCHEMA)
    assert output.entities == [[], []]
    assert output.data == [{}, {"name": "Bob", "tags": []}]
    assert output.input_token_counts == [0, 3]
    assert output.errors is not None
    assert output.errors[0] is not None
    assert output.errors[1] is None
    assert output.errors[0].code == "INFERENCE_ERROR"
    assert output.errors[0].message == ERROR
    model.batch_extract_json.assert_called_once()
    for index, count in enumerate([0, 3]):
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
            output_schema=SCHEMA,
        )
        outcome = _extract_success_outcome(adapter, batch_item, items[index], worker_result)
        decoded = msgpack.unpackb(outcome.result_msgpack, raw=False)
        assert outcome.units is not None
        assert outcome.units.input_tokens == count
        assert decoded["data"] == output.data[index]
        if index == 0:
            assert decoded["error"]["code"] == "INFERENCE_ERROR"
            assert decoded["error"]["message"] == ERROR
        else:
            assert "error" not in decoded


def test_schema_error_precedes_document_and_model_work():
    adapter, model = adapter_with_results([])
    cast("Any", adapter)._read = MagicMock(side_effect=AssertionError("document read"))
    schema = {**SCHEMA, "required": ["undeclared"]}
    with pytest.raises(InvalidInputError, match="required"):
        adapter.extract([Item(text="Alice")], output_schema=schema)
    model.batch_extract_json.assert_not_called()
    cast("Any", adapter)._read.assert_not_called()


def test_structured_failure_preserves_length_rejection_and_valid_sibling():
    adapter, model = adapter_with_results([{"_sie_root": [{"name": None}]}, {"_sie_root": [{"name": "Bob"}]}])
    del cast("Any", adapter)._read
    adapter._lower_text_first = False
    adapter._word_splitter = WindowedSplitter(
        LinearWordSplitter(lower_text_first=False),
        max_words=3,
        max_subwords=100,
        count_subwords=lambda words: [1 for _ in words],
    )
    result = adapter.extract(
        [Item(text=text) for text in ["Alice", "too many words for this window", "Bob"]], output_schema=SCHEMA
    )
    assert result.data == [{}, {}, {"name": "Bob"}]
    assert result.input_token_counts == [0, 0, 2]
    assert result.errors is not None
    assert [error.code if error else None for error in result.errors] == ["INFERENCE_ERROR", "INPUT_TOO_LONG", None]
    assert model.batch_extract_json.call_args.args[0] == ["Alice", "Bob"]


def test_structured_mixed_output_survives_actual_sdk_worker_path():
    adapter, model = adapter_with_results([])
    model.batch_extract_json.side_effect = lambda texts, *_args, **_kwargs: [
        {"_sie_root": [{"name": None if text == "Alice" else text, "tags": None}]} for text in texts
    ]
    worker = ModelWorker(adapter, WorkerConfig(), model_name="structured")
    config = ModelConfig.model_validate(
        {
            "sie_id": "structured",
            "hf_id": "generated",
            "tasks": {"extract": {}},
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.gliner2.adapter:GLiNER2Adapter",
                }
            },
        }
    )
    registry = Mock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_config.return_value = config
    registry.get.return_value = adapter
    registry.start_worker = AsyncMock(return_value=worker)
    registry.device = "cpu"

    @asynccontextmanager
    async def lifespan(_app):
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.state.registry = registry
    statuses = []
    responses = []
    with TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as transport:
        transport.event_hooks["response"].append(lambda response: statuses.append(response.status_code))
        transport.event_hooks["response"].append(responses.append)
        with SIEClient("http://127.0.0.1", http_client=transport) as client:
            results = client.extract(
                "structured", [{"id": "bad", "text": "Alice"}, {"id": "good", "text": "Bob"}], output_schema=SCHEMA
            )
    assert statuses == [200]
    assert [result["id"] for result in results] == ["bad", "good"]
    # The existing SDK omits an empty optional data field; the wire and worker
    # output still carry the explicit empty object.
    wire = msgpack.unpackb(responses[0].content, raw=False)
    assert wire["items"][0]["data"] == {}
    assert results[0].get("data", {}) == {}
    assert results[0]["error"] == {"code": "INFERENCE_ERROR", "message": ERROR}
    assert results[1]["data"] == {"name": "Bob"}
    assert "error" not in results[1]


def test_repeated_ner_mentions_keep_each_distinct_source_offset():
    text = "Alice met Alice."
    adapter, model = adapter_with_results([])
    model.extract_entities.return_value = {
        "entities": {
            "person": [
                {"text": "Alice", "start": 10, "end": 15, "confidence": 0.8},
                {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9},
            ]
        }
    }
    result = adapter.extract([Item(text=text)], labels=["person"])
    assert [(entity["text"], entity["start"], entity["end"]) for entity in result.entities[0]] == [
        ("Alice", 0, 5),
        ("Alice", 10, 15),
    ]
    model.extract_entities.assert_called_once()


def test_real_package_formatter_and_adapter_preserve_available_mention_offsets():
    spans = [
        {"text": "Alice", "start": 0, "end": 5, "confidence": 0.9},
        {"text": "Alice", "start": 10, "end": 15, "confidence": 0.8},
    ]
    formatted = GLiNER2._format_entity_dict(None, {"person": spans}, include_confidence=True)
    expected = spans if version("gliner2") == "2.0.0" else spans[:1]
    assert formatted == {"person": expected}
    result = GLiNER2Adapter._flatten_entities({"entities": formatted}, text="Alice met Alice.")
    assert [(entity["start"], entity["end"]) for entity in result] == [
        (span["start"], span["end"]) for span in expected
    ]
