"""Entity-only admission at the actual queued worker entrypoint."""

import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgpack
import pytest
from sie_server.adapters.gliner2.entity_lora import GLiNER2EntityLoraAdapter
from sie_server.adapters.lora import LoraUnavailableError
from sie_server.config.model import ModelConfig
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.ipc_types import ExtractBatchItem, ProcessExtractBatchRequest
from sie_server.queue_executor import QueueExecutor


def batch_item(name: str, *, lora: str | None, denial: str | None = None) -> ExtractBatchItem:
    options: dict[str, Any] = {"lora": lora}
    item: dict[str, Any] = {"text": name, "id": name}
    kwargs: dict[str, Any] = {}
    if denial == "schema":
        kwargs["output_schema"] = {}
    elif denial == "instruction":
        kwargs["instruction"] = ""
    elif denial == "relation":
        item["metadata"] = {"entities": []}
    elif denial == "legacy":
        options["lora_id"] = "org/a"
    elif denial == "undeclared":
        options["lora"] = "org/not-declared"
    elif denial:
        options[{"classification": "classification_task"}.get(denial, denial)] = "topic"
    return ExtractBatchItem(
        work_item_id=name,
        request_id=name,
        item_index=0,
        total_items=1,
        timestamp=time.time(),
        item=item,
        labels=["person"],
        options=options,
        **kwargs,
    )


def setup():
    adapter = GLiNER2EntityLoraAdapter("unloaded", lora_paths={"a": "org/a"})

    def select(name):
        if name is not None:
            raise LoraUnavailableError("not resident")

    adapter.set_active_lora = Mock(side_effect=select)
    adapter.fetch_lora = Mock(side_effect=AssertionError("invalid input reached fetch"))
    adapter.extract = Mock(side_effect=lambda items, **kwargs: ExtractOutput(entities=[[] for _ in items]))
    callbacks = Mock()
    callbacks.request_lora_load.return_value = 7
    worker = ModelWorker(adapter, WorkerConfig(), model_name="entity", registry_callbacks=callbacks)
    registry = Mock()
    registry.get.return_value = adapter
    registry.get_config.return_value = ModelConfig.model_validate(
        {
            "sie_id": "entity",
            "hf_id": "unloaded",
            "tasks": {"extract": {}},
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter",
                    "adapter_options": {"loadtime": {"lora_paths": {"a": "org/a"}}},
                }
            },
        }
    )
    registry.start_worker = AsyncMock(return_value=worker)
    return adapter, worker, callbacks, QueueExecutor(registry)


@pytest.mark.asyncio
@pytest.mark.parametrize("lora", [None, "a"])
@pytest.mark.parametrize(
    "denial",
    [
        "classification",
        "multi_label",
        "positive_label",
        "task",
        "schema",
        "instruction",
        "relation",
        "legacy",
        "undeclared",
    ],
)
async def test_denied_queue_task_is_invalid_input_without_selection_or_fetch(lora: str | None, denial: str) -> None:
    adapter, worker, callbacks, executor = setup()
    await worker.start()
    try:
        outcome = await executor.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="entity",
                items=[batch_item("denied", lora=lora, denial=denial)],
            )
        )
        assert len(outcome.outcomes) == 1
        item = outcome.outcomes[0]
        assert item.disposition == "publish_error_and_ack"
        assert item.error_code == "INVALID_INPUT"
        adapter.set_active_lora.assert_not_called()
        adapter.fetch_lora.assert_not_called()
        adapter.extract.assert_not_called()
        callbacks.request_lora_load.assert_not_called()
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_mixed_queue_keeps_original_order_and_only_valid_cold_load_request() -> None:
    adapter, worker, callbacks, executor = setup()
    await worker.start()
    try:
        outcome = await executor.process_extract_batch(
            ProcessExtractBatchRequest(
                model_id="entity",
                items=[
                    batch_item("denied-base", lora=None, denial="classification"),
                    batch_item("valid-base", lora=None),
                    batch_item("denied-cold", lora="a", denial="legacy"),
                    batch_item("valid-cold", lora="a"),
                ],
            )
        )
        assert [item.work_item_id for item in outcome.outcomes] == [
            "denied-base",
            "valid-base",
            "denied-cold",
            "valid-cold",
        ]
        assert [item.disposition for item in outcome.outcomes] == [
            "publish_error_and_ack",
            "publish_and_ack",
            "publish_error_and_ack",
            "publish_error_and_ack",
        ]
        assert outcome.outcomes[0].error_code == outcome.outcomes[2].error_code == "INVALID_INPUT"
        assert outcome.outcomes[3].error_code == "LORA_LOADING"
        assert outcome.outcomes[3].retry_after_s == 7
        assert msgpack.unpackb(outcome.outcomes[1].result_msgpack, raw=False)["id"] == "valid-base"
        assert [item.text for item in adapter.extract.call_args.args[0]] == ["valid-base"]
        callbacks.request_lora_load.assert_called_once_with("entity", "a")
        adapter.fetch_lora.assert_not_called()
    finally:
        await worker.stop()
