"""Extraction admission must precede LoRA selection and load-on-miss."""

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest
from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.model_worker import PreformedExtractRequest
from sie_server.types.inputs import InvalidInputError, Item


class PlainExtractAdapter(BaseAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("json",))

    def __init__(self) -> None:
        self.selections: list[str | None] = []
        self.served: list[str] = []

    def load(self, device: str) -> None:
        pass

    def supports_lora(self) -> bool:
        return True

    def set_active_lora(self, name: str | None) -> None:
        self.selections.append(name)
        if name == "cold":
            raise ValueError("declared adapter is not resident")

    def extract(self, items: list[Item], **kwargs: Any) -> ExtractOutput:
        self.served.extend(item.text or "" for item in items)
        return ExtractOutput(entities=[[] for _ in items])


class RestrictedExtractAdapter(PlainExtractAdapter):
    def validate_extract_request(self, items: list[Item], **kwargs: Any) -> None:
        if kwargs.get("output_schema") is not None or (kwargs.get("options") or {}).get("classification_task"):
            raise InvalidInputError("entity-only profile")
        if any(item.metadata and "entities" in item.metadata for item in items):
            raise InvalidInputError("entity-only profile")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["submit_extract", "submit_extract_preformed", "submit_extract_preformed_batch"])
@pytest.mark.parametrize("lora", [None, "cold"])
@pytest.mark.parametrize("denial", ["classification", "schema", "relation"])
async def test_denied_requests_never_select_or_request_lora(method: str, lora: str | None, denial: str) -> None:
    adapter = RestrictedExtractAdapter()
    callbacks = Mock()
    worker = ModelWorker(adapter, WorkerConfig(), model_name="entity", registry_callbacks=callbacks)
    items = [Item(text="Alice", metadata={"entities": []} if denial == "relation" else None)]
    prepared = build_extract_prepared_items(items)
    options: dict[str, Any] = {"lora": lora}
    if denial == "classification":
        options["classification_task"] = "topic"
    schema = {} if denial == "schema" else None
    await worker.start()
    try:
        if method == "submit_extract_preformed_batch":
            futures = await worker.submit_extract_preformed_batch(
                [PreformedExtractRequest(prepared, items, output_schema=schema, options=options)], lora=lora
            )
            with pytest.raises(InvalidInputError, match="entity-only"):
                await futures[0]
        else:
            with pytest.raises(InvalidInputError, match="entity-only"):
                await getattr(worker, method)(prepared, items, output_schema=schema, options=options)
        assert adapter.selections == []
        assert adapter.served == []
        callbacks.request_lora_load.assert_not_called()
        assert worker._batchers[None].pending_count == 0
        assert len(worker._batchers) == 1
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_mixed_preformed_admission_preserves_future_order_and_valid_requests() -> None:
    adapter = RestrictedExtractAdapter()
    worker = ModelWorker(adapter, WorkerConfig())
    requests = []
    for index, denied in enumerate([False, True, False, True]):
        items = [Item(text=f"item-{index}")]
        requests.append(
            PreformedExtractRequest(
                build_extract_prepared_items(items),
                items,
                options={"classification_task": "topic"} if denied else None,
            )
        )
    await worker.start()
    try:
        futures = await worker.submit_extract_preformed_batch(requests, lora=None)
        outcomes = await asyncio.gather(*futures, return_exceptions=True)
        assert len(outcomes) == 4
        assert [isinstance(outcome, InvalidInputError) for outcome in outcomes] == [False, True, False, True]
        assert adapter.served == ["item-0", "item-2"]
        assert adapter.selections == [None]
    finally:
        await worker.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["submit_extract", "submit_extract_preformed", "submit_extract_preformed_batch"])
async def test_ordinary_adapter_keeps_its_original_multitask_options(method: str) -> None:
    adapter = PlainExtractAdapter()
    worker = ModelWorker(adapter, WorkerConfig(max_batch_wait_ms=1))
    items = [Item(text="original")]
    prepared = build_extract_prepared_items(items)
    options = {"classification_task": "topic"}
    assert worker._extract_admission is None
    await worker.start()
    try:
        if method == "submit_extract_preformed_batch":
            futures = await worker.submit_extract_preformed_batch(
                [PreformedExtractRequest(prepared, items, options=options)], lora=None
            )
            await asyncio.gather(*futures)
        else:
            future = await getattr(worker, method)(prepared, items, options=options)
            await asyncio.wait_for(future, 5)
        assert adapter.served == ["original"]
        assert adapter.selections == [None]
    finally:
        await worker.stop()
