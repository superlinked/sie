"""Real HTTP admission for the opt-in entity-only profile, without model IO."""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_sdk.client.errors import RequestError
from sie_server.adapters.gliner2.entity_lora import GLiNER2EntityLoraAdapter
from sie_server.api.extract import router
from sie_server.config.model import ModelConfig
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker import ModelWorker, WorkerConfig


@pytest.mark.parametrize("lora", [None, "a"])
@pytest.mark.parametrize(
    "denied",
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
def test_sdk_http_invalid_task_is_400_without_selection_or_fetch(lora: str | None, denied: str) -> None:
    adapter = GLiNER2EntityLoraAdapter("unloaded", lora_paths={"a": "org/a"})
    adapter.set_active_lora = Mock(side_effect=AssertionError("invalid input reached selection"))
    adapter.fetch_lora = Mock(side_effect=AssertionError("invalid input reached fetch"))
    adapter.load_lora = Mock(side_effect=AssertionError("invalid input reached attach"))
    callbacks = Mock()
    worker = ModelWorker(adapter, WorkerConfig(), model_name="entity", registry_callbacks=callbacks)
    config = ModelConfig.model_validate(
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
    registry = Mock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_config.return_value = config
    registry.get.return_value = adapter
    registry.start_worker = AsyncMock(return_value=worker)
    registry.device = "cpu"

    @asynccontextmanager
    async def lifespan(app):
        await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.state.registry = registry
    item: dict[str, Any] = {"text": "Alice met Bob"}
    options: dict[str, Any] = {"lora": lora}
    kwargs: dict[str, Any] = {}
    if denied == "schema":
        kwargs["output_schema"] = {}
    elif denied == "instruction":
        kwargs["instruction"] = ""
    elif denied == "relation":
        item["metadata"] = {"entities": []}
    elif denied == "legacy":
        options["lora_id"] = "org/a"
    elif denied == "undeclared":
        options["lora"] = "org/not-declared"
    else:
        options[{"classification": "classification_task"}.get(denied, denied)] = "topic"
    statuses = []
    with TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as transport:
        transport.event_hooks["response"].append(lambda response: statuses.append(response.status_code))
        with SIEClient("http://127.0.0.1", http_client=transport) as client:
            with pytest.raises(RequestError) as error:
                client.extract("entity", item, labels=["person"], options=options, **kwargs)
        assert error.value.status_code == 400
    assert statuses == [400]
    adapter.set_active_lora.assert_not_called()
    adapter.fetch_lora.assert_not_called()
    adapter.load_lora.assert_not_called()
    callbacks.request_lora_load.assert_not_called()
