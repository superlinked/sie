"""Native ingress preserves profile defaults and rejects invalid request selectors."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk._msgpack import packb
from sie_server.api import encode, extract, score
from sie_server.config.model import ModelConfig
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.postprocessor_registry import PostprocessorRegistry
from sie_server.core.registry import ModelRegistry


@pytest.mark.parametrize("endpoint", ["encode", "extract", "score"])
@pytest.mark.parametrize("wire", ["json", "msgpack"])
@pytest.mark.parametrize(
    "options",
    [
        {"lora_adapter": None, "lora": "org/legal"},
        {"lora_adapter": "org/legal", "lora_id": "other"},
        {"lora": "legal", "lora_id": "org/legal"},
        {"lora_adapter": []},
        {"lora_adapter": " "},
        {"lora_id": "legal"},
        {"lora_adapter": "org/undeclared"},
    ],
)
def test_bad_lora_options_stop_before_routing_loading_or_worker(endpoint, wire, options, monkeypatch):
    config = ModelConfig.model_validate(
        {
            "sie_id": "m",
            "hf_id": "unloaded",
            "inputs": {"text": True},
            "tasks": {"encode": {"dense": {"dim": 8}}, "extract": {}, "score": {}},
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                    "adapter_options": {"loadtime": {"lora_paths": {"legal": "org/legal"}}},
                }
            },
        }
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_config.return_value = config
    registry.device = "cpu"
    app = FastAPI()
    module = {"encode": encode, "extract": extract, "score": score}[endpoint]
    app.include_router(module.router)
    app.state.registry = registry
    route = AsyncMock(side_effect=AssertionError("bad selection reached route"))
    monkeypatch.setattr(module, "route_request", route)
    body = {"items": [{"text": "x"}], "params": {"labels": ["label"], "options": options}}
    if endpoint == "score":
        body = {"query": {"text": "q"}, "items": [{"text": "x"}], "options": options}
    with TestClient(app) as client:
        if wire == "json":
            response = client.post(f"/v1/{endpoint}/m", json=body)
        else:
            response = client.post(
                f"/v1/{endpoint}/m", content=packb(body), headers={"content-type": "application/msgpack"}
            )
    assert response.status_code == 400, response.text
    route.assert_not_awaited()
    registry.ensure_lora_loaded_async.assert_not_called()
    registry.start_worker.assert_not_called()


@pytest.mark.parametrize("wire", ["json", "msgpack"])
@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ({}, "legal"),
        ({"lora_adapter": None}, None),
        ({"lora": None}, None),
        ({"lora_id": None}, None),
        ({"lora_adapter": "org/legal"}, "org/legal"),
    ],
)
def test_http_profile_alias_and_source_default_reaches_adapter(wire, selection, expected):
    config = ModelConfig.model_validate(
        {
            "sie_id": "m",
            "hf_id": "unloaded",
            "tasks": {"encode": {"dense": {"dim": 8}}},
            "profiles": {
                "default": {"max_batch_tokens": 8192, "adapter_path": "test:TestAdapter"},
                "legal": {
                    "extends": "default",
                    "adapter_options": {
                        "loadtime": {"lora_paths": {"legal": {"id": "org/legal", "revision": "a" * 40}}},
                        "runtime": {"lora": "legal", "lora_id": "org/legal"},
                    },
                },
            },
        }
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_config.return_value = config
    registry.device = "cpu"
    registry.preprocessor_registry = MagicMock()
    registry.preprocessor_registry.has_tokenizer.return_value = False
    registry.preprocessor_registry.has_preprocessor.return_value = False
    registry.ensure_lora_loaded_async = AsyncMock(return_value=(True, False))
    adapter = MagicMock()
    adapter.encode.return_value = EncodeOutput(dense=np.ones((1, 8), dtype=np.float32), batch_size=1, dense_dim=8)
    registry.get.return_value = adapter
    app = FastAPI()
    app.include_router(encode.router)
    app.state.registry = registry
    body = {"items": [{"text": "x"}], "params": {"options": {"profile": "legal", **selection}}}
    with ThreadPoolExecutor(max_workers=1) as pool:
        registry.postprocessor_registry = PostprocessorRegistry(pool)
        with TestClient(app) as client:
            if wire == "json":
                response = client.post("/v1/encode/m", json=body, headers={"accept": "application/json"})
            else:
                response = client.post(
                    "/v1/encode/m",
                    content=packb(body),
                    headers={"content-type": "application/msgpack", "accept": "application/json"},
                )

    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["dense"]["values"] == [1.0] * 8
    adapter.encode.assert_called_once()
    assert adapter.encode.call_args.kwargs["options"] == {"lora": expected, "output_dtype": "float32"}
    if expected is None:
        registry.ensure_lora_loaded_async.assert_not_awaited()
    else:
        registry.ensure_lora_loaded_async.assert_awaited_once_with("m", expected)


@pytest.mark.parametrize("endpoint", ["encode", "extract", "score"])
@pytest.mark.parametrize("field", [None, "lora_adapter", "lora", "lora_id"])
def test_profile_selection_does_not_authorize_sibling_source_before_http_dispatch(endpoint, field, monkeypatch):
    config = ModelConfig.model_validate(
        {
            "sie_id": "m",
            "hf_id": "unloaded",
            "tasks": {"encode": {"dense": {"dim": 8}}, "extract": {}, "score": {}},
            "profiles": {
                "default": {
                    "max_batch_tokens": 8192,
                    "adapter_path": "test:TestAdapter",
                    "adapter_options": {"runtime": {"lora": "org/legal"}},
                },
                "legal": {
                    "extends": "default",
                    "adapter_options": {"loadtime": {"lora_paths": {"legal": "org/legal"}}},
                },
            },
        }
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = registry.is_loaded.return_value = True
    registry.is_loading.return_value = registry.is_unloading.return_value = registry.is_failed.return_value = False
    registry.get_config.return_value = config
    registry.device = "cpu"
    app = FastAPI()
    module = {"encode": encode, "extract": extract, "score": score}[endpoint]
    app.include_router(module.router)
    app.state.registry = registry
    route = AsyncMock(side_effect=AssertionError("undeclared source reached route"))
    monkeypatch.setattr(module, "route_request", route)
    options = {} if field is None else {field: "org/legal"}
    body = {"items": [{"text": "x"}], "params": {"labels": ["label"], "options": options}}
    if endpoint == "score":
        body = {"query": {"text": "q"}, "items": [{"text": "x"}], "options": options}
    with TestClient(app) as client:
        response = client.post(f"/v1/{endpoint}/m", json=body)
    assert response.status_code == 400, response.text
    assert "declar" in response.text
    route.assert_not_awaited()
    registry.ensure_lora_loaded_async.assert_not_called()
    registry.start_worker.assert_not_called()
