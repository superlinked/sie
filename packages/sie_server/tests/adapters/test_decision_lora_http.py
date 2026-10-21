"""Public selectors reach a real native decision model and its runtime loader."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, Mock

import msgspec
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_sdk import SIEClient
from sie_server.api import decisions, extract, systemone
from sie_server.config.model import ModelConfig
from sie_server.core.model_loader import ModelLoader
from sie_server.core.registry import ModelRegistry
from sie_server.types.inputs import Item

from . import test_decision_lora as native

# Reuse the generated native model and adapter fixtures without downloading assets.
laya_checkpoint = native.laya_checkpoint
loaded_laya = native.loaded_laya
offline = native.offline
QUESTIONS = native.QUESTIONS

TEXT = "alice wants a refund"
OPENAI_QUESTIONS = [
    {"type": "predicate", "name": "urgent", "instructions": "is this urgent"},
    {
        "type": "choice",
        "name": "intent",
        "instructions": "classify the request",
        "choices": [{"value": "refund"}, {"value": "support"}],
    },
]


def assert_answers(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            assert_answers(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_answers(left, right)
    elif isinstance(expected, float):
        # Worker coalescing can change the tiny FP32 matrix batch shape.
        assert actual == pytest.approx(expected, abs=1e-6)
    else:
        assert actual == expected


@pytest.mark.parametrize("surface", ["native", "systemone", "decisions"])
@pytest.mark.parametrize("profile_default", [None, "a"])
def test_real_http_lazy_load_aliases_defaults_and_isolation(loaded_laya, monkeypatch, surface, profile_default):
    adapter, paths = loaded_laya
    fetch = Mock(wraps=adapter.fetch_lora)
    monkeypatch.setattr(adapter, "fetch_lora", fetch)
    config = ModelConfig.model_validate(
        {
            "sie_id": "decision",
            "hf_id": adapter._model_name_or_path,
            "tasks": {"extract": {}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.laya.runtime_lora:LayaLoraAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {
                        "loadtime": {"lora_paths": paths, "max_loaded_loras": 2},
                        "runtime": {"lora": profile_default},
                    },
                }
            },
        }
    )
    registry = ModelRegistry(device="cpu")
    loader = ModelLoader(
        preprocessor_registry=MagicMock(),
        postprocessor_registry=MagicMock(),
        all_configs={"decision": config},
        registry_callbacks=registry,
    )
    loaded = loader._finish_load("decision", "cpu", adapter, config)
    registry._configs["decision"] = config
    registry._loaded["decision"] = loaded

    @asynccontextmanager
    async def lifespan(app):
        await loaded.worker.start()
        try:
            yield
        finally:
            if registry._background_tasks:
                await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)
            await loaded.worker.stop()

    app = FastAPI(lifespan=lifespan)
    for router in (extract.router, systemone.router, decisions.router):
        app.include_router(router)
    app.state.registry = registry

    def request_body(selection):
        if surface == "native":
            return {
                "items": [{"text": TEXT}],
                "params": {"output_schema": QUESTIONS, "options": selection},
            }
        if surface == "systemone":
            return {"model": "decision", "state": TEXT, "questions": QUESTIONS, **selection}
        return {"model": "decision", "input": TEXT, "questions": OPENAI_QUESTIONS, **selection}

    endpoint = "/v1/extract/decision" if surface == "native" else f"/v1/{surface}"
    with TestClient(app, base_url="http://127.0.0.1", follow_redirects=False) as transport:

        def raw(selection):
            return transport.post(
                endpoint,
                json=request_body(selection),
                headers={"accept": "application/json", "content-type": "application/json"},
            )

        async def wait_for_load():
            if registry._background_tasks:
                await asyncio.wait_for(asyncio.gather(*registry._background_tasks), 5)

        # No request replay: each cold admission returns once, and a fresh
        # request is made only after the existing registry loader is ready.
        for name in ("a", "b"):
            cold = raw({"lora_adapter": name})
            assert cold.status_code == 503, cold.text
            assert cold.headers.get("retry-after")
            assert transport.portal is not None
            transport.portal.call(wait_for_load)
            assert paths[name] in loaded.loras
        assert fetch.call_count == 2

        if surface == "decisions":
            normalized, questions = decisions.normalize_request(request_body({}))
            schema = normalized.questions
        else:
            schema = QUESTIONS

        expected = {}
        for name in (None, "a", "b"):
            output = adapter.extract([Item(text=TEXT)], output_schema=schema, options={"lora": name})
            data = msgspec.to_builtins(output.data[0])
            if surface == "systemone":
                expected[name] = {key: systemone.to_answer(value) for key, value in data.items()}
            elif surface == "decisions":
                expected[name] = decisions.response_from_systemone(
                    {
                        "model": "decision",
                        "answers": {key: systemone.to_answer(value) for key, value in data.items()},
                        "usage": {"input_tokens": output.input_token_counts[0]},
                    },
                    questions,
                )["answers"]
            else:
                expected[name] = data
        assert expected["a"] != expected[None]
        assert expected["b"] != expected[None]
        assert expected["a"] != expected["b"]

        with SIEClient("http://127.0.0.1", http_client=transport) as client:

            def answer(selection):
                if surface == "native":
                    value = client.extract(
                        "decision",
                        {"text": TEXT},
                        output_schema=QUESTIONS,
                        options=selection,
                        wait_for_capacity=False,
                    )
                    assert client.last_retry_count == 0
                    return value["data"]
                response = raw(selection)
                assert response.status_code == 200, response.text
                return response.json()["answers"]

            assert_answers(answer({}), expected[profile_default])
            assert_answers(answer({"lora_adapter": None}), expected[None])
            assert_answers(answer({"lora": "a"}), expected["a"])
            assert_answers(answer({"lora_id": paths["b"]}), expected["b"])
            names = ["a", None, "b", "b", "a", None]
            with ThreadPoolExecutor(max_workers=3) as pool:
                actual = list(pool.map(lambda name: answer({"lora_adapter": name}), names))
            for value, name in zip(actual, names, strict=True):
                assert_answers(value, expected[name])
            assert_answers(answer({"lora_adapter": None}), expected[None])

            # Invalid requests do not fetch the still-cold third adapter and
            # cannot clear or switch the selected adapter implicitly.
            fetch.reset_mock()
            for selection in (
                {"lora_adapter": "c", "lora": "a"},
                {"lora_adapter": None, "lora_id": paths["a"]},
                {"lora_id": "a"},
                {"lora_adapter": "undeclared"},
            ):
                refused = raw(selection)
                assert refused.status_code == (422 if surface == "systemone" else 400), refused.text
            fetch.assert_not_called()
            assert paths["c"] not in loaded.loras
            assert_answers(answer({"lora_adapter": None}), expected[None])
