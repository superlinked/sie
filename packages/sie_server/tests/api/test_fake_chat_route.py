"""The fake engine's scripted chat through the single-node ``/v1/chat/completions`` route.

The model configs and the adapter come from the catalog, as ``sie-server serve``
loads them; only the registry is a stub, so the route's validation, the
adapter-owned chat path and its output checks all run for real.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.api.openai_local import router
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter, load_model_configs

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
QUESTION = "What is the weather in Paris?"
RESULT = '{"city": "Paris", "temperature_c": 18}'


@cache
def _config(model: str) -> ModelConfig:
    return load_model_configs(MODELS_DIR)[model]


def _client(model: str = "sie-fake:chat") -> TestClient:
    config = _config(model)
    adapter = load_adapter(config, MODELS_DIR, device="cpu")
    adapter.load("cpu")
    registry = MagicMock()
    registry.device = "cpu"
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = False
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    app = FastAPI()
    app.include_router(router)
    app.state.registry = registry
    return TestClient(app)


def _chat(client: TestClient, messages: list[dict[str, Any]], **fields: Any) -> Any:
    model = fields.pop("model", "sie-fake:chat")
    return client.post("/v1/chat/completions", json={"model": model, "messages": messages, **fields})


def _events(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    data = [block.removeprefix("data: ") for block in response.text.split("\n\n") if block.startswith("data: ")]
    assert data[-1] == "[DONE]"
    return [json.loads(item) for item in data[:-1]]


def _deltas(events: list[dict[str, Any]], key: str) -> list[Any]:
    return [choice["delta"][key] for event in events for choice in event["choices"] if key in choice["delta"]]


def _finish(events: list[dict[str, Any]]) -> list[str]:
    return [choice["finish_reason"] for event in events for choice in event["choices"] if choice["finish_reason"]]


def test_a_tools_turn_streams_one_call_with_its_id_and_name_first() -> None:
    response = _chat(
        _client(),
        [{"role": "user", "content": QUESTION}],
        tools=[WEATHER],
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
        include_reasoning=True,
        max_tokens=64,
    )
    events = _events(response)
    pieces = [piece for calls in _deltas(events, "tool_calls") for piece in calls]
    reasoning = _deltas(events, "reasoning_content")

    assert response.headers["x-sie-served-by"] == "local"
    assert pieces[0]["id"].startswith("call_")
    assert pieces[0]["function"]["name"] == "get_weather"
    assert all("id" not in piece and "name" not in piece["function"] for piece in pieces[1:])
    assert json.loads("".join(piece["function"]["arguments"] for piece in pieces)) == {"city": QUESTION}
    assert "".join(reasoning).startswith("Tools are declared")
    assert _finish(events) == ["tool_calls"]
    assert events[-1]["choices"] == []
    assert events[-1]["usage"]["completion_tokens"] == len(reasoning) + len(pieces)


def test_a_tool_result_is_answered_in_text_that_carries_it() -> None:
    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": RESULT},
    ]
    response = _chat(_client(), messages, tools=[WEATHER], max_tokens=64)

    assert response.status_code == 200, response.text
    [choice] = response.json()["choices"]
    assert choice["message"]["content"] == f"Tool result: {RESULT}"
    assert "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


@pytest.mark.parametrize("stream", [True, False])
def test_reasoning_reaches_the_caller_only_with_include_reasoning(stream: bool) -> None:
    client = _client()
    messages = [{"role": "user", "content": "Which is larger, 9.11 or 9.9?"}]
    fields = {"stream": stream, "stream_options": {"include_usage": True}} if stream else {}
    answers = [_chat(client, messages, max_tokens=64, include_reasoning=flag, **fields) for flag in (True, False)]
    if stream:
        reasoning = ["".join(_deltas(_events(answer), "reasoning_content")) for answer in answers]
        usage = [_events(answer)[-1]["usage"] for answer in answers]
    else:
        reasoning = [answer.json()["choices"][0]["message"].get("reasoning_content", "") for answer in answers]
        usage = [answer.json()["usage"] for answer in answers]

    assert reasoning[0] == "No tool call applies, so answer in text."
    assert reasoning[1] == ""
    assert usage[0] == usage[1]


def test_required_tool_choice_forces_a_call_on_a_cpu_device() -> None:
    response = _chat(
        _client(),
        [{"role": "user", "content": "Say hello."}],
        tools=[WEATHER],
        tool_choice="required",
        max_tokens=64,
    )

    assert response.status_code == 200, response.text
    [choice] = response.json()["choices"]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Say hello."}


def test_a_tool_call_is_never_cut_by_max_tokens() -> None:
    client = _client()
    messages = [{"role": "user", "content": QUESTION}]
    needed = _chat(client, messages, tools=[WEATHER], max_tokens=64).json()["usage"]["completion_tokens"]
    exact = _chat(client, messages, tools=[WEATHER], max_tokens=needed)
    short = _chat(client, messages, tools=[WEATHER], max_tokens=needed - 1, stream=True)

    assert exact.status_code == 200, exact.text
    [choice] = exact.json()["choices"]
    assert choice["finish_reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": QUESTION}
    assert short.status_code == 400
    assert short.json()["error"]["code"] == "invalid_request"


def test_tool_choice_none_answers_in_text() -> None:
    response = _chat(_client(), [{"role": "user", "content": QUESTION}], tools=[WEATHER], tool_choice="none")

    assert response.status_code == 200, response.text
    [choice] = response.json()["choices"]
    assert "tool_calls" not in choice["message"]
    assert choice["message"]["content"].startswith("tok_")
    assert choice["finish_reason"] == "stop"


def test_max_tokens_cuts_the_answer_with_length() -> None:
    response = _chat(_client(), [{"role": "user", "content": "Write a long answer."}], max_tokens=3)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3


def test_the_window_bounds_the_prompt_and_the_answer() -> None:
    client = _client("sie-fake")
    window = _config("sie-fake").max_sequence_length
    assert window is not None
    near = _chat(client, [{"role": "user", "content": "w " * (window - 5)}], model="sie-fake", max_tokens=64)
    full = _chat(client, [{"role": "user", "content": "w " * window}], model="sie-fake", max_tokens=64)

    assert near.status_code == 200, near.text
    assert near.json()["usage"] == {"prompt_tokens": window - 4, "completion_tokens": 4, "total_tokens": window}
    assert near.json()["choices"][0]["finish_reason"] == "length"
    assert full.status_code == 400
    assert full.json()["error"] == {
        "message": f"prompt_tokens ({window + 1}) exceeds context_length ({window}) for model 'sie-fake'",
        "type": "invalid_request_error",
        "param": "prompt",
        "code": "INPUT_TOO_LONG",
    }


def test_an_image_is_refused_as_unsupported_even_when_the_model_declares_image_input() -> None:
    client = _client()
    config = _config("sie-fake:chat")
    registry = client.app.state.registry
    registry.device = "cuda:0"
    registry.get_config.return_value = config.model_copy(
        update={"inputs": config.inputs.model_copy(update={"image": True})}
    )
    pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    response = _chat(client, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": pixel}}]}])

    assert response.status_code == 400, response.text
    assert response.json()["error"] == {
        "message": "Model 'sie-fake:chat' does not support image input",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "unsupported_field",
    }
    registry.get.assert_not_called()


def test_a_profile_without_thinking_never_returns_reasoning() -> None:
    response = _chat(_client("sie-fake"), [{"role": "user", "content": "hi"}], model="sie-fake", include_reasoning=True)

    assert response.status_code == 200, response.text
    [choice] = response.json()["choices"]
    assert "reasoning_content" not in choice["message"]
    assert response.json()["usage"]["completion_tokens"] == len(choice["message"]["content"].split())
