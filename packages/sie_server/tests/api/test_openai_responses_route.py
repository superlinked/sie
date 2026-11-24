"""``POST /v1/responses`` through the server's own app and chat route, on the weightless fake.

The app comes from :class:`AppFactory`, so its middleware and routers run as
``sie-server serve`` runs them. The model configs and the fake adapter come
from the catalog; only the registry is a stub. So each request is translated,
served by the real ``/v1/chat/completions`` code and translated back.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from functools import cache
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from sie_server.adapters.errors import UPSTREAM_FAILURE_MESSAGES
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.api import openai_responses
from sie_server.api.lora import LeasedStreamingResponse
from sie_server.api.responses_translation import StreamMachine, responses_to_chat
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter, load_model_configs

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
VECTORS = Path(__file__).resolve().parents[4] / "conformance" / "openai_responses"
MODEL = "sie-fake:chat"
QUESTION = "What is the weather in Paris?"
WEATHER = {
    "type": "function",
    "name": "get_weather",
    "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
APPLY_PATCH = {"type": "custom", "name": "apply_patch", "description": "Apply a patch to the workspace."}
_TERMINAL = ("response.completed", "response.incomplete", "response.failed")


@cache
def _configs() -> dict[str, ModelConfig]:
    return load_model_configs(MODELS_DIR)


def _registry(loading: frozenset[str] = frozenset()) -> MagicMock:
    configs = {name: _configs()[name] for name in ("sie-fake", MODEL)}
    adapters = {}
    for name, config in configs.items():
        adapter = load_adapter(config, MODELS_DIR, device="cpu")
        adapter.load("cpu")
        adapters[name] = adapter
    registry = MagicMock()
    registry.device = "cpu"
    registry.model_names = list(configs)
    registry.has_model.side_effect = configs.__contains__
    registry.get_config.side_effect = configs.__getitem__
    registry.get.side_effect = adapters.__getitem__
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.side_effect = loading.__contains__
    registry.is_loaded.return_value = True
    registry.adapters = adapters
    return registry


@pytest.fixture
def registry() -> MagicMock:
    return _registry()


def _client(registry: MagicMock) -> TestClient:
    app = AppFactory.create_app(AppStateConfig(auth_mode="none"))
    app.state.registry = registry
    return TestClient(app)


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    return _client(registry)


def _body(prompt: str = "Which is larger, 9.11 or 9.9?", **fields: Any) -> dict[str, Any]:
    return {"model": MODEL, "input": prompt, "max_output_tokens": 64, **fields}


def _post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/v1/responses", json=body)


def _chat(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/v1/chat/completions", json=responses_to_chat(body).chat)


def _events(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.split("\n\n"):
        if not frame.strip() or frame.startswith(":"):
            continue
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"]
        events.append(data)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))
    assert [event["type"] for event in events].count(events[-1]["type"]) == 1
    assert events[-1]["type"] in _TERMINAL
    assert "[DONE]" not in response.text
    return events


def _chat_events(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    data = [block.removeprefix("data: ") for block in response.text.split("\n\n") if block.startswith("data: ")]
    assert data[-1] == "[DONE]"
    return [json.loads(item) for item in data[:-1]]


def _names(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def _error(response: Any, status: int) -> dict[str, Any]:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert set(body) == {"error"}
    return body["error"]


def _usage(chat_usage: dict[str, int]) -> dict[str, Any]:
    return {
        "input_tokens": chat_usage["prompt_tokens"],
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": chat_usage["completion_tokens"],
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": chat_usage["total_tokens"],
    }


def test_a_buffered_response_matches_the_chat_answer_to_the_same_conversation(client: TestClient) -> None:
    body = _body(
        "Reply with a word.",
        instructions="Be brief.",
        store=False,
        include=["reasoning.encrypted_content"],
        prompt_cache_key="session-1",
        client_metadata={"turn": "1"},
        reasoning={"summary": "auto"},
        text={"verbosity": "low"},
    )
    response = _post(client, body)
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "max_completion_tokens": 64,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Reply with a word."}],
        },
    ).json()

    assert response.status_code == 200, response.text
    assert response.headers["x-sie-served-by"] == "local"
    answer = response.json()
    assert answer["id"].startswith("resp-")
    assert isinstance(answer["created_at"], int)
    request_id = answer["id"].removeprefix("resp-")
    assert answer == {
        "id": f"resp-{request_id}",
        "object": "response",
        "created_at": answer["created_at"],
        "model": MODEL,
        "status": "completed",
        "incomplete_details": None,
        "error": None,
        "store": False,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "output": [
            {
                "type": "message",
                "id": f"msg-{request_id}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": chat["choices"][0]["message"]["content"], "annotations": []}
                ],
            }
        ],
        "usage": _usage(chat["usage"]),
    }


def test_a_streamed_response_matches_the_chat_stream_of_the_same_conversation(client: TestClient) -> None:
    body = _body("Reply with a word.", stream=True)
    events = _events(_post(client, body))
    chat = _chat_events(_chat(client, body))

    text = "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    assert _names(events)[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert _names(events)[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert text == "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chat if chunk["choices"])
    completed = events[-1]["response"]
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"][0]["text"] == text
    assert completed["usage"] == _usage(chat[-1]["usage"])


def test_a_function_tool_loop_calls_the_tool_then_answers_with_its_output(client: TestClient) -> None:
    first = _post(client, _body(QUESTION, tools=[WEATHER], tool_choice="auto"))

    assert first.status_code == 200, first.text
    [call] = first.json()["output"]
    request_id = first.json()["id"].removeprefix("resp-")
    assert call["type"] == "function_call"
    assert call["id"] == f"fc-{request_id}-0"
    assert call["call_id"].startswith("call_")
    assert call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": QUESTION}
    assert call["status"] == "completed"
    assert first.json()["status"] == "completed"
    streamed = _events(_post(client, _body(QUESTION, stream=True, tools=[WEATHER], tool_choice="auto")))
    arguments = "".join(e["delta"] for e in streamed if e["type"] == "response.function_call_arguments.delta")
    [done] = [event for event in streamed if event["type"] == "response.function_call_arguments.done"]
    assert arguments == done["arguments"] == call["arguments"]
    assert streamed[-1]["response"]["output"][0]["call_id"] == call["call_id"]

    result = '{"city": "Paris", "temperature_c": 18}'
    second = _post(
        client,
        _body(
            tools=[WEATHER],
            input=[
                {"role": "user", "content": QUESTION},
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {key: call[key] for key in ("type", "id", "call_id", "name", "arguments", "status")},
                {"type": "function_call_output", "call_id": call["call_id"], "output": result},
            ],
        ),
    )

    assert second.status_code == 200, second.text
    [message] = second.json()["output"]
    assert message["content"][0]["text"] == f"Tool result: {result}"


def test_a_streamed_custom_tool_call_carries_the_decoded_input(client: TestClient) -> None:
    events = _events(_post(client, _body(QUESTION, stream=True, tools=[APPLY_PATCH])))
    added = next(event for event in events if event["type"] == "response.output_item.added")
    [done] = [event for event in events if event["type"] == "response.custom_tool_call_input.done"]

    assert added["item"]["type"] == "custom_tool_call"
    assert added["item"]["input"] == ""
    assert done["input"] == QUESTION
    assert "response.function_call_arguments.delta" not in _names(events)
    assert events[-1]["response"]["output"][0]["input"] == QUESTION


def test_max_output_tokens_cuts_the_answer_as_incomplete(client: TestClient) -> None:
    buffered = _post(client, _body("Write a long answer.", max_output_tokens=3)).json()
    streamed = _events(_post(client, _body("Write a long answer.", max_output_tokens=3, stream=True)))

    for response in (buffered, streamed[-1]["response"]):
        assert response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
        assert response["usage"]["output_tokens"] == 3
    assert streamed[-1]["type"] == "response.incomplete"


def test_a_body_that_is_not_json_or_is_too_large_is_refused(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = client.post("/v1/responses", content=b"{", headers={"content-type": "application/json"})
    monkeypatch.setattr(openai_responses, "_MAX_BODY_BYTES", 16)
    large = _post(client, _body())

    error = _error(invalid, 400)
    assert error["message"] == "body: invalid JSON"
    assert (error["param"], error["code"]) == ("body", "invalid_request")
    assert _error(large, 413) == {
        "message": "request body too large",
        "type": "invalid_request_error",
        "param": None,
        "code": "INPUT_TOO_LONG",
    }


def test_a_translation_refusal_reaches_no_engine(client: TestClient, registry: MagicMock) -> None:
    registry.get.reset_mock()
    response = _post(client, _body(store=True))

    assert _error(response, 400) == {
        "message": "store: SIE does not store responses; send the full conversation in input",
        "type": "invalid_request_error",
        "param": "store",
        "code": "unsupported_field",
    }
    registry.get.assert_not_called()


@pytest.mark.parametrize(
    ("body", "status", "param"),
    [
        (_body(model="sie-conformance/no-such-model"), 404, None),
        (_body(max_output_tokens=5000), 400, "max_output_tokens"),
        (_body(tools=[{**WEATHER, "name": "get weather"}]), 400, "tools[0].name"),
        (_body(reasoning={"effort": "extreme"}), 400, "reasoning.effort"),
    ],
    ids=["unknown model", "output cap", "tool name", "reasoning effort"],
)
def test_a_chat_refusal_comes_back_with_the_field_the_caller_sent(
    client: TestClient, body: dict[str, Any], status: int, param: str | None
) -> None:
    for stream in (False, True):
        chat = _chat(client, {**body, "stream": stream})
        error = _error(_post(client, {**body, "stream": stream}), status)
        assert chat.status_code == status
        assert error == {**chat.json()["error"], "param": param}


def test_a_loading_model_answers_the_chat_error_with_its_retry_after() -> None:
    client = _client(_registry(loading=frozenset({MODEL})))
    response = _post(client, _body(stream=True))
    chat = _chat(client, _body(stream=True))

    assert _error(response, 503) == chat.json()["error"]
    assert response.json()["error"]["code"] == "MODEL_LOADING"
    assert response.headers["retry-after"] == chat.headers["retry-after"] == "5"


@pytest.mark.parametrize(
    "case",
    json.loads((VECTORS / "error_vectors.json").read_text(encoding="utf-8"))["errors"],
    ids=lambda case: case["name"],
)
@pytest.mark.parametrize("stream", [False, True])
def test_an_error_vector_from_the_chat_route_is_answered_at_its_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, case: dict[str, Any], stream: bool
) -> None:
    async def chat_error(*_: Any) -> JSONResponse:
        return JSONResponse(status_code=case["status"], content=case["body"], headers=case.get("headers"))

    monkeypatch.setattr(openai_responses, "serve_chat_completions", chat_error)
    response = _post(client, _body(stream=stream))

    expect = case["expect"]
    assert response.status_code == expect["status"]
    assert response.headers["content-type"] == "application/json"
    for name, value in (expect.get("headers") or {}).items():
        assert response.headers[name] == value, name
    for name in expect.get("absent_headers") or []:
        assert name not in response.headers, name
    assert response.json() == expect["body"]


def test_stored_response_routes_are_not_served(client: TestClient) -> None:
    assert client.get("/v1/responses/resp-1").status_code == 404
    assert client.post("/v1/responses/resp-1/cancel").status_code == 404
    assert client.get("/v1/responses").status_code == 405


def _scripted_stream(adapter: Any, *, after: int, ending: str) -> None:
    original = adapter.chat_completion_stream

    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        sent = 0
        async for payload in original(*args, **kwargs):
            if sent == after:
                if ending == "fail":
                    raise RemoteUpstreamError("the scripted engine failed", upstream_reason="upstream_unavailable")
                return
            sent += 1
            yield payload

    adapter.chat_completion_stream = stream


def test_a_failure_after_output_fails_the_response(registry: MagicMock, client: TestClient) -> None:
    _scripted_stream(registry.adapters[MODEL], after=3, ending="fail")
    events = _events(_post(client, _body(stream=True)))

    assert "response.output_text.delta" in _names(events)
    assert _names(events)[-1] == "response.failed"
    failed = events[-1]["response"]
    assert failed["status"] == "failed"
    assert failed["output"] == []
    assert failed["usage"] is None
    assert failed["error"] == {"code": "upstream_error", "message": UPSTREAM_FAILURE_MESSAGES["upstream_unavailable"]}
    assert "response.completed" not in _names(events)


def test_a_stream_ending_without_a_finish_fails_the_response(registry: MagicMock, client: TestClient) -> None:
    _scripted_stream(registry.adapters[MODEL], after=3, ending="stop")
    events = _events(_post(client, _body(stream=True)))

    assert "response.output_text.delta" in _names(events)
    assert _names(events)[-1] == "response.failed"
    assert events[-1]["response"]["error"] == {
        "code": "server_error",
        "message": "the generation stream ended without a finish reason",
    }


class _Lease:
    def __init__(self) -> None:
        self.released = False
        self.body_closed = False

    async def body(self) -> AsyncIterator[bytes]:
        try:
            yield b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n'
            await asyncio.get_running_loop().create_future()
        finally:
            self.body_closed = True

    async def answer(self) -> LeasedStreamingResponse:
        lifetime = AsyncExitStack()
        lifetime.callback(setattr, self, "released", True)
        inner = StreamingResponse(self.body(), media_type="text/event-stream")
        return LeasedStreamingResponse(inner, lifetime)


async def _serve(response: Any, *, fail_on: str, after: int = 0) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == fail_on and sum(m["type"] == fail_on for m in sent) >= after:
            raise OSError("the client went away")
        sent.append(message)

    async def receive() -> dict[str, Any]:
        await asyncio.get_running_loop().create_future()
        return {}

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"}}
    with pytest.raises(Exception):  # noqa: B017, PT011 - Starlette reports the disconnect in its own type
        await asyncio.wait_for(response(scope, receive, send), 5)
    return sent


def _machine() -> StreamMachine:
    return StreamMachine(responses_to_chat({"model": "m", "input": "x", "stream": True}).shape, "c", 0)


async def test_a_client_that_goes_away_mid_stream_ends_the_chat_answer() -> None:
    lease = _Lease()
    stream = openai_responses._ResponsesStream(await lease.answer(), _machine())

    sent = await _serve(stream, fail_on="http.response.body", after=4)

    assert any(b"response.output_item.added" in message.get("body", b"") for message in sent)
    assert lease.body_closed
    assert lease.released


async def test_a_stream_never_read_still_releases_the_chat_answer() -> None:
    lease = _Lease()
    stream = openai_responses._ResponsesStream(await lease.answer(), _machine())

    await _serve(stream, fail_on="http.response.start")

    assert lease.released
