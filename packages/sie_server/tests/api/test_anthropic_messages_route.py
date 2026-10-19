"""``POST /v1/messages`` through the server's own app and chat route, on the weightless fake.

The app comes from :class:`AppFactory`, so its middleware, routers and the
Messages envelope run as ``sie-server serve`` runs them. The model configs and
the fake adapter come from the catalog; only the registry is a stub. So each
request is translated, served by the real ``/v1/chat/completions`` code and
translated back.
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
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.api import anthropic_messages
from sie_server.api.anthropic_translation import StreamMachine
from sie_server.api.lora import LeasedStreamingResponse
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter, load_model_configs

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL = "sie-fake:chat"
QUESTION = "What is the weather in Paris?"
WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}
HEADERS = {"anthropic-version": "2023-06-01", "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14"}
TOKEN = "messages-route-token"  # noqa: S105 - a test credential


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


def _client(registry: MagicMock, monkeypatch: pytest.MonkeyPatch, *, token: str | None = None) -> TestClient:
    if token is not None:
        monkeypatch.setenv("SIE_AUTH_TOKEN", token)
    app = AppFactory.create_app(AppStateConfig(auth_mode="token" if token else "none"))
    app.state.registry = registry
    return TestClient(app)


@pytest.fixture
def client(registry: MagicMock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    return _client(registry, monkeypatch)


def _body(text: str = "Which is larger, 9.11 or 9.9?", **fields: Any) -> dict[str, Any]:
    return {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": text}], **fields}


def _post(client: TestClient, body: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    return client.post("/v1/messages?beta=true", json=body, headers=headers if headers is not None else HEADERS)


def _events(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.split("\n\n"):
        if not frame.strip():
            continue
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"]
        events.append(data)
    return events


def _names(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def _chat_events(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 200, response.text
    data = [block.removeprefix("data: ") for block in response.text.split("\n\n") if block.startswith("data: ")]
    assert data[-1] == "[DONE]"
    return [json.loads(item) for item in data[:-1]]


def _error(response: Any, status: int, error_type: str) -> str:
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == error_type
    assert body["request_id"] is None
    return body["error"]["message"]


def test_a_buffered_message_matches_the_chat_answer_to_the_same_conversation(client: TestClient) -> None:
    body = _body(
        "Reply with a word.",
        metadata={"user_id": "opaque"},
        system=[{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
        cache_control={"type": "ephemeral"},
    )
    body["messages"][0]["content"] = [
        {"type": "text", "text": "Reply with a word.", "cache_control": {"type": "ephemeral"}}
    ]
    response = _post(client, body)
    chat = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "max_tokens": 64,
            "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Reply with a word."}],
        },
    ).json()

    assert response.status_code == 200, response.text
    assert response.headers["x-sie-served-by"] == "local"
    assert "request-id" not in response.headers
    message = response.json()
    assert message["id"].startswith("msg_")
    assert len(message["id"]) > len("msg_")
    assert message["type"] == "message"
    assert message["role"] == "assistant"
    assert message["model"] == MODEL
    assert message["content"] == [{"type": "text", "text": chat["choices"][0]["message"]["content"]}]
    assert message["stop_reason"] == "end_turn"
    assert message["stop_sequence"] is None
    assert message["usage"] == {
        "input_tokens": chat["usage"]["prompt_tokens"],
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": chat["usage"]["completion_tokens"],
    }


def test_a_streamed_tool_turn_opens_thinking_then_tool_use_with_the_chat_usage(client: TestClient) -> None:
    response = _post(
        client,
        _body(QUESTION, stream=True, tools=[WEATHER], tool_choice={"type": "auto"}, thinking={"type": "adaptive"}),
    )
    chat = _chat_events(
        client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "max_tokens": 64,
                "stream": True,
                "stream_options": {"include_usage": True},
                "include_reasoning": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": WEATHER["description"],
                            "parameters": WEATHER["input_schema"],
                        },
                    }
                ],
                "tool_choice": "auto",
                "messages": [{"role": "user", "content": QUESTION}],
            },
        )
    )
    events = _events(response)
    starts = [event for event in events if event["type"] == "content_block_start"]
    deltas = [event["delta"] for event in events if event["type"] == "content_block_delta"]
    [message_delta] = [event for event in events if event["type"] == "message_delta"]

    assert _names(events)[0] == "message_start"
    assert events[0]["message"]["usage"]["input_tokens"] == 0
    assert [start["content_block"]["type"] for start in starts] == ["thinking", "tool_use"]
    assert starts[1]["content_block"]["id"].startswith("call_")
    assert starts[1]["content_block"]["name"] == "get_weather"
    assert "".join(delta["thinking"] for delta in deltas if delta["type"] == "thinking_delta").startswith(
        "Tools are declared"
    )
    assert {"type": "signature_delta", "signature": "sie-unsigned"} in deltas
    arguments = "".join(delta["partial_json"] for delta in deltas if delta["type"] == "input_json_delta")
    assert json.loads(arguments) == {"city": QUESTION}
    assert message_delta["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}
    assert message_delta["usage"]["input_tokens"] == chat[-1]["usage"]["prompt_tokens"]
    assert message_delta["usage"]["output_tokens"] == chat[-1]["usage"]["completion_tokens"]
    assert _names(events)[-2:] == ["message_delta", "message_stop"]


def test_a_tool_result_is_answered_in_text_and_replayed_thinking_does_not_reach_the_model(client: TestClient) -> None:
    result = '{"city": "Paris", "temperature_c": 18}'
    response = _post(
        client,
        _body(
            tools=[WEATHER],
            messages=[
                {"role": "user", "content": QUESTION},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should call it.", "signature": "sie-unsigned"},
                        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": result}]},
            ],
        ),
    )

    assert response.status_code == 200, response.text
    message = response.json()
    assert message["content"] == [{"type": "text", "text": f"Tool result: {result}"}]
    assert message["stop_reason"] == "end_turn"


@pytest.mark.parametrize("thinking", [None, {"type": "disabled"}, {"type": "adaptive", "display": "omitted"}])
def test_reasoning_stays_private_without_the_opt_in(client: TestClient, thinking: dict[str, Any] | None) -> None:
    fields = {} if thinking is None else {"thinking": thinking}
    streamed = _events(_post(client, _body(stream=True, **fields)))
    buffered = _post(client, _body(**fields)).json()

    assert all(event.get("content_block", {}).get("type") != "thinking" for event in streamed)
    assert [block["type"] for block in buffered["content"]] == ["text"]


def test_thinking_opts_into_the_reasoning_on_both_paths(client: TestClient) -> None:
    streamed = _events(_post(client, _body(stream=True, thinking={"type": "adaptive"})))
    buffered = _post(client, _body(thinking={"type": "enabled", "budget_tokens": 1024}, max_tokens=2048)).json()
    reasoning = "".join(
        event["delta"]["thinking"]
        for event in streamed
        if event["type"] == "content_block_delta" and event["delta"]["type"] == "thinking_delta"
    )

    assert reasoning == "No tool call applies, so answer in text."
    assert buffered["content"][0] == {"type": "thinking", "thinking": reasoning, "signature": "sie-unsigned"}
    assert buffered["content"][1]["type"] == "text"


def test_max_tokens_cuts_the_answer_with_a_max_tokens_stop(client: TestClient) -> None:
    message = _post(client, _body("Write a long answer.", max_tokens=3)).json()

    assert message["stop_reason"] == "max_tokens"
    assert message["usage"]["output_tokens"] == 3


def test_a_translation_refusal_reaches_no_engine(client: TestClient, registry: MagicMock) -> None:
    registry.get.reset_mock()
    response = _post(client, _body(output_config={"effort": "high"}))

    assert _error(response, 400, "invalid_request_error") == "output_config.effort: Extra inputs are not permitted"
    assert response.headers["x-sie-error-code"] == "INVALID_REQUEST"
    registry.get.assert_not_called()


@pytest.mark.parametrize(
    ("body", "status", "error_type", "message"),
    [
        (_body(model="sie-conformance/no-such-model"), 404, "not_found_error", None),
        (
            _body(max_tokens=5000),
            400,
            "invalid_request_error",
            "max_tokens: 5000 > 4096, which is the maximum allowed number of output tokens for this model",
        ),
        (_body(" ".join(["w"] * 600), model="sie-fake"), 400, "invalid_request_error", "prompt is too long"),
        (
            _body(messages=[{"role": "user", "content": "hi"}, {"role": "system", "content": "short answers"}]),
            400,
            "invalid_request_error",
            'messages.1.role: role "system" is not supported in messages; use the top-level system parameter',
        ),
        ({"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}, 400, "invalid_request_error", None),
    ],
    ids=["unknown model", "output cap", "prompt too long", "system turn", "missing max_tokens"],
)
def test_refusals_come_back_in_the_anthropic_envelope(
    client: TestClient, body: dict[str, Any], status: int, error_type: str, message: str | None
) -> None:
    for stream in (False, True):
        refused = _error(_post(client, {**body, "stream": stream}), status, error_type)
        if message is not None:
            assert refused.startswith(message), refused


def test_a_loading_model_is_overloaded_with_its_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(_registry(loading=frozenset({MODEL})), monkeypatch)
    response = _post(client, _body(stream=True))

    assert _error(response, 503, "overloaded_error") == f"Model '{MODEL}' is loading, please retry"
    assert response.headers["retry-after"] == "5"
    assert response.headers["x-should-retry"] == "true"
    assert response.headers["x-sie-error-code"] == "MODEL_LOADING"


def test_paths_below_messages_take_the_envelope_and_other_paths_do_not(client: TestClient) -> None:
    count = client.post("/v1/messages/count_tokens", json=_body(), headers=HEADERS)
    wrong_method = client.get("/v1/messages")
    other = client.post("/v1/unknown", json=_body())

    assert _error(count, 404, "not_found_error") == "Not Found"
    assert _error(wrong_method, 405, "invalid_request_error") == "Method Not Allowed"
    assert other.status_code == 404
    assert other.json() == {"detail": "Not Found"}


def test_token_auth_takes_either_header_and_refuses_in_the_anthropic_envelope(
    registry: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(registry, monkeypatch, token=TOKEN)
    missing = _post(client, _body())
    chat_missing = client.post(
        "/v1/chat/completions", json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    )

    assert _error(missing, 401, "authentication_error").startswith("Missing credentials")
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.headers["x-sie-error-code"] == "UNAUTHORIZED"
    assert chat_missing.status_code == 401
    assert chat_missing.json()["detail"]["code"] == "UNAUTHORIZED"
    for credentials in (
        {"x-api-key": TOKEN},
        {"authorization": f"Bearer {TOKEN}"},
        {"authorization": f"Bearer {TOKEN}", "x-api-key": "wrong"},
    ):
        assert _post(client, _body(), {**HEADERS, **credentials}).status_code == 200, credentials
    wrong = _post(client, _body(), {**HEADERS, "authorization": "Bearer wrong", "x-api-key": TOKEN})
    assert _error(wrong, 401, "authentication_error") == "Invalid token"


def _delayed_stream(adapter: Any, delay_s: float, *, after: int = 1, fail: bool = False) -> None:
    original = adapter.chat_completion_stream

    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        sent = 0
        async for payload in original(*args, **kwargs):
            if sent == after:
                await asyncio.sleep(delay_s)
                if fail:
                    raise RemoteUpstreamError("the scripted engine failed", upstream_reason="upstream_unavailable")
            sent += 1
            yield payload

    adapter.chat_completion_stream = stream


def test_a_silent_stream_is_kept_alive_with_pings(
    registry: MagicMock, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(anthropic_messages, "PING_INTERVAL_S", 0.02)
    _delayed_stream(registry.adapters[MODEL], 0.2)
    events = _events(_post(client, _body(stream=True)))
    names = _names(events)

    assert names[0] == "message_start"
    assert names.count("ping") >= 2, names
    assert names[-2:] == ["message_delta", "message_stop"]


def test_a_failure_after_output_is_an_error_event(registry: MagicMock, client: TestClient) -> None:
    _delayed_stream(registry.adapters[MODEL], 0.0, after=2, fail=True)
    events = _events(_post(client, _body(stream=True, thinking={"type": "adaptive"})))

    assert _names(events)[:2] == ["message_start", "content_block_start"]
    assert _names(events)[-1] == "error"
    assert events[-1]["error"]["type"] == "api_error"
    assert "message_stop" not in _names(events)


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


async def test_a_client_that_goes_away_mid_stream_ends_the_chat_answer() -> None:
    lease = _Lease()
    stream = anthropic_messages._MessagesStream(await lease.answer(), StreamMachine("msg_c", "m"), None)

    sent = await _serve(stream, fail_on="http.response.body", after=2)

    assert any(b"content_block_start" in message.get("body", b"") for message in sent)
    assert lease.body_closed
    assert lease.released


async def test_a_stream_never_read_still_releases_the_chat_answer() -> None:
    lease = _Lease()
    stream = anthropic_messages._MessagesStream(await lease.answer(), StreamMachine("msg_n", "m"), None)

    await _serve(stream, fail_on="http.response.start")

    assert lease.released
