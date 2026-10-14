"""The fake adapter's scripted chat: properties of the script, never pinned hash tokens."""

from __future__ import annotations

import json
from typing import Any

import pytest
from sie_server.adapters._generation_base import GenerationInputTooLongError
from sie_server.adapters.fake.adapter import FakeAdapter


def _tool(name: str, *required: str) -> dict[str, Any]:
    parameters = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in required},
        "required": list(required),
    }
    return {"type": "function", "function": {"name": name, "parameters": parameters}}


def _body(content: str = "Look it up.", **fields: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], "max_tokens": 256, **fields}


@pytest.fixture
def adapter() -> FakeAdapter:
    fake = FakeAdapter(max_seq_length=1024)
    fake.load("cpu")
    return fake


async def _streamed(adapter: FakeAdapter, body: dict[str, Any], **options: Any) -> list[dict[str, Any]]:
    return [event async for event in adapter.chat_completion_stream(body, requested_model="m", **options)]


async def test_the_stream_and_the_answer_agree(adapter: FakeAdapter) -> None:
    body = _body(tools=[_tool("lookup", "query")], chat_template_kwargs={"enable_thinking": True})
    answer = await adapter.chat_completion(body, requested_model="m", keep_reasoning=True)
    events = await _streamed(adapter, body, keep_reasoning=True)
    deltas = [choice["delta"] for event in events for choice in event["choices"]]
    pieces = [piece for delta in deltas for piece in delta.get("tool_calls", [])]
    [message] = [choice["message"] for choice in answer["choices"]]

    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == message["reasoning_content"]
    assert pieces[0]["id"] == message["tool_calls"][0]["id"]
    assert (
        "".join(piece["function"]["arguments"] for piece in pieces) == message["tool_calls"][0]["function"]["arguments"]
    )
    assert events[-1]["usage"] == answer["usage"]


async def test_reasoning_is_spent_but_kept_private_without_keep_reasoning(adapter: FakeAdapter) -> None:
    body = _body(chat_template_kwargs={"enable_thinking": True})
    kept = await adapter.chat_completion(body, requested_model="m", keep_reasoning=True)
    private = await adapter.chat_completion(body, requested_model="m")
    events = await _streamed(adapter, body)

    assert kept["choices"][0]["message"]["reasoning_content"]
    assert "reasoning_content" not in private["choices"][0]["message"]
    assert not any("reasoning_content" in choice["delta"] for event in events for choice in event["choices"])
    assert private["usage"] == kept["usage"]


async def test_the_same_request_gets_the_same_answer_and_another_request_another(adapter: FakeAdapter) -> None:
    first = await adapter.chat_completion(_body("one"), requested_model="m")
    again = await adapter.chat_completion(_body("one"), requested_model="m")
    other = await adapter.chat_completion(_body("two"), requested_model="m")

    assert first["choices"] == again["choices"]
    assert first["choices"][0]["message"]["content"] != other["choices"][0]["message"]["content"]


async def test_a_named_tool_choice_calls_that_tool_with_the_last_user_message(adapter: FakeAdapter) -> None:
    body = _body(
        "Find flights.",
        tools=[_tool("first"), _tool("second", "origin", "destination")],
        tool_choice={"type": "function", "function": {"name": "second"}},
    )
    [choice] = (await adapter.chat_completion(body, requested_model="m"))["choices"]
    [call] = choice["message"]["tool_calls"]

    assert call["function"]["name"] == "second"
    assert json.loads(call["function"]["arguments"]) == {"origin": "Find flights.", "destination": "Find flights."}


async def test_a_forced_call_follows_a_tool_result(adapter: FakeAdapter) -> None:
    messages = [
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "c", "content": "found"},
    ]
    tools = [_tool("lookup", "query")]
    auto = await adapter.chat_completion({"messages": messages, "max_tokens": 64, "tools": tools}, requested_model="m")
    forced = await adapter.chat_completion(
        {"messages": messages, "max_tokens": 64, "tools": tools, "tool_choice": "required"}, requested_model="m"
    )

    assert auto["choices"][0]["message"]["content"] == "Tool result: found"
    assert forced["choices"][0]["finish_reason"] == "tool_calls"


async def test_n_repeats_the_turn_and_its_usage(adapter: FakeAdapter) -> None:
    single = await adapter.chat_completion(_body(), requested_model="m")
    double = await adapter.chat_completion(_body(n=2), requested_model="m")
    events = await _streamed(adapter, _body(n=2))

    assert [choice["index"] for choice in double["choices"]] == [0, 1]
    assert double["choices"][1]["message"] == single["choices"][0]["message"]
    assert double["usage"]["completion_tokens"] == 2 * single["usage"]["completion_tokens"]
    assert sorted(c["index"] for e in events for c in e["choices"] if c["finish_reason"]) == [0, 1]


async def test_a_prompt_that_leaves_no_room_is_refused(adapter: FakeAdapter) -> None:
    with pytest.raises(GenerationInputTooLongError):
        await adapter.chat_completion(_body("w " * 1023), requested_model="m")
    with pytest.raises(GenerationInputTooLongError):
        await _streamed(adapter, _body("w " * 1023))


async def test_chat_requires_load() -> None:
    with pytest.raises(RuntimeError):
        await FakeAdapter().chat_completion(_body(), requested_model="m")
