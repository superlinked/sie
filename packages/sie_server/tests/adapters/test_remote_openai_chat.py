import json

import pytest
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote._openai_chat import ChatStreamParser

USAGE = {"prompt_tokens": 37, "completion_tokens": 2, "total_tokens": 39}


def wire(choices: list[dict], **fields: object) -> bytes:
    return json.dumps({"choices": choices, **fields}).encode()


def choice(*, index: int = 0, stream: bool = False, finish: object = "stop", **fields: object) -> dict:
    container = {"role": "assistant", "content": "answer", **fields}
    return {"index": index, "delta" if stream else "message": container, "finish_reason": finish}


def test_buffered_chat_replaces_identity_and_discards_untrusted_metadata() -> None:
    parser = ChatStreamParser("requested/model")
    result = parser.completion(
        wire(
            [choice(reasoning_content="private", metadata={"key": "secret"})],
            usage={**USAGE, "prompt_tokens_details": {"cached_tokens": 3, "secret": "discard"}},
            id="secret-upstream-id",
            model="secret-model",
            system_fingerprint="secret",
        )
    )
    assert result["id"].startswith("chatcmpl-")
    assert result["model"] == "requested/model"
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "answer"}
    assert result["usage"] == {**USAGE, "prompt_tokens_details": {"cached_tokens": 3}}
    assert "secret" not in json.dumps(result)


def test_stream_requires_finish_usage_and_done_and_reuses_local_identity() -> None:
    parser = ChatStreamParser("requested/model")
    first = parser.parse(wire([choice(stream=True, finish=None, content="ans")]))
    last = parser.parse(wire([choice(stream=True, content="wer")]))
    usage = parser.parse(wire([], usage=USAGE))
    assert first is not None
    assert last is not None
    assert usage is not None
    assert first["id"] == last["id"] == usage["id"]
    assert first["created"] == last["created"]
    assert usage["usage"] == USAGE
    assert parser.parse(b"[DONE]") is None
    parser.finish()


@pytest.mark.parametrize(
    "events",
    [
        [],
        [wire([choice(stream=True, finish=None)])],
        [wire([choice(stream=True)])],
    ],
)
def test_done_without_finished_choices_and_usage_fails(events: list[bytes]) -> None:
    parser = ChatStreamParser("model")
    for event in events:
        parser.parse(event)
    with pytest.raises(RemoteUpstreamError):
        parser.parse(b"[DONE]")


def test_finished_stream_without_done_is_truncated() -> None:
    parser = ChatStreamParser("model")
    parser.parse(wire([choice(stream=True)], usage=USAGE))
    with pytest.raises(RemoteUpstreamError, match="terminal event"):
        parser.finish()


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {**USAGE, "prompt_tokens": True},
        {**USAGE, "completion_tokens": -1},
        {**USAGE, "total_tokens": 38},
        {**USAGE, "prompt_tokens_details": {"cached_tokens": 38}},
    ],
)
def test_buffered_chat_refuses_missing_and_invalid_exact_usage(usage: object) -> None:
    with pytest.raises(RemoteUpstreamError):
        ChatStreamParser("model").completion(wire([choice()], usage=usage))


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [choice(index=True)],
        [choice(index=1)],
        [choice(finish=None)],
        [choice(finish=[])],
        [choice(finish="unknown")],
        [choice(role="tool")],
        [choice(content={"private": "secret"})],
        [choice(), choice()],
    ],
)
def test_buffered_chat_refuses_invalid_choices(choices: list[dict]) -> None:
    with pytest.raises(RemoteUpstreamError):
        ChatStreamParser("model").completion(wire(choices, usage=USAGE))


def test_multiple_choices_must_all_finish() -> None:
    parser = ChatStreamParser("model", choices=2)
    parser.parse(wire([choice(index=0, stream=True)]))
    with pytest.raises(RemoteUpstreamError):
        parser.parse(wire([], usage=USAGE))


def test_duplicate_finished_choice_and_duplicate_usage_fail() -> None:
    parser = ChatStreamParser("model")
    parser.parse(wire([choice(stream=True)], usage=USAGE))
    with pytest.raises(RemoteUpstreamError):
        parser.parse(wire([choice(stream=True)]))
    with pytest.raises(RemoteUpstreamError):
        parser.parse(wire([], usage=USAGE))


def test_tool_calls_preserve_only_function_wire_fields() -> None:
    tools = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Budapest"}', "private": "discard"},
            "secret": "discard",
        }
    ]
    payload = ChatStreamParser("model").completion(
        wire([choice(finish="tool_calls", content=None, tool_calls=tools)], usage=USAGE)
    )
    assert payload["choices"][0]["message"]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Budapest"}'}}
    ]


def test_partial_tool_delta_and_nullable_optional_fields_are_valid() -> None:
    parser = ChatStreamParser("model")
    parser.parse(wire([choice(stream=True, finish=None, role=None, tool_calls=None)]))
    payload = parser.parse(
        wire([choice(stream=True, finish=None, tool_calls=[{"index": 0, "id": "call_1", "type": "function"}])])
    )
    assert payload is not None
    assert payload["choices"][0]["delta"]["tool_calls"] == [{"index": 0, "id": "call_1", "type": "function"}]
    parser.parse(
        wire(
            [
                choice(
                    stream=True,
                    finish=None,
                    tool_calls=[
                        {"index": 0, "id": None, "type": None, "function": {"name": "weather", "arguments": ""}}
                    ],
                )
            ]
        )
    )
    parser.parse(
        wire(
            [
                choice(
                    stream=True,
                    finish="tool_calls",
                    tool_calls=[{"index": 0, "id": None, "type": None, "function": {"name": None, "arguments": "{}"}}],
                )
            ],
            usage=USAGE,
        )
    )
    assert parser.parse(b"[DONE]") is None
    parser.finish()


@pytest.mark.parametrize("missing", ["id", "type", "name", "arguments", "all"])
def test_incomplete_streamed_tool_calls_cannot_finish(missing: str) -> None:
    tool = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": ""}}
    if missing in ("id", "type"):
        del tool[missing]
    elif missing == "all":
        tool = {"index": 0, "function": {"arguments": "{}"}}
    else:
        del tool["function"][missing]
    parser = ChatStreamParser("model")
    parser.parse(wire([choice(stream=True, finish=None, tool_calls=[tool])]))
    with pytest.raises(RemoteUpstreamError):
        parser.parse(wire([choice(stream=True, finish="tool_calls")], usage=USAGE))


def test_tool_finish_without_any_calls_is_invalid() -> None:
    with pytest.raises(RemoteUpstreamError):
        ChatStreamParser("model").parse(wire([choice(stream=True, finish="tool_calls")], usage=USAGE))


def test_buffered_tool_requires_explicit_type() -> None:
    tool = {"id": "call_1", "function": {"name": "weather", "arguments": "{}"}}
    with pytest.raises(RemoteUpstreamError):
        ChatStreamParser("model").completion(wire([choice(finish="tool_calls", tool_calls=[tool])], usage=USAGE))


@pytest.mark.parametrize("field", ["id", "name"])
def test_streamed_tool_headers_are_bounded_across_fragments(field: str) -> None:
    parser = ChatStreamParser("model")
    for text in ("a" * 256, "b"):
        tool = {"index": 0, field: text} if field == "id" else {"index": 0, "function": {field: text}}
        if text == "b":
            with pytest.raises(RemoteUpstreamError):
                parser.parse(wire([choice(stream=True, finish=None, tool_calls=[tool])]))
        else:
            parser.parse(wire([choice(stream=True, finish=None, tool_calls=[tool])]))


def test_tool_state_is_scoped_to_each_choice_and_allows_argument_fragments() -> None:
    parser = ChatStreamParser("model", choices=2)
    for index in (0, 1):
        header = {"index": 0, "id": f"call_{index}", "type": "function", "function": {"name": "weather"}}
        parser.parse(wire([choice(index=index, stream=True, finish=None, content=None, tool_calls=[header])]))
        arguments = {"index": 0, "function": {"arguments": "not valid JSON"}}
        parser.parse(
            wire([choice(index=index, stream=True, finish="tool_calls", content=None, tool_calls=[arguments])])
        )
    parser.parse(wire([], usage=USAGE))
    assert parser.parse(b"[DONE]") is None
    parser.finish()


def test_valid_unicode_remains_wire_encodable() -> None:
    result = ChatStreamParser("model").completion(wire([choice(content="🌤 Budapest")], usage=USAGE))
    assert "🌤 Budapest".encode() in json.dumps(result, ensure_ascii=False).encode()


def test_nullable_cache_count_is_omitted() -> None:
    result = ChatStreamParser("model").completion(
        wire([choice()], usage={**USAGE, "prompt_tokens_details": {"cached_tokens": None}})
    )
    assert result["usage"] == USAGE


@pytest.mark.parametrize("field", ["content", "refusal", "id", "name", "arguments", "token"])
@pytest.mark.parametrize("stream", [False, True])
def test_retained_strings_must_be_utf8_encodable(field: str, stream: bool) -> None:
    item = choice(stream=stream, finish=None if stream else "tool_calls")
    if field in ("content", "refusal"):
        item["delta" if stream else "message"][field] = "\ud800"
    elif field == "token":
        item["logprobs"] = {"content": [{"token": "\ud800", "logprob": -1}]}
    else:
        tool = {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
        if stream:
            tool["index"] = 0
        if field == "id":
            tool[field] = "\ud800"
        else:
            tool["function"][field] = "\ud800"
        item["delta" if stream else "message"]["tool_calls"] = [tool]
    parser = ChatStreamParser("model")
    with pytest.raises(RemoteUpstreamError):
        if stream:
            parser.parse(wire([item]))
        else:
            parser.completion(wire([item], usage=USAGE))


@pytest.mark.parametrize("logprob", [True, float("nan"), float("inf"), 0.5, -(10**500)])
def test_invalid_logprobs_fail_with_a_fixed_error(logprob: object) -> None:
    item = choice()
    item["logprobs"] = {"content": [{"token": "text", "logprob": logprob}]}
    with pytest.raises(RemoteUpstreamError):
        ChatStreamParser("model").completion(wire([item], usage=USAGE))


@pytest.mark.parametrize("data", [b"{", b'{"error":{"message":"secret"}}', b"x" * ((1 << 20) + 1)])
def test_invalid_and_oversized_events_never_expose_upstream_errors(data: bytes) -> None:
    with pytest.raises(RemoteUpstreamError) as raised:
        ChatStreamParser("model").parse(data)
    assert "secret" not in str(raised.value)
