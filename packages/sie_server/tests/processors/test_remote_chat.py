import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import msgpack
import pytest
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_client
from sie_server.processors.streaming import StreamingProcessor

MODEL = "queued/chat"
USAGE = {"prompt_tokens": 37, "completion_tokens": 2, "total_tokens": 39}
TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}


def choice(content: str = "answer", *, index: int = 0, finish: str | None = None, **extra: Any) -> dict:
    return {"choices": [{"index": index, "delta": {"content": content}, "finish_reason": finish, **extra}]}


def events() -> list[dict | str]:
    return [choice(), choice("", finish="stop"), {"choices": [], "usage": USAGE}, "[DONE]"]


class ChatStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[dict | str], *, disconnect: bool = False, hold: bool = False) -> None:
        self.frames = frames
        self.disconnect = disconnect
        self.hold = hold
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self.frames:
            data = frame if isinstance(frame, str) else json.dumps(frame)
            wire = f"data: {data}\n\n".encode()
            yield wire[:5]
            yield wire[5:]
        self.started.set()
        if self.hold:
            await asyncio.Event().wait()
        if self.disconnect:
            raise httpx.ReadError("PRIVATE_UPSTREAM_DETAIL")

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(params=["sie", "openai"])
async def remote(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple]:
    cls = SieUpstreamAdapter if request.param == "sie" else OpenAIUpstreamAdapter
    upstream = Upstream.model_validate(
        {
            "kind": request.param,
            "base_url": "http://127.0.0.1:8088/prefix",
            **({"endpoints": ["chat"]} if request.param == "openai" else {}),
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"queued-chat": upstream})
    adapter = cls(upstream="queued-chat", upstream_model="operator/model")
    adapter.load("cpu")
    config = ModelConfig.model_validate(
        {
            "sie_id": MODEL,
            "remote_backed": True,
            "inputs": {"text": True},
            "tasks": {
                "generate": {
                    "context_length": 4096,
                    "max_output_tokens": 64,
                    "capabilities": {"tools": True, "grammar": ["json_schema", "regex"]},
                }
            },
            "profiles": {
                "default": {
                    "adapter_path": f"{cls.__module__}:{cls.__name__}",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "queued-chat", "upstream_model": "operator/model"}},
                }
            },
        }
    )
    registry = MagicMock()
    registry.device = "cpu"
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.get_config.return_value = config
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    monkeypatch.setattr(
        proc, "_render_chat_template", AsyncMock(side_effect=AssertionError("remote chat loaded a local template"))
    )
    monkeypatch.setattr(
        proc, "_check_context_length", AsyncMock(side_effect=AssertionError("remote chat loaded a local tokenizer"))
    )
    requests: list[httpx.Request] = []
    try:
        yield adapter, proc, nc, requests
    finally:
        await adapter.aclose_client()
        adapter.unload()
        install_upstreams({})


def respond(
    remote: tuple,
    *,
    frames: list[dict | str] | None = None,
    payload: dict | None = None,
    status: int = 200,
    stream: ChatStream | None = None,
) -> ChatStream | None:
    adapter, _proc, _nc, requests = remote
    if payload is None:
        stream = stream or ChatStream(events() if frames is None else frames)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            headers={
                "content-type": "application/json" if payload is not None else "text/event-stream",
                "retry-after": "19",
            },
            stream=httpx.ByteStream(json.dumps(payload).encode()) if payload is not None else stream,
        )

    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(handler))
    return stream


async def run(remote: tuple, **fields: Any) -> tuple[AsyncMock, list[dict]]:
    _adapter, proc, nc, _requests = remote
    msg = AsyncMock()
    msg.data = msgpack.packb(
        {
            "request_id": "req-1",
            "work_item_id": "req-1.0",
            "item_index": 0,
            "total_items": 1,
            "operation": "generate",
            "model_id": MODEL,
            "profile_id": "default",
            "pool_name": "default",
            "router_id": "router-1",
            "reply_subject": "_INBOX.router-1.req-1",
            "timestamp": time.time(),
            "generate": {
                "messages": [{"role": "user", "content": "question"}],
                "max_new_tokens": 9,
                "stream": True,
                **fields,
            },
        },
        use_bin_type=True,
    )
    await proc.process(msg, MODEL)
    chunks = [msgpack.unpackb(call.args[1], raw=False) for call in nc.publish.await_args_list]
    return msg, chunks


async def test_queue_chat_pins_model_preserves_usage_and_closes(remote: tuple) -> None:
    stream = respond(remote)
    msg, chunks = await run(remote, seed=4)
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == "answer"
    assert chunks[-1]["done"]
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == USAGE
    assert json.loads(remote[3][0].content)["model"] == "operator/model"
    assert json.loads(remote[3][0].content)["seed"] == 4
    msg.ack.assert_awaited_once()
    msg.nak.assert_not_awaited()
    assert stream is not None
    assert stream.closed
    assert remote[1].in_flight_count() == 0
    assert upstream_limiter("queued-chat")._in_flight == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 3),
        ("min_tokens", 1),
        ("repetition_penalty", 1.1),
        ("chat_template_kwargs", {"enable_thinking": False}),
        ("best_of", 2),
    ],
)
async def test_unsupported_chat_fields_refuse_without_dispatch(remote: tuple, field: str, value: Any) -> None:
    respond(remote)
    msg, chunks = await run(remote, **{field: value})
    assert chunks[-1]["error"]["code"] == "unsupported_field"
    assert remote[3] == []
    msg.ack.assert_awaited_once()


@pytest.mark.parametrize(
    ("status", "code"), [(429, "RESOURCE_EXHAUSTED"), (503, "RESOURCE_EXHAUSTED"), (400, "inference_error")]
)
async def test_remote_refusal_is_single_attempt_and_sanitized(remote: tuple, status: int, code: str) -> None:
    respond(remote, status=status, payload={"error": {"message": "PRIVATE_UPSTREAM_DETAIL"}})
    msg, chunks = await run(remote)
    assert chunks[-1]["error"]["code"] == code
    assert "PRIVATE" not in json.dumps(chunks)
    assert len(remote[3]) == 1
    msg.ack.assert_awaited_once()


@pytest.mark.parametrize("missing", ["done", "usage", "finish"])
async def test_truncated_chat_never_settles_as_success(remote: tuple, missing: str) -> None:
    frames = events()
    del frames[{"done": 3, "usage": 2, "finish": 1}[missing]]
    stream = respond(remote, frames=frames)
    msg, chunks = await run(remote)
    assert chunks[-1]["finish_reason"] == "error"
    assert chunks[-1]["error"]["code"] == "inference_error"
    msg.ack.assert_awaited_once()
    assert stream is not None
    assert stream.closed


def running(completion_tokens: int) -> dict:
    return {"prompt_tokens": 37, "completion_tokens": completion_tokens, "total_tokens": 37 + completion_tokens}


async def test_running_usage_on_every_chunk_settles_on_the_last_final_usage(remote: tuple) -> None:
    frames: list[dict | str] = [
        {**choice(""), "usage": running(0)},
        {**choice(), "usage": running(1)},
        {**choice("", finish="stop"), "usage": running(1)},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]
    respond(remote, frames=frames)
    msg, chunks = await run(remote)
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == "answer"
    assert chunks[-1]["done"]
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == USAGE
    msg.ack.assert_awaited_once()


async def test_running_usage_without_a_final_usage_never_settles(remote: tuple) -> None:
    frames: list[dict | str] = [{**choice(), "usage": running(1)}, choice("", finish="stop"), "[DONE]"]
    respond(remote, frames=frames)
    _msg, chunks = await run(remote)
    assert chunks[-1]["finish_reason"] == "error"
    assert chunks[-1]["error"]["code"] == "inference_error"


async def test_cancel_closes_upstream_and_settles_once(remote: tuple) -> None:
    stream = respond(remote, stream=ChatStream([choice()], hold=True))
    task = asyncio.create_task(run(remote))
    assert stream is not None
    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert remote[1].signal_cancel("req-1")
    msg, chunks = await asyncio.wait_for(task, timeout=2)
    assert chunks[-1]["finish_reason"] == "cancelled"
    assert sum(chunk.get("done", False) for chunk in chunks) == 1
    assert stream is not None
    assert stream.closed
    msg.ack.assert_awaited_once()
    assert remote[1].in_flight_count() == 0


async def test_strict_schema_is_sent_and_finished_output_verified(remote: tuple) -> None:
    respond(remote)
    grammar = {"kind": "json_schema", "value": {"type": "object"}, "strict": True, "label": "result"}
    _msg, chunks = await run(remote, grammar=grammar)
    assert chunks[-1]["error"]["code"] == "MODEL_OUTPUT_PARSE_ERROR"
    body = json.loads(remote[3][0].content)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "result", "schema": {"type": "object"}, "strict": True},
    }


async def test_regex_refuses_without_dispatch(remote: tuple) -> None:
    respond(remote)
    _msg, chunks = await run(remote, grammar={"kind": "regex", "value": "[a-z]+"})
    assert chunks[-1]["error"]["code"] == "unsupported_field"
    assert remote[3] == []


async def test_tool_fragments_and_required_choice_preserved(remote: tuple) -> None:
    frames = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-a",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{"},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]
    respond(remote, frames=frames)
    msg, chunks = await run(remote, tools=[TOOL], tool_choice="required")
    assert chunks[-1]["finish_reason"] == "tool_calls"
    tool_chunks = [chunk for chunk in chunks if chunk.get("tool_calls")]
    assert tool_chunks[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert "".join(chunk["tool_calls"][0]["function"].get("arguments", "") for chunk in tool_chunks) == "{}"
    msg.ack.assert_awaited_once()


async def test_required_tool_omitted_is_terminal_error(remote: tuple) -> None:
    respond(remote)
    _msg, chunks = await run(remote, tools=[TOOL], tool_choice="required")
    assert chunks[-1]["finish_reason"] == "error"


@pytest.mark.parametrize("streaming", [False, True])
async def test_multiple_choices_preserve_indices_and_exact_usage(remote: tuple, streaming: bool) -> None:
    if streaming:
        frames = [
            choice("first", index=0),
            choice("second", index=1),
            choice("", index=1, finish="length"),
            choice("", index=0, finish="stop"),
            {"choices": [], "usage": USAGE},
            "[DONE]",
        ]
        respond(remote, frames=frames)
    else:
        respond(
            remote,
            payload={
                "choices": [
                    {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "length"},
                    {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
                ],
                "usage": USAGE,
            },
        )
    msg, chunks = await run(remote, n=2, stream=streaming)
    assert chunks[-1]["done"]
    assert chunks[-1]["usage"] == USAGE
    if streaming:
        assert {chunk.get("choice_index", 0) for chunk in chunks if not chunk.get("done")} == {0, 1}
    else:
        assert [item["text"] for item in chunks[-1]["candidates"]] == ["first", "second"]
    msg.ack.assert_awaited_once()


async def test_private_reasoning_logprobs_do_not_reach_queue(remote: tuple) -> None:
    frame = choice(
        "answer", logprobs={"content": [{"token": "PRIVATE", "logprob": -0.1, "bytes": None, "top_logprobs": []}]}
    )
    frame["choices"][0]["delta"]["reasoning_content"] = "PRIVATE"
    respond(remote, frames=[frame, *events()[1:]])
    _msg, chunks = await run(remote, logprobs=True)
    assert "PRIVATE" not in json.dumps(chunks)


async def test_buffered_chat_preserves_tools_and_exact_usage(remote: tuple) -> None:
    respond(
        remote,
        payload={
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": USAGE,
        },
    )
    msg, chunks = await run(remote, stream=False, tools=[TOOL], tool_choice="required")
    assert chunks[-1]["usage"] == USAGE
    assert chunks[-1]["finish_reason"] == "tool_calls"
    assert next(chunk for chunk in chunks if chunk.get("tool_calls"))["tool_calls"][0]["function"]["name"] == "lookup"
    msg.ack.assert_awaited_once()


@pytest.mark.parametrize(
    ("code", "expected"), [("INVALID_INPUT", "invalid_request"), ("INPUT_TOO_LONG", "INPUT_TOO_LONG")]
)
async def test_known_input_refusal_preserves_type(remote: tuple, code: str, expected: str) -> None:
    respond(remote, status=400, payload={"error": {"code": code, "message": "PRIVATE"}})
    _msg, chunks = await run(remote)
    assert chunks[-1]["error"]["code"] == expected
    assert len(remote[3]) == 1
    assert "PRIVATE" not in json.dumps(chunks)


async def test_post_output_disconnect_is_final(remote: tuple) -> None:
    stream = respond(remote, stream=ChatStream([choice()], disconnect=True))
    _msg, chunks = await run(remote)
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == "answer"
    assert chunks[-1]["error"]["code"] == "inference_error"
    assert "PRIVATE" not in json.dumps(chunks)
    assert len(remote[3]) == 1
    assert stream is not None
    assert stream.closed


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 4097, "completion_tokens": 2, "total_tokens": 4099},
        {"prompt_tokens": 37, "completion_tokens": 10, "total_tokens": 47},
    ],
)
async def test_usage_over_context_or_output_cap_fails_closed(remote: tuple, usage: dict) -> None:
    frames = events()
    frames[2] = {"choices": [], "usage": usage}
    respond(remote, frames=frames)
    _msg, chunks = await run(remote)
    assert chunks[-1]["finish_reason"] == "error"


async def test_unrequested_tool_is_rejected(remote: tuple) -> None:
    respond(
        remote,
        frames=[
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-a",
                                    "type": "function",
                                    "function": {"name": "unrequested", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        ],
    )
    _msg, chunks = await run(remote, tools=[TOOL])
    assert chunks[-1]["finish_reason"] == "error"
    assert not any(chunk.get("tool_calls") for chunk in chunks)


@pytest.mark.parametrize(("fragments", "success"), [(["look", "up"], True), (["lookup", "lookup"], False)])
async def test_tool_name_fragments_validate_the_accumulated_name(
    remote: tuple, fragments: list[str], success: bool
) -> None:
    frames = []
    for position, fragment in enumerate(fragments):
        tool: dict[str, Any] = {"index": 0, "function": {"name": fragment, "arguments": "{}" if position == 0 else ""}}
        if position == 0:
            tool.update({"id": "call-a", "type": "function"})
        frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [tool]}, "finish_reason": None}]})
    frames += [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]
    respond(remote, frames=frames)
    _msg, chunks = await run(remote, tools=[TOOL], tool_choice="required")
    assert chunks[-1]["finish_reason"] == ("tool_calls" if success else "error")
    names = "".join(chunk["tool_calls"][0]["function"].get("name", "") for chunk in chunks if chunk.get("tool_calls"))
    assert names == "lookup"


@pytest.mark.parametrize("content", [None, "final text"])
async def test_multi_choice_tools_arrive_before_their_finish(remote: tuple, content: str | None) -> None:
    frames = []
    for index in range(2):
        frames.append(
            {
                "choices": [
                    {
                        "index": index,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
    if content is not None:
        for frame in frames:
            frame["choices"][0]["delta"]["content"] = content
    frames += [{"choices": [], "usage": USAGE}, "[DONE]"]
    respond(remote, frames=frames)
    msg, chunks = await run(remote, tools=[TOOL], tool_choice="required", n=2)
    for index in range(2):
        per_choice = [chunk for chunk in chunks if not chunk.get("done") and chunk.get("choice_index", 0) == index]
        tool_chunk = next(chunk for chunk in per_choice if chunk.get("tool_calls"))
        assert tool_chunk["tool_calls"][0]["function"]["arguments"] == "{}"
        assert per_choice[-1]["finish_reason"] == "tool_calls"
        assert sum(chunk.get("finish_reason") is not None for chunk in per_choice) == 1
    assert chunks[-1]["finish_reason"] == "tool_calls"
    msg.ack.assert_awaited_once()


@pytest.mark.parametrize("remote", ["openai"], indirect=True)
async def test_onboarded_raw_completions_use_local_template(remote: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, proc, _nc, _requests = remote
    assert isinstance(adapter, OpenAIUpstreamAdapter)
    assert adapter._upstream is not None
    upstream = adapter._upstream.model_copy(update={"endpoints": frozenset({"chat", "completions"})})
    adapter.unload()
    install_upstreams({"queued-chat": upstream})
    adapter.load("cpu")
    config_data = proc._registry.get_config(MODEL).model_dump()
    config_data.update({"remote_backed": False, "hf_id": "local/model"})
    proc._registry.get_config.return_value = ModelConfig.model_validate(config_data)
    monkeypatch.setattr(proc, "_get_tokenizer", AsyncMock(return_value=MagicMock(chat_template="local-template")))
    render = AsyncMock(return_value="LOCALLY RENDERED PROMPT")
    monkeypatch.setattr(proc, "_render_chat_template", render)
    monkeypatch.setattr(proc, "_check_context_length", AsyncMock(return_value=None))
    frames = [
        {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]},
        {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]
    respond(remote, frames=frames)
    _msg, chunks = await run(remote)
    assert chunks[-1]["finish_reason"] == "stop"
    assert remote[3][0].url.path.endswith("/completions")
    assert not remote[3][0].url.path.endswith("/chat/completions")
    assert json.loads(remote[3][0].content)["prompt"] == "LOCALLY RENDERED PROMPT"
    render.assert_awaited_once()


async def test_onboarded_without_local_template_uses_chat(remote: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, proc, _nc, _requests = remote
    if isinstance(adapter, OpenAIUpstreamAdapter):
        assert adapter._upstream is not None
        upstream = adapter._upstream.model_copy(update={"endpoints": frozenset({"chat", "completions"})})
        adapter.unload()
        install_upstreams({"queued-chat": upstream})
        adapter.load("cpu")
    config_data = proc._registry.get_config(MODEL).model_dump()
    config_data.update({"remote_backed": False, "hf_id": "local/model"})
    proc._registry.get_config.return_value = ModelConfig.model_validate(config_data)
    tokenizer = AsyncMock(side_effect=RuntimeError("no local tokenizer"))
    monkeypatch.setattr(proc, "_get_tokenizer", tokenizer)
    respond(remote)
    _msg, chunks = await run(remote)
    assert chunks[-1]["finish_reason"] == "stop"
    assert remote[3][0].url.path.endswith("/chat/completions")
    tokenizer.assert_awaited_once()


async def test_onboarded_strict_tools_keep_declared_chat(remote: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, proc, _nc, requests = remote
    if isinstance(adapter, OpenAIUpstreamAdapter):
        assert adapter._upstream is not None
        upstream = adapter._upstream.model_copy(update={"endpoints": frozenset({"chat", "completions"})})
        adapter.unload()
        install_upstreams({"queued-chat": upstream})
        adapter.load("cpu")
    config_data = proc._registry.get_config(MODEL).model_dump()
    config_data.update({"remote_backed": False, "hf_id": "local/model"})
    proc._registry.get_config.return_value = ModelConfig.model_validate(config_data)
    tokenizer = AsyncMock(side_effect=RuntimeError("no local tokenizer"))
    monkeypatch.setattr(proc, "_get_tokenizer", tokenizer)
    tool = {**TOOL, "function": {**TOOL["function"], "strict": True}}
    respond(remote)
    _msg, chunks = await run(remote, tools=[tool])
    assert chunks[-1]["finish_reason"] == "stop"
    assert requests[0].url.path.endswith("/chat/completions")
    assert json.loads(requests[0].content)["tools"][0]["function"]["strict"] is True
    counts = isinstance(adapter, OpenAIUpstreamAdapter)
    assert tokenizer.await_count == int(counts), "only an OpenAI upstream consults the tokenizer, to count"


@pytest.mark.parametrize("remote", ["sie"], indirect=True)
async def test_onboarded_native_sie_preserves_already_suppressed_answer(remote: tuple, monkeypatch) -> None:
    _adapter, proc, _nc, requests = remote
    config_data = proc._registry.get_config(MODEL).model_dump()
    config_data.update({"remote_backed": False, "hf_id": "local/model"})
    config_data["tasks"]["generate"]["chat_template_kwargs"] = {"enable_thinking": False}
    proc._registry.get_config.return_value = ModelConfig.model_validate(config_data)
    monkeypatch.setattr(proc, "_get_tokenizer", AsyncMock(return_value=MagicMock(chat_template="local-template")))
    monkeypatch.setattr(proc, "_render_chat_template", AsyncMock(return_value="LOCALLY RENDERED<think>"))
    monkeypatch.setattr(proc, "_check_context_length", AsyncMock(return_value=None))
    respond(
        remote,
        frames=[{"text_delta": "answer", "done": False}, {"done": True, "finish_reason": "stop", "usage": USAGE}],
    )
    _msg, chunks = await run(remote)
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == "answer"
    assert chunks[-1]["finish_reason"] == "stop"
    assert requests[0].url.path.endswith("/generate/operator__model")
