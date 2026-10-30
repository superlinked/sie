import asyncio
import base64
import io
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import msgpack
import pytest
from PIL import Image
from sie_server.adapters._generation_base import GenerationChunk, ToolCallDelta, UpstreamTokenUsage
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.tokenizer import load_image_token_id
from sie_server.core.upstream_client import upstream_client
from sie_server.processors.hybrid_usage import HybridCount, count_hybrid_usage
from sie_server.processors.streaming import StreamingProcessor

MODEL = "local/hybrid"
IMAGE_TOKEN_ID = 151655
UPSTREAM_USAGE = {
    "prompt_tokens": 37,
    "completion_tokens": 9,
    "total_tokens": 46,
    "prompt_tokens_details": {"cached_tokens": 30},
}


async def _words(text: str) -> int:
    return len(text.split())


def _count(*, prompt_tokens: int = 2, completion_limit: int = 100) -> HybridCount:
    return HybridCount(prompt_tokens=prompt_tokens, completion_limit=completion_limit, count_tokens=_words)


class _Chunks:
    def __init__(self, chunks: list[GenerationChunk]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> "_Chunks":
        return self

    async def __anext__(self) -> GenerationChunk:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True


async def _drain(chunks: list[GenerationChunk], count: HybridCount) -> tuple[list[GenerationChunk], _Chunks]:
    source = _Chunks(chunks)
    return [chunk async for chunk in count_hybrid_usage(source, count)], source


def _terminal(**fields: Any) -> GenerationChunk:
    terminal: dict[str, Any] = {
        "text_delta": "",
        "done": True,
        "finish_reason": "stop",
        "prompt_tokens": 37,
        "completion_tokens": 9,
        "cached_tokens": 30,
    }
    return GenerationChunk(**{**terminal, **fields})


async def test_terminal_reports_the_worker_count_and_keeps_the_upstream_figure_beside_it() -> None:
    out, source = await _drain(
        [
            GenerationChunk(text_delta="", reasoning_delta="think one two "),
            GenerationChunk(text_delta="answer here", is_first=True),
            _terminal(),
        ],
        _count(),
    )
    assert [chunk.text_delta for chunk in out] == ["", "answer here", ""]
    assert all(not chunk.reasoning_delta for chunk in out)
    terminal = out[-1]
    assert (terminal.prompt_tokens, terminal.completion_tokens, terminal.cached_tokens) == (2, 5, None)
    assert terminal.upstream_usage == UpstreamTokenUsage(prompt_tokens=37, completion_tokens=9, cached_tokens=30)
    assert source.closed


async def test_reasoning_on_a_visible_chunk_is_counted_and_stripped() -> None:
    out, _ = await _drain(
        [GenerationChunk(text_delta="answer", reasoning_delta="why "), _terminal()],
        _count(),
    )
    assert out[0] == GenerationChunk(text_delta="answer")
    assert out[-1].completion_tokens == 2


async def test_tool_calls_and_candidates_are_counted_per_choice() -> None:
    call = ToolCallDelta(index=0, id="call-a", function_name="lookup", arguments_delta='{"q": "x"}')
    streamed, _ = await _drain(
        [GenerationChunk(text_delta="", tool_call_delta=call), _terminal(finish_reason="tool_calls")],
        _count(),
    )
    assert streamed[-1].completion_tokens == 2
    candidates = (
        {"text": "first answer", "finish_reason": "stop"},
        {
            "text": "",
            "finish_reason": "tool_calls",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
        },
    )
    buffered, _ = await _drain(
        [GenerationChunk(text_delta="", choice_index=1, reasoning_delta="private "), _terminal(candidates=candidates)],
        _count(),
    )
    assert buffered[-1].completion_tokens == 4


async def test_completion_is_bounded_and_never_zero_when_the_upstream_counted_one() -> None:
    bounded, _ = await _drain([GenerationChunk(text_delta="a b c d e"), _terminal()], _count(completion_limit=3))
    assert bounded[-1].completion_tokens == 3
    empty, _ = await _drain([_terminal()], _count())
    assert empty[-1].completion_tokens == 1


async def test_a_terminal_without_upstream_counts_passes_through() -> None:
    cancelled = GenerationChunk(text_delta="", done=True, finish_reason="cancelled")
    out, _ = await _drain([GenerationChunk(text_delta="partial"), cancelled], _count())
    assert out[-1] == cancelled


class WordTokenizer:
    chat_template = "template"

    def __init__(self, *, fail: bool = False, words: int | None = None) -> None:
        self.fail = fail
        self.words = words
        self.rendered_with: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool, **kwargs: Any):
        assert tokenize is False
        assert add_generation_prompt is True
        self.rendered_with.append(kwargs)
        if self.fail:
            raise ValueError("template rejects these messages")
        if self.words is not None:
            return "w " * self.words
        rendered = " ".join(_render_content(message["content"]) for message in messages)
        return rendered + (" tools" if kwargs.get("tools") else "") + " reply:"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [IMAGE_TOKEN_ID if word == "<|image_pad|>" else 1 for word in text.split()]


def _render_content(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    return " ".join(
        "<|vision_start|> <|image_pad|> <|vision_end|>" if part["type"] == "image" else part["text"] for part in content
    )


class ChatStream(httpx.AsyncByteStream):
    def __init__(self, frames: list[dict | str]) -> None:
        self.frames = frames
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self.frames:
            data = frame if isinstance(frame, str) else json.dumps(frame)
            yield f"data: {data}\n\n".encode()

    async def aclose(self) -> None:
        self.closed = True


def _frames() -> list[dict | str]:
    return [
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think one two "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "answer here"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": UPSTREAM_USAGE},
        "[DONE]",
    ]


def _answer() -> dict:
    message = {"role": "assistant", "content": "answer here", "reasoning_content": "think one two "}
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": UPSTREAM_USAGE}


def _processor(kind: str, endpoints: list[str]) -> tuple:
    cls = SieUpstreamAdapter if kind == "sie" else OpenAIUpstreamAdapter
    upstream = Upstream.model_validate(
        {
            "kind": kind,
            "base_url": "http://127.0.0.1:8088/prefix",
            **({"endpoints": endpoints} if kind == "openai" else {}),
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"hybrid-chat": upstream})
    adapter = cls(upstream="hybrid-chat", upstream_model="operator/model")
    adapter.load("cpu")
    config = ModelConfig.model_validate(
        {
            "sie_id": MODEL,
            "hf_id": "local/hybrid",
            "hf_revision": "a" * 40,
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64, "capabilities": {"tools": True}}},
            "profiles": {
                "default": {
                    "adapter_path": f"{cls.__module__}:{cls.__name__}",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "hybrid-chat", "upstream_model": "operator/model"}},
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
    tokenizer = WordTokenizer()
    proc._get_tokenizer = AsyncMock(return_value=tokenizer)  # type: ignore[method-assign]
    return adapter, proc, nc, [], tokenizer


async def _close(adapter: Any) -> None:
    await adapter.aclose_client()
    adapter.unload()
    install_upstreams({})


@pytest.fixture
async def hybrid() -> AsyncIterator[tuple]:
    setup = _processor("openai", ["chat"])
    try:
        yield setup
    finally:
        await _close(setup[0])


def respond(hybrid: tuple, *, frames: list[dict | str] | None = None, payload: dict | None = None) -> None:
    adapter, _proc, _nc, requests, _tokenizer = hybrid

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if payload is not None:
            return httpx.Response(
                200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(json.dumps(payload).encode())
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=ChatStream(frames or _frames())
        )

    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(handler))


async def run(hybrid: tuple, **fields: Any) -> list[dict]:
    _adapter, proc, nc, _requests, _tokenizer = hybrid
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
    msg.ack.assert_awaited_once()
    return [msgpack.unpackb(call.args[1], raw=False) for call in nc.publish.await_args_list]


@pytest.mark.parametrize("streaming", [True, False])
async def test_queued_upstream_chat_reports_the_worker_count(hybrid: tuple, streaming: bool) -> None:
    if streaming:
        respond(hybrid)
    else:
        respond(hybrid, payload=_answer())
    chunks = await run(hybrid, stream=streaming)
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == "answer here"
    assert "think" not in json.dumps(chunks)
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 5,
        "total_tokens": 7,
        "upstream_usage": {"prompt_tokens": 37, "completion_tokens": 9, "cached_tokens": 30},
    }
    sent = json.loads(hybrid[3][0].content)
    assert sent["messages"] == [{"role": "user", "content": "question"}]


async def test_private_upstream_reasoning_keeps_the_queued_stream_publishing(
    hybrid: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sie_server.processors.streaming._FLUSH_INTERVAL_S", 0)
    reasoning = [
        {"choices": [{"index": 0, "delta": {"reasoning_content": f"step {step} "}, "finish_reason": None}]}
        for step in range(3)
    ]
    respond(hybrid, frames=[*reasoning, *_frames()[1:]])
    chunks = await run(hybrid)
    answer = next(index for index, chunk in enumerate(chunks) if chunk.get("text_delta"))
    assert answer >= len(reasoning)
    assert all(chunk["text_delta"] == "" and not chunk["done"] for chunk in chunks[:answer])
    assert chunks[answer]["text_delta"] == "answer here"
    assert "step" not in json.dumps(chunks)


async def test_a_platform_upstream_reports_its_own_count() -> None:
    setup = _processor("sie", [])
    setup[1]._get_tokenizer.side_effect = AssertionError("a platform upstream counts with its own tokenizer")
    frames = [
        {"choices": [{"index": 0, "delta": {"content": "first"}, "finish_reason": "stop"}]},
        {"choices": [{"index": 1, "delta": {"content": "second"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": UPSTREAM_USAGE},
        "[DONE]",
    ]
    try:
        respond(setup, frames=frames)
        chunks = await run(setup, n=2)
    finally:
        await _close(setup[0])
    assert chunks[-1]["usage"] == UPSTREAM_USAGE


async def test_queued_upstream_chat_renders_tools_into_the_counted_prompt(hybrid: tuple) -> None:
    respond(hybrid)
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    chunks = await run(hybrid, tools=[tool])
    assert chunks[-1]["usage"]["prompt_tokens"] == 3


@pytest.mark.parametrize(
    ("tokenizer", "code"),
    [(WordTokenizer(fail=True), "invalid_request"), (WordTokenizer(words=4090), "context_exceeded")],
)
async def test_a_prompt_the_worker_cannot_count_is_refused_before_dispatch(
    hybrid: tuple, monkeypatch: pytest.MonkeyPatch, tokenizer: WordTokenizer, code: str
) -> None:
    monkeypatch.setattr(hybrid[1], "_get_tokenizer", AsyncMock(return_value=tokenizer))
    respond(hybrid)
    chunks = await run(hybrid)
    assert chunks[-1]["error"]["code"] == code
    assert hybrid[3] == []


async def test_without_a_local_tokenizer_the_upstream_counts_are_reported(hybrid: tuple) -> None:
    hybrid[1]._get_tokenizer.side_effect = RuntimeError("no local tokenizer")
    respond(hybrid)
    chunks = await run(hybrid)
    assert chunks[-1]["usage"] == UPSTREAM_USAGE
    assert "think" not in json.dumps(chunks)


async def test_a_remote_backed_model_reports_the_upstream_figure(hybrid: tuple) -> None:
    config = hybrid[1]._registry.get_config.return_value
    data = config.model_dump()
    data.update(remote_backed=True, hf_id=None, hf_revision=None)
    hybrid[1]._registry.get_config.return_value = ModelConfig.model_validate(data)
    hybrid[1]._get_tokenizer.side_effect = AssertionError("a remote-backed model has no local tokenizer")
    respond(hybrid)
    chunks = await run(hybrid)
    assert chunks[-1]["usage"] == UPSTREAM_USAGE


@pytest.mark.parametrize("tools", [False, True])
async def test_upstream_raw_completion_reports_the_worker_count(tools: bool) -> None:
    setup = _processor("openai", ["chat", "completions"])
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    frames = [
        {"choices": [{"index": 0, "text": "one two ", "finish_reason": None}]},
        {"choices": [{"index": 0, "text": "three", "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}},
        "[DONE]",
    ]
    try:
        respond(setup, frames=frames)
        chunks = await run(setup, **({"tools": [tool]} if tools else {}))
    finally:
        await _close(setup[0])
    requests = setup[3]
    prompt = "question tools reply:" if tools else "question reply:"
    assert requests[0].url.path == "/prefix/completions"
    assert json.loads(requests[0].content)["prompt"] == prompt
    assert chunks[-1]["usage"] == {
        "prompt_tokens": len(prompt.split()),
        "completion_tokens": 3,
        "total_tokens": len(prompt.split()) + 3,
        "upstream_usage": {"prompt_tokens": 11, "completion_tokens": 4},
    }


async def test_cancellation_closes_the_counted_upstream_chat(hybrid: tuple) -> None:
    started = asyncio.Event()

    class Held(ChatStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield f"data: {json.dumps(_frames()[1])}\n\n".encode()
            started.set()
            await asyncio.Event().wait()

    stream = Held([])
    adapter, proc = hybrid[0], hybrid[1]

    def handler(request: httpx.Request) -> httpx.Response:
        hybrid[3].append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(handler))
    task = asyncio.create_task(run(hybrid))
    await asyncio.wait_for(started.wait(), timeout=2)
    assert proc.signal_cancel("req-1")
    chunks = await asyncio.wait_for(task, timeout=2)
    assert chunks[-1]["finish_reason"] == "cancelled"
    assert "usage" not in chunks[-1]
    assert stream.closed


async def test_keep_reasoning_counts_it_and_leaves_it_on_the_chunks() -> None:
    source = _Chunks(
        [
            GenerationChunk(text_delta="", reasoning_delta="think one two "),
            GenerationChunk(text_delta="answer here", is_first=True),
            _terminal(),
        ]
    )

    out = [chunk async for chunk in count_hybrid_usage(source, _count(), keep_reasoning=True)]

    assert [chunk.reasoning_delta for chunk in out] == ["think one two ", "", ""]
    assert [chunk.text_delta for chunk in out] == ["", "answer here", ""]
    assert out[-1].completion_tokens == 5


def _declare(proc: StreamingProcessor, **generate: Any) -> None:
    data = proc._registry.get_config.return_value.model_dump()
    data["inputs"]["image"] = True
    data["tasks"]["generate"].update(generate)
    proc._registry.get_config.return_value = ModelConfig.model_validate(data)


async def test_queued_upstream_chat_renders_with_the_model_chat_template_kwargs(hybrid: tuple) -> None:
    _declare(hybrid[1], chat_template_kwargs={"enable_thinking": False})
    respond(hybrid)

    chunks = await run(hybrid)

    assert chunks[-1]["usage"]["prompt_tokens"] == 2
    assert json.loads(hybrid[3][0].content)["chat_template_kwargs"] == {"enable_thinking": False}
    assert hybrid[4].rendered_with == [{"enable_thinking": False}]


async def test_request_chat_template_kwargs_are_refused_before_dispatch(hybrid: tuple) -> None:
    _declare(hybrid[1], chat_template_kwargs={"enable_thinking": False})
    respond(hybrid)

    chunks = await run(hybrid, chat_template_kwargs={"enable_thinking": True})

    assert chunks[-1]["error"]["code"] == "unsupported_field"
    assert hybrid[3] == []


@pytest.mark.parametrize(
    ("declared", "expected"),
    [({"image_token_id": 7}, 7), ({"image_token_index": 9}, 9), ({"image_token_id": True}, None), ({}, None)],
)
def test_the_image_token_is_read_from_the_model_config(tmp_path: Path, declared: dict, expected: int | None) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2", **declared}))
    assert load_image_token_id(tmp_path) == expected


def _weights(proc: StreamingProcessor, tmp_path: Path, *, image_token_id: int | None = IMAGE_TOKEN_ID) -> None:
    declared = {} if image_token_id is None else {"image_token_id": image_token_id}
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2", **declared}))
    data = proc._registry.get_config.return_value.model_dump()
    data.update(hf_id=None, hf_revision=None, weights_path=str(tmp_path))
    data["inputs"]["image"] = True
    proc._registry.get_config.return_value = ModelConfig.model_validate(data)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = base64.b64encode(_png()).decode()
IMAGE_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
        {"type": "text", "text": "here"},
    ],
}
IMAGE_USAGE = {
    "prompt_tokens": 160,
    "completion_tokens": 9,
    "total_tokens": 169,
    "prompt_tokens_details": {"image_tokens": 120},
}


def _image_frames(usage: dict) -> list[dict | str]:
    return [*_frames()[:3], {"choices": [], "usage": usage}, "[DONE]"]


@pytest.mark.parametrize("streaming", [True, False])
async def test_an_image_prompt_counts_the_render_without_its_placeholder_plus_the_upstream_image_tokens(
    hybrid: tuple, tmp_path: Path, streaming: bool
) -> None:
    """The local template renders ``<|vision_start|> <|image_pad|> <|vision_end|>`` for the image.

    The upstream's ``image_tokens`` is the expanded ``image_token_id`` run, so
    the rendered pad is left out of SIE's count and the vision wrappers stay
    in it: ``look``, ``<|vision_start|>``, ``<|vision_end|>``, ``here`` and
    ``reply:`` are 5 tokens, plus 120 image tokens.
    """
    _weights(hybrid[1], tmp_path)
    if streaming:
        respond(hybrid, frames=_image_frames(IMAGE_USAGE))
    else:
        respond(hybrid, payload={**_answer(), "usage": IMAGE_USAGE})

    chunks = await run(hybrid, stream=streaming, messages=[IMAGE_MESSAGE])

    assert chunks[-1]["usage"] == {
        "prompt_tokens": 125,
        "completion_tokens": 5,
        "total_tokens": 130,
        "images": 1,
        "upstream_usage": {"prompt_tokens": 160, "completion_tokens": 9},
    }
    sent = json.loads(hybrid[3][0].content)
    assert sent["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                {"type": "text", "text": "here"},
            ],
        }
    ]


@pytest.mark.parametrize("image_tokens", [None, -1, "120", True, 161])
async def test_an_image_prompt_without_exact_upstream_image_tokens_fails_closed(
    hybrid: tuple, tmp_path: Path, image_tokens: Any
) -> None:
    _weights(hybrid[1], tmp_path)
    details = {} if image_tokens is None else {"prompt_tokens_details": {"image_tokens": image_tokens}}
    respond(hybrid, frames=_image_frames({**IMAGE_USAGE, "prompt_tokens_details": None, **details}))
    chunks = await run(hybrid, messages=[IMAGE_MESSAGE])

    respond(hybrid, frames=[*_frames()[:3], "[DONE]"])
    without_usage = await run(hybrid, messages=[IMAGE_MESSAGE])

    assert "usage" not in chunks[-1]
    assert chunks[-1]["error"] == without_usage[-1]["error"]
    assert chunks[-1]["finish_reason"] == without_usage[-1]["finish_reason"]


@pytest.mark.parametrize("image_token_id", [None, 42])
async def test_an_image_the_worker_cannot_place_in_its_count_is_refused_before_dispatch(
    hybrid: tuple, tmp_path: Path, image_token_id: int | None
) -> None:
    _weights(hybrid[1], tmp_path, image_token_id=image_token_id)
    respond(hybrid, frames=_image_frames(IMAGE_USAGE))

    chunks = await run(hybrid, messages=[IMAGE_MESSAGE])

    assert chunks[-1]["error"]["code"] == "unsupported_field"
    assert hybrid[3] == []


async def test_a_remote_backed_image_prompt_reports_the_upstream_figure(hybrid: tuple) -> None:
    data = hybrid[1]._registry.get_config.return_value.model_dump()
    data.update(remote_backed=True, hf_id=None, hf_revision=None)
    data["inputs"]["image"] = True
    hybrid[1]._registry.get_config.return_value = ModelConfig.model_validate(data)
    hybrid[1]._get_tokenizer.side_effect = AssertionError("a remote-backed model has no local tokenizer")
    respond(hybrid, frames=_image_frames(IMAGE_USAGE))

    chunks = await run(hybrid, messages=[IMAGE_MESSAGE])

    assert chunks[-1]["usage"] == {"prompt_tokens": 160, "completion_tokens": 9, "total_tokens": 169, "images": 1}


async def test_request_text_carrying_the_image_token_is_refused_before_dispatch(hybrid: tuple, tmp_path: Path) -> None:
    _weights(hybrid[1], tmp_path)
    respond(hybrid, frames=_image_frames(IMAGE_USAGE))
    message = {**IMAGE_MESSAGE, "content": [*IMAGE_MESSAGE["content"], {"type": "text", "text": " <|image_pad|>"}]}

    chunks = await run(hybrid, messages=[message])

    assert chunks[-1]["error"]["code"] == "invalid_request"
    assert chunks[-1]["error"]["message"] == "request text must not contain the model's image placeholder token"
    assert hybrid[3] == []
