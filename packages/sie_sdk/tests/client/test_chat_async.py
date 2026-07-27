"""Tests for the async chat-completions + streaming SDK surface.

Mocks the aiohttp session; exercises the async mirror of ``test_chat.py``:
buffered ``chat_completions``, ``stream_chat_completions``, ``stream_generate``,
mid-stream error -> ``ServerError``, and a 503 PROVISIONING pre-stream retry.
"""

from __future__ import annotations

import json
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from sie_sdk import SIEAsyncClient
from sie_sdk.client.errors import RequestError, ServerError


class _FakeRaw:
    """Stand-in for an aiohttp response used as an async context manager."""

    def __init__(
        self,
        *,
        status: int = 200,
        line_bytes: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
        body: Any | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {"content-type": "text/event-stream"}
        self._line_bytes = line_bytes or []
        self._body = {} if body is None else body
        # ``content`` async-iterates byte lines, mirroring aiohttp.StreamReader.
        self.content = self._aiter_bytes()

    async def _aiter_bytes(self):
        for b in self._line_bytes:
            yield b

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


def _sse_bytes(*chunks: dict[str, Any]) -> list[bytes]:
    out: list[bytes] = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n".encode())
        out.append(b"\n")
    out.append(b"data: [DONE]\n")
    return out


def _chat_chunk(content: str, *, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish, "logprobs": None}],
    }


def _patch_session(client: SIEAsyncClient, *, post_returns=None, post_side_effect=None) -> MagicMock:
    """Install a mock aiohttp session whose ``post`` returns _FakeRaw context managers."""
    session = MagicMock()
    if post_side_effect is not None:
        session.post = MagicMock(side_effect=post_side_effect)
    else:
        session.post = MagicMock(return_value=post_returns)
    session.close = AsyncMock()
    client._session = session
    return session


@pytest.mark.asyncio
async def test_async_chat_completions_parses_and_sends_json() -> None:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop", "logprobs": None}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(
        client,
        post_returns=_FakeRaw(
            status=200,
            body=payload,
            headers={
                "x-sie-request-id": "req-chat",
                "x-sie-units-output-tokens": "1",
                "x-sie-credits-debited": "7",
            },
        ),
    )
    out = await client.chat_completions("m", [{"role": "user", "content": "hi"}], max_completion_tokens=16)
    assert out["choices"][0]["message"]["content"] == "Hi"
    assert out["request"] == {
        "id": "req-chat",
        "usage": {"output_tokens": 1},
        "credits_debited": 7,
    }
    call = session.post.call_args
    assert call.args[0] == "/v1/chat/completions"
    sent = json.loads(call.kwargs["data"].decode("utf-8"))
    assert sent["model"] == "m"
    assert "stream" not in sent
    assert call.kwargs["headers"]["accept"] == "application/json"
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_chat_yields_chunks() -> None:
    raw = _FakeRaw(status=200, line_bytes=_sse_bytes(_chat_chunk("He"), _chat_chunk("llo", finish="stop")))
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=raw)
    out = [c async for c in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}])]
    assert [c["choices"][0]["delta"].get("content") for c in out] == ["He", "llo"]
    sent = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert sent["stream"] is True
    assert session.post.call_args.kwargs["headers"]["accept"] == "text/event-stream"
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_generate_yields_and_normalizes_path() -> None:
    token_logprobs = [{"token": "He", "logprob": -0.25, "top_logprobs": []}]
    chunk0 = {
        "request_id": "r",
        "seq": 0,
        "text_delta": "He",
        "logprobs": token_logprobs,
        "done": False,
    }
    term = {
        "request_id": "r",
        "seq": 1,
        "text_delta": "llo",
        "done": True,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(chunk0, term)))
    out = [
        c
        async for c in client.stream_generate(
            "Qwen/Qwen3-4B-Instruct",
            "hi",
            max_new_tokens=8,
            images=[{"data": b"\x89PNG\r\n\x1a\npayload", "format": "png"}],
            frequency_penalty=0.25,
            presence_penalty=-0.5,
            grammar={"regex": "[a-z]+", "label": None, "strict": None},
            seed=-3,
            logit_bias={"123": 1.5},
            logprobs=True,
            top_logprobs=3,
            routing_key="tenant-7",
            prompt_cache_key="prompt-9",
            safety_identifier="safety-3",
            lora_adapter="sql-adapter",
            extra_body={"prompt": "wrong", "max_new_tokens": 999, "stream": False},
        )
    ]
    assert "".join(c["text_delta"] for c in out) == "Hello"
    assert out[0]["logprobs"] == token_logprobs
    assert out[-1]["done"] is True
    assert session.post.call_args.args[0] == "/v1/generate/Qwen__Qwen3-4B-Instruct"
    sent = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert sent == {
        "prompt": "hi",
        "max_new_tokens": 8,
        "images": [{"data": "iVBORw0KGgpwYXlsb2Fk", "format": "png"}],
        "stream": True,
        "frequency_penalty": 0.25,
        "presence_penalty": -0.5,
        "grammar": {"regex": "[a-z]+", "label": None, "strict": None},
        "seed": -3,
        "logit_bias": {"123": 1.5},
        "top_logprobs": 3,
        "routing_key": "tenant-7",
        "prompt_cache_key": "prompt-9",
        "safety_identifier": "safety-3",
        "lora_adapter": "sql-adapter",
        "logprobs": True,
    }
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_generate_validates_extra_body_grammar_before_request() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=_FakeRaw())
    with pytest.raises(ValueError, match="exactly one"):
        _ = [
            chunk
            async for chunk in client.stream_generate(
                "m",
                "hi",
                max_new_tokens=8,
                extra_body={"grammar": {"regex": "x", "ebnf": 'root ::= "x"'}},
            )
        ]
    session.post.assert_not_called()
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_raises_on_error_chunk() -> None:
    err = {
        "request_id": "r",
        "seq": 0,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": {"code": "inference_error", "message": "boom"},
    }
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]
    assert ei.value.code == "inference_error"
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_chat_retries_503_provisioning_then_streams() -> None:
    s503 = _FakeRaw(
        status=503,
        headers={"Retry-After": "0.01", "content-type": "application/json"},
        body={"error": {"code": "PROVISIONING", "message": "prov"}},
    )
    s200 = _FakeRaw(status=200, line_bytes=_sse_bytes(_chat_chunk("ok", finish="stop")))
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_side_effect=[s503, s200])
    out = [
        c
        async for c in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}], provision_timeout_s=5.0)
    ]
    assert [c["choices"][0]["delta"].get("content") for c in out] == ["ok"]
    assert session.post.call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_async_chat_completions_504_retains_terminal_request_metadata_without_retry() -> None:
    timeout = _FakeRaw(
        status=504,
        headers={
            "content-type": "application/json",
            "x-sie-request-id": "req-async-chat-timeout",
            "x-sie-units-output-tokens": "5",
            "x-sie-credits-debited": "13",
        },
        body={"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}},
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=timeout)

    with pytest.raises(ServerError) as excinfo:
        await client.chat_completions("m", [{"role": "user", "content": "hi"}])

    assert session.post.call_count == 1
    assert excinfo.value.request == {
        "id": "req-async-chat-timeout",
        "usage": {"output_tokens": 5},
        "credits_debited": 13,
    }
    await client.close()


@pytest.mark.asyncio
async def test_async_chat_completions_non_dict_response_retains_request_metadata() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(
        client,
        post_returns=_FakeRaw(
            status=200,
            headers={
                "x-sie-request-id": "req-async-chat-malformed",
                "x-sie-units-output-tokens": "3",
                "x-sie-credits-debited": "8",
            },
            body=["not a dict"],
        ),
    )
    with pytest.raises(RequestError) as excinfo:
        await client.chat_completions("m", [{"role": "user", "content": "hi"}])
    assert excinfo.value.request == {
        "id": "req-async-chat-malformed",
        "usage": {"output_tokens": 3},
        "credits_debited": 8,
    }
    await client.close()


# M7 (async mirror of the sync test): every newly typed kwarg on the
# async chat-completion surface must land on the wire under its
# snake_case name.
@pytest.mark.asyncio
async def test_async_chat_completions_forwards_all_m7_typed_params() -> None:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop", "logprobs": None}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=_FakeRaw(status=200, body=payload))
    await client.chat_completions(
        "m",
        [{"role": "user", "content": "hi"}],
        n=2,
        best_of=4,
        logprobs=True,
        top_logprobs=5,
        lora_adapter="my-lora",
        top_k=40,
        repetition_penalty=1.1,
        logit_bias={"1234": 5.0, "9999": -7.5},
        user="end-user-42",
        safety_identifier="safety-tier-A",
        parallel_tool_calls=False,
        seed=42,
    )
    sent = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert sent["n"] == 2
    assert sent["best_of"] == 4
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 5
    assert sent["lora_adapter"] == "my-lora"
    assert sent["top_k"] == 40
    assert sent["repetition_penalty"] == 1.1
    assert sent["logit_bias"] == {"1234": 5.0, "9999": -7.5}
    assert sent["user"] == "end-user-42"
    assert sent["safety_identifier"] == "safety-tier-A"
    assert sent["parallel_tool_calls"] is False
    assert sent["seed"] == 42
    assert "stream" not in sent
    await client.close()


@pytest.mark.asyncio
async def test_async_chat_completions_extra_body_still_works() -> None:
    """Backwards-compat mirror of the sync test."""
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop", "logprobs": None}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=_FakeRaw(status=200, body=payload))
    await client.chat_completions(
        "m",
        [{"role": "user", "content": "hi"}],
        extra_body={"hypothetical_future_field": "future-value", "top_k": 99},
    )
    sent = json.loads(session.post.call_args.kwargs["data"].decode("utf-8"))
    assert sent["hypothetical_future_field"] == "future-value"
    assert sent["top_k"] == 99
    await client.close()
