"""Tests for the async chat-completions + streaming SDK surface.

Mocks the aiohttp session; exercises the async mirror of ``test_chat.py``:
buffered ``chat_completions``, ``stream_chat_completions``, ``stream_generate``,
mid-stream error -> ``ServerError``, and a 503 PROVISIONING pre-stream retry.
"""

from __future__ import annotations

import json
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient
from sie_sdk.client.errors import RequestError, ResourceExhaustedError, ServerError


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
async def test_async_chat_completions_consumes_modal_continuation_without_reposting() -> None:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = SIEAsyncClient("https://gateway.example.test")
    session = _patch_session(
        client,
        post_returns=_FakeRaw(
            status=303,
            body={},
            headers={"Location": "/v1/chat/completions?__modal_attempt_token=opaque"},
        ),
    )
    session.get = MagicMock(
        return_value=_FakeRaw(status=200, body=payload, headers={"content-type": "application/json"})
    )

    out = await client.chat_completions("m", [{"role": "user", "content": "hi"}])
    await client.close()

    assert out["choices"][0]["message"]["content"] == "Hi"
    assert session.post.call_count == 1
    assert session.get.call_args.args[0] == "/v1/chat/completions?__modal_attempt_token=opaque"


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
        "error": {"code": "unsupported_field", "message": "boom", "param": "top_k"},
    }
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]
    assert ei.value.code == "unsupported_field"
    assert ei.value.param == "top_k"
    assert ei.value.request == {"id": "r"}
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_generate_surfaces_empty_model_output_code_and_request_id() -> None:
    """#3136: async mirror — the typed ``empty_model_output`` terminal must
    surface its code and in-band request id to the caller.
    """
    err = {
        "request_id": "req-empty-1",
        "seq": 0,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        "error": {"code": "empty_model_output", "message": "model produced no visible output text"},
    }
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]
    assert ei.value.code == "empty_model_output"
    assert ei.value.request == {"id": "req-empty-1"}
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_generate_error_after_output_retains_request_id() -> None:
    delta = {"request_id": "req-mid-1", "seq": 0, "text_delta": "Hi", "done": False}
    err = {
        "request_id": "req-mid-1",
        "seq": 1,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": {"code": "unsupported_field", "message": "boom", "param": "top_k"},
    }
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(delta, err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]
    assert ei.value.code == "unsupported_field"
    assert ei.value.param == "top_k"
    assert ei.value.request == {"id": "req-mid-1"}
    await client.close()


def _chat_error_chunk(code: str, message: str, request_id: str) -> dict[str, Any]:
    """Chat-shape stream error chunk as the gateway emits it (#3136): the
    OpenAI envelope plus a top-level ``error`` block and the additive
    ``request_id`` member.
    """
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None, "logprobs": None}],
        "error": {"message": message, "type": "server_error", "param": None, "code": code},
        "request_id": request_id,
    }


def _capacity_error_chunk(surface: str, retry_after_s: object) -> dict[str, Any]:
    error: dict[str, Any] = {
        "message": "scheduler full",
        "type": "server_error",
        "param": "model",
        "code": "RESOURCE_EXHAUSTED",
        "retry_after_s": retry_after_s,
    }
    if surface == "chat":
        chunk = _chat_error_chunk("RESOURCE_EXHAUSTED", "scheduler full", "req-capacity")
        chunk["error"] = error
        return chunk
    return {
        "request_id": "req-capacity",
        "seq": 1,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": error,
    }


def _stream_surface(client: SIEAsyncClient, surface: str, *, wait_for_capacity: bool = True):
    if surface == "chat":
        return client.stream_chat_completions(
            "m",
            [{"role": "user", "content": "hi"}],
            wait_for_capacity=wait_for_capacity,
        )
    return client.stream_generate(
        "m",
        "hi",
        max_new_tokens=8,
        wait_for_capacity=wait_for_capacity,
    )


@pytest.mark.parametrize("surface", ["chat", "native"])
@pytest.mark.parametrize(
    ("retry_after_s", "expected"),
    [pytest.param(12, 12.0, id="valid"), pytest.param(True, None, id="malformed")],
)
@pytest.mark.asyncio
async def test_async_stream_capacity_give_up_preserves_validated_retry_hint(
    surface: str,
    retry_after_s: object,
    expected: float | None,
) -> None:
    error = _capacity_error_chunk(surface, retry_after_s)
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(error)))

    with pytest.raises(ResourceExhaustedError) as exc_info:
        _ = [chunk async for chunk in _stream_surface(client, surface, wait_for_capacity=False)]

    assert exc_info.value.code == "RESOURCE_EXHAUSTED"
    assert exc_info.value.param == "model"
    assert exc_info.value.request == {"id": "req-capacity"}
    assert exc_info.value.retry_after == expected
    await client.close()


@pytest.mark.parametrize("surface", ["chat", "native"])
@pytest.mark.parametrize(
    ("retry_after_s", "expected"),
    [pytest.param(12, 12.0, id="valid"), pytest.param(True, None, id="malformed")],
)
@pytest.mark.asyncio
async def test_async_stream_capacity_after_output_preserves_hint_without_retry(
    surface: str,
    retry_after_s: object,
    expected: float | None,
) -> None:
    delta = (
        _chat_chunk("partial")
        if surface == "chat"
        else {
            "request_id": "req-capacity",
            "seq": 0,
            "text_delta": "partial",
            "done": False,
        }
    )
    error = _capacity_error_chunk(surface, retry_after_s)
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(delta, error)))

    with pytest.raises(ServerError) as exc_info:
        _ = [chunk async for chunk in _stream_surface(client, surface)]

    assert exc_info.value.code == "RESOURCE_EXHAUSTED"
    assert exc_info.value.param == "model"
    assert exc_info.value.request == {"id": "req-capacity"}
    assert exc_info.value.retry_after == expected
    assert session.post.call_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_chat_error_chunk_surfaces_request_id() -> None:
    """#3136: async mirror — a chat-shape stream error must surface the
    in-band gateway request id.
    """
    err = _chat_error_chunk("first_chunk_timeout", "Generation aborted: first_chunk timeout", "req-chat-1")
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}])]
    assert ei.value.code == "first_chunk_timeout"
    assert ei.value.request == {"id": "req-chat-1"}
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_chat_error_after_output_retains_request_id() -> None:
    err = _chat_error_chunk("inference_error", "boom", "req-chat-2")
    client = SIEAsyncClient("http://localhost:8080")
    _patch_session(client, post_returns=_FakeRaw(status=200, line_bytes=_sse_bytes(_chat_chunk("Hi"), err)))
    with pytest.raises(ServerError) as ei:
        _ = [c async for c in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}])]
    assert ei.value.code == "inference_error"
    assert ei.value.request == {"id": "req-chat-2"}
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
async def test_async_stream_chat_retries_first_sse_model_loading_then_streams() -> None:
    loading = {
        "error": {"code": "MODEL_LOADING", "message": "loading"},
    }
    s200_loading = _FakeRaw(status=200, line_bytes=_sse_bytes(loading))
    s200_ok = _FakeRaw(status=200, line_bytes=_sse_bytes(_chat_chunk("ok", finish="stop")))
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_side_effect=[s200_loading, s200_ok])

    with patch("sie_sdk.client.async_.asyncio.sleep"):
        out = [
            c
            async for c in client.stream_chat_completions(
                "m", [{"role": "user", "content": "hi"}], provision_timeout_s=5.0
            )
        ]

    assert [c["choices"][0]["delta"].get("content") for c in out] == ["ok"]
    assert session.post.call_count == 2
    await client.close()


@pytest.mark.parametrize(
    ("code", "retry_after_s", "expected_hint", "expected_delay"),
    [
        pytest.param("RESOURCE_EXHAUSTED", 12, 12.0, 12.0, id="valid"),
        pytest.param("RESOURCE_EXHAUSTED", None, None, 0.25, id="missing"),
        pytest.param("RESOURCE_EXHAUSTED", True, None, 0.25, id="boolean"),
        pytest.param("RESOURCE_EXHAUSTED", 12.5, None, 0.25, id="fractional"),
        pytest.param("RESOURCE_EXHAUSTED", 61, None, 0.25, id="out-of-domain"),
        pytest.param("MODEL_LOADING", 12, None, 5.0, id="wrong-code"),
    ],
)
@pytest.mark.asyncio
async def test_async_stream_chat_first_sse_retry_hint_is_validated(
    code: str,
    retry_after_s: object,
    expected_hint: float | None,
    expected_delay: float,
) -> None:
    error: dict[str, Any] = {"code": code, "message": "capacity unavailable"}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    capacity = _FakeRaw(status=200, line_bytes=_sse_bytes({"error": error}))
    success = _FakeRaw(status=200, line_bytes=_sse_bytes(_chat_chunk("ok", finish="stop")))
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_side_effect=[capacity, success])

    def backoff(retry_after: float | None, attempt: int) -> float:
        assert retry_after == expected_hint
        assert attempt == 0
        return retry_after if retry_after is not None else 0.25

    with (
        patch("sie_sdk.client.async_.asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch("sie_sdk.client._shared.compute_oom_backoff", side_effect=backoff) as oom_backoff,
    ):
        out = [
            chunk
            async for chunk in client.stream_chat_completions(
                "m",
                [{"role": "user", "content": "hi"}],
                provision_timeout_s=60.0,
            )
        ]

    assert [chunk["choices"][0]["delta"].get("content") for chunk in out] == ["ok"]
    sleep.assert_awaited_once_with(expected_delay)
    if code == "RESOURCE_EXHAUSTED":
        oom_backoff.assert_called_once()
    else:
        oom_backoff.assert_not_called()
    assert session.post.call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_chat_does_not_retry_sse_model_loading_after_output() -> None:
    loading = {"error": {"code": "MODEL_LOADING", "message": "loading"}}
    partial_then_loading = _FakeRaw(
        status=200,
        line_bytes=_sse_bytes(_chat_chunk("partial"), loading),
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_session(client, post_returns=partial_then_loading)

    with pytest.raises(ServerError, match="loading"):
        _ = [c async for c in client.stream_chat_completions("m", [{"role": "user", "content": "hi"}])]

    assert session.post.call_count == 1
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
