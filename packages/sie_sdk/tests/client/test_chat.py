"""Tests for the sync chat-completions + streaming SDK surface.

Mocks the httpx layer; exercises:

- ``chat_completions`` buffered JSON round-trip + request shape.
- ``stream_chat_completions`` SSE chunk yielding + ``stream:true`` body.
- ``stream_generate`` SSE chunk yielding + HF-id path normalization.
- A mid-stream error chunk raises ``ServerError``.
- A 503 PROVISIONING pre-stream response is retried, then the stream is consumed.
"""

from __future__ import annotations

import json
from typing import Any, Self
from unittest.mock import MagicMock, patch

import pytest
from sie_sdk import SIEClient
from sie_sdk.client.errors import RequestError, ResourceExhaustedError, ServerError


def _ok_json(payload: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.content = json.dumps(payload).encode("utf-8")
    resp.json.return_value = payload
    return resp


class _FakeStream:
    """Stand-in for ``httpx.Client.stream(...)`` used as a context manager."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {"content-type": "text/event-stream"}
        self._json = json_body or {}
        self.content = json.dumps(self._json).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self):
        yield from self._lines

    def read(self) -> bytes:
        return self.content

    def json(self) -> dict[str, Any]:
        return self._json

    @property
    def text(self) -> str:
        return json.dumps(self._json)


def _sse(*chunks: dict[str, Any]) -> list[str]:
    """Render chunks as SSE lines terminated by ``[DONE]``."""
    lines: list[str] = []
    for c in chunks:
        lines.append(f"data: {json.dumps(c)}")
        lines.append("")
    lines.append("data: [DONE]")
    return lines


def _chat_chunk(content: str, *, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "system_fingerprint": None,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish, "logprobs": None}],
    }


def test_chat_completions_parses_and_sends_json_shape() -> None:
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
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.post.return_value = _ok_json(payload)
        mc.return_value.post.return_value.headers = {
            "x-sie-request-id": "req-chat",
            "x-sie-units-output-tokens": "1",
            "x-sie-credits-debited": "7",
        }
        client = SIEClient("http://localhost:8080")
        out = client.chat_completions("m", [{"role": "user", "content": "hi"}], max_completion_tokens=16)
        assert out["choices"][0]["message"]["content"] == "Hi"
        assert out["request"] == {
            "id": "req-chat",
            "usage": {"output_tokens": 1},
            "credits_debited": 7,
        }
        call = mc.return_value.post.call_args
        assert call.args[0] == "/v1/chat/completions"
        sent = json.loads(call.kwargs["content"].decode("utf-8"))
        assert sent["model"] == "m"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
        assert sent["max_completion_tokens"] == 16
        assert "stream" not in sent
        assert call.kwargs["headers"]["accept"] == "application/json"
        client.close()


def test_chat_completions_consumes_modal_continuation_without_reposting() -> None:
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    redirect = MagicMock(
        status_code=303,
        headers={"Location": "/v1/chat/completions?__modal_attempt_token=opaque"},
        content=b"",
    )
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.post.return_value = redirect
        mc.return_value.get.return_value = _ok_json(payload)
        client = SIEClient("https://gateway.example.test")
        out = client.chat_completions("m", [{"role": "user", "content": "hi"}])
        client.close()

    assert out["choices"][0]["message"]["content"] == "Hi"
    assert mc.return_value.post.call_count == 1
    assert mc.return_value.get.call_args.args[0] == "/v1/chat/completions?__modal_attempt_token=opaque"


def test_stream_chat_completions_yields_chunks_and_sets_stream_flag() -> None:
    lines = _sse(_chat_chunk("He"), _chat_chunk("llo", finish="stop"))
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=lines)
        client = SIEClient("http://localhost:8080")
        out = list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]))
        assert [c["choices"][0]["delta"].get("content") for c in out] == ["He", "llo"]
        call = mc.return_value.stream.call_args
        assert call.args[0] == "POST"
        assert call.args[1] == "/v1/chat/completions"
        sent = json.loads(call.kwargs["content"].decode("utf-8"))
        assert sent["stream"] is True
        assert call.kwargs["headers"]["accept"] == "text/event-stream"
        client.close()


def test_stream_generate_yields_chunks_and_normalizes_model_path() -> None:
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
        "ttft_ms": 5.0,
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(chunk0, term))
        client = SIEClient("http://localhost:8080")
        out = list(
            client.stream_generate(
                "Qwen/Qwen3-4B-Instruct",
                "hi",
                max_new_tokens=8,
                images=[{"data": b"\x89PNG\r\n\x1a\npayload", "format": "png"}],
                frequency_penalty=0.25,
                presence_penalty=-0.5,
                grammar={"regex": "[a-z]+", "label": None, "strict": None},
                seed=-2,
                logit_bias={"123": 1.5},
                logprobs=True,
                top_logprobs=3,
                routing_key="tenant-7",
                prompt_cache_key="prompt-9",
                safety_identifier="safety-3",
                lora_adapter="sql-adapter",
                extra_body={"prompt": "wrong", "max_new_tokens": 999, "stream": False},
            )
        )
        assert "".join(c["text_delta"] for c in out) == "Hello"
        assert out[0]["logprobs"] == token_logprobs
        assert out[-1]["done"] is True
        assert out[-1]["usage"]["completion_tokens"] == 2
        assert mc.return_value.stream.call_args.args[1] == "/v1/generate/Qwen__Qwen3-4B-Instruct"
        sent = json.loads(mc.return_value.stream.call_args.kwargs["content"].decode("utf-8"))
        assert sent == {
            "prompt": "hi",
            "max_new_tokens": 8,
            "images": [{"data": "iVBORw0KGgpwYXlsb2Fk", "format": "png"}],
            "stream": True,
            "frequency_penalty": 0.25,
            "presence_penalty": -0.5,
            "grammar": {"regex": "[a-z]+", "label": None, "strict": None},
            "seed": -2,
            "logit_bias": {"123": 1.5},
            "top_logprobs": 3,
            "routing_key": "tenant-7",
            "prompt_cache_key": "prompt-9",
            "safety_identifier": "safety-3",
            "lora_adapter": "sql-adapter",
            "logprobs": True,
        }
        client.close()


def test_stream_generate_validates_extra_body_grammar_before_request() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ValueError, match="exactly one"):
            list(
                client.stream_generate(
                    "m",
                    "hi",
                    max_new_tokens=8,
                    extra_body={"grammar": {"regex": "x", "ebnf": 'root ::= "x"'}},
                )
            )
        mc.return_value.stream.assert_not_called()
        client.close()


def test_stream_raises_server_error_on_error_chunk() -> None:
    err = {
        "request_id": "r",
        "seq": 0,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": {"code": "unsupported_field", "message": "boom", "param": "top_k"},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_generate("m", "hi", max_new_tokens=8))
        assert ei.value.code == "unsupported_field"
        assert ei.value.param == "top_k"
        assert ei.value.request == {"id": "r"}
        client.close()


def test_stream_generate_surfaces_empty_model_output_code_and_request_id() -> None:
    """#3136: the typed ``empty_model_output`` terminal (PR #3139) must reach
    the caller as a distinguishable code plus the in-band gateway request id —
    streamed responses carry no terminal headers, so the chunk is the only
    source.
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
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_generate("m", "hi", max_new_tokens=8))
        assert ei.value.code == "empty_model_output"
        assert ei.value.request == {"id": "req-empty-1"}
        client.close()


def test_stream_generate_drops_malformed_request_id() -> None:
    """A non-ASCII / padded in-band id must be dropped before it reaches the
    synthetic HTTP headers — HTTPX raises ``UnicodeEncodeError`` on non-ASCII
    header values, which would bypass the typed-error path entirely (#3136).
    """
    err = {
        "request_id": " r\u00e9q-bad ",
        "seq": 0,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": {"code": "empty_model_output", "message": "model produced no visible output text"},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_generate("m", "hi", max_new_tokens=8))
        assert ei.value.code == "empty_model_output"
        assert ei.value.request is None
        client.close()


def test_stream_generate_error_after_output_retains_request_id() -> None:
    delta = {"request_id": "req-mid-1", "seq": 0, "text_delta": "Hi", "done": False}
    err = {
        "request_id": "req-mid-1",
        "seq": 1,
        "text_delta": "",
        "done": True,
        "finish_reason": "error",
        "error": {"code": "unsupported_field", "message": "boom", "param": "top_k"},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(delta, err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_generate("m", "hi", max_new_tokens=8))
        assert ei.value.code == "unsupported_field"
        assert ei.value.param == "top_k"
        assert ei.value.request == {"id": "req-mid-1"}
        client.close()


def _chat_error_chunk(code: str, message: str, request_id: str) -> dict[str, Any]:
    """Chat-shape stream error chunk as the gateway emits it (#3136): the
    OpenAI envelope plus a top-level ``error`` block and the additive
    ``request_id`` member (the ``chatcmpl-*`` id is not the correlation key
    gateway logs use).
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


def _stream_surface(client: SIEClient, surface: str, *, wait_for_capacity: bool = True):
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
def test_stream_capacity_give_up_preserves_validated_retry_hint(
    surface: str,
    retry_after_s: object,
    expected: float | None,
) -> None:
    error = _capacity_error_chunk(surface, retry_after_s)
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(error))
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ResourceExhaustedError) as exc_info:
            list(_stream_surface(client, surface, wait_for_capacity=False))

        assert exc_info.value.code == "RESOURCE_EXHAUSTED"
        assert exc_info.value.param == "model"
        assert exc_info.value.request == {"id": "req-capacity"}
        assert exc_info.value.retry_after == expected
        client.close()


@pytest.mark.parametrize("surface", ["chat", "native"])
@pytest.mark.parametrize(
    ("retry_after_s", "expected"),
    [pytest.param(12, 12.0, id="valid"), pytest.param(True, None, id="malformed")],
)
def test_stream_capacity_after_output_preserves_hint_without_retry(
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
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(delta, error))
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError) as exc_info:
            list(_stream_surface(client, surface))

        assert exc_info.value.code == "RESOURCE_EXHAUSTED"
        assert exc_info.value.param == "model"
        assert exc_info.value.request == {"id": "req-capacity"}
        assert exc_info.value.retry_after == expected
        assert mc.return_value.stream.call_count == 1
        client.close()


def test_stream_chat_error_chunk_surfaces_request_id() -> None:
    """#3136: a chat-shape stream error before any output must surface the
    in-band gateway request id — streamed responses carry no terminal headers,
    so the chunk is the only source.
    """
    err = _chat_error_chunk("first_chunk_timeout", "Generation aborted: first_chunk timeout", "req-chat-1")
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]))
        assert ei.value.code == "first_chunk_timeout"
        assert ei.value.request == {"id": "req-chat-1"}
        client.close()


def test_stream_chat_error_after_output_retains_request_id() -> None:
    err = _chat_error_chunk("inference_error", "boom", "req-chat-2")
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = _FakeStream(lines=_sse(_chat_chunk("Hi"), err))
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as ei:
            list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]))
        assert ei.value.code == "inference_error"
        assert ei.value.request == {"id": "req-chat-2"}
        client.close()


def test_stream_chat_retries_503_provisioning_then_streams() -> None:
    s503 = _FakeStream(
        status_code=503,
        headers={"Retry-After": "0.01", "content-type": "application/json"},
        json_body={"error": {"code": "PROVISIONING", "message": "prov"}},
    )
    s200 = _FakeStream(lines=_sse(_chat_chunk("ok", finish="stop")))
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.side_effect = [s503, s200]
        client = SIEClient("http://localhost:8080")
        out = list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}], provision_timeout_s=5.0))
        assert [c["choices"][0]["delta"].get("content") for c in out] == ["ok"]
        assert mc.return_value.stream.call_count == 2
        assert client.last_retry_count == 1
        client.close()


def test_stream_chat_retries_first_sse_model_loading_then_streams() -> None:
    loading = {
        "error": {"code": "MODEL_LOADING", "message": "loading"},
    }
    s200_loading = _FakeStream(lines=_sse(loading))
    s200_ok = _FakeStream(lines=_sse(_chat_chunk("ok", finish="stop")))
    with patch("sie_sdk.client.sync.httpx.Client") as mc, patch("sie_sdk.client.sync.time.sleep"):
        mc.return_value.stream.side_effect = [s200_loading, s200_ok]
        client = SIEClient("http://localhost:8080")

        out = list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}], provision_timeout_s=5.0))

        assert [c["choices"][0]["delta"].get("content") for c in out] == ["ok"]
        assert mc.return_value.stream.call_count == 2
        assert client.last_retry_count == 1
        client.close()


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
def test_stream_chat_first_sse_retry_hint_is_validated(
    code: str,
    retry_after_s: object,
    expected_hint: float | None,
    expected_delay: float,
) -> None:
    error: dict[str, Any] = {"code": code, "message": "capacity unavailable"}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    capacity = _FakeStream(lines=_sse({"error": error}))
    success = _FakeStream(lines=_sse(_chat_chunk("ok", finish="stop")))

    def backoff(retry_after: float | None, attempt: int) -> float:
        assert retry_after == expected_hint
        assert attempt == 0
        return retry_after if retry_after is not None else 0.25

    with (
        patch("sie_sdk.client.sync.httpx.Client") as mc,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
        patch("sie_sdk.client._shared.compute_oom_backoff", side_effect=backoff) as oom_backoff,
    ):
        mc.return_value.stream.side_effect = [capacity, success]
        client = SIEClient("http://localhost:8080")

        out = list(
            client.stream_chat_completions(
                "m",
                [{"role": "user", "content": "hi"}],
                provision_timeout_s=60.0,
            )
        )

        assert [chunk["choices"][0]["delta"].get("content") for chunk in out] == ["ok"]
        sleep.assert_called_once_with(expected_delay)
        if code == "RESOURCE_EXHAUSTED":
            oom_backoff.assert_called_once()
        else:
            oom_backoff.assert_not_called()
        client.close()


def test_stream_chat_does_not_retry_sse_model_loading_after_output() -> None:
    loading = {"error": {"code": "MODEL_LOADING", "message": "loading"}}
    partial_then_loading = _FakeStream(lines=_sse(_chat_chunk("partial"), loading))
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.stream.return_value = partial_then_loading
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError, match="loading"):
            list(client.stream_chat_completions("m", [{"role": "user", "content": "hi"}]))

        assert mc.return_value.stream.call_count == 1
        assert client.last_retry_count == 0
        client.close()


def test_chat_completions_reports_pre_execution_retry() -> None:
    retry = MagicMock()
    retry.status_code = 503
    retry.headers = {"Retry-After": "0.01", "content-type": "application/json"}
    retry.json.return_value = {"error": {"code": "PROVISIONING", "message": "prov"}}
    retry.content = json.dumps(retry.json.return_value).encode()
    payload = {
        "model": "m",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    with patch("sie_sdk.client.sync.httpx.Client") as mc, patch("sie_sdk.client.sync.time.sleep"):
        mc.return_value.post.side_effect = [retry, _ok_json(payload)]
        client = SIEClient("http://localhost:8080")

        client.chat_completions("m", [{"role": "user", "content": "hi"}], provision_timeout_s=5.0)

        assert client.last_retry_count == 1
        client.close()


def test_chat_completions_504_retains_terminal_request_metadata_without_retry() -> None:
    timeout = MagicMock()
    timeout.status_code = 504
    timeout.headers = {
        "content-type": "application/json",
        "x-sie-request-id": "req-chat-timeout",
        "x-sie-units-output-tokens": "5",
        "x-sie-credits-debited": "13",
    }
    timeout.json.return_value = {"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}}
    timeout.content = json.dumps(timeout.json.return_value).encode()

    with patch("sie_sdk.client.sync.httpx.Client") as mc, patch("sie_sdk.client.sync.time.sleep") as sleep:
        mc.return_value.post.return_value = timeout
        client = SIEClient("http://localhost:8080")
        with pytest.raises(ServerError) as excinfo:
            client.chat_completions("m", [{"role": "user", "content": "hi"}])

        assert mc.return_value.post.call_count == 1
        sleep.assert_not_called()
        assert excinfo.value.request == {
            "id": "req-chat-timeout",
            "usage": {"output_tokens": 5},
            "credits_debited": 13,
        }
        client.close()


def test_chat_completions_non_dict_response_retains_request_metadata() -> None:
    response = MagicMock()
    response.status_code = 200
    response.headers = {
        "x-sie-request-id": "req-chat-malformed",
        "x-sie-units-output-tokens": "3",
        "x-sie-credits-debited": "8",
    }
    response.json.return_value = ["not a dict"]

    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.post.return_value = response
        client = SIEClient("http://localhost:8080")
        with pytest.raises(RequestError) as excinfo:
            client.chat_completions("m", [{"role": "user", "content": "hi"}])
        assert excinfo.value.request == {
            "id": "req-chat-malformed",
            "usage": {"output_tokens": 3},
            "credits_debited": 8,
        }
        client.close()


# M7: every newly typed kwarg on the sync chat-completion surface must land
# on the wire under its snake_case name. A regression here means callers
# either silently lose the kwarg or have to keep routing it through
# extra_body — defeating the typed surface.
def test_chat_completions_forwards_all_m7_typed_params() -> None:
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
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.post.return_value = _ok_json(payload)
        client = SIEClient("http://localhost:8080")
        client.chat_completions(
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
        sent = json.loads(mc.return_value.post.call_args.kwargs["content"].decode("utf-8"))
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
        # Non-streaming surface must NOT set stream.
        assert "stream" not in sent
        client.close()


def test_chat_completions_extra_body_still_works_for_unknown_fields() -> None:
    """Backwards-compat: callers who routed forward-compat fields through
    ``extra_body`` before the typed kwargs landed must keep working. The
    typed kwargs win when both set; ``extra_body`` supplies anything not
    yet on the typed surface.
    """
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
    with patch("sie_sdk.client.sync.httpx.Client") as mc:
        mc.return_value.post.return_value = _ok_json(payload)
        client = SIEClient("http://localhost:8080")
        client.chat_completions(
            "m",
            [{"role": "user", "content": "hi"}],
            extra_body={"hypothetical_future_field": "future-value", "top_k": 99},
        )
        sent = json.loads(mc.return_value.post.call_args.kwargs["content"].decode("utf-8"))
        # extra_body merges last, so it overrides typed kwargs absent → its
        # own values land verbatim.
        assert sent["hypothetical_future_field"] == "future-value"
        assert sent["top_k"] == 99
        client.close()
