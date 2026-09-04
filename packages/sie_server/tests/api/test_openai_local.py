"""Contract tests for the direct OpenAI routes (/v1/chat/completions, /v1/rerank).

These run without a real model subprocess. Mock transports cover the CUDA
SGLang proxy and ordinary validation; the MLX happy path remains covered live
by ``mise run mac-smoke`` on Apple Silicon.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from sie_server.adapters.mlx.generation import MLXGenerationAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.api import openai_local
from sie_server.api.openai_local import _validate_mlx_seed, router
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker import WorkerResult

_GEMMA_OPEN = "<" + "|channel" + ">" + "thought\n"
_GEMMA_CLOSE = "<" + "channel|" + ">"


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    # Registry is only reached AFTER input validation, so a stub is fine for 4xx paths.
    app.state.registry = MagicMock()
    return TestClient(app, raise_server_exceptions=False)


class _ChunkedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _cuda_chat_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    reasoning_parser: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    default_sampling: dict[str, Any] | None = None,
    max_output_tokens: int = 64,
    server_url: str = "http://127.0.0.1:30400",
) -> tuple[TestClient, MagicMock]:
    adapter = SGLangGenerationAdapter(
        model_name_or_path="upstream/repo",
        served_model_name="upstream-served-model",
        reasoning_parser=reasoning_parser,
    )
    adapter._server_url = server_url
    generate_task = SimpleNamespace(
        max_output_tokens=max_output_tokens,
        chat_template_kwargs=chat_template_kwargs or {},
    )
    profile = SimpleNamespace(
        runtime={
            "default_sampling": default_sampling or {},
            "stop_tokens": ["<configured-stop>"],
        },
        loadtime={},
    )
    config = SimpleNamespace(
        tasks=SimpleNamespace(generate=generate_task, score=None),
        resolve_profile=MagicMock(return_value=profile),
    )
    registry = MagicMock()
    registry.device = "cuda:0"
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = False
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter

    def _client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(openai_local, "_new_proxy_client", _client_factory)
    client = _client()
    client.app.state.registry = registry
    return client, registry


def _mlx_chat_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    chat_template_kwargs: dict[str, Any],
    default_sampling: dict[str, Any] | None = None,
    max_output_tokens: int = 64,
) -> tuple[TestClient, MagicMock]:
    adapter = MLXGenerationAdapter(
        model_name_or_path="Qwen/Qwen3.5-4B",
        mlx_repo="mlx-community/Qwen3.5-4B-4bit",
    )
    adapter._server_url = "http://127.0.0.1:30401"
    generate_task = SimpleNamespace(
        max_output_tokens=max_output_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    profile = SimpleNamespace(
        runtime={
            "default_sampling": default_sampling or {},
            "stop_tokens": ["<configured-stop>"],
        },
        loadtime={},
    )
    config = SimpleNamespace(
        tasks=SimpleNamespace(generate=generate_task, score=None),
        resolve_profile=MagicMock(return_value=profile),
    )
    registry = MagicMock()
    registry.device = "mps"
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = False
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter

    def _client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(openai_local, "_new_proxy_client", _client_factory)
    client = _client()
    client.app.state.registry = registry
    return client, registry


# -- /v1/chat/completions validation -----------------------------------------


def test_chat_requires_object_body() -> None:
    r = _client().post("/v1/chat/completions", content=b"[]", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_chat_requires_model() -> None:
    r = _client().post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_chat_requires_nonempty_messages() -> None:
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages"


@pytest.mark.parametrize(
    ("messages", "param"),
    [
        ([0], "messages[0]"),
        ([{"role": "invalid", "content": "hi"}], "messages[0].role"),
        ([{"role": "user", "content": 7}], "messages[0].content"),
    ],
)
def test_chat_rejects_malformed_message_structure_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[Any],
    param: str,
) -> None:
    client, registry = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("malformed messages must not reach upstream"),
    )
    response = client.post("/v1/chat/completions", json={"model": "Qwen/Qwen3.5-4B", "messages": messages})

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    registry.get_config.assert_not_called()
    registry.get.assert_not_called()


def test_chat_rejects_non_generation_model_before_load() -> None:
    # An embedding/reranker model (no generate task) must be rejected BEFORE ensure_loaded()
    # so a chat request can't kick off a real model load that only 501s afterwards.
    client = _client()
    client.app.state.registry.get_config.return_value.tasks.generate = None
    r = client.post(
        "/v1/chat/completions", json={"model": "embed-model", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 400
    assert "generation" in r.json()["error"]["message"].lower()


def test_chat_rejects_non_bool_stream() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "stream": "false"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "stream"


def test_chat_rejects_unsupported_streaming_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    client, registry = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("unsupported streaming must not reach the child"),
    )
    generate_task = registry.get_config.return_value.tasks.generate
    generate_task.capabilities = SimpleNamespace(streaming=False)
    registry.is_loaded.return_value = False

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "stream"
    registry.start_load_async.assert_not_called()
    registry.get.assert_not_called()


def test_chat_rejects_invalid_max_tokens() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    for bad in (0, -5, "100", True):
        r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "max_tokens": bad})
        assert r.status_code == 400, bad
        assert r.json()["error"]["param"] == "max_tokens"


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (True, "'seed' must be an integer"),
        ("1", "'seed' must be an integer"),
        (1.5, "'seed' must be an integer"),
        (-(1 << 63) - 1, "'seed' is outside the supported integer range"),
        (1 << 63, "'seed' is outside the supported integer range"),
    ],
)
def test_chat_rejects_invalid_seed(bad: object, message: str) -> None:
    msgs = [{"role": "user", "content": "hi"}]
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": msgs, "seed": bad})
    assert r.status_code == 400, bad
    assert r.json()["error"] == {
        "code": "INVALID_INPUT",
        "message": message,
        "param": "seed",
        "type": "invalid_request_error",
    }


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (-(1 << 63), 1 << 63),
        (-1, (1 << 64) - 1),
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
    ],
)
def test_validate_mlx_seed_preserves_signed_bit_pattern(seed: int, expected: int) -> None:
    assert _validate_mlx_seed(seed) == expected


def test_validate_mlx_seed_preserves_absent_value() -> None:
    assert _validate_mlx_seed(None) is None


@pytest.mark.parametrize(
    ("requested_model", "enable_thinking", "child_content", "request_seed", "expected_seed"),
    [
        (
            "Qwen/Qwen3.5-4B",
            False,
            "<think>private default reasoning</think>Visible answer",
            None,
            (1 << 64) - 1,
        ),
        (
            "Qwen/Qwen3.5-4B:thinking",
            True,
            "private enabled reasoning</think>Visible answer",
            -2,
            (1 << 64) - 2,
        ),
    ],
)
def test_mlx_chat_applies_profile_defaults_and_hides_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    requested_model: str,
    enable_thinking: bool,
    child_content: str,
    request_seed: int | None,
    expected_seed: int,
) -> None:
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-mlx",
                "object": "chat.completion",
                "created": 1,
                "model": "mlx-community/Qwen3.5-4B-4bit",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": child_content},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client, registry = _mlx_chat_client(
        monkeypatch,
        _handler,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        default_sampling={
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "seed": -1,
        },
    )
    request_body: dict[str, Any] = {
        "model": requested_model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 8,
        "temperature": 0.2,
        "stop": ["client-stop"],
        "chat_template_kwargs": {
            "enable_thinking": not enable_thinking,
            "guardian_config": {"risk_name": "harm"},
        },
    }
    if request_seed is not None:
        request_body["seed"] = request_seed
    response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:30401/v1/chat/completions"
    assert seen["body"] == {
        "model": "mlx-community/Qwen3.5-4B-4bit",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8,
        "temperature": 0.2,
        "top_p": 0.8,
        "seed": expected_seed,
        "stop": ["client-stop", "<configured-stop>"],
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "guardian_config": {"risk_name": "harm"},
        },
    }
    assert response.json()["choices"][0]["message"]["content"] == "Visible answer"
    assert "private" not in response.text
    registry.touch_lru.assert_called_once_with(requested_model)


def test_mlx_chat_rejects_child_process_controls_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, registry = _mlx_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("rejected request must not reach upstream"),
        chat_template_kwargs={"enable_thinking": False},
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Hello"}],
            "draft_model": "untrusted/model",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "unsupported_field",
        "message": "field 'draft_model' is not supported",
        "param": "draft_model",
        "type": "invalid_request_error",
    }
    registry.is_loaded.assert_not_called()
    registry.get.assert_not_called()


@pytest.mark.parametrize(
    ("chat_template_kwargs", "code"),
    [
        ({"arbitrary_tokenizer_kwarg": True}, "unsupported_field"),
        ({"enable_thinking": "false"}, "invalid_request"),
        ({"guardian_config": {"risk_name": "harm", "extra": True}}, "unsupported_field"),
    ],
)
def test_chat_rejects_unbounded_template_kwargs_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
    chat_template_kwargs: dict[str, Any],
    code: str,
) -> None:
    client, registry = _mlx_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("rejected request must not reach upstream"),
        chat_template_kwargs={"enable_thinking": False},
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Hello"}],
            "chat_template_kwargs": chat_template_kwargs,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == "chat_template_kwargs"
    registry.is_loaded.assert_not_called()
    registry.get.assert_not_called()


def test_cuda_chat_blocking_proxies_profile_defaults_and_returns_openai_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream-served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "private chain</think>Visible answer",
                            "reasoning_content": "private chain",
                        },
                        "logprobs": {
                            "content": [{"token": "private", "logprob": -0.1, "bytes": None, "top_logprobs": []}]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )

    client, registry = _cuda_chat_client(
        monkeypatch,
        _handler,
        chat_template_kwargs={"enable_thinking": True},
        default_sampling={"temperature": 0.7, "top_p": 0.8, "min_new_tokens": 2},
    )
    requested_model = "Qwen/Qwen3-0.6B:thinking"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": requested_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 8,
            "temperature": 0.2,
            "stop": ["client-stop"],
            "chat_template_kwargs": {
                "enable_thinking": False,
                "guardian_config": {"risk_name": "harm"},
            },
        },
    )

    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:30400/v1/chat/completions"
    assert seen["body"] == {
        "model": "upstream-served-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 8,
        "temperature": 0.2,
        "top_p": 0.8,
        "min_tokens": 2,
        "stop": ["client-stop", "<configured-stop>"],
        "chat_template_kwargs": {
            "enable_thinking": True,
            "guardian_config": {"risk_name": "harm"},
        },
        "separate_reasoning": True,
        "stream_reasoning": True,
    }
    payload = response.json()
    parsed = ChatCompletion.model_validate(payload)
    assert parsed.model == requested_model
    assert parsed.choices[0].message.content == "Visible answer"
    assert "reasoning_content" not in payload["choices"][0]["message"]
    assert payload["choices"][0]["logprobs"] is None
    registry.get.assert_called_once_with(requested_model)
    registry.touch_lru.assert_called_once_with(requested_model)


def test_cuda_chat_stream_sanitizes_split_reasoning_and_closes_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_model = "Qwen/Qwen3-0.6B:thinking"
    events = [
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": "private", "content": "private chain</thi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [{"index": 0, "delta": {"content": "nk>Visible answer"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        },
    ]
    wire = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"
    upstream_stream = _ChunkedAsyncStream([wire[:37], wire[37:121], wire[121:]])

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream_stream, headers={"content-type": "text/event-stream"})

    client, _ = _cuda_chat_client(
        monkeypatch,
        _handler,
        chat_template_kwargs={"enable_thinking": True},
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": requested_model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert data_lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in data_lines[:-1]]
    parsed = [ChatCompletionChunk.model_validate(chunk) for chunk in chunks]
    visible = "".join(choice.delta.content or "" for chunk in parsed for choice in chunk.choices)
    assert visible == "Visible answer"
    assert all(chunk.model == requested_model for chunk in parsed)
    assert "reasoning_content" not in response.text
    assert "private chain" not in response.text
    assert "</think>" not in response.text
    assert upstream_stream.closed is True


def test_cuda_chat_reasoning_parser_field_is_private_but_final_content_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-parser",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream-served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Visible answer",
                            "reasoning_content": "private chain",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            },
        )

    client, _ = _cuda_chat_client(
        monkeypatch,
        _handler,
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": True},
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B:thinking",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "Visible answer"}


def test_cuda_chat_blocking_hides_gemma_reasoning_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-gemma",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream-served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _GEMMA_OPEN + "private chain" + _GEMMA_CLOSE + "Visible answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client, _ = _cuda_chat_client(
        monkeypatch,
        _handler,
        reasoning_parser="gemma4",
        chat_template_kwargs={"enable_thinking": True},
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "google/gemma-4-E2B-it:thinking",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Visible answer"
    assert "private chain" not in response.text


def test_cuda_chat_stream_hides_split_gemma_reasoning_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _GEMMA_OPEN + "private chain" + _GEMMA_CLOSE + "Visible answer"
    content_chunks = [raw[:6], raw[6:18], raw[18:31], raw[31:]]
    events = [
        {
            "id": "chatcmpl-gemma-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        for content in content_chunks
    ]
    events.append(
        {
            "id": "chatcmpl-gemma-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "upstream-served-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    wire = b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"
    upstream_stream = _ChunkedAsyncStream([wire[:41], wire[41:137], wire[137:]])

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream_stream, headers={"content-type": "text/event-stream"})

    client, _ = _cuda_chat_client(
        monkeypatch,
        _handler,
        reasoning_parser="gemma4",
        chat_template_kwargs={"enable_thinking": True},
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "google/gemma-4-E2B-it:thinking",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    data_lines = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    chunks = [json.loads(line) for line in data_lines[:-1]]
    visible = "".join(
        choice.get("delta", {}).get("content", "") for chunk in chunks for choice in chunk.get("choices", [])
    )
    assert visible == "Visible answer"
    assert "private chain" not in response.text
    assert upstream_stream.closed is True


def test_cuda_chat_stream_rejects_non_sse_success_and_closes_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream_stream = _ChunkedAsyncStream([b'{"unexpected":"json"}'])

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream_stream, headers={"content-type": "application/json"})

    client, _ = _cuda_chat_client(monkeypatch, _handler)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "generation child returned an invalid streaming response"
    assert upstream_stream.closed is True


def test_cuda_chat_rejects_non_loopback_child_without_opening_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("unsafe upstream must not be contacted"),
        server_url="http://example.com:8080",
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


@pytest.mark.parametrize("stream", [False, True])
def test_cuda_chat_does_not_log_upstream_error_body(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stream: bool,
) -> None:
    private_marker = "private-message-must-not-be-logged"

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": [{"input": private_marker}]})

    client, _ = _cuda_chat_client(monkeypatch, _handler)
    with caplog.at_level("ERROR", logger="sie_server.api.openai_local"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": private_marker}],
                "stream": stream,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "upstream generation request failed"
    assert private_marker not in caplog.text


def test_cuda_chat_rejects_engine_control_and_output_above_model_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("rejected request must not reach upstream"),
        max_output_tokens=32,
    )
    messages = [{"role": "user", "content": "Hello"}]

    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3.5-4B", "messages": messages, "routed_dp_rank": 2},
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "unsupported_field",
        "message": "field 'routed_dp_rank' is not supported",
        "param": "routed_dp_rank",
        "type": "invalid_request_error",
    }

    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3.5-4B", "messages": messages, "max_completion_tokens": 33},
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "max_completion_tokens"


def test_mlx_chat_rejects_output_above_model_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _mlx_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("rejected request must not reach upstream"),
        chat_template_kwargs={"enable_thinking": False},
        max_output_tokens=32,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 33,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "max_completion_tokens"


def test_cuda_chat_rejects_remote_media_before_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    client, registry = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("remote media request must not reach upstream"),
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "http://169.254.169.254/latest/meta-data/"},
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "messages[0].content[1].image_url"
    registry.get.assert_not_called()


def test_cuda_chat_preserves_visible_logprobs_when_reasoning_is_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible_logprobs = {"content": [{"token": "answer", "logprob": -0.1, "bytes": None, "top_logprobs": []}]}

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-visible",
                "object": "chat.completion",
                "created": 1,
                "model": "upstream-served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Visible answer",
                            "reasoning_content": None,
                        },
                        "logprobs": visible_logprobs,
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client, _ = _cuda_chat_client(
        monkeypatch,
        _handler,
        reasoning_parser="qwen3",
        chat_template_kwargs={"enable_thinking": False},
    )
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Visible answer"}
    assert choice["logprobs"] == visible_logprobs


# -- /v1/rerank validation ----------------------------------------------------


def test_rerank_requires_model() -> None:
    r = _client().post("/v1/rerank", json={"query": "q", "documents": ["a"]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "model"


def test_rerank_requires_query() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "documents": ["a"]})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "query"


def test_rerank_rejects_blank_model_and_query() -> None:
    for body, param in [
        ({"model": "   ", "query": "q", "documents": ["a"]}, "model"),
        ({"model": "m", "query": "   ", "documents": ["a"]}, "query"),
    ]:
        r = _client().post("/v1/rerank", json=body)
        assert r.status_code == 400
        assert r.json()["error"]["param"] == param


def test_rerank_requires_nonempty_string_documents() -> None:
    for docs in ([], "not-a-list", [1, 2], ["ok", 3], ["   "]):
        r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": docs})
        assert r.status_code == 400, docs
        assert r.json()["error"]["param"] == "documents"


def test_rerank_top_n_must_be_positive_int() -> None:
    for bad in (0, -1, "3", True):
        r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "top_n": bad})
        assert r.status_code == 400, bad
        assert r.json()["error"]["param"] == "top_n"


def test_rerank_rejects_too_many_documents() -> None:
    docs = ["d"] * (openai_local._MAX_RERANK_DOCS + 1)
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": docs})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "documents"


def test_rerank_rejects_non_bool_return_documents() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "return_documents": "true"})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "return_documents"


def test_rerank_rejects_unknown_fields() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"], "priority": 1})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "priority"


# -- top-level OpenAI error envelope (no {"detail": ...} wrapper) -------------


def test_chat_unknown_model_returns_top_level_openai_error() -> None:
    client = _client()
    client.app.state.registry.has_model.return_value = False
    r = client.post(
        "/v1/chat/completions",
        json={"model": "missing-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["code"] == "MODEL_NOT_FOUND"
    assert error["type"] == "invalid_request_error"
    assert "message" in error


def test_chat_invalid_input_returns_top_level_openai_error() -> None:
    r = _client().post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 400
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["param"] == "messages"
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "INVALID_INPUT"
    assert "message" in error


def test_chat_unloading_model_returns_top_level_openai_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, registry = _cuda_chat_client(
        monkeypatch,
        lambda _request: pytest.fail("unloading model must not reach upstream"),
    )
    registry.is_unloading.return_value = True
    r = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["code"] == "MODEL_NOT_LOADED"
    assert error["type"] == "server_error"
    assert "message" in error


def _rerank_client() -> tuple[TestClient, MagicMock]:
    """Rerank client whose registry passes all model-state checks."""
    profile = SimpleNamespace(runtime={}, loadtime={})
    config = SimpleNamespace(
        tasks=SimpleNamespace(generate=None, score=SimpleNamespace()),
        resolve_profile=MagicMock(return_value=profile),
    )
    registry = MagicMock()
    registry.device = "cpu"
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = False
    registry.is_loaded.return_value = True
    client = _client()
    client.app.state.registry = registry
    return client, registry


def test_rerank_unknown_model_returns_top_level_openai_error() -> None:
    client = _client()
    client.app.state.registry.has_model.return_value = False
    r = client.post("/v1/rerank", json={"model": "missing-model", "query": "q", "documents": ["a"]})
    assert r.status_code == 404
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["code"] == "MODEL_NOT_FOUND"
    assert error["type"] == "invalid_request_error"
    assert "message" in error


def test_rerank_invalid_input_returns_top_level_openai_error() -> None:
    r = _client().post("/v1/rerank", json={"model": "m", "query": "q", "documents": []})
    assert r.status_code == 400
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["param"] == "documents"
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "INVALID_INPUT"
    assert "message" in error


def test_rerank_inference_error_returns_top_level_openai_500() -> None:
    client, registry = _rerank_client()

    async def _boom() -> None:
        raise RuntimeError("sensitive reranker detail")

    worker = MagicMock()
    worker.submit_score = AsyncMock(return_value=_boom())
    registry.start_worker = AsyncMock(return_value=worker)

    r = client.post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a"]})
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body
    error = body["error"]
    assert error["code"] == "inference_error"
    assert error["type"] == "server_error"
    assert error["message"] == "internal error during reranking"
    assert "sensitive reranker detail" not in r.text


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_rerank_non_finite_scores_return_top_level_openai_500(bad: float) -> None:
    """NaN/inf model scores fail closed as an enveloped 500 naming the model.

    Regression for pass-2 audit A2: on this JSON-only surface a NaN score
    crashed json.dumps ("Out of range float values are not JSON compliant")
    into a bare, un-enveloped 500 with no OpenAI error object.
    """
    client, registry = _rerank_client()

    async def _result() -> WorkerResult:
        return WorkerResult(
            output=ScoreOutput(scores=np.array([0.9, bad], dtype=np.float32)),
            timing=RequestTiming(),
        )

    worker = MagicMock()
    worker.submit_score = AsyncMock(return_value=_result())
    registry.start_worker = AsyncMock(return_value=worker)

    r = client.post("/v1/rerank", json={"model": "m", "query": "q", "documents": ["a", "b"]})
    assert r.status_code == 500
    body = r.json()
    assert "detail" not in body  # enveloped OpenAI error, not FastAPI's {"detail": ...}
    error = body["error"]
    assert error["code"] == "INFERENCE_ERROR"
    assert error["type"] == "server_error"
    assert "m" in error["message"]
    assert "non-finite" in error["message"]
