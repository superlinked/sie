"""Regression tests for direct ``/v1/completions`` generation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationAdapter,
    GenerationCapacityError,
    GenerationChunk,
    GenerationDrainingError,
    GenerationError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.api import openai_completions
from sie_server.api.openai_completions import _CompletionError, _stream_completion, router
from sie_server.config.engine import EngineConfig
from sie_server.config.model import AdapterOptions, GenerateTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

_GEMMA_OPEN = "<" + "|channel" + ">" + "thought\n"
_GEMMA_CLOSE = "<" + "channel|" + ">"


class _UntrustedUnsupportedFieldError(GenerationUnsupportedFieldError):
    code = "SENSITIVE_UNSUPPORTED_CODE"


class _UntrustedCapacityError(GenerationCapacityError):
    code = "SENSITIVE_CAPACITY_CODE"


class _FakeCompletionAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None
        self.preflight_call: dict[str, object] | None = None
        self.preflight_stream: bool | None = None
        self.preflight_error: GenerationError | None = None
        self.generate_error_after_chunks: int | None = None
        self.closed = False
        self.text_chunks = ["hello", " world"]
        self.finish_reason: FinishReason = "stop"
        self.error_code: str | None = None
        self.error_message: str | None = None

    def load(self, device: str) -> None:  # pragma: no cover - registry is mocked loaded
        _ = device

    def preflight_generate(self, parameters: Mapping[str, object], *, stream: bool) -> None:
        self.preflight_call = dict(parameters)
        self.preflight_stream = stream
        if self.preflight_error is not None:
            raise self.preflight_error

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
        grammar: GrammarSpec | None = None,
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        images: list[ImageInput] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        self.last_call = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "top_k": top_k,
            "min_new_tokens": min_new_tokens,
            "seed": seed,
        }
        _ = repetition_penalty, grammar, logit_bias, logprobs, top_logprobs, images
        try:
            if self.generate_error_after_chunks == 0:
                raise GenerationUnsupportedFieldError("top_k")
            for index, text in enumerate(self.text_chunks):
                yield GenerationChunk(text_delta=text, is_first=index == 0)
                if self.generate_error_after_chunks == index + 1:
                    raise GenerationUnsupportedFieldError("top_k")
            yield GenerationChunk(
                text_delta="",
                done=True,
                finish_reason=self.finish_reason,
                prompt_tokens=3,
                completion_tokens=2,
                error_code=self.error_code,
                error_message=self.error_message,
            )
        finally:
            self.closed = True


def _config(model_id: str = "Qwen/Qwen3-4B-Instruct") -> ModelConfig:
    return ModelConfig(
        sie_id=model_id,
        hf_id=model_id,
        tasks=Tasks(generate=GenerateTask(context_length=32768, max_output_tokens=64)),
        profiles={
            "default": ProfileConfig(
                adapter_path="test:FakeCompletionAdapter",
                max_batch_tokens=8192,
                kv_budget_tokens=4096,
                adapter_options=AdapterOptions(
                    loadtime={"reasoning_parser": "gemma4" if model_id.startswith("google/gemma") else "qwen3"},
                    runtime={
                        "default_sampling": {
                            "temperature": 0.25,
                            "top_p": 0.8,
                            "frequency_penalty": 0.75,
                            "top_k": 12,
                            "min_new_tokens": 2,
                            "seed": 23,
                        },
                        "stop_tokens": ["</s>"],
                    },
                ),
            )
        },
    )


@pytest.fixture
def adapter() -> _FakeCompletionAdapter:
    return _FakeCompletionAdapter()


@pytest.fixture
def registry(adapter: _FakeCompletionAdapter) -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.has_model.return_value = True
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    reg.is_unloading.return_value = False
    reg.is_failed.return_value = False
    reg.get_failure.return_value = None
    reg.get_config.return_value = _config()
    reg.get.return_value = adapter
    reg.device = "cpu"
    return reg


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.registry = registry
    return TestClient(app)


def test_blocking_completion_uses_direct_adapter_and_openai_shape(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": ["Continue this"],
            "max_tokens": 8,
            "frequency_penalty": 0.5,
            "seed": -1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert body["model"] == "Qwen/Qwen3-4B-Instruct"
    assert body["choices"] == [{"text": "hello world", "index": 0, "finish_reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert body["system_fingerprint"].startswith("fp_")
    assert adapter.last_call == {
        "prompt": "Continue this",
        "max_new_tokens": 8,
        "temperature": 0.25,
        "top_p": 0.8,
        "stop": ["</s>"],
        "frequency_penalty": 0.5,
        "presence_penalty": None,
        "top_k": 12,
        "min_new_tokens": 2,
        "seed": -1,
    }
    assert adapter.preflight_call == adapter.last_call
    assert adapter.preflight_stream is False


def test_completion_uses_profile_frequency_penalty_and_seed_defaults(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
        },
    )

    assert response.status_code == 200, response.text
    assert adapter.last_call is not None
    assert adapter.last_call["frequency_penalty"] == 0.75
    assert adapter.last_call["seed"] == 23


def test_completion_request_frequency_penalty_and_seed_override_profile_defaults(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "frequency_penalty": -0.5,
            "seed": -1,
        },
    )

    assert response.status_code == 200, response.text
    assert adapter.last_call is not None
    assert adapter.last_call["frequency_penalty"] == -0.5
    assert adapter.last_call["seed"] == -1


def test_streaming_completion_emits_openai_chunks_usage_and_done(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200, response.text
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    assert [event["choices"][0]["text"] for event in events[:-1]] == ["hello", " world", ""]
    assert events[0]["choices"][0]["finish_reason"] is None
    assert events[2]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert len({event["id"] for event in events}) == 1


def test_streaming_completion_surfaces_cancelled_terminal_as_error(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    adapter.finish_reason = "cancelled"

    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200, response.text
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    terminal = events[-1]
    assert terminal["choices"] == [{"text": "", "index": 0, "finish_reason": None}]
    assert terminal["error"] == {
        "message": "generation was cancelled before completion",
        "type": "server_error",
        "param": None,
        "code": "generation_cancelled",
    }
    assert all("usage" not in event for event in events)


def test_completion_preflight_preserves_exact_parameter_and_skips_dispatch(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    adapter.preflight_error = GenerationUnsupportedFieldError("top_k")

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "top_k"
    assert adapter.preflight_call is not None
    assert adapter.last_call is None


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (_UntrustedUnsupportedFieldError("top_k", "private unsupported detail"), 400),
        (_UntrustedCapacityError("private capacity detail"), 503),
    ],
)
def test_completion_sanitizes_untrusted_typed_error_codes(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
    error: GenerationError,
    status_code: int,
) -> None:
    adapter.preflight_error = error

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x"},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == "inference_error"
    assert "SENSITIVE_" not in response.text


def test_completion_preserves_typed_synchronous_dispatch_error(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**parameters: object) -> AsyncIterator[GenerationChunk]:
        _ = parameters
        raise GenerationUnsupportedFieldError("top_k")

    monkeypatch.setattr(adapter, "generate", refuse)

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "top_k"


@pytest.mark.parametrize("error_after", [0, 1])
def test_blocking_completion_preserves_typed_iterator_error_and_closes(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
    error_after: int,
) -> None:
    adapter.generate_error_after_chunks = error_after

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "top_k"
    assert adapter.closed is True


def test_streaming_completion_preserves_typed_iterator_error_and_closes(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
) -> None:
    adapter.generate_error_after_chunks = 1

    response = client.post(
        "/v1/completions",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "prompt": "x",
            "stream": True,
        },
    )

    assert response.status_code == 200
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    error = json.loads(payloads[-2])
    assert error["choices"][0]["finish_reason"] is None
    assert error["error"]["code"] == "unsupported_field"
    assert error["error"]["param"] == "top_k"
    assert payloads[-1] == "[DONE]"
    assert adapter.closed is True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("code", "message", "expected_code", "expected_message"),
    [
        (
            "inference_error",
            "SENSITIVE_COMPLETION_RUNTIME",
            "inference_error",
            "internal error during generation",
        ),
        (
            "grammar_compile_failed",
            "SENSITIVE_COMPLETION_GRAMMAR",
            "grammar_compile_failed",
            "internal error compiling grammar",
        ),
        (
            "empty_model_output",
            "model produced no visible output",
            "empty_model_output",
            "model produced no visible output",
        ),
        (
            "SENSITIVE_COMPLETION_ERROR_CODE",
            "SENSITIVE_COMPLETION_ERROR_CODE",
            "inference_error",
            "generation terminated with an upstream error",
        ),
    ],
)
def test_completion_classifies_adapter_yielded_terminal_errors(
    client: TestClient,
    adapter: _FakeCompletionAdapter,
    stream: bool,
    code: str,
    message: str,
    expected_code: str,
    expected_message: str,
) -> None:
    adapter.finish_reason = "error"
    adapter.error_code = code
    adapter.error_message = message

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x", "stream": stream},
    )

    if stream:
        assert response.status_code == 200, response.text
        payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
        error_event = json.loads(payloads[-2])
        assert error_event["choices"][0]["finish_reason"] is None
        assert error_event["error"]["code"] == expected_code
        assert error_event["error"]["message"] == expected_message
        assert payloads[-1] == "[DONE]"
    else:
        assert response.status_code == 500, response.text
        assert response.json()["error"]["code"] == expected_code
        assert response.json()["error"]["message"] == expected_message
    if message.startswith("SENSITIVE_"):
        assert message not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("after_delta", [False, True])
@pytest.mark.parametrize(
    ("error", "expected_retry_after_s"),
    [
        (
            _CompletionError(
                "scheduler full",
                status_code=503,
                code="RESOURCE_EXHAUSTED",
                headers={"Retry-After": "12"},
            ),
            12,
        ),
        (
            _CompletionError(
                "scheduler full",
                status_code=503,
                code="RESOURCE_EXHAUSTED",
                headers={"Retry-After": "malformed"},
            ),
            None,
        ),
        (
            _CompletionError(
                "wrong code",
                status_code=503,
                code="MODEL_LOADING",
                headers={"Retry-After": "12"},
            ),
            None,
        ),
        (
            _CompletionError(
                "scheduler draining",
                status_code=503,
                code="MODEL_LOADING",
                headers={"Retry-After": "5"},
            ),
            None,
        ),
    ],
)
async def test_streaming_completion_in_band_retry_authority_matches_gateway_shape(
    monkeypatch: pytest.MonkeyPatch,
    after_delta: bool,
    error: _CompletionError,
    expected_retry_after_s: int | None,
) -> None:
    async def chunks() -> AsyncIterator[GenerationChunk]:
        if after_delta:
            yield GenerationChunk(text_delta="partial")
        if error.code == "MODEL_LOADING" and error.message == "scheduler draining":
            raise GenerationDrainingError(error.message)
        raise GenerationCapacityError(error.message)

    monkeypatch.setattr(openai_completions, "_from_generation_error", lambda _exc, _registry: error)

    raw = [
        event
        async for event in _stream_completion(
            chunks(),
            completion_id="cmpl-test",
            created=1,
            model="test/model",
            include_usage=False,
            registry=MagicMock(),
        )
    ]
    payloads = [line.removeprefix("data: ").strip() for block in raw for line in block.splitlines() if line]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    if after_delta:
        assert events[0]["choices"][0]["text"] == "partial"
    error_event = events[-1]
    assert error_event["choices"][0]["finish_reason"] is None
    assert error_event["error"]["code"] == error.code
    assert error_event["error"]["type"] == error.error_type
    assert error_event["error"]["param"] == error.param
    if expected_retry_after_s is None:
        assert "retry_after_s" not in error_event["error"]
    else:
        assert error_event["error"]["retry_after_s"] == expected_retry_after_s


@pytest.mark.parametrize("after_delta", [False, True])
def test_streaming_completion_capacity_error_uses_configured_retry_authority(
    client: TestClient,
    registry: MagicMock,
    adapter: _FakeCompletionAdapter,
    monkeypatch: pytest.MonkeyPatch,
    after_delta: bool,
) -> None:
    registry.engine_config = EngineConfig.model_validate({"oom_recovery": {"retry_after_s": 12}})

    async def refuse(**_parameters: object) -> AsyncIterator[GenerationChunk]:
        if after_delta:
            yield GenerationChunk(text_delta="partial")
        raise GenerationCapacityError("scheduler full")

    monkeypatch.setattr(adapter, "generate", refuse)

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x", "stream": True},
    )

    assert response.status_code == 200, response.text
    payloads = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(payload) for payload in payloads[:-1]]
    if after_delta:
        assert events[0]["choices"][0]["text"] == "partial"
    error_event = events[-1]
    assert error_event["choices"][0]["finish_reason"] is None
    assert error_event["error"] == {
        "message": "scheduler full",
        "type": "server_error",
        "param": None,
        "code": "RESOURCE_EXHAUSTED",
        "retry_after_s": 12,
    }


def test_completion_rejects_unsupported_streaming_before_load(
    client: TestClient,
    registry: MagicMock,
) -> None:
    config = _config()
    assert config.tasks.generate is not None
    config.tasks.generate.capabilities.streaming = False
    registry.get_config.return_value = config
    registry.is_loaded.return_value = False

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x", "stream": True},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "stream"
    registry.start_load_async.assert_not_called()
    registry.get.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enable_thinking", [False, True])
@pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it"])
def test_completion_hides_reasoning_for_every_resolved_profile(
    client: TestClient,
    registry: MagicMock,
    adapter: _FakeCompletionAdapter,
    stream: bool,
    enable_thinking: bool,
    model_id: str,
) -> None:
    config = _config(model_id)
    assert config.tasks.generate is not None
    config.tasks.generate.chat_template_kwargs = {"enable_thinking": enable_thinking}
    registry.get_config.return_value = config
    if model_id.startswith("google/gemma"):
        raw = _GEMMA_OPEN + "secret" + _GEMMA_CLOSE + "answer"
        adapter.text_chunks = [raw[:7], raw[7:19], raw[19:]]
    else:
        adapter.text_chunks = ["<thi", "nk>secret</th", "ink>answer"]

    response = client.post(
        "/v1/completions",
        json={
            "model": model_id,
            "prompt": "Continue this",
            "max_tokens": 8,
            "stream": stream,
        },
    )

    assert response.status_code == 200, response.text
    assert "secret" not in response.text
    assert "answer" in response.text


@pytest.mark.parametrize(
    ("body", "param", "code"),
    [
        ({"model": "m", "prompt": "x", "tools": []}, "tools", "unsupported_field"),
        ({"model": "m", "prompt": ["x", "y"]}, "prompt", "unsupported_field"),
        ({"model": "m", "prompt": "x", "n": 2}, "n", "unsupported_field"),
        ({"model": "m", "prompt": "x", "max_tokens": True}, "max_tokens", "invalid_request"),
        ({"model": "m", "prompt": "x", "temperature": 1e100}, "temperature", "invalid_request"),
        ({"model": "m", "prompt": "x", "top_p": 0}, "top_p", "invalid_request"),
        ({"model": "m", "prompt": "x", "stop": ""}, "stop", "invalid_request"),
        ({"model": "m", "prompt": "x", "stop": ["ok", ""]}, "stop", "invalid_request"),
    ],
)
def test_invalid_requests_fail_before_registry_lookup(
    client: TestClient,
    registry: MagicMock,
    body: dict[str, object],
    param: str,
    code: str,
) -> None:
    response = client.post("/v1/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["param"] == param
    assert response.json()["error"]["code"] == code
    registry.has_model.assert_not_called()


def test_model_output_cap_rejects_before_load(client: TestClient, registry: MagicMock) -> None:
    registry.is_loaded.return_value = False

    response = client.post(
        "/v1/completions",
        json={"model": "Qwen/Qwen3-4B-Instruct", "prompt": "x", "max_tokens": 65},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_exceeded"
    registry.start_load_async.assert_not_called()


def test_model_state_errors_use_openai_error_envelope(client: TestClient, registry: MagicMock) -> None:
    registry.has_model.return_value = False

    response = client.post(
        "/v1/completions",
        json={"model": "missing/model", "prompt": "x"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "Model 'missing/model' not found",
            "type": "invalid_request_error",
            "param": None,
            "code": "MODEL_NOT_FOUND",
        }
    }
