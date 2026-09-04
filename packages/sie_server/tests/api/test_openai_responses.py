"""Regression tests for direct ``/v1/responses`` generation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationCapacityError,
    GenerationChunk,
    GenerationError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.api import openai_responses
from sie_server.api.openai_completions import _CompletionError
from sie_server.config.model import AdapterOptions, GenerateTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

_VECTORS = Path(__file__).parents[4] / "conformance" / "openai_generation" / "responses_request_vectors.json"


class _UntrustedUnsupportedFieldError(GenerationUnsupportedFieldError):
    code = "SENSITIVE_UNSUPPORTED_CODE"


class _UntrustedCapacityError(GenerationCapacityError):
    code = "SENSITIVE_CAPACITY_CODE"


class _FakeResponsesAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None
        self.preflight_call: dict[str, object] | None = None
        self.preflight_error: GenerationError | None = None
        self.generate_error_after_chunks: int | None = None
        self.closed = False
        self.text_chunks = ["hello", " world"]
        self.include_usage = True
        self.error_code: str | None = None
        self.error_message: str | None = None

    def load(self, device: str) -> None:  # pragma: no cover - registry is mocked loaded
        _ = device

    def preflight_generate(self, parameters: Mapping[str, object], *, stream: bool) -> None:
        assert stream is False
        self.preflight_call = dict(parameters)
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
                finish_reason="stop",
                prompt_tokens=3 if self.include_usage else None,
                completion_tokens=2 if self.include_usage else None,
                error_code=self.error_code,
                error_message=self.error_message,
            )
        finally:
            self.closed = True


def _config(*, chat_template_kwargs: dict[str, object] | None = None) -> ModelConfig:
    return ModelConfig(
        sie_id="Qwen/Qwen3-4B-Instruct",
        hf_id="Qwen/Qwen3-4B-Instruct",
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=64,
                chat_template_kwargs=chat_template_kwargs or {},
            )
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="test:FakeResponsesAdapter",
                max_batch_tokens=8192,
                kv_budget_tokens=4096,
                adapter_options=AdapterOptions(
                    loadtime={"reasoning_parser": "qwen3"},
                    runtime={
                        "default_sampling": {
                            "temperature": 0.25,
                            "top_p": 0.8,
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
def adapter() -> _FakeResponsesAdapter:
    return _FakeResponsesAdapter()


@pytest.fixture
def registry(adapter: _FakeResponsesAdapter) -> MagicMock:
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
    app.include_router(openai_responses.router)
    app.state.registry = registry
    return TestClient(app)


def test_blocking_responses_uses_direct_adapter_and_openai_shape(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "input": "Say hello",
            "max_output_tokens": 8,
            "temperature": 0.5,
            "seed": -1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"].startswith("resp-")
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "Qwen/Qwen3-4B-Instruct"
    assert body["output"][0]["content"] == [{"type": "output_text", "text": "hello world", "annotations": []}]
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    assert adapter.last_call == {
        "prompt": "Say hello",
        "max_new_tokens": 8,
        "temperature": 0.5,
        "top_p": 0.8,
        "stop": ["</s>"],
        "frequency_penalty": None,
        "presence_penalty": None,
        "top_k": 12,
        "min_new_tokens": 2,
        "seed": -1,
    }
    assert adapter.preflight_call == adapter.last_call


def test_message_input_uses_model_native_template(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    render = AsyncMock(return_value="<rendered>Hi there</rendered>")
    monkeypatch.setattr(openai_responses, "_render_native_messages_prompt", render)

    response = client.post(
        "/v1/responses",
        json={
            "model": "Qwen/Qwen3-4B-Instruct",
            "input": [
                {"role": "developer", "content": "Be concise"},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hi "},
                        {"type": "text", "text": "there"},
                    ],
                },
            ],
        },
    )

    assert response.status_code == 200, response.text
    render.assert_awaited_once()
    assert render.await_args.args[1] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hi there"},
    ]
    assert adapter.last_call is not None
    assert adapter.last_call["prompt"] == "<rendered>Hi there</rendered>"


def test_responses_hides_family_reasoning_without_changing_usage(
    client: TestClient,
    registry: MagicMock,
    adapter: _FakeResponsesAdapter,
) -> None:
    registry.get_config.return_value = _config(chat_template_kwargs={"enable_thinking": True})
    adapter.text_chunks = ["<thi", "nk>private</think>answer"]

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["output"][0]["content"][0]["text"] == "answer"
    assert body["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_shared_responses_request_conformance_vectors() -> None:
    vectors = json.loads(_VECTORS.read_text())

    for case in vectors["accepted"]:
        params = openai_responses._parse_params(case["body"])
        observed = {
            "model": params.model,
            "prompt": params.prompt,
            "messages": params.messages,
            "max_output_tokens": params.max_output_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
        }
        assert observed == case["expected"], case["name"]

    for case in vectors["rejected"]:
        with pytest.raises(_CompletionError) as exc_info:
            openai_responses._parse_params(case["body"])
        assert exc_info.value.param == case["param"], case["name"]
        assert exc_info.value.code == case["code"], case["name"]


def test_responses_rejects_model_output_cap(client: TestClient) -> None:
    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x", "max_output_tokens": 65},
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "max_output_tokens"


def test_responses_rejects_terminal_event_without_usage(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
) -> None:
    adapter.include_usage = False

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "malformed_worker_response"


@pytest.mark.parametrize(
    ("code", "message", "expected_code", "expected_message"),
    [
        (
            "inference_error",
            "SENSITIVE_RESPONSES_RUNTIME",
            "inference_error",
            "internal error during generation",
        ),
        (
            "grammar_compile_failed",
            "SENSITIVE_RESPONSES_GRAMMAR",
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
            "SENSITIVE_RESPONSES_ERROR_CODE",
            "SENSITIVE_RESPONSES_ERROR_CODE",
            "inference_error",
            "generation terminated with an upstream error",
        ),
    ],
)
def test_responses_classifies_adapter_yielded_terminal_errors(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
    code: str,
    message: str,
    expected_code: str,
    expected_message: str,
) -> None:
    adapter.error_code = code
    adapter.error_message = message

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["message"] == expected_message
    if message.startswith("SENSITIVE_"):
        assert message not in response.text


def test_responses_rejects_oversized_rendered_input(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openai_responses, "_MAX_PROMPT_BYTES", 4)

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "12345"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["param"] == "input"
    assert response.json()["error"]["code"] == "context_exceeded"


def test_responses_preflight_preserves_exact_parameter_and_skips_dispatch(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
) -> None:
    adapter.preflight_error = GenerationUnsupportedFieldError("top_k")

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
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
def test_responses_sanitizes_untrusted_typed_error_codes(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
    error: GenerationError,
    status_code: int,
) -> None:
    adapter.preflight_error = error

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == "inference_error"
    assert "SENSITIVE_" not in response.text


def test_responses_preserves_typed_synchronous_dispatch_error(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**parameters: object) -> AsyncIterator[GenerationChunk]:
        _ = parameters
        raise GenerationUnsupportedFieldError("top_k")

    monkeypatch.setattr(adapter, "generate", refuse)

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "top_k"


@pytest.mark.parametrize("error_after", [0, 1])
def test_responses_preserves_typed_iterator_error_and_closes(
    client: TestClient,
    adapter: _FakeResponsesAdapter,
    error_after: int,
) -> None:
    adapter.generate_error_after_chunks = error_after

    response = client.post(
        "/v1/responses",
        json={"model": "Qwen/Qwen3-4B-Instruct", "input": "x"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_field"
    assert response.json()["error"]["param"] == "top_k"
    assert adapter.closed is True
