"""Tests for the direct ``/v1/generate/{model}`` HTTP route (walking-skeleton local-dev path).

Mirrors :mod:`tests.api.test_score` but targets the new local-dev path that
calls the adapter directly (no NATS, no gateway). The gateway-side handler
``proxy_generate`` is covered by Rust inline tests in
``packages/sie_gateway/src/handlers/proxy.rs``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationCapacityError,
    GenerationChunk,
    GenerationDrainingError,
    GenerationError,
    GenerationInputTooLongError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.api import generate as generate_api
from sie_server.api.generate import router as generate_router
from sie_server.config.engine import EngineConfig
from sie_server.config.model import (
    AdapterOptions,
    GenerateCapabilities,
    GenerateTask,
    InputModalities,
    ModelConfig,
    ProfileConfig,
    Tasks,
)
from sie_server.core.registry import ModelRegistry
from sie_server.types.grammar import GrammarSpec
from sie_server.types.inputs import ImageInput

_GEMMA_OPEN = "<" + "|channel" + ">" + "thought\n"
_GEMMA_CLOSE = "<" + "channel|" + ">"


class _UntrustedUnsupportedFieldError(GenerationUnsupportedFieldError):
    code = "SENSITIVE_UNSUPPORTED_CODE"


class _UntrustedCapacityError(GenerationCapacityError):
    code = "SENSITIVE_CAPACITY_CODE"


class _FutureUnsupportedFieldError(GenerationUnsupportedFieldError):
    pass


@pytest.mark.parametrize(
    ("error", "expected_param"),
    [
        (GenerationUnsupportedFieldError("n", "n refusal"), "n"),
        (GenerationUnsupportedFieldError("best_of", "best_of refusal"), "best_of"),
        (
            GenerationUnsupportedFieldError(
                "lora_path",
                "'lora_adapter' is not supported by this generation backend",
            ),
            "lora_adapter",
        ),
    ],
)
def test_generation_http_exception_preserves_public_refusal_param(
    error: GenerationUnsupportedFieldError,
    expected_param: str,
) -> None:
    exception = generate_api._generation_http_exception(error)

    assert exception.status_code == 400
    assert exception.detail["param"] == expected_param
    assert exception.detail["param"] != "lora_path"
    assert "lora_path" not in str(exception.detail)


def test_generation_http_exception_redacts_future_refusal_subclass_param() -> None:
    exception = generate_api._generation_http_exception(_FutureUnsupportedFieldError("n", "future refusal"))

    assert exception.status_code == 400
    assert exception.detail == {"code": "unsupported_field", "message": "future refusal"}


class _FakeGenAdapter(GenerationAdapter):
    """Minimal in-memory GenerationAdapter for route tests."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None
        self.last_call: dict | None = None

    def load(self, device: str) -> None:  # pragma: no cover — registry-mocked
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    # The terminal finish_reason the fake yields; tests flip this to
    # exercise the route's error/cancelled → non-200 mapping (BUG: a
    # terminal error/cancelled chunk must NOT become an HTTP 200).
    finish_reason: str = "stop"
    # Typed terminal error fields; tests set these to exercise the
    # route's verbatim-code fail-closed mapping (#3104/#3136).
    error_code: str | None = None
    error_message: str | None = None

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
            "repetition_penalty": repetition_penalty,
            "min_new_tokens": min_new_tokens,
            "grammar": grammar,
            "seed": seed,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if images is not None:
            self.last_call["images"] = images
        # Yield one delta + a terminal chunk so the local-dev route can
        # drain the iterator into the walking-skeleton-shaped aggregate response.
        yield GenerationChunk(text_delta=f"echo:{prompt}", is_first=True)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=self.finish_reason,  # type: ignore[arg-type]
            prompt_tokens=len(prompt.split()),
            completion_tokens=2,
            error_code=self.error_code,
            error_message=self.error_message,
        )


class _LegacyTextGenAdapter(_FakeGenAdapter):
    """Third-party-style adapter implementing the pre-grammar call signature."""

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
        seed: int | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
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
            "repetition_penalty": repetition_penalty,
            "min_new_tokens": min_new_tokens,
            "seed": seed,
            "logit_bias": logit_bias,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        yield GenerationChunk(text_delta=f"echo:{prompt}", is_first=True)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=len(prompt.split()),
            completion_tokens=2,
        )


class _ThinkingGenAdapter(_FakeGenAdapter):
    """Script reasoning delimiters across engine chunk boundaries."""

    def __init__(self, *, prompt_seeded: bool = False, gemma: bool = False) -> None:
        super().__init__()
        self._prompt_seeded = prompt_seeded
        self._opening = _GEMMA_OPEN if gemma else "<think>"
        self._closing = _GEMMA_CLOSE if gemma else "</think>"

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        **kwargs: object,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, max_new_tokens, kwargs)
        text_chunks = (
            ("private", " reasoning" + self._closing[:5], self._closing[5:] + "Visible answer")
            if self._prompt_seeded
            else (
                self._opening[:5],
                self._opening[5:] + "private",
                " reasoning" + self._closing[:5],
                self._closing[5:] + "Visible answer",
            )
        )
        for text in text_chunks:
            yield GenerationChunk(text_delta=text)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=8,
        )


class _PreflightAdapter(_FakeGenAdapter):
    def __init__(self, error: Exception | None = None, *, raise_during_generate: bool = False) -> None:
        super().__init__()
        self.error = error
        self.raise_during_generate = raise_during_generate
        self.preflight_parameters: dict[str, object] | None = None
        self.preflight_stream: bool | None = None
        self.dispatched_parameters: dict[str, object] | None = None
        self.events: list[str] = []

    def preflight_generate(self, parameters: Mapping[str, object], *, stream: bool) -> None:
        self.events.append("preflight")
        self.preflight_parameters = dict(parameters)
        self.preflight_stream = stream
        if self.error is not None and not self.raise_during_generate:
            raise self.error

    async def generate(self, prompt: str, **kwargs: object) -> AsyncIterator[GenerationChunk]:
        self.events.append("generate")
        self.dispatched_parameters = {"prompt": prompt, **kwargs}
        if self.error is not None and self.raise_during_generate:
            raise self.error
        async for chunk in super().generate(prompt, **kwargs):  # type: ignore[arg-type]
            yield chunk


def _make_config(
    *,
    model_id: str = "Qwen/Qwen3-4B-Instruct",
    grammar: list[Literal["json_schema", "regex", "ebnf"]] | None = None,
    streaming: bool = True,
) -> ModelConfig:
    return ModelConfig(
        sie_id=model_id,
        hf_id=model_id,
        tasks=Tasks(
            generate=GenerateTask(
                context_length=32768,
                max_output_tokens=4096,
                capabilities=GenerateCapabilities(grammar=grammar or [], streaming=streaming),
            ),
        ),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang:SGLangGenerationAdapter",
                max_batch_tokens=16384,
                kv_budget_tokens=8192,
                adapter_options=AdapterOptions(
                    loadtime={"reasoning_parser": "gemma4" if model_id.startswith("google/gemma") else "qwen3"}
                ),
            ),
        },
    )


@pytest.fixture
def fake_adapter() -> _FakeGenAdapter:
    return _FakeGenAdapter()


@pytest.fixture
def registry(fake_adapter: _FakeGenAdapter) -> MagicMock:
    reg = MagicMock(spec=ModelRegistry)
    reg.has_model.return_value = True
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    reg.is_unloading.return_value = False
    reg.is_failed.return_value = False
    reg.get_failure.return_value = None
    reg.get.return_value = fake_adapter
    reg.get_config.return_value = _make_config()
    reg.device = "cpu"
    reg.engine_config = None
    # Required by ``ensure_loaded`` short-circuit when already loaded.
    return reg


@pytest.fixture
def client(registry: MagicMock) -> TestClient:
    app = FastAPI()
    app.include_router(generate_router)
    app.state.registry = registry
    return TestClient(app)


class TestGenerateEndpoint:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("enable_thinking", [False, True])
    @pytest.mark.parametrize("model_id", ["Qwen/Qwen3.5-4B", "google/gemma-4-E2B-it"])
    def test_reasoning_output_is_hidden_for_every_resolved_profile(
        self,
        client: TestClient,
        registry: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        stream: bool,
        enable_thinking: bool,
        model_id: str,
    ) -> None:
        config = _make_config(model_id=model_id)
        gemma = model_id.startswith("google/gemma")
        opening = _GEMMA_OPEN if gemma else "<think>"
        assert config.tasks.generate is not None
        config.tasks.generate.chat_template_kwargs = {"enable_thinking": enable_thinking}
        if enable_thinking:
            config.inputs = InputModalities(text=True, image=True)

            async def render(_config: object, prompt: str, image_count: int) -> str:
                assert prompt == "Hello"
                assert image_count == 1
                return "rendered thinking prompt" + opening

            monkeypatch.setattr(generate_api, "_render_native_image_prompt", render)
        registry.get_config.return_value = config
        registry.get.return_value = _ThinkingGenAdapter(prompt_seeded=enable_thinking, gemma=gemma)

        request: dict[str, object] = {"prompt": "Hello", "max_new_tokens": 16, "stream": stream}
        if enable_thinking:
            request["images"] = [{"data": "aGVsbG8=", "format": "png"}]

        response = client.post(
            f"/v1/generate/{model_id.replace('/', '__')}",
            json=request,
        )

        assert response.status_code == 200, response.text
        if stream:
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            streamed_text = "".join(event.get("text_delta", "") for event in events)
            assert streamed_text == "Visible answer"
            assert "private reasoning" not in streamed_text
        else:
            assert response.json()["text"] == "Visible answer"

    @pytest.mark.parametrize("stream", [False, True])
    def test_text_only_generate_preserves_legacy_adapter_call_signature(
        self,
        client: TestClient,
        registry: MagicMock,
        stream: bool,
    ) -> None:
        legacy_adapter = _LegacyTextGenAdapter()
        registry.get.return_value = legacy_adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hello", "max_new_tokens": 8, "stream": stream},
        )

        assert response.status_code == 200, response.text
        assert legacy_adapter.last_call is not None
        assert legacy_adapter.last_call["prompt"] == "Hello"
        if stream:
            assert '"finish_reason": "error"' not in response.text
            assert "data: [DONE]" in response.text

    def test_request_body_is_rejected_before_unbounded_aggregation(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(generate_api, "_MAX_GENERATE_BODY_BYTES", 64)

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "x" * 128, "max_new_tokens": 8},
        )

        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "INPUT_TOO_LONG"

    @pytest.mark.parametrize("stream", [False, True])
    def test_native_images_render_model_prompt_and_reach_adapter_as_bytes(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
        monkeypatch: pytest.MonkeyPatch,
        stream: bool,
    ) -> None:
        config = _make_config()
        config.inputs = InputModalities(text=True, image=True)
        registry.get_config.return_value = config

        async def render(_config: object, prompt: str, image_count: int) -> str:
            assert prompt == "Read the image"
            assert image_count == 1
            return "<image>Read the image"

        monkeypatch.setattr("sie_server.api.generate._render_native_image_prompt", render)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Read the image",
                "images": [{"data": "aGVsbG8=", "format": "PNG"}],
                "max_new_tokens": 8,
                "stream": stream,
            },
        )

        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["prompt"] == "<image>Read the image"
        assert fake_adapter.last_call["images"] == [{"data": b"hello", "format": "png"}]

    @pytest.mark.parametrize("enable_thinking", [False, True])
    def test_native_image_prompt_uses_pinned_trusted_model_tokenizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        enable_thinking: bool,
    ) -> None:
        config = _make_config()
        config.hf_revision = "0123456789abcdef0123456789abcdef01234567"
        assert config.tasks.generate is not None
        config.tasks.generate.chat_template_kwargs = {"enable_thinking": enable_thinking}
        captured: dict[str, object] = {}

        class _Tokenizer:
            def apply_chat_template(self, messages: object, **kwargs: object) -> str:
                captured["messages"] = messages
                captured["template_kwargs"] = kwargs
                return "<image>Read"

        def load(source: object, **kwargs: object) -> _Tokenizer:
            captured["source"] = source
            captured["load_kwargs"] = kwargs
            return _Tokenizer()

        generate_api._load_native_tokenizer_cached.cache_clear()
        monkeypatch.setattr(generate_api, "load_tokenizer", load)
        rendered = asyncio.run(generate_api._render_native_image_prompt(config, "Read", 1))

        assert rendered == "<image>Read"
        assert captured["source"] == "Qwen/Qwen3-4B-Instruct"
        assert captured["load_kwargs"] == {
            "trust_remote_code": True,
            "revision": "0123456789abcdef0123456789abcdef01234567",
        }
        assert captured["messages"] == [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read"}]}
        ]
        assert captured["template_kwargs"] == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }

    def test_native_image_prompt_coalesces_and_caches_tokenizer_loads(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _make_config()
        config.hf_revision = "0123456789abcdef0123456789abcdef01234567"
        loads = 0

        class _Tokenizer:
            def apply_chat_template(self, _messages: object, **_kwargs: object) -> str:
                return "<image>Read"

        def load(_source: object, **_kwargs: object) -> _Tokenizer:
            nonlocal loads
            loads += 1
            return _Tokenizer()

        async def render_twice() -> list[str]:
            return await asyncio.gather(
                generate_api._render_native_image_prompt(config, "Read", 1),
                generate_api._render_native_image_prompt(config, "Read", 1),
            )

        generate_api._load_native_tokenizer_cached.cache_clear()
        monkeypatch.setattr(generate_api, "load_tokenizer", load)
        rendered = asyncio.run(render_twice())

        assert rendered == ["<image>Read", "<image>Read"]
        assert loads == 1

    @pytest.mark.parametrize(
        ("images", "expected_param"),
        [
            ([], "images"),
            ([{"data": "!!!"}], "images[0].data"),
            ([{"data": "aGk"}], "images[0].data"),
            ([{"data": "__8="}], "images[0].data"),
            ([{"data": "AB=="}], "images[0].data"),
            ([{"data": "aGVsbG8=", "url": "https://example.com/a.png"}], "images[0].url"),
            ([{"data": "aGVsbG8=", "format": "png;bad"}], "images[0].format"),
            ([{"data": "aGk="}] * 17, "images"),
        ],
    )
    def test_native_images_reject_malformed_envelopes_before_load(
        self,
        client: TestClient,
        registry: MagicMock,
        images: object,
        expected_param: str,
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Read", "images": images, "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["param"] == expected_param
        registry.load_async.assert_not_called()

    def test_native_images_reject_nonvision_model_before_load(
        self,
        client: TestClient,
        registry: MagicMock,
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Read", "images": [{"data": "aGVsbG8="}], "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "unsupported_field",
            "message": "Model 'Qwen__Qwen3-4B-Instruct' does not support image input",
            "param": "images",
        }
        registry.load_async.assert_not_called()

    def test_native_image_template_failure_is_rejected_before_model_load(
        self,
        client: TestClient,
        registry: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _make_config()
        config.inputs = InputModalities(text=True, image=True)
        registry.get_config.return_value = config
        registry.is_loaded.return_value = False
        registry.load_async = AsyncMock()

        async def reject_template(_config: object, _prompt: str, _image_count: int) -> str:
            raise generate_api._bad_request("image prompt template failed", param="images")

        monkeypatch.setattr(generate_api, "_render_native_image_prompt", reject_template)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Read", "images": [{"data": "aGVsbG8="}], "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "images"
        registry.load_async.assert_not_awaited()

    def test_happy_path_returns_text_finish_reason_usage(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hello", "max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
        )
        assert response.status_code == 200
        data = response.json()
        # Response echoes the canonical (slash-form) model id, not the raw
        # ``__``-form path param, so it round-trips with what the SDK sent.
        assert data["model"] == "Qwen/Qwen3-4B-Instruct"
        assert data["text"] == "echo:Hello"
        assert data["finish_reason"] == "stop"
        assert data["usage"]["completion_tokens"] == 2
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + 2

        # Adapter received the parsed sampling params verbatim.
        assert fake_adapter.last_call == {
            "prompt": "Hello",
            "max_new_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "top_k": None,
            "repetition_penalty": None,
            "min_new_tokens": None,
            "grammar": None,
            "seed": None,
            "logit_bias": None,
            "logprobs": False,
            "top_logprobs": None,
        }

    def test_runtime_options_apply_profile_then_request_then_typed_fields(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        config = _make_config()
        config.profiles["default"].adapter_options = AdapterOptions(
            runtime={
                "default_sampling": {"temperature": 0.7, "top_p": 0.8},
                "stop_tokens": ["</s>"],
            }
        )
        registry.get_config.return_value = config

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hello",
                "max_new_tokens": 32,
                "temperature": 1.0,
                "options": {
                    "default_sampling": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_new_tokens": 2},
                    "stop_tokens": ["END"],
                },
            },
        )

        assert response.status_code == 200
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["temperature"] == 1.0
        assert fake_adapter.last_call["top_p"] == 0.9
        assert fake_adapter.last_call["stop"] == ["END"]
        assert fake_adapter.last_call["top_k"] == 40
        assert fake_adapter.last_call["min_new_tokens"] == 2

    def test_profile_default_min_new_tokens_caps_to_explicit_max_before_adapter(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        config = _make_config()
        config.profiles["default"].adapter_options = AdapterOptions(
            runtime={"default_sampling": {"min_new_tokens": 10}}
        )
        registry.get_config.return_value = config

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hello", "max_new_tokens": 1},
        )

        assert response.status_code == 200
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["max_new_tokens"] == 1
        assert fake_adapter.last_call["min_new_tokens"] == 1

    def test_explicit_min_new_tokens_above_max_rejects_before_load(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        registry.is_loaded.return_value = False

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hello",
                "max_new_tokens": 1,
                "options": {"default_sampling": {"min_new_tokens": 10}},
            },
        )

        assert response.status_code == 400
        assert "min_new_tokens' (10) must not exceed max_new_tokens (1)" in response.json()["detail"]["message"]
        registry.load_async.assert_not_awaited()
        assert fake_adapter.last_call is None

    def test_non_default_options_profile_rejects_before_load(self, client: TestClient, registry: MagicMock) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hello", "max_new_tokens": 32, "options": {"profile": "fast"}},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "options.profile"
        assert "model:profile" in response.json()["detail"]["message"]
        registry.load_async.assert_not_called()

    def test_registry_lookup_uses_denormalized_slash_key(self, client: TestClient, registry: MagicMock) -> None:
        # Regression: the registry keys on the canonical slash ``sie_id``
        # (``ModelConfig.name``), so the ``__`` path segment must be
        # denormalized before lookup or every real model 404s.
        response = client.post(
            "/v1/generate/Qwen__Qwen3.5-4B",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200
        registry.has_model.assert_called_with("Qwen/Qwen3.5-4B")
        registry.get_config.assert_called_with("Qwen/Qwen3.5-4B")
        registry.get.assert_called_with("Qwen/Qwen3.5-4B")

    def test_slash_in_model_path_returns_400_with_suggestion(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen/Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 400
        body = response.json()
        # The suggested SIE-safe id should appear in the message.
        assert "Qwen__Qwen3-4B-Instruct" in body["detail"]["message"]

    def test_unknown_model_returns_404(self, client: TestClient, registry: MagicMock) -> None:
        registry.has_model.return_value = False
        response = client.post(
            "/v1/generate/unknown__model",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 404

    def test_missing_prompt_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"max_new_tokens": 8},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "prompt"

    def test_zero_max_new_tokens_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 0},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "max_new_tokens"

    def test_max_new_tokens_exceeds_cap_returns_400(self, client: TestClient) -> None:
        # The config caps at 4096; ask for 5000.
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 5000},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["code"] == "context_exceeded"
        assert body["detail"]["param"] == "max_new_tokens"

    def test_unsupported_field_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "tools": []},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["detail"]["code"] == "unsupported_field"
        assert body["detail"]["param"] == "tools"

    def test_non_generation_adapter_returns_400(self, client: TestClient, registry: MagicMock) -> None:
        # Registry returns a non-GenerationAdapter (e.g. an embedding adapter).
        registry.get.return_value = MagicMock(spec=[])  # plain MagicMock — not GenerationAdapter
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 400

    def test_model_loading_returns_503(self, client: TestClient, registry: MagicMock) -> None:
        registry.is_loading.return_value = True
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 503

    def test_stop_must_be_list_of_strings(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stop": "not-a-list"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "stop"

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("temperature", "0.7"),
            ("temperature", True),
            ("top_p", "0.9"),
            ("top_p", False),
        ],
    )
    def test_sampling_params_must_be_json_numbers(self, client: TestClient, param: str, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, param: value},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == param

    @pytest.mark.parametrize("param", ["routing_key", "prompt_cache_key", "safety_identifier"])
    def test_routing_hints_reject_non_string_values(self, client: TestClient, param: str) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, param: {"malformed": True}},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["param"] == param

    def test_schema_nullable_fields_are_treated_as_omitted(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hi",
                "max_new_tokens": 8,
                "temperature": None,
                "top_p": None,
                "routing_key": None,
                "prompt_cache_key": None,
                "safety_identifier": None,
            },
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["temperature"] == 1.0
        assert fake_adapter.last_call["top_p"] == 1.0

    def test_response_model_is_canonical_slash_id(self, client: TestClient) -> None:
        # The request path uses the SIE-safe ``__`` form, but the response
        # ``model`` field must be the canonical slash id so it round-trips
        # with what the SDK sent.
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "Qwen/Qwen3-4B-Instruct"

    def test_oversized_prompt_returns_413(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        # Shrink the cap so the test doesn't have to build a 4 MiB string.
        monkeypatch.setattr("sie_server.api.generate._MAX_PROMPT_BYTES", 16)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "x" * 64, "max_new_tokens": 8},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["detail"]["param"] == "prompt"
        assert body["detail"]["code"] == "INPUT_TOO_LONG"

    # ── Penalty forwarding and unsupported direct-worker grammar ──────

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("value", [999, -999, "x", True])
    def test_penalty_out_of_range_or_wrong_type_returns_400(
        self, client: TestClient, field: str, value: object
    ) -> None:
        """BUG 12: penalties must be validated identically to the gateway —
        finite number in [-2.0, 2.0]; reject out-of-range / string / bool.
        Previously these were whitelisted but never validated → 200.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, field: value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == field

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_penalty_nan_inf_returns_400(self, client: TestClient, field: str, literal: str) -> None:
        """NaN / inf (non-finite) penalties reject with 400 (gateway parity)."""
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            data=f'{{"prompt": "Hi", "max_new_tokens": 8, "{field}": {literal}}}',
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == field

    @pytest.mark.parametrize("field", ["frequency_penalty", "presence_penalty"])
    def test_valid_penalty_is_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter, field: str) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, field: 0.5},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call[field] == 0.5

    @pytest.mark.parametrize(
        ("grammar_body", "expected"),
        [
            (
                {
                    "json_schema": {
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                    "label": "document",
                    "strict": True,
                },
                GrammarSpec(
                    kind="json_schema",
                    value={
                        "type": "object",
                        "properties": {"title": {"type": "string"}},
                        "required": ["title"],
                    },
                    label="document",
                    strict=True,
                ),
            ),
            ({"regex": r"\d{3}-\d{4}"}, GrammarSpec(kind="regex", value=r"\d{3}-\d{4}")),
            ({"ebnf": 'root ::= "yes" | "no"'}, GrammarSpec(kind="ebnf", value='root ::= "yes" | "no"')),
        ],
    )
    @pytest.mark.parametrize("stream", [False, True])
    def test_native_grammar_reaches_adapter(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
        grammar_body: dict[str, object],
        expected: GrammarSpec,
        stream: bool,
    ) -> None:
        registry.get_config.return_value = _make_config(grammar=[expected.kind])

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Return structured output",
                "grammar": grammar_body,
                "max_new_tokens": 8,
                "stream": stream,
            },
        )

        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["grammar"] == expected

    def test_native_grammar_dereferences_internal_schema_refs(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        registry.get_config.return_value = _make_config(grammar=["json_schema"])

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Return structured output",
                "grammar": {
                    "json_schema": {
                        "$defs": {"Title": {"type": "string", "minLength": 1}},
                        "type": "object",
                        "properties": {"title": {"$ref": "#/$defs/Title"}},
                    }
                },
                "max_new_tokens": 8,
            },
        )

        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        grammar = fake_adapter.last_call["grammar"]
        assert isinstance(grammar, GrammarSpec)
        assert grammar.value == {
            "type": "object",
            "properties": {"title": {"type": "string", "minLength": 1}},
        }

    @pytest.mark.parametrize(
        "list_index",
        ["²", "9" * 5000],
        ids=["non-ascii", "overlong-ascii"],
    )
    def test_native_grammar_rejects_invalid_json_pointer_list_index(self, list_index: str) -> None:
        pointer = f"/{list_index}"
        with pytest.raises(KeyError) as key_error:
            generate_api._json_pointer(["value"], pointer)
        assert key_error.value.args == (pointer,)

        with pytest.raises(HTTPException) as exc_info:
            generate_api._parse_native_grammar(
                {
                    "json_schema": {
                        "$defs": [{"type": "string"}],
                        "$ref": f"#/$defs/{list_index}",
                    }
                }
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["param"] == "grammar.json_schema.$ref"

    def test_native_grammar_allows_property_names_that_match_unsupported_keywords(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        registry.get_config.return_value = _make_config(grammar=["json_schema"])
        schema = {
            "type": "object",
            "properties": {
                "if": {"type": "string"},
                "then": {"type": "integer"},
            },
        }

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Return structured output",
                "grammar": {"json_schema": schema},
                "max_new_tokens": 8,
            },
        )

        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        grammar = fake_adapter.last_call["grammar"]
        assert isinstance(grammar, GrammarSpec)
        assert grammar.value == schema

    def test_schema_helpers_reject_raw_traversal_depth_before_python_recursion(self) -> None:
        for helper in (generate_api._dereference_schema_refs, generate_api._validate_schema_shape):
            schema: dict[str, object] = {}
            cursor = schema
            for _ in range(generate_api._MAX_SCHEMA_TRAVERSAL_DEPTH + 1):
                child: dict[str, object] = {}
                cursor["unknown"] = child
                cursor = child

            with pytest.raises(HTTPException) as exc_info:
                helper(schema)

            assert exc_info.value.status_code == 400
            assert "traversal depth exceeds limit" in exc_info.value.detail["message"]

    @pytest.mark.parametrize(
        "grammar",
        [
            "not-an-object",
            {},
            {"json_schema": {}, "regex": "x"},
            {"regex": 123},
            {"ebnf": "root", "unknown": True},
            {"json_schema": {"$ref": "https://example.com/schema.json"}},
        ],
    )
    def test_native_grammar_rejects_invalid_shape_before_adapter(
        self,
        client: TestClient,
        registry: MagicMock,
        fake_adapter: _FakeGenAdapter,
        grammar: object,
    ) -> None:
        registry.get_config.return_value = _make_config(grammar=["json_schema", "regex", "ebnf"])

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "grammar": grammar, "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert fake_adapter.last_call is None

    def test_native_grammar_requires_model_capability(
        self,
        client: TestClient,
        fake_adapter: _FakeGenAdapter,
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "grammar": {"regex": "[a-z]+"}, "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["param"] == "grammar.regex"
        assert fake_adapter.last_call is None

    def test_prompt_at_cap_is_accepted(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        # A prompt exactly at the byte cap is allowed (boundary check).
        monkeypatch.setattr("sie_server.api.generate._MAX_PROMPT_BYTES", 16)
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "x" * 16, "max_new_tokens": 8},
        )
        assert response.status_code == 200

    # ── Adapter-supported seed / logit_bias / streaming logprobs ──────

    def test_seed_is_accepted_and_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        """``seed`` is whitelisted (the adapter forwards it) and reaches the
        adapter — previously a schema-compliant ``seed`` body 400'd as
        ``unsupported_field``.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": 1234},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["seed"] == 1234

    @pytest.mark.parametrize(
        "value",
        [
            -(1 << 63),
            -1,
            0,
            (1 << 63) - 1,
        ],
    )
    def test_seed_boundaries_match_gateway_signed_i64_contract(
        self,
        client: TestClient,
        fake_adapter: _FakeGenAdapter,
        value: int,
    ) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": value},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["seed"] == value

    @pytest.mark.parametrize("value", [-(1 << 63) - 1, 1 << 63])
    def test_seed_outside_gateway_integer_range_returns_400(self, client: TestClient, value: int) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == {
            "code": "INVALID_REQUEST",
            "message": "'seed' is outside the supported integer range",
            "param": "seed",
        }

    @pytest.mark.parametrize("value", ["x", 1.5, True])
    def test_seed_wrong_type_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "seed": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == {
            "code": "INVALID_REQUEST",
            "message": "'seed' must be an integer",
            "param": "seed",
        }

    def test_logit_bias_is_accepted_and_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logit_bias": {"123": 1.5, "456": -2.0}},
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["logit_bias"] == {"123": 1.5, "456": -2.0}

    @pytest.mark.parametrize(
        "value",
        [
            "not-an-object",
            {"abc": 1.0},  # non-integer key
            {"123": 999.0},  # out of [-100, 100]
            {"123": "x"},  # non-numeric value
        ],
    )
    def test_logit_bias_malformed_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logit_bias": value},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "logit_bias"

    def test_blocking_logprobs_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "logprobs": True, "top_logprobs": 5},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == {
            "code": "unsupported_field",
            "message": "'logprobs' is supported only with 'stream: true' on the native endpoint",
            "param": "logprobs",
        }

    def test_logprobs_wrong_type_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stream": True, "logprobs": "yes"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "logprobs"

    @pytest.mark.parametrize("value", [-1, 21, 1.5, True])
    def test_top_logprobs_out_of_range_returns_400(self, client: TestClient, value: object) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hi",
                "max_new_tokens": 8,
                "stream": True,
                "logprobs": True,
                "top_logprobs": value,
            },
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "top_logprobs"

    def test_top_logprobs_requires_logprobs_true(self, client: TestClient) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stream": True, "top_logprobs": 5},
        )
        assert response.status_code == 400, response.text
        assert response.json()["detail"]["param"] == "top_logprobs"

    def test_invalid_sampler_does_not_start_model_load(self, client: TestClient, registry: MagicMock) -> None:
        registry.is_loaded.return_value = False
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "temperature": -1},
        )
        assert response.status_code == 400, response.text
        assert registry.start_load_async.called is False

    def test_streaming_logprobs_are_forwarded(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={
                "prompt": "Hi",
                "max_new_tokens": 8,
                "stream": True,
                "logprobs": True,
                "top_logprobs": 5,
            },
        )
        assert response.status_code == 200, response.text
        assert fake_adapter.last_call is not None
        assert fake_adapter.last_call["logprobs"] is True
        assert fake_adapter.last_call["top_logprobs"] == 5

    # ── FIX 5: a terminal finish_reason of error / cancelled must NOT be
    # an HTTP 200 with partial text ──────────────────────────────────

    def test_terminal_error_finish_reason_returns_500(self, client: TestClient, fake_adapter: _FakeGenAdapter) -> None:
        """A stream that ends with ``finish_reason="error"`` (adapter caught
        an upstream failure and surfaced it as a terminal chunk rather than
        raising) must map to HTTP 500, not a 200 with partial text.
        """
        fake_adapter.finish_reason = "error"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 500, response.text
        assert response.json()["detail"]["code"] == "inference_error"

    def test_terminal_empty_model_output_returns_500_with_verbatim_code(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        """#3104/#3136: an ``empty_model_output`` terminal keeps a
        ``stop``/``length`` finish_reason, so the buffered route must key on
        the typed ``error_code`` — not answer HTTP 200 with empty text — and
        surface the code/message verbatim, exactly like the streaming shape.
        """
        fake_adapter.finish_reason = "length"
        fake_adapter.error_code = "empty_model_output"
        fake_adapter.error_message = "model produced no visible output text"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 500, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "empty_model_output"
        assert detail["message"] == "model produced no visible output text"
        assert "retry-after" not in {k.lower() for k in response.headers}

    def test_terminal_error_with_typed_code_is_not_flattened_to_inference_error(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        """A parser-style ``finish_reason="error"`` terminal that carries its
        own typed code must surface that code verbatim, not the hardcoded
        ``inference_error`` fallback (#3136).
        """
        fake_adapter.finish_reason = "error"
        fake_adapter.error_code = "tool_call_parse_error"
        fake_adapter.error_message = "model emitted a malformed tool call"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 500, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "tool_call_parse_error"
        assert detail["message"] == "model emitted a malformed tool call"

    @pytest.mark.parametrize(
        ("code", "message", "expected_code", "expected_message"),
        [
            (
                "inference_error",
                "SENSITIVE_NATIVE_BUFFERED_RUNTIME",
                "inference_error",
                "internal error during generation",
            ),
            (
                "grammar_compile_failed",
                "SENSITIVE_NATIVE_BUFFERED_GRAMMAR",
                "grammar_compile_failed",
                "internal error compiling grammar",
            ),
            (
                "MODEL_OUTPUT_PARSE_ERROR",
                "invalid model JSON",
                "MODEL_OUTPUT_PARSE_ERROR",
                "invalid model JSON",
            ),
            (
                "SENSITIVE_NATIVE_BUFFERED_ERROR_CODE",
                "SENSITIVE_NATIVE_BUFFERED_ERROR_CODE",
                "inference_error",
                "generation terminated with an upstream error",
            ),
        ],
    )
    def test_terminal_error_message_is_classified_at_buffered_boundary(
        self,
        client: TestClient,
        fake_adapter: _FakeGenAdapter,
        code: str,
        message: str,
        expected_code: str,
        expected_message: str,
    ) -> None:
        fake_adapter.finish_reason = "error"
        fake_adapter.error_code = code
        fake_adapter.error_message = message

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 500, response.text
        assert response.json()["detail"] == {"code": expected_code, "message": expected_message}
        if message.startswith("SENSITIVE_"):
            assert message not in response.text

    def test_terminal_cancelled_finish_reason_returns_503(
        self, client: TestClient, fake_adapter: _FakeGenAdapter
    ) -> None:
        """A stream that ends with ``finish_reason="cancelled"`` must map to a
        non-2xx (503), not a 200 with partial text.
        """
        fake_adapter.finish_reason = "cancelled"
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == "generation_cancelled"

    def test_terminal_stop_finish_reason_still_returns_200(self, client: TestClient) -> None:
        """Sanity: the normal ``stop`` terminator is unaffected by the
        error/cancelled mapping and still 200s.
        """
        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )
        assert response.status_code == 200, response.text

    @pytest.mark.parametrize("stream", [False, True])
    def test_preflight_receives_exact_dispatched_parameters(
        self,
        client: TestClient,
        registry: MagicMock,
        stream: bool,
    ) -> None:
        adapter = _PreflightAdapter()
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stream": stream},
        )

        assert response.status_code == 200, response.text
        assert adapter.events[:2] == ["preflight", "generate"]
        assert adapter.preflight_stream is stream
        assert adapter.preflight_parameters == adapter.dispatched_parameters
        if stream:
            assert adapter.preflight_parameters is not None
            assert adapter.preflight_parameters["logprobs"] is False
            assert adapter.preflight_parameters["top_logprobs"] is None
        else:
            assert adapter.preflight_parameters is not None
            assert "logprobs" not in adapter.preflight_parameters
            assert "top_logprobs" not in adapter.preflight_parameters

    def test_streaming_capability_rejects_before_adapter_lookup(
        self,
        client: TestClient,
        registry: MagicMock,
    ) -> None:
        registry.get_config.return_value = _make_config(streaming=False)

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8, "stream": True},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "unsupported_field",
            "message": "Model 'Qwen__Qwen3-4B-Instruct' does not support streaming generation",
            "param": "stream",
        }
        registry.get.assert_not_called()

    def test_preflight_unsupported_field_is_exact_400(
        self,
        client: TestClient,
        registry: MagicMock,
    ) -> None:
        adapter = _PreflightAdapter(GenerationUnsupportedFieldError("top_k"))
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "unsupported_field"
        assert response.json()["detail"]["param"] == "top_k"
        assert adapter.events == ["preflight"]

    def test_preflight_input_too_long_is_exact_400(
        self,
        client: TestClient,
        registry: MagicMock,
    ) -> None:
        adapter = _PreflightAdapter(
            GenerationInputTooLongError(
                "prompt token count (129) exceeds max_input_len (128)",
            )
        )
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "code": "INPUT_TOO_LONG",
            "message": "prompt token count (129) exceeds max_input_len (128)",
            "param": "prompt",
        }
        assert adapter.events == ["preflight"]

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            (_UntrustedUnsupportedFieldError("top_k", "private unsupported detail"), 400),
            (_UntrustedCapacityError("private capacity detail"), 503),
        ],
    )
    def test_typed_generation_errors_sanitize_untrusted_codes(
        self,
        client: TestClient,
        registry: MagicMock,
        error: GenerationError,
        status_code: int,
    ) -> None:
        adapter = _PreflightAdapter(error)
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == status_code
        assert response.json()["detail"]["code"] == "inference_error"
        assert "SENSITIVE_" not in response.text

    @pytest.mark.parametrize(
        ("raise_during_generate", "message", "events"),
        [
            (False, "internal error during generation preflight", ["preflight"]),
            (True, "internal error during generation", ["preflight", "generate"]),
        ],
    )
    def test_unexpected_generation_errors_do_not_leak_exception_text(
        self,
        client: TestClient,
        registry: MagicMock,
        raise_during_generate: bool,
        message: str,
        events: list[str],
    ) -> None:
        adapter = _PreflightAdapter(
            RuntimeError("sensitive internal generation detail"),
            raise_during_generate=raise_during_generate,
        )
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 500
        assert response.json()["detail"] == {"code": "inference_error", "message": message}
        assert "sensitive internal generation detail" not in response.text
        assert adapter.events == events

    @pytest.mark.parametrize("raise_during_generate", [False, True])
    def test_generic_typed_generation_error_does_not_leak_exception_text(
        self,
        client: TestClient,
        registry: MagicMock,
        raise_during_generate: bool,
    ) -> None:
        sentinel = "SENSITIVE_TYPED_NATIVE_BUFFERED"
        adapter = _PreflightAdapter(
            GenerationError(sentinel),
            raise_during_generate=raise_during_generate,
        )
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 500
        assert response.json()["detail"] == {
            "code": "inference_error",
            "message": "internal error during generation",
        }
        assert sentinel not in response.text

    @pytest.mark.parametrize(
        ("error", "code", "retry_after"),
        [
            (GenerationCapacityError("scheduler full"), "RESOURCE_EXHAUSTED", "5"),
            (GenerationDrainingError("scheduler draining"), "MODEL_LOADING", "5"),
        ],
    )
    def test_typed_capacity_after_preflight_remains_retryable(
        self,
        client: TestClient,
        registry: MagicMock,
        error: GenerationError,
        code: str,
        retry_after: str,
    ) -> None:
        adapter = _PreflightAdapter(error, raise_during_generate=True)
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == code
        assert response.headers["retry-after"] == retry_after
        assert adapter.events == ["preflight", "generate"]

    def test_generation_capacity_uses_configured_resource_exhausted_retry_hint(
        self,
        client: TestClient,
        registry: MagicMock,
    ) -> None:
        registry.engine_config = EngineConfig.model_validate({"oom_recovery": {"retry_after_s": 12}})
        adapter = _PreflightAdapter(GenerationCapacityError("scheduler full"))
        registry.get.return_value = adapter

        response = client.post(
            "/v1/generate/Qwen__Qwen3-4B-Instruct",
            json={"prompt": "Hi", "max_new_tokens": 8},
        )

        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "RESOURCE_EXHAUSTED"
        assert response.headers["retry-after"] == "12"
