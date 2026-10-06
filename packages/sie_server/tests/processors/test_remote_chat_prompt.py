"""Onboarded direct chat retains SIE's own template/parser without replay."""

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from sie_server.adapters._generation_base import (
    GenerationChunk,
    GenerationInvalidRequestError,
    GenerationUnsupportedFieldError,
)
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import ModelConfig
from sie_server.processors import remote_chat_prompt


class RawAdapter(OpenAIUpstreamAdapter):
    def __init__(self, chunks: list[GenerationChunk]) -> None:
        super().__init__(upstream="team", upstream_model="operator/model")
        self.chunks = chunks
        self.parameters: dict[str, Any] = {}
        self.closed = False

    @property
    def supports_raw_completions(self) -> bool:
        return True

    def preflight_generate(self, parameters, *, stream):
        self.parameters = dict(parameters)

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True


def model(tool_call_parser: str = "hermes") -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "local/model",
            "hf_id": "onboarded/model",
            "hf_revision": "a" * 40,
            "tasks": {
                "generate": {
                    "context_length": 4096,
                    "max_output_tokens": 64,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            },
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.sglang:SGLangGenerationAdapter",
                    "max_batch_tokens": 8192,
                    "kv_budget_tokens": 8192,
                    "adapter_options": {"loadtime": {"tool_call_parser": tool_call_parser}},
                }
            },
        }
    )


def body(**changes) -> dict:
    return {"messages": [{"role": "user", "content": "question"}], "max_tokens": 64, **changes}


@pytest.fixture
def template(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    def render(messages, **kwargs):
        observed.update(messages=messages, kwargs=kwargs)
        return "rendered<think>"

    monkeypatch.setattr(
        remote_chat_prompt,
        "_tokenizer",
        lambda *args: SimpleNamespace(chat_template="template", apply_chat_template=render),
    )
    return observed


async def prepare(adapter, request, config=None):
    return await remote_chat_prompt.prepare_rendered_chat(
        adapter, request, config=config or model(), requested_model="local/model", max_response_bytes=1024
    )


@pytest.mark.asyncio
async def test_template_defaults_and_buffered_privacy_usage(template) -> None:
    adapter = RawAdapter(
        [
            GenerationChunk(text_delta="private</think>answer"),
            GenerationChunk(
                text_delta="", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=3, cached_tokens=2
            ),
        ]
    )
    events = await prepare(
        adapter, body(chat_template_kwargs={"enable_thinking": True, "extra": True}, temperature=None, top_p=None)
    )
    assert events is not None
    payload = await remote_chat_prompt.collect_rendered_chat(events)
    assert payload["choices"][0]["message"]["content"] == "answer"
    assert payload["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    assert template["kwargs"]["enable_thinking"] is False
    assert template["kwargs"]["extra"] is True
    assert adapter.parameters["prompt"] == "rendered<think>"
    assert adapter.parameters["temperature"] == 1.0
    assert adapter.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"n": 2},
        {"tool_choice": "required"},
        {"response_format": {"type": "json_object"}},
        {"logprobs": True},
        {"logit_bias": {"2": 1}},
        {"min_p": 0.5},
        {"repetition_context_size": 128},
        {"role_mapping": {"user": "system"}},
        {"user": "application-user"},
        {"safety_identifier": "safety-id"},
        {"tools": [{"type": "function", "function": {"name": "lookup", "strict": True}}]},
    ],
)
async def test_modes_requiring_chat_keep_chat_ownership(template, changes) -> None:
    adapter = RawAdapter([])
    assert await prepare(adapter, body(**changes)) is None
    assert not adapter.parameters
    assert not template


@pytest.mark.asyncio
async def test_local_tool_parser_and_prior_arguments(template) -> None:
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    adapter = RawAdapter(
        [
            GenerationChunk(text_delta='</think><tool_call>{"name":"lookup","arguments":{"x":1}}</tool_call>'),
            GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=3),
        ]
    )
    request = body(
        tools=[tool],
        messages=[
            {"role": "developer", "content": "system"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "old", "type": "function", "function": {"name": "lookup", "arguments": '{"x":0}'}}
                ],
            },
        ],
    )
    events = await prepare(adapter, request)
    assert events is not None
    payload = await remote_chat_prompt.collect_rendered_chat(events)
    call = payload["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"x": 1}
    assert template["messages"][0]["role"] == "system"
    assert template["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"x": 0}
    assert request["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"x":0}'
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_rendered_chat_tool_arguments_follow_the_request_schema(template) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "edit",
            "parameters": {"type": "object", "properties": {"old": {"type": "string"}, "line": {"type": "integer"}}},
        },
    }
    adapter = RawAdapter(
        [
            GenerationChunk(
                text_delta="</think><tool_call>\n<function=edit>\n<parameter=old>\n    return 1.10\n</parameter>\n"
                "<parameter=line>\n7\n</parameter>\n</function>\n</tool_call>"
            ),
            GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=3),
        ]
    )
    events = await prepare(adapter, body(tools=[tool]), config=model("qwen3_coder"))
    assert events is not None
    payload = await remote_chat_prompt.collect_rendered_chat(events)
    call = payload["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"old": "    return 1.10", "line": 7}


@pytest.mark.asyncio
async def test_none_choice_hides_tools_from_template(template) -> None:
    adapter = RawAdapter(
        [
            GenerationChunk(
                text_delta="</think>answer", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=1
            )
        ]
    )
    events = await prepare(
        adapter, body(tool_choice="none", tools=[{"type": "function", "function": {"name": "lookup"}}])
    )
    assert events is not None
    await remote_chat_prompt.collect_rendered_chat(events)
    assert "tools" not in template["kwargs"]


@pytest.mark.asyncio
async def test_closing_visible_stream_closes_generation(template) -> None:
    adapter = RawAdapter([GenerationChunk(text_delta="</think>answer"), GenerationChunk(text_delta="late")])
    events = await prepare(adapter, body(stream=True))
    assert events is not None
    event = await anext(events)
    assert event["choices"][0]["delta"]["content"] == "answer"
    await events.aclose()
    assert adapter.closed


@pytest.mark.asyncio
async def test_missing_exact_usage_never_produces_buffered_success(template) -> None:
    adapter = RawAdapter([GenerationChunk(text_delta="</think>answer", done=True, finish_reason="stop")])
    events = await prepare(adapter, body())
    assert events is not None
    with pytest.raises(RemoteUpstreamError, match="exact usage"):
        await remote_chat_prompt.collect_rendered_chat(events)
    assert adapter.closed


@pytest.mark.asyncio
async def test_render_failure_is_fixed_and_does_not_dispatch(template, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("private checkpoint detail")

    monkeypatch.setattr(
        remote_chat_prompt,
        "_tokenizer",
        lambda *args: SimpleNamespace(chat_template="template", apply_chat_template=fail),
    )
    adapter = RawAdapter([])
    with pytest.raises(GenerationInvalidRequestError, match="model-native") as failure:
        await prepare(adapter, body())
    assert "private" not in str(failure.value)
    assert not adapter.parameters


@pytest.mark.asyncio
async def test_native_sie_answer_already_owns_prompt_reasoning_suppression(
    template, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = SieUpstreamAdapter(upstream="team", upstream_model="operator/model")
    monkeypatch.setattr(adapter, "preflight_generate", lambda *args, **kwargs: None)

    async def generate(*args, **kwargs):
        yield GenerationChunk(text_delta="answer")
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=3)

    monkeypatch.setattr(adapter, "generate", generate)
    events = await prepare(adapter, body())
    assert events is not None
    payload = await remote_chat_prompt.collect_rendered_chat(events)
    assert payload["choices"][0]["message"]["content"] == "answer"


@pytest.mark.asyncio
async def test_string_stop_is_normalized_before_native_sie_preflight(template, monkeypatch) -> None:
    adapter = SieUpstreamAdapter(upstream="team", upstream_model="operator/model")
    observed = {}
    monkeypatch.setattr(adapter, "preflight_generate", lambda parameters, **kwargs: observed.update(parameters))
    events = await prepare(adapter, body(stop="END"))
    assert events is not None
    assert observed["stop"] == ["END"]
    await events.aclose()


@pytest.mark.asyncio
async def test_unsupported_raw_field_selects_chat_without_dispatch(template, monkeypatch) -> None:
    adapter = RawAdapter([])

    def preflight(parameters, **kwargs):
        raise GenerationUnsupportedFieldError("top_k", "unsupported by raw completions")

    monkeypatch.setattr(adapter, "preflight_generate", preflight)
    assert await prepare(adapter, body(top_k=4)) is None
    assert not adapter.parameters
    assert not adapter.closed


@pytest.mark.asyncio
async def test_nullable_controls_retain_default_parallel_and_legacy_output_cap(template) -> None:
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    adapter = RawAdapter(
        [
            GenerationChunk(
                text_delta='</think><tool_call>{"name":"lookup","arguments":{}}</tool_call><tool_call>{"name":"lookup","arguments":{}}</tool_call>'
            ),
            GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=12, completion_tokens=3),
        ]
    )
    request = body(max_completion_tokens=None, parallel_tool_calls=None, tools=[tool])
    events = await prepare(adapter, request)
    assert events is not None
    payload = await remote_chat_prompt.collect_rendered_chat(events)
    assert len(payload["choices"][0]["message"]["tool_calls"]) == 2
    assert adapter.parameters["max_new_tokens"] == 64
    assert request["parallel_tool_calls"] is None
