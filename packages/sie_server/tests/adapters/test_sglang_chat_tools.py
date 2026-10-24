"""The single-server chat route and the queued worker must render one request to the same prompt."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import msgpack
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, model_serializer
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.api import openai_local
from sie_server.processors import streaming as streaming_mod
from sie_server.processors.streaming import StreamingProcessor
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

_COMPAT_DIR = Path(__file__).parents[2] / "src/sie_server/adapters/sglang/_compat"
_SERVING_CHAT = "sglang.srt.entrypoints.openai.serving_chat"
_MODEL = "Qwen/Qwen3.8-27B-FP8"
_TEMPLATE_KWARGS = {"enable_thinking": False}
_UNKNOWN = "[UNK]"

# Each template keeps only the tool and message handling of one family the
# catalog serves with tools: the Qwen3.x templates write every tool with
# ``tojson``, GLM-5.3 writes the function's keys in order without ``strict``
# and ``defer_loading``, and Gemma 4 reads the function's fields by name.
_QWEN_TEMPLATE = """\
{%- if tools %}<|im_start|>system
# Tools

<tools>{% for tool in tools %}
{{ tool | tojson }}{% endfor %}
</tools>{% if messages[0].role == 'system' %}

{{ messages[0].content }}{% endif %}<|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for message in messages %}
{%- if message.role == 'user' %}<|im_start|>user
{{ message.content }}<|im_end|>
{% elif message.role == 'assistant' %}<|im_start|>assistant
{{ message.content }}{% for call in message.tool_calls or [] %}<tool_call>
<function={{ call.function.name }}>{% for key, value in call.function.arguments | items %}
<parameter={{ key }}>{{ value }}</parameter>{% endfor %}
</function></tool_call>{% endfor %}<|im_end|>
{% elif message.role == 'tool' %}<|im_start|>user
<tool_response>{{ message.content }}</tool_response><|im_end|>
{% elif message.role != 'system' or not loop.first %}{{ raise_exception('Unexpected message role.') }}
{%- endif %}
{%- endfor %}<|im_start|>assistant
{% if enable_thinking is false %}<think>

</think>

{% endif %}"""

_GLM_TEMPLATE = """\
{%- if tools %}<|system|>
<tools>{% for tool in tools %}{% if 'function' in tool %}{% set tool = tool['function'] %}{% endif %}
{ {%- for key, value in tool.items() if key not in ('defer_loading', 'strict') %}{% if not loop.first %}, {% endif %}\
"{{ key }}": {{ value | tojson(ensure_ascii=False) }}{% endfor %}}{% endfor %}
</tools>{% endif %}
{%- for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}<|assistant|>"""

_GEMMA_TEMPLATE = """\
{%- for tool in tools or [] %}<|tool>declaration:{{ tool['function']['name'] }}\
{description:{{ tool['function']['description'] }}}<tool|>{% endfor %}
{%- for message in messages %}<turn>{{ message.role }}
{{ message.content }}<turn|>{% endfor %}<turn>model
"""

_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, for example Paris"}},
            "required": ["city"],
        },
    },
}
_PING = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}
_STRICT = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look one key up.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]},
        "strict": True,
    },
}
_ASK = {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}
_CALL = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
    ],
}
_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "18 C and sunny"}

_REQUESTS: dict[str, dict[str, Any]] = {
    "one_tool": {"messages": [_ASK], "tools": [_WEATHER]},
    "tool_exchange": {
        "messages": [{"role": "system", "content": "Answer briefly."}, _ASK, _CALL, _RESULT],
        "tools": [_WEATHER, _PING],
    },
    "strict_tool": {"messages": [_ASK], "tools": [_STRICT, _WEATHER]},
    "named_tool_choice": {
        "messages": [_ASK],
        "tools": [_PING, _WEATHER],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    },
    "tool_choice_none": {"messages": [_ASK], "tools": [_WEATHER], "tool_choice": "none"},
    "developer_message": {"messages": [{"role": "developer", "content": "Answer briefly."}, _ASK], "tools": [_WEATHER]},
}


def _load_hook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sie_chat_tools_test", _COMPAT_DIR / "sie_chat_tools.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tokenizer(template: str) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({_UNKNOWN: 0}, unk_token=_UNKNOWN))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token=_UNKNOWN)
    tokenizer.chat_template = template
    return tokenizer


def _sglang_serving_chat(version: str, tokenizer: PreTrainedTokenizerFast) -> ModuleType:
    """The request models and chat-template path of the pinned SGLang releases."""

    class Function(BaseModel):
        description: str | None = Field(default=None)
        name: str
        parameters: object | None = None
        strict: bool = False

    class Tool(BaseModel):
        type: str = Field(default="function")
        function: Function

    class Function0520(Function):
        defer_loading: bool | None = None

        @model_serializer(mode="wrap")
        def _serialize(self, handler: Any) -> Any:
            data = handler(self)
            if self.defer_loading is None:
                data.pop("defer_loading", None)
            return data

    class Tool0520(BaseModel):
        type: str = Field(default="function")
        function: Function0520
        defer_loading: bool | None = None

    class FunctionResponse(BaseModel):
        name: str | None = None
        arguments: str | dict[str, Any] | None = None

    class ToolCall(BaseModel):
        id: str | None = None
        index: int | None = None
        type: Literal["function"] = "function"
        function: FunctionResponse

    class Message(BaseModel):
        role: str
        content: str | None = None
        tool_call_id: str | None = None
        name: str | None = None
        reasoning_content: str | None = None
        tool_calls: list[ToolCall] | None = None

    class ToolChoiceFunction(BaseModel):
        name: str

    class ToolChoice(BaseModel):
        type: Literal["function"] = "function"
        function: ToolChoiceFunction

    tool_model = Tool0520 if version == "0.5.20" else Tool

    class ChatCompletionRequest(BaseModel):
        model_config = ConfigDict(extra="ignore")

        messages: list[Message]
        tools: list[tool_model] | None = None  # type: ignore[valid-type]
        tool_choice: Literal["none", "auto", "required"] | ToolChoice = "auto"
        chat_template_kwargs: dict[str, Any] | None = None

    class OpenAIServingChat:
        def __init__(self) -> None:
            self._chat_template_cache: dict[str, str] = {}

        async def handle_request(self, request: ChatCompletionRequest, raw_request: Any) -> str:
            return self._convert_to_internal_request(request, raw_request)

        def _convert_to_internal_request(self, request: ChatCompletionRequest, raw_request: Any) -> str:
            return self._process_messages(request, False)

        def _process_messages(self, request: ChatCompletionRequest, is_multimodal: bool) -> str:
            tools = None
            if request.tools and request.tool_choice != "none":
                if not isinstance(request.tool_choice, str):
                    tools = [
                        item.model_dump()
                        for item in request.tools
                        if item.function.name == request.tool_choice.function.name
                    ]
                else:
                    tools = [item.model_dump() for item in request.tools]
            return self._apply_jinja_template(request, tools, is_multimodal)

        def _apply_jinja_template(self, request: ChatCompletionRequest, tools: Any, is_multimodal: bool) -> str:
            messages = []
            for message in request.messages:
                rendered = message.model_dump()
                if rendered["content"] is None:
                    rendered["content"] = ""
                for call in rendered["tool_calls"] or []:
                    if isinstance(call["function"]["arguments"], str):
                        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                messages.append(rendered)
            template_kwargs = dict(request.chat_template_kwargs or {})
            if version == "0.5.20":
                return self._render_and_encode_chat_template(
                    messages, tools=tools, template_kwargs=template_kwargs, encode_kwargs={}, use_cache=True
                )
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, tools=tools, **template_kwargs
            )

        def _render_and_encode_chat_template(
            self,
            messages: list[dict[str, Any]],
            *,
            tools: Any,
            template_kwargs: dict[str, Any],
            encode_kwargs: dict[str, Any],
            use_cache: bool,
        ) -> str:
            key = json.dumps((messages, tools, template_kwargs, encode_kwargs), sort_keys=True) if use_cache else None
            if key is not None and key in self._chat_template_cache:
                return self._chat_template_cache[key]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, tools=tools, **template_kwargs
            )
            if key is not None:
                self._chat_template_cache[key] = rendered
            return rendered

    module = ModuleType(_SERVING_CHAT)
    module.ChatCompletionRequest = ChatCompletionRequest  # type: ignore[attr-defined]
    module.OpenAIServingChat = OpenAIServingChat  # type: ignore[attr-defined]
    return module


class _RawRequest:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def json(self) -> Any:
        return json.loads(self._body)


def _single_server_prompts(
    monkeypatch: pytest.MonkeyPatch, requests: list[dict[str, Any]], serving_chat: ModuleType, *, hook: bool
) -> list[str]:
    """Send the requests through the single-server route and render what reaches one child."""
    proxied: list[bytes] = []

    def _child(child_request: httpx.Request) -> httpx.Response:
        proxied.append(child_request.content)
        message = {"role": "assistant", "content": "ok"}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "served",
                "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    adapter = SGLangGenerationAdapter(model_name_or_path=_MODEL, served_model_name="served")
    adapter._server_url = "http://127.0.0.1:30400"
    profile = SimpleNamespace(runtime={"default_sampling": {}}, loadtime={})
    config = SimpleNamespace(
        tasks=SimpleNamespace(
            generate=SimpleNamespace(
                max_output_tokens=64,
                chat_template_kwargs=dict(_TEMPLATE_KWARGS),
                capabilities=SimpleNamespace(tools=True),
            ),
            score=None,
        ),
        resolve_profile=MagicMock(return_value=profile),
        inputs=SimpleNamespace(video=False),
        profiles={},
        routing=None,
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
    monkeypatch.setattr(
        openai_local, "_new_proxy_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(_child))
    )
    app = FastAPI()
    app.include_router(openai_local.router)
    app.state.registry = registry
    client = TestClient(app)
    for request in requests:
        response = client.post("/v1/chat/completions", json={"model": _MODEL, **request})
        assert response.status_code == 200, response.text

    if hook:
        monkeypatch.setitem(sys.modules, _SERVING_CHAT, serving_chat)
        _load_hook().install_chat_tools_as_sent()
    serving = serving_chat.OpenAIServingChat()
    return [
        asyncio.run(
            serving.handle_request(serving_chat.ChatCompletionRequest.model_validate_json(body), _RawRequest(body))
        )
        for body in proxied
    ]


class _CapturingAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None
        self.prompts: list[str] = []

    def load(self, device: str) -> None:  # pragma: no cover - registry-mocked
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        self.prompts.append(prompt)
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")


def _queued_prompt(monkeypatch: pytest.MonkeyPatch, request: dict[str, Any], tokenizer: Any) -> str:
    """Publish the request as the gateway does and return the prompt the worker renders."""
    messages = []
    for message in request["messages"]:
        forwarded = {"role": "system" if message["role"] == "developer" else message["role"]}
        forwarded["content"] = message.get("content") or ""
        for key in ("tool_calls", "tool_call_id"):
            if message.get(key) is not None:
                forwarded[key] = copy.deepcopy(message[key])
        messages.append(forwarded)
    generate: dict[str, Any] = {"messages": messages, "max_new_tokens": 64, "tools": copy.deepcopy(request["tools"])}
    if "tool_choice" in request:
        generate["tool_choice"] = request["tool_choice"]
    work_item = {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": _MODEL,
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": 0.0,
        "generate": generate,
    }
    msg = AsyncMock()
    msg.data = msgpack.packb(work_item, use_bin_type=True)

    adapter = _CapturingAdapter()
    registry = MagicMock()
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.device = "cuda:0"
    config = MagicMock()
    config.hf_id = _MODEL
    config.name = _MODEL
    config.hf_revision = None
    config.weights_path = None
    config.tasks.generate.chat_template_kwargs = dict(_TEMPLATE_KWARGS)
    config.tasks.generate.context_length = 32768
    config.tasks.generate.max_output_tokens = 512
    profile = MagicMock()
    profile.loadtime = {"tool_call_parser": "qwen3_coder"}
    config.resolve_profile.return_value = profile
    registry.get_config.return_value = config
    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *_args, **_kwargs: tokenizer)

    processor = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")
    asyncio.run(processor.process(msg, _MODEL))
    assert len(adapter.prompts) == 1, msgpack.unpackb(processor._nc.publish.await_args.args[1], raw=False)
    return adapter.prompts[0]


@pytest.mark.parametrize("version", ["0.5.10.post1", "0.5.20"])
@pytest.mark.parametrize("template", [_QWEN_TEMPLATE, _GLM_TEMPLATE, _GEMMA_TEMPLATE], ids=["qwen", "glm", "gemma"])
@pytest.mark.parametrize("name", sorted(_REQUESTS))
def test_single_server_and_queued_routes_render_the_same_prompt(
    monkeypatch: pytest.MonkeyPatch, name: str, template: str, version: str
) -> None:
    request = _REQUESTS[name]
    tokenizer = _tokenizer(template)

    queued = _queued_prompt(monkeypatch, request, tokenizer)
    (single,) = _single_server_prompts(monkeypatch, [request], _sglang_serving_chat(version, tokenizer), hook=True)

    assert single == queued


def test_sglang_own_tool_dump_changes_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _tokenizer(_QWEN_TEMPLATE)
    request = _REQUESTS["one_tool"]

    (single,) = _single_server_prompts(monkeypatch, [request], _sglang_serving_chat("0.5.20", tokenizer), hook=False)

    assert single != _queued_prompt(monkeypatch, request, tokenizer)
    assert '"strict": false}, "defer_loading": null}' in single


@pytest.mark.parametrize(
    ("template", "reordered"),
    [(_QWEN_TEMPLATE, "tools"), (_GLM_TEMPLATE, "tools"), (_QWEN_TEMPLATE, "arguments")],
    ids=["qwen-tools", "glm-tools", "qwen-arguments"],
)
def test_requests_differing_only_in_key_order_keep_their_own_prompts(
    monkeypatch: pytest.MonkeyPatch, template: str, reordered: str
) -> None:
    tokenizer = _tokenizer(template)
    first = copy.deepcopy(_REQUESTS["tool_exchange"])
    first["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"city": "Paris", "unit": "C"}'
    second = copy.deepcopy(first)
    if reordered == "tools":
        second["tools"] = json.loads(json.dumps(first["tools"], sort_keys=True))
    else:
        second["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"unit": "C", "city": "Paris"}'

    prompts = _single_server_prompts(monkeypatch, [first, second], _sglang_serving_chat("0.5.20", tokenizer), hook=True)

    assert prompts == [_queued_prompt(monkeypatch, first, tokenizer), _queued_prompt(monkeypatch, second, tokenizer)]
    assert prompts[0] != prompts[1]


def test_tools_as_sent_keeps_sglang_tools_for_another_body() -> None:
    hook = _load_hook()
    serving_chat = _sglang_serving_chat("0.5.20", _tokenizer(_QWEN_TEMPLATE))
    parsed = serving_chat.ChatCompletionRequest.model_validate({"messages": [_ASK], "tools": [_WEATHER]}).tools
    rendered = [tool.model_dump() for tool in parsed]

    assert hook.tools_as_sent([_WEATHER], parsed, rendered) == [_WEATHER]
    assert hook.tools_as_sent([_WEATHER], parsed, None) is None
    assert hook.tools_as_sent(None, parsed, rendered) is rendered
    assert hook.tools_as_sent([_PING], parsed, rendered) is rendered
    assert hook.tools_as_sent([_WEATHER, _PING], parsed, rendered) is rendered
    anthropic = [{"name": "get_weather", "input_schema": _WEATHER["function"]["parameters"]}]
    assert hook.tools_as_sent(anthropic, parsed, rendered) is rendered


def test_child_sitecustomize_installs_the_hook(tmp_path: Path) -> None:
    package = tmp_path / "sglang/srt/entrypoints/openai"
    package.mkdir(parents=True)
    for directory in (package, *package.parents):
        if directory == tmp_path:
            break
        (directory / "__init__.py").write_text("")
    (package / "serving_chat.py").write_text(
        "class OpenAIServingChat:\n"
        "    async def handle_request(self, request, raw_request):\n"
        "        return self._apply_jinja_template(request, ['dumped'], False)\n"
        "    def _apply_jinja_template(self, request, tools, is_multimodal):\n"
        "        return tools\n"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("SIE_SGLANG_")}
    environment["PYTHONPATH"] = os.pathsep.join([str(_COMPAT_DIR), str(tmp_path)])
    environment["SIE_SGLANG_CHAT_TOOLS_AS_SENT"] = "1"
    result = subprocess.run(  # noqa: S603 - executes the fixed local interpreter
        [
            sys.executable,
            "-c",
            "import sys\n"
            f"assert {_SERVING_CHAT!r} not in sys.modules\n"
            f"from {_SERVING_CHAT} import OpenAIServingChat\n"
            "assert OpenAIServingChat._sie_chat_tools_as_sent\n",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "Error in sitecustomize" not in result.stderr
