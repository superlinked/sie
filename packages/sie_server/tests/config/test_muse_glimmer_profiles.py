from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi import HTTPException
from sie_server.adapters._generation_base import resolve_reasoning_format, thinking_blocks_must_be_hidden
from sie_server.adapters.sglang.cuda13 import SGLangMuseAdapter
from sie_server.adapters.sglang.muse_channels import translate_muse_text
from sie_server.api.openai_local import _prepare_sglang_body
from sie_server.config.model import ModelConfig
from sie_server.core.loader import _build_adapter_kwargs
from sie_server.processors.remote_chat_prompt import _tool_format
from sie_server.processors.streaming import StreamingProcessor
from sie_server.processors.tool_call_grammar import build_tool_choice_grammar
from sie_server.processors.tool_call_parser import _parse_xml_tool_call, tool_parameter_schemas

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "meta-models__Muse-Glimmer-30B.yaml"
PROMPT = "<|start|>user<|message|>Hi<|eot|><|start|>assistant"
TOOLS = (
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}},
)


class _OneEventStream:
    def __init__(self) -> None:
        self.status_code = 200

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aiter_lines(self):
        yield (
            'data: {"text": " to=self<|message|>x<|eom|><|start|>assistant to=user<|message|>Hi", '
            '"meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}'
        )

    def raise_for_status(self) -> None:
        return None


def _channel(name: str, *others: str) -> str:
    invokes = "".join(f'<atem:invoke name="{called}"></atem:invoke>' for called in (name, *others))
    return f" to={name}<|message|><atem:function_calls>{invokes}</atem:function_calls>"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(MODEL_FILE.read_text()))


def _adapter(config: ModelConfig) -> SGLangMuseAdapter:
    assert config.resolve_profile("default").adapter_path == "sie_server.adapters.sglang.cuda13:SGLangMuseAdapter"
    return SGLangMuseAdapter(**_build_adapter_kwargs(config, "bfloat16"))


def test_muse_glimmer_claims_tools_grammar_and_private_reasoning() -> None:
    config = _config()
    assert config.tasks.generate is not None
    capabilities = config.tasks.generate.capabilities

    assert capabilities.tools is True
    assert capabilities.grammar == ["json_schema", "regex"]
    assert thinking_blocks_must_be_hidden(config) is True


def test_muse_glimmer_resolves_to_its_formats_on_the_worker_routes() -> None:
    config = _config()
    registry = MagicMock()
    registry.get_config.return_value = config
    processor = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")

    assert processor._resolve_tool_call_format(config.sie_id) == "muse_atem"
    assert _tool_format(config) == "muse_atem"
    # The adapter rewrites the reasoning channel into <think> blocks.
    assert resolve_reasoning_format(config) == "qwen3"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_muse_glimmer_launches_the_engine_with_its_parsers_and_no_remote_code(
    mock_find_port: MagicMock, mock_requests_get: MagicMock, mock_popen: MagicMock
) -> None:
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    _adapter(_config()).load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert cmd[cmd.index("--tool-call-parser") + 1] == "muse"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "muse"
    assert cmd[cmd.index("--grammar-backend") + 1] == "xgrammar"
    assert "--trust-remote-code" not in cmd
    assert cmd[cmd.index("--context-length") + 1] == "8192"
    assert cmd[cmd.index("--speculative-algorithm") + 1] == "DFLASH"
    assert cmd[cmd.index("--speculative-draft-model-path") + 1] == "meta-models/Muse-Glimmer-30B-assistant"
    assert len(cmd[cmd.index("--speculative-draft-model-revision") + 1]) == 40
    assert mock_popen.call_args.kwargs["env"]["SGLANG_ENABLE_SPEC_V2"] == "1"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_muse_glimmer_keeps_channel_markers_at_the_engine_and_hides_them_from_callers(
    mock_async_client: MagicMock,
) -> None:
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.stream = MagicMock(return_value=_OneEventStream())
    client.is_closed = False
    mock_async_client.return_value = client
    adapter = _adapter(_config())
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> str:
        return "".join([chunk.text_delta async for chunk in adapter.generate(prompt=PROMPT, max_new_tokens=8)])

    text = asyncio.run(_drain())

    body = client.stream.call_args.kwargs["json"]
    assert body["sampling_params"]["skip_special_tokens"] is False
    assert "require_reasoning" not in body
    assert text == "<think>x</think>Hi"


def test_forced_tool_choice_regex_accepts_only_declared_tools() -> None:
    spec = build_tool_choice_grammar(TOOLS, "required", "muse_atem")
    assert spec is not None
    assert spec.kind == "regex"
    pattern = re.compile(spec.value)
    weather = (
        ' to=get_weather<|message|><atem:function_calls>\n<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">a < b</atem:parameter>\n</atem:invoke>\n</atem:function_calls>'
    )

    assert pattern.fullmatch(weather)
    assert pattern.fullmatch(_channel("get_time"))
    assert pattern.fullmatch(_channel("get_time", "get_weather"))
    assert not pattern.fullmatch(weather.replace("get_weather", "delete_everything"))
    assert not pattern.fullmatch("I cannot help with that.")
    assert not pattern.fullmatch(' to=get_time<|message|><atem:function_calls><atem:invoke name="get_time">')


def test_forced_tool_choice_regex_has_no_optional_whitespace_around_the_channel() -> None:
    # Greedy decoding filled optional leading whitespace up to the token limit.
    spec = build_tool_choice_grammar(TOOLS, "required", "muse_atem")
    assert spec is not None
    pattern = re.compile(spec.value)

    assert not pattern.fullmatch(" " + _channel("get_time"))
    assert not pattern.fullmatch(_channel("get_time") + " ")


def test_named_tool_choice_regex_pins_the_function() -> None:
    spec = build_tool_choice_grammar(TOOLS, {"type": "function", "function": {"name": "get_time"}}, "muse_atem")
    assert spec is not None
    pattern = re.compile(spec.value)

    assert pattern.fullmatch(_channel("get_time"))
    assert not pattern.fullmatch(_channel("get_weather"))


def test_channels_the_regex_accepts_parse_after_translation() -> None:
    spec = build_tool_choice_grammar(TOOLS, "required", "muse_atem")
    assert spec is not None
    pattern = re.compile(spec.value)
    schemas = tool_parameter_schemas(TOOLS)
    with_argument = (
        " to=get_weather<|message|><atem:function_calls>\n"
        '<atem:invoke name="get_weather">\n'
        '<atem:parameter name="city">Paris</atem:parameter>\n'
        '<atem:parameter name="first name">Ada</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls>"
    )
    expected_calls = {
        with_argument: [("get_weather", {"city": "Paris", "first name": "Ada"})],
        _channel("get_time"): [("get_time", {})],
        _channel("get_time", "get_weather"): [("get_time", {}), ("get_weather", {})],
    }

    for channel, calls in expected_calls.items():
        assert pattern.fullmatch(channel)
        bodies = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", translate_muse_text(channel), re.DOTALL)
        assert [_parse_xml_tool_call(body, schemas) for body in bodies] == calls


def test_direct_route_refuses_a_tool_choice_the_engine_cannot_force() -> None:
    config = _config()
    adapter = _adapter(config)
    body = {"messages": [{"role": "user", "content": "Hi"}], "tools": list(TOOLS)}

    for choice in ("required", {"type": "function", "function": {"name": "get_time"}}):
        with pytest.raises(HTTPException) as refused:
            _prepare_sglang_body(
                {**body, "tool_choice": choice},
                config=config,
                adapter=adapter,
                max_completion_tokens=None,
                max_tokens=8,
                seed=None,
            )
        assert refused.value.status_code == 400
        assert refused.value.detail["param"] == "tool_choice"

    for choice in (None, "auto", "none"):
        prepared = _prepare_sglang_body(
            {**body, "tool_choice": choice},
            config=config,
            adapter=adapter,
            max_completion_tokens=None,
            max_tokens=8,
            seed=None,
        )
        assert prepared["model"] == config.sie_id


def test_native_window_profile_keeps_the_default_launch_shape() -> None:
    config = _config()
    default = config.resolve_profile("default")
    native = config.resolve_profile("h100-128k")

    assert native.adapter_path == default.adapter_path
    assert native.kv_budget_tokens == 131072
    assert native.loadtime["max_seq_length"] == 131072
    # At 0.85 the draft leaves room for only 92,938 tokens of KV cache.
    assert native.loadtime["mem_fraction_static"] == 0.9
    assert native.loadtime["speculative"] == default.loadtime["speculative"]
    assert native.loadtime["tool_call_parser"] == "muse"
    assert native.loadtime["reasoning_parser"] == "muse"
    assert native.runtime["stop_tokens"] == ["<|eot|>"]
    assert native.runtime["default_sampling"] == default.runtime["default_sampling"]
    assert default.kv_budget_tokens == 8192
