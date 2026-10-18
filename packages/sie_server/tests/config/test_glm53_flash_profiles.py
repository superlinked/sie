from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from sie_server.adapters.sglang.cuda13 import SGLangCuda13Adapter
from sie_server.config.model import ModelConfig
from sie_server.core.loader import _build_adapter_kwargs
from sie_server.processors.remote_chat_prompt import _tool_format
from sie_server.processors.streaming import StreamingProcessor
from sie_server.processors.tool_call_grammar import build_tool_choice_grammar

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "zai-org__GLM-5.3-Flash.yaml"
THINKING_PROMPT = "[gMASK]<sop><|user|>Hi<|assistant|><think>"
TOOLS = ({"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},)


class _OneEventStream:
    def __init__(self) -> None:
        self.status_code = 200

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aiter_lines(self):
        yield 'data: {"text": "x", "meta_info": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"}}}'

    def raise_for_status(self) -> None:
        return None


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(MODEL_FILE.read_text()))


def _adapter(config: ModelConfig) -> SGLangCuda13Adapter:
    assert config.resolve_profile("default").adapter_path == "sie_server.adapters.sglang.cuda13:SGLangCuda13Adapter"
    return SGLangCuda13Adapter(**_build_adapter_kwargs(config, "bfloat16"))


def test_glm53_flash_claims_tool_calling_without_grammar() -> None:
    config = _config()
    assert config.tasks.generate is not None
    capabilities = config.tasks.generate.capabilities

    assert capabilities.tools is True
    assert capabilities.grammar == []


def test_glm53_flash_tool_calls_resolve_to_the_glm_format_on_the_worker_routes() -> None:
    config = _config()
    registry = MagicMock()
    registry.get_config.return_value = config
    processor = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")

    assert processor._resolve_tool_call_format(config.sie_id) == "glm_xml"
    assert _tool_format(config) == "glm_xml"


@patch("sie_server.adapters.sglang._server.subprocess.Popen")
@patch("sie_server.adapters.sglang._server.requests.get")
@patch("sie_server.adapters.sglang._server.find_free_port")
def test_glm53_flash_launches_the_engine_with_its_glm_parsers(
    mock_find_port: MagicMock, mock_requests_get: MagicMock, mock_popen: MagicMock
) -> None:
    mock_find_port.return_value = 30005
    mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
    mock_requests_get.return_value = MagicMock(status_code=200)

    _adapter(_config()).load("cuda:0")

    cmd = mock_popen.call_args[0][0]
    assert cmd[cmd.index("--tool-call-parser") + 1] == "glm47"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "glm45"
    assert cmd[cmd.index("--grammar-backend") + 1] == "xgrammar"


@patch("sie_server.adapters.sglang.generation.httpx.AsyncClient")
def test_glm53_flash_sends_a_forced_tool_choice_grammar_with_require_reasoning(mock_async_client: MagicMock) -> None:
    config = _config()
    grammar = build_tool_choice_grammar(TOOLS, "required", _tool_format(config))
    assert grammar is not None
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.stream = MagicMock(return_value=_OneEventStream())
    client.is_closed = False
    mock_async_client.return_value = client
    adapter = _adapter(config)
    adapter._server_url = "http://localhost:30005"

    async def _drain() -> None:
        async for _ in adapter.generate(prompt=THINKING_PROMPT, max_new_tokens=8, grammar=grammar):
            pass

    asyncio.run(_drain())

    body = client.stream.call_args.kwargs["json"]
    assert body["require_reasoning"] is True
    assert body["sampling_params"]["regex"] == grammar.value


def test_glm53_flash_local_profile_keeps_its_measured_window() -> None:
    config = _config()
    assert config.tasks.generate is not None
    default = config.resolve_profile("default")

    assert config.tasks.generate.context_length == 16384
    assert config.tasks.generate.max_output_tokens == 4096
    assert config.max_sequence_length == 16384
    assert default.loadtime["max_seq_length"] == 16384
