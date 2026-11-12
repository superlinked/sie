"""Gemma 4 models resolve to the native ``gemma4`` tool-call format."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from sie_server.config.model import ModelConfig
from sie_server.processors.remote_chat_prompt import _tool_format
from sie_server.processors.streaming import StreamingProcessor
from sie_server.processors.tool_call_grammar import build_tool_choice_grammar

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
GEMMA4_MODELS = ("E2B-it", "E4B-it", "26B-A4B-it", "31B-it")
TOOLS = ({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},)


def _config(variant: str) -> ModelConfig:
    path = MODELS_DIR / f"google__gemma-4-{variant}.yaml"
    return ModelConfig.model_validate(yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("variant", GEMMA4_MODELS)
def test_gemma4_declares_tools_and_resolves_to_its_format(variant: str) -> None:
    config = _config(variant)
    assert config.tasks.generate is not None
    assert config.tasks.generate.capabilities.tools is True

    registry = MagicMock()
    registry.get_config.return_value = config
    processor = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")
    resolved = processor._resolve_tool_call_format(config.sie_id)

    assert resolved == "gemma4"
    assert _tool_format(config) == "gemma4"
    spec = build_tool_choice_grammar(TOOLS, "required", resolved)
    assert spec is not None
    assert spec.kind == "regex"
