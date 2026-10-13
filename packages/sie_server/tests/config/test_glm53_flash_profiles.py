from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml
from sie_server.config.model import ModelConfig
from sie_server.processors.remote_chat_prompt import _tool_format
from sie_server.processors.streaming import StreamingProcessor

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "zai-org__GLM-5.3-Flash.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(MODEL_FILE.read_text()))


def test_glm53_flash_claims_tool_calling_without_grammar() -> None:
    config = _config()
    assert config.tasks.generate is not None
    capabilities = config.tasks.generate.capabilities

    assert capabilities.tools is True
    assert capabilities.grammar == []


def test_glm53_flash_tool_calls_resolve_to_the_glm_format_on_every_route() -> None:
    config = _config()
    registry = MagicMock()
    registry.get_config.return_value = config
    processor = StreamingProcessor(nc=AsyncMock(), registry=registry, worker_id="w1")

    assert config.resolve_profile("default").loadtime["tool_call_parser"] == "glm47"
    assert processor._resolve_tool_call_format(config.sie_id) == "glm_xml"
    assert _tool_format(config) == "glm_xml"


def test_glm53_flash_local_profile_keeps_its_measured_window() -> None:
    config = _config()
    assert config.tasks.generate is not None
    default = config.resolve_profile("default")

    assert config.tasks.generate.context_length == 16384
    assert config.tasks.generate.max_output_tokens == 4096
    assert config.max_sequence_length == 16384
    assert default.loadtime["max_seq_length"] == 16384
