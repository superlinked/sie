"""Every catalog model that declares tools has a tool-call format the worker understands.

A ``tool_call_parser`` that maps to ``auto`` leaves the worker guessing the
on-wire format per block and refuses forced ``tool_choice``, so a model that
advertises tools must name a parser with an explicit format in every profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import ModelConfig
from sie_server.processors.tool_call_parser import tool_call_format_for_parser

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Models that declare tools but whose profiles name no ``tool_call_parser``.
# These predate this guard and are listed so it can land without changing them;
# remove an entry once its model names a parser (the test fails if one is stale).
_KNOWN_UNMAPPED = frozenset(
    {
        # The fake test model: a stub adapter with no on-wire tool-call format.
        "sie-fake",
    }
)


def _tool_models() -> list[ModelConfig]:
    configs = [
        ModelConfig.model_validate(yaml.safe_load(path.read_text())) for path in sorted(MODELS_DIR.glob("*.yaml"))
    ]
    return [
        config for config in configs if config.tasks.generate is not None and config.tasks.generate.capabilities.tools
    ]


def _unmapped_profiles(config: ModelConfig) -> list[str]:
    return [
        f"{name} (tool_call_parser={config.resolve_profile(name).loadtime.get('tool_call_parser')!r})"
        for name in config.profiles
        if tool_call_format_for_parser(config.resolve_profile(name).loadtime.get("tool_call_parser")) == "auto"
    ]


def test_tool_models_name_a_known_tool_call_parser_in_every_profile() -> None:
    models = _tool_models()
    assert models, f"no tool-capable model found in {MODELS_DIR}"
    offenders = {
        config.sie_id: unmapped
        for config in models
        if config.sie_id not in _KNOWN_UNMAPPED and (unmapped := _unmapped_profiles(config))
    }
    assert offenders == {}, "tool-capable models whose tool_call_parser has no explicit format"


def test_known_unmapped_entries_are_still_needed() -> None:
    by_id = {config.sie_id: config for config in _tool_models()}
    stale = sorted(
        model_id for model_id in _KNOWN_UNMAPPED if model_id not in by_id or not _unmapped_profiles(by_id[model_id])
    )
    assert stale == [], "remove these from _KNOWN_UNMAPPED: they no longer need an exception"


@pytest.mark.parametrize(
    ("parser", "expected"),
    [
        ("qwen25", "hermes_json"),
        ("hermes", "hermes_json"),
        ("Hermes", "hermes_json"),
        ("qwen3_coder", "qwen_xml"),
        ("qwen3", "qwen_xml"),
        ("glm47", "glm_xml"),
        ("glm45", "glm_xml"),
        ("muse", "muse_atem"),
        ("gemma4", "gemma4"),
        ("GEMMA4", "gemma4"),
        ("gemma3", "auto"),
        ("llama3", "auto"),
        ("", "auto"),
        (None, "auto"),
        (4, "auto"),
    ],
)
def test_tool_call_format_for_parser(parser: object, expected: str) -> None:
    assert tool_call_format_for_parser(parser) == expected
