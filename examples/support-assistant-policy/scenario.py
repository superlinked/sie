"""The support-rules scenario: the store's system prompt, the 55 scripted conversations, and the six arms.

Standard library only, so score.py and `run.py --show` work on a bare python3.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = HERE / "scenario" / "system_prompt.md"
CONVERSATIONS_PATH = HERE / "scenario" / "conversations.json"

RULES = ("commitments", "knowledge", "confidentiality", "escalation", "scope")
RULE_NAMES = {
    "commitments": "Commitments",
    "knowledge": "Knowledge",
    "confidentiality": "Confidentiality",
    "escalation": "Escalation",
    "scope": "Scope and tone",
}

# Every arm had the same output cap. It is far above a normal support reply, so no
# reply is cut short into something that looks compliant.
MAX_OUTPUT_TOKENS = 400

SIE_MODEL = "Qwen/Qwen3.8-27B-FP8"
SIE_BASE_URL = "https://api.superlinked.com"

# The six arms, each at the lowest reasoning setting its API accepts, and temperature 0
# where the API accepts one. `provider` picks the client run.py uses.
ARMS: dict[str, dict[str, Any]] = {
    SIE_MODEL: {"name": "SIE Qwen3.8 27B", "provider": "sie", "reasoning": "none", "temperature": 0},
    "claude-sonnet-5": {
        "name": "Claude Sonnet 5",
        "provider": "anthropic",
        "reasoning": "none",
        # The API rejects `temperature` for this model, so it runs at the API default.
        "temperature": None,
    },
    "claude-haiku-4-5-20251001": {
        "name": "Claude Haiku 4.5",
        "provider": "anthropic",
        "reasoning": "none",
        "temperature": 0,
    },
    "gpt-6-luna": {"name": "GPT-6 Luna", "provider": "openai", "reasoning": "none", "temperature": 0},
    "gpt-6-sol": {"name": "GPT-6 Sol", "provider": "openai", "reasoning": "none", "temperature": 0},
    # Astra rejects reasoning effort "none"; "low" is its lowest accepted value, and
    # the API takes no temperature alongside it.
    "gpt-6-astra": {"name": "GPT-6 Astra", "provider": "openai", "reasoning": "low", "temperature": None},
}

# Two conversations for a cheap first run: twelve calls, one on a commitment the
# assistant must not make and one on internal notes it must not reveal.
SMOKE = ("commitments__late-laptop-credit", "confidentiality__fraud-hold")


def file_stem(model: str) -> str:
    """The file name a model's transcript is written under: `Qwen/Qwen3.8-27B-FP8` -> `Qwen_Qwen3.8-27B-FP8`."""
    return model.replace("/", "_")


def system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def conversations() -> list[dict[str, Any]]:
    corpus = json.loads(CONVERSATIONS_PATH.read_text(encoding="utf-8"))
    rows = corpus["conversations"]
    if len(rows) != 55 or {row["rule"] for row in rows} != set(RULES):
        raise SystemExit(f"{CONVERSATIONS_PATH} is not the 55-conversation, five-rule scenario")
    return rows
