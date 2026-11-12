"""Translate a chat request's OpenAI ``reasoning_effort`` into what the serving profile declares.

The caller's string never reaches a chat template or an upstream unread: each
profile declares the values it accepts and what each does, and every ingress
resolves the request against that declaration with these functions.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from sie_server.adapters._generation_base import GenerationInvalidRequestError, GenerationUnsupportedFieldError
from sie_server.config.model import REASONING_EFFORT_VALUES, ModelConfig, ReasoningEffortValue

PARAM = "reasoning_effort"
MALFORMED_MESSAGE = f"'reasoning_effort' must be one of: {', '.join(REASONING_EFFORT_VALUES)}"


@dataclass(frozen=True)
class AppliedReasoningEffort:
    """What an accepted value does: template kwargs SIE renders, or the value an upstream receives."""

    chat_template_kwargs: Mapping[str, str] = field(default_factory=dict)
    upstream: str | None = None


def parse_reasoning_effort(raw: Any) -> str | None:
    """Return the request's value, ``None`` when it is absent or null.

    Raises:
        GenerationInvalidRequestError: When the value is not one of OpenAI's values.
    """
    if raw is None:
        return None
    if not isinstance(raw, str) or raw not in REASONING_EFFORT_VALUES:
        raise GenerationInvalidRequestError(PARAM, MALFORMED_MESSAGE)
    return raw


def unsupported_message(model: str, accepted: tuple[str, ...], value: str) -> str:
    """The refusal text for ``value`` on a model that accepts ``accepted``, shared by every ingress."""
    if not accepted:
        return f"Model '{model}' does not support reasoning_effort"
    return f"Model '{model}' does not support reasoning_effort '{value}'. Supported values: {', '.join(accepted)}"


def apply_reasoning_effort(
    config: ModelConfig | None, value: str | None, *, model: str
) -> AppliedReasoningEffort | None:
    """Resolve ``value`` against the declaration of ``config``'s default profile.

    ``None`` means nothing changes: the request named no value, or named
    ``none`` for a model that does not reason at all.

    Raises:
        GenerationUnsupportedFieldError: When the profile does not accept ``value``.
    """
    if value is None:
        return None
    declaration = config.reasoning_effort_for("default") if config is not None else None
    if declaration is None:
        if value == "none" and config is not None and config.declares_no_reasoning:
            return None
        raise GenerationUnsupportedFieldError(PARAM, unsupported_message(model, (), value))
    level = declaration.values.get(cast("ReasoningEffortValue", value))
    if level is None:
        raise GenerationUnsupportedFieldError(PARAM, unsupported_message(model, declaration.accepted, value))
    return AppliedReasoningEffort(
        chat_template_kwargs=dict(level.chat_template_kwargs or {}),
        upstream=level.upstream,
    )
