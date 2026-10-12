"""Show a JSON Schema response format to the model, as OpenAI-compatible servers do.

A JSON Schema grammar only constrains decoding. It removes the tokens that would
break the schema, but the model never reads the schema: field descriptions and
the options of an ``enum`` stay invisible, and a model that has not been told
the options tends to pick the first one the grammar allows. OpenAI and
Anthropic put the schema in the model's context, so OpenAI-compatible callers
expect the model to see it.

When SIE renders the chat template itself, a ``json_schema`` response format is
therefore appended to the system message as the ``# Response Formats`` section
of OpenAI's harmony prompt format (the one gpt-oss models are trained on), with
the schema as compact JSON::

    <caller's system message>

    # Response Formats

    ## <response_format.json_schema.name>

    {"type":"object","properties":{...}}

The grammar still constrains decoding. Without a system message the section
becomes one. A chat template that cannot render a system message (it raises,
or drops the text) gets the section at the start of the first user message
instead, the way such templates fold a system prompt themselves.

The section is left out when:

* the request opts out (``schema_in_prompt: false``) or the serving profile
  does (``adapter_options.runtime.schema_in_prompt: false``);
* a message already contains the schema as JSON. Whitespace is ignored and
  keys may be in the schema's own or sorted order, so ``json.dumps`` and
  ``JSON.stringify`` renderings at any indent are found. A schema written any
  other way (YAML, prose, other key orders, a ``$ref`` the gateway inlined) is
  not detected; such callers opt out per request;
* the grammar says nothing beyond "a JSON object" (``json_object`` mode).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sie_server.types.grammar import GrammarSpec

logger = logging.getLogger(__name__)

RESPONSE_FORMATS_HEADING = "# Response Formats"
SCHEMA_IN_PROMPT_OPTION = "schema_in_prompt"

_DEFAULT_FORMAT_NAME = "response"
_SYSTEM_ROLES = frozenset({"system", "developer"})
# The schemas ``response_format: {"type": "json_object"}`` stands for. They
# carry nothing a model could use beyond "answer with a JSON object".
_GENERIC_OBJECT_SCHEMAS: tuple[dict[str, Any], ...] = (
    {"type": "object"},
    {"type": "object", "additionalProperties": True},
)
_SYSTEM_ROLE_PROBE = "sie-system-role-probe-6f1d2c"


def response_format_section(grammar: GrammarSpec | None) -> str | None:
    """Render ``grammar`` as a ``# Response Formats`` prompt section.

    Returns ``None`` for anything other than an informative JSON Schema.
    """
    if grammar is None or grammar.kind != "json_schema" or not isinstance(grammar.value, dict):
        return None
    if grammar.value in _GENERIC_OBJECT_SCHEMAS:
        return None
    name = " ".join((grammar.label or "").split()) or _DEFAULT_FORMAT_NAME
    schema = json.dumps(grammar.value, ensure_ascii=False, separators=(",", ":"))
    return f"{RESPONSE_FORMATS_HEADING}\n\n## {name}\n\n{schema}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def _squash(text: str) -> str:
    return "".join(text.split())


def schema_in_messages(messages: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> bool:
    """Whether a message's text already contains ``schema`` as JSON.

    Whitespace is ignored on both sides, and the schema is looked for in its
    own key order and in sorted order, with and without ASCII escaping.
    """
    needles = {
        _squash(json.dumps(schema, ensure_ascii=ascii_only, sort_keys=sort_keys))
        for ascii_only in (False, True)
        for sort_keys in (False, True)
    }
    for message in messages:
        text = _squash(_content_text(message.get("content")))
        if any(needle in text for needle in needles):
            return True
    return False


def _append_text(content: Any, section: str) -> Any:
    if isinstance(content, list):
        separator = "\n\n" if _content_text(content).strip() else ""
        return [*content, {"type": "text", "text": separator + section}]
    text = content if isinstance(content, str) else ""
    return f"{text.rstrip()}\n\n{section}" if text.strip() else section


def _prepend_text(content: Any, section: str) -> Any:
    if isinstance(content, list):
        return [{"type": "text", "text": f"{section}\n\n"}, *content]
    text = content if isinstance(content, str) else ""
    return f"{section}\n\n{text}" if text else section


def with_response_format(
    messages: Sequence[Mapping[str, Any]],
    section: str,
    *,
    system_role: bool,
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` that carries ``section``.

    The section is appended to a leading system (or developer) message, else
    it becomes a new leading system message. When the template cannot render
    a system message (``system_role`` false) it opens the first user message
    instead. The input messages are not modified.
    """
    out = [dict(message) for message in messages]
    if out and out[0].get("role") in _SYSTEM_ROLES:
        out[0]["content"] = _append_text(out[0].get("content"), section)
        return out
    if system_role:
        return [{"role": "system", "content": section}, *out]
    for message in out:
        if message.get("role") == "user":
            message["content"] = _prepend_text(message.get("content"), section)
            return out
    return out


def starts_with_system_message(messages: Sequence[Mapping[str, Any]]) -> bool:
    return bool(messages) and messages[0].get("role") in _SYSTEM_ROLES


def renders_system_role(apply_chat_template: Callable[..., Any]) -> bool:
    """Whether a chat template keeps the text of a leading system message.

    Some templates raise on a system role, others drop it silently; both mean
    the section has to travel in a user message.
    """
    probe = [{"role": "system", "content": _SYSTEM_ROLE_PROBE}, {"role": "user", "content": "Hi"}]
    try:
        rendered = apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 - a template that cannot render a system message
        logger.debug("chat template does not render a system role", exc_info=True)
        return False
    return isinstance(rendered, str) and _SYSTEM_ROLE_PROBE in rendered


def profile_schema_in_prompt(config: Any) -> bool:
    """The serving profile's default; ``adapter_options.runtime.schema_in_prompt: false`` turns it off."""
    try:
        runtime = config.resolve_profile("default").runtime
    except (AttributeError, KeyError, ValueError):
        return True
    if not isinstance(runtime, Mapping):
        return True
    return runtime.get(SCHEMA_IN_PROMPT_OPTION) is not False


def json_schema_response_format(response_format: Any) -> GrammarSpec | None:
    """The JSON Schema of an OpenAI ``response_format``, or ``None``."""
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return None
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict) or not isinstance(json_schema.get("schema"), dict):
        return None
    name = json_schema.get("name")
    return GrammarSpec(kind="json_schema", value=json_schema["schema"], label=name if isinstance(name, str) else None)
