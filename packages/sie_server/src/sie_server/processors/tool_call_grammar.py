"""Build constrained-decoding grammars that enforce OpenAI ``tool_choice``.

The ``tool_choice: "auto"`` path lets the model decide whether to emit
``<tool_call>`` blocks. ``"required"`` (the model MUST call some tool)
and the named form (``{"type":"function","function":{"name": X}}`` — the
model MUST call tool ``X``) need *enforcement*. The SIE worker renders
the chat template itself and calls SGLang's raw ``/generate`` endpoint,
which has no OpenAI ``tool_choice`` parameter — so the only lever is
constrained decoding via a grammar on ``sampling_params`` (``regex`` /
``ebnf`` / ``json_schema``, see the SGLang generation adapter).

This module builds a ``regex`` :class:`GrammarSpec` that constrains the
output to one or more well-formed tool-call blocks in the model's
configured on-wire format (:data:`ToolCallFormat`), optionally pinned to
a single function name. Regex (rather than EBNF) keeps the constraint
small and predictable; both are supported by the xgrammar backend.

GPU-VERIFY: the patterns here are correct by construction and unit-tested
for shape, but have NOT been run end-to-end against xgrammar + Qwen3.5 on
GPU. One verification pass should confirm the backend accepts the pattern and the model
terminates cleanly before relying on forced ``tool_choice`` in
production. The failure mode if a pattern is rejected is a clean
``grammar_compile_failed`` terminal chunk, not a crash.
"""

from __future__ import annotations

import re
from typing import Any

from sie_server.processors.tool_call_parser import ToolCallFormat
from sie_server.types.grammar import GrammarSpec


class ToolChoiceError(ValueError):
    """Raised for an unsatisfiable / unsupported ``tool_choice`` request.

    The worker surfaces this as a terminal ``invalid_request`` chunk —
    e.g. a named choice that does not appear in ``tools``, or a format
    for which forcing is not implemented.
    """


def normalize_tool_choice(
    tool_choice: dict[str, Any] | str | None,
) -> tuple[str, str | None]:
    """Reduce an OpenAI ``tool_choice`` to ``(mode, function_name)``.

    ``mode`` is one of ``"auto"`` / ``"none"`` / ``"required"`` /
    ``"named"``; ``function_name`` is set only for ``"named"``. ``None``
    and unrecognised shapes default to ``"auto"`` (model decides) — the
    gateway is the authority for shape validation, so the worker treats
    anything it doesn't understand as the permissive default rather than
    failing the request.
    """
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice, None
        return "auto", None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "auto", None


def extract_tool_names(tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None) -> list[str]:
    """Pull the ``function.name`` from each tool definition, in order."""
    names: list[str] = []
    for tool in tools or ():
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def build_tool_choice_grammar(
    tools: tuple[dict[str, Any], ...] | None,
    tool_choice: dict[str, Any] | str | None,
    tool_call_format: ToolCallFormat,
) -> GrammarSpec | None:
    """Build a forcing grammar for ``tool_choice``, or ``None`` if no
    forcing is needed.

    Returns ``None`` for ``auto`` / ``none`` / absent (the model decides,
    or tools are off). For ``required`` the grammar admits a call to any
    tool in ``tools``; for the named form it pins the single function.

    Raises :class:`ToolChoiceError` when the request cannot be satisfied
    (named function absent from ``tools``, empty tool set, or a format
    for which forcing is unimplemented).
    """
    mode, name = normalize_tool_choice(tool_choice)
    if mode in ("auto", "none"):
        return None

    names = extract_tool_names(tools)
    if not names:
        raise ToolChoiceError(f"tool_choice={mode!r} requires a non-empty 'tools' array with named functions")

    if mode == "named":
        if name not in names:
            raise ToolChoiceError(f"tool_choice names function {name!r}, which is not in tools {names!r}")
        allowed = [name]
    else:  # required
        allowed = names

    if tool_call_format == "qwen_xml":
        pattern = _qwen_xml_force_regex(allowed)
    elif tool_call_format == "hermes_json":
        pattern = _hermes_json_force_regex(allowed)
    elif tool_call_format == "glm_xml":
        pattern = _glm_xml_force_regex(allowed)
    elif tool_call_format == "auto":
        # ``auto`` means the model's on-wire tool-call format could not be
        # confidently resolved from config. Guessing Qwen XML here would
        # force a non-Qwen model (e.g. a Hermes-JSON model) to emit XML it
        # never learned — producing garbage or a hang. Refuse to force
        # instead of guessing; the caller surfaces this as
        # ``invalid_request``.
        raise ToolChoiceError(
            f"tool_choice={mode!r} requires forcing, but the model's tool-call format "
            "could not be resolved (tool_call_format='auto'); configure the model's "
            "tool_call_parser to enable forced tool_choice"
        )
    else:  # pragma: no cover - exhaustive over ToolCallFormat
        raise ToolChoiceError(f"tool_choice forcing is not implemented for format {tool_call_format!r}")

    return GrammarSpec(kind="regex", value=pattern, label=f"tool_choice:{mode}")


# A tool-call block's inner body — anything up to the closing tag. The
# constrained-decoding backend compiles the regex to an FSM, so the lazy
# quantifier only affects matching, not the accepted language.
_INNER = r"[\s\S]*?"


def _name_alternation(names: list[str]) -> str:
    """Regex-escaped ``(a|b|c)`` over the allowed function names."""
    return "(?:" + "|".join(re.escape(n) for n in names) + ")"


def _qwen_xml_force_regex(names: list[str]) -> str:
    """Force one-or-more Qwen3(-Coder) XML tool-call blocks.

    A single block looks like::

        <tool_call>
        <function=NAME>
        <parameter=KEY>
        VALUE
        </parameter>
        </function>
        </tool_call>

    The pattern pins ``<function=NAME>`` to the allowed set and leaves
    the parameter body free (the schema is already rendered into the
    prompt; the goal of forcing is to guarantee a call happens, not to
    re-enforce argument types).
    """
    block = r"<tool_call>\s*<function=" + _name_alternation(names) + r">" + _INNER + r"</function>\s*</tool_call>"
    # Whole-output match: optional surrounding whitespace, one or more blocks.
    return r"\s*(?:" + block + r"\s*)+"


def _hermes_json_force_regex(names: list[str]) -> str:
    """Force one-or-more Hermes JSON tool-call blocks.

    A single block looks like::

        <tool_call>{"name": "NAME", "arguments": {...}}</tool_call>

    The Hermes/Qwen templates emit ``name`` before ``arguments``; the
    pattern pins the name to the allowed set and leaves the arguments
    object free.
    """
    block = (
        r'<tool_call>\s*\{\s*"name"\s*:\s*"'
        + _name_alternation(names)
        + r'"\s*,\s*"arguments"\s*:\s*'
        + _INNER
        + r"\}\s*</tool_call>"
    )
    return r"\s*(?:" + block + r"\s*)+"


def _glm_xml_force_regex(names: list[str]) -> str:
    """Force one-or-more GLM tool-call blocks.

    A single block looks like::

        <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>

    The name ends at the first ``<arg_key>`` or at ``</tool_call>`` for a call
    without arguments, which pins it to the allowed set. Arguments are complete
    key/value pairs so every accepted block parses; values stay free, as in the
    other formats, and keys exclude ``<``/``>`` as the parser requires.
    """
    pair = r"<arg_key>[^<>]*</arg_key>\s*<arg_value>" + _INNER + r"</arg_value>\s*"
    block = r"<tool_call>\s*" + _name_alternation(names) + r"\s*(?:" + pair + r")*</tool_call>"
    return r"\s*(?:" + block + r"\s*)+"
