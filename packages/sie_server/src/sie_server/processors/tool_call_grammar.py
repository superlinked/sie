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
``grammar_compile_failed`` terminal chunk, not a crash. The ``gemma4``
pattern has had that pass with xgrammar and Gemma 4 E2B: forced and named
calls parse and stop cleanly, including when the model does not want the call
(see :func:`_gemma4_force_regex`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterator
from typing import Any

import orjson
from jsonschema import Draft202012Validator
from jsonschema.exceptions import UndefinedTypeCheck

from sie_server.adapters._generation_base import REASONING_CONTROL_TOKENS
from sie_server.processors.strict_grammar import (
    iter_subschemas,
    unverifiable_schema_keyword,
    without_schema_keywords,
)
from sie_server.processors.tool_call_parser import (
    MAX_XML_PARAMS,
    XML_PARAM_CLOSE,
    ToolCallFormat,
    resolve_schema_ref,
    schema_allows_null,
    schema_conversion_type,
)
from sie_server.types.grammar import GrammarSpec


class ToolChoiceError(ValueError):
    """Raised for an unsatisfiable / unsupported ``tool_choice`` request.

    The worker surfaces this as a terminal ``invalid_request`` chunk —
    e.g. a named choice that does not appear in ``tools``, or a format
    for which forcing is not implemented.
    """


class StrictToolError(ValueError):
    """Raised when a ``strict`` function tool cannot be enforced.

    OpenAI rejects a strict schema it cannot honour instead of silently
    generating unconstrained arguments, so the worker surfaces this as a
    terminal ``unsupported_field`` chunk naming ``param``.
    """

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param


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
    elif tool_call_format == "muse_atem":
        pattern = _muse_atem_force_regex(allowed)
    elif tool_call_format == "gemma4":
        pattern = _gemma4_force_regex(allowed)
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


def _muse_atem_force_regex(names: list[str]) -> str:
    """Force one Muse Glimmer tool channel holding one-or-more calls.

    A channel looks like::

         to=NAME<|message|><atem:function_calls>
        <atem:invoke name="NAME">
        <atem:parameter name="KEY">VALUE</atem:parameter>
        </atem:invoke>
        </atem:function_calls>

    The generation prompt ends at ``<|start|>assistant``, so the output starts at
    the recipient and constraining it skips the reasoning channel. Nothing optional
    surrounds the channel: greedy decoding fills optional leading whitespace up to
    the token limit. Parameters are complete so every accepted block parses;
    values stay free, as in the other formats.
    """
    name = _name_alternation(names)
    parameter = r'<atem:parameter name="[^"<>]+">' + _INNER + r"</atem:parameter>\s*"
    invoke = r'<atem:invoke name="' + name + r'">\s*(?:' + parameter + r")*</atem:invoke>"
    return r" to=" + name + r"<\|message\|><atem:function_calls>\s*(?:" + invoke + r"\s*)+</atem:function_calls>"


def _gemma4_force_regex(names: list[str]) -> str:
    """Force one-or-more consecutive Gemma 4 tool-call blocks.

    A single block looks like::

        <|tool_call>call:NAME{KEY:<|"|>VALUE<|"|>,KEY:42}<tool_call|>

    The pattern pins ``NAME`` to the allowed set in every block. Nothing
    optional precedes, separates or follows the blocks: greedy decoding fills
    optional whitespace up to the token limit.

    Unlike the Qwen and Hermes patterns, the argument body is not free text.
    With a free body, a Gemma 4 model that was forced to call a tool it did not
    want wrote ``}</tool_call>`` and then prose up to the token limit, which
    never closes the block. So outside ``<|"|>`` strings the body admits no
    whitespace, no ``"`` and no ``<``: after the arguments the only way on is
    the closing ``<tool_call|>``. Strings stay free except that they cannot
    contain ``<|`` or end with ``<``, which keeps the string delimiter
    unambiguous without lookahead. Values are not otherwise checked; the
    parser still validates the call.
    """
    string = r'<\|"\|>(?:[^<]|<+[^|<])*<\|"\|>'
    body = r'(?:[^\s<"]|' + string + r")*"
    block = r"<\|tool_call>call:" + _name_alternation(names) + r"\{" + body + r"\}<tool_call\|>"
    return r"(?:" + block + r")+"


# ---------------------------------------------------------------------------
# Strict function tools
# ---------------------------------------------------------------------------
#
# A tool with ``"strict": true`` promises that every generated call's
# arguments match its ``parameters`` schema. The regex forcing above pins only
# the function name, so strict tools compile to an XGrammar structural tag
# instead: the call layout is fixed text and each argument value is
# constrained by its own schema. The tag also covers ``tool_choice: "auto"``,
# where a ``triggered_tags`` format lets the model write free text until it
# opens a call. Reasoning stays unconstrained: SGLang applies the grammar only
# after the reasoning block when the request expects one.

_JSON_KINDS = frozenset({"integer", "number", "boolean", "object", "array", "null"})
_SCALAR_KINDS = frozenset({"integer", "number", "boolean"})
# Object-level keywords the Qwen XML layout cannot express on the parameters
# object, which it writes as one tag per declared property.
_UNSUPPORTED_PARAMETERS_KEYWORDS = (
    "anyOf",
    "allOf",
    "patternProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
    "const",
    "enum",
    # A root reference would hide the properties the layout reads.
    "$ref",
)
_DEFINITION_KEYWORDS = ("$defs", "definitions")
# Keys XGrammar 0.1.32 drops from every object when it caches a subschema,
# property names included (measured).
_XGRAMMAR_CACHE_SKIPPED_KEYS = frozenset(
    {"description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly", "$schema"}
)
_DEFINITION_REF_PREFIXES = ("#/$defs/", "#/definitions/")
# Parameter names the XML parser reads back intact (see ``_XML_PARAM_OPEN_RE``).
_XML_PARAMETER_NAME = re.compile(r"[^<>\s](?:[^<>\n]*[^<>\s])?")
# Characters XGrammar's regex dialect treats as syntax. ``re.escape`` is not
# used because it also escapes spaces and writes a backslash before a literal
# newline, escapes outside that dialect's documented set.
_REGEX_SPECIAL = frozenset("\\^$.|?*+()[]{}/-")
_REGEX_ESCAPES = {"\n": r"\n", "\t": r"\t", "\r": r"\r"}
# Bound on nested unions followed when checking a scalar parameter.
_MAX_UNION_DEPTH = 8
# Every call tag begins with the trigger, as ``triggered_tags`` requires.
_TRIGGER = "<tool_call>"
_QWEN_CALL_START = _TRIGGER + "\n<function="
_QWEN_CALL_END = "</function>\n</tool_call>"
_HERMES_CALL_START = _TRIGGER + '\n{"name": '
_HERMES_CALL_END = "}\n</tool_call>"
# Text a raw XML string value cannot contain: the closing tags the parsers end on.
_RAW_VALUE_EXCLUDES = (XML_PARAM_CLOSE, "</tool_call>")
_REASONING_DELIMITERS = frozenset(token for pair in REASONING_CONTROL_TOKENS.values() for token in pair)


def is_strict_tool(tool: Any) -> bool:
    """Return whether one OpenAI tool definition sets ``function.strict: true``."""
    function = tool.get("function") if isinstance(tool, dict) else None
    return isinstance(function, dict) and function.get("strict") is True


def has_strict_tools(tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None) -> bool:
    """Return whether any tool in ``tools`` is strict."""
    return any(is_strict_tool(tool) for tool in tools or ())


def strict_tool_schemas(tools: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Map each strict tool's name to the parameters schema its arguments must satisfy."""
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or ():
        if not is_strict_tool(tool):
            continue
        function = tool["function"]
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and name:
            schemas[name] = parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}}
    return schemas


def tools_require_grammar(tools: Any, tool_choice: Any) -> bool:
    """Return whether ``tools`` / ``tool_choice`` make the worker constrain decoding.

    A forced ``tool_choice`` (``required`` or a named function) and any
    ``strict`` tool both compile a grammar, so such a request must run on the
    model's grammar-safe profile: speculative decoding bypasses the grammar
    backend. Mirrors the gateway's routing rule for ``/v1/chat/completions``.
    """
    if not isinstance(tools, list | tuple) or not tools:
        return False
    mode, _ = normalize_tool_choice(tool_choice)
    return mode in ("required", "named") or (mode != "none" and has_strict_tools(tools))


def request_constrains_decoding(grammar: Any, tools: Any, tool_choice: Any) -> bool:
    """Return whether a generation request runs under a grammar.

    True for a ``response_format`` / native grammar and for the tool grammars
    of :func:`tools_require_grammar`. Every worker ingress uses this to pick
    the grammar-safe profile and the profile's grammar reasoning effort.
    """
    return grammar is not None or tools_require_grammar(tools, tool_choice)


def build_strict_tool_grammar(
    tools: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    tool_choice: dict[str, Any] | str | None,
    tool_call_format: ToolCallFormat,
    *,
    parallel_tool_calls: bool = True,
) -> GrammarSpec | None:
    """Build the structural-tag grammar that enforces strict tool arguments.

    Every strict tool's arguments are constrained by its schema; a non-strict
    tool in the same request keeps a free argument body. ``auto`` admits free
    text or calls, ``required`` and the named form admit only calls, and
    ``parallel_tool_calls=False`` stops after the first call. Returns ``None``
    for ``tool_choice: "none"``.

    Raises:
        StrictToolError: The format, or a strict tool's schema, cannot be
            enforced.
        ToolChoiceError: The ``tool_choice`` cannot be satisfied.
    """
    mode, name = normalize_tool_choice(tool_choice)
    if mode == "none":
        return None
    build_tag = _STRICT_TAG_BUILDERS.get(tool_call_format)
    if build_tag is None:
        raise StrictToolError(
            "tools",
            f"strict function tools are not supported for this model's tool-call format ({tool_call_format!r})",
        )
    names = extract_tool_names(tools)
    if not names:
        raise ToolChoiceError("strict function tools require a non-empty 'tools' array with named functions")
    if mode == "named" and name not in names:
        raise ToolChoiceError(f"tool_choice names function {name!r}, which is not in tools {names!r}")

    duplicates = sorted({tool_name for tool_name in names if names.count(tool_name) > 1})
    if duplicates:
        # The parser converts each call by one schema per name.
        raise StrictToolError("tools", f"function names must be unique for strict tools; repeated: {duplicates!r}")

    tags: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            continue
        # Every strict schema is checked, even one the named choice excludes:
        # an unenforceable strict tool is rejected whichever call is chosen.
        tag = build_tag(function, strict=is_strict_tool(tool), path=f"tools[{index}].function.parameters")
        if mode != "named" or function["name"] == name:
            tags.append(tag)
    _check_schema_cache_collisions(tags)

    stop_after_first = not parallel_tool_calls
    if mode == "auto":
        structure: dict[str, Any] = {
            "type": "triggered_tags",
            "triggers": [_TRIGGER],
            "tags": tags,
            "stop_after_first": stop_after_first,
        }
    else:
        # Calls only. Up to two leading newlines match the chat templates,
        # which open a call on a fresh line; the bound keeps greedy decoding
        # from padding the reply with whitespace.
        structure = {
            "type": "sequence",
            "elements": [
                {"type": "regex", "pattern": r"\n{0,2}"},
                {
                    "type": "tags_with_separator",
                    "tags": tags,
                    "separator": "\n",
                    "at_least_one": True,
                    "stop_after_first": stop_after_first,
                },
            ],
        }
    return GrammarSpec(
        kind="structural_tag",
        value={"type": "structural_tag", "format": structure},
        label=f"tool_strict:{mode}",
    )


def _check_schema_cache_collisions(tags: list[dict[str, Any]]) -> None:
    """Reject schemas XGrammar would conflate in its schema cache.

    XGrammar 0.1.32 keys cached subschemas with ``_XGRAMMAR_CACHE_SKIPPED_KEYS``
    removed from every object, property names included. Two subschemas that
    differ only under a property such as ``description`` therefore share one
    grammar, and the second is constrained by the first one's types.
    """
    first_with_key: dict[bytes, dict[str, Any]] = {}
    for schema in _json_schema_contents(tags):
        for subschema in iter_subschemas(schema):
            cache_key = orjson.dumps(
                _without_keys(subschema, _XGRAMMAR_CACHE_SKIPPED_KEYS), option=orjson.OPT_SORT_KEYS
            )
            first = first_with_key.setdefault(cache_key, subschema)
            if first is not subschema and _schema_meaning(first) != _schema_meaning(subschema):
                raise StrictToolError(
                    "tools",
                    "strict tool schemas contain subschemas that differ only under a property named "
                    f"one of {sorted(_XGRAMMAR_CACHE_SKIPPED_KEYS)}, which this model's grammar backend "
                    "cannot tell apart",
                )


def _schema_meaning(schema: dict[str, Any]) -> bytes:
    """``schema`` without annotation keywords, property names kept."""
    return orjson.dumps(without_schema_keywords(schema, _XGRAMMAR_CACHE_SKIPPED_KEYS), option=orjson.OPT_SORT_KEYS)


def _json_schema_contents(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if value.get("type") == "json_schema" and isinstance(value.get("json_schema"), dict):
            yield value["json_schema"]
            return
        for child in value.values():
            yield from _json_schema_contents(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_schema_contents(child)


def _without_keys(value: Any, keys: frozenset[str]) -> Any:
    if isinstance(value, dict):
        return {key: _without_keys(child, keys) for key, child in value.items() if key not in keys}
    if isinstance(value, list):
        return [_without_keys(child, keys) for child in value]
    return value


def _function_parameters(function: dict[str, Any], path: str) -> dict[str, Any]:
    parameters = function.get("parameters")
    if parameters is None:
        return {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise StrictToolError(path, f"'{path}' must be a JSON Schema object")
    return parameters


def _hermes_json_tool_tag(function: dict[str, Any], *, strict: bool, path: str) -> dict[str, Any]:
    """Return the tag for one Hermes JSON call.

    The call is ``<tool_call>``, a newline, ``{"name": "NAME", "arguments": ARGS}``,
    a newline, and ``</tool_call>``.
    """
    begin = _HERMES_CALL_START + orjson.dumps(function["name"]).decode() + ', "arguments": '
    if strict:
        schema = _function_parameters(function, path)
        _check_enforceable(schema, path)
        if schema.get("type") not in (None, "object"):
            raise StrictToolError(path, f"'{path}' must describe an object for a strict tool")
        # Strict arguments never carry undeclared keys, as with Qwen XML; an
        # open object would let the backend generate them.
        schema = {**schema, "additionalProperties": False}
    else:
        # XGrammar treats an object without ``additionalProperties`` as closed.
        schema = {"type": "object", "additionalProperties": True}
    return {
        "type": "tag",
        "begin": begin,
        "content": {"type": "json_schema", "json_schema": schema},
        "end": _HERMES_CALL_END,
    }


def _qwen_xml_tool_tag(function: dict[str, Any], *, strict: bool, path: str) -> dict[str, Any]:
    """Return the tag for one Qwen3(-Coder) XML call.

    The layout is exactly what the chat template renders::

        <tool_call>
        <function=NAME>
        <parameter=KEY>
        VALUE
        </parameter>
        </function>
        </tool_call>

    A strict tool's parameters appear in schema order; required ones are
    mandatory and undeclared ones cannot appear.
    """
    begin = _QWEN_CALL_START + function["name"] + ">\n"
    if not strict:
        # A free body must not close the call and open another, which would
        # let a strict call appear without its grammar.
        free = {"type": "any_text", "excludes": [_TRIGGER, "</tool_call>"]}
        return {"type": "tag", "begin": begin, "content": free, "end": _QWEN_CALL_END}
    parameters = _function_parameters(function, path)
    _check_enforceable(parameters, path)
    properties, required = _object_properties(parameters, path)
    if len(properties) > MAX_XML_PARAMS:
        # The parser reads at most this many parameters from one call.
        raise StrictToolError(path, f"'{path}' declares more than {MAX_XML_PARAMS} properties")
    elements: list[dict[str, Any]] = []
    for key, schema in properties.items():
        if not _XML_PARAMETER_NAME.fullmatch(key):
            raise StrictToolError(path, f"parameter name {key!r} cannot be written in this model's tool-call format")
        parameter: dict[str, Any] = {
            "type": "tag",
            "begin": f"<parameter={key}>\n",
            "content": _qwen_xml_value_format(schema, parameters, f"{path}.properties.{key}"),
            "end": "\n</parameter>\n",
        }
        if key not in required:
            parameter = {"type": "or", "elements": [parameter, {"type": "const_string", "value": ""}]}
        elements.append(parameter)
    content: dict[str, Any] = (
        {"type": "sequence", "elements": elements} if elements else {"type": "const_string", "value": ""}
    )
    return {"type": "tag", "begin": begin, "content": content, "end": _QWEN_CALL_END}


def _check_enforceable(parameters: dict[str, Any], path: str) -> None:
    """Reject a strict schema that cannot be served, not one the grammar only loosens.

    Every strict call's arguments are validated against the full schema when
    the turn ends (:func:`~sie_server.processors.strict_grammar.enforce_strict_tool_arguments`),
    so a keyword the grammar backend drops (``multipleOf``, ``pattern`` beside
    ``format``, an assertion beside ``$ref``) can at worst fail a turn, never
    return arguments that break the schema. What remains rejected here is a
    keyword the verifier itself cannot evaluate, a reference the per-parameter
    grammar would resolve elsewhere, and listed values no argument can match.
    """
    keyword = unverifiable_schema_keyword(parameters)
    if keyword is not None:
        raise StrictToolError(path, f"'{keyword}' in '{path}' is not supported for strict tools")
    for subschema in iter_subschemas(parameters):
        _check_reference(subschema.get("$ref"), path)
        _check_listed_values(subschema, path)


def _check_reference(ref: Any, path: str) -> None:
    """Accept only references into the definition maps.

    A parameter's schema is compiled on its own, with the tool's definitions
    attached, so a pointer to any other location would resolve elsewhere.
    """
    if ref is not None and not (isinstance(ref, str) and ref.startswith(_DEFINITION_REF_PREFIXES)):
        raise StrictToolError(path, f"'$ref' {ref!r} in '{path}' must point into '$defs' for strict tools")
    # XGrammar looks up escaped pointer segments (``~0`` / ``~1``) literally.
    if isinstance(ref, str) and "~" in ref:
        raise StrictToolError(path, f"'$ref' {ref!r} in '{path}' uses an escaped pointer, unsupported for strict tools")


def _json_key(value: Any) -> Hashable:
    """A key under which two values are equal exactly when JSON Schema finds them equal.

    Numbers compare by value (``1`` equals ``1.0``) and never equal a boolean
    (``true`` is not ``1``), at every nesting level.
    """
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int | float):
        return ("number", value)
    if isinstance(value, list):
        return ("array", tuple(_json_key(item) for item in value))
    if isinstance(value, dict):
        return ("object", frozenset((key, _json_key(item)) for key, item in value.items()))
    return (type(value).__name__, value)


def _check_listed_values(subschema: dict[str, Any], path: str) -> None:
    """Reject ``enum`` / ``const`` values no argument can match.

    A ``const`` outside its ``enum`` or a listed value of the wrong ``type``
    leaves no conforming argument, so every call would fail validation.
    """
    if "enum" not in subschema and "const" not in subschema:
        return
    enum = subschema.get("enum")
    values = enum if isinstance(enum, list) else []
    if "const" in subschema:
        if "enum" in subschema and _json_key(subschema["const"]) not in {_json_key(value) for value in values}:
            raise StrictToolError(path, f"'const' in '{path}' is not one of its 'enum' values")
        values = [subschema["const"]]
    # A profile that hides reasoning strips these delimiters from the output
    # before the tool parser reads it, which would alter the declared value.
    for value in values:
        if isinstance(value, str) and any(token in value for token in _REASONING_DELIMITERS):
            raise StrictToolError(path, f"listed value {value!r} in '{path}' contains a reasoning delimiter")
    declared = subschema.get("type")
    if declared is None:
        return
    kinds = {declared} if isinstance(declared, str) else set(declared) if isinstance(declared, list) else set()
    if subschema.get("nullable") is True:
        kinds.add("null")
    for value in values:
        if not any(_json_value_has_type(value, kind) for kind in kinds):
            raise StrictToolError(path, f"value {value!r} in '{path}' does not match its 'type'")


def _json_value_has_type(value: Any, kind: Any) -> bool:
    """JSON Schema's type test, the one the argument validator applies (``1.0`` is an integer)."""
    try:
        return Draft202012Validator.TYPE_CHECKER.is_type(value, kind)
    except UndefinedTypeCheck:
        return False


def _object_properties(parameters: dict[str, Any], path: str) -> tuple[dict[str, Any], set[str]]:
    """Return a strict tool's declared properties and required keys."""
    if parameters.get("type") not in (None, "object"):
        raise StrictToolError(path, f"'{path}' must describe an object for a strict tool")
    for keyword in _UNSUPPORTED_PARAMETERS_KEYWORDS:
        if keyword in parameters:
            raise StrictToolError(path, f"'{keyword}' in '{path}' is not supported for strict tools on this model")
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise StrictToolError(path, f"'{path}.properties' must be an object")
    required = parameters.get("required", [])
    if not isinstance(required, list) or not all(isinstance(key, str) for key in required):
        raise StrictToolError(path, f"'{path}.required' must be an array of strings")
    missing = [key for key in required if key not in properties]
    if missing:
        raise StrictToolError(path, f"'{path}' requires undeclared properties {missing!r}")
    return properties, {str(key) for key in required}


def _qwen_xml_value_format(schema: Any, root: dict[str, Any], path: str) -> dict[str, Any]:
    """Constrain one XML parameter value so the parser converts it to a conforming value.

    The tool-call parser converts each raw value by the type it infers from the
    schema (:func:`schema_conversion_type`): string-typed values stay raw text,
    a bare ``null`` becomes JSON null where the schema allows it, and every
    other type is read as JSON. The value grammar follows the same split, so an
    accepted value always converts to a conforming argument.
    """
    resolved = resolve_schema_ref(schema, root)
    if resolved is True or resolved == {}:
        return _raw_text_format()
    if not isinstance(resolved, dict):
        raise StrictToolError(path, f"'{path}' must be a JSON Schema object with resolvable references")
    kind = schema_conversion_type(resolved, root)
    if kind is None or kind == "string":
        return _raw_string_format(resolved, root, path)
    if kind not in _JSON_KINDS:
        raise StrictToolError(path, f"type {kind!r} at '{path}' is not supported for strict tools")
    if kind in _SCALAR_KINDS and not _is_scalar_or_null(resolved, root, kind):
        # The parser reads a scalar with ``int()`` / ``float()`` / true|false,
        # so a union with a non-null type would not convert back to JSON.
        raise StrictToolError(path, f"'{path}' mixes {kind} with other types, which strict tools cannot enforce here")
    value_schema = dict(resolved)
    if any("$ref" in subschema for subschema in iter_subschemas(value_schema)):
        # References point into the tool's definitions, but XGrammar resolves
        # them against the value schema's own root, so that root carries the
        # tool's definitions in place of any the value declares itself.
        for keyword in _DEFINITION_KEYWORDS:
            if keyword in root:
                value_schema[keyword] = root[keyword]
            else:
                value_schema.pop(keyword, None)
    return {"type": "json_schema", "json_schema": value_schema}


def _is_scalar_or_null(
    schema: Any, root: dict[str, Any], kind: str, depth: int = 0, seen: dict[tuple[int, int], bool] | None = None
) -> bool:
    """Return whether every value ``schema`` admits is a ``kind`` scalar or null.

    ``seen`` memoizes each resolved schema: unions over shared definitions would
    otherwise be walked once per path, exponentially in their depth.
    """
    seen = {} if seen is None else seen
    schema = resolve_schema_ref(schema, root)
    if not isinstance(schema, dict) or depth > _MAX_UNION_DEPTH or "allOf" in schema or "not" in schema:
        return False
    key = (id(schema), depth)
    if key in seen:
        return seen[key]
    if "anyOf" in schema:
        variants = schema["anyOf"]
        seen[key] = isinstance(variants, list) and all(
            _is_scalar_or_null(variant, root, kind, depth + 1, seen) for variant in variants
        )
        return seen[key]
    declared = schema.get("type")
    if declared is None:
        # A typeless branch admits exactly its listed values, and the parser
        # converts them by ``kind``.
        enum = schema.get("enum")
        values = [schema["const"]] if "const" in schema else enum if isinstance(enum, list) else None
        return values is not None and all(value is None or _json_value_has_type(value, kind) for value in values)
    if isinstance(declared, str):
        return declared in (kind, "null")
    return isinstance(declared, list) and set(declared) <= {kind, "null"}


def _raw_string_format(schema: dict[str, Any], root: dict[str, Any], path: str) -> dict[str, Any]:
    """Raw string value: free text, or one of the declared string values (plus ``null`` when allowed)."""
    # Optional strings (``anyOf: [{type: string}, {type: null}]``) and unions of
    # string enums, possibly behind ``$ref``, stay raw text. Assertions the raw
    # grammar cannot express (``pattern``, ``format``, lengths, ...) are left to
    # the end-of-turn argument validation.
    variants: list[Any] = [schema]
    if isinstance(schema.get("anyOf"), list):
        variants = [resolve_schema_ref(variant, root) for variant in schema["anyOf"]]
    choices: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise StrictToolError(path, f"'{path}' must be a JSON Schema object with resolvable references")
        declared = variant.get("type")
        if declared in ("null", ["null"]):
            values = _allowed_values(variant, path)
            if values is None or None in values:
                choices.append("null")
            continue
        if declared not in (None, "string") and not (
            isinstance(declared, list) and set(declared) <= {"string", "null"}
        ):
            raise StrictToolError(path, f"'{path}' mixes string with other types, which strict tools cannot enforce")
        values = _allowed_values(variant, path)
        if values is None:
            # Free text; the parser maps a bare ``null`` to JSON null when the
            # schema allows it, and keeps it as the string otherwise.
            return _raw_text_format()
        # ``enum`` / ``const`` restrict the values independently of ``type``,
        # so ``null`` is allowed only when it is listed and the type admits it.
        null_type = (
            declared is None or variant.get("nullable") is True or (isinstance(declared, list) and "null" in declared)
        )
        for value in values:
            if isinstance(value, str):
                if any(tag in value for tag in _RAW_VALUE_EXCLUDES):
                    raise StrictToolError(path, f"'{path}' enum values cannot contain a closing tag")
                if value.strip().lower() == "null" and schema_allows_null(schema, root):
                    # The parser reads a bare ``null`` as JSON null here.
                    raise StrictToolError(path, f"'{path}' allows null, so its enum cannot list the string {value!r}")
                choices.append(value)
            elif value is None and null_type:
                choices.append("null")
    if not choices:
        raise StrictToolError(path, f"'{path}' admits no string value")
    pattern = "(?:" + "|".join(_regex_literal(choice) for choice in dict.fromkeys(choices)) + ")"
    return {"type": "regex", "pattern": pattern}


def _allowed_values(variant: dict[str, Any], path: str) -> list[Any] | None:
    """Return the values ``enum`` and ``const`` together allow, or ``None`` when neither is set."""
    enum = variant.get("enum")
    if enum is not None and (
        not isinstance(enum, list) or not all(isinstance(value, str) or value is None for value in enum)
    ):
        raise StrictToolError(path, f"'{path}' enum values must be strings")
    if "const" not in variant:
        return enum
    const = variant["const"]
    if not (isinstance(const, str) or const is None):
        raise StrictToolError(path, f"'{path}' const must be a string")
    # ``_check_listed_values`` already rejected a ``const`` outside its ``enum``.
    return [const]


def _raw_text_format() -> dict[str, Any]:
    # The parser ends a value at the first ``</parameter>``, and the stream
    # parser ends the whole call at the first ``</tool_call>``.
    return {"type": "any_text", "excludes": list(_RAW_VALUE_EXCLUDES)}


def _regex_literal(text: str) -> str:
    return "".join("\\" + char if char in _REGEX_SPECIAL else _REGEX_ESCAPES.get(char, char) for char in text)


# The tool-call formats whose strict tools can be enforced, and the tag each builds.
_STRICT_TAG_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "qwen_xml": _qwen_xml_tool_tag,
    "hermes_json": _hermes_json_tool_tag,
}
