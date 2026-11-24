"""Strict function tools compile to an XGrammar structural tag.

Shape tests run everywhere. Acceptance tests compile the tag with XGrammar,
the backend SGLang uses for it, and run when ``xgrammar`` is installed.
"""

from __future__ import annotations

from typing import Any

import orjson
import pytest
from sie_server.processors.tool_call_grammar import (
    StrictToolError,
    ToolChoiceError,
    build_strict_tool_grammar,
    build_tool_choice_grammar,
    has_strict_tools,
    tools_require_grammar,
)
from sie_server.processors.tool_call_parser import MAX_XML_PARAMS, _tool_call_deltas, tool_parameter_schemas
from sie_server.processors.tool_call_parser import _parse_xml_tool_call as parse_xml_tool_call

SEARCH = {
    "type": "object",
    "properties": {
        "keywords": {"type": "array", "items": {"type": "string"}},
        "exact": {"type": "boolean"},
        "filter": {
            "type": "object",
            "properties": {"measures": {"type": "array", "items": {"type": "string"}}},
            "required": ["measures"],
            "additionalProperties": False,
        },
        "limit": {"type": "integer"},
    },
    "required": ["keywords", "exact", "filter", "limit"],
    "additionalProperties": False,
}

# OpenAI strict schemas express optional fields as required and nullable.
# Same shapes as Pydantic ``model_json_schema()`` output.
LOOKUP = {
    "type": "object",
    "$defs": {
        "Kind": {"type": "string", "enum": ["metric", "fact", "label"]},
        "Spec": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "type": {"type": "string", "const": "label"}},
            "required": ["id", "type"],
            "additionalProperties": False,
        },
    },
    "properties": {
        "keywords": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        "kinds": {"type": "array", "items": {"$ref": "#/$defs/Kind"}},
        "limit": {"anyOf": [{"type": "integer", "minimum": 1, "maximum": 100}, {"type": "null"}]},
        "spec": {"anyOf": [{"$ref": "#/$defs/Spec"}, {"type": "null"}]},
        "eligible": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "kind": {"anyOf": [{"$ref": "#/$defs/Kind"}, {"type": "null"}]},
        "explicit": {"type": "boolean"},
    },
    "required": ["keywords", "kinds", "limit", "spec", "eligible", "note", "kind", "explicit"],
    "additionalProperties": False,
}

_PART_NAMES = ("text", "chart", "table", "doc", "kda", "alert", "patch", "board", "what_if", "results")
ANSWER = {
    "type": "object",
    "$defs": {
        **{
            f"{name.title()}Part": {
                "type": "object",
                "properties": {f"{name}_ref": {"type": "string"}},
                "required": [f"{name}_ref"],
            }
            for name in _PART_NAMES
        },
        "Mood": {"type": "string", "enum": ["positive", "negative", "neutral"]},
    },
    "properties": {
        "parts": {
            "type": "array",
            "items": {"anyOf": [{"$ref": f"#/$defs/{name.title()}Part"} for name in _PART_NAMES]},
        },
        "status": {"type": "string", "enum": ["partial", "final"], "default": "final"},
        "mood": {"anyOf": [{"$ref": "#/$defs/Mood"}, {"type": "null"}], "default": None},
    },
    "required": ["parts"],
}


def _tool(name: str, parameters: dict[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    function: dict[str, Any] = {"name": name, "strict": strict}
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


TOOLS = [_tool("search", SEARCH), _tool("lookup", LOOKUP), _tool("final_answer", ANSWER)]


def _call(name: str, arguments: list[tuple[str, str]]) -> str:
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in arguments)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


GOOD_SEARCH = _call(
    "search",
    [("keywords", '["revenue"]'), ("exact", "true"), ("filter", '{"measures": ["revenue"]}'), ("limit", "5")],
)
# The probe's violating call: a string for a boolean, a string for an object, no limit.
BAD_SEARCH = _call("search", [("keywords", '["revenue"]'), ("exact", "yes"), ("filter", "revenue")])
LOOKUP_VALUES = {
    "keywords": '["revenue"]',
    "kinds": '["metric", "fact"]',
    "limit": "10",
    "spec": "null",
    "eligible": "true",
    "note": "null",
    "kind": "metric",
    "explicit": "false",
}
GOOD_LOOKUP = _call("lookup", list(LOOKUP_VALUES.items()))
GOOD_ANSWER = _call(
    "final_answer",
    [("parts", '[{"text_ref": "t1"}, {"chart_ref": "c1"}]'), ("status", "final"), ("mood", "null")],
)


def _format(spec: Any) -> dict[str, Any]:
    assert spec is not None
    assert spec.kind == "structural_tag"
    return spec.value["format"]


def _calls_format(spec: Any) -> dict[str, Any]:
    """The format holding the call tags (``required`` wraps it after a newline prefix)."""
    fmt = _format(spec)
    return fmt["elements"][1] if fmt["type"] == "sequence" else fmt


def _tags(spec: Any) -> list[dict[str, Any]]:
    return _calls_format(spec)["tags"]


def _parameter_contents(tag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contents: dict[str, dict[str, Any]] = {}
    for element in tag["content"]["elements"]:
        parameter = element["elements"][0] if element["type"] == "or" else element
        key = parameter["begin"].removeprefix("<parameter=").removesuffix(">\n")
        contents[key] = parameter["content"]
    return contents


# -- detection and routing ------------------------------------------------------


def test_has_strict_tools() -> None:
    assert has_strict_tools(TOOLS)
    assert not has_strict_tools([_tool("search", SEARCH, strict=False)])
    assert not has_strict_tools(None)


@pytest.mark.parametrize(
    ("tools", "choice", "expected"),
    [
        ([_tool("search", SEARCH)], None, True),
        ([_tool("search", SEARCH)], "auto", True),
        ([_tool("search", SEARCH)], "none", False),
        ([_tool("search", SEARCH, strict=False)], "auto", False),
        ([_tool("search", SEARCH, strict=False)], "required", True),
        ([_tool("search", SEARCH, strict=False)], {"type": "function", "function": {"name": "search"}}, True),
        (None, "required", False),
        ([], "required", False),
    ],
)
def test_tools_require_grammar(tools: Any, choice: Any, expected: bool) -> None:
    assert tools_require_grammar(tools, choice) is expected


# -- structure ------------------------------------------------------------------


def test_auto_uses_triggered_tags_over_every_tool() -> None:
    spec = build_strict_tool_grammar(TOOLS, "auto", "qwen_xml")
    fmt = _format(spec)
    assert fmt["type"] == "triggered_tags"
    assert fmt["triggers"] == ["<tool_call>"]
    assert [tag["begin"] for tag in fmt["tags"]] == [
        "<tool_call>\n<function=search>\n",
        "<tool_call>\n<function=lookup>\n",
        "<tool_call>\n<function=final_answer>\n",
    ]
    assert fmt["stop_after_first"] is False
    assert spec is not None
    assert spec.label == "tool_strict:auto"


def test_required_admits_only_calls() -> None:
    fmt = _format(build_strict_tool_grammar(TOOLS, "required", "qwen_xml"))
    assert fmt["type"] == "sequence"
    calls = fmt["elements"][1]
    assert calls["type"] == "tags_with_separator"
    assert calls["at_least_one"] is True


def test_named_pins_one_tool() -> None:
    spec = build_strict_tool_grammar(TOOLS, {"type": "function", "function": {"name": "lookup"}}, "qwen_xml")
    assert [tag["begin"] for tag in _tags(spec)] == ["<tool_call>\n<function=lookup>\n"]


@pytest.mark.parametrize("choice", ["auto", "required"])
def test_parallel_tool_calls_false_stops_after_first_call(choice: str) -> None:
    spec = build_strict_tool_grammar(TOOLS, choice, "qwen_xml", parallel_tool_calls=False)
    assert _calls_format(spec)["stop_after_first"] is True


def test_none_builds_no_grammar() -> None:
    assert build_strict_tool_grammar(TOOLS, "none", "qwen_xml") is None


def test_named_tool_must_exist() -> None:
    with pytest.raises(ToolChoiceError):
        build_strict_tool_grammar(TOOLS, {"type": "function", "function": {"name": "nope"}}, "qwen_xml")


def test_qwen_xml_parameters_follow_schema_order_and_requiredness() -> None:
    tag = _tags(build_strict_tool_grammar([_tool("final_answer", ANSWER)], "required", "qwen_xml"))[0]
    elements = tag["content"]["elements"]
    assert elements[0]["type"] == "tag"
    assert elements[0]["begin"] == "<parameter=parts>\n"
    assert elements[0]["end"] == "\n</parameter>\n"
    # ``status`` and ``mood`` are optional: each may be skipped.
    assert [element["type"] for element in elements[1:]] == ["or", "or"]
    assert elements[1]["elements"][1] == {"type": "const_string", "value": ""}
    assert tag["end"] == "</function>\n</tool_call>"


def test_qwen_xml_value_rules_per_type() -> None:
    contents = _parameter_contents(_tags(build_strict_tool_grammar([_tool("lookup", LOOKUP)], "auto", "qwen_xml"))[0])
    # Nullable array, object behind ``$ref`` and the bounded integer are JSON.
    assert contents["keywords"]["type"] == "json_schema"
    assert contents["spec"]["type"] == "json_schema"
    assert contents["spec"]["json_schema"]["$defs"] == LOOKUP["$defs"]
    assert contents["limit"]["json_schema"]["anyOf"][0] == {"type": "integer", "minimum": 1, "maximum": 100}
    assert contents["eligible"]["type"] == "json_schema"
    assert contents["explicit"] == {"type": "json_schema", "json_schema": {"type": "boolean"}}
    # Strings stay raw text; a string enum behind ``$ref`` is an alternation.
    assert contents["note"] == {"type": "any_text", "excludes": ["</parameter>", "</tool_call>"]}
    assert contents["kind"] == {"type": "regex", "pattern": "(?:metric|fact|label|null)"}


def test_enum_values_are_regex_escaped() -> None:
    schema = {"type": "object", "properties": {"op": {"enum": ["a.b", "c|d", "(x)"]}}, "required": ["op"]}
    contents = _parameter_contents(_tags(build_strict_tool_grammar([_tool("f", schema)], "required", "qwen_xml"))[0])
    assert contents["op"]["pattern"] == r"(?:a\.b|c\|d|\(x\))"


def test_tool_without_parameters_admits_empty_call() -> None:
    tag = _tags(build_strict_tool_grammar([_tool("ping", None)], "required", "qwen_xml"))[0]
    assert tag["content"] == {"type": "const_string", "value": ""}


def test_non_strict_tool_keeps_free_arguments() -> None:
    tools = [_tool("search", SEARCH), _tool("free", {"type": "object"}, strict=False)]
    tags = _tags(build_strict_tool_grammar(tools, "auto", "qwen_xml"))
    assert tags[1]["content"] == {"type": "any_text", "excludes": ["<tool_call>", "</tool_call>"]}


def test_hermes_json_constrains_arguments_by_schema() -> None:
    tag = _tags(build_strict_tool_grammar([_tool("search", SEARCH)], "required", "hermes_json"))[0]
    assert tag["begin"] == '<tool_call>\n{"name": "search", "arguments": '
    assert tag["content"] == {"type": "json_schema", "json_schema": SEARCH}
    assert tag["end"] == "}\n</tool_call>"


def test_hermes_json_closes_an_open_strict_arguments_object() -> None:
    open_schema = {**SEARCH, "additionalProperties": True}
    tag = _tags(build_strict_tool_grammar([_tool("search", open_schema)], "required", "hermes_json"))[0]
    assert tag["content"]["json_schema"]["additionalProperties"] is False
    assert open_schema["additionalProperties"] is True  # the request's schema is not mutated


def test_parser_rejects_a_call_cut_inside_a_value() -> None:
    """A ``</tool_call>`` inside a JSON string ends the block early.

    When a ``</parameter>`` sits inside the cut string, taking it as the
    value's end would report a wrong value, so the call fails. Without one the
    scan stops as for any unclosed parameter, and the end-of-turn schema check
    covers strict calls.
    """
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})]
    schemas = tool_parameter_schemas(tools)
    with pytest.raises(ValueError, match="cut inside a JSON string"):
        parse_xml_tool_call('<function=f>\n<parameter=tags>\n["a</parameter>b', schemas)
    assert parse_xml_tool_call('<function=f>\n<parameter=tags>\n["a', schemas) == ("f", {})


@pytest.mark.parametrize("kind", ["dict", "list[str]"])
def test_parser_reads_closing_tags_inside_json_for_container_type_names(kind: str) -> None:
    value = '{"k": "a</parameter>b"}' if kind == "dict" else '["a</parameter>b"]'
    tools = [_tool("f", {"type": "object", "properties": {"x": {"type": kind}}}, strict=False)]
    body = f"<function=f>\n<parameter=x>\n{value}\n</parameter>\n</function>"
    _, arguments = parse_xml_tool_call(body, tool_parameter_schemas(tools))
    assert arguments["x"] == orjson.loads(value)


def test_structural_tag_serializes() -> None:
    spec = build_strict_tool_grammar(TOOLS, "auto", "qwen_xml")
    assert spec is not None
    assert orjson.loads(orjson.dumps(spec.value)) == spec.value


def test_non_strict_requests_keep_the_forcing_regex() -> None:
    spec = build_tool_choice_grammar([_tool("search", SEARCH, strict=False)], "required", "qwen_xml")
    assert spec is not None
    assert spec.kind == "regex"


# -- rejection ------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["glm_xml", "muse_atem", "gemma4", "auto"])
def test_unsupported_format_is_rejected(fmt: Any) -> None:
    with pytest.raises(StrictToolError) as excinfo:
        build_strict_tool_grammar(TOOLS, "auto", fmt)
    assert excinfo.value.param == "tools"


@pytest.mark.parametrize(
    ("schema", "fragment"),
    [
        ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "mixes string"),
        ({"anyOf": [{"type": "integer"}, {"type": "object"}]}, "mixes integer"),
        ({"enum": ["a", 1]}, "enum values must be strings"),
        ({"$ref": "#/$defs/missing"}, "resolvable references"),
        ({"type": "uuid"}, "type 'uuid'"),
        ({"type": "string", "enum": ["a</parameter>b"]}, "cannot contain"),
        ({"type": "string", "enum": ["a</tool_call>b"]}, "cannot contain"),
        ({"type": "string", "const": 1}, "does not match its 'type'"),
        ({"type": "string", "enum": ["a"], "const": "b"}, "is not one of its 'enum' values"),
        ({"type": "string", "enum": ["x", None]}, "does not match its 'type'"),
        ({"type": "integer", "enum": [1, "a"]}, "does not match its 'type'"),
        ({"$ref": "#/properties/other"}, "must point into"),
        ({"type": "array", "items": {"$ref": "#/$defs/T~1"}}, "escaped pointer"),
        ({"const": True, "enum": [1]}, "is not one of its 'enum' values"),
        ({"const": [True], "enum": [[1]]}, "is not one of its 'enum' values"),
        ({"anyOf": [{"type": "integer"}, {"const": True}]}, "mixes integer"),
        ({"type": "string", "enum": ["a<think>b"]}, "reasoning delimiter"),
        ({"type": "array", "items": {"const": "</think>"}}, "reasoning delimiter"),
        ({"type": ["string", "null"], "enum": ["null"]}, "cannot list the string 'null'"),
        ({"anyOf": [{"type": "string", "enum": ["NULL"]}, {"type": "null"}]}, "cannot list the string 'NULL'"),
        ({"anyOf": [{"type": "string", "enum": ["x"]}, {"type": "null", "enum": ["x"]}]}, "does not match"),
        ({"type": "object", "properties": {"a": {}}, "unevaluatedProperties": False}, "'unevaluatedProperties'"),
    ],
)
def test_unservable_parameter_is_rejected(schema: dict[str, Any], fragment: str) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    with pytest.raises(StrictToolError) as excinfo:
        build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml")
    assert fragment in str(excinfo.value)
    assert excinfo.value.param.startswith("tools[0].function.parameters")


# Keywords XGrammar 0.1.32 compiles without enforcing. They are accepted: the
# grammar still constrains what it can, and every call's arguments are checked
# against the full schema when the turn ends (``enforce_strict_tool_arguments``).
_LOOSENED_BY_THE_BACKEND = [
    {"type": "string", "pattern": "^a+$"},
    {"type": "string", "format": "date"},
    {"type": "string", "maxLength": 3},
    {"type": "integer", "enum": [1, 2], "minimum": 2},
    {"type": "string", "const": "abc", "maxLength": 1},
    {"type": "integer", "minimum": 10, "exclusiveMinimum": 0},
    {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
    {"type": "array", "items": {"type": "string", "pattern": "^a+$", "maxLength": 1}},
    {"type": "array", "items": {"minimum": 10}},
    {"type": "array", "items": {"type": "string", "format": "date", "pattern": "^2025-"}},
    {"type": "integer", "multipleOf": 2},
    {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
    {"type": "array", "items": {"type": "integer"}, "contains": {"const": 5}},
    {"type": "object", "properties": {"a": {"type": "integer"}}, "not": {"required": ["a"]}},
    {"type": "object", "properties": {"a": {"type": "integer"}}, "dependentRequired": {"a": ["a"]}},
    {"type": "object", "if": {"required": ["a"]}, "then": {"required": ["b"]}},
    {"type": "array", "items": {"oneOf": [{"enum": [1, 2]}, {"enum": [2, 3]}]}},
    {"type": "array", "items": {"$ref": "#/$defs/I", "maximum": 3}},
    {"anyOf": [{"type": "integer"}, {"type": "null"}], "maximum": 3},
    {"$ref": "#/$defs/I", "maximum": 3},
]


@pytest.mark.parametrize("schema", _LOOSENED_BY_THE_BACKEND)
@pytest.mark.parametrize("fmt", ["qwen_xml", "hermes_json"])
def test_keywords_the_backend_loosens_are_accepted(schema: dict[str, Any], fmt: Any) -> None:
    parameters = {
        "type": "object",
        "$defs": {"S": {"type": "string"}, "I": {"type": "integer"}},
        "properties": {"value": schema},
        "required": ["value"],
    }
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", fmt) is not None


def test_annotations_beside_a_reference_or_union_are_allowed() -> None:
    parameters = {
        "type": "object",
        "$defs": {"S": {"type": "string"}},
        "properties": {
            "a": {"$ref": "#/$defs/S", "description": "a string"},
            "b": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "B"},
        },
        "required": ["a", "b"],
    }
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml") is not None


def test_scalar_check_over_shared_references_is_not_exponential() -> None:
    defs: dict[str, Any] = {"A0": {"type": "integer"}}
    for level in range(1, 9):
        defs[f"A{level}"] = {"anyOf": [{"$ref": f"#/$defs/A{level - 1}"}] * 20}
    parameters = {"type": "object", "$defs": defs, "properties": {"n": {"$ref": "#/$defs/A8"}}, "required": ["n"]}
    contents = _parameter_contents(_tags(build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml"))[0])
    assert contents["n"]["type"] == "json_schema"


def test_keyword_names_used_as_property_names_are_not_rejected() -> None:
    parameters = {
        "type": "object",
        "properties": {"not": {"type": "string"}, "contains": {"type": "string", "enum": ["oneOf"]}},
        "required": ["not", "contains"],
    }
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml") is not None


@pytest.mark.parametrize(
    ("schema", "pattern"),
    [
        ({"type": "string", "enum": ["a", "b"], "const": "a"}, "(?:a)"),
        ({"type": ["string", "null"], "enum": ["x"]}, "(?:x)"),
        ({"type": ["string", "null"], "enum": ["x", None]}, "(?:x|null)"),
        ({"anyOf": [{"type": "string", "enum": ["x"]}, {"type": "null"}]}, "(?:x|null)"),
        ({"anyOf": [{"type": "string", "enum": ["x"]}, {"type": "null", "enum": [None]}]}, "(?:x|null)"),
        ({"type": ["null"]}, "(?:null)"),
        # Without null in the schema the parser keeps the text ``null`` as a string.
        ({"type": "string", "enum": ["null"]}, "(?:null)"),
    ],
)
def test_string_choices_honour_enum_const_and_null(schema: dict[str, Any], pattern: str) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    contents = _parameter_contents(_tags(build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml"))[0])
    assert contents["value"] == {"type": "regex", "pattern": pattern}


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "array"},
        {"anyOf": [{"type": "object"}]},
        {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
        {"type": "object", "properties": {}, "required": ["missing"]},
        {"type": "object", "properties": {"bad name": {"type": "string"}, "a<b": {"type": "string"}}},
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "const": {"x": "a"}},
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "enum": [{"x": "a"}]},
        {"type": "object", "properties": {f"p{index}": {"type": "string"} for index in range(MAX_XML_PARAMS + 1)}},
        {
            "$ref": "#/$defs/Args",
            "$defs": {"Args": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
        },
    ],
)
def test_unenforceable_parameters_object_is_rejected(parameters: dict[str, Any]) -> None:
    with pytest.raises(StrictToolError):
        build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml")


def test_unservable_strict_tool_is_rejected_even_when_not_named() -> None:
    tools = [_tool("search", SEARCH), _tool("bad", {"type": "object", "properties": {"d": {"type": "uuid"}}})]
    with pytest.raises(StrictToolError):
        build_strict_tool_grammar(tools, {"type": "function", "function": {"name": "search"}}, "qwen_xml")


# -- the parser reads every accepted value back as a conforming argument --------


def test_parser_converts_constrained_values_to_schema_types() -> None:
    schemas = tool_parameter_schemas(TOOLS)
    body = GOOD_LOOKUP.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert parse_xml_tool_call(body, schemas) == (
        "lookup",
        {
            "keywords": ["revenue"],
            "kinds": ["metric", "fact"],
            "limit": 10,
            "spec": None,
            "eligible": True,
            "note": None,
            "kind": "metric",
            "explicit": False,
        },
    )


def test_parser_reads_a_closing_tag_inside_a_json_string() -> None:
    """A JSON string may contain ``</parameter>``; the value ends where the JSON is complete."""
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})]
    call = _call("f", [("tags", '["a</parameter>b", "c"]')])
    body = call.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"tags": ["a</parameter>b", "c"]})


def test_parser_keeps_a_closed_malformed_array_value_raw() -> None:
    """A closed call whose array value is not JSON keeps the raw text instead of failing."""
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array"}}}, strict=False)]
    body = '<function=f>\n<parameter=tags>\n["a, b]\n</parameter>\n</function>'
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"tags": '["a, b]'})


@pytest.mark.parametrize("schema", [{"type": "integer", "exclusiveMinimum": 0}, {"type": "number", "minimum": 0}])
def test_bounds_the_backend_enforces_are_accepted(schema: dict[str, Any]) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None


def test_parser_reads_a_typeless_null_const_as_json_null() -> None:
    tools = [_tool("f", {"type": "object", "properties": {"x": {"const": None}}, "required": ["x"]})]
    contents = _parameter_contents(_tags(build_strict_tool_grammar(tools, "auto", "qwen_xml"))[0])
    assert contents["x"] == {"type": "regex", "pattern": "(?:null)"}
    body = "<function=f>\n<parameter=x>\nnull\n</parameter>\n</function>"
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"x": None})


@pytest.mark.parametrize(
    ("payload", "fmt"),
    [
        ('{"name": "f", "arguments": {"x": 1e999}}', "hermes_json"),
        ("<function=f>\n<parameter=xs>\n[1e999]\n</parameter>\n</function>", "qwen_xml"),
    ],
)
def test_an_overflowing_number_fails_the_call_instead_of_emitting_infinity(payload: str, fmt: Any) -> None:
    tools = [_tool("f", {"type": "object", "properties": {"x": {"type": "number"}, "xs": {"type": "array"}}})]
    with pytest.raises(ValueError, match="not JSON-serialisable"):
        _tool_call_deltas(payload, 0, fmt, tool_parameter_schemas(tools))


@pytest.mark.parametrize(("value", "accepted"), [(1, True), (1.0, True), (1.5, False), (True, False)])
def test_integer_listed_values_follow_json_schema_equality(value: Any, accepted: bool) -> None:
    parameters = {"type": "object", "properties": {"x": {"type": "integer", "const": value}}, "required": ["x"]}
    if accepted:
        assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None
    else:
        with pytest.raises(StrictToolError, match="does not match its 'type'"):
            build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json")


def test_duplicate_function_names_are_rejected() -> None:
    tools = [_tool("f", {"type": "object", "properties": {"x": {"type": "integer"}}}), _tool("f", None, strict=False)]
    with pytest.raises(StrictToolError, match="unique"):
        build_strict_tool_grammar(tools, "auto", "hermes_json")


def _box(kind: str, description: str = "") -> dict[str, Any]:
    field: dict[str, Any] = {"type": kind}
    if description:
        field["description"] = description
    return {"type": "object", "properties": {"description": field}, "required": ["description"]}


def test_subschemas_the_backend_cache_would_conflate_are_rejected() -> None:
    """XGrammar's cache key drops property names such as ``description``, so these two would share one grammar."""
    parameters = {"type": "object", "properties": {"a": _box("integer"), "b": _box("string")}, "required": ["a", "b"]}
    with pytest.raises(StrictToolError, match="cannot tell apart"):
        build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json")


def test_subschemas_differing_only_in_annotations_are_accepted() -> None:
    parameters = {
        "type": "object",
        "properties": {"a": _box("integer", "first"), "b": _box("integer", "second"), "c": _box("string")},
        "required": ["a", "b", "c"],
    }
    parameters["properties"]["c"] = {"type": "object", "properties": {"description": {"type": "string"}, "x": {}}}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None


def test_a_single_property_named_description_is_accepted() -> None:
    parameters = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None


@pytest.mark.parametrize(("text", "expected"), [("1.5", 1.5), ("2", 2)])
def test_parser_keeps_finite_numbers(text: str, expected: float) -> None:
    tools = [_tool("f", {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]})]
    body = f"<function=f>\n<parameter=x>\n{text}\n</parameter>\n</function>"
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"x": expected})


def test_parser_reads_many_closing_tags_inside_a_json_string() -> None:
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})]
    tag = "</parameter>" * 40
    call = _call("f", [("tags", orjson.dumps([tag, "c"]).decode()), ("tail", "x")])
    body = call.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"tags": [tag, "c"], "tail": "x"})


def test_values_resolve_references_against_the_tool_definitions() -> None:
    """A parameter's own ``$defs`` cannot shadow the tool's: ``#`` is the tool's parameters object."""
    parameters = {
        "type": "object",
        "$defs": {"Base": {"type": "integer"}},
        "properties": {
            "box": {
                "type": "object",
                "$defs": {"Base": {"type": "string"}},
                "properties": {"copy": {"$ref": "#/$defs/Base"}},
                "required": ["copy"],
            }
        },
        "required": ["box"],
    }
    contents = _parameter_contents(_tags(build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml"))[0])
    assert contents["box"]["json_schema"]["$defs"] == {"Base": {"type": "integer"}}


@pytest.mark.parametrize(
    "schema",
    [{"const": 1, "enum": [1.0]}, {"type": "array", "const": [1, {"a": 2}], "enum": [[1.0, {"a": 2.0}]]}],
)
def test_const_numerically_equal_to_an_enum_value_is_accepted(schema: dict[str, Any]) -> None:
    parameters = {"type": "object", "properties": {"value": schema}, "required": ["value"]}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "qwen_xml") is not None


# -- acceptance with the real backend -------------------------------------------


@pytest.fixture(scope="module")
def xgr() -> Any:
    return pytest.importorskip("xgrammar")


def _accepts(xgr: Any, spec: Any, text: str) -> bool:
    from xgrammar.testing import _is_grammar_accept_string

    grammar = xgr.Grammar.from_structural_tag(orjson.dumps(spec.value).decode())
    return bool(_is_grammar_accept_string(grammar, text))


@pytest.mark.parametrize("choice", ["auto", "required"])
@pytest.mark.parametrize("text", [GOOD_SEARCH, GOOD_LOOKUP, GOOD_ANSWER])
def test_backend_accepts_conforming_calls(xgr: Any, choice: str, text: str) -> None:
    assert _accepts(xgr, build_strict_tool_grammar(TOOLS, choice, "qwen_xml"), text)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(BAD_SEARCH, id="probe-violation"),
        pytest.param(GOOD_SEARCH.replace("<parameter=limit>\n5\n</parameter>\n", ""), id="missing-required"),
        pytest.param(GOOD_SEARCH.replace("</function>", "<parameter=extra>\nx\n</parameter>\n</function>"), id="extra"),
        pytest.param(GOOD_LOOKUP.replace("\n10\n", "\n500\n"), id="integer-above-maximum"),
        pytest.param(GOOD_LOOKUP.replace("\nmetric\n", "\nmeasure\n"), id="enum-outside-ref"),
        pytest.param(GOOD_LOOKUP.replace('["metric", "fact"]', '["dashboard"]'), id="array-item-enum"),
        pytest.param(GOOD_ANSWER.replace('{"chart_ref": "c1"}', '{"nope": 1}'), id="union-without-match"),
        pytest.param(GOOD_ANSWER.replace("\nfinal\n", "\ndone\n"), id="string-enum"),
        pytest.param(
            GOOD_LOOKUP.replace("<parameter=note>\nnull\n", "<parameter=note>\na</parameter>b\n"),
            id="closing-tag-in-raw-string",
        ),
        pytest.param(
            GOOD_LOOKUP.replace("<parameter=note>\nnull\n", "<parameter=note>\na</tool_call>b\n"),
            id="call-closing-tag-in-raw-string",
        ),
    ],
)
@pytest.mark.parametrize("choice", ["auto", "required"])
def test_backend_rejects_violating_calls(xgr: Any, choice: str, text: str) -> None:
    assert not _accepts(xgr, build_strict_tool_grammar(TOOLS, choice, "qwen_xml"), text)


def test_backend_and_parser_round_trip_a_closing_tag_inside_a_json_string(xgr: Any) -> None:
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})]
    call = _call("f", [("tags", '["a</parameter>b"]')])
    assert _accepts(xgr, build_strict_tool_grammar(tools, "required", "qwen_xml"), call)
    body = call.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"tags": ["a</parameter>b"]})


def test_backend_and_parser_round_trip_many_closing_tags_inside_a_json_string(xgr: Any) -> None:
    tools = [_tool("f", {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}})]
    tags = ["</parameter>" * 20]
    call = _call("f", [("tags", orjson.dumps(tags).decode())])
    assert _accepts(xgr, build_strict_tool_grammar(tools, "required", "qwen_xml"), call)
    body = call.removeprefix("<tool_call>\n").removesuffix("\n</tool_call>")
    assert parse_xml_tool_call(body, tool_parameter_schemas(tools)) == ("f", {"tags": tags})


def test_backend_resolves_references_against_the_tool_definitions(xgr: Any) -> None:
    parameters = {
        "type": "object",
        "$defs": {"Base": {"type": "integer"}},
        "properties": {
            "box": {
                "type": "object",
                "$defs": {"Base": {"type": "string"}},
                "properties": {"copy": {"$ref": "#/$defs/Base"}},
                "required": ["copy"],
            }
        },
        "required": ["box"],
    }
    spec = build_strict_tool_grammar([_tool("f", parameters)], "required", "qwen_xml")
    assert _accepts(xgr, spec, _call("f", [("box", '{"copy": 1}')]))
    assert not _accepts(xgr, spec, _call("f", [("box", '{"copy": "bad"}')]))


@pytest.mark.parametrize("schema", _LOOSENED_BY_THE_BACKEND)
@pytest.mark.parametrize("fmt", ["qwen_xml", "hermes_json"])
def test_backend_compiles_schemas_it_only_loosens(xgr: Any, schema: dict[str, Any], fmt: Any) -> None:
    parameters = {
        "type": "object",
        "$defs": {"S": {"type": "string"}, "I": {"type": "integer"}},
        "properties": {"value": schema},
        "required": ["value"],
    }
    spec = build_strict_tool_grammar([_tool("f", parameters)], "required", fmt)
    assert not _accepts(xgr, spec, "plain text")


def test_backend_auto_admits_text_and_required_does_not(xgr: Any) -> None:
    assert _accepts(xgr, build_strict_tool_grammar(TOOLS, "auto", "qwen_xml"), "The answer is 42.")
    assert not _accepts(xgr, build_strict_tool_grammar(TOOLS, "required", "qwen_xml"), "The answer is 42.")


def test_backend_required_admits_parallel_calls_and_a_leading_newline(xgr: Any) -> None:
    spec = build_strict_tool_grammar(TOOLS, "required", "qwen_xml")
    assert _accepts(xgr, spec, "\n" + GOOD_SEARCH + "\n" + GOOD_ANSWER)
    single = build_strict_tool_grammar(TOOLS, "required", "qwen_xml", parallel_tool_calls=False)
    assert not _accepts(xgr, single, GOOD_SEARCH + "\n" + GOOD_ANSWER)


def test_const_equal_to_an_enum_value_is_accepted() -> None:
    parameters = {"type": "object", "properties": {"flag": {"const": True, "enum": [True]}}, "required": ["flag"]}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None


def test_backend_non_strict_body_cannot_smuggle_a_strict_call(xgr: Any) -> None:
    """A free body may not close its call and open a strict one that skips required arguments."""
    tools = [_tool("free", {"type": "object"}, strict=False), _tool("lookup", LOOKUP)]
    missing_limit = _call("lookup", [(key, value) for key, value in LOOKUP_VALUES.items() if key != "limit"])
    smuggled = "<tool_call>\n<function=free>\nx\n</tool_call>\n" + missing_limit + "\n</function>\n</tool_call>"
    spec = build_strict_tool_grammar(tools, "auto", "qwen_xml")
    assert not _accepts(xgr, spec, smuggled)
    assert _accepts(
        xgr, spec, "<tool_call>\n<function=free>\n<parameter=q>\nx\n</parameter>\n</function>\n</tool_call>"
    )


def test_backend_named_rejects_other_tools(xgr: Any) -> None:
    spec = build_strict_tool_grammar(TOOLS, {"type": "function", "function": {"name": "final_answer"}}, "qwen_xml")
    assert _accepts(xgr, spec, GOOD_ANSWER)
    assert not _accepts(xgr, spec, GOOD_SEARCH)


def test_backend_accepts_omitted_optional_parameters(xgr: Any) -> None:
    spec = build_strict_tool_grammar(TOOLS, "required", "qwen_xml")
    assert _accepts(xgr, spec, _call("final_answer", [("parts", "[]")]))
    assert _accepts(xgr, spec, _call("final_answer", [("parts", "[]"), ("mood", "neutral")]))
    assert _accepts(xgr, build_strict_tool_grammar([_tool("ping", None)], "required", "qwen_xml"), _call("ping", []))


def test_backend_hermes_non_strict_tool_keeps_free_arguments(xgr: Any) -> None:
    tools = [_tool("search", SEARCH), _tool("free", {"type": "object"}, strict=False)]
    spec = build_strict_tool_grammar(tools, "auto", "hermes_json")
    free_call = '<tool_call>\n{"name": "free", "arguments": {"query": "x", "n": 2}}\n</tool_call>'
    assert _accepts(xgr, spec, free_call)


def test_hermes_strict_parameters_must_describe_an_object() -> None:
    with pytest.raises(StrictToolError, match="must describe an object"):
        build_strict_tool_grammar([_tool("f", {"type": "array"})], "auto", "hermes_json")


def test_hermes_compiles_the_whole_arguments_schema_without_xml_layout_limits() -> None:
    """Hermes compiles the arguments object as one JSON schema, so a root union is allowed (and validated)."""
    parameters = {"anyOf": [{"type": "object", "properties": {"a": {"type": "integer"}}}]}
    assert build_strict_tool_grammar([_tool("f", parameters)], "auto", "hermes_json") is not None


def test_backend_hermes_json(xgr: Any) -> None:
    spec = build_strict_tool_grammar(TOOLS, "required", "hermes_json")
    good = '{"keywords": ["a"], "exact": true, "filter": {"measures": []}, "limit": 1}'
    bad = '{"keywords": ["a"], "exact": "yes", "filter": "x"}'
    assert _accepts(xgr, spec, '<tool_call>\n{"name": "search", "arguments": ' + good + "}\n</tool_call>")
    assert not _accepts(xgr, spec, '<tool_call>\n{"name": "search", "arguments": ' + bad + "}\n</tool_call>")
