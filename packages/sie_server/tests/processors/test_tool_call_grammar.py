from __future__ import annotations

import re

import pytest
from sie_server.processors.tool_call_grammar import (
    ToolChoiceError,
    build_tool_choice_grammar,
    extract_tool_names,
    normalize_tool_choice,
)

_TOOLS = (
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {}}},
)


# ── normalize_tool_choice ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ("auto", None)),
        ("auto", ("auto", None)),
        ("none", ("none", None)),
        ("required", ("required", None)),
        ("garbage", ("auto", None)),
        ({"type": "function", "function": {"name": "get_weather"}}, ("named", "get_weather")),
        ({"type": "function", "function": {}}, ("auto", None)),
        ({"nonsense": 1}, ("auto", None)),
    ],
)
def test_normalize_tool_choice(value: object, expected: tuple[str, str | None]) -> None:
    assert normalize_tool_choice(value) == expected  # type: ignore[arg-type]


def test_extract_tool_names() -> None:
    assert extract_tool_names(_TOOLS) == ["get_weather", "get_time"]
    assert extract_tool_names(None) == []
    assert extract_tool_names(({"type": "function"},)) == []


# ── build_tool_choice_grammar: no-forcing cases ────────────────────


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_no_grammar_for_non_forcing_choices(choice: object) -> None:
    assert build_tool_choice_grammar(_TOOLS, choice, "qwen_xml") is None  # type: ignore[arg-type]


# ── build_tool_choice_grammar: error cases ─────────────────────────


def test_required_without_tools_raises() -> None:
    with pytest.raises(ToolChoiceError):
        build_tool_choice_grammar(None, "required", "qwen_xml")


def test_named_function_not_in_tools_raises() -> None:
    with pytest.raises(ToolChoiceError):
        build_tool_choice_grammar(_TOOLS, {"type": "function", "function": {"name": "missing"}}, "qwen_xml")


# ── build_tool_choice_grammar: regex shape (qwen_xml) ──────────────


def test_required_qwen_xml_regex_matches_any_tool() -> None:
    spec = build_tool_choice_grammar(_TOOLS, "required", "qwen_xml")
    assert spec is not None
    assert spec.kind == "regex"
    pat = re.compile(spec.value)  # value is a valid regex
    weather = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
    time = "<tool_call><function=get_time></function></tool_call>"
    assert pat.fullmatch(weather)
    assert pat.fullmatch(time)
    # A different (unlisted) function must NOT satisfy the constraint.
    assert not pat.fullmatch("<tool_call><function=delete_everything></function></tool_call>")
    # Plain prose must not match.
    assert not pat.fullmatch("I cannot help with that.")


def test_named_qwen_xml_regex_pins_single_function() -> None:
    spec = build_tool_choice_grammar(_TOOLS, {"type": "function", "function": {"name": "get_weather"}}, "qwen_xml")
    assert spec is not None
    pat = re.compile(spec.value)
    assert pat.fullmatch("<tool_call><function=get_weather></function></tool_call>")
    # The other declared tool is not allowed for a named choice.
    assert not pat.fullmatch("<tool_call><function=get_time></function></tool_call>")


def test_required_qwen_xml_regex_allows_multiple_calls() -> None:
    spec = build_tool_choice_grammar(_TOOLS, "required", "qwen_xml")
    assert spec is not None
    pat = re.compile(spec.value)
    two = (
        "<tool_call><function=get_weather></function></tool_call><tool_call><function=get_time></function></tool_call>"
    )
    assert pat.fullmatch(two)


# ── build_tool_choice_grammar: regex shape (hermes_json) ───────────


def test_required_hermes_json_regex() -> None:
    spec = build_tool_choice_grammar(_TOOLS, "required", "hermes_json")
    assert spec is not None
    assert spec.kind == "regex"
    pat = re.compile(spec.value)
    block = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    assert pat.fullmatch(block)
    bad = '<tool_call>{"name": "nope", "arguments": {}}</tool_call>'
    assert not pat.fullmatch(bad)


# ── build_tool_choice_grammar: regex shape (glm_xml) ───────────────


def test_required_glm_xml_regex_matches_any_tool() -> None:
    spec = build_tool_choice_grammar(_TOOLS, "required", "glm_xml")
    assert spec is not None
    assert spec.kind == "regex"
    pat = re.compile(spec.value)
    weather = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>"
    spaced = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Tokyo</arg_value>\n</tool_call>"
    assert pat.fullmatch(weather)
    assert pat.fullmatch(spaced)
    assert pat.fullmatch("<tool_call>get_time</tool_call>")
    assert pat.fullmatch(weather + "\n<tool_call>get_time</tool_call>")
    assert not pat.fullmatch("<tool_call>delete_everything</tool_call>")
    assert not pat.fullmatch("I cannot help with that.")


def test_glm_xml_regex_accepts_only_complete_argument_pairs() -> None:
    from sie_server.processors.tool_call_parser import _parse_glm_tool_call

    spec = build_tool_choice_grammar(_TOOLS, "required", "glm_xml")
    assert spec is not None
    pat = re.compile(spec.value)
    two = (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
        "\n<arg_key>days</arg_key>\n<arg_value>3</arg_value></tool_call>"
    )
    assert pat.fullmatch(two)
    assert _parse_glm_tool_call(two.removeprefix("<tool_call>").removesuffix("</tool_call>")) == (
        "get_weather",
        {"city": "Tokyo", "days": 3},
    )
    for dangling in (
        "<tool_call>get_weather<arg_key>city</tool_call>",
        "<tool_call>get_weather<arg_key>city</arg_key></tool_call>",
        "<tool_call>get_weather<arg_key>ci<ty</arg_key><arg_value>x</arg_value></tool_call>",
        "<tool_call>get_weather<arg_value>x</arg_value></tool_call>",
    ):
        assert not pat.fullmatch(dangling), dangling


def test_named_glm_xml_regex_pins_the_whole_name() -> None:
    tools = (
        {"type": "function", "function": {"name": "get"}},
        {"type": "function", "function": {"name": "get_weather"}},
    )
    spec = build_tool_choice_grammar(tools, {"type": "function", "function": {"name": "get"}}, "glm_xml")
    assert spec is not None
    pat = re.compile(spec.value)
    assert pat.fullmatch("<tool_call>get<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>")
    assert not pat.fullmatch("<tool_call>get_weather<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>")
    assert not pat.fullmatch("<tool_call>get_weather</tool_call>")


def test_auto_format_refuses_to_force_instead_of_guessing_qwen_xml() -> None:
    # Regression (fix #7): when the format is unresolved ("auto"), forcing a
    # tool_choice must NOT silently assume Qwen XML — a non-Qwen model would
    # be constrained to emit XML it never learned. Refuse instead.
    with pytest.raises(ToolChoiceError, match="could not be resolved"):
        build_tool_choice_grammar(_TOOLS, "required", "auto")
    with pytest.raises(ToolChoiceError, match="could not be resolved"):
        build_tool_choice_grammar(_TOOLS, {"type": "function", "function": {"name": "get_weather"}}, "auto")


def test_regex_escapes_special_chars_in_names() -> None:
    tools = ({"type": "function", "function": {"name": "a.b+c"}},)
    spec = build_tool_choice_grammar(tools, {"type": "function", "function": {"name": "a.b+c"}}, "qwen_xml")
    assert spec is not None
    pat = re.compile(spec.value)
    assert pat.fullmatch("<tool_call><function=a.b+c></function></tool_call>")
    # The unescaped regex meaning (a or b, etc.) must not leak through.
    assert not pat.fullmatch("<tool_call><function=axbxc></function></tool_call>")
