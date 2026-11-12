"""Forced ``tool_choice`` grammar for the Gemma 4 tool-call format."""

from __future__ import annotations

import re

import pytest
from sie_server.processors.tool_call_grammar import build_tool_choice_grammar
from sie_server.processors.tool_call_parser import _parse_candidate_text

_TOOLS = (
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {}}},
)
_Q = '<|"|>'
_WEATHER = f"<|tool_call>call:get_weather{{city:{_Q}Paris{_Q},days:[1,{{x:null}}]}}<tool_call|>"
_TIME = f"<|tool_call>call:get_time{{zone:{_Q}CET{_Q}}}<tool_call|>"
_PING_WEATHER = "<|tool_call>call:get_weather{}<tool_call|>"


def _pattern(tool_choice: object, tools: tuple = _TOOLS) -> str:
    spec = build_tool_choice_grammar(tools, tool_choice, "gemma4")  # type: ignore[arg-type]
    assert spec is not None
    assert spec.kind == "regex"
    return spec.value


def _accepts(pattern: str, text: str) -> bool:
    accepted = re.fullmatch(pattern, text) is not None
    if accepted:
        # Everything the forcing grammar admits in these tests is a well-formed call.
        remaining, calls = _parse_candidate_text(text, "gemma4")
        assert remaining == ""
        assert calls
    return accepted


@pytest.mark.parametrize(
    "tool_choice",
    ["required", {"type": "function", "function": {"name": "get_weather"}}],
)
def test_forcing_returns_a_regex(tool_choice: object) -> None:
    spec = build_tool_choice_grammar(_TOOLS, tool_choice, "gemma4")  # type: ignore[arg-type]

    assert spec is not None
    assert spec.kind == "regex"
    assert spec.label.startswith("tool_choice:")


@pytest.mark.parametrize("tool_choice", ["auto", "none", None])
def test_no_forcing_for_auto_or_none(tool_choice: object) -> None:
    assert build_tool_choice_grammar(_TOOLS, tool_choice, "gemma4") is None  # type: ignore[arg-type]


def test_required_accepts_any_allowed_call_and_consecutive_calls() -> None:
    pattern = _pattern("required")

    assert _accepts(pattern, _WEATHER)
    assert _accepts(pattern, _TIME)
    assert _accepts(pattern, _WEATHER + _TIME)
    assert _accepts(pattern, _PING_WEATHER)


def test_named_pins_the_function() -> None:
    pattern = _pattern({"type": "function", "function": {"name": "get_weather"}})

    assert _accepts(pattern, _WEATHER)
    assert _accepts(pattern, _WEATHER + _WEATHER)
    assert not _accepts(pattern, _TIME)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("<|tool_call>call:unknown{}<tool_call|>", id="unknown-name"),
        pytest.param("<|tool_call>call:get_weather2{}<tool_call|>", id="extended-name"),
        pytest.param("<|tool_call>call:get_weathe{}<tool_call|>", id="prefix-name"),
        pytest.param(" " + _WEATHER, id="leading-space"),
        pytest.param("\n" + _WEATHER, id="leading-newline"),
        pytest.param("Sure. " + _WEATHER, id="leading-prose"),
        pytest.param(_WEATHER + " ", id="trailing-space"),
        pytest.param('<tool_call>{"name":"get_weather","arguments":{}}</tool_call>', id="qwen-style"),
        pytest.param("<|tool_call>get_weather{}<tool_call|>", id="missing-call-prefix"),
        pytest.param("", id="empty"),
    ],
)
def test_required_rejects(text: str) -> None:
    assert not _accepts(_pattern("required"), text)


# Raw outputs a Gemma 4 model produced on GPU, with and without forcing.
_OBSERVED = [
    f"<|tool_call>call:get_weather{{city:{_Q}Paris{_Q},unit:{_Q}celsius{_Q}}}<tool_call|>",
    f"<|tool_call>call:get_time{{timezone:{_Q}America/New_York{_Q}}}<tool_call|>",
    f"<|tool_call>call:get_weather{{city:{_Q}Paris{_Q},unit:{_Q}celsius{_Q}}}<tool_call|>"
    f"<|tool_call>call:get_weather{{city:{_Q}Tokyo{_Q},unit:{_Q}celsius{_Q}}}<tool_call|>",
    f"<|tool_call>call:get_weather{{all_day:false,attendees:[{{name:{_Q}Ana{_Q}}},{{name:{_Q}Ben{_Q}}}],"
    f"duration_minutes:45,location:{{floor:3,room:{_Q}Blue{_Q}}},"
    f'notes:{_Q}He said "ship it", then left.{_Q},title:{_Q}Q4 review: goals, risks {{and}} budget{_Q}}}<tool_call|>',
]


@pytest.mark.parametrize("text", _OBSERVED)
def test_required_accepts_observed_model_output(text: str) -> None:
    assert _accepts(_pattern("required"), text)


@pytest.mark.parametrize(
    "tail",
    [
        pytest.param("</tool_call>\nWhy don't scientists trust atoms?", id="wrong-close-tag-then-prose"),
        pytest.param("\nWhy don't scientists trust atoms?", id="prose-after-arguments"),
        pytest.param(" ", id="space-after-arguments"),
        pytest.param("<tool_call>", id="other-tag"),
    ],
)
def test_nothing_but_the_close_tag_can_follow_the_arguments(tail: str) -> None:
    """A free body let a forced model write prose after ``}`` up to the token limit."""
    pattern = re.compile(_pattern("required"))
    prefix = f"<|tool_call>call:get_weather{{city:{_Q}London{_Q}}}"

    assert pattern.fullmatch(prefix + "<tool_call|>")
    # No continuation of the rejected tail can ever complete a match.
    for suffix in ("", "<tool_call|>", "}<tool_call|>", f"{_Q}}}<tool_call|>", f"{_Q}x{_Q}}}<tool_call|>"):
        assert pattern.fullmatch(prefix + tail + suffix) is None


def test_a_second_block_is_pinned_to_an_allowed_name() -> None:
    pattern = _pattern({"type": "function", "function": {"name": "get_weather"}})

    assert not _accepts(pattern, _WEATHER + _TIME)
    assert not _accepts(pattern, _WEATHER + "<|tool_call>call:unknown{}<tool_call|>")


@pytest.mark.parametrize(
    "value",
    ["a < b", "x<y>z", "<<a", "tag </tool_call> text", '"quoted"', "line\nbreak", "{[,:]}", ""],
)
def test_strings_stay_free_inside_the_delimiters(value: str) -> None:
    assert _accepts(_pattern("required"), f"<|tool_call>call:get_weather{{note:{_Q}{value}{_Q}}}<tool_call|>")


@pytest.mark.parametrize("value", ["a <| b", "ends with <"])
def test_strings_that_could_hide_a_delimiter_are_not_generated(value: str) -> None:
    assert not _accepts(_pattern("required"), f"<|tool_call>call:get_weather{{note:{_Q}{value}{_Q}}}<tool_call|>")


def test_names_with_regex_metacharacters_are_escaped() -> None:
    tools = ({"type": "function", "function": {"name": "a.b+c", "parameters": {}}},)
    pattern = _pattern("required", tools)

    assert _accepts(pattern, "<|tool_call>call:a.b+c{}<tool_call|>")
    assert not _accepts(pattern, "<|tool_call>call:axbbc{}<tool_call|>")
