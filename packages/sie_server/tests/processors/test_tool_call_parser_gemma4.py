"""Gemma 4 tool-call parsing: ``<|tool_call>call:NAME{key:value,…}<tool_call|>``."""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator

import pytest
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.processors.tool_call_parser import parse_tool_call_stream

_OPEN = "<|tool_call>"
_CLOSE = "<tool_call|>"
_Q = '<|"|>'
_TAG_FRAGMENTS = ("<|tool_call", "tool_call>", "<tool_call|", "tool_call|>")


async def _chunks(items: list[GenerationChunk]) -> AsyncIterator[GenerationChunk]:
    for item in items:
        yield item


async def _run(parts: list[str], **kwargs: object) -> list[GenerationChunk]:
    kwargs.setdefault("tool_call_format", "gemma4")
    items = [GenerationChunk(text_delta=part) for part in parts]
    items.append(GenerationChunk(text_delta="", done=True, finish_reason="stop"))
    return [item async for item in parse_tool_call_stream(_chunks(items), **kwargs)]  # type: ignore[arg-type]


async def _collect(text: str, **kwargs: object) -> list[GenerationChunk]:
    return await _run([text], **kwargs)


def _text(out: list[GenerationChunk]) -> str:
    return "".join(chunk.text_delta for chunk in out)


def _calls(out: list[GenerationChunk]) -> list[tuple[int, str, object]]:
    """Reassemble ``(index, name, arguments)`` per call from the announce/arguments delta pairs."""
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    for chunk in out:
        delta = chunk.tool_call_delta
        if delta is None:
            continue
        if delta.function_name:
            names[delta.index] = delta.function_name
        arguments[delta.index] = arguments.get(delta.index, "") + delta.arguments_delta
    return [(index, names[index], json.loads(arguments[index])) for index in sorted(names)]


def _deltas(out: list[GenerationChunk]) -> list:
    return [chunk.tool_call_delta for chunk in out if chunk.tool_call_delta is not None]


def _tool(name: str, **properties: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": properties}},
    }


def _call(name: str, body: str) -> str:
    return f"{_OPEN}call:{name}{{{body}}}{_CLOSE}"


def _assert_parse_error(out: list[GenerationChunk]) -> None:
    assert _deltas(out) == []
    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "error"
    assert terminal.error_code == "MODEL_OUTPUT_PARSE_ERROR"


# ── well-formed calls ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_call() -> None:
    out = await _collect(f"{_OPEN}call:get_weather{{city:{_Q}Paris{_Q},unit:{_Q}celsius{_Q}}}{_CLOSE}")

    deltas = _deltas(out)
    assert len(deltas) == 2
    assert deltas[0].index == 0
    assert deltas[0].function_name == "get_weather"
    assert deltas[0].id is not None
    assert deltas[0].id.startswith("call_")
    assert deltas[1].arguments_delta == '{"city":"Paris","unit":"celsius"}'
    assert _text(out) == ""
    assert out[-1].done is True
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_several_calls_in_one_turn_get_distinct_indices() -> None:
    out = await _collect(
        _call("get_weather", f"city:{_Q}Paris{_Q}") + _call("get_time", f"zone:{_Q}CET{_Q}") + _call("ping", "")
    )

    assert _calls(out) == [
        (0, "get_weather", {"city": "Paris"}),
        (1, "get_time", {"zone": "CET"}),
        (2, "ping", {}),
    ]
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_text_before_a_call_is_preserved() -> None:
    out = await _collect("Let me check the weather. " + _call("get_weather", f"city:{_Q}Oslo{_Q}"))

    assert out[0].text_delta == "Let me check the weather. "
    assert _text(out) == "Let me check the weather. "
    assert _calls(out) == [(0, "get_weather", {"city": "Oslo"})]


@pytest.mark.asyncio
async def test_call_without_arguments() -> None:
    out = await _collect(f"{_OPEN}call:ping{{}}{_CLOSE}")

    assert _calls(out) == [(0, "ping", {})]
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_nested_values_keep_native_types() -> None:
    body = (
        "a:{},b:[],c:[{x:1},{y:[2,3]}],d:{e:[{f:{g:null}}]},"
        "i:42,j:-7,k:0,l:3.25,m:-0.5,n:1e3,o:2.5E-2,p:-1e+2,"
        "t:true,u:false,v:null,w:[true,false,null,0,-1.5]"
    )
    out = await _collect(_call("f", body))

    [(_, _, arguments)] = _calls(out)
    assert arguments == {
        "a": {},
        "b": [],
        "c": [{"x": 1}, {"y": [2, 3]}],
        "d": {"e": [{"f": {"g": None}}]},
        "i": 42,
        "j": -7,
        "k": 0,
        "l": 3.25,
        "m": -0.5,
        "n": 1000.0,
        "o": 0.025,
        "p": -100.0,
        "t": True,
        "u": False,
        "v": None,
        "w": [True, False, None, 0, -1.5],
    }
    assert isinstance(arguments["i"], int)
    assert isinstance(arguments["n"], float)


@pytest.mark.asyncio
async def test_strings_keep_structural_text_exactly() -> None:
    tricky = "{a}[b],c:\"d\"'e'\nline two ü 日本 </tool_call><tool_call>call:x{y:1}  "
    out = await _collect(_call("f", f"s:{_Q}{tricky}{_Q},n:[{_Q}{tricky}{_Q}],o:{{k:{_Q}{tricky}{_Q}}}"))

    assert _calls(out) == [(0, "f", {"s": tricky, "n": [tricky], "o": {"k": tricky}})]
    assert _text(out) == ""


@pytest.mark.asyncio
async def test_empty_string_value() -> None:
    out = await _collect(_call("f", f"s:{_Q}{_Q}"))

    assert _calls(out) == [(0, "f", {"s": ""})]


@pytest.mark.asyncio
async def test_quoted_keys() -> None:
    out = await _collect(_call("f", f"{_Q}my key{_Q}:1,{_Q}a,b:c{_Q}:{_Q}x{_Q},o:{{{_Q}in ner{_Q}:true}}"))

    assert _calls(out) == [(0, "f", {"my key": 1, "a,b:c": "x", "o": {"in ner": True}})]


@pytest.mark.asyncio
async def test_whitespace_between_tokens_is_ignored() -> None:
    out = await _collect(
        f"{_OPEN}\n call:f {{ a : 1 ,\n b : [ 1 , 2 ] , c : {{ d : {_Q} s {_Q} }} , e : [ ] }} \n{_CLOSE}"
    )

    assert _calls(out) == [(0, "f", {"a": 1, "b": [1, 2], "c": {"d": " s "}, "e": []})]


# ── stream splitting ───────────────────────────────────────────────

_SPLIT_SAMPLE = (
    "Let me check. "
    + _call("get_weather", f'city:{_Q}Paris, "FR"{_Q},days:[1,2,{{x:-1.5e3,y:[]}}],opts:{{metric:true,tz:null}}')
    + _call("get_time", f"zone:{_Q}Europe/Paris{_Q}")
    + " Done."
)
_SPLIT_TEXT = "Let me check.  Done."
_SPLIT_CALLS = [
    (
        0,
        "get_weather",
        {"city": 'Paris, "FR"', "days": [1, 2, {"x": -1500.0, "y": []}], "opts": {"metric": True, "tz": None}},
    ),
    (1, "get_time", {"zone": "Europe/Paris"}),
]


def _assert_no_tag_leak(out: list[GenerationChunk]) -> None:
    for chunk in out:
        for fragment in _TAG_FRAGMENTS:
            assert fragment not in chunk.text_delta


@pytest.mark.asyncio
async def test_unsplit_sample() -> None:
    out = await _collect(_SPLIT_SAMPLE)

    assert _text(out) == _SPLIT_TEXT
    assert _calls(out) == _SPLIT_CALLS
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_every_two_chunk_split_matches_the_unsplit_parse() -> None:
    for cut in range(len(_SPLIT_SAMPLE) + 1):
        out = await _run([_SPLIT_SAMPLE[:cut], _SPLIT_SAMPLE[cut:]])
        assert _text(out) == _SPLIT_TEXT, cut
        assert _calls(out) == _SPLIT_CALLS, cut
        assert out[-1].finish_reason == "tool_calls", cut
        _assert_no_tag_leak(out)


@pytest.mark.asyncio
async def test_one_character_per_chunk_matches_the_unsplit_parse() -> None:
    out = await _run(list(_SPLIT_SAMPLE))

    assert _text(out) == _SPLIT_TEXT
    assert _calls(out) == _SPLIT_CALLS
    assert out[-1].finish_reason == "tool_calls"
    _assert_no_tag_leak(out)


# ── schema-directed conversion of top-level arguments ──────────────

_LOOKUP_TOOL = _tool(
    "lookup",
    zip={"type": "string"},
    code={"type": "string"},
    flag={"type": "string"},
    count={"type": "integer"},
    ratio={"type": "number"},
    tags={"type": "array"},
    nested={"type": "object"},
    note={"type": ["string", "null"]},
)


@pytest.mark.asyncio
async def test_top_level_arguments_follow_declared_types() -> None:
    out = await _collect(
        _call(
            "lookup",
            f"zip:02134,code:-12.50,flag:true,count:{_Q}5{_Q},ratio:{_Q}0.5{_Q},"
            f"tags:{_Q}[1, 2]{_Q},extra:{_Q}7{_Q},nested:{{n:{_Q}5{_Q},m:7}},note:null",
        ),
        tools=[_LOOKUP_TOOL],
    )

    assert _calls(out) == [
        (
            0,
            "lookup",
            {
                "zip": "02134",
                "code": "-12.50",
                "flag": "true",
                "count": 5,
                "ratio": 0.5,
                "tags": [1, 2],
                "extra": "7",
                "nested": {"n": "5", "m": 7},
                "note": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_unconvertible_quoted_value_stays_a_string() -> None:
    out = await _collect(_call("lookup", f"count:{_Q}many{_Q},zip:{_Q}02134{_Q}"), tools=[_LOOKUP_TOOL])

    assert _calls(out) == [(0, "lookup", {"count": "many", "zip": "02134"})]


@pytest.mark.asyncio
async def test_quoted_null_for_a_nullable_string_parameter_is_the_word() -> None:
    out = await _collect(_call("lookup", f"note:{_Q}null{_Q},zip:{_Q}null{_Q}"), tools=[_LOOKUP_TOOL])

    assert _calls(out) == [(0, "lookup", {"note": "null", "zip": "null"})]


@pytest.mark.asyncio
async def test_native_values_for_non_string_parameters_are_kept() -> None:
    out = await _collect(_call("lookup", "count:5,ratio:2,tags:[1,{a:2}],nested:{a:[]}"), tools=[_LOOKUP_TOOL])

    assert _calls(out) == [(0, "lookup", {"count": 5, "ratio": 2, "tags": [1, {"a": 2}], "nested": {"a": []}})]


@pytest.mark.asyncio
async def test_string_parameter_still_takes_a_structured_value_natively() -> None:
    out = await _collect(_call("lookup", "zip:{a:1},code:[1]"), tools=[_LOOKUP_TOOL])

    assert _calls(out) == [(0, "lookup", {"zip": {"a": 1}, "code": [1]})]


@pytest.mark.asyncio
async def test_undeclared_tool_keeps_native_values() -> None:
    out = await _collect(_call("other", f"zip:02,count:{_Q}5{_Q}"), tools=[_LOOKUP_TOOL])

    _assert_parse_error(out)  # ``02`` is not a number and no string type applies
    out = await _collect(_call("other", f"zip:2,count:{_Q}5{_Q}"), tools=[_LOOKUP_TOOL])
    assert _calls(out) == [(0, "other", {"zip": 2, "count": "5"})]


@pytest.mark.asyncio
async def test_bare_word_for_a_string_parameter_must_still_be_one_token() -> None:
    out = await _collect(_call("lookup", "zip:two words"), tools=[_LOOKUP_TOOL])

    _assert_parse_error(out)


# ── malformed blocks ───────────────────────────────────────────────

_DEEP_OK = "[" * 31 + "]" * 31
_DEEP_BAD = "[" * 32 + "]" * 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(f"get_weather{{city:{_Q}Paris{_Q}}}", id="missing-call-prefix"),
        pytest.param("call:{}", id="empty-name"),
        pytest.param("call:get weather{}", id="name-with-space"),
        pytest.param("call:get_weather", id="no-argument-object"),
        pytest.param(f"call:f{{a {_Q}x{_Q}}}", id="missing-colon"),
        pytest.param(f"call:f{{a:{_Q}x{_Q} b:1}}", id="missing-comma"),
        pytest.param("call:get_weather{unit:celsius}", id="bare-word"),
        pytest.param("call:f{a:1,}", id="trailing-comma"),
        pytest.param("call:f{,a:1}", id="leading-comma"),
        pytest.param("call:f{a:[1,]}", id="trailing-comma-in-array"),
        pytest.param("call:f{a:[1 2]}", id="missing-comma-in-array"),
        pytest.param("call:f{a:{b:1}", id="unbalanced-brace"),
        pytest.param("call:f{a:[1}", id="unbalanced-bracket"),
        pytest.param("call:f{a:1}}", id="extra-closing-brace"),
        pytest.param("call:f{a:1} trailing", id="text-after-object"),
        pytest.param("call:f{a:}", id="missing-value"),
        pytest.param("call:f{a:1,a:2}", id="duplicate-key"),
        pytest.param("call:f{o:{a:1,a:2}}", id="duplicate-nested-key"),
        pytest.param(f"call:f{{a:{_Q}abc}}", id="unterminated-string"),
        pytest.param(f"call:f{{{_Q}k:1}}", id="unterminated-quoted-key"),
        pytest.param("call:f{a:1.}", id="number-trailing-dot"),
        pytest.param("call:f{a:.5}", id="number-leading-dot"),
        pytest.param("call:f{a:01}", id="number-leading-zero"),
        pytest.param("call:f{a:+1}", id="number-plus-sign"),
        pytest.param("call:f{a:-}", id="lone-minus"),
        pytest.param("call:f{a:NaN}", id="nan"),
        pytest.param("call:f{a:Infinity}", id="infinity"),
        pytest.param("call:f{a:1e999}", id="non-finite-number"),
        pytest.param("call:f{a:True}", id="capitalised-true"),
        pytest.param("call:f{a:'x'}", id="single-quoted"),
        pytest.param('call:f{a:"x"}', id="json-quoted"),
        pytest.param(f"call:f{{a:{_DEEP_BAD}}}", id="too-deep"),
        pytest.param("call:f{" + ",".join(f"k{i}:{i}" for i in range(257)) + "}", id="too-many-arguments"),
    ],
)
async def test_malformed_block_is_a_terminal_parse_error(body: str) -> None:
    out = await _collect(f"{_OPEN}{body}{_CLOSE}")

    _assert_parse_error(out)
    assert (out[-1].error_message or "").startswith("malformed Gemma tool-call:")


@pytest.mark.asyncio
async def test_depth_at_the_bound_is_accepted() -> None:
    out = await _collect(_call("f", f"a:{_DEEP_OK}"))

    [(_, _, arguments)] = _calls(out)
    value = arguments["a"]
    for _ in range(30):
        value = value[0]
    assert value == []


@pytest.mark.asyncio
async def test_argument_count_at_the_bound_is_accepted() -> None:
    out = await _collect(_call("f", ",".join(f"k{i}:{i}" for i in range(256))))

    [(_, _, arguments)] = _calls(out)
    assert len(arguments) == 256


@pytest.mark.asyncio
async def test_huge_integer_is_a_parse_error_not_a_crash() -> None:
    out = await _collect(_call("f", "a:" + "9" * 5000))

    _assert_parse_error(out)


@pytest.mark.asyncio
async def test_unterminated_block_is_a_parse_error() -> None:
    out = await _collect(f"Sure. {_OPEN}call:f{{a:1}}")

    _assert_parse_error(out)
    assert out[-1].error_message == "unterminated <tool_call> block"


@pytest.mark.asyncio
async def test_unterminated_block_over_the_buffer_cap_is_a_parse_error() -> None:
    out = await _run([f"{_OPEN}call:f{{a:{_Q}", "x" * (256 * 1024 + 16)])

    _assert_parse_error(out)
    assert "exceeded" in (out[-1].error_message or "")


@pytest.mark.asyncio
async def test_parse_error_after_a_good_call_stops_the_stream() -> None:
    out = await _collect(_call("a", "x:1") + _call("b", "y:oops"))

    assert _calls(out) == [(0, "a", {"x": 1})]
    assert out[-1].error_code == "MODEL_OUTPUT_PARSE_ERROR"
    assert out[-1].finish_reason == "error"


# ── parallel_tool_calls and format isolation ───────────────────────


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_rejects_a_second_block() -> None:
    out = await _collect(_call("a", "x:1") + _call("b", "y:2"), parallel_tool_calls=False)

    assert {d.index for d in _deltas(out)} == {0}
    assert out[-1].done is True
    assert out[-1].finish_reason == "error"
    assert out[-1].error_code == "parallel_tool_calls_violated"


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_allows_one_block() -> None:
    out = await _collect(_call("a", "x:1"), parallel_tool_calls=False)

    assert _calls(out) == [(0, "a", {"x": 1})]
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_qwen_style_block_is_plain_text_under_gemma4() -> None:
    text = 'Use <tool_call>{"name":"f","arguments":{}}</tool_call> here.'
    out = await _collect(text)

    assert _deltas(out) == []
    assert _text(out) == text
    assert out[-1].finish_reason == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_call_format", ["qwen_xml", "hermes_json", "auto"])
async def test_gemma_block_is_plain_text_under_other_formats(tool_call_format: str) -> None:
    text = "Use " + _call("f", "x:1") + " here."
    out = await _collect(text, tool_call_format=tool_call_format)

    assert _deltas(out) == []
    assert _text(out) == text
    assert out[-1].finish_reason == "stop"


# ── non-streaming n>1 candidates ───────────────────────────────────


@pytest.mark.asyncio
async def test_candidates_carry_their_own_gemma_calls() -> None:
    malformed = f"{_OPEN}call:f{{a:oops}}{_CLOSE}"
    candidates = (
        {"text": "Hi " + _call("get_weather", f"city:{_Q}Tokyo{_Q}") + _call("ping", ""), "finish_reason": "stop"},
        {"text": "plain answer", "finish_reason": "stop"},
        {"text": "Bad " + malformed + " tail", "finish_reason": "stop"},
    )
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks([GenerationChunk(text_delta="", done=True, finish_reason="stop", candidates=candidates)]),
            tool_call_format="gemma4",
        )
    ]

    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "tool_calls"
    first, second, third = terminal.candidates or ()
    assert first["text"] == "Hi "
    assert first["finish_reason"] == "tool_calls"
    assert [call["function"]["name"] for call in first["tool_calls"]] == ["get_weather", "ping"]
    assert json.loads(first["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert first["tool_calls"][0]["id"].startswith("call_")
    assert second["text"] == "plain answer"
    assert "tool_calls" not in second
    assert third["text"] == "Bad " + malformed + " tail"
    assert "tool_calls" not in third
    assert third["finish_reason"] == "stop"


# ── round trip through the chat template's rendering ───────────────


def _render(value: object) -> str:
    """Mirror the chat template's ``format_argument(value, escape_keys=False)``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"{_Q}{value}{_Q}"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{_render(item)}" for key, item in sorted(value.items())) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_render(item) for item in value) + "]"
    return str(value)


def _render_call(name: str, arguments: dict[str, object]) -> str:
    # The template renders the top-level arguments with the same macro as nested objects.
    return f"{_OPEN}call:{name}{_render(arguments)}{_CLOSE}"


_STRING_ALPHABET = "ab xyz{}[],:\"'\n\t<>|/-_.ü日本0123456789"


def _random_value(rng: random.Random, depth: int) -> object:
    kind = rng.choice(["str", "int", "float", "bool", "null", "list", "dict"] if depth < 4 else ["str", "int", "float"])
    if kind == "str":
        return "".join(rng.choice(_STRING_ALPHABET) for _ in range(rng.randint(0, 12)))
    if kind == "int":
        return rng.randint(-(10**12), 10**12)
    if kind == "float":
        return rng.choice([0.5, -2.25, 1e-07, 3.0, 1.5e20, -0.0, rng.uniform(-1e6, 1e6)])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "null":
        return None
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return _random_object(rng, depth + 1)


def _random_object(rng: random.Random, depth: int) -> dict[str, object]:
    keys = {f"{rng.choice('abcxyz')}{rng.choice(['', '_1', 'Key', '.v', '-w'])}" for _ in range(rng.randint(0, 5))}
    return {key: _random_value(rng, depth) for key in keys}


_ROUND_TRIP_CASES: list[dict[str, object]] = [
    {},
    {"city": "Paris", "unit": "celsius"},
    {"query": {"filters": [{"field": "age", "gte": 18}, {"field": "tags", "in": ["a", "b"]}], "limit": 10}},
    {"matrix": [[1, 2], [3, 4], []], "empty": {}, "flag": False, "none": None, "pi": 3.14159},
    {"text": "line one\nline two: {braces}, [brackets], \"quotes\", 'single' </tool_call>"},
]


@pytest.mark.asyncio
async def test_round_trip_of_template_rendered_arguments() -> None:
    rng = random.Random(4)  # noqa: S311 - deterministic test data, not security
    cases = [*_ROUND_TRIP_CASES, *(_random_object(rng, 0) for _ in range(200))]
    for arguments in cases:
        out = await _collect(_render_call("fn", arguments))
        assert out[-1].error_code is None, (arguments, out[-1].error_message)
        assert _calls(out) == [(0, "fn", arguments)], arguments
