from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.processors.tool_call_parser import parse_tool_call_stream


async def _chunks(items: list[GenerationChunk]) -> AsyncIterator[GenerationChunk]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_parse_tool_call_stream_emits_text_and_tool_delta() -> None:
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(
                        text_delta='Use this: <tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>'
                    ),
                    GenerationChunk(
                        text_delta="", done=True, finish_reason="stop", prompt_tokens=5, completion_tokens=7
                    ),
                ]
            )
        )
    ]

    assert out[0].text_delta == "Use this: "
    assert out[1].tool_call_delta is not None
    assert out[1].tool_call_delta.index == 0
    assert out[1].tool_call_delta.id is not None
    assert out[1].tool_call_delta.function_name == "get_weather"
    assert out[2].tool_call_delta is not None
    assert out[2].tool_call_delta.arguments_delta == '{"city":"Paris"}'
    assert out[-1].done is True
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_handles_split_open_tag() -> None:
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta="<tool"),
                    GenerationChunk(text_delta='_call>{"name":"x","arguments":{}}</tool_call>'),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]

    assert any(item.tool_call_delta is not None for item in out)
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_handles_split_close_tag() -> None:
    """BUG 1 regression: a ``</tool_call>`` split across a chunk boundary
    must still close the block and emit a tool_call_delta, not a parse error.
    """
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta='<tool_call>{"name":"f","arguments":{}}</tool_'),
                    GenerationChunk(text_delta="call>"),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]

    deltas = [c.tool_call_delta for c in out if c.tool_call_delta is not None]
    assert deltas, "expected a tool_call_delta, got none"
    assert deltas[0].function_name == "f"
    # No parse error emitted.
    assert all(c.error_code is None for c in out)
    assert out[-1].done is True
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_handles_close_tag_split_mid_tag() -> None:
    """BUG 1 regression: close tag split in the MIDDLE of a longer split.

    Boundary lands several characters into ``</tool_call>`` so the retained
    tail must be re-scanned across more than one chunk.
    """
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta='<tool_call>{"name":"f","arguments":{}}</to'),
                    GenerationChunk(text_delta="ol_ca"),
                    GenerationChunk(text_delta="ll>"),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]

    deltas = [c.tool_call_delta for c in out if c.tool_call_delta is not None]
    assert deltas, "expected a tool_call_delta, got none"
    assert deltas[0].function_name == "f"
    assert all(c.error_code is None for c in out)
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_args_value_with_literal_close_substring() -> None:
    """Contract pin: a tool args value containing a literal ``</tool_call>``
    substring inside a JSON string still parses; the FIRST real close tag
    terminates the block (the literal is JSON-escaped on the wire).
    """
    # The literal ``</tool_call>`` appears as escaped text inside a JSON
    # string value; the actual closing tag is the trailing one.
    raw = '<tool_call>{"name":"f","arguments":{"q":"a<\\/tool_call>b"}}</tool_call>'
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta=raw),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]
    deltas = [c.tool_call_delta for c in out if c.tool_call_delta is not None]
    assert deltas, "expected a tool_call_delta, got none"
    assert deltas[0].function_name == "f"
    args = "".join(d.arguments_delta for d in deltas if d.arguments_delta)
    import json as _json

    assert _json.loads(args) == {"q": "a</tool_call>b"}
    assert all(c.error_code is None for c in out)
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_malformed_json_is_terminal_error() -> None:
    out = [
        item
        async for item in parse_tool_call_stream(_chunks([GenerationChunk(text_delta="<tool_call>{bad}</tool_call>")]))
    ]

    assert len(out) == 1
    assert out[0].done is True
    assert out[0].finish_reason == "error"
    assert out[0].error_code == "MODEL_OUTPUT_PARSE_ERROR"
    assert "malformed" in (out[0].error_message or "")


@pytest.mark.asyncio
async def test_parse_tool_call_stream_passthrough_when_no_tool_call() -> None:
    """Plain prose flows through untouched, finish_reason stays 'stop'."""
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta="Hello "),
                    GenerationChunk(text_delta="world!"),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]
    # First two chunks may be coalesced by the parser's flush-on-lookahead,
    # but the concatenated text_delta across non-terminal chunks must
    # equal the input, and the terminal must keep finish_reason="stop"
    # (no tool calls observed).
    text = "".join(c.text_delta for c in out if not c.done)
    assert text == "Hello world!"
    assert out[-1].done is True
    assert out[-1].finish_reason == "stop"
    assert all(c.tool_call_delta is None for c in out)


@pytest.mark.asyncio
async def test_parse_tool_call_stream_parallel_calls_use_distinct_indices() -> None:
    """Two adjacent <tool_call> blocks produce ToolCallDeltas with index 0 and 1."""
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(
                        text_delta=(
                            '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
                            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>'
                        )
                    ),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]
    deltas = [c.tool_call_delta for c in out if c.tool_call_delta is not None]
    assert len(deltas) == 4  # 2 calls × (announcement + arguments)
    indices = sorted({d.index for d in deltas})
    assert indices == [0, 1]
    names = {d.function_name for d in deltas if d.function_name is not None}
    assert names == {"a", "b"}
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_terminal_propagates_usage_and_tool_calls_reason() -> None:
    """Terminal chunk after a tool call keeps usage and overrides finish_reason."""
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta='<tool_call>{"name":"f","arguments":{}}</tool_call>'),
                    GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason="stop",
                        prompt_tokens=11,
                        completion_tokens=13,
                    ),
                ]
            )
        )
    ]
    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "tool_calls"
    assert terminal.prompt_tokens == 11
    assert terminal.completion_tokens == 13


@pytest.mark.asyncio
async def test_parse_tool_call_stream_preserves_terminal_error_metadata() -> None:
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason="error",
                        error_code="grammar_invalid",
                        error_message="invalid grammar",
                    )
                ]
            )
        )
    ]

    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "error"
    assert terminal.error_code == "grammar_invalid"
    assert terminal.error_message == "invalid grammar"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_candidate_rewrite_preserves_terminal_error_metadata() -> None:
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason="stop",
                        candidates=({"text": "plain answer", "finish_reason": "stop", "logprobs": None},),
                        error_code="empty_model_output",
                        error_message="model produced no visible output text",
                    )
                ]
            )
        )
    ]

    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "stop"
    assert terminal.error_code == "empty_model_output"
    assert terminal.error_message == "model produced no visible output text"
    assert terminal.candidates is not None
    assert terminal.candidates[0]["text"] == "plain answer"


# ── Qwen XML format ────────────────────────────────────────────────


async def _collect(text: str, **kwargs: object) -> list[GenerationChunk]:
    return [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta=text),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            ),
            **kwargs,  # type: ignore[arg-type]
        )
    ]


def _deltas(out: list[GenerationChunk]) -> list:
    return [c.tool_call_delta for c in out if c.tool_call_delta is not None]


def _args_for(out: list[GenerationChunk], index: int = 0) -> str:
    """Concatenate the arguments_delta for a given tool-call index."""
    return "".join(d.arguments_delta for d in _deltas(out) if d.index == index and d.arguments_delta)


@pytest.mark.asyncio
async def test_xml_format_autodetected() -> None:
    """Qwen XML form is parsed by the runtime heuristic (auto)."""
    out = await _collect(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>"
    )
    deltas = _deltas(out)
    assert deltas[0].function_name == "get_weather"
    import json

    assert json.loads(_args_for(out)) == {"city": "Tokyo"}
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_explicit_qwen_xml_format() -> None:
    out = await _collect(
        "<tool_call><function=f><parameter=a>\n1\n</parameter></function></tool_call>",
        tool_call_format="qwen_xml",
    )
    import json

    assert _deltas(out)[0].function_name == "f"
    assert json.loads(_args_for(out)) == {"a": 1}


@pytest.mark.asyncio
async def test_explicit_hermes_json_format() -> None:
    out = await _collect(
        '<tool_call>{"name":"g","arguments":{"k":"v"}}</tool_call>',
        tool_call_format="hermes_json",
    )
    assert _deltas(out)[0].function_name == "g"
    assert _args_for(out) == '{"k":"v"}'


@pytest.mark.asyncio
async def test_xml_argument_coercion_edge_cases() -> None:
    """XML parameter values coerce to typed JSON where possible, else string."""
    import json

    raw = (
        "<tool_call><function=f>"
        "<parameter=num>\n5\n</parameter>"
        "<parameter=flag>\ntrue\n</parameter>"
        '<parameter=nested>\n{"a":[1,2]}\n</parameter>'
        "<parameter=jsonish>\nnot json {oops\n</parameter>"
        "<parameter=empty>\n\n</parameter>"
        "<parameter=unicode>\ncafé ☕\n</parameter>"
        "<parameter=multiline>\nline1\nline2\n</parameter>"
        "</function></tool_call>"
    )
    out = await _collect(raw, tool_call_format="qwen_xml")
    args = json.loads(_args_for(out))
    assert args["num"] == 5
    assert args["flag"] is True
    assert args["nested"] == {"a": [1, 2]}
    assert args["jsonish"] == "not json {oops"
    assert args["empty"] == ""
    assert args["unicode"] == "café ☕"
    assert args["multiline"] == "line1\nline2"


@pytest.mark.asyncio
async def test_multiple_xml_blocks_distinct_indices() -> None:
    out = await _collect(
        "<tool_call><function=a><parameter=x>\n1\n</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>\n2\n</parameter></function></tool_call>",
        tool_call_format="qwen_xml",
    )
    deltas = _deltas(out)
    assert sorted({d.index for d in deltas}) == [0, 1]
    assert {d.function_name for d in deltas if d.function_name} == {"a", "b"}
    assert out[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_rejects_second_call() -> None:
    """With ``parallel_tool_calls=false`` a second ``<tool_call>`` block is
    rejected with a terminal error chunk rather than silently dropped.

    The first call's deltas are still flushed (it's a legal call); the
    second opener trips the enforcement and ends the stream as ``error``
    with ``parallel_tool_calls_violated``.
    """
    out = await _collect(
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>',
        parallel_tool_calls=False,
    )
    # First call's deltas (announce + arguments) are emitted before the
    # violation is detected on the second opener.
    deltas = _deltas(out)
    assert {d.index for d in deltas} == {0}
    assert {d.function_name for d in deltas if d.function_name} == {"a"}
    # The second call never reaches the parser — no index 1 deltas.
    assert all(d.index == 0 for d in deltas)
    # Terminal chunk carries the enforcement error.
    terminal = out[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "error"
    assert terminal.error_code == "parallel_tool_calls_violated"
    assert "parallel_tool_calls" in (terminal.error_message or "")


@pytest.mark.asyncio
async def test_parallel_tool_calls_false_single_call_passthrough() -> None:
    """A single tool call under ``parallel_tool_calls=false`` is unaffected:
    deltas flow normally and the stream finishes as ``tool_calls``.
    """
    out = await _collect(
        '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>',
        parallel_tool_calls=False,
    )
    deltas = _deltas(out)
    assert {d.function_name for d in deltas if d.function_name} == {"a"}
    assert out[-1].done is True
    assert out[-1].finish_reason == "tool_calls"
    assert out[-1].error_code is None


@pytest.mark.asyncio
async def test_parallel_tool_calls_true_keeps_all_calls() -> None:
    out = await _collect(
        '<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{}}</tool_call>',
        parallel_tool_calls=True,
    )
    assert sorted({d.index for d in _deltas(out)}) == [0, 1]


# ── fix #6: linear (non-backtracking) XML parameter scan ───────────────


def test_xml_param_scan_matches_well_formed_input() -> None:
    """The linear ``str.find`` scan reproduces the old regex behavior for
    well-formed input — multiple params, typed coercion preserved.
    """
    from sie_server.processors.tool_call_parser import _parse_xml_tool_call

    raw = (
        "<function=f>"
        "<parameter=a>\n1\n</parameter>"
        "<parameter=b>\nhello\n</parameter>"
        '<parameter=c>\n{"k":[1,2]}\n</parameter>'
        "</function>"
    )
    name, args = _parse_xml_tool_call(raw)
    assert name == "f"
    assert args == {"a": 1, "b": "hello", "c": {"k": [1, 2]}}


def test_xml_param_scan_stops_at_unterminated_parameter() -> None:
    """An opener with no closing ``</parameter>`` stops the scan instead of
    swallowing the rest of the buffer (and never backtracks).
    """
    from sie_server.processors.tool_call_parser import _parse_xml_tool_call

    raw = "<function=f><parameter=ok>\nv\n</parameter><parameter=dangling>\nno close here"
    name, args = _parse_xml_tool_call(raw)
    assert name == "f"
    assert args == {"ok": "v"}


def test_xml_param_scan_bounds_param_count() -> None:
    """Adversarial buffer with a huge number of openers parses at most
    ``_MAX_XML_PARAMS`` and returns in linear time (no O(n^2) backtracking).
    """
    import time

    from sie_server.processors.tool_call_parser import _MAX_XML_PARAMS, _parse_xml_tool_call

    # Far more params than the cap, each well-formed.
    n = _MAX_XML_PARAMS + 5_000
    body = "".join(f"<parameter=p{i}>\n{i}\n</parameter>" for i in range(n))
    raw = "<function=f>" + body + "</function>"
    t0 = time.monotonic()
    name, args = _parse_xml_tool_call(raw)
    elapsed = time.monotonic() - t0
    assert name == "f"
    assert len(args) == _MAX_XML_PARAMS
    # Generous ceiling: linear parse of this buffer is milliseconds.
    assert elapsed < 2.0


def test_xml_param_scan_garbled_openers_no_close_is_fast() -> None:
    """A 256 KiB buffer of openers with NO closing tags is the worst case for
    the old ``(.*?)...</parameter>`` regex; the linear scan returns promptly
    with zero params.
    """
    import time

    from sie_server.processors.tool_call_parser import _parse_xml_tool_call

    garbage = "<parameter=x>" * (256 * 1024 // len("<parameter=x>"))
    raw = "<function=f>" + garbage
    t0 = time.monotonic()
    name, args = _parse_xml_tool_call(raw)
    elapsed = time.monotonic() - t0
    assert name == "f"
    assert args == {}  # no closer found for any opener
    assert elapsed < 2.0


# ── Per-choice (n>1) streaming (H5) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_parse_tool_call_stream_per_choice_state_isolated() -> None:
    """H5: streaming ``n>1`` runs each candidate through its own parser
    state. Two interleaved candidates each containing their own
    ``<tool_call>`` block must produce two independent tool-call deltas,
    each tagged with the right ``choice_index``.
    """
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta="A:", choice_index=0),
                    GenerationChunk(text_delta="B:", choice_index=1),
                    GenerationChunk(
                        text_delta='<tool_call>{"name":"fa","arguments":{}}</tool_call>',
                        choice_index=0,
                    ),
                    GenerationChunk(
                        text_delta='<tool_call>{"name":"fb","arguments":{}}</tool_call>',
                        choice_index=1,
                    ),
                    # Per-choice finishes (non-terminal with finish_reason).
                    GenerationChunk(text_delta="", choice_index=0, finish_reason="stop"),
                    GenerationChunk(text_delta="", choice_index=1, finish_reason="length"),
                    GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason="stop",
                        prompt_tokens=3,
                        completion_tokens=8,
                    ),
                ]
            )
        )
    ]

    tool_calls_by_choice: dict[int, list] = {}
    for item in out:
        if item.tool_call_delta is not None:
            tool_calls_by_choice.setdefault(item.choice_index, []).append(item.tool_call_delta)

    # Each candidate produced its own tool-call announce+body pair.
    assert 0 in tool_calls_by_choice
    assert 1 in tool_calls_by_choice
    names_0 = {tcd.function_name for tcd in tool_calls_by_choice[0] if tcd.function_name}
    names_1 = {tcd.function_name for tcd in tool_calls_by_choice[1] if tcd.function_name}
    assert names_0 == {"fa"}
    assert names_1 == {"fb"}

    # Per-choice closures carry the right finish_reason — coerced to
    # ``tool_calls`` because each emitted a tool call.
    closures = [item for item in out if not item.done and item.finish_reason is not None and not item.tool_call_delta]
    closure_by_choice = {c.choice_index: c.finish_reason for c in closures}
    assert closure_by_choice.get(0) == "tool_calls"
    assert closure_by_choice.get(1) == "tool_calls"


@pytest.mark.asyncio
async def test_parse_tool_call_stream_per_choice_logprobs_pass_through() -> None:
    """H4: a chunk with ``logprobs`` set and an empty text_delta surfaces as
    its own logprobs-bearing chunk through the parser, preserving
    ``choice_index``. Required for streaming ``n>1`` + logprobs.
    """
    lp_entry = {"token": "t", "logprob": -0.5, "bytes": [116], "top_logprobs": []}
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(text_delta="t", choice_index=2, logprobs=(lp_entry,)),
                    GenerationChunk(text_delta="", done=True, finish_reason="stop"),
                ]
            )
        )
    ]
    lp_chunks = [c for c in out if c.logprobs]
    assert lp_chunks, "logprobs must survive the parser wrap"
    assert any(c.choice_index == 2 for c in lp_chunks)


@pytest.mark.asyncio
async def test_parse_tool_call_stream_non_streaming_per_candidate_tool_calls() -> None:
    """H5 non-streaming: the parser injects per-candidate ``tool_calls`` onto
    the terminal's ``candidates`` array by parsing each candidate's full
    text for ``<tool_call>`` blocks.
    """
    candidates = (
        {
            "text": 'Hi <tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>',
            "finish_reason": "stop",
            "logprobs": None,
        },
        {
            "text": "plain answer",
            "finish_reason": "stop",
            "logprobs": None,
        },
    )
    out = [
        item
        async for item in parse_tool_call_stream(
            _chunks(
                [
                    GenerationChunk(
                        text_delta="",
                        done=True,
                        finish_reason="stop",
                        prompt_tokens=4,
                        completion_tokens=10,
                        candidates=candidates,
                    ),
                ]
            )
        )
    ]
    terminal = out[-1]
    assert terminal.done is True
    assert terminal.candidates is not None
    cands = list(terminal.candidates)
    assert "tool_calls" in cands[0]
    assert cands[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    # Per-candidate finish_reason coerced to ``tool_calls`` when a call was parsed.
    assert cands[0]["finish_reason"] == "tool_calls"
    assert "tool_calls" not in cands[1] or not cands[1].get("tool_calls")
    assert cands[1]["finish_reason"] == "stop"
