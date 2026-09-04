"""Strict streaming parser for Qwen/Hermes-style tool-call tags.

The SGLang adapter yields text deltas. Qwen3 chat templates encode tool calls as
``<tool_call>{"name":"...","arguments":{...}}</tool_call>`` inside that text
stream. This module converts those tagged regions into the worker/gateway
``tool_call_delta`` shape while preserving surrounding prose as normal text.

This first implementation emits arguments atomically when the closing tag arrives.
It is intentionally strict: malformed JSON or missing ``name``/``arguments``
turns into a terminal ``MODEL_OUTPUT_PARSE_ERROR`` chunk.

For streaming ``n>1`` the parser maintains independent per-candidate state
keyed by ``chunk.choice_index`` (H5): each candidate's tool-call deltas
surface tagged with the same ``choice_index`` they came in on, so the
gateway can fan tool calls out per candidate.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, cast

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationChunk,
    ToolCallDelta,
    aclose_with_error_precedence,
)

logger = logging.getLogger(__name__)

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_PARSE_ERROR = "MODEL_OUTPUT_PARSE_ERROR"
# Terminal error code emitted when a model ignores ``parallel_tool_calls=false``
# and tries to open a second ``<tool_call>`` block in the same turn. We refuse
# to silently truncate (the prior behavior leaked a successful response with
# hidden missing tool calls); the client gets an explicit enforcement error.
_PARALLEL_TOOL_CALLS_VIOLATED = "parallel_tool_calls_violated"

# On-the-wire tool-call encodings that can appear inside
# ``<tool_call>…</tool_call>``:
#   - ``qwen_xml``    — Qwen3(-Coder): ``<function=NAME><parameter=K>V</parameter>…</function>``
#   - ``hermes_json`` — Hermes: ``{"name": "...", "arguments": {...}}``
#   - ``auto``        — runtime heuristic (XML if the block starts with
#                       ``<function=``, else JSON). Kept as a fallback for
#                       callers that cannot resolve the model's configured
#                       parser, but the worker now drives this from the
#                       model config (``tasks.generate`` → adapter
#                       ``tool_call_parser``) so production traffic uses an
#                       explicit format rather than guessing per block.
ToolCallFormat = Literal["auto", "qwen_xml", "hermes_json"]

# A model that emits ``<tool_call>`` and never closes it would let
# ``tool_buffer`` grow without bound (same for free-form prose with no
# opener filling ``text_buffer``). 256 KiB per buffer is comfortably
# larger than any realistic tool argument blob while still capping the
# worker's memory cost per stuck request — surfacing the malformed
# stream as a parse error instead of an OOM.
_MAX_TOOL_BUFFER_CHARS = 256 * 1024
_MAX_TEXT_BUFFER_CHARS = 256 * 1024
# Cap on the deserialised ``arguments`` blob's re-serialised length.
# Defends against pathological deeply-nested JSON that survives
# ``json.loads`` (Python's default recursion limit) but takes seconds
# to ``json.dumps`` back out.
_MAX_TOOL_ARGUMENTS_CHARS = 64 * 1024


async def parse_tool_call_stream(
    chunks: AsyncIterator[GenerationChunk],
    *,
    tool_call_format: ToolCallFormat = "auto",
    parallel_tool_calls: bool = True,
) -> AsyncIterator[GenerationChunk]:
    """Tool-call parsing wrapper that guarantees the upstream is closed.

    Closing this wrapper (``aclose()``, e.g. on client cancel) does **not**
    by itself finalize the wrapped ``chunks`` generator — Python only GCs it
    later. For the SGLang adapter that means its ``except GeneratorExit ->
    POST /abort_request`` cleanup wouldn't run promptly, orphaning a
    generation on the GPU. Wrap iteration in ``finally`` so the upstream is
    always ``aclose()``d (on cancel, early parse-error ``return``, and normal
    completion alike).
    """
    terminal_outcome_selected = False
    try:
        async for out in _parse_tool_call_stream_impl(
            chunks,
            tool_call_format=tool_call_format,
            parallel_tool_calls=parallel_tool_calls,
        ):
            if out.done:
                terminal_outcome_selected = True
            yield out
    finally:
        await aclose_with_error_precedence(
            chunks,
            outcome_selected=terminal_outcome_selected,
            context="tool-call parser upstream iterator",
        )


@dataclass
class _ChoiceState:
    """Per-candidate parser state for streaming ``n>1`` (H5).

    The original parser ran with these as locals; refactoring them into a
    state object lets the impl maintain N independent parsers keyed by
    ``chunk.choice_index``. For ``n=1`` (the default) exactly one state
    is created with key ``0``, preserving the prior behaviour bit-for-bit.
    """

    text_buffer: str = ""
    in_tool_call: bool = False
    tool_buffer: str = ""
    tool_index: int = 0
    emitted_tool_call: bool = False
    # ``is_first`` is a one-shot marker on the very first user-visible
    # chunk; latched per-choice so each candidate's first delta carries
    # the marker (otherwise the n>1 stream would mark only candidate 0).
    first_emitted: bool = False
    # Captured from the per-choice "finish" delta (non-terminal chunk
    # with ``finish_reason`` set on the multi-candidate streaming path)
    # so a subsequent global terminal still surfaces this choice's
    # finish reason on the per-choice closure emission.
    pending_finish: str | None = None
    pending_prompt_tokens: int | None = None
    pending_completion_tokens: int | None = None
    # Set once the choice's terminal closure has been emitted so the
    # global terminal does not double-emit.
    closed: bool = False
    # Tool-call deltas observed so far (for parallel_tool_calls=false
    # enforcement, currently sufficient as the boolean
    # ``emitted_tool_call`` flag). Reserved field kept implicit.
    _reserved: list[ToolCallDelta] = field(default_factory=list)


def _mark_first(state: _ChoiceState, is_first_hint: bool) -> bool:
    """Latch first-yield marker per-choice."""
    if state.first_emitted or not is_first_hint:
        return False
    state.first_emitted = True
    return True


def _process_text(
    state: _ChoiceState,
    *,
    incoming: str,
    is_first_hint: bool,
    choice_index: int,
    tool_call_format: ToolCallFormat,
    parallel_tool_calls: bool,
) -> tuple[list[GenerationChunk], bool]:
    """Run one text delta through the per-choice parser.

    Returns ``(emitted, terminal)``: ``terminal`` is True when a parse
    error / parallel-violation chunk was produced and the parser for
    this choice (and the whole stream, since today's contract is
    stream-wide on parse failure) should stop.
    """
    out: list[GenerationChunk] = []
    if not incoming:
        return out, False
    incoming = state.text_buffer + incoming
    state.text_buffer = ""
    cursor = 0
    while cursor < len(incoming):
        if state.in_tool_call:
            close_idx = incoming.find(_CLOSE, cursor)
            if close_idx == -1:
                # See the in-line comment in the original implementation
                # below — preserve the close-tag-straddle handling.
                keep = len(_CLOSE) - 1
                absorb_end = max(cursor, len(incoming) - keep)
                state.tool_buffer += incoming[cursor:absorb_end]
                if len(state.tool_buffer) + (len(incoming) - absorb_end) > _MAX_TOOL_BUFFER_CHARS:
                    out.append(
                        _parse_error_chunk(
                            f"unterminated <tool_call> block exceeded "
                            f"{_MAX_TOOL_BUFFER_CHARS} chars without a closing tag"
                        )
                    )
                    return out, True
                state.text_buffer = incoming[absorb_end:]
                cursor = len(incoming)
                continue
            state.tool_buffer += incoming[cursor:close_idx]
            try:
                deltas = _tool_call_deltas(state.tool_buffer, state.tool_index, tool_call_format)
            except ValueError as exc:
                out.append(_parse_error_chunk(str(exc)))
                return out, True
            for delta in deltas:
                out.append(
                    GenerationChunk(
                        text_delta="",
                        done=False,
                        tool_call_delta=delta,
                        choice_index=choice_index,
                    )
                )
            state.emitted_tool_call = True
            state.tool_index += 1
            state.tool_buffer = ""
            state.in_tool_call = False
            cursor = close_idx + len(_CLOSE)
            continue

        open_idx = incoming.find(_OPEN, cursor)
        if open_idx == -1:
            state.text_buffer += incoming[cursor:]
            cursor = len(incoming)
            if len(state.text_buffer) > _MAX_TEXT_BUFFER_CHARS:
                out.append(
                    _parse_error_chunk(
                        f"text buffer exceeded {_MAX_TEXT_BUFFER_CHARS} chars without producing a tool-call boundary"
                    )
                )
                return out, True
            flush_len = max(0, len(state.text_buffer) - (len(_OPEN) - 1))
            if flush_len:
                out.append(
                    GenerationChunk(
                        text_delta=state.text_buffer[:flush_len],
                        done=False,
                        is_first=_mark_first(state, is_first_hint),
                        choice_index=choice_index,
                    )
                )
                state.text_buffer = state.text_buffer[flush_len:]
            continue

        state.text_buffer += incoming[cursor:open_idx]
        if state.text_buffer:
            out.append(
                GenerationChunk(
                    text_delta=state.text_buffer,
                    done=False,
                    is_first=_mark_first(state, is_first_hint),
                    choice_index=choice_index,
                )
            )
            state.text_buffer = ""
        # ``parallel_tool_calls=false`` enforcement (per-choice scoped).
        if not parallel_tool_calls and state.emitted_tool_call:
            logger.info(
                "parallel_tool_calls=false: model emitted a second <tool_call> block on choice %d; "
                "terminating stream with %s",
                choice_index,
                _PARALLEL_TOOL_CALLS_VIOLATED,
            )
            out.append(_parallel_tool_calls_violation_chunk())
            return out, True
        state.in_tool_call = True
        cursor = open_idx + len(_OPEN)
    return out, False


def _close_choice(state: _ChoiceState, choice_index: int, fallback_finish: str | None) -> list[GenerationChunk]:
    """Emit the per-choice closure chunk for one candidate.

    Used both on the per-choice finish-delta path (multi-candidate
    streaming, when the worker observed an SGLang ``finish_reason`` on a
    specific index) and on the global terminal (single-candidate path,
    or any choice that did not see a per-choice finish event).
    """
    out: list[GenerationChunk] = []
    if state.in_tool_call:
        out.append(_parse_error_chunk("unterminated <tool_call> block"))
        return out
    if state.text_buffer:
        out.append(
            GenerationChunk(
                text_delta=state.text_buffer,
                done=False,
                is_first=_mark_first(state, False),
                choice_index=choice_index,
            )
        )
        state.text_buffer = ""
    finish: FinishReason = (
        "tool_calls"
        if state.emitted_tool_call
        else cast("FinishReason", state.pending_finish or fallback_finish or "stop")
    )
    out.append(
        GenerationChunk(
            text_delta="",
            # Per-choice closure rides as ``done=False`` with a populated
            # ``finish_reason`` so the processor's done-path (which
            # closes the whole stream) is only triggered by the global
            # terminal. For ``n=1`` the global terminal IS this same
            # chunk — see :func:`_parse_tool_call_stream_impl`.
            done=False,
            finish_reason=finish,
            choice_index=choice_index,
            prompt_tokens=state.pending_prompt_tokens,
            completion_tokens=state.pending_completion_tokens,
        )
    )
    state.closed = True
    return out


async def _parse_tool_call_stream_impl(
    chunks: AsyncIterator[GenerationChunk],
    *,
    tool_call_format: ToolCallFormat = "auto",
    parallel_tool_calls: bool = True,
) -> AsyncIterator[GenerationChunk]:
    """Convert tagged text chunks into OpenAI-compatible tool-call deltas.

    ``tool_call_format`` selects the encoding inside each
    ``<tool_call>`` block (see :data:`ToolCallFormat`). The worker
    resolves it from the model config so the choice is explicit; the
    default ``"auto"`` preserves the original per-block heuristic for
    callers that don't.

    ``parallel_tool_calls`` mirrors the OpenAI request flag. When
    ``False`` only one tool call is permitted in a turn — if the model
    ignores the single-call instruction and opens a second
    ``<tool_call>`` block we emit a terminal error chunk with code
    ``parallel_tool_calls_violated`` instead of silently dropping the
    extra call. The prior behavior (drop second call, finish as
    ``tool_calls``) returned a "successful" response with hidden missing
    data; clients now get an explicit enforcement error they can act on.

    Per-choice (H5): when the upstream stream carries
    ``chunk.choice_index`` for multi-candidate runs each candidate gets
    an independent parser state. A non-terminal chunk with a
    ``finish_reason`` set is treated as the *per-choice* closure event
    (multi-candidate streaming pattern) — the wrapper flushes that
    choice's buffered text/tool-call and emits a per-choice closure
    chunk with the right ``finish_reason``. The global ``done=True``
    terminal then closes any choices that did not see a per-choice
    closure and surfaces aggregate usage.
    """
    states: dict[int, _ChoiceState] = {}

    def _state(idx: int) -> _ChoiceState:
        s = states.get(idx)
        if s is None:
            s = _ChoiceState()
            states[idx] = s
        return s

    async for chunk in chunks:
        idx = chunk.choice_index
        state = _state(idx)
        incoming = chunk.text_delta

        # Per-choice closure on the multi-candidate streaming path:
        # SGLang emits a non-terminal event with ``finish_reason`` set
        # when a specific candidate hits stop/length. Capture the
        # closure metadata, flush text first (via _process_text on any
        # delta riding the same chunk), then emit the per-choice
        # closure marker. The global ``done=True`` is handled below.
        is_per_choice_finish = (chunk.finish_reason is not None) and not chunk.done

        if incoming or (is_per_choice_finish and state.in_tool_call):
            emitted, terminal = _process_text(
                state,
                incoming=incoming,
                is_first_hint=chunk.is_first,
                choice_index=idx,
                tool_call_format=tool_call_format,
                parallel_tool_calls=parallel_tool_calls,
            )
            for ev in emitted:
                yield ev
            if terminal:
                return
        # Forward the chunk's logprobs slice (already tagged with
        # choice_index) as its own chunk so streaming logprobs survive
        # the parser wrap. Tool-call deltas from the model surface via
        # the text path above; this branch only fires when the upstream
        # adapter already attached pre-parsed logprobs to the chunk
        # (single- and multi-candidate streaming both do).
        if chunk.logprobs:
            yield GenerationChunk(
                text_delta="",
                done=False,
                choice_index=idx,
                logprobs=chunk.logprobs,
            )

        if is_per_choice_finish:
            state.pending_finish = chunk.finish_reason
            state.pending_prompt_tokens = chunk.prompt_tokens
            state.pending_completion_tokens = chunk.completion_tokens
            for ev in _close_choice(state, idx, chunk.finish_reason):
                yield ev

        if chunk.done:
            # Non-streaming ``n>1`` + tools (H5 non-streaming side): the
            # adapter ships a single terminal with ``candidates=[{text,
            # finish_reason, logprobs}, ...]``. Run each candidate's text
            # through a one-shot parser so per-candidate ``tool_calls``
            # surface on the terminal's candidates array. Mutating the
            # passed-in tuple is forbidden (frozen dataclass slots);
            # build a new tuple and re-yield the terminal with it set.
            if chunk.candidates:
                updated_candidates: list[dict] = []
                any_tool_call = False
                for cand in chunk.candidates:
                    cand_text = cand.get("text", "") if isinstance(cand, dict) else ""
                    parsed_text, parsed_calls = _parse_candidate_text(cand_text, tool_call_format)
                    new_cand = dict(cand) if isinstance(cand, dict) else {}
                    if parsed_calls:
                        # OpenAI non-streaming shape: message.content=null,
                        # message.tool_calls=[{id, type, function:{name, arguments}}].
                        # The gateway candidate builder surfaces this as
                        # ``choices[i].message.tool_calls``.
                        new_cand["tool_calls"] = parsed_calls
                        new_cand["text"] = parsed_text
                        # If the candidate had a non-tool_calls
                        # finish_reason (e.g. ``length`` because the
                        # model ran out of tokens after emitting a
                        # complete tool block) OpenAI canonicalises
                        # this to ``tool_calls``.
                        new_cand["finish_reason"] = "tool_calls"
                        any_tool_call = True
                    else:
                        new_cand["text"] = parsed_text
                    updated_candidates.append(new_cand)
                # Replace the chunk's candidates by re-emitting via a
                # frozen-dataclass swap; downstream sees the new list.
                # ``GenerationChunk`` is frozen, so build a fresh
                # instance with the updated tuple.
                chunk = GenerationChunk(
                    text_delta=chunk.text_delta,
                    done=True,
                    is_first=chunk.is_first,
                    finish_reason="tool_calls" if any_tool_call else chunk.finish_reason,
                    prompt_tokens=chunk.prompt_tokens,
                    completion_tokens=chunk.completion_tokens,
                    candidates=tuple(updated_candidates),
                    logprobs=chunk.logprobs,
                    error_code=chunk.error_code,
                    error_message=chunk.error_message,
                )
            # Global terminal. Close any choices that did not see a
            # per-choice finish event (single-candidate path: this is
            # the only closure). The terminal itself carries aggregate
            # usage and rides as ``done=True``.
            # Ensure choice 0 exists so the single-candidate path
            # always emits its closure here (matches prior behaviour).
            if not states:
                _state(0)
            for cidx, st in list(states.items()):
                if st.closed:
                    continue
                # Mid-tool-call at terminal → error (mirrors prior impl).
                if st.in_tool_call:
                    yield _parse_error_chunk("unterminated <tool_call> block")
                    return
                if st.text_buffer:
                    yield GenerationChunk(
                        text_delta=st.text_buffer,
                        done=False,
                        is_first=_mark_first(st, chunk.is_first),
                        choice_index=cidx,
                    )
                    st.text_buffer = ""
            # Emit the global terminal. Preserve the single-candidate
            # contract: ``finish_reason`` here is the global stream
            # finish (worker terminal). For ``n=1`` clients consume
            # this as the choice's finish_reason too — the gateway SSE
            # path maps both per-choice and global finish reasons.
            global_finish = chunk.finish_reason
            # Single-candidate emitted a tool call? Surface tool_calls
            # on the global terminal too so n=1 behaviour matches the
            # original wrapper exactly.
            if len(states) == 1 and 0 in states and states[0].emitted_tool_call:
                global_finish = "tool_calls"
            elif global_finish is None:
                global_finish = "tool_calls" if any(s.emitted_tool_call for s in states.values()) else "stop"
            yield GenerationChunk(
                text_delta="",
                done=True,
                is_first=_mark_first(_state(0), chunk.is_first),
                finish_reason=global_finish,  # type: ignore[arg-type]
                prompt_tokens=chunk.prompt_tokens,
                completion_tokens=chunk.completion_tokens,
                error_code=chunk.error_code,
                error_message=chunk.error_message,
                # Preserve ``candidates`` (with any per-candidate
                # ``tool_calls`` injected above for the non-streaming
                # ``n>1`` + tools path) through the wrap.
                candidates=chunk.candidates,
            )
            return

    # Upstream iterator ended without a terminal — close every choice's
    # tail (mirrors the original behaviour) and synthesize a terminal.
    for cidx, st in list(states.items()):
        if st.closed:
            continue
        if st.in_tool_call:
            yield _parse_error_chunk("unterminated <tool_call> block")
            return
        if st.text_buffer:
            yield GenerationChunk(
                text_delta=st.text_buffer,
                done=False,
                is_first=_mark_first(st, False),
                choice_index=cidx,
            )
    any_tool = any(s.emitted_tool_call for s in states.values()) if states else False
    yield GenerationChunk(text_delta="", done=True, finish_reason="tool_calls" if any_tool else "stop")


def _parse_candidate_text(text: str, tool_call_format: ToolCallFormat) -> tuple[str, list[dict]]:
    """Parse a single candidate's full text for ``<tool_call>`` blocks.

    Returns ``(text_outside_tool_blocks, tool_calls)`` where
    ``tool_calls`` is the OpenAI non-streaming wire shape:
    ``[{id, type, function: {name, arguments}}, ...]``.

    Used by the non-streaming ``n>1`` + tools path: the worker's
    multi-candidate adapter ships one terminal carrying the full text
    of each candidate, and each candidate needs its own tool-call
    aggregation (H5 non-streaming side). On malformed blocks the
    candidate's surrounding text is preserved verbatim and the
    malformed block is skipped — the non-streaming path doesn't have a
    natural channel for a per-candidate parse error, so we degrade
    gracefully rather than failing the whole multi-candidate batch.
    """
    if not text or _OPEN not in text:
        return text, []
    out_text_parts: list[str] = []
    tool_calls: list[dict] = []
    cursor = 0
    tool_index = 0
    while cursor < len(text):
        open_idx = text.find(_OPEN, cursor)
        if open_idx == -1:
            out_text_parts.append(text[cursor:])
            break
        out_text_parts.append(text[cursor:open_idx])
        body_start = open_idx + len(_OPEN)
        close_idx = text.find(_CLOSE, body_start)
        if close_idx == -1:
            # Unterminated block: bail; surface the rest as text so the
            # candidate's prose isn't silently lost.
            out_text_parts.append(text[open_idx:])
            break
        body = text[body_start:close_idx]
        try:
            deltas = _tool_call_deltas(body, tool_index, tool_call_format)
        except ValueError:
            # Malformed tool body — drop it and keep going. The candidate
            # batch is non-streaming, so partial failure of one block in
            # one candidate must not poison the rest.
            cursor = close_idx + len(_CLOSE)
            continue
        # ``_tool_call_deltas`` returns two deltas per call: announcement
        # (id+name) and body (arguments). Re-assemble into the
        # non-streaming OpenAI shape.
        if len(deltas) == 2:  # noqa: PLR2004 — paired (announce, body) shape contract
            announce, body_delta = deltas[0], deltas[1]
            tool_calls.append(
                {
                    "id": announce.id or "",
                    "type": "function",
                    "function": {
                        "name": announce.function_name or "",
                        "arguments": body_delta.arguments_delta,
                    },
                }
            )
        tool_index += 1
        cursor = close_idx + len(_CLOSE)
    return "".join(out_text_parts), tool_calls


def _tool_call_deltas(raw: str, index: int, tool_call_format: ToolCallFormat = "auto") -> list[ToolCallDelta]:
    raw = raw.strip()
    # Two on-the-wire formats appear inside <tool_call>…</tool_call>:
    #   1. Hermes JSON:  {"name": "...", "arguments": {...}}
    #   2. Qwen3(-Coder) XML:  <function=NAME><parameter=K>V</parameter>…</function>
    # Qwen3.5's chat template emits format (2); other models emit (1).
    # ``tool_call_format`` makes the choice explicit (config-driven);
    # ``"auto"`` falls back to the original "starts-with-<function=" heuristic.
    if tool_call_format == "qwen_xml":
        name, arguments = _parse_xml_tool_call(raw)
    elif tool_call_format == "hermes_json":
        name, arguments = _parse_hermes_tool_call(raw)
    elif raw.startswith("<function="):
        name, arguments = _parse_xml_tool_call(raw)
    else:
        name, arguments = _parse_hermes_tool_call(raw)
    if not isinstance(name, str) or not name:
        raise ValueError("tool-call payload missing string 'name'")
    if arguments is None:
        arguments = {}
    try:
        arguments_json = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool-call 'arguments' is not JSON-serialisable: {exc}") from exc
    if len(arguments_json) > _MAX_TOOL_ARGUMENTS_CHARS:
        raise ValueError(
            f"tool-call 'arguments' serialised to {len(arguments_json)} chars, exceeds {_MAX_TOOL_ARGUMENTS_CHARS}"
        )
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    return [
        ToolCallDelta(index=index, id=call_id, function_name=name, arguments_delta=""),
        ToolCallDelta(index=index, arguments_delta=arguments_json),
    ]


def _parse_hermes_tool_call(raw: str) -> tuple[object, object]:
    """Parse the Hermes JSON tool-call form ``{"name": ..., "arguments": ...}``.

    Returns the raw ``(name, arguments)`` (untyped) for the shared
    validation/serialisation tail in :func:`_tool_call_deltas` to check.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed tool-call JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("tool-call payload must be a JSON object")
    return value.get("name"), value.get("arguments")


_XML_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>")
# Matches only a parameter *opener* — ``<parameter=name>`` — with a bounded
# character class, so it cannot backtrack. The value and closing
# ``</parameter>`` are located with a linear ``str.find`` below rather than a
# ``(.*?)...</parameter>`` group, which on a 256 KiB buffer of garbled output
# (many openers, no closers) is O(n²) and runs INLINE on the event loop.
_XML_PARAM_OPEN_RE = re.compile(r"<parameter=([^>\s]+)\s*>")
_XML_PARAM_CLOSE = "</parameter>"
# Hard cap on parameters parsed from one XML tool call. Defends against an
# adversarial buffer packed with openers; well-formed calls have a handful.
_MAX_XML_PARAMS = 256


def _parse_xml_tool_call(raw: str) -> tuple[str, dict[str, object]]:
    """Parse the Qwen3(-Coder) XML tool-call form.

    Example::

        <function=get_weather>
        <parameter=city>
        Tokyo
        </parameter>
        </function>

    Returns ``(name, arguments)``. Parameter values are coerced from text to
    JSON scalars where possible (so ``"5"`` → ``5``, ``"true"`` → ``True``),
    falling back to the trimmed string — which matches how OpenAI clients
    expect typed function arguments.
    """
    fm = _XML_FUNC_RE.search(raw)
    if not fm:
        raise ValueError("malformed XML tool-call: missing <function=...>")
    name = fm.group(1)
    arguments: dict[str, object] = {}
    # Linear scan: find each opener, then the next ``</parameter>`` via
    # ``str.find`` (no regex backtracking). Worst case is O(n) over ``raw``.
    pos = 0
    count = 0
    while count < _MAX_XML_PARAMS:
        om = _XML_PARAM_OPEN_RE.search(raw, pos)
        if om is None:
            break
        key = om.group(1)
        val_start = om.end()
        close_idx = raw.find(_XML_PARAM_CLOSE, val_start)
        if close_idx == -1:
            # Unterminated parameter — stop; well-formed input always closes.
            break
        val = raw[val_start:close_idx].strip()
        try:
            arguments[key] = json.loads(val)
        except (json.JSONDecodeError, ValueError):
            arguments[key] = val
        pos = close_idx + len(_XML_PARAM_CLOSE)
        count += 1
    return name, arguments


def _parse_error_chunk(message: str) -> GenerationChunk:
    return GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="error",
        tool_call_delta=None,
        error_code=_PARSE_ERROR,
        error_message=message,
    )


def _parallel_tool_calls_violation_chunk() -> GenerationChunk:
    """Terminal chunk emitted when the model opens a second
    ``<tool_call>`` block under ``parallel_tool_calls=false``.

    Shape mirrors :func:`_parse_error_chunk` so downstream publishers
    (``streaming.py``) surface it through the same terminal-error path
    used for malformed JSON, transport failures, etc.
    """
    return GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="error",
        tool_call_delta=None,
        error_code=_PARALLEL_TOOL_CALLS_VIOLATED,
        error_message=(
            "model attempted a second tool call but parallel_tool_calls=false was set; "
            "refusing to silently drop additional calls"
        ),
    )
