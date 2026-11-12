"""Strict streaming parser for model tool-call tags.

The SGLang adapter yields text deltas. Qwen3 chat templates encode tool calls as
``<tool_call>{"name":"...","arguments":{...}}</tool_call>`` inside that text
stream. This module converts those tagged regions into the worker/gateway
``tool_call_delta`` shape while preserving surrounding prose as normal text.

This first implementation emits arguments atomically when the closing tag arrives.
While a block is open, each text delta it absorbs is passed on as an empty
progress chunk, so a consumer that bounds the gap between chunks sees a call that
is still being generated. It is intentionally strict: malformed JSON or missing
``name``/``arguments`` turns into a terminal ``MODEL_OUTPUT_PARSE_ERROR`` chunk.

The XML forms carry every argument value as text. Each value is converted by
the JSON-schema type its parameter declares in the request's ``tools``: a
string parameter keeps its exact text, and a value that does not convert, or
that belongs to an undeclared tool or parameter, stays a string.

Gemma 4 uses its own tags, ``<|tool_call>call:NAME{key:value,…}<tool_call|>``,
and writes typed values (strings between ``<|"|>`` delimiters, numbers,
booleans, ``null``, nested objects and arrays), so its values keep their native
types; only a top-level argument whose parameter is declared as a string keeps
the exact text of a bare token.

For streaming ``n>1`` the parser maintains independent per-candidate state
keyed by ``chunk.choice_index`` (H5): each candidate's tool-call deltas
surface tagged with the same ``choice_index`` they came in on, so the
gateway can fan tool calls out per candidate.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from sie_server.adapters._generation_base import (
    FinishReason,
    GenerationChunk,
    ToolCallDelta,
    aclose_with_error_precedence,
)

logger = logging.getLogger(__name__)

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
# Gemma 4 delimits a call with its own special tokens instead of the tags above.
_GEMMA_OPEN = "<|tool_call>"
_GEMMA_CLOSE = "<tool_call|>"
_PARSE_ERROR = "MODEL_OUTPUT_PARSE_ERROR"
# Terminal error code emitted when a model ignores ``parallel_tool_calls=false``
# and tries to open a second ``<tool_call>`` block in the same turn. We refuse
# to silently truncate (the prior behavior leaked a successful response with
# hidden missing tool calls); the client gets an explicit enforcement error.
_PARALLEL_TOOL_CALLS_VIOLATED = "parallel_tool_calls_violated"

# On-the-wire tool-call encodings that can appear inside
# ``<tool_call>…</tool_call>`` (``gemma4`` uses its own tags):
#   - ``qwen_xml``    — Qwen3(-Coder): ``<function=NAME><parameter=K>V</parameter>…</function>``
#   - ``hermes_json`` — Hermes: ``{"name": "...", "arguments": {...}}``
#   - ``glm_xml``     — GLM: ``NAME<arg_key>K</arg_key><arg_value>V</arg_value>…``
#   - ``muse_atem``   — Muse Glimmer: its adapter rewrites ``atem:invoke`` blocks
#                       into the ``qwen_xml`` form; only forcing sees the original.
#   - ``gemma4``      — Gemma 4: ``<|tool_call>call:NAME{K:V,…}<tool_call|>`` with
#                       typed values and ``<|"|>``-delimited strings.
#   - ``auto``        — runtime heuristic (XML if the block starts with
#                       ``<function=``, else JSON). Kept as a fallback for
#                       callers that cannot resolve the model's configured
#                       parser, but the worker now drives this from the
#                       model config (``tasks.generate`` → adapter
#                       ``tool_call_parser``) so production traffic uses an
#                       explicit format rather than guessing per block.
ToolCallFormat = Literal["auto", "qwen_xml", "hermes_json", "glm_xml", "muse_atem", "gemma4"]


def tool_call_format_for_parser(parser: object) -> ToolCallFormat:
    """Map a model's configured ``tool_call_parser`` name to its on-wire format.

    The name is the same value passed to SGLang's ``--tool-call-parser``.
    ``qwen25`` (Qwen2.5-style Hermes JSON) and ``hermes*`` → ``hermes_json``;
    any other ``qwen*`` (e.g. ``qwen3_coder``) → ``qwen_xml``; ``glm*`` →
    ``glm_xml``; ``muse`` → ``muse_atem``; ``gemma4`` → ``gemma4``. A missing
    or unknown name maps to ``auto`` so out-of-tree models keep the runtime
    heuristic.
    """
    if not isinstance(parser, str):
        return "auto"
    name = parser.lower()
    if name == "qwen25" or "hermes" in name:
        return "hermes_json"
    if name.startswith("qwen"):
        return "qwen_xml"
    if name.startswith("glm"):
        return "glm_xml"
    if name == "muse":
        return "muse_atem"
    if name == "gemma4":
        return "gemma4"
    return "auto"


def _tool_call_tags(tool_call_format: ToolCallFormat) -> tuple[str, str]:
    """Return the ``(open, close)`` tags that delimit one call in this format."""
    if tool_call_format == "gemma4":
        return _GEMMA_OPEN, _GEMMA_CLOSE
    return _OPEN, _CLOSE


# A model that emits ``<tool_call>`` and never closes it would let
# ``tool_buffer`` grow without bound (same for free-form prose with no
# opener filling ``text_buffer``). 256 KiB per buffer is comfortably
# larger than any realistic tool argument blob while still capping the
# worker's memory cost per stuck request — surfacing the malformed
# stream as a parse error instead of an OOM.
_MAX_TOOL_BUFFER_CHARS = 256 * 1024
_MAX_TEXT_BUFFER_CHARS = 256 * 1024
# A call's serialised ``arguments`` cross the queue as one chunk, and one queue
# message carries at most 1 MiB by default. JSON escaping at most doubles
# printable ASCII, so any such block that fits the tool buffer fits this bound.
_MAX_TOOL_ARGUMENTS_BYTES = 2 * _MAX_TOOL_BUFFER_CHARS

ToolSchemas = Mapping[str, Mapping[str, Any]]

_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})
_INTEGER_TYPE_PREFIXES = ("int", "uint", "long", "short", "unsigned")
_NUMBER_TYPE_PREFIXES = ("num", "float")
_CONTAINER_TYPES = frozenset({"object", "array", "arr"})
_CONTAINER_TYPE_PREFIXES = ("dict", "list")
_MAX_SCHEMA_DEPTH = 32


def tool_parameter_schemas(tools: Sequence[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    """Map each declared function name in an OpenAI ``tools`` array to its ``parameters`` schema."""
    schemas: dict[str, Mapping[str, Any]] = {}
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        schemas[name] = parameters if isinstance(parameters, Mapping) else {}
    return schemas


def _resolve_ref(schema: Any, root: Mapping[str, Any]) -> Any:
    """Follow local ``#/...`` JSON pointers. An unresolvable reference yields ``None``."""
    for _ in range(_MAX_SCHEMA_DEPTH):
        if not isinstance(schema, Mapping):
            return schema
        ref = schema.get("$ref")
        if not isinstance(ref, str):
            return schema
        if ref == "#":
            schema = root
            continue
        if not ref.startswith("#/"):
            return None
        node: Any = root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        schema = node
    return None


def _enum_type(values: list[object]) -> str:
    kinds: set[str] = set()
    for value in values:
        if value is None:
            kinds.add("null")
        elif isinstance(value, bool):
            kinds.add("boolean")
        elif isinstance(value, int):
            kinds.add("integer")
        elif isinstance(value, float):
            kinds.add("number")
        elif isinstance(value, str):
            kinds.add("string")
        elif isinstance(value, list):
            kinds.add("array")
        elif isinstance(value, dict):
            kinds.add("object")
    return kinds.pop() if len(kinds) == 1 else "string"


def _schema_type(schema: Any, root: Mapping[str, Any], depth: int = 0) -> str | None:
    """Infer one conversion type from a parameter schema, as SGLang's ``infer_type_from_json_schema`` does."""
    schema = _resolve_ref(schema, root)
    if not isinstance(schema, Mapping) or depth > _MAX_SCHEMA_DEPTH:
        return None
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared.strip().lower()
    if isinstance(declared, list) and declared:
        non_null = [str(kind) for kind in declared if kind != "null"]
        return non_null[0].strip().lower() if non_null else "string"
    variants = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(variants, list):
        kinds = [kind for kind in (_schema_type(variant, root, depth + 1) for variant in variants) if kind]
        if kinds:
            distinct = set(kinds)
            if len(distinct) == 1:
                return kinds[0]
            if len(distinct) == 2 and "null" in distinct:  # noqa: PLR2004 - optional type pair
                return next(kind for kind in kinds if kind != "null")
            return "string" if "string" in distinct else kinds[0]
    enum = schema.get("enum")
    if isinstance(enum, list):
        return _enum_type(enum) if enum else "string"
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for variant in all_of:
            kind = _schema_type(variant, root, depth + 1)
            if kind and kind != "string":
                return kind
        return "string"
    if "properties" in schema:
        return "object"
    if "items" in schema:
        return "array"
    return None


def _schema_allows_null(schema: Any, root: Mapping[str, Any], depth: int = 0) -> bool:
    schema = _resolve_ref(schema, root)
    if not isinstance(schema, Mapping) or depth > _MAX_SCHEMA_DEPTH:
        return False
    declared = schema.get("type")
    if schema.get("nullable") is True or declared == "null" or (isinstance(declared, list) and "null" in declared):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    return any(
        isinstance(variants, list) and any(_schema_allows_null(variant, root, depth + 1) for variant in variants)
        for variants in (schema.get("anyOf"), schema.get("oneOf"))
    )


def _reject_constant(constant: str) -> object:
    raise ValueError(f"non-finite JSON constant {constant}")


def _loads_json(text: str) -> object:
    return json.loads(text, parse_constant=_reject_constant)


def _convert_argument(raw: str, schema: Any, root: Mapping[str, Any]) -> object:
    """Convert one XML argument value by its declared type.

    ``schema is None`` marks an undeclared tool or parameter, which keeps the raw
    text. A value that does not convert to its declared type also keeps the raw
    text, so the client's own validation can report it to the model.
    """
    if schema is None:
        return raw
    kind = _schema_type(schema, root) or "string"
    text = raw.strip()
    if text.lower() == "null" and (kind not in _STRING_TYPES or _schema_allows_null(schema, root)):
        return None
    if kind in _STRING_TYPES:
        return raw
    if kind.startswith(_INTEGER_TYPE_PREFIXES):
        try:
            return int(text)
        except ValueError:
            return raw
    if kind.startswith(_NUMBER_TYPE_PREFIXES):
        if "." not in text and "e" not in text.lower():
            try:
                return int(text)
            except ValueError:
                pass
        try:
            number = float(text)
        except ValueError:
            return raw
        return number if math.isfinite(number) else raw
    if kind in ("boolean", "bool", "binary"):
        lowered = text.lower()
        return lowered == "true" if lowered in ("true", "false") else raw
    try:
        return _loads_json(text)
    except (ValueError, RecursionError):
        pass
    if kind in _CONTAINER_TYPES or kind.startswith(_CONTAINER_TYPE_PREFIXES):
        try:
            value = ast.literal_eval(text)
            if isinstance(value, dict | list):
                json.dumps(value, allow_nan=False)
                return value
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            pass
    return raw


def _parameter_schema(schemas: ToolSchemas, name: str, key: str) -> tuple[Any, Mapping[str, Any]]:
    """Return ``(parameter schema or None, tool parameters root)`` for one argument."""
    root = schemas.get(name)
    if root is None:
        return None, {}
    properties = root.get("properties")
    if not isinstance(properties, Mapping):
        return None, root
    return properties.get(key), root


async def parse_tool_call_stream(
    chunks: AsyncIterator[GenerationChunk],
    *,
    tool_call_format: ToolCallFormat = "auto",
    parallel_tool_calls: bool = True,
    tools: Sequence[Mapping[str, Any]] | None = None,
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
            schemas=tool_parameter_schemas(tools),
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
    schemas: ToolSchemas,
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
    open_tag, close_tag = _tool_call_tags(tool_call_format)
    incoming = state.text_buffer + incoming
    state.text_buffer = ""
    cursor = 0
    while cursor < len(incoming):
        if state.in_tool_call:
            close_idx = incoming.find(close_tag, cursor)
            if close_idx == -1:
                # See the in-line comment in the original implementation
                # below — preserve the close-tag-straddle handling.
                keep = len(close_tag) - 1
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
                deltas = _tool_call_deltas(state.tool_buffer, state.tool_index, tool_call_format, schemas)
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
            cursor = close_idx + len(close_tag)
            continue

        open_idx = incoming.find(open_tag, cursor)
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
            flush_len = max(0, len(state.text_buffer) - (len(open_tag) - 1))
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
        cursor = open_idx + len(open_tag)
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
    schemas: ToolSchemas | None = None,
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
    schemas = schemas or {}
    states: dict[int, _ChoiceState] = {}

    def _state(idx: int) -> _ChoiceState:
        s = states.get(idx)
        if s is None:
            s = _ChoiceState()
            states[idx] = s
        return s

    async for chunk in chunks:
        if not chunk.text_delta and not chunk.done and chunk.finish_reason is None and not chunk.logprobs:
            yield chunk
            continue
        idx = chunk.choice_index
        if chunk.reasoning_delta:
            yield GenerationChunk(text_delta="", choice_index=idx, reasoning_delta=chunk.reasoning_delta)
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
                schemas=schemas,
            )
            for ev in emitted:
                yield ev
            if terminal:
                return
            if not (emitted or chunk.reasoning_delta or chunk.logprobs or chunk.finish_reason or chunk.done):
                yield GenerationChunk(text_delta="", choice_index=idx)
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
                    parsed_text, parsed_calls = _parse_candidate_text(cand_text, tool_call_format, schemas)
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
                    cached_tokens=chunk.cached_tokens,
                    upstream_usage=chunk.upstream_usage,
                    candidates=tuple(updated_candidates),
                    logprobs=chunk.logprobs,
                    error_code=chunk.error_code,
                    error_message=chunk.error_message,
                    upstream_reason=chunk.upstream_reason,
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
                cached_tokens=chunk.cached_tokens,
                upstream_usage=chunk.upstream_usage,
                error_code=chunk.error_code,
                error_message=chunk.error_message,
                upstream_reason=chunk.upstream_reason,
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


def _parse_candidate_text(
    text: str,
    tool_call_format: ToolCallFormat,
    schemas: ToolSchemas | None = None,
) -> tuple[str, list[dict]]:
    """Parse a single candidate's full text for ``<tool_call>`` blocks.

    Returns ``(text_outside_tool_blocks, tool_calls)`` where
    ``tool_calls`` is the OpenAI non-streaming wire shape:
    ``[{id, type, function: {name, arguments}}, ...]``.

    Used by the non-streaming ``n>1`` + tools path: the worker's
    multi-candidate adapter ships one terminal carrying the full text
    of each candidate, and each candidate needs its own tool-call
    aggregation (H5 non-streaming side). A malformed block stays in the
    candidate's text verbatim: the non-streaming path has no channel for
    a per-candidate parse error, and dropping the block would let a
    candidate whose only call was malformed finish as an ordinary answer.
    """
    open_tag, close_tag = _tool_call_tags(tool_call_format)
    if not text or open_tag not in text:
        return text, []
    out_text_parts: list[str] = []
    tool_calls: list[dict] = []
    cursor = 0
    tool_index = 0
    while cursor < len(text):
        open_idx = text.find(open_tag, cursor)
        if open_idx == -1:
            out_text_parts.append(text[cursor:])
            break
        out_text_parts.append(text[cursor:open_idx])
        body_start = open_idx + len(open_tag)
        close_idx = text.find(close_tag, body_start)
        if close_idx == -1:
            # Unterminated block: bail; surface the rest as text so the
            # candidate's prose isn't silently lost.
            out_text_parts.append(text[open_idx:])
            break
        body = text[body_start:close_idx]
        try:
            deltas = _tool_call_deltas(body, tool_index, tool_call_format, schemas)
        except ValueError:
            out_text_parts.append(text[open_idx : close_idx + len(close_tag)])
            cursor = close_idx + len(close_tag)
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
        cursor = close_idx + len(close_tag)
    return "".join(out_text_parts), tool_calls


def _tool_call_deltas(
    raw: str,
    index: int,
    tool_call_format: ToolCallFormat = "auto",
    schemas: ToolSchemas | None = None,
) -> list[ToolCallDelta]:
    raw = raw.strip()
    # Two on-the-wire formats appear inside <tool_call>…</tool_call>:
    #   1. Hermes JSON:  {"name": "...", "arguments": {...}}
    #   2. Qwen3(-Coder) XML:  <function=NAME><parameter=K>V</parameter>…</function>
    # Qwen3.5's chat template emits format (2); other models emit (1).
    # ``tool_call_format`` makes the choice explicit (config-driven);
    # ``"auto"`` falls back to the original "starts-with-<function=" heuristic.
    if tool_call_format in ("qwen_xml", "muse_atem"):
        name, arguments = _parse_xml_tool_call(raw, schemas)
    elif tool_call_format == "hermes_json":
        name, arguments = _parse_hermes_tool_call(raw)
    elif tool_call_format == "glm_xml":
        name, arguments = _parse_glm_tool_call(raw, schemas)
    elif tool_call_format == "gemma4":
        name, arguments = _parse_gemma4_tool_call(raw, schemas)
    elif raw.startswith("<function="):
        name, arguments = _parse_xml_tool_call(raw, schemas)
    else:
        name, arguments = _parse_hermes_tool_call(raw)
    if not isinstance(name, str) or not name:
        raise ValueError("tool-call payload missing string 'name'")
    if arguments is None:
        arguments = {}
    try:
        arguments_json = json.dumps(arguments, separators=(",", ":"), ensure_ascii=False)
        size = len(arguments_json.encode())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool-call 'arguments' is not JSON-serialisable: {exc}") from exc
    if size > _MAX_TOOL_ARGUMENTS_BYTES:
        raise ValueError(f"tool-call 'arguments' serialised to {size} bytes, exceeds {_MAX_TOOL_ARGUMENTS_BYTES}")
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
# Matches only a parameter *opener* — ``<parameter=name>``. A name may contain
# spaces, as JSON Schema allows. Its character classes stop at the next ``<`` or
# ``>``, so an opener that never closes cannot rescan the rest of the buffer. The
# value and closing ``</parameter>`` are located with a linear ``str.find`` below
# rather than a ``(.*?)...</parameter>`` group, which on a 256 KiB buffer of garbled
# output (many openers, no closers) is O(n²) and runs INLINE on the event loop.
_XML_PARAM_OPEN_RE = re.compile(r"<parameter=([^<>]*[^<>\s])\s*>")
_XML_PARAM_CLOSE = "</parameter>"
# Hard cap on parameters parsed from one XML tool call. Defends against an
# adversarial buffer packed with openers; well-formed calls have a handful.
_MAX_XML_PARAMS = 256


def _parse_xml_tool_call(raw: str, schemas: ToolSchemas | None = None) -> tuple[str, dict[str, object]]:
    """Parse the Qwen3(-Coder) XML tool-call form.

    Example::

        <function=get_weather>
        <parameter=city>
        Tokyo
        </parameter>
        </function>

    Returns ``(name, arguments)``. The chat template wraps every value in one
    newline on each side, so exactly one is removed from each end and the rest
    of the text is kept. Values are then converted by their declared parameter
    type in ``schemas``.
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
        val = raw[val_start:close_idx].removeprefix("\n").removesuffix("\n")
        schema, root = _parameter_schema(schemas or {}, name, key)
        arguments[key] = _convert_argument(val, schema, root)
        pos = close_idx + len(_XML_PARAM_CLOSE)
        count += 1
    return name, arguments


_GLM_KEY_OPEN = "<arg_key>"
_GLM_KEY_CLOSE = "</arg_key>"
_GLM_VALUE_OPEN = "<arg_value>"
_GLM_VALUE_CLOSE = "</arg_value>"


def _skip_whitespace(raw: str, pos: int) -> int:
    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    return pos


def _parse_glm_tool_call(raw: str, schemas: ToolSchemas | None = None) -> tuple[str, dict[str, object]]:
    """Parse the GLM tool-call form.

    Example::

        get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value>

    Returns ``(name, arguments)``; the name is the text before the first
    ``<arg_key>``. Only whitespace may separate the name and the argument
    pairs, and a name or key carrying a tag is rejected, so an unpaired or
    misplaced tag is a parse error rather than part of the call. The chat
    template writes ``<arg_value>{value}</arg_value>`` with nothing around the
    value, so the text is kept verbatim and converted by its declared parameter
    type in ``schemas``. The scan is linear, and a call with more than
    ``_MAX_XML_PARAMS`` pairs is rejected rather than truncated.
    """
    first_key = raw.find(_GLM_KEY_OPEN)
    name = (raw if first_key == -1 else raw[:first_key]).strip()
    if not name or any(char.isspace() or char in "<>" for char in name):
        raise ValueError("malformed GLM tool-call: invalid function name")
    arguments: dict[str, object] = {}
    pos = len(raw) if first_key == -1 else first_key
    count = 0
    while (pos := _skip_whitespace(raw, pos)) < len(raw) and count < _MAX_XML_PARAMS:
        if not raw.startswith(_GLM_KEY_OPEN, pos):
            raise ValueError("malformed GLM tool-call: expected <arg_key>")
        key_close = raw.find(_GLM_KEY_CLOSE, pos)
        if key_close == -1:
            raise ValueError("malformed GLM tool-call: unterminated argument")
        value_open = _skip_whitespace(raw, key_close + len(_GLM_KEY_CLOSE))
        if not raw.startswith(_GLM_VALUE_OPEN, value_open):
            raise ValueError("malformed GLM tool-call: expected <arg_value>")
        value_close = raw.find(_GLM_VALUE_CLOSE, value_open)
        if value_close == -1:
            raise ValueError("malformed GLM tool-call: unterminated argument")
        key = raw[pos + len(_GLM_KEY_OPEN) : key_close].strip()
        if "<" in key or ">" in key:
            raise ValueError("malformed GLM tool-call: invalid argument key")
        value = raw[value_open + len(_GLM_VALUE_OPEN) : value_close]
        schema, root = _parameter_schema(schemas or {}, name, key)
        arguments[key] = _convert_argument(value, schema, root)
        pos = value_close + len(_GLM_VALUE_CLOSE)
        count += 1
    if _skip_whitespace(raw, pos) < len(raw):
        raise ValueError(f"malformed GLM tool-call: more than {_MAX_XML_PARAMS} arguments")
    return name, arguments


_GEMMA_CALL_PREFIX = "call:"
_GEMMA_STRING = '<|"|>'
# Characters that end a bare name, key or scalar token. A token never contains
# ``<``, so it cannot swallow the start of a ``<|"|>`` string delimiter.
_GEMMA_NAME_STOP = frozenset("{}[]<>:,")
_GEMMA_TOKEN_STOP = frozenset('{}[]<>:,"')
_GEMMA_LITERALS: dict[str, object] = {"true": True, "false": False, "null": None}
# The JSON number grammar: no leading zeros, no bare ``1.``/``.5``, no ``NaN``.
_GEMMA_NUMBER_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
# Nesting bound for objects and arrays (the argument object is depth 1).
_MAX_GEMMA_DEPTH = _MAX_SCHEMA_DEPTH


def _gemma_error(reason: str) -> ValueError:
    return ValueError(f"malformed Gemma tool-call: {reason}")


class _GemmaArgumentParser:
    """Single-pass recursive-descent reader for a Gemma 4 argument object.

    Each method takes the position to read from and returns ``(value, next
    position)``. Every character is read once, nesting is bounded by
    ``_MAX_GEMMA_DEPTH`` and any deviation from the grammar raises.
    """

    def __init__(self, text: str, schemas: ToolSchemas, name: str) -> None:
        self.text = text
        self.schemas = schemas
        self.name = name

    def skip(self, pos: int) -> int:
        return _skip_whitespace(self.text, pos)

    def token(self, pos: int) -> tuple[str, int]:
        """Read a bare token: a run of characters that are neither whitespace nor structural."""
        text = self.text
        end = pos
        while end < len(text) and not text[end].isspace() and text[end] not in _GEMMA_TOKEN_STOP:
            end += 1
        return text[pos:end], end

    def string(self, pos: int) -> tuple[str, int]:
        """Read a ``<|"|>``-delimited string. The format has no escapes, so the text is kept exactly."""
        start = pos + len(_GEMMA_STRING)
        end = self.text.find(_GEMMA_STRING, start)
        if end == -1:
            raise _gemma_error("unterminated string")
        return self.text[start:end], end + len(_GEMMA_STRING)

    def key(self, pos: int) -> tuple[str, int]:
        if self.text.startswith(_GEMMA_STRING, pos):
            return self.string(pos)
        key, end = self.token(pos)
        if not key:
            raise _gemma_error(f"expected an argument key at offset {pos}")
        return key, end

    def read_object(self, pos: int, depth: int, *, top_level: bool = False) -> tuple[dict[str, object], int]:
        if depth > _MAX_GEMMA_DEPTH:
            raise _gemma_error(f"nesting deeper than {_MAX_GEMMA_DEPTH} levels")
        result: dict[str, object] = {}
        pos = self.skip(pos + 1)
        if self.text.startswith("}", pos):
            return result, pos + 1
        while True:
            key, pos = self.key(pos)
            pos = self.skip(pos)
            if not self.text.startswith(":", pos):
                raise _gemma_error(f"expected ':' after key {key[:64]!r}")
            pos = self.skip(pos + 1)
            if key in result:
                raise _gemma_error(f"duplicate key {key[:64]!r}")
            if top_level:
                if len(result) >= _MAX_XML_PARAMS:
                    raise _gemma_error(f"more than {_MAX_XML_PARAMS} arguments")
                result[key], pos = self.argument(pos, key)
            else:
                result[key], pos = self.value(pos, depth)
            pos = self.skip(pos)
            if self.text.startswith(",", pos):
                pos = self.skip(pos + 1)
                continue
            if self.text.startswith("}", pos):
                return result, pos + 1
            raise _gemma_error(f"expected ',' or '}}' at offset {pos}")

    def read_array(self, pos: int, depth: int) -> tuple[list[object], int]:
        if depth > _MAX_GEMMA_DEPTH:
            raise _gemma_error(f"nesting deeper than {_MAX_GEMMA_DEPTH} levels")
        result: list[object] = []
        pos = self.skip(pos + 1)
        if self.text.startswith("]", pos):
            return result, pos + 1
        while True:
            item, pos = self.value(pos, depth)
            result.append(item)
            pos = self.skip(pos)
            if self.text.startswith(",", pos):
                pos = self.skip(pos + 1)
                continue
            if self.text.startswith("]", pos):
                return result, pos + 1
            raise _gemma_error(f"expected ',' or ']' at offset {pos}")

    def value(self, pos: int, depth: int) -> tuple[object, int]:
        """Read one value held by a container at ``depth``, keeping its native type."""
        if self.text.startswith(_GEMMA_STRING, pos):
            return self.string(pos)
        if self.text.startswith("{", pos):
            return self.read_object(pos, depth + 1)
        if self.text.startswith("[", pos):
            return self.read_array(pos, depth + 1)
        token, end = self.token(pos)
        return _gemma_scalar(token, pos), end

    def argument(self, pos: int, key: str) -> tuple[object, int]:
        """Read one top-level argument, converted by its declared parameter type.

        A quoted string for a string or undeclared parameter keeps its exact
        text: unlike the XML formats, Gemma writes ``null`` unquoted, so a
        quoted ``null`` is the word. A quoted string for any other declared
        type goes through :func:`_convert_argument`, as in the XML formats. A
        bare token for a parameter declared as a string keeps its exact text
        (``zip:02134`` stays ``"02134"``); every other value keeps its native
        type.
        """
        schema, root = _parameter_schema(self.schemas, self.name, key)
        kind = None if schema is None else _schema_type(schema, root)
        if self.text.startswith(_GEMMA_STRING, pos):
            text, end = self.string(pos)
            if kind is None or kind in _STRING_TYPES:
                return text, end
            return _convert_argument(text, schema, root), end
        if kind in _STRING_TYPES:
            token, end = self.token(pos)
            if token:
                return _convert_argument(token, schema, root), end
        return self.value(pos, 1)


def _gemma_scalar(word: str, pos: int) -> object:
    """Parse a bare scalar: ``true``, ``false``, ``null`` or a JSON number. Nothing else is guessed."""
    if word in _GEMMA_LITERALS:
        return _GEMMA_LITERALS[word]
    if not word:
        raise _gemma_error(f"expected a value at offset {pos}")
    match = _GEMMA_NUMBER_RE.fullmatch(word)
    if match is None:
        raise _gemma_error(f"unquoted value {word[:64]!r}")
    try:
        if match.group(1) is None and match.group(2) is None:
            return int(word)
        number = float(word)
    except ValueError as exc:  # an integer longer than Python's digit limit
        raise _gemma_error(f"unrepresentable number {word[:64]!r}") from exc
    if not math.isfinite(number):
        raise _gemma_error(f"non-finite number {word[:64]!r}")
    return number


def _parse_gemma4_tool_call(raw: str, schemas: ToolSchemas | None = None) -> tuple[str, dict[str, object]]:
    """Parse the Gemma 4 tool-call form (the text between ``<|tool_call>`` and ``<tool_call|>``).

    Example::

        call:get_weather{city:<|"|>Paris<|"|>,days:3,opts:{metric:true}}

    The chat template renders arguments with sorted bare keys, strings between
    ``<|"|>`` delimiters (with no escape mechanism), ``true``/``false``/``null``,
    numbers, and nested ``{…}`` objects and ``[…]`` arrays. Keys may also be
    ``<|"|>``-quoted, as the template's key-escaping form writes them, and
    whitespace between tokens is ignored. The parse is strict: anything else,
    including an unquoted word, a trailing comma or text after the argument
    object, raises instead of guessing. Top-level arguments are converted by
    their declared parameter type (see :meth:`_GemmaArgumentParser.argument`);
    nested values keep their native types.

    Known limitation: the block's end is located by searching for
    ``<tool_call|>`` before this parse, so a string argument that contains that
    text ends as a parse error, never as corrupted arguments.
    """
    text = raw.strip()
    if not text.startswith(_GEMMA_CALL_PREFIX):
        raise _gemma_error("expected 'call:NAME{...}'")
    brace = text.find("{", len(_GEMMA_CALL_PREFIX))
    if brace == -1:
        raise _gemma_error("missing argument object")
    name = text[len(_GEMMA_CALL_PREFIX) : brace].rstrip()
    if not name or any(char.isspace() or char in _GEMMA_NAME_STOP for char in name):
        raise _gemma_error("invalid function name")
    parser = _GemmaArgumentParser(text, schemas or {}, name)
    arguments, end = parser.read_object(brace, 1, top_level=True)
    if parser.skip(end) != len(text):
        raise _gemma_error(f"unexpected text after the argument object at offset {end}")
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
