#!/usr/bin/env python3
"""Agent-loop conformance check for an OpenAI-compatible chat endpoint.

    uv sync
    uv run python conformance.py --model Qwen/Qwen3.5-4B:long-context          # local sie-server
    SIE_API_KEY=... uv run python conformance.py \
        --base-url "$SIE_BASE_URL/v1" --model zai-org/GLM-5.3-Flash:relay

Drives ``/v1/chat/completions`` the way a coding agent such as OpenCode does:
streaming on, tools declared on every turn, tool results sent back as
``role: tool`` messages. When the endpoint accepts ``include_reasoning``, every
tools turn carries it and the earlier turn's reasoning goes back as
``reasoning_content``, as OpenCode sends it with the SIE Cloud recipe. Every
case records what came back, so a failure says what the wire did rather than
only that it was wrong.

The check goes through the ``openai`` package on purpose. The thing under test
is the OpenAI-compatible wire surface that OpenCode's provider speaks, so the
reference client for that surface is the right instrument here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
import openai

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_REPORT = Path("run-output/conformance.json")
REQUEST_TIMEOUT_S = 900.0
WARM_UP_TIMEOUT_S = 900.0
WARM_UP_POLL_S = 5.0
WARM_UP_CODES = frozenset({"MODEL_LOADING", "provisioning"})
BUSY_CODES = frozenset({"QUEUE_FULL", "RESOURCE_EXHAUSTED", "rate_limit_exceeded"})
BUSY_RETRIES = 4
BUSY_WAIT_CAP_S = 180.0
RETRY_AFTER_DEFAULT_S = 5.0
RETRY_AFTER_MAX_S = 60.0
GENERATION_BUDGET = 4096
HIDDEN_TOKEN_SLACK = 16
LONG_PROMPT_MIN_TOKENS = 65_536
LONG_PROMPT_WORDS = 68_000
NEEDLE = "The access code is 7741-ZEBRA."
RECORDED_HEADERS = ("x-sie-served-by", "x-sie-model-revision")
REASONING_QUESTION = "Which is larger, 9.11 or 9.9? Answer with just the larger number."
LONG_CALL_MIN_BYTES = 65_536
LONG_CALL_BUDGET = 32_000
LONG_CALL_LINES = 1_320
LONG_CALL_SENTENCE = "The quick brown fox jumps over the lazy dog."
LONG_CALL_PROMPT = (
    f"Use the write_file tool once to create the file notes/pangrams.txt. Its content must be {LONG_CALL_LINES:,} "
    f"lines numbered 0001 to {LONG_CALL_LINES:04d}: each line is its four-digit number, one space and the sentence "
    f'"{LONG_CALL_SENTENCE}", about {LONG_CALL_LINES * (len(LONG_CALL_SENTENCE) + 6):,} characters in all. Write '
    "every line out in full in that one call: no ellipsis, no placeholder, no summary and no code that generates "
    "the lines."
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, for example Paris"}},
            "required": ["city"],
        },
    },
}

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or replace one text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the project root"},
                "content": {"type": "string", "description": "The whole file content"},
            },
            "required": ["path", "content"],
        },
    },
}

FILLER_WORDS = [
    "the",
    "ledger",
    "records",
    "every",
    "shipment",
    "that",
    "left",
    "the",
    "warehouse",
    "before",
    "noon",
    "and",
    "the",
    "clerk",
    "signs",
    "each",
    "page",
    "once",
    "the",
    "totals",
    "agree",
    "with",
    "the",
    "manifest",
    "while",
    "the",
    "driver",
    "waits",
    "by",
    "the",
    "gate",
    "for",
    "the",
    "stamped",
    "copy",
]


@dataclass
class ToolCall:
    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    first_delta_complete: bool = False
    deltas: int = 0

    def parsed_arguments(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


@dataclass
class StreamResult:
    content: str = ""
    reasoning: str = ""
    reasoning_deltas: int = 0
    tool_calls: dict[int, ToolCall] = field(default_factory=dict)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    usage_on_last_chunk: bool = False
    chunks: int = 0
    headers: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    retries: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    first_chunk_s: float | None = None
    stream_s: float = 0.0
    longest_gap_s: float = 0.0

    def calls(self) -> list[ToolCall]:
        return [self.tool_calls[index] for index in sorted(self.tool_calls)]

    def observed(self) -> dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "chunks": self.chunks,
            "content": self.content[:200],
            "reasoning_deltas": self.reasoning_deltas,
            "reasoning_chars": len(self.reasoning),
            "tool_calls": [
                {
                    "index": call.index,
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments[:200],
                    "deltas": call.deltas,
                    "first_delta_has_id_and_name": call.first_delta_complete,
                }
                for call in self.calls()
            ],
            "usage": self.usage,
            "usage_on_last_chunk": self.usage_on_last_chunk,
            "seconds": round(self.seconds, 2),
            "error": self.error,
            "retries": self.retries,
            "headers": self.headers,
        }

    def hidden_output_tokens(self) -> int | None:
        completion = (self.usage or {}).get("completion_tokens")
        if not isinstance(completion, int):
            return None
        texts = [self.content, self.reasoning, *(call.arguments for call in self.calls())]
        return completion - sum(len(text.encode()) for text in texts)


@dataclass
class Case:
    name: str
    status: str
    detail: str
    observed: dict[str, Any]


def _extra(obj: Any, key: str) -> Any:
    value = getattr(obj, key, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None)
    return extra.get(key) if isinstance(extra, dict) else None


def _seconds(value: Any) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _retry_after(value: str | None) -> float | None:
    seconds = _seconds(value)
    if seconds is not None or value is None:
        return seconds
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    remaining = (when - datetime.now(UTC)).total_seconds()
    return remaining if remaining > 0 else None


def _error_payload(exc: Exception) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    body = body if isinstance(body, dict) else {}
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    status_error = isinstance(exc, openai.APIStatusError)
    retry_after = _retry_after(exc.response.headers.get("retry-after")) if status_error else None
    return {
        "status": exc.status_code if status_error else None,
        "code": inner.get("code") or (None if status_error else type(exc).__name__),
        "param": inner.get("param"),
        "message": inner.get("message") or str(exc),
        "retry_after_s": retry_after if retry_after is not None else _seconds(inner.get("retry_after_s")),
    }


def _is_busy(error: dict[str, Any] | None) -> bool:
    return error is not None and error["status"] in (None, 429, 503) and error["code"] in BUSY_CODES


def _delivered(result: StreamResult) -> bool:
    return bool(result.content or result.reasoning or result.tool_calls)


def _busy_retry(result: StreamResult, retries: list[dict[str, Any]]) -> dict[str, Any] | None:
    error = result.error
    if error is None or not _is_busy(error) or _delivered(result) or len(retries) >= BUSY_RETRIES:
        return None
    hint = error["retry_after_s"]
    wait = min(RETRY_AFTER_DEFAULT_S * 2 ** len(retries) if hint is None else hint, RETRY_AFTER_MAX_S)
    if sum(retry["waited_s"] for retry in retries) + wait > BUSY_WAIT_CAP_S:
        return None
    return {**error, "waited_s": wait}


def _retrying(send: Callable[[], StreamResult]) -> StreamResult:
    started = time.monotonic()
    retries: list[dict[str, Any]] = []
    while True:
        result = send()
        retry = _busy_retry(result, retries)
        if retry is None:
            result.retries = retries
            result.seconds = time.monotonic() - started
            return result
        retries.append(retry)
        time.sleep(retry["waited_s"])


def _still_busy(result: StreamResult) -> str:
    error = result.error or {}
    refusal = f"{error.get('status')} {error.get('code')}: {error.get('message')}"
    if _delivered(result):
        return f"busy after some output, not retried: {refusal}"
    waited = sum(retry["waited_s"] for retry in result.retries)
    return f"still busy after {len(result.retries)} retries and {waited:g} s of waiting: {refusal}"


def _keep_headers(result: StreamResult, raw: Any) -> None:
    result.headers = {name: raw.headers[name] for name in RECORDED_HEADERS if name in raw.headers}


def stream_chat(client: openai.OpenAI, model: str, messages: list[dict[str, Any]], **params: Any) -> StreamResult:
    return _retrying(lambda: _stream_once(client, model, messages, **params))


def complete_chat(client: openai.OpenAI, model: str, messages: list[dict[str, Any]], **params: Any) -> StreamResult:
    return _retrying(lambda: _complete_once(client, model, messages, **params))


def _stream_once(client: openai.OpenAI, model: str, messages: list[dict[str, Any]], **params: Any) -> StreamResult:
    result = StreamResult()
    started = previous = time.monotonic()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        _keep_headers(result, raw)
        for chunk in raw.parse():
            now = time.monotonic()
            if result.first_chunk_s is None:
                result.first_chunk_s = now - started
            else:
                result.longest_gap_s = max(result.longest_gap_s, now - previous)
                result.stream_s = now - started - result.first_chunk_s
            previous = now
            result.chunks += 1
            result.usage_on_last_chunk = False
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                result.usage = usage.model_dump(exclude_none=True)
                result.usage_on_last_chunk = True
            for choice in chunk.choices or []:
                if choice.index != 0:
                    continue
                if choice.finish_reason:
                    result.finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.content:
                    result.content += delta.content
                reasoning = _extra(delta, "reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    result.reasoning += reasoning
                    result.reasoning_deltas += 1
                for piece in delta.tool_calls or []:
                    call = result.tool_calls.setdefault(piece.index, ToolCall(index=piece.index))
                    call.deltas += 1
                    if call.deltas == 1:
                        call.first_delta_complete = bool(piece.id) and bool(piece.function and piece.function.name)
                    if piece.id:
                        call.id = piece.id
                    if piece.function and piece.function.name:
                        call.name = piece.function.name
                    if piece.function and piece.function.arguments:
                        call.arguments += piece.function.arguments
    except (openai.APIError, httpx.HTTPError) as exc:
        result.error = _error_payload(exc)
    return result


def _complete_once(client: openai.OpenAI, model: str, messages: list[dict[str, Any]], **params: Any) -> StreamResult:
    result = StreamResult()
    try:
        raw = client.chat.completions.with_raw_response.create(model=model, messages=messages, **params)
        _keep_headers(result, raw)
        completion = raw.parse()
        if completion.usage is not None:
            result.usage = completion.usage.model_dump(exclude_none=True)
        for choice in completion.choices or []:
            if choice.index != 0:
                continue
            result.finish_reason = choice.finish_reason
            result.content = choice.message.content or ""
            reasoning = _extra(choice.message, "reasoning_content")
            if isinstance(reasoning, str):
                result.reasoning = reasoning
    except (openai.APIError, httpx.HTTPError) as exc:
        result.error = _error_payload(exc)
    return result


def _describe_calls(result: StreamResult) -> str:
    return ", ".join(f"{call.name}({call.arguments})" for call in result.calls()) or "no tool call"


def _fail_on_error(name: str, result: StreamResult) -> Case | None:
    if result.error is None:
        return None
    if _is_busy(result.error):
        return Case(name, "capacity", _still_busy(result), result.observed())
    error = result.error
    return Case(
        name, "fail", f"request failed: {error['status']} {error['code']}: {error['message']}", result.observed()
    )


def _is_refusal(error: dict[str, Any]) -> bool:
    return error.get("status") in (400, 422)


def _refuses_field(error: dict[str, Any], name: str) -> bool:
    return _is_refusal(error) and (error.get("param") == name or name in str(error.get("message") or ""))


def case_streaming_tool_call_deltas(
    client: openai.OpenAI, model: str, tool_params: dict[str, Any]
) -> tuple[Case, StreamResult]:
    name = "streaming_tool_call_deltas"
    messages = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
    result = stream_chat(
        client, model, messages, tools=[WEATHER_TOOL], max_tokens=GENERATION_BUDGET, temperature=0, **tool_params
    )
    failed = _fail_on_error(name, result)
    if failed:
        return failed, result
    calls = result.calls()
    problems = []
    if not calls:
        problems.append("no tool call came back")
    for call in calls:
        if not call.id:
            problems.append(f"call {call.index} has no id")
        if call.name != "get_weather":
            problems.append(f"call {call.index} named {call.name!r}")
        parsed = call.parsed_arguments()
        if parsed is None:
            problems.append(f"call {call.index} arguments are not a JSON object: {call.arguments[:80]!r}")
        elif "city" not in parsed:
            problems.append(f"call {call.index} arguments lack 'city': {parsed}")
        if not call.first_delta_complete:
            problems.append(f"call {call.index}: first delta lacks id or name")
    if result.finish_reason != "tool_calls":
        problems.append(f"finish_reason is {result.finish_reason!r}, expected 'tool_calls'")
    if problems:
        return Case(name, "fail", "; ".join(problems), result.observed()), result
    deltas = sum(call.deltas for call in calls)
    detail = f"{len(calls)} call: {_describe_calls(result)}; {deltas} deltas; finish_reason=tool_calls"
    return Case(name, "pass", detail, result.observed()), result


def case_parallel_tool_calls(client: openai.OpenAI, model: str, tool_params: dict[str, Any]) -> Case:
    name = "parallel_tool_calls"
    messages = [
        {
            "role": "user",
            "content": "Get the current weather in Paris and in Tokyo. Call get_weather once per city, both in this turn.",
        }
    ]
    result = stream_chat(
        client, model, messages, tools=[WEATHER_TOOL], max_tokens=GENERATION_BUDGET, temperature=0, **tool_params
    )
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    calls = result.calls()
    ids = [call.id for call in calls]
    cities = sorted(
        str((call.parsed_arguments() or {}).get("city", "")).lower() for call in calls if call.parsed_arguments()
    )
    observed = result.observed()
    if len(calls) < 2:
        return Case(name, "fail", f"{len(calls)} tool call in the turn: {_describe_calls(result)}", observed)
    if len(set(ids)) != len(ids) or None in ids:
        return Case(name, "fail", f"tool-call ids are not distinct: {ids}", observed)
    missing = [city for city in ("paris", "tokyo") if not any(city in c for c in cities)]
    if missing:
        return Case(name, "fail", f"{len(calls)} calls but cities {cities} miss {missing}", observed)
    return Case(name, "pass", f"{len(calls)} calls in one turn: {_describe_calls(result)}", observed)


def case_tool_result_round_trip(
    client: openai.OpenAI, model: str, first: StreamResult, tool_params: dict[str, Any]
) -> Case:
    name = "tool_result_round_trip"
    calls = first.calls()
    if _is_busy(first.error):
        return Case(name, "capacity", "not run, streaming_tool_call_deltas stayed busy", first.observed())
    if first.error or not calls:
        return Case(name, "fail", "no tool call from streaming_tool_call_deltas to send a result for", first.observed())
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": first.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    }
    if first.reasoning or "extra_body" in tool_params:
        assistant["reasoning_content"] = first.reasoning
    tool_messages = [
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps({"city": "Paris", "temperature_c": 18, "conditions": "overcast"}),
        }
        for call in calls
    ]
    messages = [
        {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."},
        assistant,
        *tool_messages,
    ]
    result = stream_chat(
        client, model, messages, tools=[WEATHER_TOOL], max_tokens=GENERATION_BUDGET, temperature=0, **tool_params
    )
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    observed = result.observed()
    observed["reasoning_content_sent_back"] = assistant.get("reasoning_content")
    problems = []
    if result.calls():
        problems.append(f"answered with another tool call: {_describe_calls(result)}")
    if result.finish_reason != "stop":
        problems.append(f"finish_reason is {result.finish_reason!r}, expected 'stop'")
    if not result.content.strip():
        problems.append("empty final answer")
    elif "18" not in result.content:
        problems.append(f"final answer does not carry the tool's 18: {result.content[:120]!r}")
    if problems:
        return Case(name, "fail", "; ".join(problems), observed)
    return Case(name, "pass", f"final answer used the tool result: {result.content.strip()[:100]!r}", observed)


def long_prompt() -> str:
    words = [FILLER_WORDS[i % len(FILLER_WORDS)] for i in range(LONG_PROMPT_WORDS)]
    lines = [" ".join(words[i : i + 20]) + "." for i in range(0, len(words), 20)]
    lines.insert(int(len(lines) * 0.6), NEEDLE)
    return "\n".join(lines)


def declared_window(entry: dict[str, Any] | None) -> int | None:
    for key in ("max_sequence_length", "context_length"):
        value = (entry or {}).get(key)
        if isinstance(value, int):
            return value
    return None


def case_prompt_above_64k_tokens(client: openai.OpenAI, model: str, entry: dict[str, Any] | None) -> Case:
    name = "prompt_above_64k_tokens"
    document = long_prompt()
    messages = [
        {
            "role": "user",
            "content": f"Read this warehouse log, then answer.\n\n{document}\n\nWhat is the access code? Reply with just the code.",
        }
    ]
    result = stream_chat(client, model, messages, max_tokens=GENERATION_BUDGET, temperature=0)
    observed = result.observed()
    observed["prompt_chars"] = len(messages[0]["content"])
    observed["needle_found"] = "7741-ZEBRA" in result.content
    window = declared_window(entry)
    observed["declared_window"] = window
    if window is not None and window < LONG_PROMPT_MIN_TOKENS:
        error = result.error
        if error is None:
            prompt_tokens = (result.usage or {}).get("prompt_tokens")
            detail = f"declared window {window} is under {LONG_PROMPT_MIN_TOKENS}, yet {prompt_tokens} prompt tokens were accepted"
            return Case(name, "recorded", detail, observed)
        if _is_refusal(error):
            detail = f"declared window {window} is under {LONG_PROMPT_MIN_TOKENS}; refused: {error['status']} {error['code']}: {error['message']}"
            return Case(name, "recorded", detail, observed)
    failed = _fail_on_error(name, result)
    if failed:
        failed.observed = observed
        return failed
    prompt_tokens = (result.usage or {}).get("prompt_tokens")
    if prompt_tokens is None:
        return Case(name, "fail", "accepted, but no usage.prompt_tokens to confirm the prompt size", observed)
    if prompt_tokens < LONG_PROMPT_MIN_TOKENS:
        return Case(name, "fail", f"prompt counted as {prompt_tokens} tokens, under {LONG_PROMPT_MIN_TOKENS}", observed)
    if not result.content.strip() or result.finish_reason != "stop":
        return Case(
            name, "fail", f"accepted {prompt_tokens} tokens but finish_reason={result.finish_reason!r}", observed
        )
    needle = "needle found" if observed["needle_found"] else "needle not found"
    return Case(name, "pass", f"{prompt_tokens} prompt tokens accepted and answered; {needle}", observed)


def case_finish_reason_stop(client: openai.OpenAI, model: str) -> tuple[Case, StreamResult]:
    name = "finish_reason_stop"
    messages = [{"role": "user", "content": "Reply with the single word: ready."}]
    result = stream_chat(client, model, messages, max_tokens=GENERATION_BUDGET, temperature=0)
    failed = _fail_on_error(name, result)
    if failed:
        return failed, result
    if result.finish_reason != "stop":
        return Case(name, "fail", f"finish_reason is {result.finish_reason!r}", result.observed()), result
    return Case(name, "pass", f"finish_reason=stop after {result.content.strip()[:40]!r}", result.observed()), result


def case_finish_reason_tool_calls(first: Case) -> Case:
    name = "finish_reason_tool_calls"
    finish = first.observed.get("finish_reason")
    if first.status == "capacity":
        return Case(name, "capacity", "not judged, the tool-call turn stayed busy", {"finish_reason": finish})
    if finish == "tool_calls":
        return Case(name, "pass", "finish_reason=tool_calls on the tool-call turn", {"finish_reason": finish})
    return Case(name, "fail", f"tool-call turn ended with finish_reason={finish!r}", {"finish_reason": finish})


def case_finish_reason_length(client: openai.OpenAI, model: str) -> Case:
    name = "finish_reason_length"
    messages = [{"role": "user", "content": "Write three paragraphs about the sea."}]
    result = stream_chat(client, model, messages, max_tokens=8, temperature=0)
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    if result.finish_reason != "length":
        return Case(name, "fail", f"finish_reason is {result.finish_reason!r} with max_tokens=8", result.observed())
    completion = (result.usage or {}).get("completion_tokens")
    return Case(
        name, "pass", f"finish_reason=length at max_tokens=8 (completion_tokens={completion})", result.observed()
    )


def case_usage_on_final_chunk(stop: StreamResult) -> Case:
    name = "usage_on_final_chunk"
    observed = {"usage": stop.usage, "usage_on_last_chunk": stop.usage_on_last_chunk, "chunks": stop.chunks}
    if _is_busy(stop.error):
        return Case(name, "capacity", "not judged, the stop stream stayed busy", observed)
    if stop.error:
        return Case(name, "fail", "the stop stream failed, see finish_reason_stop", observed)
    usage = stop.usage or {}
    if not usage:
        return Case(name, "fail", "no usage object on any chunk with stream_options.include_usage", observed)
    if not stop.usage_on_last_chunk:
        return Case(name, "fail", "usage arrived, but not on the final chunk", observed)
    missing = [key for key in ("prompt_tokens", "completion_tokens") if not usage.get(key)]
    if missing:
        return Case(name, "fail", f"usage lacks {missing}: {usage}", observed)
    return Case(name, "pass", f"final chunk carries usage {usage}", observed)


def _hid_reasoning(result: StreamResult) -> bool:
    hidden = result.hidden_output_tokens()
    return hidden is not None and hidden > HIDDEN_TOKEN_SLACK


def _too_short_to_have_reasoned(result: StreamResult) -> bool:
    completion = (result.usage or {}).get("completion_tokens")
    return isinstance(completion, int) and completion <= HIDDEN_TOKEN_SLACK


def case_include_reasoning(client: openai.OpenAI, model: str) -> tuple[Case, bool]:
    name = "include_reasoning"
    messages = [{"role": "user", "content": REASONING_QUESTION}]
    flag = {"extra_body": {"include_reasoning": True}}
    with_flag = stream_chat(client, model, messages, max_tokens=GENERATION_BUDGET, temperature=0, **flag)
    without_flag = stream_chat(client, model, messages, max_tokens=GENERATION_BUDGET, temperature=0)
    whole = complete_chat(client, model, messages, max_tokens=GENERATION_BUDGET, temperature=0, **flag)
    contents = (with_flag.content, without_flag.content, whole.content)
    observed = {
        "with_include_reasoning": with_flag.observed(),
        "without_include_reasoning": without_flag.observed(),
        "non_streamed_with_include_reasoning": {
            "finish_reason": whole.finish_reason,
            "content": whole.content[:200],
            "reasoning_chars": len(whole.reasoning),
            "usage": whole.usage,
            "error": whole.error,
            "retries": whole.retries,
        },
        "content_has_think_tags": any("<think>" in text or "</think>" in text for text in contents),
    }
    accepted = with_flag.error is None
    for label, result in (("include_reasoning", with_flag), ("plain request", without_flag)):
        if result.error and not _is_busy(result.error):
            error = result.error
            verdict = "refused" if label == "include_reasoning" else "failed"
            detail = f"{label} {verdict}: {error['status']} {error['code']}: {error['message']}"
            return Case(name, "fail", detail, observed), accepted
    if whole.error and not _is_busy(whole.error):
        error = whole.error
        detail = f"non-streamed include_reasoning failed: {error['status']} {error['code']}: {error['message']}"
        return Case(name, "fail", detail, observed), accepted
    busy = next((result for result in (with_flag, without_flag, whole) if _is_busy(result.error)), None)
    if busy is not None:
        return Case(name, "capacity", _still_busy(busy), observed), accepted
    if observed["content_has_think_tags"]:
        return Case(name, "fail", "raw <think> tags leaked into content", observed), accepted
    if without_flag.reasoning_deltas:
        detail = f"reasoning_content arrived without the flag ({without_flag.reasoning_deltas} deltas)"
        return Case(name, "fail", detail, observed), accepted
    results = (with_flag, without_flag, whole)
    reasoned = bool(with_flag.reasoning or whole.reasoning) or any(_hid_reasoning(result) for result in results)
    if not reasoned and not all(_too_short_to_have_reasoned(result) for result in results):
        observed["model_reasoned"] = None
        spent = [(result.usage or {}).get("completion_tokens") for result in results]
        detail = (
            f"no reasoning came back, and the visible text could account for the completion tokens {spent}, "
            "so whether the model reasoned is unknown"
        )
        return Case(name, "recorded", detail, observed), accepted
    observed["model_reasoned"] = reasoned
    if not reasoned:
        completion = (without_flag.usage or {}).get("completion_tokens")
        detail = f"the model did not reason here ({completion} completion tokens for {without_flag.content.strip()[:20]!r}); no reasoning to return"
        return Case(name, "pass", detail, observed), accepted
    problems = []
    if not with_flag.reasoning:
        problems.append("no delta.reasoning_content in the stream")
    if not whole.reasoning:
        problems.append("no message.reasoning_content in the non-streamed answer")
    if problems:
        hidden = max((r.hidden_output_tokens() or 0) for r in results)
        detail = f"the model reasoned (at least {hidden} completion tokens beyond the visible text) but with include_reasoning: {'; '.join(problems)}"
        return Case(name, "fail", detail, observed), accepted
    detail = (
        f"{with_flag.reasoning_deltas} reasoning_content deltas, {len(with_flag.reasoning)} chars; "
        f"message.reasoning_content {len(whole.reasoning)} chars; none without the flag"
    )
    return Case(name, "pass", detail, observed), accepted


def case_tool_choice_required(client: openai.OpenAI, model: str) -> Case:
    name = "tool_choice_required"
    messages = [{"role": "user", "content": "Say hello."}]
    result = stream_chat(
        client,
        model,
        messages,
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_tokens=GENERATION_BUDGET,
        temperature=0,
    )
    observed = result.observed()
    if result.error:
        error = result.error
        if _is_busy(error):
            return Case(name, "capacity", _still_busy(result), observed)
        if _is_refusal(error):
            observed["support"] = "refused"
            return Case(name, "recorded", f"refused: {error['status']} {error['code']}: {error['message']}", observed)
        return Case(name, "fail", f"request failed: {error['status']} {error['code']}: {error['message']}", observed)
    if result.calls():
        observed["support"] = "supported"
        return Case(name, "recorded", f"supported: forced {_describe_calls(result)}", observed)
    observed["support"] = "accepted but not forced"
    return Case(
        name, "fail", f"accepted tool_choice=required but answered without a call: {result.content[:80]!r}", observed
    )


def long_call_line(number: int) -> str:
    return f"{number:04d} {LONG_CALL_SENTENCE}"


def case_long_tool_call(client: openai.OpenAI, model: str, tool_params: dict[str, Any]) -> Case:
    name = "long_tool_call"
    messages = [{"role": "user", "content": LONG_CALL_PROMPT}]
    result = stream_chat(
        client, model, messages, tools=[WRITE_TOOL], max_tokens=LONG_CALL_BUDGET, temperature=0, **tool_params
    )
    calls = result.calls()
    size = sum(len(call.arguments.encode()) for call in calls)
    observed = result.observed()
    observed.update(
        max_tokens=LONG_CALL_BUDGET,
        argument_bytes=size,
        first_chunk_s=None if result.first_chunk_s is None else round(result.first_chunk_s, 2),
        stream_s=round(result.stream_s, 2),
        longest_gap_s=round(result.longest_gap_s, 2),
    )
    timing = f"streamed over {result.stream_s:.0f} s, longest gap between chunks {result.longest_gap_s:.1f} s"
    if result.error and _refuses_field(result.error, "max_tokens"):
        error = result.error
        detail = (
            f"not run: max_tokens={LONG_CALL_BUDGET} refused: {error['status']} {error['code']}: {error['message']}"
        )
        return Case(name, "recorded", detail, observed)
    failed = _fail_on_error(name, result)
    if failed:
        failed.observed = observed
        failed.detail = f"{failed.detail} ({timing})"
        return failed
    if not calls:
        detail = f"the model answered without a tool call: {result.content.strip()[:80]!r}"
        return Case(name, "recorded", detail, observed)
    arguments = [call.parsed_arguments() for call in calls]
    if result.finish_reason == "length":
        detail = f"max_tokens={LONG_CALL_BUDGET} ended the turn after {size:,} bytes of arguments ({timing})"
        return Case(name, "recorded", detail, observed)
    if None in arguments:
        detail = f"{size:,} bytes of arguments, but not a JSON object: truncated or corrupted ({timing})"
        return Case(name, "fail", detail, observed)
    if result.finish_reason != "tool_calls":
        detail = f"finish_reason is {result.finish_reason!r} after a complete call, expected 'tool_calls'"
        return Case(name, "fail", detail, observed)
    content = next((args.get("content") for args in arguments if args and isinstance(args.get("content"), str)), "")
    observed["content_lines"] = len(content.splitlines())
    observed["lines_as_asked"] = sum(
        line.strip() == long_call_line(number) for number, line in enumerate(content.splitlines(), 1)
    )
    kib = size / 1024
    if size < LONG_CALL_MIN_BYTES:
        detail = f"the model wrote {kib:.1f} KiB of arguments, under the {LONG_CALL_MIN_BYTES // 1024} KiB asked for"
        return Case(name, "recorded", detail, observed)
    detail = f"{kib:.1f} KiB of arguments in {len(calls)} call ({observed['content_lines']} lines), {timing}"
    return Case(name, "pass", detail, observed)


def model_entry(base_url: str, api_key: str, model: str) -> dict[str, Any] | None:
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=30.0)
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if isinstance(payload, dict):
        entries = payload.get("models") or payload.get("data") or []
    else:
        entries = payload
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and (entry.get("name") == model or entry.get("id") == model):
            return {
                key: entry.get(key)
                for key in ("name", "id", "capabilities", "max_sequence_length", "context_length", "max_output_tokens")
                if key in entry
            }
    return None


def wait_until_loaded(client: openai.OpenAI, model: str) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    messages = [{"role": "user", "content": "Reply with the single word: ready."}]
    while True:
        attempts += 1
        result = stream_chat(client, model, messages, max_tokens=4, temperature=0)
        elapsed = time.monotonic() - started
        if (result.error or {}).get("code") not in WARM_UP_CODES or elapsed > WARM_UP_TIMEOUT_S:
            return {"attempts": attempts, "seconds": round(elapsed, 1), "error": result.error}
        time.sleep(WARM_UP_POLL_S)


def run(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    client = openai.OpenAI(base_url=base_url, api_key=api_key or "none", timeout=REQUEST_TIMEOUT_S, max_retries=0)
    started = datetime.now(UTC)
    entry = model_entry(base_url, api_key, model)
    warm_up = wait_until_loaded(client, model)
    cases: list[Case] = []

    reasoning_case, reasoning_accepted = case_include_reasoning(client, model)
    tool_params: dict[str, Any] = {"tool_choice": "auto"}
    if reasoning_accepted:
        tool_params["extra_body"] = {"include_reasoning": True}
    tool_case, tool_result = case_streaming_tool_call_deltas(client, model, tool_params)
    if tool_result.error and _refuses_field(tool_result.error, "tool_choice"):
        error = tool_result.error
        detail = (
            "OpenCode sends tool_choice=auto on every tools turn; "
            f"refused: {error['status']} {error['code']}: {error['message']}"
        )
        cases.append(Case("tool_choice_auto", "fail", detail, tool_result.observed()))
        tool_params.pop("tool_choice")
        tool_case, tool_result = case_streaming_tool_call_deltas(client, model, tool_params)
    elif tool_result.error:
        error = tool_result.error
        detail = f"not judged, the tools turn failed: {error['status']} {error['code']}: {error['message']}"
        cases.append(Case("tool_choice_auto", "recorded", detail, tool_result.observed()))
    else:
        observed = {"tool_choice": "auto", "finish_reason": tool_result.finish_reason}
        cases.append(Case("tool_choice_auto", "pass", "tool_choice=auto accepted on the tools turn", observed))
    cases.append(tool_case)
    cases.append(case_parallel_tool_calls(client, model, tool_params))
    cases.append(case_tool_result_round_trip(client, model, tool_result, tool_params))
    cases.append(case_prompt_above_64k_tokens(client, model, entry))
    stop_case, stop_result = case_finish_reason_stop(client, model)
    cases.append(stop_case)
    cases.append(case_finish_reason_tool_calls(tool_case))
    cases.append(case_finish_reason_length(client, model))
    cases.append(case_usage_on_final_chunk(stop_result))
    cases.append(reasoning_case)
    cases.append(case_tool_choice_required(client, model))
    cases.append(case_long_tool_call(client, model, tool_params))

    statuses = ("pass", "fail", "recorded", "capacity")
    counts = {status: sum(1 for case in cases if case.status == status) for status in statuses}
    return {
        "base_url": base_url,
        "model": model,
        "model_entry": entry,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "served_by": tool_result.headers.get("x-sie-served-by"),
        "warm_up": warm_up,
        "tools_turn_params": tool_params,
        "cases": [{**case.__dict__, "retries": _retries_in(case.observed)} for case in cases],
        "summary": counts,
    }


def _retries_in(observed: dict[str, Any]) -> int:
    nested = sum(_retries_in(value) for value in observed.values() if isinstance(value, dict))
    return len(observed.get("retries") or []) + nested


def print_table(report: dict[str, Any]) -> None:
    width = max(len(case["name"]) for case in report["cases"])
    print(f"model {report['model']} at {report['base_url']}")
    if report.get("served_by"):
        print(f"served by {report['served_by']}")
    warm_up = report["warm_up"]
    print(f"warm-up: {warm_up['attempts']} attempt(s), {warm_up['seconds']}s")
    print()
    for case in report["cases"]:
        retried = f" ({case['retries']} busy retries)" if case["retries"] else ""
        print(f"  {case['name']:<{width}}  {case['status']:<8}  {case['detail']}{retried}")
    summary = report["summary"]
    capacity = f", {summary['capacity']} capacity" if summary["capacity"] else ""
    print()
    print(f"{summary['pass']} pass, {summary['fail']} fail, {summary['recorded']} recorded{capacity}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})"
    )
    parser.add_argument("--api-key", default=os.environ.get("SIE_API_KEY", ""), help="Bearer key (env SIE_API_KEY)")
    parser.add_argument("--model", required=True, help="model id, for example zai-org/GLM-5.3-Flash:relay")
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help=f"JSON report path (default {DEFAULT_REPORT})"
    )
    args = parser.parse_args()

    report = run(args.base_url, args.api_key, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"report written to {args.report}")
    return 1 if report["summary"]["fail"] or report["summary"]["capacity"] else 0


if __name__ == "__main__":
    sys.exit(main())
