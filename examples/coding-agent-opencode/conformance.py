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

``--api messages`` checks ``/v1/messages`` instead, the Anthropic Messages wire
that Claude Code speaks: the same agent-loop cases over ``tool_use`` and
``tool_result`` blocks, plus the event grammar, the inert fields Claude Code
sends and the refusal wording it recovers from. It reads the raw event stream
with ``httpx``, because the event sequence is what the client depends on.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, cast

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
MESSAGES_HEADERS = {
    "anthropic-version": "2023-06-01",
    "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14",
}
MESSAGES_BUSY_TYPES = frozenset({"overloaded_error", "rate_limit_error"})
MESSAGES_WARM_UP_CODES = frozenset({"MODEL_LOADING", "PROVISIONING"})
MISSING_MODEL = "sie-conformance/no-such-model"
EFFORT_REFUSAL = "output_config.effort: Extra inputs are not permitted"
SYSTEM_TURN_REFUSAL = re.compile(r"role .{0,2}system", re.IGNORECASE)
PROMPT_TOO_LONG = re.compile(r"^prompt is too long(: \d+ tokens > \d+ maximum)?$")

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
    if error is None or error["status"] not in (None, 429, 503, 529):
        return False
    return error["code"] in BUSY_CODES or error.get("type") in MESSAGES_BUSY_TYPES


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


# --- Anthropic Messages (`--api messages`) ---------------------------------

MESSAGES_TOOL = {
    "name": "get_weather",
    "description": WEATHER_TOOL["function"]["description"],
    "input_schema": WEATHER_TOOL["function"]["parameters"],
}


@dataclass
class MessagesResult(StreamResult):
    """A ``/v1/messages`` answer. ``finish_reason`` holds ``stop_reason``, and
    ``usage_on_last_chunk`` says that usage arrived on ``message_delta``."""

    events: list[dict[str, Any]] = field(default_factory=list)
    pings: int = 0
    start_usage: dict[str, Any] | None = None
    status: int | None = None

    def observed(self) -> dict[str, Any]:
        observed = super().observed()
        observed["stop_reason"] = observed.pop("finish_reason")
        observed["usage_on_message_delta"] = observed.pop("usage_on_last_chunk")
        observed["status"] = self.status
        observed["message_start_usage"] = self.start_usage
        observed["events"] = len(self.events)
        observed["pings"] = self.pings
        return observed


def _messages_error(status: int | None, body: Any, headers: httpx.Headers | None) -> dict[str, Any]:
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
    sie_code = headers.get("x-sie-error-code") if headers is not None else None
    retry_after = _retry_after(headers.get("retry-after")) if headers is not None else None
    return {
        "status": status,
        "code": sie_code or error.get("type"),
        "type": error.get("type"),
        "param": None,
        "message": error.get("message") or (json.dumps(body)[:200] if body else None),
        "retry_after_s": retry_after,
    }


def _messages_event(result: MessagesResult, name: str, data: dict[str, Any]) -> None:
    summary: dict[str, Any] = {"event": name}
    if "index" in data:
        summary["index"] = data["index"]
    if name == "ping":
        result.pings += 1
    elif name == "message_start":
        result.start_usage = (data.get("message") or {}).get("usage")
    elif name == "content_block_start":
        block = data.get("content_block") or {}
        summary["block"] = block.get("type")
        if block.get("type") == "tool_use":
            index = data.get("index", len(result.tool_calls))
            result.tool_calls[index] = ToolCall(
                index=index,
                id=block.get("id"),
                name=block.get("name"),
                first_delta_complete=bool(block.get("id")) and bool(block.get("name")),
            )
    elif name == "content_block_delta":
        delta = data.get("delta") or {}
        summary["delta"] = delta.get("type")
        if delta.get("type") == "text_delta":
            result.content += delta.get("text", "")
        elif delta.get("type") == "thinking_delta":
            result.reasoning += delta.get("thinking", "")
            result.reasoning_deltas += 1
        elif delta.get("type") == "input_json_delta":
            call = result.tool_calls.setdefault(data.get("index", 0), ToolCall(index=data.get("index", 0)))
            call.arguments += delta.get("partial_json", "")
            call.deltas += 1
    elif name == "message_delta":
        result.finish_reason = (data.get("delta") or {}).get("stop_reason")
        result.usage = data.get("usage")
        result.usage_on_last_chunk = True
    elif name == "error":
        error = data.get("error") or {}
        result.error = {
            "status": None,
            "code": error.get("type"),
            "type": error.get("type"),
            "param": None,
            "message": error.get("message"),
            "retry_after_s": None,
        }
    if name not in ("message_delta", "message_stop", "ping"):
        result.usage_on_last_chunk = False
    result.events.append(summary)


def _messages_once(client: httpx.Client, body: dict[str, Any], headers: dict[str, str] | None = None) -> MessagesResult:
    result = MessagesResult()
    try:
        with client.stream("POST", "/messages", json=body, headers=headers or MESSAGES_HEADERS) as response:
            result.status = response.status_code
            result.headers = {name: response.headers[name] for name in RECORDED_HEADERS if name in response.headers}
            if response.status_code != 200:
                response.read()
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                result.error = _messages_error(response.status_code, payload, response.headers)
                return result
            if not body.get("stream"):
                response.read()
                message = response.json()
                for block in message.get("content") or []:
                    if block.get("type") == "text":
                        result.content += block.get("text", "")
                    elif block.get("type") == "thinking":
                        result.reasoning += block.get("thinking", "")
                    elif block.get("type") == "tool_use":
                        index = len(result.tool_calls)
                        result.tool_calls[index] = ToolCall(
                            index=index,
                            id=block.get("id"),
                            name=block.get("name"),
                            arguments=json.dumps(block.get("input")),
                            first_delta_complete=True,
                        )
                result.finish_reason = message.get("stop_reason")
                result.usage = message.get("usage")
                return result
            name = ""
            for line in response.iter_lines():
                if not line:
                    name = ""
                elif line.startswith("event:"):
                    name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    result.chunks += 1
                    _messages_event(result, name, json.loads(line[len("data:") :]))
    except (httpx.HTTPError, ValueError) as exc:
        result.error = {
            "status": None,
            "code": type(exc).__name__,
            "type": None,
            "param": None,
            "message": str(exc),
            "retry_after_s": None,
        }
    return result


@dataclass
class MessagesSession:
    """The client for one run, and every answer it got, for the grammar check."""

    client: httpx.Client
    results: list[MessagesResult] = field(default_factory=list)

    def send(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> MessagesResult:
        result = cast(MessagesResult, _retrying(lambda: _messages_once(self.client, body, headers)))
        self.results.append(result)
        return result


def _messages_body(model: str, text: str, **params: Any) -> dict[str, Any]:
    params.setdefault("max_tokens", GENERATION_BUDGET)
    params.setdefault("stream", True)
    return {"model": model, "messages": [{"role": "user", "content": text}], **params}


def grammar_problems(events: list[dict[str, Any]]) -> list[str]:
    """What breaks the Messages event grammar Anthropic clients parse."""
    names = [event["event"] for event in events if event["event"] != "ping"]
    if not names:
        return ["no events"]
    problems = []
    if names[0] != "message_start":
        problems.append(f"first event is {names[0]}, not message_start")
    open_index: int | None = None
    next_index = 0
    for event in events:
        name = event["event"]
        if name == "content_block_start":
            if open_index is not None:
                problems.append(f"block {event.get('index')} opened while {open_index} is open")
            if event.get("index") != next_index:
                problems.append(f"block index {event.get('index')}, expected {next_index}")
            open_index = event.get("index")
            next_index += 1
        elif name == "content_block_delta" and event.get("index") != open_index:
            problems.append(f"delta for block {event.get('index')} while {open_index} is open")
        elif name == "content_block_stop":
            if event.get("index") != open_index:
                problems.append(f"stop for block {event.get('index')} while {open_index} is open")
            open_index = None
    if names[-1] == "error":
        return problems
    if names[-2:] != ["message_delta", "message_stop"]:
        problems.append(f"ends with {names[-2:]}, not message_delta then message_stop")
    if open_index is not None:
        problems.append(f"block {open_index} never stopped")
    return problems


def messages_case_streaming_tool_use(session: MessagesSession, model: str) -> tuple[Case, MessagesResult]:
    name = "streaming_tool_use"
    body = _messages_body(
        model,
        "What is the weather in Paris right now? Use the get_weather tool.",
        tools=[MESSAGES_TOOL],
        tool_choice={"type": "auto"},
        thinking={"type": "adaptive"},
        temperature=0,
    )
    result = session.send(body)
    failed = _fail_on_error(name, result)
    if failed:
        return failed, result
    calls = result.calls()
    problems = []
    if not calls:
        problems.append("no tool_use block came back")
    for call in calls:
        if not call.first_delta_complete:
            problems.append(f"block {call.index}: content_block_start lacks id or name")
        if call.name != "get_weather":
            problems.append(f"block {call.index} named {call.name!r}")
        parsed = call.parsed_arguments()
        if parsed is None:
            problems.append(f"block {call.index} input is not a JSON object: {call.arguments[:80]!r}")
        elif "city" not in parsed:
            problems.append(f"block {call.index} input lacks 'city': {parsed}")
    if result.finish_reason != "tool_use":
        problems.append(f"stop_reason is {result.finish_reason!r}, expected 'tool_use'")
    if problems:
        return Case(name, "fail", "; ".join(problems), result.observed()), result
    deltas = sum(call.deltas for call in calls)
    detail = f"{len(calls)} tool_use: {_describe_calls(result)}; {deltas} input_json_delta; stop_reason=tool_use"
    return Case(name, "pass", detail, result.observed()), result


def messages_case_parallel_tool_use(session: MessagesSession, model: str) -> Case:
    name = "parallel_tool_use"
    body = _messages_body(
        model,
        "Get the current weather in Paris and in Tokyo. Call get_weather once per city, both in this turn.",
        tools=[MESSAGES_TOOL],
        temperature=0,
    )
    result = session.send(body)
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    calls = result.calls()
    ids = [call.id for call in calls]
    cities = sorted(str((call.parsed_arguments() or {}).get("city", "")).lower() for call in calls)
    observed = result.observed()
    if len(calls) < 2:
        return Case(name, "fail", f"{len(calls)} tool_use in the turn: {_describe_calls(result)}", observed)
    if len(set(ids)) != len(ids) or None in ids:
        return Case(name, "fail", f"tool_use ids are not distinct: {ids}", observed)
    missing = [city for city in ("paris", "tokyo") if not any(city in c for c in cities)]
    if missing:
        return Case(name, "fail", f"{len(calls)} calls but cities {cities} miss {missing}", observed)
    return Case(name, "pass", f"{len(calls)} tool_use in one turn: {_describe_calls(result)}", observed)


def messages_case_tool_result_round_trip(session: MessagesSession, model: str, first: MessagesResult) -> Case:
    name = "tool_result_round_trip"
    calls = first.calls()
    if _is_busy(first.error):
        return Case(name, "capacity", "not run, streaming_tool_use stayed busy", first.observed())
    if first.error or not calls:
        return Case(name, "fail", "no tool_use from streaming_tool_use to answer", first.observed())
    assistant: list[dict[str, Any]] = []
    if first.reasoning:
        assistant.append({"type": "thinking", "thinking": first.reasoning, "signature": "sie-unsigned"})
    if first.content:
        assistant.append({"type": "text", "text": first.content})
    assistant += [
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.parsed_arguments() or {}} for call in calls
    ]
    results = [
        {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps({"city": "Paris", "temperature_c": 18, "conditions": "overcast"}),
        }
        for call in calls
    ]
    body = {
        "model": model,
        "max_tokens": GENERATION_BUDGET,
        "stream": True,
        "temperature": 0,
        "tools": [MESSAGES_TOOL],
        "messages": [
            {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."},
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": results},
        ],
    }
    result = session.send(body)
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    observed = result.observed()
    problems = []
    if result.calls():
        problems.append(f"answered with another tool_use: {_describe_calls(result)}")
    if result.finish_reason != "end_turn":
        problems.append(f"stop_reason is {result.finish_reason!r}, expected 'end_turn'")
    if not result.content.strip():
        problems.append("empty final answer")
    elif "18" not in result.content:
        problems.append(f"final answer does not carry the tool's 18: {result.content[:120]!r}")
    if problems:
        return Case(name, "fail", "; ".join(problems), observed)
    return Case(name, "pass", f"final answer used the tool result: {result.content.strip()[:100]!r}", observed)


def messages_case_prompt_above_64k_tokens(session: MessagesSession, model: str, entry: dict[str, Any] | None) -> Case:
    name = "prompt_above_64k_tokens"
    text = f"Read this warehouse log, then answer.\n\n{long_prompt()}\n\nWhat is the access code? Reply with just the code."
    result = session.send(_messages_body(model, text, temperature=0))
    observed = result.observed()
    observed["needle_found"] = "7741-ZEBRA" in result.content
    window = declared_window(entry)
    observed["declared_window"] = window
    error = result.error
    if window is not None and window < LONG_PROMPT_MIN_TOKENS and error is not None and _is_refusal(error):
        wording = (
            "Claude Code's compaction wording" if PROMPT_TOO_LONG.match(error["message"] or "") else "other wording"
        )
        detail = (
            f"declared window {window} is under {LONG_PROMPT_MIN_TOKENS}; refused with {wording}: {error['message']}"
        )
        return Case(name, "recorded", detail, observed)
    failed = _fail_on_error(name, result)
    if failed:
        failed.observed = observed
        return failed
    usage = result.usage or {}
    prompt_tokens = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0)
    if prompt_tokens < LONG_PROMPT_MIN_TOKENS:
        return Case(name, "fail", f"prompt counted as {prompt_tokens} tokens, under {LONG_PROMPT_MIN_TOKENS}", observed)
    if not result.content.strip() or result.finish_reason != "end_turn":
        return Case(name, "fail", f"accepted {prompt_tokens} tokens but stop_reason={result.finish_reason!r}", observed)
    needle = "needle found" if observed["needle_found"] else "needle not found"
    return Case(name, "pass", f"{prompt_tokens} prompt tokens accepted and answered; {needle}", observed)


def messages_case_end_turn(session: MessagesSession, model: str) -> tuple[Case, MessagesResult]:
    name = "stop_reason_end_turn"
    result = session.send(_messages_body(model, "Reply with the single word: ready.", temperature=0))
    failed = _fail_on_error(name, result)
    if failed:
        return failed, result
    if result.finish_reason != "end_turn":
        return Case(name, "fail", f"stop_reason is {result.finish_reason!r}", result.observed()), result
    detail = f"stop_reason=end_turn after {result.content.strip()[:40]!r}"
    return Case(name, "pass", detail, result.observed()), result


def messages_case_stop_tool_use(first: Case) -> Case:
    name = "stop_reason_tool_use"
    stop = first.observed.get("stop_reason")
    if first.status == "capacity":
        return Case(name, "capacity", "not judged, the tool turn stayed busy", {"stop_reason": stop})
    if stop == "tool_use":
        return Case(name, "pass", "stop_reason=tool_use on the tool turn", {"stop_reason": stop})
    return Case(name, "fail", f"tool turn ended with stop_reason={stop!r}", {"stop_reason": stop})


def messages_case_max_tokens(session: MessagesSession, model: str) -> Case:
    name = "stop_reason_max_tokens"
    result = session.send(_messages_body(model, "Write three paragraphs about the sea.", max_tokens=8))
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    if result.finish_reason != "max_tokens":
        return Case(name, "fail", f"stop_reason is {result.finish_reason!r} with max_tokens=8", result.observed())
    output = (result.usage or {}).get("output_tokens")
    return Case(name, "pass", f"stop_reason=max_tokens at max_tokens=8 (output_tokens={output})", result.observed())


def messages_case_usage_on_message_delta(stop: MessagesResult) -> Case:
    name = "usage_on_message_delta"
    observed = {
        "usage": stop.usage,
        "message_start_usage": stop.start_usage,
        "on_message_delta": stop.usage_on_last_chunk,
    }
    if _is_busy(stop.error):
        return Case(name, "capacity", "not judged, the end_turn stream stayed busy", observed)
    if stop.error:
        return Case(name, "fail", "the end_turn stream failed, see stop_reason_end_turn", observed)
    usage = stop.usage or {}
    if not stop.usage_on_last_chunk or not usage:
        return Case(name, "fail", "no usage on message_delta", observed)
    if not isinstance(stop.start_usage, dict):
        return Case(name, "fail", "message_start carries no usage object", observed)
    missing = [key for key in ("input_tokens", "output_tokens") if not usage.get(key)]
    if missing:
        return Case(name, "fail", f"message_delta usage lacks {missing}: {usage}", observed)
    return Case(name, "pass", f"message_delta carries usage {usage}", observed)


def messages_case_thinking(session: MessagesSession, model: str) -> Case:
    name = "thinking_blocks"
    with_thinking = session.send(
        _messages_body(model, REASONING_QUESTION, temperature=0, thinking={"type": "adaptive"})
    )
    without = session.send(_messages_body(model, REASONING_QUESTION, temperature=0))
    whole = session.send(
        _messages_body(model, REASONING_QUESTION, temperature=0, stream=False, thinking={"type": "adaptive"})
    )
    results = (with_thinking, without, whole)
    observed = {
        "with_thinking": with_thinking.observed(),
        "without_thinking": without.observed(),
        "non_streamed_with_thinking": whole.observed(),
    }
    for label, result in (("thinking", with_thinking), ("plain request", without), ("non-streamed thinking", whole)):
        if result.error and not _is_busy(result.error):
            error = result.error
            return Case(
                name, "fail", f"{label} failed: {error['status']} {error['type']}: {error['message']}", observed
            )
    busy = next((result for result in results if _is_busy(result.error)), None)
    if busy is not None:
        return Case(name, "capacity", _still_busy(busy), observed)
    if any("<think>" in result.content or "</think>" in result.content for result in results):
        return Case(name, "fail", "raw <think> tags leaked into text", observed)
    if without.reasoning_deltas:
        return Case(name, "fail", f"thinking arrived without asking ({without.reasoning_deltas} deltas)", observed)
    if not (with_thinking.reasoning or whole.reasoning):
        return Case(
            name,
            "recorded",
            "no thinking block came back; the model did not reason here or reasons privately",
            observed,
        )
    if not with_thinking.reasoning or not whole.reasoning:
        missing = "streamed" if not with_thinking.reasoning else "buffered"
        return Case(name, "fail", f"thinking came back on one path but not the {missing} one", observed)
    detail = f"{with_thinking.reasoning_deltas} thinking_delta, {len(with_thinking.reasoning)} chars; buffered thinking block {len(whole.reasoning)} chars; none without asking"
    return Case(name, "pass", detail, observed)


def messages_case_tool_choice_any(session: MessagesSession, model: str) -> Case:
    name = "tool_choice_any"
    body = _messages_body(model, "Say hello.", tools=[MESSAGES_TOOL], tool_choice={"type": "any"}, temperature=0)
    result = session.send(body)
    observed = result.observed()
    if result.error:
        error = result.error
        if _is_busy(error):
            return Case(name, "capacity", _still_busy(result), observed)
        if _is_refusal(error):
            return Case(name, "recorded", f"refused: {error['status']} {error['type']}: {error['message']}", observed)
        return Case(name, "fail", f"request failed: {error['status']} {error['type']}: {error['message']}", observed)
    if result.calls():
        return Case(name, "recorded", f"supported: forced {_describe_calls(result)}", observed)
    return Case(
        name, "fail", f"accepted tool_choice any but answered without a call: {result.content[:80]!r}", observed
    )


def messages_case_inert_fields(session: MessagesSession, model: str) -> Case:
    name = "inert_fields_accepted"
    body = {
        "model": model,
        "max_tokens": GENERATION_BUDGET,
        "stream": True,
        "metadata": {"user_id": "conformance-check"},
        "cache_control": {"type": "ephemeral"},
        "system": [{"type": "text", "text": "Answer briefly.", "cache_control": {"type": "ephemeral"}}],
        "tools": [{**MESSAGES_TOOL, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Reply with the single word: ready.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ],
    }
    result = session.send(body)
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    return Case(name, "pass", "metadata, cache_control and anthropic-beta accepted", result.observed())


def _refusal_case(name: str, result: MessagesResult, wanted: Callable[[str], bool], what: str) -> Case:
    observed = result.observed()
    error = result.error
    if error is None:
        return Case(name, "fail", f"accepted, expected a 400 with {what}", observed)
    if _is_busy(error):
        return Case(name, "capacity", _still_busy(result), observed)
    if error["status"] != 400 or error["type"] != "invalid_request_error":
        return Case(
            name, "fail", f"answered {error['status']} {error['type']}, expected 400 invalid_request_error", observed
        )
    if not wanted(error["message"] or ""):
        return Case(name, "fail", f"400 without {what}: {error['message']!r}", observed)
    return Case(name, "pass", f"400 invalid_request_error: {error['message']}", observed)


def messages_case_effort_refusal(session: MessagesSession, model: str) -> Case:
    body = _messages_body(model, "Reply with the single word: ready.", output_config={"effort": "high"})
    result = session.send(body)
    return _refusal_case(
        "effort_refusal",
        result,
        lambda message: message == EFFORT_REFUSAL,
        "the wording Claude Code retries without effort on",
    )


def messages_case_system_turn_refusal(session: MessagesSession, model: str) -> Case:
    body = {
        "model": model,
        "max_tokens": GENERATION_BUDGET,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Reply with the single word: ready."},
            {"role": "system", "content": "The user prefers short answers."},
        ],
    }
    result = session.send(body)

    def wanted(message: str) -> bool:
        return "not supported" in message and SYSTEM_TURN_REFUSAL.search(message) is not None

    return _refusal_case(
        "system_turn_refusal", result, wanted, "the wording Claude Code resends without system turns on"
    )


def messages_case_unknown_model(session: MessagesSession) -> Case:
    name = "unknown_model"
    result = session.send(_messages_body(MISSING_MODEL, "hi"))
    observed = result.observed()
    error = result.error
    if error is None:
        return Case(name, "fail", f"{MISSING_MODEL} was served", observed)
    if error["status"] == 404 and error["type"] == "not_found_error":
        return Case(name, "pass", f"404 not_found_error: {error['message']}", observed)
    return Case(name, "fail", f"answered {error['status']} {error['type']}, expected 404 not_found_error", observed)


def messages_case_buffered(session: MessagesSession, model: str) -> Case:
    name = "buffered_message"
    result = session.send(_messages_body(model, "Reply with the single word: ready.", stream=False, temperature=0))
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    usage = result.usage or {}
    if result.finish_reason != "end_turn" or not usage.get("input_tokens") or not usage.get("output_tokens"):
        return Case(name, "fail", f"stop_reason={result.finish_reason!r}, usage={usage}", result.observed())
    return Case(name, "pass", f"stop_reason=end_turn, usage {usage}", result.observed())


def messages_case_event_grammar(results: list[MessagesResult]) -> Case:
    name = "event_grammar"
    streams = [result for result in results if result.events]
    problems = {f"stream {position}": grammar_problems(result.events) for position, result in enumerate(streams)}
    problems = {key: value for key, value in problems.items() if value}
    observed = {"streams": len(streams), "problems": problems}
    if not streams:
        return Case(name, "fail", "no stream to check", observed)
    if problems:
        first = next(iter(problems.values()))
        return Case(name, "fail", f"{len(problems)} of {len(streams)} streams break the grammar: {first[0]}", observed)
    return Case(
        name,
        "pass",
        f"{len(streams)} streams: message_start, one block at a time, message_delta then message_stop",
        observed,
    )


def messages_case_pings(results: list[MessagesResult]) -> Case:
    pings = sum(result.pings for result in results)
    observed = {"pings": pings, "streams": sum(1 for result in results if result.events)}
    return Case("ping_events", "recorded", f"{pings} ping events across the streams", observed)


def messages_case_count_tokens(session: MessagesSession, model: str) -> Case:
    name = "count_tokens"
    body = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    try:
        response = session.client.post("/messages/count_tokens", json=body, headers=MESSAGES_HEADERS)
        status = response.status_code
        try:
            payload = response.json()
        except ValueError:
            payload = None
    except httpx.HTTPError as exc:
        return Case(name, "recorded", f"request failed: {type(exc).__name__}", {})
    observed = {"status": status, "body": payload}
    if status == 200:
        return Case(name, "recorded", f"served: {payload}", observed)
    return Case(name, "recorded", f"not served: {status}; Claude Code then estimates locally", observed)


def wait_until_messages_served(client: httpx.Client, model: str) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        result = _messages_once(client, _messages_body(model, "Reply with the single word: ready.", max_tokens=4))
        elapsed = time.monotonic() - started
        code = str((result.error or {}).get("code") or "").upper()
        if code not in MESSAGES_WARM_UP_CODES or elapsed > WARM_UP_TIMEOUT_S:
            return {"attempts": attempts, "seconds": round(elapsed, 1), "error": result.error}
        time.sleep(WARM_UP_POLL_S)


def run_messages(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=REQUEST_TIMEOUT_S)
    started = datetime.now(UTC)
    entry = model_entry(base_url, api_key, model)
    warm_up = wait_until_messages_served(client, model)
    session = MessagesSession(client)
    cases: list[Case] = []

    tool_case, tool_result = messages_case_streaming_tool_use(session, model)
    cases.append(tool_case)
    cases.append(messages_case_parallel_tool_use(session, model))
    cases.append(messages_case_tool_result_round_trip(session, model, tool_result))
    cases.append(messages_case_prompt_above_64k_tokens(session, model, entry))
    stop_case, stop_result = messages_case_end_turn(session, model)
    cases.append(stop_case)
    cases.append(messages_case_stop_tool_use(tool_case))
    cases.append(messages_case_max_tokens(session, model))
    cases.append(messages_case_usage_on_message_delta(stop_result))
    cases.append(messages_case_thinking(session, model))
    cases.append(messages_case_tool_choice_any(session, model))
    cases.append(messages_case_buffered(session, model))
    cases.append(messages_case_inert_fields(session, model))
    cases.append(messages_case_effort_refusal(session, model))
    cases.append(messages_case_system_turn_refusal(session, model))
    cases.append(messages_case_unknown_model(session))
    cases.append(messages_case_event_grammar(session.results))
    cases.append(messages_case_pings(session.results))
    cases.append(messages_case_count_tokens(session, model))
    client.close()

    statuses = ("pass", "fail", "recorded", "capacity")
    counts = {status: sum(1 for case in cases if case.status == status) for status in statuses}
    return {
        "api": "messages",
        "base_url": base_url,
        "model": model,
        "model_entry": entry,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "served_by": tool_result.headers.get("x-sie-served-by"),
        "warm_up": warm_up,
        "cases": [{**case.__dict__, "retries": _retries_in(case.observed)} for case in cases],
        "summary": counts,
    }


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
    parser.add_argument(
        "--api",
        choices=("chat", "messages"),
        default="chat",
        help="wire to check: chat completions, as OpenCode speaks it (default), or Anthropic Messages",
    )
    args = parser.parse_args()

    check = run_messages if args.api == "messages" else run
    report = check(args.base_url, args.api_key, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_table(report)
    print(f"report written to {args.report}")
    return 1 if report["summary"]["fail"] or report["summary"]["capacity"] else 0


if __name__ == "__main__":
    sys.exit(main())
