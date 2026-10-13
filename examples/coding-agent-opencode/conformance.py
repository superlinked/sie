#!/usr/bin/env python3
"""Agent-loop conformance check for an OpenAI-compatible chat endpoint.

    uv sync
    uv run python conformance.py --model Qwen/Qwen3.5-4B                       # local sie-server
    SIE_API_KEY=... uv run python conformance.py \
        --base-url https://api.superlinked.com/v1 --model zai-org/GLM-5.3-Flash:relay

Drives ``/v1/chat/completions`` the way a coding agent such as OpenCode does:
streaming on, tools declared on every turn, tool results sent back as
``role: tool`` messages. Every case records what came back, so a failure
says what the wire did rather than only that it was wrong.

The check goes through the ``openai`` package on purpose. The thing under test
is the OpenAI-compatible wire surface that OpenCode's provider speaks, so the
reference client for that surface is the right instrument here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai

DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_REPORT = Path("run-output/conformance.json")
REQUEST_TIMEOUT_S = 900.0
WARM_UP_TIMEOUT_S = 900.0
WARM_UP_POLL_S = 5.0
LONG_PROMPT_MIN_TOKENS = 65_536
LONG_PROMPT_WORDS = 60_000
NEEDLE = "The access code is 7741-ZEBRA."

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
            "headers": self.headers,
        }


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


def _error_payload(exc: openai.APIStatusError) -> dict[str, Any]:
    body = exc.body if isinstance(exc.body, dict) else {}
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    return {
        "status": exc.status_code,
        "code": inner.get("code"),
        "message": inner.get("message") or str(exc),
    }


def stream_chat(client: openai.OpenAI, model: str, messages: list[dict[str, Any]], **params: Any) -> StreamResult:
    result = StreamResult()
    started = time.monotonic()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        result.headers = {
            name: raw.headers[name]
            for name in ("x-sie-served-by", "x-sie-upstream", "x-sie-model-revision")
            if name in raw.headers
        }
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
    except openai.APIStatusError as exc:
        result.error = _error_payload(exc)
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        result.error = {"status": None, "code": type(exc).__name__, "message": str(exc)}
    result.seconds = time.monotonic() - started
    return result


def _describe_calls(result: StreamResult) -> str:
    return ", ".join(f"{call.name}({call.arguments})" for call in result.calls()) or "no tool call"


def _fail_on_error(name: str, result: StreamResult) -> Case | None:
    if result.error is None:
        return None
    error = result.error
    return Case(
        name, "fail", f"request failed: {error['status']} {error['code']}: {error['message']}", result.observed()
    )


def _refuses_field(error: dict[str, Any], name: str) -> bool:
    status = error.get("status")
    return status is not None and 400 <= status < 500 and name in str(error.get("message") or "")


def case_streaming_tool_call_deltas(
    client: openai.OpenAI, model: str, tool_params: dict[str, Any]
) -> tuple[Case, StreamResult]:
    name = "streaming_tool_call_deltas"
    messages = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
    result = stream_chat(client, model, messages, tools=[WEATHER_TOOL], max_tokens=512, temperature=0, **tool_params)
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
    result = stream_chat(client, model, messages, tools=[WEATHER_TOOL], max_tokens=512, temperature=0, **tool_params)
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
    if first.error or not calls:
        return Case(name, "fail", "no tool call from streaming_tool_call_deltas to send a result for", first.observed())
    assistant = {
        "role": "assistant",
        "content": first.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    }
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
    result = stream_chat(client, model, messages, tools=[WEATHER_TOOL], max_tokens=256, temperature=0, **tool_params)
    failed = _fail_on_error(name, result)
    if failed:
        return failed
    observed = result.observed()
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


def case_prompt_above_64k_tokens(client: openai.OpenAI, model: str) -> Case:
    name = "prompt_above_64k_tokens"
    document = long_prompt()
    messages = [
        {
            "role": "user",
            "content": f"Read this warehouse log, then answer.\n\n{document}\n\nWhat is the access code? Reply with just the code.",
        }
    ]
    result = stream_chat(client, model, messages, max_tokens=64, temperature=0)
    observed = result.observed()
    observed["prompt_chars"] = len(messages[0]["content"])
    observed["needle_found"] = "7741-ZEBRA" in result.content
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
    result = stream_chat(client, model, messages, max_tokens=64, temperature=0)
    failed = _fail_on_error(name, result)
    if failed:
        return failed, result
    if result.finish_reason != "stop":
        return Case(name, "fail", f"finish_reason is {result.finish_reason!r}", result.observed()), result
    return Case(name, "pass", f"finish_reason=stop after {result.content.strip()[:40]!r}", result.observed()), result


def case_finish_reason_tool_calls(first: Case) -> Case:
    name = "finish_reason_tool_calls"
    finish = first.observed.get("finish_reason")
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


def case_include_reasoning(client: openai.OpenAI, model: str) -> Case:
    name = "include_reasoning"
    messages = [{"role": "user", "content": "Which is larger, 9.11 or 9.9? Answer with just the larger number."}]
    with_flag = stream_chat(
        client, model, messages, max_tokens=2048, temperature=0, extra_body={"include_reasoning": True}
    )
    without_flag = stream_chat(client, model, messages, max_tokens=2048, temperature=0)
    observed = {
        "with_include_reasoning": with_flag.observed(),
        "without_include_reasoning": {
            "reasoning_deltas": without_flag.reasoning_deltas,
            "error": without_flag.error,
        },
        "content_has_think_tags": "<think>" in with_flag.content or "</think>" in with_flag.content,
    }
    if with_flag.error:
        error = with_flag.error
        return Case(
            name, "fail", f"include_reasoning refused: {error['status']} {error['code']}: {error['message']}", observed
        )
    if without_flag.error:
        error = without_flag.error
        return Case(
            name, "fail", f"plain request failed: {error['status']} {error['code']}: {error['message']}", observed
        )
    if observed["content_has_think_tags"]:
        return Case(name, "fail", "raw <think> tags leaked into content", observed)
    if without_flag.reasoning_deltas:
        return Case(
            name,
            "fail",
            f"reasoning_content arrived without the flag ({without_flag.reasoning_deltas} deltas)",
            observed,
        )
    if with_flag.reasoning_deltas:
        detail = f"{with_flag.reasoning_deltas} reasoning_content deltas, {len(with_flag.reasoning)} chars; none without the flag"
    else:
        detail = "no reasoning_content deltas: the model declares no reasoning, or the field is not served here"
    return Case(name, "pass", detail, observed)


def case_tool_choice_required(client: openai.OpenAI, model: str) -> Case:
    name = "tool_choice_required"
    messages = [{"role": "user", "content": "Say hello."}]
    result = stream_chat(
        client, model, messages, tools=[WEATHER_TOOL], tool_choice="required", max_tokens=256, temperature=0
    )
    observed = result.observed()
    if result.error:
        error = result.error
        if error["status"] is not None and 400 <= error["status"] < 500:
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
    entries = payload.get("models") or payload.get("data") or []
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
        if (result.error or {}).get("code") != "MODEL_LOADING" or elapsed > WARM_UP_TIMEOUT_S:
            return {"attempts": attempts, "seconds": round(elapsed, 1), "error": result.error}
        time.sleep(WARM_UP_POLL_S)


def run(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    client = openai.OpenAI(base_url=base_url, api_key=api_key or "none", timeout=REQUEST_TIMEOUT_S, max_retries=0)
    started = datetime.now(UTC)
    warm_up = wait_until_loaded(client, model)
    cases: list[Case] = []

    tool_params: dict[str, Any] = {"tool_choice": "auto"}
    tool_case, tool_result = case_streaming_tool_call_deltas(client, model, tool_params)
    if tool_result.error and _refuses_field(tool_result.error, "tool_choice"):
        error = tool_result.error
        detail = (
            "OpenCode sends tool_choice=auto on every tools turn; "
            f"refused: {error['status']} {error['code']}: {error['message']}"
        )
        cases.append(Case("tool_choice_auto", "fail", detail, tool_result.observed()))
        tool_params = {}
        tool_case, tool_result = case_streaming_tool_call_deltas(client, model, tool_params)
    elif tool_result.error:
        error = tool_result.error
        detail = f"request failed: {error['status']} {error['code']}: {error['message']}"
        cases.append(Case("tool_choice_auto", "fail", detail, tool_result.observed()))
    else:
        observed = {"tool_choice": "auto", "finish_reason": tool_result.finish_reason}
        cases.append(Case("tool_choice_auto", "pass", "tool_choice=auto accepted on the tools turn", observed))
    cases.append(tool_case)
    cases.append(case_parallel_tool_calls(client, model, tool_params))
    cases.append(case_tool_result_round_trip(client, model, tool_result, tool_params))
    cases.append(case_prompt_above_64k_tokens(client, model))
    stop_case, stop_result = case_finish_reason_stop(client, model)
    cases.append(stop_case)
    cases.append(case_finish_reason_tool_calls(tool_case))
    cases.append(case_finish_reason_length(client, model))
    cases.append(case_usage_on_final_chunk(stop_result))
    cases.append(case_include_reasoning(client, model))
    cases.append(case_tool_choice_required(client, model))

    counts = {status: sum(1 for case in cases if case.status == status) for status in ("pass", "fail", "recorded")}
    return {
        "base_url": base_url,
        "model": model,
        "model_entry": model_entry(base_url, api_key, model),
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "served_by": tool_result.headers.get("x-sie-served-by"),
        "warm_up": warm_up,
        "tools_turn_params": tool_params,
        "cases": [case.__dict__ for case in cases],
        "summary": counts,
    }


def print_table(report: dict[str, Any]) -> None:
    width = max(len(case["name"]) for case in report["cases"])
    print(f"model {report['model']} at {report['base_url']}")
    if report.get("served_by"):
        print(f"served by {report['served_by']}")
    warm_up = report["warm_up"]
    print(f"warm-up: {warm_up['attempts']} attempt(s), {warm_up['seconds']}s")
    print()
    for case in report["cases"]:
        print(f"  {case['name']:<{width}}  {case['status']:<8}  {case['detail']}")
    summary = report["summary"]
    print()
    print(f"{summary['pass']} pass, {summary['fail']} fail, {summary['recorded']} recorded")


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
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
