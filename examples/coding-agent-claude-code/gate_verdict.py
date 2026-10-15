#!/usr/bin/env python3
"""Judge a headless Claude Code run from its ``--output-format stream-json`` events.

    python gate_verdict.py run-output/gate-.../claude-events.jsonl

The run passes when Claude Code finished without an error, took at least
``MIN_TURNS`` turns, read a non-zero input token count from the endpoint's
usage, and retried no request that the endpoint refused as invalid. Prints a
JSON summary of the run either way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MIN_TURNS = 3


def _events(lines: list[str]) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    unreadable = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        if isinstance(event, dict):
            events.append(event)
    return events, unreadable


def _tool_uses(events: list[dict[str, Any]]) -> list[str]:
    names = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(str(block.get("name")))
    return names


def judge(lines: list[str]) -> tuple[bool, dict[str, Any]]:
    events, unreadable = _events(lines)
    results = [event for event in events if event.get("type") == "result"]
    retries = [event for event in events if event.get("type") == "system" and event.get("subtype") == "api_retry"]
    invalid_retries = [
        event
        for event in retries
        if event.get("error_status") == 400 or event.get("error") in ("invalid_request", "invalid_request_error")
    ]
    result = results[-1] if results else {}
    usage = result.get("usage") or {}
    problems = []
    if not results:
        problems.append("no result event")
    else:
        if result.get("is_error"):
            problems.append(f"the run ended in an error: {str(result.get('result'))[:200]}")
        if result.get("subtype") != "success":
            problems.append(f"result subtype is {result.get('subtype')!r}")
        if (result.get("num_turns") or 0) < MIN_TURNS:
            problems.append(f"{result.get('num_turns')} turns, fewer than {MIN_TURNS}")
        if not usage.get("input_tokens"):
            problems.append("Claude Code read no input tokens from the endpoint's usage")
    if invalid_retries:
        problems.append(f"{len(invalid_retries)} request(s) refused as invalid were retried")
    summary = {
        "turns": result.get("num_turns"),
        "duration_ms": result.get("duration_ms"),
        "usage": {key: usage.get(key) for key in ("input_tokens", "cache_read_input_tokens", "output_tokens")},
        "tool_uses": _tool_uses(events),
        "api_retries": len(retries),
        "unreadable_lines": unreadable,
        "problems": problems,
    }
    return not problems, summary


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    passed, summary = judge(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines())
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
