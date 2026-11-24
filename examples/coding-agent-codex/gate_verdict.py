"""Judge a headless Codex run from its ``codex exec --json`` events.

    python gate_verdict.py run-output/gate-.../codex-events.jsonl

The run passes when Codex completed its turn with no failed turn and no error
event, ran at least ``MIN_COMMANDS`` commands, and read a non-zero input token
count from the endpoint's usage. A dropped stream that Codex retries is an
error event whose message starts with ``Reconnecting...``; those are counted,
not failed. Items of type ``error``, such as the warning that Codex has no
metadata for the model, are warnings. Prints a JSON summary of the run either
way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MIN_COMMANDS = 1
RECONNECTING = "Reconnecting..."


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


def judge(lines: list[str]) -> tuple[bool, dict[str, Any]]:
    events, unreadable = _events(lines)
    completed = [event for event in events if event.get("type") == "turn.completed"]
    failed = [event for event in events if event.get("type") == "turn.failed"]
    errors = [str(event.get("message")) for event in events if event.get("type") == "error"]
    reconnects = [message for message in errors if message.startswith(RECONNECTING)]
    others = [message for message in errors if not message.startswith(RECONNECTING)]
    items = [event.get("item") or {} for event in events if event.get("type") == "item.completed"]
    commands = [item for item in items if item.get("type") == "command_execution"]
    usage = (completed[-1].get("usage") if completed else None) or {}
    problems = []
    if failed:
        problems.append(f"the turn failed: {str((failed[-1].get('error') or {}).get('message'))[:200]}")
    elif others:
        problems.append(f"{len(others)} error event(s): {others[0][:200]}")
    if not completed:
        problems.append("no turn.completed event")
    else:
        if len(commands) < MIN_COMMANDS:
            problems.append(f"{len(commands)} commands run, fewer than {MIN_COMMANDS}")
        if not usage.get("input_tokens"):
            problems.append("Codex read no input tokens from the endpoint's usage")
    summary = {
        "commands": len(commands),
        "items": [item.get("type") for item in items],
        "usage": {key: usage.get(key) for key in ("input_tokens", "cached_input_tokens", "output_tokens")},
        "reconnects": len(reconnects),
        "warnings": [str(item.get("message"))[:200] for item in items if item.get("type") == "error"],
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
