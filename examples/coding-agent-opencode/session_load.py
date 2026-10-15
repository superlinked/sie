#!/usr/bin/env python3
"""Agent-shaped session load for an OpenAI-compatible chat endpoint.

    uv sync
    uv run python session_load.py --model Qwen/Qwen3.8-27B-FP8:h100-256k-batch --sessions 1,2,4,6,8

A coding agent sends its whole transcript on every turn: the same system
prompt, then everything read and said so far, then the new input. Each
session here does the same. Every session shares one system prompt of
``--prefix-tokens``; each turn adds ``--turn-tokens`` of new input, unique to
the session, and asks for an answer that runs to ``--output-tokens``; the
answer goes back into the transcript as it was streamed. For each count in
``--sessions``, that many sessions run their ``--turns`` turns at once.

Each turn records the time to the first streamed token, the decode rate after
it, the prompt tokens and, when the server reports them, the cached prompt
tokens. A returning turn is any turn after a session's first: with a working
prefix cache only its new input is prefilled. The summary for each session
count gives the median and p90 of both, where the p90 decode rate is the rate
nine turns in ten reached or beat, and whether that count meets the bar of at
least 30 tok/s and at most 3 s to the first token on returning turns, both at
p90, with no failed turn.

The input is built from common English words in random order, sized with the endpoint's own
token count of the shared prompt, so the sizes hold on any tokenizer to within
a few percent; the report keeps the prompt tokens the server counted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import openai

from conformance import DEFAULT_BASE_URL, FILLER_WORDS, model_entry, wait_until_loaded

DEFAULT_REPORT = Path("run-output/session-load.json")
REQUEST_TIMEOUT_S = 1800.0
MIN_DECODE_TPS = 30.0
MAX_RETURNING_TTFT_S = 3.0
WORDS_PER_LINE = 16


@dataclass
class Turn:
    sessions: int
    session: int
    turn: int
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    ttft_s: float | None = None
    decode_tps: float | None = None
    seconds: float = 0.0
    finish_reason: str | None = None
    error: str | None = None


def words_text(rng: random.Random, words: int, label: str) -> str:
    lines = [label]
    for start in range(0, words, WORDS_PER_LINE):
        count = min(WORDS_PER_LINE, words - start)
        lines.append(" ".join(rng.choice(FILLER_WORDS) for _ in range(count)) + ".")
    return "\n".join(lines)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def stream_turn(
    client: openai.OpenAI, model: str, messages: list[dict[str, Any]], record: Turn, max_tokens: int
) -> str:
    started = time.monotonic()
    first = last = None
    content = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump(exclude_none=True)
                record.prompt_tokens = usage.get("prompt_tokens")
                record.completion_tokens = usage.get("completion_tokens")
                record.cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            for choice in chunk.choices or []:
                if choice.finish_reason:
                    record.finish_reason = choice.finish_reason
                text = choice.delta.content if choice.delta else None
                if text:
                    now = time.monotonic()
                    first = now if first is None else first
                    last = now
                    content += text
    except (openai.APIError, httpx.HTTPError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"[:300]
    record.seconds = time.monotonic() - started
    if first is not None:
        record.ttft_s = first - started
        if last is not None and last > first and record.completion_tokens and record.completion_tokens > 1:
            record.decode_tps = (record.completion_tokens - 1) / (last - first)
    return content


def run_session(
    client: openai.OpenAI,
    model: str,
    system: str,
    words_per_turn: int,
    args: argparse.Namespace,
    sessions: int,
    index: int,
    start: threading.Barrier,
    records: list[Turn],
) -> None:
    rng = random.Random(f"{args.seed}-{sessions}-{index}")
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    start.wait()
    for turn in range(1, args.turns + 1):
        label = f"Session {index + 1} of {sessions}, turn {turn}. The agent read this file:"
        body = words_text(rng, words_per_turn, label)
        ask = f"Explain what the file above changes and why, in at least {args.output_tokens * 2} words."
        messages.append({"role": "user", "content": f"{body}\n\n{ask}"})
        record = Turn(sessions=sessions, session=index + 1, turn=turn)
        answer = stream_turn(client, model, messages, record, args.output_tokens)
        records.append(record)
        if record.error:
            return
        messages.append({"role": "assistant", "content": answer})


def summarise(sessions: int, turns: list[Turn], wall_s: float) -> dict[str, Any]:
    done = [turn for turn in turns if turn.error is None]
    returning = [turn for turn in done if turn.turn > 1]
    rates = [turn.decode_tps for turn in done if turn.decode_tps is not None]
    first_ttft = [turn.ttft_s for turn in done if turn.turn == 1 and turn.ttft_s is not None]
    returning_ttft = [turn.ttft_s for turn in returning if turn.ttft_s is not None]
    reported = [turn for turn in returning if turn.cached_tokens is not None and turn.prompt_tokens]
    cached_share = (
        sum(turn.cached_tokens or 0 for turn in reported) / sum(turn.prompt_tokens or 0 for turn in reported)
        if reported
        else None
    )
    decode_p90 = _percentile(rates, 0.1)
    ttft_p90 = _percentile(returning_ttft, 0.9)
    errors = len(turns) - len(done)
    meets = (
        errors == 0
        and decode_p90 is not None
        and ttft_p90 is not None
        and decode_p90 >= MIN_DECODE_TPS
        and ttft_p90 <= MAX_RETURNING_TTFT_S
    )
    return {
        "sessions": sessions,
        "turns": len(turns),
        "errors": errors,
        "max_prompt_tokens": max((turn.prompt_tokens or 0 for turn in done), default=0),
        "decode_tps_p50": _round(_percentile(rates, 0.5), 1),
        "decode_tps_p90": _round(decode_p90, 1),
        "first_turn_ttft_p50_s": _round(_percentile(first_ttft, 0.5)),
        "returning_ttft_p50_s": _round(_percentile(returning_ttft, 0.5)),
        "returning_ttft_p90_s": _round(ttft_p90),
        "returning_cached_share": _round(cached_share, 3),
        "wall_s": round(wall_s, 1),
        "meets_bar": meets,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = openai.OpenAI(
        base_url=args.base_url, api_key=args.api_key or "none", timeout=REQUEST_TIMEOUT_S, max_retries=0
    )
    started = datetime.now(UTC)
    entry = model_entry(args.base_url, args.api_key, args.model)
    warm_up = wait_until_loaded(client, args.model)
    probe_words = args.prefix_tokens
    probe = words_text(random.Random(f"{args.seed}-system"), probe_words, "You are a coding agent. Project notes:")
    sizing = Turn(sessions=0, session=0, turn=0)
    probe_messages = [{"role": "system", "content": probe}, {"role": "user", "content": "Reply with the word ready."}]
    stream_turn(client, args.model, probe_messages, sizing, 1)
    if sizing.error or not sizing.prompt_tokens:
        raise SystemExit(f"could not size the prompt: {sizing.error or 'no usage.prompt_tokens'}")
    tokens_per_word = sizing.prompt_tokens / probe_words
    system_words = round(args.prefix_tokens / tokens_per_word)
    system = words_text(random.Random(f"{args.seed}-system"), system_words, "You are a coding agent. Project notes:")
    words_per_turn = round(args.turn_tokens / tokens_per_word)
    levels = []
    records: list[Turn] = []
    for sessions in args.sessions:
        level: list[Turn] = []
        barrier = threading.Barrier(sessions)
        threads = [
            threading.Thread(
                target=run_session,
                args=(client, args.model, system, words_per_turn, args, sessions, index, barrier, level),
            )
            for index in range(sessions)
        ]
        began = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        summary = summarise(sessions, level, time.monotonic() - began)
        levels.append(summary)
        records.extend(level)
        print_level(summary)
    return {
        "base_url": args.base_url,
        "model": args.model,
        "model_entry": entry,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "warm_up": warm_up,
        "shape": {
            "prefix_tokens": args.prefix_tokens,
            "turn_tokens": args.turn_tokens,
            "turns": args.turns,
            "output_tokens": args.output_tokens,
            "tokens_per_word": round(tokens_per_word, 3),
        },
        "bar": {"min_decode_tps_p90": MIN_DECODE_TPS, "max_returning_ttft_p90_s": MAX_RETURNING_TTFT_S},
        "levels": levels,
        "turns": [asdict(record) for record in records],
    }


def print_level(level: dict[str, Any]) -> None:
    cached = level["returning_cached_share"]
    share = "not reported" if cached is None else f"{cached:.1%}"
    print(
        f"{level['sessions']:>2} sessions  {level['turns']:>3} turns  {level['errors']} errors  "
        f"max prompt {level['max_prompt_tokens']:>7}  decode p50/p90 {level['decode_tps_p50']}/{level['decode_tps_p90']} tok/s  "
        f"returning TTFT p50/p90 {level['returning_ttft_p50_s']}/{level['returning_ttft_p90_s']} s  "
        f"cached {share}  {'meets' if level['meets_bar'] else 'misses'} the bar",
        flush=True,
    )


def _counts(text: str) -> list[int]:
    try:
        counts = [int(part) for part in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated whole numbers, got {text!r}") from exc
    if not counts or min(counts) < 1:
        raise argparse.ArgumentTypeError("every session count must be at least 1")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})"
    )
    parser.add_argument("--api-key", default=os.environ.get("SIE_API_KEY", ""), help="Bearer key (env SIE_API_KEY)")
    parser.add_argument("--model", required=True, help="model id, for example Qwen/Qwen3.8-27B-FP8:h100-256k-batch")
    parser.add_argument("--sessions", type=_counts, default=[1, 2, 4], help="session counts to run, e.g. 1,2,4,6,8")
    parser.add_argument("--prefix-tokens", type=int, default=16_000, help="shared system prompt size (default 16000)")
    parser.add_argument("--turn-tokens", type=int, default=6_000, help="new input per turn (default 6000)")
    parser.add_argument("--turns", type=int, default=7, help="turns per session (default 7, about 61K at the end)")
    parser.add_argument("--output-tokens", type=int, default=500, help="answer length per turn (default 500)")
    parser.add_argument("--seed", default="session-load", help="seed for the generated input")
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, help=f"JSON report path (default {DEFAULT_REPORT})"
    )
    args = parser.parse_args()
    for name in ("prefix_tokens", "turn_tokens", "turns", "output_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report written to {args.report}")
    return 1 if any(level["errors"] for level in report["levels"]) else 0


if __name__ == "__main__":
    sys.exit(main())
