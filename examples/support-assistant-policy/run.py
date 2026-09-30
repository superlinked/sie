#!/usr/bin/env python3
"""Replay the 55 scripted support conversations against one model, turn by turn.

    uv run python run.py --smoke                        # SIE, 2 conversations, 12 calls
    uv run python run.py                                # SIE, all 55 conversations, 330 calls
    uv run python run.py --model claude-sonnet-5        # a rival arm, through its own SDK
    uv run python run.py --model my-model --provider openai --base-url http://localhost:8000/v1
    python3 run.py --show commitments__late-laptop-credit 2   # print a request, no network

Each conversation is six customer turns. Turn N sends the system prompt, every
earlier customer turn and every reply the model actually gave, so each model
keeps its own history. The customer turns never change.

Providers and their keys, read from the environment and never written out:

    sie        SIE_API_KEY, and SIE_BASE_URL to point at another SIE server
               (default https://api.superlinked.com). Sent through sie_sdk.SIEClient.
    openai     OPENAI_API_KEY. --base-url sends the same calls to any
               OpenAI-compatible endpoint, with OPENAI_API_KEY as its key.
    anthropic  ANTHROPIC_API_KEY.

A call that still fails after its retries is recorded with its error, and the
customer's next turn follows "(no reply)", as it would in a real widget. The
scorer counts a failed push as broken.

The output has the same shape as the recorded transcripts on the dataset, so
judge.py and score.py read both.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scenario

HERE = Path(__file__).resolve().parent
MAX_ATTEMPTS = 6
MAX_BACKOFF_S = 60
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500


class Backend:
    """One chat call for one arm: reply text, prompt tokens, output tokens, cached prompt tokens."""

    def __init__(self, model: str, provider: str, base_url: str | None) -> None:
        arm = scenario.ARMS.get(model, {})
        self.model = model
        self.provider = provider
        self.reasoning = arm.get("reasoning", "none")
        self.temperature = arm.get("temperature", 0)
        if provider == "sie":
            from sie_sdk import SIEClient

            self.base_url = base_url or os.environ.get("SIE_BASE_URL", scenario.SIE_BASE_URL)
            self.client: Any = SIEClient(self.base_url, api_key=require("SIE_API_KEY"), timeout_s=600)
        elif provider == "openai":
            import openai

            self.base_url = base_url or "https://api.openai.com/v1"
            self.client = openai.OpenAI(
                api_key=require("OPENAI_API_KEY"), base_url=self.base_url, max_retries=0, timeout=300
            )
        elif provider == "anthropic":
            import anthropic

            self.base_url = "https://api.anthropic.com"
            self.client = anthropic.Anthropic(api_key=require("ANTHROPIC_API_KEY"), max_retries=0, timeout=300)
        else:
            raise SystemExit(f"unknown provider {provider}")

    def call(self, messages: list[dict[str, str]]) -> tuple[str, int, int, int | None]:
        if self.provider == "sie":
            response = self.client.chat_completions(
                self.model,
                messages,
                max_tokens=scenario.MAX_OUTPUT_TOKENS,
                temperature=0,
                top_p=1,
            )
            usage = response.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            return (
                response["choices"][0]["message"].get("content") or "",
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                None if cached is None else int(cached),
            )
        if self.provider == "openai":
            extra: dict[str, Any] = {}
            if self.model in scenario.ARMS:
                extra["reasoning_effort"] = self.reasoning
            if self.temperature is not None:
                extra["temperature"] = self.temperature
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, max_completion_tokens=scenario.MAX_OUTPUT_TOKENS, **extra
            )
            # Some OpenAI-compatible servers omit usage; count those tokens as zero.
            usage = response.usage
            return (
                response.choices[0].message.content or "",
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
                None,
            )
        # The SDK takes no typed temperature for these models, so it goes in the body as the API accepts it.
        extra = (
            {"thinking": {"type": "disabled"}}
            if self.temperature is None
            else {"extra_body": {"temperature": self.temperature}}
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=scenario.MAX_OUTPUT_TOKENS,
            system=messages[0]["content"],
            messages=messages[1:],
            **extra,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens, None


def require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name}. To check the published figures without a key, run score.py instead.")
    return value


def is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == HTTP_TOO_MANY_REQUESTS or status >= HTTP_SERVER_ERROR
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectError"}


def call_with_retries(backend: Backend, messages: list[dict[str, str]]) -> tuple[str, int, int, int | None, str | None]:
    for attempt in range(MAX_ATTEMPTS):
        try:
            text, tokens_in, tokens_out, cached = backend.call(messages)
            return text, tokens_in, tokens_out, cached, None
        except (TypeError, AttributeError, KeyError, NameError):
            # A bug in this script, not a failed call: stop rather than record it against the model.
            raise
        except Exception as error:  # noqa: BLE001 - classified here, recorded when final
            message = f"{type(error).__name__}: {error}"[:300]
            if not is_transient(error) or attempt == MAX_ATTEMPTS - 1:
                return "", 0, 0, None, message
            time.sleep(min(2**attempt, MAX_BACKOFF_S))
    raise AssertionError("unreachable")


def messages_for(system: str, conversation: dict[str, Any], replies: list[str], index: int) -> list[dict[str, str]]:
    """The request for turn `index` (1-based), given the replies the model gave to the turns before it."""
    messages = [{"role": "system", "content": system}]
    for turn, reply in zip(conversation["turns"][: index - 1], replies, strict=True):
        messages.append({"role": "user", "content": turn["text"]})
        messages.append({"role": "assistant", "content": reply or "(no reply)"})
    messages.append({"role": "user", "content": conversation["turns"][index - 1]["text"]})
    return messages


def run_conversation(backend: Backend, system: str, conversation: dict[str, Any]) -> dict[str, Any]:
    replies: list[str] = []
    turns = []
    for turn in conversation["turns"]:
        messages = messages_for(system, conversation, replies, turn["index"])
        text, tokens_in, tokens_out, cached, error = call_with_retries(backend, messages)
        replies.append(text)
        row = {**turn, "reply": text, "error": error, "tokensIn": tokens_in, "tokensOut": tokens_out}
        if cached is not None:
            row["cachedTokensIn"] = cached
        turns.append(row)
        print(f"  {conversation['slug']} turn {turn['index']}: {'error' if error else 'ok'}", flush=True)
    return {"slug": conversation["slug"], "rule": conversation["rule"], "turns": turns}


def recorded_replies(slug: str) -> list[str]:
    """The SIE arm's recorded replies to one conversation, for --show on a turn past the first."""
    path = HERE / "evidence" / "transcripts" / f"{scenario.file_stem(scenario.SIE_MODEL)}.json"
    if not path.exists():
        raise SystemExit("Run `python3 fetch.py` first: a turn past the first carries the replies recorded before it.")
    run = json.loads(path.read_text(encoding="utf-8"))
    conversation = next(c for c in run["conversations"] if c["slug"] == slug)
    return [turn["reply"] for turn in conversation["turns"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the support-rules conversations against one model")
    parser.add_argument("--model", default=scenario.SIE_MODEL, help=f"default {scenario.SIE_MODEL}")
    parser.add_argument("--provider", choices=("sie", "openai", "anthropic"), help="needed for a model not listed")
    parser.add_argument("--base-url", help="another endpoint for the sie or openai provider")
    parser.add_argument("--smoke", action="store_true", help=f"only {' and '.join(scenario.SMOKE)}")
    parser.add_argument("--conversation", action="append", default=[], help="a conversation slug; repeatable")
    parser.add_argument("--concurrency", type=int, default=4, help="conversations in flight at once")
    parser.add_argument("--output", type=Path, help="default runs/<model>.json")
    parser.add_argument("--show", nargs=2, metavar=("SLUG", "TURN"), help="print one request and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = scenario.system_prompt()
    corpus = {row["slug"]: row for row in scenario.conversations()}

    if args.show:
        slug, raw_index = args.show
        if slug not in corpus:
            raise SystemExit(f"Unknown conversation {slug}. Known: {', '.join(corpus)}")
        index = int(raw_index)
        if not 1 <= index <= len(corpus[slug]["turns"]):
            raise SystemExit(f"{slug} has turns 1 to {len(corpus[slug]['turns'])}")
        replies = recorded_replies(slug)[: index - 1] if index > 1 else []
        body = {
            "model": scenario.SIE_MODEL,
            "messages": messages_for(system, corpus[slug], replies, index),
            "max_tokens": scenario.MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "top_p": 1,
        }
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    selected = list(scenario.SMOKE) if args.smoke else (args.conversation or list(corpus))
    unknown = [slug for slug in selected if slug not in corpus]
    if unknown:
        raise SystemExit(f"Unknown conversation(s): {', '.join(unknown)}")
    provider = args.provider or scenario.ARMS.get(args.model, {}).get("provider")
    if provider is None:
        raise SystemExit(f"{args.model} is not one of the six arms; pass --provider")

    backend = Backend(args.model, provider, args.base_url)
    print(f"{args.model} via {provider} at {backend.base_url}: {len(selected)} conversations")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(pool.map(lambda slug: run_conversation(backend, system, corpus[slug]), selected))

    out: dict[str, Any] = {
        "model": args.model,
        "settings": {
            "reasoning": backend.reasoning,
            "temperature": backend.temperature,
            "maxTokens": scenario.MAX_OUTPUT_TOKENS,
        },
        "runDate": datetime.now(timezone.utc).date().isoformat(),  # noqa: UP017 - --show runs on python3 < 3.11
        "conversations": results,
    }
    if provider == "sie":
        out["endpoint"] = {"url": backend.base_url, "servedModel": args.model}
    output = args.output or HERE / "runs" / f"{scenario.file_stem(args.model)}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    errors = sum(1 for c in results for t in c["turns"] if t["error"])
    print(f"{len(results)} conversations, {errors} failed turns -> {output}")
    print(f"Next: uv run python judge.py {output}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
