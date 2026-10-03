#!/usr/bin/env python3
"""Screen the study's prompts again with one of SIE's two guard models, writing rows in the recorded shape.

    uv run python run.py --arm gliguard --set toxicchat --limit 5        # SIE's hosted API, 5 prompts
    uv run python run.py --arm gliguard --set aegis                      # all 1,915 Aegis 2.0 prompts
    uv run python run.py --arm qwen3guard --base-url http://localhost:8080 --set toxicchat --limit 20
    python3 score.py --run runs

Arms:

    gliguard    fastino/gliguard-LLMGuardrails-300M through /v1/extract with labels safe and unsafe, the
                call the study made. Defaults to SIE's hosted API, https://api.superlinked.com.
    qwen3guard  Qwen/Qwen3Guard-Gen-4B through /v1/chat/completions: the prompt as the only user message,
                temperature 0, max_tokens 64. The model's chat template wraps it in the moderation policy.
                It is not on the hosted API yet, so point --base-url at an SIE server that serves it.

SIE_API_KEY is read from the environment and never written out; a local server without auth needs none.
SIE_BASE_URL, or --base-url, picks the server. Calls go through sie_sdk.SIEClient.

Rows land in runs/<set>__<stem>.jsonl, one per prompt, with the fields the recorded rows carry: index
(the prompt's position in the set's source file), output, refusal, latency_s, tokens_in, tokens_out,
attempts, error. A call that still fails after its retries is recorded with its error; the rules score a
failed GLiGuard row as safe and a failed Qwen3Guard row as harmful, as the study did.

The prompts come from the set files fetch.py downloads. Nothing here prints them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import study

MAX_ATTEMPTS = 6
MAX_BACKOFF_S = 30
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR = 500
STEMS = {"gliguard": "sie_fastino__gliguard-LLMGuardrails-300M", "qwen3guard": "chat_Qwen__Qwen3Guard-Gen-4B"}


def is_transient(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == HTTP_TOO_MANY_REQUESTS or status >= HTTP_SERVER_ERROR
    return type(error).__name__ in {"ProvisioningError", "ServerError", "TimeoutError", "ConnectError"}


class Screen:
    def __init__(self, arm: str, base_url: str) -> None:
        # Imported here: only the live run needs the SDK; fetch.py and score.py run on the standard library.
        from sie_sdk import SIEClient

        api_key = os.environ.get("SIE_API_KEY", "").strip() or None
        self.arm = arm
        self.client = SIEClient(base_url, api_key=api_key, timeout_s=300)

    def call(self, text: str) -> tuple[str, int, int]:
        if self.arm == "gliguard":
            result = self.client.extract(study.GLIGUARD, {"text": text}, labels=study.GLIGUARD_LABELS)
            classes = [{"label": c["label"], "score": c["score"]} for c in result.get("classifications") or []]
            return json.dumps({"entities": [], "classifications": classes}), 0, 0
        response = self.client.chat_completions(
            study.QWEN3GUARD,
            [{"role": "user", "content": text}],
            max_tokens=study.QWEN3GUARD_MAX_TOKENS,
            temperature=0,
        )
        usage = response.get("usage") or {}
        content = response["choices"][0]["message"].get("content") or ""
        return content, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def row(self, row: study.Row) -> dict[str, Any]:
        error: str | None = None
        for attempt in range(MAX_ATTEMPTS):
            started = time.monotonic()
            try:
                output, tokens_in, tokens_out = self.call(row.text)
            except (TypeError, AttributeError, KeyError, NameError):
                raise  # a bug in this script, not a failed call
            except Exception as exc:  # noqa: BLE001 - recorded per row once it is final
                error = f"{type(exc).__name__}: {getattr(exc, 'status_code', '')}".strip()
                if not is_transient(exc):
                    break
                time.sleep(min(2**attempt, MAX_BACKOFF_S))
                continue
            return {
                "index": row.index,
                "output": output,
                "refusal": None,
                "latency_s": time.monotonic() - started,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "attempts": attempt + 1,
                "error": None,
            }
        return {
            "index": row.index,
            "output": "",
            "refusal": None,
            "latency_s": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "attempts": attempt + 1,
            "error": error,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen the study's prompts with GLiGuard or Qwen3Guard on SIE")
    parser.add_argument("--arm", choices=sorted(STEMS), required=True)
    parser.add_argument("--set", choices=[s.key for s in study.SETS], required=True)
    parser.add_argument("--base-url", default=os.environ.get("SIE_BASE_URL", study.SIE_BASE_URL))
    parser.add_argument("--limit", type=int, help="the first N prompts of the set only")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=study.HERE / "runs")
    args = parser.parse_args()
    if args.arm == "qwen3guard" and args.base_url.rstrip("/") == study.SIE_BASE_URL:
        raise SystemExit("Qwen3Guard is not on SIE's hosted API yet; pass --base-url for an SIE server that serves it")

    rows = study.load_set(args.set)[: args.limit]
    screen = Screen(args.arm, args.base_url)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.set}__{STEMS[args.arm]}.jsonl"
    print(f"{args.arm} on {args.set}: {len(rows)} prompts to {args.base_url}")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(screen.row, rows))
    target.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    failed = sum(r["error"] is not None for r in records)
    print(f"Wrote {len(records)} rows to {target}, {failed} failed")
    print(f"Now run: python3 score.py --run {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
