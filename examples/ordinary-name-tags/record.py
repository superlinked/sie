"""Collect one bounded native extraction attempt per frozen paragraph."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from score import LABELS, MODELS, ROOT, load_frame, tag_set


class OneAttempt:
    """An HTTP request hook that refuses a second POST before it reaches the wire."""

    def __init__(self) -> None:
        self.posts = 0

    def __call__(self, request: Any) -> None:
        if request.method == "POST":
            if self.posts:
                raise RuntimeError("A second extraction attempt is disabled for this case")
            self.posts += 1


def public_entities(result: dict[str, Any], text: str) -> list[dict[str, Any]]:
    if result.get("error"):
        raise ValueError("The extraction item failed")
    entities = [
        {key: entity[key] for key in ("text", "label", "start", "end", "score")} for entity in result["entities"]
    ]
    tag_set(text, entities, spans=True)
    return entities


def public_usage(result: dict[str, Any]) -> dict[str, int]:
    """Retain token counters without exporting request IDs or other metadata."""
    usage = result.get("request", {}).get("usage", {})
    return {
        key: usage[key]
        for key in ("input_tokens", "output_tokens", "cached_input_tokens")
        if type(usage.get(key)) is int and usage[key] >= 0
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Explicit endpoint serving the requested model")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "inputs/cases.jsonl")
    parser.add_argument("--variants", type=Path, default=ROOT / "inputs/acceptable-tag-representations.json")
    parser.add_argument("--wall-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 3000:
        parser.error("--wall-seconds must be between 1 and 3000")
    cases, _ = load_frame(args.cases, args.variants)
    if len(cases) != 24:
        parser.error("This recording recipe is limited to the frozen 24-source frame")
    if args.output.exists():
        parser.error("Use a new output path; existing recordings are never overwritten")

    # These dependencies are only needed for explicitly requested inference.
    import httpx
    from sie_sdk import SIEClient

    settings = MODELS[args.model]
    deadline = time.monotonic() + args.wall_seconds
    stopped = False
    with args.output.open("x", encoding="utf-8") as output:
        for case in cases:
            row: dict[str, Any] = {
                "id": case["id"],
                "model": args.model,
                "text_sha256": case["text_sha256"],
                "expected_checkpoint_revision": settings["checkpoint_revision"],
                "status": "unattempted",
            }
            remaining = deadline - time.monotonic()
            if not stopped and remaining > 0:
                fence = OneAttempt()
                request = {
                    "model": args.model,
                    "items": [{"text": case["text"]}],
                    "params": {"labels": list(LABELS), "options": {"threshold": settings["threshold"]}},
                }
                row["request"] = request
                transport = httpx.Client(base_url=args.url, follow_redirects=False, event_hooks={"request": [fence]})
                try:
                    with SIEClient(
                        args.url,
                        api_key=os.environ.get("SIE_API_KEY", ""),
                        http_client=transport,
                        timeout_s=min(remaining, 60),
                    ) as client:
                        result = client.extract(
                            args.model,
                            {"text": case["text"]},
                            labels=list(LABELS),
                            options={"threshold": settings["threshold"]},
                            wait_for_capacity=False,
                            max_oom_retries=0,
                            provision_timeout_s=min(remaining, 60),
                        )
                    if result.get("model") != args.model:
                        raise ValueError("The response does not identify the requested model")
                    row.update(status="ok", entities=public_entities(result, case["text"]), usage=public_usage(result))
                except Exception as error:
                    transport.close()
                    # Exception text can contain headers or infrastructure addresses.
                    row.update(status="failed", error_type=type(error).__name__)
                    stopped = True
                row["physical_posts"] = fence.posts
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
