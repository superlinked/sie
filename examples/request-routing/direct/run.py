"""Make one native routing call; print native scores and a separate caller selection."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence

from sie_sdk import SIEClient

MODELS = ("fastino/GLiNER2.5-Decide", "knowledgator/gliclass-instruct-large-v1.0")
TEXT = "remove fencing from my calendar for may 7th"
LABELS = (
    "view calendar events",
    "edit calendar events",
    "view meeting schedule",
    "book a meeting room",
    "view reminders",
    "edit reminders",
    "view to-do list",
    "edit to-do list",
)


class RetryDisabled(RuntimeError):
    """An SDK branch attempted an additional inference send."""


class SingleSendClient(SIEClient):
    def _record_retry(self) -> None:
        # The pinned SDK calls this before any automatic resend.
        raise RetryDisabled


def select_handler(result: Mapping, labels: Sequence[str]) -> str:
    scores = result.get("classifications")
    if not isinstance(scores, list) or len(scores) != len(labels):
        raise ValueError("Expected one native score per supplied handler")
    seen: set[str] = set()
    checked: list[tuple[str, float]] = []
    for entry in scores:
        if not isinstance(entry, Mapping):
            raise TypeError("Malformed native score")
        label, score = entry.get("label"), entry.get("score")
        if not isinstance(label, str) or label not in labels or label in seen:
            raise ValueError("Native handler coverage differs from the supplied labels")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Native score is not finite and bounded")
        seen.add(label)
        checked.append((label, score))
    checked.sort(key=lambda item: item[1], reverse=True)
    if len(checked) < 2 or checked[0][1] == checked[1][1]:
        raise ValueError("There is no unique highest-scored handler")
    return checked[0][0]


def route(client: SIEClient, model: str) -> dict:
    native = client.extract(
        model,
        {"text": TEXT},
        labels=list(LABELS),
        options={"overflow_policy": "error"},
        wait_for_capacity=False,
        provision_timeout_s=30,
        max_oom_retries=0,
    )
    selected = select_handler(native, LABELS)
    return {"native": native, "caller": {"selected_handler": selected}}


def main() -> int:
    parser = argparse.ArgumentParser(description="One native SIE extract call. No handler is executed.")
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    args = parser.parse_args()
    endpoint = os.environ.get("SIE_BASE_URL", "").strip()
    if not endpoint:
        parser.error("SIE_BASE_URL is required")
    try:
        with SingleSendClient(
            endpoint,
            api_key=os.environ.get("SIE_API_KEY") or None,
            timeout_s=30,
            max_connections=1,
        ) as client:
            print(json.dumps(route(client, args.model), indent=2, allow_nan=False))
        return 0
    except Exception:  # noqa: BLE001 — SDK error details can contain endpoint credentials.
        print("The routing call or native score validation failed. No handler was executed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
