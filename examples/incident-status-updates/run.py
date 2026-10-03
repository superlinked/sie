#!/usr/bin/env python3
"""Turn internal incident reports into public status updates with SIE Cloud.

    uv run python run.py                      # every pinned case
    uv run python run.py --case sessionstore  # one case
    python3 run.py --show sessionstore        # print a request, no network

One call per case, the call the /chat task page shows:

    POST https://api.superlinked.com/v1/chat/completions
    {"model": "Qwen/Qwen3.8-27B-FP8",
     "messages": [<instruction>, <report>],
     "max_completion_tokens": 256}

No sampling fields are sent, so the answer is whatever the model profile
defaults produce. Results go to --output as a manifest.json and a calls.json in
the dataset's own shape, one entry per call holding the request, the response,
the HTTP status, the served model revision and the round-trip time. The key
comes from SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import prompt

HERE = Path(__file__).resolve().parent


def record(client: Any, case: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Send one case and return its calls.json entry."""
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    response = client.chat_completions(
        body["model"],
        body["messages"],
        max_completion_tokens=body["max_completion_tokens"],
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so `response` stays the server's own envelope and nothing else.
    envelope = {key: value for key, value in response.items() if key != "request"}
    # Stop at the case that broke rather than at the end. score.py rejects an
    # answerless entry or a missing revision later, but by then the remaining
    # calls have already been paid for.
    choices = envelope.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], dict) else None
    if not isinstance(content, str) or not content.strip():
        raise SystemExit(f"{case['slug']}: the response carried no answer text")
    if not client.last_model_revision:
        raise SystemExit(f"{case['slug']}: the response reported no served model revision")
    return {
        "slug": case["slug"],
        "requested_at": requested_at,
        "request": {
            "method": "POST",
            # The URL the SDK actually used, not this module's default, so a run
            # against a regional endpoint records where it really went.
            "url": client.base_url.rstrip("/") + prompt.CHAT_COMPLETIONS_PATH,
            "body": body,
        },
        "status": 200,
        "response": envelope,
        "response_sha256": prompt.sha256_bytes(prompt.compact_json(envelope)),
        "model_revision": client.last_model_revision,
        "timing": {"duration_ms": elapsed_ms},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record status updates from pinned incident reports")
    parser.add_argument("--case", action="append", default=[], help="slug to run; repeatable, default all")
    parser.add_argument("--show", metavar="SLUG", help="print one request body and exit, without calling anything")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="directory for manifest.json and calls.json"
    )
    return parser.parse_args()


def main() -> int:
    import json

    args = parse_args()
    cases_doc = prompt.load_cases()
    cases = {case["slug"]: case for case in cases_doc["cases"]}

    if args.show:
        case = cases.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}. Known: {', '.join(cases)}")
        body = prompt.request_body(cases_doc, prompt.wikitext(case), case)
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    # Imported only on the path that calls the API, so `--show` runs on a bare
    # python3 with nothing installed.
    from sie_sdk import SIEClient

    selected = args.case or list(cases)
    unknown = [slug for slug in selected if slug not in cases]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    duplicated = sorted({slug for slug in selected if selected.count(slug) > 1})
    if duplicated:
        raise SystemExit(f"Repeated --case: {', '.join(duplicated)}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figure without a key, run score.py instead.")

    client = SIEClient(os.environ.get("SIE_BASE_URL", prompt.ENDPOINT), api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{prompt.CHAT_COMPLETIONS_PATH}")
    print(f"model    {cases_doc['model']}")

    entries = []
    for slug in selected:
        case = cases[slug]
        entry = record(client, case, prompt.request_body(cases_doc, prompt.wikitext(case), case))
        entries.append(entry)
        revision = entry["model_revision"] or "not reported"
        print(f"{slug:<32} {entry['timing']['duration_ms']:>8.0f} ms  revision {revision}")

    revisions = sorted({entry["model_revision"] for entry in entries if entry["model_revision"]})
    manifest = {
        "task": "chat",
        "page": "https://superlinked.com/sre",
        "endpoint": base_url,
        "path": prompt.CHAT_COMPLETIONS_PATH,
        "model": cases_doc["model"],
        "model_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run_date": datetime.now(UTC).date().isoformat(),
        "recorded_by": "examples/incident-status-updates/run.py",
        "sampling": "model profile defaults; no temperature, top_p or seed was sent",
        "calls_recorded": len(entries),
        "response_sha256": (
            "sha256 of json.dumps(response, ensure_ascii=False, separators=(',', ':')).encode('utf-8')"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("manifest.json", manifest), ("calls.json", {"calls": entries})):
        (args.output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}/manifest.json and {args.output}/calls.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
