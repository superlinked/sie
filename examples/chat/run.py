#!/usr/bin/env python3
"""Record ten-turn support conversations against SIE Cloud, three rules fixed.

    uv run python run.py                      # all six conversations
    uv run python run.py --conversation yose  # one conversation, ten turns
    python3 run.py --show yose 8              # print a request, no network

One call per turn, the call the /chat task page shows:

    POST https://api.superlinked.com/v1/chat/completions
    {"model": "Qwen/Qwen3.8-27B-FP8",
     "messages": [<system>, <user>, <assistant>, <user>, ...],
     "max_completion_tokens": 300}

The system message carries the park document and the three standing rules, and
it is sent once, at the head of every turn. Turn N carries every earlier
customer turn and every earlier reply the model actually gave, so turn ten
carries nine exchanges. Nothing re-states the rules after turn one.

No sampling fields are sent, so the reply is whatever the model profile
defaults produce. Results go to --output as a manifest.json and a calls.json in
the dataset's own shape, one entry per call holding the request, the response,
the HTTP status, the served model revision and the round-trip time. The key
comes from SIE_API_KEY and is never written out.

A run that dies part way through still writes what it recorded, to
manifest.partial.json and calls.partial.json, and the manifest says which turn
it stopped on. Those calls were paid for and their replies cannot be obtained
again, because no sampling fields are sent and the same request returns
different text next time.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import conversation as corpus_module

HERE = Path(__file__).resolve().parent


def reply_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def record(
    client: Any,
    conversation: dict[str, Any],
    index: int,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Send one turn and return its calls.json entry."""
    slug = corpus_module.turn_slug(conversation, index)
    requested_at = datetime.now(UTC).isoformat()
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
    # Stop at the turn that broke rather than at the end. A conversation is
    # sequential: turn N+1 has to carry turn N's reply, so continuing past a
    # turn with no reply text would record a history that never happened.
    if not reply_text(envelope).strip():
        raise SystemExit(f"{slug}: the response carried no reply text")
    if not client.last_model_revision:
        raise SystemExit(f"{slug}: no served model revision reported")
    return {
        "slug": slug,
        "role": "evidence",
        "run": "chat-completions-multi-turn",
        "conversation": conversation["slug"],
        "turn": index,
        "turn_role": conversation["turns"][index - 1]["role"],
        "requested_at": requested_at,
        # The URL the SDK actually used, not this module's default, so a run
        # against a regional endpoint records where it really went.
        "endpoint": corpus_module.CHAT_COMPLETIONS_PATH,
        "model": body["model"],
        "http_status": 200,
        "client_latency_ms": elapsed_ms,
        "model_revision": client.last_model_revision,
        "request": body,
        "response": envelope,
        "request_sha256": corpus_module.sha256_text(json.dumps(body)),
        "response_sha256": corpus_module.sha256_text(json.dumps(envelope, separators=(",", ":"))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record ten-turn conversations with fixed rules")
    parser.add_argument(
        "--conversation",
        action="append",
        default=[],
        help="conversation slug to run; repeatable, default all",
    )
    parser.add_argument(
        "--show",
        nargs=2,
        metavar=("SLUG", "TURN"),
        help="print one request body and exit, without calling anything",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "run-output",
        help="directory for manifest.json and calls.json",
    )
    return parser.parse_args()


def write_output(
    output: Path,
    entries: list[dict[str, Any]],
    base_url: str,
    turns_per_conversation: int,
    stopped_at: str | None,
) -> str:
    """Write the recorded calls and their manifest, and say which pair it wrote.

    `stopped_at` names the turn a failed run died on. A partial run is written
    under its own names and says so in the manifest, so it can never be read as
    a finished recording. Every count comes from the entries rather than from
    what was selected, because on a partial run those two differ.
    """
    complete = stopped_at is None
    revisions = sorted({entry["model_revision"] for entry in entries})
    stamps = sorted(entry["requested_at"] for entry in entries)
    recorded_conversations = {entry["conversation"] for entry in entries}
    manifest = {
        "task": "chat",
        "page": "https://superlinked.com/chat",
        "endpoint": base_url,
        "path": corpus_module.CHAT_COMPLETIONS_PATH,
        "model": corpus_module.MODEL,
        "model_revision": revisions[0] if len(revisions) == 1 else revisions,
        "run": "chat-completions-multi-turn",
        "run_complete": complete,
        "run_date": stamps[0][:10],
        "run_started_utc": stamps[0],
        "run_completed_utc": stamps[-1],
        "recorded_by": "examples/chat/run.py",
        "sampling": "model profile defaults; no temperature, top_p or seed was sent",
        "max_completion_tokens": corpus_module.MAX_COMPLETION_TOKENS,
        "max_words": corpus_module.MAX_WORDS,
        "conversations": len(recorded_conversations),
        "turns_per_conversation": turns_per_conversation,
        "calls_recorded": len(entries),
        "corpus_sha256": corpus_module.sha256_bytes(corpus_module.CORPUS_PATH.read_bytes()),
    }
    if not complete:
        manifest["partial"] = (
            f"The run stopped at {stopped_at}. These are the calls that had already been made, "
            "kept because their replies cannot be obtained again. This is not a recording of the "
            "run: score.py requires a recorded call for every pinned turn and refuses a set that "
            "is short of one."
        )
    suffix = "" if complete else ".partial"
    output.mkdir(parents=True, exist_ok=True)
    for name, value in ((f"manifest{suffix}.json", manifest), (f"calls{suffix}.json", {"calls": entries})):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return f"{output}/manifest{suffix}.json and {output}/calls{suffix}.json"


def recorded_replies() -> dict[str, str]:
    """Replies already on the record, for --show on a turn past the first.

    Turn eight's request contains turns one to seven and the replies they got,
    so showing it without the recording would mean inventing seven replies.
    """
    try:
        calls = corpus_module.read_json(corpus_module.CALLS_PATH)
    except corpus_module.InputError:
        return {}
    return {entry["slug"]: reply_text(entry.get("response") or {}) for entry in calls["calls"]}


def main() -> int:
    args = parse_args()
    doc = corpus_module.load_corpus()
    conversations = {item["slug"]: item for item in doc["conversations"]}

    if args.show:
        slug, raw_index = args.show
        item = conversations.get(slug)
        if item is None:
            raise SystemExit(f"Unknown conversation: {slug}. Known: {', '.join(sorted(conversations))}")
        index = int(raw_index)
        if not 1 <= index <= len(item["turns"]):
            raise SystemExit(f"{slug} has turns 1 to {len(item['turns'])}, not {index}")
        replies = recorded_replies() if index > 1 else {}
        try:
            body = corpus_module.request_body(item, index, replies)
        except corpus_module.InputError as error:
            raise SystemExit(f"{error} Run `python3 fetch.py` to show a turn past the first.") from error
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return 0

    selected = args.conversation or list(conversations)
    unknown = [slug for slug in selected if slug not in conversations]
    if unknown:
        raise SystemExit(f"Unknown conversation(s): {', '.join(unknown)}")
    repeated = sorted({slug for slug in selected if selected.count(slug) > 1})
    if repeated:
        raise SystemExit(f"Repeated --conversation: {', '.join(repeated)}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figures without a key, run score.py instead.")

    # Imported only on the path that calls the API, so `--show` and every
    # offline path run on a bare python3 with nothing installed. The key check
    # comes first, so a reader who forgot it gets that answer rather than an
    # import error about a dependency they may not need.
    from sie_sdk import SIEClient

    client = SIEClient(
        os.environ.get("SIE_BASE_URL", corpus_module.ENDPOINT),
        api_key=api_key,
        timeout_s=900,
    )
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{corpus_module.CHAT_COMPLETIONS_PATH}")
    print(f"model    {corpus_module.MODEL}")

    turns_per_conversation = len(conversations[selected[0]]["turns"])
    entries: list[dict[str, Any]] = []
    replies: dict[str, str] = {}
    stopped_at: str | None = None
    try:
        for slug in selected:
            item = conversations[slug]
            for turn in item["turns"]:
                index = turn["index"]
                stopped_at = corpus_module.turn_slug(item, index)
                body = corpus_module.request_body(item, index, replies)
                entry = record(client, item, index, body)
                entries.append(entry)
                replies[entry["slug"]] = reply_text(entry["response"])
                shown = replies[entry["slug"]].replace("\n", " / ")
                print(f"{entry['slug']:<14} {entry['client_latency_ms']:>8.0f} ms  {shown[:100]}")
    except BaseException:
        # Every entry here is a call that was made and paid for, and its reply
        # cannot be obtained again: no sampling fields are sent, so a rerun of
        # the same request returns different text. Losing them to an error on a
        # later turn is the expensive failure, so write them before re-raising.
        # `BaseException`, because record() exits with SystemExit, which is not
        # an Exception, and a keyboard interrupt costs the same calls.
        if entries:
            written = write_output(args.output, entries, base_url, turns_per_conversation, stopped_at)
            print(f"Stopped at {stopped_at}. Wrote {len(entries)} recorded turn(s) to {written}", file=sys.stderr)
        raise

    written = write_output(args.output, entries, base_url, turns_per_conversation, None)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
