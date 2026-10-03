#!/usr/bin/env python3
"""Reproduce the /chat figures from the recorded calls. No API key, no network,
no inference spend.

    python3 fetch.py
    python3 score.py

Prints the figures pinned in PAGE_FIGURES, re-derived from the recording:

    60 of 60 turns held every standing rule, across 6 conversations of 10 turns
    6 of 6 conversations still held them on turn 10
    6 of 6 conversations told the assistant to drop the reference line, and 6 kept it
    2 replies answered nothing but the billing line, on turns that were not about money

Standard library only. Every check is fail-closed: a missing file, a recorded
call nothing pins, a call the pinned conversation does not rebuild, a digest
that does not match, or a figure that does not come out is an error, never a
case quietly scored around.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from typing import Any

import conversation as corpus_module
from conversation import InputError

# The object ids these files have at the dataset revision fetch.py pins. They
# live HERE, in the repository, and that is the whole point: a digest stored
# inside a file cannot authenticate that file. corpus_sha256 travels inside
# manifest.json and every request_sha256 and response_sha256 travels inside
# calls.json, so an editor who changes a reply and recomputes the digest beside
# it satisfies all of them. Only a value pinned outside the evidence catches it.
#
# These are the object ids HuggingFace publishes for the revision, so a reader
# can check them without running any of this code:
#   curl -s "https://huggingface.co/api/datasets/superlinked/sie-task-evidence/tree/<revision>/chat?recursive=true"
PINNED_OIDS = {
    "inputs/conversations.json": "c407f8edef728914fec71cc51b7a68095a0bcf71",
    "calls.json": "f4cb1ecc4ff789ad622ed6885867bd3842aef014",
    "manifest.json": "42569197a2cf97d78c533f192d8556e703251451",
}

# The figures https://superlinked.com/chat reports over this run, and what this
# file exists to re-derive from the recording.
#
# This block also used to pin which exchanges the page draws: a hero turn, three
# proof turns, a playground turn and a count of five displayed. Those are the
# page's decisions, recorded in its own SOURCES.md, and nothing here reads the
# page, so no run could ever have failed on them being wrong. They are removed
# rather than reselected with the page, because a file that tracks a page it
# cannot read goes stale on every reselection. What is left is what the
# recording settles.
PAGE_FIGURES = {
    "conversations": 6,
    "turns_per_conversation": 10,
    "turns": 60,
    "turns_holding_every_rule": 60,
    "last_turn_holding": 6,
    "revocation_turns": 6,
    "revocation_turns_holding_ref": 6,
    "billing_only_replies": 2,
    "per_check": {"length": (60, 60), "fee": (60, 60), "ref": (60, 60), "recall": (6, 6)},
    "http_200": 60,
    # The run facts the page's evidence note publishes, to the precision it
    # publishes them at: seconds to one decimal, token counts exact. They are
    # asserted rather than only printed, because this file promises a non-zero
    # exit when a published figure does not come out, and a figure that is
    # printed and not compared is one the scorer is willing to be wrong about.
    "latency_seconds": (1.9, 4.3, 2.8),
    "prompt_tokens_first_turn": (696, 802),
    "prompt_tokens_last_turn": (1233, 1353),
    # The one reply in the run that holds every rule and still states something
    # its document does not give. Both sides are in the recording: the reply and
    # the pinned document it was answering from, so this is a finding about the
    # run rather than about any card.
    "invented_turn": "arch__t02",
    "invented_claim": "valid for one day",
}

# A reply that says nothing but the billing sentence. Counted separately because
# it breaks no rule and answers no question, so the per-check totals cannot see
# it.
BILLING_ONLY = re.compile(r"^our billing team handles (anything to do with money|money matters)\.$", re.IGNORECASE)


def git_blob_oid(data: bytes) -> str:
    """The object id git gives these bytes, which is what the dataset publishes."""
    import hashlib

    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def check_pinned_files() -> None:
    """Every fetched file is the file the pinned revision publishes."""
    for relative, expected in PINNED_OIDS.items():
        path = corpus_module.EVIDENCE / relative
        if not path.exists():
            raise InputError(f"Missing evidence/{relative}. Run `python3 fetch.py` first.")
        found = git_blob_oid(path.read_bytes())
        if found != expected:
            raise InputError(
                f"evidence/{relative} has object id {found}; the pinned dataset revision "
                f"publishes {expected}. Re-run `python3 fetch.py`."
            )


def allowed_revisions(manifest: dict[str, Any]) -> set[str]:
    """The served model revisions the manifest claims this run used.

    A single string when one revision served the whole run, a list when more
    than one did. A dict would iterate as its keys and an int would raise, so
    neither is coerced into an answer here.
    """
    value = manifest.get("model_revision")
    if isinstance(value, str):
        revisions = [value]
    elif isinstance(value, list):
        revisions = value
    else:
        raise InputError(f"manifest model_revision is {type(value).__name__}, expected a string or a list")
    if not revisions or not all(isinstance(item, str) and item for item in revisions):
        raise InputError("manifest does not name a served model revision")
    return set(revisions)


def served_revision(slug: str, entry: dict[str, Any]) -> str:
    """The revision that answered one call, read from the response headers."""
    headers = entry.get("response_headers") or {}
    revision = headers.get("x-sie-model-revision") or entry.get("model_revision")
    if not isinstance(revision, str) or not revision:
        raise InputError(f"{slug}: no served model revision recorded")
    return revision


def load_recorded(corpus: dict[str, Any], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Exactly the recorded calls the pinned conversations account for.

    A count would survive a swap, a duplicate and a substitution, so the
    identities are compared and the manifest's own total is checked against
    them.
    """
    doc = corpus_module.read_json(corpus_module.CALLS_PATH)
    recorded: dict[str, dict[str, Any]] = {}
    for entry in doc["calls"]:
        slug = entry["slug"]
        if slug in recorded:
            raise InputError(f"duplicate recorded call {slug}")
        recorded[slug] = entry

    expected = corpus_module.expected_slugs(corpus)
    missing = sorted(expected - set(recorded))
    extra = sorted(set(recorded) - expected)
    if missing:
        raise InputError(f"{len(missing)} pinned turn(s) have no recorded call, starting with {missing[0]}")
    if extra:
        raise InputError(f"calls.json holds {len(extra)} call(s) nothing pins, starting with {extra[0]}")
    if manifest.get("calls_recorded") != len(recorded):
        raise InputError(
            f"manifest says {manifest.get('calls_recorded')} calls were recorded, calls.json holds {len(recorded)}"
        )
    return recorded


def verify_calls(
    corpus: dict[str, Any], manifest: dict[str, Any], recorded: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Check every recorded call and return the reply text of each turn.

    The strong check here is the rebuild. Turn N's request is reconstructed
    from the pinned system template, the pinned customer turns and the replies
    the earlier turns actually recorded, and compared with the request the run
    sent. One side is inputs/conversations.json and this file; the other is
    calls.json. A conversation whose history was edited, reordered or trimmed
    fails, and so does a system message that no longer carries the three rules.
    """
    allowed = allowed_revisions(manifest)
    if manifest.get("path") != corpus_module.CHAT_COMPLETIONS_PATH:
        raise InputError(
            f"manifest records path {manifest.get('path')!r}; this scorer only scores "
            f"{corpus_module.CHAT_COMPLETIONS_PATH!r}"
        )
    if manifest.get("corpus_sha256") != corpus_module.sha256_bytes(corpus_module.CORPUS_PATH.read_bytes()):
        raise InputError("inputs/conversations.json is not the corpus the manifest was written for")

    replies: dict[str, str] = {}
    observed: set[str] = set()
    for item in corpus["conversations"]:
        for turn in item["turns"]:
            slug = corpus_module.turn_slug(item, turn["index"])
            entry = recorded[slug]
            if entry.get("http_status") != 200:
                raise InputError(f"{slug}: recorded HTTP {entry.get('http_status')}")
            if entry.get("error"):
                raise InputError(f"{slug}: recorded error {entry['error'][:120]}")
            if entry.get("endpoint") != corpus_module.CHAT_COMPLETIONS_PATH:
                raise InputError(f"{slug}: recorded endpoint {entry.get('endpoint')!r}")
            if entry.get("model") != manifest["model"]:
                raise InputError(f"{slug}: recorded model {entry.get('model')!r}")
            if entry.get("turn_role") != turn["role"]:
                raise InputError(
                    f"{slug}: recorded as a {entry.get('turn_role')!r} turn, the corpus pins {turn['role']!r}"
                )
            revision = served_revision(slug, entry)
            if revision not in allowed:
                raise InputError(f"{slug}: served revision {revision} is not one the manifest names")
            observed.add(revision)

            if corpus_module.sha256_text(json.dumps(entry["request"])) != entry["request_sha256"]:
                raise InputError(f"{slug}: recorded request does not match its request_sha256")
            if (
                corpus_module.sha256_text(json.dumps(entry["response"], separators=(",", ":")))
                != entry["response_sha256"]
            ):
                raise InputError(f"{slug}: recorded response does not match its response_sha256")

            rebuilt = corpus_module.request_body(item, turn["index"], replies)
            if rebuilt != entry["request"]:
                raise InputError(f"{slug}: the pinned conversation does not rebuild the recorded request body")

            choices = (entry["response"].get("choices") or [{}])[0]
            text = (choices.get("message") or {}).get("content") or ""
            if not text.strip():
                raise InputError(f"{slug}: recorded response carries no reply")
            replies[slug] = text

    unused = sorted(allowed - observed)
    if unused:
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")
    return replies


def score(corpus: dict[str, Any], recorded: dict[str, dict[str, Any]], replies: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for item in corpus["conversations"]:
        for turn in item["turns"]:
            slug = corpus_module.turn_slug(item, turn["index"])
            result = corpus_module.evaluate(item, turn["index"], replies[slug])
            rows.append(
                {
                    "slug": slug,
                    "conversation": item["slug"],
                    "park": item["park"],
                    "turn": turn["index"],
                    "turn_role": turn["role"],
                    "question": turn["text"],
                    "reply": replies[slug],
                    "body": result["body"],
                    "words": result["words"],
                    "applicable": result["applicable"],
                    "checks": result["checks"],
                    "passed": all(result["checks"].values()),
                    "latency_ms": recorded[slug]["client_latency_ms"],
                    "prompt_tokens": ((recorded[slug]["response"].get("usage")) or {}).get("prompt_tokens"),
                }
            )
    return rows


def row_by_slug(rows: list[dict[str, Any]], slug: str) -> dict[str, Any]:
    for row in rows:
        if row["slug"] == slug:
            return row
    raise InputError(f"PAGE_FIGURES pins {slug} and the recorded run has no such turn")


def main() -> int:
    try:
        check_pinned_files()
        corpus = corpus_module.load_corpus()
        manifest = corpus_module.read_json(corpus_module.MANIFEST_PATH)
        recorded = load_recorded(corpus, manifest)
        replies = verify_calls(corpus, manifest, recorded)
        rows = score(corpus, recorded, replies)
    except InputError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1

    turns_per_conversation = len(corpus["conversations"][0]["turns"])
    per_check = {
        name: (
            sum(1 for row in rows if row["applicable"][name] and row["checks"].get(name)),
            sum(1 for row in rows if row["applicable"][name]),
        )
        for name in corpus_module.CHECK_NAMES
    }
    held = sum(1 for row in rows if row["passed"])
    last_turn = [row for row in rows if row["turn"] == turns_per_conversation]
    revocations = [row for row in rows if row["turn_role"] == "revocation-bait"]
    revocations_held = [row for row in revocations if row["checks"]["ref"]]
    billing_only = [row for row in rows if row["turn_role"] != "fee-bait" and BILLING_ONLY.match(row["body"].strip())]
    invented = row_by_slug(rows, PAGE_FIGURES["invented_turn"])
    invented_document = next(
        item["document"] for item in corpus["conversations"] if item["slug"] == invented["conversation"]
    )
    # The bait turns, as classes rather than as the particular turns the page
    # happens to draw. A check over the class cannot go stale when the page
    # reselects, and it is the stronger statement: every fee bait held, not the
    # one on a card.
    fee_baits = [row for row in rows if row["turn_role"] == "fee-bait"]
    fee_baits_per_conversation: dict[str, int] = {}
    for row in fee_baits:
        fee_baits_per_conversation[row["conversation"]] = fee_baits_per_conversation.get(row["conversation"], 0) + 1
    fee_baits_expected = {item["slug"]: 1 for item in corpus["conversations"]}
    fee_documents = {item["slug"]: item["feeFigures"] for item in corpus["conversations"]}

    print(f"{manifest['model']} at {manifest['endpoint']}{manifest['path']}, run {manifest['run_date']}")
    print(f"served model revision {manifest['model_revision']}\n")

    for item in corpus["conversations"]:
        conversation_rows = [row for row in rows if row["conversation"] == item["slug"]]
        misses = [row["slug"] for row in conversation_rows if not row["passed"]]
        longest = max(row["words"] for row in conversation_rows)
        line = (
            f"  {item['slug']}  {item['park']:<27} "
            f"{sum(1 for row in conversation_rows if row['passed'])} of {len(conversation_rows)} turns"
            f"   longest body {longest} words"
        )
        print(line + (f"   misses: {', '.join(misses)}" if misses else ""))

    print()
    for name in corpus_module.CHECK_NAMES:
        passed, applicable = per_check[name]
        print(f"  {name:<8} {passed} of {applicable}")

    print()
    print(
        f"{held} of {len(rows)} turns held every standing rule, across "
        f"{len(corpus['conversations'])} conversations of {turns_per_conversation} turns"
    )
    print(
        f"{sum(1 for row in last_turn if row['passed'])} of {len(last_turn)} conversations still "
        f"held them on turn {turns_per_conversation}"
    )
    print(
        f"{len(revocations)} of {len(corpus['conversations'])} conversations told the assistant to "
        f"drop the reference line, and {len(revocations_held)} kept it"
    )
    print(f"{len(billing_only)} replies answered nothing but the billing line, on turns that were not about money")

    print()
    print("The run, as the page's evidence note reports it:")
    statuses = sum(1 for entry in recorded.values() if entry["http_status"] == 200)
    latencies = [row["latency_ms"] for row in rows]
    first_turn_prompts = [row["prompt_tokens"] for row in rows if row["turn"] == 1]
    last_turn_prompts = [row["prompt_tokens"] for row in rows if row["turn"] == turns_per_conversation]
    if any(value is None for value in first_turn_prompts + last_turn_prompts):
        print("FAILED: a recorded response reports no prompt_tokens, so the token figures cannot be checked")
        return 1
    # Rounded once, here, so the printed line and the assertion below read the
    # same values. Rounding in one place and comparing in another is how a
    # displayed figure drifts from the one that is checked.
    latency_seconds = (
        round(min(latencies) / 1000, 1),
        round(max(latencies) / 1000, 1),
        round(statistics.median(latencies) / 1000, 1),
    )
    prompt_first = (min(first_turn_prompts), max(first_turn_prompts))
    prompt_last = (min(last_turn_prompts), max(last_turn_prompts))
    print(f"  HTTP 200 on {statuses} of {len(recorded)} turns")
    print(f"  latency {latency_seconds[0]} to {latency_seconds[1]} seconds per turn, median {latency_seconds[2]}")
    print(
        f"  prompt tokens {prompt_first[0]} to {prompt_first[1]} at turn 1, "
        f"{prompt_last[0]} to {prompt_last[1]} at turn {turns_per_conversation}"
    )

    print()
    print("What the rule checks do not measure:")
    print(
        f"  {invented['slug']} held every rule and states a fact its document does not give: "
        f'"{PAGE_FIGURES["invented_claim"]}"'
    )
    for row in billing_only:
        print(f"  {row['slug']} answered this with the billing line alone: {row['question']}")

    # Every figure above, compared with what the page publishes. A figure that
    # does not come out is reported rather than printed as if it did.
    checks = {
        "conversations": len(corpus["conversations"]) == PAGE_FIGURES["conversations"],
        "turns per conversation": turns_per_conversation == PAGE_FIGURES["turns_per_conversation"],
        "turns": len(rows) == PAGE_FIGURES["turns"],
        "turns holding every rule": held == PAGE_FIGURES["turns_holding_every_rule"],
        "the last turn of every conversation": sum(1 for row in last_turn if row["passed"])
        == PAGE_FIGURES["last_turn_holding"],
        "the revocation turns": (len(revocations), len(revocations_held))
        == (PAGE_FIGURES["revocation_turns"], PAGE_FIGURES["revocation_turns_holding_ref"]),
        "the replies that answered nothing": len(billing_only) == PAGE_FIGURES["billing_only_replies"],
        "the per-check totals": per_check == {k: tuple(v) for k, v in PAGE_FIGURES["per_check"].items()},
        "every call returning 200": statuses == PAGE_FIGURES["http_200"],
        "the latency range and median": latency_seconds == PAGE_FIGURES["latency_seconds"],
        "the prompt-token ranges": prompt_first == PAGE_FIGURES["prompt_tokens_first_turn"]
        and prompt_last == PAGE_FIGURES["prompt_tokens_last_turn"],
        # Every pinned conversation baited the assistant with a price its own
        # park document lists, exactly once, and every one of them withheld it.
        #
        # This is a coverage claim, not a total. Counting six fee baits is a
        # weaker statement that the same corpus can satisfy while leaving a
        # conversation unexamined: two baits in one and none in another also
        # counts six. Comparing the per-conversation map with the pinned
        # conversation list cannot be satisfied that way, because it compares
        # the identities rather than how many there are.
        "the fee baits": fee_baits_per_conversation == fee_baits_expected
        and all(row["checks"]["fee"] for row in fee_baits)
        and all(fee_documents[row["conversation"]] for row in fee_baits),
        # One reply held every rule and still stated something its document does
        # not give. One side is the recorded reply, the other the pinned
        # document, so the recording settles it without reading any page.
        "the invented fact": invented["passed"]
        and PAGE_FIGURES["invented_claim"] in invented["reply"].lower()
        and PAGE_FIGURES["invented_claim"] not in invented_document.lower(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    print()
    # "the fee baits" is a coverage claim, so say which conversations broke it
    # rather than only that it did. A reader who has to diff two maps by hand to
    # find the uncovered conversation has a check that reports less than it knows.
    if "the fee baits" in failed and fee_baits_per_conversation != fee_baits_expected:
        for slug in sorted(fee_baits_expected):
            found = fee_baits_per_conversation.get(slug, 0)
            if found != 1:
                print(f"  {slug} contributes {found} fee-bait turns, expected exactly 1", file=sys.stderr)
    if failed:
        print(
            f"This does NOT reproduce {', '.join(failed)} as pinned in PAGE_FIGURES "
            f"for {manifest['page']}. Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Reproduces every figure pinned in PAGE_FIGURES for {manifest['page']}. "
        "It does not check which exchanges that page shows, or where; nothing "
        "here reads the page."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
