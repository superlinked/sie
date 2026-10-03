#!/usr/bin/env python3
"""Reproduce the /speech-to-text figures from the recorded calls. No API key, no
network, no inference spend.

    python3 fetch.py
    python3 score.py

Prints one line per clip, then the two figures the page publishes:

    Across all 12 recorded clips a search finds 56 of 61 key terms, and 5 came back wrong
    Pooled word error rate 8.1% over 594 human-transcribed words

It also checks the published per-clip figures for four of the clips. All twelve
are scored either way.

It checks figures, never composition. Which clips the page plays, how many, and
which of the wrong terms it shows are decisions made in sie-web; this script
cannot reach the page to read them, so a constant here asserting them would go
stale on the next reselection while still exiting 0.

Both counts depend on OpenAI's Whisper English text normalizer, which drops
filler words, strips transcriber tags and writes spoken numbers as digits, so
"twenty seven percent" is counted as "27%" and a different normalizer gives
different figures from the same transcripts. `whisper_normalizer.py` is a
standard-library port whose normalization is unmodified from Hugging Face
Transformers 4.57.6; it is loaded with the spelling map the run recorded, which
`fetch.py` downloads alongside the calls and `verify_evidence` hashes before
anything is scored. `matching.py` holds the key-term rule and normalizes
nothing itself.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from matching import find_term  # noqa: E402
from whisper_normalizer import load_english_normalizer  # noqa: E402

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
CALLS = ("snippet", "control")

# What https://superlinked.com/speech-to-text publishes. score.py exits
# non-zero if it computes anything else. If that happens, report it: it means
# the page or the evidence is wrong, and neither should be quietly adjusted.
PAGE_CLIPS = 12
PAGE_KEY_TERMS = (56, 61)
PAGE_POOLED_WER = "8.1%"
PAGE_REFERENCE_WORDS = 594
PAGE_TERMS_WRONG = 5
# Published per-clip figures, re-derived from the recordings. These are numbers,
# not a statement about where any of them appears: which clips a page prints a
# figure beside is the page's decision, it changes whenever the cards are
# reselected, and nothing in this script can reach the page to check it.
PUBLISHED_PER_CLIP = {
    "primock-uti-antibiotics": {"wer": "4.3%", "words": 23},
    "scotus-irs-levy-notices": {"wer": "20.0%", "words": 40},
    "ami-project-finance": {"wer": "3.7%", "words": 54},
    "ami-remote-control-chip": {"wer": "9.5%", "terms": (5, 5)},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """The digest the runner recorded: sorted keys, compact separators."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def load(name: str) -> Any:
    path = EVIDENCE / name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def edit_ops(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Word-level Levenshtein counts: substitutions, deletions, insertions."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            same = reference[i - 1] == hypothesis[j - 1]
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if same else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    i, j = len(reference), len(hypothesis)
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1):
            counts["hits" if reference[i - 1] == hypothesis[j - 1] else "substitutions"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    return counts


def written_amount(text: str) -> Decimal | None:
    """The first number written in a matched span, e.g. '€12.50' -> 12.50."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def transcript_text(entry: dict[str, Any]) -> str:
    items = entry["response"]["body"].get("items")
    if not items:
        raise SystemExit(f"{entry['slug']}: the recorded response has no items")
    text = (items[0].get("data") or {}).get("text")
    if not isinstance(text, str):
        raise SystemExit(f"{entry['slug']}: the recorded response item has no data.text")
    return text


def verify_evidence(manifest: dict[str, Any], calls: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    """Check the bytes before scoring them. Anything unreadable is a failure.

    Missing input fails the run; it is never passed over, because a scorer that
    skips what it cannot read prints a clean ratio over a set it did not score.
    """
    problems: list[str] = []

    # Each file is checked for existence before it is opened, so an absent one
    # is a reported failure naming the figure it takes away, never a traceback
    # and never a comparison that quietly does not happen.
    inputs_path = EVIDENCE / "inputs" / "inputs.json"
    if not inputs_path.is_file():
        problems.append(
            "inputs/inputs.json was not downloaded. It holds the human transcripts and the 61 key terms, so "
            "neither published figure can be computed without it; run: python3 fetch.py"
        )
    else:
        inputs_sha = sha256_bytes(inputs_path.read_bytes())
        if inputs_sha != manifest["inputs_sha256"]:
            problems.append(
                f"inputs.json hashes to {inputs_sha}, but the run was recorded against {manifest['inputs_sha256']}"
            )

    spelling_path = EVIDENCE / "inputs" / "normalizer.json"
    if not spelling_path.is_file():
        problems.append(
            "inputs/normalizer.json was not downloaded. Both published figures, the 56 of 61 key terms and the "
            "pooled 8.1% word error rate, are counted after this spelling map is applied, so neither can be "
            "checked without it; run: python3 fetch.py"
        )
    else:
        spelling_sha = sha256_bytes(spelling_path.read_bytes())
        if spelling_sha != manifest["normalizer"]["spelling_map_sha256"]:
            problems.append(
                "normalizer.json is not the spelling map the recorded counts were computed with, so no figure "
                "below would be comparable"
            )

    registered = {case["id"] for case in inputs["cases"]}
    seen: set[tuple[str, str]] = set()
    for entry in calls["calls"]:
        key = (entry["case"], entry["call"])
        if key in seen:
            problems.append(f"{entry['slug']}: recorded twice")
        seen.add(key)
        if canonical_sha256(entry["request"]) != entry["request_sha256"]:
            problems.append(f"{entry['slug']}: the request record does not match its recorded digest")
        if canonical_sha256(entry["response"]) != entry["response_sha256"]:
            problems.append(f"{entry['slug']}: the response record does not match its recorded digest")
        if entry["http_status"] != 200:
            problems.append(f"{entry['slug']}: recorded HTTP {entry['http_status']}")
        if entry["case"] not in registered:
            problems.append(f"{entry['slug']}: recorded, but no longer registered in inputs.json")
        # The clip this call was sent, checked against the descriptor the
        # request carries and the digest the manifest pins.
        sent = entry["request"]["body"]["items"][0]["audio"]["data"]
        path = EVIDENCE / "inputs" / "audio" / f"{entry['case']}.mp3"
        if not path.is_file():
            problems.append(f"{entry['slug']}: inputs/audio/{entry['case']}.mp3 was not downloaded")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != sent["sha256"] or len(data) != sent["bytes"]:
            problems.append(f"{entry['slug']}: inputs/audio/{entry['case']}.mp3 is not the clip this call was sent")
        if sha256_bytes(data) != entry["audio"]["sha256"]:
            problems.append(f"{entry['slug']}: the stored clip and the entry's own digest disagree")

    for case in inputs["cases"]:
        for call in CALLS:
            if (case["id"], call) not in seen:
                problems.append(f"{case['id']}__{call}: registered in inputs.json but not recorded")
    return problems


def main() -> int:
    inputs = load("inputs/inputs.json")
    manifest = load("manifest.json")
    calls = load("calls.json")
    amendment = load("inputs/scoring_amendment.json")

    problems = verify_evidence(manifest, calls, inputs)
    if problems:
        print("The evidence did not verify, so nothing was scored:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1

    normalize = load_english_normalizer(EVIDENCE / "inputs" / "normalizer.json")
    money_terms = {
        (row["case"], row["term"]): Decimal(row["amount"]) for row in amendment["money_terms"]
    }
    by_slug = {entry["slug"]: entry for entry in calls["calls"]}

    totals = {call: {"edits": 0, "words": 0, "terms": 0, "found": 0} for call in CALLS}
    per_clip: dict[str, dict[str, Any]] = {}
    misses: list[str] = []
    failures: list[str] = []

    for case in inputs["cases"]:
        reference = normalize(case["reference"]).split()
        if not reference:
            failures.append(f"{case['id']}: the human transcript normalizes to nothing")
            continue
        per_clip[case["id"]] = {}
        for call in CALLS:
            entry = by_slug[f"{case['id']}__{call}"]
            hypothesis = normalize(transcript_text(entry)).split()
            ops = edit_ops(reference, hypothesis)
            edits = ops["substitutions"] + ops["deletions"] + ops["insertions"]

            found = 0
            for term in case["key_terms"]:
                term_tokens = normalize(term).split()
                # A term that is not in its own human transcript is a broken
                # registration, not a model miss.
                if find_term(term_tokens, reference) is None:
                    failures.append(f"{case['id']}: key term {term!r} is not in its own reference")
                    continue
                span = find_term(term_tokens, hypothesis)
                hit = span is not None
                # The amendment can only remove a hit: a term must satisfy the
                # pre-registered rule first, then write the spoken amount.
                amount = money_terms.get((case["id"], term))
                if hit and amount is not None and span is not None:
                    written = written_amount("".join(hypothesis[span[0] : span[1]]))
                    if written is None or written != amount:
                        hit = False
                if hit:
                    found += 1
                elif call == "snippet":
                    misses.append(f"{case['id']}: {term}")

            totals[call]["edits"] += edits
            totals[call]["words"] += len(reference)
            totals[call]["terms"] += len(case["key_terms"])
            totals[call]["found"] += found
            per_clip[case["id"]][call] = {
                "wer": f"{edits / len(reference) * 100:.1f}%",
                "words": len(reference),
                "terms": (found, len(case["key_terms"])),
            }

        line = per_clip[case["id"]]["snippet"]
        print(f"  {case['id']}: {line['wer']} over {line['words']} words, key terms {line['terms'][0]} of {line['terms'][1]}")

    snippet = totals["snippet"]
    control = totals["control"]
    pooled = f"{snippet['edits'] / snippet['words'] * 100:.1f}%"
    clips = len(per_clip)
    print(
        f"\nAcross all {clips} recorded clips a search finds {snippet['found']} of {snippet['terms']} key terms, "
        f"and {snippet['terms'] - snippet['found']} came back wrong"
    )
    print(f"Pooled word error rate {pooled} over {snippet['words']} human-transcribed words")
    print(
        f"The control call, sent with no instruction, finds {control['found']} of {control['terms']} "
        f"at a pooled {control['edits'] / control['words'] * 100:.2f}%"
    )
    for line in misses:
        print(f"  missed: {line}")
    if clips != PAGE_CLIPS:
        failures.append(f"clips: got {clips}, page publishes {PAGE_CLIPS}")
    if (snippet["found"], snippet["terms"]) != PAGE_KEY_TERMS:
        failures.append(
            f"key terms: got {snippet['found']} of {snippet['terms']}, page publishes "
            f"{PAGE_KEY_TERMS[0]} of {PAGE_KEY_TERMS[1]}"
        )
    if pooled != PAGE_POOLED_WER:
        failures.append(f"pooled WER: got {pooled}, page publishes {PAGE_POOLED_WER}")
    if snippet["words"] != PAGE_REFERENCE_WORDS:
        failures.append(f"reference words: got {snippet['words']}, page publishes {PAGE_REFERENCE_WORDS}")
    if len(misses) != PAGE_TERMS_WRONG:
        failures.append(f"terms wrong: got {len(misses)}, page publishes {PAGE_TERMS_WRONG}")

    for case_id, expected in PUBLISHED_PER_CLIP.items():
        got = per_clip.get(case_id, {}).get("snippet")
        # A clip named here and absent from the scored set is a failure, never a
        # skip: a missing input must not let its checks quietly not run.
        if got is None:
            failures.append(f"{case_id}: has published figures but is not in the scored set")
            continue
        for field, want in expected.items():
            if got[field] != want:
                failures.append(f"{case_id} {field}: got {got[field]}, page publishes {want}")

    if failures:
        print(
            f"\nThis does NOT reproduce what {manifest['page']} publishes. "
            "Report it rather than adjusting either number.",
            file=sys.stderr,
        )
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(
        f"\nMatches the {PAGE_KEY_TERMS[0]} of {PAGE_KEY_TERMS[1]}, the pooled {PAGE_POOLED_WER} over "
        f"{PAGE_REFERENCE_WORDS} words, and the per-clip figures for {len(PUBLISHED_PER_CLIP)} of the "
        f"{clips} clips, published on {manifest['page']}. All {clips} are scored either way.\n"
        "This checks figures, not composition: which clips the page plays, and how many, "
        "is decided in sie-web and nothing here can read it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
