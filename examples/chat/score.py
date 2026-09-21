#!/usr/bin/env python3
"""Score the recorded answers offline. No network, no API key, no install.

    python3 score.py

Reads the pinned passages and questions from evidence/inputs/cases.json, the
recorded calls from evidence/calls.json, and applies the four acceptance
checks to each recorded answer. The checks were written before the run that
produced calls.json.

Every check is fail-closed. A missing input, a missing recorded call, a
recorded request the pinned input does not rebuild, or a response digest that
does not match is a failure, never a skipped case.
"""

from __future__ import annotations

import string
import sys
import unicodedata
from typing import Any

import prompt
from prompt import InputError

# The four acceptance checks, in the order the /chat page lists them.
CHECK_NAMES = ("format", "quote", "reference", "length")

# SQuAD's official normalizer lowercases, drops articles and punctuation and
# collapses whitespace. Two documented additions: Unicode dashes fold to "-"
# and Unicode quotes to "'" before punctuation is dropped, so a reference
# answer written "100-150" and one written "100<en dash>150" compare equal.
# Without that fold the two differ only in a code point no reader can see.
_DASHES = "‐‑‒–—―−"
_QUOTES = "‘’“”"
_PUNCTUATION = set(string.punctuation) | set(_DASHES) | set(_QUOTES)
_ARTICLES = {"a", "an", "the"}


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization, used on both sides of the reference check."""
    folded = unicodedata.normalize("NFC", text).lower()
    folded = "".join("" if character in _PUNCTUATION else character for character in folded)
    return " ".join(word for word in folded.split() if word not in _ARTICLES)


def answer_text(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def parse_reply(text: str) -> dict[str, Any]:
    # No filtering and no stripping. Dropping blank lines let a reply of
    # "Answer: ...", a blank line and "Quote: ..." satisfy the documented
    # "exactly two lines", and `text.strip()` did the same for a leading or
    # trailing blank line. `splitlines()` still reads "a\nb\n" as two lines,
    # so a single trailing newline is not counted as a third.
    lines = text.splitlines()
    answer = ""
    quote = ""
    for line in lines:
        if not answer and line.startswith("Answer: "):
            answer = line[len("Answer: ") :].strip()
        elif not quote and line.startswith("Quote: "):
            quote = line[len("Quote: ") :].strip()
    return {"lines": lines, "answer": answer, "quote": quote}


def evaluate(cases_doc: dict[str, Any], case: dict[str, Any], question: dict[str, Any], text: str) -> dict[str, Any]:
    """Apply the four acceptance checks to one recorded answer."""
    abstention = cases_doc["abstention"]
    max_words = cases_doc["max_words"]
    parsed = parse_reply(text)
    lines = parsed["lines"]
    answer = parsed["answer"]
    quote = parsed["quote"]

    format_ok = len(lines) == 2 and lines[0].startswith("Answer: ") and lines[1].startswith("Quote: ")
    abstained = answer == abstention
    # Exact containment. Collapsing whitespace on both sides would let a quote
    # that differs in spaces, tabs or line breaks pass a check the page
    # describes as "character for character".
    quote_ok = quote == "none" if abstained else bool(quote) and quote in case["context"]

    if question["kind"] == "unanswerable":
        reference_ok = abstained
    else:
        normalized = normalize_answer(answer)
        reference_ok = bool(normalized) and any(normalize_answer(gold) in normalized for gold in question["gold"])

    words = len(answer.split())
    length_ok = 0 < words <= max_words

    return {
        "checks": {
            "format": format_ok,
            "quote": quote_ok,
            "reference": reference_ok,
            "length": length_ok,
        },
        "details": {
            "line_count": len(lines),
            "answer": answer,
            "quote": quote,
            "abstained": abstained,
            "answer_words": words,
            "gold": question["gold"],
        },
    }


def expected_url(manifest: dict[str, Any], cases_doc: dict[str, Any]) -> str:
    """The URL every recorded call must carry.

    The path comes from this file, not from the manifest, so a recording cannot
    tell the scorer which endpoint it is allowed to have used. The host does
    come from the manifest, because a run against a regional endpoint or a
    self-hosted cluster is legitimate.
    """
    recorded_path = manifest.get("path")
    if recorded_path != prompt.CHAT_COMPLETIONS_PATH:
        raise InputError(
            f"manifest records path {recorded_path!r}; this scorer only scores {prompt.CHAT_COMPLETIONS_PATH!r}"
        )
    if manifest.get("model") != cases_doc["model"]:
        raise InputError("manifest model and the pinned cases disagree")
    return manifest["endpoint"].rstrip("/") + prompt.CHAT_COMPLETIONS_PATH


def score() -> dict[str, Any]:
    """Score every pinned question against its recorded call."""
    cases_doc = prompt.load_cases()
    manifest = prompt.read_json(prompt.MANIFEST_PATH)
    calls_doc = prompt.read_json(prompt.CALLS_PATH)
    recorded: dict[str, Any] = {}
    for entry in calls_doc["calls"]:
        if entry["slug"] in recorded:
            raise InputError(f"Duplicate recorded call {entry['slug']}")
        recorded[entry["slug"]] = entry

    url = expected_url(manifest, cases_doc)
    allowed = prompt.allowed_revisions(manifest)
    prompt.check_call_set(set(recorded), prompt.expected_slugs(cases_doc), manifest, "call")
    observed_revisions: set[str] = set()
    scored_slugs = {case["slug"] for case in prompt.scored_cases(cases_doc)}
    results = []
    for case in cases_doc["cases"]:
        for question in case["questions"]:
            slug = prompt.question_slug(case, question)
            entry = recorded[slug]
            if entry["status"] != 200:
                raise InputError(f"{slug}: recorded HTTP {entry['status']}")
            if prompt.sha256_bytes(prompt.compact_json(entry["response"])) != entry["response_sha256"]:
                raise InputError(f"{slug}: recorded response does not match its response_sha256")
            if entry["request"]["url"] != url:
                raise InputError(f"{slug}: recorded URL is {entry['request']['url']}, not {url}")
            observed_revisions.add(prompt.check_revision(slug, entry, allowed))

            # The pinned passage and question must rebuild the request that was
            # sent. One side is inputs/cases.json, the other is calls.json, and
            # neither is derived from the other.
            if prompt.request_body(cases_doc, case, question) != entry["request"]["body"]:
                raise InputError(f"{slug}: pinned input does not rebuild the recorded request body")

            text = answer_text(entry["response"])
            if not text.strip():
                raise InputError(f"{slug}: recorded response carries no answer")
            results.append(
                {
                    "slug": slug,
                    "scored": case["slug"] in scored_slugs,
                    **evaluate(cases_doc, case, question, text),
                }
            )

    if observed_revisions != allowed:
        unused = sorted(allowed - observed_revisions)
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")

    counted = [result for result in results if result["scored"]]
    # `quote` passes on an abstention, where the recorded quote is `none` and
    # nothing was cited. Reporting that as "24 of 24 quoted the passage" would
    # count nine answers that quoted nothing, so the two are separated here and
    # in the printed summary.
    cited = [result for result in counted if result["details"]["quote"] != "none"]
    return {
        "questions_scored": len(counted),
        "cited": len(cited),
        "declined": len(counted) - len(cited),
        "passing_all_checks": sum(all(result["checks"].values()) for result in counted),
        "check_passes": {name: sum(result["checks"][name] for result in counted) for name in CHECK_NAMES},
        "results": results,
    }


def main() -> int:
    try:
        summary = score()
    except InputError as error:
        print(f"FAILED: {error}")
        return 1

    for result in summary["results"]:
        passed = sum(result["checks"].values())
        misses = [name for name, ok in result["checks"].items() if not ok]
        suffix = f"  misses: {', '.join(misses)}" if misses else ""
        tail = "" if result["scored"] else "  (playground, not counted)"
        print(f"{result['slug']:<40} {passed}/{len(CHECK_NAMES)}{suffix}{tail}")

    total = summary["questions_scored"]
    passing = summary["passing_all_checks"]
    print()
    for name in CHECK_NAMES:
        print(f"{name:<24} {summary['check_passes'][name]}/{total}")
    print()
    print(f"{summary['cited']} of {total} answers cited a sentence, and every one of those is in its passage")
    print(f"{summary['declined']} of {total} declined and cited nothing")
    print(f"{passing} of {total} answers passed all four checks")

    # The published headline is that no answer quoted text absent from its
    # passage, so that is what this exits on. The reference figure is printed
    # above and is not 24 of 24; four answers miss it, and README.md says which
    # and why.
    if summary["check_passes"]["quote"] != total:
        print("FAILED: an answer quoted text that is not in its passage")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
