#!/usr/bin/env python3
"""Score the recorded status updates offline. No network, no API key, no install.

    python3 score.py

Reads the pinned incident reports from data/, the recorded calls from
calls.json, and applies the four acceptance checks to each recorded answer.
The checks were written before the run that produced calls.json.

Every check is fail-closed. A missing source file, a missing recorded call, a
recorded request the pinned input does not rebuild, or a response digest that
does not match is a failure, never a skipped case.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import prompt
from prompt import InputError

# The four acceptance checks, in the order the /chat page lists them.
CHECK_NAMES = ("format", "window", "length", "no_internal_identifiers")
LINE_LABELS = ("Status", "Impact", "Window", "Cause")
WINDOW_LINE = re.compile(r"^Window: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC$")
MAX_WORDS = 25
# Case-insensitive, like the responder-name check beside them. Written
# lowercase-only, these let `Gerrit1003` and `t393034` through a check whose
# whole job is that no such identifier reaches a status page.
LEAK_PATTERNS = {
    "hostname": re.compile(r"\b[a-z][a-z-]*\d{3,4}(?:\.[a-z0-9-]+)*\b", re.IGNORECASE),
    "numbered instance": re.compile(r"\b[a-z]+(?:-[a-z]+)*-\d{1,2}\b", re.IGNORECASE),
    "internal domain": re.compile(r"\.(?:wmnet|wikimedia\.cloud)\b", re.IGNORECASE),
    "Phabricator task": re.compile(r"\bT\d{5,6}\b", re.IGNORECASE),
}
NAME_STOPWORDS = {"round", "oncallers", "wmf", "wmde", "user", "n/a", "unknown"}


def people_to_check(case: dict[str, Any], source: str) -> list[str]:
    """Staff names the update must not contain, read from the report itself."""
    fields = prompt.scorecard_fields(source)
    raw = " ".join(fields.get(key, "") for key in ("coordinators", "responders-num"))
    raw = re.sub(r"User:", " ", raw)
    names = set(case.get("people", []))
    for match in re.finditer(r"Incident opened\.\s+(\S+(?: [A-Z]\S+)?) becomes IC", source):
        raw += " " + match.group(1)
    for token in re.split(r"[\s,:;()\[\]]+", raw):
        token = token.strip().strip("'")
        # Check the exact handle (Amir1) and the handle without its digits
        # (Amir), so neither form can pass. A token whose digit-stripped form
        # is a stopword is a label, not a handle: "round2" names nobody.
        if re.sub(r"\d+$", "", token).lower() in NAME_STOPWORDS:
            continue
        for candidate in {token, re.sub(r"\d+$", "", token)}:
            for piece in candidate.split("-"):
                if len(piece) >= 3 and piece.lower() not in NAME_STOPWORDS and not piece.isdigit():
                    names.add(piece)
    return sorted(names, key=lambda name: (name.lower(), name))


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


def evaluate(case: dict[str, Any], source: str, text: str) -> dict[str, Any]:
    """Apply the four acceptance checks to one recorded answer."""
    lines = text.strip().split("\n")
    format_ok = (
        len(lines) == 4
        and all(line.startswith(f"{label}: ") for line, label in zip(lines, LINE_LABELS))
        and lines[0] == "Status: Resolved"
        and bool(WINDOW_LINE.match(lines[2]))
    )
    fields: dict[str, str] = {}
    for line in lines:
        label, _, value = line.partition(": ")
        if label in LINE_LABELS and label not in fields:
            fields[label] = value.strip()
    answer_window = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", fields.get("Window", ""))
    window_ok = len(answer_window) == 2 and answer_window[0] in case["start"] and answer_window[1] in case["end"]
    word_counts = {label: len(fields.get(label, "").split()) for label in ("Impact", "Cause")}
    length_ok = all(label in fields and 0 < count <= MAX_WORDS for label, count in word_counts.items())
    leaks = []
    for kind, pattern in LEAK_PATTERNS.items():
        leaks.extend({"kind": kind, "text": m.group(0)} for m in pattern.finditer(text))
    people = people_to_check(case, source)
    for name in people:
        leaks.extend(
            {"kind": "person", "text": m.group(0)} for m in re.finditer(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
        )
    return {
        "checks": {
            "format": format_ok,
            "window": window_ok,
            "length": length_ok,
            "no_internal_identifiers": not leaks,
        },
        "details": {
            "line_count": len(lines),
            "answer_window": answer_window,
            "word_counts": word_counts,
            "leaks": leaks,
            "people_checked": people,
        },
    }


def expected_url(manifest: dict[str, Any]) -> str:
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
    if manifest.get("model") != prompt.read_json(prompt.CASES_PATH)["model"]:
        raise InputError("manifest model and the pinned cases disagree")
    return manifest["endpoint"].rstrip("/") + prompt.CHAT_COMPLETIONS_PATH


def score() -> dict[str, Any]:
    """Score every pinned case against its recorded call."""
    cases_doc = prompt.load_cases()
    manifest = prompt.read_json(prompt.MANIFEST_PATH)
    calls_doc = prompt.read_json(prompt.CALLS_PATH)
    recorded: dict[str, Any] = {}
    for entry in calls_doc["calls"]:
        if entry["slug"] in recorded:
            raise InputError(f"Duplicate recorded call {entry['slug']}")
        recorded[entry["slug"]] = entry

    url = expected_url(manifest)
    allowed = prompt.allowed_revisions(manifest)
    prompt.check_call_set(set(recorded), {case["slug"] for case in cases_doc["cases"]}, manifest, "call")
    observed_revisions: set[str] = set()
    results = []
    for case in cases_doc["cases"]:
        slug = case["slug"]
        entry = recorded.get(slug)
        if entry is None:
            raise InputError(f"{slug}: no recorded call in calls.json")
        if entry["status"] != 200:
            raise InputError(f"{slug}: recorded HTTP {entry['status']}")
        if prompt.sha256_bytes(prompt.compact_json(entry["response"])) != entry["response_sha256"]:
            raise InputError(f"{slug}: recorded response does not match its response_sha256")
        if entry["request"]["url"] != url:
            raise InputError(f"{slug}: recorded URL is {entry['request']['url']}, not {url}")
        observed_revisions.add(prompt.check_revision(slug, entry, allowed))

        # The pinned wikitext must rebuild the request that was sent. One side
        # is inputs/, the other is calls.json, and neither is derived from the
        # other.
        source = prompt.wikitext(case)
        if prompt.request_body(cases_doc, source, case) != entry["request"]["body"]:
            raise InputError(f"{slug}: pinned input does not rebuild the recorded request body")

        text = answer_text(entry["response"])
        if not text.strip():
            raise InputError(f"{slug}: recorded response carries no answer")
        results.append({"slug": slug, **evaluate(case, source, text)})

    if observed_revisions != allowed:
        unused = sorted(allowed - observed_revisions)
        raise InputError(f"manifest names revision {unused[0]}, which no recorded call used")

    return {
        "reports_scored": len(results),
        "passing_all_checks": sum(all(result["checks"].values()) for result in results),
        "check_passes": {name: sum(result["checks"][name] for result in results) for name in CHECK_NAMES},
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
        print(f"{result['slug']:<32} {passed}/{len(CHECK_NAMES)}{suffix}")

    total = summary["reports_scored"]
    passing = summary["passing_all_checks"]
    print()
    for name in CHECK_NAMES:
        print(f"{name:<24} {summary['check_passes'][name]}/{total}")
    print()
    print(f"{passing} of {total} incident reports passed all four checks")

    if passing != total:
        print("FAILED: not every recorded status update passes")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
