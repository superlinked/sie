"""Pinned inputs and the request they build. Standard library only.

Both `run.py` (which calls SIE Cloud) and `score.py` (which never does) read
the prompt from here, so the request the scorer rebuilds is the request the
runner sends.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "data" / "cases.json"
CALLS_PATH = HERE / "calls.json"

ENDPOINT = "https://api.superlinked.com"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# Scorecard keys the report keeps, and the label each becomes.
SCORECARD_LABELS = {
    "task": "Task",
    "paged-num": "People paged",
    "responders-num": "Responders",
    "coordinators": "Coordinators",
    "start": "Start",
    "end": "End",
    "metrics": "Metrics",
    "impact": "Impact",
}


class InputError(Exception):
    """A pinned input this example refuses to use."""


def read_json(path: Path) -> Any:
    if not path.exists():
        raise InputError(f"Missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_json(value: Any) -> bytes:
    """The encoding calls.json documents for response_sha256."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_cases() -> dict[str, Any]:
    """Load data/cases.json and confirm every source file is the pinned bytes."""
    doc = read_json(CASES_PATH)
    seen: set[str] = set()
    for case in doc["cases"]:
        slug = case["slug"]
        if slug in seen:
            raise InputError(f"Duplicate case {slug}")
        seen.add(slug)
        path = HERE / case["source_file"]
        if not path.exists():
            raise InputError(f"{slug}: missing {case['source_file']}")
        body = path.read_bytes()
        if len(body) != case["source_bytes"]:
            raise InputError(f"{slug}: {case['source_file']} is {len(body)} bytes, pinned at {case['source_bytes']}")
        if sha256_bytes(body) != case["source_sha256"]:
            raise InputError(f"{slug}: {case['source_file']} does not match its pinned SHA-256")
    return doc


def wikitext(case: dict[str, Any]) -> str:
    return (HERE / case["source_file"]).read_text(encoding="utf-8")


def scorecard_fields(text: str) -> dict[str, str]:
    match = re.search(r"\{\{Incident scorecard(.*?)\}\}", text, re.DOTALL)
    if not match:
        return {}
    fields = {}
    for part in re.split(r"\n\s*\|", "\n" + match.group(1)):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def plain_text(source: str) -> str:
    """Wikitext to the plain report the model receives. Deterministic."""
    text = re.split(r"\n==\s*Scorecard\s*==", source)[0]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    def scorecard(match: re.Match[str]) -> str:
        fields = scorecard_fields(match.group(0))
        lines = [f"{label}: {fields[key]}" for key, label in SCORECARD_LABELS.items() if fields.get(key)]
        return "\n".join(lines) + "\n\n"

    text = re.sub(r"\{\{Incident scorecard.*?\}\}", scorecard, text, flags=re.DOTALL)
    text = re.sub(r"<nowiki>(.*?)</nowiki>", lambda m: html.escape(m.group(1)), text, flags=re.DOTALL)
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text)
    text = re.sub(r"^(?:File|Image):.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[(?:https?:)?//\S+ ([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[(?:https?:)?//[^\]\s]+\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^\*+\s*", "- ", text, flags=re.MULTILINE)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def report_text(source: str, case: dict[str, Any]) -> str:
    """The user message: the page title, then the derived report body.

    Some scorecards give times without a date, so the title is the only place
    the report states the incident date. Every user message carries it.
    """
    title = case["title"].removeprefix("Incidents/")
    return f"Title: {title}\n\n{plain_text(source)}"


def system_instruction(cases_doc: dict[str, Any]) -> str:
    return cases_doc["prompt_template"].split("\n\nIncident report:\n")[0].strip()


def request_body(cases_doc: dict[str, Any], source: str, case: dict[str, Any]) -> dict[str, Any]:
    """The request body, keys in the order the /chat task page sends them."""
    return {
        "model": cases_doc["model"],
        "messages": [
            {"role": "system", "content": system_instruction(cases_doc)},
            {"role": "user", "content": report_text(source, case).strip()},
        ],
        "max_completion_tokens": cases_doc["max_completion_tokens"],
    }
