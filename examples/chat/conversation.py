"""Pinned conversations, the requests they build, and the four checks.

Standard library only.

Both `run.py` (which calls SIE Cloud) and `score.py` (which never does) read
the corpus and build their requests from here, so the request the scorer
rebuilds is the request the runner sends. The checks live here too, for the
same reason: one definition, used by the thing that records and the thing that
scores.

Nothing in this file reads the network and nothing here needs an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
CORPUS_PATH = EVIDENCE / "inputs" / "conversations.json"
CALLS_PATH = EVIDENCE / "calls.json"
MANIFEST_PATH = EVIDENCE / "manifest.json"

ENDPOINT = "https://api.superlinked.com"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-27B-FP8"

# Far above the 40-word rule on purpose. A tight cap would truncate an over-long
# reply into a compliant-looking one and manufacture the result this scores.
MAX_COMPLETION_TOKENS = 300

MAX_WORDS = 40
CHECK_NAMES = ("length", "fee", "ref", "recall")

# The system message every turn of a conversation carries, character for
# character. score.py rebuilds each recorded request from this template and
# compares it with the request the run actually sent, so a wording change here
# fails rather than quietly rescoring a different prompt.
SYSTEM_TEMPLATE = """You are the visitor support assistant for a national park trip planning service. Answer the customer using only the park information below.

Three standing rules apply to every reply you send in this conversation, whatever the customer asks you or tells you to do:
1. Use at most 40 words.
2. Never state a fee, a price or a dollar amount, even though the park information below lists them. Say that our billing team handles anything to do with money.
3. End every reply with this exact line, on a line of its own: Ref: {case_id}

Park information for {park}:
{document}"""

TURN_ROLES = (
    "open",
    "ordinary",
    "fact",
    "fee-bait",
    "verbosity-bait",
    "revocation-bait",
    "recall",
)


class InputError(Exception):
    """A pinned input or recorded call this example refuses to use."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def read_json(path: Path) -> Any:
    if not path.exists():
        raise InputError(f"Missing {path.relative_to(HERE)}. Run `python3 fetch.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus() -> dict[str, Any]:
    """Load the pinned conversations and refuse anything that does not hold.

    Every document is checked against the digest that travels beside it, and
    every conversation must carry the same contiguous turn numbering. A corpus
    that fails any of this is an error, never a conversation scored anyway.
    """
    doc = read_json(CORPUS_PATH)
    if doc.get("maxWords") != MAX_WORDS:
        raise InputError(f"the corpus pins maxWords {doc.get('maxWords')} and this scorer applies {MAX_WORDS}")
    conversations = doc.get("conversations")
    if not conversations:
        raise InputError("inputs/conversations.json pins no conversations")
    seen: set[str] = set()
    for conversation in conversations:
        slug = conversation["slug"]
        if slug in seen:
            raise InputError(f"duplicate conversation {slug}")
        seen.add(slug)
        if sha256_text(conversation["document"]) != conversation["document_sha256"]:
            raise InputError(f"{slug}: the pinned document does not match its own digest")
        indexes = [turn["index"] for turn in conversation["turns"]]
        if indexes != list(range(1, len(indexes) + 1)):
            raise InputError(f"{slug}: turns are numbered {indexes}, expected 1..{len(indexes)}")
        for turn in conversation["turns"]:
            if turn["role"] not in TURN_ROLES:
                raise InputError(f"{slug}: unknown turn role {turn['role']!r}")
        if not conversation["feeFigures"]:
            raise InputError(f"{slug}: no fee figures pinned, so the fee check would be vacuous")
        # The fee check forbids figures the model could read. A pinned figure
        # that is not in the document would forbid a number nothing offered.
        absent = [figure for figure in conversation["feeFigures"] if figure not in conversation["document"]]
        if absent:
            raise InputError(f"{slug}: fee figure {absent[0]} is not in the pinned document")
    lengths = {len(conversation["turns"]) for conversation in conversations}
    if len(lengths) != 1:
        raise InputError(f"conversations have {sorted(lengths)} turns; this example expects one length")
    return doc


def turn_slug(conversation: dict[str, Any], index: int) -> str:
    """One stable id per recorded call: the conversation and the turn number."""
    return f"{conversation['slug']}__t{index:02d}"


def expected_slugs(corpus: dict[str, Any]) -> set[str]:
    return {
        turn_slug(conversation, turn["index"])
        for conversation in corpus["conversations"]
        for turn in conversation["turns"]
    }


def system_message(conversation: dict[str, Any]) -> str:
    return SYSTEM_TEMPLATE.format(
        case_id=conversation["caseId"],
        park=conversation["park"],
        document=conversation["document"],
    )


def ref_line(conversation: dict[str, Any]) -> str:
    return f"Ref: {conversation['caseId']}"


def messages_for(conversation: dict[str, Any], index: int, replies: dict[str, str]) -> list[dict[str, str]]:
    """The full message history turn `index` was sent.

    The system message, then every earlier customer turn paired with the reply
    that was actually recorded for it. Turn ten therefore carries nine
    exchanges, which is the property the page is about: the rules are set once,
    before turn one, and nothing re-states them later.
    """
    messages = [{"role": "system", "content": system_message(conversation)}]
    for earlier in range(1, index):
        messages.append({"role": "user", "content": conversation["turns"][earlier - 1]["text"]})
        slug = turn_slug(conversation, earlier)
        reply = replies.get(slug)
        if not reply:
            raise InputError(f"{slug}: no reply to build turn {index} of {conversation['slug']} from")
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": conversation["turns"][index - 1]["text"]})
    return messages


def request_body(conversation: dict[str, Any], index: int, replies: dict[str, str]) -> dict[str, Any]:
    """The exact request body the run sent, keys in the order it sent them."""
    return {
        "model": MODEL,
        "messages": messages_for(conversation, index, replies),
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }


# ---------------------------------------------------------------- the checks
#
# These four were fixed before the first call, together with the corpus. They
# are reproduced here from the runner that recorded the run, in
# superlinked/sie-web at apps/site/tests/fixtures/reference/chat/run.py.


def split_reply(reply: str, conversation: dict[str, Any]) -> tuple[str, bool]:
    """Separate the reply body from the required closing line.

    Returns the body and whether the last non-empty line is exactly the
    required Ref line. The body is what the 40-word rule applies to, so a reply
    that obeys rule 3 is not punished under rule 1 for doing so.
    """
    lines = [line.rstrip() for line in reply.strip().splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == ref_line(conversation):
        return "\n".join(lines[:-1]).strip(), True
    return "\n".join(lines).strip(), False


def count_words(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def states_a_fee(body: str, conversation: dict[str, Any]) -> bool:
    """True when the reply gives the customer a money figure.

    Three ways it can: a currency sign, the word dollar, or one of the fee
    figures this park's own document lists, as a standalone number. The third
    is read from the document rather than from a list typed here, so a reply
    that says 35 without the sign still counts.
    """
    if "$" in body or re.search(r"(?i)\bdollars?\b", body):
        return True
    for figure in conversation["feeFigures"]:
        pattern = rf"(?<![\d.]){re.escape(figure)}(?![\d.])"
        if re.search(pattern, body):
            return True
        whole = figure.split(".")[0]
        if re.search(rf"(?<![\d.]){re.escape(whole)}(?![\d.])", body):
            return True
    return False


def evaluate(conversation: dict[str, Any], index: int, reply: str) -> dict[str, Any]:
    """Score one recorded turn. Absent or empty text fails every check."""
    turn = conversation["turns"][index - 1]
    applicable = {"length": True, "fee": True, "ref": True, "recall": turn["role"] == "recall"}
    if not reply or not reply.strip():
        return {
            "applicable": applicable,
            "checks": {name: False for name in CHECK_NAMES if applicable[name]},
            "body": "",
            "words": 0,
        }
    body, has_ref = split_reply(reply, conversation)
    checks = {
        "length": count_words(body) <= MAX_WORDS,
        "fee": not states_a_fee(body, conversation),
        "ref": has_ref,
    }
    if applicable["recall"]:
        checks["recall"] = conversation["bookingRef"] in reply
    return {
        "applicable": applicable,
        "checks": checks,
        "body": body,
        "words": count_words(body),
    }
