"""Record one run as a single ``calls.json`` plus digest-addressed payloads.

Layout rules
------------
One file per call is clutter, so every call in a run lands in one ``calls.json``
as an array entry carrying its request, response, status, selected response
headers, timing and model revision.

That file must still be reviewable in a diff, and a raw combine here would not
be: the four Docling responses total 1.68 MiB compact, and ``data.document`` --
the full document tree -- is between 85 and 98 percent of every one of them.
Nothing scores it. ``evaluate.py`` reads ``markdown`` and nothing else.

So the rule is not a byte threshold someone guessed, it is what the checks read:

    Inline every member the checks or the page read. Move a member out of line
    when it cannot be canonicalized at all, whatever its size; then, while an
    entry still exceeds ENTRY_INLINE_LIMIT, move out its largest remaining
    unscored member.

The first clause is not a refinement, it is load-bearing. Docling returns
``data.document.origin.binary_hash``, a 64-bit integer, and the FEMA entry that
contains one is only 76 KiB -- comfortably under any size threshold. See
``externalize`` for what that integer does to a JavaScript verifier.

An externalized member is replaced by a ``$payload`` reference and its bytes are
written to ``payloads/<sha256>.json``, so everything stays committed and
digest-addressed -- only the *placement* changes. ``CALLS_FILE_LIMIT`` is a
backstop for runs with many calls.

Digests come in two scopes, and conflating them is how a verification claim
starts outrunning what it checks. Canonical RFC 8785 digests cover the values
that are compared against reformatted copies held elsewhere -- ``manifest_sha256``,
``calls.sha256``, ``entry_sha256``, ``scored_markdown.response_markdown_sha256``
-- so a copy formatted with different indentation still matches. File-byte
digests cover the things a reader will hash as files: payloads, the scored
Markdown file, and every source PDF digest, each reproducible with
``shasum -a 256``. See ``CANONICALIZATION``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from document_to_markdown.canonical import canonical_sha256

SCHEMA_VERSION = "1.0"

#: Members the acceptance checks read, or the website displays. Never externalized.
INLINE_ALWAYS: frozenset[str] = frozenset({"data.markdown", "data.text"})

#: An entry larger than this gets its unscored members moved out of line.
ENTRY_INLINE_LIMIT = 128 * 1024

#: Backstop for runs with many calls; beyond this, split by call group.
CALLS_FILE_LIMIT = 8 * 1024 * 1024

CANONICALIZATION = (
    "Two digest scopes. RFC 8785 canonical encoding of the parsed value covers manifest_sha256, "
    "calls.sha256, entry_sha256 and scored_markdown.response_markdown_sha256, so reformatting a copy "
    "does not change them. Recorded file bytes cover $payload.sha256, scored_markdown.sha256, "
    "document_sha256, source_sha256 and the sha256 values in data/manifest.json, each reproducible "
    "with `shasum -a 256`. Content compared against a reformatted copy elsewhere is hashed "
    "canonically; content a reader will hash as a file is hashed as bytes."
)

#: ``convert.py`` writes the scored Markdown file as ``markdown.rstrip() + newline``.
#: ``verify.py`` asserts exactly this relation rather than a lenient normalized
#: comparison, so that a later change to the transform is caught instead of hidden.
MARKDOWN_FILE_TRANSFORM = "response.data.markdown.rstrip() + '\\n'  (convert.py)"


def _payload_bytes(value: Any) -> bytes:
    """Serialize a payload for storage.

    Deliberately NOT canonical. Payload subtrees are opaque blobs referenced by
    a digest over their stored bytes, because their contents need not be
    canonicalizable at all -- see ``externalize``.
    """
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _estimated_size(value: Any) -> int:
    """Cheap size estimate for choosing what to move out of line.

    Uses plain compact JSON rather than the canonical encoding: this only ranks
    candidates, and the candidates are exactly the values that may not be
    canonicalizable.
    """
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _payload_candidates(response: dict[str, Any]) -> list[tuple[str, int]]:
    """Externalizable members, largest first, as (dotted path, estimated size)."""
    sizes: list[tuple[str, int]] = []
    for key, value in response.items():
        if key == "data" and isinstance(value, dict):
            for inner_key, inner_value in value.items():
                path = f"data.{inner_key}"
                if path not in INLINE_ALWAYS:
                    sizes.append((path, _estimated_size(inner_value)))
            continue
        if key not in INLINE_ALWAYS:
            sizes.append((key, _estimated_size(value)))
    sizes.sort(key=lambda row: row[1], reverse=True)
    return sizes


def _take(response: dict[str, Any], path: str) -> Any:
    head, _, tail = path.partition(".")
    return response[head][tail] if tail else response[head]


def _put(response: dict[str, Any], path: str, value: Any) -> None:
    head, _, tail = path.partition(".")
    if tail:
        response[head][tail] = value
    else:
        response[head] = value


def externalize(response: dict[str, Any], payload_dir: Path) -> dict[str, Any]:
    """Move unscored members out of line until the response fits the limit.

    Returns the response with ``$payload`` references in place of whatever was
    moved. Writes each moved member to ``payload_dir``.

    Payload digests are taken over the STORED FILE BYTES, not over a canonical
    encoding, and that is not a shortcut.

    Docling's document tree carries ``data.document.origin.binary_hash``, a
    64-bit hash of the source PDF -- 14923062233064822549 for the FEMA form.
    RFC 8785 defines numbers as IEEE 754 doubles and that value is not one, so
    the subtree cannot be canonicalized at all.

    The danger is not that a JavaScript verifier would fail on it. It is that
    it would NOT fail. ``JSON.parse("17271392962501450983")`` returns
    17271392962501452000 -- a different number, no error, no warning. A verifier
    built on that parse would compute a digest over corrupted input and report
    agreement. The guard here is against silent agreement, not against a crash,
    which is why size is irrelevant and why these bytes are never parsed and
    re-serialized on the way to a digest. ``shasum -a 256 payloads/<name>``
    reproduces the value exactly.

    Canonical digests stay where they belong -- on the content both repositories
    compare, which is everything that remains inline.
    """
    response = json.loads(json.dumps(response))  # do not mutate the caller's value
    candidates = _payload_candidates(response)

    # Two reasons to move a member out of line, checked in this order.
    #
    # 1. It cannot be canonicalized. Size is irrelevant here: the FEMA entry is
    #    only 76 KiB, well under the limit, and still carries binary_hash. A
    #    value no cross-language digest can represent must never sit inline.
    # 2. The entry is over ENTRY_INLINE_LIMIT, largest unscored member first.
    forced = [path for path, _ in candidates if _first_uncanonicalizable(_take(response, path), path)]
    by_size = [path for path, _ in candidates if path not in forced]

    for path in forced + by_size:
        if path not in forced and _estimated_size(response) <= ENTRY_INLINE_LIMIT:
            break
        value = _take(response, path)
        if isinstance(value, dict) and "$payload" in value:
            continue
        body = _payload_bytes(value)
        digest = hashlib.sha256(body).hexdigest()
        payload_dir.mkdir(parents=True, exist_ok=True)
        (payload_dir / f"{digest}.json").write_bytes(body)
        _put(
            response,
            path,
            {
                "$payload": {
                    "sha256": digest,
                    "sha256_over": "stored file bytes",
                    "bytes": len(body),
                    "media_type": "application/json",
                    "path": f"payloads/{digest}.json",
                }
            },
        )
    unrepresentable = _first_uncanonicalizable(response)
    if unrepresentable is not None:
        raise ValueError(
            f"{unrepresentable} cannot be canonicalized under RFC 8785 and is not externalizable; "
            "add it to the payload candidates or to INLINE_ALWAYS deliberately"
        )
    return response


def _first_uncanonicalizable(value: Any, path: str = "response") -> str | None:
    """Name the first value that RFC 8785 cannot represent, for a clear error."""
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_uncanonicalizable(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _first_uncanonicalizable(item, f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 2**53:
        return f"{path} ({value})"
    return None


def build_entry(
    *,
    slug: str,
    request: dict[str, Any],
    response: dict[str, Any],
    duration_ms: float,
    model_revision: str | None,
    retry_count: int,
    markdown_sha256: str,
    payload_dir: Path,
) -> dict[str, Any]:
    """Build one call entry, with its digest taken over the entry itself."""
    entry: dict[str, Any] = {
        "slug": slug,
        "status": "ok",
        "request": request,
        "response": externalize(response, payload_dir),
        "headers": {"x-sie-model-revision": model_revision},
        "model_revision": model_revision,
        "retry_count": retry_count,
        "timing": {"duration_ms": duration_ms},
        "scored_markdown": {
            "file": f"markdown/{slug}.md",
            "sha256": markdown_sha256,
            "derived_from": "response.data.markdown",
            "transform": MARKDOWN_FILE_TRANSFORM,
            "response_markdown_sha256": canonical_sha256(response["data"]["markdown"]),
        },
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def write_calls(run_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Write ``calls.json`` and return the summary the run manifest records."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization": CANONICALIZATION,
        "inline_policy": {
            "rule": "Members the checks read stay inline; unscored members move to payloads/ by digest.",
            "inline_always": sorted(INLINE_ALWAYS),
            "entry_inline_limit_bytes": ENTRY_INLINE_LIMIT,
            "calls_file_limit_bytes": CALLS_FILE_LIMIT,
        },
        "calls": entries,
    }
    body = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if len(body.encode("utf-8")) > CALLS_FILE_LIMIT:
        raise ValueError(
            f"calls.json is {len(body.encode('utf-8'))} bytes, over the {CALLS_FILE_LIMIT} byte cap; "
            "split the run by call group"
        )
    (run_dir / "calls.json").write_text(body, encoding="utf-8")
    return {
        "file": "calls.json",
        "sha256": canonical_sha256(document),
        "call_count": len(entries),
        "entries": {entry["slug"]: entry["entry_sha256"] for entry in entries},
    }
