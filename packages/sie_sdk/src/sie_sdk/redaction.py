"""Shared log-redaction helpers (#2339).

One canonical implementation for the two redaction shapes that were
previously re-implemented per surface:

- :func:`mask_token` — token masking for logs/audit records, matching the
  Rust gateway's ``middleware::auth::mask_token`` (last 4 kept, rest
  starred; 4 chars or fewer fully starred).
- :func:`endpoint_origin_for_log` — credential- and query-free endpoint
  origin for diagnostics, consumed by the managed worker runtime and the
  dispatcher telemetry setup.
- :func:`redact_secrets` — scrubs credential- and PII-shaped substrings out
  of free text (a remote error body, a captured stderr) whose contents the
  caller does not control, before it reaches a terminal or a log.

This lives in ``sie_sdk`` (not ``sie_server``) because every Python
surface that logs — server, managed worker runtime, dispatcher — has the
SDK in its dependency closure, while the dispatcher deliberately does not
depend on ``sie_server``.

The benchmark campaign serializer keeps its own equivalent pattern tuple on
purpose: those patterns feed content-addressed campaign artifact digests, so
re-pointing them at this module would let an unrelated pattern change rewrite
committed artifact hashes.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

REDACTED = "[REDACTED]"

# Credential and PII shapes. Over-redaction is the safe failure mode here: the
# consumer is a diagnostic, and a scrubbed word costs less than a leaked token.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    # AKIA is a long-lived access key, ASIA an STS temporary one. Temporary
    # does not mean harmless: an ASIA id still pairs with a live secret and
    # session token for the life of the session.
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # A JWT, i.e. base64url header then payload then signature, dot separated.
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
    # key=value / "key": "value" for any key that names a credential.
    re.compile(
        r"(?i)\b([\w.-]*(?:api[_-]?key|token|password|passwd|secret|credential)[\w.-]*)\"?\s*[:=]\s*\"?[^\s,;\"}]+"
    ),
    # Email addresses are the PII shape that actually shows up in error bodies.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]*\w\b"),
)


def _redaction_replacement(match: re.Match[str]) -> str:
    """Keep a matched key name, drop everything else."""
    if match.lastindex:
        return f"{match.group(1)}={REDACTED}"
    return REDACTED


def redact_secrets(text: str) -> str:
    """Scrub credential- and PII-shaped substrings out of untrusted free text.

    Intended for text the caller did not author — a remote HTTP error body,
    captured subprocess output — that is about to be printed or logged. This
    is a best-effort scrubber, not a certified one: it is a last line of
    defence, never a licence to route secrets through a log in the first
    place.
    """
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_redaction_replacement, text)
    return text


def mask_token(token: str) -> str:
    """Mask a token for logs: keep the last 4 characters, star the rest."""
    if len(token) <= 4:
        return "****"
    return "*" * (len(token) - 4) + token[-4:]


def endpoint_origin_for_log(endpoint: str) -> str:
    """Return a credential- and query-free endpoint origin for diagnostics."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return "<redacted>"
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return "<redacted>"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}{f':{port}' if port is not None else ''}"
