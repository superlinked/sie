"""Shared log-redaction helpers (#2339).

One canonical implementation for the two redaction shapes that were
previously re-implemented per surface:

- :func:`mask_token` — token masking for logs/audit records, matching the
  Rust gateway's ``middleware::auth::mask_token`` (last 4 kept, rest
  starred; 4 chars or fewer fully starred).
- :func:`endpoint_origin_for_log` — credential- and query-free endpoint
  origin for diagnostics, consumed by the managed worker runtime and the
  dispatcher telemetry setup.

This lives in ``sie_sdk`` (not ``sie_server``) because every Python
surface that logs — server, managed worker runtime, dispatcher — has the
SDK in its dependency closure, while the dispatcher deliberately does not
depend on ``sie_server``.
"""

from __future__ import annotations

from urllib.parse import urlsplit


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
