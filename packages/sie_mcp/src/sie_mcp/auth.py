"""Connector-secret authentication at the MCP edge.

A per-user connector secret (Bearer) is validated here and mapped to a stable
user identity, so per-user metering can attach later. The cluster
credential the service uses downstream is held server-side and never travels
through this edge.

Implemented as pure ASGI (not ``BaseHTTPMiddleware``) so it does not buffer the
streaming MCP responses.
"""

import re
from ipaddress import IPv6Address
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from sie_mcp.config import MCPConfig

# The OAuth bridge endpoints bootstrap auth for claude.ai connectors, so they sit
# in front of the connector-secret gate. Metadata + DCR + authorize + token
# must all be reachable unauthenticated.
_EXEMPT_PATHS = frozenset(
    {
        "/healthz",
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-protected-resource/mcp",
        "/.well-known/oauth-authorization-server",
        "/register",
        "/authorize",
        "/token",
    }
)


_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "[::1]"})

_MAX_PORT = 65535

# RFC 9110 s7.2: a Host header is "uri-host [ ':' port ]" and carries no userinfo,
# path, query, or fragment.
_HOST_RE = re.compile(
    r"""
    \A
    (?:
        \[(?P<ipv6>[0-9a-f:.]+)\]
      | (?P<regname>[0-9a-z._~!$&'()*+,;=%-]+)
    )
    (?::(?P<port>[0-9]{1,5}))?
    \Z
    """,
    re.VERBOSE,
)


def _split_host_port(host: str) -> tuple[str, str | None] | None:
    """Split a lowercased ``Host`` into ``(hostname, port)``, or ``None`` if malformed."""
    match = _HOST_RE.fullmatch(host)
    if match is None:
        return None
    port = match["port"]
    if port is not None and not 1 <= int(port) <= _MAX_PORT:
        return None
    ipv6 = match["ipv6"]
    if ipv6 is not None:
        try:
            IPv6Address(ipv6)
        except ValueError:
            return None
    return (f"[{ipv6}]" if ipv6 is not None else match["regname"]), port


def _host_trusted(config: MCPConfig, host: str) -> bool:
    host = host.lower()
    parsed = _split_host_port(host)
    if parsed is None:
        return False
    hostname, port = parsed
    if hostname in _LOOPBACK_HOSTNAMES:
        return True
    for allowed in (entry.lower() for entry in config.allowed_hosts):
        if host == allowed:
            return True
        if allowed.endswith(":*") and port is not None and hostname == allowed[:-2]:
            return True
    return False


def base_url(config: MCPConfig, *, scheme: str, headers: Headers) -> str | None:
    """Resolve the externally reachable origin for OAuth metadata URLs.

    Prefers the pinned ``SIE_MCP_PUBLIC_URL``. Unpinned, the request's own scheme and
    ``Host`` are used only when the host is loopback or listed in
    ``SIE_MCP_ALLOWED_HOSTS``; otherwise ``None``. This origin names the authorization
    server clients trust, so caller-controlled ``Host`` and ``X-Forwarded-*`` values are
    never advertised. A proxy-set scheme is honoured only through uvicorn's
    ``FORWARDED_ALLOW_IPS`` trust list, which rewrites ``scheme`` upstream.
    """
    if config.public_base_url:
        return config.public_base_url
    host = headers.get("host") or ""
    if not host or not _host_trusted(config, host):
        return None
    return f"{scheme}://{host}"


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    value = authorization.strip()
    if not value:
        return None
    if value[:7].lower() == "bearer ":
        return value[7:].strip() or None
    # A bare "Bearer" scheme with no token is not a credential.
    if value.lower() == "bearer":
        return None
    # Otherwise accept a raw token sent without the scheme prefix.
    return value


def authenticate(config: MCPConfig, token: str | None) -> str | None:
    """Return a user identity for the token, or ``None`` to reject the request."""
    if config.connector_secrets:
        if token and token in config.connector_secrets:
            return config.connector_secrets[token]
        if not token and config.allow_anonymous:
            return "anonymous"
        return None
    # No secrets configured: open only when anonymous access is explicitly allowed.
    return "anonymous" if config.allow_anonymous else None


class ConnectorSecretAuthMiddleware:
    """Pure-ASGI auth gate that maps a connector secret to a user identity."""

    def __init__(self, app: ASGIApp, config: MCPConfig) -> None:
        self._app = app
        self._config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in _EXEMPT_PATHS:
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        token = bearer_token(headers.get("authorization"))
        identity = authenticate(self._config, token)
        if identity is None:
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers=self._challenge_headers(scope, headers),
            )
            await response(scope, receive, send)
            return

        state: dict[str, Any] = dict(scope.get("state") or {})
        state["user_id"] = identity
        scope["state"] = state
        await self._app(scope, receive, send)

    def _challenge_headers(self, scope: Scope, headers: Headers) -> dict[str, str]:
        """Point unauthenticated clients at the OAuth bridge (RFC 9728)."""
        if not self._config.oauth_enabled:
            return {}
        origin = base_url(self._config, scheme=scope.get("scheme", "http"), headers=headers)
        if origin is None:
            return {}
        metadata = f"{origin}/.well-known/oauth-protected-resource"
        return {"WWW-Authenticate": f'Bearer resource_metadata="{metadata}"'}
