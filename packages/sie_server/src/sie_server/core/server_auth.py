"""Token authentication for ``sie-server``.

One vocabulary with the gateway: ``SIE_AUTH_MODE`` selects the mode,
``SIE_AUTH_TOKENS`` (or ``SIE_AUTH_TOKEN``) holds comma-separated tokens, and a
request presents one in ``Authorization: Bearer`` or ``x-api-key``.
"""

from __future__ import annotations

import errno
import hmac
import ipaddress
import os
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson
from starlette.types import ASGIApp, Receive, Scope, Send

from sie_server.core.data_dir import ensure_data_dir, resolve_data_dir

AUTH_MODE_ENV = "SIE_AUTH_MODE"
AUTH_TOKENS_ENV = "SIE_AUTH_TOKENS"
AUTH_TOKEN_ENV = "SIE_AUTH_TOKEN"  # noqa: S105 - an environment variable name
TOKEN_FILE_NAME = "auth-token"  # noqa: S105 - a file name

EXEMPT_PATHS = frozenset({"/healthz", "/livez", "/readyz"})

AuthMode = Literal["none", "token"]

_TOKEN_MODES = frozenset({"token", "static"})
_KNOWN_MODES = frozenset({"none", "auto"} | _TOKEN_MODES)
_MAX_TOKEN_FILE_BYTES = 4096
_GENERATED_TOKEN_BYTES = 32
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


class AuthConfigError(ValueError):
    """The auth environment has no valid interpretation."""


class TokenFileError(RuntimeError):
    """The persisted token cannot be created or trusted."""


@dataclass(frozen=True)
class StoredToken:
    token: str
    path: Path
    created: bool


def configured_tokens(env: Mapping[str, str]) -> tuple[str, ...]:
    """Tokens from ``SIE_AUTH_TOKENS``, else ``SIE_AUTH_TOKEN``, as the gateway reads them."""
    for name in (AUTH_TOKENS_ENV, AUTH_TOKEN_ENV):
        tokens = tuple(part.strip() for part in env.get(name, "").split(",") if part.strip())
        if tokens:
            return tokens
    return ()


def requested_mode(env: Mapping[str, str]) -> str | None:
    """The normalised ``SIE_AUTH_MODE``, or ``None`` when it is unset or empty."""
    raw = env.get(AUTH_MODE_ENV, "").strip().lower()
    return raw or None


def resolve_auth_mode(requested: str | None, *, has_tokens: bool, exposed: bool) -> AuthMode:
    """Decide whether requests need a token.

    A configured token is never dead configuration: it turns auth on unless
    ``SIE_AUTH_MODE=none`` says otherwise, and that combination is refused.
    ``auto`` also turns auth on for a bind that other machines can reach.
    """
    if requested is not None and requested not in _KNOWN_MODES:
        raise AuthConfigError(f"{AUTH_MODE_ENV} must be one of none, token, static or auto.")
    if requested in _TOKEN_MODES:
        return "token"
    if requested == "none":
        if has_tokens:
            raise AuthConfigError(
                f"{AUTH_TOKENS_ENV} or {AUTH_TOKEN_ENV} is set but {AUTH_MODE_ENV}=none turns authentication "
                f"off. Unset {AUTH_MODE_ENV} or set it to token to enforce the token, or remove the token."
            )
        return "none"
    if has_tokens:
        return "token"
    if requested == "auto" and exposed:
        return "token"
    return "none"


def is_loopback_host(host: str) -> bool:
    """Whether a bind address is reachable only from this machine."""
    candidate = host.strip().strip("[]")
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def token_path(directory: Path) -> Path:
    return directory / TOKEN_FILE_NAME


def load_or_create_token(directory: Path) -> StoredToken:
    """Read the persisted token, generating it on first use."""
    path = token_path(directory)
    try:
        ensure_data_dir(directory)
    except OSError as exc:
        raise TokenFileError(f"Cannot create {directory}: {exc.strerror or exc}") from exc
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    except FileExistsError:
        return StoredToken(token=read_stored_token(directory), path=path, created=False)
    except OSError as exc:
        raise TokenFileError(f"Cannot create {path}: {exc.strerror or exc}") from exc
    token = secrets.token_urlsafe(_GENERATED_TOKEN_BYTES)
    try:
        try:
            os.write(fd, f"{token}\n".encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise TokenFileError(f"Cannot write {path}: {exc.strerror or exc}") from exc
    return StoredToken(token=token, path=path, created=True)


def read_stored_token(directory: Path) -> str:
    """Read the persisted token, refusing a file that others could read or replace."""
    path = token_path(directory)
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TokenFileError(f"{path} is a symbolic link; it must be a regular file.") from exc
        raise TokenFileError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise TokenFileError(f"{path} must be a regular file.")
        if info.st_uid != os.getuid():
            raise TokenFileError(f"{path} must be owned by the user that runs sie-server.")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise TokenFileError(f"{path} must not be readable or writable by others (chmod 600 {path}).")
        raw = os.read(fd, _MAX_TOKEN_FILE_BYTES + 1)
    finally:
        os.close(fd)
    if len(raw) > _MAX_TOKEN_FILE_BYTES:
        raise TokenFileError(f"{path} is too large to be a token.")
    token = raw.decode("ascii", errors="replace").strip()
    if not token or any(not ("!" <= ch <= "~") or ch == "," for ch in token):
        raise TokenFileError(f"{path} must hold one token: printable ASCII without spaces or commas.")
    return token


def server_tokens(env: Mapping[str, str]) -> tuple[str, ...]:
    """The tokens a server in ``token`` mode accepts: the configured ones, else the persisted one."""
    tokens = configured_tokens(env)
    if tokens:
        return tokens
    directory = resolve_data_dir()
    if directory is None:
        raise TokenFileError(f"No token is configured and there is no data directory; set {AUTH_TOKEN_ENV}.")
    return (read_stored_token(directory),)


def request_credentials(headers: Iterable[tuple[bytes, bytes]]) -> set[bytes]:
    """Every distinct token a request presents in ``Authorization`` or ``x-api-key``.

    ``Authorization`` accepts ``Bearer <token>`` (any case) or the bare token,
    as the gateway does. Empty values are ignored.
    """
    presented: set[bytes] = set()
    for name, value in headers:
        if name == b"authorization":
            credential = value.strip()
            if credential[:7].lower() == b"bearer ":
                credential = credential[7:].strip()
        elif name == b"x-api-key":
            credential = value.strip()
        else:
            continue
        if credential:
            presented.add(credential)
    return presented


class ServerAuthMiddleware:
    """Require a token on every HTTP and WebSocket route except the probes."""

    def __init__(self, app: ASGIApp, tokens: Sequence[str]) -> None:
        if not tokens:
            raise ValueError("ServerAuthMiddleware needs at least one token")
        self.app = app
        self._tokens = tuple(token.encode() for token in tokens)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or scope["path"] in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        rejection = self._rejection(request_credentials(scope["headers"]))
        if rejection is None:
            await self.app(scope, receive, send)
            return
        if scope["type"] == "http":
            await _send_unauthorized(send, rejection)
        else:
            await _reject_websocket(scope, receive, send, rejection)

    def _rejection(self, presented: set[bytes]) -> str | None:
        if not presented:
            return "Missing credentials: send Authorization: Bearer <token> or x-api-key: <token>"
        if len(presented) > 1:
            return "Authorization and x-api-key carry different tokens"
        (credential,) = presented
        matches = [hmac.compare_digest(credential, token) for token in self._tokens]
        if any(matches):
            return None
        return "Invalid token"


def _unauthorized_body(message: str) -> bytes:
    return orjson.dumps({"detail": {"code": "UNAUTHORIZED", "message": message}})


_UNAUTHORIZED_HEADERS = [
    (b"content-type", b"application/json"),
    (b"www-authenticate", b"Bearer"),
]


async def _send_unauthorized(send: Send, message: str) -> None:
    body = _unauthorized_body(message)
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [*_UNAUTHORIZED_HEADERS, (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _reject_websocket(scope: Scope, receive: Receive, send: Send, message: str) -> None:
    connect = await receive()
    if connect["type"] != "websocket.connect":
        return
    if "websocket.http.response" in (scope.get("extensions") or {}):
        body = _unauthorized_body(message)
        await send(
            {
                "type": "websocket.http.response.start",
                "status": 401,
                "headers": [*_UNAUTHORIZED_HEADERS, (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "websocket.http.response.body", "body": body})
        return
    await send({"type": "websocket.close", "code": 1008})
