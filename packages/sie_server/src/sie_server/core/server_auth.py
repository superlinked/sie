"""Token authentication for ``sie-server``.

One vocabulary with the gateway: ``SIE_AUTH_MODE`` selects the mode,
``SIE_AUTH_TOKENS`` (or ``SIE_AUTH_TOKEN``) holds comma-separated tokens, and a
request presents one in ``Authorization: Bearer`` or ``x-api-key``, with
``Authorization`` taking precedence when both are sent.
"""

from __future__ import annotations

import errno
import hmac
import ipaddress
import os
import secrets
import stat
import time
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
LOOPBACK_HOST_NAMES = frozenset({"localhost", "127.0.0.1", "[::1]"})

AuthMode = Literal["none", "token"]

_TOKEN_MODES = frozenset({"token", "static"})
_KNOWN_MODES = frozenset({"none", "auto"} | _TOKEN_MODES)
_MAX_TOKEN_FILE_BYTES = 4096
_GENERATED_TOKEN_BYTES = 32
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_SETTLE_ATTEMPTS = 40
_SETTLE_INTERVAL_S = 0.025


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
    """``SIE_AUTH_MODE`` without surrounding whitespace, or ``None`` when it is unset or empty.

    Values are case-sensitive, as the gateway reads them.
    """
    raw = env.get(AUTH_MODE_ENV, "").strip()
    return raw or None


def declares_tokens(env: Mapping[str, str]) -> bool:
    """Whether ``SIE_AUTH_TOKENS`` or ``SIE_AUTH_TOKEN`` is present, even with a blank value."""
    return AUTH_TOKENS_ENV in env or AUTH_TOKEN_ENV in env


def resolve_auth_mode(
    requested: str | None, *, has_tokens: bool, exposed: bool, declares_tokens: bool = False
) -> AuthMode:
    """Decide whether requests need a token.

    A configured token is never dead configuration: it turns auth on unless
    ``SIE_AUTH_MODE=none`` says otherwise, and that combination is refused.
    A token variable that is present but holds no token is refused too, unless
    the mode is ``none``, because it reads as an attempt to set a token.
    ``auto`` also turns auth on for a bind that other machines can reach.
    """
    if requested is not None and requested not in _KNOWN_MODES:
        raise AuthConfigError(f"{AUTH_MODE_ENV} must be one of none, token, static or auto (lowercase).")
    if declares_tokens and not has_tokens and requested != "none":
        raise AuthConfigError(
            f"{AUTH_TOKENS_ENV} or {AUTH_TOKEN_ENV} is set but holds no token, for example an empty or unset "
            f"variable passed through. Set it to a token, or remove it."
        )
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


def loopback_host_allowlist(bind_host: str) -> frozenset[str]:
    """Host header values a loopback server answers: the loopback names plus its own bind address."""
    candidate = bind_host.strip().lower().strip("[]")
    if not candidate:
        return LOOPBACK_HOST_NAMES
    return LOOPBACK_HOST_NAMES | {f"[{candidate}]" if ":" in candidate else candidate}


def request_host(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    """The request's Host without its port, lowercased; ``None`` when absent or malformed."""
    for name, value in headers:
        if name != b"host":
            continue
        raw = value.decode("latin-1").strip().lower()
        if raw.startswith("["):
            host, bracket, rest = raw.partition("]")
            host += bracket
        else:
            host, _, port = raw.partition(":")
            rest = f":{port}" if port else ""
        if not _bracket_ok(host) or (rest and not (rest.startswith(":") and rest[1:].isdigit())):
            return None
        return host
    return None


def _bracket_ok(host: str) -> bool:
    return bool(host) and (not host.startswith("[") or host.endswith("]"))


def token_path(directory: Path) -> Path:
    return directory / TOKEN_FILE_NAME


def load_or_create_token(directory: Path) -> StoredToken:
    """Read the persisted token, generating it on first use.

    The token appears at its path complete or not at all: it is written to a
    private temporary file and hard-linked into place, which fails if another
    process created the token first.
    """
    path = token_path(directory)
    try:
        ensure_data_dir(directory)
    except OSError as exc:
        raise TokenFileError(f"Cannot create {directory}: {exc.strerror or exc}") from exc
    check_private_directory(directory)
    token = secrets.token_urlsafe(_GENERATED_TOKEN_BYTES)
    staged = directory / f".{TOKEN_FILE_NAME}.{secrets.token_hex(8)}"
    try:
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        try:
            os.write(fd, f"{token}\n".encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(staged, path)
    except FileExistsError:
        _wait_for_settled_token(path)
        return StoredToken(token=read_stored_token(directory), path=path, created=False)
    except OSError as exc:
        raise TokenFileError(f"Cannot write {path}: {exc.strerror or exc}") from exc
    finally:
        staged.unlink(missing_ok=True)
    return StoredToken(token=token, path=path, created=True)


def _wait_for_settled_token(path: Path) -> None:
    """Give a concurrent first start time to drop its staging link."""
    for _ in range(_SETTLE_ATTEMPTS):
        try:
            if os.lstat(path).st_nlink == 1:
                return
        except OSError:
            return
        time.sleep(_SETTLE_INTERVAL_S)


def check_private_directory(directory: Path) -> None:
    """Refuse a token directory that another user owns or can write to."""
    try:
        info = directory.stat()
    except OSError as exc:
        raise TokenFileError(f"Cannot read {directory}: {exc.strerror or exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise TokenFileError(f"{directory} must be a directory.")
    if info.st_uid != os.getuid():
        raise TokenFileError(f"{directory} must be owned by the user that runs sie-server.")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise TokenFileError(f"{directory} must not be writable by others (chmod 700 {directory}).")


def read_stored_token(directory: Path) -> str:
    """Read the persisted token, refusing a file that others could read or replace."""
    check_private_directory(directory)
    path = token_path(directory)
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TokenFileError(f"{path} is a symbolic link; it must be a regular file.") from exc
        raise TokenFileError(f"Cannot read {path}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise TokenFileError(f"{path} must be a regular file.")
        if info.st_nlink != 1 and not _drop_staging_links(path, info):
            raise TokenFileError(
                f"{path} must not have other hard links. Remove the other link, or delete {path} to have "
                f"sie-server generate a new token."
            )
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
        raise TokenFileError(
            f"{path} must hold one token: printable ASCII without spaces or commas. "
            f"Delete {path} to have sie-server generate a new token."
        )
    return token


def _drop_staging_links(path: Path, info: os.stat_result) -> bool:
    """Remove the staging links an interrupted first start left on the token.

    True when every extra hard link of the token is such a staging file; any
    other link is left in place and the token is refused.
    """
    staged = []
    for entry in path.parent.glob(f".{TOKEN_FILE_NAME}.*"):
        try:
            entry_info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(entry_info.st_mode) and (entry_info.st_dev, entry_info.st_ino) == (info.st_dev, info.st_ino):
            staged.append(entry)
    if len(staged) != info.st_nlink - 1:
        return False
    for entry in staged:
        entry.unlink(missing_ok=True)
    return True


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
    """The distinct tokens a request presents, as the gateway reads them.

    ``Authorization`` (``Bearer <token>`` in any case, or the bare token) takes
    precedence: when it carries a token, ``x-api-key`` is ignored. Otherwise the
    ``x-api-key`` values count. Empty values are ignored. More than one distinct
    token means repeated headers that disagree.
    """
    authorization: set[bytes] = set()
    api_key: set[bytes] = set()
    for name, value in headers:
        if name == b"authorization":
            credential = value.strip()
            if credential[:7].lower() == b"bearer ":
                credential = credential[7:].strip()
            if credential:
                authorization.add(credential)
        elif name == b"x-api-key":
            credential = value.strip()
            if credential:
                api_key.add(credential)
    return authorization or api_key


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
        await _reject(scope, receive, send, _unauthorized(rejection))

    def _rejection(self, presented: set[bytes]) -> str | None:
        if not presented:
            return "Missing credentials: send Authorization: Bearer <token> or x-api-key: <token>"
        if len(presented) > 1:
            return "Repeated credential headers carry different tokens"
        (credential,) = presented
        matches = [hmac.compare_digest(credential, token) for token in self._tokens]
        if any(matches):
            return None
        return "Invalid token"


class LoopbackHostMiddleware:
    """Answer only Host headers that name this machine, so a DNS-rebinding page cannot reach the server."""

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self._allowed = frozenset(allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or request_host(scope["headers"]) in self._allowed:
            await self.app(scope, receive, send)
            return
        rejection = _Rejection(
            400,
            "INVALID_INPUT",
            "This server listens on loopback and answers only requests addressed to localhost, 127.0.0.1 or [::1].",
        )
        await _reject(scope, receive, send, rejection)


@dataclass(frozen=True)
class _Rejection:
    status: int
    code: str
    message: str


def _unauthorized(message: str) -> _Rejection:
    return _Rejection(401, "UNAUTHORIZED", message)


async def _reject(scope: Scope, receive: Receive, send: Send, rejection: _Rejection) -> None:
    body = orjson.dumps({"detail": {"code": rejection.code, "message": rejection.message}})
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if rejection.status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": rejection.status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
        return
    connect = await receive()
    if connect["type"] != "websocket.connect":
        return
    if "websocket.http.response" in (scope.get("extensions") or {}):
        await send({"type": "websocket.http.response.start", "status": rejection.status, "headers": headers})
        await send({"type": "websocket.http.response.body", "body": body})
        return
    await send({"type": "websocket.close", "code": 1008})
