"""Token authentication for ``sie-server``.

One vocabulary with the gateway: ``SIE_AUTH_MODE`` selects the mode,
``SIE_AUTH_TOKENS`` (or ``SIE_AUTH_TOKEN``) holds comma-separated tokens, and a
request presents one in ``Authorization: Bearer`` or ``x-api-key``, with
``Authorization`` taking precedence when both carry a token. Repeated headers
are read differently until 0.13.0: the gateway reads only the first
``Authorization`` value and uses ``x-api-key`` when that value is empty or
blank, while this server reads every value, skips empty ones and refuses values that
carry different tokens.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import hmac
import ipaddress
import logging
import os
import re
import secrets
import shlex
import stat
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import orjson
from starlette.types import ASGIApp, Receive, Scope, Send

from sie_server.core.data_dir import (
    DATA_DIR_ENV,
    ensure_data_dir,
    owned_files,
    remove_stale_files,
    resolve_data_dir,
)

logger = logging.getLogger(__name__)

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
_TEMP_PREFIX = f".{TOKEN_FILE_NAME}.sie-server-"
_TEMP_NAME = re.compile(rf"{re.escape(_TEMP_PREFIX)}[0-9a-f]{{16}}")
_READ_ONLY_ERRNOS = frozenset({errno.EACCES, errno.EROFS})
_NO_ACL_ERRNOS = frozenset({errno.ENOENT, errno.ENOTSUP, errno.EOPNOTSUPP})
_ACL_TYPE_EXTENDED = 0x100
_ACL_FIRST_ENTRY = 0
_ACL_NEXT_ENTRY = -1
_ACL_EXTENDED_DENY = 2
_HOST_CHECK_RELEASE = "0.13.0"
_MAX_WARNED_HOSTS = 64
_MAX_LOGGED_HOST_CHARS = 100
_LINK_SCAN_ATTEMPTS = 3


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

    An existing token is read without writing to the directory, so a read-only
    directory or volume works once it holds the token. A new token appears at
    its path complete or not at all: it is written to a private temporary file
    and hard-linked into place, which fails if another process created the
    token first.
    """
    path = token_path(directory)
    try:
        ensure_data_dir(directory)
    except OSError as exc:
        raise TokenFileError(f"Cannot create {directory}: {exc.strerror or exc}") from exc
    check_private_directory(directory)
    remove_stale_files(directory, _TEMP_NAME)
    if os.path.lexists(path):
        return StoredToken(token=read_stored_token(directory), path=path, created=False)
    token = secrets.token_urlsafe(_GENERATED_TOKEN_BYTES)
    staged = directory / f"{_TEMP_PREFIX}{secrets.token_hex(8)}"
    try:
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    except OSError as exc:
        if exc.errno in _READ_ONLY_ERRNOS and os.path.lexists(path):
            return StoredToken(token=read_stored_token(directory), path=path, created=False)
        raise TokenFileError(f"Cannot create {staged}: {exc.strerror or exc}") from exc
    try:
        try:
            os.write(fd, f"{token}\n".encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(staged, path)
    except FileExistsError:
        return StoredToken(token=read_stored_token(directory), path=path, created=False)
    except OSError as exc:
        raise TokenFileError(f"Cannot create {path} from {staged}: {exc.strerror or exc}") from exc
    finally:
        staged.unlink(missing_ok=True)
    return StoredToken(token=token, path=path, created=True)


def check_private_directory(directory: Path) -> None:
    """Refuse a token directory that another user owns or can write to, including through a macOS access control list."""
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except NotADirectoryError as exc:
        raise TokenFileError(f"{directory} must be a directory.") from exc
    except OSError as exc:
        raise TokenFileError(f"Cannot read {directory}: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(fd)
        quoted = shlex.quote(str(directory))
        if info.st_uid != os.getuid():
            raise TokenFileError(f"{directory} must be owned by the user that runs sie-server.")
        if stat.S_IMODE(info.st_mode) & 0o022:
            raise TokenFileError(f"{directory} must not be writable by others (chmod 700 {quoted}).")
        if _acl_allows_access(fd, directory):
            raise TokenFileError(
                f"{directory} has an access control list entry that allows access, which its mode does not show. "
                f'List the entries with ls -led {quoted} and remove each allow entry with chmod -a "<entry>" '
                f"{quoted}, giving the entry's text as listed, without its number or the word inherited. "
                f"Or set {DATA_DIR_ENV} to a directory of its own."
            )
    finally:
        os.close(fd)


@functools.cache
def _libsystem() -> ctypes.CDLL:
    libsystem = ctypes.CDLL(None, use_errno=True)
    libsystem.acl_get_fd_np.restype = ctypes.c_void_p
    libsystem.acl_get_fd_np.argtypes = (ctypes.c_int, ctypes.c_int)
    libsystem.acl_get_entry.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
    libsystem.acl_get_tag_type.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_int))
    libsystem.acl_free.argtypes = (ctypes.c_void_p,)
    return libsystem


def _acl_allows_access(fd: int, target: Path) -> bool:
    try:
        return acl_allows_access(fd)
    except OSError as exc:
        raise TokenFileError(f"Cannot read the access control list of {target}: {exc.strerror}") from exc


def acl_allows_access(fd: int) -> bool:
    """Whether the macOS access control list of ``fd`` has an entry other than a deny entry.

    Such an entry can grant access that the mode bits do not show. Deny entries, such as the one macOS puts on a
    home folder, only take access away. An entry whose kind cannot be read counts as allowing access, and a list
    that cannot be read raises ``OSError``. Always false on other platforms.
    """
    if sys.platform != "darwin":
        return False
    libsystem = _libsystem()
    ctypes.set_errno(0)
    acl = libsystem.acl_get_fd_np(fd, _ACL_TYPE_EXTENDED)
    if not acl:
        code = ctypes.get_errno()
        if code in _NO_ACL_ERRNOS:
            return False
        raise OSError(code, os.strerror(code))
    try:
        entry = ctypes.c_void_p()
        tag = ctypes.c_int()
        position = _ACL_FIRST_ENTRY
        while libsystem.acl_get_entry(acl, position, ctypes.byref(entry)) == 0:
            position = _ACL_NEXT_ENTRY
            if libsystem.acl_get_tag_type(entry, ctypes.byref(tag)) != 0 or tag.value != _ACL_EXTENDED_DENY:
                return True
        return False
    finally:
        libsystem.acl_free(acl)


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
        if info.st_nlink != 1 and not _drop_temporary_links(path, fd, info):
            raise TokenFileError(
                f"{path} must not have other hard links. Remove the other link, or delete {path} to have "
                f"sie-server generate a new token."
            )
        if info.st_uid != os.getuid():
            raise TokenFileError(f"{path} must be owned by the user that runs sie-server.")
        quoted = shlex.quote(str(path))
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise TokenFileError(f"{path} must not be readable or writable by others (chmod 600 {quoted}).")
        if _acl_allows_access(fd, path):
            raise TokenFileError(
                f"{path} has an access control list entry that allows access, so others may have read the token. "
                f"Delete the file (rm {quoted}) to have sie-server generate a new token. Removing the entries "
                f"with chmod -N {quoted} instead keeps the current token."
            )
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


def _drop_temporary_links(path: Path, fd: int, token: os.stat_result) -> bool:
    """Remove the temporary links a concurrent or interrupted first start left on the token.

    True when every extra hard link of the token is such a file, even if a
    read-only directory keeps them in place; any other link is left alone and
    the token is refused. The token's link count is read before and after each
    scan, and one link after the scan means the temporary links are gone. An
    unchanged count equal to the temporary links found plus one means they may
    be the only extra links, so they are removed and the count is read again.
    It must then be one, because a link renamed, added or removed while the
    directory is listed can balance the count; only when a read-only directory
    keeps every one of them in place does the balanced count decide. A count
    that changes during the scan means a concurrent start is still removing
    its link, so the scan is repeated a few times before the token is refused.
    """
    before = token.st_nlink
    for _ in range(_LINK_SCAN_ATTEMPTS):
        try:
            linked = {
                entry.name: entry
                for entry, info in owned_files(path.parent, _TEMP_NAME)
                if os.path.samestat(info, token)
            }
        except OSError:
            return False
        after = os.fstat(fd).st_nlink
        if after == 1:
            return True
        if after == before == len(linked) + 1:
            kept = [_unlink_token_link(entry, token) for entry in linked.values()]
            remaining = os.fstat(fd).st_nlink
            return remaining == 1 or (remaining == after and all(code in _READ_ONLY_ERRNOS for code in kept))
        if after == before:
            return False
        before = after
    return False


def _unlink_token_link(entry: Path, token: os.stat_result) -> int | None:
    """Unlink ``entry`` if it is still a link of the token; the error number that kept it in place, else ``None``."""
    try:
        if os.path.samestat(entry.lstat(), token):
            entry.unlink()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc.errno
    return None


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
    """The distinct tokens a request presents.

    ``Authorization`` (``Bearer <token>`` in any case, or the bare token) takes
    precedence, as on the gateway: when it carries a token, ``x-api-key`` is
    ignored. Otherwise the ``x-api-key`` values count. Every value of a
    repeated header is read and empty values are ignored, so an empty
    ``Authorization`` followed by a valid one is accepted here, while the
    gateway ignores the later value and uses ``x-api-key`` or refuses the
    request. More than one distinct token means repeated headers that
    disagree.
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


def _describe_hosts(allowed: frozenset[str]) -> str:
    names = ["localhost", "127.0.0.1", "[::1]", *sorted(allowed - LOOPBACK_HOST_NAMES)]
    return f"{', '.join(names[:-1])} or {names[-1]}"


class LoopbackHostMiddleware:
    """Watch for Host headers that do not name this machine, which is what a DNS-rebinding page sends.

    Until 0.13.0 such a request is answered, and a warning is logged once per
    Host. With ``enforce`` it is refused with 400.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str], *, enforce: bool = False) -> None:
        self.app = app
        self._allowed = frozenset(allowed_hosts)
        self._enforce = enforce
        self._hosts = _describe_hosts(self._allowed)
        self._warned: set[str | None] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        host = request_host(scope["headers"])
        if host in self._allowed:
            await self.app(scope, receive, send)
            return
        if self._enforce:
            message = f"This server listens on loopback and answers only requests addressed to {self._hosts}."
            await _reject(scope, receive, send, _Rejection(400, "INVALID_INPUT", message))
            return
        self._warn(host)
        await self.app(scope, receive, send)

    def _warn(self, host: str | None) -> None:
        key = None if host is None else host[:_MAX_LOGGED_HOST_CHARS]
        if key in self._warned or len(self._warned) >= _MAX_WARNED_HOSTS:
            return
        self._warned.add(key)
        shown = "(none)" if key is None else repr(key)
        logger.warning(
            "A request with Host %s reached this loopback server, which has authentication off. From %s "
            "sie-server refuses such requests with 400, because a web page can send them through DNS "
            "rebinding. Address the server as %s, or turn on token authentication.",
            shown,
            _HOST_CHECK_RELEASE,
            self._hosts,
        )


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
