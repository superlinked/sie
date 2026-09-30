"""Operator-defined upstreams for remote profiles.

An upstream is a named endpoint outside this deployment. It is defined only in
the server's startup configuration (``--upstreams-file`` or
``SIE_UPSTREAMS_FILE``), never through the model configuration API. Its
credential is the name of an environment variable. The value is read when a
request is sent and is never stored on the parsed configuration.

Validation messages never repeat the offending value, because a rejected URL or
a literal credential is exactly what must not reach a log.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml.constructor import ConstructorError

from sie_server.config.model import UPSTREAM_NAME_PATTERN, is_remote_adapter_path

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig

UPSTREAMS_FILE_ENV = "SIE_UPSTREAMS_FILE"
REMOTE_SERVING_ENV = "SIE_REMOTE_SERVING"

_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_VISIBLE_ASCII = re.compile(r"^[\x21-\x7e]+$")
_DEFAULT_PORTS = {"http": 80, "https": 443}


class UpstreamConfigError(ValueError):
    """An upstreams file or an upstream definition is invalid."""


class _DuplicateKeyError(yaml.YAMLError):
    def __init__(self, line: int) -> None:
        super().__init__(f"duplicate key on line {line}")
        self.line = line


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that refuses a repeated mapping key instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError:
                raise ConstructorError(problem="found an unhashable mapping key") from None
            if key in seen:
                raise _DuplicateKeyError(key_node.start_mark.line + 1)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class UpstreamCredentialError(RuntimeError):
    """The environment variable holding an upstream's credential is not set."""


class RemoteServingDisabledError(RuntimeError):
    """Remote serving is switched off for this server."""


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_upstream_url(url: str, *, field: str = "base_url", require_tls: bool = True) -> str:
    """Return ``url`` without a trailing slash, or raise :class:`UpstreamConfigError`.

    A URL that carries credentials, a query or a fragment is rejected. With
    ``require_tls``, plain HTTP is accepted only for a loopback host.
    """
    if not _VISIBLE_ASCII.fullmatch(url):
        raise UpstreamConfigError(f"{field} must be printable ASCII without spaces; use punycode for IDN hosts")
    if "?" in url or "#" in url:
        raise UpstreamConfigError(f"{field} must not carry a query or a fragment")
    try:
        parts = urlsplit(url)
        port = parts.port
        sent = httpx.URL(url)
    except (ValueError, httpx.InvalidURL):
        raise UpstreamConfigError(f"{field} is not a valid URL") from None
    if parts.scheme not in {"http", "https"}:
        raise UpstreamConfigError(f"{field} must be an http or https URL")
    if "@" in parts.netloc:
        raise UpstreamConfigError(f"{field} must not carry credentials")
    if "%" in parts.netloc:
        raise UpstreamConfigError(f"{field} must not percent-encode the host")
    if not parts.hostname:
        raise UpstreamConfigError(f"{field} must name a host")
    # The URL is checked with one parser and sent with another; both must agree
    # on where it goes.
    default_port = _DEFAULT_PORTS.get(parts.scheme)
    sent_host = sent.raw_host.decode("ascii").lower()
    if (sent.scheme, sent_host, sent.port or default_port) != (parts.scheme, parts.hostname, port or default_port):
        raise UpstreamConfigError(f"{field} is parsed inconsistently")
    if require_tls and parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise UpstreamConfigError(f"{field} must use https outside loopback")
    return url.rstrip("/")


class UpstreamKind(StrEnum):
    OPENAI = "openai"
    SIE = "sie"


class RateCap(BaseModel):
    """Per-replica limits on calls to one upstream."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    requests_per_minute: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)


class Upstream(BaseModel):
    """One named upstream from the startup configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: UpstreamKind
    base_url: str
    api_key_secret: str | None = None
    """Name of the environment variable that holds the credential."""
    rate_cap: RateCap
    proxy_url: str | None = None
    """The only egress proxy used for this upstream. Ambient proxy variables are ignored."""

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: str) -> str:
        return validate_upstream_url(value)

    @field_validator("proxy_url")
    @classmethod
    def _proxy_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_upstream_url(value, field="proxy_url", require_tls=False)

    @field_validator("api_key_secret")
    @classmethod
    def _api_key_secret(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME.fullmatch(value):
            raise UpstreamConfigError("api_key_secret must name an environment variable, not hold a credential")
        return value

    @model_validator(mode="after")
    def _proxy_only_over_tls(self) -> Upstream:
        # A plain-HTTP request through a proxy reaches the proxy in cleartext,
        # bearer token included. Over TLS the proxy only sees a CONNECT tunnel.
        if self.proxy_url is not None and not self.base_url.startswith("https://"):
            raise UpstreamConfigError("proxy_url requires an https base_url")
        return self

    def api_key(self) -> str | None:
        """Read the credential now. ``None`` when the upstream declares none.

        Surrounding whitespace, such as the newline a secret file often ends
        with, is removed. Any other non-printable character is refused without
        repeating the value, because an HTTP library would quote it in its error.
        """
        if self.api_key_secret is None:
            return None
        value = os.environ.get(self.api_key_secret, "").strip()
        if not value:
            raise UpstreamCredentialError(f"environment variable {self.api_key_secret} is not set")
        if not _VISIBLE_ASCII.fullmatch(value):
            raise UpstreamCredentialError(
                f"environment variable {self.api_key_secret} holds characters a credential cannot contain"
            )
        return value


class UpstreamsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    upstreams: dict[str, Upstream] = Field(default_factory=dict)

    @field_validator("upstreams")
    @classmethod
    def _names(cls, value: dict[str, Upstream]) -> dict[str, Upstream]:
        for name in value:
            if not UPSTREAM_NAME_PATTERN.fullmatch(name):
                raise UpstreamConfigError(
                    "upstream names are lowercase letters, digits and hyphens, at most 63 characters"
                )
        return value


def load_upstreams(path: str | Path) -> dict[str, Upstream]:
    """Parse and validate an upstreams file. Raises :class:`UpstreamConfigError`."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise UpstreamConfigError(f"upstreams file {path} could not be read: {exc.strerror}") from None
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader subclass
    except _DuplicateKeyError as exc:
        raise UpstreamConfigError(f"upstreams file {path} repeats a key on line {exc.line}") from None
    except yaml.YAMLError:
        raise UpstreamConfigError(f"upstreams file {path} is not valid YAML") from None
    if raw is None:
        return {}
    try:
        parsed = UpstreamsFile.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'file'}: {error['msg']}"
            for error in exc.errors(include_url=False, include_input=False)
        )
        raise UpstreamConfigError(f"upstreams file {path} is invalid: {problems}") from None
    return dict(parsed.upstreams)


class _InstalledUpstreams:
    """The upstreams this process loaded at startup, and the switch over all of them.

    The model registry and the remote adapters read them here, because the
    upstreams are server configuration and neither receives the app state.
    """

    def __init__(self) -> None:
        self.upstreams: Mapping[str, Upstream] = MappingProxyType({})
        self.remote_serving = True


_INSTALLED = _InstalledUpstreams()


def remote_serving_from_env(raw: str | None) -> bool:
    """Parse ``SIE_REMOTE_SERVING``. Unset means on; anything unrecognised means off."""
    return raw is None or raw.strip().lower() in {"1", "true", "yes", "on"}


def install_upstreams(upstreams: Mapping[str, Upstream], *, remote_serving: bool = True) -> None:
    """Install the startup upstreams. ``remote_serving=False`` refuses every remote profile."""
    _INSTALLED.upstreams = MappingProxyType(dict(upstreams))
    _INSTALLED.remote_serving = remote_serving


def installed_upstreams() -> Mapping[str, Upstream]:
    return _INSTALLED.upstreams


def upstream_for_serving(name: str) -> Upstream:
    """The upstream a remote profile may call now. Raises when serving is refused."""
    if not _INSTALLED.remote_serving:
        raise RemoteServingDisabledError(f"remote serving is switched off ({REMOTE_SERVING_ENV})")
    upstream = _INSTALLED.upstreams.get(name)
    if upstream is None:
        raise UpstreamConfigError(f"upstream {name!r} is not defined in the startup configuration")
    return upstream


def validate_profile_upstreams(config: ModelConfig) -> None:
    """Reject a model whose remote profile names an upstream this server does not define.

    With remote serving off, every remote profile is refused at load anyway,
    so the names are not checked and a stale model does not stop the server.
    """
    if not _INSTALLED.remote_serving:
        return
    for profile_name in config.profiles:
        resolved = config.resolve_profile(profile_name)
        if not is_remote_adapter_path(resolved.adapter_path):
            continue
        upstream = resolved.loadtime.get("upstream")
        if upstream not in _INSTALLED.upstreams:
            msg = f"Profile '{profile_name}' of '{config.sie_id}' names an undefined upstream {upstream!r}"
            raise ValueError(msg)
