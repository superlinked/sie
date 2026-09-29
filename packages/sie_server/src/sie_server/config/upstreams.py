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
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

UPSTREAMS_FILE_ENV = "SIE_UPSTREAMS_FILE"

_UPSTREAM_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


class UpstreamConfigError(ValueError):
    """An upstreams file or an upstream definition is invalid."""


class UpstreamCredentialError(RuntimeError):
    """The environment variable holding an upstream's credential is not set."""


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
    if "?" in url or "#" in url:
        raise UpstreamConfigError(f"{field} must not carry a query or a fragment")
    try:
        parts = urlsplit(url)
        _ = parts.port
    except ValueError:
        raise UpstreamConfigError(f"{field} is not a valid URL") from None
    if parts.scheme not in {"http", "https"}:
        raise UpstreamConfigError(f"{field} must be an http or https URL")
    if "@" in parts.netloc:
        raise UpstreamConfigError(f"{field} must not carry credentials")
    if not parts.hostname:
        raise UpstreamConfigError(f"{field} must name a host")
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

    def api_key(self) -> str | None:
        """Read the credential now. ``None`` when the upstream declares none."""
        if self.api_key_secret is None:
            return None
        value = os.environ.get(self.api_key_secret)
        if not value:
            raise UpstreamCredentialError(f"environment variable {self.api_key_secret} is not set")
        return value


class UpstreamsFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    upstreams: dict[str, Upstream] = Field(default_factory=dict)

    @field_validator("upstreams")
    @classmethod
    def _names(cls, value: dict[str, Upstream]) -> dict[str, Upstream]:
        for name in value:
            if not _UPSTREAM_NAME.fullmatch(name):
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
        raw = yaml.safe_load(text)
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
