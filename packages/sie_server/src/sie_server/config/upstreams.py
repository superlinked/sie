"""Operator-defined upstreams for remote profiles.

An upstream is a named endpoint outside this deployment. It is defined only in
the server's startup configuration (``--upstreams-file`` or
``SIE_UPSTREAMS_FILE``), never through the model configuration API. Its
credential is the name of an environment variable. The value is read when a
request is sent and is never stored on the parsed configuration.

An upstream of kind ``openai`` declares the endpoints it offers, and the
operator may set or strip request fields on every call to it. The fields that
carry the caller's input or that the server reads the answer by can be neither
set nor stripped.

Validation messages never repeat the offending value, because a rejected URL or
a literal credential is exactly what must not reach a log.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator
from yaml.constructor import ConstructorError

from sie_server.config.model import UPSTREAM_NAME_PATTERN, is_remote_adapter_path

if TYPE_CHECKING:
    from sie_server.config.model import ModelConfig

UPSTREAMS_FILE_ENV = "SIE_UPSTREAMS_FILE"
REMOTE_SERVING_ENV = "SIE_REMOTE_SERVING"

_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_VISIBLE_ASCII = re.compile(r"^[\x21-\x7e]+$")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_MAX_PARAMS = 32
_MAX_PARAM_DEPTH = 8
_MAX_SET_PARAMS_BYTES = 16 << 10
_MAX_EQUIVALENCE_MODEL_LENGTH = 256
_MAX_EQUIVALENCE_PATH_LENGTH = 4096

RESERVED_UPSTREAM_PARAMS = frozenset(
    {
        "documents",
        "encoding_format",
        "input",
        "messages",
        "model",
        "n",
        "prompt",
        "query",
        "stream",
        "stream_options",
        "top_n",
    }
)
"""Request fields an operator can neither set nor strip, because the server sends them itself."""


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


class UpstreamEndpoint(StrEnum):
    """An OpenAI-compatible endpoint that an upstream of kind ``openai`` offers."""

    COMPLETIONS = "completions"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"


ENDPOINT_OUTPUTS: Mapping[UpstreamEndpoint, str] = MappingProxyType(
    {
        UpstreamEndpoint.COMPLETIONS: "tokens",
        UpstreamEndpoint.CHAT: "tokens",
        UpstreamEndpoint.EMBEDDINGS: "dense",
        UpstreamEndpoint.RERANK: "score",
    }
)
"""The model output each endpoint produces, named as in ``ModelConfig.outputs``."""


def _refuse_repeats(items: Sequence[object], *, field: str) -> None:
    seen: set[str] = set()
    for item in items:
        if isinstance(item, str):
            if item in seen:
                raise UpstreamConfigError(f"{field} lists the same entry more than once")
            seen.add(item)


def _check_param_name(name: str, *, field: str, action: str) -> None:
    if not _PARAM_NAME.fullmatch(name):
        raise UpstreamConfigError(
            f"{field} names are letters, digits and underscores, at most 64 characters, and do not start with a digit"
        )
    if name in RESERVED_UPSTREAM_PARAMS:
        raise UpstreamConfigError(f"{field} cannot {action} {name!r}, which the server sends itself")


def _json_value_problem(value: object) -> str | None:
    """Why ``value`` is not a bounded JSON value, or ``None``.

    The walk adds up a lower bound of the JSON size, one byte per value plus
    one per string character or container entry, and stops at the size limit.
    That keeps it short even when YAML aliases repeat one subtree many times.
    """
    pending: list[tuple[object, int]] = [(value, 0)]
    size = 0
    while pending:
        item, depth = pending.pop()
        size += 1 + (len(item) if isinstance(item, str | dict | list) else 0)
        if size > _MAX_SET_PARAMS_BYTES:
            return f"is larger than {_MAX_SET_PARAMS_BYTES} bytes as JSON"
        if item is None or isinstance(item, bool | int | str):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                return "holds a number that is not finite"
            continue
        if not isinstance(item, dict | list):
            return "holds a value that is not JSON"
        if depth >= _MAX_PARAM_DEPTH:
            return f"nests deeper than {_MAX_PARAM_DEPTH} levels"
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                return "holds an object key that is not a string"
            pending.extend((child, depth + 1) for child in item.values())
        else:
            pending.extend((child, depth + 1) for child in item)
    return None


class RateCap(BaseModel):
    """Per-replica limits on calls to one upstream."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    requests_per_minute: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)


class Breaker(BaseModel):
    """When an upstream that keeps failing stops receiving calls, and for how long.

    The circuit opens for ``cooldown_s`` once ``failures`` calls in a row, all
    within ``window_s``, found the upstream unavailable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    failures: int = Field(default=5, gt=0, le=1000)
    window_s: float = Field(default=30.0, gt=0, le=3600)
    cooldown_s: float = Field(default=60.0, gt=0, le=3600)


class EquivalencePolicy(BaseModel):
    """Operator-owned records for hybrid embedding/scoring admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_age_s: int = Field(gt=0, le=604_800, strict=True)
    record_files: dict[str, str] = Field(min_length=1, max_length=256)

    @field_validator("record_files")
    @classmethod
    def _record_files(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not name
            or len(name) > _MAX_EQUIVALENCE_MODEL_LENGTH
            or not path
            or len(path) > _MAX_EQUIVALENCE_PATH_LENGTH
            or not Path(path).is_absolute()
            for name, path in value.items()
        ):
            raise UpstreamConfigError(
                "equivalence record names must be bounded; file paths must be bounded and absolute"
            )
        return value


class Upstream(BaseModel):
    """One named upstream from the startup configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: UpstreamKind
    base_url: str
    api_key_secret: str | None = None
    """Name of the environment variable that holds the credential."""
    rate_cap: RateCap
    breaker: Breaker = Field(default_factory=Breaker)
    proxy_url: str | None = None
    """The only egress proxy used for this upstream. Ambient proxy variables are ignored."""
    endpoints: frozenset[UpstreamEndpoint] = frozenset()
    """For kind ``openai``: the endpoints the upstream offers."""
    set_params: dict[str, Any] = Field(default_factory=dict, repr=False)
    """For kind ``openai``: request fields added to every call, in place of any value the server sends."""
    strip_params: frozenset[str] = frozenset()
    """For kind ``openai``: request fields removed from every call."""
    equivalence: EquivalencePolicy | None = Field(default=None, repr=False)
    """For kind ``openai``: local files containing exact-model measured evidence."""

    @field_validator("base_url")
    @classmethod
    def _base_url(cls, value: str) -> str:
        return validate_upstream_url(value)

    @field_validator("endpoints", "strip_params", mode="before")
    @classmethod
    def _no_repeats(cls, value: object, info: ValidationInfo) -> object:
        if isinstance(value, list | tuple):
            _refuse_repeats(value, field=str(info.field_name))
        return value

    @field_validator("set_params")
    @classmethod
    def _set_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > _MAX_PARAMS:
            raise UpstreamConfigError(f"set_params sets at most {_MAX_PARAMS} fields")
        for name, item in value.items():
            _check_param_name(name, field="set_params", action="set")
            problem = _json_value_problem(item)
            if problem is not None:
                raise UpstreamConfigError(f"set_params[{name!r}] {problem}")
        if len(json.dumps(value, separators=(",", ":"))) > _MAX_SET_PARAMS_BYTES:
            raise UpstreamConfigError(f"set_params is larger than {_MAX_SET_PARAMS_BYTES} bytes as JSON")
        return value

    @field_validator("strip_params")
    @classmethod
    def _strip_params(cls, value: frozenset[str]) -> frozenset[str]:
        if len(value) > _MAX_PARAMS:
            raise UpstreamConfigError(f"strip_params strips at most {_MAX_PARAMS} fields")
        for name in sorted(value):
            _check_param_name(name, field="strip_params", action="strip")
        return value

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

    @model_validator(mode="after")
    def _openai_fields(self) -> Upstream:
        if self.kind is UpstreamKind.SIE:
            if self.equivalence is not None:
                raise UpstreamConfigError("kind sie uses immutable identity, not OpenAI equivalence records")
            declared = [name for name in ("endpoints", "set_params", "strip_params") if getattr(self, name)]
            if declared:
                raise UpstreamConfigError(f"kind sie takes no {' or '.join(declared)}")
        elif not self.endpoints:
            raise UpstreamConfigError("kind openai must declare the endpoints it offers")
        both = sorted(self.set_params.keys() & self.strip_params)
        if both:
            raise UpstreamConfigError(f"set_params and strip_params both name {', '.join(map(repr, both))}")
        return self

    @property
    def outputs(self) -> frozenset[str] | None:
        """The model outputs a remote profile on this upstream can produce. ``None`` when the kind sets no limit."""
        if self.kind is UpstreamKind.SIE:
            return None
        return frozenset(ENDPOINT_OUTPUTS[endpoint] for endpoint in self.endpoints)

    def apply_params(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """``body`` as one call to this upstream sends it: operator fields set, then stripped."""
        sent = dict(body)
        sent.update(copy.deepcopy(self.set_params))
        for name in self.strip_params:
            sent.pop(name, None)
        return sent

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


def remote_serving_enabled() -> bool:
    """Whether this server may serve any request through a remote profile."""
    return _INSTALLED.remote_serving


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

    A remote profile on an ``openai`` upstream is also rejected when the
    upstream declares no endpoint for any output of the model. A request for
    an output the upstream cannot produce is refused when it arrives.

    With remote serving off, every remote profile is refused at load anyway,
    so the names are not checked and a stale model does not stop the server.
    """
    if not _INSTALLED.remote_serving:
        return
    for profile_name in config.profiles:
        resolved = config.resolve_profile(profile_name)
        if not is_remote_adapter_path(resolved.adapter_path):
            continue
        name = resolved.loadtime.get("upstream")
        upstream = _INSTALLED.upstreams.get(name) if isinstance(name, str) else None
        if upstream is None:
            msg = f"Profile '{profile_name}' of '{config.sie_id}' names an undefined upstream {name!r}"
            raise ValueError(msg)
        produced = upstream.outputs
        if produced is not None and produced.isdisjoint(config.outputs):
            usable = sorted(endpoint.value for endpoint, output in ENDPOINT_OUTPUTS.items() if output in config.outputs)
            remedy = (
                f"declare {' or '.join(usable)}" if usable else "an upstream of kind openai has no endpoint for them"
            )
            msg = (
                f"Profile '{profile_name}' of '{config.sie_id}' names upstream {name!r}, whose endpoints produce "
                f"none of the model's outputs ({', '.join(config.outputs)}); {remedy}"
            )
            raise ValueError(msg)
