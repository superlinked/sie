"""Fresh, bounded SDK observations of an immutable SIE execution profile."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx
from sie_sdk import RequestError, ServerError, SIEClient, SIEConnectionError

from sie_server.adapters.errors import UpstreamUnavailableError
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.config.engine import EngineConfig
from sie_server.config.equivalence import canonical_digest
from sie_server.config.model import UPSTREAM_MODEL_PATTERN, ModelConfig, is_immutable_revision
from sie_server.config.upstreams import (
    Upstream,
    UpstreamCredentialError,
    UpstreamKind,
    installed_upstreams,
    remote_serving_enabled,
)
from sie_server.core.profile_identity import local_profile_identity
from sie_server.core.upstream_client import upstream_sync_client
from sie_server.core.upstream_deadline import DEADLINE_EXTENSION, DeadlineTransport

_MAX_METADATA_BYTES = 64 << 10
_METADATA_DEADLINE_S = 5.0
_IDENTITY_AGE_S = 30.0
_REFUSAL_AGE_S = 2.0
_MAX_CACHE_ENTRIES = 128
_IDENTITY = re.compile(r"^v[12]:sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"[0-9a-f]{40}")


class _MetadataReadError(httpx.HTTPError):
    """Fixed metadata failure without upstream content or credentials."""


class _BoundedMetadataTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport) -> None:
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        request.extensions[DEADLINE_EXTENSION] = started + _METADATA_DEADLINE_S
        response = self._transport.handle_request(request)
        try:
            if response.status_code != httpx.codes.OK:
                return httpx.Response(response.status_code, headers=response.headers, content=b"", request=request)
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                raise _MetadataReadError("upstream identity metadata encoding is refused")
            data = bytearray()
            if not isinstance(response.stream, httpx.SyncByteStream):
                raise _MetadataReadError("upstream identity metadata stream is invalid")
            for part in response.stream:
                if len(data) + len(part) > _MAX_METADATA_BYTES or time.monotonic() - started > _METADATA_DEADLINE_S:
                    raise _MetadataReadError("upstream identity metadata exceeded its read bound")
                data.extend(part)
            if time.monotonic() - started > _METADATA_DEADLINE_S:
                raise _MetadataReadError("upstream identity metadata exceeded its read bound")
            return httpx.Response(200, headers=response.headers, content=bytes(data), request=request)
        finally:
            response.close()

    def close(self) -> None:
        self._transport.close()


def _read_identity(upstream_name: str, upstream: Upstream, remote_model: str) -> tuple[str, str] | None:
    if not remote_serving_enabled() or installed_upstreams().get(upstream_name) is not upstream:
        return None
    model, separator, profile = remote_model.partition(":")
    profile = profile if separator else "default"
    transport = _BoundedMetadataTransport(DeadlineTransport(proxy=upstream.proxy_url))
    with upstream_sync_client(upstream, timeout=httpx.Timeout(1.0, connect=3.0), transport=transport) as http_client:
        http_client.headers["Accept-Encoding"] = "identity"
        with (
            SIEClient(upstream.base_url, api_key="", http_client=http_client) as client,
            upstream_limiter(upstream_name).call(),
        ):
            try:
                metadata: Any = client.get_model(model)
            except RequestError:
                return None
            except (ServerError, SIEConnectionError, httpx.HTTPError, ValueError, RecursionError, OSError) as exc:
                raise UpstreamUnavailableError(
                    upstream_name, "unavailable", retry_after_s=1, reason="identity metadata could not be validated"
                ) from exc
    if not isinstance(metadata, dict) or metadata.get("name") != model:
        return None
    profiles = metadata.get("profiles")
    selected = profiles.get(profile) if isinstance(profiles, dict) else None
    revision = metadata.get("revision")
    identity = selected.get("identity") if isinstance(selected, dict) else None
    if (
        not isinstance(selected, dict)
        or not isinstance(revision, str)
        or not is_immutable_revision(revision)
        or not isinstance(identity, str)
        or not _IDENTITY.fullmatch(identity)
        or selected.get("remote_contract_sha256") is not None
    ):
        return None
    return revision, identity


@dataclass
class _Observation:
    upstream: Upstream
    value: tuple[str, str] | None = None
    checked_at: float = 0.0
    loading: bool = True


_LOCK = threading.Lock()
_OBSERVATIONS: OrderedDict[tuple[int, str, str], _Observation] = OrderedDict()


def _fresh_identity(upstream_name: str, upstream: Upstream, remote_model: str) -> tuple[str, str] | None:
    key = id(upstream), upstream_name, remote_model
    with _LOCK:
        existing = _OBSERVATIONS.get(key)
        if existing is not None:
            _OBSERVATIONS.move_to_end(key)
            age = time.monotonic() - existing.checked_at
            if existing.loading:
                return None
            if 0 <= age < (_IDENTITY_AGE_S if existing.value is not None else _REFUSAL_AGE_S):
                return existing.value
        elif len(_OBSERVATIONS) >= _MAX_CACHE_ENTRIES:
            candidate = next((candidate for candidate, entry in _OBSERVATIONS.items() if not entry.loading), None)
            if candidate is None:
                return None
            del _OBSERVATIONS[candidate]
        observation = _Observation(upstream)
        _OBSERVATIONS[key] = observation
    value = None
    try:
        value = _read_identity(upstream_name, upstream, remote_model)
    except (UpstreamUnavailableError, UpstreamCredentialError, httpx.HTTPError, OSError, ValueError, RecursionError):
        pass
    finally:
        with _LOCK:
            observation.value = value
            observation.checked_at = time.monotonic()
            observation.loading = False
    return value


def _shown(value: object, pattern: re.Pattern[str]) -> str:
    return value if isinstance(value, str) and pattern.fullmatch(value) else "<invalid>"


def _shown_identity(revision: object, identity: object) -> str:
    return f"(hf_revision={_shown(revision, _REVISION)}, identity={_shown(identity, _IDENTITY)})"


def sie_identity_refusal(config: ModelConfig, *, device: str, engine_config: EngineConfig | None = None) -> str | None:
    """Admit only fresh matching weights and a known local execution identity."""
    if not remote_serving_enabled():
        return "remote serving is disabled"
    routing = config.routing
    if routing is None or routing.policy not in {"fallback", "threshold"} or routing.fallback_profile is None:
        return "model does not declare a hybrid remote profile"
    if not device or device == "cuda" or (device.startswith("cuda:") and not device.partition(":")[2].isdigit()):
        return "hybrid execution device is ambiguous"
    remote = config.resolve_profile(routing.fallback_profile)
    upstream_name, remote_model = remote.loadtime.get("upstream"), remote.loadtime.get("upstream_model")
    upstream = installed_upstreams().get(upstream_name) if isinstance(upstream_name, str) else None
    if not isinstance(upstream_name, str) or upstream is None or upstream.kind is not UpstreamKind.SIE:
        return "hybrid profile requires an SIE upstream"
    if not isinstance(remote_model, str) or not UPSTREAM_MODEL_PATTERN.fullmatch(remote_model):
        return "hybrid upstream model is invalid"
    if ":" not in remote_model:
        return "hybrid SIE upstream must name an explicit local profile"
    if upstream.set_params or upstream.strip_params:
        return "hybrid SIE identity cannot authorize upstream request transforms"
    defaults = config.resolve_profile("default").runtime
    if defaults.get("output_dtype", "float32") != "float32":
        return "hybrid SIE identity requires the float32 wire contract"
    if defaults.get("muvera") is not None:
        return "hybrid SIE identity cannot authorize repeated postprocessing"
    try:
        if any(
            key not in defaults or canonical_digest(value) != canonical_digest(defaults[key])
            for key, value in remote.runtime.items()
        ):
            return "hybrid remote profile differs from local runtime defaults"
    except (TypeError, ValueError, RecursionError):
        return "hybrid runtime defaults cannot be identified"
    identity = local_profile_identity(config, "default", device=device, engine_config=engine_config)
    if identity is None:
        return "hybrid local execution cannot be identified"
    observed = _fresh_identity(upstream_name, upstream, remote_model)
    if observed is None:
        return "hybrid upstream identity is unavailable or outside its age"
    if observed != (config.hf_revision, identity):
        return (
            "hybrid upstream weights or execution profile differs from local: "
            f"upstream {_shown_identity(*observed)}, local {_shown_identity(config.hf_revision, identity)}"
        )
    return None
