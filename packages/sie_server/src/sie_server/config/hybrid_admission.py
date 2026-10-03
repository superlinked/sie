"""Runtime admission of exact, fresh operator-owned hybrid evidence.

Reading a record never makes it authoritative for a different model, endpoint,
profile or local runtime. The check is repeated before each bridged request,
so replacing a record or letting it expire cannot keep an old admission alive.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from sie_sdk.types import DEFAULT_OUTPUT_DTYPE

from sie_server.config.engine import EngineConfig
from sie_server.config.equivalence import (
    EquivalenceRecord,
    canonical_digest,
    model_contract_digest,
    remote_profile_contract_digest,
    upstream_contract_digest,
)
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import UpstreamKind, installed_upstreams
from sie_server.core.profile_identity import local_profile_identity, runtime_instance_id

_MAX_RECORD_BYTES = 512 << 10


def _read_record(path: str) -> EquivalenceRecord:
    # Startup configuration owns this path. Refuse devices/FIFOs and bound the
    # actual read rather than trusting a prior stat or a changing file size.
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("equivalence record must be a regular file")
        data = stream.read(_MAX_RECORD_BYTES + 1)
    if len(data) > _MAX_RECORD_BYTES:
        raise ValueError("equivalence record exceeds the byte limit")
    return EquivalenceRecord.model_validate_json(data)


def openai_equivalence_refusal(
    config: ModelConfig,
    *,
    device: str,
    engine_config: EngineConfig | None = None,
    now: datetime | None = None,
) -> str | None:
    """Return a fixed refusal reason, or ``None`` for passing current evidence."""
    if device == "cuda" or (device.startswith("cuda:") and not device.partition(":")[2].isdigit()):
        return "hybrid execution device is ambiguous"
    routing = config.routing
    if routing is None or routing.policy not in {"fallback", "threshold"}:
        return "model does not declare a hybrid policy"
    profile_name = routing.fallback_profile
    if profile_name is None:
        return "hybrid policy has no remote profile"
    upstreams = installed_upstreams()
    profile = config.resolve_profile(profile_name)
    upstream_name = profile.loadtime.get("upstream")
    upstream_model = profile.loadtime.get("upstream_model")
    upstream = upstreams.get(upstream_name) if isinstance(upstream_name, str) else None
    if upstream is None or upstream.kind is not UpstreamKind.OPENAI:
        return "hybrid profile requires immutable SIE identity or OpenAI proof"
    policy = upstream.equivalence
    if policy is None or config.sie_id not in policy.record_files:
        return "hybrid profile has no operator-owned equivalence record"
    if profile.runtime.keys() - config.resolve_profile("default").runtime.keys():
        return "hybrid remote profile adds unmeasured runtime defaults"
    if config.resolve_profile("default").runtime.get("output_dtype", DEFAULT_OUTPUT_DTYPE) != DEFAULT_OUTPUT_DTYPE:
        return "hybrid output dtype differs from the measured float32 contract"
    identity = local_profile_identity(config, "default", device=device, engine_config=engine_config)
    if identity is None:
        return "hybrid local execution cannot be identified"
    try:
        record = _read_record(policy.record_files[config.sie_id])
    except (OSError, ValueError, ValidationError, RecursionError):
        return "hybrid equivalence record cannot be validated"
    if not record.passed or not record.is_fresh(max_age_s=policy.max_age_s, now=now):
        return "hybrid equivalence record failed or is outside its configured age"
    outputs = frozenset(set(config.outputs) & {"dense", "sparse", "multivector", "score"})
    if (
        record.model != config.sie_id
        or record.local_profile != "default"
        or record.remote_profile != profile_name
        or record.upstream_name != upstream_name
        or record.upstream_model != upstream_model
        or record.context_length != config.max_sequence_length
        or record.outputs != outputs
        or record.local_identity != identity
        or record.local_instance_id != runtime_instance_id()
        or record.runtime_options_sha256 != canonical_digest(dict(config.resolve_profile("default").runtime))
        or record.model_contract_sha256 != model_contract_digest(config)
        or record.upstream_contract_sha256 != upstream_contract_digest(upstream)
        or record.remote_contract_sha256 != remote_profile_contract_digest(config, profile_name, upstreams)
        or record.local_observation_sha256 != canonical_digest({"identity": identity, "revision": config.hf_revision})
    ):
        return "hybrid equivalence record differs from the current serving contract"
    return None


def hybrid_request_refusal(config: ModelConfig, request_options: Mapping[str, object] | None) -> str | None:
    """Keep valid local numerical overrides off a bridge measured at defaults."""
    defaults = {"output_dtype": DEFAULT_OUTPUT_DTYPE, **config.resolve_profile("default").runtime}
    try:
        differs = any(
            key not in {"profile", "is_query"}
            and (key not in defaults or canonical_digest(value) != canonical_digest(defaults[key]))
            for key, value in (request_options or {}).items()
        )
    except (TypeError, ValueError, RecursionError):
        differs = True
    if differs:
        return "hybrid request differs from the measured runtime options"
    return None
