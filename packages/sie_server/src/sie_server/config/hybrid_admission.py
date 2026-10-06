"""Runtime admission of exact, fresh operator-owned hybrid evidence.

Reading a record never makes it authoritative for a different model, endpoint,
profile or local execution identity. The check is repeated before each bridged
request, so replacing a record or letting it expire cannot keep an old
admission alive.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
from sie_server.config.fleet_equivalence import equivalence_record_digest, read_equivalence_evidence
from sie_server.config.model import ModelConfig
from sie_server.config.sie_identity import sie_upstream_identity
from sie_server.config.upstreams import UpstreamKind, installed_upstreams
from sie_server.core.profile_identity import local_profile_identity, serving_code_digest

_NUMERICAL_OUTPUTS = frozenset({"dense", "sparse", "multivector", "score"})
_MAX_CACHED_EVIDENCE_FILES = 64
_SETTLED_EVIDENCE_NS = 2_000_000_000
_evidence_lock = threading.Lock()
_evidence_cache: dict[str, tuple[tuple[int, ...], tuple[tuple[EquivalenceRecord, str], ...]]] = {}


@dataclass(frozen=True)
class NumericalAdmission:
    """The local execution identities that a remote profile's current evidence covers."""

    sha256: str
    kind: str
    local_identities: frozenset[str]
    model_contract_sha256: str
    outputs: frozenset[str]
    expires_at: datetime


def _evidence_version(path: str) -> tuple[int, ...]:
    status = Path(path).stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def _digested_evidence(path: str) -> tuple[tuple[EquivalenceRecord, str], ...]:
    """Each record of an evidence file with its digest, parsed once per version of the file."""
    version = _evidence_version(path)
    with _evidence_lock:
        cached = _evidence_cache.get(path)
    if cached is not None and cached[0] == version:
        return cached[1]
    evidence = tuple((record, equivalence_record_digest(record)) for record in read_equivalence_evidence(path))
    # A file rewritten within the filesystem's timestamp granularity can keep
    # its version, so only a version older than that is reused.
    if _evidence_version(path) == version and time.time_ns() - max(version[-2:]) > _SETTLED_EVIDENCE_NS:
        with _evidence_lock:
            if path not in _evidence_cache and len(_evidence_cache) >= _MAX_CACHED_EVIDENCE_FILES:
                _evidence_cache.pop(next(iter(_evidence_cache)))
            _evidence_cache[path] = (version, evidence)
    return evidence


def openai_admission(config: ModelConfig, *, now: datetime | None = None) -> NumericalAdmission | str:
    """Return the admission of the model's remote profile, or a fixed refusal reason.

    The admission depends on the installed evidence, this process's upstream and
    code, and the model configuration. It does not depend on the process asking.
    """
    routing = config.routing
    if routing is None or routing.policy not in {"fallback", "threshold"}:
        return "model does not declare a hybrid policy"
    profile_name = routing.fallback_profile
    if profile_name is None:
        return "hybrid policy has no remote profile"
    upstreams = installed_upstreams()
    profile = config.resolve_profile(profile_name)
    upstream_name = profile.loadtime.get("upstream")
    upstream = upstreams.get(upstream_name) if isinstance(upstream_name, str) else None
    if upstream is None or upstream.kind is not UpstreamKind.OPENAI:
        return "hybrid profile requires immutable SIE identity or OpenAI proof"
    policy = upstream.equivalence
    if policy is None or config.sie_id not in policy.record_files:
        return "hybrid profile has no operator-owned equivalence record"
    default_runtime = config.resolve_profile("default").runtime
    if default_runtime.get("output_dtype", DEFAULT_OUTPUT_DTYPE) != DEFAULT_OUTPUT_DTYPE:
        return "hybrid output dtype differs from the measured float32 contract"
    try:
        if canonical_digest(dict(profile.runtime)) != canonical_digest(dict(default_runtime)):
            return "hybrid remote profile runtime differs from the measured local runtime"
    except (TypeError, ValueError, RecursionError):
        return "hybrid runtime defaults cannot be identified"
    remote_execution = serving_code_digest()
    if remote_execution is None:
        return "hybrid remote execution cannot be identified"
    try:
        evidence = _digested_evidence(policy.record_files[config.sie_id])
    except (OSError, ValueError, ValidationError, RecursionError):
        return "hybrid equivalence record cannot be validated"
    remote_contract = remote_profile_contract_digest(config, profile_name, upstreams)
    contract = {
        "model": config.sie_id,
        "local_profile": "default",
        "remote_profile": profile_name,
        "upstream_name": upstream_name,
        "upstream_model": profile.loadtime.get("upstream_model"),
        "context_length": config.max_sequence_length,
        "outputs": frozenset(set(config.outputs) & _NUMERICAL_OUTPUTS),
        "runtime_options_sha256": canonical_digest(dict(default_runtime)),
        "model_contract_sha256": model_contract_digest(config),
        "upstream_contract_sha256": upstream_contract_digest(upstream),
        "remote_contract_sha256": remote_contract,
        "remote_execution_sha256": remote_execution,
    }
    matching = [
        (record, digest)
        for record, digest in evidence
        if all(getattr(record, field) == value for field, value in contract.items())
        and record.local_observation_sha256
        == canonical_digest({"identity": record.local_identity, "revision": config.hf_revision})
    ]
    if not matching:
        return "hybrid equivalence record differs from the current serving contract"
    observed = now or datetime.now(UTC)
    admitted = [
        (record, digest)
        for record, digest in matching
        if record.passed and record.is_fresh(max_age_s=policy.max_age_s, now=observed)
    ]
    if not admitted:
        return "hybrid equivalence record failed or is outside its configured age"
    return NumericalAdmission(
        sha256=canonical_digest(
            {
                "version": 1,
                "kind": UpstreamKind.OPENAI.value,
                "model": config.sie_id,
                "remote_profile": profile_name,
                "remote_contract_sha256": remote_contract,
                "records": sorted(digest for _, digest in admitted),
            }
        ),
        kind=UpstreamKind.OPENAI.value,
        local_identities=frozenset(record.local_identity for record, _ in admitted),
        model_contract_sha256=contract["model_contract_sha256"],
        outputs=contract["outputs"],
        expires_at=min(record.measured_at for record, _ in admitted) + timedelta(seconds=policy.max_age_s),
    )


def sie_admission(config: ModelConfig, *, wait: bool = True, now: datetime | None = None) -> NumericalAdmission | str:
    """Return the admission an SIE upstream's fresh identity grants, or a fixed refusal reason.

    The upstream's execution identity is the one identity admitted. Without
    ``wait`` the upstream is never contacted on the caller's thread.
    """
    observed = sie_upstream_identity(config, wait=wait)
    if isinstance(observed, str):
        return observed
    revision, identity, remaining_s = observed
    if revision != config.hf_revision:
        return "hybrid upstream weights differ from local"
    outputs = frozenset(set(config.outputs) & _NUMERICAL_OUTPUTS)
    if not outputs:
        return "hybrid model declares no numerical outputs"
    profile_name = config.routing.fallback_profile if config.routing is not None else None
    if profile_name is None:
        return "model does not declare a hybrid remote profile"
    return NumericalAdmission(
        sha256=canonical_digest(
            {
                "version": 1,
                "kind": UpstreamKind.SIE.value,
                "model": config.sie_id,
                "remote_profile": profile_name,
                "remote_contract_sha256": remote_profile_contract_digest(config, profile_name, installed_upstreams()),
                "revision": revision,
                "identity": identity,
            }
        ),
        kind=UpstreamKind.SIE.value,
        local_identities=frozenset({identity}),
        model_contract_sha256=model_contract_digest(config),
        outputs=outputs,
        expires_at=(now or datetime.now(UTC)) + timedelta(seconds=remaining_s),
    )


def remote_admission(
    config: ModelConfig, *, wait: bool = True, now: datetime | None = None
) -> NumericalAdmission | str:
    """Return the admission of the model's remote profile on this process's upstream, or a refusal reason."""
    routing = config.routing
    if routing is None or routing.policy not in {"fallback", "threshold"} or routing.fallback_profile is None:
        return "model does not declare a hybrid remote profile"
    upstream_name = config.resolve_profile(routing.fallback_profile).loadtime.get("upstream")
    upstream = installed_upstreams().get(upstream_name) if isinstance(upstream_name, str) else None
    if upstream is None:
        return "hybrid remote profile names no installed upstream"
    if upstream.kind is UpstreamKind.SIE:
        return sie_admission(config, wait=wait, now=now)
    return openai_admission(config, now=now)


def openai_equivalence_refusal(
    config: ModelConfig,
    *,
    device: str,
    engine_config: EngineConfig | None = None,
    now: datetime | None = None,
) -> str | None:
    """Return a fixed refusal reason, or ``None`` when current evidence covers this process."""
    if device == "cuda" or (device.startswith("cuda:") and not device.partition(":")[2].isdigit()):
        return "hybrid execution device is ambiguous"
    admission = openai_admission(config, now=now)
    if isinstance(admission, str):
        return admission
    identity = local_profile_identity(config, "default", device=device, engine_config=engine_config)
    if identity is None:
        return "hybrid local execution cannot be identified"
    if identity not in admission.local_identities:
        return "hybrid equivalence record does not cover this execution identity"
    return None


def hybrid_request_refusal(
    config: ModelConfig, request_options: Mapping[str, object] | None, *, instruction: str | None = None
) -> str | None:
    """Keep valid local numerical overrides and request instructions off a bridge measured at defaults."""
    if instruction is not None:
        return "hybrid request sets an instruction"
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


def bridged_item_refusal(
    config: ModelConfig,
    admission: NumericalAdmission,
    *,
    token: str | None,
    outputs: Collection[str],
    request_options: Mapping[str, object] | None,
    instruction: str | None = None,
) -> str | None:
    """Refuse a bridged item unless it names the current admission and stays within it."""
    if token != admission.sha256:
        return "bridged item names a different numerical admission"
    if not outputs or not set(outputs) <= admission.outputs:
        return "bridged item requests outputs outside its numerical admission"
    return hybrid_request_refusal(config, request_options, instruction=instruction)
