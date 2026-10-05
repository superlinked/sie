"""Bounded, versioned evidence from two local runs and one remote run.

Records are operator artifacts, never caller assertions. They contain hashes
and measured errors, not request bodies, vectors or credentials. A record binds
one local execution identity, not the process that was measured, so every
process reporting that identity is covered. This module does not admit hybrid
routing; the deployment/runtime gate consumes the record.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, UpstreamKind

ProbeCategory = Literal[
    "short",
    "long",
    "boundary_before",
    "boundary_after",
    "query_prefix",
    "query_default",
    "empty_prefix",
    "document_prefix",
    "score_scale",
]
ProbeOperation = Literal["encode", "score"]
ProbeOutput = Literal["dense", "sparse", "multivector", "score"]
ProbeOutcome = Literal["ok", "invalid_input", "input_too_long", "shape_mismatch"]
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_MAX_VALUES = 100_000_000
_MAX_TOKENS = 100_000
_MAX_RECORD_BYTES = 512 << 10
_MAX_EVIDENCE_BYTES = 8 << 20
_CATEGORIES = frozenset(
    {
        "short",
        "long",
        "boundary_before",
        "boundary_after",
        "query_prefix",
        "query_default",
        "empty_prefix",
        "document_prefix",
    }
)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_equivalence_bytes(path: str | Path, *, max_bytes: int) -> bytes:
    """Read bounded operator evidence without blocking on a device or FIFO."""
    if type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_EVIDENCE_BYTES:
        raise ValueError("equivalence read limit is out of range")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("equivalence record must be a regular file")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("equivalence record exceeds the byte limit")
    return data


def read_equivalence_record(path: str | Path) -> EquivalenceRecord:
    return EquivalenceRecord.model_validate_json(read_equivalence_bytes(path, max_bytes=_MAX_RECORD_BYTES))


def model_contract_digest(config: ModelConfig) -> str:
    """Bind serving settings, independent of the policy enabled after measurement."""
    # A probe runs before hybrid admission. Changing only the dispatch policy
    # must not invalidate the very evidence needed to enable that policy.
    return canonical_digest(config.model_dump(mode="json", exclude={"routing"}))


def upstream_contract_digest(upstream: Upstream) -> str:
    """Bind endpoint, tenant credential reference and operator request transforms."""
    return canonical_digest(upstream.model_dump(mode="json", exclude={"rate_cap", "breaker", "equivalence"}))


def remote_profile_contract_digest(
    config: ModelConfig, profile_name: str, upstreams: Mapping[str, Upstream]
) -> str | None:
    """Report the server's actual operator-bound endpoint/model contract, never credentials."""
    kinds = {
        "sie_server.adapters.remote.sie:SieUpstreamAdapter": UpstreamKind.SIE,
        "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter": UpstreamKind.OPENAI,
    }
    try:
        profile = config.resolve_profile(profile_name)
        name = profile.loadtime.get("upstream")
        upstream = upstreams.get(name) if isinstance(name, str) else None
        if upstream is None or kinds.get(profile.adapter_path) != upstream.kind:
            return None
        return canonical_digest(
            {
                "version": 1,
                "model_contract": model_contract_digest(config),
                "profile": profile_name,
                "upstream_contract": upstream_contract_digest(upstream),
            }
        )
    except (TypeError, ValueError, RecursionError):
        return None


class ErrorMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    noise_floor: float = Field(ge=0, allow_inf_nan=False)
    remote_error: float = Field(ge=0, allow_inf_nan=False)
    values: int = Field(gt=0, le=_MAX_VALUES)

    @property
    def passed(self) -> bool:
        return self.remote_error <= self.noise_floor


def measure_values(first: np.ndarray, second: np.ndarray, remote: np.ndarray) -> ErrorMeasurement:
    """Compare absolute error to the actual two-run local noise floor.

    There is no caller-selected tolerance. The remote run must be within the
    measured floor of both local observations. Layout changes fail separately.
    """
    if first.shape != second.shape or first.shape != remote.shape or first.size == 0 or first.size > _MAX_VALUES:
        raise ValueError("probe output layout differs or exceeds the limit")
    if any(
        value.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
        for value in (first, second, remote)
    ):
        raise ValueError("probe output must use a supported floating precision")
    arrays = [np.asarray(value, dtype=np.float64) for value in (first, second, remote)]
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("probe output is not finite")
    a, b, r = arrays
    return ErrorMeasurement(
        noise_floor=float(np.max(np.abs(a - b))),
        remote_error=float(max(np.max(np.abs(a - r)), np.max(np.abs(b - r)))),
        values=int(a.size),
    )


class ProbeCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    operation: ProbeOperation
    category: ProbeCategory
    input_sha256: str = Field(pattern=_HASH_PATTERN)
    token_counts: tuple[int, ...] = Field(min_length=1, max_length=32)
    outcomes: tuple[ProbeOutcome, ProbeOutcome, ProbeOutcome]
    measurements: dict[ProbeOutput, ErrorMeasurement] = Field(max_length=4)

    @model_validator(mode="after")
    def coherent(self) -> ProbeCase:
        if any(value < 0 or value > _MAX_TOKENS for value in self.token_counts):
            raise ValueError("probe token count is out of range")
        if self.outcomes == ("ok", "ok", "ok") and not self.measurements:
            raise ValueError("successful runs require measured evidence")
        if self.outcomes != ("ok", "ok", "ok") and self.measurements:
            raise ValueError("refused or incompatible outputs cannot report measured errors")
        return self

    @property
    def passed(self) -> bool:
        if self.outcomes == ("ok", "ok", "ok"):
            return all(value.passed for value in self.measurements.values())
        return len(set(self.outcomes)) == 1 and self.outcomes[0] in ("invalid_input", "input_too_long")


class EquivalenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[3] = 3
    measured_at: AwareDatetime
    upstream_name: str = Field(min_length=1, max_length=128)
    upstream_model: str = Field(min_length=1, max_length=256)
    upstream_contract_sha256: str = Field(pattern=_HASH_PATTERN)
    model_contract_sha256: str = Field(pattern=_HASH_PATTERN)
    probe_sources_sha256: str = Field(pattern=_HASH_PATTERN)
    remote_contract_sha256: str = Field(pattern=_HASH_PATTERN)
    remote_execution_sha256: str = Field(pattern=_HASH_PATTERN)
    local_observation_sha256: str = Field(pattern=_HASH_PATTERN)
    runtime_options_sha256: str = Field(pattern=_HASH_PATTERN)
    output_dtype: Literal["float32"]
    local_identity: str = Field(pattern=r"^v[12]:sha256:[0-9a-f]{64}$")
    model: str = Field(min_length=1, max_length=256)
    local_profile: str = "default"
    remote_profile: str = Field(min_length=1, max_length=128)
    context_length: int = Field(gt=1, le=100_000)
    outputs: frozenset[ProbeOutput] = Field(min_length=1, max_length=4)
    cases: tuple[ProbeCase, ...] = Field(min_length=8, max_length=20)

    @model_validator(mode="after")
    def complete_suite(self) -> EquivalenceRecord:
        if self.local_profile != "default" or self.remote_profile == "default":
            raise ValueError("probe must compare the local default to an explicit remote profile")
        if len({(case.operation, case.category) for case in self.cases}) != len(self.cases):
            raise ValueError("probe cases must not repeat")
        operations = {"score" if output == "score" else "encode" for output in self.outputs}
        if {case.operation for case in self.cases} != operations:
            raise ValueError("probe operations do not cover the declared outputs")
        for operation in operations:
            selected = [case for case in self.cases if case.operation == operation]
            required = _CATEGORIES | ({"score_scale"} if operation == "score" else set())
            if {case.category for case in selected} != required:
                raise ValueError("probe suite must cover lengths, truncation and both instruction prefixes")
            expected = {output for output in self.outputs if (output == "score") == (operation == "score")}
            for case in selected:
                if case.outcomes == ("ok", "ok", "ok") and set(case.measurements) != expected:
                    raise ValueError("every successful probe must measure every declared output")
                if case.category == "boundary_before" and max(case.token_counts) > self.context_length:
                    raise ValueError("before-boundary probe must fit the context")
                if case.category == "boundary_after" and max(case.token_counts) <= self.context_length:
                    raise ValueError("after-boundary probe must cross the context")
        return self

    @property
    def passed(self) -> bool:
        # A set of matching refusals does not establish numerical equivalence.
        return all(case.passed for case in self.cases) and all(
            case.outcomes == ("ok", "ok", "ok")
            for case in self.cases
            if case.category in ("short", "long", "score_scale", "query_default", "empty_prefix")
        )

    def is_fresh(self, *, max_age_s: int, now: datetime | None = None) -> bool:
        observed = now or datetime.now(UTC)
        if observed.tzinfo is None or observed.utcoffset() is None:
            return False
        age = observed - self.measured_at
        return max_age_s > 0 and timedelta(0) <= age <= timedelta(seconds=max_age_s)
