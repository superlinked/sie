"""Evidence for one remote profile across local execution identities.

A bundle holds one record per local execution identity, all measuring the same
comparison contract. Parsing grants no routing authority. Failed and expired
records remain diagnostic evidence, never admission proof.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from sie_server.config.equivalence import EquivalenceRecord, canonical_digest, read_equivalence_bytes

MAX_FLEET_RECORDS = 256
MAX_FLEET_BYTES = 8 << 20
MAX_EVIDENCE_AGE_S = 604_800
_IDENTITY_FIELDS = {"measured_at", "local_identity", "local_observation_sha256", "cases"}


def _canonical_record(record: EquivalenceRecord) -> dict[str, Any]:
    data = record.model_dump(mode="json")
    data["outputs"] = sorted(record.outputs)
    data["measured_at"] = record.measured_at.astimezone(UTC).isoformat()
    data["cases"] = sorted(data["cases"], key=lambda case: (case["operation"], case["category"]))
    return data


def equivalence_record_digest(record: EquivalenceRecord) -> str:
    """Bind the entire measurement, including failures, identity and timestamp."""
    return canonical_digest(_canonical_record(record))


def _comparison_contract(record: EquivalenceRecord) -> dict[str, Any]:
    data = _canonical_record(record)
    contract = {key: value for key, value in data.items() if key not in _IDENTITY_FIELDS}
    contract["cases"] = [
        {key: case[key] for key in ("operation", "category", "input_sha256", "token_counts")} for case in data["cases"]
    ]
    return contract


class FleetEquivalenceRecord(BaseModel):
    """One comparison contract, measured once for each listed execution identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[2] = 2
    records: tuple[EquivalenceRecord, ...] = Field(min_length=1, max_length=MAX_FLEET_RECORDS)

    @model_validator(mode="after")
    def coherent(self) -> FleetEquivalenceRecord:
        if len({record.local_identity for record in self.records}) != len(self.records):
            raise ValueError("fleet evidence must not repeat an execution identity")
        contract = _comparison_contract(self.records[0])
        if any(_comparison_contract(record) != contract for record in self.records[1:]):
            raise ValueError("fleet evidence must measure the same comparison contract")
        return self

    @property
    def passed(self) -> bool:
        return all(record.passed for record in self.records)

    @property
    def record_digests(self) -> dict[str, str]:
        return {record.local_identity: equivalence_record_digest(record) for record in self.records}

    @property
    def digest(self) -> str:
        return canonical_digest({"version": self.version, "records": self.record_digests})

    def is_fresh(self, *, max_age_s: int, now: datetime | None = None) -> bool:
        if type(max_age_s) is not int or not 1 <= max_age_s <= MAX_EVIDENCE_AGE_S:
            return False
        observed = now or datetime.now(UTC)
        return all(record.is_fresh(max_age_s=max_age_s, now=observed) for record in self.records)


_EVIDENCE = TypeAdapter(FleetEquivalenceRecord | EquivalenceRecord)


def read_fleet_equivalence_record(path: str | Path) -> FleetEquivalenceRecord:
    """Read a bounded regular operator artifact, including failed measurements."""
    return FleetEquivalenceRecord.model_validate_json(read_equivalence_bytes(path, max_bytes=MAX_FLEET_BYTES))


def read_equivalence_evidence(path: str | Path) -> tuple[EquivalenceRecord, ...]:
    """Read one record, or a bundle of records, from a bounded regular file."""
    evidence = _EVIDENCE.validate_json(read_equivalence_bytes(path, max_bytes=MAX_FLEET_BYTES))
    return evidence.records if isinstance(evidence, FleetEquivalenceRecord) else (evidence,)
