"""The planner's report: what each placement costs and what would make it cheaper."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from sie_placement.schema import (
    SCHEMA_VERSION,
    Cell,
    ContainerShape,
    PlanIdentity,
    PlanLane,
    ServiceLevel,
    SourceRef,
    Warmth,
)

REPORT_KIND = "placement-report"


class IdentityStatus(StrEnum):
    PLACED = "placed"
    CO_PACKED = "co-packed"
    KEPT = "kept"
    UNPLACED = "unplaced"


class UnplacedReason(StrEnum):
    NO_RECEIPT = "no-receipt"
    WARM_BOUND_MISSED = "warm-bound-missed"
    WARM_BOUND_UNDECLARED = "warm-bound-undeclared"
    CURRENT_UNMEASURED = "current-unmeasured"
    CURRENT_UNPRICED = "current-unpriced"


class ComeAliveBasis(StrEnum):
    WARM_FLOOR = "warm-floor"
    BOOT_EVIDENCE = "boot-evidence"
    BORROWED_BOOT_EVIDENCE = "borrowed-boot-evidence"


class Measurement(StrEnum):
    RECEIPT = "receipt"
    BOOT_EVIDENCE = "boot-evidence"
    CO_RESIDENCY_RECEIPT = "co-residency-receipt"


class RequestReason(StrEnum):
    UNMEASURED = "unmeasured"
    REFUSED = "refused"
    FAILED = "failed"


class RefusedRecord(StrEnum):
    RECEIPT = "receipt"
    BOOT_EVIDENCE = "boot-evidence"
    CO_RESIDENCY_RECEIPT = "co-residency-receipt"


def _money(value: float) -> float:
    return round(value, 2) + 0.0


def _rate(value: float) -> float:
    return round(value, 4) + 0.0


def _compact(fields: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


@dataclass(frozen=True, slots=True, kw_only=True)
class Blocker:
    field: str
    bound: int | None
    measured: float | None
    identity: str | None = None

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {"identity": self.identity, "field": self.field, "bound": self.bound, "measured": self.measured}
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Option:
    cell: Cell
    container_shape: ContainerShape
    concurrency: int
    warmth: Warmth
    monthly_cost: float
    come_alive_s: float
    come_alive_basis: ComeAliveBasis
    receipt: SourceRef
    blocked_by: tuple[Blocker, ...] = ()
    saving: float | None = None
    lane: str | None = None

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "cell": self.cell.to_dict(),
                "containerShape": self.container_shape.to_dict(),
                "concurrency": self.concurrency,
                "lane": self.lane,
                "warmth": self.warmth.value,
                "monthlyCost": _money(self.monthly_cost),
                "saving": None if self.saving is None else _money(self.saving),
                "comeAliveS": self.come_alive_s,
                "comeAliveBasis": self.come_alive_basis.value,
                "blockedBy": [blocker.to_dict() for blocker in self.blocked_by] or None,
                "receipt": self.receipt.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityReport:
    identity: str
    status: IdentityStatus
    service_level: ServiceLevel
    demand_requests: int
    lane: str | None = None
    monthly_cost: float | None = None
    chosen: Option | None = None
    joined: Option | None = None
    peak_containers: int | None = None
    next_cheaper: Option | None = None
    current: PlanIdentity | None = None
    unplaced_reason: UnplacedReason | None = None
    blocked_by: tuple[Blocker, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "identity": self.identity,
                "status": self.status.value,
                "serviceLevel": self.service_level.to_dict(),
                "demandRequests": self.demand_requests,
                "lane": self.lane,
                "monthlyCost": None if self.monthly_cost is None else _money(self.monthly_cost),
                "blockedBy": [blocker.to_dict() for blocker in self.blocked_by] or None,
                "chosen": None if self.chosen is None else self.chosen.to_dict(),
                "joined": None if self.joined is None else self.joined.to_dict(),
                "peakContainers": self.peak_containers,
                "nextCheaper": None if self.next_cheaper is None else self.next_cheaper.to_dict(),
                "unplacedReason": None if self.unplaced_reason is None else self.unplaced_reason.value,
                "current": None if self.current is None else self.current.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LaneReport:
    name: str
    identities: tuple[str, ...]
    hourly_cost: float | None = None
    monthly_cost: float | None = None
    floor_monthly_cost: float | None = None
    envelope_max_monthly_cost: float | None = None
    unpriced_reason: str | None = None
    unpriced_identities: tuple[str, ...] = ()
    co_residency_receipt: SourceRef | None = None

    @property
    def unpriced(self) -> bool:
        return self.unpriced_reason is not None or bool(self.unpriced_identities)

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "name": self.name,
                "identities": list(self.identities),
                "hourlyCost": None if self.hourly_cost is None else _rate(self.hourly_cost),
                "monthlyCost": None if self.monthly_cost is None else _money(self.monthly_cost),
                "floorMonthlyCost": None if self.floor_monthly_cost is None else _money(self.floor_monthly_cost),
                "envelopeMaxMonthlyCost": (
                    None if self.envelope_max_monthly_cost is None else _money(self.envelope_max_monthly_cost)
                ),
                "unpricedReason": self.unpriced_reason,
                "unpricedIdentities": list(self.unpriced_identities) or None,
                "coResidencyReceipt": None
                if self.co_residency_receipt is None
                else self.co_residency_receipt.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedChartLane:
    queue_pool: str
    machine_profile: str
    bundle: str
    lanes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "queuePool": self.queue_pool,
            "machineProfile": self.machine_profile,
            "bundle": self.bundle,
            "lanes": list(self.lanes),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementRequest:
    identity: str
    cell: Cell
    measure: tuple[Measurement, ...]
    warmth: Warmth
    reason: RequestReason
    cost_bound: float
    saving_bound: float | None
    container_shape: ContainerShape | None = None
    concurrency: int | None = None
    lane: str | None = None
    co_residents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "identity": self.identity,
                "cell": self.cell.to_dict(),
                "containerShape": None if self.container_shape is None else self.container_shape.to_dict(),
                "concurrency": self.concurrency,
                "lane": self.lane,
                "coResidents": list(self.co_residents) or None,
                "measure": [measurement.value for measurement in self.measure],
                "warmth": self.warmth.value,
                "reason": self.reason.value,
                "costBound": _money(self.cost_bound),
                "savingBound": None if self.saving_bound is None else _money(self.saving_bound),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Lever:
    identity: str
    cell: Cell
    container_shape: ContainerShape
    warmth: Warmth
    come_alive_s: float
    come_alive_basis: ComeAliveBasis
    bound_s: int
    monthly_cost: float
    saving_under_bound: float
    permitted_bound_s: int
    saving_at_permitted_bound: float

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "cell": self.cell.to_dict(),
            "containerShape": self.container_shape.to_dict(),
            "warmth": self.warmth.value,
            "comeAliveS": self.come_alive_s,
            "comeAliveBasis": self.come_alive_basis.value,
            "comeAliveBoundS": self.bound_s,
            "monthlyCost": _money(self.monthly_cost),
            "savingUnderBound": _money(self.saving_under_bound),
            "permittedBoundS": self.permitted_bound_s,
            "savingAtPermittedBound": _money(self.saving_at_permitted_bound),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Refusal:
    record: RefusedRecord
    identity: str | None
    cell: Cell
    source: SourceRef
    reason: str
    residents: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "record": self.record.value,
                "identity": self.identity,
                "residents": list(self.residents) or None,
                "cell": self.cell.to_dict(),
                "source": self.source.to_dict(),
                "reason": self.reason,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CoPackingMove:
    identity: str
    from_lane: str
    from_warmth: Warmth
    to_lane: str
    co_residents: tuple[str, ...]
    receipt: SourceRef
    gpu_memory_headroom_mib: int
    saving: float
    blocked_by: tuple[Blocker, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "identity": self.identity,
                "fromLane": self.from_lane,
                "fromWarmth": self.from_warmth.value,
                "toLane": self.to_lane,
                "coResidents": list(self.co_residents),
                "receipt": self.receipt.to_dict(),
                "gpuMemoryHeadroomMiB": self.gpu_memory_headroom_mib,
                "saving": _money(self.saving),
                "blockedBy": [blocker.to_dict() for blocker in self.blocked_by] or None,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IdentityChange:
    identity: str
    before: PlanIdentity | None
    after: PlanIdentity | None

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "identity": self.identity,
                "before": None if self.before is None else self.before.to_dict(),
                "after": None if self.after is None else self.after.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LaneChange:
    name: str
    before: PlanLane | None
    after: PlanLane | None

    def to_dict(self) -> dict[str, object]:
        return _compact(
            {
                "name": self.name,
                "before": None if self.before is None else self.before.to_dict(),
                "after": None if self.after is None else self.after.to_dict(),
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    placement: str
    currency: str
    parameters: Mapping[str, object]
    identities: tuple[IdentityReport, ...]
    lanes: tuple[LaneReport, ...]
    shared_chart_lanes: tuple[SharedChartLane, ...]
    measurement_requests: tuple[MeasurementRequest, ...]
    come_alive_levers: tuple[Lever, ...]
    queue_wait_unattained: tuple[str, ...]
    refusals: tuple[Refusal, ...]
    identity_changes: tuple[IdentityChange, ...]
    lane_changes: tuple[LaneChange, ...]
    co_packing_moves: tuple[CoPackingMove, ...] = ()
    blocked_co_packing_moves: tuple[CoPackingMove, ...] = ()
    warm_bound_undeclared: tuple[str, ...] = ()

    @property
    def monthly_cost(self) -> float:
        return sum(lane.monthly_cost for lane in self.lanes if lane.monthly_cost is not None)

    @property
    def envelope_max_monthly_cost(self) -> float:
        return sum(lane.envelope_max_monthly_cost for lane in self.lanes if lane.envelope_max_monthly_cost is not None)

    @property
    def unpriced_lanes(self) -> tuple[str, ...]:
        return tuple(lane.name for lane in self.lanes if lane.unpriced)

    @property
    def unchanged(self) -> bool:
        return not self.identity_changes and not self.lane_changes

    def identity(self, identity: str) -> IdentityReport:
        for entry in self.identities:
            if entry.identity == identity:
                return entry
        raise KeyError(identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": REPORT_KIND,
            "schemaVersion": SCHEMA_VERSION,
            "placement": self.placement,
            "currency": self.currency,
            "parameters": dict(self.parameters),
            "totals": {
                "monthlyCost": _money(self.monthly_cost),
                "envelopeMaxMonthlyCost": _money(self.envelope_max_monthly_cost),
                "unpricedLanes": list(self.unpriced_lanes),
            },
            "identities": [entry.to_dict() for entry in self.identities],
            "lanes": [lane.to_dict() for lane in self.lanes],
            "sharedChartLanes": [shared.to_dict() for shared in self.shared_chart_lanes],
            "coPackingMoves": [move.to_dict() for move in self.co_packing_moves],
            "blockedCoPackingMoves": [move.to_dict() for move in self.blocked_co_packing_moves],
            "measurementRequests": [request.to_dict() for request in self.measurement_requests],
            "comeAliveLevers": [lever.to_dict() for lever in self.come_alive_levers],
            "queueWaitUnattained": list(self.queue_wait_unattained),
            "warmBoundUndeclared": list(self.warm_bound_undeclared),
            "refusals": [refusal.to_dict() for refusal in self.refusals],
            "changes": {
                "identities": [change.to_dict() for change in self.identity_changes],
                "lanes": [change.to_dict() for change in self.lane_changes],
            },
        }


def render(document: Mapping[str, object]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"
