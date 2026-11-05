"""Measured-only placement: the cheapest measured cell and warmth that meet each identity's service level."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from sie_placement.report import (
    Blocker,
    CapacityScope,
    CapacityUse,
    ComeAliveBasis,
    IdentityChange,
    IdentityReport,
    IdentityStatus,
    LaneChange,
    LaneReport,
    Lever,
    Measurement,
    MeasurementRequest,
    Option,
    Refusal,
    RefusedRecord,
    Report,
    RequestReason,
    Scarcity,
    ScarcityReason,
    SharedChartLane,
    UnplacedReason,
)
from sie_placement.schema import (
    BOOT_WARMTHS,
    WARM_FLOOR_WARMTHS,
    BootEvidence,
    BootOutcome,
    Capacity,
    Cell,
    ContainerShape,
    Demand,
    Envelope,
    MachineProfilePrice,
    Operation,
    Plan,
    PlanIdentity,
    PlanLane,
    PriceList,
    Receipt,
    SchemaError,
    ServiceLevel,
    ServiceLevels,
    SourceRef,
    Warmth,
)

SECONDS_PER_HOUR = 3600
HOURS_PER_MONTH = 730.0
PINNED_FLOOR = 1
BOOT_PERCENTILE = 95
BOOT_PERCENTILE_MIN_ROWS = 20
PLACEMENT_FAILURE_RATE_LIMIT = 0.05
PLACEMENT_FAILURE_FIELD = "placementFailureRate"
MEMORY_HEADROOM_FRACTION = 0.8
COLD_WARMTHS = (Warmth.COLD_WITH_BOUND, Warmth.SNAPSHOT_COLD)

_SLUG = re.compile(r"[^a-z0-9]+")
_LEVEL_FIELDS = {
    "comeAliveP95S": "come_alive_p95_s",
    "warmP99Ms": "warm_p99_ms",
    "ttftP99Ms": "ttft_p99_ms",
    "itlP99Ms": "itl_p99_ms",
}


class PlannerInputError(ValueError):
    pass


def _require_name(name: str, value: str) -> None:
    if not value or any(character.isspace() for character in value):
        raise SchemaError(name, f"must be a non-empty string without whitespace, got {value!r}")


def _require_minimum(name: str, value: float, minimum: int) -> None:
    if value < minimum:
        raise SchemaError(name, f"must be at least {minimum}, got {value!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannerParameters:
    placement: str
    max_containers: int
    fallback_floor: int
    cooldown_s: int
    buffer_containers: int = 0
    hold_budget_s: int = 0
    hours_per_month: float = HOURS_PER_MONTH
    queue_pool: str = "default"
    bundle: str = "default"
    bundles: Mapping[str, str] = field(default_factory=dict)
    unshared: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _require_name("placement", self.placement)
        _require_minimum("maxContainers", self.max_containers, 1)
        _require_minimum("fallbackFloor", self.fallback_floor, 0)
        _require_minimum("cooldownS", self.cooldown_s, 0)
        _require_minimum("bufferContainers", self.buffer_containers, 0)
        _require_minimum("holdBudgetS", self.hold_budget_s, 0)
        if not (math.isfinite(self.hours_per_month) and self.hours_per_month > 0):
            raise SchemaError("hoursPerMonth", f"must be greater than 0, got {self.hours_per_month!r}")
        _require_name("queuePool", self.queue_pool)
        _require_name("bundle", self.bundle)
        for identity, bundle in self.bundles.items():
            _require_name("bundles", identity)
            _require_name(f"bundles[{identity!r}]", bundle)
        for identity in self.unshared:
            _require_name("unshared", identity)

    @property
    def active_floor(self) -> int:
        return max(1, min(self.fallback_floor, self.max_containers))

    def bundle_of(self, identity: str) -> str:
        return self.bundles.get(identity, self.bundle)

    def to_dict(self) -> dict[str, object]:
        document: dict[str, object] = {
            "placement": self.placement,
            "maxContainers": self.max_containers,
            "fallbackFloor": self.fallback_floor,
            "cooldownS": self.cooldown_s,
            "bufferContainers": self.buffer_containers,
            "holdBudgetS": self.hold_budget_s,
            "hoursPerMonth": self.hours_per_month,
            "queuePool": self.queue_pool,
            "bundle": self.bundle,
        }
        if self.bundles:
            document["bundles"] = dict(sorted(self.bundles.items()))
        if self.unshared:
            document["unshared"] = sorted(self.unshared)
        return document


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannerResult:
    plan: Plan
    report: Report


class CapacityExceededError(PlannerInputError):
    def __init__(self, result: PlannerResult) -> None:
        self.result = result
        self.exceeded = tuple(use for use in result.report.capacity if use.exceeded)
        super().__init__("; ".join(use.describe() for use in self.exceeded))


def container_hourly_cost(price: MachineProfilePrice, shape: ContainerShape, region_multiplier: float) -> float:
    return region_multiplier * (
        price.accelerator_per_hour * shape.gpu_count
        + price.core_per_hour * shape.cpu_cores
        + price.memory_gib_per_hour * shape.memory_mib / 1024
    )


def envelope_for(warmth: Warmth, parameters: PlannerParameters) -> Envelope:
    floor_min = PINNED_FLOOR if warmth in WARM_FLOOR_WARMTHS else 0
    raised = floor_min + parameters.buffer_containers
    if raised and not floor_min:
        raised = max(raised, parameters.fallback_floor)
    return Envelope(floor_min=floor_min, floor_max=min(parameters.max_containers, raised))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Window:
    hours: int
    hours_per_month: float
    ceiling: int
    active_floor: int
    cooldown_s: int

    @property
    def scale(self) -> float:
        return self.hours_per_month / self.hours

    @staticmethod
    def needed(requests: int, requests_per_second: float) -> int:
        return math.ceil(requests / (requests_per_second * SECONDS_PER_HOUR))

    def above_floor_hours(self, requests: Sequence[int], requests_per_second: float, floor: int) -> float:
        above = sum(max(0, min(self.ceiling, self.needed(count, requests_per_second)) - floor) for count in requests)
        return above * self.scale

    def warm_container_hours(self, requests: Sequence[int], requests_per_second: float) -> float:
        return PINNED_FLOOR * self.hours_per_month + self.above_floor_hours(requests, requests_per_second, PINNED_FLOOR)

    def cold_container_hours(self, requests: Sequence[int], requests_per_second: float | None) -> float:
        seconds = 0.0
        for count in requests:
            tail = count * self.active_floor * self.cooldown_s
            if requests_per_second is None:
                seconds += min(self.active_floor * SECONDS_PER_HOUR, tail)
                continue
            containers = max(self.active_floor, min(self.ceiling, self.needed(count, requests_per_second)))
            seconds += min(containers * SECONDS_PER_HOUR, count / requests_per_second + tail)
        return seconds / SECONDS_PER_HOUR * self.scale


@dataclass(frozen=True, slots=True, kw_only=True)
class _Placement:
    wait_s: float | None
    measured: bool
    failure_rate: float
    borrowed: bool


_UNMEASURED_PLACEMENT = _Placement(wait_s=0.0, measured=False, failure_rate=0.0, borrowed=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Boot:
    boot_s: float | None
    borrowed: bool
    placement: _Placement = _UNMEASURED_PLACEMENT
    booted: bool = True

    @property
    def seconds(self) -> float | None:
        if self.boot_s is None or self.placement.wait_s is None:
            return None
        if not self.placement.measured:
            return self.boot_s
        return round(self.boot_s + self.placement.wait_s, 3)

    @property
    def placement_wait_s(self) -> float | None:
        return self.placement.wait_s if self.placement.measured else None

    @property
    def basis(self) -> ComeAliveBasis:
        borrowed = self.borrowed or (self.placement.measured and self.placement.borrowed)
        return ComeAliveBasis.BORROWED_BOOT_EVIDENCE if borrowed else ComeAliveBasis.BOOT_EVIDENCE

    @property
    def reason(self) -> RequestReason:
        if not self.booted or (self.boot_s is not None and self.placement.wait_s is None):
            return RequestReason.UNPLACED
        return RequestReason.FAILED


type _BootKey = tuple[str, Cell, ContainerShape, Warmth]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Option:
    receipt: Receipt
    order: str
    warmth: Warmth
    cost: float
    come_alive_s: float
    basis: ComeAliveBasis
    blockers: tuple[Blocker, ...]
    current: bool
    host: str | None = None
    host_owner: str | None = None
    fallbacks: tuple[Receipt, ...] = ()
    boot_s: float | None = None
    placement_wait_s: float | None = None

    @property
    def feasible(self) -> bool:
        return not self.blockers

    @property
    def cells(self) -> tuple[Cell, ...]:
        return (self.receipt.cell, *(receipt.cell for receipt in self.fallbacks))

    @property
    def pricing_receipt(self) -> Receipt:
        return min((self.receipt, *self.fallbacks), key=lambda receipt: receipt.throughput.requests_per_second)

    @property
    def rank(self) -> tuple[float, bool, float, int, str, str, str]:
        return (
            self.cost,
            not self.current,
            self.come_alive_s,
            len(self.fallbacks),
            self.order,
            self.warmth.value,
            self.host or "",
        )

    def summary(self, saving: float | None = None) -> Option:
        return Option(
            cell=self.receipt.cell,
            container_shape=self.receipt.container_shape,
            concurrency=self.receipt.concurrency,
            warmth=self.warmth,
            monthly_cost=self.cost,
            come_alive_s=self.come_alive_s,
            come_alive_basis=self.basis,
            receipt=self.receipt.source,
            blocked_by=self.blockers,
            saving=saving,
            lane=self.host,
            fallback_cells=self.cells[1:],
            placement_wait_s=self.placement_wait_s,
            co_residents=self.receipt.co_residents,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Memory:
    fits: bool | None
    need: int | None = None
    limit: float | None = None


_UNPACKABLE = _Memory(fits=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layout:
    plan: Plan
    names: dict[str, str]
    kept: dict[str, tuple[PlanIdentity, PlanLane]]
    receipts: dict[str, Receipt | None]
    owners: dict[str, str]


type _CoKey = tuple[str, Cell, ContainerShape, int]
type _Packing = tuple[dict[str, _Option], tuple[Measurement, ...], tuple[str, ...], tuple[tuple[str, Warmth], ...]]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Choice:
    identity: str
    level: ServiceLevel
    requests: tuple[int, ...]
    options: tuple[_Option, ...]
    chosen: _Option | None
    boot_requests: tuple[MeasurementRequest, ...]
    has_valid_receipt: bool
    held: float | None = None
    gaps: tuple[MeasurementRequest, ...] = ()
    unpriced: bool = False
    withdrawn: tuple[_Option, ...] = ()
    beside: frozenset[str] = frozenset()

    @property
    def baseline(self) -> float | None:
        return self.held if self.chosen is None else self.chosen.cost

    @property
    def unplaced_reason(self) -> UnplacedReason:
        if any(option.feasible for option in self.options):
            return UnplacedReason.CURRENT_UNPRICED if self.unpriced else UnplacedReason.CURRENT_UNMEASURED
        if not self.has_valid_receipt:
            return UnplacedReason.NO_RECEIPT
        if not _declares_warm_bound(self.level):
            return UnplacedReason.WARM_BOUND_UNDECLARED
        if any(_fields(option.blockers) == [PLACEMENT_FAILURE_FIELD] for option in self.options):
            return UnplacedReason.PLACEMENT_UNRELIABLE
        return UnplacedReason.WARM_BOUND_MISSED


def _canonical(document: Mapping[str, object]) -> str:
    return json.dumps(document, sort_keys=True)


def _slug(value: str) -> str:
    return _SLUG.sub("-", value.lower()).strip("-")


def _digest(value: str, width: int = 8) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:width]


def _refusal_reason(cell: Cell, prices: Mapping[str, MachineProfilePrice]) -> str | None:
    price = prices.get(cell.machine_profile)
    if price is None:
        return f"machine profile {cell.machine_profile!r} has no price"
    if price.gpu_product != cell.gpu_product:
        return (
            f"realized GPU product {cell.gpu_product!r} differs from {price.gpu_product!r}, "
            f"the product machine profile {cell.machine_profile!r} requests"
        )
    return None


def _lane_refusal(lane: PlanLane, prices: Mapping[str, MachineProfilePrice]) -> str | None:
    return next((reason for cell in lane.cells if (reason := _refusal_reason(cell, prices)) is not None), None)


def _fields(blockers: Iterable[Blocker]) -> list[str]:
    return [blocker.field for blocker in blockers]


def _incomparable_reason(level: ServiceLevel, receipt: Receipt) -> str | None:
    generate = receipt.request_shape.operation is Operation.GENERATE
    if level.is_generation and not generate:
        return "the receipt reports one latency and the service level declares ttftP99Ms and itlP99Ms"
    if level.warm_p99_ms is not None and generate:
        return "the receipt reports ttft and itl and the service level declares warmP99Ms"
    return None


def _warm_blockers(level: ServiceLevel, receipt: Receipt) -> tuple[Blocker, ...]:
    checks: list[tuple[str, int | None, float]] = []
    if receipt.latency is not None:
        checks.append(("warmP99Ms", level.warm_p99_ms, receipt.latency.p99_ms))
    if receipt.ttft is not None:
        checks.append(("ttftP99Ms", level.ttft_p99_ms, receipt.ttft.p99_ms))
    if receipt.itl is not None:
        checks.append(("itlP99Ms", level.itl_p99_ms, receipt.itl.p99_ms))
    return tuple(
        Blocker(field=name, bound=bound, measured=measured)
        for name, bound, measured in checks
        if bound is not None and measured > bound
    )


def _declares_warm_bound(level: ServiceLevel) -> bool:
    return level.warm_p99_ms is not None or level.is_generation


def _undeclared_blockers(level: ServiceLevel, receipt: Receipt) -> tuple[Blocker, ...]:
    if _declares_warm_bound(level):
        return ()
    measured = (("warmP99Ms", receipt.latency), ("ttftP99Ms", receipt.ttft), ("itlP99Ms", receipt.itl))
    return tuple(
        Blocker(field=name, bound=None, measured=values.p99_ms) for name, values in measured if values is not None
    )


def _come_alive_blocker(level: ServiceLevel, seconds: float) -> Blocker | None:
    bound = level.come_alive_p95_s
    if bound is None or (bound > 0 and seconds <= bound):
        return None
    return Blocker(field="comeAliveP95S", bound=bound, measured=seconds)


def _percentile(samples: Sequence[float | None]) -> float | None:
    succeeded = sorted(value for value in samples if value is not None)
    if len(samples) < BOOT_PERCENTILE_MIN_ROWS:
        return succeeded[-1] if len(succeeded) == len(samples) else None
    rank = (len(samples) * BOOT_PERCENTILE + 99) // 100
    return succeeded[rank - 1] if rank <= len(succeeded) else None


def _boot_sample(record: BootEvidence) -> float | None:
    if record.outcome is not BootOutcome.SUCCEEDED or record.boot_seconds is None:
        return None
    return float(record.boot_seconds)


def _placement_index(records: Iterable[BootEvidence]) -> dict[str, _Placement]:
    grouped: defaultdict[str, list[BootEvidence]] = defaultdict(list)
    for record in records:
        if record.warmth in COLD_WARMTHS:
            grouped[record.cell.gpu_product].append(record)
    index: dict[str, _Placement] = {}
    for product, group in grouped.items():
        own = [record for record in group if not record.borrowed]
        used = own or group
        samples = [
            None if record.outcome is BootOutcome.UNPLACED else float(record.placement_wait_s or 0.0) for record in used
        ]
        measured = any(record.outcome is BootOutcome.UNPLACED or record.placement_wait_s is not None for record in used)
        unplaced = sum(record.outcome is BootOutcome.UNPLACED for record in used)
        index[product] = _Placement(
            wait_s=_percentile(samples) if measured else 0.0,
            measured=measured,
            failure_rate=unplaced / len(used),
            borrowed=not own,
        )
    return index


def _boot_index(records: Iterable[BootEvidence], placements: Mapping[str, _Placement]) -> dict[_BootKey, _Boot]:
    grouped: defaultdict[_BootKey, list[BootEvidence]] = defaultdict(list)
    for record in records:
        grouped[(record.identity, record.cell, record.container_shape, record.warmth)].append(record)
    index: dict[_BootKey, _Boot] = {}
    for key, group in grouped.items():
        _, cell, _, warmth = key
        placement = placements.get(cell.gpu_product, _UNMEASURED_PLACEMENT)
        if warmth not in COLD_WARMTHS:
            placement = _UNMEASURED_PLACEMENT
        booted = [record for record in group if record.outcome is not BootOutcome.UNPLACED]
        own = [record for record in booted if not record.borrowed]
        used = own or booted
        if not used:
            index[key] = _Boot(
                boot_s=None, borrowed=all(record.borrowed for record in group), placement=placement, booted=False
            )
            continue
        index[key] = _Boot(
            boot_s=_percentile([_boot_sample(record) for record in used]), borrowed=not own, placement=placement
        )
    return index


def _hourly_requests(demand: Demand) -> dict[str, tuple[int, ...]]:
    hours: defaultdict[str, Counter[datetime]] = defaultdict(Counter)
    for series in demand.series:
        for hour in series.hours:
            hours[series.identity][hour.start] += hour.requests
    return {
        identity: tuple(count for _, count in sorted(counts.items()) if count) for identity, counts in hours.items()
    }


def _runs_on(receipt: Receipt, lane: PlanLane) -> bool:
    return (
        receipt.cell in lane.cells
        and lane.container_shape == receipt.container_shape
        and lane.concurrency == receipt.concurrency
    )


class _Planner:
    def __init__(
        self,
        service_levels: ServiceLevels,
        receipts: Iterable[Receipt],
        boot_evidence: Iterable[BootEvidence],
        price_list: PriceList,
        demand: Demand,
        current_plan: Plan | None,
        parameters: PlannerParameters,
        capacity: Capacity | None = None,
    ) -> None:
        receipts = tuple(receipts)
        boot_evidence = tuple(boot_evidence)
        placements = {placement.name: placement for placement in price_list.placements}
        placement = placements.get(parameters.placement)
        if placement is None:
            raise PlannerInputError(f"placement {parameters.placement!r} is not in the price list")
        if current_plan is not None and current_plan.placement != parameters.placement:
            raise PlannerInputError(
                f"the current plan is for placement {current_plan.placement!r}, not {parameters.placement!r}"
            )
        self.parameters = parameters
        self.capacity = capacity
        self.service_levels = service_levels
        self.receipt_records = receipts
        self.boot_records = boot_evidence
        self.demand = demand
        self.price_list = price_list
        self.multiplier = placement.region_multiplier
        self.prices = {profile.name: profile for profile in price_list.machine_profiles}
        self.levels = {entry.identity: service_levels.effective(entry.identity) for entry in service_levels.identities}
        self.window = _Window(
            hours=int((demand.end - demand.start).total_seconds()) // SECONDS_PER_HOUR,
            hours_per_month=parameters.hours_per_month,
            ceiling=parameters.max_containers,
            active_floor=parameters.active_floor,
            cooldown_s=parameters.cooldown_s,
        )
        self.requests = _hourly_requests(demand)
        self.current_plan = current_plan
        self.current: dict[str, tuple[PlanIdentity, PlanLane]] = {}
        self.current_lanes: dict[str, PlanLane] = {}
        self.anchors: dict[str, frozenset[str]] = {}
        if current_plan is not None:
            self.current_lanes = {lane.name: lane for lane in current_plan.lanes}
            self.current = {
                entry.identity: (entry, self.current_lanes[entry.lane]) for entry in current_plan.identities
            }
        self.current_pinned: defaultdict[str, set[str]] = defaultdict(set)
        self.current_lazy: defaultdict[str, set[str]] = defaultdict(set)
        if current_plan is not None:
            for entry in current_plan.identities:
                if entry.warmth is Warmth.PINNED:
                    self.current_pinned[entry.lane].add(entry.identity)
                elif entry.warmth is Warmth.LAZY_RESIDENT:
                    self.current_lazy[entry.lane].add(entry.identity)
        self.refusals: list[Refusal] = []
        self.refused_profiles: set[tuple[str, str]] = set()
        self.savings: dict[tuple[str, tuple[tuple[str, int], ...]], float] = {}
        self.co_receipts: defaultdict[_CoKey, list[tuple[str, Receipt]]] = defaultdict(list)
        self.excluded: set[tuple[str, str]] = set()
        self.gpu_memory_undeclared: set[str] = set()
        self.receipts = self._accept_receipts(receipts)
        self.placements = _placement_index(
            record for record in boot_evidence if _refusal_reason(record.cell, self.prices) is None
        )
        accepted = self._accept_boot_evidence(boot_evidence)
        self.boots = _boot_index(accepted, self.placements)
        self.footprints: dict[tuple[str, Cell], int] = {}
        self.working: dict[tuple[str, Cell, ContainerShape, int], int] = {}
        measured = [receipt for pairs in (*self.receipts.values(), *self.co_receipts.values()) for _, receipt in pairs]
        for receipt in measured:
            self._footprint(receipt.identity, receipt.cell, receipt.footprint_mib)
            if receipt.working_memory_mib is not None:
                key = (receipt.identity, receipt.cell, receipt.container_shape, receipt.concurrency)
                self.working[key] = max(receipt.working_memory_mib, self.working.get(key, 0))
        for record in accepted:
            self._footprint(record.identity, record.cell, record.footprint_mib)
        self.solitary = {identity for identity, level in self.levels.items() if level.is_generation}
        self.solitary.update(
            receipt.identity for receipt in measured if receipt.request_shape.operation is Operation.GENERATE
        )
        self.solitary.update(parameters.unshared)

    def _footprint(self, identity: str, cell: Cell, footprint: int | None) -> None:
        if footprint is not None:
            self.footprints[(identity, cell)] = max(footprint, self.footprints.get((identity, cell), 0))

    def _accept_receipts(self, receipts: Iterable[Receipt]) -> dict[str, list[tuple[str, Receipt]]]:
        accepted: defaultdict[str, list[tuple[str, Receipt]]] = defaultdict(list)
        for receipt in receipts:
            level = self.levels.get(receipt.identity)
            if level is None:
                continue
            reason = _refusal_reason(receipt.cell, self.prices) or _incomparable_reason(level, receipt)
            if reason is not None:
                self._refuse(RefusedRecord.RECEIPT, receipt.identity, receipt.cell, receipt.source, reason)
                if not receipt.co_residents:
                    self.refused_profiles.add((receipt.identity, receipt.cell.machine_profile))
                continue
            order = _canonical(receipt.to_dict())
            if receipt.co_residents:
                key = (receipt.identity, receipt.cell, receipt.container_shape, receipt.concurrency)
                self.co_receipts[key].append((order, receipt))
                continue
            accepted[receipt.identity].append((order, receipt))
        return accepted

    def _accept_boot_evidence(self, records: Iterable[BootEvidence]) -> list[BootEvidence]:
        accepted = []
        for record in records:
            if record.identity not in self.levels:
                continue
            reason = _refusal_reason(record.cell, self.prices)
            if reason is not None:
                self._refuse(RefusedRecord.BOOT_EVIDENCE, record.identity, record.cell, record.source, reason)
                continue
            accepted.append(record)
        return accepted

    def _refuse(self, record: RefusedRecord, identity: str, cell: Cell, source: SourceRef, reason: str) -> None:
        self.refusals.append(Refusal(record=record, identity=identity, cell=cell, source=source, reason=reason))

    def is_current(
        self,
        identity: str,
        receipt: Receipt,
        warmth: Warmth,
        fallbacks: tuple[Cell, ...] = (),
        beside: frozenset[str] | None = None,
    ) -> bool:
        current = self.current.get(identity)
        if current is None:
            return False
        entry, lane = current
        context = self.pinned_beside(identity) if beside is None else beside
        if warmth is Warmth.PINNED and not _stands_beside(receipt, context):
            return False
        return (
            entry.warmth is warmth
            and lane.queue_pool == self.parameters.queue_pool
            and lane.bundle == self.parameters.bundle_of(identity)
            and lane.cells == (receipt.cell, *fallbacks)
            and _runs_on(receipt, lane)
        )

    def pinned_beside(self, identity: str) -> frozenset[str]:
        current = self.current.get(identity)
        if current is None or current[0].warmth is not Warmth.PINNED:
            return frozenset()
        return frozenset(self.current_pinned[current[1].name] - {identity})

    def co_receipt(self, identity: str, lane: PlanLane, beside: frozenset[str]) -> tuple[str, Receipt] | None:
        found = [
            pair
            for pair in self.co_receipts.get((identity, lane.cell, lane.container_shape, lane.concurrency), [])
            if _stands_beside(pair[1], beside)
        ]
        return min(
            found,
            key=lambda pair: (
                len(pair[1].co_residents),
                -_p99(pair[1]),
                pair[1].throughput.requests_per_second,
                pair[0],
            ),
            default=None,
        )

    def hosts_solitary(self, identities: Iterable[str]) -> bool:
        return any(identity in self.solitary for identity in identities)

    def memory(self, lane: PlanLane, pinned: Iterable[str], lazy: Iterable[str]) -> _Memory:
        pinned, lazy = tuple(sorted(pinned)), tuple(sorted(lazy))
        price = self.prices.get(lane.cell.machine_profile)
        residents = (*pinned, *lazy)
        if not residents or price is None or lane.container_shape.gpu_count != 1 or lane.fallback_cells:
            return _UNPACKABLE
        if price.gpu_memory_mib is None:
            self.gpu_memory_undeclared.add(price.name)
            return _UNPACKABLE
        limit = price.gpu_memory_mib * MEMORY_HEADROOM_FRACTION
        config = (lane.cell, lane.container_shape, lane.concurrency)
        if any(
            (identity, lane.cell) not in self.footprints or (identity, *config) not in self.working
            for identity in residents
        ):
            return _Memory(fits=None, limit=limit)
        need = (
            sum(self.footprints[(identity, lane.cell)] for identity in pinned)
            + max((self.footprints[(identity, lane.cell)] for identity in lazy), default=0)
            + max(self.working[(identity, *config)] for identity in residents)
        )
        return _Memory(fits=need <= limit, need=need, limit=limit)

    def lane_memory(self, lane: PlanLane, residents: Mapping[str, Warmth]) -> _Memory:
        return self.memory(
            lane,
            pinned=(identity for identity, warmth in residents.items() if warmth is Warmth.PINNED),
            lazy=(identity for identity, warmth in residents.items() if warmth is Warmth.LAZY_RESIDENT),
        )

    def memory_gaps(self, lane: PlanLane, residents: Iterable[str]) -> tuple[str, ...]:
        config = (lane.cell, lane.container_shape, lane.concurrency)
        return tuple(
            identity
            for identity in sorted(set(residents))
            if (identity, lane.cell) not in self.footprints or (identity, *config) not in self.working
        )

    @staticmethod
    def footprint_request(
        identity: str, lane: PlanLane, warmth: Warmth, cost: float, saving: float | None = None
    ) -> MeasurementRequest:
        return MeasurementRequest(
            identity=identity,
            cell=lane.cell,
            container_shape=lane.container_shape,
            concurrency=lane.concurrency,
            lane=lane.name,
            measure=(Measurement.FOOTPRINT,),
            warmth=warmth,
            reason=RequestReason.UNMEASURED,
            cost_bound=cost,
            saving_bound=saving,
        )

    def hourly(self, receipt: Receipt) -> float:
        return container_hourly_cost(
            self.prices[receipt.cell.machine_profile], receipt.container_shape, self.multiplier
        )

    def lane_hourly(self, lane: PlanLane) -> float:
        return max(
            container_hourly_cost(self.prices[cell.machine_profile], lane.container_shape, self.multiplier)
            for cell in lane.cells
        )

    def tight(self, level: ServiceLevel) -> bool:
        bound = level.come_alive_p95_s
        return bound is not None and bound <= self.parameters.hold_budget_s

    def unreliable(self, gpu_product: str) -> bool:
        placement = self.placements.get(gpu_product)
        return placement is not None and placement.failure_rate > PLACEMENT_FAILURE_RATE_LIMIT

    def reliability_blockers(self, level: ServiceLevel, cells: Sequence[Cell]) -> tuple[Blocker, ...]:
        first = cells[0].gpu_product
        if not self.tight(level) or not self.unreliable(first):
            return ()
        if any(cell.gpu_product != first for cell in cells[1:]):
            return ()
        rate = self.placements[first].failure_rate
        return (Blocker(field=PLACEMENT_FAILURE_FIELD, bound=PLACEMENT_FAILURE_RATE_LIMIT, measured=rate),)

    @staticmethod
    def warm_blockers(level: ServiceLevel, receipt: Receipt, current_class: Cell | None) -> tuple[Blocker, ...]:
        warm = _warm_blockers(level, receipt)
        if receipt.cell != current_class:
            warm = (*warm, *_undeclared_blockers(level, receipt))
        return warm

    def current_class(self, identity: str) -> Cell | None:
        current = self.current.get(identity)
        return None if current is None else current[1].cell

    def keepable(self, identity: str) -> tuple[PlanIdentity, PlanLane] | None:
        current = self.current.get(identity)
        if current is None:
            return None
        lane = current[1]
        if lane.queue_pool != self.parameters.queue_pool or lane.bundle != self.parameters.bundle_of(identity):
            return None
        return current

    def staying_anchors(self, choices: Iterable[_Choice]) -> dict[str, frozenset[str]]:
        anchors: defaultdict[str, set[str]] = defaultdict(set)
        for choice in choices:
            current = self.current.get(choice.identity)
            if current is None or current[0].warmth is not Warmth.PINNED:
                continue
            if choice.chosen is None or choice.chosen.current:
                anchors[current[1].name].add(choice.identity)
        return {lane: frozenset(identities) for lane, identities in anchors.items()}

    def floor_paid(self, lane: str, identity: str) -> bool:
        return bool(self.anchors.get(lane, frozenset()) - {identity})

    def lazy_hosts(self, identity: str) -> tuple[PlanLane, ...]:
        current = self.current.get(identity)
        home = None if current is None else current[1].name
        lazy = current is not None and current[0].warmth is Warmth.LAZY_RESIDENT
        bundle = self.parameters.bundle_of(identity)
        return tuple(
            lane
            for name, lane in sorted(self.current_lanes.items())
            if (lazy if name == home else self.floor_paid(name, identity))
            and (identity, name) not in self.excluded
            and lane.envelope.floor_min >= PINNED_FLOOR
            and not lane.fallback_cells
            and not self.hosts_solitary((*self.current_pinned[name], *self.current_lazy[name]))
            and lane.queue_pool == self.parameters.queue_pool
            and lane.bundle == bundle
        )

    def resident_hours(self, lane: PlanLane, identity: str, requests: Sequence[int], rate: float | None) -> float:
        capped = self.within_ceiling(lane)
        floor = (
            0.0 if self.floor_paid(lane.name, identity) else capped.envelope.floor_min * self.parameters.hours_per_month
        )
        if rate is None:
            return floor
        window = replace(self.window, ceiling=capped.max_containers)
        return floor + window.above_floor_hours(requests, rate, capped.envelope.floor_min)

    def current_cost(self, identity: str, entry: PlanIdentity, lane: PlanLane) -> float | None:
        if _lane_refusal(lane, self.prices) is not None:
            return None
        hourly = self.lane_hourly(lane)
        requests = self.requests.get(identity, ())
        rate = max(
            (
                receipt.throughput.requests_per_second
                for _, receipt in self.receipts.get(identity, [])
                if _runs_on(receipt, lane)
            ),
            default=None,
        )
        if entry.warmth is Warmth.PINNED:
            hours = (
                PINNED_FLOOR * self.parameters.hours_per_month
                if rate is None
                else self.window.warm_container_hours(requests, rate)
            )
        elif entry.warmth is Warmth.LAZY_RESIDENT:
            hours = self.resident_hours(lane, identity, requests, rate)
        else:
            hours = self.window.cold_container_hours(requests, rate)
        return hourly * hours

    def current_gaps(
        self,
        identity: str,
        entry: PlanIdentity,
        lane: PlanLane,
        cost: float | None,
        beside: frozenset[str] | None = None,
    ) -> tuple[MeasurementRequest, ...]:
        if cost is None:
            return ()
        missing: list[tuple[Measurement, RequestReason]] = []
        beside = self.pinned_beside(identity) if beside is None else beside
        if beside:
            if self.co_receipt(identity, lane, beside) is None:
                missing.append((Measurement.CO_RESIDENCY_RECEIPT, RequestReason.UNMEASURED))
        elif self.slowest_receipt(identity, lane) is None:
            refused = (identity, lane.cell.machine_profile) in self.refused_profiles
            missing.append((Measurement.RECEIPT, RequestReason.REFUSED if refused else RequestReason.UNMEASURED))
        if entry.warmth in BOOT_WARMTHS and (identity, lane.cell, lane.container_shape, entry.warmth) not in self.boots:
            missing.append((Measurement.BOOT_EVIDENCE, RequestReason.UNMEASURED))
        requests = [
            MeasurementRequest(
                identity=identity,
                cell=lane.cell,
                container_shape=lane.container_shape,
                concurrency=lane.concurrency,
                lane=lane.name,
                measure=(measurement,),
                warmth=entry.warmth,
                reason=reason,
                cost_bound=cost,
                saving_bound=None,
                co_residents=tuple(sorted(beside)) if measurement is Measurement.CO_RESIDENCY_RECEIPT else (),
            )
            for measurement, reason in missing
        ]
        if entry.warmth is Warmth.LAZY_RESIDENT:
            pinned = self.current_pinned[lane.name] - {identity}
            lazy = self.current_lazy[lane.name] | {identity}
            if self.memory(lane, pinned=pinned, lazy=lazy).fits is None:
                requests.extend(
                    self.footprint_request(
                        resident, lane, Warmth.PINNED if resident in pinned else Warmth.LAZY_RESIDENT, cost
                    )
                    for resident in self.memory_gaps(lane, (*pinned, *lazy))
                )
        return tuple(requests)

    def choose(self, identity: str, beside: frozenset[str] | None = None) -> _Choice:
        beside = self.pinned_beside(identity) if beside is None else beside
        level = self.levels[identity]
        requests = self.requests.get(identity, ())
        options: list[_Option] = []
        withdrawn: list[_Option] = []
        boot_requests: dict[tuple[Cell, ContainerShape], MeasurementRequest] = {}
        load_requests: dict[tuple[str, str], MeasurementRequest] = {}
        receipts = self.receipts.get(identity, [])
        current_class = self.current_class(identity)
        hosts = () if identity in self.solitary else self.lazy_hosts(identity)
        for order, receipt in receipts:
            hourly = self.hourly(receipt)
            rate = receipt.throughput.requests_per_second
            blocked = (
                *self.warm_blockers(level, receipt, current_class),
                *self.reliability_blockers(level, (receipt.cell,)),
            )
            options.append(
                _Option(
                    receipt=receipt,
                    order=order,
                    warmth=Warmth.PINNED,
                    cost=hourly * self.window.warm_container_hours(requests, rate),
                    come_alive_s=0.0,
                    basis=ComeAliveBasis.WARM_FLOOR,
                    blockers=blocked,
                    current=self.is_current(identity, receipt, Warmth.PINNED, beside=beside),
                )
            )
            cold_cost = hourly * self.window.cold_container_hours(requests, rate)
            for warmth in COLD_WARMTHS:
                boot = self.boots.get((identity, receipt.cell, receipt.container_shape, warmth))
                if boot is None or boot.seconds is None:
                    if warmth is Warmth.COLD_WITH_BOUND and not blocked and level.come_alive_p95_s != 0:
                        key = (receipt.cell, receipt.container_shape)
                        known = boot_requests.get(key)
                        if known is None or cold_cost < known.cost_bound:
                            boot_requests[key] = MeasurementRequest(
                                identity=identity,
                                cell=receipt.cell,
                                container_shape=receipt.container_shape,
                                measure=(Measurement.BOOT_EVIDENCE,),
                                warmth=warmth,
                                reason=RequestReason.UNMEASURED if boot is None else boot.reason,
                                cost_bound=cold_cost,
                                saving_bound=None,
                            )
                    if boot is not None and boot.boot_s is not None and not blocked:
                        withdrawn.append(
                            _Option(
                                receipt=receipt,
                                order=order,
                                warmth=warmth,
                                cost=cold_cost,
                                come_alive_s=math.inf,
                                basis=boot.basis,
                                blockers=(Blocker(field="comeAliveP95S", bound=level.come_alive_p95_s, measured=None),),
                                current=False,
                                boot_s=boot.boot_s,
                            )
                        )
                    continue
                blocker = _come_alive_blocker(level, boot.seconds)
                options.append(
                    _Option(
                        receipt=receipt,
                        order=order,
                        warmth=warmth,
                        cost=cold_cost,
                        come_alive_s=boot.seconds,
                        basis=boot.basis,
                        blockers=blocked if blocker is None else (*blocked, blocker),
                        current=self.is_current(identity, receipt, warmth),
                        boot_s=boot.boot_s,
                        placement_wait_s=boot.placement_wait_s,
                    )
                )
            for host in hosts:
                if not _runs_on(receipt, host):
                    continue
                lazy_cost = hourly * self.resident_hours(host, identity, requests, rate)
                load = self.boots.get((identity, host.cell, host.container_shape, Warmth.LAZY_RESIDENT))
                pinned = self.current_pinned[host.name] - {identity}
                lazy = self.current_lazy[host.name] | {identity}
                memory = self.memory(host, pinned=pinned, lazy=lazy)
                if memory.fits is False:
                    continue
                lacking = () if memory.fits else self.memory_gaps(host, (*pinned, *lazy))
                missing = (
                    *(() if load is not None and load.seconds is not None else (Measurement.BOOT_EVIDENCE,)),
                    *((Measurement.FOOTPRINT,) if identity in lacking else ()),
                )
                if missing or lacking:
                    if blocked or level.come_alive_p95_s == 0:
                        continue
                    found = [
                        self.footprint_request(
                            resident, host, Warmth.PINNED if resident in pinned else Warmth.LAZY_RESIDENT, lazy_cost
                        )
                        for resident in lacking
                        if resident != identity
                    ]
                    if missing:
                        found.append(
                            MeasurementRequest(
                                identity=identity,
                                cell=host.cell,
                                container_shape=host.container_shape,
                                concurrency=host.concurrency,
                                lane=host.name,
                                measure=missing,
                                warmth=Warmth.LAZY_RESIDENT,
                                reason=(
                                    RequestReason.FAILED
                                    if load is not None and load.seconds is None
                                    else RequestReason.UNMEASURED
                                ),
                                cost_bound=lazy_cost,
                                saving_bound=None,
                            )
                        )
                    for request in found:
                        known = load_requests.get((host.name, request.identity))
                        if known is None or lazy_cost < known.cost_bound:
                            load_requests[(host.name, request.identity)] = request
                    continue
                if load is None or load.seconds is None:
                    continue
                blocker = _come_alive_blocker(level, load.seconds)
                options.append(
                    _Option(
                        receipt=receipt,
                        order=order,
                        warmth=Warmth.LAZY_RESIDENT,
                        cost=lazy_cost,
                        come_alive_s=load.seconds,
                        basis=load.basis,
                        blockers=blocked if blocker is None else (*blocked, blocker),
                        current=self.is_current(identity, receipt, Warmth.LAZY_RESIDENT)
                        and self.current[identity][1].name == host.name,
                        host=host.name,
                        boot_s=load.boot_s,
                    )
                )
        options.extend(self.beside_options(identity, level, requests, current_class, beside))
        if self.tight(level):
            options.extend(self.candidate_lists(identity, level, receipts, requests, current_class, beside))
        feasible = [option for option in options if option.feasible]
        chosen = min(feasible, key=lambda option: option.rank) if feasible else None
        held = None
        unpriced = False
        gaps: tuple[MeasurementRequest, ...] = ()
        current = self.keepable(identity)
        if current is not None and not any(option.current for option in options):
            entry, lane = current
            if not self.kept_blockers(identity, entry, lane, self.slowest_receipt(identity, lane)):
                cost = self.current_cost(identity, entry, lane)
                cheaper = [option for option in feasible if cost is not None and round(cost - option.cost, 2) > 0]
                chosen = min(cheaper, key=lambda option: option.rank) if cheaper else None
                if chosen is None:
                    held = cost
                    unpriced = cost is None
                    gaps = self.current_gaps(identity, entry, lane, cost, beside)
        return _Choice(
            identity=identity,
            level=level,
            requests=requests,
            options=tuple(options),
            chosen=chosen,
            boot_requests=_ungapped(identity, (*boot_requests.values(), *load_requests.values()), gaps),
            has_valid_receipt=bool(receipts),
            held=held,
            gaps=gaps,
            unpriced=unpriced,
            withdrawn=tuple(withdrawn),
            beside=beside,
        )

    def beside_options(
        self,
        identity: str,
        level: ServiceLevel,
        requests: Sequence[int],
        current_class: Cell | None,
        beside: frozenset[str],
    ) -> list[_Option]:
        current = self.current.get(identity)
        if current is None or current[0].warmth is not Warmth.PINNED:
            return []
        lane = current[1]
        if not beside and any(_runs_on(receipt, lane) for _, receipt in self.receipts.get(identity, [])):
            return []
        found = self.co_receipt(identity, lane, beside)
        if found is None or not self.is_current(identity, found[1], Warmth.PINNED, beside=beside):
            return []
        order, receipt = found
        return [
            _Option(
                receipt=receipt,
                order=order,
                warmth=Warmth.PINNED,
                cost=self.hourly(receipt)
                * self.window.warm_container_hours(requests, receipt.throughput.requests_per_second),
                come_alive_s=0.0,
                basis=ComeAliveBasis.WARM_FLOOR,
                blockers=(
                    *self.warm_blockers(level, receipt, current_class),
                    *self.reliability_blockers(level, lane.cells),
                ),
                current=True,
            )
        ]

    def candidate_lists(
        self,
        identity: str,
        level: ServiceLevel,
        receipts: Sequence[tuple[str, Receipt]],
        requests: Sequence[int],
        current_class: Cell | None,
        beside: frozenset[str],
    ) -> list[_Option]:
        lists = []
        for order, receipt in receipts:
            if not self.unreliable(receipt.cell.gpu_product) or self.warm_blockers(level, receipt, current_class):
                continue
            for second_order, second in receipts:
                if (
                    second.cell.gpu_product == receipt.cell.gpu_product
                    or self.hourly(second) < self.hourly(receipt)
                    or second.container_shape != receipt.container_shape
                    or second.concurrency != receipt.concurrency
                    or self.warm_blockers(level, second, current_class)
                ):
                    continue
                hourly = max(self.hourly(receipt), self.hourly(second))
                rate = min(receipt.throughput.requests_per_second, second.throughput.requests_per_second)
                lists.append(
                    _Option(
                        receipt=receipt,
                        fallbacks=(second,),
                        order=f"{order}\n{second_order}",
                        warmth=Warmth.PINNED,
                        cost=hourly * self.window.warm_container_hours(requests, rate),
                        come_alive_s=0.0,
                        basis=ComeAliveBasis.WARM_FLOOR,
                        blockers=(),
                        current=self.is_current(identity, receipt, Warmth.PINNED, (second.cell,), beside),
                    )
                )
        return lists

    def receipt_requests(self, choice: _Choice, baseline: float | None) -> list[MeasurementRequest]:
        measured = {receipt.cell.machine_profile for _, receipt in self.receipts.get(choice.identity, [])}
        measured.update(
            request.cell.machine_profile
            for request in choice.gaps
            if Measurement.RECEIPT in request.measure or Measurement.CO_RESIDENCY_RECEIPT in request.measure
        )
        bound = choice.level.come_alive_p95_s
        tight = self.tight(choice.level)
        requests = []
        for profile in sorted(self.price_list.machine_profiles, key=lambda price: price.name):
            if profile.name in measured or (tight and self.unreliable(profile.gpu_product)):
                continue
            accelerator = profile.accelerator_per_hour * self.multiplier
            candidates = [(accelerator * self.window.warm_container_hours((), 1.0), Warmth.PINNED)]
            if bound != 0:
                cold = accelerator * self.window.cold_container_hours(choice.requests, None)
                candidates.append((cold, Warmth.COLD_WITH_BOUND))
            cost, warmth = min(candidates, key=lambda candidate: (candidate[0], candidate[1] is not Warmth.PINNED))
            saving = None if baseline is None else baseline - cost
            if saving is not None and saving <= 0:
                continue
            requests.append(
                MeasurementRequest(
                    identity=choice.identity,
                    cell=Cell(machine_profile=profile.name, gpu_product=profile.gpu_product),
                    measure=(
                        (Measurement.RECEIPT,)
                        if warmth is Warmth.PINNED
                        else (Measurement.RECEIPT, Measurement.BOOT_EVIDENCE)
                    ),
                    warmth=warmth,
                    reason=(
                        RequestReason.REFUSED
                        if (choice.identity, profile.name) in self.refused_profiles
                        else RequestReason.UNMEASURED
                    ),
                    cost_bound=cost,
                    saving_bound=saving,
                )
            )
        return requests

    @staticmethod
    def boot_requests(choice: _Choice, baseline: float | None) -> list[MeasurementRequest]:
        if baseline is None:
            return []
        return [
            replace(request, saving_bound=baseline - request.cost_bound)
            for request in choice.boot_requests
            if baseline - request.cost_bound > 0
        ]

    def replanned_total(
        self,
        current: Plan,
        *,
        service_levels: ServiceLevels | None = None,
        boot_evidence: Iterable[BootEvidence] | None = None,
    ) -> float:
        planner = _Planner(
            self.service_levels if service_levels is None else service_levels,
            self.receipt_records,
            self.boot_records if boot_evidence is None else boot_evidence,
            self.price_list,
            self.demand,
            current,
            self.parameters,
            self.capacity,
        )
        report = planner.run(price_savings=False).report
        if any(use.exceeded for use in report.capacity):
            return math.inf
        return report.monthly_cost

    def loosened(self, identity: str, bounds: Mapping[str, int]) -> ServiceLevels:
        fields = {_LEVEL_FIELDS[name]: bound for name, bound in bounds.items()}
        return replace(
            self.service_levels,
            identities=tuple(
                replace(entry, level=replace(entry.level, **fields)) if entry.identity == identity else entry
                for entry in self.service_levels.identities
            ),
        )

    def saving_when_loosened(self, identity: str, bounds: Mapping[str, int], written: Plan, total: float) -> float:
        key = (identity, tuple(sorted(bounds.items())))
        if key not in self.savings:
            replanned = self.replanned_total(written, service_levels=self.loosened(identity, bounds))
            self.savings[key] = _saving(total, replanned)
        return self.savings[key]

    def loosening(self, blockers: Iterable[Blocker]) -> dict[str, int]:
        bounds: dict[str, int] = {}
        for blocker in blockers:
            if blocker.field == PLACEMENT_FAILURE_FIELD:
                name, bound = "comeAliveP95S", self.parameters.hold_budget_s + 1
            elif blocker.measured is not None:
                name, bound = blocker.field, _permitted(blocker.measured)
            else:
                continue
            bounds[name] = max(bound, bounds.get(name, 0))
        return bounds

    def booted_within(self, identity: str, option: _Option, bound: int) -> tuple[BootEvidence, ...]:
        key = (identity, option.receipt.cell, option.receipt.container_shape, option.warmth)
        budget = max(0.0, bound - (option.placement_wait_s or 0.0))
        return tuple(
            replace(record, boot_seconds=min(record.boot_seconds, budget))
            if (record.identity, record.cell, record.container_shape, record.warmth) == key
            and record.boot_seconds is not None
            else record
            for record in self.boot_records
        )

    def levers(self, choice: _Choice, written: Plan, total: float) -> list[Lever]:
        baseline = choice.baseline
        bound = choice.level.come_alive_p95_s
        if baseline is None or bound is None:
            return []
        best: dict[tuple[Cell, ContainerShape, Warmth], _Option] = {}
        for option in choice.options:
            if option.warmth is Warmth.PINNED or option.cost >= baseline:
                continue
            if [blocker.field for blocker in option.blockers] != ["comeAliveP95S"]:
                continue
            key = (option.receipt.cell, option.receipt.container_shape, option.warmth)
            known = best.get(key)
            if known is None or option.rank < known.rank:
                best[key] = option
        levers = []
        for option in best.values():
            permitted = _permitted(option.come_alive_s)
            saving = self.saving_when_loosened(choice.identity, {"comeAliveP95S": permitted}, written, total)
            if not saving:
                continue
            under = 0.0
            if bound > 0:
                replanned = self.replanned_total(
                    written, boot_evidence=self.booted_within(choice.identity, option, bound)
                )
                under = _saving(total, replanned)
            levers.append(
                Lever(
                    identity=choice.identity,
                    cell=option.receipt.cell,
                    container_shape=option.receipt.container_shape,
                    warmth=option.warmth,
                    come_alive_s=option.come_alive_s,
                    come_alive_basis=option.basis,
                    bound_s=bound,
                    monthly_cost=option.cost,
                    saving_under_bound=under,
                    permitted_bound_s=permitted,
                    saving_at_permitted_bound=saving,
                    placement_wait_s=option.placement_wait_s,
                )
            )
        return levers

    def next_cheaper(self, choice: _Choice, baseline: float | None, written: Plan, total: float) -> Option | None:
        if baseline is None:
            return None
        cheaper = [option for option in choice.options if not option.feasible and option.cost < baseline]
        if not cheaper:
            return None
        cost = max(option.cost for option in cheaper)
        nearest = min((option for option in cheaper if option.cost == cost), key=lambda option: option.rank)
        bounds = self.loosening(nearest.blockers)
        return nearest.summary(saving=self.saving_when_loosened(choice.identity, bounds, written, total))

    def scarcity(self, choice: _Choice) -> list[Scarcity]:
        baseline = choice.baseline
        if baseline is None:
            return []
        bound = choice.level.come_alive_p95_s
        candidates = [(option, _scarcity_reason(option, bound)) for option in choice.options]
        candidates.extend(
            (option, ScarcityReason.UNPLACED)
            for option in choice.withdrawn
            if bound is None or (bound > 0 and (option.boot_s or 0.0) <= bound)
        )
        best: dict[tuple[Cell, ContainerShape, Warmth, str], tuple[_Option, ScarcityReason]] = {}
        for option, reason in candidates:
            if reason is None or round(baseline - option.cost, 2) <= 0:
                continue
            key = (option.receipt.cell, option.receipt.container_shape, option.warmth, option.host or "")
            known = best.get(key)
            if known is None or option.rank < known[0].rank:
                best[key] = (option, reason)
        return [
            Scarcity(
                identity=choice.identity,
                reason=reason,
                cell=option.receipt.cell,
                container_shape=option.receipt.container_shape,
                warmth=option.warmth,
                monthly_cost=option.cost,
                placement_failure_rate=self.placements.get(
                    option.receipt.cell.gpu_product, _UNMEASURED_PLACEMENT
                ).failure_rate,
                come_alive_bound_s=bound,
                boot_s=option.boot_s,
                placement_wait_s=option.placement_wait_s,
                lane=option.host,
            )
            for option, reason in best.values()
        ]

    def capacity_use(self, plan: Plan) -> tuple[CapacityUse, ...]:
        capacity = self.capacity
        if capacity is None:
            return ()

        def held(lanes: Iterable[PlanLane]) -> tuple[int, int, tuple[str, ...]]:
            lanes = tuple(lanes)
            envelope = sum(lane.envelope.floor_max * lane.container_shape.gpu_count for lane in lanes)
            peak = sum(lane.max_containers * lane.container_shape.gpu_count for lane in lanes)
            return envelope, peak, tuple(lane.name for lane in lanes)

        total, peak, names = held(plan.lanes)
        uses = []
        if capacity.total is not None:
            uses.append(
                CapacityUse(
                    scope=CapacityScope.TOTAL,
                    ceiling=capacity.total,
                    accelerators=total,
                    peak_accelerators=peak,
                    lanes=names,
                )
            )
        for ceiling in sorted(capacity.types, key=lambda entry: entry.gpu_product):
            accelerators, peak_accelerators, lanes = held(
                lane for lane in plan.lanes if any(cell.gpu_product == ceiling.gpu_product for cell in lane.cells)
            )
            uses.append(
                CapacityUse(
                    scope=CapacityScope.TYPE,
                    gpu_product=ceiling.gpu_product,
                    ceiling=ceiling.accelerators,
                    accelerators=accelerators,
                    peak_accelerators=peak_accelerators,
                    lanes=lanes,
                )
            )
        if capacity.shared_quota is not None:
            quota = capacity.shared_quota
            uses.append(
                CapacityUse(
                    scope=CapacityScope.SHARED_QUOTA,
                    ceiling=quota.accelerators,
                    accelerators=total + quota.held_elsewhere,
                    peak_accelerators=peak + quota.held_elsewhere,
                    held_elsewhere=quota.held_elsewhere,
                    lanes=names,
                )
            )
        return tuple(uses)

    @staticmethod
    def lane_names(placed: Mapping[str, _Option], reserved: set[str]) -> dict[str, str]:
        bases = {
            identity: _slug("-".join((identity, *(cell.machine_profile for cell in option.cells)))) or "lane"
            for identity, option in placed.items()
        }
        counts = Counter(bases.values())
        taken = set(reserved)
        names: dict[str, str] = {}
        for identity, base in bases.items():
            name = base if counts[base] == 1 and base not in reserved else f"{base}-{_digest(identity)}"
            if name in taken:
                name = f"{base}-{_digest(identity, width=64)}"
            if name in taken:
                raise PlannerInputError(f"lane name {name!r} for {identity!r} is already taken")
            taken.add(name)
            names[identity] = name
        return names

    def placed_lane(self, name: str, identity: str, option: _Option) -> PlanLane:
        receipt = option.receipt
        return PlanLane(
            name=name,
            queue_pool=self.parameters.queue_pool,
            bundle=self.parameters.bundle_of(identity),
            cell=receipt.cell,
            fallback_cells=option.cells[1:],
            container_shape=receipt.container_shape,
            concurrency=receipt.concurrency,
            max_containers=self.parameters.max_containers,
            envelope=envelope_for(option.warmth, self.parameters),
        )

    def within_ceiling(self, lane: PlanLane) -> PlanLane:
        ceiling = self.parameters.max_containers
        if lane.max_containers <= ceiling:
            return lane
        envelope = replace(
            lane.envelope,
            floor_min=min(lane.envelope.floor_min, ceiling),
            floor_max=min(lane.envelope.floor_max, ceiling),
        )
        return replace(lane, max_containers=ceiling, envelope=envelope)

    def slowest_receipt(self, identity: str, lane: PlanLane) -> Receipt | None:
        matching = [(order, receipt) for order, receipt in self.receipts.get(identity, []) if _runs_on(receipt, lane)]
        if not matching:
            return None
        return min(matching, key=lambda pair: (pair[1].throughput.requests_per_second, pair[0]))[1]

    def kept_blockers(
        self, identity: str, entry: PlanIdentity, lane: PlanLane, receipt: Receipt | None
    ) -> tuple[Blocker, ...]:
        level = self.levels[identity]
        blockers = [] if receipt is None else list(_warm_blockers(level, receipt))
        if entry.warmth in BOOT_WARMTHS:
            boot = self.boots.get((identity, lane.cell, lane.container_shape, entry.warmth))
            if boot is not None and boot.seconds is not None:
                blocker = _come_alive_blocker(level, boot.seconds)
                blockers.extend(() if blocker is None else (blocker,))
            elif boot is not None or level.come_alive_p95_s == 0:
                blockers.append(Blocker(field="comeAliveP95S", bound=level.come_alive_p95_s, measured=None))
        blockers.extend(self.reliability_blockers(level, lane.cells))
        return tuple(blockers)

    def demand_hours(self, lane: PlanLane, identity: str, receipt: Receipt | None) -> float | None:
        requests = self.requests.get(identity, ())
        if not requests:
            return 0.0
        if receipt is None:
            return None
        window = replace(
            self.window,
            ceiling=lane.max_containers,
            active_floor=max(1, min(self.parameters.fallback_floor, lane.max_containers)),
        )
        rate = receipt.throughput.requests_per_second
        if lane.envelope.floor_min:
            return window.above_floor_hours(requests, rate, lane.envelope.floor_min)
        return window.cold_container_hours(requests, rate)

    def assemble(self, choices: Mapping[str, _Choice]) -> _Layout:
        placed = {identity: choice.chosen for identity, choice in choices.items() if choice.chosen is not None}
        kept = {
            identity: self.current[identity]
            for identity, choice in choices.items()
            if choice.chosen is None and identity in self.current
        }
        staying = {identity: self.current[identity][1].name for identity, option in placed.items() if option.current}
        joining = {
            identity: host
            for identity, option in placed.items()
            if (host := option.host) is not None and identity not in staying
        }
        riding = {
            identity: owner
            for identity, option in placed.items()
            if (owner := option.host_owner) is not None and identity not in staying
        }
        reserved = {lane.name for _, lane in kept.values()} | set(staying.values()) | set(joining.values())
        moving = {
            identity: option
            for identity, option in placed.items()
            if identity not in staying and identity not in joining and identity not in riding
        }
        names = staying | joining | self.lane_names(moving, reserved)
        names.update((identity, names[owner]) for identity, owner in riding.items())
        carried = [lane for _, lane in kept.values()]
        carried.extend(self.current_lanes[name] for name in (*staying.values(), *joining.values()))
        lanes = {lane.name: self.within_ceiling(lane) for lane in carried}
        for identity, option in moving.items():
            lanes[names[identity]] = self.placed_lane(names[identity], identity, option)
        identities = [entry for entry, _ in kept.values()]
        identities.extend(
            PlanIdentity(identity=identity, lane=names[identity], warmth=option.warmth)
            for identity, option in placed.items()
        )
        plan = Plan(
            placement=self.parameters.placement,
            lanes=tuple(lanes[name] for name in sorted(lanes)),
            identities=tuple(sorted(identities, key=lambda entry: entry.identity)),
        )
        receipts: dict[str, Receipt | None] = {identity: option.pricing_receipt for identity, option in placed.items()}
        receipts.update({identity: self.slowest_receipt(identity, lane) for identity, (_, lane) in kept.items()})
        return _Layout(
            plan=plan,
            names=names,
            kept=kept,
            receipts=receipts,
            owners={names[identity]: identity for identity in moving},
        )

    def total(self, layout: _Layout) -> float:
        lane_reports, _ = self._pricing(layout.plan, layout.receipts)
        return sum(lane.monthly_cost for lane in lane_reports if lane.monthly_cost is not None)

    def stay(self, choice: _Choice) -> _Choice | None:
        identity = choice.identity
        current = self.keepable(identity)
        if current is None:
            return None
        standing = [option for option in choice.options if option.current]
        feasible = [option for option in standing if option.feasible]
        if feasible:
            return replace(choice, chosen=min(feasible, key=lambda option: option.rank))
        entry, lane = current
        if standing or self.kept_blockers(identity, entry, lane, self.slowest_receipt(identity, lane)):
            return None
        cost = self.current_cost(identity, entry, lane)
        gaps = self.current_gaps(identity, entry, lane, cost, choice.beside)
        return replace(
            choice,
            chosen=None,
            held=cost,
            unpriced=cost is None,
            gaps=gaps,
            boot_requests=_ungapped(identity, choice.boot_requests, gaps),
        )

    @staticmethod
    def stays(identity: str, choice: _Choice, layout: _Layout) -> bool:
        if choice.chosen is None:
            return identity in layout.kept
        return choice.chosen.current

    def stay_beside(self, choices: dict[str, _Choice], originals: Mapping[str, _Choice]) -> dict[str, _Choice]:
        layout = self.assemble(choices)
        total = self.total(layout)
        left: defaultdict[str, dict[str, _Choice]] = defaultdict(dict)
        for identity in sorted(choices):
            if identity not in self.current or self.stays(identity, choices[identity], layout):
                continue
            stay = self.stay(originals[identity])
            if stay is not None:
                left[self.current[identity][1].name][identity] = stay
        for lane in sorted(left):
            movers = left[lane]
            groups = [{identity: stay} for identity, stay in movers.items()]
            if len(movers) > 1:
                groups.append(movers)
            for group in groups:
                trial = self.settle({**choices, **group})
                trial_layout = self.assemble(trial)
                trial_total = self.total(trial_layout)
                if round(trial_total - total, 2) <= 0:
                    choices, layout, total = trial, trial_layout, trial_total
        return choices

    def riding_cost(self, lane: PlanLane, identity: str, receipt: Receipt) -> float:
        window = replace(self.window, ceiling=lane.max_containers)
        requests = self.requests.get(identity, ())
        rate = receipt.throughput.requests_per_second
        return self.lane_hourly(lane) * window.above_floor_hours(requests, rate, lane.envelope.floor_min)

    def in_use(self, lane: PlanLane, pinned: Iterable[str], lazy: Iterable[str]) -> int:
        pinned_use = sum(self.footprints.get((identity, lane.cell), 0) for identity in pinned)
        return pinned_use + max((self.footprints.get((identity, lane.cell), 0) for identity in lazy), default=0)

    def pack_targets(self, identity: str, layout: _Layout) -> list[tuple[PlanLane, frozenset[str], frozenset[str]]]:
        pinned: defaultdict[str, set[str]] = defaultdict(set)
        lazy: defaultdict[str, set[str]] = defaultdict(set)
        for entry in layout.plan.identities:
            (pinned if entry.warmth is Warmth.PINNED else lazy)[entry.lane].add(entry.identity)
        targets = [
            (lane, frozenset(pinned[lane.name]), frozenset(lazy[lane.name]))
            for lane in layout.plan.lanes
            if lane.name != layout.names[identity]
            and (identity, lane.name) not in self.excluded
            and lane.envelope.floor_min >= PINNED_FLOOR
            and not lane.fallback_cells
            and lane.container_shape.gpu_count == 1
            and lane.queue_pool == self.parameters.queue_pool
            and lane.bundle == self.parameters.bundle_of(identity)
            and pinned[lane.name]
            and not self.hosts_solitary((*pinned[lane.name], *lazy[lane.name]))
            and self.slowest_receipt(identity, lane) is not None
        ]
        return sorted(
            targets,
            key=lambda target: (
                not any(self.levels[resident].come_alive_p95_s == 0 for resident in target[1]),
                -self.in_use(*target),
                target[0].name,
            ),
        )

    def packings(
        self,
        identity: str,
        target: tuple[PlanLane, frozenset[str], frozenset[str]],
        layout: _Layout,
        choices: Mapping[str, _Choice],
    ) -> list[_Packing]:
        lane, pinned, lazy = target
        level = self.levels[identity]
        order, receipt = min(
            (pair for pair in self.receipts.get(identity, []) if _runs_on(pair[1], lane)),
            key=lambda pair: (pair[1].throughput.requests_per_second, pair[0]),
        )
        owner = layout.owners.get(lane.name)
        host = None if owner is not None else lane.name
        offered = choices[identity]
        hosts = {option.host for option in offered.options if option.warmth is Warmth.LAZY_RESIDENT}
        hosts.update(request.lane for request in offered.boot_requests if request.warmth is Warmth.LAZY_RESIDENT)
        packings: list[_Packing] = []
        current_class = self.current_class(identity)
        blocked = (*self.warm_blockers(level, receipt, current_class), *self.reliability_blockers(level, lane.cells))
        load = self.boots.get((identity, lane.cell, lane.container_shape, Warmth.LAZY_RESIDENT))
        if (
            level.come_alive_p95_s != 0
            and not blocked
            and lane.name not in hosts
            and any(self.levels[resident].come_alive_p95_s == 0 for resident in pinned)
            and (load is None or load.seconds is not None)
            and (load is None or load.seconds is None or _come_alive_blocker(level, load.seconds) is None)
        ):
            memory = self.memory(lane, pinned=pinned, lazy=lazy | {identity})
            if memory.fits is not False:
                lacking = () if memory.fits else self.memory_gaps(lane, (*pinned, *lazy, identity))
                missing = (
                    *(() if load is not None else (Measurement.BOOT_EVIDENCE,)),
                    *((Measurement.FOOTPRINT,) if identity in lacking else ()),
                )
                option = _Option(
                    receipt=receipt,
                    order=order,
                    warmth=Warmth.LAZY_RESIDENT,
                    cost=self.riding_cost(lane, identity, receipt),
                    come_alive_s=0.0 if load is None or load.seconds is None else load.seconds,
                    basis=ComeAliveBasis.BOOT_EVIDENCE if load is None else load.basis,
                    blockers=(),
                    current=False,
                    boot_s=None if load is None else load.boot_s,
                    host=host,
                    host_owner=owner,
                )
                others = tuple(
                    (resident, Warmth.PINNED if resident in pinned else Warmth.LAZY_RESIDENT)
                    for resident in lacking
                    if resident != identity
                )
                packings.append(({identity: option}, missing, (), others))
        members = pinned | {identity}
        if all(choices[member].chosen is not None for member in pinned):
            together = {member: self.co_receipt(member, lane, members - {member}) for member in members}
            measured = {member: pair for member, pair in together.items() if pair is not None}
            if not any(
                (
                    *self.warm_blockers(self.levels[member], pair[1], self.current_class(member)),
                    *self.reliability_blockers(self.levels[member], lane.cells),
                )
                for member, pair in measured.items()
            ):
                memory = self.memory(lane, pinned=members, lazy=lazy)
                if memory.fits is not False:
                    lacking = () if memory.fits else self.memory_gaps(lane, (*members, *lazy))
                    missing = (
                        *(() if len(measured) == len(members) else (Measurement.CO_RESIDENCY_RECEIPT,)),
                        *((Measurement.FOOTPRINT,) if identity in lacking else ()),
                    )
                    others = tuple(
                        (resident, Warmth.PINNED if resident in members else Warmth.LAZY_RESIDENT)
                        for resident in lacking
                        if resident != identity
                    )
                    own_order, own = measured.get(identity, (order, receipt))
                    updates = {
                        identity: _Option(
                            receipt=own,
                            order=own_order,
                            warmth=Warmth.PINNED,
                            cost=self.riding_cost(lane, identity, own),
                            come_alive_s=0.0,
                            basis=ComeAliveBasis.WARM_FLOOR,
                            blockers=(),
                            current=False,
                            host=host,
                            host_owner=owner,
                        )
                    }
                    if not missing and not others:
                        for member in sorted(pinned):
                            chosen = choices[member].chosen
                            member_order, member_receipt = measured[member]
                            if chosen is not None:
                                rate = member_receipt.throughput.requests_per_second
                                updates[member] = replace(
                                    chosen,
                                    receipt=member_receipt,
                                    order=member_order,
                                    cost=self.hourly(member_receipt)
                                    * self.window.warm_container_hours(choices[member].requests, rate),
                                )
                    packings.append((updates, missing, tuple(sorted(pinned)), others))
        return packings

    def pack(self, choices: dict[str, _Choice]) -> tuple[dict[str, _Choice], list[MeasurementRequest]]:
        layout = self.assemble(choices)
        total = self.total(layout)
        lanes = {lane.name: lane for lane in layout.plan.lanes}
        hosted = Counter(entry.lane for entry in layout.plan.identities)
        hours = self.parameters.hours_per_month

        def floor_cost(identity: str) -> float:
            lane = lanes[layout.names[identity]]
            if _lane_refusal(lane, self.prices) is not None:
                return 0.0
            return round(self.lane_hourly(lane) * lane.envelope.floor_min * hours, 6)

        candidates = [
            identity
            for identity, choice in choices.items()
            if (chosen := choice.chosen) is not None
            and chosen.host in (None, layout.names[identity])
            and chosen.host_owner is None
            and identity not in self.solitary
            and hosted[layout.names[identity]] == 1
        ]
        requests: dict[tuple[str, str | None, Warmth, tuple[Measurement, ...]], MeasurementRequest] = {}
        for identity in sorted(candidates, key=lambda identity: (-floor_cost(identity), identity)):
            if hosted.get(layout.names.get(identity, ""), 0) != 1:
                continue
            moved = False
            for target in self.pack_targets(identity, layout):
                for updates, missing, beside, lacking in self.packings(identity, target, layout, choices):
                    members = frozenset(beside) | {identity}
                    trial = {
                        **choices,
                        **{
                            member: replace(choices[member], chosen=option)
                            if member == identity
                            else replace(choices[member], chosen=option, beside=members - {member})
                            for member, option in updates.items()
                        },
                    }
                    trial_layout = self.assemble(trial)
                    trial_total = self.total(trial_layout)
                    saving = total - trial_total
                    if round(saving, 2) <= 0:
                        continue
                    option = updates[identity]
                    if missing or lacking:
                        found = [
                            self.footprint_request(resident, target[0], warmth, option.cost, saving)
                            for resident, warmth in lacking
                        ]
                        if missing:
                            found.append(
                                MeasurementRequest(
                                    identity=identity,
                                    cell=target[0].cell,
                                    container_shape=target[0].container_shape,
                                    concurrency=target[0].concurrency,
                                    lane=target[0].name,
                                    measure=missing,
                                    warmth=option.warmth,
                                    reason=RequestReason.UNMEASURED,
                                    cost_bound=option.cost,
                                    saving_bound=saving,
                                    co_residents=beside,
                                )
                            )
                        for request in found:
                            key = (request.identity, request.lane, request.warmth, request.measure)
                            known = requests.get(key)
                            if known is None or saving > (known.saving_bound or 0.0):
                                requests[key] = request
                        continue
                    choices, layout, total = trial, trial_layout, trial_total
                    hosted = Counter(entry.lane for entry in layout.plan.identities)
                    moved = True
                    break
                if moved:
                    break
        return choices, list(requests.values())

    def stale_beside(self, choices: Mapping[str, _Choice]) -> dict[str, frozenset[str]]:
        layout = self.assemble(choices)
        pinned: defaultdict[str, set[str]] = defaultdict(set)
        for entry in layout.plan.identities:
            if entry.warmth is Warmth.PINNED:
                pinned[entry.lane].add(entry.identity)
        stale: dict[str, frozenset[str]] = {}
        for identity, choice in choices.items():
            current = self.current.get(identity)
            if current is None or current[0].warmth is not Warmth.PINNED:
                continue
            if self.stays(identity, choice, layout):
                beside = frozenset(pinned[current[1].name] - {identity})
                if beside != choice.beside:
                    stale[identity] = beside
        return stale

    def settle(self, choices: dict[str, _Choice]) -> dict[str, _Choice]:
        for _ in range(len(self.levels) + 1):
            stale = self.stale_beside(choices)
            if not stale:
                return choices
            choices = {**choices, **{identity: self.choose(identity, beside) for identity, beside in stale.items()}}
        return choices

    def overflowing(self, choices: Mapping[str, _Choice]) -> list[tuple[str, str]]:
        layout = self.assemble(choices)
        before: defaultdict[str, set[str]] = defaultdict(set)
        if self.current_plan is not None:
            for entry in self.current_plan.identities:
                before[entry.lane].add(entry.identity)
        residents: defaultdict[str, dict[str, Warmth]] = defaultdict(dict)
        for entry in layout.plan.identities:
            residents[entry.lane][entry.identity] = entry.warmth
        rejected: list[tuple[str, str]] = []
        for lane in layout.plan.lanes:
            hosted = residents[lane.name]
            joiners = sorted(
                (
                    identity
                    for identity in hosted
                    if identity not in before[lane.name]
                    and (chosen := choices[identity].chosen) is not None
                    and (chosen.host is not None or chosen.host_owner is not None)
                ),
                key=lambda identity: (-self.footprints.get((identity, lane.cell), 0), identity),
            )
            if not joiners or self.lane_memory(lane, hosted).fits:
                continue
            admitted = {identity: warmth for identity, warmth in hosted.items() if identity not in joiners}
            for identity in joiners:
                trial = {**admitted, identity: hosted[identity]}
                if self.lane_memory(lane, trial).fits:
                    admitted = trial
                else:
                    rejected.append((identity, lane.name))
        return rejected

    def fit(self, choices: dict[str, _Choice]) -> dict[str, _Choice]:
        for _ in range(len(self.levels) * (len(self.current_lanes) + len(self.levels)) + 1):
            choices = self.settle(choices)
            rejected = self.overflowing(choices)
            if not rejected:
                return choices
            self.excluded.update(rejected)
            choices = {**choices, **{identity: self.choose(identity) for identity, _ in rejected}}
        return self.settle(choices)

    def decide(self) -> tuple[dict[str, _Choice], list[MeasurementRequest]]:
        never_cold = [identity for identity in sorted(self.levels) if self.levels[identity].come_alive_p95_s == 0]
        self.anchors = self.staying_anchors(self.choose(identity) for identity in never_cold)
        originals = {identity: self.choose(identity) for identity in sorted(self.levels)}
        decided = self.fit(self.stay_beside(self.settle(dict(originals)), originals))
        decided, requests = self.pack(decided)
        return self.fit(decided), requests

    def run(self, *, price_savings: bool = True) -> PlannerResult:
        decided, packing_requests = self.decide()
        choices = [decided[identity] for identity in sorted(decided)]
        layout = self.assemble(decided)
        new_plan, names, kept, receipts = layout.plan, layout.names, layout.kept, layout.receipts
        lane_reports, costs = self._pricing(new_plan, receipts)
        total = sum(lane.monthly_cost for lane in lane_reports if lane.monthly_cost is not None)
        reports: list[IdentityReport] = []
        requests: list[MeasurementRequest] = list(packing_requests)
        levers: list[Lever] = []
        scarcity: list[Scarcity] = []
        for choice in choices:
            baseline = choice.baseline
            scarcity.extend(self.scarcity(choice))
            requests.extend(choice.gaps)
            requests.extend(self.receipt_requests(choice, baseline))
            requests.extend(self.boot_requests(choice, baseline))
            nearest = None
            if price_savings:
                levers.extend(self.levers(choice, new_plan, total))
                nearest = self.next_cheaper(choice, baseline, new_plan, total)
            reports.append(
                self._identity_report(
                    choice, names, kept, costs.get(choice.identity), receipts.get(choice.identity), nearest
                )
            )
        identity_changes, lane_changes = _changes(self.current_plan, new_plan)
        report = Report(
            placement=self.parameters.placement,
            currency=self.price_list.currency,
            parameters=self.parameters.to_dict(),
            identities=tuple(reports),
            lanes=lane_reports,
            shared_chart_lanes=_shared_chart_lanes(new_plan),
            measurement_requests=_distinct(requests),
            come_alive_levers=tuple(sorted(levers, key=_lever_rank)),
            queue_wait_unattained=tuple(
                identity for identity in sorted(self.levels) if self.levels[identity].queue_wait_p99_ms is not None
            ),
            refusals=tuple(sorted(self.refusals, key=_refusal_rank)),
            identity_changes=identity_changes,
            lane_changes=lane_changes,
            warm_bound_undeclared=tuple(
                identity
                for identity in sorted(self.levels)
                if not _declares_warm_bound(self.levels[identity]) and self.receipts.get(identity)
            ),
            capacity=self.capacity_use(new_plan),
            scarcity=tuple(sorted(scarcity, key=_scarcity_rank)),
            gpu_memory_undeclared=tuple(sorted(self.gpu_memory_undeclared)),
        )
        return PlannerResult(plan=new_plan, report=report)

    def _identity_report(
        self,
        choice: _Choice,
        names: Mapping[str, str],
        kept: Mapping[str, tuple[PlanIdentity, PlanLane]],
        cost: float | None,
        receipt: Receipt | None,
        next_cheaper: Option | None,
    ) -> IdentityReport:
        current = self.current.get(choice.identity)
        report = IdentityReport(
            identity=choice.identity,
            status=IdentityStatus.UNPLACED,
            service_level=choice.level,
            demand_requests=sum(choice.requests),
            current=None if current is None else current[0],
        )
        chosen = choice.chosen
        if chosen is None:
            reason = choice.unplaced_reason
            if choice.identity in kept:
                entry, lane = kept[choice.identity]
                return replace(
                    report,
                    status=IdentityStatus.KEPT,
                    lane=entry.lane,
                    monthly_cost=cost,
                    unplaced_reason=reason,
                    blocked_by=self.kept_blockers(choice.identity, entry, lane, receipt),
                    next_cheaper=next_cheaper,
                )
            return replace(report, unplaced_reason=reason)
        rate = chosen.pricing_receipt.throughput.requests_per_second
        summary = chosen.summary()
        if chosen.host_owner is not None:
            summary = replace(summary, lane=names[choice.identity])
        return replace(
            report,
            status=IdentityStatus.PLACED,
            lane=names[choice.identity],
            monthly_cost=cost,
            chosen=summary,
            peak_containers=max((_Window.needed(count, rate) for count in choice.requests), default=0),
            next_cheaper=next_cheaper,
        )

    def _pricing(
        self, plan: Plan, receipts: Mapping[str, Receipt | None]
    ) -> tuple[tuple[LaneReport, ...], dict[str, float]]:
        residents: defaultdict[str, list[str]] = defaultdict(list)
        warmths = {entry.identity: entry.warmth for entry in plan.identities}
        for entry in plan.identities:
            residents[entry.lane].append(entry.identity)
        hours = self.parameters.hours_per_month
        reports: list[LaneReport] = []
        costs: dict[str, float] = {}
        for lane in plan.lanes:
            hosted = tuple(sorted(residents[lane.name]))
            reason = _lane_refusal(lane, self.prices)
            if reason is not None:
                reports.append(LaneReport(name=lane.name, identities=hosted, unpriced_reason=reason))
                continue
            hourly = self.lane_hourly(lane)
            floor = lane.envelope.floor_min * hours
            demand = {identity: self.demand_hours(lane, identity, receipts.get(identity)) for identity in hosted}
            priced = {identity: value for identity, value in demand.items() if value is not None}
            costs.update((identity, hourly * (floor + value)) for identity, value in priced.items())
            memory = _UNPACKABLE
            if len(hosted) > 1:
                memory = self.memory(
                    lane,
                    pinned=(identity for identity in hosted if warmths[identity] is Warmth.PINNED),
                    lazy=(identity for identity in hosted if warmths[identity] is Warmth.LAZY_RESIDENT),
                )
            reports.append(
                LaneReport(
                    name=lane.name,
                    identities=hosted,
                    hourly_cost=hourly,
                    monthly_cost=hourly * (floor + sum(priced.values())),
                    floor_monthly_cost=hourly * lane.envelope.floor_min * hours,
                    envelope_max_monthly_cost=hourly * lane.envelope.floor_max * hours,
                    unpriced_identities=tuple(identity for identity in hosted if identity not in priced),
                    memory_mib=memory.need,
                    memory_limit_mib=memory.limit if memory.need is not None else None,
                )
            )
        return tuple(reports), costs


def _ungapped(
    identity: str, requests: Iterable[MeasurementRequest], gaps: Iterable[MeasurementRequest]
) -> tuple[MeasurementRequest, ...]:
    gapped = {
        (request.cell, request.container_shape, request.warmth)
        for request in gaps
        if request.identity == identity and Measurement.BOOT_EVIDENCE in request.measure
    }
    return tuple(
        request
        for request in requests
        if request.identity != identity or (request.cell, request.container_shape, request.warmth) not in gapped
    )


def _distinct(requests: Iterable[MeasurementRequest]) -> tuple[MeasurementRequest, ...]:
    kept: dict[tuple[object, ...], MeasurementRequest] = {}
    for request in sorted(requests, key=_request_rank):
        key = (
            request.identity,
            request.cell,
            request.container_shape,
            request.concurrency,
            request.lane,
            request.measure,
            request.warmth,
            request.co_residents,
        )
        kept.setdefault(key, request)
    return tuple(kept.values())


def _scarcity_reason(option: _Option, bound: int | None) -> ScarcityReason | None:
    fields = _fields(option.blockers)
    if fields == [PLACEMENT_FAILURE_FIELD]:
        return ScarcityReason.UNRELIABLE_TYPE
    if option.warmth not in COLD_WARMTHS or fields != ["comeAliveP95S"] or not option.placement_wait_s:
        return None
    if not bound or option.boot_s is None or option.boot_s > bound:
        return None
    return ScarcityReason.PLACEMENT_WAIT


def _stands_beside(receipt: Receipt, beside: frozenset[str]) -> bool:
    return beside <= frozenset(receipt.co_residents) if receipt.co_residents else not beside


def _p99(receipt: Receipt) -> float:
    return max(
        percentiles.p99_ms for percentiles in (receipt.latency, receipt.ttft, receipt.itl) if percentiles is not None
    )


def _saving(total: float, replanned: float) -> float:
    saving = total - replanned
    return saving if round(saving, 2) > 0 else 0.0


def _permitted(measured: float) -> int:
    return max(1, math.ceil(measured))


def _shared_chart_lanes(plan: Plan) -> tuple[SharedChartLane, ...]:
    grouped: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for lane in plan.lanes:
        grouped[(lane.queue_pool, lane.cell.machine_profile, lane.bundle)].append(lane.name)
    return tuple(
        SharedChartLane(queue_pool=pool, machine_profile=profile, bundle=bundle, lanes=tuple(sorted(names)))
        for (pool, profile, bundle), names in sorted(grouped.items())
        if len(names) > 1
    )


def _changes(current: Plan | None, new: Plan) -> tuple[tuple[IdentityChange, ...], tuple[LaneChange, ...]]:
    before_identities = {} if current is None else {entry.identity: entry for entry in current.identities}
    after_identities = {entry.identity: entry for entry in new.identities}
    before_lanes = {} if current is None else {lane.name: lane for lane in current.lanes}
    after_lanes = {lane.name: lane for lane in new.lanes}
    identity_changes = tuple(
        IdentityChange(identity=identity, before=before_identities.get(identity), after=after_identities.get(identity))
        for identity in sorted(before_identities.keys() | after_identities.keys())
        if before_identities.get(identity) != after_identities.get(identity)
    )
    lane_changes = tuple(
        LaneChange(name=name, before=before_lanes.get(name), after=after_lanes.get(name))
        for name in sorted(before_lanes.keys() | after_lanes.keys())
        if before_lanes.get(name) != after_lanes.get(name)
    )
    return identity_changes, lane_changes


def _shape_key(shape: ContainerShape | None) -> tuple[int, float, int]:
    return (0, 0.0, 0) if shape is None else (shape.gpu_count, shape.cpu_cores, shape.memory_mib)


def _request_rank(request: MeasurementRequest) -> tuple[object, ...]:
    return (
        request.saving_bound is not None,
        -(request.saving_bound or 0.0),
        request.identity,
        request.cell.machine_profile,
        request.cell.gpu_product,
        _shape_key(request.container_shape),
        tuple(measurement.value for measurement in request.measure),
        request.lane or "",
        request.concurrency or 0,
    )


def _lever_rank(lever: Lever) -> tuple[object, ...]:
    return (
        -lever.saving_under_bound,
        -lever.saving_at_permitted_bound,
        lever.identity,
        lever.cell.machine_profile,
        lever.cell.gpu_product,
        _shape_key(lever.container_shape),
        lever.warmth.value,
    )


def _scarcity_rank(entry: Scarcity) -> tuple[object, ...]:
    return (
        entry.identity,
        entry.reason.value,
        entry.cell.machine_profile,
        entry.cell.gpu_product,
        _shape_key(entry.container_shape),
        entry.warmth.value,
        entry.lane or "",
    )


def _refusal_rank(refusal: Refusal) -> tuple[str, ...]:
    return (
        refusal.record.value,
        refusal.identity,
        refusal.cell.machine_profile,
        refusal.cell.gpu_product,
        refusal.source.uri,
        refusal.source.sha256 or "",
        refusal.reason,
    )


def plan(
    service_levels: ServiceLevels,
    receipts: Iterable[Receipt],
    boot_evidence: Iterable[BootEvidence],
    price_list: PriceList,
    demand: Demand,
    current_plan: Plan | None,
    parameters: PlannerParameters,
    *,
    capacity: Capacity | None = None,
) -> PlannerResult:
    result = _Planner(
        service_levels, receipts, boot_evidence, price_list, demand, current_plan, parameters, capacity
    ).run()
    if any(use.exceeded for use in result.report.capacity):
        raise CapacityExceededError(result)
    return result
