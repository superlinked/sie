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
    SharedChartLane,
    UnplacedReason,
)
from sie_placement.schema import (
    WARM_FLOOR_WARMTHS,
    BootEvidence,
    BootOutcome,
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
    hours_per_month: float = HOURS_PER_MONTH
    queue_pool: str = "default"
    bundle: str = "default"
    bundles: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_name("placement", self.placement)
        _require_minimum("maxContainers", self.max_containers, 1)
        _require_minimum("fallbackFloor", self.fallback_floor, 0)
        _require_minimum("cooldownS", self.cooldown_s, 0)
        _require_minimum("bufferContainers", self.buffer_containers, 0)
        if not (math.isfinite(self.hours_per_month) and self.hours_per_month > 0):
            raise SchemaError("hoursPerMonth", f"must be greater than 0, got {self.hours_per_month!r}")
        _require_name("queuePool", self.queue_pool)
        _require_name("bundle", self.bundle)
        for identity, bundle in self.bundles.items():
            _require_name("bundles", identity)
            _require_name(f"bundles[{identity!r}]", bundle)

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
            "hoursPerMonth": self.hours_per_month,
            "queuePool": self.queue_pool,
            "bundle": self.bundle,
        }
        if self.bundles:
            document["bundles"] = dict(sorted(self.bundles.items()))
        return document


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannerResult:
    plan: Plan
    report: Report


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


@dataclass(frozen=True, slots=True)
class _Boot:
    seconds: float | None
    borrowed: bool

    @property
    def basis(self) -> ComeAliveBasis:
        return ComeAliveBasis.BORROWED_BOOT_EVIDENCE if self.borrowed else ComeAliveBasis.BOOT_EVIDENCE


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

    @property
    def feasible(self) -> bool:
        return not self.blockers

    @property
    def rank(self) -> tuple[float, bool, float, str, str]:
        return (self.cost, not self.current, self.come_alive_s, self.order, self.warmth.value)

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
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Choice:
    identity: str
    level: ServiceLevel
    requests: tuple[int, ...]
    options: tuple[_Option, ...]
    chosen: _Option | None
    boot_requests: tuple[MeasurementRequest, ...]
    has_valid_receipt: bool


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


def _come_alive_blocker(level: ServiceLevel, seconds: float) -> Blocker | None:
    bound = level.come_alive_p95_s
    if bound is None or (bound > 0 and seconds <= bound):
        return None
    return Blocker(field="comeAliveP95S", bound=bound, measured=seconds)


def _boot_index(records: Iterable[BootEvidence]) -> dict[_BootKey, _Boot]:
    grouped: defaultdict[_BootKey, list[BootEvidence]] = defaultdict(list)
    for record in records:
        grouped[(record.identity, record.cell, record.container_shape, record.warmth)].append(record)
    index: dict[_BootKey, _Boot] = {}
    for key, group in grouped.items():
        own = [record for record in group if not record.borrowed]
        used = own or group
        if any(record.outcome is not BootOutcome.SUCCEEDED for record in used):
            index[key] = _Boot(seconds=None, borrowed=not own)
        else:
            index[key] = _Boot(seconds=float(max(record.boot_seconds for record in used)), borrowed=not own)
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
        lane.cell == receipt.cell
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
        if current_plan is not None:
            lanes = {lane.name: lane for lane in current_plan.lanes}
            self.current = {entry.identity: (entry, lanes[entry.lane]) for entry in current_plan.identities}
        self.refusals: list[Refusal] = []
        self.refused_profiles: set[tuple[str, str]] = set()
        self.savings: dict[tuple[str, tuple[tuple[str, int], ...]], float] = {}
        self.receipts = self._accept_receipts(receipts)
        self.boots = _boot_index(self._accept_boot_evidence(boot_evidence))

    def _accept_receipts(self, receipts: Iterable[Receipt]) -> dict[str, list[tuple[str, Receipt]]]:
        accepted: defaultdict[str, list[tuple[str, Receipt]]] = defaultdict(list)
        for receipt in receipts:
            level = self.levels.get(receipt.identity)
            if level is None:
                continue
            reason = _refusal_reason(receipt.cell, self.prices) or _incomparable_reason(level, receipt)
            if reason is not None:
                self._refuse(RefusedRecord.RECEIPT, receipt.identity, receipt.cell, receipt.source, reason)
                self.refused_profiles.add((receipt.identity, receipt.cell.machine_profile))
                continue
            accepted[receipt.identity].append((_canonical(receipt.to_dict()), receipt))
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

    def is_current(self, identity: str, receipt: Receipt, warmth: Warmth) -> bool:
        current = self.current.get(identity)
        if current is None:
            return False
        entry, lane = current
        return (
            entry.warmth is warmth
            and lane.queue_pool == self.parameters.queue_pool
            and lane.bundle == self.parameters.bundle_of(identity)
            and _runs_on(receipt, lane)
        )

    def choose(self, identity: str) -> _Choice:
        level = self.levels[identity]
        requests = self.requests.get(identity, ())
        options: list[_Option] = []
        boot_requests: dict[tuple[Cell, ContainerShape], MeasurementRequest] = {}
        receipts = self.receipts.get(identity, [])
        for order, receipt in receipts:
            hourly = container_hourly_cost(
                self.prices[receipt.cell.machine_profile], receipt.container_shape, self.multiplier
            )
            rate = receipt.throughput.requests_per_second
            warm = _warm_blockers(level, receipt)
            options.append(
                _Option(
                    receipt=receipt,
                    order=order,
                    warmth=Warmth.PINNED,
                    cost=hourly * self.window.warm_container_hours(requests, rate),
                    come_alive_s=0.0,
                    basis=ComeAliveBasis.WARM_FLOOR,
                    blockers=warm,
                    current=self.is_current(identity, receipt, Warmth.PINNED),
                )
            )
            cold_cost = hourly * self.window.cold_container_hours(requests, rate)
            for warmth in COLD_WARMTHS:
                boot = self.boots.get((identity, receipt.cell, receipt.container_shape, warmth))
                if boot is None or boot.seconds is None:
                    if warmth is Warmth.COLD_WITH_BOUND and not warm and level.come_alive_p95_s != 0:
                        key = (receipt.cell, receipt.container_shape)
                        known = boot_requests.get(key)
                        if known is None or cold_cost < known.cost_bound:
                            boot_requests[key] = MeasurementRequest(
                                identity=identity,
                                cell=receipt.cell,
                                container_shape=receipt.container_shape,
                                measure=(Measurement.BOOT_EVIDENCE,),
                                warmth=warmth,
                                reason=RequestReason.UNMEASURED if boot is None else RequestReason.FAILED,
                                cost_bound=cold_cost,
                                saving_bound=None,
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
                        blockers=warm if blocker is None else (*warm, blocker),
                        current=self.is_current(identity, receipt, warmth),
                    )
                )
        feasible = [option for option in options if option.feasible]
        chosen = min(feasible, key=lambda option: option.rank) if feasible else None
        return _Choice(
            identity=identity,
            level=level,
            requests=requests,
            options=tuple(options),
            chosen=chosen,
            boot_requests=tuple(boot_requests.values()),
            has_valid_receipt=bool(receipts),
        )

    def receipt_requests(self, choice: _Choice) -> list[MeasurementRequest]:
        measured = {receipt.cell.machine_profile for _, receipt in self.receipts.get(choice.identity, [])}
        bound = choice.level.come_alive_p95_s
        requests = []
        for profile in sorted(self.price_list.machine_profiles, key=lambda price: price.name):
            if profile.name in measured:
                continue
            accelerator = profile.accelerator_per_hour * self.multiplier
            candidates = [(accelerator * self.window.warm_container_hours((), 1.0), Warmth.PINNED)]
            if bound != 0:
                cold = accelerator * self.window.cold_container_hours(choice.requests, None)
                candidates.append((cold, Warmth.COLD_WITH_BOUND))
            cost, warmth = min(candidates, key=lambda candidate: (candidate[0], candidate[1] is not Warmth.PINNED))
            saving = None if choice.chosen is None else choice.chosen.cost - cost
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

    def boot_requests(self, choice: _Choice) -> list[MeasurementRequest]:
        if choice.chosen is None:
            return []
        chosen = choice.chosen
        return [
            replace(request, saving_bound=chosen.cost - request.cost_bound)
            for request in choice.boot_requests
            if chosen.cost - request.cost_bound > 0
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
        )
        return planner.run(price_savings=False).report.monthly_cost

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

    def booted_within(self, identity: str, option: _Option, bound: int) -> tuple[BootEvidence, ...]:
        key = (identity, option.receipt.cell, option.receipt.container_shape, option.warmth)
        return tuple(
            replace(record, boot_seconds=min(record.boot_seconds, bound))
            if (record.identity, record.cell, record.container_shape, record.warmth) == key
            else record
            for record in self.boot_records
        )

    def levers(self, choice: _Choice, written: Plan, total: float) -> list[Lever]:
        chosen = choice.chosen
        bound = choice.level.come_alive_p95_s
        if chosen is None or bound is None:
            return []
        best: dict[tuple[Cell, ContainerShape, Warmth], _Option] = {}
        for option in choice.options:
            if option.warmth not in COLD_WARMTHS or option.cost >= chosen.cost:
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
                )
            )
        return levers

    def next_cheaper(self, choice: _Choice, written: Plan, total: float) -> Option | None:
        chosen = choice.chosen
        if chosen is None:
            return None
        cheaper = [option for option in choice.options if not option.feasible and option.cost < chosen.cost]
        if not cheaper:
            return None
        cost = max(option.cost for option in cheaper)
        nearest = min((option for option in cheaper if option.cost == cost), key=lambda option: option.rank)
        bounds = {
            blocker.field: _permitted(blocker.measured) for blocker in nearest.blockers if blocker.measured is not None
        }
        return nearest.summary(saving=self.saving_when_loosened(choice.identity, bounds, written, total))

    @staticmethod
    def lane_names(placed: Mapping[str, _Option], reserved: set[str]) -> dict[str, str]:
        bases = {
            identity: _slug(f"{identity}-{option.receipt.cell.machine_profile}") or "lane"
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
        if entry.warmth in COLD_WARMTHS and level.come_alive_p95_s is not None:
            boot = self.boots.get((identity, lane.cell, lane.container_shape, entry.warmth))
            if boot is not None and boot.seconds is not None:
                blocker = _come_alive_blocker(level, boot.seconds)
                blockers.extend(() if blocker is None else (blocker,))
            elif level.come_alive_p95_s == 0:
                blockers.append(Blocker(field="comeAliveP95S", bound=0, measured=None))
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

    def run(self, *, price_savings: bool = True) -> PlannerResult:
        choices = [self.choose(identity) for identity in sorted(self.levels)]
        placed = {choice.identity: choice.chosen for choice in choices if choice.chosen is not None}
        kept = {
            choice.identity: self.current[choice.identity]
            for choice in choices
            if choice.chosen is None and choice.identity in self.current
        }
        staying = {identity: self.current[identity][1].name for identity, option in placed.items() if option.current}
        reserved = {lane.name for _, lane in kept.values()} | set(staying.values())
        moving = {identity: option for identity, option in placed.items() if identity not in staying}
        names = staying | self.lane_names(moving, reserved)
        lanes = {lane.name: self.within_ceiling(lane) for _, lane in kept.values()}
        for identity, option in placed.items():
            lanes[names[identity]] = self.placed_lane(names[identity], identity, option)
        identities = [entry for entry, _ in kept.values()]
        identities.extend(
            PlanIdentity(identity=identity, lane=names[identity], warmth=option.warmth)
            for identity, option in placed.items()
        )
        new_plan = Plan(
            placement=self.parameters.placement,
            lanes=tuple(lanes[name] for name in sorted(lanes)),
            identities=tuple(sorted(identities, key=lambda entry: entry.identity)),
        )
        receipts: dict[str, Receipt | None] = {identity: option.receipt for identity, option in placed.items()}
        receipts.update({identity: self.slowest_receipt(identity, lane) for identity, (_, lane) in kept.items()})
        lane_reports, costs = self._pricing(new_plan, receipts)
        total = sum(lane.monthly_cost for lane in lane_reports if lane.monthly_cost is not None)
        reports: list[IdentityReport] = []
        requests: list[MeasurementRequest] = []
        levers: list[Lever] = []
        for choice in choices:
            requests.extend(self.receipt_requests(choice))
            requests.extend(self.boot_requests(choice))
            nearest = None
            if price_savings:
                levers.extend(self.levers(choice, new_plan, total))
                nearest = self.next_cheaper(choice, new_plan, total)
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
            measurement_requests=tuple(sorted(requests, key=_request_rank)),
            come_alive_levers=tuple(sorted(levers, key=_lever_rank)),
            queue_wait_unattained=tuple(
                identity for identity in sorted(self.levels) if self.levels[identity].queue_wait_p99_ms is not None
            ),
            refusals=tuple(sorted(self.refusals, key=_refusal_rank)),
            identity_changes=identity_changes,
            lane_changes=lane_changes,
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
            reason = UnplacedReason.WARM_BOUND_MISSED if choice.has_valid_receipt else UnplacedReason.NO_RECEIPT
            if choice.identity in kept:
                entry, lane = kept[choice.identity]
                return replace(
                    report,
                    status=IdentityStatus.KEPT,
                    lane=entry.lane,
                    monthly_cost=cost,
                    unplaced_reason=reason,
                    blocked_by=self.kept_blockers(choice.identity, entry, lane, receipt),
                )
            return replace(report, unplaced_reason=reason)
        rate = chosen.receipt.throughput.requests_per_second
        return replace(
            report,
            status=IdentityStatus.PLACED,
            lane=names[choice.identity],
            monthly_cost=cost,
            chosen=chosen.summary(),
            peak_containers=max((_Window.needed(count, rate) for count in choice.requests), default=0),
            next_cheaper=next_cheaper,
        )

    def _pricing(
        self, plan: Plan, receipts: Mapping[str, Receipt | None]
    ) -> tuple[tuple[LaneReport, ...], dict[str, float]]:
        residents: defaultdict[str, list[str]] = defaultdict(list)
        for entry in plan.identities:
            residents[entry.lane].append(entry.identity)
        hours = self.parameters.hours_per_month
        reports: list[LaneReport] = []
        costs: dict[str, float] = {}
        for lane in plan.lanes:
            hosted = tuple(sorted(residents[lane.name]))
            reason = _refusal_reason(lane.cell, self.prices)
            if reason is not None:
                reports.append(LaneReport(name=lane.name, identities=hosted, unpriced_reason=reason))
                continue
            hourly = container_hourly_cost(
                self.prices[lane.cell.machine_profile], lane.container_shape, self.multiplier
            )
            floor = lane.envelope.floor_min * hours
            demand = {identity: self.demand_hours(lane, identity, receipts.get(identity)) for identity in hosted}
            priced = {identity: value for identity, value in demand.items() if value is not None}
            costs.update((identity, hourly * (floor + value)) for identity, value in priced.items())
            reports.append(
                LaneReport(
                    name=lane.name,
                    identities=hosted,
                    hourly_cost=hourly,
                    monthly_cost=hourly * (floor + sum(priced.values())),
                    floor_monthly_cost=hourly * lane.envelope.floor_min * hours,
                    envelope_max_monthly_cost=hourly * lane.envelope.floor_max * hours,
                    unpriced_identities=tuple(identity for identity in hosted if identity not in priced),
                )
            )
        return tuple(reports), costs


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
) -> PlannerResult:
    return _Planner(service_levels, receipts, boot_evidence, price_list, demand, current_plan, parameters).run()
