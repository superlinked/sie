"""Keep each plan lane's warm floor inside its envelope through one gateway floor-carrier pool per lane."""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sie_placement.schema import Plan, PlanLane, Warmth
from sie_placement.tuner import Observation, Settings, Step, step

CARRIER_PREFIX = "placement-floor"
DEFAULT_QUEUE_POOL = "default"
DEFAULT_MAX_POOL_FLOOR = 4
DEFAULT_MAX_POOLS = 64
MAX_POOL_NAME_LENGTH = 128
PROMETHEUS_TIMEOUT_S = 10.0
SNAPSHOT_FRESHNESS_S = 20

QUEUE_DEPTH = "queueDepth"
PENDING_DEMAND = "pendingDemand"
WARM_FLOOR = "warmFloor"
RUNNERS = "runners"

UNREALIZED_WARMTHS = {
    Warmth.BRIDGED: "the overlay configures no remote bridge to serve it while the lane boots",
    Warmth.SNAPSHOT_COLD: "the chart restores no snapshot, so the lane would boot cold",
    Warmth.SCHEDULED_WARM: "the chart keeps no warm window, so the lane would stay cold inside it",
}

_LANE_TOKEN = re.compile(r"[a-z0-9_-]{1,63}")
_SELECTOR_FORBIDDEN = re.compile(r"[{}\r\n]")
_LANE_LABELS = "pool, machine_profile, bundle"
_PRODUCER_LABELS = "producer_instance, collector_generation"
_LANE_TOKEN_COUNT = 3
_VALUE_PAIR = 2


class PlanRefusedError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = tuple(problems)


class GatewayError(RuntimeError):
    pass


class PrometheusError(RuntimeError):
    pass


class CarrierConflictError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LaneKey:
    queue_pool: str
    machine_profile: str
    bundle: str

    @classmethod
    def canonical(cls, queue_pool: str, machine_profile: str, bundle: str) -> LaneKey | None:
        tokens = [value.strip().lower() for value in (queue_pool, machine_profile, bundle)]
        if any(_LANE_TOKEN.fullmatch(token) is None for token in tokens):
            return None
        return cls(queue_pool=tokens[0], machine_profile=tokens[1], bundle=tokens[2])

    @property
    def otel_lane(self) -> str:
        return f"{self.queue_pool}|{self.machine_profile}|{self.bundle}"

    @property
    def chart_lane(self) -> str:
        return f"{self.queue_pool}/{self.machine_profile}/{self.bundle}"


@dataclass(frozen=True, slots=True)
class Carrier:
    lane: PlanLane
    key: LaneKey

    @property
    def name(self) -> str:
        return f"{CARRIER_PREFIX}-{self.key.queue_pool}-{self.key.machine_profile}-{self.key.bundle}"

    @property
    def gpus(self) -> dict[str, int]:
        return {self.key.machine_profile: 0}


def carriers(
    plan: Plan, *, max_pool_floor: int = DEFAULT_MAX_POOL_FLOOR, max_pools: int = DEFAULT_MAX_POOLS
) -> tuple[Carrier, ...]:
    problems: list[str] = []
    if len(plan.lanes) > max_pools:
        problems.append(
            f"$.lanes: {len(plan.lanes)} lanes need as many floor-carrier pools, "
            f"above the gateway's limit of {max_pools} API pools"
        )
    result: list[Carrier] = []
    first_index: dict[str, int] = {}
    pins: dict[str, dict[str, int]] = {}
    for index, entry in enumerate(plan.identities):
        if entry.warmth is Warmth.PINNED:
            pins.setdefault(entry.lane, {})[entry.identity] = index
    for index, lane in enumerate(plan.lanes):
        path = f"$.lanes[{index}]"
        key = LaneKey.canonical(lane.queue_pool, lane.cell.machine_profile, lane.bundle)
        if key is None:
            problems.append(
                f"{path}: queuePool, cell.machineProfile and bundle must each be 1 to 63 letters, digits, '_' or '-'"
            )
            continue
        pinned = pins.get(lane.name)
        if key.queue_pool == DEFAULT_QUEUE_POOL and pinned:
            identity, pin = min(pinned.items(), key=lambda item: item[1])
            problems.append(
                f"$.identities[{pin}].warmth: pins {identity!r} on chart lane {key.chart_lane!r}, but the gateway "
                f"keeps no static queue pool for the {DEFAULT_QUEUE_POOL!r} queue pool, so its pinnedModels have no "
                "carrier; place pinned identities on a named queuePool"
            )
        if lane.envelope.floor_max > max_pool_floor:
            problems.append(
                f"{path}.envelope.floorMax: {lane.envelope.floor_max} exceeds the gateway's "
                f"per-pool floor limit of {max_pool_floor}"
            )
        carrier = Carrier(lane=lane, key=key)
        if len(carrier.name) > MAX_POOL_NAME_LENGTH:
            problems.append(
                f"{path}: floor-carrier pool name {carrier.name!r} is longer than {MAX_POOL_NAME_LENGTH} characters"
            )
        earlier = first_index.setdefault(carrier.name, index)
        if earlier != index:
            problems.append(f"{path}: needs the same floor-carrier pool {carrier.name!r} as $.lanes[{earlier}]")
        result.append(carrier)
    for index, entry in enumerate(plan.identities):
        reason = UNREALIZED_WARMTHS.get(entry.warmth)
        if reason is not None:
            problems.append(
                f"$.identities[{index}].warmth: is {entry.warmth.value!r} for {entry.identity!r} on lane "
                f"{entry.lane!r}, which the sie-cluster chart does not realize: {reason}; plan the identity pinned, "
                "lazy-resident or cold-with-bound"
            )
    if problems:
        raise PlanRefusedError(problems)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class Sample:
    labels: Mapping[str, str]
    value: float


class Prometheus(Protocol):
    def query(self, promql: str) -> Sequence[Sample]: ...


def _object(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    fields: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        fields[key] = item
    return fields


def parse_vector(document: object) -> tuple[Sample, ...]:
    response = _object(document)
    if response is None or response.get("status") != "success":
        raise PrometheusError("the query did not succeed")
    data = _object(response.get("data"))
    if data is None or data.get("resultType") != "vector":
        raise PrometheusError("the query did not return an instant vector")
    result = data.get("result")
    if not isinstance(result, list):
        raise PrometheusError("the query result is not a list")
    samples: list[Sample] = []
    for item in result:
        entry = _object(item) or {}
        metric = _object(entry.get("metric"))
        value = entry.get("value")
        if metric is None or not isinstance(value, list) or len(value) != _VALUE_PAIR:
            raise PrometheusError("the query returned a malformed sample")
        text = value[1]
        try:
            number = float(text) if isinstance(text, str) else None
        except ValueError:
            number = None
        if number is None:
            raise PrometheusError("the query returned a sample that is not a number")
        samples.append(Sample(labels={str(name): str(label) for name, label in metric.items()}, value=number))
    return tuple(samples)


def require_base_url(label: str, base_url: str) -> None:
    try:
        parts = urllib.parse.urlsplit(base_url)
        scheme, hostname, username, password = parts.scheme, parts.hostname, parts.username, parts.password
    except ValueError:
        raise ValueError(f"the {label} URL is not a valid URL") from None
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError(f"the {label} URL must be an absolute http or https URL")
    if username is not None or password is not None:
        raise ValueError(f"the {label} URL must not embed credentials")
    if parts.query or parts.fragment:
        raise ValueError(f"the {label} URL must not carry a query or a fragment")
    try:
        parts.port  # noqa: B018
    except ValueError:
        raise ValueError(f"the {label} URL has an invalid port") from None


class PrometheusHttp:
    def __init__(self, base_url: str, *, timeout_s: float = PROMETHEUS_TIMEOUT_S) -> None:
        require_base_url("Prometheus", base_url)
        self._endpoint = f"{base_url.rstrip('/')}/api/v1/query"
        self._timeout_s = timeout_s

    def query(self, promql: str) -> tuple[Sample, ...]:
        url = f"{self._endpoint}?{urllib.parse.urlencode({'query': promql})}"
        try:
            with urllib.request.urlopen(url, timeout=self._timeout_s) as response:  # noqa: S310
                document = json.load(response)
        except urllib.error.HTTPError as error:
            raise PrometheusError(f"the query failed with HTTP {error.code}") from None
        except (OSError, ValueError) as error:
            raise PrometheusError(f"the query failed: {error}") from None
        except (http.client.HTTPException, RecursionError) as error:
            raise PrometheusError(f"the query failed: {type(error).__name__}") from None
        return parse_vector(document)


def series_queries(selector: str = "") -> dict[str, str]:
    if _SELECTOR_FORBIDDEN.search(selector):
        raise ValueError("the series selector must be label matchers without braces or line breaks")

    def matchers(*extra: str) -> str:
        return ",".join(part for part in (selector.strip(), *extra) if part)

    def fresh(metric: str, timestamp: str, join_labels: str) -> str:
        return (
            f"max by ({_LANE_LABELS}) ({metric}{{{matchers()}}} and on({join_labels}) "
            f"(abs(time() - {timestamp}{{{matchers()}}}) < {SNAPSHOT_FRESHNESS_S}))"
        )

    capacity = "sie_gateway_capacity_snapshot_timestamp_seconds"
    return {
        QUEUE_DEPTH: fresh(
            "sie_gateway_lane_queue_depth",
            "sie_gateway_lane_queue_snapshot_timestamp_seconds",
            f"{_PRODUCER_LABELS}, {_LANE_LABELS}",
        ),
        PENDING_DEMAND: fresh("sie_gateway_pending_demand", capacity, _PRODUCER_LABELS),
        WARM_FLOOR: fresh("sie_gateway_pool_warm_floor", capacity, _PRODUCER_LABELS),
        RUNNERS: (
            "count by (lane) (max by (lane, producer_instance) "
            f"(sie_worker_gpu_slots{{{matchers('state="ready"')}}}) > 0)"
        ),
    }


def _sample_lane(labels: Mapping[str, str], *, otel_lane: bool) -> LaneKey | None:
    if otel_lane:
        tokens = labels.get("lane", "").split("|")
        if len(tokens) != _LANE_TOKEN_COUNT:
            return None
        return LaneKey.canonical(*tokens)
    return LaneKey.canonical(labels.get("pool", ""), labels.get("machine_profile", ""), labels.get("bundle", ""))


def _count(value: float) -> int | None:
    if not math.isfinite(value) or value < 0:
        return None
    return round(value)


def _lane_values(samples: Sequence[Sample], *, otel_lane: bool) -> dict[LaneKey, int]:
    values: dict[LaneKey, int] = {}
    for sample in samples:
        key = _sample_lane(sample.labels, otel_lane=otel_lane)
        count = _count(sample.value)
        if key is not None and count is not None:
            values[key] = max(count, values.get(key, 0))
    return values


@dataclass(frozen=True, slots=True, kw_only=True)
class LaneTelemetry:
    queue_depth: int | None = None
    pending_demand: int | None = None
    warm_floor: int | None = None
    runners: int | None = None

    def observation(self, current_floor: int) -> Observation:
        if self.queue_depth is None or self.pending_demand is None or self.runners is None:
            return Observation(runners=current_floor, backlog=0, running=1)
        return Observation(runners=self.runners, backlog=self.pending_demand, running=self.queue_depth)

    def to_dict(self) -> dict[str, object]:
        return {
            QUEUE_DEPTH: self.queue_depth,
            PENDING_DEMAND: self.pending_demand,
            WARM_FLOOR: self.warm_floor,
            RUNNERS: self.runners,
        }


@dataclass(frozen=True, slots=True)
class Telemetry:
    series: Mapping[str, Mapping[LaneKey, int] | None]
    errors: tuple[str, ...] = ()

    def _value(self, name: str, key: LaneKey, absent: int | None) -> int | None:
        values = self.series.get(name)
        if values is None:
            return None
        return values.get(key, absent)

    def lane(self, key: LaneKey) -> LaneTelemetry:
        return LaneTelemetry(
            queue_depth=self._value(QUEUE_DEPTH, key, None),
            pending_demand=self._value(PENDING_DEMAND, key, None),
            warm_floor=self._value(WARM_FLOOR, key, None),
            runners=self._value(RUNNERS, key, 0),
        )


def read_telemetry(prometheus: Prometheus, selector: str = "") -> Telemetry:
    series: dict[str, Mapping[LaneKey, int] | None] = {}
    errors: list[str] = []
    for name, promql in series_queries(selector).items():
        try:
            samples = prometheus.query(promql)
        except PrometheusError as error:
            errors.append(f"{name}: {error}")
            series[name] = None
            continue
        series[name] = _lane_values(samples, otel_lane=name == RUNNERS)
    return Telemetry(series=series, errors=tuple(errors))


class PoolApi(Protocol):
    def get_pool(self, name: str) -> Mapping[str, object] | None: ...

    def create_pool(
        self, name: str, gpus: dict[str, int], *, bundle: str, minimum_worker_count: int, queue_pool: str
    ) -> None: ...


def _matches(value: object, expected: str) -> bool:
    return isinstance(value, str) and value.strip().lower() == expected


def carrier_floor(carrier: Carrier, pool: Mapping[str, object]) -> int:
    spec = _object(pool.get("spec"))
    if spec is None:
        raise CarrierConflictError("it has no spec")
    if not _matches(spec.get("queue_pool", "default"), carrier.key.queue_pool):
        raise CarrierConflictError(f"it does not draw from queue pool {carrier.key.queue_pool!r}")
    if not _matches(spec.get("bundle"), carrier.key.bundle):
        raise CarrierConflictError(f"it is not filtered to bundle {carrier.key.bundle!r}")
    gpus = _object(spec.get("gpus"))
    if not (
        gpus is not None
        and len(gpus) == 1
        and all(
            _matches(profile, carrier.key.machine_profile) and count == 0 and not isinstance(count, bool)
            for profile, count in gpus.items()
        )
    ):
        raise CarrierConflictError(f"its gpus are not {carrier.gpus!r}")
    for field in ("gpu_caps", "pinned_models", "ttl_seconds"):
        if spec.get(field):
            raise CarrierConflictError(f"it sets {field}")
    floor = spec.get("minimum_worker_count", 0)
    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 0:
        raise CarrierConflictError("its minimum_worker_count is not a non-negative integer")
    return floor


@dataclass(frozen=True, slots=True, kw_only=True)
class LaneRecord:
    carrier: Carrier
    telemetry: LaneTelemetry
    current_floor: int | None = None
    step: Step | None = None
    applied: bool = False
    error: str | None = None

    @property
    def lane_floor_above_envelope(self) -> bool:
        warm_floor = self.telemetry.warm_floor
        return warm_floor is not None and warm_floor > self.carrier.lane.envelope.floor_max

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.carrier.lane.name,
            "queuePool": self.carrier.key.queue_pool,
            "machineProfile": self.carrier.key.machine_profile,
            "bundle": self.carrier.key.bundle,
            "carrier": self.carrier.name,
            "telemetry": self.telemetry.to_dict(),
            "currentFloor": self.current_floor,
            "action": None if self.step is None else self.step.action.value,
            "floor": None if self.step is None else self.step.settings.floor,
            "applied": self.applied,
            "laneFloorAboveEnvelope": self.lane_floor_above_envelope,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Tick:
    at: datetime
    apply: bool
    records: tuple[LaneRecord, ...]
    series_errors: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return bool(self.series_errors) or any(record.error is not None for record in self.records)

    def audit_rows(self) -> list[dict[str, object]]:
        at = self.at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        mode = "apply" if self.apply else "shadow"
        return [{"at": at, "mode": mode, **record.to_dict()} for record in self.records]


def _tune_lane(carrier: Carrier, telemetry: LaneTelemetry, reader: PoolApi, writer: PoolApi | None) -> LaneRecord:
    try:
        pool = reader.get_pool(carrier.name)
    except GatewayError as error:
        return LaneRecord(carrier=carrier, telemetry=telemetry, error=f"reading pool {carrier.name!r} failed: {error}")
    try:
        current = 0 if pool is None else carrier_floor(carrier, pool)
    except CarrierConflictError as error:
        return LaneRecord(
            carrier=carrier,
            telemetry=telemetry,
            error=f"pool {carrier.name!r} is not this lane's floor carrier ({error}) and is left unchanged",
        )
    result = step(carrier.lane.envelope, telemetry.observation(current), Settings(floor=current))
    record = LaneRecord(carrier=carrier, telemetry=telemetry, current_floor=current, step=result)
    if writer is None or (pool is None and result.settings.floor == 0):
        return record
    try:
        writer.create_pool(
            carrier.name,
            carrier.gpus,
            bundle=carrier.key.bundle,
            minimum_worker_count=result.settings.floor,
            queue_pool=carrier.key.queue_pool,
        )
    except GatewayError as error:
        return LaneRecord(
            carrier=carrier,
            telemetry=telemetry,
            current_floor=current,
            step=result,
            error=f"writing pool {carrier.name!r} failed: {error}",
        )
    return LaneRecord(carrier=carrier, telemetry=telemetry, current_floor=current, step=result, applied=True)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def tick(
    lanes: Sequence[Carrier],
    *,
    prometheus: Prometheus,
    reader: PoolApi,
    writer: PoolApi | None = None,
    selector: str = "",
    now: Callable[[], datetime] = _utc_now,
) -> Tick:
    at = now()
    telemetry = read_telemetry(prometheus, selector)
    records = tuple(_tune_lane(carrier, telemetry.lane(carrier.key), reader, writer) for carrier in lanes)
    return Tick(at=at, apply=writer is not None, records=records, series_errors=telemetry.errors)
