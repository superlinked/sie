"""Values for the sie-cluster Helm chart that realize a placement plan."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from sie_placement.schema import ContainerShape, Plan, PlanLane, SchemaError, Warmth

DEFAULT_QUEUE_POOL = "default"
RESERVED_POOL_NAME = "_default"
LANE_TOKEN_MAX_LENGTH = 63
EXTERNAL_DEPENDENCIES_LANE_LIMIT = 192
BUNDLED_DEPENDENCIES_LANE_LIMIT = 96
DEFAULT_FALLBACK_REPLICAS = 2
MILLICORES_PER_CORE = 1000

_LANE_TOKEN = re.compile(r"[A-Za-z0-9_-]+")
_IDENTITY_TOKEN = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?")
_GO_SPACE = (
    "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)


class UnrealizablePlanError(ValueError):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


@dataclass(frozen=True, slots=True, kw_only=True)
class ChartParameters:
    fallback_replicas: int = DEFAULT_FALLBACK_REPLICAS
    bundled_dependencies: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.fallback_replicas, bool) or self.fallback_replicas < 1:
            raise SchemaError("fallbackReplicas", f"must be at least 1, got {self.fallback_replicas!r}")

    @property
    def lane_limit(self) -> int:
        return BUNDLED_DEPENDENCIES_LANE_LIMIT if self.bundled_dependencies else EXTERNAL_DEPENDENCIES_LANE_LIMIT

    @property
    def dependency_topology(self) -> str:
        return "bundled KEDA or Prometheus" if self.bundled_dependencies else "external KEDA and Prometheus"

    def fallback_floor(self, min_replicas: int, max_replicas: int) -> int:
        return min(max_replicas, max(min_replicas, self.fallback_replicas))


def lane_token(path: str, value: str) -> str:
    raw = value.strip(_GO_SPACE)
    if not raw:
        raise UnrealizablePlanError(path, "value must not be empty or whitespace")
    if len(raw.encode()) > LANE_TOKEN_MAX_LENGTH:
        raise UnrealizablePlanError(
            path, f"value {raw!r} exceeds the {LANE_TOKEN_MAX_LENGTH}-character physical lane limit"
        )
    if _LANE_TOKEN.fullmatch(raw) is None:
        raise UnrealizablePlanError(path, f"value {raw!r} must match ^[A-Za-z0-9_-]+$")
    return raw.lower()


def _require_identity_token(path: str, role: str, value: str) -> None:
    if _IDENTITY_TOKEN.fullmatch(value) is None:
        raise UnrealizablePlanError(path, f"{role} {value!r} must be a lowercase Kubernetes DNS-1123 label")


def _cpu_quantity(path: str, cores: float) -> str:
    millicores = round(cores * MILLICORES_PER_CORE)
    if not math.isclose(millicores, cores * MILLICORES_PER_CORE, rel_tol=0, abs_tol=1e-6) or millicores < 1:
        raise UnrealizablePlanError(path, f"must be a whole number of millicores, got {cores!r}")
    if millicores % MILLICORES_PER_CORE == 0:
        return str(millicores // MILLICORES_PER_CORE)
    return f"{millicores}m"


def _resources(path: str, shape: ContainerShape) -> dict[str, str]:
    return {"cpu": _cpu_quantity(f"{path}.cpuCores", shape.cpu_cores), "memory": f"{shape.memory_mib}Mi"}


@dataclass(slots=True, kw_only=True)
class _ChartLane:
    index: int
    lane: PlanLane
    queue_pool: str
    machine_profile: str
    bundle: str
    pins: dict[str, int] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return f"$.lanes[{self.index}]"

    @property
    def name(self) -> str:
        return f"{self.queue_pool}/{self.machine_profile}/{self.bundle}"


def _pins_by_lane(plan: Plan) -> dict[str, dict[str, int]]:
    pins: dict[str, dict[str, int]] = defaultdict(dict)
    for index, entry in enumerate(plan.identities):
        if entry.warmth is Warmth.PINNED:
            pins[entry.lane][entry.identity] = index
    return pins


def _require_realizable_floor(path: str, lane: PlanLane, parameters: ChartParameters) -> None:
    envelope = lane.envelope
    if envelope.floor_min or not envelope.floor_max:
        return
    floor = parameters.fallback_floor(0, lane.max_containers)
    if envelope.floor_max < floor:
        raise UnrealizablePlanError(
            f"{path}.envelope.floorMax",
            f"must be 0 or at least {floor}: a lane with floorMin 0 realizes a raised floor as at least the "
            f"chart's fallback floor, min(maxContainers, fallbackReplicas), got {envelope.floor_max}",
        )


def _require_same_chart_lane(path: str, lane: PlanLane, first: _ChartLane) -> None:
    other = first.lane
    differences = (
        ("cell.gpuProduct", lane.cell.gpu_product, other.cell.gpu_product),
        ("containerShape", lane.container_shape.to_dict(), other.container_shape.to_dict()),
        ("maxContainers", lane.max_containers, other.max_containers),
        ("envelope.floorMin", lane.envelope.floor_min, other.envelope.floor_min),
    )
    for key, value, expected in differences:
        if value != expected:
            raise UnrealizablePlanError(
                f"{path}.{key}",
                f"names chart lane {first.name!r} with {first.path}, which has {expected!r}, got {value!r}; "
                "one chart lane carries one value",
            )


def _chart_lanes(plan: Plan, parameters: ChartParameters) -> list[_ChartLane]:
    pins = _pins_by_lane(plan)
    lanes: dict[tuple[str, str, str], _ChartLane] = {}
    for index, lane in enumerate(plan.lanes):
        path = f"$.lanes[{index}]"
        key = (
            lane_token(f"{path}.queuePool", lane.queue_pool),
            lane_token(f"{path}.cell.machineProfile", lane.cell.machine_profile),
            lane_token(f"{path}.bundle", lane.bundle),
        )
        _require_identity_token(f"{path}.cell.machineProfile", "worker group key", key[1])
        _require_identity_token(f"{path}.bundle", "bundle key", key[2])
        _require_realizable_floor(path, lane, parameters)
        chart_lane = lanes.get(key)
        if chart_lane is None:
            chart_lane = _ChartLane(index=index, lane=lane, queue_pool=key[0], machine_profile=key[1], bundle=key[2])
            lanes[key] = chart_lane
        else:
            _require_same_chart_lane(path, lane, chart_lane)
        chart_lane.pins.update(pins.get(lane.name, {}))
    if len(lanes) > parameters.lane_limit:
        raise UnrealizablePlanError(
            "$.lanes",
            f"names {len(lanes)} chart lanes, above the {parameters.lane_limit} the sie-cluster chart admits "
            f"with {parameters.dependency_topology}",
        )
    return list(lanes.values())


def _worker_group(lanes: list[_ChartLane]) -> dict[str, object]:
    first = lanes[0]
    for other in lanes[1:]:
        differences = (
            ("queuePool", other.queue_pool, first.queue_pool),
            ("cell.gpuProduct", other.lane.cell.gpu_product, first.lane.cell.gpu_product),
            ("containerShape", other.lane.container_shape.to_dict(), first.lane.container_shape.to_dict()),
        )
        for key, value, expected in differences:
            if value != expected:
                raise UnrealizablePlanError(
                    f"{other.path}.{key}",
                    f"shares worker group {first.machine_profile!r} with {first.path}, which has {expected!r}, "
                    f"got {value!r}; one worker group per machine profile carries one value",
                )
    shape = first.lane.container_shape
    return {
        "enabled": True,
        "machineProfile": first.machine_profile,
        "queuePool": first.queue_pool,
        "gpu": {"count": shape.gpu_count, "product": first.lane.cell.gpu_product},
        "resources": {
            "requests": _resources(f"{first.path}.containerShape", shape),
            "limits": _resources(f"{first.path}.containerShape", shape),
        },
        "bundles": {
            lane.bundle: {
                "enabled": True,
                "minReplicas": lane.lane.envelope.floor_min,
                "maxReplicas": lane.lane.max_containers,
            }
            for lane in sorted(lanes, key=lambda lane: lane.bundle)
        },
    }


def _static_queue_pool(lanes: list[_ChartLane]) -> dict[str, object]:
    first = lanes[0]
    if first.queue_pool == RESERVED_POOL_NAME:
        raise UnrealizablePlanError(
            f"{first.path}.queuePool",
            f"{RESERVED_POOL_NAME!r} is reserved by the gateway and cannot name the lane's static queue pool",
        )
    for other in lanes[1:]:
        if other.lane.envelope.floor_min != first.lane.envelope.floor_min:
            raise UnrealizablePlanError(
                f"{other.path}.envelope.floorMin",
                f"shares static queue pool {first.queue_pool!r} with {first.path}, whose floorMin is "
                f"{first.lane.envelope.floor_min}, got {other.lane.envelope.floor_min}; a static queue pool "
                "carries one minimumWorkerCount for every lane on it",
            )
        if set(other.pins) != set(first.pins):
            raise UnrealizablePlanError(
                f"{other.path}.name",
                f"shares static queue pool {first.queue_pool!r} with {first.path}, which pins "
                f"{sorted(first.pins)!r}, but pins {sorted(other.pins)!r}; a static queue pool pins one set on "
                "every lane on it",
            )
    bundles = sorted({lane.bundle for lane in lanes})
    pool: dict[str, object] = {}
    if len(bundles) == 1:
        pool["bundle"] = bundles[0]
    elif first.lane.envelope.floor_min or first.pins:
        other = next(lane for lane in lanes if lane.bundle != first.bundle)
        raise UnrealizablePlanError(
            f"{other.path}.bundle",
            f"shares static queue pool {first.queue_pool!r} with {first.path}, whose bundle is "
            f"{first.bundle!r}, got {other.bundle!r}; a static queue pool carries a floor or pins for one bundle",
        )
    pool["gpus"] = dict.fromkeys(sorted({lane.machine_profile for lane in lanes}), 0)
    pool["minimumWorkerCount"] = first.lane.envelope.floor_min
    pool["pinnedModels"] = sorted(first.pins)
    return pool


def _require_no_default_pool_pins(lanes: Iterable[_ChartLane]) -> None:
    for lane in lanes:
        if lane.pins:
            identity, index = min(lane.pins.items(), key=lambda item: item[1])
            raise UnrealizablePlanError(
                f"$.identities[{index}].warmth",
                f"pins {identity!r} on chart lane {lane.name!r}, but the gateway keeps no static queue pool for "
                f"the {DEFAULT_QUEUE_POOL!r} queue pool, so its pinnedModels have no carrier; place pinned "
                "identities on a named queuePool",
            )


def helm_values(plan: Plan, parameters: ChartParameters | None = None) -> dict[str, object]:
    parameters = ChartParameters() if parameters is None else parameters
    lanes = _chart_lanes(plan, parameters)
    by_profile: dict[str, list[_ChartLane]] = defaultdict(list)
    by_queue_pool: dict[str, list[_ChartLane]] = defaultdict(list)
    for lane in lanes:
        by_profile[lane.machine_profile].append(lane)
        by_queue_pool[lane.queue_pool].append(lane)
    pools = {profile: _worker_group(by_profile[profile]) for profile in sorted(by_profile)}
    static_queue_pools: dict[str, object] = {}
    for queue_pool in sorted(by_queue_pool):
        if queue_pool == DEFAULT_QUEUE_POOL:
            _require_no_default_pool_pins(by_queue_pool[queue_pool])
            continue
        static_queue_pools[queue_pool] = _static_queue_pool(by_queue_pool[queue_pool])
    values: dict[str, object] = {"workers": {"pools": pools}}
    if static_queue_pools:
        values["queueRouting"] = {"staticQueuePools": static_queue_pools}
    return values
