from __future__ import annotations

import json
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sie_placement import planner
from sie_placement.helm import (
    BUNDLED_DEPENDENCIES_LANE_LIMIT,
    EXTERNAL_DEPENDENCIES_LANE_LIMIT,
    UNREALIZED_WARMTHS,
    ChartParameters,
    UnrealizablePlanError,
    chart_pools,
    helm_values,
    lane_token,
    plan_refusals,
)
from sie_placement.report import render
from sie_placement.schema import (
    WARM_FLOOR_WARMTHS,
    BootEvidence,
    BootOutcome,
    Cell,
    ContainerShape,
    Demand,
    Envelope,
    IdentityServiceLevel,
    MachineProfilePrice,
    Operation,
    Percentiles,
    Placement,
    Plan,
    PlanIdentity,
    PlanLane,
    PriceList,
    Receipt,
    RequestShape,
    SchemaError,
    ServiceLevel,
    ServiceLevels,
    SourceRef,
    Throughput,
    Warmth,
)

FIXTURES = Path(__file__).parent / "fixtures"
HELM_FIXTURES = FIXTURES / "helm"
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
WIDE = ContainerShape(gpu_count=1, cpu_cores=8, memory_mib=32768)
PAIR = ContainerShape(gpu_count=2, cpu_cores=16, memory_mib=65536)
GOLDEN_PARAMETERS = ChartParameters(device_groups={"h100": True})
PLANNER_PARAMETERS = planner.PlannerParameters(
    placement="us-central1",
    max_containers=4,
    fallback_floor=2,
    cooldown_s=900,
    buffer_containers=1,
    bundles={"Qwen/Qwen3-8B": "sglang"},
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lane(
    name: str,
    *,
    queue_pool: str = "placement",
    bundle: str = "default",
    cell: Cell = L4,
    shape: ContainerShape = SHAPE,
    max_containers: int = 4,
    floor_min: int = 1,
    floor_max: int | None = None,
) -> PlanLane:
    return PlanLane(
        name=name,
        queue_pool=queue_pool,
        bundle=bundle,
        cell=cell,
        container_shape=shape,
        concurrency=8,
        max_containers=max_containers,
        envelope=Envelope(floor_min=floor_min, floor_max=floor_min if floor_max is None else floor_max),
    )


def plan_of(*lanes: PlanLane, warmths: dict[str, Warmth] | None = None) -> Plan:
    warmths = warmths or {}
    identities = tuple(
        PlanIdentity(
            identity=f"org/{entry.name}",
            lane=entry.name,
            warmth=warmths.get(entry.name, Warmth.PINNED if entry.envelope.floor_min else Warmth.COLD_WITH_BOUND),
        )
        for entry in lanes
    )
    return Plan(placement="us-central1", lanes=lanes, identities=identities)


def refusal(plan: Plan, parameters: ChartParameters | None = None) -> UnrealizablePlanError:
    with pytest.raises(UnrealizablePlanError) as caught:
        helm_values(plan, parameters)
    return caught.value


def golden_plan() -> Plan:
    return Plan.from_dict(load(HELM_FIXTURES / "plan.json"))


def golden_values() -> dict:
    return load(HELM_FIXTURES / "values.json")


def test_golden_plan_emits_the_golden_values() -> None:
    assert render(helm_values(golden_plan(), GOLDEN_PARAMETERS)) == (HELM_FIXTURES / "values.json").read_text(
        encoding="utf-8"
    )


def test_values_do_not_depend_on_document_order() -> None:
    plan = golden_plan()
    expected = (HELM_FIXTURES / "values.json").read_text(encoding="utf-8")
    shuffle = random.Random(5)  # noqa: S311 -- reproducible test data
    for _ in range(20):
        lanes = list(plan.lanes)
        identities = list(plan.identities)
        shuffle.shuffle(lanes)
        shuffle.shuffle(identities)
        shuffled = replace(plan, lanes=tuple(lanes), identities=tuple(identities))
        assert render(helm_values(shuffled, GOLDEN_PARAMETERS)) == expected


def test_the_default_base_is_the_charts_own_worker_groups() -> None:
    assert ChartParameters().base_pools == chart_pools() == load(HELM_FIXTURES / "chart-pools.json")
    assert chart_pools() is not chart_pools()


def test_an_emitted_worker_group_disables_the_base_bundles_the_plan_does_not_place() -> None:
    bundles = golden_values()["workers"]["pools"]["a100-40gb"]["bundles"]
    assert bundles == {"default": {"enabled": False}, "sglang": {"enabled": True, "minReplicas": 0, "maxReplicas": 2}}
    base = chart_pools() | {
        "l4": {"enabled": False, "bundles": {"default": {}, "sglang": None, "old": {"enabled": False}}},
    }
    values = helm_values(plan_of(lane("a", bundle="candle")), ChartParameters(base_pools=base))
    assert values["workers"]["pools"]["l4"]["bundles"] == {
        "candle": {"enabled": True, "minReplicas": 1, "maxReplicas": 4},
        "default": {"enabled": False},
        "sglang": {"enabled": False},
    }


@pytest.mark.parametrize(
    ("candidate", "path", "message"),
    [
        (
            plan_of(
                lane("a"), lane("b", queue_pool="PLACEMENT", bundle="Default", cell=replace(L4, machine_profile="L4"))
            ),
            "$.lanes[1]",
            "names chart lane 'placement/l4/default', which $.lanes[0] also names;",
        ),
        (
            plan_of(lane("a", queue_pool="default", floor_min=0), lane("b", queue_pool="default", floor_min=0)),
            "$.lanes[1]",
            "names chart lane 'default/l4/default', which $.lanes[0] also names;",
        ),
        (
            plan_of(lane("a", bundle="sglang"), lane("b"), lane("c", bundle="SGLang")),
            "$.lanes[2]",
            "names chart lane 'placement/l4/sglang', which $.lanes[0] also names;",
        ),
    ],
)
def test_two_plan_lanes_on_one_chart_lane_are_refused(candidate: Plan, path: str, message: str) -> None:
    error = refusal(candidate)
    assert error.path == path
    assert error.message.startswith(message)
    assert error.message.endswith("so each plan lane needs its own queuePool, machineProfile and bundle")


@pytest.mark.parametrize(
    ("raw", "token"),
    [
        ("l4", "l4"),
        (" L4 ", "l4"),
        ("\tA100_80GB\n", "a100_80gb"),
        ("\u3000Default\xa0", "default"),
        ("\u2028q-pool\u205f", "q-pool"),
        ("a" * 63, "a" * 63),
    ],
)
def test_lane_token_canonicalizes_like_the_chart(raw: str, token: str) -> None:
    assert lane_token("$.x", raw) == token


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (" \u3000", "value must not be empty or whitespace"),
        ("a" * 64, "exceeds the 63-character physical lane limit"),
        ("l4.x", "must match ^[A-Za-z0-9_-]+$"),
        ("\x1cl4", "must match ^[A-Za-z0-9_-]+$"),
        ("\u01424", "must match ^[A-Za-z0-9_-]+$"),
        ("\u0142" * 32, "exceeds the 63-character physical lane limit"),
    ],
)
def test_lane_token_refuses_what_the_chart_refuses(raw: str, message: str) -> None:
    with pytest.raises(UnrealizablePlanError, match=r"^\$\.x: ") as caught:
        lane_token("$.x", raw)
    assert message in caught.value.message


@pytest.mark.parametrize(
    ("changed", "path", "message"),
    [
        ({"cell": Cell(machine_profile="l4_spot", gpu_product="NVIDIA-L4")}, "cell.machineProfile", "worker group key"),
        ({"cell": Cell(machine_profile="-l4", gpu_product="NVIDIA-L4")}, "cell.machineProfile", "worker group key"),
        ({"bundle": "sglang_cu130"}, "bundle", "bundle key"),
        ({"bundle": "sglang-"}, "bundle", "bundle key"),
    ],
)
def test_worker_group_and_bundle_keys_are_dns_labels(changed: dict, path: str, message: str) -> None:
    error = refusal(plan_of(replace(lane("a"), **changed)))
    assert error.path == f"$.lanes[0].{path}"
    assert f"{message} " in error.message
    assert "must be a lowercase Kubernetes DNS-1123 label" in error.message


def test_zero_floor_lane_refuses_a_raised_floor_below_the_fallback_floor() -> None:
    error = refusal(plan_of(lane("a", queue_pool="default", floor_min=0, floor_max=1)))
    assert error.path == "$.lanes[0].envelope.floorMax"
    assert error.message.startswith("must be 0 or at least 2:")
    error = refusal(
        plan_of(lane("a", queue_pool="default", floor_min=0, floor_max=2)), ChartParameters(fallback_replicas=3)
    )
    assert error.message.startswith("must be 0 or at least 3:")


@pytest.mark.parametrize(
    ("floor_max", "max_containers", "fallback_replicas"),
    [(0, 4, 2), (2, 4, 2), (4, 4, 2), (1, 4, 1), (1, 1, 2), (2, 2, 5)],
)
def test_zero_floor_lane_accepts_a_realizable_floor(
    floor_max: int, max_containers: int, fallback_replicas: int
) -> None:
    plan = plan_of(lane("a", queue_pool="default", floor_min=0, floor_max=floor_max, max_containers=max_containers))
    values = helm_values(plan, ChartParameters(fallback_replicas=fallback_replicas))
    assert values["workers"]["pools"]["l4"]["bundles"]["default"] == {
        "enabled": True,
        "minReplicas": 0,
        "maxReplicas": max_containers,
    }


def test_schema_fixture_plan_has_fallback_cells_and_an_unrealizable_zero_floor() -> None:
    plan = Plan.from_dict(load(FIXTURES / "plan.json"))
    assert refusal(plan).path == "$.lanes[1].fallbackCells"
    error = refusal(replace(plan, lanes=tuple(replace(entry, fallback_cells=()) for entry in plan.lanes)))
    assert error.path == "$.lanes[1].envelope.floorMax"
    assert error.message.startswith("must be 0 or at least 2:")


def test_planner_estate_plan_puts_two_queue_pools_on_one_worker_group() -> None:
    error = refusal(Plan.from_dict(load(FIXTURES / "planner" / "expected-plan.json")))
    assert error.path == "$.lanes[1].queuePool"
    assert error.message.startswith(
        "shares worker group 'l4' with $.lanes[0], which has 'default', got 'default-l4-default';"
    )


def planner_estate(identities: set[str], parameters: planner.PlannerParameters) -> planner.PlannerResult:
    service_levels = ServiceLevels.from_dict(load(FIXTURES / "planner" / "service-levels.json"))
    receipts = [Receipt.from_dict(document) for document in load(FIXTURES / "planner" / "receipts.json")]
    boots = [BootEvidence.from_dict(document) for document in load(FIXTURES / "planner" / "boot-evidence.json")]
    return planner.plan(
        replace(service_levels, identities=tuple(e for e in service_levels.identities if e.identity in identities)),
        [entry for entry in receipts if entry.identity in identities],
        [entry for entry in boots if entry.identity in identities],
        PriceList.from_dict(load(FIXTURES / "planner" / "price-list.json")),
        Demand.from_dict(load(FIXTURES / "planner" / "demand.json")),
        None,
        parameters,
    )


def test_a_planner_plan_with_one_lane_per_chart_lane_emits_its_lanes() -> None:
    result = planner_estate({"BAAI/bge-m3:sparse", "Qwen/Qwen3-8B"}, PLANNER_PARAMETERS)
    assert {entry.warmth for entry in result.plan.identities} == {Warmth.COLD_WITH_BOUND}
    values = helm_values(result.plan, ChartParameters(fallback_replicas=PLANNER_PARAMETERS.fallback_floor))
    emitted = {
        (group["queuePool"], profile, bundle): (settings["minReplicas"], settings["maxReplicas"])
        for profile, group in values["workers"]["pools"].items()
        for bundle, settings in group["bundles"].items()
        if settings["enabled"]
    }
    assert emitted == {
        (entry.queue_pool, entry.cell.machine_profile, entry.bundle): (entry.envelope.floor_min, entry.max_containers)
        for entry in result.plan.lanes
    }
    assert len(emitted) == 2
    assert "queueRouting" not in values


@pytest.mark.parametrize("queue_pool", ["default", "placement"])
def test_a_planner_plan_that_pins_beside_another_lane_emits_the_pinned_lane_on_its_own_queue_pool(
    queue_pool: str,
) -> None:
    result = planner_estate({"BAAI/bge-m3", "Qwen/Qwen3-8B"}, replace(PLANNER_PARAMETERS, queue_pool=queue_pool))
    assert [entry.warmth for entry in result.plan.identities] == [Warmth.PINNED, Warmth.COLD_WITH_BOUND]
    pinned, cold = (entry.lane for entry in result.plan.identities)
    assert {entry.name: entry.queue_pool for entry in result.plan.lanes} == {pinned: pinned, cold: queue_pool}
    values = helm_values(result.plan, ChartParameters(fallback_replicas=PLANNER_PARAMETERS.fallback_floor))
    assert static_pools(values)[pinned]["pinnedModels"] == ["BAAI/bge-m3"]
    assert set(static_pools(values)) == {pinned} | ({queue_pool} - {"default"})


def bundle_lanes(count: int) -> Plan:
    return plan_of(
        *(lane(f"lane-{index}", queue_pool="default", bundle=f"b{index}", floor_min=0) for index in range(count))
    )


def base_lanes(count: int, *, enabled: bool = True, more: dict | None = None) -> dict[str, object]:
    bundles: dict[str, dict] = {f"b{index}": {} for index in range(count)}
    return chart_pools() | {"lanes": {"enabled": enabled, "bundles": bundles | (more or {})}}


@pytest.mark.parametrize(
    ("parameters", "limit", "topology"),
    [
        (ChartParameters(), EXTERNAL_DEPENDENCIES_LANE_LIMIT, "external KEDA and Prometheus"),
        (ChartParameters(bundled_dependencies=True), BUNDLED_DEPENDENCIES_LANE_LIMIT, "bundled KEDA or Prometheus"),
    ],
)
def test_chart_lane_count_is_bounded_by_the_chart_limit(parameters: ChartParameters, limit: int, topology: str) -> None:
    assert (EXTERNAL_DEPENDENCIES_LANE_LIMIT, BUNDLED_DEPENDENCIES_LANE_LIMIT) == (192, 96)
    assert parameters.lane_limit == limit
    bundles = helm_values(bundle_lanes(limit), parameters)["workers"]["pools"]["l4"]["bundles"]
    assert bundles.pop("default") == {"enabled": False}
    assert len(bundles) == limit
    assert all(settings["enabled"] for settings in bundles.values())
    error = refusal(bundle_lanes(limit + 1), parameters)
    assert error.path == "$.lanes"
    assert error.message == (
        f"names {limit + 1} chart lanes, above the {limit} the sie-cluster chart admits with {topology}"
    )


@pytest.mark.parametrize(
    ("bundled_dependencies", "limit", "topology"),
    [
        (False, EXTERNAL_DEPENDENCIES_LANE_LIMIT, "external KEDA and Prometheus"),
        (True, BUNDLED_DEPENDENCIES_LANE_LIMIT, "bundled KEDA or Prometheus"),
    ],
)
def test_lanes_the_base_values_enable_count_toward_the_chart_limit(
    bundled_dependencies: bool, limit: int, topology: str
) -> None:
    planned = sum(
        settings["enabled"]
        for group in golden_values()["workers"]["pools"].values()
        for settings in group["bundles"].values()
    )
    assert planned == len(golden_plan().lanes)
    accepted = replace(
        GOLDEN_PARAMETERS, bundled_dependencies=bundled_dependencies, base_pools=base_lanes(limit - planned)
    )
    assert helm_values(golden_plan(), accepted) == golden_values()
    refused = replace(accepted, base_pools=base_lanes(limit - planned + 1))
    error = refusal(golden_plan(), refused)
    assert error.path == "$.lanes"
    assert error.message == (
        f"names {planned} chart lanes and the base values enable {limit - planned + 1} more, {limit + 1} in all, "
        f"above the {limit} the sie-cluster chart admits with {topology}"
    )


def test_only_enabled_lanes_of_worker_groups_the_overlay_leaves_out_count_toward_the_chart_limit() -> None:
    limit = EXTERNAL_DEPENDENCIES_LANE_LIMIT
    disabled = replace(GOLDEN_PARAMETERS, base_pools=base_lanes(limit, enabled=False))
    assert helm_values(golden_plan(), disabled) == golden_values()
    l4 = {"l4": {"enabled": True, "bundles": {"default": {}, "extra": {}}}}
    one = plan_of(lane("a", queue_pool="default", floor_min=0))
    base = base_lanes(limit - 1, more={"off": {"enabled": False}}) | l4
    assert helm_values(one, ChartParameters(base_pools=base))["workers"]["pools"]["l4"]["bundles"] == {
        "default": {"enabled": True, "minReplicas": 0, "maxReplicas": 4},
        "extra": {"enabled": False},
    }
    base = base_lanes(limit - 1, more={"off": {"enabled": False}, "on": {"enabled": True}}) | l4
    assert refusal(one, ChartParameters(base_pools=base)).message.startswith(
        f"names 1 chart lanes and the base values enable {limit} more, {limit + 1} in all"
    )


@pytest.mark.parametrize(
    ("base_pools", "path"),
    [
        ([], "basePools"),
        ({"l4": []}, "basePools['l4']"),
        ({1: {}}, "basePools[1]"),
        ({"l4": {"bundles": []}}, "basePools['l4'].bundles"),
        ({"l4": {"bundles": {"default": []}}}, "basePools['l4'].bundles['default']"),
    ],
)
def test_base_pools_must_be_worker_group_values(base_pools: Any, path: str) -> None:
    with pytest.raises(SchemaError) as caught:
        ChartParameters(base_pools=base_pools)
    assert caught.value.path == path


def test_pins_on_the_default_queue_pool_are_refused() -> None:
    plan = plan_of(lane("a", queue_pool="default"), lane("b", queue_pool="DEFAULT", bundle="sglang"))
    error = refusal(plan)
    assert error.path == "$.identities[0].warmth"
    assert "pins 'org/a' on chart lane 'default/l4/default'" in error.message
    assert "place pinned identities on a named queuePool" in error.message


def test_a_warm_lane_without_pins_on_the_default_queue_pool_keeps_its_floor() -> None:
    plan = plan_of(lane("a", queue_pool="default", floor_min=2, floor_max=3), warmths={"a": Warmth.LAZY_RESIDENT})
    values = helm_values(plan)
    assert values == {
        "workers": {
            "pools": {
                "l4": {
                    "enabled": True,
                    "machineProfile": "l4",
                    "queuePool": "default",
                    "gpu": {"count": 1, "deviceGroup": False, "product": "NVIDIA-L4"},
                    "resources": {
                        "requests": {"cpu": "4", "memory": "16384Mi"},
                        "limits": {"cpu": "4", "memory": "16384Mi"},
                    },
                    "bundles": {"default": {"enabled": True, "minReplicas": 2, "maxReplicas": 4}},
                }
            }
        }
    }


def test_a_named_queue_pool_carries_one_floor_for_every_lane_on_it() -> None:
    warmths = {"a": Warmth.LAZY_RESIDENT, "b": Warmth.LAZY_RESIDENT}
    plan = plan_of(lane("a"), lane("b", cell=A100, shape=WIDE), warmths=warmths)
    assert helm_values(plan)["queueRouting"] == {
        "staticQueuePools": {
            "placement": {
                "bundle": "default",
                "gpus": {"a100-80gb": 0, "l4": 0},
                "minimumWorkerCount": 1,
                "pinnedModels": [],
            }
        }
    }


def test_a_named_queue_pool_without_floor_or_pins_admits_every_bundle() -> None:
    plan = plan_of(lane("a", floor_min=0), lane("b", bundle="sglang", cell=A100, shape=WIDE, floor_min=0))
    assert helm_values(plan)["queueRouting"]["staticQueuePools"] == {
        "placement": {"gpus": {"a100-80gb": 0, "l4": 0}, "minimumWorkerCount": 0, "pinnedModels": []}
    }


@pytest.mark.parametrize(
    ("second", "warmths", "path", "message"),
    [
        (lane("b", cell=A100, floor_min=0), {}, "envelope.floorMin", "carries one minimumWorkerCount"),
        (lane("b", cell=A100), {"b": Warmth.LAZY_RESIDENT}, "name", "pins one set on every lane"),
        (
            lane("b", cell=A100, bundle="sglang"),
            {"a": Warmth.LAZY_RESIDENT, "b": Warmth.LAZY_RESIDENT},
            "bundle",
            "carries a floor or pins for one bundle",
        ),
    ],
)
def test_lanes_on_one_static_queue_pool_must_agree(
    second: PlanLane, warmths: dict[str, Warmth], path: str, message: str
) -> None:
    error = refusal(plan_of(lane("a"), second, warmths=warmths))
    assert error.path == f"$.lanes[1].{path}"
    assert error.message.startswith("shares static queue pool 'placement' with $.lanes[0]")
    assert message in error.message


def test_two_pinned_lanes_need_two_named_queue_pools() -> None:
    error = refusal(plan_of(lane("a"), lane("b", cell=A100, shape=WIDE)))
    assert error.path == "$.lanes[1].name"
    assert "which pins ['org/a'], but pins ['org/b']" in error.message


@pytest.mark.parametrize(
    ("changed", "path"),
    [
        ({"queue_pool": "other"}, "queuePool"),
        ({"cell": Cell(machine_profile="l4", gpu_product="NVIDIA-L4-24GB")}, "cell.gpuProduct"),
        ({"container_shape": WIDE}, "containerShape"),
    ],
)
def test_lanes_on_one_machine_profile_share_one_worker_group(changed: dict, path: str) -> None:
    error = refusal(plan_of(lane("a", floor_min=0), replace(lane("b", bundle="sglang", floor_min=0), **changed)))
    assert error.path == f"$.lanes[1].{path}"
    assert error.message.startswith("shares worker group 'l4' with $.lanes[0]")


@pytest.mark.parametrize(
    ("cores", "quantity"),
    [(4, "4"), (0.5, "500m"), (7.5, "7500m"), (2.3, "2300m"), (0.001, "1m"), (12.25, "12250m")],
)
def test_cpu_cores_render_as_kubernetes_quantities(cores: float, quantity: str) -> None:
    shape = ContainerShape(gpu_count=1, cpu_cores=cores, memory_mib=1536)
    group = helm_values(plan_of(lane("a", queue_pool="default", floor_min=0, shape=shape)))["workers"]["pools"]["l4"]
    assert group["gpu"] == {"count": 1, "deviceGroup": False, "product": "NVIDIA-L4"}
    assert group["resources"]["requests"] == group["resources"]["limits"] == {"cpu": quantity, "memory": "1536Mi"}


@pytest.mark.parametrize("device_group", [True, False])
def test_a_multi_gpu_container_is_the_device_group_the_parameters_name(device_group: bool) -> None:
    plan = plan_of(lane("a", queue_pool="default", floor_min=0, shape=PAIR), lane("b", cell=A100, shape=PAIR))
    values = helm_values(plan, ChartParameters(device_groups={"l4": device_group, "a100-80gb": not device_group}))
    pools = values["workers"]["pools"]
    assert pools["l4"]["gpu"] == {"count": 2, "deviceGroup": device_group, "product": "NVIDIA-L4"}
    assert pools["a100-80gb"]["gpu"] == {"count": 2, "deviceGroup": not device_group, "product": A100.gpu_product}


def test_a_multi_gpu_container_needs_its_device_group_named() -> None:
    plan = plan_of(lane("a", queue_pool="default", floor_min=0), lane("b", cell=A100, shape=PAIR))
    error = refusal(plan, ChartParameters(device_groups={"l4": True, "A100-80GB": True}))
    assert error.path == "$.lanes[1].containerShape.gpuCount"
    assert error.message == (
        "is 2: the chart runs a container of 2 GPUs as one device group or as 2 single-GPU workers, and the plan "
        "does not say which; name machine profile 'a100-80gb' in device_groups"
    )


def test_a_single_gpu_container_is_never_a_device_group() -> None:
    values = helm_values(plan_of(lane("a")), ChartParameters(device_groups={"l4": True}))
    assert values["workers"]["pools"]["l4"]["gpu"]["deviceGroup"] is False


@pytest.mark.parametrize("device_groups", [{"l4": 1}, {1: True}, [("l4", True)]])
def test_device_groups_map_machine_profiles_to_booleans(device_groups: Any) -> None:
    with pytest.raises(SchemaError, match="deviceGroups"):
        ChartParameters(device_groups=device_groups)


@pytest.mark.parametrize("cores", [0.0005, 1.0001])
def test_cpu_cores_finer_than_a_millicore_are_refused(cores: float) -> None:
    shape = ContainerShape(gpu_count=1, cpu_cores=cores, memory_mib=1024)
    error = refusal(plan_of(lane("a", queue_pool="default", floor_min=0, shape=shape)))
    assert error.path == "$.lanes[0].containerShape.cpuCores"


@pytest.mark.parametrize("queue_pool", ["_default", "_DEFAULT"])
def test_the_reserved_pool_name_cannot_carry_a_static_queue_pool(queue_pool: str) -> None:
    error = refusal(plan_of(lane("a", queue_pool=queue_pool, floor_min=0)))
    assert error.path == "$.lanes[0].queuePool"
    assert "is reserved by the gateway" in error.message


@pytest.mark.parametrize("fallback_replicas", [0, -1, True])
def test_chart_parameters_need_a_fallback_floor(fallback_replicas: int) -> None:
    with pytest.raises(SchemaError, match="fallbackReplicas"):
        ChartParameters(fallback_replicas=fallback_replicas)


def test_every_warm_floor_warmth_is_carried_by_bundle_min_replicas() -> None:
    for warmth in WARM_FLOOR_WARMTHS:
        plan = plan_of(lane("a", floor_min=3), warmths={"a": warmth})
        values = helm_values(plan)
        assert values["workers"]["pools"]["l4"]["bundles"]["default"]["minReplicas"] == 3
        assert values["queueRouting"]["staticQueuePools"]["placement"]["minimumWorkerCount"] == 3
        pinned = ["org/a"] if warmth is Warmth.PINNED else []
        assert values["queueRouting"]["staticQueuePools"]["placement"]["pinnedModels"] == pinned


def residents(*lanes: PlanLane, placed: dict[str, tuple[str, Warmth]]) -> Plan:
    identities = tuple(
        PlanIdentity(identity=identity, lane=name, warmth=warmth) for identity, (name, warmth) in placed.items()
    )
    return Plan(placement="us-central1", lanes=lanes, identities=identities)


def static_pools(values: dict) -> dict:
    return values["queueRouting"]["staticQueuePools"]


START = datetime(2026, 11, 2, tzinfo=UTC)
NO_DEMAND = Demand(start=START, end=START + timedelta(hours=24), series=())
PACKING_PRICES = PriceList(
    currency="USD",
    machine_profiles=(
        MachineProfilePrice(
            name="l4",
            gpu_product="NVIDIA-L4",
            gpu_memory_mib=23034,
            accelerator_per_hour=1.0,
            core_per_hour=0.0,
            memory_gib_per_hour=0.0,
        ),
        MachineProfilePrice(
            name="a100-80gb",
            gpu_product="NVIDIA-A100-SXM4-80GB",
            gpu_memory_mib=81920,
            accelerator_per_hour=3.0,
            core_per_hour=0.0,
            memory_gib_per_hour=0.0,
        ),
    ),
    placements=(Placement(name="us-central1", region_multiplier=1),),
)
PACKING_PARAMETERS = planner.PlannerParameters(
    placement="us-central1", max_containers=4, fallback_floor=2, cooldown_s=900, queue_pool="placement"
)


def packing_receipt(
    identity: str, cell: Cell = L4, *, beside: tuple[str, ...] = (), footprint: int = 5000, working: int = 1000
) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        concurrency=8,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=8.0),
        latency=Percentiles(p99_ms=150.0),
        co_residents=beside,
        footprint_mib=footprint,
        working_memory_mib=working,
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/{'+'.join(beside)}"),
    )


def never_cold(identity: str, bound: int = 0) -> IdentityServiceLevel:
    return IdentityServiceLevel(identity=identity, level=ServiceLevel(come_alive_p95_s=bound, warm_p99_ms=250))


def packed_estate(queue_pool: str) -> planner.PlannerResult:
    receipts = [
        packing_receipt("org/a", footprint=6000),
        packing_receipt("org/b"),
        packing_receipt("org/a", beside=("org/b",), footprint=6000, working=1500),
        packing_receipt("org/b", beside=("org/a",), working=1500),
        packing_receipt("org/c", footprint=4000, working=1500),
    ]
    load = BootEvidence(
        identity="org/c",
        cell=L4,
        container_shape=SHAPE,
        warmth=Warmth.LAZY_RESIDENT,
        outcome=BootOutcome.SUCCEEDED,
        boot_seconds=20.0,
        source=SourceRef(uri="loads/org/c/l4"),
    )
    return planner.plan(
        ServiceLevels(identities=(never_cold("org/a"), never_cold("org/b"), never_cold("org/c", 60))),
        receipts,
        [load],
        PACKING_PRICES,
        NO_DEMAND,
        None,
        replace(PACKING_PARAMETERS, queue_pool=queue_pool),
    )


def scarce_type_attempts(cell: Cell) -> list[BootEvidence]:
    return [
        BootEvidence(
            identity="org/other",
            cell=cell,
            container_shape=SHAPE,
            warmth=Warmth.COLD_WITH_BOUND,
            outcome=BootOutcome.SUCCEEDED if index < 2 else BootOutcome.UNPLACED,
            boot_seconds=30.0 if index < 2 else None,
            source=SourceRef(uri=f"boots/org/other/{cell.machine_profile}/{index}"),
        )
        for index in range(3)
    ]


def test_a_lane_with_fallback_cells_is_refused() -> None:
    error = refusal(plan_of(replace(lane("a"), fallback_cells=(A100,))))
    assert error.path == "$.lanes[0].fallbackCells"
    assert error.message.startswith(
        "lists 'a100-80gb' with 'NVIDIA-A100-SXM4-80GB' after the lane's cell, 'l4' with 'NVIDIA-L4', but a worker "
        "group of the sie-cluster chart runs on one machine profile and GPU product and has no ordered candidate list"
    )
    assert error.message.endswith("so the chart would place the lane on its cell only; plan the lane on one cell")
    later = plan_of(
        lane("a", floor_min=0), replace(lane("b", queue_pool="default", floor_min=0), fallback_cells=(A100,))
    )
    assert refusal(later).path == "$.lanes[1].fallbackCells"


def test_a_planner_candidate_list_is_refused_rather_than_emitted_on_its_first_cell() -> None:
    receipts = [packing_receipt("org/model"), packing_receipt("org/model", A100)]
    boots = [*scarce_type_attempts(L4), *scarce_type_attempts(A100)]
    result = planner.plan(
        ServiceLevels(identities=(never_cold("org/model"),)),
        receipts,
        boots,
        PACKING_PRICES,
        NO_DEMAND,
        None,
        replace(PACKING_PARAMETERS, hold_budget_s=20),
    )
    (planned,) = result.plan.lanes
    assert (planned.cell, planned.fallback_cells) == (L4, (A100,))
    error = refusal(result.plan)
    assert error.path == "$.lanes[0].fallbackCells"


@pytest.mark.parametrize("warmth", sorted(UNREALIZED_WARMTHS))
def test_a_warmth_the_chart_does_not_realize_is_refused(warmth: Warmth) -> None:
    plan = plan_of(lane("a", floor_min=0), lane("b", queue_pool="default", floor_min=0), warmths={"b": warmth})
    error = refusal(plan)
    assert error.path == "$.identities[1].warmth"
    assert error.message.startswith(
        f"is {warmth.value!r} for 'org/b' on lane 'b', which the sie-cluster chart does not realize: "
        f"{UNREALIZED_WARMTHS[warmth]};"
    )
    assert set(UNREALIZED_WARMTHS) == {Warmth.BRIDGED, Warmth.SNAPSHOT_COLD, Warmth.SCHEDULED_WARM}


def test_a_lane_pinning_several_identities_carries_them_all_on_its_own_static_queue_pool() -> None:
    plan = residents(
        lane("shared", floor_max=2),
        lane("cold", queue_pool="default", cell=A100, floor_min=0),
        placed={
            "org/b": ("shared", Warmth.PINNED),
            "org/a": ("shared", Warmth.PINNED),
            "org/c": ("shared", Warmth.LAZY_RESIDENT),
            "org/d": ("cold", Warmth.COLD_WITH_BOUND),
        },
    )
    values = helm_values(plan)
    assert static_pools(values) == {
        "placement": {
            "bundle": "default",
            "gpus": {"l4": 0},
            "minimumWorkerCount": 1,
            "pinnedModels": ["org/a", "org/b"],
        }
    }
    assert values["workers"]["pools"]["l4"]["bundles"] == {
        "default": {"enabled": True, "minReplicas": 1, "maxReplicas": 4}
    }
    assert values["workers"]["pools"]["a100-80gb"]["queuePool"] == "default"


def co_pinned(name: str, profile: str, queue_pool: str) -> PlanLane:
    return lane(name, queue_pool=queue_pool, cell=Cell(machine_profile=profile, gpu_product="NVIDIA-L4"))


def test_two_co_pinned_lanes_on_one_gpu_type_emit_through_their_own_machine_profiles_and_queue_pools() -> None:
    placed = {
        "org/a": ("first", Warmth.PINNED),
        "org/b": ("first", Warmth.PINNED),
        "org/c": ("second", Warmth.PINNED),
        "org/d": ("second", Warmth.PINNED),
        "org/e": ("second", Warmth.LAZY_RESIDENT),
    }
    values = helm_values(
        residents(co_pinned("first", "l4", "first"), co_pinned("second", "l4-b", "second"), placed=placed)
    )
    assert {profile: group["queuePool"] for profile, group in values["workers"]["pools"].items()} == {
        "l4": "first",
        "l4-b": "second",
    }
    assert {name: pool["pinnedModels"] for name, pool in static_pools(values).items()} == {
        "first": ["org/a", "org/b"],
        "second": ["org/c", "org/d"],
    }
    error = refusal(residents(co_pinned("first", "l4", "first"), co_pinned("second", "l4", "second"), placed=placed))
    assert error.path == "$.lanes[1].queuePool"
    assert error.message.startswith("shares worker group 'l4' with $.lanes[0], which has 'first', got 'second';")
    shared_pool = refusal(
        residents(co_pinned("first", "l4", "shared"), co_pinned("second", "l4-b", "shared"), placed=placed)
    )
    assert shared_pool.path == "$.lanes[1].name"
    assert "which pins ['org/a', 'org/b'], but pins ['org/c', 'org/d']" in shared_pool.message


def test_a_lane_pinning_several_identities_on_the_default_queue_pool_is_refused() -> None:
    placed = {"org/a": ("shared", Warmth.PINNED), "org/b": ("shared", Warmth.PINNED)}
    error = refusal(residents(lane("shared", queue_pool="default"), placed=placed))
    assert error.path == "$.identities[0].warmth"
    assert "pins 'org/a' on chart lane 'default/l4/default'" in error.message


def test_a_planner_lane_pinning_several_identities_emits_on_a_named_queue_pool() -> None:
    result = packed_estate("placement")
    (planned,) = result.plan.lanes
    assert {entry.identity: (entry.lane, entry.warmth) for entry in result.plan.identities} == {
        "org/a": (planned.name, Warmth.PINNED),
        "org/b": (planned.name, Warmth.PINNED),
        "org/c": (planned.name, Warmth.LAZY_RESIDENT),
    }
    values = helm_values(result.plan)
    assert planned.queue_pool == planned.name
    assert static_pools(values) == {
        planned.name: {
            "bundle": "default",
            "gpus": {"l4": 0},
            "minimumWorkerCount": planned.envelope.floor_min,
            "pinnedModels": ["org/a", "org/b"],
        }
    }
    assert values["workers"]["pools"]["l4"]["bundles"]["default"] == {
        "enabled": True,
        "minReplicas": planned.envelope.floor_min,
        "maxReplicas": planned.max_containers,
    }
    assert helm_values(packed_estate("default").plan) == values


def test_a_planner_estate_with_several_pinned_lanes_emits_a_named_queue_pool_for_each() -> None:
    l4_b = Cell(machine_profile="l4-b", gpu_product="NVIDIA-L4")
    second_l4 = replace(PACKING_PRICES.machine_profiles[0], name="l4-b")
    prices = replace(PACKING_PRICES, machine_profiles=(*PACKING_PRICES.machine_profiles, second_l4))
    boot = BootEvidence(
        identity="org/d",
        cell=A100,
        container_shape=SHAPE,
        warmth=Warmth.COLD_WITH_BOUND,
        outcome=BootOutcome.SUCCEEDED,
        boot_seconds=60.0,
        source=SourceRef(uri="boots/org/d/a100-80gb"),
    )
    result = planner.plan(
        ServiceLevels(identities=(never_cold("org/a"), never_cold("org/b"), never_cold("org/d", 600))),
        [packing_receipt("org/a"), packing_receipt("org/b", l4_b), packing_receipt("org/d", A100)],
        [boot],
        prices,
        NO_DEMAND,
        None,
        replace(PACKING_PARAMETERS, queue_pool="default"),
    )
    assert {entry.identity: (entry.lane, entry.warmth) for entry in result.plan.identities} == {
        "org/a": ("org-a-l4", Warmth.PINNED),
        "org/b": ("org-b-l4-b", Warmth.PINNED),
        "org/d": ("org-d-a100-80gb", Warmth.COLD_WITH_BOUND),
    }
    values = helm_values(result.plan, ChartParameters(fallback_replicas=PACKING_PARAMETERS.fallback_floor))
    assert {profile: group["queuePool"] for profile, group in values["workers"]["pools"].items()} == {
        "a100-80gb": "default",
        "l4": "org-a-l4",
        "l4-b": "org-b-l4-b",
    }
    assert static_pools(values) == {
        "org-a-l4": {"bundle": "default", "gpus": {"l4": 0}, "minimumWorkerCount": 1, "pinnedModels": ["org/a"]},
        "org-b-l4-b": {"bundle": "default", "gpus": {"l4-b": 0}, "minimumWorkerCount": 1, "pinnedModels": ["org/b"]},
    }
    replanned = planner.plan(
        ServiceLevels(identities=(never_cold("org/a"), never_cold("org/b"), never_cold("org/d", 600))),
        [packing_receipt("org/a"), packing_receipt("org/b", l4_b), packing_receipt("org/d", A100)],
        [boot],
        prices,
        NO_DEMAND,
        result.plan,
        replace(PACKING_PARAMETERS, queue_pool="default"),
    )
    assert replanned.plan == result.plan
    assert helm_values(replanned.plan, ChartParameters(fallback_replicas=PACKING_PARAMETERS.fallback_floor)) == values


def test_plan_refusals_list_every_refusal_helm_values_makes_whatever_the_chart_parameters() -> None:
    candidates = (
        plan_of(replace(lane("a", floor_min=0), fallback_cells=(A100,))),
        plan_of(lane("a", queue_pool="Default")),
        plan_of(lane("a", floor_min=0), warmths={"a": Warmth.SNAPSHOT_COLD}),
    )
    for candidate in candidates:
        (only,) = plan_refusals(candidate)
        assert str(only) == str(refusal(candidate))
    combined = plan_of(
        replace(lane("a", floor_min=0), fallback_cells=(A100,)),
        lane("b", queue_pool="default", cell=A100),
        lane("c", floor_min=0, bundle="c"),
        warmths={"c": Warmth.BRIDGED},
    )
    assert [error.path for error in plan_refusals(combined)] == [
        "$.lanes[0].fallbackCells",
        "$.identities[1].warmth",
        "$.identities[2].warmth",
    ]
    assert plan_refusals(golden_plan()) == ()
