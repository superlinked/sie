from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
from sie_placement.helm import (
    BUNDLED_DEPENDENCIES_LANE_LIMIT,
    EXTERNAL_DEPENDENCIES_LANE_LIMIT,
    ChartParameters,
    UnrealizablePlanError,
    helm_values,
    lane_token,
)
from sie_placement.report import render
from sie_placement.schema import (
    WARM_FLOOR_WARMTHS,
    Cell,
    ContainerShape,
    Envelope,
    Plan,
    PlanIdentity,
    PlanLane,
    SchemaError,
    Warmth,
)

FIXTURES = Path(__file__).parent / "fixtures"
HELM_FIXTURES = FIXTURES / "helm"
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
WIDE = ContainerShape(gpu_count=1, cpu_cores=8, memory_mib=32768)


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


def test_golden_plan_emits_the_golden_values() -> None:
    plan = Plan.from_dict(load(HELM_FIXTURES / "plan.json"))
    assert render(helm_values(plan)) == (HELM_FIXTURES / "values.json").read_text(encoding="utf-8")


def test_values_do_not_depend_on_document_order() -> None:
    plan = Plan.from_dict(load(HELM_FIXTURES / "plan.json"))
    expected = helm_values(plan)
    shuffle = random.Random(5)  # noqa: S311 -- reproducible test data
    for _ in range(20):
        lanes = list(plan.lanes)
        identities = list(plan.identities)
        shuffle.shuffle(lanes)
        shuffle.shuffle(identities)
        assert helm_values(replace(plan, lanes=tuple(lanes), identities=tuple(identities))) == expected


def test_lanes_naming_one_chart_lane_union_their_pins() -> None:
    values = helm_values(Plan.from_dict(load(HELM_FIXTURES / "plan.json")))
    assert list(values["workers"]["pools"]["l4"]["bundles"]) == ["default"]
    assert values["queueRouting"]["staticQueuePools"]["placement"]["pinnedModels"] == [
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
        "intfloat/e5-base-v2",
    ]


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


def test_schema_fixture_plan_has_an_unrealizable_zero_floor() -> None:
    error = refusal(Plan.from_dict(load(FIXTURES / "plan.json")))
    assert error.path == "$.lanes[1].envelope.floorMax"
    assert error.message.startswith("must be 0 or at least 2:")


def test_planner_estate_plan_names_one_chart_lane_with_two_floors() -> None:
    error = refusal(Plan.from_dict(load(FIXTURES / "planner" / "expected-plan.json")))
    assert error.path == "$.lanes[1].envelope.floorMin"
    assert error.message.startswith("names chart lane 'default/l4/default' with $.lanes[0], which has 0, got 1")


def bundle_lanes(count: int, *, duplicate: bool = False) -> Plan:
    lanes = [lane(f"lane-{index}", queue_pool="default", bundle=f"b{index}", floor_min=0) for index in range(count)]
    if duplicate:
        lanes.append(lane("lane-again", queue_pool="default", bundle="B0", floor_min=0))
    return plan_of(*lanes)


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
    assert len(helm_values(bundle_lanes(limit), parameters)["workers"]["pools"]["l4"]["bundles"]) == limit
    assert (
        len(helm_values(bundle_lanes(limit, duplicate=True), parameters)["workers"]["pools"]["l4"]["bundles"]) == limit
    )
    error = refusal(bundle_lanes(limit + 1), parameters)
    assert error.path == "$.lanes"
    assert error.message == (
        f"names {limit + 1} chart lanes, above the {limit} the sie-cluster chart admits with {topology}"
    )


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
                    "gpu": {"count": 1, "product": "NVIDIA-L4"},
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


@pytest.mark.parametrize(
    ("changed", "path", "expected"),
    [
        ({"cell": Cell(machine_profile="L4", gpu_product="NVIDIA-L4-24GB")}, "cell.gpuProduct", "'NVIDIA-L4'"),
        ({"container_shape": WIDE}, "containerShape", "{'gpuCount': 1, 'cpuCores': 4, 'memoryMiB': 16384}"),
        ({"max_containers": 5}, "maxContainers", "4"),
        ({"envelope": Envelope(floor_min=2, floor_max=2)}, "envelope.floorMin", "1"),
    ],
)
def test_lanes_naming_one_chart_lane_must_agree(changed: dict, path: str, expected: str) -> None:
    error = refusal(plan_of(lane("a"), replace(lane("b", queue_pool="PLACEMENT", bundle="Default"), **changed)))
    assert error.path == f"$.lanes[1].{path}"
    assert error.message.startswith(f"names chart lane 'placement/l4/default' with $.lanes[0], which has {expected}")


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
    shape = ContainerShape(gpu_count=2, cpu_cores=cores, memory_mib=1536)
    group = helm_values(plan_of(lane("a", queue_pool="default", floor_min=0, shape=shape)))["workers"]["pools"]["l4"]
    assert group["gpu"] == {"count": 2, "product": "NVIDIA-L4"}
    assert group["resources"]["requests"] == group["resources"]["limits"] == {"cpu": quantity, "memory": "1536Mi"}


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
