from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sie_placement.planner import PlannerParameters, plan
from sie_placement.report import (
    Blocker,
    ComeAliveBasis,
    IdentityStatus,
    Measurement,
    Option,
    RefusedRecord,
    RequestReason,
    render,
)
from sie_placement.schema import (
    BootEvidence,
    BootOutcome,
    Cell,
    ContainerShape,
    CoResidencyReceipt,
    CoResident,
    Demand,
    DemandHour,
    DemandSeries,
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

FIXTURES = Path(__file__).parent / "fixtures" / "planner"
SEEDS = 60
START = datetime(2026, 11, 2, tzinfo=UTC)
MONTH = 730.0
SCALE = MONTH / 24
L4 = Cell(machine_profile="l4", gpu_product="NVIDIA-L4")
A100 = Cell(machine_profile="a100-80gb", gpu_product="NVIDIA-A100-SXM4-80GB")
SHAPE = ContainerShape(gpu_count=1, cpu_cores=4, memory_mib=16384)
WIDE = ContainerShape(gpu_count=1, cpu_cores=8, memory_mib=32768)
BOUND = ServiceLevel(come_alive_p95_s=0, warm_p99_ms=250)
GENERATION_BOUND = ServiceLevel(come_alive_p95_s=0, ttft_p99_ms=1500, itl_p99_ms=80)
GPU_MEMORY = 23034


def price(cell: Cell, rate: float) -> MachineProfilePrice:
    return MachineProfilePrice(
        name=cell.machine_profile,
        gpu_product=cell.gpu_product,
        accelerator_per_hour=rate,
        core_per_hour=0.0,
        memory_gib_per_hour=0.0,
    )


PRICES = PriceList(
    currency="USD",
    machine_profiles=(price(L4, 1.0), price(A100, 3.0)),
    placements=(Placement(name="test", region_multiplier=1),),
)


def receipt(identity, cell=L4, *, shape=SHAPE, concurrency=8, rps=10.0, p99=100.0) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=rps),
        latency=Percentiles(p99_ms=p99),
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/{shape.memory_mib}/{concurrency}"),
    )


def generation_receipt(identity, cell=L4, *, concurrency=16, ttft=1000.0, itl=40.0) -> Receipt:
    return Receipt(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        concurrency=concurrency,
        request_shape=RequestShape(operation=Operation.GENERATE, items_per_request=1, output_tokens=256),
        throughput=Throughput(requests_per_second=2.0, output_tokens_per_second=512.0),
        ttft=Percentiles(p99_ms=ttft),
        itl=Percentiles(p99_ms=itl),
        source=SourceRef(uri=f"receipts/{identity}/{cell.machine_profile}/generate"),
    )


def resident(identity, *, p99=200.0, rps=10.0) -> CoResident:
    return CoResident(
        identity=identity,
        request_shape=RequestShape(operation=Operation.ENCODE, items_per_request=1),
        throughput=Throughput(requests_per_second=rps),
        latency=Percentiles(p99_ms=p99),
    )


def generation_resident(identity, *, ttft=1100.0, itl=60.0) -> CoResident:
    return CoResident(
        identity=identity,
        request_shape=RequestShape(operation=Operation.GENERATE, items_per_request=1, output_tokens=256),
        throughput=Throughput(requests_per_second=2.0, output_tokens_per_second=512.0),
        ttft=Percentiles(p99_ms=ttft),
        itl=Percentiles(p99_ms=itl),
    )


def co_receipt(*residents, cell=L4, shape=SHAPE, concurrency=8, peak=12000, uri=None) -> CoResidencyReceipt:
    members = tuple(entry if isinstance(entry, CoResident) else resident(entry) for entry in residents)
    names = "+".join(sorted(entry.identity for entry in members))
    return CoResidencyReceipt(
        cell=cell,
        container_shape=shape,
        concurrency=concurrency,
        gpu_memory_mib=GPU_MEMORY,
        peak_gpu_memory_mib=peak,
        residents=members,
        source=SourceRef(uri=uri or f"co-residency/{cell.machine_profile}/{concurrency}/{names}"),
    )


def boot(identity, seconds=60.0, cell=L4) -> BootEvidence:
    return BootEvidence(
        identity=identity,
        cell=cell,
        container_shape=SHAPE,
        warmth=Warmth.COLD_WITH_BOUND,
        boot_seconds=seconds,
        outcome=BootOutcome.SUCCEEDED,
        source=SourceRef(uri=f"boots/{identity}/{cell.machine_profile}"),
    )


def levels(*identities, overrides=None) -> ServiceLevels:
    entries = dict.fromkeys(identities, BOUND) | (overrides or {})
    return ServiceLevels(
        identities=tuple(IdentityServiceLevel(identity=name, level=level) for name, level in sorted(entries.items()))
    )


def demand(hourly=None) -> Demand:
    return Demand(
        start=START,
        end=START + timedelta(hours=24),
        series=tuple(
            DemandSeries(
                identity=identity,
                operation=Operation.ENCODE,
                hours=tuple(
                    DemandHour(start=START + timedelta(hours=hour), requests=requests)
                    for hour, requests in sorted(counts.items())
                ),
            )
            for identity, counts in sorted((hourly or {}).items())
        ),
    )


def parameters(**overrides) -> PlannerParameters:
    values = {"placement": "test", "max_containers": 4, "fallback_floor": 2, "cooldown_s": 900} | overrides
    return PlannerParameters(**values)


def run(service_levels, receipts, co=(), *, boots=(), hourly=None, current=None, prices=PRICES, **overrides):
    return plan(
        service_levels,
        receipts,
        boots,
        prices,
        demand(hourly),
        current,
        parameters(**overrides),
        co_residency_receipts=co,
    )


def co_requests(report):
    return [
        request for request in report.measurement_requests if request.measure == (Measurement.CO_RESIDENCY_RECEIPT,)
    ]


def lanes_of(result) -> dict[str, tuple[str, ...]]:
    hosted: dict[str, list[str]] = {}
    for entry in result.plan.identities:
        hosted.setdefault(entry.lane, []).append(entry.identity)
    return {name: tuple(sorted(identities)) for name, identities in hosted.items()}


def lane(name, cell=L4, *, queue_pool="default", floor=1, concurrency=8) -> PlanLane:
    return PlanLane(
        name=name,
        queue_pool=queue_pool,
        bundle="default",
        cell=cell,
        container_shape=SHAPE,
        concurrency=concurrency,
        max_containers=4,
        envelope=Envelope(floor_min=floor, floor_max=floor),
    )


def rendered(result) -> tuple[str, str]:
    return render(result.plan.to_dict()), render(result.report.to_dict())


PAIR = (levels("org/a", "org/b"), [receipt("org/a", A100), receipt("org/b")])


def test_a_pinned_identity_joins_a_pinned_lane_that_a_co_residency_receipt_covers():
    measured = co_receipt("org/a", "org/b")
    result = run(*PAIR, [measured])
    assert result.plan.lanes == (lane("org-b-l4"),)
    assert result.plan.identities == (
        PlanIdentity(identity="org/a", lane="org-b-l4", warmth=Warmth.PINNED, co_packed=True),
        PlanIdentity(identity="org/b", lane="org-b-l4", warmth=Warmth.PINNED),
    )
    assert Plan.from_dict(json.loads(render(result.plan.to_dict()))) == result.plan
    (move,) = result.report.co_packing_moves
    assert move.to_dict() == {
        "identity": "org/a",
        "fromLane": "org-a-a100-80gb",
        "fromWarmth": "pinned",
        "toLane": "org-b-l4",
        "coResidents": ["org/b"],
        "receipt": {"uri": "co-residency/l4/8/org/a+org/b"},
        "gpuMemoryHeadroomMiB": GPU_MEMORY - 12000,
        "saving": 3 * MONTH,
    }
    moved = result.report.identity("org/a")
    assert (moved.status, moved.lane, moved.monthly_cost) == (IdentityStatus.CO_PACKED, "org-b-l4", MONTH)
    assert (moved.chosen.cell, moved.chosen.monthly_cost) == (A100, 3 * MONTH)
    assert moved.joined == Option(
        cell=L4,
        container_shape=SHAPE,
        concurrency=8,
        warmth=Warmth.PINNED,
        monthly_cost=MONTH,
        come_alive_s=0.0,
        come_alive_basis=ComeAliveBasis.WARM_FLOOR,
        receipt=measured.source,
    )
    assert result.report.identity("org/b").status is IdentityStatus.PLACED
    assert result.report.identity("org/b").joined is None
    (lane_report,) = result.report.lanes
    assert lane_report.co_residency_receipt == measured.source
    assert lane_report.to_dict()["coResidencyReceipt"] == {"uri": "co-residency/l4/8/org/a+org/b"}
    assert result.report.monthly_cost == pytest.approx(MONTH)
    assert result.report.measurement_requests == ()


def test_a_co_packing_pass_without_receipts_changes_nothing():
    without = run(*PAIR)
    assert lanes_of(without) == {"org-a-a100-80gb": ("org/a",), "org-b-l4": ("org/b",)}
    assert without.report.co_packing_moves == ()
    assert all(entry.co_packed is False for entry in without.plan.identities)
    assert without.report.to_dict()["coPackingMoves"] == []


def test_identities_are_visited_in_decreasing_floor_cost():
    service_levels = levels("org/x", "org/y", "org/z")
    receipts = [receipt("org/x", A100), receipt("org/y"), receipt("org/z")]
    co = [co_receipt("org/x", "org/y"), co_receipt("org/x", "org/z"), co_receipt("org/y", "org/z")]
    result = run(service_levels, receipts, co)
    assert [(move.identity, move.to_lane) for move in result.report.co_packing_moves] == [("org/x", "org-y-l4")]
    assert lanes_of(result) == {"org-y-l4": ("org/x", "org/y"), "org-z-l4": ("org/z",)}
    (request,) = co_requests(result.report)
    assert (request.identity, request.lane, request.co_residents) == ("org/z", "org-y-l4", ("org/x", "org/y"))
    assert request.saving_bound == pytest.approx(MONTH)


@pytest.mark.parametrize(("first", "second", "host"), [("org/a", "org/b", "org-b-l4"), ("org/b", "org/c", "org-c-l4")])
def test_equal_floors_are_visited_in_identity_order(first, second, host):
    result = run(levels(first, second), [receipt(first), receipt(second)], [co_receipt(first, second)])
    (move,) = result.report.co_packing_moves
    assert (move.identity, move.to_lane, move.co_residents) == (first, host, (second,))


@pytest.mark.parametrize(
    ("host_p99", "mover_p99", "moved"),
    [(300.0, 200.0, False), (200.0, 300.0, False), (250.0, 250.0, True), (100.0, 100.0, True)],
)
def test_a_move_is_kept_only_when_every_resident_meets_its_bound_under_the_receipt(host_p99, mover_p99, moved):
    measured = co_receipt(resident("org/a", p99=mover_p99), resident("org/b", p99=host_p99))
    result = run(*PAIR, [measured])
    assert bool(result.report.co_packing_moves) is moved
    reverse = [] if moved else [("org/b", "org-a-a100-80gb")]
    assert [(request.identity, request.lane) for request in co_requests(result.report)] == reverse
    expected = [
        Blocker(identity=identity, field="warmP99Ms", bound=250, measured=p99)
        for identity, p99 in (("org/a", mover_p99), ("org/b", host_p99))
        if p99 > 250
    ]
    assert [move.blocked_by for move in result.report.blocked_co_packing_moves] == ([] if moved else [tuple(expected)])
    if not moved:
        (blocked,) = result.report.blocked_co_packing_moves
        assert (blocked.identity, blocked.to_lane, blocked.co_residents) == ("org/a", "org-b-l4", ("org/b",))
        assert blocked.saving == pytest.approx(3 * MONTH)
        assert blocked.to_dict()["blockedBy"][0] == {
            "identity": expected[0].identity,
            "field": "warmP99Ms",
            "bound": 250,
            "measured": expected[0].measured,
        }


@pytest.mark.parametrize(("itl", "moved"), [(90.0, False), (80.0, True)])
def test_a_generation_resident_keeps_its_token_bounds_under_the_receipt(itl, moved):
    service_levels = levels("org/embed", overrides={"org/gen": GENERATION_BOUND})
    receipts = [receipt("org/embed", A100, concurrency=16), generation_receipt("org/gen")]
    measured = co_receipt("org/embed", generation_resident("org/gen", itl=itl), concurrency=16)
    result = run(service_levels, receipts, [measured])
    assert [move.identity for move in result.report.co_packing_moves] == (["org/embed"] if moved else [])
    if moved:
        assert lanes_of(result) == {"org-gen-l4": ("org/embed", "org/gen")}


@pytest.mark.parametrize(("peak", "moved"), [(GPU_MEMORY - 4095, False), (GPU_MEMORY - 4096, True)])
def test_a_move_keeps_the_reviewed_gpu_memory_headroom_free(peak, moved):
    result = run(*PAIR, [co_receipt("org/a", "org/b", peak=peak)], gpu_memory_headroom_mib=4096)
    assert [move.gpu_memory_headroom_mib for move in result.report.co_packing_moves] == ([4096] if moved else [])
    assert [request.identity for request in co_requests(result.report)] == ([] if moved else ["org/b"])
    blocked = [(move.identity, move.blocked_by) for move in result.report.blocked_co_packing_moves]
    shortfall = Blocker(field="gpuMemoryHeadroomMiB", bound=4096, measured=GPU_MEMORY - peak)
    assert blocked == ([] if moved else [("org/a", (shortfall,))])
    if not moved:
        assert result.report.to_dict()["blockedCoPackingMoves"][0]["blockedBy"] == [
            {"field": "gpuMemoryHeadroomMiB", "bound": 4096, "measured": 4095}
        ]


def test_a_move_without_a_co_residency_receipt_is_a_measurement_request():
    result = run(*PAIR)
    assert result.report.co_packing_moves == ()
    first, second = co_requests(result.report)
    assert first.to_dict() == {
        "identity": "org/a",
        "cell": L4.to_dict(),
        "containerShape": SHAPE.to_dict(),
        "concurrency": 8,
        "lane": "org-b-l4",
        "coResidents": ["org/b"],
        "measure": ["co-residency-receipt"],
        "warmth": "pinned",
        "reason": "unmeasured",
        "costBound": 0.0,
        "savingBound": 3 * MONTH,
    }
    assert (second.identity, second.cell, second.lane, second.co_residents) == (
        "org/b",
        A100,
        "org-a-a100-80gb",
        ("org/a",),
    )
    assert second.saving_bound == pytest.approx(MONTH)
    ranked = [request.measure for request in result.report.measurement_requests]
    assert ranked[0] == (Measurement.CO_RESIDENCY_RECEIPT,)


@pytest.mark.parametrize("mismatch", [{"concurrency": 16}, {"shape": WIDE}])
def test_a_receipt_for_another_lane_shape_does_not_cover_the_move(mismatch):
    result = run(*PAIR, [co_receipt("org/a", "org/b", **mismatch)])
    assert result.report.co_packing_moves == ()
    assert [(request.identity, request.reason) for request in co_requests(result.report)] == [
        ("org/a", RequestReason.UNMEASURED),
        ("org/b", RequestReason.UNMEASURED),
    ]


def test_a_receipt_for_a_subset_or_a_superset_does_not_cover_the_move():
    service_levels = levels("org/a", "org/b", "org/c", "org/d")
    current = Plan(
        placement="test",
        lanes=(lane("shared"),),
        identities=(
            PlanIdentity(identity="org/b", lane="shared", warmth=Warmth.PINNED),
            PlanIdentity(identity="org/c", lane="shared", warmth=Warmth.PINNED),
        ),
    )
    receipts = [receipt("org/a", A100), receipt("org/b"), receipt("org/c")]
    co = [co_receipt("org/a", "org/b"), co_receipt("org/a", "org/b", "org/c", "org/d")]
    result = run(service_levels, receipts, co, current=current)
    assert result.report.refusals == ()
    assert result.report.identity("org/d").status is IdentityStatus.UNPLACED
    assert result.report.co_packing_moves == ()
    assert result.report.blocked_co_packing_moves == ()
    (request,) = co_requests(result.report)
    assert (request.identity, request.lane, request.co_residents) == ("org/a", "shared", ("org/b", "org/c"))


def test_a_refused_co_residency_receipt_is_listed_and_asked_for_again():
    other = Cell(machine_profile="l4", gpu_product="NVIDIA-A10G")
    result = run(*PAIR, [co_receipt("org/a", "org/b", cell=other)])
    assert result.report.co_packing_moves == ()
    (refusal,) = result.report.refusals
    assert refusal.to_dict() == {
        "record": "co-residency-receipt",
        "residents": ["org/a", "org/b"],
        "cell": other.to_dict(),
        "source": {"uri": "co-residency/l4/8/org/a+org/b"},
        "reason": "realized GPU product 'NVIDIA-A10G' differs from 'NVIDIA-L4', the product machine profile 'l4' requests",
    }
    assert refusal.record is RefusedRecord.CO_RESIDENCY_RECEIPT
    assert [(request.identity, request.reason) for request in co_requests(result.report)] == [
        ("org/a", RequestReason.REFUSED),
        ("org/b", RequestReason.UNMEASURED),
    ]


def test_a_co_residency_receipt_that_cannot_answer_a_service_level_is_refused():
    result = run(*PAIR, [co_receipt(generation_resident("org/a"), "org/b")])
    (refusal,) = result.report.refusals
    assert refusal.reason == (
        "resident 'org/a': the receipt reports ttft and itl and the service level declares warmP99Ms"
    )
    assert result.report.co_packing_moves == ()


def test_co_residency_receipts_naming_an_identity_without_a_service_level_are_ignored():
    other = Cell(machine_profile="l4", gpu_product="NVIDIA-A10G")
    co = [co_receipt("org/a", "org/b", "org/stranger"), co_receipt("org/a", "org/stranger", cell=other)]
    result = run(*PAIR, co)
    assert result.report.refusals == ()
    assert result.report.co_packing_moves == ()


def test_a_lane_of_another_bundle_hosts_no_move():
    result = run(*PAIR, [co_receipt("org/a", "org/b")], bundles={"org/a": "sglang"})
    assert result.report.co_packing_moves == ()
    assert co_requests(result.report) == []


def test_a_zero_floor_lane_hosts_no_move():
    service_levels = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    result = run(service_levels, PAIR[1], [co_receipt("org/a", "org/b")], boots=[boot("org/b")])
    assert result.report.identity("org/b").chosen.warmth is Warmth.COLD_WITH_BOUND
    assert result.report.co_packing_moves == ()
    assert lanes_of(result) == {"org-a-a100-80gb": ("org/a",), "org-b-l4": ("org/b",)}
    assert co_requests(result.report) == []


@pytest.mark.parametrize(("queue_pool", "moved"), [("default", True), ("batch", False)])
def test_a_kept_pinned_lane_of_the_same_queue_pool_hosts_a_move(queue_pool, moved):
    legacy = lane("legacy", queue_pool=queue_pool)
    current = Plan(
        placement="test",
        lanes=(legacy,),
        identities=(PlanIdentity(identity="org/kept", lane="legacy", warmth=Warmth.PINNED),),
    )
    measured = co_receipt("org/a", "org/kept")
    result = run(levels("org/a", "org/kept"), [receipt("org/a", A100)], [measured], current=current)
    assert result.report.identity("org/kept").status is IdentityStatus.KEPT
    assert {lane.name: lane for lane in result.plan.lanes}["legacy"] == legacy
    if moved:
        assert lanes_of(result) == {"legacy": ("org/a", "org/kept")}
        assert result.report.identity("org/kept").monthly_cost == pytest.approx(MONTH)
        assert result.report.unpriced_lanes == ()
    else:
        assert result.report.co_packing_moves == ()
        assert co_requests(result.report) == []


def test_a_lane_with_a_resident_that_is_not_pinned_hosts_no_move():
    current = Plan(
        placement="test",
        lanes=(lane("legacy"),),
        identities=(
            PlanIdentity(identity="org/kept", lane="legacy", warmth=Warmth.PINNED),
            PlanIdentity(identity="org/resident", lane="legacy", warmth=Warmth.LAZY_RESIDENT),
        ),
    )
    service_levels = levels("org/a", "org/kept", "org/resident")
    result = run(
        service_levels, [receipt("org/a", A100)], [co_receipt("org/a", "org/kept", "org/resident")], current=current
    )
    assert result.report.co_packing_moves == ()
    assert co_requests(result.report) == []


def test_an_identity_that_shares_its_lane_does_not_move():
    current = Plan(
        placement="test",
        lanes=(lane("shared"),),
        identities=(
            PlanIdentity(identity="org/a", lane="shared", warmth=Warmth.PINNED),
            PlanIdentity(identity="org/b", lane="shared", warmth=Warmth.PINNED),
        ),
    )
    service_levels = levels("org/a", "org/b", "org/c")
    receipts = [receipt("org/a"), receipt("org/b"), receipt("org/c", A100)]
    co = [co_receipt("org/a", "org/c", cell=A100), co_receipt("org/b", "org/c", cell=A100)]
    result = run(service_levels, receipts, co, current=current)
    assert lanes_of(result) == {"shared": ("org/a", "org/b"), "org-c-a100-80gb": ("org/c",)}
    assert result.report.co_packing_moves == ()


def test_a_cold_identity_with_demand_joins_a_pinned_lane_and_turns_pinned():
    service_levels = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    receipts = [receipt("org/a"), receipt("org/b")]
    hourly = {"org/b": {0: 10}}
    result = run(service_levels, receipts, [co_receipt("org/a", "org/b")], boots=[boot("org/b")], hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert (move.identity, move.from_warmth, move.to_lane) == ("org/b", Warmth.COLD_WITH_BOUND, "org-a-l4")
    assert move.saving == pytest.approx(2 * SCALE)
    assert result.plan.identities[1] == PlanIdentity(
        identity="org/b", lane="org-a-l4", warmth=Warmth.PINNED, co_packed=True
    )
    assert result.report.monthly_cost == pytest.approx(MONTH)


def test_an_idle_cold_identity_saves_nothing_by_moving_and_is_not_asked_about():
    service_levels = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    result = run(
        service_levels, [receipt("org/a"), receipt("org/b")], [co_receipt("org/a", "org/b")], boots=[boot("org/b")]
    )
    assert result.report.co_packing_moves == ()
    assert co_requests(result.report) == []
    service_levels = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    unmeasured = run(service_levels, [receipt("org/a"), receipt("org/b")], boots=[boot("org/b")])
    assert co_requests(unmeasured.report) == []


def test_a_move_that_raises_the_total_is_not_kept():
    hourly = {identity: dict.fromkeys(range(24), 30000) for identity in ("org/a", "org/b")}
    slow = co_receipt(resident("org/a", rps=5.0), resident("org/b", rps=5.0))
    result = run(levels("org/a", "org/b"), [receipt("org/a"), receipt("org/b")], [slow], hourly=hourly)
    assert result.report.co_packing_moves == ()
    assert co_requests(result.report) == []
    assert result.report.monthly_cost == pytest.approx(2 * MONTH)


def test_a_co_packed_lane_prices_the_combined_load_of_its_residents():
    hourly = {"org/a": {3: 21600}, "org/b": {3: 21600}}
    result = run(*PAIR, [co_receipt("org/a", "org/b")], hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert move.saving == pytest.approx(3 * MONTH - SCALE)
    (lane_report,) = result.report.lanes
    assert lane_report.monthly_cost == pytest.approx(MONTH + SCALE)
    assert result.report.monthly_cost == pytest.approx(MONTH + SCALE)
    assert result.report.identity("org/a").monthly_cost == pytest.approx(MONTH)
    assert result.report.identity("org/b").monthly_cost == pytest.approx(MONTH)


def test_a_co_packed_lane_prices_each_resident_at_its_co_resident_throughput():
    hourly = {"org/a": {3: 54000}, "org/b": {3: 1}}
    result = run(*PAIR, [co_receipt(resident("org/a", rps=5.0), "org/b")], hourly=hourly)
    assert result.report.co_packing_moves[0].identity == "org/a"
    assert result.report.lanes[0].monthly_cost == pytest.approx(MONTH + (4 - 1) * SCALE)
    assert result.report.identity("org/a").monthly_cost == pytest.approx(MONTH + (3 - 1) * SCALE)
    assert result.report.identity("org/b").monthly_cost == pytest.approx(MONTH)


def test_the_move_that_saves_most_wins():
    service_levels = levels("org/a", "org/c", "org/d")
    receipts = [receipt("org/a", A100), receipt("org/c"), receipt("org/d")]
    hourly = {"org/a": {3: 21600}, "org/c": {3: 21600}}
    result = run(service_levels, receipts, [co_receipt("org/a", "org/c"), co_receipt("org/a", "org/d")], hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert (move.identity, move.to_lane) == ("org/a", "org-d-l4")
    assert move.saving == pytest.approx(3 * MONTH)
    (request,) = co_requests(result.report)
    assert (request.identity, request.lane, request.co_residents) == ("org/c", "org-d-l4", ("org/a", "org/d"))


def test_a_request_after_a_move_is_bounded_by_what_the_move_left_unsaved():
    service_levels = levels("org/a", "org/b", "org/c")
    receipts = [receipt("org/a", A100), receipt("org/b"), receipt("org/c")]
    hourly = {"org/a": {3: 21600}, "org/b": {3: 21600}}
    result = run(service_levels, receipts, [co_receipt("org/a", "org/b")], hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert (move.to_lane, move.saving) == ("org-b-l4", pytest.approx(3 * MONTH - SCALE))
    requests = {(request.identity, request.lane): request for request in co_requests(result.report)}
    assert requests[("org/a", "org-c-l4")].saving_bound == pytest.approx(SCALE)


def test_a_shared_lane_that_is_not_pinned_keeps_its_own_receipts_pricing():
    cold = ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)
    current = Plan(
        placement="test",
        lanes=(lane("cold-shared", floor=0),),
        identities=(
            PlanIdentity(identity="org/a", lane="cold-shared", warmth=Warmth.COLD_WITH_BOUND),
            PlanIdentity(identity="org/b", lane="cold-shared", warmth=Warmth.COLD_WITH_BOUND),
        ),
    )
    service_levels = levels(overrides={"org/a": cold, "org/b": cold})
    receipts = [receipt("org/a"), receipt("org/b")]
    boots = [boot("org/a"), boot("org/b")]
    hourly = {"org/a": {3: 10}, "org/b": {3: 10}}
    fast = co_receipt(resident("org/a", rps=100.0), resident("org/b", rps=100.0))
    result = run(service_levels, receipts, [fast], boots=boots, hourly=hourly, current=current)
    assert result.plan == current
    (lane_report,) = result.report.lanes
    assert lane_report.co_residency_receipt is None
    assert lane_report.monthly_cost == pytest.approx(2 * 2 * SCALE)
    assert result.report.co_packing_moves == ()


@pytest.mark.parametrize("reverse", [False, True])
def test_the_cheapest_admitted_receipt_for_a_set_prices_its_lane(reverse):
    hourly = {"org/a": {3: 21600}, "org/b": {3: 21600}}
    blocked = co_receipt(resident("org/a", p99=300.0, rps=40.0), resident("org/b", rps=40.0), uri="co/blocked")
    slow = co_receipt("org/a", "org/b", uri="co/slow")
    fast = co_receipt(resident("org/a", rps=20.0), resident("org/b", rps=20.0), uri="co/fast")
    co = [blocked, slow, fast]
    result = run(*PAIR, co[::-1] if reverse else co, hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert move.receipt.uri == "co/fast"
    assert result.report.lanes[0].co_residency_receipt.uri == "co/fast"
    assert result.report.monthly_cost == pytest.approx(MONTH)


def test_a_co_packed_identity_is_measured_against_what_it_adds_to_its_new_lane():
    service_levels = levels("org/a", "org/b", overrides={"org/a": ServiceLevel(come_alive_p95_s=60, warm_p99_ms=250)})
    receipts = [receipt("org/a", A100), receipt("org/b")]
    boots = [boot("org/a", seconds=240.0, cell=A100)]
    hourly = {"org/a": {3: 100}}
    alone = run(service_levels, receipts, boots=boots, hourly=hourly, prices=PRICES)
    assert [(request.identity, request.measure) for request in alone.report.measurement_requests][:1] == [
        ("org/a", (Measurement.CO_RESIDENCY_RECEIPT,))
    ]
    assert any(
        request.identity == "org/a" and Measurement.RECEIPT in request.measure
        for request in alone.report.measurement_requests
    )
    assert [lever.identity for lever in alone.report.come_alive_levers] == ["org/a"]
    packed = run(service_levels, receipts, [co_receipt("org/a", "org/b")], boots=boots, hourly=hourly)
    assert packed.report.identity("org/a").status is IdentityStatus.CO_PACKED
    assert packed.report.measurement_requests == ()
    assert packed.report.come_alive_levers == ()
    assert packed.report.identity("org/a").next_cheaper is None


@pytest.mark.parametrize("cell", [A100, L4])
def test_replanning_a_co_packed_plan_changes_nothing(cell):
    service_levels = levels("org/a", "org/b")
    receipts = [receipt("org/a", cell), receipt("org/b")]
    first = run(service_levels, receipts, [co_receipt("org/a", "org/b")])
    assert [move.identity for move in first.report.co_packing_moves] == ["org/a"]
    second = run(service_levels, receipts, [co_receipt("org/a", "org/b")], current=first.plan)
    assert second.plan == first.plan
    assert second.report.unchanged
    assert rendered(second)[0] == rendered(first)[0]
    assert second.report.identity("org/a").status is IdentityStatus.CO_PACKED
    assert second.report.co_packing_moves == first.report.co_packing_moves


def test_a_co_packed_identity_whose_receipt_is_withdrawn_returns_to_a_lane_of_its_own():
    first = run(*PAIR, [co_receipt("org/a", "org/b")])
    second = run(*PAIR, current=first.plan)
    assert lanes_of(second) == {"org-a-a100-80gb": ("org/a",), "org-b-l4": ("org/b",)}
    (change,) = second.report.identity_changes
    assert change.before == PlanIdentity(identity="org/a", lane="org-b-l4", warmth=Warmth.PINNED, co_packed=True)
    assert change.after == PlanIdentity(identity="org/a", lane="org-a-a100-80gb", warmth=Warmth.PINNED)


def test_a_kept_co_packed_identity_left_alone_on_its_lane_loses_the_mark():
    current = Plan(
        placement="test",
        lanes=(lane("shared"),),
        identities=(
            PlanIdentity(identity="org/a", lane="shared", warmth=Warmth.PINNED, co_packed=True),
            PlanIdentity(identity="org/b", lane="shared", warmth=Warmth.PINNED),
        ),
    )
    result = run(levels("org/a", "org/b"), [receipt("org/b", A100)], current=current)
    assert result.report.identity("org/a").status is IdentityStatus.KEPT
    assert lanes_of(result) == {"org-b-a100-80gb": ("org/b",), "shared": ("org/a",)}
    assert {entry.identity: entry for entry in result.plan.identities}["org/a"] == PlanIdentity(
        identity="org/a", lane="shared", warmth=Warmth.PINNED
    )
    kept_together = run(levels("org/a", "org/b"), [receipt("org/b")], current=current)
    assert kept_together.plan == current


def test_a_host_is_measured_net_of_the_moves_its_lane_carries():
    receipts = [receipt("org/a", A100), receipt("org/b")]
    boots = [boot("org/b", seconds=120.0)]
    co = [co_receipt("org/a", "org/b")]
    tight = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=60, warm_p99_ms=250)})
    result = run(tight, receipts, co, boots=boots)
    assert lanes_of(result) == {"org-b-l4": ("org/a", "org/b")}
    host = result.report.identity("org/b")
    assert host.chosen.warmth is Warmth.PINNED
    assert host.next_cheaper is None
    assert result.report.come_alive_levers == ()
    assert result.report.measurement_requests == ()
    loose = levels("org/a", overrides={"org/b": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    loosened = run(loose, receipts, co, boots=boots, current=result.plan)
    assert loosened.report.monthly_cost > result.report.monthly_cost


def test_a_host_lever_claims_what_loosening_its_bound_saves_the_plan():
    receipts = [receipt("org/h", A100), receipt("org/m")]
    boots = [boot("org/h", seconds=120.0, cell=A100)]
    co = [co_receipt("org/h", "org/m", cell=A100)]
    tight = levels("org/m", overrides={"org/h": ServiceLevel(come_alive_p95_s=60, warm_p99_ms=250)})
    result = run(tight, receipts, co, boots=boots)
    assert lanes_of(result) == {"org-h-a100-80gb": ("org/h", "org/m")}
    (lever,) = result.report.come_alive_levers
    assert (lever.identity, lever.warmth, lever.permitted_bound_s) == ("org/h", Warmth.COLD_WITH_BOUND, 120)
    assert lever.saving_at_permitted_bound == pytest.approx(2 * MONTH)
    assert result.report.identity("org/h").next_cheaper.saving == pytest.approx(2 * MONTH)
    (request,) = co_requests(result.report)
    assert (request.identity, request.lane, request.saving_bound) == ("org/h", "org-m-l4", pytest.approx(2 * MONTH))
    loose = levels("org/m", overrides={"org/h": ServiceLevel(come_alive_p95_s=120, warm_p99_ms=250)})
    loosened = run(loose, receipts, co, boots=boots, current=result.plan)
    assert result.report.monthly_cost - loosened.report.monthly_cost == pytest.approx(lever.saving_at_permitted_bound)
    measured = run(tight, receipts, [*co, co_receipt("org/h", "org/m")], boots=boots, current=result.plan)
    assert result.report.monthly_cost - measured.report.monthly_cost == pytest.approx(request.saving_bound)


def test_a_co_residency_request_names_the_lane_as_the_pass_found_it():
    service_levels = levels("org/x", "org/y", "org/z")
    receipts = [receipt("org/x", A100), receipt("org/y"), receipt("org/z")]
    co = [co_receipt("org/y", "org/z")]
    result = run(service_levels, receipts, co)
    assert lanes_of(result) == {"org-x-a100-80gb": ("org/x",), "org-z-l4": ("org/y", "org/z")}
    requests = {request.lane: request for request in co_requests(result.report)}
    assert sorted(requests) == ["org-y-l4", "org-z-l4"]
    assert requests["org-y-l4"].to_dict() == {
        "identity": "org/x",
        "cell": L4.to_dict(),
        "containerShape": SHAPE.to_dict(),
        "concurrency": 8,
        "lane": "org-y-l4",
        "coResidents": ["org/y"],
        "measure": ["co-residency-receipt"],
        "warmth": "pinned",
        "reason": "unmeasured",
        "costBound": 0.0,
        "savingBound": 3 * MONTH,
    }
    assert requests["org-z-l4"].co_residents == ("org/z",)
    joined_y = run(service_levels, receipts, [*co, co_receipt("org/x", "org/y")], current=result.plan)
    assert lanes_of(joined_y) == {"org-y-l4": ("org/x", "org/y"), "org-z-l4": ("org/z",)}
    joined_z = run(service_levels, receipts, [*co, co_receipt("org/x", "org/z")], current=result.plan)
    assert lanes_of(joined_z) == {"org-y-l4": ("org/y",), "org-z-l4": ("org/x", "org/z")}


def test_two_lanes_that_need_one_receipt_ask_for_it_once():
    hourly = {"org/b": {3: 54000}}
    result = run(levels("org/a", "org/b"), [receipt("org/a"), receipt("org/b")], hourly=hourly)
    (request,) = co_requests(result.report)
    assert (request.identity, request.lane, request.co_residents) == ("org/b", "org-a-l4", ("org/a",))
    assert request.saving_bound == pytest.approx(MONTH + SCALE)
    service_levels = levels("org/a", "org/b", "org/c")
    three = run(service_levels, [receipt("org/a"), receipt("org/b"), receipt("org/c")])
    sets = [frozenset((request.identity, *request.co_residents)) for request in co_requests(three.report)]
    assert sorted(sorted(members) for members in sets) == [["org/a", "org/b"], ["org/a", "org/c"], ["org/b", "org/c"]]


@pytest.mark.parametrize(("rps", "requests"), [(0.3, 1080), (1.13, 4068), (2.01, 14472)])
def test_joint_and_solo_pricing_count_containers_at_the_stated_rate(rps, requests):
    service_levels = levels("org/a", "org/b")
    receipts = [receipt("org/a", A100), receipt("org/b", rps=rps)]
    hourly = {"org/b": {3: requests}}
    alone = run(service_levels, receipts, hourly=hourly)
    needed = requests // round(rps * 3600)
    assert alone.report.identity("org/b").peak_containers == needed
    solo = {lane.name: lane for lane in alone.report.lanes}["org-b-l4"].monthly_cost
    assert solo == pytest.approx(MONTH + (needed - 1) * SCALE)
    packed = run(service_levels, receipts, [co_receipt("org/a", resident("org/b", rps=rps))], hourly=hourly)
    (lane_report,) = packed.report.lanes
    assert lane_report.monthly_cost == pytest.approx(solo)
    assert packed.report.identity("org/b").monthly_cost == pytest.approx(solo)
    assert packed.report.co_packing_moves[0].saving == pytest.approx(3 * MONTH)


def test_a_co_packed_identity_counts_its_peak_at_its_co_resident_throughput():
    hourly = {"org/a": {3: 144000}}
    result = run(*PAIR, [co_receipt(resident("org/a", rps=5.0), "org/b")], hourly=hourly)
    moved = result.report.identity("org/a")
    assert moved.status is IdentityStatus.CO_PACKED
    assert moved.peak_containers == 8
    assert moved.chosen.cell == A100
    assert (moved.joined.cell, moved.joined.warmth, moved.joined.come_alive_basis) == (
        L4,
        Warmth.PINNED,
        ComeAliveBasis.WARM_FLOOR,
    )
    assert run(*PAIR, hourly=hourly).report.identity("org/a").peak_containers == 4


def test_ties_between_hosts_go_to_the_lane_whose_residents_sort_first():
    current = Plan(
        placement="test",
        lanes=(lane("aaa"),),
        identities=(PlanIdentity(identity="org/z", lane="aaa", warmth=Warmth.PINNED),),
    )
    service_levels = levels("org/a", "org/b", "org/z")
    co = [co_receipt("org/a", "org/z"), co_receipt("org/a", "org/b")]
    result = run(service_levels, [receipt("org/a", A100), receipt("org/b")], co, current=current)
    assert result.report.identity("org/z").status is IdentityStatus.KEPT
    (move,) = result.report.co_packing_moves
    assert (move.to_lane, move.co_residents) == ("org-b-l4", ("org/b",))
    assert move.saving == pytest.approx(3 * MONTH)


@pytest.mark.parametrize("measured", [False, True])
def test_a_lane_on_an_unpriced_cell_hosts_no_move_and_no_request(measured):
    t4 = Cell(machine_profile="t4", gpu_product="NVIDIA-T4")
    current = Plan(
        placement="test",
        lanes=(lane("legacy", cell=t4),),
        identities=(PlanIdentity(identity="org/kept", lane="legacy", warmth=Warmth.PINNED),),
    )
    co = [co_receipt("org/a", "org/kept", cell=t4)] if measured else []
    result = run(levels("org/a", "org/kept"), [receipt("org/a", A100)], co, current=current)
    assert result.report.unpriced_lanes == ("legacy",)
    assert result.report.co_packing_moves == ()
    assert result.report.blocked_co_packing_moves == ()
    assert co_requests(result.report) == []
    assert [refusal.residents for refusal in result.report.refusals] == ([("org/a", "org/kept")] if measured else [])


@pytest.mark.parametrize(("rate", "asked"), [(0.000001, False), (0.00001, True)])
def test_a_co_residency_request_that_cannot_save_a_cent_is_dropped(rate, asked):
    prices = PriceList(
        currency="USD",
        machine_profiles=(price(L4, rate), price(A100, 3.0)),
        placements=(Placement(name="test", region_multiplier=1),),
    )
    result = run(levels("org/a", "org/b"), [receipt("org/a"), receipt("org/b")], prices=prices)
    assert [(request.identity, request.lane) for request in co_requests(result.report)] == (
        [("org/a", "org-b-l4")] if asked else []
    )


def test_identities_kept_apart_are_never_co_packed_or_asked_about():
    apart = frozenset({frozenset({"org/a", "org/b"})})
    measured = run(*PAIR, [co_receipt("org/a", "org/b")], apart=apart)
    assert measured.report.co_packing_moves == ()
    assert measured.report.blocked_co_packing_moves == ()
    assert co_requests(measured.report) == []
    assert measured.report.parameters["apart"] == [["org/a", "org/b"]]
    assert co_requests(run(*PAIR, apart=apart).report) == []
    service_levels = levels("org/a", "org/b", "org/c")
    receipts = [receipt("org/a", A100), receipt("org/b"), receipt("org/c")]
    co = [co_receipt("org/a", "org/b"), co_receipt("org/a", "org/c")]
    result = run(service_levels, receipts, co, apart=apart)
    assert lanes_of(result) == {"org-b-l4": ("org/b",), "org-c-l4": ("org/a", "org/c")}
    assert not any(
        {"org/a", "org/b"} <= {request.identity, *request.co_residents} for request in co_requests(result.report)
    )


@pytest.mark.parametrize(
    "apart",
    [
        frozenset({frozenset({"org/a"})}),
        frozenset({("org/a", "org/b")}),
        frozenset({frozenset({"org/a", "two words"})}),
    ],
)
def test_apart_holds_pairs_of_identities(apart):
    with pytest.raises(SchemaError, match="apart"):
        parameters(apart=apart)


def test_a_blocked_move_is_listed_by_what_it_would_save_beyond_the_plan():
    service_levels = levels("org/a", "org/b", "org/c")
    receipts = [receipt("org/a", A100), receipt("org/b"), receipt("org/c")]
    hourly = {"org/a": {3: 72000}}
    admitted = co_receipt(resident("org/a", rps=10.0), "org/b")
    faster = co_receipt(resident("org/a", rps=20.0, p99=300.0), "org/c")
    result = run(service_levels, receipts, [admitted, faster], hourly=hourly)
    (move,) = result.report.co_packing_moves
    assert move.to_lane == "org-b-l4"
    (blocked,) = result.report.blocked_co_packing_moves
    assert (blocked.identity, blocked.from_lane, blocked.to_lane, blocked.co_residents) == (
        "org/a",
        "org-a-a100-80gb",
        "org-c-l4",
        ("org/c",),
    )
    assert blocked.receipt == faster.source
    assert blocked.blocked_by == (Blocker(identity="org/a", field="warmP99Ms", bound=250, measured=300.0),)
    assert blocked.saving == pytest.approx(SCALE)
    slower = co_receipt(resident("org/a", rps=5.0, p99=300.0), "org/c")
    assert run(service_levels, receipts, [admitted, slower], hourly=hourly).report.blocked_co_packing_moves == ()


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def estate(co_residency=True) -> dict:
    return {
        "service_levels": ServiceLevels.from_dict(load("service-levels.json")),
        "receipts": [Receipt.from_dict(document) for document in load("receipts.json")],
        "boot_evidence": [BootEvidence.from_dict(document) for document in load("boot-evidence.json")],
        "price_list": PriceList.from_dict(load("price-list.json")),
        "demand": Demand.from_dict(load("demand.json")),
        "current_plan": Plan.from_dict(load("current-plan.json")),
        "parameters": PlannerParameters(
            placement="us-central1",
            max_containers=4,
            fallback_floor=2,
            cooldown_s=900,
            buffer_containers=1,
            bundles={"Qwen/Qwen3-8B": "sglang"},
        ),
        "co_residency_receipts": (
            [CoResidencyReceipt.from_dict(document) for document in load("co-residency-receipts.json")]
            if co_residency
            else []
        ),
    }


def test_the_co_packed_estate_plan_and_report_are_byte_stable():
    plan_text, report_text = rendered(plan(**estate()))
    assert plan_text == (FIXTURES / "expected-copack-plan.json").read_text(encoding="utf-8")
    assert report_text == (FIXTURES / "expected-copack-report.json").read_text(encoding="utf-8")
    shuffler = random.Random(5)  # noqa: S311 -- reproducible test data
    for _ in range(5):
        inputs = estate()
        shuffler.shuffle(inputs["receipts"])
        shuffler.shuffle(inputs["co_residency_receipts"])
        shuffler.shuffle(inputs["boot_evidence"])
        assert rendered(plan(**inputs)) == (plan_text, report_text)


def test_the_co_packed_estate_states_its_moves():
    result = plan(**estate())
    report = result.report
    (move,) = report.co_packing_moves
    assert (move.identity, move.from_lane, move.to_lane) == (
        "Qwen/Qwen3-Embedding-4B",
        "qwen-qwen3-embedding-4b-l4",
        "default-l4-default",
    )
    assert move.co_residents == ("BAAI/bge-m3", "intfloat/e5-base-v2")
    assert move.saving == pytest.approx(1.04 * MONTH)
    shared = {lane.name: lane for lane in report.lanes}["default-l4-default"]
    assert shared.identities == ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B", "intfloat/e5-base-v2")
    assert shared.unpriced_identities == ()
    assert report.identity("intfloat/e5-base-v2").monthly_cost == pytest.approx(1.04 * MONTH)
    sparse = [request for request in co_requests(report) if request.identity == "BAAI/bge-m3:sparse"]
    assert [(request.lane, request.co_residents) for request in sparse] == [
        ("default-l4-default", ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B", "intfloat/e5-base-v2"))
    ]
    (refusal,) = [refusal for refusal in report.refusals if refusal.record is RefusedRecord.CO_RESIDENCY_RECEIPT]
    assert refusal.residents == ("BAAI/bge-m3", "Qwen/Qwen3-Embedding-4B")
    without = plan(**estate(co_residency=False)).report
    assert report.monthly_cost == pytest.approx(without.monthly_cost - move.saving)
    again = plan(**{**estate(), "current_plan": result.plan})
    assert again.plan == result.plan
    assert again.report.unchanged


def random_estate(seed: int) -> dict:
    rng = random.Random(seed)  # noqa: S311 -- reproducible test data
    identities = [f"org/model-{index}" for index in range(rng.randint(2, 6))]
    service_levels = levels(
        overrides={
            identity: ServiceLevel(
                come_alive_p95_s=rng.choice([0, 0, 120]), warm_p99_ms=rng.choice([None, 150, 250, 400])
            )
            for identity in identities
        },
    )
    receipts = [
        receipt(
            identity,
            cell,
            concurrency=rng.choice([8, 16]),
            rps=rng.choice([1.0, 10.0]),
            p99=rng.choice([100.0, 200.0]),
        )
        for identity in identities
        for cell in (L4, A100)
        if rng.random() < 0.7
    ]
    boots = [
        boot(identity, rng.choice([30.0, 300.0]), cell)
        for identity in identities
        for cell in (L4, A100)
        if rng.random() < 0.3
    ]
    co = []
    for _ in range(rng.randint(0, 12)):
        members = rng.sample(identities, rng.randint(2, min(4, len(identities))))
        template = rng.choice([entry for entry in receipts if entry.identity in members] or [receipt(members[0])])
        co.append(
            co_receipt(
                *(
                    resident(member, p99=rng.choice([120.0, 240.0, 380.0]), rps=rng.choice([2.0, 10.0]))
                    for member in members
                ),
                cell=template.cell,
                concurrency=template.concurrency,
                peak=rng.choice([8000, 20000]),
                uri=f"co/{seed}/{len(co)}",
            )
        )
    hourly = {
        identity: {hour: rng.choice([0, 10, 30000]) for hour in rng.sample(range(24), rng.randint(0, 4))}
        for identity in identities
    }
    return {
        "identities": identities,
        "service_levels": service_levels,
        "receipts": receipts,
        "co": co,
        "boots": boots,
        "hourly": hourly,
        "headroom": rng.choice([0, 4096]),
        "rng": rng,
    }


def test_the_random_estates_exercise_moves_onto_shared_lanes():
    moved = shared = 0
    for seed in range(SEEDS):
        inputs = random_estate(seed)
        result = run(
            inputs["service_levels"],
            inputs["receipts"],
            inputs["co"],
            boots=inputs["boots"],
            hourly=inputs["hourly"],
            gpu_memory_headroom_mib=inputs["headroom"],
        )
        moved += len(result.report.co_packing_moves)
        shared += sum(len(lane.identities) > 2 for lane in result.report.lanes if lane.co_residency_receipt)
    assert moved >= SEEDS // 3
    assert shared >= 1


@pytest.mark.parametrize("seed", range(SEEDS))
def test_random_estates_co_pack_only_measured_sets_and_replan_to_themselves(seed):
    inputs = random_estate(seed)
    identities, service_levels, receipts = inputs["identities"], inputs["service_levels"], inputs["receipts"]
    co, boots, hourly, headroom, rng = (
        inputs["co"],
        inputs["boots"],
        inputs["hourly"],
        inputs["headroom"],
        inputs["rng"],
    )
    result = run(service_levels, receipts, co, boots=boots, hourly=hourly, gpu_memory_headroom_mib=headroom)
    assert Plan.from_dict(result.plan.to_dict()) == result.plan
    effective = {identity: service_levels.effective(identity) for identity in identities}
    by_uri = {receipt.source.uri: receipt for receipt in co}
    warmths = {entry.identity: entry for entry in result.plan.identities}
    for lane_report in result.report.lanes:
        if lane_report.co_residency_receipt is None:
            continue
        measured = by_uri[lane_report.co_residency_receipt.uri]
        assert measured.identities == frozenset(lane_report.identities)
        assert measured.gpu_memory_headroom_mib >= headroom
        for member in measured.residents:
            assert warmths[member.identity].warmth is Warmth.PINNED
            bound = effective[member.identity].warm_p99_ms
            assert bound is None or member.latency.p99_ms <= bound
    co_packed = {entry.identity for entry in result.plan.identities if entry.co_packed}
    assert co_packed == {move.identity for move in result.report.co_packing_moves}
    covered = {
        identity
        for lane_report in result.report.lanes
        if lane_report.co_residency_receipt is not None
        for identity in lane_report.identities
    }
    assert co_packed <= covered
    without = run(service_levels, receipts, boots=boots, hourly=hourly, gpu_memory_headroom_mib=headroom)
    floors = {
        lane_report.identities[0]: lane_report.floor_monthly_cost
        for lane_report in without.report.lanes
        if len(lane_report.identities) == 1
    }
    visits = [(-floors[move.identity], move.identity) for move in result.report.co_packing_moves]
    assert visits == sorted(visits)
    assert all(round(move.saving, 2) > 0 for move in result.report.co_packing_moves)
    assert result.report.monthly_cost == pytest.approx(
        without.report.monthly_cost - sum(move.saving for move in result.report.co_packing_moves)
    )
    for request in co_requests(result.report):
        assert request.saving_bound is not None
        assert request.saving_bound > 0
    again = run(
        service_levels, receipts, co, boots=boots, hourly=hourly, gpu_memory_headroom_mib=headroom, current=result.plan
    )
    assert again.plan == result.plan
    assert again.report.unchanged
    shuffled = list(co)
    rng.shuffle(shuffled)
    reordered = run(
        service_levels, list(reversed(receipts)), shuffled, boots=boots, hourly=hourly, gpu_memory_headroom_mib=headroom
    )
    assert rendered(reordered) == rendered(result)
